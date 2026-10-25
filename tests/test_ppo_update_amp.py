# ruff: noqa: E402 -- filter upstream import deprecations before importing TorchRL.
"""`ppo.update_amp_dtype`: off is bit-identical, bf16 keeps float32 state."""

from __future__ import annotations

import copy
import warnings

warnings.filterwarnings(
    "ignore",
    message="`torch.jit.script_method` is deprecated.*",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message="Creating .* which inherits from WeightUpdaterBase is deprecated.*",
    category=DeprecationWarning,
)
import pytest
import torch
from tensordict import TensorDict
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torchrl.data import Composite, Unbounded
from torchrl.envs import EnvBase, StepCounter, TransformedEnv

from rlopt.agent.ipmd.ipmd import IPMD, IPMDRLOptConfig
from rlopt.agent.ppo.mixed_precision import (
    AMP_OLD_LOG_PROB_KEY,
    resolve_update_amp_dtype,
    trailing_linear,
    update_autocast,
)
from rlopt.agent.ppo.ppo import PPO, PPOConfig, PPORLOptConfig
from rlopt.env_utils import make_parallel_env
from rlopt.models import GaussianPolicyHead

LOSS_KEYS = ("loss_objective", "loss_critic", "loss_entropy")


def _configure(cfg, tmp_path, amp: str | None) -> None:
    cfg.env.env_name = "Pendulum-v1"
    cfg.env.device = cfg.device = "cpu"
    cfg.seed = 0
    cfg.collector.frames_per_batch = 64
    cfg.collector.total_frames = 64
    # Policy-sampled rows, so the stored loc/scale are the actor's own.
    cfg.collector.init_random_frames = 0
    cfg.replay_buffer.size = 64
    cfg.loss.mini_batch_size = 64
    cfg.loss.epochs = 1
    cfg.compile.compile = False
    cfg.logger.backend = ""
    cfg.logger.log_dir = str(tmp_path)
    cfg.policy.input_keys = ["observation"]
    cfg.policy.num_cells = [32, 32]
    cfg.policy.normalize_input = True
    cfg.value_function.input_keys = ["observation"]
    cfg.value_function.num_cells = [32, 32]
    cfg.value_function.normalize_input = True
    cfg.optim.max_grad_norm = 0.5
    cfg.ppo.update_amp_dtype = amp


def _make_agent(
    kind: str, tmp_path, amp: str | None = None, ipmd_amp: str | None = None
):
    if kind == "ipmd":
        cfg = IPMDRLOptConfig()
        _configure(cfg, tmp_path, amp)
        cfg.ipmd.update_amp_dtype = ipmd_amp
        cfg.ipmd.use_latent_command = False
        cfg.ipmd.reward_input_keys = ["observation"]
        cfg.ipmd.reward_loss_coeff = 0
        cfg.ipmd.reward_l2_coeff = cfg.ipmd.reward_grad_penalty_coeff = (
            cfg.ipmd.reward_logit_reg_coeff
        ) = 0
        cfg.ipmd.reward_param_weight_decay_coeff = cfg.ipmd.bc_coef = (
            cfg.ipmd.rollout_bc_coef
        ) = cfg.ipmd.diversity_bonus_coeff = 0
        return IPMD(make_parallel_env(cfg), cfg)
    cfg = PPORLOptConfig()
    _configure(cfg, tmp_path, amp)
    return PPO(make_parallel_env(cfg), cfg)


def _first_batch(agent) -> TensorDict:
    rollout = next(iter(agent.collector))
    agent.actor_critic.train()
    if isinstance(agent, IPMD):
        agent._prepare_rollout_rewards(rollout)
    agent.data_buffer.empty()
    agent.pre_iteration_compute(rollout)
    return next(iter(agent.data_buffer))


def _agent_update(agent, batch: TensorDict) -> TensorDict:
    # TanhNormal has no closed-form entropy: TorchRL samples it, so pin the RNG.
    torch.manual_seed(1234)
    if isinstance(agent, IPMD):
        loss, _ = agent.update(
            batch,
            0,
            TensorDict({}, batch_size=batch.batch_size),
            torch.zeros(()),
        )
        return loss
    loss, _ = agent.update(batch, 0)
    return loss


def _reference_update(agent, batch: TensorDict) -> TensorDict:
    """The PPO-terms gradient step exactly as written before the key existed."""
    torch.manual_seed(1234)
    agent.optim.zero_grad(set_to_none=True)
    loss = agent.loss_module(batch)
    extra, _ = agent._extra_actor_loss(batch)
    (
        loss["loss_critic"] + loss["loss_objective"] + loss["loss_entropy"] + extra
    ).backward()
    max_norm = getattr(agent, "_max_grad_norm", agent.config.optim.max_grad_norm)
    grad_norm = clip_grad_norm_(agent._grad_clip_params, float(max_norm))
    agent.optim.step()
    out = loss.detach()
    out.set("grad_norm", grad_norm.detach())
    return out


def _snapshot(agent):
    return (
        copy.deepcopy(agent.actor_critic.state_dict()),
        copy.deepcopy(agent.optim.state_dict()),
    )


def _restore(agent, snapshot) -> None:
    model_state, optim_state = snapshot
    agent.actor_critic.load_state_dict(model_state)
    agent.optim.load_state_dict(optim_state)


def _params(agent) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in agent.actor_critic.named_parameters()}


def test_resolve_update_amp_dtype():
    assert PPOConfig().update_amp_dtype is None
    for off in (None, "none", "fp32", "float32", "FP32"):
        assert resolve_update_amp_dtype(off) is None
    assert resolve_update_amp_dtype("bf16") is torch.bfloat16
    assert resolve_update_amp_dtype("bfloat16") is torch.bfloat16
    with pytest.raises(ValueError, match="GradScaler"):
        resolve_update_amp_dtype("fp16")
    with pytest.raises(ValueError, match="update_amp_dtype"):
        resolve_update_amp_dtype("int8")


@pytest.mark.parametrize("kind", ["ppo", "ipmd"])
def test_update_amp_off_is_bit_identical_to_reference(kind, tmp_path):
    agent = _make_agent(kind, tmp_path, amp=None)
    assert agent._update_amp_hooks == []
    batch = _first_batch(agent)
    start = _snapshot(agent)

    reference = _reference_update(agent, batch.clone())
    reference_params = _params(agent)
    reference_optim = copy.deepcopy(agent.optim.state_dict()["state"])

    _restore(agent, start)
    loss = _agent_update(agent, batch.clone())
    for key in (*LOSS_KEYS, "grad_norm"):
        assert torch.equal(loss[key], reference[key]), key
    for name, value in _params(agent).items():
        assert torch.equal(value, reference_params[name]), name
    for idx, state in agent.optim.state_dict()["state"].items():
        for key, value in state.items():
            if torch.is_tensor(value):
                assert torch.equal(value, reference_optim[idx][key]), (idx, key)


@pytest.mark.parametrize("kind", ["ppo", "ipmd"])
def test_update_amp_bf16_keeps_fp32_state_and_finite_losses(kind, tmp_path):
    agent = _make_agent(kind, tmp_path, amp="bf16")
    # Two output projections re-run in float32 + two float32 output casts.
    assert len(agent._update_amp_hooks) == 4
    batch = _first_batch(agent)
    start = _snapshot(agent)

    reference = _reference_update(agent, batch.clone())
    _restore(agent, start)

    head = next(m for m in agent.policy.modules() if isinstance(m, GaussianPolicyHead))
    trunk = head.base.module
    first_linear = trunk[0]
    last_linear = trailing_linear(head.base)
    assert last_linear is trunk[-1]
    seen: dict[str, list[torch.dtype]] = {"first": [], "last": []}
    probes = [
        first_linear.register_forward_hook(
            lambda _m, _a, out: seen["first"].append(out.dtype)
        ),
        last_linear.register_forward_hook(
            lambda _m, _a, out: seen["last"].append(out.dtype)
        ),
    ]
    try:
        loss = _agent_update(agent, batch.clone())
    finally:
        for probe in probes:
            probe.remove()

    # The hidden trunk ran in bf16; the projection and everything after it in fp32.
    assert seen["first"] == [torch.bfloat16]
    assert seen["last"] == [torch.float32]
    for key in (*LOSS_KEYS, "grad_norm", "clip_fraction", "kl_approx"):
        assert loss[key].dtype == torch.float32, key
        assert torch.isfinite(loss[key]).all(), key
    for name, param in agent.actor_critic.named_parameters():
        assert param.dtype == torch.float32, name
        if param.grad is not None:
            assert param.grad.dtype == torch.float32, name
    for state in agent.optim.state_dict()["state"].values():
        for key, value in state.items():
            if torch.is_tensor(value) and value.is_floating_point():
                assert value.dtype == torch.float32, key
    # Autocast engaged (the numbers moved) but stayed close to float32.
    assert not all(torch.equal(loss[k], reference[k]) for k in LOSS_KEYS)
    for key in LOSS_KEYS:
        torch.testing.assert_close(loss[key], reference[key], rtol=0.05, atol=0.05)
    # The first step's loss is at the old weights: the bf16 old log-prob
    # makes the ratio exactly 1 (full batch, same row count).
    assert loss["max_ratio"].item() == 1.0
    assert loss["kl_approx"].item() == 0.0
    assert loss["clip_fraction"].item() == 0.0


def test_update_amp_hooks_are_inert_outside_autocast(tmp_path):
    agent = _make_agent("ipmd", tmp_path, amp="bf16")
    batch = _first_batch(agent)
    obs = batch.select("observation")
    policy = agent.actor_critic.get_policy_operator()
    value = agent.actor_critic.get_value_operator()
    with torch.no_grad():
        hooked = policy(obs.clone())
        hooked_value = value(obs.clone())
        for handle in agent._update_amp_hooks:
            handle.remove()
        plain = policy(obs.clone())
        plain_value = value(obs.clone())
    for key in ("loc", "scale"):
        assert hooked[key].dtype == torch.float32
        assert torch.equal(hooked[key], plain[key]), key
    assert torch.equal(hooked_value["state_value"], plain_value["state_value"])


def test_update_amp_refuses_update_paths_that_ignore_it():
    class _OwnUpdate(PPO):
        def update(self, batch, num_network_updates):  # pragma: no cover
            return super().update(batch, num_network_updates)

    class _OwnPPOTerms(IPMD):
        def _backward_ppo_terms(self, batch):  # pragma: no cover
            return super()._backward_ppo_terms(batch)

    class _Inherits(IPMD):
        pass

    for cls in (_OwnUpdate, _OwnPPOTerms):
        with pytest.raises(ValueError, match="inert"):
            object.__new__(cls)._validate_update_amp_support()
    for cls in (PPO, IPMD, _Inherits):
        object.__new__(cls)._validate_update_amp_support()


def test_ipmd_section_copy_of_the_key_is_refused(tmp_path):
    # IPMDConfig inherits PPOConfig, so `ipmd.update_amp_dtype` exists but is
    # not what the update reads.
    with pytest.raises(ValueError, match=r"set ppo\.update_amp_dtype"):
        _make_agent("ipmd", tmp_path, ipmd_amp="bf16")


def test_kl_sample_size_subsamples_without_touching_global_rng(tmp_path):
    agent = _make_agent("ipmd", tmp_path)
    batch = _first_batch(agent)
    policy = agent.actor_critic.get_policy_operator()

    assert agent.config.optim.kl_sample_size is None
    full = agent._prepare_kl_context(batch, policy)
    assert full["obs_td"].batch_size == batch.batch_size
    assert torch.equal(full["old_loc"], batch["loc"])

    agent.config.optim.kl_sample_size = 16
    rng_before = torch.random.get_rng_state()
    first = agent._prepare_kl_context(batch, policy)
    second = agent._prepare_kl_context(batch, policy)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    for ctx in (first, second):
        assert ctx["obs_td"].batch_size == torch.Size([16])
        assert ctx["old_loc"].shape[0] == 16
        # Every sampled row is a row of the minibatch, with its own loc.
        obs = ctx["obs_td"]["observation"]
        match = (obs[:, None, :] == batch["observation"][None, :, :]).all(-1)
        assert match.any(dim=1).all()
        rows = match.float().argmax(dim=1)
        assert torch.equal(ctx["old_loc"], batch["loc"][rows])
    # A fresh subset per update.
    assert not torch.equal(
        first["obs_td"]["observation"], second["obs_td"]["observation"]
    )
    kl = agent._compute_kl_after_update(first, policy)
    assert kl is not None
    assert torch.isfinite(kl)

    agent.config.optim.kl_sample_size = 0
    with pytest.raises(ValueError, match="kl_sample_size"):
        agent._prepare_kl_context(batch, policy)


# --------------------------------------------------------------------------
# CUDA: the PPO ratio is exactly 1 at the old weights under bf16.
# --------------------------------------------------------------------------

_CUDA_ENVS = 256
_CUDA_STEPS = 8
_CUDA_ROWS = _CUDA_ENVS * _CUDA_STEPS


class _FakeVecEnv(EnvBase):
    """48-D observations, 8-D unbounded actions, random dynamics, on CUDA."""

    def __init__(self, num_envs: int, device: str) -> None:
        super().__init__(batch_size=torch.Size([num_envs]), device=device)
        self.observation_spec = Composite(
            observation=Unbounded((num_envs, 48)), shape=(num_envs,)
        )
        self.action_spec = Unbounded((num_envs, 8))
        self.reward_spec = Unbounded((num_envs, 1))

    def _flags(self) -> dict[str, Tensor]:
        shape = (*self.batch_size, 1)
        return {
            "done": torch.zeros(shape, dtype=torch.bool, device=self.device),
            "terminated": torch.zeros(shape, dtype=torch.bool, device=self.device),
        }

    def _reset(self, tensordict):  # noqa: ARG002
        return TensorDict(
            {
                "observation": torch.randn(*self.batch_size, 48, device=self.device),
                **self._flags(),
            },
            batch_size=self.batch_size,
            device=self.device,
        )

    def _step(self, tensordict):  # noqa: ARG002
        return TensorDict(
            {
                "observation": torch.randn(*self.batch_size, 48, device=self.device),
                "reward": torch.randn(*self.batch_size, 1, device=self.device),
                **self._flags(),
            },
            batch_size=self.batch_size,
            device=self.device,
        )

    def _set_seed(self, seed):
        return seed


def _make_cuda_agent(kind: str, tmp_path, amp: str | None, mini_batch_size: int):
    cfg = IPMDRLOptConfig() if kind == "ipmd" else PPORLOptConfig()
    cfg.env.device = cfg.device = "cuda"
    cfg.seed = 0
    cfg.collector.frames_per_batch = _CUDA_ROWS
    cfg.collector.total_frames = _CUDA_ROWS
    cfg.collector.init_random_frames = 0
    cfg.replay_buffer.size = _CUDA_ROWS
    cfg.loss.mini_batch_size = mini_batch_size
    cfg.loss.epochs = 1
    cfg.compile.compile = False
    cfg.logger.backend = ""
    cfg.logger.log_dir = str(tmp_path)
    for net in (cfg.policy, cfg.value_function):
        net.input_keys = ["observation"]
        net.num_cells = [512, 512, 256]
        net.activation_fn = "silu"
        net.normalize_input = True
    # A sharp policy: the bf16 trunk noise in loc is large against sigma.
    cfg.ppo.log_std_init = -3.0
    cfg.ppo.update_amp_dtype = amp
    if kind == "ipmd":
        cfg.ipmd.use_latent_command = False
        cfg.ipmd.reward_input_keys = ["observation"]
        cfg.ipmd.reward_loss_coeff = 0
        cfg.ipmd.reward_l2_coeff = cfg.ipmd.reward_grad_penalty_coeff = (
            cfg.ipmd.reward_logit_reg_coeff
        ) = 0
        cfg.ipmd.reward_param_weight_decay_coeff = cfg.ipmd.bc_coef = (
            cfg.ipmd.rollout_bc_coef
        ) = cfg.ipmd.diversity_bonus_coeff = 0
    env = TransformedEnv(_FakeVecEnv(_CUDA_ENVS, "cuda"), StepCounter(1000))
    return (IPMD if kind == "ipmd" else PPO)(env, cfg)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kind", ["ppo", "ipmd"])
@pytest.mark.parametrize("minibatches", [1, 4])
def test_update_amp_ratio_is_exactly_one_at_old_weights_cuda(
    kind, minibatches, tmp_path
):
    torch.manual_seed(0)
    agent = _make_cuda_agent(kind, tmp_path, "bf16", _CUDA_ROWS // minibatches)
    rollout = next(iter(agent.collector))
    agent.actor_critic.train()
    if isinstance(agent, IPMD):
        agent._prepare_rollout_rewards(rollout)
    behaviour_log_prob = rollout["action_log_prob"].clone()
    behaviour_loc = rollout["loc"].clone()
    agent.data_buffer.empty()
    agent.pre_iteration_compute(rollout)
    # The stored behaviour log-prob and loc are untouched.
    assert torch.equal(rollout["action_log_prob"], behaviour_log_prob)
    assert torch.equal(rollout["loc"], behaviour_loc)

    # One epoch of the update's own minibatches: random row subsets.
    batches = list(agent.data_buffer)
    assert len(batches) == minibatches
    assert all(int(b.batch_size[0]) == _CUDA_ROWS // minibatches for b in batches)
    for batch in batches:
        loss = agent._update_loss_forward(batch)
        assert loss["clip_fraction"].item() == 0.0
        assert loss["max_ratio"].item() == 1.0
        assert loss["mean_ratio"].item() == 1.0
        assert loss["kl_approx"].item() == 0.0
        # The bf16 old log-prob really differs from the behaviour one.
        assert not torch.equal(batch[AMP_OLD_LOG_PROB_KEY], batch["action_log_prob"])
        # The KL diagnostic still compares against the behaviour loc/scale.
        policy = agent.actor_critic.get_policy_operator()
        ctx = agent._prepare_kl_context(batch, policy)
        assert torch.equal(ctx["old_loc"], batch["loc"])

    # Control: the stored float32 behaviour log-prob is NOT exact under bf16.
    with update_autocast("cuda", torch.bfloat16):
        raw = agent.loss_module(batches[0])
    assert raw["max_ratio"].item() != 1.0
    assert raw["kl_approx"].item() != 0.0
