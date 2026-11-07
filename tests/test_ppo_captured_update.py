# ruff: noqa: E402 -- filter upstream import deprecations before importing TorchRL.
"""`ppo.capture_update`: the CUDA-graph update is bit-identical to the eager one.

Every check runs real PPO / IPMD iterations (collect -> GAE -> epochs ->
normalizer update -> collector weight sync) on a random CUDA env, twice from
the same seed, eager and captured, and compares parameters, module buffers,
optimizer state, per-iteration learning rates and logged ``train/`` metrics,
and the CUDA / CPU RNG state afterwards.
"""

from __future__ import annotations

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
from torchrl.data import Composite, Unbounded
from torchrl.envs import EnvBase, StepCounter, TransformedEnv

from rlopt.agent.ipmd.ipmd import IPMD, IPMDRLOptConfig
from rlopt.agent.ppo.captured_update import (
    exact_adam_scalars,
    exact_adam_step,
    resolve_capture_update_optimizer,
)
from rlopt.agent.ppo.ppo import PPO, PPOConfig, PPORLOptConfig

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA graphs need a CUDA device"
)

NUM_ENVS = 256
STEPS = 8
ROWS = NUM_ENVS * STEPS
OBS = 48
ACT = 8


class _FakeVecEnv(EnvBase):
    """Random observations (actor + critic-only keys), unbounded actions, CUDA."""

    def __init__(self, num_envs: int, device: str) -> None:
        super().__init__(batch_size=torch.Size([num_envs]), device=device)
        self.observation_spec = Composite(
            observation=Unbounded((num_envs, OBS)),
            critic_extra=Unbounded((num_envs, 16)),
            shape=(num_envs,),
        )
        self.action_spec = Unbounded((num_envs, ACT))
        self.reward_spec = Unbounded((num_envs, 1))

    def _flags(self) -> dict[str, Tensor]:
        shape = (*self.batch_size, 1)
        return {
            "done": torch.zeros(shape, dtype=torch.bool, device=self.device),
            "terminated": torch.zeros(shape, dtype=torch.bool, device=self.device),
        }

    def _obs(self) -> dict[str, Tensor]:
        return {
            "observation": 2.0 * torch.randn(*self.batch_size, OBS, device=self.device)
            + 0.5,
            "critic_extra": torch.randn(*self.batch_size, 16, device=self.device),
        }

    def _reset(self, tensordict):  # noqa: ARG002
        return TensorDict(
            {**self._obs(), **self._flags()},
            batch_size=self.batch_size,
            device=self.device,
        )

    def _step(self, tensordict):
        action = tensordict.get("action")
        reward = -(action.square().mean(-1, keepdim=True)) + 0.1 * torch.randn(
            *self.batch_size, 1, device=self.device
        )
        return TensorDict(
            {**self._obs(), "reward": reward, **self._flags()},
            batch_size=self.batch_size,
            device=self.device,
        )

    def _set_seed(self, seed):
        return seed


def _make_agent(
    kind: str,
    log_dir: str,
    *,
    capture: bool,
    minibatches: int = 1,
    epochs: int = 3,
    iterations: int = 4,
    kl_mode: str = "iteration",
    amp: str | None = None,
    normalizer: str = "after_rollout",
    optimizer: str = "exact",
    mini_batch_size: int | None = None,
):
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    cfg = IPMDRLOptConfig() if kind == "ipmd" else PPORLOptConfig()
    cfg.env.device = cfg.device = "cuda"
    cfg.seed = 0
    cfg.collector.frames_per_batch = ROWS
    cfg.collector.total_frames = ROWS * iterations
    cfg.collector.init_random_frames = 0
    cfg.replay_buffer.size = ROWS
    cfg.loss.mini_batch_size = mini_batch_size or ROWS // minibatches
    cfg.loss.epochs = epochs
    cfg.loss.gamma = 0.97
    cfg.compile.compile = False
    cfg.logger.backend = ""
    cfg.logger.log_dir = log_dir
    cfg.policy.input_keys = ["observation"]
    cfg.value_function.input_keys = ["observation", "critic_extra"]
    for net in (cfg.policy, cfg.value_function):
        net.num_cells = [256, 256, 128]
        net.activation_fn = "silu"
        net.normalize_input = True
    cfg.ppo.entropy_coeff = 0.0
    cfg.ppo.clip_value = True
    cfg.ppo.clip_log_std = False
    cfg.ppo.update_amp_dtype = amp
    cfg.ppo.capture_update = capture
    cfg.ppo.capture_update_optimizer = optimizer
    if normalizer == "frozen":
        cfg.ppo.freeze_normalizers = True
    elif normalizer == "live":
        cfg.ppo.update_normalizers_after_rollout = False
    cfg.optim.optimizer = "adamw"
    cfg.optim.weight_decay = 1.0e-2
    cfg.optim.max_grad_norm = 1.0
    cfg.optim.scheduler = "adaptive"
    cfg.optim.kl_adapt_step = kl_mode
    # A tiny target: the KL rule fires every iteration, so the LR keeps moving.
    cfg.optim.desired_kl = 1.0e-5
    cfg.optim.min_lr = 1.0e-5
    cfg.optim.max_lr = 1.0e-3
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
        # The tuned recipe's optimizer layout: actor / critic / log_std groups,
        # a per-iteration linear critic LR on top of the KL rule.
        cfg.ipmd.actor_learning_rate = 1.0e-3
        cfg.ipmd.critic_learning_rate = 1.0e-3
        cfg.ipmd.critic_lr_schedule = "linear"
        cfg.ipmd.critic_lr_final = 1.0e-5
    else:
        cfg.optim.lr = 1.0e-3
    env = TransformedEnv(_FakeVecEnv(NUM_ENVS, "cuda"), StepCounter(1000))
    return (IPMD if kind == "ipmd" else PPO)(env, cfg)


def _run(agent, iterations: int, metadata=None) -> tuple[list[dict], object]:
    """`train()` without logging or checkpoints."""
    metadata = metadata or agent.init_metadata()
    records = []
    for it in range(iterations):
        iteration = agent.collect(metadata, it)
        agent.prepare(iteration, metadata)
        agent.iterate(iteration, metadata)
        agent._update_rollout_normalizers(iteration.rollout)
        agent.collector.update_policy_weights_()
        records.append(
            {
                "metrics": {
                    k: v for k, v in iteration.metrics.items() if k.startswith("train/")
                },
                "lrs": [float(g["lr"]) for g in agent.optim.param_groups],
                "updates_completed": int(metadata.updates_completed),
            }
        )
    return records, metadata


def _snapshot(agent, records) -> dict:
    # TorchRL's collector makes its own CUDA stream current and leaves it so;
    # each agent clones on a different stream. Settle all of them first.
    torch.cuda.synchronize()
    snap = {
        "params": {
            k: v.detach().clone() for k, v in agent.actor_critic.named_parameters()
        },
        "buffers": {
            k: v.detach().clone() for k, v in agent.actor_critic.named_buffers()
        },
        "optim": {
            idx: {
                k: (v.detach().clone() if isinstance(v, Tensor) else v)
                for k, v in state.items()
            }
            for idx, state in agent.optim.state_dict()["state"].items()
        },
        "records": records,
        "cuda_rng": torch.cuda.get_rng_state().clone(),
        "cpu_rng": torch.get_rng_state().clone(),
    }
    torch.cuda.synchronize()
    return snap


def _mismatches(a: dict, b: dict) -> list[str]:
    torch.cuda.synchronize()
    bad = []
    for section in ("params", "buffers"):
        bad += [
            f"{section}:{k}"
            for k in a[section]
            if not torch.equal(a[section][k], b[section][k])
        ]
    for idx, state in a["optim"].items():
        for key, value in state.items():
            other = b["optim"][idx][key]
            if isinstance(value, Tensor):
                same = value.device == other.device and torch.equal(value, other)
            else:
                same = value == other
            if not same:
                bad.append(f"optim:{idx}:{key}")
    for i, (ra, rb) in enumerate(zip(a["records"], b["records"], strict=True)):
        for key in ("lrs", "updates_completed"):
            if ra[key] != rb[key]:
                bad.append(f"iter{i}:{key} {ra[key]} != {rb[key]}")
        if set(ra["metrics"]) != set(rb["metrics"]):
            bad.append(f"iter{i}:metric keys {set(ra['metrics']) ^ set(rb['metrics'])}")
        for key in set(ra["metrics"]) & set(rb["metrics"]):
            if ra["metrics"][key] != rb["metrics"][key]:
                bad.append(f"iter{i}:{key}")
    for key in ("cuda_rng", "cpu_rng"):
        if not torch.equal(a[key], b[key]):
            bad.append(key)
    return bad


def _train(tmp_path, name: str, iterations: int = 4, **kwargs):
    agent = _make_agent(log_dir=str(tmp_path / name), iterations=iterations, **kwargs)
    records, _ = _run(agent, iterations)
    snap = _snapshot(agent, records)
    engine = getattr(agent, "_captured_update_engine", None)
    agent.collector.shutdown()
    return snap, engine


# ---------------------------------------------------------------------------
# Defaults and validation
# ---------------------------------------------------------------------------


def test_capture_update_is_off_by_default():
    assert PPOConfig().capture_update is False
    assert PPOConfig().capture_update_optimizer == "exact"
    assert resolve_capture_update_optimizer("Fused") == "fused"
    with pytest.raises(ValueError, match="capture_update_optimizer"):
        resolve_capture_update_optimizer("capturable")


def test_ipmd_section_copy_of_the_flag_is_refused():
    cfg = IPMDRLOptConfig()
    cfg.ipmd.capture_update = True
    cfg.env.device = cfg.device = "cuda"
    with pytest.raises(ValueError, match=r"set ppo\.capture_update"):
        IPMD(TransformedEnv(_FakeVecEnv(4, "cuda"), StepCounter(10)), cfg)


# ---------------------------------------------------------------------------
# The exact optimizer replica on its own
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("decoupled", [True, False])
def test_exact_adam_step_matches_torch_foreach_adam(decoupled):
    gen = torch.Generator(device="cuda").manual_seed(0)
    shapes = [(257, 129), (257,), (64, 257), (8,)]
    init = [torch.randn(s, device="cuda", generator=gen) * 0.1 for s in shapes]
    ref = [torch.nn.Parameter(t.clone()) for t in init]
    mine = [t.clone() for t in init]
    opt_cls = torch.optim.AdamW if decoupled else torch.optim.Adam
    groups = [
        {"params": ref[:2], "lr": 1e-3, "weight_decay": 1e-2},
        {"params": ref[2:3], "lr": 3e-4, "weight_decay": 1e-2},
        {"params": ref[3:], "lr": 1e-3, "weight_decay": 0.0},
    ]
    opt = opt_cls(groups, betas=(0.9, 0.999), eps=1e-8)
    exp_avgs = [torch.zeros_like(t) for t in mine]
    exp_avg_sqs = [torch.zeros_like(t) for t in mine]
    slices = [slice(0, 2), slice(2, 3), slice(3, 4)]
    lr_scale = [1.0, 1.0, 1 / 1.5, 1 / 1.5**2, 1 / 1.5, 1.5e-2, 1.0, 0.3]
    for step, scale in enumerate(lr_scale, start=1):
        grads = [
            torch.randn(s, device="cuda", generator=gen) * 10.0 ** (step % 4 - 2)
            for s in shapes
        ]
        for group, base in zip(opt.param_groups, (1e-3, 3e-4, 1e-3), strict=True):
            group["lr"] = base * scale
        for p, g in zip(ref, grads, strict=True):
            p.grad = g.clone()
        opt.step()
        for group, sl in zip(opt.param_groups, slices, strict=True):
            row = exact_adam_scalars(
                group["lr"], float(step), 0.9, 0.999, group["weight_decay"]
            )
            scalars = torch.tensor(row, dtype=torch.float64).to(
                device="cuda", dtype=torch.float32
            )
            exact_adam_step(
                mine[sl],
                grads[sl],
                exp_avgs[sl],
                exp_avg_sqs[sl],
                scalars,
                beta1=0.9,
                beta2=0.999,
                eps=1e-8,
                weight_decay=group["weight_decay"],
                decoupled=decoupled,
            )
    for p, q, m, v in zip(ref, mine, exp_avgs, exp_avg_sqs, strict=True):
        assert torch.equal(p.detach(), q)
        assert torch.equal(opt.state[p]["exp_avg"], m)
        assert torch.equal(opt.state[p]["exp_avg_sq"], v)


# ---------------------------------------------------------------------------
# Captured vs eager training, bit for bit
# ---------------------------------------------------------------------------

_BITWISE_CASES = {
    "ppo_full_batch": {"kind": "ppo"},
    "ipmd_full_batch": {"kind": "ipmd"},
    "ipmd_two_epochs": {"kind": "ipmd", "epochs": 2},
    "ipmd_even_minibatches": {"kind": "ipmd", "minibatches": 4, "epochs": 2},
    "ipmd_kl_rule_per_update": {"kind": "ipmd", "kl_mode": "update"},
    "ipmd_live_normalizers": {"kind": "ipmd", "normalizer": "live"},
    "ipmd_frozen_normalizers": {"kind": "ipmd", "normalizer": "frozen"},
    "ipmd_bf16": {"kind": "ipmd", "amp": "bf16"},
    "ppo_bf16_minibatches": {
        "kind": "ppo",
        "amp": "bf16",
        "minibatches": 4,
        "epochs": 2,
    },
}


@pytest.mark.parametrize("case", sorted(_BITWISE_CASES))
def test_captured_update_is_bit_identical_to_eager(case, tmp_path):
    kwargs = _BITWISE_CASES[case]
    eager, no_engine = _train(tmp_path, "eager", capture=False, **kwargs)
    captured, engine = _train(tmp_path, "captured", capture=True, **kwargs)
    assert no_engine is None
    assert engine is not None
    assert engine.disabled_reason is None
    # One capture for the whole run: the LR moved every iteration (KL rule,
    # critic schedule) without a recapture.
    assert engine.captures == 1
    epochs = kwargs.get("epochs", 3)
    assert engine.replays == 4 * epochs * kwargs.get("minibatches", 1)
    lrs = [tuple(r["lrs"]) for r in eager["records"]]
    assert len(set(lrs)) == len(lrs), "the test must move the learning rate"
    assert _mismatches(eager, captured) == []


def test_captured_update_is_bit_identical_with_tf32_matmuls(tmp_path):
    """scripts/rlopt/train_impl.py trains with float32 matmul precision "high"."""
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        eager, _ = _train(tmp_path, "eager", kind="ipmd", capture=False)
        captured, engine = _train(tmp_path, "captured", kind="ipmd", capture=True)
    finally:
        torch.set_float32_matmul_precision(previous)
    assert engine.captures == 1
    assert _mismatches(eager, captured) == []


def test_resume_moves_step_counters_to_the_device_and_recaptures(tmp_path):
    """A checkpoint restore (map_location=cuda) puts AdamW's step counters on
    the device and replaces every optimizer-state tensor: the engine recaptures
    and keeps a host mirror of the counters instead of syncing on them."""

    def train(name: str, capture: bool):
        agent = _make_agent("ipmd", str(tmp_path / name), capture=capture, iterations=4)
        records, metadata = _run(agent, 2)
        path = tmp_path / f"{name}.pt"
        agent.save_model(path)
        agent.load_model(str(path))
        assert all(state["step"].is_cuda for state in agent.optim.state.values()), (
            "load_model is expected to map the step counters to the device"
        )
        more, _ = _run(agent, 2, metadata)
        snap = _snapshot(agent, records + more)
        engine = getattr(agent, "_captured_update_engine", None)
        agent.collector.shutdown()
        return snap, engine

    eager, _ = train("eager", capture=False)
    captured, engine = train("captured", capture=True)
    assert engine.captures == 2
    assert all(spec.device_steps for spec in engine.specs)
    assert _mismatches(eager, captured) == []


# ---------------------------------------------------------------------------
# The non-exact option and the fallbacks
# ---------------------------------------------------------------------------


def test_fused_optimizer_option_runs_close_but_not_exact(tmp_path):
    eager, _ = _train(tmp_path, "eager", kind="ipmd", capture=False)
    fused, engine = _train(
        tmp_path, "fused", kind="ipmd", capture=True, optimizer="fused"
    )
    assert engine.captures == 1
    assert engine.disabled_reason is None
    for key, value in eager["params"].items():
        torch.testing.assert_close(fused["params"][key], value, rtol=1e-4, atol=1e-5)
    for idx, state in eager["optim"].items():
        assert torch.equal(state["step"], fused["optim"][idx]["step"])


def test_uneven_minibatches_fall_back_to_eager_with_a_reason(tmp_path):
    kwargs = {"kind": "ipmd", "epochs": 2, "mini_batch_size": ROWS // 3}
    eager, _ = _train(tmp_path, "eager", capture=False, **kwargs)
    fallback, engine = _train(tmp_path, "fallback", capture=True, **kwargs)
    assert engine.captures == 0
    assert engine.replays == 0
    assert any("does not divide" in reason for reason in engine.fallback_reasons)
    assert _mismatches(eager, fallback) == []


@pytest.mark.parametrize(
    ("setting", "reason"),
    [
        ("lcp", "lcp_coeff"),
        ("compile", "compile.compile"),
        ("kl_sample", "kl_sample_size"),
        ("critic_extra", "critic_extra_epochs"),
    ],
)
def test_unsupported_configurations_run_eagerly(setting, reason, tmp_path):
    agent = _make_agent("ipmd", str(tmp_path), capture=True, iterations=1)
    if setting == "lcp":
        agent.config.ppo.lcp_coeff = 0.1
    elif setting == "compile":
        agent.config.compile.compile = True
    elif setting == "critic_extra":
        agent.config.ipmd.critic_extra_epochs = 1
    else:
        agent.config.optim.kl_sample_size = 64
    engine = agent._captured_update()
    metadata = agent.init_metadata()
    assert reason in engine._unsupported_reason(metadata)
    agent.collector.shutdown()
