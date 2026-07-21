"""Tests for IPMD algorithm components."""

from __future__ import annotations

import warnings
from functools import lru_cache
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from torchrl.data import Unbounded


@lru_cache(maxsize=1)
def _rlopt() -> SimpleNamespace:
    warnings.filterwarnings(
        "ignore",
        message="Creating .* which inherits from WeightUpdaterBase is deprecated.*",
        category=DeprecationWarning,
        append=False,
    )
    from rlopt.agent import IPMD, IPMDBilinear, IPMDBilinearRLOptConfig, IPMDRLOptConfig
    from rlopt.config_base import NetworkConfig
    from rlopt.env_utils import make_parallel_env

    return SimpleNamespace(
        IPMD=IPMD,
        IPMDBilinear=IPMDBilinear,
        IPMDBilinearRLOptConfig=IPMDBilinearRLOptConfig,
        IPMDRLOptConfig=IPMDRLOptConfig,
        NetworkConfig=NetworkConfig,
        make_parallel_env=make_parallel_env,
    )


def _apply_obs_input_keys(cfg) -> None:
    """Force observation-based keys for smoke tests across torchrl versions."""
    cfg.logger.backend = ""
    cfg.policy.input_keys = ["observation"]
    if cfg.value_function is not None:
        cfg.value_function.input_keys = ["observation"]
    cfg.ipmd.reward_input_keys = ["observation"]
    cfg.ipmd.latent_key = "observation"
    cfg.ipmd.latent_dim = 3
    cfg.ipmd.bc_coef = 0.0
    cfg.ipmd.latent_learning.method = "patch_autoencoder"
    cfg.ipmd.diversity_bonus_coeff = 0.0


def _apply_nonlatent_obs_input_keys(cfg) -> None:
    """Configure observation-only IPMD tests that do not exercise latent commands."""
    cfg.logger.backend = ""
    cfg.policy.input_keys = ["observation"]
    if cfg.value_function is not None:
        cfg.value_function.input_keys = ["observation"]
    cfg.ipmd.use_latent_command = False
    cfg.ipmd.reward_input_keys = ["observation"]
    cfg.ipmd.bc_coef = 0.0
    cfg.ipmd.diversity_bonus_coeff = 0.0


def _make_test_expert_sampler(expert_data: TensorDict):
    """Return a private expert sampler override matching the new agent contract."""

    def _sample(batch_size: int, required_keys):
        batch = expert_data
        if batch.numel() > batch_size:
            batch = batch[:batch_size]
        return batch.select(*required_keys).clone()

    return _sample


def create_synthetic_expert_data(env, num_transitions: int = 100) -> TensorDict:
    """Create synthetic expert demonstration data by random sampling."""
    obs_dim = env.observation_spec["observation"].shape[-1]
    act_dim = env.action_spec.shape[-1]

    return TensorDict(
        {
            "observation": torch.randn(num_transitions, obs_dim),
            "action": torch.randn(num_transitions, act_dim),
            ("next", "observation"): torch.randn(num_transitions, obs_dim),
            "reward": torch.randn(num_transitions),
            "done": torch.zeros(num_transitions, dtype=torch.bool),
            "terminated": torch.zeros(num_transitions, dtype=torch.bool),
            "truncated": torch.zeros(num_transitions, dtype=torch.bool),
        },
        batch_size=[num_transitions],
    )


def _make_env_reward_only_ipmd_agent(*, split_actor_critic_lr: bool = False):
    rlopt = _rlopt()
    cfg = rlopt.IPMDRLOptConfig()
    cfg.env.env_name = "Pendulum-v1"
    cfg.env.device = "cpu"
    cfg.device = "cpu"
    cfg.collector.frames_per_batch = 4
    cfg.collector.total_frames = 4
    cfg.replay_buffer.size = 64
    cfg.loss.mini_batch_size = 2
    cfg.compile.compile = False
    cfg.logger.backend = ""
    cfg.policy.input_keys = ["observation"]
    if cfg.value_function is not None:
        cfg.value_function.input_keys = ["observation"]
    cfg.ipmd.use_latent_command = False
    cfg.ipmd.reward_input_keys = ["observation"]
    cfg.ipmd.reward_loss_coeff = 0.0
    cfg.ipmd.reward_l2_coeff = 0.0
    cfg.ipmd.reward_grad_penalty_coeff = 0.0
    cfg.ipmd.bc_coef = 0.0
    cfg.ipmd.diversity_bonus_coeff = 0.0
    if split_actor_critic_lr:
        cfg.ipmd.actor_learning_rate = 2.0e-5
        cfg.ipmd.critic_learning_rate = 1.0e-3
        cfg.optim.scheduler = "adaptive"
        cfg.optim.min_lr = 1.0e-5
        cfg.optim.max_lr = 2.0e-4

    env = rlopt.make_parallel_env(cfg)
    return rlopt.IPMD(env, cfg, logger=None), env


def test_ipmd_split_actor_critic_learning_rates_and_adaptation():
    agent, env = _make_env_reward_only_ipmd_agent(split_actor_critic_lr=True)
    try:
        groups = {group["name"]: group for group in agent.optim.param_groups}
        assert groups["actor"]["lr"] == pytest.approx(2.0e-5)
        assert groups["critic"]["lr"] == pytest.approx(1.0e-3)

        agent._maybe_adjust_lr(torch.tensor(1.0e-4), agent.config.optim)

        assert groups["actor"]["lr"] == pytest.approx(3.0e-5)
        assert groups["critic"]["lr"] == pytest.approx(1.0e-3)
    finally:
        env.close()


def test_ipmd_split_learning_rates_must_be_configured_together():
    cfg = _rlopt().IPMDRLOptConfig()
    cfg.ipmd.actor_learning_rate = 2.0e-5
    with pytest.raises(ValueError, match="must be configured together"):
        cfg.ipmd.validate()


def test_silu_activation_is_available():
    from rlopt.utils import get_activation_class

    assert get_activation_class("silu") is torch.nn.SiLU
    assert get_activation_class("swish") is torch.nn.SiLU


def test_sonic_running_mean_std_normalizes_concatenated_critic_input():
    from rlopt.agent.ppo.ppo import RunningMeanStdCatInputs

    normalizer = RunningMeanStdCatInputs(
        torch.nn.Identity(),
        feature_dim=2,
        epsilon=1.0e-5,
        clip=5.0,
    )
    first = torch.tensor([[1.0], [3.0]])
    second = torch.tensor([[3.0], [7.0]])
    output = normalizer(first, second)

    torch.testing.assert_close(
        output,
        torch.tensor([[1.0, 3.0], [3.0, 5.0]]),
        atol=6.0e-5,
        rtol=0.0,
    )
    assert normalizer.count.item() == pytest.approx(3.0)

    running_mean = normalizer.running_mean.clone()
    running_var = normalizer.running_var.clone()
    normalizer.eval()
    normalized = normalizer(torch.tensor([[2.0]]), torch.tensor([[5.0]]))
    torch.testing.assert_close(
        normalized,
        (torch.tensor([[2.0, 5.0]]) - running_mean) / torch.sqrt(running_var + 1.0e-5),
    )
    torch.testing.assert_close(normalizer.running_mean, running_mean)
    torch.testing.assert_close(normalizer.running_var, running_var)

    vmapped = torch.vmap(normalizer)(torch.tensor([[[2.0, 5.0]], [[3.0, 7.0]]]))
    assert vmapped.shape == (2, 1, 2)
    torch.testing.assert_close(normalizer.running_mean, running_mean)
    torch.testing.assert_close(normalizer.running_var, running_var)


def test_sonic_advantage_normalization_uses_the_complete_rollout():
    from rlopt.agent.ppo.ppo import _normalize_advantage_over_rollout

    advantage = torch.arange(6, dtype=torch.float32).reshape(2, 3, 1)
    normalized = _normalize_advantage_over_rollout(advantage)

    torch.testing.assert_close(normalized.mean(dim=(0, 1)), torch.zeros(1))
    torch.testing.assert_close(normalized.std(dim=(0, 1)), torch.ones(1))


def _make_bilinear_offline_cfg():
    rlopt = _rlopt()
    cfg = rlopt.IPMDBilinearRLOptConfig()
    cfg.env.env_name = "Pendulum-v1"
    cfg.env.device = "cpu"
    cfg.device = "cpu"
    cfg.collector.frames_per_batch = 4
    cfg.collector.total_frames = 4
    cfg.replay_buffer.size = 64
    cfg.loss.mini_batch_size = 2
    cfg.compile.compile = False
    cfg.logger.backend = ""
    cfg.policy.input_keys = ["observation"]
    cfg.policy.num_cells = [16]
    if cfg.value_function is not None:
        cfg.value_function.input_keys = ["observation"]
        cfg.value_function.num_cells = [16]
    cfg.ipmd.use_latent_command = False
    cfg.ipmd.reward_input_keys = ["observation"]
    cfg.ipmd.reward_num_cells = (16,)
    cfg.ipmd.reward_loss_coeff = 0.0
    cfg.ipmd.reward_l2_coeff = 0.0
    cfg.ipmd.reward_grad_penalty_coeff = 0.0
    cfg.ipmd.bc_coef = 0.0
    cfg.ipmd.diversity_bonus_coeff = 0.0
    cfg.bilinear.obs_keys = ["observation"]
    cfg.bilinear.next_obs_keys = ["observation"]
    cfg.bilinear.feature_dim = 3
    cfg.bilinear.embed_dim = 8
    cfg.bilinear.g_hidden_dims = (16,)
    cfg.bilinear.mu_hidden_dims = (16,)
    cfg.bilinear.sr_batch_size = 4
    cfg.bilinear.history_buffer_size = 64
    cfg.bilinear.num_noises = 2
    cfg.bilinear.sample_eval_interval = 0
    cfg.bilinear.offline_pretrain.enabled = True
    cfg.bilinear.offline_pretrain.num_updates = 2
    cfg.bilinear.offline_pretrain.batch_size = 8
    cfg.bilinear.offline_pretrain.log_interval = 1
    return cfg


class _BilinearTestPolicyHead(torch.nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.loc = torch.nn.Linear(obs_dim, action_dim)
        self.log_std = torch.nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        loc = self.loc(obs)
        scale = self.log_std.exp().expand_as(loc)
        return loc, scale


def _make_bilinear_test_policy_head(env) -> _BilinearTestPolicyHead:
    obs_dim = env.observation_spec["observation"].shape[-1]
    action_dim = env.action_spec.shape[-1]
    return _BilinearTestPolicyHead(obs_dim, action_dim)


def _make_bilinear_expert_batch(env, num_transitions: int = 32) -> TensorDict:
    obs_dim = env.observation_spec["observation"].shape[-1]
    act_dim = env.action_spec.shape[-1]
    return TensorDict(
        {
            "observation": torch.randn(num_transitions, obs_dim),
            ("next", "observation"): torch.randn(num_transitions, obs_dim),
            "expert_action": torch.tanh(torch.randn(num_transitions, act_dim)),
        },
        batch_size=[num_transitions],
    )


def test_bilinear_policy_head_splits_command_from_representation_state() -> None:
    """Policy command keys should not be concatenated into the SR state input."""
    _rlopt()
    from rlopt.agent.ipmd.ipmd_bilinear import BilinearPolicyHead
    from rlopt.agent.ipmd.module import build_bilinear_sr

    bilinear_rep = build_bilinear_sr(
        "diffsr",
        obs_dim=5,
        next_obs_dim=4,
        action_dim=2,
        feature_dim=3,
        embed_dim=4,
        g_hidden_dims=(8,),
        mu_hidden_dims=(8,),
        num_noises=2,
        use_ema_for_policy=True,
        device="cpu",
    )
    head = BilinearPolicyHead(
        bilinear_rep=bilinear_rep,
        num_cells=[8],
        activation_fn="elu",
        action_dim=2,
        num_command_inputs=2,
        command_dim=7,
        device="cpu",
    )

    loc, scale = head(
        torch.randn(6, 2),
        torch.randn(6, 5),
        torch.randn(6, 5),
    )

    assert loc.shape == (6, 2)
    assert scale.shape == (6, 2)
    assert head.base[0].in_features == 7

    raw_state_head = BilinearPolicyHead(
        bilinear_rep=bilinear_rep,
        num_cells=[8],
        activation_fn="elu",
        action_dim=2,
        num_command_inputs=2,
        command_dim=7,
        include_raw_state=True,
        device="cpu",
    )
    loc, scale = raw_state_head(
        torch.randn(6, 2),
        torch.randn(6, 5),
        torch.randn(6, 5),
    )

    assert loc.shape == (6, 2)
    assert scale.shape == (6, 2)
    assert raw_state_head.base[0].in_features == 12


def test_ipmd_prepare_rollout_rewards_ignores_estimator_when_disabled():
    """Disabled estimated PPO rewards should leave env rewards untouched."""
    agent, env = _make_env_reward_only_ipmd_agent()
    obs_dim = env.observation_spec["observation"].shape[-1]
    act_dim = env.action_spec.shape[-1]
    env_reward = torch.tensor([[1.0], [2.0]], device=agent.device)
    rollout = TensorDict(
        {
            "observation": torch.randn(2, obs_dim, device=agent.device),
            "action": torch.randn(2, act_dim, device=agent.device),
            ("next", "observation"): torch.randn(2, obs_dim, device=agent.device),
            ("next", "reward"): env_reward.clone(),
            ("next", "done"): torch.zeros(2, 1, dtype=torch.bool, device=agent.device),
            ("next", "truncated"): torch.zeros(
                2, 1, dtype=torch.bool, device=agent.device
            ),
        },
        batch_size=[2],
        device=agent.device,
    )

    def _unexpected_reward_forward(*_args, **_kwargs):
        msg = "reward estimator should not run"
        raise AssertionError(msg)

    agent._reward_from_td = _unexpected_reward_forward

    metrics = agent._prepare_rollout_rewards(rollout)

    assert torch.equal(rollout.get(("next", "reward")), env_reward)
    assert ("next", "est_reward") not in rollout.keys(True)
    assert "train/est_reward_mean" not in metrics


def test_ipmd_disabled_reward_update_disables_expert_minibatch_path():
    """Pure-PPO IPMD configs should skip the expert minibatch path entirely."""
    agent, _ = _make_env_reward_only_ipmd_agent()
    assert agent._expert_minibatch_update_enabled is False


def test_ipmd_bilinear_offline_pretrain_requires_expert_sampler() -> None:
    """Offline SR pretraining should fail at construction without a sampler."""
    rlopt = _rlopt()
    cfg = _make_bilinear_offline_cfg()
    env = rlopt.make_parallel_env(cfg)

    with pytest.raises(ValueError, match="sample_expert_batch"):
        rlopt.IPMDBilinear(
            env,
            cfg,
            policy_net=_make_bilinear_test_policy_head(env),
            logger=None,
        )


def test_ipmd_bilinear_offline_pretrain_validates_config() -> None:
    """Offline SR pretraining validates positive update, batch, and log values."""
    rlopt = _rlopt()
    cfg = _make_bilinear_offline_cfg()
    cfg.bilinear.offline_pretrain.batch_size = 0
    env = rlopt.make_parallel_env(cfg)
    env.sample_expert_batch = lambda batch_size, _required_keys: TensorDict(
        {},
        batch_size=[batch_size],
    )

    with pytest.raises(ValueError, match=r"offline_pretrain\.batch_size"):
        rlopt.IPMDBilinear(
            env,
            cfg,
            policy_net=_make_bilinear_test_policy_head(env),
            logger=None,
        )


def test_ipmd_bilinear_offline_pretrain_constructs_default_policy_head() -> None:
    """Offline preflight should work with the production bilinear policy head."""
    rlopt = _rlopt()
    cfg = _make_bilinear_offline_cfg()
    env = rlopt.make_parallel_env(cfg)
    expert_data = _make_bilinear_expert_batch(env)
    env.sample_expert_batch = _make_test_expert_sampler(expert_data)

    agent = rlopt.IPMDBilinear(env, cfg, logger=None)

    assert agent.policy is not None
    assert len(agent._sr_history_buffer) == 0


def test_ipmd_bilinear_offline_pretrain_uses_expert_action_and_next_obs() -> None:
    """Offline SR pretraining should request explicit next-state keys and expert actions."""
    rlopt = _rlopt()
    cfg = _make_bilinear_offline_cfg()
    env = rlopt.make_parallel_env(cfg)
    expert_data = _make_bilinear_expert_batch(env)
    requested: list[list[str | tuple[str, ...]]] = []

    def _sample_expert_batch(batch_size: int, required_keys):
        requested.append(list(required_keys))
        return expert_data[:batch_size].select(*required_keys).clone()

    env.sample_expert_batch = _sample_expert_batch
    agent = rlopt.IPMDBilinear(
        env,
        cfg,
        policy_net=_make_bilinear_test_policy_head(env),
        logger=None,
    )

    sr_before = {
        name: param.detach().clone()
        for name, param in agent.bilinear_rep.named_parameters()
    }
    non_sr_actor_before = {
        name: param.detach().clone()
        for name, param in agent.actor_critic.named_parameters()
        if "bilinear_rep" not in name
    }
    reward_before = {
        name: param.detach().clone()
        for name, param in agent.reward_estimator.named_parameters()
    }

    agent._offline_pretrain_spectral_representation()

    assert requested
    assert requested[0] == ["observation", ("next", "observation"), "expert_action"]
    assert len(agent._sr_history_buffer) == (
        cfg.bilinear.offline_pretrain.num_updates
        * cfg.bilinear.offline_pretrain.batch_size
    )
    assert agent.bilinear_rep.obs_norm.count.item() == len(agent._sr_history_buffer)
    assert any(
        not torch.allclose(param, sr_before[name])
        for name, param in agent.bilinear_rep.named_parameters()
    )
    for name, param in agent.actor_critic.named_parameters():
        if "bilinear_rep" not in name:
            assert torch.allclose(param, non_sr_actor_before[name])
    for name, param in agent.reward_estimator.named_parameters():
        assert torch.allclose(param, reward_before[name])
    assert agent.bilinear_rep.state_net_ema is not None
    for online, ema in zip(
        agent.bilinear_rep.state_net.parameters(),
        agent.bilinear_rep.state_net_ema.parameters(),
        strict=True,
    ):
        assert torch.allclose(online, ema)


def test_ipmd_bilinear_offline_policy_bc_updates_actor_after_sr_pretrain() -> None:
    """Offline policy BC should update the same feature-only actor used online."""
    rlopt = _rlopt()
    cfg = _make_bilinear_offline_cfg()
    cfg.bilinear.policy_include_raw_state = False
    cfg.bilinear.offline_pretrain.policy_bc_updates = 2
    cfg.bilinear.offline_pretrain.policy_bc_batch_size = 8
    env = rlopt.make_parallel_env(cfg)
    expert_data = _make_bilinear_expert_batch(env)
    requested: list[list[str | tuple[str, ...]]] = []

    def _sample_expert_batch(batch_size: int, required_keys):
        requested.append(list(required_keys))
        return expert_data[:batch_size].select(*required_keys).clone()

    env.sample_expert_batch = _sample_expert_batch
    agent = rlopt.IPMDBilinear(env, cfg, logger=None)
    non_sr_actor_before = {
        name: param.detach().clone()
        for name, param in agent.actor_critic.named_parameters()
        if "bilinear_rep" not in name
    }

    agent._offline_pretrain_spectral_representation()

    assert requested
    assert ["observation", ("next", "observation"), "expert_action"] in requested
    assert any(
        not torch.allclose(param, non_sr_actor_before[name])
        for name, param in agent.actor_critic.named_parameters()
        if "bilinear_rep" not in name
    )


def test_ipmd_bilinear_offline_pretrain_handles_distinct_next_obs_dim() -> None:
    """Offline SR pretraining should support s and s' with different feature widths."""
    rlopt = _rlopt()
    cfg = _make_bilinear_offline_cfg()
    cfg.bilinear.next_obs_keys = ["next_observation"]
    cfg.bilinear.sample_eval_interval = 1
    env = rlopt.make_parallel_env(cfg)
    next_obs_dim = 2
    env.output_spec.unlock_()
    env.output_spec["full_observation_spec", "next_observation"] = Unbounded(
        shape=(*env.batch_size, next_obs_dim),
        device="cpu",
    )
    env.output_spec.lock_()
    obs_dim = env.observation_spec["observation"].shape[-1]
    act_dim = env.action_spec.shape[-1]
    expert_data = TensorDict(
        {
            "observation": torch.randn(32, obs_dim),
            ("next", "next_observation"): torch.randn(32, next_obs_dim),
            "expert_action": torch.randn(32, act_dim),
        },
        batch_size=[32],
    )
    env.sample_expert_batch = _make_test_expert_sampler(expert_data)
    agent = rlopt.IPMDBilinear(
        env,
        cfg,
        policy_net=_make_bilinear_test_policy_head(env),
        logger=None,
    )

    agent._offline_pretrain_spectral_representation()

    assert len(agent._sr_history_buffer) == (
        cfg.bilinear.offline_pretrain.num_updates
        * cfg.bilinear.offline_pretrain.batch_size
    )


def test_ipmd_bilinear_offline_missing_required_keys_fails_at_construction() -> None:
    """A malformed expert batch should fail during offline pretrain preflight."""
    rlopt = _rlopt()
    cfg = _make_bilinear_offline_cfg()
    env = rlopt.make_parallel_env(cfg)
    expert_data = _make_bilinear_expert_batch(env)

    def _sample_expert_batch(batch_size: int, required_keys):
        del required_keys
        return expert_data[:batch_size].select("observation").clone()

    env.sample_expert_batch = _sample_expert_batch

    with pytest.raises(RuntimeError, match="missing required keys"):
        rlopt.IPMDBilinear(
            env,
            cfg,
            policy_net=_make_bilinear_test_policy_head(env),
            logger=None,
        )


def test_ipmd_prepare_rollout_rewards_honors_estimated_reward_gate():
    """Estimated PPO rewards should be mixed only when the gate is enabled."""
    agent, env = _make_env_reward_only_ipmd_agent()
    agent.config.ipmd.use_estimated_rewards_for_ppo = True
    agent._use_estimated_rewards_for_ppo = True
    agent.config.ipmd.env_reward_weight = 0.25
    agent.config.ipmd.est_reward_weight = 0.75
    agent.config.ipmd.estimated_reward_done_penalty = 0.5

    obs_dim = env.observation_spec["observation"].shape[-1]
    act_dim = env.action_spec.shape[-1]
    rollout = TensorDict(
        {
            "observation": torch.randn(2, obs_dim, device=agent.device),
            "action": torch.randn(2, act_dim, device=agent.device),
            ("next", "observation"): torch.randn(2, obs_dim, device=agent.device),
            ("next", "reward"): torch.ones(2, 1, device=agent.device),
            ("next", "done"): torch.ones(2, 1, dtype=torch.bool, device=agent.device),
            ("next", "truncated"): torch.tensor([[False], [True]], device=agent.device),
        },
        batch_size=[2],
        device=agent.device,
    )

    def _fixed_reward_forward(*_args, **_kwargs):
        return torch.full((2, 1), 2.0, device=agent.device)

    agent._reward_from_td = _fixed_reward_forward

    metrics = agent._prepare_rollout_rewards(rollout)

    expected_est_reward = torch.tensor([[1.5], [2.0]], device=agent.device)
    expected_reward = (
        0.25 * torch.ones_like(expected_est_reward) + 0.75 * expected_est_reward
    )
    assert torch.allclose(rollout.get(("next", "est_reward")), expected_est_reward)
    assert torch.allclose(rollout.get(("next", "reward")), expected_reward)
    assert metrics["train/est_reward_mean"] == pytest.approx(1.75)


def test_ipmd_initialization():
    """Test IPMD agent initialization."""
    rlopt = _rlopt()
    cfg = rlopt.IPMDRLOptConfig()
    cfg.env.env_name = "Pendulum-v1"
    cfg.env.device = "cpu"
    cfg.device = "cpu"
    cfg.collector.frames_per_batch = 4
    cfg.collector.total_frames = 4
    cfg.collector.init_random_frames = 0
    cfg.replay_buffer.size = 64
    cfg.loss.mini_batch_size = 2
    cfg.compile.compile = False
    cfg.ipmd.reward_num_cells = (64, 64)
    _apply_nonlatent_obs_input_keys(cfg)
    cfg.q_function = rlopt.NetworkConfig(
        num_cells=[64, 64],
        activation_fn="relu",
        output_dim=1,
        input_keys=["observation"],
    )

    env = rlopt.make_parallel_env(cfg)
    agent = rlopt.IPMD(env, cfg, logger=None)

    assert agent is not None
    assert agent.__class__.__name__ == "IPMD"
    assert hasattr(agent, "reward_estimator")
    assert hasattr(agent, "actor_critic")
    assert hasattr(agent, "loss_module")


def test_ipmd_latent_mode_owns_controller_without_mixin_inheritance():
    """Latent-enabled IPMD should own a controller instead of inheriting behavior."""
    rlopt = _rlopt()
    cfg = rlopt.IPMDRLOptConfig()
    cfg.env.env_name = "Pendulum-v1"
    cfg.env.device = "cpu"
    cfg.device = "cpu"
    cfg.collector.frames_per_batch = 4
    cfg.collector.total_frames = 4
    cfg.collector.init_random_frames = 0
    cfg.replay_buffer.size = 64
    cfg.loss.mini_batch_size = 2
    cfg.compile.compile = False
    _apply_obs_input_keys(cfg)

    env = rlopt.make_parallel_env(cfg)
    agent = rlopt.IPMD(env, cfg, logger=None)

    from rlopt.agent.imitation.latent_commands import (
        LatentCommandController,
        LatentCommandMixin,
    )

    assert isinstance(agent._latent_command_controller, LatentCommandController)
    assert not isinstance(agent, LatentCommandMixin)
    assert all(block.kind != "latent" for block in agent._reward_input_blocks)


def test_ipmd_checkpoint_cadence_crosses_sample_boundary_once() -> None:
    """Checkpoint cadence should trigger when an iteration crosses save_interval."""
    agent, _ = _make_env_reward_only_ipmd_agent()
    agent.config.save_interval = 100

    assert not agent._should_save_checkpoint(
        frames_processed=99,
        frames_in_iteration=4,
    )
    assert agent._should_save_checkpoint(
        frames_processed=100,
        frames_in_iteration=4,
    )
    assert not agent._should_save_checkpoint(
        frames_processed=120,
        frames_in_iteration=20,
    )
    assert agent._should_save_checkpoint(
        frames_processed=205,
        frames_in_iteration=90,
    )


def test_ipmd_reward_estimator():
    """Test IPMD reward estimator network."""
    rlopt = _rlopt()
    cfg = rlopt.IPMDRLOptConfig()
    cfg.env.env_name = "Pendulum-v1"
    cfg.env.device = "cpu"
    cfg.device = "cpu"
    cfg.collector.frames_per_batch = 4
    cfg.collector.total_frames = 4
    cfg.replay_buffer.size = 64
    cfg.loss.mini_batch_size = 2
    cfg.compile.compile = False
    cfg.ipmd.reward_num_cells = (32, 32)
    _apply_nonlatent_obs_input_keys(cfg)

    env = rlopt.make_parallel_env(cfg)
    agent = rlopt.IPMD(env, cfg, logger=None)

    # Create a sample batch
    obs_dim = env.observation_spec["observation"].shape[-1]
    act_dim = env.action_spec.shape[-1]
    batch_size = 8

    td = TensorDict(
        {
            "observation": torch.randn(batch_size, obs_dim),
            "action": torch.randn(batch_size, act_dim),
            ("next", "observation"): torch.randn(batch_size, obs_dim),
        },
        batch_size=[batch_size],
    )

    # Compute estimated rewards
    rewards = agent._reward_from_td(td, batch_role="rollout")
    assert rewards.shape[0] == batch_size
    assert not torch.isnan(rewards).any()
    assert not torch.isinf(rewards).any()


def test_ipmd_reward_input_spec_matches_reward_mode() -> None:
    """Reward-network shape and required keys should come from one shared spec."""
    rlopt = _rlopt()
    cfg = rlopt.IPMDRLOptConfig()
    cfg.env.env_name = "Pendulum-v1"
    cfg.env.device = "cpu"
    cfg.device = "cpu"
    cfg.collector.frames_per_batch = 4
    cfg.collector.total_frames = 4
    cfg.replay_buffer.size = 64
    cfg.loss.mini_batch_size = 2
    cfg.compile.compile = False
    cfg.logger.backend = ""
    cfg.policy.input_keys = ["observation"]
    if cfg.value_function is not None:
        cfg.value_function.input_keys = ["observation"]
    cfg.ipmd.use_latent_command = False
    cfg.ipmd.reward_input_keys = ["observation"]
    cfg.ipmd.reward_input_type = "sa"
    cfg.ipmd.reward_loss_coeff = 0.0
    cfg.ipmd.reward_l2_coeff = 0.0
    cfg.ipmd.reward_grad_penalty_coeff = 0.0
    cfg.ipmd.bc_coef = 0.0

    env = rlopt.make_parallel_env(cfg)
    agent = rlopt.IPMD(env, cfg, logger=None)

    obs_dim = env.observation_spec["observation"].shape[-1]
    act_dim = env.action_spec.shape[-1]
    assert [block.kind for block in agent._reward_input_blocks] == ["obs", "action"]
    assert agent._reward_input_dim == obs_dim + act_dim
    assert agent._reward_required_batch_keys() == ["observation", "action"]


def test_ipmd_grouped_reward_model_builds_heads_from_reward_keys() -> None:
    """Grouped mode should infer reference context and one head per remaining key."""
    rlopt = _rlopt()
    cfg = rlopt.IPMDRLOptConfig()
    cfg.env.env_name = "Pendulum-v1"
    cfg.env.device = "cpu"
    cfg.device = "cpu"
    cfg.collector.frames_per_batch = 4
    cfg.collector.total_frames = 4
    cfg.replay_buffer.size = 64
    cfg.loss.mini_batch_size = 2
    cfg.compile.compile = False
    cfg.logger.backend = ""
    cfg.policy.input_keys = ["observation"]
    if cfg.value_function is not None:
        cfg.value_function.input_keys = ["observation"]
    cfg.ipmd.use_latent_command = False
    cfg.ipmd.reward_model_type = "grouped"
    cfg.ipmd.reward_input_type = "s"
    cfg.ipmd.reward_input_keys = [
        ("reward_state", "reference_command"),
        ("reward_state", "joint_pos"),
        ("reward_state", "joint_vel"),
    ]
    cfg.ipmd.reward_group_head_weights = [1.0, 0.5]
    cfg.ipmd.reward_loss_coeff = 0.0
    cfg.ipmd.reward_l2_coeff = 0.0
    cfg.ipmd.reward_grad_penalty_coeff = 0.0
    cfg.ipmd.bc_coef = 0.0

    env = rlopt.make_parallel_env(cfg)
    spec = env.observation_spec.clone()
    for key in cfg.ipmd.reward_input_keys:
        spec.set(key, spec["observation"].clone())
    env.observation_spec = spec
    agent = rlopt.IPMD(env, cfg, logger=None)

    assert isinstance(agent.reward_estimator, torch.nn.ModuleDict)
    assert agent._reward_group_context_keys == (("reward_state", "reference_command"),)
    assert [spec.name for spec in agent._reward_group_head_specs] == [
        "joint_pos",
        "joint_vel",
    ]
    assert [spec.weight for spec in agent._reward_group_head_specs] == [1.0, 0.5]

    obs_dim = env.observation_spec["observation"].shape[-1]
    batch_size = 5
    td = TensorDict({}, batch_size=[batch_size], device=agent.device)
    for key in cfg.ipmd.reward_input_keys:
        td.set(key, torch.randn(batch_size, obs_dim, device=agent.device))

    rewards = agent._reward_from_td(td, batch_role="rollout")
    assert rewards.shape == (batch_size, 1)
    assert not torch.isnan(rewards).any()


def _make_latent_grouped_reward_agent():
    rlopt = _rlopt()
    cfg = rlopt.IPMDRLOptConfig()
    cfg.env.env_name = "Pendulum-v1"
    cfg.env.device = "cpu"
    cfg.device = "cpu"
    cfg.collector.frames_per_batch = 4
    cfg.collector.total_frames = 4
    cfg.replay_buffer.size = 64
    cfg.loss.mini_batch_size = 2
    cfg.compile.compile = False
    cfg.logger.backend = ""
    cfg.policy.input_keys = [("command", "policy_command")]
    if cfg.value_function is not None:
        cfg.value_function.input_keys = ["observation"]
    cfg.ipmd.use_latent_command = True
    cfg.ipmd.command_source = "posterior"
    cfg.ipmd.latent_key = ("command", "policy_command")
    cfg.ipmd.latent_dim = 3
    cfg.ipmd.latent_learning.method = "patch_autoencoder"
    cfg.ipmd.latent_learning.posterior_input_keys = ["observation"]
    cfg.ipmd.latent_learning.recon_coeff = 0.0
    cfg.ipmd.reward_model_type = "grouped"
    cfg.ipmd.reward_input_type = "s"
    cfg.ipmd.reward_input_keys = [
        ("command", "policy_command"),
        ("reward_state", "joint_pos"),
    ]
    cfg.ipmd.reward_group_context_keys = [("command", "policy_command")]
    cfg.ipmd.reward_group_head_weights = [1.0]
    cfg.ipmd.reward_loss_coeff = 1.0
    cfg.ipmd.reward_l2_coeff = 0.0
    cfg.ipmd.reward_grad_penalty_coeff = 0.0
    cfg.ipmd.reward_logit_reg_coeff = 0.0
    cfg.ipmd.bc_coef = 0.0
    cfg.ipmd.diversity_bonus_coeff = 0.0

    env = rlopt.make_parallel_env(cfg)
    spec = env.observation_spec.clone()
    spec.set(("command", "policy_command"), spec["observation"].clone())
    spec.set(("reward_state", "joint_pos"), spec["observation"].clone())
    env.observation_spec = spec
    return rlopt.IPMD(env, cfg, logger=None), env


def test_ipmd_latent_grouped_reward_context_is_not_a_head() -> None:
    """Latent reward context should resolve as shared grouped context only."""
    agent, _env = _make_latent_grouped_reward_agent()

    assert agent._reward_infers_latent_condition is True
    assert agent._reward_group_context_keys == (("command", "policy_command"),)
    assert [spec.obs_key for spec in agent._reward_group_head_specs] == [
        ("reward_state", "joint_pos")
    ]


def test_ipmd_grouped_reward_context_normalizes_hydra_list_keys() -> None:
    """Hydra list materialization should not break grouped context matching."""
    rlopt = _rlopt()
    cfg = rlopt.IPMDRLOptConfig()
    cfg.env.env_name = "Pendulum-v1"
    cfg.env.device = "cpu"
    cfg.device = "cpu"
    cfg.collector.frames_per_batch = 4
    cfg.collector.total_frames = 4
    cfg.replay_buffer.size = 64
    cfg.loss.mini_batch_size = 2
    cfg.compile.compile = False
    cfg.logger.backend = ""
    cfg.policy.input_keys = ["observation"]
    if cfg.value_function is not None:
        cfg.value_function.input_keys = ["observation"]
    cfg.ipmd.use_latent_command = False
    cfg.ipmd.reward_model_type = "grouped"
    cfg.ipmd.reward_input_type = "s"
    cfg.ipmd.reward_input_keys = [
        ("command", "policy_command"),
        ("reward_state", "joint_pos"),
    ]
    cfg.ipmd.reward_group_context_keys = [["command", "policy_command"]]
    cfg.ipmd.reward_group_head_weights = [1.0]
    cfg.ipmd.bc_coef = 0.0
    cfg.ipmd.diversity_bonus_coeff = 0.0

    env = rlopt.make_parallel_env(cfg)
    spec = env.observation_spec.clone()
    spec.set(("command", "policy_command"), spec["observation"].clone())
    spec.set(("reward_state", "joint_pos"), spec["observation"].clone())
    env.observation_spec = spec
    agent = rlopt.IPMD(env, cfg, logger=None)

    assert agent._reward_group_context_keys == (("command", "policy_command"),)
    assert [spec.obs_key for spec in agent._reward_group_head_specs] == [
        ("reward_state", "joint_pos")
    ]


def test_ipmd_latent_reward_expert_sampler_uses_posterior_not_command() -> None:
    """Expert reward batches should request posterior inputs and inject command latents."""
    agent, env = _make_latent_grouped_reward_agent()

    class _IdentityLatent(torch.nn.Module):
        def forward(self, obs_features: torch.Tensor) -> torch.Tensor:
            return obs_features

    assert agent._latent_learner is not None
    agent._latent_learner.encoder = _IdentityLatent().to(agent.device)
    obs_dim = env.observation_spec["observation"].shape[-1]
    batch_size = 2
    expert_obs = torch.randn(batch_size, obs_dim, device=agent.device)
    expert_data = TensorDict(
        {
            "observation": expert_obs.clone(),
            ("reward_state", "joint_pos"): torch.randn(
                batch_size, obs_dim, device=agent.device
            ),
        },
        batch_size=[batch_size],
        device=agent.device,
    )
    requested_keys: list[list] = []

    def _sample(sample_batch_size: int, required_keys):
        requested_keys.append(list(required_keys))
        batch = expert_data[:sample_batch_size]
        return batch.select(*required_keys).clone()

    agent._set_test_expert_batch_sampler(_sample)
    policy_obs = torch.randn(batch_size, obs_dim, device=agent.device)
    policy_batch = TensorDict(
        {
            "observation": policy_obs.clone(),
            ("reward_state", "joint_pos"): torch.randn(
                batch_size, obs_dim, device=agent.device
            ),
        },
        batch_size=[batch_size],
        device=agent.device,
    )

    expert_batch, _has_expert = agent._expert_batch_for_update(policy_batch)
    assert requested_keys == [[("reward_state", "joint_pos"), "observation"]]
    assert ("command", "policy_command") not in expert_batch.keys(True)

    metrics = agent._run_reward_updates(
        policy_batch,
        expert_batch,
        policy_update_idx=0,
    )

    assert torch.allclose(policy_batch.get(("command", "policy_command")), policy_obs)
    assert torch.allclose(expert_batch.get(("command", "policy_command")), expert_obs)
    assert torch.isfinite(metrics["reward_condition_policy_norm"])
    assert torch.isfinite(metrics["reward_condition_expert_norm"])


def test_ipmd_pre_policy_reward_updates_preserve_rollout_command() -> None:
    """Reward-training latent recompute must not overwrite stored rollout commands."""
    from rlopt.agent.ppo.ppo import PPOTrainingMetadata

    agent, env = _make_latent_grouped_reward_agent()

    class _IdentityLatent(torch.nn.Module):
        def forward(self, obs_features: torch.Tensor) -> torch.Tensor:
            return obs_features

    assert agent._latent_learner is not None
    agent._latent_learner.encoder = _IdentityLatent().to(agent.device)
    obs_dim = env.observation_spec["observation"].shape[-1]
    batch_size = 2
    expert_data = TensorDict(
        {
            "observation": torch.randn(batch_size, obs_dim, device=agent.device),
            ("reward_state", "joint_pos"): torch.randn(
                batch_size, obs_dim, device=agent.device
            ),
        },
        batch_size=[batch_size],
        device=agent.device,
    )

    def _sample(sample_batch_size: int, required_keys):
        batch = expert_data[:sample_batch_size]
        return batch.select(*required_keys).clone()

    agent._set_test_expert_batch_sampler(_sample)
    stored_command = torch.randn(batch_size, obs_dim, device=agent.device)
    rollout_flat = TensorDict(
        {
            "observation": torch.randn(batch_size, obs_dim, device=agent.device),
            ("command", "policy_command"): stored_command.clone(),
            ("reward_state", "joint_pos"): torch.randn(
                batch_size, obs_dim, device=agent.device
            ),
        },
        batch_size=[batch_size],
        device=agent.device,
    )
    metadata = PPOTrainingMetadata(
        collector_iter=iter(()),
        total_iterations=1,
        policy_operator=agent.actor_critic.get_policy_operator(),
        updates_completed=0,
        minibatches_per_epoch=1,
        epochs_per_rollout=1,
        anneal_clip_epsilon=False,
        base_clip_epsilon=agent.config.ppo.clip_epsilon,
    )

    metrics = agent._run_pre_policy_reward_updates(rollout_flat, metadata)

    assert metrics["reward_updates_applied"].item() == pytest.approx(1.0)
    assert torch.allclose(
        rollout_flat.get(("command", "policy_command")),
        stored_command,
    )


def test_ipmd_grouped_reward_terms_emit_per_head_metrics() -> None:
    """Grouped reward updates should expose policy/expert/diff/GP per head."""
    rlopt = _rlopt()
    cfg = rlopt.IPMDRLOptConfig()
    cfg.env.env_name = "Pendulum-v1"
    cfg.env.device = "cpu"
    cfg.device = "cpu"
    cfg.collector.frames_per_batch = 4
    cfg.collector.total_frames = 4
    cfg.replay_buffer.size = 64
    cfg.loss.mini_batch_size = 2
    cfg.compile.compile = False
    cfg.logger.backend = ""
    cfg.policy.input_keys = ["observation"]
    if cfg.value_function is not None:
        cfg.value_function.input_keys = ["observation"]
    cfg.ipmd.use_latent_command = False
    cfg.ipmd.reward_model_type = "grouped"
    cfg.ipmd.reward_input_type = "s"
    cfg.ipmd.reward_input_keys = [
        ("reward_state", "reference_command"),
        ("reward_state", "joint_pos"),
        ("reward_state", "root_quat"),
    ]
    cfg.ipmd.reward_loss_coeff = 1.0
    cfg.ipmd.reward_l2_coeff = 0.5
    cfg.ipmd.reward_grad_penalty_coeff = 1.0
    cfg.ipmd.reward_logit_reg_coeff = 0.1
    cfg.ipmd.bc_coef = 0.0

    env = rlopt.make_parallel_env(cfg)
    spec = env.observation_spec.clone()
    for key in cfg.ipmd.reward_input_keys:
        spec.set(key, spec["observation"].clone())
    env.observation_spec = spec
    agent = rlopt.IPMD(env, cfg, logger=None)

    obs_dim = env.observation_spec["observation"].shape[-1]
    batch_size = 4
    batch = TensorDict({}, batch_size=[batch_size], device=agent.device)
    expert_batch = TensorDict({}, batch_size=[batch_size], device=agent.device)
    for key in cfg.ipmd.reward_input_keys:
        batch.set(key, torch.randn(batch_size, obs_dim, device=agent.device))
        expert_batch.set(key, torch.randn(batch_size, obs_dim, device=agent.device))

    assert agent.reward_optim is not None
    agent.reward_optim.zero_grad(set_to_none=True)
    metrics = agent._backward_reward_terms(batch, expert_batch)

    for head in ("joint_pos", "root_quat"):
        for suffix in ("policy_mean", "expert_mean", "diff", "grad_penalty"):
            key = f"reward_head_{head}_{suffix}"
            assert key in metrics
            assert torch.isfinite(metrics[key])
    assert torch.isfinite(metrics["loss_reward_grad_penalty"])
    assert torch.isfinite(metrics["loss_reward_logit_reg"])


def test_ipmd_grouped_reward_model_requires_state_only_inputs() -> None:
    """Grouped reward heads should fail fast for transition/action input modes."""
    rlopt = _rlopt()
    cfg = rlopt.IPMDRLOptConfig()
    cfg.env.env_name = "Pendulum-v1"
    cfg.env.device = "cpu"
    cfg.device = "cpu"
    cfg.collector.frames_per_batch = 4
    cfg.collector.total_frames = 4
    cfg.compile.compile = False
    _apply_nonlatent_obs_input_keys(cfg)
    cfg.ipmd.reward_model_type = "grouped"
    cfg.ipmd.reward_input_type = "sas"

    env = rlopt.make_parallel_env(cfg)
    with pytest.raises(ValueError, match=r"requires .*reward_input_type='s'"):
        rlopt.IPMD(env, cfg, logger=None)


def test_ipmd_invalid_command_source_fails_at_agent_construction() -> None:
    """Invalid post-construction config mutations should fail before training starts."""
    rlopt = _rlopt()
    cfg = rlopt.IPMDRLOptConfig()
    cfg.env.env_name = "Pendulum-v1"
    cfg.env.device = "cpu"
    cfg.device = "cpu"
    cfg.collector.frames_per_batch = 4
    cfg.collector.total_frames = 4
    cfg.compile.compile = False
    _apply_obs_input_keys(cfg)
    cfg.ipmd.command_source = "rollout_posterior"

    env = rlopt.make_parallel_env(cfg)
    with pytest.raises(ValueError, match="command_source"):
        rlopt.IPMD(env, cfg, logger=None)


def test_ipmd_reward_update_schedule_runs_separate_optimizer_steps() -> None:
    """Reward updates should honor the configured cadence independently of PPO."""
    rlopt = _rlopt()
    cfg = rlopt.IPMDRLOptConfig()
    cfg.env.env_name = "Pendulum-v1"
    cfg.env.device = "cpu"
    cfg.device = "cpu"
    cfg.collector.frames_per_batch = 4
    cfg.collector.total_frames = 4
    cfg.replay_buffer.size = 64
    cfg.loss.mini_batch_size = 2
    cfg.compile.compile = False
    cfg.logger.backend = ""
    cfg.policy.input_keys = ["observation"]
    if cfg.value_function is not None:
        cfg.value_function.input_keys = ["observation"]
    cfg.ipmd.use_latent_command = False
    cfg.ipmd.reward_input_keys = ["observation"]
    cfg.ipmd.reward_input_type = "s"
    cfg.ipmd.reward_loss_coeff = 1.0
    cfg.ipmd.reward_l2_coeff = 0.0
    cfg.ipmd.reward_grad_penalty_coeff = 0.0
    cfg.ipmd.bc_coef = 0.0
    cfg.ipmd.reward_updates_per_policy_update = 2
    cfg.ipmd.reward_update_interval = 3
    cfg.ipmd.diversity_bonus_coeff = 0.0

    env = rlopt.make_parallel_env(cfg)
    agent = rlopt.IPMD(env, cfg, logger=None)
    assert agent.reward_optim is not None

    batch = TensorDict({}, batch_size=[2], device=agent.device)
    expert_batch = TensorDict({}, batch_size=[2], device=agent.device)
    backward_calls: list[int] = []
    step_calls: list[int] = []

    def _fake_backward_reward_terms(_batch, _expert_batch):
        backward_calls.append(1)
        return {
            "loss_reward_diff": torch.zeros((), device=agent.device),
            "loss_reward_l2": torch.zeros((), device=agent.device),
            "loss_reward_grad_penalty": torch.zeros((), device=agent.device),
            "loss_reward_grad_penalty_batch": torch.zeros((), device=agent.device),
            "loss_reward_grad_penalty_expert": torch.zeros((), device=agent.device),
            "loss_reward_logit_reg": torch.zeros((), device=agent.device),
            "loss_reward_param_weight_decay": torch.zeros((), device=agent.device),
        }

    def _fake_reward_step():
        step_calls.append(1)

    agent._backward_reward_terms = _fake_backward_reward_terms
    agent.reward_optim.step = _fake_reward_step

    skipped = agent._run_reward_updates(batch, expert_batch, policy_update_idx=1)
    assert skipped["reward_updates_applied"].item() == pytest.approx(0.0)
    assert len(backward_calls) == 0
    assert len(step_calls) == 0

    applied = agent._run_reward_updates(batch, expert_batch, policy_update_idx=3)
    assert applied["reward_updates_applied"].item() == pytest.approx(2.0)
    assert len(backward_calls) == 2
    assert len(step_calls) == 2


def test_ipmd_iterate_updates_reward_before_reward_recompute_and_advantage() -> None:
    """Pre-PPO reward updates should precede reward recompute and GAE."""
    from rlopt.agent.ppo.ppo import PPOIterationData, PPOTrainingMetadata

    agent, env = _make_env_reward_only_ipmd_agent()
    order: list[str] = []

    def _fake_reward_updates(_rollout_flat, _metadata):
        order.append("reward_update")
        return {}

    def _fake_prepare_rollout_rewards(_rollout):
        order.append("reward_recompute")
        return {}

    def _fake_pre_iteration_compute(_rollout):
        order.append("advantage")
        msg = "stop after advantage"
        raise RuntimeError(msg)

    agent._reward_update_enabled = True
    agent._run_pre_policy_reward_updates = _fake_reward_updates
    agent._prepare_rollout_rewards = _fake_prepare_rollout_rewards
    agent.pre_iteration_compute = _fake_pre_iteration_compute

    obs_dim = env.observation_spec["observation"].shape[-1]
    rollout = TensorDict(
        {
            "observation": torch.randn(2, obs_dim, device=agent.device),
            ("next", "reward"): torch.ones(2, 1, device=agent.device),
            ("next", "done"): torch.zeros(2, 1, dtype=torch.bool, device=agent.device),
            ("next", "truncated"): torch.zeros(
                2, 1, dtype=torch.bool, device=agent.device
            ),
        },
        batch_size=[2],
        device=agent.device,
    )
    iteration = PPOIterationData(iteration_idx=0, frames=2, rollout=rollout)
    metadata = PPOTrainingMetadata(
        collector_iter=iter(()),
        total_iterations=1,
        policy_operator=agent.actor_critic.get_policy_operator(),
        updates_completed=0,
        minibatches_per_epoch=1,
        epochs_per_rollout=1,
        anneal_clip_epsilon=False,
        base_clip_epsilon=agent.config.ppo.clip_epsilon,
    )

    with pytest.raises(RuntimeError, match="stop after advantage"):
        agent.iterate(iteration, metadata)

    assert order == ["reward_update", "reward_recompute", "advantage"]


def test_ipmd_reward_next_obs_requires_aligned_expert_transitions() -> None:
    """Reward next-state expert inputs should fail fast without aligned transitions."""
    rlopt = _rlopt()
    cfg = rlopt.IPMDRLOptConfig()
    cfg.env.env_name = "Pendulum-v1"
    cfg.env.device = "cpu"
    cfg.device = "cpu"
    cfg.collector.frames_per_batch = 4
    cfg.collector.total_frames = 4
    cfg.compile.compile = False
    cfg.logger.backend = ""
    cfg.policy.input_keys = ["observation"]
    if cfg.value_function is not None:
        cfg.value_function.input_keys = ["observation"]
    cfg.ipmd.use_latent_command = False
    cfg.ipmd.reward_input_keys = ["observation"]
    cfg.ipmd.reward_input_type = "s'"
    cfg.ipmd.bc_coef = 0.0

    env = rlopt.make_parallel_env(cfg)
    env._reference_has_aligned_next = False

    def _sample_expert_batch(*, batch_size: int, required_keys):
        del batch_size, required_keys
        return TensorDict({}, batch_size=[0])

    env.sample_expert_batch = _sample_expert_batch

    with pytest.raises(ValueError, match="transition-aligned next observations"):
        rlopt.IPMD(env, cfg, logger=None)


def test_ipmd_private_expert_sampler_integration():
    """Test IPMD integration with the private expert sampler override."""
    rlopt = _rlopt()
    cfg = rlopt.IPMDRLOptConfig()
    cfg.env.env_name = "Pendulum-v1"
    cfg.env.device = "cpu"
    cfg.device = "cpu"
    cfg.collector.frames_per_batch = 4
    cfg.collector.total_frames = 4
    cfg.replay_buffer.size = 64
    cfg.loss.mini_batch_size = 2
    cfg.compile.compile = False
    cfg.ipmd.expert_batch_size = 4
    _apply_nonlatent_obs_input_keys(cfg)

    env = rlopt.make_parallel_env(cfg)
    agent = rlopt.IPMD(env, cfg, logger=None)

    # Create synthetic expert data
    expert_data = create_synthetic_expert_data(env, num_transitions=50)
    agent._set_test_expert_batch_sampler(_make_test_expert_sampler(expert_data))

    # Sample from expert buffer
    expert_batch = agent._next_expert_batch(batch_size=cfg.ipmd.expert_batch_size)
    assert isinstance(expert_batch, TensorDict)
    assert expert_batch.numel() > 0


def test_ipmd_update_with_expert_data():
    """Test IPMD update step with expert data."""
    rlopt = _rlopt()
    cfg = rlopt.IPMDRLOptConfig()
    cfg.env.env_name = "Pendulum-v1"
    cfg.env.device = "cpu"
    cfg.device = "cpu"
    cfg.collector.frames_per_batch = 4
    cfg.collector.total_frames = 4
    cfg.replay_buffer.size = 64
    cfg.loss.mini_batch_size = 4
    cfg.compile.compile = False
    cfg.ipmd.reward_num_cells = (32, 32)
    cfg.ipmd.expert_batch_size = 4
    _apply_nonlatent_obs_input_keys(cfg)

    env = rlopt.make_parallel_env(cfg)
    agent = rlopt.IPMD(env, cfg, logger=None)

    # Create synthetic expert data
    expert_data = create_synthetic_expert_data(env, num_transitions=50)
    agent._set_test_expert_batch_sampler(_make_test_expert_sampler(expert_data))

    # Create policy data batch
    obs_dim = env.observation_spec["observation"].shape[-1]
    act_dim = env.action_spec.shape[-1]
    batch_size = 4

    policy_batch = TensorDict(
        {
            "observation": torch.randn(batch_size, obs_dim),
            "action": torch.randn(batch_size, act_dim),
            "action_log_prob": torch.zeros(batch_size),
            ("next", "observation"): torch.randn(batch_size, obs_dim),
            ("next", "reward"): torch.randn(batch_size, 1),
            ("next", "done"): torch.zeros(batch_size, 1, dtype=torch.bool),
            ("next", "terminated"): torch.zeros(batch_size, 1, dtype=torch.bool),
            ("next", "truncated"): torch.zeros(batch_size, 1, dtype=torch.bool),
        },
        batch_size=[batch_size],
    )
    with torch.no_grad():
        policy_batch = agent.adv_module(policy_batch)

    # Fixed-signature update (torch.compile / CUDA graph)
    num_network_updates = torch.zeros((), dtype=torch.int64, device=agent.device)
    expert_batch = agent._next_expert_batch(batch_size=cfg.ipmd.expert_batch_size)
    has_expert = torch.tensor(1.0, device=agent.device, dtype=torch.float32)
    loss_td, _ = agent.update(
        policy_batch, num_network_updates, expert_batch, has_expert
    )

    # Check loss outputs (PPO-based IPMD)
    assert "loss_critic" in loss_td
    assert "loss_objective" in loss_td
    assert "loss_entropy" in loss_td
    assert "loss_reward_diff" not in loss_td
    assert "loss_reward_l2" not in loss_td
    # Check that losses are finite
    for key in tuple(loss_td.keys()):
        value = loss_td[key]
        assert not torch.isnan(value).any(), f"NaN in {key}"
        assert not torch.isinf(value).any(), f"Inf in {key}"


def test_ipmd_bc_pretrain_updates_skip_then_restore_ppo() -> None:
    """BC pretraining should suppress PPO gradients only for its fixed prefix."""
    rlopt = _rlopt()
    cfg = rlopt.IPMDRLOptConfig()
    cfg.env.env_name = "Pendulum-v1"
    cfg.env.device = "cpu"
    cfg.device = "cpu"
    cfg.collector.frames_per_batch = 4
    cfg.collector.total_frames = 4
    cfg.replay_buffer.size = 64
    cfg.loss.mini_batch_size = 4
    cfg.compile.compile = False
    cfg.ipmd.expert_batch_size = 4
    _apply_nonlatent_obs_input_keys(cfg)
    cfg.ipmd.bc_coef = 1.0
    cfg.ipmd.bc_pretrain_updates = 1

    env = rlopt.make_parallel_env(cfg)
    agent = rlopt.IPMD(env, cfg, logger=None)
    obs_dim = env.observation_spec["observation"].shape[-1]
    act_dim = env.action_spec.shape[-1]
    batch_size = 4
    policy_batch = TensorDict(
        {
            "observation": torch.randn(batch_size, obs_dim),
            "action": torch.randn(batch_size, act_dim),
            "action_log_prob": torch.zeros(batch_size),
            ("next", "observation"): torch.randn(batch_size, obs_dim),
            ("next", "reward"): torch.randn(batch_size, 1),
            ("next", "done"): torch.zeros(batch_size, 1, dtype=torch.bool),
            ("next", "terminated"): torch.zeros(batch_size, 1, dtype=torch.bool),
            ("next", "truncated"): torch.zeros(batch_size, 1, dtype=torch.bool),
        },
        batch_size=[batch_size],
    )
    with torch.no_grad():
        policy_batch = agent.adv_module(policy_batch)
    expert_batch = TensorDict(
        {
            "observation": torch.randn(batch_size, obs_dim),
            "expert_action": torch.randn(batch_size, act_dim),
        },
        batch_size=[batch_size],
    )
    has_expert = torch.tensor(1.0, device=agent.device)

    warmup_loss, _ = agent.update(
        policy_batch.clone(), 0, expert_batch.clone(), has_expert
    )
    normal_loss, _ = agent.update(
        policy_batch.clone(), 1, expert_batch.clone(), has_expert
    )

    assert warmup_loss["bc_pretrain_active"].item() == 1.0
    assert normal_loss["bc_pretrain_active"].item() == 0.0
    assert torch.isfinite(warmup_loss["loss_bc"])
    assert torch.isfinite(normal_loss["loss_bc"])


def test_ipmd_bc_pretrain_requires_positive_bc_coefficient() -> None:
    cfg = _rlopt().IPMDRLOptConfig()
    cfg.ipmd.bc_coef = 0.0
    cfg.ipmd.bc_pretrain_updates = 1
    with pytest.raises(
        ValueError,
        match=r"requires ipmd\.bc_coef > 0 or ipmd\.rollout_bc_coef > 0",
    ):
        cfg.ipmd.validate()


def test_ipmd_rollout_bc_uses_live_policy_observations() -> None:
    """Rollout BC should train on rollout states without an expert sampler."""
    rlopt = _rlopt()
    cfg = rlopt.IPMDRLOptConfig()
    cfg.env.env_name = "Pendulum-v1"
    cfg.env.device = "cpu"
    cfg.device = "cpu"
    cfg.collector.frames_per_batch = 4
    cfg.collector.total_frames = 4
    cfg.replay_buffer.size = 64
    cfg.loss.mini_batch_size = 4
    cfg.compile.compile = False
    _apply_nonlatent_obs_input_keys(cfg)
    cfg.ipmd.rollout_bc_coef = 1.0
    cfg.ipmd.rollout_bc_action_key = ["policy_supervision", "expert_action"]
    cfg.ipmd.bc_pretrain_updates = 1

    env = rlopt.make_parallel_env(cfg)
    action_spec = env.action_spec.clone()
    spec = env.observation_spec.clone()
    spec.set(("policy_supervision", "expert_action"), action_spec)
    env.observation_spec = spec
    agent = rlopt.IPMD(env, cfg, logger=None)

    obs_dim = env.observation_spec["observation"].shape[-1]
    act_dim = env.action_spec.shape[-1]
    batch_size = 4
    policy_batch = TensorDict(
        {
            "observation": torch.randn(batch_size, obs_dim),
            ("policy_supervision", "expert_action"): torch.tanh(
                torch.randn(batch_size, act_dim)
            ),
            "action": torch.tanh(torch.randn(batch_size, act_dim)),
            "action_log_prob": torch.zeros(batch_size),
            ("next", "observation"): torch.randn(batch_size, obs_dim),
            ("next", "reward"): torch.randn(batch_size, 1),
            ("next", "done"): torch.zeros(batch_size, 1, dtype=torch.bool),
            ("next", "terminated"): torch.zeros(batch_size, 1, dtype=torch.bool),
            ("next", "truncated"): torch.zeros(batch_size, 1, dtype=torch.bool),
        },
        batch_size=[batch_size],
    )
    with torch.no_grad():
        policy_batch = agent.adv_module(policy_batch)
    no_expert_batch = TensorDict({}, batch_size=[batch_size])
    has_expert = torch.tensor(0.0, device=agent.device)

    warmup_loss, _ = agent.update(policy_batch.clone(), 0, no_expert_batch, has_expert)
    normal_loss, _ = agent.update(policy_batch.clone(), 1, no_expert_batch, has_expert)

    assert agent._rollout_bc_action_key == (
        "policy_supervision",
        "expert_action",
    )
    assert warmup_loss["bc_pretrain_active"].item() == 1.0
    assert normal_loss["bc_pretrain_active"].item() == 0.0
    for key in (
        "loss_rollout_bc",
        "rollout_bc_nll",
        "rollout_bc_policy_action_mae",
        "rollout_bc_policy_action_rmse",
    ):
        assert torch.isfinite(warmup_loss[key])
        assert torch.isfinite(normal_loss[key])
    assert "loss_bc" not in warmup_loss


def test_ipmd_rollout_bc_mse_supervises_policy_mean() -> None:
    """MSE rollout BC should be independent of the policy exploration std."""
    rlopt = _rlopt()
    cfg = rlopt.IPMDRLOptConfig()
    cfg.env.env_name = "Pendulum-v1"
    cfg.env.device = "cpu"
    cfg.device = "cpu"
    cfg.collector.frames_per_batch = 4
    cfg.collector.total_frames = 4
    cfg.replay_buffer.size = 64
    cfg.loss.mini_batch_size = 4
    cfg.compile.compile = False
    _apply_nonlatent_obs_input_keys(cfg)
    cfg.ipmd.rollout_bc_coef = 0.01
    cfg.ipmd.rollout_bc_loss_type = "mse"
    cfg.ipmd.rollout_bc_action_key = ["policy_supervision", "expert_action"]

    env = rlopt.make_parallel_env(cfg)
    spec = env.observation_spec.clone()
    spec.set(("policy_supervision", "expert_action"), env.action_spec.clone())
    env.observation_spec = spec
    agent = rlopt.IPMD(env, cfg, logger=None)

    obs_dim = env.observation_spec["observation"].shape[-1]
    act_dim = env.action_spec.shape[-1]
    batch = TensorDict(
        {
            "observation": torch.randn(4, obs_dim),
            ("policy_supervision", "expert_action"): torch.randn(4, act_dim),
            "action": torch.randn(4, act_dim),
            "action_log_prob": torch.zeros(4),
            ("next", "observation"): torch.randn(4, obs_dim),
            ("next", "reward"): torch.randn(4, 1),
            ("next", "done"): torch.zeros(4, 1, dtype=torch.bool),
            ("next", "terminated"): torch.zeros(4, 1, dtype=torch.bool),
            ("next", "truncated"): torch.zeros(4, 1, dtype=torch.bool),
        },
        batch_size=[4],
    )
    with torch.no_grad():
        batch = agent.adv_module(batch)
    loss, _ = agent.update(
        batch,
        0,
        TensorDict({}, batch_size=[4]),
        torch.tensor(0.0),
    )

    assert torch.isfinite(loss["rollout_bc_mse"])
    assert torch.allclose(
        loss["loss_rollout_bc"],
        0.01 * loss["rollout_bc_mse"],
    )


def test_ipmd_rollout_bc_rejects_unknown_loss_type() -> None:
    cfg = _rlopt().IPMDRLOptConfig()
    cfg.ipmd.rollout_bc_loss_type = "cosine"
    with pytest.raises(ValueError, match="rollout_bc_loss_type"):
        cfg.ipmd.validate()


def test_ipmd_rollout_bc_label_cannot_be_a_model_input() -> None:
    """The rollout action label must remain outside every learned model input."""
    rlopt = _rlopt()
    cfg = rlopt.IPMDRLOptConfig()
    cfg.env.env_name = "Pendulum-v1"
    cfg.env.device = "cpu"
    cfg.device = "cpu"
    cfg.collector.frames_per_batch = 4
    cfg.loss.mini_batch_size = 2
    cfg.compile.compile = False
    _apply_nonlatent_obs_input_keys(cfg)
    cfg.ipmd.rollout_bc_coef = 1.0
    cfg.ipmd.rollout_bc_action_key = "expert_action"
    cfg.policy.input_keys = ["observation", "expert_action"]

    env = rlopt.make_parallel_env(cfg)
    spec = env.observation_spec.clone()
    spec.set("expert_action", env.action_spec.clone())
    env.observation_spec = spec

    with pytest.raises(ValueError, match="training-only label"):
        rlopt.IPMD(env, cfg, logger=None)


def test_ipmd_missing_sampler_raises() -> None:
    """IPMD expert sampling should fail fast when no sampler is available."""
    rlopt = _rlopt()
    cfg = rlopt.IPMDRLOptConfig()
    cfg.env.env_name = "Pendulum-v1"
    cfg.env.device = "cpu"
    cfg.device = "cpu"
    cfg.compile.compile = False
    cfg.collector.frames_per_batch = 10
    cfg.collector.total_frames = 100  # Divisible by frames_per_batch
    _apply_nonlatent_obs_input_keys(cfg)

    env = rlopt.make_parallel_env(cfg)
    agent = rlopt.IPMD(env, cfg, logger=None)
    agent._expert_batch_sampler = None

    with pytest.raises(RuntimeError, match="sample_expert_batch"):
        agent._next_expert_batch(batch_size=4)


def test_ipmd_sampler_size_mismatch_raises() -> None:
    """Expert sampler must return exactly the requested batch size."""
    rlopt = _rlopt()
    cfg = rlopt.IPMDRLOptConfig()
    cfg.env.env_name = "Pendulum-v1"
    cfg.env.device = "cpu"
    cfg.device = "cpu"
    cfg.compile.compile = False
    cfg.collector.frames_per_batch = 4
    cfg.collector.total_frames = 4
    _apply_nonlatent_obs_input_keys(cfg)

    env = rlopt.make_parallel_env(cfg)
    agent = rlopt.IPMD(env, cfg, logger=None)
    expert_data = create_synthetic_expert_data(env, num_transitions=8)

    def _sample(_batch_size: int, required_keys):
        return expert_data.select(*required_keys).clone()

    agent._set_test_expert_batch_sampler(_sample)

    with pytest.raises(RuntimeError, match="exactly the requested batch size"):
        agent._next_expert_batch(batch_size=4)


def test_ipmd_sampler_missing_required_keys_raises() -> None:
    """Expert sampler should fail fast when required keys are absent."""
    rlopt = _rlopt()
    cfg = rlopt.IPMDRLOptConfig()
    cfg.env.env_name = "Pendulum-v1"
    cfg.env.device = "cpu"
    cfg.device = "cpu"
    cfg.compile.compile = False
    cfg.collector.frames_per_batch = 4
    cfg.collector.total_frames = 4
    _apply_nonlatent_obs_input_keys(cfg)

    env = rlopt.make_parallel_env(cfg)
    agent = rlopt.IPMD(env, cfg, logger=None)
    obs_dim = env.observation_spec["observation"].shape[-1]
    expert_data = TensorDict(
        {"observation": torch.randn(4, obs_dim)},
        batch_size=[4],
    )

    def _sample(_batch_size: int, required_keys):
        return expert_data.select(
            *(key for key in required_keys if key in expert_data.keys(True))
        ).clone()

    agent._set_test_expert_batch_sampler(_sample)

    with pytest.raises(RuntimeError, match="missing required keys"):
        agent._next_expert_batch(batch_size=4, required_keys=["action"])


def test_ipmd_posterior_keys_default_to_reward_inputs() -> None:
    """Patch posterior inputs should default to the configured reward keys."""
    rlopt = _rlopt()
    cfg = rlopt.IPMDRLOptConfig()
    cfg.env.env_name = "Pendulum-v1"
    cfg.env.device = "cpu"
    cfg.device = "cpu"
    cfg.compile.compile = False
    cfg.collector.frames_per_batch = 4
    cfg.collector.total_frames = 4
    _apply_obs_input_keys(cfg)

    env = rlopt.make_parallel_env(cfg)
    agent = rlopt.IPMD(env, cfg, logger=None)

    assert agent._posterior_obs_keys == ["observation"]


def test_ipmd_pre_iteration_compute_stays_mi_free() -> None:
    """Prepared latent IPMD rollouts should not attach removed MI targets."""
    rlopt = _rlopt()
    cfg = rlopt.IPMDRLOptConfig()
    cfg.env.env_name = "Pendulum-v1"
    cfg.env.device = "cpu"
    cfg.device = "cpu"
    cfg.collector.frames_per_batch = 4
    cfg.collector.total_frames = 4
    cfg.collector.init_random_frames = 0
    cfg.loss.mini_batch_size = 2
    cfg.compile.compile = False
    _apply_obs_input_keys(cfg)

    env = rlopt.make_parallel_env(cfg)
    agent = rlopt.IPMD(env, cfg, logger=None)
    agent._set_test_expert_batch_sampler(
        _make_test_expert_sampler(create_synthetic_expert_data(env, num_transitions=20))
    )

    rollout = next(iter(agent.collector))
    agent._prepare_rollout_rewards(rollout)

    prepared = agent.pre_iteration_compute(rollout)
    assert "mi_reward" not in prepared.keys(True)
    assert "mi_value" not in prepared.keys(True)
    assert "mi_returns" not in prepared.keys(True)
    assert "mi_advantage" not in prepared.keys(True)


def test_ipmd_posterior_collector_latents_are_recomputed_each_step() -> None:
    """Posterior collection should publish the current batch latent each step."""
    rlopt = _rlopt()
    cfg = rlopt.IPMDRLOptConfig()
    cfg.env.env_name = "Pendulum-v1"
    cfg.env.device = "cpu"
    cfg.device = "cpu"
    cfg.collector.frames_per_batch = 4
    cfg.collector.total_frames = 4
    cfg.collector.init_random_frames = 0
    cfg.loss.mini_batch_size = 2
    cfg.compile.compile = False
    _apply_obs_input_keys(cfg)
    cfg.ipmd.command_source = "posterior"
    cfg.ipmd.latent_steps_min = 10
    cfg.ipmd.latent_steps_max = 10
    cfg.ipmd.latent_learning.posterior_input_keys = ["observation"]

    env = rlopt.make_parallel_env(cfg)
    agent = rlopt.IPMD(env, cfg, logger=None)

    class _NormalizeIdentity(torch.nn.Module):
        def forward(self, obs_features: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.normalize(obs_features, dim=-1, eps=1.0e-6)

    normalize_identity = _NormalizeIdentity().to(agent.device)
    assert agent._latent_learner is not None
    agent._latent_learner.encoder = normalize_identity
    rollout_obs_1 = torch.tensor(
        [[0.2, 0.3, 0.4], [0.9, -0.1, 0.1]],
        device=agent.device,
        dtype=torch.float32,
    )
    rollout_obs_2 = torch.tensor(
        [[-0.3, 0.1, 0.7], [0.4, 0.4, -0.2]],
        device=agent.device,
        dtype=torch.float32,
    )
    rollout_td_1 = TensorDict(
        {"observation": rollout_obs_1.clone()},
        batch_size=[rollout_obs_1.shape[0]],
        device=agent.device,
    )
    rollout_td_2 = TensorDict(
        {"observation": rollout_obs_2.clone()},
        batch_size=[rollout_obs_2.shape[0]],
        device=agent.device,
    )
    agent._inject_latent_command(rollout_td_1)
    agent._inject_latent_command(rollout_td_2)

    expected_1 = torch.nn.functional.normalize(rollout_obs_1, dim=-1, eps=1.0e-6)
    expected_2 = torch.nn.functional.normalize(rollout_obs_2, dim=-1, eps=1.0e-6)
    assert torch.allclose(rollout_td_1.get("observation"), expected_1)
    assert torch.allclose(rollout_td_2.get("observation"), expected_2)
    assert not torch.allclose(
        rollout_td_1.get("observation"),
        rollout_td_2.get("observation"),
    )


def test_ipmd_vqvae_phase_config_validates_command_width() -> None:
    """VQVAE code width plus phase features should be validated up front."""
    rlopt = _rlopt()
    cfg = rlopt.IPMDRLOptConfig()
    cfg.ipmd.latent_dim = 66
    cfg.ipmd.latent_learning.command_phase_mode = "sin_cos"
    cfg.ipmd.latent_learning.code_latent_dim = 64
    cfg.ipmd.validate()

    bad_cfg = rlopt.IPMDRLOptConfig()
    bad_cfg.ipmd.latent_dim = 65
    bad_cfg.ipmd.latent_learning.command_phase_mode = "sin_cos"
    bad_cfg.ipmd.latent_learning.code_latent_dim = 64
    with pytest.raises(ValueError, match="code_latent_dim plus phase"):
        bad_cfg.ipmd.validate()

    bad_quantizer = rlopt.IPMDRLOptConfig()
    bad_quantizer.ipmd.latent_learning.quantizer = "not_a_quantizer"
    with pytest.raises(ValueError, match="quantizer"):
        bad_quantizer.ipmd.validate()


def test_fsq_quantizer_uses_all_even_level_codes() -> None:
    """Even FSQ levels should map to all integer bins without half-step collapse."""
    from rlopt.agent.imitation.latent_learning import FSQQuantizer

    quantizer = FSQQuantizer([8])
    z_e = torch.linspace(-20.0, 20.0, steps=4096).unsqueeze(-1)
    _, code, _ = quantizer(z_e)

    assert torch.equal(torch.unique(code), torch.arange(8))


def test_patch_vqvae_collector_holds_code_and_reports_phase() -> None:
    """The collector hook should hold z_q while sin/cos phase advances."""
    rlopt = _rlopt()
    from rlopt.agent.imitation.latent_learning import PatchVQVAELatentLearner

    cfg = rlopt.IPMDRLOptConfig()
    cfg.ipmd.latent_dim = 5
    cfg.ipmd.latent_learning.method = "patch_vqvae"
    cfg.ipmd.latent_learning.quantizer = "identity"
    cfg.ipmd.latent_learning.code_latent_dim = 3
    cfg.ipmd.latent_learning.command_phase_mode = "sin_cos"
    cfg.ipmd.latent_learning.code_period = 3
    cfg.ipmd.latent_learning.posterior_input_keys = ["observation"]
    cfg.ipmd.validate()

    fake_agent = SimpleNamespace(
        config=cfg,
        device=torch.device("cpu"),
        _latent_dim=5,
        _obs_feature_dims={"observation": 3},
        _posterior_obs_keys=["observation"],
        _action_feature_dim=2,
    )
    learner = PatchVQVAELatentLearner()
    learner.initialize(fake_agent)
    learner.encoder = torch.nn.Identity()

    td0 = TensorDict(
        {"observation": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])},
        batch_size=[2],
    )
    td1 = TensorDict(
        {"observation": torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])},
        batch_size=[2],
    )
    td3 = TensorDict(
        {"observation": torch.tensor([[7.0, 8.0, 9.0], [1.0, 3.0, 5.0]])},
        batch_size=[2],
    )

    cmd0 = learner.infer_collector_latents(td0)
    cmd1 = learner.infer_collector_latents(td1)
    cmd2 = learner.infer_collector_latents(td1)
    cmd3 = learner.infer_collector_latents(td3)

    assert torch.allclose(cmd0[:, :3], td0["observation"])
    assert torch.allclose(cmd1[:, :3], td0["observation"])
    assert torch.allclose(cmd2[:, :3], td0["observation"])
    assert torch.allclose(cmd3[:, :3], td3["observation"])

    phase0 = torch.tensor([0.0, 1.0]).repeat(2, 1)
    angle1 = torch.tensor(2.0 * torch.pi / 3.0)
    phase1 = torch.stack((torch.sin(angle1), torch.cos(angle1))).repeat(2, 1)
    angle2 = torch.tensor(4.0 * torch.pi / 3.0)
    phase2 = torch.stack((torch.sin(angle2), torch.cos(angle2))).repeat(2, 1)
    assert torch.allclose(cmd0[:, -2:], phase0)
    assert torch.allclose(cmd1[:, -2:], phase1)
    assert torch.allclose(cmd2[:, -2:], phase2)
    assert torch.allclose(cmd3[:, -2:], phase0)


def test_future_cvae_trains_distinct_posterior_prior_and_future_target() -> None:
    """Future CVAE should learn from explicit current and future feature groups."""
    rlopt = _rlopt()
    from rlopt.agent.imitation.latent_learning import (
        FutureCVAELatentLearner,
        build_latent_learner,
    )

    cfg = rlopt.IPMDRLOptConfig()
    cfg.ipmd.latent_dim = 2
    cfg.ipmd.latent_learning.method = "future_cvae"
    cfg.ipmd.latent_learning.posterior_input_keys = ["current", "future"]
    cfg.ipmd.latent_learning.prior_input_keys = ["current"]
    cfg.ipmd.latent_learning.reconstruction_target_keys = ["future"]
    cfg.ipmd.latent_learning.encoder_hidden_dims = [8]
    cfg.ipmd.latent_learning.prior_hidden_dims = [8]
    cfg.ipmd.latent_learning.decoder_hidden_dims = [8]
    cfg.ipmd.latent_learning.recon_coeff = 1.0
    cfg.ipmd.latent_learning.kl_coeff = 0.1
    cfg.ipmd.latent_learning.posterior_command_period = 3
    cfg.ipmd.validate()

    expert_data = TensorDict(
        {
            "current": torch.randn(6, 2),
            "future": torch.randn(6, 4),
        },
        batch_size=[6],
    )

    def _obs_features_from_td(td, keys, *, next_obs, detach):
        assert next_obs is False
        parts = []
        for key in keys:
            value = td.get(key)
            value = value.detach() if detach else value
            parts.append(value.reshape(value.shape[0], -1))
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)

    def _next_expert_batch(batch_size=6, *, required_keys):
        return expert_data[:batch_size].select(*required_keys).clone()

    fake_agent = SimpleNamespace(
        config=cfg,
        device=torch.device("cpu"),
        _latent_dim=2,
        _obs_feature_dims={"current": 2, "future": 4},
        _posterior_obs_keys=["current", "future"],
        _prior_obs_keys=["current"],
        _reconstruction_target_obs_keys=["future"],
        _obs_features_from_td=_obs_features_from_td,
        _next_expert_batch=_next_expert_batch,
    )
    learner = build_latent_learner("future_cvae")
    assert isinstance(learner, FutureCVAELatentLearner)
    learner.initialize(fake_agent)

    assert learner.required_expert_batch_keys() == ["current", "future"]
    metrics = learner.update(TensorDict({}, batch_size=[0]))
    assert metrics["ipmd/latent_recon_loss"] >= 0.0
    assert metrics["ipmd/latent_kl_loss"] >= 0.0
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())

    latents = learner.infer_expert_latents(expert_data, detach=True)
    reconstruction = learner.reconstruct_batch_features(
        expert_data,
        detach=True,
        context="test",
    )
    prior_samples = learner.sample_expert_prior_latents(
        batch_size=3,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert latents.shape == (6, 2)
    assert reconstruction is not None
    assert reconstruction.shape == (6, 4)
    assert prior_samples is not None
    assert prior_samples.shape == (3, 2)


def test_future_cvae_collector_holds_posterior_mean_for_command_period() -> None:
    """The oracle CVAE command should refresh only at its configured boundary."""
    rlopt = _rlopt()
    from rlopt.agent.imitation.latent_learning import FutureCVAELatentLearner

    cfg = rlopt.IPMDRLOptConfig()
    cfg.ipmd.latent_dim = 2
    cfg.ipmd.latent_learning.method = "future_cvae"
    cfg.ipmd.latent_learning.posterior_input_keys = ["current", "future"]
    cfg.ipmd.latent_learning.prior_input_keys = ["current"]
    cfg.ipmd.latent_learning.reconstruction_target_keys = ["future"]
    cfg.ipmd.latent_learning.posterior_command_period = 3
    cfg.ipmd.validate()

    def _obs_features_from_td(td, keys, *, next_obs, detach):
        del next_obs
        parts = [td.get(key).reshape(td.numel(), -1) for key in keys]
        features = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
        return features.detach() if detach else features

    fake_agent = SimpleNamespace(
        config=cfg,
        device=torch.device("cpu"),
        _latent_dim=2,
        _obs_feature_dims={"current": 2, "future": 4},
        _posterior_obs_keys=["current", "future"],
        _prior_obs_keys=["current"],
        _reconstruction_target_obs_keys=["future"],
        _obs_features_from_td=_obs_features_from_td,
    )
    learner = FutureCVAELatentLearner()
    learner.initialize(fake_agent)

    class _CurrentAsMean(torch.nn.Module):
        def forward(self, features):
            mean = features[:, :2]
            return mean, torch.zeros_like(mean)

    learner.posterior = _CurrentAsMean()
    td0 = TensorDict(
        {"current": torch.tensor([[1.0, 2.0]]), "future": torch.zeros(1, 4)},
        batch_size=[1],
    )
    td1 = TensorDict(
        {"current": torch.tensor([[5.0, 6.0]]), "future": torch.ones(1, 4)},
        batch_size=[1],
    )

    command0 = learner.infer_collector_latents(td0).clone()
    command1 = learner.infer_collector_latents(td1).clone()
    command2 = learner.infer_collector_latents(td1).clone()
    command3 = learner.infer_collector_latents(td1).clone()

    assert torch.allclose(command0, torch.tensor([[1.0, 2.0]]))
    assert torch.allclose(command1, command0)
    assert torch.allclose(command2, command0)
    assert torch.allclose(command3, torch.tensor([[5.0, 6.0]]))


def test_per_step_vq_sequence_consumes_one_token_per_control_step() -> None:
    """A horizon packet should publish consecutive tokens, not one held code."""
    rlopt = _rlopt()
    from rlopt.agent.imitation.latent_learning import (
        PerStepVQSequenceLatentLearner,
    )

    cfg = rlopt.IPMDRLOptConfig()
    cfg.ipmd.latent_dim = 2
    cfg.ipmd.latent_learning.method = "per_step_vq_sequence"
    cfg.ipmd.latent_learning.quantizer = "fsq"
    cfg.ipmd.latent_learning.fsq_levels = [3, 3]
    cfg.ipmd.latent_learning.code_latent_dim = 2
    cfg.ipmd.latent_learning.command_phase_mode = "none"
    cfg.ipmd.latent_learning.token_sequence_horizon = 3
    cfg.ipmd.latent_learning.posterior_input_keys = ["motion", "anchor"]
    cfg.ipmd.validate()

    fake_agent = SimpleNamespace(
        config=cfg,
        device=torch.device("cpu"),
        _latent_dim=2,
        _obs_feature_dims={"motion": 6, "anchor": 3},
        _posterior_obs_keys=["motion", "anchor"],
        _action_feature_dim=2,
    )
    learner = PerStepVQSequenceLatentLearner()
    learner.initialize(fake_agent)

    class _SelectTwo(torch.nn.Module):
        def forward(self, features):
            return features[..., :2]

    learner.encoder = _SelectTwo()
    td0 = TensorDict(
        {
            "motion": torch.tensor(
                [
                    [0.1, 0.2, 0.7, -0.4, -0.8, 0.9],
                    [-0.3, 0.5, 0.6, 0.7, -0.2, -0.1],
                ]
            ),
            "anchor": torch.zeros(2, 3),
        },
        batch_size=[2],
    )
    td1 = TensorDict(
        {
            "motion": torch.tensor(
                [
                    [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1],
                    [0.9, 0.7, 0.5, 0.3, 0.1, -0.1],
                ]
            ),
            "anchor": torch.ones(2, 3),
        },
        batch_size=[2],
    )

    token_ids0, packet0 = learner.infer_token_packet(td0, detach=True)
    token_ids1, packet1 = learner.infer_token_packet(td1, detach=True)
    command0 = learner.infer_collector_latents(td0).clone()
    command1 = learner.infer_collector_latents(td1).clone()
    command2 = learner.infer_collector_latents(td1).clone()
    command3 = learner.infer_collector_latents(td1).clone()

    assert token_ids0.shape == (2, 3)
    assert torch.allclose(learner.decode_token_ids(token_ids0), packet0)
    assert torch.allclose(learner.decode_token_ids(token_ids1), packet1)
    assert torch.allclose(command0, packet0[:, 0])
    assert torch.allclose(command1, packet0[:, 1])
    assert torch.allclose(command2, packet0[:, 2])
    assert torch.allclose(command3, packet1[:, 0])


def test_per_step_vq_sequence_reconstructs_per_step_features() -> None:
    """The sequence VQ update should train on B by horizon by feature tensors."""
    rlopt = _rlopt()
    from rlopt.agent.imitation.latent_learning import (
        PerStepVQSequenceLatentLearner,
    )

    cfg = rlopt.IPMDRLOptConfig()
    cfg.ipmd.latent_dim = 2
    cfg.ipmd.latent_learning.method = "per_step_vq_sequence"
    cfg.ipmd.latent_learning.quantizer = "fsq"
    cfg.ipmd.latent_learning.fsq_levels = [3, 3]
    cfg.ipmd.latent_learning.code_latent_dim = 2
    cfg.ipmd.latent_learning.token_sequence_horizon = 3
    cfg.ipmd.latent_learning.posterior_input_keys = ["window"]
    cfg.ipmd.latent_learning.encoder_hidden_dims = [8]
    cfg.ipmd.latent_learning.decoder_hidden_dims = [8]
    cfg.ipmd.latent_learning.recon_coeff = 1.0
    cfg.ipmd.validate()

    expert_data = TensorDict(
        {"window": torch.randn(5, 9)},
        batch_size=[5],
    )

    def _next_expert_batch(batch_size=5, *, required_keys):
        return expert_data[:batch_size].select(*required_keys).clone()

    fake_agent = SimpleNamespace(
        config=cfg,
        device=torch.device("cpu"),
        _latent_dim=2,
        _obs_feature_dims={"window": 9},
        _posterior_obs_keys=["window"],
        _action_feature_dim=2,
        _next_expert_batch=_next_expert_batch,
    )
    learner = PerStepVQSequenceLatentLearner()
    learner.initialize(fake_agent)
    metrics = learner.update(TensorDict({}, batch_size=[0]))
    reconstruction = learner.reconstruct_batch_features(
        expert_data,
        detach=True,
        context="test",
    )

    assert metrics["ipmd/token_sequence_horizon"] == 3.0
    assert metrics["ipmd/vqvae_recon_mse"] >= 0.0
    assert reconstruction is not None
    assert reconstruction.shape == (5, 3, 3)


def test_causal_categorical_planner_uses_shared_transformer_backbone_settings() -> None:
    """Discrete token planning should keep the continuous planner backbone size."""
    from rlopt.agent.causal_interface_planner import (
        CausalInterfaceTransformerCategoricalPlanner,
        CausalInterfaceTransformerFlowPlanner,
    )

    common = {
        "state_dim": 12,
        "d_model": 16,
        "num_layers": 2,
        "num_heads": 4,
        "feedforward_dim": 32,
        "num_state_tokens": 2,
        "dropout": 0.0,
    }
    categorical = CausalInterfaceTransformerCategoricalPlanner(
        **common,
        token_horizon=3,
        codebook_size=7,
    )
    continuous = CausalInterfaceTransformerFlowPlanner(
        **common,
        target_dim=6,
        patch_dim=2,
    )
    state = torch.randn(5, 12)
    token_ids = torch.randint(0, 7, (5, 3))

    logits = categorical.logits(state)
    prediction = categorical(state)
    loss = categorical.categorical_loss(state, token_ids)
    categorical_config = categorical.config_dict()
    continuous_config = continuous.config_dict()

    assert logits.shape == (5, 3, 7)
    assert prediction.shape == (5, 3)
    assert torch.isfinite(loss)
    for key in (
        "state_dim",
        "d_model",
        "num_layers",
        "num_heads",
        "feedforward_dim",
        "num_state_tokens",
        "dropout",
    ):
        assert categorical_config[key] == continuous_config[key]


def test_causal_continuous_planner_accepts_optional_language_token() -> None:
    from rlopt.agent.causal_interface_planner import (
        CausalInterfaceTransformerFlowPlanner,
    )

    planner = CausalInterfaceTransformerFlowPlanner(
        state_dim=12,
        target_dim=6,
        d_model=16,
        num_layers=1,
        num_heads=4,
        feedforward_dim=32,
        patch_dim=2,
        num_state_tokens=2,
        language_dim=5,
        num_language_tokens=1,
    )
    state = torch.randn(4, 12)
    language = torch.randn(4, 5)
    target = torch.randn(4, 6)

    prediction = planner(
        state,
        language=language,
        num_inference_steps=2,
    )
    loss = planner.flow_matching_loss(state, target, language=language)
    restored = CausalInterfaceTransformerFlowPlanner.from_config(planner.config_dict())
    restored.load_state_dict(planner.state_dict())

    assert prediction.shape == target.shape
    assert torch.isfinite(loss)
    assert restored.language_dim == 5
    assert restored.num_language_tokens == 1
    with pytest.raises(ValueError, match="requires a language embedding"):
        planner(state, num_inference_steps=1)


def test_frozen_token_planner_decodes_and_consumes_predicted_packet(tmp_path) -> None:
    """Planner deployment should decode IDs once and consume the packet in order."""
    from rlopt.agent.causal_interface_planner import (
        CausalInterfaceTransformerCategoricalPlanner,
        FrozenCategoricalTokenPlannerSampler,
    )

    planner = CausalInterfaceTransformerCategoricalPlanner(
        state_dim=12,
        token_horizon=3,
        codebook_size=3,
        d_model=12,
        num_layers=1,
        num_heads=3,
        feedforward_dim=24,
        num_state_tokens=1,
    )
    checkpoint_path = tmp_path / "token_planner.pt"
    torch.save(
        {
            "planner_config": planner.config_dict(),
            "planner_state_dict": planner.state_dict(),
            "target_spec": {
                "interface": "per_step_token_sequence",
                "term_names": ["token_ids"],
                "term_widths": [3],
                "target_dim": 3,
            },
            "metadata": {
                "target_encoding": {
                    "kind": "categorical_sequence",
                    "horizon": 3,
                    "codebook_size": 3,
                },
                "planner_observation_spec": {
                    "history_frames": 2,
                    "frame_dim": 6,
                    "flat_dim": 12,
                },
                "sample_metadata": {
                    "state_history_steps": 1,
                    "planner_observation_spec": {
                        "history_frames": 2,
                        "frame_dim": 6,
                        "flat_dim": 12,
                    },
                },
            },
        },
        checkpoint_path,
    )

    class _FakeEnv:
        def __init__(self):
            self.calls = 0

        def causal_planner_observation_spec(self, *, history_steps):
            assert history_steps == 1
            return {"history_frames": 2, "frame_dim": 6, "flat_dim": 12}

        def current_causal_planner_observation(self, *, env_ids, history_steps):
            assert history_steps == 1
            self.calls += 1
            return TensorDict(
                {("planner", "state_history"): torch.zeros(env_ids.numel(), 2, 6)},
                batch_size=[env_ids.numel()],
            )

    env = _FakeEnv()
    sampler = FrozenCategoricalTokenPlannerSampler(
        env=env,
        checkpoint_path=checkpoint_path,
        decode_token_ids=lambda ids: torch.nn.functional.one_hot(
            ids, num_classes=3
        ).float(),
        latent_dim=3,
        discover_env_method=lambda obj, name: getattr(obj, name, None),
        device="cpu",
    )

    class _FixedPacket(torch.nn.Module):
        def forward(self, state):
            return torch.tensor([0, 1, 2]).expand(state.shape[0], -1)

    sampler.planner = _FixedPacket()
    td = TensorDict({}, batch_size=[2])
    commands = [
        sampler.sample_for_step(td, device=torch.device("cpu"), dtype=torch.float32)
        for _ in range(4)
    ]

    assert torch.equal(commands[0], torch.tensor([[1, 0, 0], [1, 0, 0]]).float())
    assert torch.equal(commands[1], torch.tensor([[0, 1, 0], [0, 1, 0]]).float())
    assert torch.equal(commands[2], torch.tensor([[0, 0, 1], [0, 0, 1]]).float())
    assert torch.equal(commands[3], commands[0])
    assert env.calls == 2


def test_frozen_continuous_planner_holds_command_for_configured_period(
    tmp_path,
) -> None:
    """Continuous planner deployment should renew only after the hold period."""
    from rlopt.agent.causal_interface_planner import (
        CausalInterfaceTransformerFlowPlanner,
        FrozenContinuousInterfacePlannerSampler,
    )

    planner = CausalInterfaceTransformerFlowPlanner(
        state_dim=12,
        target_dim=3,
        term_widths=(3,),
        d_model=12,
        num_layers=1,
        num_heads=3,
        feedforward_dim=24,
        patch_dim=3,
        num_state_tokens=1,
    )
    checkpoint_path = tmp_path / "continuous_planner.pt"
    observation_spec = {"history_frames": 2, "frame_dim": 6, "flat_dim": 12}
    torch.save(
        {
            "planner_config": planner.config_dict(),
            "planner_state_dict": planner.state_dict(),
            "target_spec": {
                "interface": "future_cvae",
                "term_names": ["z"],
                "term_widths": [3],
                "target_dim": 3,
            },
            "metadata": {
                "flow_num_inference_steps": 2,
                "flow_inference_noise_std": 0.0,
                "planner_observation_spec": observation_spec,
                "sample_metadata": {
                    "state_history_steps": 1,
                    "planner_interval_steps": 3,
                    "planner_observation_spec": observation_spec,
                },
            },
        },
        checkpoint_path,
    )

    class _FakeEnv:
        def __init__(self):
            self.calls = 0

        def causal_planner_observation_spec(self, *, history_steps):
            assert history_steps == 1
            return observation_spec

        def current_causal_planner_observation(self, *, env_ids, history_steps):
            assert history_steps == 1
            self.calls += 1
            return TensorDict(
                {("planner", "state_history"): torch.zeros(env_ids.numel(), 2, 6)},
                batch_size=[env_ids.numel()],
            )

    env = _FakeEnv()
    sampler = FrozenContinuousInterfacePlannerSampler(
        env=env,
        checkpoint_path=checkpoint_path,
        latent_dim=3,
        discover_env_method=lambda obj, name: getattr(obj, name, None),
        device="cpu",
    )

    class _FixedCommand(torch.nn.Module):
        def forward(self, state, **_kwargs):
            return torch.tensor([1.0, 2.0, 3.0]).expand(state.shape[0], -1)

    sampler.planner = _FixedCommand()
    td = TensorDict({}, batch_size=[2])
    commands = [
        sampler.sample_for_step(td, device=torch.device("cpu"), dtype=torch.float32)
        for _ in range(4)
    ]

    expected = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    assert all(torch.equal(command, expected) for command in commands)
    assert env.calls == 2


def test_ipmd_patch_autoencoder_uses_only_posterior_expert_keys() -> None:
    """Patch autoencoder latent updates should be reconstruction-only expert batches."""
    rlopt = _rlopt()
    cfg = rlopt.IPMDRLOptConfig()
    cfg.env.env_name = "Pendulum-v1"
    cfg.env.device = "cpu"
    cfg.device = "cpu"
    cfg.collector.frames_per_batch = 4
    cfg.collector.total_frames = 4
    cfg.collector.init_random_frames = 0
    cfg.loss.mini_batch_size = 2
    cfg.compile.compile = False
    _apply_obs_input_keys(cfg)
    cfg.ipmd.reward_loss_coeff = 0.0
    cfg.ipmd.reward_l2_coeff = 0.0
    cfg.ipmd.reward_grad_penalty_coeff = 0.0
    cfg.ipmd.diversity_bonus_coeff = 0.0
    cfg.ipmd.latent_learning.method = "patch_autoencoder"
    cfg.ipmd.latent_learning.posterior_input_keys = ["observation"]
    cfg.ipmd.latent_learning.recon_coeff = 1.0
    cfg.ipmd.latent_learning.uniformity_coeff = 0.0
    cfg.ipmd.latent_learning.weight_decay_coeff = 0.0

    env = rlopt.make_parallel_env(cfg)
    agent = rlopt.IPMD(env, cfg, logger=None)
    obs_dim = env.observation_spec["observation"].shape[-1]
    expert_data = TensorDict(
        {"observation": torch.randn(4, obs_dim, device=agent.device)},
        batch_size=[4],
        device=agent.device,
    )
    requested_keys = []

    def _sample(batch_size: int, required_keys):
        requested_keys.append(list(required_keys))
        batch = expert_data[:batch_size]
        return batch.select(*required_keys).clone()

    agent._set_test_expert_batch_sampler(_sample)
    assert agent._latent_learner is not None

    metrics = agent._latent_learner.update(
        TensorDict({}, batch_size=[0], device=agent.device)
    )

    assert requested_keys == [["observation"]]
    assert "ipmd/latent_recon_loss" in metrics
    assert metrics["ipmd/latent_recon_loss"] >= 0.0


def test_ipmd_rejects_removed_posterior_command_source_aliases() -> None:
    """Removed command-source aliases should now fail at construction-time validation."""
    rlopt = _rlopt()

    assert rlopt.IPMD._normalize_command_source("posterior") == "posterior"
    with pytest.raises(ValueError, match="Unsupported IPMD command_source"):
        rlopt.IPMD._normalize_command_source("rollout_posterior")
    with pytest.raises(ValueError, match="Unsupported IPMD command_source"):
        rlopt.IPMD._normalize_command_source("expert_posterior")


def test_ipmd_accepts_configured_posterior_keys_present_in_obs_spec() -> None:
    """Posterior keys only need to exist in the observation spec at construction time."""
    rlopt = _rlopt()
    cfg = rlopt.IPMDRLOptConfig()
    cfg.env.env_name = "Pendulum-v1"
    cfg.env.device = "cpu"
    cfg.device = "cpu"
    cfg.collector.frames_per_batch = 4
    cfg.collector.total_frames = 4
    cfg.collector.init_random_frames = 0
    cfg.loss.mini_batch_size = 2
    cfg.compile.compile = False
    _apply_obs_input_keys(cfg)
    cfg.ipmd.latent_learning.posterior_input_keys = [("policy", "observation")]

    env = rlopt.make_parallel_env(cfg)
    spec = env.observation_spec.clone()
    spec.set(("policy", "observation"), spec["observation"].clone())
    env.observation_spec = spec

    agent = rlopt.IPMD(env, cfg, logger=None)
    assert agent._posterior_obs_keys == [("policy", "observation")]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
