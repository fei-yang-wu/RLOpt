"""Tests for IPMD algorithm components."""

from __future__ import annotations

import warnings
from functools import lru_cache
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict


@lru_cache(maxsize=1)
def _rlopt() -> SimpleNamespace:
    warnings.filterwarnings(
        "ignore",
        message="Creating .* which inherits from WeightUpdaterBase is deprecated.*",
        category=DeprecationWarning,
        append=False,
    )
    from rlopt.agent import IPMD, IPMDRLOptConfig
    from rlopt.config_base import NetworkConfig
    from rlopt.env_utils import make_parallel_env

    return SimpleNamespace(
        IPMD=IPMD,
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


def _make_env_reward_only_ipmd_agent():
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

    env = rlopt.make_parallel_env(cfg)
    return rlopt.IPMD(env, cfg, logger=None), env


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
    with pytest.raises(ValueError, match="requires .*reward_input_type='s'"):
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
