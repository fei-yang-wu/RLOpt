"""PPO owns the on-policy loop; IPMD with reward learning off IS that loop.

The strongest statement of that contract is bit-identity: given the same
initial weights and the same seed, ``PPO.train()`` and ``IPMD.train()`` with
every reward/BC/latent term off must end on the same weights, the same running
input statistics and the same learning rates. The two loops used to be separate
code and drifted; these tests are what keeps them from doing that again.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from rlopt.agent.ipmd.ipmd import IPMD, IPMDRLOptConfig
from rlopt.agent.ppo.ppo import (
    PPO,
    PPOConfig,
    PPORLOptConfig,
    RunningMeanStdCatInputs,
)
from rlopt.env_utils import make_parallel_env

FRAMES_PER_BATCH = 64
ITERATIONS = 4

TUNED_UPDATE_KEYS = {
    "actor_learning_rate": 3.0e-4,
    "critic_learning_rate": 1.0e-3,
    "critic_lr_schedule": "linear",
    "critic_lr_final": 1.0e-4,
    "first_layer_bias_lr_scale": 2.0,
    "critic_extra_epochs": 1,
    "critic_extra_minibatches": 2,
}


def _configure(cfg, tmp_path: Path) -> None:
    cfg.env.env_name = "Pendulum-v1"
    cfg.env.device = cfg.device = "cpu"
    cfg.seed = 0
    cfg.collector.frames_per_batch = FRAMES_PER_BATCH
    cfg.collector.total_frames = FRAMES_PER_BATCH * ITERATIONS
    cfg.collector.init_random_frames = 0
    cfg.replay_buffer.size = FRAMES_PER_BATCH
    cfg.loss.mini_batch_size = FRAMES_PER_BATCH // 2
    cfg.loss.epochs = 2
    cfg.compile.compile = False
    cfg.logger.backend = ""
    cfg.logger.log_dir = str(tmp_path)
    cfg.save_interval = 0
    cfg.trainer.progress_bar = False
    cfg.policy.input_keys = ["observation"]
    cfg.policy.num_cells = [16, 16]
    cfg.policy.normalize_input = True
    cfg.value_function.input_keys = ["observation"]
    cfg.value_function.num_cells = [16, 16]
    cfg.value_function.normalize_input = True
    cfg.optim.scheduler = "adaptive"
    cfg.optim.kl_adapt_step = "iteration"
    cfg.optim.desired_kl = 1.0e-4
    cfg.optim.min_lr = 1.0e-5
    cfg.optim.max_lr = 1.0e-2
    cfg.optim.weight_decay = 1.0e-2
    cfg.optim.max_grad_norm = 0.5
    cfg.ppo.normalize_advantage_global = True
    cfg.ppo.entropy_coeff = 0.0


def _ppo(tmp_path: Path, update_keys: dict | None = None) -> PPO:
    cfg = PPORLOptConfig()
    _configure(cfg, tmp_path)
    for key, value in (update_keys or {}).items():
        setattr(cfg.ppo, key, value)
    return PPO(make_parallel_env(cfg), cfg)


def _ipmd(tmp_path: Path, update_keys: dict | None = None) -> IPMD:
    cfg = IPMDRLOptConfig()
    _configure(cfg, tmp_path)
    for key, value in (update_keys or {}).items():
        setattr(cfg.ipmd, key, value)
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


def _state(agent: PPO) -> dict[str, torch.Tensor]:
    state = {f"policy.{k}": v for k, v in agent.policy.state_dict().items()}
    state.update(
        {f"value.{k}": v for k, v in agent.value_function.state_dict().items()}
    )
    return {key: value.detach().clone() for key, value in state.items()}


def _groups(agent: PPO) -> list[tuple]:
    return [
        (
            group.get("name"),
            float(group["lr"]),
            float(group["weight_decay"]),
            bool(group.get("adaptive_lr", True)),
            len(group["params"]),
        )
        for group in agent.optim.param_groups
    ]


def _train_from_same_start(tmp_path: Path, update_keys: dict | None):
    ppo = _ppo(tmp_path / "ppo", update_keys)
    ipmd = _ipmd(tmp_path / "ipmd", update_keys)
    # The contract is about the update, so pin the start rather than rely on the
    # two constructors drawing the same initial weights.
    ipmd.policy.load_state_dict(ppo.policy.state_dict())
    ipmd.value_function.load_state_dict(ppo.value_function.state_dict())
    start = _state(ppo)
    torch.manual_seed(1234)
    ppo.train()
    torch.manual_seed(1234)
    ipmd.train()
    return ppo, ipmd, start


@pytest.mark.parametrize(
    "update_keys",
    [None, TUNED_UPDATE_KEYS],
    ids=["single_group", "split_lr_schedule_extra_critic"],
)
def test_ppo_is_bit_identical_to_ipmd_with_reward_learning_off(tmp_path, update_keys):
    ppo, ipmd, start = _train_from_same_start(tmp_path, update_keys)

    ppo_state, ipmd_state = _state(ppo), _state(ipmd)
    assert ppo_state.keys() == ipmd_state.keys()
    for key, value in ppo_state.items():
        torch.testing.assert_close(
            value, ipmd_state[key], rtol=0, atol=0, msg=f"{key} diverged"
        )
    assert _groups(ppo) == _groups(ipmd)
    # Guard the test itself: identical because both trained, not because neither did.
    assert any(
        not torch.equal(value, start[key])
        for key, value in ppo_state.items()
        if key.endswith("weight")
    )


def test_parity_covers_the_running_input_statistics(tmp_path):
    ppo, ipmd, _ = _train_from_same_start(tmp_path, TUNED_UPDATE_KEYS)
    for agent in (ppo, ipmd):
        counts = [
            module.count.item()
            for network in (agent.policy, agent.value_function)
            for module in network.modules()
            if isinstance(module, RunningMeanStdCatInputs)
        ]
        # Prior count 1, then each collected frame exactly once: no double count
        # from the forward pass, GAE, the KL diagnostic or the extra critic steps.
        assert counts == [1.0 + FRAMES_PER_BATCH * ITERATIONS] * 2


def test_split_groups_match_the_ipmd_recipe_shape(tmp_path):
    agent = _ppo(tmp_path, TUNED_UPDATE_KEYS)
    groups = {group.get("name"): group for group in agent.optim.param_groups}
    assert set(groups) == {
        "actor",
        "critic",
        "actor_log_std",
        "actor_first_bias",
        "critic_first_bias",
    }
    # The KL rule owns the actor only, and AdamW must not pull log_std to 0.
    assert groups["actor"]["adaptive_lr"] is True
    assert groups["actor_log_std"]["adaptive_lr"] is True
    assert groups["critic"]["adaptive_lr"] is False
    assert groups["actor_log_std"]["weight_decay"] == 0.0
    assert groups["actor"]["weight_decay"] == 1.0e-2
    assert groups["critic"]["weight_decay"] == 1.0e-2


def test_defaults_keep_the_single_group_optimizer(tmp_path):
    agent = _ppo(tmp_path)
    assert len(agent.optim.param_groups) == 1
    assert agent.optim.param_groups[0].get("name") is None
    assert agent._max_grad_norm == 0.5


def test_kl_rule_leaves_the_critic_rate_alone_under_split_rates(tmp_path):
    agent = _ppo(
        tmp_path,
        {"actor_learning_rate": 3.0e-4, "critic_learning_rate": 1.0e-3},
    )
    agent._maybe_adjust_lr(torch.tensor(1.0), agent.config.optim)
    rates = {group["name"]: group["lr"] for group in agent.optim.param_groups}
    assert rates["actor"] < 3.0e-4
    assert rates["critic"] == 1.0e-3


def test_critic_extra_steps_leave_the_actor_untouched(tmp_path):
    agent = _ppo(
        tmp_path,
        {
            "actor_learning_rate": 3.0e-4,
            "critic_learning_rate": 1.0e-3,
            "critic_extra_epochs": 2,
            "critic_extra_minibatches": 2,
        },
    )
    rollout = next(iter(agent.collector))
    with torch.no_grad():
        rollout = agent.adv_module(rollout)
    actor_before = {k: v.clone() for k, v in agent.policy.state_dict().items()}
    critic_before = {k: v.clone() for k, v in agent.value_function.state_dict().items()}
    metrics = agent._run_critic_extra_updates(rollout)
    agent.collector.shutdown()
    assert metrics["critic_extra_steps"] == 4.0
    for key, value in agent.policy.state_dict().items():
        torch.testing.assert_close(value, actor_before[key], rtol=0, atol=0)
    assert any(
        not torch.equal(value, critic_before[key])
        for key, value in agent.value_function.state_dict().items()
    )


def test_ppo_checkpoint_carries_the_frame_budget_and_scheduler(tmp_path):
    cfg = PPORLOptConfig()
    _configure(cfg, tmp_path)
    cfg.optim.scheduler = "linearlr"
    cfg.optim.scheduler_kwargs = {
        "start_factor": 1.0,
        "end_factor": 0.1,
        "total_iters": 100,
    }
    agent = PPO(make_parallel_env(cfg), cfg)
    agent.optim.step()
    for _ in range(7):
        agent.lr_scheduler.step()
    agent.save_model(path=tmp_path / "ckpt", step=4096)
    agent.collector.shutdown()
    checkpoint = tmp_path / "ckpt" / "model_step_4096.pt"
    saved = torch.load(checkpoint, weights_only=False)
    assert saved["cumulative_env_frames"] == 4096
    assert saved["lr_scheduler_state_dict"]["last_epoch"] == 7

    restored_cfg = PPORLOptConfig()
    _configure(restored_cfg, tmp_path / "restored")
    restored_cfg.optim.scheduler = "linearlr"
    restored_cfg.optim.scheduler_kwargs = dict(cfg.optim.scheduler_kwargs)
    restored_cfg.collector.total_frames = 1 << 20
    restored = PPO(make_parallel_env(restored_cfg), restored_cfg)
    restored.load_model(str(checkpoint))
    assert restored.lr_scheduler.last_epoch == 7
    metadata = restored.init_metadata()
    assert metadata.frames_processed == 4096
    metadata.progress_bar.close()
    restored.collector.shutdown()


def test_ppo_loads_a_checkpoint_from_before_the_split_groups(tmp_path):
    legacy = _ppo(tmp_path / "legacy")
    legacy.save_model(path=tmp_path / "legacy.pt")
    legacy.collector.shutdown()
    split = _ppo(
        tmp_path / "split",
        {"actor_learning_rate": 3.0e-4, "critic_learning_rate": 1.0e-3},
    )
    split.load_model(str(tmp_path / "legacy.pt"))
    split.collector.shutdown()
    for key, value in legacy.policy.state_dict().items():
        torch.testing.assert_close(
            value, split.policy.state_dict()[key], rtol=0, atol=0
        )
    # The live optimizer keeps its own groups rather than raising.
    assert {group["name"] for group in split.optim.param_groups} >= {"actor", "critic"}


def test_compile_wraps_the_loss_not_the_update(tmp_path):
    cfg = PPORLOptConfig()
    _configure(cfg, tmp_path)
    cfg.compile.compile = True
    cfg.compile.compile_mode = "default"
    agent = PPO(make_parallel_env(cfg), cfg)
    assert type(agent.loss_module).__name__ == "OptimizedModule"
    assert type(agent.adv_module).__name__ == "OptimizedModule"
    # The optimizer's Python-float rates would guard-fail on every KL adaptation.
    assert agent.update.__func__ is PPO.update
    agent.collector.shutdown()


def test_ppo_aborts_on_a_nonfinite_policy(tmp_path):
    agent = _ppo(tmp_path)
    next(iter(agent.policy.parameters())).data.fill_(float("nan"))
    iteration = type("Iteration", (), {"metrics": {}})()
    metadata = type("Metadata", (), {"frames_processed": 64})()
    with pytest.raises(RuntimeError, match="non-finite"):
        agent._abort_on_nonfinite(iteration, metadata)
    agent.collector.shutdown()


def test_ppo_row_carries_the_env_reward_series(tmp_path):
    agent = _ppo(tmp_path)
    rows: list[dict] = []
    original = agent.log_training_metrics

    def capture(metrics, **kwargs):
        rows.append(dict(metrics))
        return original(metrics, **kwargs)

    agent.log_training_metrics = capture
    agent.train()
    assert len(rows) == ITERATIONS
    for row in rows:
        assert row["train/env_reward_mean"] == row["train/step_reward_mean"]


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("actor_learning_rate", -1.0, r"ppo\.actor_learning_rate must be > 0"),
        ("critic_learning_rate", 0.0, r"ppo\.critic_learning_rate must be > 0"),
        ("first_layer_bias_lr_scale", 0.0, r"ppo\.first_layer_bias_lr_scale"),
        ("critic_lr_schedule", "exponential", r"ppo\.critic_lr_schedule must be"),
        ("critic_extra_epochs", -1, r"ppo\.critic_extra_epochs must be >= 0"),
    ],
)
def test_ppo_config_rejects_bad_update_keys(key, value, match):
    cfg = PPOConfig()
    setattr(cfg, key, value)
    with pytest.raises(ValueError, match=match):
        cfg.validate_update_schedule()


def test_ppo_config_requires_the_rates_together():
    cfg = PPOConfig(actor_learning_rate=1.0e-3)
    with pytest.raises(ValueError, match="must be configured together"):
        cfg.validate_update_schedule()


@pytest.mark.parametrize(
    "keys",
    [
        {"critic_lr_schedule": "linear", "critic_lr_final": 1.0e-5},
        {"critic_extra_epochs": 1},
        {"first_layer_bias_lr_scale": 2.0},
    ],
)
def test_ppo_config_refuses_schedules_without_split_rates(keys):
    cfg = PPOConfig(**keys)
    with pytest.raises(ValueError, match="learning rate"):
        cfg.validate_update_schedule()


def test_ppo_agent_validates_before_building_the_optimizer(tmp_path):
    cfg = PPORLOptConfig()
    _configure(cfg, tmp_path)
    cfg.ppo.actor_learning_rate = 1.0e-3
    with pytest.raises(ValueError, match="must be configured together"):
        PPO(make_parallel_env(cfg), cfg)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("actor_learning_rate", 1.0e-3),
        ("critic_lr_schedule", "linear"),
        ("critic_extra_epochs", 1),
    ],
)
def test_ipmd_refuses_the_update_key_it_would_ignore(tmp_path, key, value):
    cfg = IPMDRLOptConfig()
    _configure(cfg, tmp_path)
    setattr(cfg.ppo, key, value)
    cfg.ipmd.use_latent_command = False
    cfg.ipmd.reward_input_keys = ["observation"]
    with pytest.raises(ValueError, match=rf"ipmd\.{key}"):
        IPMD(make_parallel_env(cfg), cfg)
