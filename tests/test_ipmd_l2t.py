"""Focused tests for privileged-teacher IPMD distillation."""

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
    warnings.filterwarnings(
        "ignore",
        message=(
            "`torch.jit.script_method` is deprecated. Please switch to "
            "`torch.compile` or `torch.export`."
        ),
        category=DeprecationWarning,
    )
    from rlopt.agent import IPMD, IPMDL2T, IPMDL2TRLOptConfig
    from rlopt.agent.ipmd.ipmd import IPMD as IPMDBase
    from rlopt.env_utils import make_parallel_env

    return SimpleNamespace(
        IPMD=IPMD,
        IPMDBase=IPMDBase,
        IPMDL2T=IPMDL2T,
        IPMDL2TRLOptConfig=IPMDL2TRLOptConfig,
        make_parallel_env=make_parallel_env,
    )


def _config(*, split_learning_rates: bool = True):
    cfg = _rlopt().IPMDL2TRLOptConfig()
    cfg.env.env_name = "Pendulum-v1"
    cfg.env.device = "cpu"
    cfg.device = "cpu"
    cfg.collector.frames_per_batch = 4
    cfg.collector.total_frames = 4
    cfg.replay_buffer.size = 64
    cfg.loss.mini_batch_size = 4
    cfg.compile.compile = False
    cfg.logger.backend = ""
    cfg.policy.input_keys = ["observation"]
    assert cfg.value_function is not None
    cfg.value_function.input_keys = ["observation"]
    cfg.ipmd_l2t.student_policy.input_keys = ["observation"]
    cfg.ipmd.use_latent_command = False
    cfg.ipmd.reward_input_keys = ["observation"]
    cfg.ipmd.reward_loss_coeff = 0.0
    cfg.ipmd.reward_l2_coeff = 0.0
    cfg.ipmd.reward_grad_penalty_coeff = 0.0
    cfg.ipmd.reward_logit_reg_coeff = 0.0
    cfg.ipmd.reward_param_weight_decay_coeff = 0.0
    cfg.ipmd.bc_coef = 0.0
    cfg.ipmd.rollout_bc_coef = 0.0
    cfg.ipmd.diversity_bonus_coeff = 0.0
    cfg.optim.max_grad_norm = 0.5
    cfg.ipmd_l2t.student_max_grad_norm = 0.25
    if split_learning_rates:
        cfg.ipmd.actor_learning_rate = 2.0e-5
        cfg.ipmd.critic_learning_rate = 1.0e-3
        cfg.optim.scheduler = "adaptive"
        cfg.optim.min_lr = 1.0e-5
        cfg.optim.max_lr = 2.0e-4
    return cfg


def _make_agent(*, split_learning_rates: bool = True):
    rlopt = _rlopt()
    cfg = _config(split_learning_rates=split_learning_rates)
    env = rlopt.make_parallel_env(cfg)
    return rlopt.IPMDL2T(env, cfg, logger=None), env


def _policy_batch(agent, batch_size: int = 4) -> TensorDict:
    obs_dim = agent.env.observation_spec["observation"].shape[-1]
    act_dim = agent.env.action_spec.shape[-1]
    batch = TensorDict(
        {
            "observation": torch.randn(batch_size, obs_dim),
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
        return agent.adv_module(batch)


def _state_dict_clone(module) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in module.state_dict().items()}


def _assert_state_dict_equal(module, expected: dict[str, torch.Tensor]) -> None:
    actual = module.state_dict()
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        assert torch.equal(actual[key], value), key


def test_collection_is_teacher_only_and_student_is_deployment_policy() -> None:
    agent, env = _make_agent()
    try:
        collector_param_ids = {
            id(param) for param in agent.collector_policy.parameters()
        }
        teacher_param_ids = {id(param) for param in agent.teacher_policy.parameters()}
        student_param_ids = {id(param) for param in agent.student_policy.parameters()}
        assert collector_param_ids == teacher_param_ids
        assert collector_param_ids.isdisjoint(student_param_ids)
        assert agent.deployment_policy is agent.student_policy
        assert agent.teacher_policy is not agent.student_policy
        assert agent._student_obs_keys == ["observation"]
        monitored_ids = {id(param) for _, param in agent._parameter_monitor}
        assert {id(param) for param in agent.student_policy.parameters()}.issubset(
            monitored_ids
        )
    finally:
        env.close()


def test_teacher_keys_must_exactly_match_critic_keys() -> None:
    rlopt = _rlopt()
    cfg = _config()
    env = rlopt.make_parallel_env(cfg)
    try:
        assert cfg.value_function is not None
        cfg.value_function.input_keys = [("critic", "privileged")]
        with pytest.raises(ValueError, match="exactly match"):
            rlopt.IPMDL2T(env, cfg, logger=None)
    finally:
        env.close()


def test_student_architecture_must_match_teacher_actor() -> None:
    rlopt = _rlopt()
    cfg = _config()
    cfg.ipmd_l2t.student_policy.num_cells = [32, 32]
    env = rlopt.make_parallel_env(cfg)
    try:
        with pytest.raises(ValueError, match="match the teacher actor architecture"):
            rlopt.IPMDL2T(env, cfg, logger=None)
    finally:
        env.close()


def test_student_backward_detaches_teacher_action_and_stops_teacher_gradients() -> None:
    agent, env = _make_agent()
    try:
        obs_dim = env.observation_spec["observation"].shape[-1]
        act_dim = env.action_spec.shape[-1]
        target = torch.randn(4, act_dim, requires_grad=True)
        observation = torch.randn(4, obs_dim, requires_grad=True)
        batch = TensorDict(
            {
                "observation": observation,
                "action": target,
            },
            batch_size=[4],
        )
        teacher_before = _state_dict_clone(agent.teacher_policy)
        student_before = _state_dict_clone(agent.student_policy)

        agent.optim.zero_grad(set_to_none=True)
        metrics = agent._backward_student_imitation(batch)

        assert target.grad is None
        assert observation.grad is None
        assert all(param.grad is None for param in agent.teacher_policy.parameters())
        assert any(
            param.grad is not None for param in agent.student_policy.parameters()
        )
        assert all(torch.isfinite(value) for value in metrics.values())

        agent.optim.step()
        _assert_state_dict_equal(agent.teacher_policy, teacher_before)
        assert any(
            not torch.equal(value, student_before[key])
            for key, value in agent.student_policy.state_dict().items()
        )
    finally:
        env.close()


def test_student_optimizer_is_nonadaptive_and_parameter_sets_are_disjoint() -> None:
    agent, env = _make_agent()
    try:
        groups = {group["name"]: group for group in agent.optim.param_groups}
        assert groups["actor"]["lr"] == pytest.approx(2.0e-5)
        assert groups["critic"]["lr"] == pytest.approx(1.0e-3)
        assert groups["student"]["lr"] == pytest.approx(2.0e-5)
        assert groups["student"]["adaptive_lr"] is False

        teacher_ids = {id(param) for param in agent._grad_clip_params}
        student_ids = {id(param) for param in agent._student_grad_clip_params}
        assert teacher_ids
        assert student_ids
        assert teacher_ids.isdisjoint(student_ids)

        agent._maybe_adjust_lr(torch.tensor(1.0e-4), agent.config.optim)
        assert groups["actor"]["lr"] == pytest.approx(3.0e-5)
        assert groups["critic"]["lr"] == pytest.approx(1.0e-3)
        assert groups["student"]["lr"] == pytest.approx(2.0e-5)
    finally:
        env.close()


def test_update_clips_roles_separately_and_emits_finite_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent, env = _make_agent()
    calls: list[tuple[set[int], float]] = []
    import rlopt.agent.ipmd.ipmd_l2t as l2t_module

    original_clip = l2t_module.clip_grad_norm_

    def _recording_clip(parameters, max_norm):
        params = list(parameters)
        calls.append(({id(param) for param in params}, float(max_norm)))
        return original_clip(params, max_norm)

    monkeypatch.setattr(l2t_module, "clip_grad_norm_", _recording_clip)
    try:
        student_before = _state_dict_clone(agent.student_policy)
        batch = _policy_batch(agent)
        loss, update_idx = agent.update(
            batch,
            0,
            TensorDict({}, batch_size=[4]),
            torch.tensor(0.0),
        )

        assert update_idx == 1
        assert len(calls) == 2
        assert calls[0][0] == {id(param) for param in agent._grad_clip_params}
        assert calls[0][1] == pytest.approx(0.5)
        assert calls[1][0] == {id(param) for param in agent._student_grad_clip_params}
        assert calls[1][1] == pytest.approx(0.25)
        for key in (
            "loss_student_imitation",
            "student_action_mae",
            "student_action_rmse",
            "student_action_abs_mean",
            "teacher_action_abs_mean",
            "student_grad_norm",
        ):
            assert key in loss
            assert torch.isfinite(loss[key])
        assert any(
            not torch.equal(value, student_before[key])
            for key, value in agent.student_policy.state_dict().items()
        )
    finally:
        env.close()


def test_latent_command_is_mirrored_tensor_identically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rlopt = _rlopt()
    teacher_key = ("critic", "latent_command")
    student_key = ("policy", "latent_command")
    latent = torch.randn(3, 8)

    def _fake_teacher_injection(self, td):
        td.set(self._latent_key, latent)

    monkeypatch.setattr(
        rlopt.IPMDBase,
        "_inject_latent_command",
        _fake_teacher_injection,
    )
    agent = object.__new__(rlopt.IPMDL2T)
    agent._use_latent_command = True
    agent._latent_key = teacher_key
    agent._student_latent_key = student_key
    td = TensorDict({}, batch_size=[3])

    rlopt.IPMDL2T._inject_latent_command(agent, td)

    teacher_value = td.get(teacher_key)
    student_value = td.get(student_key)
    assert isinstance(teacher_value, torch.Tensor)
    assert isinstance(student_value, torch.Tensor)
    assert torch.equal(teacher_value, student_value)
    assert teacher_value.data_ptr() == student_value.data_ptr()


def test_checkpoint_round_trip_and_standard_ipmd_deployment_load(tmp_path) -> None:
    rlopt = _rlopt()
    agent, env = _make_agent()
    restored = ordinary = None
    restored_env = ordinary_env = None
    try:
        student_state = _state_dict_clone(agent.student_policy)
        teacher_state = _state_dict_clone(agent.teacher_policy)
        checkpoint_path = tmp_path / "ipmd_l2t.pt"
        agent.save_model(checkpoint_path)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert payload["checkpoint_metadata"] == {
            "algorithm": "IPMD_L2T",
            "primary_policy_role": "student",
        }
        assert "teacher_policy_state_dict" in payload

        restored_cfg = _config()
        restored_env = rlopt.make_parallel_env(restored_cfg)
        restored = rlopt.IPMDL2T(restored_env, restored_cfg, logger=None)
        restored.load_model(str(checkpoint_path))
        _assert_state_dict_equal(restored.student_policy, student_state)
        _assert_state_dict_equal(restored.teacher_policy, teacher_state)

        ordinary_cfg = _config()
        ordinary_env = rlopt.make_parallel_env(ordinary_cfg)
        ordinary = rlopt.IPMD(ordinary_env, ordinary_cfg, logger=None)
        ordinary.load_model(str(checkpoint_path))
        _assert_state_dict_equal(ordinary.policy, student_state)

        plain_path = tmp_path / "ordinary_ipmd.pt"
        torch.save({"policy_state_dict": teacher_state}, plain_path)
        with pytest.raises(ValueError, match="teacher_policy_state_dict"):
            restored.load_model(str(plain_path))
    finally:
        for close_env in (ordinary_env, restored_env, env):
            if close_env is not None:
                close_env.close()
