# ruff: noqa: E402 -- third-party deprecation filters must precede TorchRL imports.
"""DAgger handoff, frozen teacher, and PPO initialization/resume boundaries."""

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
from tensordict.nn import TensorDictModule
from torchrl.modules import IndependentNormal, ProbabilisticActor

from rlopt.agent.ipmd.dagger_config import DAggerConfig
from rlopt.agent.ipmd.ipmd import IPMD, IPMDRLOptConfig
from rlopt.agent.ipmd.ipmd_dagger import IPMDDagger, TeacherMixturePolicy
from rlopt.env_utils import make_parallel_env


def test_exponential_schedule():
    c = DAggerConfig()
    assert c.teacher_rate(0) == 1
    assert c.teacher_rate(200_000_000) == 1
    assert c.teacher_rate(300_000_000) == pytest.approx(0.1)
    assert c.teacher_rate(399_999_999) == pytest.approx(0.01)
    assert c.teacher_rate(400_000_000) == 0
    assert c.teacher_rate(500_000_000) == 0


class Head(torch.nn.Module):
    def __init__(self, multiplier):
        super().__init__()
        self.multiplier = multiplier

    def forward(self, x):
        return x * self.multiplier, torch.ones_like(x)


def test_student_control_still_gets_teacher_labels():
    def actor(multiplier):
        return ProbabilisticActor(
            TensorDictModule(Head(multiplier), ["observation"], ["loc", "scale"]),
            in_keys=["loc", "scale"],
            distribution_class=IndependentNormal,
        )

    mixture = TeacherMixturePolicy(actor(2), actor(3)).eval()
    for rate, multiplier in [(0.0, 2), (1.0, 3)]:
        mixture.teacher_rate = rate
        td = mixture(TensorDict({"observation": torch.tensor([[4.0], [5.0]])}, [2]))
        torch.testing.assert_close(td["action"], td["observation"] * multiplier)
        torch.testing.assert_close(td["teacher_action"], td["observation"] * 3)
        assert not td["teacher_action"].requires_grad


def config(path):
    c = IPMDRLOptConfig()
    c.env.env_name = "Pendulum-v1"
    c.env.device = c.device = "cpu"
    c.collector.frames_per_batch = 4
    c.collector.total_frames = 12
    c.replay_buffer.size = 32
    c.loss.mini_batch_size = 4
    c.compile.compile = False
    c.logger.backend = ""
    c.logger.log_dir = str(path)
    c.save_interval = 4
    c.policy.input_keys = ["observation"]
    c.policy.num_cells = [8]
    c.policy.normalize_input = True
    c.value_function.input_keys = ["observation"]
    c.value_function.num_cells = [8]
    c.ipmd.use_latent_command = False
    c.ipmd.reward_input_keys = ["observation"]
    c.ipmd.reward_loss_coeff = 0.0
    c.ipmd.reward_l2_coeff = 0.0
    c.ipmd.reward_grad_penalty_coeff = 0.0
    c.ipmd.reward_logit_reg_coeff = 0.0
    c.ipmd.reward_param_weight_decay_coeff = 0.0
    c.ipmd.bc_coef = 0.0
    c.ipmd.rollout_bc_coef = 0.0
    c.ipmd.diversity_bonus_coeff = 0.0
    return c


def test_training_export_and_resume(tmp_path):
    cfg = config(tmp_path / "teacher")
    env = make_parallel_env(cfg)
    teacher = IPMD(env, cfg)
    teacher_path = tmp_path / "teacher.pt"
    teacher.save_model(teacher_path, step=None)
    env.close()
    cfg = config(tmp_path / "dagger")
    cfg.policy.num_cells = [16, 16]
    cfg.dagger = DAggerConfig(
        enabled=True,
        teacher_checkpoint=str(teacher_path),
        teacher_num_cells=[8],
        teacher_only_frames=4,
        handoff_end_frames=12,
        batch_size=4,
        updates_per_rollout=2,
        replay_capacity=16,
        initialization_path=str(tmp_path / "init.pt"),
    )
    env = make_parallel_env(cfg)
    agent = IPMDDagger(env, cfg)
    original_teacher = {
        k: v.clone() for k, v in agent._dagger_policy.teacher.state_dict().items()
    }
    original_critic = {
        k: v.clone() for k, v in agent.value_function.state_dict().items()
    }
    agent.train()
    for k, v in agent._dagger_policy.teacher.state_dict().items():
        torch.testing.assert_close(v, original_teacher[k], rtol=0, atol=0)
    for k, v in agent.value_function.state_dict().items():
        torch.testing.assert_close(v, original_critic[k], rtol=0, atol=0)
    out = torch.load(tmp_path / "init.pt", weights_only=False)
    assert out["cumulative_env_frames"] == 0
    assert out["distillation_env_frames"] == 12
    assert "optimizer_state_dict" not in out
    assert "dagger_state" not in out
    snapshots = list((tmp_path / "dagger").rglob("model_step_*.pt"))
    assert snapshots
    checkpoint = max(snapshots, key=lambda p: int(p.stem.split("_")[-1]))
    env2 = make_parallel_env(cfg)
    resumed = IPMDDagger(env2, cfg)
    resumed.load_model(str(checkpoint))
    resumed.validate_training()
    assert resumed._resume_frame_offset == 12
    assert len(resumed._dagger_replay) == 12
    assert resumed._dagger_optimizer.state_dict()["state"]
    env2.close()
    cfg.dagger.enabled = False
    env3 = make_parallel_env(cfg)
    ppo = IPMD(env3, cfg)
    ppo.load_model(str(tmp_path / "init.pt"))
    assert ppo._resume_frame_offset == 0
    assert not ppo.optim.state_dict()["state"]
    env3.close()
