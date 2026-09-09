# ruff: noqa: E402 -- filter upstream import deprecations before importing TorchRL.
"""Normalization must not change the policy during PPO loss/KL evaluation."""

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
import torch

from rlopt.agent.ipmd.ipmd import IPMD, IPMDRLOptConfig
from rlopt.agent.ppo.ppo import RunningMeanStdCatInputs
from rlopt.env_utils import make_parallel_env


def test_frozen_forward_keeps_likelihood_and_gradients():
    torch.manual_seed(0)
    norm = RunningMeanStdCatInputs(torch.nn.Linear(3, 2), 3, update_on_forward=False)
    x = torch.randn(12, 3) + 2
    norm.train()
    first = norm(x)
    actions = first.detach() + torch.randn_like(first)
    old_logp = torch.distributions.Normal(first.detach(), 1).log_prob(actions).sum(-1)
    for _ in range(6):
        mean = norm(x)
        logp = torch.distributions.Normal(mean, 1).log_prob(actions).sum(-1)
        torch.testing.assert_close(logp, old_logp, rtol=0, atol=0)
    assert norm.count.item() == 1
    (-logp.mean()).backward()
    assert norm.module.weight.grad.abs().sum() > 0
    norm._update(x)
    assert norm.count.item() == 13
    assert not torch.equal(norm(x), first)


def test_historical_forward_updates_remain_default():
    norm = RunningMeanStdCatInputs(torch.nn.Identity(), 3).train()
    norm(torch.ones(4, 3))
    assert norm.count.item() == 5


def test_ipmd_updates_once_after_optimization_and_before_save(tmp_path):
    cfg = IPMDRLOptConfig()
    cfg.env.env_name = "Pendulum-v1"
    cfg.env.device = cfg.device = "cpu"
    cfg.collector.frames_per_batch = 4
    cfg.collector.total_frames = 12
    cfg.replay_buffer.size = 4
    cfg.loss.mini_batch_size = 4
    cfg.loss.epochs = 3
    cfg.compile.compile = False
    cfg.logger.backend = ""
    cfg.logger.log_dir = str(tmp_path)
    cfg.save_interval = 4
    cfg.policy.input_keys = ["observation"]
    cfg.policy.num_cells = [8]
    cfg.policy.normalize_input = True
    cfg.value_function.input_keys = ["observation"]
    cfg.value_function.num_cells = [8]
    cfg.value_function.normalize_input = True
    cfg.ipmd.use_latent_command = False
    cfg.ipmd.reward_input_keys = ["observation"]
    cfg.ipmd.reward_loss_coeff = 0
    cfg.ipmd.reward_l2_coeff = cfg.ipmd.reward_grad_penalty_coeff = (
        cfg.ipmd.reward_logit_reg_coeff
    ) = 0
    cfg.ipmd.reward_param_weight_decay_coeff = cfg.ipmd.bc_coef = (
        cfg.ipmd.rollout_bc_coef
    ) = cfg.ipmd.diversity_bonus_coeff = 0
    cfg.ppo.update_normalizers_after_rollout = True
    cfg.optim.scheduler = "adaptive"
    env = make_parallel_env(cfg)
    agent = IPMD(env, cfg)
    norms = [
        m
        for net in (agent.policy, agent.value_function)
        for m in net.modules()
        if isinstance(m, RunningMeanStdCatInputs)
    ]
    assert len(norms) == 2
    initial = [m.count.clone() for m in norms]
    original_iterate = agent.iterate
    calls = []

    def checked_iterate(iteration, metadata):
        before = [{k: v.clone() for k, v in m.named_buffers()} for m in norms]
        original_iterate(iteration, metadata)
        for m, saved in zip(norms, before, strict=False):
            for k, v in m.named_buffers():
                torch.testing.assert_close(v, saved[k], rtol=0, atol=0)
        calls.append(metadata.frames_processed)

    agent.iterate = checked_iterate
    agent.train()
    assert calls == [4, 8, 12]
    for m, start in zip(norms, initial, strict=False):
        torch.testing.assert_close(m.count, start + 12, rtol=0, atol=0)
    snapshots = list(tmp_path.rglob("model_step_12.pt"))
    assert len(snapshots) == 1
    saved = torch.load(snapshots[0], weights_only=False)
    for key in ("policy_state_dict", "value_state_dict"):
        counts = [v for k, v in saved[key].items() if k.endswith(".count")]
        assert len(counts) == 1
        assert counts[0].item() == 13
