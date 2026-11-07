"""The real ``WMBPTT`` and ``G1V3TrackingTask`` on a stand-in for the Isaac env.

No simulator and no dataset: the bank is a closed-form pair of clips, and the
env replays it with noise and random episode ends. The agent goes through its
real ``__init__`` and ``train``, so construction, the loop, logging and
checkpoints all run.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import Tensor

pytest.importorskip("imitation_experiments")

from imitation_experiments.teleop.g1_kinematics import G1_ISAACLAB_JOINT_NAMES
from imitation_experiments.world_model.g1_v3_math import (
    G1V3ReferenceLibrary,
    joint_limits,
)

from rlopt.agent.wm_bptt.agent import WMBPTT
from rlopt.agent.wm_bptt.config import WMBPTTConfig, WMBPTTRLOptConfig

G1_V3_TASK = "imitation_experiments.world_model.g1_v3_task:G1V3TrackingTask"
SMALL = {
    "wm_hidden": [64, 64, 64],
    "wm_batches_per_iter": 2,
    "wm_batch_size": 32,
    "wm_warmup_iters": 1,
    "horizon": 8,
    "imagine_batch": 32,
    "actor_hidden": [64, 64, 32, 32],
    "critic_hidden": [32, 32],
    "critic_epochs": 2,
    "critic_minibatches": 2,
    "replay_steps": 80,
    "min_replay_steps": 40,
    "log_interval_iters": 2,
    "eval_window": 8,
    "eval_batch": 16,
    "real_window_recent_steps": 8,
}
"""Sizes only: every value picks how much work a code path does, not which path."""


def action_scale(device: torch.device | str) -> Tensor:
    """A left-right symmetric stand-in for the joint_pos action scale (the env's is
    ``UNITREE_G1_29DOF_MIMIC_ACTION_SCALE``, which needs Isaac Lab to import)."""
    sides = ("left_", "right_")
    names = [next((n[len(s):] for s in sides if n.startswith(s)), n) for n in G1_ISAACLAB_JOINT_NAMES]
    return torch.tensor([0.1 + 0.02 * len(name) for name in names], device=device)


def synthetic_library(device: torch.device | str) -> G1V3ReferenceLibrary:
    """Two clips of a walking pose, in closed form (bit-stable, no RNG)."""
    device = torch.device(device)
    mid = joint_limits("cpu").mean(-1)
    rows, lengths = [], [90, 70]
    for clip, length in enumerate(lengths):
        t = torch.arange(length, dtype=torch.float32) * 0.02
        yaw = 0.2 * clip + 0.1 * t
        pos = torch.stack((0.8 * t, 0.1 * t * clip, 0.75 + 0.02 * torch.sin(4.0 * t)), -1)
        quat = torch.stack((torch.cos(yaw / 2), torch.zeros_like(t), torch.zeros_like(t), torch.sin(yaw / 2)), -1)
        phase = torch.arange(29, dtype=torch.float32) * 0.3 + clip
        q = mid + 0.15 * torch.sin(5.0 * t.unsqueeze(-1) + phase)
        rows.append(torch.cat((pos, quat, q), -1))
    return G1V3ReferenceLibrary.from_qpos(torch.cat(rows).to(device), lengths)


class FakeIsaacEnv:
    """Device-only env: reference rows plus noise, random terminations.

    Provides what ``WMBPTT`` and ``G1V3TrackingTask`` read: the ``wm`` group,
    ``step``, ``reset`` and the attributes the task unwraps.
    """

    def __init__(self, num_envs: int, device: torch.device | str) -> None:
        self.device = torch.device(device)
        self.n = int(num_envs)
        self.batch_size = torch.Size([self.n])
        self.v3_library = synthetic_library(self.device)
        self.step_dt = 0.02
        self.trajectory_manager = object()
        self.reward_manager = object()
        q0 = joint_limits("cpu").mean(-1).to(self.device).unsqueeze(0)
        robot = SimpleNamespace(
            data=SimpleNamespace(default_joint_pos=q0),
            find_joints=lambda names, **_kwargs: (list(range(29)), names),
        )
        self.scene = {"robot": robot}
        term = SimpleNamespace(_joint_names=list(G1_ISAACLAB_JOINT_NAMES), _scale=action_scale(self.device).expand(self.n, 29))
        self.action_manager = SimpleNamespace(get_term=lambda name: {"joint_pos": term}[name])
        # G1-v3's physics: one 0.02 s control step of four Newton substeps.
        self.cfg = SimpleNamespace(sim=SimpleNamespace(physics=SimpleNamespace(num_substeps=4)), decimation=1)

    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        lib, n, dev = self.v3_library, self.n, self.device
        self.rank = torch.randint(0, lib.length.numel(), (n,), device=dev)
        self.frame = (torch.rand(n, device=dev) * (lib.length[self.rank] - 40).float()).long()
        self.episode = torch.zeros(n, dtype=torch.long, device=dev)
        self.ctx = torch.zeros(n, 33, device=dev)
        self.ctx[:, 0] = 0.75
        state = lib.ref_state(self.rank, self.frame)
        self.hist_state = state[:, None].repeat(1, 10, 1).contiguous()
        self.hist_action = torch.zeros(n, 10, 29, device=dev)
        return self._obs(), {}

    def _obs(self) -> dict[str, Any]:
        n = self.n
        cursor = torch.stack((self.rank, self.frame, self.episode), -1).float()
        return {
            "wm": {
                "root_state": self.hist_state[..., :13].flatten(1),
                "joint_state": self.hist_state[..., 13:].flatten(1),
                "last_action": self.hist_action.flatten(1),
                "reference_cursor": cursor,
                "sim_context": self.ctx,
                "push_flag": torch.zeros(n, 1, device=self.device),
                "proprio_history": 0.01 * self.hist_state[:, :, :1].repeat(1, 1, 99).flatten(1),
            }
        }

    def step(self, action: Tensor) -> tuple[dict[str, Any], Tensor, Tensor, Tensor, dict[str, Any]]:
        lib, n, dev = self.v3_library, self.n, self.device
        self.frame = self.frame + 1
        self.episode = self.episode + 1
        clip_end = self.frame >= lib.length[self.rank] - 2
        failed = torch.rand(n, device=dev) < 0.02
        done = clip_end | failed
        rank = torch.where(done, torch.randint(0, lib.length.numel(), (n,), device=dev), self.rank)
        fresh = (torch.rand(n, device=dev) * (lib.length[rank] - 40).float()).long()
        self.frame = torch.where(done, fresh, self.frame)
        self.rank = rank
        self.episode = torch.where(done, torch.zeros_like(self.episode), self.episode)
        state = lib.ref_state(self.rank, self.frame)
        state = state + 0.01 * torch.randn_like(state)
        state[:, 3:7] = state[:, 3:7] / state[:, 3:7].norm(dim=-1, keepdim=True)
        self.hist_state = torch.cat((self.hist_state[:, 1:], state[:, None]), 1)
        self.hist_action = torch.cat((self.hist_action[:, 1:], action[:, None]), 1)
        self.hist_state[done] = state[done][:, None]
        self.hist_action[done] = 0
        reward = 0.05 + 0.02 * torch.cos(self.frame.float() * 0.1) - 0.001 * action.square().mean(-1)
        return self._obs(), reward, failed & ~clip_end, clip_end, {}


class FinalObsFakeEnv(FakeIsaacEnv):
    """Publishes ``final_obs`` as ``G1V3RLEnv`` does: the true post-step rows of every
    ended episode, batched, before the reset. A failed episode ends ``FALL`` below its
    reference, so its last state is no reset row and no reference row."""

    FALL = 0.25

    def step(self, action: Tensor) -> tuple[dict[str, Any], Tensor, Tensor, Tensor, dict[str, Any]]:
        lib, n, dev = self.v3_library, self.n, self.device
        self.frame = self.frame + 1
        self.episode = self.episode + 1
        clip_end = self.frame >= lib.length[self.rank] - 2
        failed = (torch.rand(n, device=dev) < 0.02) & ~clip_end
        done = clip_end | failed
        state = lib.ref_state(self.rank, self.frame)
        state = state + 0.01 * torch.randn_like(state)
        state[:, 3:7] = state[:, 3:7] / state[:, 3:7].norm(dim=-1, keepdim=True)
        state[:, 2] -= self.FALL * failed.float()
        self.true_next_state = state  # what a test compares against
        self.hist_state = torch.cat((self.hist_state[:, 1:], state[:, None]), 1)
        self.hist_action = torch.cat((self.hist_action[:, 1:], action[:, None]), 1)
        extras: dict[str, Any] = {}
        ids = done.nonzero().squeeze(-1)
        if ids.numel():
            wm = {key: value[ids] for key, value in self._obs()["wm"].items()}
            extras["final_obs"] = {"_env_ids": ids, "obs": {"wm": wm}}
        self.rank = torch.where(done, torch.randint(0, lib.length.numel(), (n,), device=dev), self.rank)
        fresh = (torch.rand(n, device=dev) * (lib.length[self.rank] - 40).float()).long()
        self.frame = torch.where(done, fresh, self.frame)
        self.episode = torch.where(done, torch.zeros_like(self.episode), self.episode)
        start = lib.ref_state(self.rank, self.frame)
        self.hist_state[done] = start[done][:, None]
        self.hist_action[done] = 0
        reward = 0.05 + 0.02 * torch.cos(self.frame.float() * 0.1) - 0.001 * action.square().mean(-1)
        return self._obs(), reward, failed, clip_end, extras


class SyncFreeFinalObsEnv(FakeIsaacEnv):
    """Ends the first four envs on every second step, with no host read of its own.
    The agent's reads are then the only ones a sync census sees."""

    def __init__(self, num_envs: int, device: torch.device | str) -> None:
        super().__init__(num_envs, device)
        self.steps = 0

    def step(self, _action: Tensor) -> tuple[dict[str, Any], Tensor, Tensor, Tensor, dict[str, Any]]:
        zeros = torch.zeros(self.n, device=self.device)
        self.steps += 1
        self.frame = self.frame + 1
        self.episode = self.episode + 1
        extras: dict[str, Any] = {}
        done = zeros.bool()
        if self.steps % 2 == 0:
            done = torch.arange(self.n, device=self.device) < 4
            ids = torch.arange(4, device=self.device)
            wm = {key: value[ids] for key, value in self._obs()["wm"].items()}
            extras["final_obs"] = {"_env_ids": ids, "obs": {"wm": wm}}
        self.frame = torch.where(done | (self.frame >= self.v3_library.length[self.rank] - 2), 0, self.frame)
        self.episode = torch.where(done, 0, self.episode)
        return self._obs(), zeros, done, zeros.bool(), extras


def make_agent(
    *,
    device: str = "cpu",
    num_envs: int = 8,
    seed: int = 7,
    frames: int = 0,
    save_interval: int = 0,
    log_dir: str | Path | None = None,
    small: bool = True,
    env_cls: type[FakeIsaacEnv] = FakeIsaacEnv,
    **wm: Any,
) -> WMBPTT:
    """A real agent. ``wm`` overrides ``WMBPTTConfig`` (after ``SMALL``)."""
    # G1V3RLOptWMBPTTConfig's fields, which need Isaac Lab to import.
    fields = {"task_model": G1_V3_TASK, "goal": "explicit", "action_clip": 20.0, **(SMALL if small else {}), **wm}
    config = WMBPTTRLOptConfig(wm=WMBPTTConfig(**fields))
    config.device = device
    config.seed = seed
    config.save_interval = save_interval
    config.collector.total_frames = frames
    config.collector.frames_per_batch = 1
    config.trainer.progress_bar = False
    config.logger.backend = ""
    config.logger.log_dir = str(log_dir or tempfile.mkdtemp(prefix="wm_bptt_"))
    return WMBPTT(env=env_cls(num_envs, device), config=config)  # type: ignore[arg-type]


def frames_for(iterations: int, *, num_envs: int = 8, wm: WMBPTTConfig | None = None) -> int:
    """``collector.total_frames`` of a run of exactly ``iterations`` iterations."""
    wm = wm or WMBPTTConfig(min_replay_steps=SMALL["min_replay_steps"])
    acc, steps = 0.0, 0
    for i in range(iterations):
        acc += wm.env_steps_per_iter
        n = int(acc)
        acc -= n
        steps += max(n, wm.min_replay_steps) if i == 0 else n
    return steps * num_envs


def _tensor_bytes(t: Tensor) -> bytes:
    t = t.detach().to("cpu").contiguous().reshape(-1)
    return b"%s|%s|" % (str(t.dtype).encode(), str(tuple(t.shape)).encode()) + t.view(torch.uint8).numpy().tobytes()


def _feed(h: Any, value: Any) -> None:
    if isinstance(value, Tensor):
        h.update(_tensor_bytes(value))
    elif isinstance(value, dict):
        for key in sorted(value, key=str):
            h.update(str(key).encode())
            _feed(h, value[key])
    elif isinstance(value, (list, tuple)):
        for item in value:
            _feed(h, item)
    elif isinstance(value, float):
        h.update(float(value).hex().encode())
    else:
        h.update(repr(value).encode())


def digest(value: Any) -> str:
    """SHA-256 of a nest of tensors and scalars: exact bits, order-independent keys."""
    h = hashlib.sha256()
    _feed(h, value)
    return h.hexdigest()


def replay_tensors(agent: WMBPTT) -> dict[str, Tensor]:
    rp = agent.replay
    assert rp is not None
    names = ("state", "action", "prev_action", "context", "rank", "frame", "episode_step", "push", "policy_action", "env_reward", "terminated")
    return {name: getattr(rp, name) for name in names}


def fingerprint(agent: WMBPTT) -> dict[str, str]:
    """Digest per group, so a drift names what moved."""
    groups: dict[str, Any] = {
        "actor": agent.actor.state_dict(),
        "critic": agent.critic.state_dict(),
        "dynamics": agent.dynamics.state_dict(),
        "optim_wm": agent.wm_optim.state_dict(),
        "optim_actor": agent.actor_optim.state_dict(),
        "optim_critic": agent.critic_optim.state_dict(),
        "rng_cpu": torch.get_rng_state(),
        "replay": replay_tensors(agent),
        "counters": {
            "iteration": agent._iteration,
            "frames": agent._frames_processed,
            "step_acc": agent._step_acc,
            "noise_seed": agent._noise_seed,
            "replay_steps": agent.replay.steps,  # type: ignore[union-attr]
            "episode_sums": agent._episode_sums,
        },
    }
    if agent.device.type == "cuda":
        groups["rng_cuda"] = torch.cuda.get_rng_state(agent.device)
    return {name: digest(value) for name, value in groups.items()}


def state_equal(a: Any, b: Any) -> bool:
    """Bitwise equality of two nests (``digest`` with the keys kept)."""
    return digest(a) == digest(b)
