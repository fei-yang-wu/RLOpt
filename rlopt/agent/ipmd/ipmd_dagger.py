"""Frozen-teacher DAgger warm start using IPMD's latent collection and PPO phases.

Unlike IPMDL2T, the teacher is frozen, both actors see the same deployable
inputs, and student-visited states are labeled with teacher means. Actor
architectures may differ. No PPO or critic updates occur during this stage.
"""

from __future__ import annotations

import copy
import hashlib
import time
from pathlib import Path

import torch
from tensordict.nn import TensorDictModuleBase
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.envs.utils import ExplorationType, set_exploration_type

from rlopt.agent.ipmd.ipmd import IPMD
from rlopt.agent.ppo.ppo import PPO


class TeacherMixturePolicy(TensorDictModuleBase):
    """Query the teacher on the exact state/command seen by the student."""

    def __init__(self, student, teacher):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.in_keys = list(getattr(student, "in_keys", teacher.in_keys))
        self.out_keys = list(
            dict.fromkeys(
                [
                    *getattr(student, "out_keys", teacher.out_keys),
                    "teacher_action",
                    "teacher_used",
                    "student_action",
                ]
            )
        )
        self.teacher_rate = 1.0

    def forward(self, td):
        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
            td = self.student(td)  # Includes the frozen encoder's command injection.
            target = self.teacher.get_dist(td).deterministic_sample.detach()
            # get_dist writes loc/scale, but the student's executed candidate is
            # already in action. Never use a sampled teacher action as a label.
            candidate = td.get("action")
            use_teacher = (
                torch.rand((*td.batch_size, 1), device=candidate.device)
                < self.teacher_rate
            )
            td.set("student_action", candidate)
            td.set("teacher_action", target)
            td.set("teacher_used", use_teacher)
            td.set("action", torch.where(use_teacher, target, candidate))
        return td


class IPMDDagger(IPMD):
    """DAgger-only stage; export a zero-PPO-frame actor on completion."""

    def _construct_collector(self, env, policy):
        cfg = self.config.dagger
        cfg.validate()
        if self.config.compile.compile or self.config.compile.cudagraphs:
            msg = "DAgger mixture requires compile/cudagraphs disabled"
            raise ValueError(msg)
        teacher_cfg = copy.deepcopy(self.config.policy)
        teacher_cfg.num_cells = list(cfg.teacher_num_cells)
        teacher = self._construct_policy_from_config(teacher_cfg)
        checkpoint = torch.load(
            cfg.teacher_checkpoint, map_location=self.device, weights_only=False
        )
        teacher.load_state_dict(checkpoint["policy_state_dict"], strict=True)
        teacher.eval().requires_grad_(False)
        self._dagger_teacher_encoder = checkpoint.get(
            "hl_skill_command_sampler_state_dict"
        )
        self._dagger_teacher_sha = hashlib.sha256(
            Path(cfg.teacher_checkpoint).read_bytes()
        ).hexdigest()
        # Same input geometry: preserve the teacher's normalization and noise
        # scale; the larger mean network is the only randomly initialized part.
        student = self.policy
        teacher_buffers = dict(teacher.named_buffers())
        with torch.no_grad():
            for name, buffer in student.named_buffers():
                if name in teacher_buffers:
                    buffer.copy_(teacher_buffers[name])
            teacher_parameters = dict(teacher.named_parameters())
            for name, parameter in student.named_parameters():
                if name.endswith("log_std"):
                    parameter.copy_(teacher_parameters[name])
                    parameter.requires_grad_(False)
        self._dagger_policy = TeacherMixturePolicy(policy, teacher).eval()
        return super()._construct_collector(env, self._dagger_policy)

    def validate_training(self):
        super().validate_training()
        cfg = self.config.dagger
        sampler = self._hl_skill_command_sampler
        if sampler is not None:
            live = sampler.checkpoint_state_dict()["skill_encoder_state_dict"]
            saved = (self._dagger_teacher_encoder or {}).get(
                "skill_encoder_state_dict", {}
            )
            if live.keys() != saved.keys() or any(
                not torch.equal(v, saved[k].to(v.device)) for k, v in live.items()
            ):
                msg = "DAgger teacher and student frozen encoder weights differ"
                raise ValueError(msg)
        target = int(self.config.collector.total_frames)
        batch = int(self.config.collector.frames_per_batch)
        if not cfg.handoff_end_frames <= target < cfg.handoff_end_frames + batch:
            msg = "DAgger collector budget must equal handoff_end_frames, rounded to a rollout"
            raise ValueError(msg)
        self._dagger_policy.eval()
        if not hasattr(self, "_dagger_optimizer"):
            self._dagger_optimizer = torch.optim.Adam(
                [p for p in self.policy.parameters() if p.requires_grad],
                lr=cfg.learning_rate,
            )
            self._dagger_replay = TensorDictReplayBuffer(
                storage=LazyTensorStorage(cfg.replay_capacity, device=self.device),
                batch_size=cfg.batch_size,
            )
        pending = getattr(self, "_dagger_pending", None)
        if pending is not None:
            if pending["teacher_sha256"] != self._dagger_teacher_sha:
                msg = "DAgger resume teacher hash mismatch"
                raise ValueError(msg)
            self._dagger_optimizer.load_state_dict(pending["optimizer"])
            self._dagger_replay.load_state_dict(pending["replay"])
            torch.set_rng_state(pending["torch_rng"].cpu())
            if self.device.type == "cuda" and pending.get("cuda_rng") is not None:
                torch.cuda.set_rng_state(pending["cuda_rng"].cpu(), self.device)
            self._dagger_pending = None

    def collect(self, run, iteration_idx):
        self._dagger_policy.teacher_rate = self.config.dagger.teacher_rate(
            run.frames_processed
        )
        return super().collect(run, iteration_idx)

    prepare = PPO.prepare
    record = PPO.record

    def iterate(self, iteration, run):
        start = time.perf_counter()
        cfg = self.config.dagger
        student = self.policy
        student.eval()  # Frozen running statistics; gradients still flow.
        samples = (
            iteration.rollout.select(*self._policy_obs_keys, "teacher_action")
            .reshape(-1)
            .detach()
        )
        self._dagger_replay.extend(samples)
        losses = []
        for _ in range(cfg.updates_per_rollout):
            batch = self._dagger_replay.sample()
            dist = student.get_dist(batch.select(*self._policy_obs_keys))
            loss = (dist.deterministic_sample - batch["teacher_action"]).square().mean()
            self._dagger_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                student.parameters(), cfg.max_grad_norm, error_if_nonfinite=True
            )
            self._dagger_optimizer.step()
            losses.append(loss.detach())
        iteration.metrics.update(
            {
                "dagger/mse": torch.stack(losses).mean(),
                "dagger/rollout_mse": (
                    iteration.rollout["student_action"]
                    - iteration.rollout["teacher_action"]
                )
                .square()
                .mean(),
                "dagger/teacher_rate": self._dagger_policy.teacher_rate,
                "dagger/teacher_fraction": iteration.rollout["teacher_used"]
                .float()
                .mean(),
                "dagger/replay_rows": len(self._dagger_replay),
                "dagger/frames": run.frames_processed,
            }
        )
        iteration.learn_time = time.perf_counter() - start
        self._dagger_frames = run.frames_processed

    def _extra_checkpoint_state_dict(self):
        state = super()._extra_checkpoint_state_dict()
        if hasattr(self, "_dagger_optimizer"):
            state["dagger_state"] = {
                "teacher_sha256": self._dagger_teacher_sha,
                "optimizer": self._dagger_optimizer.state_dict(),
                "replay": self._dagger_replay.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(self.device)
                if self.device.type == "cuda"
                else None,
            }
        return state

    def _load_extra_checkpoint_state_dict(self, checkpoint):
        super()._load_extra_checkpoint_state_dict(checkpoint)
        if "dagger_state" not in checkpoint:
            msg = "DAgger resume requires distillation optimizer/replay state"
            raise ValueError(msg)
        self._dagger_pending = checkpoint["dagger_state"]

    def train(self):
        super().train()
        # The inherited loop saves resumable distillation snapshots. Only a
        # successful full-budget exit publishes PPO initialization, atomically.
        cfg = self.config.dagger
        frames = getattr(
            self, "_dagger_frames", getattr(self, "_resume_frame_offset", 0)
        )
        if frames < cfg.handoff_end_frames:
            msg = "DAgger ended before the requested handoff budget"
            raise RuntimeError(msg)
        output = Path(cfg.initialization_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "policy_state_dict": self._checkpoint_policy_state_dict(),
            "cumulative_env_frames": 0,
            "distillation_env_frames": frames,
            "initialization_metadata": {
                "method": "frozen_teacher_dagger",
                "teacher_sha256": self._dagger_teacher_sha,
            },
        }
        if self._hl_skill_command_sampler is not None:
            state["hl_skill_command_sampler_state_dict"] = (
                self._hl_skill_command_sampler.checkpoint_state_dict()
            )
        temporary = output.with_suffix(".tmp")
        torch.save(state, temporary)
        temporary.replace(output)
        self.log.info(
            "DAgger complete at %d frames; PPO initialization: %s", frames, output
        )
