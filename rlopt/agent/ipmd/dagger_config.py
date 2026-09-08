"""Frozen-teacher warm-start settings, separate from PPO's budget."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class DAggerConfig:
    enabled: bool = False
    teacher_checkpoint: str = ""
    teacher_num_cells: list[int] = field(
        default_factory=lambda: [2048, 2048, 1024, 1024, 512, 512]
    )
    teacher_only_frames: int = 200_000_000
    handoff_end_frames: int = 400_000_000
    final_teacher_rate: float = 0.01
    learning_rate: float = 1e-4
    batch_size: int = 8192
    updates_per_rollout: int = 12
    replay_capacity: int = 262144
    max_grad_norm: float = 1.0
    initialization_path: str = ""

    def validate(self):
        if not self.teacher_checkpoint or not self.initialization_path:
            msg = "DAgger requires teacher_checkpoint and initialization_path"
            raise ValueError(msg)
        if not 0 <= self.teacher_only_frames < self.handoff_end_frames:
            msg = "DAgger requires 0 <= teacher_only_frames < handoff_end_frames"
            raise ValueError(msg)
        if not 0 < self.final_teacher_rate < 1:
            msg = "final_teacher_rate must lie in (0,1)"
            raise ValueError(msg)
        for name in (
            "learning_rate",
            "batch_size",
            "updates_per_rollout",
            "replay_capacity",
            "max_grad_norm",
        ):
            if getattr(self, name) <= 0:
                msg = f"DAgger {name} must be positive"
                raise ValueError(msg)
        if self.replay_capacity < self.batch_size:
            msg = "DAgger replay_capacity must be >= batch_size"
            raise ValueError(msg)

    def teacher_rate(self, frames: int) -> float:
        if frames >= self.handoff_end_frames:
            return 0.0
        fraction = max(
            0.0,
            (frames - self.teacher_only_frames)
            / (self.handoff_end_frames - self.teacher_only_frames),
        )
        return math.exp(math.log(self.final_teacher_rate) * fraction)
