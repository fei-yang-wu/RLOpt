"""Goal-conditioned latent sequence flow model, independent of the simulator.

Goals live in world coordinates; only their model inputs are re-expressed in
the current robot heading frame. Encoder labels and causal history are always
detached. This module neither changes the encoder nor owns a PPO optimizer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn

from rlopt.agent.causal_interface_planner import CausalInterfaceTransformerFlowPlanner


def quaternion_matrix(quaternion: Tensor) -> Tensor:
    """Active rotation matrix for wxyz quaternions (including q == -q)."""
    q = torch.nn.functional.normalize(quaternion, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def heading_matrix(quaternion: Tensor) -> Tensor:
    # World-Z twist, not ZYX Euler yaw: matches SONIC and the runtime even
    # when the pelvis tilts. Input here is wxyz; Isaac's runtime uses xyzw.
    twist = torch.zeros_like(quaternion)
    twist[..., 0] = quaternion[..., 0]
    twist[..., 3] = quaternion[..., 3]
    return quaternion_matrix(twist)


@dataclass
class WorldPoseGoal:
    """Batched constraints on named bodies; mask[..., 0/1] selects position/rotation.

    ``deadline`` is an absolute time on each environment's episode clock.
    ``hold_seconds == 0`` means reach/pass through. Positive hold durations
    require evidence of staying within tolerance after arrival, not merely a
    low velocity at one frame. Position and rotation tolerance units are metres
    and radians. Body order is part of the saved planner contract.
    """

    position: Tensor  # [B, J, 3], world metres
    quaternion: Tensor  # [B, J, 4], world wxyz
    mask: Tensor  # [B, J, 2], bool
    tolerance: Tensor  # [B, J, 2], position and geodesic rotation tolerances
    deadline: Tensor  # [B], seconds
    hold_seconds: Tensor  # [B], seconds

    def validate(self) -> None:
        if self.position.ndim != 3 or self.position.shape[-1] != 3:
            msg = "Goal position must be [B, J, 3]."
            raise ValueError(msg)
        batch, bodies = self.position.shape[:2]
        expected = {
            "quaternion": (batch, bodies, 4),
            "mask": (batch, bodies, 2),
            "tolerance": (batch, bodies, 2),
            "deadline": (batch,),
            "hold_seconds": (batch,),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if tuple(value.shape) != shape or value.device != self.position.device:
                msg = f"Goal {name} must have shape {shape} on the goal device."
                raise ValueError(msg)
            if not torch.isfinite(value).all():
                msg = f"Non-finite goal {name}."
                raise ValueError(msg)
        if not torch.isfinite(self.position).all():
            msg = "Non-finite goal position."
            raise ValueError(msg)
        if self.mask.dtype != torch.bool or not self.mask.flatten(1).any(1).all():
            msg = "Each goal needs at least one boolean pose constraint."
            raise ValueError(msg)
        if (self.quaternion.norm(dim=-1) < 1e-6).any():
            msg = "Goal quaternions must be nonzero, including masked bodies."
            raise ValueError(msg)
        if (self.tolerance <= 0).any() or (self.hold_seconds < 0).any():
            msg = "Goal tolerances must be positive and hold time non-negative."
            raise ValueError(msg)

    def features(
        self, root_position: Tensor, root_quaternion: Tensor, now: Tensor
    ) -> Tensor:
        """Pure conversion: never move the world goal when the robot replans."""
        inverse = heading_matrix(root_quaternion).transpose(-1, -2)
        origin = root_position * root_position.new_tensor([1, 1, 0])
        position = torch.einsum(
            "bij,bnj->bni", inverse, self.position - origin[:, None]
        )
        rotation = inverse[:, None] @ quaternion_matrix(self.quaternion)
        # Row-major first two columns, matching the repository's rot6d layout.
        rotation6d = rotation[..., :2].flatten(-2)
        pose = torch.cat(
            (
                torch.where(self.mask[..., :1], position, 0.0),
                torch.where(self.mask[..., 1:], rotation6d, 0.0),
                self.mask.to(position.dtype),
                torch.where(self.mask, self.tolerance, 0.0),
            ),
            dim=-1,
        ).flatten(1)
        return torch.cat(
            (
                pose,
                (self.deadline - now).clamp_min(0)[:, None],
                self.hold_seconds[:, None],
            ),
            dim=-1,
        )

    def reached(self, position: Tensor, quaternion: Tensor) -> Tensor:
        position_error = (position - self.position).norm(dim=-1)
        actual = torch.nn.functional.normalize(quaternion, dim=-1)
        desired = torch.nn.functional.normalize(self.quaternion, dim=-1)
        angle = 2 * (actual * desired).sum(-1).abs().clamp(max=1).acos()
        errors = torch.stack((position_error, angle), dim=-1)
        return ((errors <= self.tolerance) | ~self.mask).flatten(1).all(1)


@dataclass(frozen=True)
class GeometricGoalPlannerConfig:
    history_steps: int = 10
    # Raw joint positions/velocities + pelvis pose in current heading frame.
    # Available both in reference motion and robot history, without invented
    # last actions or privileged future reference inputs.
    history_frame_dim: int = 67
    body_names: tuple[str, ...] = (
        "pelvis",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    )
    latent_dim: int = 64
    prediction_steps: int = 10
    command_stride: int = 1
    execute_steps: int = 1
    control_dt: float = 0.02
    d_model: int = 256
    num_layers: int = 4
    num_heads: int = 4
    feedforward_dim: int = 1024
    num_inference_steps: int = 16

    @property
    def goal_dim(self) -> int:
        return 13 * len(self.body_names) + 2

    def validate(self) -> None:
        for name in (
            "history_steps",
            "history_frame_dim",
            "latent_dim",
            "prediction_steps",
            "command_stride",
            "execute_steps",
            "num_inference_steps",
        ):
            if getattr(self, name) <= 0:
                msg = f"{name} must be positive."
                raise ValueError(msg)
        if not self.body_names or len(set(self.body_names)) != len(self.body_names):
            msg = "body_names must be nonempty and unique."
            raise ValueError(msg)
        if self.execute_steps > self.prediction_steps or self.control_dt <= 0:
            msg = "Execution prefix exceeds prediction or control_dt is invalid."
            raise ValueError(msg)


class GeometricGoalPlanner(nn.Module):
    """p(Z[t:t+K] | causal robot history, masked geometric goal, remaining time)."""

    def __init__(self, config: GeometricGoalPlannerConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.flow = CausalInterfaceTransformerFlowPlanner(
            state_dim=config.history_steps * config.history_frame_dim + config.goal_dim,
            target_dim=config.prediction_steps * config.latent_dim,
            term_widths=(config.latent_dim,) * config.prediction_steps,
            d_model=config.d_model,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            feedforward_dim=config.feedforward_dim,
            patch_dim=config.latent_dim,
            num_state_tokens=4,
        )

    def conditioning(self, history: Tensor, goal: Tensor) -> Tensor:
        batch = history.shape[0]
        expected = (batch, self.config.history_steps, self.config.history_frame_dim)
        if tuple(history.shape) != expected or tuple(goal.shape) != (
            batch,
            self.config.goal_dim,
        ):
            msg = f"Expected history {expected} and goal [B,{self.config.goal_dim}]."
            raise ValueError(msg)
        return torch.cat((history.detach().flatten(1), goal.detach()), -1)

    def loss(
        self,
        history: Tensor,
        goal: Tensor,
        target: Tensor,
        *,
        valid: Tensor | None = None,
        noise: Tensor | None = None,
        time: Tensor | None = None,
    ) -> Tensor:
        cfg = self.config
        expected = (history.shape[0], cfg.prediction_steps, cfg.latent_dim)
        if tuple(target.shape) != expected:
            msg = f"Expected latent sequence {expected}."
            raise ValueError(msg)
        condition = self.conditioning(history, goal)
        normalized = self.flow.normalize_target(target.detach().flatten(1))
        noise = (
            torch.randn_like(normalized)
            if noise is None
            else noise.detach().reshape_as(normalized)
        )
        time = (
            torch.rand_like(normalized[:, :1])
            if time is None
            else time.detach().reshape(-1, 1)
        )
        interpolated = (1 - time) * noise + time * normalized
        if valid is not None:
            if tuple(valid.shape) != expected[:2]:
                msg = "Target validity must be [B, K]."
                raise ValueError(msg)
            # Remove invalid target information from attention as well as the
            # loss. Otherwise future labels can leak through padded tokens.
            interpolated = (
                interpolated.reshape(expected)
                .masked_fill(~valid[..., None], 0)
                .flatten(1)
            )
        error = (
            self.flow.velocity(condition, interpolated, time) - (normalized - noise)
        ).square()
        error = error.reshape(expected).mean(-1)
        if valid is None:
            return error.mean()
        # Every training row contributes equally, including short goal horizons.
        weights = valid.detach().to(error.dtype)
        return ((error * weights).sum(-1) / weights.sum(-1).clamp_min(1)).mean()

    @torch.no_grad()
    def forward(
        self, history: Tensor, goal: Tensor, *, noise_std: float = 1.0
    ) -> Tensor:
        cfg = self.config
        condition = self.conditioning(history, goal)
        shape = (history.shape[0], cfg.prediction_steps, cfg.latent_dim)
        # Same zero padding as training; even an overdue goal gets one command
        # so the caller can report deadline failure without an invalid packet.
        offsets = (
            torch.arange(cfg.prediction_steps, device=goal.device)
            * cfg.command_stride
            * cfg.control_dt
        )
        valid = (
            offsets[None]
            < (goal[:, -2] + goal[:, -1]).clamp_min(cfg.control_dt)[:, None]
        )
        x = torch.randn(shape, device=history.device, dtype=history.dtype) * noise_std
        x.masked_fill_(~valid[..., None], 0)
        dt = 1.0 / cfg.num_inference_steps
        for step in range(cfg.num_inference_steps):
            time = x.new_full((len(x), 1), step * dt)
            velocity = self.flow.velocity(condition, x.flatten(1), time).reshape(shape)
            x = (x + dt * velocity).masked_fill(~valid[..., None], 0)
        return (
            self.flow.denormalize_target(x.flatten(1))
            .reshape(shape)
            .masked_fill(~valid[..., None], 0)
        )

    def checkpoint(self) -> dict[str, Any]:
        return {
            "planner_type": "geometric_goal_flow_v1",
            "config": asdict(self.config),
            "model_state_dict": self.state_dict(),
        }

    @classmethod
    def from_checkpoint(cls, checkpoint: dict[str, Any]) -> GeometricGoalPlanner:
        if checkpoint.get("planner_type") != "geometric_goal_flow_v1":
            msg = "Not a geometric goal planner checkpoint."
            raise ValueError(msg)
        config = dict(checkpoint["config"])
        config["body_names"] = tuple(config["body_names"])
        model = cls(GeometricGoalPlannerConfig(**config))
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        return model


def sample_hindsight_indices(
    lengths: Tensor,
    batch_size: int,
    *,
    history_steps: int,
    prediction_steps: int,
    command_stride: int,
    encoder_steps: int,
    min_goal_steps: int,
    max_goal_steps: int,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Uniform eligible trajectory, uniform start, then uniform future goal.

    No target window wraps or clamps at a trajectory boundary. Histories near
    reset use repeat-first padding in the data adapter, matching deployment.
    The goal horizon is independent of K; shorter goals remain valid.
    Offsets and lengths use control frames, not macro-window indices.
    """
    if (
        min(
            batch_size,
            history_steps,
            prediction_steps,
            command_stride,
            encoder_steps,
            min_goal_steps,
        )
        <= 0
        or max_goal_steps < min_goal_steps
    ):
        msg = "Invalid hindsight sampling dimensions or goal interval."
        raise ValueError(msg)
    tail = max(
        (prediction_steps - 1) * command_stride + encoder_steps - 1, min_goal_steps
    )
    available = lengths - tail
    eligible = torch.where(available > 0)[0]
    if not eligible.numel():
        msg = "No trajectories contain a complete history, target, and future goal."
        raise ValueError(msg)
    ranks = eligible[
        torch.randint(
            eligible.numel(), (batch_size,), device=lengths.device, generator=generator
        )
    ]
    starts = (
        torch.rand(batch_size, device=lengths.device, generator=generator)
        * available[ranks]
    ).long()
    upper = (lengths[ranks] - 1 - starts).clamp(max=max_goal_steps)
    offsets = (
        min_goal_steps
        + (
            torch.rand(batch_size, device=lengths.device, generator=generator)
            * (upper - min_goal_steps + 1)
        ).long()
    )
    return ranks, starts, starts + offsets
