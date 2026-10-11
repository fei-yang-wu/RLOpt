"""DiffSR chunk prior reward ``g`` for IPMD (2026-10-11).

MaxEnt IRL with a reward prior: the reward the policy optimizes is
``r = f + g``, where ``f`` is the learned residual (the reward estimator) and
``g`` a fixed function. Regularizing only ``f`` is the IRL problem with the
regularizer centred at ``g`` (``psi_g(r) = psi(r - g)``), so ``g`` needs no
expert batch and never enters the estimator's gradient
(``wiki/maxent-irl-reparameterization.md``).

Here ``g`` is built from a frozen ``hl_skill_diffsr`` checkpoint's next-chunk
head (``jepa_ntp_head='diff_chunk'``, ``jepa_ntp_chunk_span='boundary_next'``),
which models the reference's chunk conditional ``p_E(y | s, z)``:

    s = robot macro frames t-5 .. t             (the head's source window)
    z = skill code of the REFERENCE frames t .. t+9   (the commanded chunk)
    y = robot macro frames t+10 .. t+20          (what the robot did next)
    r_B = - mean_{tau, k} || eps - phi(s, z)^T mu(y_tau, tau) ||^2      (ELBO up to constants)
    g   = weight * stride * h(r_B)

``h`` is a fixed monotone map to [0, 1] (any fixed ``g`` is a valid prior):

    transform="log": h = clip(1 - log(-r_B / lo) / log(hi / lo), 0, 1)
    transform="exp": h = exp(r_B / kappa)

``r_B`` spans about -7 (reference chunks) .. -60 (a 21 mm tracker) .. -300
(an untrained policy, deterministic actions) .. -1e5 (an untrained policy
under its initial exploration noise). ``exp`` is flat zero over the last
range, so the default is ``log`` with ``(lo, hi) = (5, 1e5)``.

All frames are anchored at frame ``t``'s heading (robot frames at the robot's,
reference frames at the reference's), exactly as the head was trained and as
the offline check (``logs/gap_probe/rb_check.py``) scored them.

Online, ``g`` is evaluated once per rollout over the whole rollout tensor. A
per-env ring of the last ``SPAN - 1`` frames persists across rollouts so a
chunk can straddle a rollout boundary; a per-env episode position counter
(reset on ``done``) decides validity. Row ``T`` of the rollout (the state
observed at that step) is evaluated when its episode position ``p`` satisfies
``p >= SPAN - 1`` and ``(p - (SPAN - 1)) % stride == 0``; the value is paid on
that row. ``weight * stride`` keeps the per-step average at
``weight * exp(r_B / kappa)``.

The ring is not checkpointed: after a resume the counters restart at zero,
which only delays the first evaluations of each env by up to ``SPAN - 1``
steps.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

#: Raw per-step layout of one side: joint_pos (29), root_pos_w (3), root_quat_w xyzw (4).
SIDE_WIDTH = 36
FRAME_WIDTH = 2 * SIDE_WIDTH
MACRO_WIDTH = 38
PAST = 5
HORIZON = 10
SPAN = PAST + 1 + 2 * HORIZON  # frames t-5 .. t+20


def _quat_conj(q: Tensor) -> Tensor:
    return torch.cat([-q[..., :3], q[..., 3:4]], dim=-1)


def _quat_mul(a: Tensor, b: Tensor) -> Tensor:
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dim=-1,
    )


def _quat_rotate(q: Tensor, v: Tensor) -> Tensor:
    axis, w = q[..., :3], q[..., 3:4]
    cross = torch.linalg.cross(axis, v, dim=-1)
    return v + 2.0 * (w * cross + torch.linalg.cross(axis, cross, dim=-1))


def _rot6d_interleaved(q: Tensor) -> Tensor:
    x, y, z, w = q.unbind(-1)
    s = 2.0 / (q * q).sum(-1).clamp(min=1e-9)
    m00 = 1.0 - s * (y * y + z * z)
    m01 = s * (x * y - z * w)
    m10 = s * (x * y + z * w)
    m11 = 1.0 - s * (x * x + z * z)
    m20 = s * (x * z - y * w)
    m21 = s * (y * z + x * w)
    return torch.stack([m00, m01, m10, m11, m20, m21], dim=-1)


def heading_anchored_frames(side: Tensor, anchor_index: int) -> Tensor:
    """``[K, L, 36]`` raw frames -> ``[K, L, 38]`` macro frames anchored at one slot.

    Restates ``imitation_experiments.capacity.probe_skill_window_usage.
    heading_anchored_expert_windows`` with an arbitrary anchor slot: heading =
    the anchor quaternion with x, y zeroed and renormalized, origin = the
    anchor's planar position, every slot's root pose expressed in it.
    """
    joint_pos = side[..., :29]
    pos = side[..., 29:32]
    quat = side[..., 32:36]
    heading = quat[:, anchor_index].clone()
    heading[..., 0] = 0.0
    heading[..., 1] = 0.0
    heading = heading / torch.linalg.vector_norm(heading, dim=-1, keepdim=True).clamp_min(1e-9)
    origin = pos[:, anchor_index].clone()
    origin[..., 2] = 0.0
    inv_heading = _quat_conj(heading)[:, None, :].expand_as(quat)
    pos_b = _quat_rotate(inv_heading, pos - origin[:, None, :])
    quat_b = _quat_mul(inv_heading, quat)
    return torch.cat([joint_pos, pos_b, _rot6d_interleaved(quat_b)], dim=-1)


class DiffSRChunkPrior(nn.Module):
    """Frozen DiffSR next-chunk head turned into the prior reward ``g``."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        num_envs: int,
        device: torch.device | str,
        weight: float = 0.3,
        kappa: float = 150.0,
        stride: int = 10,
        draws: int = 1,
        batch_size: int = 512,
        transform: str = "log",
        log_range: tuple[float, float] = (5.0, 1.0e5),
    ) -> None:
        super().__init__()
        from rlopt.agent.hl_skill_diffsr import (
            HighLevelSkillDiffSRConfig,
            _build_diffsr,
            _encoder_input_window,
            _encoder_window_steps,
            _jepa_ntp_target_dim,
        )
        from rlopt.agent.hl_skill_encoder import build_skill_encoder

        self.device = torch.device(device)
        blob = torch.load(Path(checkpoint).expanduser(), map_location="cpu", weights_only=False, mmap=True)
        config = HighLevelSkillDiffSRConfig.from_dict(blob["config"])
        if int(config.source_history_steps) != PAST or int(config.horizon_steps) != HORIZON:
            raise ValueError(
                "DiffSRChunkPrior needs source_history_steps=5 and horizon_steps=10; got "
                f"{config.source_history_steps} and {config.horizon_steps}."
            )
        if str(config.jepa_ntp_head) != "diff_chunk" or str(config.jepa_ntp_chunk_span) != "boundary_next":
            raise ValueError(
                "DiffSRChunkPrior needs jepa_ntp_head='diff_chunk' with "
                "jepa_ntp_chunk_span='boundary_next'."
            )
        if str(config.encoder_window_mode) != "intermediate":
            raise ValueError("DiffSRChunkPrior supports encoder_window_mode='intermediate' only.")
        self.config = config
        self._encoder_input_window = _encoder_input_window
        head = _build_diffsr(config, MACRO_WIDTH, self.device, next_obs_dim=_jepa_ntp_target_dim(config, MACRO_WIDTH))
        head.load_state_dict(blob["jepa_state_dict"]["ntp_diffsr"], strict=True)
        self.head = head.to(self.device).eval().requires_grad_(False)
        encoder = build_skill_encoder(
            state_dim=MACRO_WIDTH,
            window_steps=_encoder_window_steps(config),
            z_dim=config.z_dim,
            hidden_dims=config.encoder_hidden_dims,
            spec=config.latent_spec(),
            activation=config.encoder_activation,
            layer_norm=config.encoder_layer_norm,
        )
        encoder.load_state_dict(blob["skill_encoder_state_dict"], strict=True)
        self.encoder = encoder.to(self.device).eval().requires_grad_(False)
        del blob
        self.weight = float(weight)
        self.kappa = float(kappa)
        self.stride = int(stride)
        self.draws = int(draws)
        self.batch_size = int(batch_size)
        if self.stride < 1 or self.draws < 1 or self.kappa <= 0.0:
            raise ValueError("stride and draws must be >= 1 and kappa > 0.")
        if transform not in {"log", "exp"}:
            raise ValueError(f"transform must be 'log' or 'exp', got {transform!r}.")
        lo, hi = (float(v) for v in log_range)
        if not 0.0 < lo < hi:
            raise ValueError(f"log_range must satisfy 0 < lo < hi, got {log_range!r}.")
        self.transform = transform
        self.log_lo, self.log_hi = lo, hi
        self.num_envs = int(num_envs)
        self.register_buffer("ring", torch.zeros(self.num_envs, SPAN - 1, FRAME_WIDTH, device=self.device))
        self.register_buffer("position", torch.zeros(self.num_envs, dtype=torch.long, device=self.device))

    # ------------------------------------------------------------------
    @torch.no_grad()
    def chunk_scores(self, windows: Tensor) -> Tensor:
        """``r_B`` for ``[K, SPAN, 72]`` raw windows (row t at slot PAST)."""
        robot = heading_anchored_frames(windows[..., :SIDE_WIDTH], PAST)
        reference = heading_anchored_frames(windows[..., SIDE_WIDTH:], PAST)
        source = robot[:, : PAST + 1].reshape(robot.shape[0], -1)
        target = robot[:, PAST + HORIZON :].reshape(robot.shape[0], -1)
        ref_chunk = reference[:, PAST : PAST + HORIZON]
        state = ref_chunk[:, 0]
        future = torch.cat([ref_chunk[:, 1:], ref_chunk[:, -1:]], dim=1)
        z = self.encoder(state, self._encoder_input_window(self.config, future))
        phi = self.head.forward_phi(s=source, a=z)
        x0 = self.head.obs_norm.normalize(target)
        total = torch.zeros(x0.shape[0], device=x0.device)
        count = 0
        for tau in range(int(self.head.num_noises)):
            alphabar = self.head.alphabars[tau]
            t = torch.full((x0.shape[0], 1), tau, device=x0.device, dtype=torch.long)
            for _ in range(self.draws):
                eps = torch.randn_like(x0)
                xt = alphabar.sqrt() * x0 + (1.0 - alphabar).sqrt() * eps
                mu = self.head.forward_mu(sp=xt, t=t, s=source)
                pred = torch.bmm(phi.unsqueeze(1), mu).squeeze(1)
                total = total + (pred - eps).pow(2).sum(-1)
                count += 1
        return -total / count

    def shape(self, scores: Tensor) -> Tensor:
        """The fixed map ``h(r_B)`` to [0, 1]."""
        if self.transform == "exp":
            return torch.exp(scores / self.kappa)
        ratio = (-scores).clamp_min(self.log_lo) / self.log_lo
        return (1.0 - torch.log(ratio) / math.log(self.log_hi / self.log_lo)).clamp(0.0, 1.0)

    @torch.no_grad()
    def compute(self, frames: Tensor, done: Tensor) -> tuple[Tensor, dict[str, float]]:
        """Prior reward ``[N, T]`` for one rollout and update the ring.

        ``frames``: ``[N, T, 72]`` raw ``macro_pair`` observations at each
        rollout row (the state before that row's action). ``done``:
        ``[N, T]`` booleans, the row's transition ended the episode.
        """
        num_envs, steps = int(frames.shape[0]), int(frames.shape[1])
        if num_envs != self.num_envs:
            raise ValueError(f"Rollout has {num_envs} envs, prior was built for {self.num_envs}.")
        frames = frames.to(self.device, torch.float32)
        done = done.to(self.device).reshape(num_envs, steps).bool()
        # Episode position of every rollout row: row 0 continues the counter,
        # a done on row t makes row t+1 the first frame of a new episode.
        positions = torch.empty(num_envs, steps, dtype=torch.long, device=self.device)
        current = self.position.clone()
        for t in range(steps):
            positions[:, t] = current
            current = torch.where(done[:, t], torch.zeros_like(current), current + 1)
        timeline = torch.cat([self.ring, frames], dim=1)  # [N, SPAN-1+T, 72]
        lead = SPAN - 1
        due = (positions >= lead) & ((positions - lead) % self.stride == 0)
        reward = torch.zeros(num_envs, steps, device=self.device)
        env_idx, row_idx = torch.nonzero(due, as_tuple=True)
        scores = torch.empty(0, device=self.device)
        if env_idx.numel() > 0:
            offsets = torch.arange(SPAN, device=self.device)
            columns = row_idx[:, None] + offsets[None, :]  # timeline index of row T is T + lead
            windows = timeline[env_idx[:, None], columns]
            parts = [self.chunk_scores(windows[i : i + self.batch_size]) for i in range(0, windows.shape[0], self.batch_size)]
            scores = torch.cat(parts)
            reward[env_idx, row_idx] = self.weight * self.stride * self.shape(scores)
        self.ring.copy_(timeline[:, -lead:])
        self.position.copy_(current)
        metrics: dict[str, float] = {"prior/eval_rows": float(env_idx.numel())}
        if scores.numel() > 0:
            metrics.update(
                {
                    "prior/r_B_mean": float(scores.mean()),
                    "prior/r_B_p10": float(torch.quantile(scores, 0.1)),
                    "prior/r_B_p90": float(torch.quantile(scores, 0.9)),
                    "prior/r_B_median": float(scores.median()),
                    "prior/h_mean": float(self.shape(scores).mean()),
                    "prior/g_per_step_mean": float(reward.mean()),
                }
            )
        return reward, metrics


__all__ = ["DiffSRChunkPrior", "FRAME_WIDTH", "SPAN", "heading_anchored_frames"]
