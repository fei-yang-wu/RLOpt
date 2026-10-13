"""Universal skill encoder: many input modalities, one token space.

SONIC-style universal tokens for a tracker that already exists. The tracker
was trained on codes ``z = E_teacher(x_robot)`` of a frozen robot-motion
encoder; this module trains ONE student encoder that maps any subset of input
views (full robot motion, VR three-point tracking, a root trajectory, body
keypoints, ...) into that same code space, so the unchanged tracker can be
driven by whichever device is available, e.g. a VR headset for teleoperation.

The student is a masked-modality MLP. Each view is a ``[B, frames, width]``
block z-scored by its own running statistics; absent views are zeroed and a
presence bit per view is appended. A *mode* is a fixed set of present views;
every training row draws a mode.

Loss per row (teacher frozen)::

    L = align_coeff * ||z_s - z_t||^2  +  ntp_coeff * L_ntp(z_s)

``||.||^2`` is summed over code dims (SONIC's latent/token alignment term).
``L_ntp`` is the teacher's own generative next-chunk head scoring the student
code against the real future (its recon term analogue): it keeps a code that
cannot be matched exactly, because the view is ambiguous, functional for the
decoder rather than merely close. The tracker is untouched; its token space is
the teacher's.

The view construction is robot-specific and injected (``view_fn``), so this
module stays robot-agnostic.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _require_positive_int(name: str, value: int) -> int:
    normalized = int(value)
    if normalized <= 0:
        msg = f"{name} must be > 0, got {value!r}."
        raise ValueError(msg)
    return normalized


def _require_non_negative_float(name: str, value: float) -> float:
    normalized = float(value)
    if normalized < 0.0:
        msg = f"{name} must be >= 0, got {value!r}."
        raise ValueError(msg)
    return normalized


@dataclass
class UniversalEncoderConfig:
    """Student architecture, the view/mode spec and the distillation knobs."""

    z_dim: int = 64
    frames: int = 10
    view_widths: dict[str, int] = field(default_factory=dict)
    """Per-frame width of every view the student can receive (order = input order)."""
    modes: dict[str, list[str]] = field(default_factory=dict)
    """Mode name -> the views present in that mode."""
    mode_probs: dict[str, float] = field(default_factory=dict)
    """Training mixture over modes; missing modes get weight 0. Empty = uniform."""
    hidden_dims: tuple[int, ...] = (2048, 1024, 512, 512)
    layer_norm: bool = True
    norm_eps: float = 1.0e-4
    align_coeff: float = 1.0
    ntp_coeff: float = 0.1
    lr: float = 3.0e-4
    weight_decay: float = 0.0
    grad_clip_norm: float | None = 1.0
    batch_size: int = 8192
    num_updates: int = 20000
    log_interval: int = 1000
    eval_batches: int = 4
    eval_batch_size: int | None = None

    def validate(self) -> None:
        self.z_dim = _require_positive_int("z_dim", self.z_dim)
        self.frames = _require_positive_int("frames", self.frames)
        self.view_widths = {str(k): _require_positive_int(f"view_widths[{k}]", v) for k, v in self.view_widths.items()}
        if not self.view_widths:
            msg = "view_widths must name at least one view."
            raise ValueError(msg)
        self.modes = {str(k): [str(v) for v in views] for k, views in self.modes.items()}
        if not self.modes:
            msg = "modes must name at least one mode."
            raise ValueError(msg)
        for mode, views in self.modes.items():
            unknown = sorted(set(views) - set(self.view_widths))
            if not views or unknown:
                msg = f"mode {mode!r} has no views or unknown views {unknown}."
                raise ValueError(msg)
        self.mode_probs = {str(k): _require_non_negative_float(f"mode_probs[{k}]", v) for k, v in self.mode_probs.items()}
        unknown_modes = sorted(set(self.mode_probs) - set(self.modes))
        if unknown_modes:
            msg = f"mode_probs names unknown modes {unknown_modes}."
            raise ValueError(msg)
        if self.mode_probs and sum(self.mode_probs.values()) <= 0.0:
            msg = "mode_probs must have positive total weight."
            raise ValueError(msg)
        self.hidden_dims = tuple(_require_positive_int("hidden_dims", d) for d in self.hidden_dims)
        self.align_coeff = _require_non_negative_float("align_coeff", self.align_coeff)
        self.ntp_coeff = _require_non_negative_float("ntp_coeff", self.ntp_coeff)
        self.batch_size = _require_positive_int("batch_size", self.batch_size)
        self.num_updates = _require_positive_int("num_updates", self.num_updates)
        self.log_interval = _require_positive_int("log_interval", self.log_interval)
        self.eval_batches = _require_positive_int("eval_batches", self.eval_batches)

    @property
    def view_order(self) -> tuple[str, ...]:
        return tuple(self.view_widths)

    @property
    def mode_order(self) -> tuple[str, ...]:
        return tuple(self.modes)

    def mode_weights(self) -> Tensor:
        weights = [float(self.mode_probs.get(m, 0.0 if self.mode_probs else 1.0)) for m in self.mode_order]
        total = sum(weights)
        return torch.tensor([w / total for w in weights], dtype=torch.float32)

    def presence_table(self) -> Tensor:
        """``[num_modes, num_views]`` bool: which views each mode shows."""
        return torch.tensor(
            [[view in self.modes[mode] for view in self.view_order] for mode in self.mode_order],
            dtype=torch.bool,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> UniversalEncoderConfig:
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in values.items() if k in known}
        if "hidden_dims" in kwargs:
            kwargs["hidden_dims"] = tuple(int(d) for d in kwargs["hidden_dims"])
        config = cls(**kwargs)
        config.validate()
        return config


class _RunningNorm(nn.Module):
    """Per-dimension running mean/variance (Chan et al. merge), checkpointed."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = float(eps)
        self.register_buffer("mean", torch.zeros(int(dim)))
        self.register_buffer("var", torch.ones(int(dim)))
        self.register_buffer("count", torch.zeros((), dtype=torch.float64))

    @torch.no_grad()
    def update(self, values: Tensor) -> None:
        batch = values.detach().to(torch.float32).reshape(-1, values.shape[-1])
        n = float(batch.shape[0])
        if n == 0:
            return
        batch_mean = batch.mean(dim=0)
        batch_var = batch.var(dim=0, unbiased=False)
        count = float(self.count.item())
        total = count + n
        delta = batch_mean - self.mean
        self.mean.add_(delta * (n / total))
        self.var.copy_((self.var * count + batch_var * n + delta.pow(2) * (count * n / total)) / total)
        self.count.fill_(total)

    def forward(self, values: Tensor) -> Tensor:
        return (values - self.mean) / torch.sqrt(self.var + self.eps)


class UniversalSkillEncoder(nn.Module):
    """Masked-modality MLP ``z = E_u({view_b * present_b}, present)``."""

    def __init__(self, config: UniversalEncoderConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.view_order = config.view_order
        self.norms = nn.ModuleDict(
            {view: _RunningNorm(config.frames * width, config.norm_eps) for view, width in config.view_widths.items()}
        )
        input_dim = sum(config.frames * w for w in config.view_widths.values()) + len(self.view_order)
        layers: list[nn.Module] = []
        previous = input_dim
        for width in config.hidden_dims:
            layers.append(nn.Linear(previous, width))
            if config.layer_norm:
                layers.append(nn.LayerNorm(width))
            layers.append(nn.SiLU())
            previous = width
        layers.append(nn.Linear(previous, config.z_dim))
        self.net = nn.Sequential(*layers)
        self.register_buffer("presence_table", config.presence_table())

    def presence_for_mode(self, mode: str, batch_size: int, device: torch.device) -> Tensor:
        index = self.config.mode_order.index(mode)
        return self.presence_table[index].to(device).unsqueeze(0).expand(batch_size, -1)

    @torch.no_grad()
    def update_norms(self, views: Mapping[str, Tensor], present: Tensor) -> None:
        for b, view in enumerate(self.view_order):
            rows = present[:, b]
            if bool(rows.any()):
                self.norms[view].update(views[view].flatten(1)[rows])

    def forward(self, views: Mapping[str, Tensor | None], present: Tensor) -> Tensor:
        """``views[name]``: ``[B, frames, width]`` (may be None when absent everywhere)."""
        batch = int(present.shape[0])
        parts: list[Tensor] = []
        for b, view in enumerate(self.view_order):
            width = self.config.frames * self.config.view_widths[view]
            block = views.get(view)
            if block is None:
                if bool(present[:, b].any()):
                    msg = f"view {view!r} is marked present but was not provided."
                    raise ValueError(msg)
                parts.append(present.new_zeros((batch, width), dtype=torch.float32))
                continue
            flat = block.reshape(batch, -1).to(torch.float32)
            if int(flat.shape[-1]) != width:
                msg = f"view {view!r} is {int(flat.shape[-1])} wide, expected {width}."
                raise ValueError(msg)
            parts.append(self.norms[view](flat) * present[:, b : b + 1].to(flat.dtype))
        parts.append(present.to(torch.float32))
        return self.net(torch.cat(parts, dim=-1))

    def encode_mode(self, views: Mapping[str, Tensor], mode: str) -> Tensor:
        some = next(v for v in views.values() if v is not None)
        present = self.presence_for_mode(mode, int(some.shape[0]), some.device)
        return self(views, present)

    def checkpoint_state_dict(self) -> dict[str, Any]:
        return {"universal_config": self.config.to_dict(), "universal_state_dict": self.state_dict()}

    @classmethod
    def from_checkpoint_state_dict(cls, state: Mapping[str, Any]) -> UniversalSkillEncoder:
        encoder = cls(UniversalEncoderConfig.from_dict(state["universal_config"]))
        encoder.load_state_dict(state["universal_state_dict"])
        return encoder


ViewFn = Callable[[Tensor], Mapping[str, Tensor]]


def load_teacher_modules(teacher: Any, checkpoint: Mapping[str, Any]) -> None:
    """Restore the frozen modules a distiller reads, without optimizer state.

    ``teacher`` is a trainer built from ``checkpoint["config"]``. Only the
    encoder, the endpoint DiffSR head (and its feature normalizer) and the
    generative NTP head are needed; skipping the optimizer state saves the
    memory of two Adam moments per parameter.
    """
    teacher.skill_encoder.load_state_dict(checkpoint["skill_encoder_state_dict"])
    teacher.diffsr.load_state_dict(checkpoint["diffsr_state_dict"])
    feature_norm_state = checkpoint.get("feature_normalization_state_dict")
    obs_norm = getattr(teacher.diffsr, "obs_norm", None)
    if isinstance(obs_norm, nn.Module) and feature_norm_state:
        obs_norm.load_state_dict(feature_norm_state)
    ntp_head = getattr(teacher, "jepa_ntp_diffsr", None)
    if ntp_head is not None:
        jepa_state = checkpoint.get("jepa_state_dict") or {}
        if "ntp_diffsr" not in jepa_state:
            msg = "teacher checkpoint has no jepa_state_dict['ntp_diffsr'] for its NTP head."
            raise ValueError(msg)
        ntp_head.load_state_dict(jepa_state["ntp_diffsr"])
    teacher.update = int(checkpoint.get("update", 0))


class UniversalEncoderDistiller:
    """Distil a frozen robot-motion encoder into a :class:`UniversalSkillEncoder`.

    ``teacher`` is a :class:`~rlopt.agent.hl_skill_diffsr.HighLevelSkillDiffSRTrainer`
    restored from the tracker's encoder checkpoint; it supplies the data windows
    (its own train/eval trajectory split), the target code and the generative
    head behind ``L_ntp``. ``view_fn`` maps the teacher-visible frames
    ``[B, frames, state_dim]`` (``s_t`` plus the encoder window) to the views.
    ``augment_fn`` (optional) maps those frames to a perturbed copy of the same
    motion, e.g. the window re-expressed in a displaced heading frame; each
    training step then also aligns the student to the teacher's code of the
    perturbed window, so the student follows the teacher off the offline
    distribution (the tracker saw the teacher's codes of robot-anchored windows).
    ``live_frames`` (optional, ``[N, frames, state_dim]``, any device/dtype) are
    windows recorded on-policy, i.e. while a student drove the tracker; each
    step draws ``live_batch_size`` of them and aligns the student to the
    teacher's code of the same window in a random mode (DAgger-style).
    """

    def __init__(
        self,
        config: UniversalEncoderConfig,
        *,
        teacher: Any,
        view_fn: ViewFn,
        augment_fn: Callable[[Tensor], Tensor] | None = None,
        live_frames: Tensor | None = None,
        live_batch_size: int = 0,
    ) -> None:
        from rlopt.agent.hl_skill_diffsr import _encoder_window_steps  # noqa: PLC0415

        config.validate()
        expected_frames = int(_encoder_window_steps(teacher.config)) + 1
        if int(config.frames) != expected_frames:
            msg = f"config.frames={config.frames} but the teacher sees {expected_frames} frames."
            raise ValueError(msg)
        if int(config.z_dim) != int(teacher.config.z_dim):
            msg = f"z_dim {config.z_dim} != teacher z_dim {teacher.config.z_dim}."
            raise ValueError(msg)
        self.config = config
        self.teacher = teacher
        self.view_fn = view_fn
        self.augment_fn = augment_fn
        self.device = teacher.device
        self.live_frames = live_frames
        self.live_batch_size = int(live_batch_size) if live_frames is not None else 0
        if self.live_frames is not None and getattr(view_fn, "uses_batch", False):
            msg = "live_frames carry no trajectory rank/frame, so they cannot feed a batch-aware view_fn."
            raise ValueError(msg)
        if self.live_frames is not None and int(self.live_frames.shape[1]) != expected_frames:
            msg = f"live_frames carry {int(self.live_frames.shape[1])} frames, the teacher sees {expected_frames}."
            raise ValueError(msg)
        for module in self._teacher_modules():
            module.eval()
            module.requires_grad_(False)
        self.student = UniversalSkillEncoder(config).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.student.parameters(), lr=float(config.lr), weight_decay=float(config.weight_decay)
        )
        self.update = 0
        self._mode_weights = config.mode_weights().to(self.device)

    def _teacher_modules(self) -> list[nn.Module]:
        modules: list[nn.Module] = [self.teacher.skill_encoder, self.teacher.diffsr]
        if getattr(self.teacher, "jepa_ntp_diffsr", None) is not None:
            modules.append(self.teacher.jepa_ntp_diffsr)
        return modules

    def _window(
        self, batch_size: int, split: str | None
    ) -> tuple[Tensor, Tensor, Callable[[Tensor], Tensor], Any]:
        """Teacher frames ``[B, frames, D]``, teacher code, ``L_ntp(z)`` and the macro batch."""
        from rlopt.agent.hl_skill_diffsr import _encoder_input_window  # noqa: PLC0415

        state, exec_window, _source, batch, loss_at_z = self.teacher._planner_eval_window(
            batch_size=int(batch_size), split=split
        )
        encoder_window = _encoder_input_window(self.teacher.config, exec_window)
        with torch.no_grad():
            z_teacher, *_ = self.teacher.skill_encoder.encode(state, encoder_window, deterministic=True)
        frames = torch.cat([state.unsqueeze(1), encoder_window], dim=1)
        return frames, z_teacher, loss_at_z, batch

    @torch.no_grad()
    def _teacher_code(self, frames: Tensor) -> Tensor:
        z, *_ = self.teacher.skill_encoder.encode(frames[:, 0], frames[:, 1:], deterministic=True)
        return z

    def _views(self, frames: Tensor, batch: Any = None) -> dict[str, Tensor]:
        """``view_fn(frames)``, or ``view_fn(frames, batch)`` when it sets ``uses_batch``.

        A batch-aware view (e.g. human motion fetched by the window's
        trajectory rank and frame) needs the macro batch the frames came from.
        """
        with torch.no_grad():
            if getattr(self.view_fn, "uses_batch", False):
                if batch is None:
                    msg = "this view_fn needs the macro batch (trajectory rank and frame)."
                    raise ValueError(msg)
                views = dict(self.view_fn(frames, batch))
            else:
                views = dict(self.view_fn(frames))
        missing = sorted(set(self.config.view_order) - set(views))
        if missing:
            msg = f"view_fn did not produce views {missing}."
            raise ValueError(msg)
        return {name: views[name] for name in self.config.view_order}

    def train_step(self) -> dict[str, float]:
        self.student.train()
        frames, z_teacher, loss_at_z, macro_batch = self._window(
            self.config.batch_size, self.teacher.config.train_split
        )
        views = self._views(frames, macro_batch)
        batch = int(frames.shape[0])
        mode_index = torch.multinomial(self._mode_weights, batch, replacement=True)
        present = self.student.presence_table[mode_index]
        self.student.update_norms(views, present)
        z_student = self.student(views, present)
        align_rows = (z_student - z_teacher).pow(2).sum(dim=-1)
        align = align_rows.mean()
        loss = float(self.config.align_coeff) * align
        align_aug = torch.zeros((), device=self.device)
        if self.augment_fn is not None:
            with torch.no_grad():
                frames_aug = self.augment_fn(frames)
            z_teacher_aug = self._teacher_code(frames_aug)
            views_aug = self._views(frames_aug, macro_batch)
            self.student.update_norms(views_aug, present)
            align_aug = (self.student(views_aug, present) - z_teacher_aug).pow(2).sum(dim=-1).mean()
            loss = loss + float(self.config.align_coeff) * align_aug
        align_live = torch.zeros((), device=self.device)
        if self.live_batch_size > 0:
            assert self.live_frames is not None
            rows = torch.randint(0, int(self.live_frames.shape[0]), (self.live_batch_size,))
            live = self.live_frames[rows].to(self.device, torch.float32)
            z_teacher_live = self._teacher_code(live)
            views_live = self._views(live)
            live_modes = torch.multinomial(self._mode_weights, self.live_batch_size, replacement=True)
            present_live = self.student.presence_table[live_modes]
            self.student.update_norms(views_live, present_live)
            align_live = (self.student(views_live, present_live) - z_teacher_live).pow(2).sum(dim=-1).mean()
            loss = loss + float(self.config.align_coeff) * align_live
        ntp = torch.zeros((), device=self.device)
        if float(self.config.ntp_coeff) > 0.0:
            ntp = loss_at_z(z_student)
            loss = loss + float(self.config.ntp_coeff) * ntp
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        metrics: dict[str, float] = {}
        if self.config.grad_clip_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=float(self.config.grad_clip_norm))
            metrics["train/grad_norm"] = float(grad_norm.item())
        self.optimizer.step()
        self.update += 1
        with torch.no_grad():
            cosine = F.cosine_similarity(z_student, z_teacher, dim=-1)
            for m, mode in enumerate(self.config.mode_order):
                rows = mode_index == m
                if bool(rows.any()):
                    metrics[f"train/{mode}/align"] = float(align_rows[rows].mean().item())
                    metrics[f"train/{mode}/cosine"] = float(cosine[rows].mean().item())
        metrics.update(
            {
                "train/loss": float(loss.detach().item()),
                "train/align": float(align.detach().item()),
                "train/align_aug": float(align_aug.detach().item()),
                "train/align_live": float(align_live.detach().item()),
                "train/ntp_loss": float(ntp.detach().item()),
            }
        )
        return metrics

    @torch.no_grad()
    def evaluate(self, *, prefix: str = "eval", split: str | None = None) -> dict[str, float]:
        """Per-mode fit to the teacher code and the NTP loss of the student code."""
        split = self.teacher.config.eval_split if split is None else split
        batch_size = int(self.config.eval_batch_size or self.config.batch_size)
        was_training = self.student.training
        self.student.eval()
        accum: dict[str, float] = {}
        for _ in range(int(self.config.eval_batches)):
            frames, z_teacher, loss_at_z, macro_batch = self._window(batch_size, split)
            views = self._views(frames, macro_batch)
            rows = int(z_teacher.shape[0])
            centered = (z_teacher - z_teacher.mean(dim=0, keepdim=True)).pow(2).sum(dim=-1).mean()
            row_metrics = {
                f"{prefix}/teacher/ntp_loss": float(loss_at_z(z_teacher).item()),
                f"{prefix}/teacher/ntp_loss_shuffled": float(
                    loss_at_z(z_teacher[torch.randperm(rows, device=z_teacher.device)]).item()
                ),
                f"{prefix}/teacher/z_rms": float(z_teacher.pow(2).mean().sqrt().item()),
            }
            for mode in self.config.mode_order:
                z = self.student.encode_mode(views, mode)
                residual = (z - z_teacher).pow(2).sum(dim=-1).mean()
                row_metrics.update(
                    {
                        f"{prefix}/{mode}/align": float(residual.item()),
                        f"{prefix}/{mode}/cosine": float(F.cosine_similarity(z, z_teacher, dim=-1).mean().item()),
                        f"{prefix}/{mode}/r2": 1.0 - float(residual.item()) / max(float(centered.item()), 1.0e-12),
                        f"{prefix}/{mode}/z_rms": float(z.pow(2).mean().sqrt().item()),
                        f"{prefix}/{mode}/ntp_loss": float(loss_at_z(z).item()),
                    }
                )
            if self.augment_fn is not None:
                frames_aug = self.augment_fn(frames)
                z_teacher_aug = self._teacher_code(frames_aug)
                views_aug = self._views(frames_aug, macro_batch)
                for mode in self.config.mode_order:
                    z = self.student.encode_mode(views_aug, mode)
                    row_metrics[f"{prefix}_aug/{mode}/align"] = float(
                        (z - z_teacher_aug).pow(2).sum(dim=-1).mean().item()
                    )
                    row_metrics[f"{prefix}_aug/{mode}/cosine"] = float(
                        F.cosine_similarity(z, z_teacher_aug, dim=-1).mean().item()
                    )
            for key, value in row_metrics.items():
                accum[key] = accum.get(key, 0.0) + value
        if was_training:
            self.student.train()
        return {key: value / float(self.config.eval_batches) for key, value in accum.items()}

    def checkpoint_state_dict(self, *, teacher_checkpoint_path: str = "", view_spec: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return {
            **self.student.checkpoint_state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "update": int(self.update),
            "teacher_checkpoint_path": str(teacher_checkpoint_path),
            "teacher_config": self.teacher.config.to_dict(),
            "view_spec": dict(view_spec or {}),
        }

    def save_checkpoint(self, path: str | Path, **kwargs: Any) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        staged = target.with_suffix(target.suffix + ".tmp")
        torch.save(self.checkpoint_state_dict(**kwargs), staged)
        staged.replace(target)

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        state = torch.load(Path(path), map_location=self.device, weights_only=False)
        self.student.load_state_dict(state["universal_state_dict"])
        if "optimizer_state_dict" in state:
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.update = int(state.get("update", 0))
        return state
