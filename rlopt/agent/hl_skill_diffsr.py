from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, cast

import torch
from tensordict import TensorDictBase
from torch import Tensor, nn

from rlopt.agent.ipmd.module import BilinearSR, build_bilinear_sr


def _require_positive_int(name: str, value: int) -> int:
    normalized = int(value)
    if normalized <= 0:
        msg = f"{name} must be > 0, got {value!r}."
        raise ValueError(msg)
    return normalized


def _require_positive_float(name: str, value: float) -> float:
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        msg = f"{name} must be > 0, got {value!r}."
        raise ValueError(msg)
    return normalized


def _require_non_negative_float(name: str, value: float) -> float:
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        msg = f"{name} must be >= 0, got {value!r}."
        raise ValueError(msg)
    return normalized


def _require_fraction(name: str, value: float) -> float:
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 < normalized < 1.0:
        msg = f"{name} must be in (0, 1), got {value!r}."
        raise ValueError(msg)
    return normalized


def _normalize_split_value(name: str, value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized == "":
        return None
    if normalized not in {"all", "train", "eval"}:
        msg = f"{name} must be one of 'all', 'train', or 'eval', got {value!r}."
        raise ValueError(msg)
    return normalized


def _normalize_encoder_window_mode(name: str, value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in {"full", "intermediate"}:
        msg = f"{name} must be one of 'full' or 'intermediate', got {value!r}."
        raise ValueError(msg)
    return normalized


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


@dataclass
class HighLevelSkillDiffSRConfig:
    """Configuration for offline high-level skill DiffSR training."""

    horizon_steps: int = 25
    z_dim: int = 256
    diffsr_feature_dim: int = 128
    diffsr_embed_dim: int = 512
    batch_size: int = 8192
    num_updates: int = 2000
    log_interval: int = 100
    eval_batches: int = 4
    eval_batch_size: int | None = None
    train_split: str | None = "train"
    eval_split: str | None = "eval"
    eval_trajectory_fraction: float = 0.1
    trajectory_split_seed: int = 0
    preflight_batch_size: int = 8
    encoder_window_mode: str = "full"
    encoder_hidden_dims: tuple[int, ...] = (1024, 512, 512)
    diffsr_f_hidden_dims: tuple[int, ...] = (512, 512)
    diffsr_g_hidden_dims: tuple[int, ...] = (512,)
    diffsr_mu_hidden_dims: tuple[int, ...] = (512,)
    diffsr_num_noises: int = 8
    diffsr_x_min: float = -10.0
    diffsr_x_max: float = 10.0
    encoder_lr: float = 3.0e-4
    diffsr_lr: float = 1.0e-4
    weight_decay: float = 0.0
    grad_clip_norm: float | None = 1.0
    z_norm_coeff: float = 1.0e-4
    reconstruction_norm_eps: float = 1.0e-6
    device: str = "auto"
    diffsr_state_output_init_std: float = 1.0e-3

    def validate(self) -> None:
        self.horizon_steps = _require_positive_int("horizon_steps", self.horizon_steps)
        self.z_dim = _require_positive_int("z_dim", self.z_dim)
        self.diffsr_feature_dim = _require_positive_int(
            "diffsr_feature_dim", self.diffsr_feature_dim
        )
        self.diffsr_embed_dim = _require_positive_int(
            "diffsr_embed_dim", self.diffsr_embed_dim
        )
        self.batch_size = _require_positive_int("batch_size", self.batch_size)
        self.num_updates = _require_positive_int("num_updates", self.num_updates)
        self.log_interval = _require_positive_int("log_interval", self.log_interval)
        self.eval_batches = _require_positive_int("eval_batches", self.eval_batches)
        if self.eval_batch_size is not None:
            self.eval_batch_size = _require_positive_int(
                "eval_batch_size", self.eval_batch_size
            )
        self.train_split = _normalize_split_value("train_split", self.train_split)
        self.eval_split = _normalize_split_value("eval_split", self.eval_split)
        self.eval_trajectory_fraction = _require_fraction(
            "eval_trajectory_fraction", self.eval_trajectory_fraction
        )
        self.trajectory_split_seed = int(self.trajectory_split_seed)
        self.preflight_batch_size = _require_positive_int(
            "preflight_batch_size", self.preflight_batch_size
        )
        self.encoder_window_mode = _normalize_encoder_window_mode(
            "encoder_window_mode", self.encoder_window_mode
        )
        if self.encoder_window_mode == "intermediate" and self.horizon_steps <= 1:
            msg = "encoder_window_mode='intermediate' requires horizon_steps > 1."
            raise ValueError(msg)
        self.encoder_hidden_dims = tuple(
            _require_positive_int("encoder_hidden_dims", dim)
            for dim in self.encoder_hidden_dims
        )
        self.diffsr_f_hidden_dims = tuple(
            _require_positive_int("diffsr_f_hidden_dims", dim)
            for dim in self.diffsr_f_hidden_dims
        )
        self.diffsr_g_hidden_dims = tuple(
            _require_positive_int("diffsr_g_hidden_dims", dim)
            for dim in self.diffsr_g_hidden_dims
        )
        self.diffsr_mu_hidden_dims = tuple(
            _require_positive_int("diffsr_mu_hidden_dims", dim)
            for dim in self.diffsr_mu_hidden_dims
        )
        self.diffsr_num_noises = _require_positive_int(
            "diffsr_num_noises", self.diffsr_num_noises
        )
        self.encoder_lr = _require_positive_float("encoder_lr", self.encoder_lr)
        self.diffsr_lr = _require_positive_float("diffsr_lr", self.diffsr_lr)
        self.weight_decay = _require_non_negative_float(
            "weight_decay", self.weight_decay
        )
        if self.grad_clip_norm is not None:
            self.grad_clip_norm = _require_positive_float(
                "grad_clip_norm", self.grad_clip_norm
            )
        self.z_norm_coeff = _require_non_negative_float(
            "z_norm_coeff", self.z_norm_coeff
        )
        self.reconstruction_norm_eps = _require_positive_float(
            "reconstruction_norm_eps", self.reconstruction_norm_eps
        )
        self.diffsr_state_output_init_std = _require_non_negative_float(
            "diffsr_state_output_init_std", self.diffsr_state_output_init_std
        )
        self.device = str(self.device)

    def to_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], _jsonable(asdict(self)))

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> HighLevelSkillDiffSRConfig:
        known_fields = {item.name for item in fields(cls)}
        kwargs = {key: values[key] for key in known_fields if key in values}
        tuple_fields = {
            "encoder_hidden_dims",
            "diffsr_f_hidden_dims",
            "diffsr_g_hidden_dims",
            "diffsr_mu_hidden_dims",
        }
        for key in tuple_fields:
            if key in kwargs:
                kwargs[key] = tuple(int(item) for item in kwargs[key])
        config = cls(**kwargs)
        config.validate()
        return config


class HighLevelSkillEncoder(nn.Module):
    """Encode a current state and future expert window into a continuous skill."""

    def __init__(
        self,
        state_dim: int,
        window_steps: int,
        z_dim: int,
        hidden_dims: tuple[int, ...] = (1024, 512, 512),
    ) -> None:
        super().__init__()
        self.state_dim = _require_positive_int("state_dim", state_dim)
        self.window_steps = _require_positive_int("window_steps", window_steps)
        self.z_dim = _require_positive_int("z_dim", z_dim)
        hidden_dims = tuple(
            _require_positive_int("hidden_dims", dim) for dim in hidden_dims
        )

        input_dim = self.state_dim * (self.window_steps + 1)
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Mish(),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, self.z_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, state: Tensor, future_window: Tensor) -> Tensor:
        if state.ndim != 2:
            msg = f"state must have shape [B, D], got {tuple(state.shape)}."
            raise ValueError(msg)
        if future_window.ndim != 3:
            msg = (
                "future_window must have shape [B, W, D], got "
                f"{tuple(future_window.shape)}."
            )
            raise ValueError(msg)
        batch_size, state_dim = state.shape
        expected_window = (batch_size, self.window_steps, state_dim)
        if (
            int(state_dim) != self.state_dim
            or tuple(future_window.shape) != expected_window
        ):
            msg = (
                "state/future_window shape mismatch: expected state "
                f"[B, {self.state_dim}] and future_window {expected_window}, "
                f"got {tuple(state.shape)} and {tuple(future_window.shape)}."
            )
            raise ValueError(msg)
        flat_window = future_window.reshape(batch_size, self.window_steps * state_dim)
        return self.net(torch.cat([state, flat_window], dim=-1))


class FrozenHighLevelSkillCommandSampler:
    """Frozen high-level encoder used as an online latent-command source."""

    def __init__(
        self,
        *,
        env: object,
        checkpoint_path: str | Path,
        latent_dim: int,
        latent_steps_min: int,
        latent_steps_max: int,
        discover_env_method: Callable[[object, str], Callable[..., Any] | None],
        horizon_steps: int | None = None,
        command_phase_mode: str = "none",
        code_latent_dim: int | None = None,
        phase_period: int | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        self.latent_dim = _require_positive_int("latent_dim", latent_dim)
        self.latent_steps_min = max(1, int(latent_steps_min))
        self.latent_steps_max = max(self.latent_steps_min, int(latent_steps_max))
        self.command_phase_mode = str(command_phase_mode).strip().lower()
        if self.command_phase_mode not in {"none", "sin_cos"}:
            msg = (
                "command_phase_mode must be 'none' or 'sin_cos', got "
                f"{command_phase_mode!r}."
            )
            raise ValueError(msg)
        self.phase_dim = 2 if self.command_phase_mode == "sin_cos" else 0
        self.phase_period = (
            _require_positive_int("phase_period", int(phase_period))
            if phase_period is not None
            else self.latent_steps_max
        )
        self.device = self._resolve_device(env, device)
        self._current_macro_sampler = discover_env_method(
            env,
            "current_expert_macro_transition_batch",
        )
        if self._current_macro_sampler is None:
            msg = (
                "command_source='hl_skill' requires the environment to expose "
                "current_expert_macro_transition_batch(...)."
            )
            raise ValueError(msg)

        checkpoint = torch.load(
            Path(checkpoint_path).expanduser(),
            map_location=self.device,
            weights_only=False,
        )
        self.config = HighLevelSkillDiffSRConfig.from_dict(checkpoint["config"])
        if (
            horizon_steps is not None
            and int(horizon_steps) != self.config.horizon_steps
        ):
            msg = (
                "Configured hl_skill_horizon_steps does not match checkpoint "
                f"horizon_steps: {int(horizon_steps)} != {self.config.horizon_steps}."
            )
            raise ValueError(msg)
        self.code_latent_dim = int(self.config.z_dim)
        if code_latent_dim is not None and int(code_latent_dim) != self.code_latent_dim:
            msg = (
                "Frozen high-level skill code_latent_dim must match checkpoint "
                f"z_dim: {int(code_latent_dim)} != {self.code_latent_dim}."
            )
            raise ValueError(msg)
        expected_latent_dim = self.code_latent_dim + self.phase_dim
        if expected_latent_dim != self.latent_dim:
            msg = (
                "Frozen high-level skill command width must match ipmd.latent_dim: "
                f"checkpoint z_dim {self.code_latent_dim} + phase_dim "
                f"{self.phase_dim} != {self.latent_dim}."
            )
            raise ValueError(msg)

        state_dict = checkpoint["skill_encoder_state_dict"]
        self.encoder_window_steps = self._encoder_window_steps(self.config)
        self.state_dim = self._state_dim_from_encoder_state(
            state_dict,
            window_steps=self.encoder_window_steps,
        )
        self.skill_encoder = HighLevelSkillEncoder(
            state_dim=self.state_dim,
            window_steps=self.encoder_window_steps,
            z_dim=self.config.z_dim,
            hidden_dims=self.config.encoder_hidden_dims,
        ).to(self.device)
        self.skill_encoder.load_state_dict(state_dict)
        self.skill_encoder.eval()
        self.skill_encoder.requires_grad_(False)
        self._codes: Tensor | None = None
        self._latent_steps: Tensor | None = None

    @staticmethod
    def _resolve_device(env: object, device: torch.device | str | None) -> torch.device:
        if device is not None:
            return torch.device(device)
        env_device = getattr(env, "device", None)
        if env_device is not None:
            return torch.device(env_device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _encoder_window_steps(config: HighLevelSkillDiffSRConfig) -> int:
        if config.encoder_window_mode == "intermediate":
            return int(config.horizon_steps) - 1
        return int(config.horizon_steps)

    @staticmethod
    def _state_dim_from_encoder_state(
        state_dict: Mapping[str, Tensor],
        *,
        window_steps: int,
    ) -> int:
        first_weight = state_dict.get("net.0.weight")
        if first_weight is None or first_weight.ndim != 2:
            msg = (
                "Checkpoint skill encoder is missing first linear weight net.0.weight."
            )
            raise ValueError(msg)
        input_dim = int(first_weight.shape[1])
        divisor = int(window_steps) + 1
        if input_dim % divisor != 0:
            msg = (
                "Checkpoint skill encoder input width is incompatible with "
                f"window_steps={window_steps}: input_dim={input_dim}."
            )
            raise ValueError(msg)
        return input_dim // divisor

    @staticmethod
    def _done_mask(
        td: TensorDictBase, *, batch_size: int, device: torch.device
    ) -> Tensor:
        done_mask = torch.zeros(batch_size, device=device, dtype=torch.bool)
        candidate_keys: list[str | tuple[str, ...]] = [
            "done",
            "terminated",
            "truncated",
            "is_init",
            ("next", "done"),
            ("next", "terminated"),
            ("next", "truncated"),
            ("next", "is_init"),
        ]
        available_keys = td.keys(True)
        for key in candidate_keys:
            if key not in available_keys:
                continue
            value = cast(Tensor, td.get(key)).reshape(-1).to(device=device).bool()
            if value.numel() == batch_size:
                done_mask |= value
        return done_mask

    def _sample_steps(self, count: int, *, device: torch.device) -> Tensor:
        if self.latent_steps_min == self.latent_steps_max:
            return torch.full(
                (count,),
                self.latent_steps_min,
                device=device,
                dtype=torch.long,
            )
        return torch.randint(
            self.latent_steps_min,
            self.latent_steps_max + 1,
            (count,),
            device=device,
        )

    def _encoder_input_window(self, future_window: Tensor) -> Tensor:
        if self.config.encoder_window_mode == "intermediate":
            return future_window[:, :-1, :]
        return future_window

    def _append_command_phase(self, code_latents: Tensor, phase: Tensor) -> Tensor:
        if self.phase_dim == 0:
            return code_latents
        angle = phase.to(device=code_latents.device, dtype=code_latents.dtype)
        angle = angle.reshape(-1) * (2.0 * math.pi)
        phase_features = torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)
        return torch.cat((code_latents, phase_features), dim=-1)

    @torch.no_grad()
    def _encode_current_latents(self, env_ids: Tensor) -> Tensor:
        batch = self._current_macro_sampler(
            horizon_steps=int(self.config.horizon_steps),
            env_ids=env_ids,
        )
        state = cast(Tensor, batch.get(("hl", "state"))).to(self.device)
        future_window = cast(Tensor, batch.get(("hl", "future_window"))).to(self.device)
        batch_size = int(env_ids.numel())
        expected_state = (batch_size, self.state_dim)
        expected_window = (batch_size, int(self.config.horizon_steps), self.state_dim)
        if tuple(state.shape) != expected_state:
            msg = (
                "Current expert macro state shape mismatch: expected "
                f"{expected_state}, got {tuple(state.shape)}."
            )
            raise ValueError(msg)
        if tuple(future_window.shape) != expected_window:
            msg = (
                "Current expert macro future_window shape mismatch: expected "
                f"{expected_window}, got {tuple(future_window.shape)}."
            )
            raise ValueError(msg)
        return self.skill_encoder(state, self._encoder_input_window(future_window))

    @torch.no_grad()
    def sample_for_step(
        self,
        td: TensorDictBase,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        batch_size = int(td.numel())
        if batch_size <= 0:
            return torch.empty(0, self.latent_dim, device=device, dtype=dtype)
        if (
            self._codes is None
            or self._latent_steps is None
            or self._codes.shape[0] != batch_size
            or self._codes.shape[1] != self.code_latent_dim
            or self._codes.device != device
            or self._codes.dtype != dtype
        ):
            self._codes = torch.zeros(
                batch_size,
                self.code_latent_dim,
                device=device,
                dtype=dtype,
            )
            self._latent_steps = torch.zeros(
                batch_size, device=device, dtype=torch.long
            )

        assert self._codes is not None
        assert self._latent_steps is not None
        renew_mask = self._done_mask(
            td,
            batch_size=batch_size,
            device=device,
        ) | (self._latent_steps <= 0)
        if bool(renew_mask.any()):
            env_ids = torch.nonzero(renew_mask, as_tuple=False).reshape(-1)
            codes = self._encode_current_latents(env_ids.to(self.device)).to(
                device=device,
                dtype=dtype,
            )
            self._codes.index_copy_(0, env_ids.to(device=device), codes)
            self._latent_steps.index_copy_(
                0,
                env_ids.to(device=device),
                self._sample_steps(int(env_ids.numel()), device=device),
            )

        phase = (self.phase_period - self._latent_steps).clamp(min=0).to(torch.float32)
        phase = phase / float(self.phase_period)
        latents = self._append_command_phase(self._codes, phase)
        self._latent_steps = self._latent_steps - 1
        return latents


@dataclass
class HighLevelSkillDiffSRTrainState:
    update: int = 0
    elapsed_seconds: float = 0.0
    last_metrics: dict[str, float] = field(default_factory=dict)


class HighLevelSkillDiffSRTrainer:
    """Offline trainer for high-level skill encoders with a DiffSR objective."""

    def __init__(self, config: HighLevelSkillDiffSRConfig, env: object) -> None:
        self.config = config
        self.config.validate()
        self.env = env
        self.device = self._resolve_device()
        preflight_size = min(self.config.batch_size, self.config.preflight_batch_size)
        state, future_window, target = self._sample_and_validate_macro_batch(
            preflight_size,
            split=self.config.train_split,
        )
        del future_window, target
        self.state_dim = int(state.shape[-1])
        self.encoder_window_steps = self._encoder_window_steps()
        self.feature_slices = self._resolve_feature_slices()

        self.skill_encoder = HighLevelSkillEncoder(
            state_dim=self.state_dim,
            window_steps=self.encoder_window_steps,
            z_dim=self.config.z_dim,
            hidden_dims=self.config.encoder_hidden_dims,
        ).to(self.device)
        self.diffsr = self._build_diffsr().to(self.device)
        self._initialize_diffsr_state_output()
        self.optimizer = torch.optim.AdamW(
            [
                {
                    "params": self.skill_encoder.parameters(),
                    "lr": self.config.encoder_lr,
                },
                {"params": self.diffsr.parameters(), "lr": self.config.diffsr_lr},
            ],
            weight_decay=self.config.weight_decay,
        )
        self.update = 0

    def _resolve_device(self) -> torch.device:
        if self.config.device.strip().lower() != "auto":
            return torch.device(self.config.device)
        env_device = getattr(self.env, "device", None)
        if env_device is not None:
            return torch.device(env_device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _encoder_window_steps(self) -> int:
        if self.config.encoder_window_mode == "intermediate":
            return int(self.config.horizon_steps) - 1
        return int(self.config.horizon_steps)

    def _encoder_input_window(self, future_window: Tensor) -> Tensor:
        if self.config.encoder_window_mode == "intermediate":
            return future_window[:, :-1, :]
        return future_window

    def _encode_skill(self, state: Tensor, future_window: Tensor) -> Tensor:
        return self.skill_encoder(state, self._encoder_input_window(future_window))

    def _build_diffsr(self) -> BilinearSR:
        return build_bilinear_sr(
            "diffsr",
            obs_dim=self.state_dim,
            next_obs_dim=self.state_dim,
            action_dim=self.config.z_dim,
            feature_dim=self.config.diffsr_feature_dim,
            embed_dim=self.config.diffsr_embed_dim,
            f_hidden_dims=self.config.diffsr_f_hidden_dims,
            g_hidden_dims=self.config.diffsr_g_hidden_dims,
            mu_hidden_dims=self.config.diffsr_mu_hidden_dims,
            num_noises=self.config.diffsr_num_noises,
            use_ema_for_policy=False,
            x_min=self.config.diffsr_x_min,
            x_max=self.config.diffsr_x_max,
            device=self.device,
        )

    def _initialize_diffsr_state_output(self) -> None:
        init_std = float(self.config.diffsr_state_output_init_std)
        if init_std <= 0.0:
            return
        state_net = getattr(self.diffsr, "state_net", None)
        fc2 = getattr(state_net, "fc2", None)
        if isinstance(fc2, nn.Linear):
            nn.init.normal_(fc2.weight, mean=0.0, std=init_std)
            nn.init.zeros_(fc2.bias)

    def _sample_macro_batch(
        self,
        batch_size: int,
        *,
        split: str | None,
    ) -> TensorDictBase:
        sampler = getattr(self.env, "sample_expert_macro_transition_batch", None)
        if not callable(sampler):
            msg = "env must expose sample_expert_macro_transition_batch(...)."
            raise ValueError(msg)
        return sampler(
            batch_size=int(batch_size),
            horizon_steps=int(self.config.horizon_steps),
            split=split,
            eval_fraction=float(self.config.eval_trajectory_fraction),
            split_seed=int(self.config.trajectory_split_seed),
        )

    def _sample_and_validate_macro_batch(
        self,
        batch_size: int,
        *,
        split: str | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch = self._sample_macro_batch(batch_size, split=split)
        return self._validate_macro_batch(batch, batch_size=batch_size)

    def _resolve_feature_slices(self) -> dict[str, tuple[int, int]]:
        provider = getattr(self.env, "expert_macro_feature_slices", None)
        if not callable(provider):
            return {}
        raw_slices = provider(horizon_steps=int(self.config.horizon_steps))
        if raw_slices is None:
            return {}
        feature_slices: dict[str, tuple[int, int]] = {}
        used_ranges: list[tuple[int, int, str]] = []
        for raw_name, raw_bounds in dict(raw_slices).items():
            name = str(raw_name)
            start, end = raw_bounds
            start = int(start)
            end = int(end)
            if start < 0 or end <= start or end > self.state_dim:
                msg = (
                    "Invalid expert macro feature slice for "
                    f"{name!r}: {(start, end)} with state_dim={self.state_dim}."
                )
                raise ValueError(msg)
            for other_start, other_end, other_name in used_ranges:
                if start < other_end and other_start < end:
                    msg = (
                        "Overlapping expert macro feature slices: "
                        f"{name!r}={(start, end)} overlaps "
                        f"{other_name!r}={(other_start, other_end)}."
                    )
                    raise ValueError(msg)
            used_ranges.append((start, end, name))
            feature_slices[name] = (start, end)
        return feature_slices

    def _validate_macro_batch(
        self,
        batch: TensorDictBase,
        *,
        batch_size: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        state = batch.get(("hl", "state"))
        future_window = batch.get(("hl", "future_window"))
        target = batch.get(("hl", "target"))
        missing = [
            name
            for name, value in (
                ("hl/state", state),
                ("hl/future_window", future_window),
                ("hl/target", target),
            )
            if value is None
        ]
        if missing:
            msg = f"Expert macro batch is missing keys: {missing}."
            raise ValueError(msg)
        state = cast(Tensor, state).to(device=self.device, dtype=torch.float32)
        future_window = cast(Tensor, future_window).to(
            device=self.device,
            dtype=torch.float32,
        )
        target = cast(Tensor, target).to(device=self.device, dtype=torch.float32)
        if state.ndim != 2:
            msg = f"hl/state must have shape [B, D], got {tuple(state.shape)}."
            raise ValueError(msg)
        state_dim = int(state.shape[-1])
        expected_state = (int(batch_size), state_dim)
        expected_window = (int(batch_size), self.config.horizon_steps, state_dim)
        if tuple(state.shape) != expected_state:
            msg = (
                "hl/state shape mismatch: expected "
                f"{expected_state}, got {tuple(state.shape)}."
            )
            raise ValueError(msg)
        if tuple(future_window.shape) != expected_window:
            msg = (
                "hl/future_window shape mismatch: expected "
                f"{expected_window}, got {tuple(future_window.shape)}."
            )
            raise ValueError(msg)
        if tuple(target.shape) != expected_state:
            msg = (
                "hl/target shape mismatch: expected "
                f"{expected_state}, got {tuple(target.shape)}."
            )
            raise ValueError(msg)
        return state, future_window, target

    @staticmethod
    def _effective_rank(z: Tensor) -> float:
        if int(z.shape[0]) < 2:
            return 0.0
        centered = z - z.mean(dim=0, keepdim=True)
        singular_values = torch.linalg.svdvals(centered)
        total = singular_values.sum()
        if float(total.item()) <= 1.0e-12:
            return 0.0
        probs = singular_values / total
        entropy = -(probs * probs.clamp_min(1.0e-12).log()).sum()
        return float(torch.exp(entropy).item())

    @classmethod
    def _z_diagnostics(cls, z: Tensor, *, prefix: str) -> dict[str, float]:
        z_std = z.std(dim=0, unbiased=False)
        return {
            f"{prefix}/z_abs_mean": float(z.abs().mean().item()),
            f"{prefix}/z_rms": float(z.pow(2).mean().sqrt().item()),
            f"{prefix}/z_dim_std_mean": float(z_std.mean().item()),
            f"{prefix}/z_dim_std_min": float(z_std.min().item()),
            f"{prefix}/z_effective_rank": cls._effective_rank(z),
        }

    def _diffsr_loss_for_z(self, state: Tensor, z: Tensor, target: Tensor) -> Tensor:
        zero_reward = torch.zeros(state.shape[0], 1, device=self.device)
        _, loss, _ = self.diffsr.compute_loss(state, z, target, zero_reward)
        return loss

    @staticmethod
    def _metric_safe_feature_name(name: str) -> str:
        return name.replace("/", "_").replace(" ", "_")

    def _state_error_metrics(
        self,
        error: Tensor,
        *,
        prefix: str,
        stem: str,
    ) -> dict[str, float]:
        metrics = {
            f"{prefix}/{stem}_l1": float(error.abs().sum(dim=-1).mean().item()),
            f"{prefix}/{stem}_mse": float(error.pow(2).sum(dim=-1).mean().item()),
            f"{prefix}/{stem}_dim_mse": float(error.pow(2).mean().item()),
        }
        for name, (start, end) in self.feature_slices.items():
            group_error = error[:, start:end]
            metric_name = self._metric_safe_feature_name(name)
            metrics.update(
                {
                    f"{prefix}/{stem}_{metric_name}_l1": float(
                        group_error.abs().sum(dim=-1).mean().item()
                    ),
                    f"{prefix}/{stem}_{metric_name}_mse": float(
                        group_error.pow(2).sum(dim=-1).mean().item()
                    ),
                    f"{prefix}/{stem}_{metric_name}_dim_mse": float(
                        group_error.pow(2).mean().item()
                    ),
                }
            )
        return metrics

    def _state_normalized_error_metrics(
        self,
        error: Tensor,
        target: Tensor,
        *,
        prefix: str,
        stem: str,
    ) -> dict[str, float]:
        variance = target.var(dim=0, unbiased=False).clamp_min(
            float(self.config.reconstruction_norm_eps)
        )
        normalized_sq = error.pow(2) / variance.unsqueeze(0)
        metrics = {
            f"{prefix}/norm_{stem}_mse": float(normalized_sq.sum(dim=-1).mean().item()),
            f"{prefix}/norm_{stem}_dim_mse": float(normalized_sq.mean().item()),
        }
        for name, (start, end) in self.feature_slices.items():
            group_normalized_sq = normalized_sq[:, start:end]
            metric_name = self._metric_safe_feature_name(name)
            metrics.update(
                {
                    f"{prefix}/norm_{stem}_{metric_name}_mse": float(
                        group_normalized_sq.sum(dim=-1).mean().item()
                    ),
                    f"{prefix}/norm_{stem}_{metric_name}_dim_mse": float(
                        group_normalized_sq.mean().item()
                    ),
                }
            )
        return metrics

    def _sample_reconstruction_metrics(
        self,
        state: Tensor,
        z: Tensor,
        target: Tensor,
        *,
        prefix: str,
    ) -> dict[str, float]:
        if not self.diffsr.supports_sampling:
            return {}
        sampled_target, _ = self.diffsr.sample(
            s=state,
            a=z,
            preserve_history=False,
        )
        error = sampled_target - target
        metrics = self._state_error_metrics(error, prefix=prefix, stem="sample_recon")
        metrics.update(
            self._state_normalized_error_metrics(
                error,
                target,
                prefix=prefix,
                stem="sample_recon",
            )
        )
        return metrics

    @staticmethod
    def _linear_probe_design(features: Tensor) -> Tensor:
        ones = torch.ones(
            features.shape[0],
            1,
            device=features.device,
            dtype=features.dtype,
        )
        return torch.cat([features, ones], dim=-1)

    @staticmethod
    def _solve_linear_probe(
        xtx: Tensor,
        xty: Tensor,
        *,
        ridge: float,
    ) -> Tensor:
        regularizer = torch.eye(
            xtx.shape[0],
            device=xtx.device,
            dtype=xtx.dtype,
        )
        regularizer[-1, -1] = 0.0
        return torch.linalg.solve(xtx + float(ridge) * regularizer, xty)

    def _window_probe_error_metrics(
        self,
        error_flat: Tensor,
        variance_flat: Tensor,
        *,
        prefix: str,
        stem: str,
    ) -> dict[str, float]:
        batch_size = int(error_flat.shape[0])
        error_window = error_flat.reshape(
            batch_size,
            self.config.horizon_steps,
            self.state_dim,
        )
        mid_step = int(self.config.horizon_steps // 2)
        normalized_sq = error_flat.pow(2) / variance_flat.unsqueeze(0)
        return {
            f"{prefix}/{stem}_mse": float(error_flat.pow(2).sum(dim=-1).mean().item()),
            f"{prefix}/{stem}_dim_mse": float(error_flat.pow(2).mean().item()),
            f"{prefix}/{stem}_norm_dim_mse": float(normalized_sq.mean().item()),
            f"{prefix}/{stem}_step_first_dim_mse": float(
                error_window[:, 0, :].pow(2).mean().item()
            ),
            f"{prefix}/{stem}_step_mid_dim_mse": float(
                error_window[:, mid_step, :].pow(2).mean().item()
            ),
            f"{prefix}/{stem}_step_final_dim_mse": float(
                error_window[:, -1, :].pow(2).mean().item()
            ),
        }

    @torch.no_grad()
    def evaluate_window_probe(
        self,
        *,
        train_batches: int = 4,
        eval_batches: int | None = None,
        batch_size: int | None = None,
        prefix: str = "train",
        train_split: str | None = None,
        eval_split: str | None = None,
        ridge: float = 1.0e-3,
    ) -> dict[str, float]:
        train_batches = _require_positive_int("train_batches", train_batches)
        eval_batches = (
            self.config.eval_batches if eval_batches is None else eval_batches
        )
        eval_batches = _require_positive_int("eval_batches", eval_batches)
        batch_size = (
            self.config.eval_batch_size or self.config.batch_size
            if batch_size is None
            else batch_size
        )
        batch_size = _require_positive_int("batch_size", batch_size)
        ridge = _require_non_negative_float("ridge", ridge)
        train_split = self.config.train_split if train_split is None else train_split
        eval_split = self.config.eval_split if eval_split is None else eval_split

        encoder_was_training = self.skill_encoder.training
        self.skill_encoder.eval()

        z_xtx: Tensor | None = None
        z_xty: Tensor | None = None
        state_xtx: Tensor | None = None
        state_xty: Tensor | None = None
        target_sum: Tensor | None = None
        target_sq_sum: Tensor | None = None
        train_samples = 0

        for _ in range(train_batches):
            state, future_window, _ = self._sample_and_validate_macro_batch(
                batch_size,
                split=train_split,
            )
            z = self._encode_skill(state, future_window)
            target_flat = future_window.reshape(batch_size, -1)
            z_design = self._linear_probe_design(z).to(torch.float64)
            state_design = self._linear_probe_design(state).to(torch.float64)
            target64 = target_flat.to(torch.float64)

            batch_z_xtx = z_design.T @ z_design
            batch_z_xty = z_design.T @ target64
            batch_state_xtx = state_design.T @ state_design
            batch_state_xty = state_design.T @ target64
            z_xtx = batch_z_xtx if z_xtx is None else z_xtx + batch_z_xtx
            z_xty = batch_z_xty if z_xty is None else z_xty + batch_z_xty
            state_xtx = (
                batch_state_xtx if state_xtx is None else state_xtx + batch_state_xtx
            )
            state_xty = (
                batch_state_xty if state_xty is None else state_xty + batch_state_xty
            )
            batch_target_sum = target64.sum(dim=0)
            batch_target_sq_sum = target64.pow(2).sum(dim=0)
            target_sum = (
                batch_target_sum
                if target_sum is None
                else target_sum + batch_target_sum
            )
            target_sq_sum = (
                batch_target_sq_sum
                if target_sq_sum is None
                else target_sq_sum + batch_target_sq_sum
            )
            train_samples += int(batch_size)

        assert z_xtx is not None
        assert z_xty is not None
        assert state_xtx is not None
        assert state_xty is not None
        assert target_sum is not None
        assert target_sq_sum is not None
        z_weights = self._solve_linear_probe(z_xtx, z_xty, ridge=ridge)
        state_weights = self._solve_linear_probe(state_xtx, state_xty, ridge=ridge)
        mean_flat = target_sum / float(train_samples)
        variance_flat = (
            target_sq_sum / float(train_samples) - mean_flat.pow(2)
        ).clamp_min(float(self.config.reconstruction_norm_eps))

        accum: dict[str, float] = {}
        eval_samples = 0
        for _ in range(eval_batches):
            state, future_window, _ = self._sample_and_validate_macro_batch(
                batch_size,
                split=eval_split,
            )
            z = self._encode_skill(state, future_window)
            if int(z.shape[0]) > 1:
                shuffled_z = z[torch.randperm(z.shape[0], device=z.device)]
            else:
                shuffled_z = z.clone()
            target_flat = future_window.reshape(batch_size, -1).to(torch.float64)
            z_design = self._linear_probe_design(z).to(torch.float64)
            shuffled_z_design = self._linear_probe_design(shuffled_z).to(torch.float64)
            state_design = self._linear_probe_design(state).to(torch.float64)

            prediction_errors = {
                "window_probe_z": z_design @ z_weights - target_flat,
                "window_probe_z_shuffled": shuffled_z_design @ z_weights - target_flat,
                "window_probe_state": state_design @ state_weights - target_flat,
                "window_probe_mean": mean_flat.unsqueeze(0) - target_flat,
            }
            batch_metrics: dict[str, float] = {}
            for stem, error_flat in prediction_errors.items():
                batch_metrics.update(
                    self._window_probe_error_metrics(
                        error_flat,
                        variance_flat,
                        prefix=prefix,
                        stem=stem,
                    )
                )
            for key, value in batch_metrics.items():
                accum[key] = accum.get(key, 0.0) + float(value) * float(batch_size)
            eval_samples += int(batch_size)

        for key in accum:
            accum[key] /= float(eval_samples)
        accum[f"{prefix}/window_probe_train_samples"] = float(train_samples)
        accum[f"{prefix}/window_probe_eval_samples"] = float(eval_samples)
        accum[f"{prefix}/window_probe_ridge"] = float(ridge)

        if encoder_was_training:
            self.skill_encoder.train()
        return accum

    def train_step(self) -> dict[str, float]:
        self.skill_encoder.train()
        self.diffsr.train()
        state, future_window, target = self._sample_and_validate_macro_batch(
            self.config.batch_size,
            split=self.config.train_split,
        )
        self.diffsr.update_obs_norm(target.detach())
        z = self._encode_skill(state, future_window)
        diffsr_loss = self._diffsr_loss_for_z(state, z, target)
        z_norm_loss = z.pow(2).mean()
        loss = diffsr_loss + self.config.z_norm_coeff * z_norm_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        metrics = self._z_diagnostics(z.detach(), prefix="train")
        if self.config.grad_clip_norm is not None:
            params = [*self.skill_encoder.parameters(), *self.diffsr.parameters()]
            grad_norm = torch.nn.utils.clip_grad_norm_(
                params,
                max_norm=float(self.config.grad_clip_norm),
            )
            metrics["train/grad_norm"] = float(grad_norm.item())
        self.optimizer.step()
        self.update += 1
        metrics.update(
            {
                "train/loss": float(loss.detach().item()),
                "train/diffsr_loss": float(diffsr_loss.detach().item()),
                "train/z_norm_loss": float(z_norm_loss.detach().item()),
            }
        )
        return metrics

    @torch.no_grad()
    def evaluate(
        self,
        *,
        num_batches: int | None = None,
        batch_size: int | None = None,
        prefix: str = "train",
        include_reconstruction: bool = False,
        split: str | None = None,
    ) -> dict[str, float]:
        num_batches = self.config.eval_batches if num_batches is None else num_batches
        batch_size = (
            self.config.eval_batch_size or self.config.batch_size
            if batch_size is None
            else batch_size
        )
        num_batches = _require_positive_int("num_batches", num_batches)
        batch_size = _require_positive_int("batch_size", batch_size)
        split = self.config.eval_split if split is None else split

        encoder_was_training = self.skill_encoder.training
        diffsr_was_training = self.diffsr.training
        self.skill_encoder.eval()
        self.diffsr.eval()
        accum: dict[str, float] = {}
        for _ in range(num_batches):
            state, future_window, target = self._sample_and_validate_macro_batch(
                batch_size,
                split=split,
            )
            z = self._encode_skill(state, future_window)
            zero_z = torch.zeros_like(z)
            if int(z.shape[0]) > 1:
                shuffled_z = z[torch.randperm(z.shape[0], device=z.device)]
            else:
                shuffled_z = z.clone()
            real_loss = self._diffsr_loss_for_z(state, z, target)
            zero_loss = self._diffsr_loss_for_z(state, zero_z, target)
            shuffled_loss = self._diffsr_loss_for_z(state, shuffled_z, target)
            batch_metrics = self._z_diagnostics(z, prefix=prefix)
            batch_metrics.update(
                {
                    f"{prefix}/loss_real_z_eval": float(real_loss.item()),
                    f"{prefix}/loss_zero_z_eval": float(zero_loss.item()),
                    f"{prefix}/loss_shuffled_z_eval": float(shuffled_loss.item()),
                }
            )
            if include_reconstruction:
                batch_metrics.update(
                    self._sample_reconstruction_metrics(
                        state,
                        z,
                        target,
                        prefix=prefix,
                    )
                )
            for key, value in batch_metrics.items():
                accum[key] = accum.get(key, 0.0) + float(value)
        for key in accum:
            accum[key] /= float(num_batches)
        if encoder_was_training:
            self.skill_encoder.train()
        if diffsr_was_training:
            self.diffsr.train()
        return accum

    def train(
        self,
        *,
        log_callback: Callable[[dict[str, float | int]], None] | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> HighLevelSkillDiffSRTrainState:
        start_time = time.perf_counter()
        state = HighLevelSkillDiffSRTrainState()
        for _ in range(self.config.num_updates):
            metrics = self.train_step()
            should_log = (
                self.update in (1, self.config.num_updates)
                or self.update % self.config.log_interval == 0
            )
            if should_log:
                metrics.update(self.evaluate(prefix="train"))
                elapsed = time.perf_counter() - start_time
                row: dict[str, float | int] = {
                    "update": int(self.update),
                    "elapsed_seconds": float(elapsed),
                    **metrics,
                }
                if log_callback is not None:
                    log_callback(row)
                state.last_metrics = {
                    key: float(value) for key, value in metrics.items()
                }
                state.elapsed_seconds = float(elapsed)
                state.update = int(self.update)
        if checkpoint_path is not None:
            self.save_checkpoint(checkpoint_path)
        return state

    def checkpoint_state_dict(self) -> dict[str, Any]:
        obs_norm = getattr(self.diffsr, "obs_norm", None)
        feature_norm_state = (
            obs_norm.state_dict() if isinstance(obs_norm, nn.Module) else {}
        )
        return {
            "skill_encoder_state_dict": self.skill_encoder.state_dict(),
            "diffsr_state_dict": self.diffsr.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "update": int(self.update),
            "feature_normalization_state_dict": feature_norm_state,
        }

    def save_checkpoint(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_state_dict(), target)

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        checkpoint = torch.load(
            Path(path), map_location=self.device, weights_only=False
        )
        loaded_config = HighLevelSkillDiffSRConfig.from_dict(checkpoint["config"])
        if loaded_config.horizon_steps != self.config.horizon_steps:
            msg = (
                "Checkpoint horizon_steps does not match trainer construction: "
                f"{loaded_config.horizon_steps} != {self.config.horizon_steps}."
            )
            raise ValueError(msg)
        if loaded_config.z_dim != self.config.z_dim:
            msg = (
                "Checkpoint z_dim does not match trainer construction: "
                f"{loaded_config.z_dim} != {self.config.z_dim}."
            )
            raise ValueError(msg)
        if loaded_config.encoder_window_mode != self.config.encoder_window_mode:
            msg = (
                "Checkpoint encoder_window_mode does not match trainer construction: "
                f"{loaded_config.encoder_window_mode!r} != {self.config.encoder_window_mode!r}."
            )
            raise ValueError(msg)
        self.config = loaded_config
        self.encoder_window_steps = self._encoder_window_steps()
        self.skill_encoder.load_state_dict(checkpoint["skill_encoder_state_dict"])
        self.diffsr.load_state_dict(checkpoint["diffsr_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        feature_norm_state = checkpoint.get("feature_normalization_state_dict")
        obs_norm = getattr(self.diffsr, "obs_norm", None)
        if isinstance(obs_norm, nn.Module) and feature_norm_state:
            obs_norm.load_state_dict(feature_norm_state)
        self.update = int(checkpoint.get("update", 0))
        return cast(dict[str, Any], checkpoint)
