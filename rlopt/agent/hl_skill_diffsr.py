from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
from tensordict import TensorDictBase
from torch import Tensor, nn

from rlopt.agent.hl_skill_encoder import (
    LATENT_MODES as _LATENT_MODES,
)
from rlopt.agent.hl_skill_encoder import (
    SkillLatentSpec,
    build_skill_encoder,
)
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


def _normalize_command_mode(value: str) -> str:
    normalized = str(value).strip().lower()
    aliases = {"fz": "phi", "z_fz": "z_phi"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"z", "phi", "z_phi"}:
        msg = (
            "command_mode must be one of 'z', 'phi', or 'z_phi' "
            f"(aliases: 'fz', 'z_fz'), got {value!r}."
        )
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


def _resolve_device(device: torch.device | str | None, env: object) -> torch.device:
    if device is not None and str(device).strip().lower() != "auto":
        return torch.device(device)
    env_device = getattr(env, "device", None)
    if env_device is not None:
        return torch.device(env_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _encoder_window_steps(config: HighLevelSkillDiffSRConfig) -> int:
    if config.encoder_window_mode == "intermediate":
        return int(config.horizon_steps) - 1
    return int(config.horizon_steps)


def _encoder_input_window(
    config: HighLevelSkillDiffSRConfig, future_window: Tensor
) -> Tensor:
    if config.encoder_window_mode == "intermediate":
        return future_window[:, :-1, :]
    return future_window


def _build_diffsr(
    config: HighLevelSkillDiffSRConfig, state_dim: int, device: torch.device
) -> BilinearSR:
    return build_bilinear_sr(
        "diffsr",
        obs_dim=state_dim,
        next_obs_dim=state_dim,
        action_dim=config.z_dim,
        feature_dim=config.diffsr_feature_dim,
        embed_dim=config.diffsr_embed_dim,
        g_hidden_dims=config.diffsr_g_hidden_dims,
        mu_hidden_dims=config.diffsr_mu_hidden_dims,
        num_noises=config.diffsr_num_noises,
        use_ema_for_policy=False,
        x_min=config.diffsr_x_min,
        x_max=config.diffsr_x_max,
        device=device,
    )


def _validate_macro_batch(
    batch: TensorDictBase,
    *,
    batch_size: int,
    horizon_steps: int,
    device: torch.device,
    state_dim: int | None = None,
    source: str = "Expert",
) -> tuple[Tensor, Tensor, Tensor]:
    """Validate and materialize ``hl/{state,future_window,target}`` from a macro batch.

    ``state_dim=None`` infers the state width from the batch (offline trainer /
    preflight); a provided ``state_dim`` is asserted (online sampler). ``source``
    only prefixes the missing-keys message.
    """
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
        msg = f"{source} macro batch is missing keys: {missing}."
        raise ValueError(msg)
    state = cast(Tensor, state).to(device=device, dtype=torch.float32)
    future_window = cast(Tensor, future_window).to(device=device, dtype=torch.float32)
    target = cast(Tensor, target).to(device=device, dtype=torch.float32)
    if state.ndim != 2:
        msg = f"hl/state must have shape [B, D], got {tuple(state.shape)}."
        raise ValueError(msg)
    resolved_dim = int(state.shape[-1]) if state_dim is None else int(state_dim)
    expected_state = (int(batch_size), resolved_dim)
    expected_window = (int(batch_size), int(horizon_steps), resolved_dim)
    if tuple(state.shape) != expected_state:
        msg = (
            f"hl/state shape mismatch: expected {expected_state}, "
            f"got {tuple(state.shape)}."
        )
        raise ValueError(msg)
    if tuple(future_window.shape) != expected_window:
        msg = (
            f"hl/future_window shape mismatch: expected {expected_window}, "
            f"got {tuple(future_window.shape)}."
        )
        raise ValueError(msg)
    if tuple(target.shape) != expected_state:
        msg = (
            f"hl/target shape mismatch: expected {expected_state}, "
            f"got {tuple(target.shape)}."
        )
        raise ValueError(msg)
    return state, future_window, target


def _effective_rank(z: Tensor) -> Tensor:
    """Participation-ratio effective rank of ``z`` as a 0-dim tensor."""
    if int(z.shape[0]) < 2:
        return z.new_zeros(())
    centered = z - z.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    total = singular_values.sum()
    if bool((total <= 1.0e-12).item()):
        return z.new_zeros(())
    probs = singular_values / total
    entropy = -(probs * probs.clamp_min(1.0e-12).log()).sum()
    return torch.exp(entropy)


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
    latent_mode: str = "deterministic"
    reg_coeff: float = 1.0e-3
    categorical_groups: int = 8
    categorical_categories: int = 32
    gaussian_logstd_min: float = -5.0
    gaussian_logstd_max: float = 2.0
    gumbel_codebook_size: int = 512
    gumbel_tau_start: float = 2.0
    gumbel_tau_end: float = 0.5
    gumbel_tau_anneal_iters: int = 2000
    gumbel_hard: bool = True
    fsq_levels: tuple[int, ...] = (8, 8, 8, 5, 5)
    vq_codebook_size: int = 512
    vq_ema_decay: float = 0.99
    vq_dead_code_reset_iters: int = 0
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
    reconstruction_norm_eps: float = 1.0e-6
    device: str = "auto"
    diffsr_state_output_init_std: float = 1.0e-3
    # Optional co-trained skill commander (System-1 planner). When enabled, a
    # SkillCommander is BC'd to the encoder's z (detached) from the current
    # state + language goal, jointly with the encoder/DiffSR pretraining.
    cotrain_commander: bool = False
    commander_language_embeddings_path: str = ""
    commander_hidden_dims: tuple[int, ...] = (1024, 512, 512)
    commander_lr: float = 3.0e-4
    commander_cosine_loss_coeff: float = 1.0
    commander_z_norm_coeff: float = 1.0e-4
    commander_state_noise_std: float = 0.0

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
        if self.latent_mode not in _LATENT_MODES:
            msg = (
                f"latent_mode must be one of {_LATENT_MODES}, got {self.latent_mode!r}."
            )
            raise ValueError(msg)
        self.reg_coeff = _require_non_negative_float("reg_coeff", self.reg_coeff)
        self.categorical_groups = _require_positive_int(
            "categorical_groups", self.categorical_groups
        )
        self.categorical_categories = _require_positive_int(
            "categorical_categories", self.categorical_categories
        )
        if (
            self.latent_mode in ("categorical", "gumbel_multicat")
            and self.z_dim % self.categorical_groups != 0
        ):
            msg = (
                f"latent_mode={self.latent_mode!r} requires z_dim divisible by "
                f"categorical_groups (per-group code dim = z_dim // groups): "
                f"z_dim={self.z_dim}, categorical_groups={self.categorical_groups}."
            )
            raise ValueError(msg)
        self.gaussian_logstd_min = float(self.gaussian_logstd_min)
        self.gaussian_logstd_max = float(self.gaussian_logstd_max)
        if self.gaussian_logstd_max <= self.gaussian_logstd_min:
            msg = (
                "gaussian_logstd_max must be > gaussian_logstd_min, got "
                f"{self.gaussian_logstd_max} <= {self.gaussian_logstd_min}."
            )
            raise ValueError(msg)
        self.gumbel_codebook_size = _require_positive_int(
            "gumbel_codebook_size", self.gumbel_codebook_size
        )
        self.gumbel_hard = bool(self.gumbel_hard)
        self.fsq_levels = tuple(
            _require_positive_int("fsq_levels", level) for level in self.fsq_levels
        )
        if self.latent_mode == "fsq" and any(level < 2 for level in self.fsq_levels):
            msg = f"fsq_levels must each be >= 2, got {self.fsq_levels!r}."
            raise ValueError(msg)
        self.vq_codebook_size = _require_positive_int(
            "vq_codebook_size", self.vq_codebook_size
        )
        self.vq_dead_code_reset_iters = int(self.vq_dead_code_reset_iters)
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
        self.reconstruction_norm_eps = _require_positive_float(
            "reconstruction_norm_eps", self.reconstruction_norm_eps
        )
        self.diffsr_state_output_init_std = _require_non_negative_float(
            "diffsr_state_output_init_std", self.diffsr_state_output_init_std
        )
        self.cotrain_commander = bool(self.cotrain_commander)
        self.commander_hidden_dims = tuple(
            _require_positive_int("commander_hidden_dims", dim)
            for dim in self.commander_hidden_dims
        )
        self.commander_lr = _require_positive_float("commander_lr", self.commander_lr)
        self.commander_cosine_loss_coeff = _require_non_negative_float(
            "commander_cosine_loss_coeff", self.commander_cosine_loss_coeff
        )
        self.commander_z_norm_coeff = _require_non_negative_float(
            "commander_z_norm_coeff", self.commander_z_norm_coeff
        )
        self.commander_state_noise_std = _require_non_negative_float(
            "commander_state_noise_std", self.commander_state_noise_std
        )
        self.commander_language_embeddings_path = str(
            self.commander_language_embeddings_path
        ).strip()
        if self.cotrain_commander and not self.commander_language_embeddings_path:
            msg = (
                "commander_language_embeddings_path is required when "
                "cotrain_commander is enabled."
            )
            raise ValueError(msg)
        self.device = str(self.device)

    def latent_spec(self) -> SkillLatentSpec:
        """Project the latent-method fields into the encoder factory's spec."""
        return SkillLatentSpec(
            latent_mode=self.latent_mode,
            gaussian_logstd_min=self.gaussian_logstd_min,
            gaussian_logstd_max=self.gaussian_logstd_max,
            categorical_groups=self.categorical_groups,
            categorical_categories=self.categorical_categories,
            gumbel_codebook_size=self.gumbel_codebook_size,
            gumbel_tau_start=self.gumbel_tau_start,
            gumbel_tau_end=self.gumbel_tau_end,
            gumbel_tau_anneal_iters=self.gumbel_tau_anneal_iters,
            gumbel_hard=self.gumbel_hard,
            fsq_levels=tuple(self.fsq_levels),
            vq_codebook_size=self.vq_codebook_size,
            vq_ema_decay=self.vq_ema_decay,
            vq_dead_code_reset_iters=self.vq_dead_code_reset_iters,
        )

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
            "fsq_levels",
            "commander_hidden_dims",
        }
        for key in tuple_fields:
            if key in kwargs:
                kwargs[key] = tuple(int(item) for item in kwargs[key])
        config = cls(**kwargs)
        config.validate()
        return config


class FrozenHighLevelSkillCommandSampler:
    """High-level encoder used as an online latent-command source.

    The default mode is frozen and preserves the original rollout behavior.  When
    online finetuning is enabled, the sampler also caches one macro input per
    renewed command so PPO minibatches can recompute skill commands with gradient
    flow into the skill encoder.
    """

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
        command_mode: str = "z",
        device: torch.device | str | None = None,
        finetune_enabled: bool = False,
        pg_coeff: float = 0.05,
        offline_diffsr_coeff: float = 1.0,
        anchor_coeff: float = 0.01,
        z_norm_coeff: float | None = None,
        lr: float = 3.0e-5,
        grad_clip_norm: float | None = 1.0,
        offline_batch_size: int = 8192,
        update_interval: int = 1,
        train_diffsr: bool = False,
    ) -> None:
        self.latent_dim = _require_positive_int("latent_dim", latent_dim)
        self.latent_steps_min = max(1, int(latent_steps_min))
        self.latent_steps_max = max(self.latent_steps_min, int(latent_steps_max))
        self.finetune_enabled = bool(finetune_enabled)
        self.pg_coeff = _require_non_negative_float("pg_coeff", pg_coeff)
        self.offline_diffsr_coeff = _require_non_negative_float(
            "offline_diffsr_coeff", offline_diffsr_coeff
        )
        self.anchor_coeff = _require_non_negative_float("anchor_coeff", anchor_coeff)
        self.lr = _require_positive_float("lr", lr)
        self.grad_clip_norm = (
            None
            if grad_clip_norm is None
            else _require_positive_float("grad_clip_norm", grad_clip_norm)
        )
        self.offline_batch_size = _require_positive_int(
            "offline_batch_size", offline_batch_size
        )
        self.update_interval = _require_positive_int("update_interval", update_interval)
        self.train_diffsr = bool(train_diffsr)
        self.command_mode = _normalize_command_mode(command_mode)
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
        self.device = _resolve_device(device, env)
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
        self._offline_macro_sampler = discover_env_method(
            env,
            "sample_expert_macro_transition_batch",
        )
        if self.finetune_enabled and self._offline_macro_sampler is None:
            msg = (
                "Online high-level skill finetuning requires the environment to "
                "expose sample_expert_macro_transition_batch(...)."
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
        self.skill_z_dim = int(self.config.z_dim)
        self.command_code_dim = self._command_code_dim_for_mode()
        if (
            code_latent_dim is not None
            and int(code_latent_dim) > 0
            and int(code_latent_dim) != self.command_code_dim
        ):
            msg = (
                "Frozen high-level skill code_latent_dim must match command-mode "
                f"pre-phase width: {int(code_latent_dim)} != "
                f"{self.command_code_dim} for command_mode={self.command_mode!r}."
            )
            raise ValueError(msg)
        expected_latent_dim = self.command_code_dim + self.phase_dim
        if expected_latent_dim != self.latent_dim:
            msg = (
                "Frozen high-level skill command width must match ipmd.latent_dim: "
                f"command_mode={self.command_mode!r} pre-phase width "
                f"{self.command_code_dim} + phase_dim {self.phase_dim} "
                f"!= {self.latent_dim}."
            )
            raise ValueError(msg)

        state_dict = checkpoint["skill_encoder_state_dict"]
        self.encoder_window_steps = _encoder_window_steps(self.config)
        self.state_dim = self._state_dim_from_encoder_state(
            state_dict,
            window_steps=self.encoder_window_steps,
        )
        self.skill_encoder = build_skill_encoder(
            state_dim=self.state_dim,
            window_steps=self.encoder_window_steps,
            z_dim=self.config.z_dim,
            hidden_dims=self.config.encoder_hidden_dims,
            spec=self.config.latent_spec(),
        ).to(self.device)
        self.skill_encoder.load_state_dict(state_dict)

        self.initial_skill_encoder = build_skill_encoder(
            state_dim=self.state_dim,
            window_steps=self.encoder_window_steps,
            z_dim=self.config.z_dim,
            hidden_dims=self.config.encoder_hidden_dims,
            spec=self.config.latent_spec(),
        ).to(self.device)
        self.initial_skill_encoder.load_state_dict(state_dict)
        self.initial_skill_encoder.eval()
        self.initial_skill_encoder.requires_grad_(False)

        self.diffsr = _build_diffsr(self.config, self.state_dim, self.device).to(
            self.device
        )
        diffsr_state = checkpoint.get("diffsr_state_dict")
        if diffsr_state is None and (self.finetune_enabled or self.command_mode != "z"):
            msg = (
                "Online high-level skill finetuning and non-z command modes "
                "require checkpoints with diffsr_state_dict."
            )
            raise ValueError(msg)
        if diffsr_state is not None:
            self.diffsr.load_state_dict(diffsr_state)
        feature_norm_state = checkpoint.get("feature_normalization_state_dict")
        obs_norm = getattr(self.diffsr, "obs_norm", None)
        if isinstance(obs_norm, nn.Module) and feature_norm_state:
            obs_norm.load_state_dict(feature_norm_state)

        self.z_norm_coeff = (
            _require_non_negative_float("z_norm_coeff", z_norm_coeff)
            if z_norm_coeff is not None
            else float(self.config.reg_coeff)
        )

        self.skill_encoder.train(self.finetune_enabled)
        self.skill_encoder.requires_grad_(self.finetune_enabled)
        self.diffsr.train(self.finetune_enabled and self.train_diffsr)
        self.diffsr.requires_grad_(self.finetune_enabled and self.train_diffsr)

        self.optimizer: torch.optim.Optimizer | None = None
        if self.finetune_enabled:
            params: list[nn.Parameter] = list(self.skill_encoder.parameters())
            if self.train_diffsr:
                params.extend(list(self.diffsr.parameters()))
            self.optimizer = torch.optim.Adam(params, lr=self.lr)
        self.finetune_updates = 0

        self._codes: Tensor | None = None
        self._latent_steps: Tensor | None = None
        self._active_macro_ids: Tensor | None = None
        self._cache_state_chunks: list[Tensor] = []
        self._cache_future_window_chunks: list[Tensor] = []
        self._cache_target_chunks: list[Tensor] = []
        self._cache_initial_z_chunks: list[Tensor] = []
        self._next_macro_id = 0

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

    def _command_code_dim_for_mode(self) -> int:
        if self.command_mode == "z":
            return int(self.config.z_dim)
        if self.command_mode == "phi":
            return int(self.config.diffsr_feature_dim)
        if self.command_mode == "z_phi":
            return int(self.config.z_dim) + int(self.config.diffsr_feature_dim)
        msg = f"Unsupported high-level skill command mode: {self.command_mode!r}."
        raise ValueError(msg)

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

    def _sample_offline_macro_batch(
        self,
        batch_size: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if self._offline_macro_sampler is None:
            msg = "sample_expert_macro_transition_batch(...) is unavailable."
            raise RuntimeError(msg)
        batch = self._offline_macro_sampler(
            batch_size=int(batch_size),
            horizon_steps=int(self.config.horizon_steps),
            split=self.config.train_split,
            eval_fraction=float(self.config.eval_trajectory_fraction),
            split_seed=int(self.config.trajectory_split_seed),
        )
        return _validate_macro_batch(
            batch,
            batch_size=int(batch_size),
            horizon_steps=int(self.config.horizon_steps),
            device=self.device,
            state_dim=self.state_dim,
            source="Offline expert",
        )

    def _command_code_from_state_z(self, state: Tensor, z: Tensor) -> Tensor:
        if self.command_mode == "z":
            return z
        phi = self.diffsr.forward_phi(state, z)
        if self.command_mode == "phi":
            return phi
        if self.command_mode == "z_phi":
            return torch.cat((z, phi), dim=-1)
        msg = f"Unsupported high-level skill command mode: {self.command_mode!r}."
        raise ValueError(msg)

    def _append_command_phase(self, code_latents: Tensor, phase: Tensor) -> Tensor:
        if self.phase_dim == 0:
            return code_latents
        angle = phase.to(device=code_latents.device, dtype=code_latents.dtype)
        angle = angle.reshape(-1) * (2.0 * math.pi)
        phase_features = torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)
        return torch.cat((code_latents, phase_features), dim=-1)

    @torch.no_grad()
    def _encode_current_macro_batch(
        self,
        env_ids: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch = self._current_macro_sampler(
            horizon_steps=int(self.config.horizon_steps),
            env_ids=env_ids,
        )
        batch_size = int(env_ids.numel())
        state, future_window, target = _validate_macro_batch(
            batch,
            batch_size=batch_size,
            horizon_steps=int(self.config.horizon_steps),
            device=self.device,
            state_dim=self.state_dim,
            source="Current expert",
        )
        z = self.skill_encoder(state, _encoder_input_window(self.config, future_window))
        initial_z = self.initial_skill_encoder(
            state,
            _encoder_input_window(self.config, future_window),
        )
        return z, state, future_window, target, initial_z

    def start_rollout_cache(self) -> None:
        """Reset one-rollout macro cache used by online skill finetuning."""
        if not self.finetune_enabled:
            return
        self._cache_state_chunks.clear()
        self._cache_future_window_chunks.clear()
        self._cache_target_chunks.clear()
        self._cache_initial_z_chunks.clear()
        self._next_macro_id = 0
        if self._latent_steps is not None:
            self._latent_steps.zero_()
        if self._active_macro_ids is not None:
            self._active_macro_ids.fill_(-1)

    def _append_rollout_cache(
        self,
        *,
        state: Tensor,
        future_window: Tensor,
        target: Tensor,
        initial_z: Tensor,
    ) -> Tensor:
        count = int(state.shape[0])
        macro_ids = torch.arange(
            self._next_macro_id,
            self._next_macro_id + count,
            device=self.device,
            dtype=torch.long,
        )
        self._next_macro_id += count
        self._cache_state_chunks.append(state.detach().clone())
        self._cache_future_window_chunks.append(future_window.detach().clone())
        self._cache_target_chunks.append(target.detach().clone())
        self._cache_initial_z_chunks.append(initial_z.detach().clone())
        return macro_ids

    def _cached_rollout_tensors(self) -> dict[str, Tensor]:
        if not self._cache_state_chunks:
            msg = "No high-level skill macro cache is available for this rollout."
            raise RuntimeError(msg)
        return {
            "state": torch.cat(self._cache_state_chunks, dim=0),
            "future_window": torch.cat(self._cache_future_window_chunks, dim=0),
            "target": torch.cat(self._cache_target_chunks, dim=0),
            "initial_z": torch.cat(self._cache_initial_z_chunks, dim=0),
        }

    def _z_diagnostics_tensors(self, z: Tensor, *, prefix: str) -> dict[str, Tensor]:
        if int(z.shape[0]) < 2:
            rank = torch.zeros((), device=z.device, dtype=z.dtype)
        else:
            centered = z - z.mean(dim=0, keepdim=True)
            singular_values = torch.linalg.svdvals(centered)
            total = singular_values.sum()
            if bool((total <= 1.0e-12).item()):
                rank = torch.zeros((), device=z.device, dtype=z.dtype)
            else:
                probs = singular_values / total
                entropy = -(probs * probs.clamp_min(1.0e-12).log()).sum()
                rank = torch.exp(entropy)
        z_std = z.std(dim=0, unbiased=False)
        return {
            f"{prefix}_z_abs_mean": z.abs().mean().detach(),
            f"{prefix}_z_rms": z.pow(2).mean().sqrt().detach(),
            f"{prefix}_z_dim_std_mean": z_std.mean().detach(),
            f"{prefix}_z_effective_rank": rank.detach(),
        }

    def latent_commands_from_rollout_batch(
        self,
        batch: TensorDictBase,
        *,
        detach: bool,
    ) -> Tensor:
        macro_id = cast(Tensor, batch.get(("hl_skill", "macro_id"))).reshape(-1)
        phase = cast(Tensor, batch.get(("hl_skill", "phase"))).reshape(-1)
        if macro_id.numel() == 0:
            return torch.empty(0, self.latent_dim, device=self.device)
        macro_id = macro_id.to(device=self.device, dtype=torch.long)
        cached = self._cached_rollout_tensors()
        if bool((macro_id < 0).any()) or bool(
            (macro_id >= cached["state"].shape[0]).any()
        ):
            msg = "Rollout contains high-level skill macro IDs outside the cache."
            raise RuntimeError(msg)
        state = cached["state"].index_select(0, macro_id)
        future_window = cached["future_window"].index_select(0, macro_id)
        if detach:
            with torch.no_grad():
                z = self.skill_encoder(
                    state,
                    _encoder_input_window(self.config, future_window),
                )
                command_code = self._command_code_from_state_z(state, z)
                command = self._append_command_phase(command_code, phase)
        else:
            z = self.skill_encoder(
                state, _encoder_input_window(self.config, future_window)
            )
            command_code = self._command_code_from_state_z(state, z)
            command = self._append_command_phase(command_code, phase)
        return command.reshape(*batch.batch_size, self.latent_dim)

    def should_update_online(self, update_idx: int) -> bool:
        return self.finetune_enabled and int(update_idx) % self.update_interval == 0

    def trainable_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = list(self.skill_encoder.parameters())
        if self.train_diffsr:
            params.extend(list(self.diffsr.parameters()))
        return params

    def compute_online_finetune_loss(
        self,
        batch: TensorDictBase,
        *,
        latent_key: str | tuple[str, ...],
        actor_loss_fn: Callable[[TensorDictBase], Tensor],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if not self.finetune_enabled:
            zero = torch.zeros((), device=self.device)
            return zero, {}

        pg_loss = torch.zeros((), device=self.device)
        if self.pg_coeff > 0.0:
            command = self.latent_commands_from_rollout_batch(batch, detach=False)
            pg_batch = batch.clone(False)
            pg_batch.set(latent_key, command)
            pg_loss = actor_loss_fn(pg_batch)

        offline_size = int(self.offline_batch_size)
        state, future_window, target = self._sample_offline_macro_batch(offline_size)
        z = self.skill_encoder(state, _encoder_input_window(self.config, future_window))
        zero_reward = torch.zeros(state.shape[0], 1, device=self.device)
        _, diffsr_loss, _ = self.diffsr.compute_loss(state, z, target, zero_reward)
        with torch.no_grad():
            initial_z = self.initial_skill_encoder(
                state,
                _encoder_input_window(self.config, future_window),
            )
        anchor_loss = F.mse_loss(z, initial_z)
        z_norm_loss = z.pow(2).mean()

        total_loss = (
            self.pg_coeff * pg_loss
            + self.offline_diffsr_coeff * diffsr_loss
            + self.anchor_coeff * anchor_loss
            + self.z_norm_coeff * z_norm_loss
        )
        metrics = {
            "hl_skill_total_loss": total_loss.detach(),
            "hl_skill_pg_loss": pg_loss.detach(),
            "hl_skill_diffsr_loss": diffsr_loss.detach(),
            "hl_skill_anchor_loss": anchor_loss.detach(),
            "hl_skill_z_norm_loss": z_norm_loss.detach(),
        }
        metrics.update(self._z_diagnostics_tensors(z.detach(), prefix="hl_skill"))
        return total_loss, metrics

    def checkpoint_state_dict(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "skill_encoder_state_dict": self.skill_encoder.state_dict(),
            "finetune_updates": int(self.finetune_updates),
        }
        if self.train_diffsr:
            state["diffsr_state_dict"] = self.diffsr.state_dict()
        if self.optimizer is not None:
            state["optimizer_state_dict"] = self.optimizer.state_dict()
        return state

    def load_checkpoint_state_dict(self, state: Mapping[str, Any]) -> None:
        if "skill_encoder_state_dict" in state:
            self.skill_encoder.load_state_dict(state["skill_encoder_state_dict"])
        if self.train_diffsr and "diffsr_state_dict" in state:
            self.diffsr.load_state_dict(state["diffsr_state_dict"])
        if self.optimizer is not None and "optimizer_state_dict" in state:
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.finetune_updates = int(state.get("finetune_updates", 0))

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
            or self._codes.shape[1] != self.command_code_dim
            or self._codes.device != device
            or self._codes.dtype != dtype
            or self._active_macro_ids is None
            or self._active_macro_ids.shape[0] != batch_size
            or self._active_macro_ids.device != device
        ):
            self._codes = torch.zeros(
                batch_size,
                self.command_code_dim,
                device=device,
                dtype=dtype,
            )
            self._latent_steps = torch.zeros(
                batch_size, device=device, dtype=torch.long
            )
            self._active_macro_ids = torch.full(
                (batch_size,),
                -1,
                device=device,
                dtype=torch.long,
            )

        assert self._codes is not None
        assert self._latent_steps is not None
        assert self._active_macro_ids is not None
        renew_mask = self._done_mask(
            td,
            batch_size=batch_size,
            device=device,
        ) | (self._latent_steps <= 0)
        if bool(renew_mask.any()):
            env_ids = torch.nonzero(renew_mask, as_tuple=False).reshape(-1)
            z, state, future_window, target, initial_z = (
                self._encode_current_macro_batch(env_ids.to(self.device))
            )
            command_codes = self._command_code_from_state_z(state, z)
            command_codes = command_codes.to(
                device=device,
                dtype=dtype,
            )
            self._codes.index_copy_(0, env_ids.to(device=device), command_codes)
            self._latent_steps.index_copy_(
                0,
                env_ids.to(device=device),
                self._sample_steps(int(env_ids.numel()), device=device),
            )
            if self.finetune_enabled:
                macro_ids = self._append_rollout_cache(
                    state=state,
                    future_window=future_window,
                    target=target,
                    initial_z=initial_z,
                ).to(device=device)
                self._active_macro_ids.index_copy_(
                    0,
                    env_ids.to(device=device),
                    macro_ids,
                )

        phase = (self.phase_period - self._latent_steps).clamp(min=0).to(torch.float32)
        phase = phase / float(self.phase_period)
        latents = self._append_command_phase(self._codes, phase)
        if self.finetune_enabled:
            td.set(
                ("hl_skill", "macro_id"),
                self._active_macro_ids.reshape(*td.batch_size),
            )
            td.set(
                ("hl_skill", "phase"),
                phase.to(device=device, dtype=torch.float32).reshape(*td.batch_size),
            )
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
        self.device = _resolve_device(self.config.device, self.env)
        preflight_size = min(self.config.batch_size, self.config.preflight_batch_size)
        state, future_window, target = self._sample_and_validate_macro_batch(
            preflight_size,
            split=self.config.train_split,
        )
        del future_window, target
        self.state_dim = int(state.shape[-1])
        self.encoder_window_steps = _encoder_window_steps(self.config)
        self.feature_slices = self._resolve_feature_slices()

        self.skill_encoder = build_skill_encoder(
            state_dim=self.state_dim,
            window_steps=self.encoder_window_steps,
            z_dim=self.config.z_dim,
            hidden_dims=self.config.encoder_hidden_dims,
            spec=self.config.latent_spec(),
        ).to(self.device)
        self.diffsr = _build_diffsr(self.config, self.state_dim, self.device).to(
            self.device
        )
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
        
        # Optional co-trained skill commander (System-1 planner). BC'd to the
        # encoder's z (detached) from current state + language goal.
        self.commander: nn.Module | None = None
        self.commander_optimizer: torch.optim.Optimizer | None = None
        self.commander_rank_embeddings: Tensor | None = None
        self.commander_lang_embed_dim = 0
        if self.config.cotrain_commander:
            self._init_commander()

    def _encode_skill(
        self,
        state: Tensor,
        future_window: Tensor,
        *,
        deterministic: bool = False,
        step: int | None = None,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        return self.skill_encoder.encode(
            state,
            _encoder_input_window(self.config, future_window),
            deterministic=deterministic,
            step=step,
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
        return _validate_macro_batch(
            batch,
            batch_size=batch_size,
            horizon_steps=int(self.config.horizon_steps),
            device=self.device,
            source="Expert",
        )

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

    @staticmethod
    def _z_diagnostics(z: Tensor, *, prefix: str) -> dict[str, float]:
        z_std = z.std(dim=0, unbiased=False)
        return {
            f"{prefix}/z_abs_mean": float(z.abs().mean().item()),
            f"{prefix}/z_rms": float(z.pow(2).mean().sqrt().item()),
            f"{prefix}/z_dim_std_mean": float(z_std.mean().item()),
            f"{prefix}/z_dim_std_min": float(z_std.min().item()),
            f"{prefix}/z_effective_rank": float(_effective_rank(z).item()),
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
            z, *_ = self._encode_skill(state, future_window, deterministic=True)
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
            z, *_ = self._encode_skill(state, future_window, deterministic=True)
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

    def _init_commander(self) -> None:
        from rlopt.agent.skill_commander import (  # noqa: PLC0415
            SkillCommander,
            build_rank_embedding_lookup,
            load_language_embedding_table,
        )

        names_provider = getattr(self.env, "expert_trajectory_motion_names", None)
        if not callable(names_provider):
            msg = "cotrain_commander requires env.expert_trajectory_motion_names()."
            raise ValueError(msg)
        table = load_language_embedding_table(
            self.config.commander_language_embeddings_path
        )
        self.commander_lang_embed_dim = int(table["embed_dim"])
        self.commander_rank_embeddings = build_rank_embedding_lookup(
            table, [str(name) for name in names_provider()], self.device
        )
        self.commander = SkillCommander(
            state_dim=self.state_dim,
            lang_embed_dim=self.commander_lang_embed_dim,
            z_dim=self.config.z_dim,
            hidden_dims=self.config.commander_hidden_dims,
        ).to(self.device)
        self.commander_optimizer = torch.optim.AdamW(
            self.commander.parameters(), lr=self.config.commander_lr
        )

    def _commander_lang_for_batch(self, batch: TensorDictBase) -> Tensor:
        traj_rank = batch.get(("hl", "traj_rank"))
        if traj_rank is None:
            msg = "cotrain_commander requires 'traj_rank' in the macro batch."
            raise ValueError(msg)
        traj_rank = traj_rank.reshape(-1).to(device=self.device, dtype=torch.long)
        assert self.commander_rank_embeddings is not None
        return self.commander_rank_embeddings.index_select(0, traj_rank)

    def _commander_train_step(
        self, batch: TensorDictBase, state: Tensor, z: Tensor
    ) -> dict[str, float]:
        assert self.commander is not None
        assert self.commander_optimizer is not None
        self.commander.train()
        lang = self._commander_lang_for_batch(batch)
        z_target = z.detach()
        cmd_state = state
        std = float(self.config.commander_state_noise_std)
        if std > 0.0 and int(state.shape[0]) > 1:
            per_dim = state.std(dim=0, keepdim=True)
            cmd_state = state + std * per_dim * torch.randn_like(state)
        z_hat = self.commander(cmd_state, lang)
        mse = F.mse_loss(z_hat, z_target)
        cosine = F.cosine_similarity(z_hat, z_target, dim=-1).mean()
        loss = (
            mse
            + self.config.commander_cosine_loss_coeff * (1.0 - cosine)
            + self.config.commander_z_norm_coeff * z_hat.pow(2).mean()
        )
        self.commander_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.config.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.commander.parameters(),
                max_norm=float(self.config.grad_clip_norm),
            )
        self.commander_optimizer.step()
        return {
            "train/commander_mse": float(mse.detach().item()),
            "train/commander_cosine": float(cosine.detach().item()),
            "train/commander_loss": float(loss.detach().item()),
        }

    @torch.no_grad()
    def _commander_eval_metrics(
        self, batch: TensorDictBase, state: Tensor, z: Tensor, *, prefix: str
    ) -> dict[str, float]:
        assert self.commander is not None
        was_training = self.commander.training
        self.commander.eval()
        lang = self._commander_lang_for_batch(batch)
        z_hat = self.commander(state, lang)
        metrics = {
            f"{prefix}/commander_cosine": float(
                F.cosine_similarity(z_hat, z, dim=-1).mean().item()
            ),
            f"{prefix}/commander_mse": float(F.mse_loss(z_hat, z).item()),
        }
        if int(lang.shape[0]) > 1:
            shuffled = lang[torch.randperm(lang.shape[0], device=lang.device)]
            z_hat_shuffled = self.commander(state, shuffled)
            metrics[f"{prefix}/commander_cosine_shuffled_lang"] = float(
                F.cosine_similarity(z_hat_shuffled, z, dim=-1).mean().item()
            )
        if was_training:
            self.commander.train()
        return metrics

    def train_step(self) -> dict[str, float]:
        self.skill_encoder.train()
        self.diffsr.train()
        batch = self._sample_macro_batch(
            self.config.batch_size, split=self.config.train_split
        )
        state, future_window, target = _validate_macro_batch(
            batch,
            batch_size=self.config.batch_size,
            horizon_steps=int(self.config.horizon_steps),
            device=self.device,
        )
        self.diffsr.update_obs_norm(target.detach())
        # reg_loss is the per-method latent regularizer (L2 / KL / commitment / 0),
        # weighted uniformly by reg_coeff; info carries method-specific diagnostics.
        z, reg_loss, info = self._encode_skill(state, future_window, step=self.update)
        diffsr_loss = self._diffsr_loss_for_z(state, z, target)
        z_norm_loss = z.pow(2).mean()
        loss = diffsr_loss + self.config.reg_coeff * reg_loss

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
        self.skill_encoder.on_after_train_step(self.update)
        self.update += 1
        metrics.update(
            {
                "train/loss": float(loss.detach().item()),
                "train/diffsr_loss": float(diffsr_loss.detach().item()),
                "train/z_norm_loss": float(z_norm_loss.detach().item()),
                "train/reg_loss": float(reg_loss.detach().item()),
                **{f"train/{k}": float(v.item()) for k, v in info.items()},
            }
        )
        if self.commander is not None:
            metrics.update(self._commander_train_step(batch, state, z))
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
            batch = self._sample_macro_batch(batch_size, split=split)
            state, future_window, target = _validate_macro_batch(
                batch,
                batch_size=batch_size,
                horizon_steps=int(self.config.horizon_steps),
                device=self.device,
            )
            z, *_ = self._encode_skill(state, future_window, deterministic=True)
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
            # Per-method diversity / collapse diagnostics.
            diversity = self.skill_encoder.diversity_metrics(
                state, _encoder_input_window(self.config, future_window)
            )
            batch_metrics.update(
                {f"{prefix}/diversity/{key}": float(value.item())
                 for key, value in diversity.items()}
            )
            if self.commander is not None:
                batch_metrics.update(
                    self._commander_eval_metrics(batch, state, z, prefix=prefix)
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
        reconstruction_eval: bool = False,
    ) -> HighLevelSkillDiffSRTrainState:
        start_time = time.perf_counter()
        state = HighLevelSkillDiffSRTrainState()
        best_eval = float("inf")
        best_path = (
            Path(checkpoint_path).with_name("best.pt")
            if checkpoint_path is not None
            else None
        )
        for _ in range(self.config.num_updates):
            metrics = self.train_step()
            should_log = (
                self.update in (1, self.config.num_updates)
                or self.update % self.config.log_interval == 0
            )
            if should_log:
                metrics.update(
                    self.evaluate(
                        prefix="train",
                        include_reconstruction=reconstruction_eval,
                    )
                )
                eval_loss = metrics.get("train/loss_real_z_eval")
                if best_path is not None and eval_loss is not None and eval_loss < best_eval:
                    best_eval = float(eval_loss)
                    self.save_checkpoint(best_path)
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
        Path(f"{target}.json").write_text(
            json.dumps({"update": int(self.update)}), encoding="utf-8"
        )

    def commander_checkpoint_state_dict(
        self, *, skill_checkpoint_path: str = ""
    ) -> dict[str, Any]:
        """SkillCommander-format checkpoint for the co-trained commander."""
        if self.commander is None:
            msg = "No co-trained commander to checkpoint."
            raise RuntimeError(msg)
        return {
            "generator_state_dict": self.commander.state_dict(),
            "config": {
                "generator_hidden_dims": list(self.config.commander_hidden_dims)
            },
            "skill_config": self.config.to_dict(),
            "skill_checkpoint_path": str(skill_checkpoint_path),
            "state_dim": int(self.state_dim),
            "lang_embed_dim": int(self.commander_lang_embed_dim),
            "z_dim": int(self.config.z_dim),
            "update": int(self.update),
        }

    def save_commander_checkpoint(
        self, path: str | Path, *, skill_checkpoint_path: str = ""
    ) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            self.commander_checkpoint_state_dict(
                skill_checkpoint_path=skill_checkpoint_path
            ),
            target,
        )

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
        self.encoder_window_steps = _encoder_window_steps(self.config)
        self.skill_encoder.load_state_dict(checkpoint["skill_encoder_state_dict"])
        self.diffsr.load_state_dict(checkpoint["diffsr_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        feature_norm_state = checkpoint.get("feature_normalization_state_dict")
        obs_norm = getattr(self.diffsr, "obs_norm", None)
        if isinstance(obs_norm, nn.Module) and feature_norm_state:
            obs_norm.load_state_dict(feature_norm_state)
        self.update = int(checkpoint.get("update", 0))
        return cast(dict[str, Any], checkpoint)
