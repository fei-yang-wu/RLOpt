"""Language-conditioned skill generator (System 1).

This module distills a frozen high-level skill encoder (see
``hl_skill_diffsr.py``) into a generator that maps ``(current_state,
language_goal) -> z``. Instead of peeking at the expert future window to encode a
skill, the generator produces the same skill latent ``z`` from the current state
and a language-goal embedding (e.g. a LAFAN1 trajectory name embedded offline by
``scripts/rlopt/build_language_goal_embeddings.py``).

Contents:

* ``SkillCommanderConfig`` - training/config dataclass.
* ``SkillCommander`` - the direct ``(state, lang) -> z`` network.
* ``FlowMatchingSkillCommander`` - conditional flow-matching latent planner.
* ``DiffusionSkillCommander`` - conditional DDPM latent planner.
* ``SkillCommanderTrainer`` - offline supervised distillation against a
  frozen ``HighLevelSkillEncoder`` (target ``z``), with held-out evaluation over
  trajectory names.
* ``FrozenSkillCommanderSampler`` - rollout-time latent-command source that
  drives the (unchanged) low-level policy from a language goal, reusing the
  command/phase/scheduling machinery of ``FrozenHighLevelSkillCommandSampler``.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler
from tensordict import TensorDictBase
from torch import Tensor, nn

from rlopt.agent.hl_skill_diffsr import (
    FrozenHighLevelSkillCommandSampler,
    HighLevelSkillDiffSRConfig,
    _build_diffsr,
    _jsonable,
    _normalize_command_mode,
    _normalize_split_value,
    _require_fraction,
    _require_non_negative_float,
    _require_positive_float,
    _require_positive_int,
    _resolve_device,
    _validate_macro_batch as _validate_hl_macro_batch,
)
from rlopt.agent.hl_skill_encoder import build_skill_encoder


def _require_probability(name: str, value: float) -> float:
    normalized = _require_non_negative_float(name, value)
    if normalized > 1.0:
        msg = f"{name} must be in [0, 1], got {value!r}."
        raise ValueError(msg)
    return normalized


def _require_non_negative_int(name: str, value: int) -> int:
    normalized = int(value)
    if normalized < 0:
        msg = f"{name} must be >= 0, got {value!r}."
        raise ValueError(msg)
    return normalized


def _coerce_bool(name: str, value: bool | str | int) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    msg = f"{name} must be a boolean value, got {value!r}."
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# Language embedding table helpers (produced by build_language_goal_embeddings)
# ---------------------------------------------------------------------------
def load_language_embedding_table(path: str | Path) -> dict[str, Any]:
    """Load and validate a ``{motion_name: embedding}`` table from disk."""
    table_path = Path(path).expanduser()
    if not table_path.is_file():
        msg = f"Language embedding table not found: {table_path}."
        raise FileNotFoundError(msg)
    table = torch.load(table_path, map_location="cpu", weights_only=False)
    required = ("names", "name_to_index", "embeddings", "embed_dim")
    missing = [key for key in required if key not in table]
    if missing:
        msg = f"Language embedding table {table_path} is missing keys: {missing}."
        raise ValueError(msg)
    embeddings = table["embeddings"]
    if not isinstance(embeddings, Tensor) or embeddings.ndim != 2:
        msg = "Language embedding table 'embeddings' must be a [N, D] tensor."
        raise ValueError(msg)
    return table


def build_rank_embedding_lookup(
    table: Mapping[str, Any],
    motion_names: list[str],
    device: torch.device | str,
) -> Tensor:
    """Build a rank-indexed ``[num_ranks, embed_dim]`` lookup from motion names."""
    name_to_index = table["name_to_index"]
    embeddings = cast(Tensor, table["embeddings"]).to(
        device=device, dtype=torch.float32
    )
    rows: list[Tensor] = []
    missing: list[str] = []
    for name in motion_names:
        index = name_to_index.get(str(name))
        if index is None:
            missing.append(str(name))
            continue
        rows.append(embeddings[int(index)])
    if missing:
        msg = (
            "Language embedding table has no entry for trajectory motion names: "
            f"{sorted(set(missing))}. Rebuild the table from the active manifest."
        )
        raise ValueError(msg)
    if not rows:
        msg = "No trajectory motion names were provided for lookup."
        raise ValueError(msg)
    return torch.stack(rows, dim=0).contiguous()


# ---------------------------------------------------------------------------
# Config + network
# ---------------------------------------------------------------------------
@dataclass
class SkillCommanderConfig:
    """Configuration for offline language-conditioned skill generator training."""

    skill_checkpoint_path: str = ""
    language_embeddings_path: str = ""
    condition_on_language: bool = True
    state_history_steps: int = 0
    planner_type: str = "mlp"
    generator_hidden_dims: tuple[int, ...] = (1024, 512, 512)
    flow_num_inference_steps: int = 16
    flow_time_embed_dim: int = 64
    flow_train_noise_std: float = 1.0
    flow_inference_noise_std: float = 1.0
    diffusion_num_train_timesteps: int = 100
    diffusion_num_inference_steps: int = 16
    diffusion_time_embed_dim: int = 64
    diffusion_beta_schedule: str = "squaredcos_cap_v2"
    diffusion_prediction_type: str = "epsilon"
    diffusion_inference_scheduler: str = "ddpm"
    diffusion_ddim_eta: float = 0.0
    diffusion_inference_noise_std: float = 1.0
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
    lr: float = 3.0e-4
    weight_decay: float = 0.0
    grad_clip_norm: float | None = 1.0
    cosine_loss_coeff: float = 1.0
    z_norm_coeff: float = 1.0e-4
    state_noise_std: float = 0.0
    state_feature_dropout_prob: float = 0.0
    state_feature_dropout_terms: tuple[str, ...] = ("expert_motion",)
    state_feature_dropout_mode: str = "shuffle"
    state_feature_dropout_warmup_updates: int = 0
    language_contrastive_coeff: float = 0.0
    language_contrastive_margin: float = 0.05
    language_contrastive_warmup_updates: int = 0
    device: str = "auto"

    def validate(self) -> None:
        self.skill_checkpoint_path = str(self.skill_checkpoint_path).strip()
        if not self.skill_checkpoint_path:
            msg = "skill_checkpoint_path is required."
            raise ValueError(msg)
        self.language_embeddings_path = str(self.language_embeddings_path).strip()
        self.condition_on_language = _coerce_bool(
            "condition_on_language", self.condition_on_language
        )
        if self.condition_on_language and not self.language_embeddings_path:
            msg = "language_embeddings_path is required."
            raise ValueError(msg)
        self.state_history_steps = _require_non_negative_int(
            "state_history_steps", self.state_history_steps
        )
        self.planner_type = str(self.planner_type).strip().lower()
        if self.planner_type not in {"mlp", "flow_matching", "diffusion_policy"}:
            msg = (
                "planner_type must be one of {'mlp', 'flow_matching', "
                "'diffusion_policy'}, got "
                f"{self.planner_type!r}."
            )
            raise ValueError(msg)
        self.generator_hidden_dims = tuple(
            _require_positive_int("generator_hidden_dims", dim)
            for dim in self.generator_hidden_dims
        )
        self.flow_num_inference_steps = _require_positive_int(
            "flow_num_inference_steps", self.flow_num_inference_steps
        )
        self.flow_time_embed_dim = _require_positive_int(
            "flow_time_embed_dim", self.flow_time_embed_dim
        )
        self.flow_train_noise_std = _require_non_negative_float(
            "flow_train_noise_std", self.flow_train_noise_std
        )
        self.flow_inference_noise_std = _require_non_negative_float(
            "flow_inference_noise_std", self.flow_inference_noise_std
        )
        self.diffusion_num_train_timesteps = _require_positive_int(
            "diffusion_num_train_timesteps", self.diffusion_num_train_timesteps
        )
        self.diffusion_num_inference_steps = _require_positive_int(
            "diffusion_num_inference_steps", self.diffusion_num_inference_steps
        )
        self.diffusion_time_embed_dim = _require_positive_int(
            "diffusion_time_embed_dim", self.diffusion_time_embed_dim
        )
        self.diffusion_beta_schedule = str(self.diffusion_beta_schedule).strip()
        if not self.diffusion_beta_schedule:
            msg = "diffusion_beta_schedule must be non-empty."
            raise ValueError(msg)
        self.diffusion_prediction_type = str(
            self.diffusion_prediction_type
        ).strip()
        if self.diffusion_prediction_type != "epsilon":
            msg = (
                "diffusion_prediction_type currently supports only 'epsilon', "
                f"got {self.diffusion_prediction_type!r}."
            )
            raise ValueError(msg)
        self.diffusion_inference_scheduler = str(
            self.diffusion_inference_scheduler
        ).strip().lower()
        if self.diffusion_inference_scheduler not in {"ddpm", "ddim"}:
            msg = (
                "diffusion_inference_scheduler must be 'ddpm' or 'ddim', "
                f"got {self.diffusion_inference_scheduler!r}."
            )
            raise ValueError(msg)
        self.diffusion_ddim_eta = _require_non_negative_float(
            "diffusion_ddim_eta", self.diffusion_ddim_eta
        )
        self.diffusion_inference_noise_std = _require_non_negative_float(
            "diffusion_inference_noise_std", self.diffusion_inference_noise_std
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
        self.lr = _require_positive_float("lr", self.lr)
        self.weight_decay = _require_non_negative_float(
            "weight_decay", self.weight_decay
        )
        if self.grad_clip_norm is not None:
            self.grad_clip_norm = _require_positive_float(
                "grad_clip_norm", self.grad_clip_norm
            )
        self.cosine_loss_coeff = _require_non_negative_float(
            "cosine_loss_coeff", self.cosine_loss_coeff
        )
        self.z_norm_coeff = _require_non_negative_float(
            "z_norm_coeff", self.z_norm_coeff
        )
        self.state_noise_std = _require_non_negative_float(
            "state_noise_std", self.state_noise_std
        )
        self.state_feature_dropout_prob = _require_probability(
            "state_feature_dropout_prob", self.state_feature_dropout_prob
        )
        self.state_feature_dropout_terms = tuple(
            str(term).strip()
            for term in self.state_feature_dropout_terms
            if str(term).strip()
        )
        self.state_feature_dropout_mode = str(
            self.state_feature_dropout_mode
        ).strip().lower()
        if self.state_feature_dropout_mode not in {"shuffle", "zero", "batch_mean"}:
            msg = (
                "state_feature_dropout_mode must be one of 'shuffle', 'zero', "
                f"or 'batch_mean', got {self.state_feature_dropout_mode!r}."
            )
            raise ValueError(msg)
        self.state_feature_dropout_warmup_updates = max(
            0, int(self.state_feature_dropout_warmup_updates)
        )
        self.language_contrastive_coeff = _require_non_negative_float(
            "language_contrastive_coeff", self.language_contrastive_coeff
        )
        if not self.condition_on_language and self.language_contrastive_coeff > 0.0:
            msg = "language_contrastive_coeff requires condition_on_language=True."
            raise ValueError(msg)
        self.language_contrastive_margin = _require_non_negative_float(
            "language_contrastive_margin", self.language_contrastive_margin
        )
        self.language_contrastive_warmup_updates = max(
            0, int(self.language_contrastive_warmup_updates)
        )
        self.device = str(self.device)

    def to_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], _jsonable(asdict(self)))

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> SkillCommanderConfig:
        known_fields = {item.name for item in fields(cls)}
        kwargs = {key: values[key] for key in known_fields if key in values}
        if "generator_hidden_dims" in kwargs:
            kwargs["generator_hidden_dims"] = tuple(
                int(item) for item in kwargs["generator_hidden_dims"]
            )
        if "state_feature_dropout_terms" in kwargs:
            kwargs["state_feature_dropout_terms"] = tuple(
                str(item) for item in kwargs["state_feature_dropout_terms"]
            )
        config = cls(**kwargs)
        config.validate()
        return config


class SkillCommander(nn.Module):
    """Map a current state and language-goal embedding to a skill latent ``z``."""

    def __init__(
        self,
        state_dim: int,
        lang_embed_dim: int,
        z_dim: int,
        hidden_dims: tuple[int, ...] = (1024, 512, 512),
    ) -> None:
        super().__init__()
        self.state_dim = _require_positive_int("state_dim", state_dim)
        self.lang_embed_dim = _require_non_negative_int("lang_embed_dim", lang_embed_dim)
        self.z_dim = _require_positive_int("z_dim", z_dim)
        hidden_dims = tuple(
            _require_positive_int("hidden_dims", dim) for dim in hidden_dims
        )

        input_dim = self.state_dim + self.lang_embed_dim
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

    def forward(self, state: Tensor, lang_emb: Tensor) -> Tensor:
        if state.ndim != 2:
            msg = f"state must have shape [B, D], got {tuple(state.shape)}."
            raise ValueError(msg)
        if lang_emb.ndim != 2:
            msg = f"lang_emb must have shape [B, L], got {tuple(lang_emb.shape)}."
            raise ValueError(msg)
        batch_size, state_dim = state.shape
        if int(state_dim) != self.state_dim:
            msg = (
                f"state width mismatch: expected [B, {self.state_dim}], "
                f"got {tuple(state.shape)}."
            )
            raise ValueError(msg)
        expected_lang = (batch_size, self.lang_embed_dim)
        if tuple(lang_emb.shape) != expected_lang:
            msg = (
                f"lang_emb shape mismatch: expected {expected_lang}, "
                f"got {tuple(lang_emb.shape)}."
            )
            raise ValueError(msg)
        return self.net(torch.cat([state, lang_emb], dim=-1))


class FlowMatchingSkillCommander(nn.Module):
    """Conditional flow-matching planner for skill latents.

    The model learns a vector field that transports Gaussian skill noise to the
    frozen encoder's target latent ``z`` conditioned on ``(state, language)``.
    At rollout time it integrates that vector field and exposes the same
    ``forward(state, lang) -> z`` API as ``SkillCommander``.
    """

    def __init__(
        self,
        state_dim: int,
        lang_embed_dim: int,
        z_dim: int,
        hidden_dims: tuple[int, ...] = (1024, 512, 512),
        *,
        time_embed_dim: int = 64,
        num_inference_steps: int = 16,
        train_noise_std: float = 1.0,
        inference_noise_std: float = 1.0,
    ) -> None:
        super().__init__()
        self.state_dim = _require_positive_int("state_dim", state_dim)
        self.lang_embed_dim = _require_non_negative_int("lang_embed_dim", lang_embed_dim)
        self.z_dim = _require_positive_int("z_dim", z_dim)
        self.time_embed_dim = _require_positive_int("time_embed_dim", time_embed_dim)
        self.num_inference_steps = _require_positive_int(
            "num_inference_steps", num_inference_steps
        )
        self.train_noise_std = _require_non_negative_float(
            "train_noise_std", train_noise_std
        )
        self.inference_noise_std = _require_non_negative_float(
            "inference_noise_std", inference_noise_std
        )
        hidden_dims = tuple(
            _require_positive_int("hidden_dims", dim) for dim in hidden_dims
        )

        input_dim = self.z_dim + self.time_embed_dim + self.state_dim + self.lang_embed_dim
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
        self.vector_field = nn.Sequential(*layers)

    def _validate_condition(self, state: Tensor, lang_emb: Tensor) -> None:
        if state.ndim != 2:
            msg = f"state must have shape [B, D], got {tuple(state.shape)}."
            raise ValueError(msg)
        if lang_emb.ndim != 2:
            msg = f"lang_emb must have shape [B, L], got {tuple(lang_emb.shape)}."
            raise ValueError(msg)
        batch_size, state_dim = state.shape
        if int(state_dim) != self.state_dim:
            msg = (
                f"state width mismatch: expected [B, {self.state_dim}], "
                f"got {tuple(state.shape)}."
            )
            raise ValueError(msg)
        expected_lang = (batch_size, self.lang_embed_dim)
        if tuple(lang_emb.shape) != expected_lang:
            msg = (
                f"lang_emb shape mismatch: expected {expected_lang}, "
                f"got {tuple(lang_emb.shape)}."
            )
            raise ValueError(msg)

    def _time_embedding(self, t: Tensor) -> Tensor:
        t = t.reshape(-1).to(dtype=torch.float32)
        half_dim = self.time_embed_dim // 2
        if half_dim == 0:
            return t.new_zeros((int(t.numel()), self.time_embed_dim))
        scale = torch.arange(half_dim, device=t.device, dtype=torch.float32)
        scale = torch.exp(
            -torch.log(torch.tensor(10000.0, device=t.device)) * scale / max(half_dim - 1, 1)
        )
        args = t[:, None] * scale[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if int(emb.shape[-1]) < self.time_embed_dim:
            emb = F.pad(emb, (0, self.time_embed_dim - int(emb.shape[-1])))
        return emb

    def _vector_field(self, z_t: Tensor, t: Tensor, state: Tensor, lang_emb: Tensor) -> Tensor:
        self._validate_condition(state, lang_emb)
        expected_z = (int(state.shape[0]), self.z_dim)
        if tuple(z_t.shape) != expected_z:
            msg = f"z_t must have shape {expected_z}, got {tuple(z_t.shape)}."
            raise ValueError(msg)
        time_emb = self._time_embedding(t).to(device=z_t.device, dtype=z_t.dtype)
        inputs = torch.cat([z_t, time_emb, state, lang_emb], dim=-1)
        return self.vector_field(inputs)

    def flow_matching_loss(
        self,
        state: Tensor,
        lang_emb: Tensor,
        z_target: Tensor,
    ) -> tuple[Tensor, dict[str, float]]:
        self._validate_condition(state, lang_emb)
        expected_z = (int(state.shape[0]), self.z_dim)
        if tuple(z_target.shape) != expected_z:
            msg = f"z_target must have shape {expected_z}, got {tuple(z_target.shape)}."
            raise ValueError(msg)
        z0 = self.train_noise_std * torch.randn_like(z_target)
        t = torch.rand(int(state.shape[0]), device=z_target.device, dtype=z_target.dtype)
        t_view = t[:, None]
        z_t = (1.0 - t_view) * z0 + t_view * z_target
        target_velocity = z_target - z0
        pred_velocity = self._vector_field(z_t, t, state, lang_emb)
        loss = F.mse_loss(pred_velocity, target_velocity)
        metrics = {
            "flow/velocity_mse": float(loss.detach().item()),
            "flow/target_velocity_rms": float(
                target_velocity.detach().pow(2).mean().sqrt().item()
            ),
            "flow/pred_velocity_rms": float(
                pred_velocity.detach().pow(2).mean().sqrt().item()
            ),
        }
        return loss, metrics

    def forward(
        self,
        state: Tensor,
        lang_emb: Tensor,
        *,
        noise: Tensor | None = None,
        num_steps: int | None = None,
    ) -> Tensor:
        self._validate_condition(state, lang_emb)
        batch_size = int(state.shape[0])
        if noise is None:
            z = self.inference_noise_std * torch.randn(
                batch_size,
                self.z_dim,
                device=state.device,
                dtype=state.dtype,
            )
        else:
            expected_noise = (batch_size, self.z_dim)
            if tuple(noise.shape) != expected_noise:
                msg = (
                    f"noise must have shape {expected_noise}, "
                    f"got {tuple(noise.shape)}."
                )
                raise ValueError(msg)
            z = noise.to(device=state.device, dtype=state.dtype)
        steps = self.num_inference_steps if num_steps is None else int(num_steps)
        steps = _require_positive_int("num_steps", steps)
        dt = 1.0 / float(steps)
        for step in range(steps):
            t = torch.full(
                (batch_size,),
                fill_value=float(step) * dt,
                device=state.device,
                dtype=state.dtype,
            )
            z = z + dt * self._vector_field(z, t, state, lang_emb)
        return z


class DiffusionSkillCommander(nn.Module):
    """Conditional DDPM planner for skill latents.

    This is the diffusion-policy variant of the language commander: it learns a
    denoising network over the skill-command action ``z`` conditioned on the
    current macro state and language embedding. The scheduler is provided by
    diffusers, while the public rollout API remains ``forward(state, lang) -> z``.
    """

    def __init__(
        self,
        state_dim: int,
        lang_embed_dim: int,
        z_dim: int,
        hidden_dims: tuple[int, ...] = (1024, 512, 512),
        *,
        time_embed_dim: int = 64,
        num_train_timesteps: int = 100,
        num_inference_steps: int = 16,
        beta_schedule: str = "squaredcos_cap_v2",
        prediction_type: str = "epsilon",
        inference_scheduler: str = "ddpm",
        ddim_eta: float = 0.0,
        inference_noise_std: float = 1.0,
    ) -> None:
        super().__init__()
        self.state_dim = _require_positive_int("state_dim", state_dim)
        self.lang_embed_dim = _require_non_negative_int("lang_embed_dim", lang_embed_dim)
        self.z_dim = _require_positive_int("z_dim", z_dim)
        self.time_embed_dim = _require_positive_int("time_embed_dim", time_embed_dim)
        self.num_train_timesteps = _require_positive_int(
            "num_train_timesteps", num_train_timesteps
        )
        self.num_inference_steps = _require_positive_int(
            "num_inference_steps", num_inference_steps
        )
        self.inference_noise_std = _require_non_negative_float(
            "inference_noise_std", inference_noise_std
        )
        self.beta_schedule = str(beta_schedule).strip()
        self.prediction_type = str(prediction_type).strip()
        if self.prediction_type != "epsilon":
            msg = (
                "DiffusionSkillCommander currently supports only epsilon "
                f"prediction, got {self.prediction_type!r}."
            )
            raise ValueError(msg)
        self.inference_scheduler_name = str(inference_scheduler).strip().lower()
        if self.inference_scheduler_name not in {"ddpm", "ddim"}:
            msg = (
                "inference_scheduler must be 'ddpm' or 'ddim', got "
                f"{self.inference_scheduler_name!r}."
            )
            raise ValueError(msg)
        self.ddim_eta = _require_non_negative_float("ddim_eta", ddim_eta)
        hidden_dims = tuple(
            _require_positive_int("hidden_dims", dim) for dim in hidden_dims
        )

        self.train_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_train_timesteps,
            beta_schedule=self.beta_schedule,
            prediction_type=self.prediction_type,
            clip_sample=False,
        )
        scheduler_cls = (
            DDIMScheduler
            if self.inference_scheduler_name == "ddim"
            else DDPMScheduler
        )
        self.inference_scheduler = scheduler_cls(
            num_train_timesteps=self.num_train_timesteps,
            beta_schedule=self.beta_schedule,
            prediction_type=self.prediction_type,
            clip_sample=False,
        )

        input_dim = self.z_dim + self.time_embed_dim + self.state_dim + self.lang_embed_dim
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
        self.noise_predictor = nn.Sequential(*layers)

    def _validate_condition(self, state: Tensor, lang_emb: Tensor) -> None:
        if state.ndim != 2:
            msg = f"state must have shape [B, D], got {tuple(state.shape)}."
            raise ValueError(msg)
        if lang_emb.ndim != 2:
            msg = f"lang_emb must have shape [B, L], got {tuple(lang_emb.shape)}."
            raise ValueError(msg)
        batch_size, state_dim = state.shape
        if int(state_dim) != self.state_dim:
            msg = (
                f"state width mismatch: expected [B, {self.state_dim}], "
                f"got {tuple(state.shape)}."
            )
            raise ValueError(msg)
        expected_lang = (batch_size, self.lang_embed_dim)
        if tuple(lang_emb.shape) != expected_lang:
            msg = (
                f"lang_emb shape mismatch: expected {expected_lang}, "
                f"got {tuple(lang_emb.shape)}."
            )
            raise ValueError(msg)

    def _time_embedding(self, timesteps: Tensor) -> Tensor:
        t = timesteps.reshape(-1).to(dtype=torch.float32)
        half_dim = self.time_embed_dim // 2
        if half_dim == 0:
            return t.new_zeros((int(t.numel()), self.time_embed_dim))
        scale = torch.arange(half_dim, device=t.device, dtype=torch.float32)
        scale = torch.exp(
            -torch.log(torch.tensor(10000.0, device=t.device))
            * scale
            / max(half_dim - 1, 1)
        )
        args = (t[:, None] / float(self.num_train_timesteps)) * scale[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if int(emb.shape[-1]) < self.time_embed_dim:
            emb = F.pad(emb, (0, self.time_embed_dim - int(emb.shape[-1])))
        return emb

    def _predict_noise(
        self,
        z_t: Tensor,
        timesteps: Tensor,
        state: Tensor,
        lang_emb: Tensor,
    ) -> Tensor:
        self._validate_condition(state, lang_emb)
        expected_z = (int(state.shape[0]), self.z_dim)
        if tuple(z_t.shape) != expected_z:
            msg = f"z_t must have shape {expected_z}, got {tuple(z_t.shape)}."
            raise ValueError(msg)
        if tuple(timesteps.reshape(-1).shape) != (int(state.shape[0]),):
            msg = (
                "timesteps must have shape "
                f"{(int(state.shape[0]),)}, got {tuple(timesteps.shape)}."
            )
            raise ValueError(msg)
        time_emb = self._time_embedding(timesteps).to(
            device=z_t.device, dtype=z_t.dtype
        )
        inputs = torch.cat([z_t, time_emb, state, lang_emb], dim=-1)
        return self.noise_predictor(inputs)

    def diffusion_loss(
        self,
        state: Tensor,
        lang_emb: Tensor,
        z_target: Tensor,
    ) -> tuple[Tensor, dict[str, float]]:
        self._validate_condition(state, lang_emb)
        expected_z = (int(state.shape[0]), self.z_dim)
        if tuple(z_target.shape) != expected_z:
            msg = f"z_target must have shape {expected_z}, got {tuple(z_target.shape)}."
            raise ValueError(msg)
        noise = torch.randn_like(z_target)
        timesteps = torch.randint(
            low=0,
            high=self.num_train_timesteps,
            size=(int(state.shape[0]),),
            device=z_target.device,
            dtype=torch.long,
        )
        z_t = self.train_scheduler.add_noise(z_target, noise, timesteps)
        pred_noise = self._predict_noise(z_t, timesteps, state, lang_emb)
        loss = F.mse_loss(pred_noise, noise)
        metrics = {
            "diffusion/noise_mse": float(loss.detach().item()),
            "diffusion/noise_rms": float(noise.detach().pow(2).mean().sqrt().item()),
            "diffusion/pred_noise_rms": float(
                pred_noise.detach().pow(2).mean().sqrt().item()
            ),
            "diffusion/timestep_mean": float(
                timesteps.detach().to(dtype=torch.float32).mean().item()
            ),
        }
        return loss, metrics

    def forward(
        self,
        state: Tensor,
        lang_emb: Tensor,
        *,
        noise: Tensor | None = None,
        num_steps: int | None = None,
    ) -> Tensor:
        self._validate_condition(state, lang_emb)
        batch_size = int(state.shape[0])
        if noise is None:
            z = self.inference_noise_std * torch.randn(
                batch_size,
                self.z_dim,
                device=state.device,
                dtype=state.dtype,
            )
        else:
            expected_noise = (batch_size, self.z_dim)
            if tuple(noise.shape) != expected_noise:
                msg = (
                    f"noise must have shape {expected_noise}, "
                    f"got {tuple(noise.shape)}."
                )
                raise ValueError(msg)
            z = noise.to(device=state.device, dtype=state.dtype)

        steps = self.num_inference_steps if num_steps is None else int(num_steps)
        steps = _require_positive_int("num_steps", steps)
        self.inference_scheduler.set_timesteps(steps, device=state.device)
        for timestep in self.inference_scheduler.timesteps:
            t_int = int(timestep.item())
            timesteps = torch.full(
                (batch_size,),
                fill_value=t_int,
                device=state.device,
                dtype=torch.long,
            )
            pred_noise = self._predict_noise(z, timesteps, state, lang_emb)
            if isinstance(self.inference_scheduler, DDIMScheduler):
                z = self.inference_scheduler.step(
                    pred_noise,
                    t_int,
                    z,
                    eta=float(self.ddim_eta),
                ).prev_sample
            else:
                z = self.inference_scheduler.step(pred_noise, t_int, z).prev_sample
        return z


def _build_skill_commander_generator(
    *,
    planner_type: str,
    state_dim: int,
    lang_embed_dim: int,
    z_dim: int,
    hidden_dims: tuple[int, ...],
    flow_num_inference_steps: int = 16,
    flow_time_embed_dim: int = 64,
    flow_train_noise_std: float = 1.0,
    flow_inference_noise_std: float = 1.0,
    diffusion_num_train_timesteps: int = 100,
    diffusion_num_inference_steps: int = 16,
    diffusion_time_embed_dim: int = 64,
    diffusion_beta_schedule: str = "squaredcos_cap_v2",
    diffusion_prediction_type: str = "epsilon",
    diffusion_inference_scheduler: str = "ddpm",
    diffusion_ddim_eta: float = 0.0,
    diffusion_inference_noise_std: float = 1.0,
) -> nn.Module:
    planner_type = str(planner_type).strip().lower()
    if planner_type == "mlp":
        return SkillCommander(
            state_dim=state_dim,
            lang_embed_dim=lang_embed_dim,
            z_dim=z_dim,
            hidden_dims=hidden_dims,
        )
    if planner_type == "flow_matching":
        return FlowMatchingSkillCommander(
            state_dim=state_dim,
            lang_embed_dim=lang_embed_dim,
            z_dim=z_dim,
            hidden_dims=hidden_dims,
            time_embed_dim=flow_time_embed_dim,
            num_inference_steps=flow_num_inference_steps,
            train_noise_std=flow_train_noise_std,
            inference_noise_std=flow_inference_noise_std,
        )
    if planner_type == "diffusion_policy":
        return DiffusionSkillCommander(
            state_dim=state_dim,
            lang_embed_dim=lang_embed_dim,
            z_dim=z_dim,
            hidden_dims=hidden_dims,
            time_embed_dim=diffusion_time_embed_dim,
            num_train_timesteps=diffusion_num_train_timesteps,
            num_inference_steps=diffusion_num_inference_steps,
            beta_schedule=diffusion_beta_schedule,
            prediction_type=diffusion_prediction_type,
            inference_scheduler=diffusion_inference_scheduler,
            ddim_eta=diffusion_ddim_eta,
            inference_noise_std=diffusion_inference_noise_std,
        )
    msg = f"Unknown skill commander planner_type: {planner_type!r}."
    raise ValueError(msg)


def _build_skill_commander_generator_from_config(
    config: SkillCommanderConfig,
    *,
    state_dim: int,
    lang_embed_dim: int,
    z_dim: int,
) -> nn.Module:
    return _build_skill_commander_generator(
        planner_type=config.planner_type,
        state_dim=state_dim,
        lang_embed_dim=lang_embed_dim,
        z_dim=z_dim,
        hidden_dims=config.generator_hidden_dims,
        flow_num_inference_steps=config.flow_num_inference_steps,
        flow_time_embed_dim=config.flow_time_embed_dim,
        flow_train_noise_std=config.flow_train_noise_std,
        flow_inference_noise_std=config.flow_inference_noise_std,
        diffusion_num_train_timesteps=config.diffusion_num_train_timesteps,
        diffusion_num_inference_steps=config.diffusion_num_inference_steps,
        diffusion_time_embed_dim=config.diffusion_time_embed_dim,
        diffusion_beta_schedule=config.diffusion_beta_schedule,
        diffusion_prediction_type=config.diffusion_prediction_type,
        diffusion_inference_scheduler=config.diffusion_inference_scheduler,
        diffusion_ddim_eta=config.diffusion_ddim_eta,
        diffusion_inference_noise_std=config.diffusion_inference_noise_std,
    )


def _build_skill_commander_generator_from_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    state_dim: int,
    lang_embed_dim: int,
    z_dim: int,
    config_overrides: Mapping[str, Any] | None = None,
) -> nn.Module:
    config = dict(checkpoint.get("config", {}))
    if config_overrides is not None:
        config.update(
            {
                str(key): value
                for key, value in dict(config_overrides).items()
                if value is not None
            }
        )
    hidden_dims = tuple(int(dim) for dim in config.get("generator_hidden_dims", (1024, 512, 512)))
    return _build_skill_commander_generator(
        planner_type=str(config.get("planner_type", "mlp")),
        state_dim=state_dim,
        lang_embed_dim=lang_embed_dim,
        z_dim=z_dim,
        hidden_dims=hidden_dims,
        flow_num_inference_steps=int(config.get("flow_num_inference_steps", 16)),
        flow_time_embed_dim=int(config.get("flow_time_embed_dim", 64)),
        flow_train_noise_std=float(config.get("flow_train_noise_std", 1.0)),
        flow_inference_noise_std=float(config.get("flow_inference_noise_std", 1.0)),
        diffusion_num_train_timesteps=int(
            config.get("diffusion_num_train_timesteps", 100)
        ),
        diffusion_num_inference_steps=int(
            config.get("diffusion_num_inference_steps", 16)
        ),
        diffusion_time_embed_dim=int(config.get("diffusion_time_embed_dim", 64)),
        diffusion_beta_schedule=str(
            config.get("diffusion_beta_schedule", "squaredcos_cap_v2")
        ),
        diffusion_prediction_type=str(config.get("diffusion_prediction_type", "epsilon")),
        diffusion_inference_scheduler=str(
            config.get("diffusion_inference_scheduler", "ddpm")
        ),
        diffusion_ddim_eta=float(config.get("diffusion_ddim_eta", 0.0)),
        diffusion_inference_noise_std=float(
            config.get("diffusion_inference_noise_std", 1.0)
        ),
    )


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
@dataclass
class SkillCommanderTrainState:
    update: int = 0
    elapsed_seconds: float = 0.0
    last_metrics: dict[str, float] = field(default_factory=dict)


class SkillCommanderTrainer:
    """Distill a frozen ``HighLevelSkillEncoder`` into a language-conditioned generator."""

    def __init__(self, config: SkillCommanderConfig, env: object) -> None:
        self.config = config
        self.config.validate()
        self.env = env
        self.device = self._resolve_device()

        # Load the frozen skill encoder + its config (defines horizon / z / window).
        skill_checkpoint = torch.load(
            Path(self.config.skill_checkpoint_path).expanduser(),
            map_location=self.device,
            weights_only=False,
        )
        self.skill_config = HighLevelSkillDiffSRConfig.from_dict(
            skill_checkpoint["config"]
        )
        self.horizon_steps = int(self.skill_config.horizon_steps)
        self.encoder_window_steps = self._encoder_window_steps()
        self.z_dim = int(self.skill_config.z_dim)

        self.condition_on_language = bool(self.config.condition_on_language)

        # Preflight a small batch to discover macro-state and planner-input widths.
        state, planner_state, _, _, _ = self._sample_and_validate_macro_batch(
            min(self.config.batch_size, self.config.preflight_batch_size),
            split=self.config.train_split,
        )
        self.state_dim = int(state.shape[-1])
        self.planner_state_dim = int(planner_state.shape[-1])

        self.skill_encoder = build_skill_encoder(
            state_dim=self.state_dim,
            window_steps=self.encoder_window_steps,
            z_dim=self.z_dim,
            hidden_dims=self.skill_config.encoder_hidden_dims,
            spec=self.skill_config.latent_spec(),
        ).to(self.device)
        self.skill_encoder.load_state_dict(skill_checkpoint["skill_encoder_state_dict"])
        self.skill_encoder.eval()
        self.skill_encoder.requires_grad_(False)
        self.state_feature_slices = self._state_feature_slices()
        self.motion_names = self._expert_trajectory_motion_names()

        # Optional language lookup: no-language planners use a [B, 0] condition.
        if self.condition_on_language:
            self.language_table = load_language_embedding_table(
                self.config.language_embeddings_path
            )
            self.lang_embed_dim = int(self.language_table["embed_dim"])
            self.rank_embeddings = build_rank_embedding_lookup(
                self.language_table,
                self.motion_names,
                self.device,
            )
        else:
            self.lang_embed_dim = 0
            self.language_table = {
                "names": list(self.motion_names),
                "phrases": list(self.motion_names),
                "name_to_index": {
                    str(name): index for index, name in enumerate(self.motion_names)
                },
                "embeddings": torch.empty(
                    (len(self.motion_names), 0), dtype=torch.float32
                ),
                "embed_dim": 0,
                "backend": "none",
                "model": None,
                "raw_names": True,
                "manifest": "no_language",
            }
            self.rank_embeddings = torch.empty(
                (len(self.motion_names), 0),
                device=self.device,
                dtype=torch.float32,
            )

        self.generator = _build_skill_commander_generator_from_config(
            self.config,
            state_dim=self.planner_state_dim,
            lang_embed_dim=self.lang_embed_dim,
            z_dim=self.z_dim,
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.generator.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        self.update = 0

    # -- setup helpers -----------------------------------------------------
    def _resolve_device(self) -> torch.device:
        if self.config.device.strip().lower() != "auto":
            return torch.device(self.config.device)
        env_device = getattr(self.env, "device", None)
        if env_device is not None:
            return torch.device(env_device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _encoder_window_steps(self) -> int:
        if self.skill_config.encoder_window_mode == "intermediate":
            return int(self.skill_config.horizon_steps) - 1
        return int(self.skill_config.horizon_steps)

    def _encoder_input_window(self, future_window: Tensor) -> Tensor:
        if self.skill_config.encoder_window_mode == "intermediate":
            return future_window[:, :-1, :]
        return future_window

    def _expert_trajectory_motion_names(self) -> list[str]:
        provider = getattr(self.env, "expert_trajectory_motion_names", None)
        if not callable(provider):
            msg = "env must expose expert_trajectory_motion_names()."
            raise ValueError(msg)
        return [str(name) for name in provider()]

    def _state_feature_slices(self) -> dict[str, tuple[int, int]]:
        provider = getattr(self.env, "expert_macro_feature_slices", None)
        if not callable(provider):
            return {}
        raw_slices = provider(int(self.horizon_steps))
        slices: dict[str, tuple[int, int]] = {}
        for name, bounds in dict(raw_slices).items():
            start, end = int(bounds[0]), int(bounds[1])
            if start < 0 or end <= start or end > self.state_dim:
                msg = (
                    "expert_macro_feature_slices returned invalid bounds for "
                    f"{name!r}: {(start, end)} with state_dim={self.state_dim}."
                )
                raise ValueError(msg)
            slices[str(name)] = (start, end)
        return slices

    # -- sampling ----------------------------------------------------------
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
            horizon_steps=int(self.horizon_steps),
            split=split,
            eval_fraction=float(self.config.eval_trajectory_fraction),
            split_seed=int(self.config.trajectory_split_seed),
            state_history_steps=int(self.config.state_history_steps),
        )

    def _sample_and_validate_macro_batch(
        self,
        batch_size: int,
        *,
        split: str | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch = self._sample_macro_batch(batch_size, split=split)
        return self._validate_macro_batch(batch, batch_size=batch_size)

    def _validate_macro_batch(
        self,
        batch: TensorDictBase,
        *,
        batch_size: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        state = batch.get(("hl", "state"))
        future_window = batch.get(("hl", "future_window"))
        target = batch.get(("hl", "target"))
        traj_rank = batch.get(("hl", "traj_rank"))
        state_history = batch.get(("hl", "state_history"))
        missing = [
            name
            for name, value in (
                ("hl/state", state),
                ("hl/future_window", future_window),
                ("hl/target", target),
            )
            if value is None
        ]
        if self.condition_on_language and traj_rank is None:
            missing.append("hl/traj_rank")
        if int(self.config.state_history_steps) > 0 and state_history is None:
            missing.append("hl/state_history")
        if missing:
            msg = f"Expert macro batch is missing keys: {missing}."
            raise ValueError(msg)
        state = cast(Tensor, state).to(device=self.device, dtype=torch.float32)
        future_window = cast(Tensor, future_window).to(
            device=self.device, dtype=torch.float32
        )
        target = cast(Tensor, target).to(device=self.device, dtype=torch.float32)
        if traj_rank is None:
            traj_rank_t = torch.zeros(
                int(batch_size), device=self.device, dtype=torch.long
            )
        else:
            traj_rank_t = cast(Tensor, traj_rank).reshape(-1).to(
                device=self.device, dtype=torch.long
            )
        if state.ndim != 2:
            msg = f"hl/state must have shape [B, D], got {tuple(state.shape)}."
            raise ValueError(msg)
        state_dim = int(state.shape[-1])
        expected_state = (int(batch_size), state_dim)
        expected_window = (int(batch_size), self.horizon_steps, state_dim)
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
        if tuple(traj_rank_t.shape) != (int(batch_size),):
            msg = (
                "hl/traj_rank shape mismatch: expected "
                f"{(int(batch_size),)}, got {tuple(traj_rank_t.shape)}."
            )
            raise ValueError(msg)

        history_steps = int(self.config.state_history_steps)
        if history_steps > 0:
            state_history_t = cast(Tensor, state_history).to(
                device=self.device, dtype=torch.float32
            )
            expected_history = (int(batch_size), history_steps + 1, state_dim)
            if tuple(state_history_t.shape) != expected_history:
                msg = (
                    "hl/state_history shape mismatch: expected "
                    f"{expected_history}, got {tuple(state_history_t.shape)}."
                )
                raise ValueError(msg)
            planner_state = state_history_t.reshape(int(batch_size), -1).contiguous()
        else:
            planner_state = state
        return state, planner_state, future_window, target, traj_rank_t

    # -- losses / diagnostics ---------------------------------------------
    def _target_z(self, state: Tensor, future_window: Tensor) -> Tensor:
        with torch.no_grad():
            return self.skill_encoder(state, self._encoder_input_window(future_window))

    def _lang_for_ranks(self, traj_rank: Tensor) -> Tensor:
        if not self.condition_on_language:
            return torch.empty(
                (int(traj_rank.numel()), 0),
                device=self.device,
                dtype=torch.float32,
            )
        return self.rank_embeddings.index_select(0, traj_rank)

    def _current_state_feature_dropout_prob(self) -> float:
        if int(self.update) < int(self.config.state_feature_dropout_warmup_updates):
            return 0.0
        return float(self.config.state_feature_dropout_prob)

    def _state_feature_dropout_slices(self) -> list[tuple[int, int]]:
        prob = self._current_state_feature_dropout_prob()
        if prob <= 0.0:
            return []
        terms = tuple(self.config.state_feature_dropout_terms)
        if len(terms) == 0:
            return []
        if any(term in {"*", "all"} for term in terms):
            return [(0, self.planner_state_dim)]
        missing = [term for term in terms if term not in self.state_feature_slices]
        if missing:
            available = sorted(self.state_feature_slices)
            msg = (
                "state_feature_dropout_terms requested unavailable macro-state "
                f"features: {missing}. Available features: {available}."
            )
            raise ValueError(msg)
        history_len = int(self.config.state_history_steps) + 1
        slices: list[tuple[int, int]] = []
        for term in terms:
            start, end = self.state_feature_slices[term]
            for history_index in range(history_len):
                offset = history_index * self.state_dim
                slices.append((offset + start, offset + end))
        return slices

    def _augment_state(self, state: Tensor) -> tuple[Tensor, dict[str, float]]:
        """Optionally corrupt the generator's state input (M3 robustness).

        Per-dim-scaled Gaussian noise forces the generator to lean on the
        language goal rather than memorize exact expert states - a first step
        toward tolerating the robot's achieved (non-expert) state at rollout.
        The distillation target z is always computed from the clean expert state.
        """
        metrics: dict[str, float] = {}
        augmented = state
        std = float(self.config.state_noise_std)
        if std > 0.0 and int(state.shape[0]) >= 2:
            per_dim = state.std(dim=0, keepdim=True)
            augmented = augmented + std * per_dim * torch.randn_like(state)
            metrics["state_noise_std"] = std

        dropout_prob = self._current_state_feature_dropout_prob()
        dropout_slices = self._state_feature_dropout_slices()
        if dropout_prob <= 0.0 or len(dropout_slices) == 0:
            return augmented, metrics

        row_mask = torch.rand(
            int(state.shape[0]), device=state.device, dtype=torch.float32
        ) < dropout_prob
        active_rows = int(row_mask.sum().item())
        if active_rows == 0:
            metrics["state_feature_dropout/prob"] = float(dropout_prob)
            metrics["state_feature_dropout/active_frac"] = 0.0
            return augmented, metrics

        augmented = augmented.clone()
        mode = str(self.config.state_feature_dropout_mode)
        for start, end in dropout_slices:
            if mode == "zero":
                replacement = torch.zeros(
                    (active_rows, end - start),
                    device=state.device,
                    dtype=state.dtype,
                )
            elif mode == "batch_mean":
                replacement = state[:, start:end].mean(dim=0, keepdim=True).expand(
                    active_rows, -1
                )
            elif mode == "shuffle":
                perm = torch.randperm(int(state.shape[0]), device=state.device)
                replacement = state.index_select(0, perm)[row_mask, start:end]
            else:
                msg = f"Unsupported state_feature_dropout_mode={mode!r}."
                raise ValueError(msg)
            augmented[row_mask, start:end] = replacement

        metrics["state_feature_dropout/prob"] = float(dropout_prob)
        metrics["state_feature_dropout/active_frac"] = float(
            row_mask.to(dtype=torch.float32).mean().item()
        )
        metrics["state_feature_dropout/num_features"] = float(
            sum(end - start for start, end in dropout_slices)
        )
        return augmented, metrics

    def _different_language_ranks(self, traj_rank: Tensor) -> tuple[Tensor, Tensor]:
        """Choose a trajectory rank with a different language embedding per row."""
        if not self.condition_on_language:
            return traj_rank.clone(), torch.zeros_like(traj_rank, dtype=torch.bool)
        num_ranks = int(self.rank_embeddings.shape[0])
        wrong_rank = traj_rank.clone()
        has_negative = torch.zeros_like(traj_rank, dtype=torch.bool)
        if num_ranks <= 1:
            return wrong_rank, has_negative

        current = self.rank_embeddings.index_select(0, traj_rank)
        unresolved = torch.ones_like(traj_rank, dtype=torch.bool)
        for offset in range(1, num_ranks):
            candidate = (traj_rank + offset) % num_ranks
            candidate_embedding = self.rank_embeddings.index_select(0, candidate)
            different = (candidate_embedding - current).abs().amax(dim=-1) > 1.0e-6
            take = unresolved & different
            wrong_rank = torch.where(take, candidate, wrong_rank)
            has_negative = has_negative | take
            unresolved = unresolved & ~take
            if not bool(unresolved.any().item()):
                break
        return wrong_rank, has_negative

    def _generator_forward_for_contrastive(
        self,
        state: Tensor,
        lang: Tensor,
        *,
        shared_noise: Tensor | None,
    ) -> Tensor:
        if isinstance(
            self.generator, (FlowMatchingSkillCommander, DiffusionSkillCommander)
        ):
            return self.generator(state, lang, noise=shared_noise)
        return self.generator(state, lang)

    def _current_language_contrastive_coeff(self) -> float:
        if int(self.update) < int(self.config.language_contrastive_warmup_updates):
            return 0.0
        return float(self.config.language_contrastive_coeff)

    def _language_contrastive_loss(
        self,
        state: Tensor,
        lang: Tensor,
        z_target: Tensor,
        traj_rank: Tensor,
    ) -> tuple[Tensor, dict[str, float]]:
        coeff = self._current_language_contrastive_coeff()
        zero = z_target.new_zeros(())
        if coeff <= 0.0 or not self.condition_on_language:
            return zero, {}

        wrong_rank, has_negative = self._different_language_ranks(traj_rank)
        active = has_negative.to(dtype=z_target.dtype)
        active_count = active.sum().clamp_min(1.0)
        if not bool(has_negative.any().item()):
            return zero, {
                "language_contrastive/active_frac": 0.0,
                "language_contrastive/loss": 0.0,
            }

        wrong_lang = self.rank_embeddings.index_select(0, wrong_rank)
        shared_noise = None
        if isinstance(
            self.generator, (FlowMatchingSkillCommander, DiffusionSkillCommander)
        ):
            shared_noise = torch.randn_like(z_target) * float(
                self.generator.inference_noise_std
            )
        z_pos = self._generator_forward_for_contrastive(
            state, lang, shared_noise=shared_noise
        )
        z_neg = self._generator_forward_for_contrastive(
            state, wrong_lang, shared_noise=shared_noise
        )
        pos_cos = F.cosine_similarity(z_pos, z_target, dim=-1)
        neg_cos = F.cosine_similarity(z_neg, z_target, dim=-1)
        margin = float(self.config.language_contrastive_margin)
        per_row = F.relu(neg_cos - pos_cos + margin)
        loss = (per_row * active).sum() / active_count
        metrics = {
            "language_contrastive/active_frac": float(active.mean().detach().item()),
            "language_contrastive/loss": float(loss.detach().item()),
            "language_contrastive/pos_cosine": float(
                ((pos_cos * active).sum() / active_count).detach().item()
            ),
            "language_contrastive/neg_cosine": float(
                ((neg_cos * active).sum() / active_count).detach().item()
            ),
            "language_contrastive/delta": float(
                (((pos_cos - neg_cos) * active).sum() / active_count).detach().item()
            ),
        }
        return loss, metrics

    @staticmethod
    def _distill_metrics(
        z_hat: Tensor, z_target: Tensor, *, prefix: str
    ) -> dict[str, float]:
        mse = F.mse_loss(z_hat, z_target)
        cosine = F.cosine_similarity(z_hat, z_target, dim=-1).mean()
        return {
            f"{prefix}/z_mse": float(mse.item()),
            f"{prefix}/z_cosine": float(cosine.item()),
            f"{prefix}/z_hat_rms": float(z_hat.pow(2).mean().sqrt().item()),
            f"{prefix}/z_target_rms": float(z_target.pow(2).mean().sqrt().item()),
        }

    # -- train / eval ------------------------------------------------------
    def train_step(self) -> dict[str, float]:
        self.generator.train()
        state, planner_state, future_window, _, traj_rank = (
            self._sample_and_validate_macro_batch(
                self.config.batch_size,
                split=self.config.train_split,
            )
        )
        z_target = self._target_z(state, future_window)
        lang = self._lang_for_ranks(traj_rank)
        cmd_state, augment_metrics = self._augment_state(planner_state)
        if isinstance(self.generator, FlowMatchingSkillCommander):
            flow_loss, flow_metrics = self.generator.flow_matching_loss(
                cmd_state, lang, z_target
            )
            contrastive_loss, contrastive_metrics = self._language_contrastive_loss(
                cmd_state, lang, z_target, traj_rank
            )
            contrastive_coeff = self._current_language_contrastive_coeff()
            loss = flow_loss + contrastive_coeff * contrastive_loss
            with torch.no_grad():
                z_hat = self.generator(planner_state, lang)
            metrics = self._distill_metrics(z_hat.detach(), z_target, prefix="train")
            metrics.update(
                {f"train/{key}": value for key, value in augment_metrics.items()}
            )
            metrics.update(
                {f"train/{key}": value for key, value in flow_metrics.items()}
            )
            metrics.update(
                {f"train/{key}": value for key, value in contrastive_metrics.items()}
            )
            loss_metrics = {
                "train/loss": float(loss.detach().item()),
                "train/flow_loss": float(flow_loss.detach().item()),
                "train/language_contrastive_weighted_loss": float(
                    (contrastive_coeff * contrastive_loss).detach().item()
                ),
            }
        elif isinstance(self.generator, DiffusionSkillCommander):
            diffusion_loss, diffusion_metrics = self.generator.diffusion_loss(
                cmd_state, lang, z_target
            )
            contrastive_loss, contrastive_metrics = self._language_contrastive_loss(
                cmd_state, lang, z_target, traj_rank
            )
            contrastive_coeff = self._current_language_contrastive_coeff()
            loss = diffusion_loss + contrastive_coeff * contrastive_loss
            with torch.no_grad():
                z_hat = self.generator(planner_state, lang)
            metrics = self._distill_metrics(z_hat.detach(), z_target, prefix="train")
            metrics.update(
                {f"train/{key}": value for key, value in augment_metrics.items()}
            )
            metrics.update(
                {f"train/{key}": value for key, value in diffusion_metrics.items()}
            )
            metrics.update(
                {f"train/{key}": value for key, value in contrastive_metrics.items()}
            )
            loss_metrics = {
                "train/loss": float(loss.detach().item()),
                "train/diffusion_loss": float(diffusion_loss.detach().item()),
                "train/language_contrastive_weighted_loss": float(
                    (contrastive_coeff * contrastive_loss).detach().item()
                ),
            }
        else:
            z_hat = self.generator(cmd_state, lang)
            mse_loss = F.mse_loss(z_hat, z_target)
            cosine_loss = 1.0 - F.cosine_similarity(z_hat, z_target, dim=-1).mean()
            z_norm_loss = z_hat.pow(2).mean()
            contrastive_loss, contrastive_metrics = self._language_contrastive_loss(
                cmd_state, lang, z_target, traj_rank
            )
            contrastive_coeff = self._current_language_contrastive_coeff()
            loss = (
                mse_loss
                + self.config.cosine_loss_coeff * cosine_loss
                + self.config.z_norm_coeff * z_norm_loss
                + contrastive_coeff * contrastive_loss
            )
            metrics = self._distill_metrics(z_hat.detach(), z_target, prefix="train")
            metrics.update(
                {f"train/{key}": value for key, value in augment_metrics.items()}
            )
            metrics.update(
                {f"train/{key}": value for key, value in contrastive_metrics.items()}
            )
            loss_metrics = {
                "train/loss": float(loss.detach().item()),
                "train/mse_loss": float(mse_loss.detach().item()),
                "train/cosine_loss": float(cosine_loss.detach().item()),
                "train/z_norm_loss": float(z_norm_loss.detach().item()),
                "train/language_contrastive_weighted_loss": float(
                    (contrastive_coeff * contrastive_loss).detach().item()
                ),
            }

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.config.grad_clip_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.generator.parameters(),
                max_norm=float(self.config.grad_clip_norm),
            )
            metrics["train/grad_norm"] = float(grad_norm.item())
        self.optimizer.step()
        self.update += 1
        metrics.update(loss_metrics)
        return metrics

    @torch.no_grad()
    def evaluate(
        self,
        *,
        num_batches: int | None = None,
        batch_size: int | None = None,
        prefix: str = "eval",
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

        was_training = self.generator.training
        self.generator.eval()
        accum: dict[str, float] = {}
        for _ in range(num_batches):
            state, planner_state, future_window, _, traj_rank = (
                self._sample_and_validate_macro_batch(
                    batch_size,
                    split=split,
                )
            )
            z_target = self._target_z(state, future_window)
            lang = self._lang_for_ranks(traj_rank)
            z_hat = self.generator(planner_state, lang)
            batch_metrics = self._distill_metrics(z_hat, z_target, prefix=prefix)
            # Wrong-language control: a different language embedding should
            # degrade the match if the generator actually uses conditioning.
            wrong_rank, has_negative = self._different_language_ranks(traj_rank)
            if bool(has_negative.any().item()):
                wrong_lang = self.rank_embeddings.index_select(0, wrong_rank)
                z_hat_wrong = self.generator(planner_state, wrong_lang)
                active = has_negative.to(dtype=z_target.dtype)
                active_count = active.sum().clamp_min(1.0)
                pos_cos = F.cosine_similarity(z_hat, z_target, dim=-1)
                wrong_cos = F.cosine_similarity(z_hat_wrong, z_target, dim=-1)
                wrong_mse = (z_hat_wrong - z_target).pow(2).mean(dim=-1)
                batch_metrics[f"{prefix}/z_cosine_wrong_lang"] = float(
                    ((wrong_cos * active).sum() / active_count).item()
                )
                batch_metrics[f"{prefix}/z_cosine_shuffled_lang"] = batch_metrics[
                    f"{prefix}/z_cosine_wrong_lang"
                ]
                batch_metrics[f"{prefix}/z_mse_wrong_lang"] = float(
                    ((wrong_mse * active).sum() / active_count).item()
                )
                batch_metrics[f"{prefix}/z_cosine_language_delta"] = float(
                    (((pos_cos - wrong_cos) * active).sum() / active_count).item()
                )
            for key, value in batch_metrics.items():
                accum[key] = accum.get(key, 0.0) + float(value)
        for key in accum:
            accum[key] /= float(num_batches)
        if was_training:
            self.generator.train()
        return accum

    def train(
        self,
        *,
        log_callback: Callable[[dict[str, float | int]], None] | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> SkillCommanderTrainState:
        start_time = time.perf_counter()
        state = SkillCommanderTrainState()
        for _ in range(self.config.num_updates):
            metrics = self.train_step()
            should_log = (
                self.update in (1, self.config.num_updates)
                or self.update % self.config.log_interval == 0
            )
            if should_log:
                metrics.update(self.evaluate(prefix="eval"))
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

    # -- checkpointing -----------------------------------------------------
    def checkpoint_state_dict(self) -> dict[str, Any]:
        return {
            "generator_state_dict": self.generator.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "skill_config": self.skill_config.to_dict(),
            "skill_checkpoint_path": str(self.config.skill_checkpoint_path),
            "language_embeddings_path": str(self.config.language_embeddings_path),
            "condition_on_language": bool(self.condition_on_language),
            "state_history_steps": int(self.config.state_history_steps),
            "state_dim": int(self.planner_state_dim),
            "macro_state_dim": int(self.state_dim),
            "planner_state_dim": int(self.planner_state_dim),
            "lang_embed_dim": int(self.lang_embed_dim),
            "z_dim": int(self.z_dim),
            "horizon_steps": int(self.horizon_steps),
            "encoder_window_mode": str(self.skill_config.encoder_window_mode),
            "update": int(self.update),
        }

    def save_checkpoint(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_state_dict(), target)


# ---------------------------------------------------------------------------
# Rollout-time latent-command source (System 1 driving System 0)
# ---------------------------------------------------------------------------
class FrozenSkillCommanderSampler(FrozenHighLevelSkillCommandSampler):
    """Skill-commander latent-command source for low-level rollouts.

    Drop-in replacement for ``FrozenHighLevelSkillCommandSampler`` that produces
    the skill latent ``z`` from ``(current_state, language_goal)`` via a trained
    ``SkillCommander`` instead of encoding the expert future window. The
    command/phase/scheduling machinery of the base class is reused unchanged;
    only the ``z`` production is overridden. Inference-only - online finetuning
    is not supported.
    """

    def __init__(
        self,
        *,
        env: object,
        checkpoint_path: str | Path,
        language_embeddings_path: str | Path,
        latent_dim: int,
        latent_steps_min: int,
        latent_steps_max: int,
        discover_env_method: Callable[[object, str], Callable[..., Any] | None],
        generator_config_overrides: Mapping[str, Any] | None = None,
        horizon_steps: int | None = None,
        command_phase_mode: str = "none",
        code_latent_dim: int | None = None,
        phase_period: int | None = None,
        command_mode: str = "z",
        use_achieved_state: bool = False,
        goal_name: str = "",
        goal_rank: int = -1,
        device: torch.device | str | None = None,
    ) -> None:
        # NOTE: we deliberately do not call super().__init__ - that loads a skill
        # encoder, whereas we load a generator. We set the attributes the reused
        # base methods rely on and disable all online-finetune machinery.
        self.latent_dim = _require_positive_int("latent_dim", latent_dim)
        self.latent_steps_min = max(1, int(latent_steps_min))
        self.latent_steps_max = max(self.latent_steps_min, int(latent_steps_max))
        self.finetune_enabled = False
        self.train_diffsr = False
        self.update_interval = 1
        self.optimizer = None
        self.finetune_updates = 0
        self.skill_encoder = None
        self.initial_skill_encoder = None
        self.z_norm_coeff = 0.0

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
            env, "current_expert_macro_transition_batch"
        )
        if self._current_macro_sampler is None:
            msg = (
                "command_source='skill_commander' requires the environment to "
                "expose current_expert_macro_transition_batch(...)."
            )
            raise ValueError(msg)
        # Closed-loop (full-M3) mode: condition the commander on the robot's
        # achieved macro state instead of the expert-reference state.
        self.use_achieved_state = bool(use_achieved_state)
        self._achieved_macro_sampler = discover_env_method(
            env, "current_achieved_macro_transition_batch"
        )
        if self.use_achieved_state and self._achieved_macro_sampler is None:
            msg = (
                "skill_commander_use_achieved_state=True requires the environment "
                "to expose current_achieved_macro_transition_batch(...)."
            )
            raise ValueError(msg)
        self._offline_macro_sampler = None
        self.forced_language_goal_name = str(goal_name).strip()
        self.forced_language_goal_rank = int(goal_rank)
        self.forced_language_embedding: Tensor | None = None
        if self.forced_language_goal_rank < -1:
            msg = "goal_rank must be >= -1."
            raise ValueError(msg)
        if self.forced_language_goal_name and self.forced_language_goal_rank >= 0:
            msg = "Set only one of goal_name or goal_rank."
            raise ValueError(msg)

        checkpoint = torch.load(
            Path(checkpoint_path).expanduser(),
            map_location=self.device,
            weights_only=False,
        )
        self.config = HighLevelSkillDiffSRConfig.from_dict(checkpoint["skill_config"])
        if (
            horizon_steps is not None
            and int(horizon_steps) != self.config.horizon_steps
        ):
            msg = (
                "Configured hl_skill_horizon_steps does not match generator "
                f"checkpoint horizon_steps: {int(horizon_steps)} != "
                f"{self.config.horizon_steps}."
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
                "Language skill code_latent_dim must match command-mode pre-phase "
                f"width: {int(code_latent_dim)} != {self.command_code_dim} for "
                f"command_mode={self.command_mode!r}."
            )
            raise ValueError(msg)
        expected_latent_dim = self.command_code_dim + self.phase_dim
        if expected_latent_dim != self.latent_dim:
            msg = (
                "Language skill command width must match ipmd.latent_dim: "
                f"command_mode={self.command_mode!r} pre-phase width "
                f"{self.command_code_dim} + phase_dim {self.phase_dim} "
                f"!= {self.latent_dim}."
            )
            raise ValueError(msg)

        config_payload = dict(checkpoint.get("config", {}))
        self.condition_on_language = _coerce_bool(
            "condition_on_language",
            config_payload.get(
                "condition_on_language",
                checkpoint.get("condition_on_language", True),
            ),
        )
        self.state_history_steps = _require_non_negative_int(
            "state_history_steps",
            config_payload.get(
                "state_history_steps", checkpoint.get("state_history_steps", 0)
            ),
        )
        self.planner_state_dim = int(
            checkpoint.get("planner_state_dim", checkpoint["state_dim"])
        )
        self.state_dim = int(
            checkpoint.get("macro_state_dim", self.planner_state_dim)
        )
        self.lang_embed_dim = int(checkpoint["lang_embed_dim"])
        self.generator = _build_skill_commander_generator_from_checkpoint(
            checkpoint,
            state_dim=self.planner_state_dim,
            lang_embed_dim=self.lang_embed_dim,
            z_dim=self.skill_z_dim,
            config_overrides=generator_config_overrides,
        ).to(self.device)
        self.generator.load_state_dict(checkpoint["generator_state_dict"])
        self.generator.eval()
        self.generator.requires_grad_(False)

        # DiffSR is only needed to expand z -> phi for non-z command modes; load
        # it from the source skill checkpoint the generator distilled against.
        self.diffsr = _build_diffsr(self.config, self.state_dim, self.device).to(
            self.device
        )
        if self.command_mode != "z":
            skill_checkpoint_path = str(checkpoint.get("skill_checkpoint_path", ""))
            skill_checkpoint = torch.load(
                Path(skill_checkpoint_path).expanduser(),
                map_location=self.device,
                weights_only=False,
            )
            diffsr_state = skill_checkpoint.get("diffsr_state_dict")
            if diffsr_state is None:
                msg = (
                    "command_mode != 'z' requires the source skill checkpoint to "
                    "contain diffsr_state_dict."
                )
                raise ValueError(msg)
            self.diffsr.load_state_dict(diffsr_state)
            feature_norm_state = skill_checkpoint.get(
                "feature_normalization_state_dict"
            )
            obs_norm = getattr(self.diffsr, "obs_norm", None)
            if isinstance(obs_norm, nn.Module) and feature_norm_state:
                obs_norm.load_state_dict(feature_norm_state)
        self.diffsr.eval()
        self.diffsr.requires_grad_(False)

        if self.condition_on_language:
            names_provider = discover_env_method(env, "expert_trajectory_motion_names")
            if names_provider is None:
                msg = (
                    "command_source='skill_commander' requires the environment to "
                    "expose expert_trajectory_motion_names()."
                )
                raise ValueError(msg)
            language_path = str(language_embeddings_path).strip() or str(
                checkpoint.get("language_embeddings_path", "")
            ).strip()
            if not language_path:
                msg = (
                    "Language-conditioned skill commander checkpoints require "
                    "language_embeddings_path."
                )
                raise ValueError(msg)
            table = load_language_embedding_table(language_path)
            self.rank_embeddings = build_rank_embedding_lookup(
                table, [str(name) for name in names_provider()], self.device
            )
            if (
                not self.forced_language_goal_name
                and self.forced_language_goal_rank >= 0
            ):
                names = table.get("names")
                if not isinstance(names, list):
                    msg = "Language embedding table is missing a list of names."
                    raise ValueError(msg)
                if self.forced_language_goal_rank >= len(names):
                    msg = (
                        f"goal_rank={self.forced_language_goal_rank} is out of "
                        f"range for {len(names)} language goals."
                    )
                    raise ValueError(msg)
                self.forced_language_goal_name = str(
                    names[self.forced_language_goal_rank]
                )
            if self.forced_language_goal_name:
                name_to_index = table["name_to_index"]
                goal_index = name_to_index.get(self.forced_language_goal_name)
                if goal_index is None:
                    msg = (
                        "Language embedding table has no entry for forced goal "
                        f"{self.forced_language_goal_name!r}."
                    )
                    raise ValueError(msg)
                embeddings = cast(Tensor, table["embeddings"]).to(
                    device=self.device, dtype=torch.float32
                )
                self.forced_language_embedding = embeddings[
                    int(goal_index)
                ].reshape(1, -1)
        else:
            self.rank_embeddings = torch.empty(
                (0, 0), device=self.device, dtype=torch.float32
            )

        # Rollout command buffers (managed by the inherited sample_for_step).
        self._codes: Tensor | None = None
        self._latent_steps: Tensor | None = None
        self._active_macro_ids: Tensor | None = None
        self._cache_state_chunks: list[Tensor] = []
        self._cache_future_window_chunks: list[Tensor] = []
        self._cache_target_chunks: list[Tensor] = []
        self._cache_initial_z_chunks: list[Tensor] = []
        self._next_macro_id = 0

    def _planner_state_from_batch(
        self,
        batch: TensorDictBase,
        state: Tensor,
        *,
        batch_size: int,
        source: str,
    ) -> Tensor:
        if int(self.state_history_steps) <= 0:
            return state
        state_history = batch.get(("hl", "state_history"))
        if state_history is None:
            msg = f"{source} macro batch is missing hl/state_history."
            raise ValueError(msg)
        state_history_t = cast(Tensor, state_history).to(
            device=self.device, dtype=torch.float32
        )
        expected_history = (
            int(batch_size),
            int(self.state_history_steps) + 1,
            self.state_dim,
        )
        if tuple(state_history_t.shape) != expected_history:
            msg = (
                f"{source} macro state_history shape mismatch: expected "
                f"{expected_history}, got {tuple(state_history_t.shape)}."
            )
            raise ValueError(msg)
        return state_history_t.reshape(int(batch_size), -1).contiguous()

    def _lang_from_batch(self, batch: TensorDictBase, *, batch_size: int) -> Tensor:
        if not self.condition_on_language:
            return torch.empty(
                (int(batch_size), 0), device=self.device, dtype=torch.float32
            )
        if self.forced_language_embedding is not None:
            return self.forced_language_embedding.expand(int(batch_size), -1)
        traj_rank = batch.get(("hl", "traj_rank"))
        if traj_rank is None:
            msg = (
                "Current expert macro batch is missing 'traj_rank' required for "
                "language-skill command generation."
            )
            raise ValueError(msg)
        traj_rank_t = cast(Tensor, traj_rank).reshape(-1).to(
            device=self.device, dtype=torch.long
        )
        expected = (int(batch_size),)
        if tuple(traj_rank_t.shape) != expected:
            msg = (
                "hl/traj_rank shape mismatch: expected "
                f"{expected}, got {tuple(traj_rank_t.shape)}."
            )
            raise ValueError(msg)
        return self.rank_embeddings.index_select(0, traj_rank_t)

    @torch.no_grad()
    def _encode_current_macro_batch(
        self,
        env_ids: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        macro_sampler = (
            self._achieved_macro_sampler
            if self.use_achieved_state
            else self._current_macro_sampler
        )
        batch = macro_sampler(
            horizon_steps=int(self.config.horizon_steps),
            env_ids=env_ids,
            state_history_steps=int(self.state_history_steps),
        )
        batch_size = int(env_ids.numel())
        state, future_window, target = _validate_hl_macro_batch(
            batch,
            batch_size=batch_size,
            horizon_steps=int(self.config.horizon_steps),
            device=self.device,
            state_dim=self.state_dim,
            source="Current skill-commander",
        )
        planner_state = self._planner_state_from_batch(
            batch,
            state,
            batch_size=batch_size,
            source="Current skill-commander",
        )
        lang = self._lang_from_batch(batch, batch_size=batch_size)
        z = self.generator(planner_state, lang)
        # initial_z mirrors z; it is only consumed by the (disabled) finetune path.
        return z, state, future_window, target, z

    def trainable_parameters(self) -> list[nn.Parameter]:
        return []

    def checkpoint_state_dict(self) -> dict[str, Any]:
        return {"generator_state_dict": self.generator.state_dict()}

    def load_checkpoint_state_dict(self, state: Mapping[str, Any]) -> None:
        if "generator_state_dict" in state:
            self.generator.load_state_dict(state["generator_state_dict"])
