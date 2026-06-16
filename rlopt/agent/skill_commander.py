"""Language-conditioned skill generator (System 2).

This module distills a frozen high-level skill encoder (see
``hl_skill_diffsr.py``) into a generator that maps ``(current_state,
language_goal) -> z``. Instead of peeking at the expert future window to encode a
skill, the generator produces the same skill latent ``z`` from the current state
and a language-goal embedding (e.g. a LAFAN1 trajectory name embedded offline by
``scripts/rlopt/build_language_goal_embeddings.py``).

Contents:

* ``SkillCommanderConfig`` - training/config dataclass.
* ``SkillCommander`` - the ``(state, lang) -> z`` network.
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
from tensordict import TensorDictBase
from torch import Tensor, nn

from rlopt.agent.hl_skill_diffsr import (
    FrozenHighLevelSkillCommandSampler,
    HighLevelSkillDiffSRConfig,
    HighLevelSkillEncoder,
    _jsonable,
    _normalize_command_mode,
    _normalize_split_value,
    _require_fraction,
    _require_non_negative_float,
    _require_positive_float,
    _require_positive_int,
)


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
    generator_hidden_dims: tuple[int, ...] = (1024, 512, 512)
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
    device: str = "auto"

    def validate(self) -> None:
        self.skill_checkpoint_path = str(self.skill_checkpoint_path).strip()
        if not self.skill_checkpoint_path:
            msg = "skill_checkpoint_path is required."
            raise ValueError(msg)
        self.language_embeddings_path = str(self.language_embeddings_path).strip()
        if not self.language_embeddings_path:
            msg = "language_embeddings_path is required."
            raise ValueError(msg)
        self.generator_hidden_dims = tuple(
            _require_positive_int("generator_hidden_dims", dim)
            for dim in self.generator_hidden_dims
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
        self.lang_embed_dim = _require_positive_int("lang_embed_dim", lang_embed_dim)
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

        # Preflight a small batch to discover the macro state dimension.
        state, _, _, _ = self._sample_and_validate_macro_batch(
            min(self.config.batch_size, self.config.preflight_batch_size),
            split=self.config.train_split,
        )
        self.state_dim = int(state.shape[-1])

        self.skill_encoder = HighLevelSkillEncoder(
            state_dim=self.state_dim,
            window_steps=self.encoder_window_steps,
            z_dim=self.z_dim,
            hidden_dims=self.skill_config.encoder_hidden_dims,
        ).to(self.device)
        self.skill_encoder.load_state_dict(skill_checkpoint["skill_encoder_state_dict"])
        self.skill_encoder.eval()
        self.skill_encoder.requires_grad_(False)

        # Language goal lookup: trajectory rank -> embedding.
        self.language_table = load_language_embedding_table(
            self.config.language_embeddings_path
        )
        self.lang_embed_dim = int(self.language_table["embed_dim"])
        self.rank_embeddings = build_rank_embedding_lookup(
            self.language_table,
            self._expert_trajectory_motion_names(),
            self.device,
        )

        self.generator = SkillCommander(
            state_dim=self.state_dim,
            lang_embed_dim=self.lang_embed_dim,
            z_dim=self.z_dim,
            hidden_dims=self.config.generator_hidden_dims,
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
        )

    def _sample_and_validate_macro_batch(
        self,
        batch_size: int,
        *,
        split: str | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch = self._sample_macro_batch(batch_size, split=split)
        return self._validate_macro_batch(batch, batch_size=batch_size)

    def _validate_macro_batch(
        self,
        batch: TensorDictBase,
        *,
        batch_size: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        state = batch.get(("hl", "state"))
        future_window = batch.get(("hl", "future_window"))
        target = batch.get(("hl", "target"))
        traj_rank = batch.get(("hl", "traj_rank"))
        missing = [
            name
            for name, value in (
                ("hl/state", state),
                ("hl/future_window", future_window),
                ("hl/target", target),
                ("hl/traj_rank", traj_rank),
            )
            if value is None
        ]
        if missing:
            msg = (
                f"Expert macro batch is missing keys: {missing}. The environment "
                "must return 'traj_rank' for language-skill training."
            )
            raise ValueError(msg)
        state = cast(Tensor, state).to(device=self.device, dtype=torch.float32)
        future_window = cast(Tensor, future_window).to(
            device=self.device, dtype=torch.float32
        )
        target = cast(Tensor, target).to(device=self.device, dtype=torch.float32)
        traj_rank = (
            cast(Tensor, traj_rank).reshape(-1).to(device=self.device, dtype=torch.long)
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
        if tuple(traj_rank.shape) != (int(batch_size),):
            msg = (
                "hl/traj_rank shape mismatch: expected "
                f"{(int(batch_size),)}, got {tuple(traj_rank.shape)}."
            )
            raise ValueError(msg)
        return state, future_window, target, traj_rank

    # -- losses / diagnostics ---------------------------------------------
    def _target_z(self, state: Tensor, future_window: Tensor) -> Tensor:
        with torch.no_grad():
            return self.skill_encoder(state, self._encoder_input_window(future_window))

    def _lang_for_ranks(self, traj_rank: Tensor) -> Tensor:
        return self.rank_embeddings.index_select(0, traj_rank)

    def _augment_state(self, state: Tensor) -> Tensor:
        """Optionally corrupt the generator's state input (M3 robustness).

        Per-dim-scaled Gaussian noise forces the generator to lean on the
        language goal rather than memorize exact expert states - a first step
        toward tolerating the robot's achieved (non-expert) state at rollout.
        The distillation target z is always computed from the clean expert state.
        """
        std = float(self.config.state_noise_std)
        if std <= 0.0 or int(state.shape[0]) < 2:
            return state
        per_dim = state.std(dim=0, keepdim=True)
        return state + std * per_dim * torch.randn_like(state)

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
        state, future_window, _, traj_rank = self._sample_and_validate_macro_batch(
            self.config.batch_size,
            split=self.config.train_split,
        )
        z_target = self._target_z(state, future_window)
        lang = self._lang_for_ranks(traj_rank)
        z_hat = self.generator(self._augment_state(state), lang)

        mse_loss = F.mse_loss(z_hat, z_target)
        cosine_loss = 1.0 - F.cosine_similarity(z_hat, z_target, dim=-1).mean()
        z_norm_loss = z_hat.pow(2).mean()
        loss = (
            mse_loss
            + self.config.cosine_loss_coeff * cosine_loss
            + self.config.z_norm_coeff * z_norm_loss
        )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        metrics = self._distill_metrics(z_hat.detach(), z_target, prefix="train")
        if self.config.grad_clip_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.generator.parameters(),
                max_norm=float(self.config.grad_clip_norm),
            )
            metrics["train/grad_norm"] = float(grad_norm.item())
        self.optimizer.step()
        self.update += 1
        metrics.update(
            {
                "train/loss": float(loss.detach().item()),
                "train/mse_loss": float(mse_loss.detach().item()),
                "train/cosine_loss": float(cosine_loss.detach().item()),
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
            state, future_window, _, traj_rank = self._sample_and_validate_macro_batch(
                batch_size,
                split=split,
            )
            z_target = self._target_z(state, future_window)
            lang = self._lang_for_ranks(traj_rank)
            z_hat = self.generator(state, lang)
            batch_metrics = self._distill_metrics(z_hat, z_target, prefix=prefix)
            # Wrong-language control: shuffling goals should degrade the match if
            # the generator actually uses the language conditioning.
            if int(lang.shape[0]) > 1:
                shuffled = lang[torch.randperm(lang.shape[0], device=lang.device)]
                z_hat_shuffled = self.generator(state, shuffled)
                batch_metrics[f"{prefix}/z_cosine_shuffled_lang"] = float(
                    F.cosine_similarity(z_hat_shuffled, z_target, dim=-1).mean().item()
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
            "state_dim": int(self.state_dim),
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
# Rollout-time latent-command source (System 2 driving System 1)
# ---------------------------------------------------------------------------
class FrozenSkillCommanderSampler(FrozenHighLevelSkillCommandSampler):
    """Language-conditioned latent-command source for low-level rollouts.

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
        horizon_steps: int | None = None,
        command_phase_mode: str = "none",
        code_latent_dim: int | None = None,
        phase_period: int | None = None,
        command_mode: str = "z",
        use_achieved_state: bool = False,
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
        self.device = self._resolve_device(env, device)

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

        self.state_dim = int(checkpoint["state_dim"])
        self.lang_embed_dim = int(checkpoint["lang_embed_dim"])
        generator_hidden_dims = tuple(
            int(dim) for dim in checkpoint["config"]["generator_hidden_dims"]
        )
        self.generator = SkillCommander(
            state_dim=self.state_dim,
            lang_embed_dim=self.lang_embed_dim,
            z_dim=self.skill_z_dim,
            hidden_dims=generator_hidden_dims,
        ).to(self.device)
        self.generator.load_state_dict(checkpoint["generator_state_dict"])
        self.generator.eval()
        self.generator.requires_grad_(False)

        # DiffSR is only needed to expand z -> phi for non-z command modes; load
        # it from the source skill checkpoint the generator distilled against.
        self.diffsr = self._build_diffsr().to(self.device)
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

        names_provider = discover_env_method(env, "expert_trajectory_motion_names")
        if names_provider is None:
            msg = (
                "command_source='skill_commander' requires the environment to "
                "expose expert_trajectory_motion_names()."
            )
            raise ValueError(msg)
        table = load_language_embedding_table(language_embeddings_path)
        self.rank_embeddings = build_rank_embedding_lookup(
            table, [str(name) for name in names_provider()], self.device
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
        )
        batch_size = int(env_ids.numel())
        state, future_window, target = self._validate_macro_batch(
            batch,
            batch_size=batch_size,
            source="Current skill-commander",
        )
        traj_rank = batch.get(("hl", "traj_rank"))
        if traj_rank is None:
            msg = (
                "Current expert macro batch is missing 'traj_rank' required for "
                "language-skill command generation."
            )
            raise ValueError(msg)
        traj_rank = traj_rank.reshape(-1).to(device=self.device, dtype=torch.long)
        lang = self.rank_embeddings.index_select(0, traj_rank)
        z = self.generator(state, lang)
        # initial_z mirrors z; it is only consumed by the (disabled) finetune path.
        return z, state, future_window, target, z

    def trainable_parameters(self) -> list[nn.Parameter]:
        return []

    def checkpoint_state_dict(self) -> dict[str, Any]:
        return {"generator_state_dict": self.generator.state_dict()}

    def load_checkpoint_state_dict(self, state: Mapping[str, Any]) -> None:
        if "generator_state_dict" in state:
            self.generator.load_state_dict(state["generator_state_dict"])
