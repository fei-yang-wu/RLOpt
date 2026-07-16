"""Shared causal Transformer-flow planner for high-level control interfaces."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _patch_term_ids(
    *, target_dim: int, patch_dim: int, term_widths: tuple[int, ...]
) -> Tensor:
    num_patches = math.ceil(int(target_dim) / int(patch_dim))
    if not term_widths:
        return torch.zeros(num_patches, dtype=torch.long)
    term_ends: list[int] = []
    running = 0
    for width in term_widths:
        running += int(width)
        term_ends.append(running)
    ids: list[int] = []
    for patch_idx in range(num_patches):
        patch_start = patch_idx * int(patch_dim)
        term_idx = 0
        while term_idx + 1 < len(term_ends) and patch_start >= term_ends[term_idx]:
            term_idx += 1
        ids.append(term_idx)
    return torch.as_tensor(ids, dtype=torch.long)


def _sinusoidal_embedding(t: Tensor, dim: int) -> Tensor:
    half = int(dim) // 2
    if half <= 0:
        return t.unsqueeze(-1)
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=t.device, dtype=t.dtype)
        / max(half - 1, 1)
    )
    args = t.unsqueeze(-1) * frequencies.unsqueeze(0)
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if int(dim) % 2 == 1:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class CausalInterfaceTransformerFlowPlanner(nn.Module):
    """One continuous planner backbone shared by every command interface.

    State is the flattened causal robot history. Targets may be DiffSR skills,
    full-body chunks, EE chunks, or other continuous packets. ``term_widths``
    changes only target token labels; it does not change the state backbone.
    """

    planner_type = "causal_interface_transformer_flow"

    def __init__(
        self,
        *,
        state_dim: int,
        target_dim: int,
        term_widths: Sequence[int] = (),
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        feedforward_dim: int = 2048,
        patch_dim: int = 32,
        num_state_tokens: int = 4,
        language_dim: int = 0,
        num_language_tokens: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.target_dim = int(target_dim)
        self.term_widths = tuple(int(width) for width in term_widths)
        self.d_model = int(d_model)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.feedforward_dim = int(feedforward_dim)
        self.patch_dim = int(patch_dim)
        self.num_state_tokens = int(num_state_tokens)
        self.language_dim = int(language_dim)
        self.num_language_tokens = int(num_language_tokens)
        self.dropout = float(dropout)
        if self.state_dim <= 0 or self.target_dim <= 0:
            msg = "state_dim and target_dim must be positive."
            raise ValueError(msg)
        if self.d_model <= 0 or self.patch_dim <= 0:
            msg = "d_model and patch_dim must be positive."
            raise ValueError(msg)
        if self.num_layers <= 0 or self.num_heads <= 0:
            msg = "num_layers and num_heads must be positive."
            raise ValueError(msg)
        if self.d_model % self.num_heads != 0:
            msg = "d_model must be divisible by num_heads."
            raise ValueError(msg)
        if self.num_state_tokens <= 0:
            msg = "num_state_tokens must be positive."
            raise ValueError(msg)
        if self.language_dim < 0:
            msg = "language_dim must be non-negative."
            raise ValueError(msg)
        if self.language_dim > 0 and self.num_language_tokens <= 0:
            msg = "num_language_tokens must be positive when language is enabled."
            raise ValueError(msg)
        if self.term_widths and sum(self.term_widths) != self.target_dim:
            msg = (
                f"term_widths sum to {sum(self.term_widths)}, "
                f"expected {self.target_dim}."
            )
            raise ValueError(msg)

        self.num_patches = math.ceil(self.target_dim / self.patch_dim)
        self.padded_target_dim = self.num_patches * self.patch_dim
        term_ids = _patch_term_ids(
            target_dim=self.target_dim,
            patch_dim=self.patch_dim,
            term_widths=self.term_widths,
        )
        self.register_buffer("patch_term_ids", term_ids, persistent=False)
        self.register_buffer("state_mean", torch.zeros(self.state_dim))
        self.register_buffer("state_std", torch.ones(self.state_dim))
        self.register_buffer("target_mean", torch.zeros(self.target_dim))
        self.register_buffer("target_std", torch.ones(self.target_dim))

        self.state_proj = nn.Sequential(
            nn.Linear(self.state_dim, self.d_model * self.num_state_tokens),
            nn.Mish(),
            nn.LayerNorm(self.d_model * self.num_state_tokens),
        )
        self.patch_proj = nn.Linear(self.patch_dim, self.d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 4),
            nn.Mish(),
            nn.Linear(self.d_model * 4, self.d_model),
        )
        self.state_token_embed = nn.Parameter(
            torch.zeros(1, self.num_state_tokens, self.d_model)
        )
        if self.language_dim > 0:
            self.language_proj: nn.Module | None = nn.Sequential(
                nn.Linear(
                    self.language_dim,
                    self.d_model * self.num_language_tokens,
                ),
                nn.Mish(),
                nn.LayerNorm(self.d_model * self.num_language_tokens),
            )
            self.language_token_embed = nn.Parameter(
                torch.zeros(1, self.num_language_tokens, self.d_model)
            )
        else:
            self.language_proj = None
            self.register_parameter("language_token_embed", None)
        self.patch_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, self.d_model)
        )
        self.term_embed = nn.Embedding(max(1, len(self.term_widths)), self.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.feedforward_dim,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=self.num_layers, enable_nested_tensor=False
        )
        self.output_norm = nn.LayerNorm(self.d_model)
        self.patch_head = nn.Linear(self.d_model, self.patch_dim)
        nn.init.normal_(self.state_token_embed, std=0.02)
        if self.language_token_embed is not None:
            nn.init.normal_(self.language_token_embed, std=0.02)
        nn.init.normal_(self.patch_pos_embed, std=0.02)

    @classmethod
    def from_config(
        cls, config: dict[str, Any]
    ) -> CausalInterfaceTransformerFlowPlanner:
        return cls(
            state_dim=int(config["state_dim"]),
            target_dim=int(config["target_dim"]),
            term_widths=tuple(int(x) for x in config.get("term_widths", ())),
            d_model=int(config.get("d_model", 512)),
            num_layers=int(config.get("num_layers", 6)),
            num_heads=int(config.get("num_heads", 8)),
            feedforward_dim=int(config.get("feedforward_dim", 2048)),
            patch_dim=int(config.get("patch_dim", 32)),
            num_state_tokens=int(config.get("num_state_tokens", 4)),
            language_dim=int(config.get("language_dim", 0)),
            num_language_tokens=int(config.get("num_language_tokens", 1)),
            dropout=float(config.get("dropout", 0.0)),
        )

    @torch.no_grad()
    def set_normalization(
        self,
        *,
        state_mean: Tensor,
        state_std: Tensor,
        target_mean: Tensor,
        target_std: Tensor,
        min_std: float = 1.0e-4,
    ) -> None:
        expected = {
            "state_mean": (state_mean, self.state_dim),
            "state_std": (state_std, self.state_dim),
            "target_mean": (target_mean, self.target_dim),
            "target_std": (target_std, self.target_dim),
        }
        for name, (value, width) in expected.items():
            if tuple(value.shape) != (width,):
                msg = f"{name} shape {tuple(value.shape)} != {(width,)}."
                raise ValueError(msg)
        self.state_mean.copy_(state_mean.to(self.state_mean))
        self.state_std.copy_(state_std.to(self.state_std).clamp_min(float(min_std)))
        self.target_mean.copy_(target_mean.to(self.target_mean))
        self.target_std.copy_(target_std.to(self.target_std).clamp_min(float(min_std)))

    def normalize_state(self, state: Tensor) -> Tensor:
        return (state - self.state_mean) / self.state_std

    def normalize_target(self, target: Tensor) -> Tensor:
        return (target - self.target_mean) / self.target_std

    def denormalize_target(self, target: Tensor) -> Tensor:
        return target * self.target_std + self.target_mean

    def _patchify(self, target: Tensor) -> Tensor:
        if int(target.shape[-1]) != self.target_dim:
            msg = (
                f"Target width mismatch: expected {self.target_dim}, "
                f"got {target.shape[-1]}."
            )
            raise ValueError(msg)
        if self.padded_target_dim != self.target_dim:
            target = F.pad(target, (0, self.padded_target_dim - self.target_dim))
        return target.reshape(target.shape[0], self.num_patches, self.patch_dim)

    def _unpatchify(self, patches: Tensor) -> Tensor:
        return patches.reshape(patches.shape[0], self.padded_target_dim)[
            :, : self.target_dim
        ].contiguous()

    def _language_tokens(
        self,
        language: Tensor | None,
        *,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor | None:
        if self.language_dim == 0:
            if language is not None and int(language.shape[-1]) != 0:
                msg = "This planner checkpoint does not accept language input."
                raise ValueError(msg)
            return None
        if language is None:
            msg = "This planner checkpoint requires a language embedding."
            raise ValueError(msg)
        expected = (int(batch_size), self.language_dim)
        if tuple(language.shape) != expected:
            msg = f"Language shape {tuple(language.shape)} != {expected}."
            raise ValueError(msg)
        assert self.language_proj is not None
        assert self.language_token_embed is not None
        projected = self.language_proj(language.to(device=device, dtype=dtype)).reshape(
            batch_size,
            self.num_language_tokens,
            self.d_model,
        )
        return projected + self.language_token_embed

    def velocity(
        self,
        state: Tensor,
        x_t: Tensor,
        t: Tensor,
        *,
        language: Tensor | None = None,
    ) -> Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        batch_size = int(state.shape[0])
        time_embed = self.time_mlp(_sinusoidal_embedding(t.reshape(-1), self.d_model))
        state_tokens = self.state_proj(self.normalize_state(state)).reshape(
            batch_size, self.num_state_tokens, self.d_model
        )
        state_tokens = state_tokens + self.state_token_embed + time_embed.unsqueeze(1)
        language_tokens = self._language_tokens(
            language,
            batch_size=batch_size,
            dtype=state_tokens.dtype,
            device=state_tokens.device,
        )
        patch_tokens = self.patch_proj(self._patchify(x_t))
        patch_tokens = (
            patch_tokens
            + self.patch_pos_embed
            + self.term_embed(self.patch_term_ids).unsqueeze(0)
            + time_embed.unsqueeze(1)
        )
        conditioning_tokens = [state_tokens]
        if language_tokens is not None:
            conditioning_tokens.append(language_tokens)
        conditioning_width = sum(int(tokens.shape[1]) for tokens in conditioning_tokens)
        encoded = self.encoder(torch.cat([*conditioning_tokens, patch_tokens], dim=1))
        return self._unpatchify(
            self.patch_head(self.output_norm(encoded[:, conditioning_width:]))
        )

    def flow_matching_loss(
        self,
        state: Tensor,
        target: Tensor,
        *,
        language: Tensor | None = None,
    ) -> Tensor:
        target_norm = self.normalize_target(target)
        noise = torch.randn_like(target_norm)
        t = torch.rand(
            (target_norm.shape[0], 1),
            device=target_norm.device,
            dtype=target_norm.dtype,
        )
        x_t = (1.0 - t) * noise + t * target_norm
        return F.mse_loss(
            self.velocity(state, x_t, t, language=language),
            target_norm - noise,
        )

    def normalized_endpoint_loss(
        self, prediction: Tensor, target: Tensor
    ) -> tuple[Tensor, Tensor]:
        prediction_norm = self.normalize_target(prediction)
        target_norm = self.normalize_target(target)
        mse = F.mse_loss(prediction_norm, target_norm)
        cosine = 1.0 - F.cosine_similarity(prediction_norm, target_norm, dim=-1).mean()
        return mse, cosine

    def forward(
        self,
        state: Tensor,
        *,
        num_inference_steps: int = 16,
        inference_noise_std: float = 0.0,
        language: Tensor | None = None,
    ) -> Tensor:
        steps = max(1, int(num_inference_steps))
        x_t = torch.zeros(
            (state.shape[0], self.target_dim), device=state.device, dtype=state.dtype
        )
        if float(inference_noise_std) > 0.0:
            x_t.normal_().mul_(float(inference_noise_std))
        dt = 1.0 / float(steps)
        for step in range(steps):
            t = torch.full(
                (state.shape[0], 1),
                float(step) / float(steps),
                device=state.device,
                dtype=state.dtype,
            )
            x_t = x_t + dt * self.velocity(
                state,
                x_t,
                t,
                language=language,
            )
        return self.denormalize_target(x_t)

    def config_dict(self) -> dict[str, Any]:
        return {
            "planner_type": self.planner_type,
            "state_dim": self.state_dim,
            "target_dim": self.target_dim,
            "term_widths": list(self.term_widths),
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "feedforward_dim": self.feedforward_dim,
            "patch_dim": self.patch_dim,
            "num_state_tokens": self.num_state_tokens,
            "language_dim": self.language_dim,
            "num_language_tokens": self.num_language_tokens,
            "dropout": self.dropout,
        }


class CausalInterfaceTransformerCategoricalPlanner(nn.Module):
    """Shared causal Transformer with a categorical per-step token head."""

    planner_type = "causal_interface_transformer_categorical"

    def __init__(
        self,
        *,
        state_dim: int,
        token_horizon: int,
        codebook_size: int,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        feedforward_dim: int = 2048,
        num_state_tokens: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.token_horizon = int(token_horizon)
        self.codebook_size = int(codebook_size)
        self.d_model = int(d_model)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.feedforward_dim = int(feedforward_dim)
        self.num_state_tokens = int(num_state_tokens)
        self.dropout = float(dropout)
        if self.state_dim <= 0:
            msg = "state_dim must be positive."
            raise ValueError(msg)
        if self.token_horizon <= 0 or self.codebook_size <= 1:
            msg = "token_horizon must be positive and codebook_size must exceed one."
            raise ValueError(msg)
        if self.d_model <= 0 or self.num_layers <= 0 or self.num_heads <= 0:
            msg = "d_model, num_layers, and num_heads must be positive."
            raise ValueError(msg)
        if self.d_model % self.num_heads != 0:
            msg = "d_model must be divisible by num_heads."
            raise ValueError(msg)
        if self.num_state_tokens <= 0:
            msg = "num_state_tokens must be positive."
            raise ValueError(msg)

        self.register_buffer("state_mean", torch.zeros(self.state_dim))
        self.register_buffer("state_std", torch.ones(self.state_dim))
        self.state_proj = nn.Sequential(
            nn.Linear(self.state_dim, self.d_model * self.num_state_tokens),
            nn.Mish(),
            nn.LayerNorm(self.d_model * self.num_state_tokens),
        )
        self.state_token_embed = nn.Parameter(
            torch.zeros(1, self.num_state_tokens, self.d_model)
        )
        self.token_query_embed = nn.Parameter(
            torch.zeros(1, self.token_horizon, self.d_model)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.feedforward_dim,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=self.num_layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(self.d_model)
        self.token_head = nn.Linear(self.d_model, self.codebook_size)
        nn.init.normal_(self.state_token_embed, std=0.02)
        nn.init.normal_(self.token_query_embed, std=0.02)

    @classmethod
    def from_config(
        cls, config: dict[str, Any]
    ) -> CausalInterfaceTransformerCategoricalPlanner:
        return cls(
            state_dim=int(config["state_dim"]),
            token_horizon=int(config["token_horizon"]),
            codebook_size=int(config["codebook_size"]),
            d_model=int(config.get("d_model", 512)),
            num_layers=int(config.get("num_layers", 6)),
            num_heads=int(config.get("num_heads", 8)),
            feedforward_dim=int(config.get("feedforward_dim", 2048)),
            num_state_tokens=int(config.get("num_state_tokens", 4)),
            dropout=float(config.get("dropout", 0.0)),
        )

    @torch.no_grad()
    def set_state_normalization(
        self,
        *,
        state_mean: Tensor,
        state_std: Tensor,
        min_std: float = 1.0e-4,
    ) -> None:
        for name, value in (("state_mean", state_mean), ("state_std", state_std)):
            if tuple(value.shape) != (self.state_dim,):
                msg = f"{name} shape {tuple(value.shape)} != {(self.state_dim,)}."
                raise ValueError(msg)
        self.state_mean.copy_(state_mean.to(self.state_mean))
        self.state_std.copy_(state_std.to(self.state_std).clamp_min(float(min_std)))

    def normalize_state(self, state: Tensor) -> Tensor:
        return (state - self.state_mean) / self.state_std

    def logits(self, state: Tensor) -> Tensor:
        batch_size = int(state.shape[0])
        state_tokens = self.state_proj(self.normalize_state(state)).reshape(
            batch_size,
            self.num_state_tokens,
            self.d_model,
        )
        state_tokens = state_tokens + self.state_token_embed
        queries = self.token_query_embed.expand(batch_size, -1, -1)
        encoded = self.encoder(torch.cat((state_tokens, queries), dim=1))
        token_features = encoded[:, self.num_state_tokens :]
        return self.token_head(self.output_norm(token_features))

    def categorical_loss(self, state: Tensor, token_ids: Tensor) -> Tensor:
        if tuple(token_ids.shape[1:]) != (self.token_horizon,):
            msg = (
                f"Token target shape {tuple(token_ids.shape)} must be "
                f"(batch, {self.token_horizon})."
            )
            raise ValueError(msg)
        return F.cross_entropy(
            self.logits(state).reshape(-1, self.codebook_size),
            token_ids.to(torch.long).reshape(-1),
        )

    def forward(self, state: Tensor) -> Tensor:
        return self.logits(state).argmax(dim=-1)

    def config_dict(self) -> dict[str, Any]:
        return {
            "planner_type": self.planner_type,
            "state_dim": self.state_dim,
            "token_horizon": self.token_horizon,
            "codebook_size": self.codebook_size,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "feedforward_dim": self.feedforward_dim,
            "num_state_tokens": self.num_state_tokens,
            "dropout": self.dropout,
        }


class FrozenCategoricalTokenPlannerSampler:
    """Consume a predicted token packet one low-level control step at a time."""

    def __init__(
        self,
        *,
        env: object,
        checkpoint_path: str | Path,
        decode_token_ids: Callable[[Tensor], Tensor],
        latent_dim: int,
        discover_env_method: Callable[[object, str], Callable[..., Any] | None],
        device: torch.device | str,
    ) -> None:
        self.device = torch.device(device)
        self.latent_dim = int(latent_dim)
        if self.latent_dim <= 0:
            msg = "latent_dim must be positive."
            raise ValueError(msg)
        self.decode_token_ids = decode_token_ids
        checkpoint = torch.load(
            Path(checkpoint_path).expanduser(),
            map_location=self.device,
            weights_only=False,
        )
        planner_config = dict(checkpoint.get("planner_config", {}))
        if (
            planner_config.get("planner_type")
            != CausalInterfaceTransformerCategoricalPlanner.planner_type
        ):
            msg = "Token planner checkpoint is not a categorical causal planner."
            raise ValueError(msg)
        target_spec = checkpoint.get("target_spec", {})
        if (
            not isinstance(target_spec, Mapping)
            or target_spec.get("interface") != "per_step_token_sequence"
        ):
            msg = "Token planner checkpoint must target per_step_token_sequence."
            raise ValueError(msg)
        metadata = checkpoint.get("metadata", {})
        if not isinstance(metadata, Mapping):
            metadata = {}
        sample_metadata = metadata.get("sample_metadata", {})
        if not isinstance(sample_metadata, Mapping):
            sample_metadata = {}
        encoding = metadata.get(
            "target_encoding", sample_metadata.get("target_encoding", {})
        )
        if (
            not isinstance(encoding, Mapping)
            or encoding.get("kind") != "categorical_sequence"
        ):
            msg = "Token planner checkpoint lacks categorical target encoding."
            raise ValueError(msg)
        self.token_horizon = int(encoding.get("horizon", -1))
        self.codebook_size = int(encoding.get("codebook_size", -1))
        self.state_history_steps = int(sample_metadata.get("state_history_steps", 0))
        self.planner_observation_spec = metadata.get(
            "planner_observation_spec",
            sample_metadata.get("planner_observation_spec"),
        )
        if not isinstance(self.planner_observation_spec, Mapping):
            msg = "Token planner checkpoint lacks planner_observation_spec."
            raise ValueError(msg)

        self.planner = CausalInterfaceTransformerCategoricalPlanner.from_config(
            planner_config
        ).to(self.device)
        self.planner.load_state_dict(checkpoint["planner_state_dict"])
        self.planner.eval()
        self.planner.requires_grad_(False)
        if self.planner.token_horizon != self.token_horizon:
            msg = "Token planner horizon does not match target encoding metadata."
            raise ValueError(msg)
        if self.planner.codebook_size != self.codebook_size:
            msg = "Token planner codebook size does not match target metadata."
            raise ValueError(msg)

        self._causal_observation = discover_env_method(
            env, "current_causal_planner_observation"
        )
        self._observation_spec = discover_env_method(
            env, "causal_planner_observation_spec"
        )
        if self._causal_observation is None or self._observation_spec is None:
            msg = (
                "command_source='token_planner' requires causal planner "
                "observation methods from the environment."
            )
            raise ValueError(msg)
        runtime_spec = dict(
            self._observation_spec(history_steps=self.state_history_steps)
        )
        if dict(self.planner_observation_spec) != runtime_spec:
            msg = (
                "Token planner observation specification does not match the "
                f"environment: {dict(self.planner_observation_spec)} != {runtime_spec}."
            )
            raise ValueError(msg)

        self._packet: Tensor | None = None
        self._cursor: Tensor | None = None

    def reset(self) -> None:
        """Discard packets decoded before a low-level checkpoint reload or reset."""
        self._packet = None
        self._cursor = None

    @staticmethod
    def _done_mask(td: Any, *, batch_size: int, device: torch.device) -> Tensor:
        result = torch.zeros(batch_size, device=device, dtype=torch.bool)
        keys = td.keys(True)
        for key in (
            "done",
            "terminated",
            "truncated",
            "is_init",
            ("next", "done"),
            ("next", "terminated"),
            ("next", "truncated"),
        ):
            if key not in keys:
                continue
            value = cast(Tensor, td.get(key)).reshape(-1).to(device=device).bool()
            if value.numel() == batch_size:
                result |= value
        return result

    def _planner_state(self, env_ids: Tensor) -> Tensor:
        assert self._causal_observation is not None
        batch = self._causal_observation(
            env_ids=env_ids,
            history_steps=self.state_history_steps,
        )
        state = batch.get(("planner", "state_history"))
        if self.state_history_steps <= 0:
            state = batch.get(("planner", "state"))
        if state is None:
            msg = "Causal observation batch lacks the configured planner state."
            raise ValueError(msg)
        state = cast(Tensor, state).to(device=self.device, dtype=torch.float32)
        return state.reshape(env_ids.numel(), -1).contiguous()

    @torch.no_grad()
    def sample_for_step(
        self,
        td: Any,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        batch_size = int(td.numel())
        if batch_size <= 0:
            return torch.empty(0, self.latent_dim, device=device, dtype=dtype)
        if (
            self._packet is None
            or self._packet.shape != (batch_size, self.token_horizon, self.latent_dim)
            or self._packet.device != self.device
        ):
            self._packet = torch.zeros(
                batch_size,
                self.token_horizon,
                self.latent_dim,
                device=self.device,
            )
            self._cursor = torch.full(
                (batch_size,),
                self.token_horizon,
                device=self.device,
                dtype=torch.long,
            )
        assert self._cursor is not None
        done = self._done_mask(td, batch_size=batch_size, device=self.device)
        renew = done | (self._cursor >= self.token_horizon)
        if bool(renew.any()):
            env_ids = renew.nonzero(as_tuple=False).reshape(-1)
            token_ids = self.planner(self._planner_state(env_ids))
            decoded = self.decode_token_ids(token_ids).to(self.device)
            expected = (env_ids.numel(), self.token_horizon, self.latent_dim)
            if tuple(decoded.shape) != expected:
                msg = (
                    f"Decoded token packet shape {tuple(decoded.shape)} != {expected}."
                )
                raise ValueError(msg)
            self._packet[env_ids] = decoded
            self._cursor[env_ids] = 0
        env_index = torch.arange(batch_size, device=self.device)
        command = self._packet[env_index, self._cursor]
        self._cursor += 1
        return command.to(device=device, dtype=dtype)


class FrozenContinuousInterfacePlannerSampler:
    """Hold a continuous causal-planner command for its low-level horizon."""

    def __init__(
        self,
        *,
        env: object,
        checkpoint_path: str | Path,
        latent_dim: int,
        discover_env_method: Callable[[object, str], Callable[..., Any] | None],
        device: torch.device | str,
    ) -> None:
        self.device = torch.device(device)
        self.latent_dim = int(latent_dim)
        checkpoint = torch.load(
            Path(checkpoint_path).expanduser(),
            map_location=self.device,
            weights_only=False,
        )
        config = dict(checkpoint.get("planner_config", {}))
        if (
            config.get("planner_type")
            != CausalInterfaceTransformerFlowPlanner.planner_type
        ):
            msg = "Continuous planner checkpoint is not the shared Transformer-flow model."
            raise ValueError(msg)
        target_spec = checkpoint.get("target_spec", {})
        if not isinstance(target_spec, Mapping) or target_spec.get("interface") not in {
            "future_cvae",
        }:
            msg = "Continuous planner checkpoint must target future_cvae."
            raise ValueError(msg)
        self.planner = CausalInterfaceTransformerFlowPlanner.from_config(config).to(
            self.device
        )
        self.planner.load_state_dict(checkpoint["planner_state_dict"])
        self.planner.eval()
        self.planner.requires_grad_(False)
        if self.planner.target_dim != self.latent_dim:
            msg = (
                f"Planner target width {self.planner.target_dim} does not match "
                f"latent_dim={self.latent_dim}."
            )
            raise ValueError(msg)

        metadata = checkpoint.get("metadata", {})
        if not isinstance(metadata, Mapping):
            metadata = {}
        sample_metadata = metadata.get("sample_metadata", {})
        if not isinstance(sample_metadata, Mapping):
            sample_metadata = {}
        self.state_history_steps = int(sample_metadata.get("state_history_steps", 0))
        self.command_period = int(
            sample_metadata.get(
                "planner_interval_steps", metadata.get("planner_interval_steps", 10)
            )
        )
        if self.command_period <= 0:
            msg = "Continuous planner command period must be positive."
            raise ValueError(msg)
        self.num_inference_steps = int(metadata.get("flow_num_inference_steps", 16))
        self.inference_noise_std = float(metadata.get("flow_inference_noise_std", 0.0))
        self.planner_observation_spec = metadata.get(
            "planner_observation_spec",
            sample_metadata.get("planner_observation_spec"),
        )
        if not isinstance(self.planner_observation_spec, Mapping):
            msg = "Continuous planner checkpoint lacks planner_observation_spec."
            raise ValueError(msg)
        self._causal_observation = discover_env_method(
            env, "current_causal_planner_observation"
        )
        self._observation_spec = discover_env_method(
            env, "causal_planner_observation_spec"
        )
        if self._causal_observation is None or self._observation_spec is None:
            msg = "continuous_planner requires causal observation environment methods."
            raise ValueError(msg)
        runtime_spec = dict(
            self._observation_spec(history_steps=self.state_history_steps)
        )
        if dict(self.planner_observation_spec) != runtime_spec:
            msg = "Continuous planner observation specification mismatch."
            raise ValueError(msg)
        self._command: Tensor | None = None
        self._steps: Tensor | None = None

    def reset(self) -> None:
        self._command = None
        self._steps = None

    @torch.no_grad()
    def sample_for_step(
        self,
        td: Any,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        batch_size = int(td.numel())
        if batch_size <= 0:
            return torch.empty(0, self.latent_dim, device=device, dtype=dtype)
        if (
            self._command is None
            or self._command.shape != (batch_size, self.latent_dim)
            or self._command.device != self.device
        ):
            self._command = torch.zeros(batch_size, self.latent_dim, device=self.device)
            self._steps = torch.zeros(batch_size, device=self.device, dtype=torch.long)
        assert self._steps is not None
        done = FrozenCategoricalTokenPlannerSampler._done_mask(
            td, batch_size=batch_size, device=self.device
        )
        renew = done | (self._steps <= 0)
        if bool(renew.any()):
            env_ids = renew.nonzero(as_tuple=False).reshape(-1)
            assert self._causal_observation is not None
            batch = self._causal_observation(
                env_ids=env_ids,
                history_steps=self.state_history_steps,
            )
            state = batch.get(("planner", "state_history"))
            if self.state_history_steps <= 0:
                state = batch.get(("planner", "state"))
            if state is None:
                msg = "Causal observation batch lacks the configured planner state."
                raise ValueError(msg)
            state = cast(Tensor, state).to(self.device, dtype=torch.float32)
            state = state.reshape(env_ids.numel(), -1)
            self._command[env_ids] = self.planner(
                state,
                num_inference_steps=self.num_inference_steps,
                inference_noise_std=self.inference_noise_std,
            )
            self._steps[env_ids] = self.command_period
        command = self._command.clone()
        self._steps -= 1
        return command.to(device=device, dtype=dtype)
