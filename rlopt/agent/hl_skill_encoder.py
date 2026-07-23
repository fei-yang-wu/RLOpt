"""High-level skill encoders: a shared trunk + pluggable latent bottleneck.

``HighLevelSkillEncoder`` is the abstract base (MLP trunk over ``[state ; future
window]``); each latent method is a small subclass. Discrete methods share a
single ``code_to_latent`` step -- the codebook *is* that map -- so they differ
only in how the discrete code is selected. The FSQ/VQ/Gumbel quantizers are
ported (trimmed) from ``rlopt.agent.imitation.latent_learning``.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

LATENT_MODES = (
    "deterministic",
    "gaussian",
    "categorical",
    "gumbel_multicat",
    "gumbel",
    "fsq",
    "vq",
)


# --------------------------------------------------------------------------- #
# Quantizers (ported from latent_learning.py, trimmed to essentials).
# --------------------------------------------------------------------------- #
class FSQQuantizer(nn.Module):
    """Finite Scalar Quantization: per-dim bounded rounding. No params, no collapse."""

    def __init__(self, levels: tuple[int, ...]) -> None:
        super().__init__()
        levels_i = torch.tensor([int(level) for level in levels], dtype=torch.long)
        if levels_i.numel() == 0 or int(levels_i.min()) < 2:
            msg = f"FSQ levels must each be >= 2, got {list(levels)!r}."
            raise ValueError(msg)
        self.register_buffer("_levels", levels_i)
        # Exact Python-int product: torch int64 prod/cumprod silently wrap for
        # large level sets (e.g. SONIC's 64 dims x 32 levels = 2^320).
        self._codebook_size = math.prod(int(level) for level in levels)
        # A flat scalar code index only exists when the product fits int64;
        # otherwise forward() returns per-dim level indices instead (the same
        # pooled-code convention the grouped categorical encoder uses).
        self.flat_code_supported = self._codebook_size <= (1 << 62)
        basis = torch.ones(levels_i.numel(), dtype=torch.long)
        if self.flat_code_supported:
            basis = torch.cumprod(
                torch.cat([torch.ones(1, dtype=torch.long), levels_i[:-1]]), dim=0
            )
        self.register_buffer("_basis", basis)
        self.code_dim = int(levels_i.numel())

    @property
    def codebook_size(self) -> int:
        return self._codebook_size

    def _bound(self, z: Tensor) -> Tensor:
        levels = self._levels.to(z.dtype)
        half = (levels - 1) * 0.5
        offset = (self._levels % 2 == 0).to(z.dtype) * 0.5
        shift = torch.atanh(offset / half.clamp(min=1.0))
        return (z + shift).tanh() * half - offset

    def forward(self, z_e: Tensor) -> tuple[Tensor, Tensor]:
        bounded = self._bound(z_e)
        rounded = torch.round(bounded)
        z_q = bounded + (rounded - bounded).detach()  # straight-through
        zhat = (rounded.long() + self._levels // 2).clamp(min=0)
        zhat = torch.minimum(zhat, self._levels - 1)
        if not self.flat_code_supported:
            return z_q, zhat
        code = (zhat * self._basis).sum(dim=-1)
        return z_q, code


class EMAVQQuantizer(nn.Module):
    """VQ-VAE quantizer: EMA codebook + commitment loss, with dead-code revival."""

    def __init__(
        self,
        codebook_size: int,
        embed_dim: int,
        *,
        decay: float = 0.99,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.embed_dim = int(embed_dim)
        self.decay, self.eps = float(decay), float(eps)
        embedding = torch.randn(self.codebook_size, self.embed_dim) * 0.01
        self.register_buffer("embedding", embedding)
        self.register_buffer("cluster_size", torch.zeros(self.codebook_size))
        self.register_buffer("cluster_sum", embedding.clone())
        self.register_buffer("_inited", torch.tensor(False))

    @torch.no_grad()
    def _kmeans_init(self, z_e: Tensor) -> None:
        flat = z_e.reshape(-1, self.embed_dim)
        if flat.shape[0] == 0:
            return
        idx = torch.randint(0, flat.shape[0], (self.codebook_size,), device=flat.device)
        self.embedding.copy_(flat[idx])
        self.cluster_sum.copy_(self.embedding)
        self.cluster_size.fill_(1.0)
        self._inited.fill_(True)

    def _quantize(self, z_e: Tensor) -> tuple[Tensor, Tensor]:
        flat = z_e.reshape(-1, self.embed_dim)
        dist = (
            flat.pow(2).sum(-1, keepdim=True)
            - 2.0 * flat @ self.embedding.t()
            + self.embedding.pow(2).sum(-1)
        )
        code = dist.argmin(dim=-1)
        z_q = F.embedding(code, self.embedding).reshape_as(z_e)
        return z_q, code.reshape(z_e.shape[:-1])

    def forward(self, z_e: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if self.training and not bool(self._inited.item()):
            self._kmeans_init(z_e.detach())
        z_q, code = self._quantize(z_e)
        commitment = F.mse_loss(z_e, z_q.detach())
        z_q_st = z_e + (z_q - z_e).detach()  # straight-through
        if self.training:
            with torch.no_grad():
                flat = z_e.detach().reshape(-1, self.embed_dim)
                onehot = F.one_hot(code.reshape(-1), self.codebook_size).to(flat.dtype)
                self.cluster_size.mul_(self.decay).add_(
                    onehot.sum(0), alpha=1.0 - self.decay
                )
                self.cluster_sum.mul_(self.decay).add_(
                    onehot.t() @ flat, alpha=1.0 - self.decay
                )
                n = self.cluster_size.sum()
                smoothed = (
                    (self.cluster_size + self.eps)
                    / (n + self.codebook_size * self.eps)
                    * n
                )
                self.embedding.copy_(self.cluster_sum / smoothed.unsqueeze(-1))
        return z_q_st, code, commitment

    @torch.no_grad()
    def revive_dead_codes(self, z_e: Tensor, *, threshold: float = 1e-3) -> int:
        flat = z_e.detach().reshape(-1, self.embed_dim)
        dead = self.cluster_size < threshold
        n_dead = int(dead.sum().item())
        if n_dead and flat.shape[0]:
            idx = torch.randint(0, flat.shape[0], (n_dead,), device=flat.device)
            self.embedding[dead] = flat[idx]
            self.cluster_sum[dead] = self.embedding[dead]
            self.cluster_size[dead] = 1.0
        return n_dead


class GumbelQuantizer(nn.Module):
    """Categorical bottleneck via Gumbel-Softmax (annealed tau)."""

    def __init__(self, input_dim: int, codebook_size: int, embed_dim: int) -> None:
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.logit_head = nn.Linear(input_dim, self.codebook_size)
        self.codebook = nn.Embedding(self.codebook_size, embed_dim)
        nn.init.normal_(self.codebook.weight, std=0.02)

    def forward(
        self, z_e: Tensor, *, tau: float, hard: bool, deterministic: bool
    ) -> tuple[Tensor, Tensor, Tensor]:
        logits = self.logit_head(z_e)
        if deterministic:
            y = F.one_hot(logits.argmax(-1), self.codebook_size).to(logits.dtype)
        else:
            y = F.gumbel_softmax(logits, tau=max(float(tau), 1e-3), hard=hard, dim=-1)
        z_q = y @ self.codebook.weight
        log_q = F.log_softmax(logits, dim=-1)
        kl = (log_q.exp() * log_q).sum(-1).mean() + math.log(self.codebook_size)
        return z_q, logits.argmax(-1), kl


def _perplexity(code: Tensor, num_codes: int) -> Tensor:
    counts = torch.bincount(code.reshape(-1), minlength=num_codes).float()
    probs = counts / counts.sum().clamp(min=1.0)
    return torch.exp(-(probs * probs.clamp_min(1e-12).log()).sum())


def _effective_rank(z: Tensor) -> Tensor:
    """Participation-ratio effective rank of ``z`` [B, D] as a 0-dim tensor."""
    if z.shape[0] < 2:
        return z.new_zeros(())
    sv = torch.linalg.svdvals(z - z.mean(0, keepdim=True))
    total = sv.sum()
    if bool((total <= 1e-12).item()):
        return z.new_zeros(())
    probs = sv / total
    return torch.exp(-(probs * probs.clamp_min(1e-12).log()).sum())


def _anneal_tau(step: int | None, start: float, end: float, iters: int) -> float:
    """Linearly anneal Gumbel temperature from ``start`` to ``end`` over ``iters``."""
    if iters <= 0 or step is None:
        return end
    progress = min(1.0, float(step) / iters)
    return start + progress * (end - start)


# --------------------------------------------------------------------------- #
# Encoders.
# --------------------------------------------------------------------------- #
class HighLevelSkillEncoder(nn.Module, ABC):
    """Shared MLP trunk over ``[state ; future_window]`` + a latent head.

    ``encode`` returns ``(z [B, z_dim], reg_loss scalar, info dict)``. ``reg_loss``
    is the latent regularizer, weighted by a single ``reg_coeff``: L2 on z
    (deterministic), KL (gaussian / categorical / gumbel), commitment (vq), 0 (fsq).
    """

    def __init__(
        self,
        *,
        state_dim: int,
        window_steps: int,
        z_dim: int,
        hidden_dims: tuple[int, ...],
        raw_dim: int,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.window_steps = int(window_steps)
        self.z_dim = int(z_dim)
        layers: list[nn.Module] = []
        prev = self.state_dim * (self.window_steps + 1)
        for hidden in hidden_dims:
            layers += [
                nn.Linear(prev, int(hidden)),
                nn.LayerNorm(int(hidden)),
                nn.Mish(),
            ]
            prev = int(hidden)
        layers.append(nn.Linear(prev, int(raw_dim)))
        self.net = nn.Sequential(*layers)

    def _raw(self, state: Tensor, future_window: Tensor) -> Tensor:
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
                f"state/future_window shape mismatch: expected state [B, {self.state_dim}] "
                f"and future_window {expected_window}, got {tuple(state.shape)} and "
                f"{tuple(future_window.shape)}."
            )
            raise ValueError(msg)
        flat_window = future_window.reshape(batch_size, self.window_steps * state_dim)
        return self.net(torch.cat([state, flat_window], dim=-1))

    @abstractmethod
    def _latent(
        self, raw: Tensor, *, deterministic: bool, step: int | None
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]: ...

    def encode(
        self,
        state: Tensor,
        future_window: Tensor,
        *,
        deterministic: bool = False,
        step: int | None = None,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        return self._latent(
            self._raw(state, future_window), deterministic=deterministic, step=step
        )

    def forward(self, state: Tensor, future_window: Tensor) -> Tensor:
        return self.encode(state, future_window, deterministic=True)[0]

    def on_after_train_step(self, step: int) -> None:
        """Called after each optimizer step (VQ overrides for dead-code revival)."""

    def _z_diversity(self, z: Tensor) -> dict[str, Tensor]:
        """Universal continuous collapse signal: rank + per-dim spread of ``z``."""
        std = z.std(dim=0, unbiased=False)
        return {
            "effective_rank": _effective_rank(z),
            "z_std_mean": std.mean(),
            "z_std_min": std.min(),
        }

    @torch.no_grad()
    def diversity_metrics(
        self, state: Tensor, future_window: Tensor
    ) -> dict[str, Tensor]:
        """Per-method diversity / collapse diagnostics (eval calls this)."""
        return self._z_diversity(self.forward(state, future_window))


class DeterministicSkillEncoder(HighLevelSkillEncoder):
    def __init__(self, **base) -> None:
        super().__init__(raw_dim=base["z_dim"], **base)

    def _latent(self, raw, **_):
        return raw, raw.pow(2).mean(), {}


class GaussianSkillEncoder(HighLevelSkillEncoder):
    def __init__(
        self, *, logstd_min: float = -5.0, logstd_max: float = 2.0, **base
    ) -> None:
        super().__init__(raw_dim=2 * base["z_dim"], **base)
        self.logstd_min, self.logstd_max = float(logstd_min), float(logstd_max)

    def _latent(self, raw, *, deterministic, **_):
        mean, logstd = raw.chunk(2, dim=-1)
        logstd = logstd.clamp(self.logstd_min, self.logstd_max)
        std = logstd.exp()
        z = mean if deterministic else mean + std * torch.randn_like(std)
        kl = 0.5 * (mean.pow(2) + std.pow(2) - 1.0 - 2.0 * logstd).sum(-1).mean()
        return z, kl, {}

    @torch.no_grad()
    def diversity_metrics(self, state, future_window):
        mean, logstd = self._raw(state, future_window).chunk(2, dim=-1)
        logstd = logstd.clamp(self.logstd_min, self.logstd_max)
        kl_per_dim = 0.5 * (mean.pow(2) + (2 * logstd).exp() - 1.0 - 2 * logstd).mean(0)
        metrics = self._z_diversity(mean)
        metrics["kl_per_dim_mean"] = kl_per_dim.mean()
        # Active units: latent dims that carry information (posterior collapse -> 0).
        metrics["active_units"] = (kl_per_dim > 1e-2).float().sum()
        metrics["post_std_mean"] = logstd.exp().mean()
        return metrics


class _DiscreteSkillEncoder(HighLevelSkillEncoder):
    """Shared discrete flow: trunk -> quantize -> code_to_latent -> z."""

    code_to_latent: nn.Module
    num_codes: int = 0

    def _pre_quantize(self, raw: Tensor) -> Tensor:
        return raw

    @abstractmethod
    def _quantize(
        self, z_e: Tensor, *, deterministic: bool, step: int | None
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        """Return ``(z_q, code, reg_loss, info)``."""

    def _latent(self, raw, *, deterministic, step):
        z_e = self._pre_quantize(raw)
        z_q, code, reg, info = self._quantize(
            z_e, deterministic=deterministic, step=step
        )
        z = self.code_to_latent(z_q).reshape(*raw.shape[:-1], self.z_dim)
        if 0 < self.num_codes <= 1 << 16:
            info = {**info, "perplexity": _perplexity(code.detach(), self.num_codes)}
        return z, reg, info

    @torch.no_grad()
    def diversity_metrics(self, state, future_window):
        raw = self._raw(state, future_window)
        z_e = self._pre_quantize(raw)
        z_q, code, _, _ = self._quantize(z_e, deterministic=True, step=None)
        z = self.code_to_latent(z_q).reshape(*raw.shape[:-1], self.z_dim)
        metrics = self._z_diversity(z)
        # Codebook collapse: effective # of codes used / fraction of the codebook.
        metrics["code_perplexity"] = _perplexity(code, self.num_codes)
        used = float(code.reshape(-1).unique().numel())
        metrics["code_usage_frac"] = z.new_tensor(used / self.num_codes)
        return metrics


class MultiCategoricalSkillEncoder(_DiscreteSkillEncoder):
    def __init__(self, *, groups: int = 8, categories: int = 32, **base) -> None:
        groups, categories = int(groups), int(categories)
        if base["z_dim"] % groups != 0:
            msg = (
                "categorical latent requires z_dim divisible by groups (code dim = "
                f"z_dim // groups): z_dim={base['z_dim']}, groups={groups}."
            )
            raise ValueError(msg)
        super().__init__(raw_dim=groups * categories, **base)
        self.groups, self.categories = groups, categories
        self.num_codes = categories
        self.code_to_latent = nn.Identity()
        code_dim = base["z_dim"] // groups
        # Per-group codebook [G, C, code_dim]; ~unit-norm codes so z scale ~ one-hot.
        self.codebook = nn.Parameter(
            torch.randn(groups, categories, code_dim) / math.sqrt(code_dim)
        )

    def _pre_quantize(self, raw):
        return raw.reshape(*raw.shape[:-1], self.groups, self.categories)  # logits

    def _quantize(self, logits, *, deterministic, **_):
        dist = torch.distributions.Independent(
            torch.distributions.OneHotCategoricalStraightThrough(logits=logits), 1
        )
        if deterministic:
            idx = logits.argmax(dim=-1)
            group_index = torch.arange(self.groups, device=idx.device)
            z_q = self.codebook[group_index, idx]
        else:
            z_q = torch.einsum("...gc,gcb->...gb", dist.rsample(), self.codebook)
        probs = dist.base_dist.probs
        log_probs = torch.log_softmax(logits, dim=-1)
        kl = (probs * (log_probs + math.log(self.categories))).sum(-1).sum(-1).mean()
        return z_q, logits.argmax(dim=-1), kl, {}


class GumbelMultiCategoricalSkillEncoder(MultiCategoricalSkillEncoder):
    """Product codebook trained with per-group Gumbel-softmax (annealed temperature).

    Same capacity and inference as ``MultiCategoricalSkillEncoder`` (G groups x C
    categories, per-group codebook, deterministic = per-group argmax), but the
    training sampler is Gumbel-softmax instead of the hard straight-through one-hot:
    the soft, temperature-annealed relaxation keeps gradients flowing to all logits
    and codes, avoiding the premature commitment / collapse of the ST sampler.
    """

    def __init__(
        self,
        *,
        groups: int = 8,
        categories: int = 32,
        tau_start: float = 2.0,
        tau_end: float = 0.5,
        tau_anneal_iters: int = 2000,
        hard: bool = True,
        **base,
    ) -> None:
        super().__init__(groups=groups, categories=categories, **base)
        self.tau_start, self.tau_end = float(tau_start), float(tau_end)
        self.tau_anneal_iters, self.hard = int(tau_anneal_iters), bool(hard)

    def _quantize(self, logits, *, deterministic, step):
        log_probs = torch.log_softmax(logits, dim=-1)
        kl = (
            (log_probs.exp() * (log_probs + math.log(self.categories)))
            .sum(-1)
            .sum(-1)
            .mean()
        )
        if deterministic:
            idx = logits.argmax(dim=-1)
            group_index = torch.arange(self.groups, device=idx.device)
            return self.codebook[group_index, idx], idx, kl, {}
        tau = _anneal_tau(step, self.tau_start, self.tau_end, self.tau_anneal_iters)
        weights = F.gumbel_softmax(logits, tau=max(tau, 1e-3), hard=self.hard, dim=-1)
        z_q = torch.einsum("...gc,gcb->...gb", weights, self.codebook)
        return z_q, logits.argmax(dim=-1), kl, {"tau": logits.new_tensor(tau)}


class GumbelSkillEncoder(_DiscreteSkillEncoder):
    def __init__(
        self,
        *,
        codebook_size: int = 512,
        tau_start: float = 2.0,
        tau_end: float = 0.5,
        tau_anneal_iters: int = 2000,
        hard: bool = True,
        **base,
    ) -> None:
        super().__init__(raw_dim=base["z_dim"], **base)
        self.num_codes = int(codebook_size)
        self.code_to_latent = nn.Identity()
        self.gumbel = GumbelQuantizer(base["z_dim"], codebook_size, base["z_dim"])
        self.tau_start, self.tau_end = float(tau_start), float(tau_end)
        self.tau_anneal_iters, self.hard = int(tau_anneal_iters), bool(hard)

    def _quantize(self, z_e, *, deterministic, step):
        tau = _anneal_tau(step, self.tau_start, self.tau_end, self.tau_anneal_iters)
        z_q, code, kl = self.gumbel(
            z_e, tau=tau, hard=self.hard, deterministic=deterministic
        )
        return z_q, code, kl, {"tau": torch.tensor(tau, device=z_e.device)}


class VQSkillEncoder(_DiscreteSkillEncoder):
    def __init__(
        self,
        *,
        codebook_size: int = 512,
        ema_decay: float = 0.99,
        dead_code_reset_iters: int = 0,
        **base,
    ) -> None:
        super().__init__(raw_dim=base["z_dim"], **base)
        self.num_codes = int(codebook_size)
        self.code_to_latent = nn.Identity()
        self.vq = EMAVQQuantizer(codebook_size, base["z_dim"], decay=ema_decay)
        self.dead_code_reset_iters = int(dead_code_reset_iters)
        self._last_z_e: Tensor | None = None

    def _quantize(self, z_e, **_):
        self._last_z_e = z_e.detach()  # nearest-neighbor; deterministic/step unused
        z_q, code, commitment = self.vq(z_e)
        return z_q, code, commitment, {}

    def on_after_train_step(self, step: int) -> None:
        if (
            self.dead_code_reset_iters > 0
            and step > 0
            and step % self.dead_code_reset_iters == 0
            and self._last_z_e is not None
        ):
            self.vq.revive_dead_codes(self._last_z_e)

    @torch.no_grad()
    def diversity_metrics(self, state, future_window):
        metrics = super().diversity_metrics(state, future_window)
        # EMA cluster usage gives a global (not per-batch) dead-code estimate.
        metrics["dead_code_frac"] = (self.vq.cluster_size < 1e-3).float().mean()
        return metrics


class FSQSkillEncoder(_DiscreteSkillEncoder):
    def __init__(self, *, levels: tuple[int, ...] = (8, 8, 8, 5, 5), **base) -> None:
        super().__init__(raw_dim=len(levels), **base)
        self.fsq = FSQQuantizer(levels)
        # When the flat code index does not fit int64, per-dim level indices
        # are the codes (pooled across dims by the usage metrics), so the
        # usage-metric vocabulary is the largest per-dim level count.
        self.num_codes = (
            self.fsq.codebook_size
            if self.fsq.flat_code_supported
            else max(int(level) for level in levels)
        )
        self.code_to_latent = nn.Linear(self.fsq.code_dim, base["z_dim"])

    def _quantize(self, z_e, **_):
        z_q, code = self.fsq(z_e)
        return z_q, code, z_e.new_zeros(()), {}


# --------------------------------------------------------------------------- #
# Spec + factory.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SkillLatentSpec:
    """Latent hyperparameters; separate from the trainer config to avoid a cycle."""

    latent_mode: str = "deterministic"
    gaussian_logstd_min: float = -5.0
    gaussian_logstd_max: float = 2.0
    categorical_groups: int = 8
    categorical_categories: int = 32
    gumbel_codebook_size: int = 512
    gumbel_tau_start: float = 2.0
    gumbel_tau_end: float = 0.5
    gumbel_tau_anneal_iters: int = 2000
    gumbel_hard: bool = True
    fsq_levels: tuple[int, ...] = (8, 8, 8, 5, 5)
    vq_codebook_size: int = 512
    vq_ema_decay: float = 0.99
    vq_dead_code_reset_iters: int = 0


def build_skill_encoder(
    *,
    state_dim: int,
    window_steps: int,
    z_dim: int,
    hidden_dims: tuple[int, ...],
    spec: SkillLatentSpec,
) -> HighLevelSkillEncoder:
    base = {
        "state_dim": state_dim,
        "window_steps": window_steps,
        "z_dim": z_dim,
        "hidden_dims": hidden_dims,
    }
    mode = spec.latent_mode
    if mode == "deterministic":
        return DeterministicSkillEncoder(**base)
    if mode == "gaussian":
        return GaussianSkillEncoder(
            logstd_min=spec.gaussian_logstd_min,
            logstd_max=spec.gaussian_logstd_max,
            **base,
        )
    if mode == "categorical":
        return MultiCategoricalSkillEncoder(
            groups=spec.categorical_groups,
            categories=spec.categorical_categories,
            **base,
        )
    if mode == "gumbel_multicat":
        return GumbelMultiCategoricalSkillEncoder(
            groups=spec.categorical_groups,
            categories=spec.categorical_categories,
            tau_start=spec.gumbel_tau_start,
            tau_end=spec.gumbel_tau_end,
            tau_anneal_iters=spec.gumbel_tau_anneal_iters,
            hard=spec.gumbel_hard,
            **base,
        )
    if mode == "gumbel":
        return GumbelSkillEncoder(
            codebook_size=spec.gumbel_codebook_size,
            tau_start=spec.gumbel_tau_start,
            tau_end=spec.gumbel_tau_end,
            tau_anneal_iters=spec.gumbel_tau_anneal_iters,
            hard=spec.gumbel_hard,
            **base,
        )
    if mode == "fsq":
        return FSQSkillEncoder(levels=tuple(spec.fsq_levels), **base)
    if mode == "vq":
        return VQSkillEncoder(
            codebook_size=spec.vq_codebook_size,
            ema_decay=spec.vq_ema_decay,
            dead_code_reset_iters=spec.vq_dead_code_reset_iters,
            **base,
        )
    msg = f"unknown latent_mode {mode!r}; expected one of {LATENT_MODES}."
    raise ValueError(msg)
