"""Language-tilted flow planner over skill codes.

The planner factor of the joint model ``p(z, Y | X, l) = p(z | X, l) p(Y | X, z)``
is given the same exponential-family shape in ``z`` as the affine DiffSR head::

    log p(z | X, l) = log p_base(z | X) + z^T b(X, l) - log Z(X, l)

* ``p_base(z | X)`` is a language-free conditional flow over encoder codes,
  trained by flow matching on ``z = E(B)``; it carries all the multimodality.
* ``exp(z^T b(X, l))`` is a log-linear language expert. By Bayes' rule it is
  exactly a classifier of the instruction given the code,
  ``p(l | X, z) ∝ exp(z^T b(X, l))``, so ``b`` is trained by cross-entropy over
  the instructions present in the batch, with no sampler and no partition
  function.
* Sampling is classifier guidance with that log-linear classifier. The
  classifier's normalizer depends on ``z``, so the guidance score is the
  CENTERED tilt, not the raw one::

      ∇_z log p(l | X, z) = b(X, l) - E_{l' ~ p(l' | X, z)}[ b(X, l') ]

      v~(z_u, u | X, l) = v_base(z_u, u | X) + beta * w(u) * [ b(X, l) - bbar(X, zhat_1) ]

  evaluated at the endpoint estimate ``zhat_1 = z_u + (1-u) v_base``, with
  ``w(u) = (1-u) sigma^2 / ((1-u)^2 sigma^2 + u^2)``: the posterior variance
  of the endpoint given the interpolant under a unit-Gaussian endpoint prior,
  divided by ``(1-u)``. The centering makes the drift vanish once the sample
  is already classified as ``l``; dropping it (a constant drift ``b``) pushes
  codes off-manifold without bound (ICE job 5999809: sample norm x8 and the
  prediction loss x600 at ``beta = 4``). For a Gaussian base and a fixed
  drift the schedule ``w(u)`` is exact (``tests/test_language_tilt_planner.py``);
  with the learned base it is the usual classifier-guidance approximation.
  ``beta = 1`` is the Bayes posterior; ``beta > 1`` sharpens.
* Instructions compose linearly in the natural parameter:
  ``log p(l1 | z) + log p(l2 | z)`` has gradient ``b1 + b2 - 2 bbar``.

The encoder is never trained through this module: every target it sees is
detached, so the code space stays the predictive-only one.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from rlopt.agent.skill_commander import FlowMatchingSkillCommander


def _require_positive_int(name: str, value: int) -> int:
    normalized = int(value)
    if normalized <= 0:
        msg = f"{name} must be > 0, got {value!r}."
        raise ValueError(msg)
    return normalized


def _require_non_negative_int(name: str, value: int) -> int:
    normalized = int(value)
    if normalized < 0:
        msg = f"{name} must be >= 0, got {value!r}."
        raise ValueError(msg)
    return normalized


def _require_positive_float(name: str, value: float) -> float:
    normalized = float(value)
    if not normalized > 0.0:
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
class LanguageTiltPlannerConfig:
    """Architecture and training knobs of the language-tilted planner."""

    flow_hidden_dims: tuple[int, ...] = (1024, 512, 512)
    flow_time_embed_dim: int = 64
    flow_num_inference_steps: int = 16
    flow_noise_std: float = 1.0
    expert_hidden_dims: tuple[int, ...] = (512, 512)
    lang_proj_dim: int = 64
    lr: float = 3.0e-4
    weight_decay: float = 0.0
    state_noise_std: float = 0.0
    eval_betas: tuple[float, ...] = (0.0, 1.0, 2.0)
    eval_num_samples: int = 8

    def validate(self) -> None:
        self.flow_hidden_dims = tuple(
            _require_positive_int("flow_hidden_dims", dim) for dim in self.flow_hidden_dims
        )
        if not self.flow_hidden_dims:
            msg = "flow_hidden_dims must not be empty."
            raise ValueError(msg)
        self.flow_time_embed_dim = _require_positive_int(
            "flow_time_embed_dim", self.flow_time_embed_dim
        )
        self.flow_num_inference_steps = _require_positive_int(
            "flow_num_inference_steps", self.flow_num_inference_steps
        )
        self.flow_noise_std = _require_positive_float(
            "flow_noise_std", self.flow_noise_std
        )
        self.expert_hidden_dims = tuple(
            _require_positive_int("expert_hidden_dims", dim)
            for dim in self.expert_hidden_dims
        )
        self.lang_proj_dim = _require_positive_int("lang_proj_dim", self.lang_proj_dim)
        self.lr = _require_positive_float("lr", self.lr)
        self.weight_decay = _require_non_negative_float(
            "weight_decay", self.weight_decay
        )
        self.state_noise_std = _require_non_negative_float(
            "state_noise_std", self.state_noise_std
        )
        self.eval_betas = tuple(
            _require_non_negative_float("eval_betas", beta) for beta in self.eval_betas
        )
        if not self.eval_betas:
            msg = "eval_betas must not be empty."
            raise ValueError(msg)
        self.eval_num_samples = _require_positive_int(
            "eval_num_samples", self.eval_num_samples
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("flow_hidden_dims", "expert_hidden_dims"):
            payload[key] = [int(item) for item in payload[key]]
        payload["eval_betas"] = [float(item) for item in payload["eval_betas"]]
        return payload

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> LanguageTiltPlannerConfig:
        known = {item.name for item in fields(cls)}
        kwargs = {key: values[key] for key in known if key in values}
        for key in ("flow_hidden_dims", "expert_hidden_dims"):
            if key in kwargs:
                kwargs[key] = tuple(int(item) for item in kwargs[key])
        if "eval_betas" in kwargs:
            kwargs["eval_betas"] = tuple(float(item) for item in kwargs["eval_betas"])
        config = cls(**kwargs)
        config.validate()
        return config


def _mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = int(input_dim)
    for width in hidden_dims:
        layers.extend([nn.Linear(prev, int(width)), nn.LayerNorm(int(width)), nn.Mish()])
        prev = int(width)
    layers.append(nn.Linear(prev, int(output_dim)))
    return nn.Sequential(*layers)


class LanguageExpert(nn.Module):
    """Log-linear language expert ``b(X, l) = Phi(X) P e(l)``.

    ``Phi(X)`` is a state-dependent ``[z_dim, proj_dim]`` matrix and ``P`` a
    linear projection of the (frozen, precomputed) sentence embedding ``e(l)``.
    The factorization makes the in-batch classifier cheap: the code is
    projected once, ``q = Phi(X)^T z``, and the logits over all instructions in
    the batch are one matmul ``q E'^T``.
    """

    def __init__(
        self,
        *,
        state_dim: int,
        lang_embed_dim: int,
        z_dim: int,
        proj_dim: int,
        hidden_dims: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.state_dim = _require_positive_int("state_dim", state_dim)
        self.lang_embed_dim = _require_positive_int("lang_embed_dim", lang_embed_dim)
        self.z_dim = _require_positive_int("z_dim", z_dim)
        self.proj_dim = _require_positive_int("proj_dim", proj_dim)
        self.state_net = _mlp(self.state_dim, hidden_dims, self.z_dim * self.proj_dim)
        self.lang_proj = nn.Linear(self.lang_embed_dim, self.proj_dim, bias=False)
        # Start near zero so the untrained tilt is a no-op on the base flow.
        last = cast(nn.Linear, self.state_net[-1])
        nn.init.normal_(last.weight, std=1.0e-3)
        nn.init.zeros_(last.bias)

    def _validate_state(self, state: Tensor) -> None:
        if state.ndim != 2 or int(state.shape[-1]) != self.state_dim:
            msg = f"state must have shape [B, {self.state_dim}], got {tuple(state.shape)}."
            raise ValueError(msg)

    def state_matrix(self, state: Tensor) -> Tensor:
        """``Phi(X)`` as ``[B, z_dim, proj_dim]``."""
        self._validate_state(state)
        return self.state_net(state).reshape(int(state.shape[0]), self.z_dim, self.proj_dim)

    def project_language(self, lang_emb: Tensor) -> Tensor:
        """``P e(l)`` as ``[N, proj_dim]``."""
        if lang_emb.ndim != 2 or int(lang_emb.shape[-1]) != self.lang_embed_dim:
            msg = (
                f"lang_emb must have shape [N, {self.lang_embed_dim}], "
                f"got {tuple(lang_emb.shape)}."
            )
            raise ValueError(msg)
        return self.lang_proj(lang_emb)

    def tilt(self, state: Tensor, lang_emb: Tensor) -> Tensor:
        """``b(X, l)`` as ``[B, z_dim]`` for row-aligned states and instructions."""
        phi = self.state_matrix(state)
        if int(lang_emb.shape[0]) != int(state.shape[0]):
            msg = "state and lang_emb must be row-aligned."
            raise ValueError(msg)
        proj = self.project_language(lang_emb)
        return torch.bmm(phi, proj.unsqueeze(-1)).squeeze(-1)

    def code_query(self, state: Tensor, z: Tensor) -> Tensor:
        """``q = Phi(X)^T z`` as ``[B, proj_dim]``."""
        phi = self.state_matrix(state)
        if tuple(z.shape) != (int(state.shape[0]), self.z_dim):
            msg = f"z must have shape [B, {self.z_dim}], got {tuple(z.shape)}."
            raise ValueError(msg)
        return torch.bmm(z.unsqueeze(1), phi).squeeze(1)

    def logits(self, state: Tensor, z: Tensor, lang_proj: Tensor) -> Tensor:
        """``z^T b(X, l_j)`` for every projected instruction ``j``: ``[B, N]``."""
        return self.code_query(state, z) @ lang_proj.transpose(0, 1)

    def classifier_drift(
        self,
        phi: Tensor,
        z: Tensor,
        target_proj: Tensor,
        candidates_proj: Tensor,
        *,
        normalizer_count: float = 1.0,
    ) -> Tensor:
        """``∇_z log p(l | X, z) = Phi(X) [e'(l) - E_{l'~p(l'|X,z)} e'(l')]``.

        ``phi`` is ``state_matrix(state)`` (``[B, z_dim, proj]``),
        ``target_proj`` the projected target instruction(s) ``[B, proj]`` and
        ``candidates_proj`` the projected instruction set ``[N, proj]`` the
        classifier normalizes over. ``normalizer_count`` scales the centering
        term: a conjunction of ``k`` instructions passes the SUM of their
        projections and ``k``.
        """
        query = torch.bmm(z.unsqueeze(1), phi).squeeze(1)  # [B, proj]
        probs = torch.softmax(query @ candidates_proj.transpose(0, 1), dim=-1)
        expected = probs @ candidates_proj  # [B, proj]
        centered = target_proj - float(normalizer_count) * expected
        return torch.bmm(phi, centered.unsqueeze(-1)).squeeze(-1)

    @staticmethod
    def unique_instructions(lang_emb: Tensor) -> tuple[Tensor, Tensor]:
        """De-duplicate instructions by embedding value; returns (unique, inverse)."""
        unique, inverse = torch.unique(lang_emb, dim=0, return_inverse=True)
        return unique, inverse.reshape(-1)

    def classification_loss(
        self, state: Tensor, z: Tensor, lang_emb: Tensor
    ) -> tuple[Tensor, dict[str, float]]:
        """Cross-entropy of the row's instruction against the batch's unique ones."""
        unique, target = self.unique_instructions(lang_emb)
        logits = self.logits(state, z, self.project_language(unique))
        loss = F.cross_entropy(logits, target)
        with torch.no_grad():
            accuracy = (logits.argmax(dim=-1) == target).to(torch.float32).mean()
        metrics = {
            "lang_loss": float(loss.detach().item()),
            "lang_acc": float(accuracy.item()),
            "lang_unique": float(unique.shape[0]),
            "lang_chance": float(1.0 / max(int(unique.shape[0]), 1)),
        }
        return loss, metrics


class LanguageTiltPlanner(nn.Module):
    """``p(z | X, l) ∝ p_base(z | X) exp(z^T b(X, l))`` with a flow base."""

    def __init__(
        self,
        *,
        state_dim: int,
        lang_embed_dim: int,
        z_dim: int,
        config: LanguageTiltPlannerConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config if config is not None else LanguageTiltPlannerConfig()
        self.config.validate()
        self.state_dim = _require_positive_int("state_dim", state_dim)
        self.lang_embed_dim = _require_positive_int("lang_embed_dim", lang_embed_dim)
        self.z_dim = _require_positive_int("z_dim", z_dim)
        self.base = FlowMatchingSkillCommander(
            state_dim=self.state_dim,
            lang_embed_dim=0,
            z_dim=self.z_dim,
            hidden_dims=self.config.flow_hidden_dims,
            time_embed_dim=self.config.flow_time_embed_dim,
            num_inference_steps=self.config.flow_num_inference_steps,
            train_noise_std=self.config.flow_noise_std,
            inference_noise_std=self.config.flow_noise_std,
        )
        self.expert = LanguageExpert(
            state_dim=self.state_dim,
            lang_embed_dim=self.lang_embed_dim,
            z_dim=self.z_dim,
            proj_dim=self.config.lang_proj_dim,
            hidden_dims=self.config.expert_hidden_dims,
        )

    # -- training -----------------------------------------------------------
    @staticmethod
    def _no_language(state: Tensor) -> Tensor:
        return state.new_zeros((int(state.shape[0]), 0))

    def flow_loss(self, state: Tensor, z: Tensor) -> tuple[Tensor, dict[str, float]]:
        """Conditional flow-matching loss of the language-free base on ``z``."""
        loss, metrics = self.base.flow_matching_loss(state, self._no_language(state), z)
        return loss, {"flow_loss": float(loss.detach().item()), **metrics}

    def language_loss(
        self, state: Tensor, z: Tensor, lang_emb: Tensor
    ) -> tuple[Tensor, dict[str, float]]:
        return self.expert.classification_loss(state, z, lang_emb)

    def losses(
        self, state: Tensor, z: Tensor, lang_emb: Tensor
    ) -> tuple[Tensor, dict[str, float]]:
        """Sum of the base flow loss and the language-expert loss."""
        flow, flow_metrics = self.flow_loss(state, z)
        lang, lang_metrics = self.language_loss(state, z, lang_emb)
        return flow + lang, {**flow_metrics, **lang_metrics}

    # -- sampling -----------------------------------------------------------
    @staticmethod
    def guidance_weight(u: Tensor, sigma: float) -> Tensor:
        """``w(u) = (1-u) sigma^2 / ((1-u)^2 sigma^2 + u^2)``.

        Posterior variance of the endpoint given ``z_u`` under a unit-Gaussian
        endpoint prior, divided by ``(1-u)`` to convert the tilted posterior
        mean shift into a velocity shift. ``w(0) = 1``, ``w(1) = 0``, peak
        ``(1 + sqrt 2) / 2`` near ``u = 0.3``.
        """
        sigma_sq = float(sigma) ** 2
        one_minus = 1.0 - u
        return one_minus * sigma_sq / (one_minus.pow(2) * sigma_sq + u.pow(2))

    def velocity(
        self,
        z_u: Tensor,
        u: Tensor,
        state: Tensor,
        beta: float,
        *,
        phi: Tensor | None = None,
        target_proj: Tensor | None = None,
        candidates_proj: Tensor | None = None,
        normalizer_count: float = 1.0,
    ) -> Tensor:
        """Guided velocity ``v_base + beta w(u) [b(X,l) - bbar(X, zhat_1)]``.

        ``u`` is ``[B]``. With ``beta == 0`` (or no target) this is the base
        velocity. The centered drift is evaluated at the endpoint estimate
        ``zhat_1 = z_u + (1-u) v_base``.
        """
        base = self.base._vector_field(z_u, u, state, self._no_language(state))
        if float(beta) == 0.0 or target_proj is None:
            return base
        assert phi is not None
        assert candidates_proj is not None
        z_hat = z_u + (1.0 - u)[:, None] * base
        drift = self.expert.classifier_drift(
            phi, z_hat, target_proj, candidates_proj, normalizer_count=normalizer_count
        )
        weight = self.guidance_weight(u, self.config.flow_noise_std).to(base.dtype)
        return base + float(beta) * weight[:, None] * drift

    @torch.no_grad()
    def sample(
        self,
        state: Tensor,
        lang_emb: Tensor,
        *,
        beta: float = 1.0,
        num_steps: int | None = None,
        noise: Tensor | None = None,
        candidates: Tensor | None = None,
        target_proj: Tensor | None = None,
        normalizer_count: float = 1.0,
    ) -> Tensor:
        """Euler-integrate the guided flow from Gaussian noise to a code.

        ``candidates`` is the instruction set (raw embeddings ``[N, L]``) the
        classifier normalizes over; default = the unique instructions in
        ``lang_emb``. At deployment pass the table's unique embeddings.
        ``target_proj`` overrides the projected target (``[B, proj]``) so a
        conjunction can be sampled from the SUM of projections with
        ``normalizer_count`` = number of instructions.
        """
        batch_size = int(state.shape[0])
        phi: Tensor | None = None
        candidates_proj: Tensor | None = None
        if float(beta) != 0.0:
            phi = self.expert.state_matrix(state)
            if candidates is None:
                candidates, _ = self.expert.unique_instructions(lang_emb)
            candidates_proj = self.expert.project_language(candidates)
            if target_proj is None:
                target_proj = self.expert.project_language(lang_emb)
            elif tuple(target_proj.shape) != (batch_size, self.expert.proj_dim):
                msg = (
                    f"target_proj must have shape [{batch_size}, {self.expert.proj_dim}], "
                    f"got {tuple(target_proj.shape)}."
                )
                raise ValueError(msg)
        if noise is None:
            z = self.config.flow_noise_std * torch.randn(
                batch_size, self.z_dim, device=state.device, dtype=state.dtype
            )
        else:
            if tuple(noise.shape) != (batch_size, self.z_dim):
                msg = f"noise must have shape [{batch_size}, {self.z_dim}], got {tuple(noise.shape)}."
                raise ValueError(msg)
            z = noise.to(device=state.device, dtype=state.dtype)
        steps = int(self.config.flow_num_inference_steps if num_steps is None else num_steps)
        steps = _require_positive_int("num_steps", steps)
        dt = 1.0 / float(steps)
        for step in range(steps):
            u = torch.full((batch_size,), float(step) * dt, device=state.device, dtype=state.dtype)
            z = z + dt * self.velocity(
                z,
                u,
                state,
                beta,
                phi=phi,
                target_proj=target_proj,
                candidates_proj=candidates_proj,
                normalizer_count=normalizer_count,
            )
        return z

    def forward(self, state: Tensor, lang_emb: Tensor, *, beta: float = 1.0) -> Tensor:
        return self.sample(state, lang_emb, beta=beta)

    # -- persistence --------------------------------------------------------
    def checkpoint_state_dict(self) -> dict[str, Any]:
        return {
            "planner_type": "language_tilt_flow",
            "planner_state_dict": self.state_dict(),
            "planner_config": self.config.to_dict(),
            "state_dim": int(self.state_dim),
            "lang_embed_dim": int(self.lang_embed_dim),
            "z_dim": int(self.z_dim),
            "guidance": (
                "v = v_base(z_u,u|X) + beta * w(u) * [b(X,l) - E_{l'~p(l'|X,zhat_1)} b(X,l')]; "
                "zhat_1 = z_u + (1-u) v_base; "
                "w(u) = (1-u) s^2 / ((1-u)^2 s^2 + u^2), s = flow_noise_std"
            ),
        }


def load_language_tilt_planner(
    checkpoint: Mapping[str, Any] | str | Path,
    *,
    device: torch.device | str = "cpu",
) -> LanguageTiltPlanner:
    """Rebuild a planner from ``LanguageTiltPlanner.checkpoint_state_dict``."""
    if not isinstance(checkpoint, Mapping):
        checkpoint = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    if checkpoint.get("planner_type") != "language_tilt_flow":
        msg = "Checkpoint is not a language_tilt_flow planner."
        raise ValueError(msg)
    planner = LanguageTiltPlanner(
        state_dim=int(checkpoint["state_dim"]),
        lang_embed_dim=int(checkpoint["lang_embed_dim"]),
        z_dim=int(checkpoint["z_dim"]),
        config=LanguageTiltPlannerConfig.from_dict(checkpoint["planner_config"]),
    )
    planner.load_state_dict(checkpoint["planner_state_dict"])
    return planner.to(device)


__all__ = [
    "LanguageExpert",
    "LanguageTiltPlanner",
    "LanguageTiltPlannerConfig",
    "load_language_tilt_planner",
]
