"""Networks of the world-model BPTT agent."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import Tensor, nn


class RunningNorm(nn.Module):
    """Running mean / std of a feature vector, updated explicitly.

    ``update`` uses the parallel (Chan) merge; ``freeze`` stops updates. The
    statistics are buffers, so they travel with ``state_dict``.
    """

    def __init__(self, dim: int, eps: float = 1e-2) -> None:
        super().__init__()
        self.eps = float(eps)
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros((), dtype=torch.float64))
        self.register_buffer("frozen", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def update(self, x: Tensor) -> None:
        if bool(self.frozen):
            return
        x = x.reshape(-1, x.shape[-1]).float()
        n = x.shape[0]
        if n == 0:
            return
        bmean = x.mean(0)
        bvar = x.var(0, unbiased=False)
        tot = self.count + n
        delta = bmean - self.mean
        w = n / tot
        self.mean += (delta * w).to(self.mean.dtype)
        m2 = self.var * self.count + bvar * n + delta.square() * (self.count * n / tot)
        self.var.copy_((m2 / tot).to(self.var.dtype))
        self.count.copy_(tot)

    def std(self) -> Tensor:
        return (self.var + self.eps**2).sqrt()

    def forward(self, x: Tensor) -> Tensor:
        return (x - self.mean) / self.std()


ACTIVATIONS: dict[str, Callable[[], nn.Module]] = {
    "elu": nn.ELU,
    "silu": nn.SiLU,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
}


def activation(name: str) -> Callable[[], nn.Module]:
    """Activation constructor by name (the IPMD ``activation_fn`` spelling)."""
    try:
        return ACTIVATIONS[name.lower()]
    except KeyError:
        msg = f"unknown activation {name!r}; one of {sorted(ACTIVATIONS)}"
        raise ValueError(msg) from None


def mlp(
    in_dim: int,
    hidden: Sequence[int],
    out_dim: int,
    *,
    act: Callable[[], nn.Module] = nn.SiLU,
    layer_norm: bool = False,
    out_gain: float = 1.0,
    zero_out: bool = False,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for h in hidden:
        layers.append(nn.Linear(d, h))
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(act())
        d = h
    last = nn.Linear(d, out_dim)
    if zero_out:
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
    elif out_gain != 1.0:
        with torch.no_grad():
            last.weight.mul_(out_gain)
            last.bias.zero_()
    layers.append(last)
    return nn.Sequential(*layers)


class DynamicsModel(nn.Module):
    """f(features(s), a, xi) -> normalized increment; LayerNorm + SiLU MLP.

    Inputs and increments are scaled by running normalizers; the last layer is
    zero, so the untrained model predicts the mean increment.

    ``structured`` (``wm.wm_structured``): the outputs are the task's semi-implicit
    targets, which ``out_norm`` fits, and ``inc_norm`` fits the plain increments,
    whose scale divides the fit's state error (NDM Gym's
    ``SemiImplicitIncrementWorldModel``). Only then is ``inc_norm`` a buffer, so a
    default model keeps its ``state_dict`` keys.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden: Sequence[int], *, structured: bool = False) -> None:
        super().__init__()
        self.in_norm = RunningNorm(in_dim, eps=1e-2)
        self.out_norm = RunningNorm(out_dim, eps=1e-4)
        self.net = mlp(in_dim, hidden, out_dim, layer_norm=True, zero_out=True)
        self.structured = bool(structured)
        if self.structured:
            self.inc_norm = RunningNorm(out_dim, eps=1e-4)

    def forward(self, x: Tensor, *, bf16: bool = False) -> Tensor:
        """Denormalized increment."""
        z = self.in_norm(x)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16 and z.is_cuda):
            y = self.net(z)
        return y.float() * self.out_norm.std() + self.out_norm.mean


class Actor(nn.Module):
    """Deterministic actor over normalized features."""

    def __init__(
        self, in_dim: int, out_dim: int, hidden: Sequence[int], out_gain: float, act: str = "elu"
    ) -> None:
        super().__init__()
        self.obs_norm = RunningNorm(in_dim, eps=1e-2)
        self.net = mlp(in_dim, hidden, out_dim, act=activation(act), out_gain=out_gain)

    def forward(self, obs: Tensor, *, bf16: bool = False) -> Tensor:
        z = self.obs_norm(obs)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16 and z.is_cuda):
            a = self.net(z)
        return a.float()

    def muon_mlps(self) -> list[nn.Module]:
        """The MLPs whose hidden matrices Muon updates (``wm.actor_opt=muon``)."""
        return [self.net]


class EncDecActor(nn.Module):
    """``wm.actor_arch=enc_dec``: NDM Gym's ``TrackingPolicy`` (paper Table A.4), ported
    from ``ndm_gym/policies/tracking_policy.py`` at ``05ea9f3`` under the MIT License
    (Copyright (c) 2026 NDM Gym Contributors; Copyright (c) 2026 ETH Zurich, Chong
    Zhang; the full text is in ``THIRD_PARTY_NOTICES.md``).

    The last ``goal_dim`` columns of the normalized input, the goal, pass an encoder
    to a latent bounded by tanh; a decoder maps (latent, the normalized rest, the
    proprio history) to the action. Both MLPs have the ``hidden`` widths; the
    decoder's last layer has the gain ``out_gain`` and no bias. The input normalizer
    is this module's one per-column statistic, as for ``Actor``, where NDM Gym keeps
    one per-frame statistic for all history slots and one for the goal.
    ``net.enc`` and ``net.dec`` are the two MLPs.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        goal_dim: int,
        hidden: Sequence[int],
        latent_dim: int,
        out_gain: float,
        act: str = "elu",
    ) -> None:
        super().__init__()
        if not 0 < goal_dim < in_dim:
            msg = f"EncDecActor splits {in_dim} input columns; a goal of {goal_dim} leaves no other input."
            raise ValueError(msg)
        self.split = in_dim - goal_dim
        self.obs_norm = RunningNorm(in_dim, eps=1e-2)
        a = activation(act)
        self.net = nn.ModuleDict(
            {
                "enc": mlp(goal_dim, hidden, latent_dim, act=a),
                "dec": mlp(latent_dim + self.split, hidden, out_dim, act=a, out_gain=out_gain),
            }
        )

    def forward(self, obs: Tensor, *, bf16: bool = False) -> Tensor:
        z = self.obs_norm(obs)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16 and z.is_cuda):
            latent = torch.tanh(self.net.enc(z[..., self.split :]).float())
            a = self.net.dec(torch.cat((latent, z[..., : self.split]), -1))
        return a.float()

    def muon_mlps(self) -> list[nn.Module]:
        """Each MLP's own hidden matrices: the encoder's last and the decoder's first
        layer are the input and output layers of their MLPs."""
        return [self.net.enc, self.net.dec]


ACTOR_ARCHS = ("flat", "enc_dec")


def build_actor(
    arch: str,
    in_dim: int,
    out_dim: int,
    hidden: Sequence[int],
    out_gain: float,
    act: str = "elu",
    *,
    goal_dim: int | None = None,
    latent_dim: int = 32,
) -> nn.Module:
    """The actor of ``wm.actor_arch``: ``Actor`` (``flat``) or ``EncDecActor``."""
    if arch == "flat":
        return Actor(in_dim, out_dim, hidden, out_gain, act)
    if arch == "enc_dec":
        if goal_dim is None:
            msg = "wm.actor_arch=enc_dec needs the goal's width (the task's policy_goal_dim)."
            raise ValueError(msg)
        return EncDecActor(in_dim, out_dim, goal_dim, hidden, latent_dim, out_gain, act)
    msg = f"wm.actor_arch must be one of {ACTOR_ARCHS}, got {arch!r}."
    raise ValueError(msg)


class Critic(nn.Module):
    def __init__(self, in_dim: int, hidden: Sequence[int], act: str = "elu") -> None:
        super().__init__()
        self.obs_norm = RunningNorm(in_dim, eps=1e-2)
        self.net = mlp(in_dim, hidden, 1, act=activation(act))

    def forward(self, obs: Tensor, *, bf16: bool = False) -> Tensor:
        z = self.obs_norm(obs)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16 and z.is_cuda):
            v = self.net(z)
        return v.float().squeeze(-1)
