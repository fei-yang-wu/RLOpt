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
    """

    def __init__(self, in_dim: int, out_dim: int, hidden: Sequence[int]) -> None:
        super().__init__()
        self.in_norm = RunningNorm(in_dim, eps=1e-2)
        self.out_norm = RunningNorm(out_dim, eps=1e-4)
        self.net = mlp(in_dim, hidden, out_dim, layer_norm=True, zero_out=True)

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
