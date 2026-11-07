# Parts of this file are adapted from NDM Gym (https://github.com/Frank-ZY-Dou/ndm-gym @ 05ea9f3: ndm_gym/algos/common.py,
# ndm_gym/algos/blinkmimic.py, ndm_gym/core/normalizer.py, ndm_gym/train.py), which ports them from BlinkMimic
# (https://github.com/fuyichen0725/blinkmimic @ d264658), under the MIT License:
#
# Copyright (c) 2026 NDM Gym Contributors
# Copyright (c) 2026 ETH Zurich, Chong Zhang
# Copyright (c) 2026 Fuyi Chen
# Copyright (c) 2026 Huy-Quang-Dao (InstantTrack, from which this software is derived)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Training-side helpers of the WMBPTT speed options (``wiki/code-opt-ndm-port.md``)."""

from __future__ import annotations

import copy
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from rlopt.agent.wm_bptt.muon import MuonAdam
from rlopt.agent.wm_bptt.networks import RunningNorm


def elementwise_3x3_products() -> None:
    """``wm.elementwise_3x3``: inside compiled functions, inductor decomposes the batched
    matrix products whose dimensions are all at most 3 (rotation matrices, 3-vectors) into
    fp32 elementwise products and sums, which it fuses with their neighbours, instead of one
    cuBLAS call each. Under TF32, cuBLAS runs some of them as tensor-core GEMMs (relative
    error 6e-4), so results change by more than rounding order.

    This patches inductor's decomposition table for the whole process, so it must run before
    the first compile. The compile caches do not key on that table, and a graph cached
    without the patch would keep the cuBLAS products, so the patch also sets a cache tag.
    """
    import torch.compiler.config  # noqa: PLC0415
    from torch._dynamo.utils import counters  # noqa: PLC0415
    from torch._inductor import decomposition  # noqa: PLC0415

    default = decomposition.decompositions[torch.ops.aten.bmm.default]
    if getattr(default, "elementwise_3x3", False):
        return  # this process has the patch: a second agent, for example
    if counters["frames"]["total"] > 0:
        msg = (
            "wm.elementwise_3x3 must apply before the first compile, and this process has compiled already: "
            "call apply_compile_options before building the env."
        )
        raise RuntimeError(msg)
    if not hasattr(torch.compiler.config, "cache_key_tag"):
        msg = "wm.elementwise_3x3 needs torch.compiler.config.cache_key_tag, or a cached graph would bypass it."
        raise RuntimeError(msg)

    def bmm(a: Tensor, b: Tensor, out_dtype: torch.dtype | None = None) -> Tensor:
        if out_dtype is None and a.dtype == torch.float32 and max(a.shape[1], a.shape[2], b.shape[2]) <= 3:
            return (a.unsqueeze(-1) * b.unsqueeze(1)).sum(2)
        return default(a, b, out_dtype)

    bmm.elementwise_3x3 = True  # type: ignore[attr-defined]
    decomposition.decompositions[torch.ops.aten.bmm.default] = bmm
    decomposition.fast_random_decomps.cache_clear()
    tag = torch.compiler.config.cache_key_tag
    torch.compiler.config.cache_key_tag = f"{tag}:rlopt:elementwise_3x3" if tag else "rlopt:elementwise_3x3"


def combo_kernels() -> None:
    """``wm.combo_kernels``: inside compiled functions, inductor runs independent kernels of
    a graph as one launch. Process-wide. The compile caches key on this setting."""
    import torch._inductor.config  # noqa: PLC0415

    torch._inductor.config.combo_kernels = True


def apply_compile_options(wm: Any) -> None:
    """The options of ``wm`` that act on the whole process. An entrypoint calls this before
    it builds the env, because env code can compile. The agent calls it again at construction,
    where it then changes nothing."""
    if wm.elementwise_3x3:
        elementwise_3x3_products()
    if wm.combo_kernels:
        combo_kernels()


def set_lr(optimizer: Any, lr: float) -> None:
    """Every group's learning rate; a tensor one (a capturable optimizer's) is
    written in place, for the CUDA graph that reads it."""
    for group in optimizer.param_groups:
        if torch.is_tensor(group["lr"]):
            group["lr"].fill_(lr)
        else:
            group["lr"] = lr


@torch.no_grad()
def update_norm_on_device(
    norm: RunningNorm, x: Tensor, valid: Tensor | None = None, gate: Tensor | None = None
) -> None:
    """``wm.no_sync``: ``RunningNorm.update`` with no host read.

    The freeze test runs on the device and ``torch.where`` writes the result in
    place. Without ``valid`` this is the same arithmetic as ``update``, so the
    statistics match it bit for bit. ``valid`` (rows,) replaces the boolean
    indexing of the caller: its rows alone count, so the batch moments are masked
    sums, equal to those of the indexed rows up to reduction order. A batch with
    no valid row changes nothing. ``gate``, a 0-dim bool, holds the statistics
    where it is false.
    """
    x = x.reshape(-1, x.shape[-1]).float()
    if x.shape[0] == 0:
        return
    live = ~norm.frozen if gate is None else gate & ~norm.frozen
    if valid is None:
        n: int | Tensor = x.shape[0]
        bmean = x.mean(0)
        bvar = x.var(0, unbiased=False)
    else:
        keep = valid.reshape(-1, 1)
        n = keep.sum(dtype=norm.count.dtype)
        live = live & (n > 0)
        rows = n.clamp_min(1.0)
        # ``where``, not a product: an excluded row may hold a non-finite value.
        bmean = (torch.where(keep, x, 0.0).sum(0) / rows).to(x.dtype)
        bvar = (torch.where(keep, x - bmean, 0.0).square().sum(0) / rows).to(x.dtype)
    tot = norm.count + n
    delta = bmean - norm.mean
    mean = norm.mean + (delta * (n / tot)).to(norm.mean.dtype)
    m2 = norm.var * norm.count + bvar * n + delta.square() * (norm.count * n / tot)
    norm.mean.copy_(torch.where(live, mean, norm.mean))
    norm.var.copy_(torch.where(live, (m2 / tot).to(norm.var.dtype), norm.var))
    norm.count.copy_(torch.where(live, tot, norm.count))


def gated_step(optimizer: Any, ok: Tensor) -> None:
    """``wm.no_sync``: ``optimizer``'s step where the 0-dim bool ``ok`` holds, skipped on
    the device elsewhere, with parameters and all optimizer state unchanged. The fused
    Adam's ``found_inf`` does this, and ``MuonAdam`` gates its Muon part. The gradients
    must be finite (``zero_nonfinite_grads``)."""
    if isinstance(optimizer, MuonAdam):
        optimizer.step(ok=ok)
        return
    if not optimizer.defaults.get("fused"):
        msg = "wm.no_sync skips a non-finite step with the fused Adam (wm.fused_adam)."
        raise ValueError(msg)
    optimizer.found_inf = (~ok).float()  # the fused kernel skips the step and undoes its step count
    try:
        optimizer.step()
    finally:
        optimizer.found_inf = None


@torch.no_grad()
def zero_nonfinite_grads(params: Iterable[Tensor], finite: Tensor) -> None:
    """``wm.no_sync``: every gradient set to zero where the 0-dim bool ``finite`` is
    false, without a host read."""
    for p in params:
        if p.grad is not None:
            p.grad.copy_(torch.where(finite, p.grad, 0.0))


class ActorAverage:
    """``wm.actor_ema``: an exponential moving average of the actor.

    The first ``update`` copies the live actor; later ones move the average by
    ``avg += (1 - decay) (live - avg)``, which is ``AveragedModel`` with
    ``get_ema_multi_avg_fn``. Only float32 tensors average (weights, normalizer
    mean and var). A normalizer's float64 ``count`` and bool ``frozen`` are not
    quantities to average, so they follow the live actor.
    """

    def __init__(self, actor: nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.module = copy.deepcopy(actor).requires_grad_(False)
        self.updates = 0
        self._live = actor
        pairs = list(zip(self._tensors(self.module), self._tensors(actor), strict=True))
        self._lerp = [(a, b) for a, b in pairs if b.dtype == torch.float32]
        self._copy = [(a, b) for a, b in pairs if b.dtype != torch.float32]

    @staticmethod
    def _tensors(module: nn.Module) -> list[Tensor]:
        return [t.detach() for t in (*module.parameters(), *module.buffers())]

    @torch.no_grad()
    def step(self) -> None:
        """One lerp, with no host-side state: the part a captured update can hold."""
        torch._foreach_lerp_([a for a, _ in self._lerp], [b for _, b in self._lerp], 1.0 - self.decay)
        for avg, live in self._copy:
            avg.copy_(live)

    @torch.no_grad()
    def update(self) -> None:
        """Call once per actor update."""
        if self.updates == 0:
            self.module.load_state_dict(self._live.state_dict())
        else:
            self.step()
        self.updates += 1


class ActorLR:
    """``wm.actor_lr_mult``: the actor's learning rate is the schedule's times a multiplier.

    From the log row where the training return first reaches ``decay_return``,
    or at iteration ``decay_by`` at the latest, the multiplier ramps linearly
    to 1 over ``decay_len`` iterations. None disables a trigger. The defaults
    are off: the multiplier is 1 at every iteration.
    """

    def __init__(
        self,
        mult: float = 1.0,
        decay_return: float | None = None,
        decay_len: int = 0,
        decay_by: int | None = None,
    ) -> None:
        self.mult, self.decay_return, self.decay_len, self.decay_by = mult, decay_return, decay_len, decay_by
        self.decay_at: int | None = None

    @property
    def on(self) -> bool:
        return self.mult != 1.0

    @property
    def watches_return(self) -> bool:
        return self.decay_return is not None

    def at(self, iteration: int) -> float:
        """The multiplier at ``iteration``."""
        if self.decay_at is None:
            return self.mult
        return self.mult + (1.0 - self.mult) * min(1.0, (iteration - self.decay_at) / self.decay_len)

    def observe(self, iteration: int, ret: float | None) -> str | None:
        """At a log row, ``iteration`` iterations done and the logged training
        return ``ret`` (None: no episode ended, the deadline alone): the
        trigger's message when it fires, which it does once, else None."""
        if self.decay_return is None or self.decay_at is not None:
            return None
        reached = ret is not None and ret >= self.decay_return
        if not reached and not (self.decay_by is not None and iteration >= self.decay_by):
            return None
        self.decay_at = iteration
        why = f"training return {ret:.3f} >= {self.decay_return}" if reached else f"deadline {self.decay_by}"
        return f"actor_lr_mult: {why} at iteration {iteration}: x{self.mult:g} -> x1 over {self.decay_len} iterations"

    def state_dict(self) -> dict[str, Any]:
        """The trigger's iteration for a checkpoint; nothing without a trigger."""
        return {"actor_lr_decay_at": self.decay_at} if self.watches_return else {}

    def load_state_dict(self, checkpoint: dict[str, Any]) -> None:
        if self.watches_return:
            if "actor_lr_decay_at" not in checkpoint:
                msg = "wm.actor_lr_decay_return: the checkpoint has no trigger state (actor_lr_decay_at)."
                raise ValueError(msg)
            self.decay_at = checkpoint["actor_lr_decay_at"]


def atomic_save(payload: dict[str, Any], target: Path) -> None:
    """``torch.save`` through a temporary name, so a reader never sees half a file."""
    tmp = target.with_name(target.name + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(target)


def cpu_snapshot(value: Any) -> Any:
    """A copy of a nest with every tensor on the CPU, safe from later in-place updates."""
    if isinstance(value, Tensor):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, dict):
        return {key: cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_snapshot(item) for item in value)
    return value


def full_checkpoint_due(elapsed: float, last: float, every_min: float) -> bool:
    """``wm.last_every_min``: whether the periodic checkpoint at ``elapsed`` training
    seconds also writes a full one, the previous one (or the run's start or resume
    point) having been at ``last``. 0 means only the run's end writes one."""
    return every_min > 0 and elapsed - last >= 60.0 * every_min


class CheckpointWriter:
    """``wm.ckpt_light``: writes checkpoints on one background thread.

    At most two writes wait behind the running one, so a slow disk stalls the
    loop rather than filling memory. A failed write raises in the loop, at the
    next ``submit``, or at ``close``. Payloads must be ``cpu_snapshot`` copies.
    """

    MAX_QUEUED = 2

    def __init__(self) -> None:
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="wm-checkpoint")
        self._pending: list[Future[None]] = []

    @staticmethod
    def _write(payload: dict[str, Any], target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        atomic_save(payload, target)

    def _collect(self, *, drain: bool = False) -> None:
        while self._pending and (drain or self._pending[0].done() or len(self._pending) > self.MAX_QUEUED):
            self._pending.pop(0).result()

    def submit(self, payload: dict[str, Any], target: Path) -> None:
        self._collect()
        self._pending.append(self._pool.submit(self._write, payload, target))

    def close(self) -> None:
        """Finish every write; the first failure raises after the thread stops."""
        try:
            self._collect(drain=True)
        finally:
            self._pool.shutdown()
