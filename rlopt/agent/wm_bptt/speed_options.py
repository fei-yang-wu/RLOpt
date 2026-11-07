# Parts of this file are adapted from NDM Gym (https://github.com/Frank-ZY-Dou/ndm-gym @ 05ea9f3: ndm_gym/algos/common.py,
# ndm_gym/algos/blinkmimic.py, ndm_gym/train.py), which ports them from BlinkMimic
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
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


def set_lr(optimizer: Any, lr: float) -> None:
    """Every group's learning rate; a tensor one (a capturable optimizer's) is
    written in place, for the CUDA graph that reads it."""
    for group in optimizer.param_groups:
        if torch.is_tensor(group["lr"]):
            group["lr"].fill_(lr)
        else:
            group["lr"] = lr


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
