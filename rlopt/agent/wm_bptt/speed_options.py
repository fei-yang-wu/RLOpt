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
