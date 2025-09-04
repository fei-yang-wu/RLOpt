from __future__ import annotations

import threading
from dataclasses import dataclass, field
from queue import Queue
from typing import Callable, Iterator, Mapping, MutableMapping, Optional

import torch
from tensordict import TensorDict


@dataclass
class StateMapper:
    """Map raw dataset transitions into IPMD's expected keys.

    The mapper can be configured with simple key copies or custom callables.
    It operates on batched transitions and should return tensors with a
    consistent batch dimension.

    Configure using dictionaries that map destination keys to either a
    source key (str) or a callable that takes a `TensorDict` and returns
    a `torch.Tensor`.
    """

    obs_map: Mapping[str, str | Callable[[TensorDict], torch.Tensor]] = field(
        default_factory=dict
    )
    action_key: str | Callable[[TensorDict], torch.Tensor] = "action"
    next_obs_map: Mapping[
        str, str | Callable[[TensorDict], torch.Tensor]
    ] = field(default_factory=dict)

    def _apply_map(
        self, raw: TensorDict, mapping: Mapping[str, str | Callable[[TensorDict], torch.Tensor]]
    ) -> TensorDict:
        out: MutableMapping[str, torch.Tensor] = {}
        for dst, spec in mapping.items():
            if isinstance(spec, str):
                out[dst] = raw.get(spec)
            else:
                out[dst] = spec(raw)
        return TensorDict(out, batch_size=raw.batch_size, device=raw.device)

    def map_transition(self, raw: TensorDict) -> TensorDict:
        """Return a TensorDict with keys: observation, action, (next, observation).

        - `observation` is built from `obs_map`. If `obs_map` is empty and
          the raw input already has `"observation"`, it is forwarded.
        - `action` can be copied from a key or computed via callable.
        - `next/observation` is built from `next_obs_map`. If empty and the raw
          input already has nested next/observation, it is forwarded.
        """
        # observation
        if self.obs_map:
            obs_td = self._apply_map(raw, self.obs_map)
        else:
            # passthrough if present
            obs_td = TensorDict({}, batch_size=raw.batch_size, device=raw.device)
            if "observation" in raw.keys():
                obs_td.set("observation", raw.get("observation"))
            else:
                raise KeyError(
                    "StateMapper: obs_map is empty and 'observation' not found in raw"
                )

        # action
        if isinstance(self.action_key, str):
            action = raw.get(self.action_key)
        else:
            action = self.action_key(raw)

        # next/observation
        if self.next_obs_map:
            next_td = self._apply_map(raw, self.next_obs_map)
        else:
            next_td = TensorDict({}, batch_size=raw.batch_size, device=raw.device)
            # support nested ('next', 'observation') or flat 'next_observation'
            if ("next", "observation") in raw.keys(True, True):
                next_td.set(("next", "observation"), raw.get(("next", "observation")))
            elif "next_observation" in raw.keys():
                next_td.set(("next", "observation"), raw.get("next_observation"))
            else:
                raise KeyError(
                    "StateMapper: next_obs_map is empty and next observation not found"
                )

        out = TensorDict({}, batch_size=raw.batch_size, device=raw.device)
        # Merge observation fields: if obs_td has multiple fields, concatenate under 'observation'
        if "observation" in obs_td.keys():
            out.set("observation", obs_td.get("observation"))
        else:
            # Concatenate all values in insertion order
            vals = [obs_td.get(k) for k in obs_td.keys()]
            out.set("observation", torch.cat(vals, dim=-1))

        out.set("action", action)

        # Merge next observation similarly
        if ("next", "observation") in next_td.keys(True, True):
            out.set(("next", "observation"), next_td.get(("next", "observation")))
        else:
            vals = [next_td.get(k) for k in next_td.keys()]
            out.set(("next", "observation"), torch.cat(vals, dim=-1))

        return out


def build_prefetch_iterator(
    source: Iterator[TensorDict],
    *,
    device: torch.device,
    prefetch: int = 2,
    non_blocking: bool = True,
) -> Iterator[TensorDict]:
    """Wrap an iterator of TensorDict batches with device prefetching.

    - Moves each batch to `device` using `TensorDict.to(device, non_blocking=...)`.
    - Buffers up to `prefetch` batches on a background thread.

    This is a low-overhead way to stream expert data into GPU-heavy training
    loops (e.g., IsaacLab) without stalling on H2D transfers.
    """

    q: Queue[Optional[TensorDict]] = Queue(maxsize=max(1, prefetch))
    stop_token = object()

    def _worker() -> None:
        try:
            for batch in source:
                # Push after device transfer to overlap copy and compute
                moved = batch.to(device, non_blocking=non_blocking)
                q.put(moved, block=True)
        finally:
            # Sentinel to terminate consumer
            q.put(None)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    while True:
        item = q.get()
        if item is None:
            break
        yield item

