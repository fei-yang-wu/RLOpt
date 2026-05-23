"""Utilities for feeding expert demonstrations into RLOpt agents.

This package provides:

- `StateMapper`: tiny adapter to map raw dataset fields into
  the `{observation, action, next/observation}` layout expected by IPMD.
- `build_prefetch_iterator`: wrap any iterator of TensorDict batches
  and prefetch + move to device asynchronously to reduce host<->device stalls.

These helpers are intentionally lightweight and do not pull extra deps.
They are meant to be used as a glue layer between external dataset libs
such as ImitationLearningTools (iltools) and RLOpt algorithms.
"""

from __future__ import annotations

from .stream import (  # noqa: F401
    OfflineExpertSampler,
    StateMapper,
    StreamingOfflineExpertSampler,
    build_offline_expert_sampler,
    build_prefetch_iterator,
)
