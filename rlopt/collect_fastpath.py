"""Switch for RLOpt's rollout-collection fast paths (opt-in).

A fast path removes host-side or dead work from the collection step and
leaves every rollout tensor bit-identical (values, dtypes, shapes, random
numbers). ``RLOPT_COLLECT_FASTPATH=1`` (or ``all``) enables them; unset, empty
or ``0`` keeps the original code paths, which is the default and what
collection equivalence tests compare against.

The value is read once, at import, so set it before the process starts.
:func:`describe` is the resolved state, recorded in the W&B run config.
"""

from __future__ import annotations

import logging
import os

ENV_VAR = "RLOPT_COLLECT_FASTPATH"


def _read_switch() -> bool:
    value = os.environ.get(ENV_VAR, "").strip().lower()
    if value in {"", "0", "false", "no", "off", "none"}:
        return False
    if value in {"1", "true", "yes", "on", "all"}:
        return True
    msg = f"{ENV_VAR} must be 0/1 (or off/on, none/all); got {value!r}."
    raise ValueError(msg)


COLLECT_FASTPATH: bool = _read_switch()
"""Whether the exactness-preserving collection fast paths are active."""


def describe() -> dict[str, object]:
    """Resolved switch state, for logs and recorded run parameters."""
    return {
        "env_var": ENV_VAR,
        "value": os.environ.get(ENV_VAR, ""),
        "enabled": COLLECT_FASTPATH,
    }


if COLLECT_FASTPATH:
    # Warning level so it reaches the job log without logging configuration.
    logging.getLogger(__name__).warning(
        "RLOpt collection fast paths ON (%s=%r): hl-skill anchor-encoder skip",
        ENV_VAR,
        os.environ.get(ENV_VAR, ""),
    )


__all__ = ["COLLECT_FASTPATH", "ENV_VAR", "describe"]
