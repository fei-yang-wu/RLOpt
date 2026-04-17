"""Backward-compatible latent-skill aliases.

Use ``latent_commands`` for rollout command plumbing and ``utils`` for pure
latent-imitation helpers. This module remains only as a compatibility shim.
"""

from __future__ import annotations

from rlopt.agent.imitation.latent_commands import (
    LatentCommandCollectorPolicy,
    LatentCommandMixin,
    RandomLatentCommandSampler,
)
from rlopt.agent.imitation.utils import (
    LatentEncoder,
    generalized_advantage_estimate,
)
from rlopt.config_utils import infer_batch_shape_from_mapping

LatentSkillCollectorPolicy = LatentCommandCollectorPolicy
LatentSkillMixin = LatentCommandMixin
RandomLatentSampler = RandomLatentCommandSampler

__all__ = [
    "LatentCommandCollectorPolicy",
    "LatentCommandMixin",
    "LatentEncoder",
    "LatentSkillCollectorPolicy",
    "LatentSkillMixin",
    "RandomLatentCommandSampler",
    "RandomLatentSampler",
    "generalized_advantage_estimate",
    "infer_batch_shape_from_mapping",
]
