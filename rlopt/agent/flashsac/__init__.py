"""FlashSAC: off-policy SAC with a norm-controlled actor/critic and a GPU replay ring."""

from __future__ import annotations

from rlopt.agent.flashsac.agent import FlashSAC, FlashSACPolicyNet
from rlopt.agent.flashsac.config import (
    FlashSACConfig,
    FlashSACReplayConfig,
    FlashSACRLOptConfig,
)
from rlopt.agent.flashsac.learner import FlashSACLearner, FlashSACLearnerConfig
from rlopt.agent.flashsac.noise import NoiseRepeat, publish_actor_state
from rlopt.agent.flashsac.replay import TimeMajorReplay

__all__ = [
    "FlashSAC",
    "FlashSACConfig",
    "FlashSACLearner",
    "FlashSACLearnerConfig",
    "FlashSACPolicyNet",
    "FlashSACRLOptConfig",
    "FlashSACReplayConfig",
    "NoiseRepeat",
    "TimeMajorReplay",
    "publish_actor_state",
]
