"""Agent algorithms module with backward compatibility."""

# New organized imports
from __future__ import annotations

from rlopt.agent.ppo import PPO, PPORLOptConfig
from rlopt.agent.sac import SAC, SACRLOptConfig
from rlopt.agent.ipmd import IPMD, IPMDRLOptConfig

__all__ = [
    "PPO",
    "SAC",
    "PPORLOptConfig",
    "SACRLOptConfig",
    "IPMD",
    "IPMDRLOptConfig",
]
