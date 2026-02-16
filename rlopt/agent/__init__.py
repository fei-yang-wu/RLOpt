"""Agent algorithms module with backward compatibility."""

# New organized imports
from __future__ import annotations

from rlopt.agent.ppo import PPO, PPORLOptConfig
from rlopt.agent.sac import SAC, SACRLOptConfig
from rlopt.agent.ipmd import IPMD, IPMDRLOptConfig
from rlopt.agent.fast_td3 import FastTD3, FastTD3RLOptConfig

__all__ = [
    "PPO",
    "SAC",
    "PPORLOptConfig",
    "SACRLOptConfig",
    "IPMD",
    "IPMDRLOptConfig",
    "FastTD3",
    "FastTD3RLOptConfig",
]
