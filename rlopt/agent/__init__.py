"""Agent algorithms module with backward compatibility."""

# New organized imports
from __future__ import annotations

from rlopt.agent.ppo import PPO, PPORLOptConfig
from rlopt.agent.sac import SAC, SACRLOptConfig
from rlopt.agent.ipmd import IPMD, IPMDRLOptConfig
from rlopt.agent.gail import AMP, AMPRLOptConfig, GAIL, GAILRLOptConfig
from rlopt.agent.ase import ASE, ASEConfig, ASERLOptConfig

__all__ = [
    "PPO",
    "SAC",
    "PPORLOptConfig",
    "SACRLOptConfig",
    "IPMD",
    "IPMDRLOptConfig",
    "GAIL",
    "GAILRLOptConfig",
    "AMP",
    "AMPRLOptConfig",
    "ASE",
    "ASEConfig",
    "ASERLOptConfig",
]
