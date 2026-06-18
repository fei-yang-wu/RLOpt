"""Agent algorithms module with backward compatibility."""

# New organized imports
from __future__ import annotations

from rlopt.agent.ase import ASE, ASEConfig, ASERLOptConfig
from rlopt.agent.fast_td3 import FastTD3, FastTD3RLOptConfig
from rlopt.agent.gail import AMP, GAIL, AMPRLOptConfig, GAILRLOptConfig
from rlopt.agent.hl_skill_diffsr import (
    FrozenHighLevelSkillCommandSampler,
    HighLevelSkillDiffSRConfig,
    HighLevelSkillDiffSRTrainer,
    HighLevelSkillEncoder,
)
from rlopt.agent.ipmd import (
    IPMD,
    IPMDSR,
    IPMDBilinear,
    IPMDBilinearRLOptConfig,
    IPMDRLOptConfig,
    IPMDSRRLOptConfig,
)
from rlopt.agent.ppo import PPO, PPORLOptConfig
from rlopt.agent.sac import SAC, FastSAC, FastSACRLOptConfig, SACRLOptConfig
from rlopt.agent.skill_commander import (
    DiffusionSkillCommander,
    FlowMatchingSkillCommander,
    FrozenSkillCommanderSampler,
    SkillCommander,
    SkillCommanderConfig,
    SkillCommanderTrainer,
)

__all__ = [
    "AMP",
    "ASE",
    "GAIL",
    "IPMD",
    "IPMDSR",
    "PPO",
    "SAC",
    "AMPRLOptConfig",
    "ASEConfig",
    "ASERLOptConfig",
    "FastSAC",
    "FastSACRLOptConfig",
    "FastTD3",
    "FastTD3RLOptConfig",
    "DiffusionSkillCommander",
    "FlowMatchingSkillCommander",
    "FrozenHighLevelSkillCommandSampler",
    "FrozenSkillCommanderSampler",
    "GAILRLOptConfig",
    "HighLevelSkillDiffSRConfig",
    "HighLevelSkillDiffSRTrainer",
    "HighLevelSkillEncoder",
    "IPMDBilinear",
    "IPMDBilinearRLOptConfig",
    "IPMDRLOptConfig",
    "IPMDSRRLOptConfig",
    "PPORLOptConfig",
    "SACRLOptConfig",
    "SkillCommander",
    "SkillCommanderConfig",
    "SkillCommanderTrainer",
]
