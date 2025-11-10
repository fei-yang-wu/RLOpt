"""Imitation learning algorithms module."""

from rlopt.agent.imitation.gail.gail import GAIL, GAILConfig, GAILRLOptConfig
from rlopt.agent.imitation.ipmd.ipmd import IPMD, IPMDConfig, IPMDRLOptConfig
from rlopt.agent.imitation.infogail.infogail import InfoGAIL, InfoGAILConfig, InfoGAILRLOptConfig
from rlopt.agent.imitation.ase.ase import ASE, ASEConfig, ASERLOptConfig

__all__ = [
    "GAIL",
    "GAILConfig",
    "GAILRLOptConfig",
    "IPMD",
    "IPMDConfig",
    "IPMDRLOptConfig",
    "InfoGAIL",
    "InfoGAILConfig",
    "InfoGAILRLOptConfig",
    "ASE",
    "ASEConfig",
    "ASERLOptConfig",
]
