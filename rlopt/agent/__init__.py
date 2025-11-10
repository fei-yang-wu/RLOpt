"""Agent algorithms module with backward compatibility."""

# New organized imports
from rlopt.agent.imitation import IPMD, IPMDConfig, IPMDRLOptConfig
from rlopt.agent.rl import PPO, PPORLOptConfig, SAC, SACRLOptConfig

__all__ = [
    # RL algorithms
    "PPO",
    "PPORLOptConfig",
    "SAC",
    "SACRLOptConfig",
    # Imitation learning algorithms
    "IPMD",
    "IPMDConfig",
    "IPMDRLOptConfig",
]
