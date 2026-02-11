"""Configuration loader for environment-specific settings."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias, cast

import yaml

if TYPE_CHECKING:
    import numpy as np
    from tensordict import TensorDict
    from torch import Tensor

ObsKey: TypeAlias = str | tuple[str, ...]
BatchKey: TypeAlias = str | tuple[str, ...]


def load_env_config(env_name: str, config_dir: str | Path = "configs/env") -> dict[str, Any]:
    """Load environment-specific configuration from YAML file.
    
    Args:
        env_name: Environment name (e.g., "HalfCheetah-v5", "Pendulum-v1")
        config_dir: Directory containing environment config files
        
    Returns:
        Dictionary with environment-specific settings
        
    Example:
        >>> cfg = load_env_config("HalfCheetah-v5")
        >>> print(cfg["sac"]["utd_ratio"])
        4.0
    """
    config_dir = Path(config_dir)
    
    # Normalize environment name for file lookup
    # HalfCheetah-v5 -> halfcheetah
    env_base = env_name.lower().split("-")[0]
    
    # Try exact match first
    config_file = config_dir / f"{env_base}.yaml"
    
    if not config_file.exists():
        # Try common variations
        variations = [
            f"{env_base}.yaml",
            f"{env_base.replace('_', '')}.yaml",
            f"{env_name.lower()}.yaml",
        ]
        
        for var in variations:
            test_file = config_dir / var
            if test_file.exists():
                config_file = test_file
                break
        else:
            # Check if it's a Mujoco environment
            mujoco_envs = ["halfcheetah", "walker2d", "walker", "hopper", "ant", "humanoid"]
            if any(mujoco in env_base for mujoco in mujoco_envs):
                config_file = config_dir / "mujoco_base.yaml"
            else:
                # Default to simple environment config
                config_file = config_dir / "pendulum.yaml"
                
    if not config_file.exists():
        raise FileNotFoundError(
            f"No config found for {env_name}. "
            f"Looked for: {config_file} and variations. "
            f"Available configs: {list(config_dir.glob('*.yaml'))}"
        )
    
    with open(config_file) as f:
        config = yaml.safe_load(f)
    
    # Override name if specified
    if "name" not in config:
        config["name"] = env_name
        
    return config


def apply_env_config(rlopt_config, env_config: dict[str, Any]) -> None:
    """Apply environment config to RLOptConfig object.
    
    Args:
        rlopt_config: RLOptConfig or subclass instance
        env_config: Environment config dict from load_env_config()
        
    Example:
        >>> cfg = SACRLOptConfig()
        >>> env_cfg = load_env_config("HalfCheetah-v5")
        >>> apply_env_config(cfg, env_cfg)
    """
    # Apply training settings
    if "training" in env_config:
        training = env_config["training"]
        if "total_frames" in training:
            rlopt_config.collector.total_frames = training["total_frames"]
        if "frames_per_batch" in training:
            rlopt_config.collector.frames_per_batch = training["frames_per_batch"]
        if "init_random_frames" in training:
            rlopt_config.collector.init_random_frames = training["init_random_frames"]
    
    # Apply replay buffer settings
    if "replay_buffer" in env_config:
        rb = env_config["replay_buffer"]
        if "size" in rb:
            rlopt_config.replay_buffer.size = rb["size"]
    
    # Apply loss settings
    if "loss" in env_config:
        loss = env_config["loss"]
        if "mini_batch_size" in loss:
            rlopt_config.loss.mini_batch_size = loss["mini_batch_size"]
    
    # Apply SAC settings
    if "sac" in env_config and hasattr(rlopt_config, "sac"):
        sac = env_config["sac"]
        if "utd_ratio" in sac:
            rlopt_config.sac.utd_ratio = sac["utd_ratio"]
    
    # Apply parallel env settings
    if "parallel" in env_config:
        parallel = env_config["parallel"]
        if "num_envs" in parallel:
            rlopt_config.env.num_envs = parallel["num_envs"]
    
    # Apply network settings (if provided)
    if "network" in env_config:
        net = env_config["network"]
        if "hidden_dims" in net:
            # Only apply if the attributes exist and are not None
            if hasattr(rlopt_config, "policy") and rlopt_config.policy is not None:
                rlopt_config.policy.num_cells = net["hidden_dims"]
            if hasattr(rlopt_config, "q_function") and rlopt_config.q_function is not None:
                rlopt_config.q_function.num_cells = net["hidden_dims"]


def get_env_type(env_name: str) -> str:
    """Get environment type (simple, mujoco, atari, etc.).
    
    Args:
        env_name: Environment name
        
    Returns:
        Environment type string
    """
    env_lower = env_name.lower()
    
    if any(x in env_lower for x in ["halfcheetah", "walker", "hopper", "ant", "humanoid"]):
        return "mujoco"
    elif any(x in env_lower for x in ["breakout", "pong", "spaceinvaders"]):
        return "atari"
    elif any(x in env_lower for x in ["pendulum", "cartpole", "mountaincar"]):
        return "simple"
    else:
        return "unknown"


def list_available_configs(config_dir: str | Path = "configs/env") -> list[str]:
    """List all available environment configurations.
    
    Args:
        config_dir: Directory containing environment config files
        
    Returns:
        List of available config names
    """
    config_dir = Path(config_dir)
    if not config_dir.exists():
        return []
    
    configs = []
    for yaml_file in config_dir.glob("*.yaml"):
        if yaml_file.stem not in ["mujoco_base"]:  # Skip base configs
            configs.append(yaml_file.stem)
    
    return sorted(configs)


def dedupe_keys(keys: list[BatchKey]) -> list[BatchKey]:
    """Return keys in insertion order, removing duplicates."""
    return list(dict.fromkeys(keys))


def next_obs_key(key: ObsKey) -> tuple[str, ...]:
    """Prefix an observation key with ``"next"`` preserving nested paths."""
    if isinstance(key, tuple):
        return ("next", *key)
    return ("next", key)


def strip_next_prefix(key: BatchKey) -> ObsKey:
    """Remove a leading ``"next"`` from a batch key if present."""
    if isinstance(key, tuple) and len(key) > 0 and key[0] == "next":
        rest = key[1:]
        if len(rest) == 1:
            return cast(ObsKey, rest[0])
        return cast(ObsKey, rest)
    return cast(ObsKey, key)


def mapping_get_obs_value(obs: Mapping[Any, Any], key: ObsKey) -> Tensor | np.ndarray:
    """Read an observation value from a flat-or-nested mapping."""
    if key in obs:
        return cast(Tensor | np.ndarray, obs[key])
    if not isinstance(key, tuple):
        return cast(Tensor | np.ndarray, obs[key])
    cur: Any = obs
    for part in key:
        cur = cur[part]
    return cast(Tensor | np.ndarray, cur)


def flatten_feature_tensor(tensor: Tensor, feature_ndim: int) -> Tensor:
    """Flatten feature dims while keeping leading batch dims unchanged."""
    if feature_ndim <= 0:
        return tensor.unsqueeze(-1)
    if feature_ndim == 1:
        return tensor
    return tensor.flatten(start_dim=tensor.ndim - feature_ndim)


def infer_batch_shape(tensor: Tensor, feature_ndim: int) -> tuple[int, ...]:
    """Infer batch shape for a tensor with known feature rank."""
    if feature_ndim <= 0:
        return tuple(int(dim) for dim in tensor.shape)
    return tuple(int(dim) for dim in tensor.shape[:-feature_ndim])


def flatten_obs_group(td: TensorDict) -> TensorDict:
    """Flatten nested ``observation`` groups to match on-policy batch layout."""
    if "observation" in td:
        obs_group = cast(TensorDict, td.pop("observation"))
        for key in list(obs_group.keys()):
            td.set(key, obs_group.get(key))
    if "next" in td:
        next_td = cast(TensorDict, td.get("next"))
        if "observation" in next_td:
            next_obs_group = cast(TensorDict, next_td.pop("observation"))
            for key in list(next_obs_group.keys()):
                next_td.set(key, next_obs_group.get(key))
    return td


def epic_distance(r_est: Tensor, r_true: Tensor) -> tuple[Tensor, Tensor]:
    """Compute EPIC-style distance between estimated and true rewards."""
    import torch

    r_est = r_est.detach().float().flatten()
    r_true = r_true.detach().float().flatten()
    est_c = r_est - r_est.mean()
    true_c = r_true - r_true.mean()
    num = (est_c * true_c).sum()
    denom = est_c.norm() * true_c.norm()
    corr = (
        torch.zeros((), device=r_est.device)
        if denom < 1e-8
        else (num / denom).clamp(-1.0, 1.0)
    )
    return corr, (0.5 * (1.0 - corr)).sqrt()
