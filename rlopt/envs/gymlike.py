# ----------------------------------------------------
# Dummy placeholders for demonstration
# ----------------------------------------------------
def make_env(env_config: DictConfig) -> EnvBase:
    """Instantiate or wrap a TorchRL environment from env_config."""
    from torchrl.envs import Compose, TransformedEnv
    from torchrl.envs.libs.gym import GymEnv

    base_env = GymEnv(env_config.get("name", "CartPole-v1"))
    env = TransformedEnv(base_env, Compose())
    return env
