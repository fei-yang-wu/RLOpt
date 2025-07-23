from __future__ import annotations


mp_context = "fork"

from torchrl.envs import (
    ClipTransform,
    DoubleToFloat,
    EnvBase,
    EnvCreator,
    ParallelEnv,
    RewardSum,
    StepCounter,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.transforms import (
    Compose,
    TransformedEnv,
)


def make_mujoco_env(
    env_name="HalfCheetah-v4", device="cpu", from_pixels: bool = False
) -> EnvBase:
    env = GymEnv(env_name, device=device, from_pixels=from_pixels, pixels_only=False)
    env = TransformedEnv(env)
    env.append_transform(VecNormV2(in_keys=["observation"], decay=0.99999, eps=1e-2))
    env.append_transform(ClipTransform(in_keys=["observation"], low=-10, high=10))
    env.append_transform(RewardSum())
    env.append_transform(StepCounter())
    env.append_transform(DoubleToFloat(in_keys=["observation"]))
    return env


def make_isaaclab_gym_env(
    env,
    num_envs: int = 4096,
    device="cuda:0",
):
    return TransformedEnv(
        env,
        Compose(
            # VecNorm(in_keys=["observation"], decay=0.99999, eps=1e-2),
            # ClipTransform(in_keys=["observation"], low=-10, high=10),
            RewardSum(),
            StepCounter(1000),  # to count the steps of each trajectory
            # DoubleToFloat(in_keys=["observation"]),
        ),
    )


def make_gym_env(
    env_name="HalfCheetah-v4",
    parallel=False,
    obs_norm_sd=None,
    num_workers=1,
    device="cpu",
    from_pixels: bool = False,
    pixels_only=True,
    convert_actions_to_numpy=True,
):
    if obs_norm_sd is None:
        obs_norm_sd = {"standard_normal": True}
    if parallel:

        def maker():
            return GymEnv(
                env_name=env_name,
                from_pixels=from_pixels,
                pixels_only=pixels_only,
                device=device,
                convert_actions_to_numpy=convert_actions_to_numpy,
            )

        base_env = ParallelEnv(
            num_workers,
            EnvCreator(maker),
            # Don't create a sub-process if we have only one worker
            serial_for_single=True,
            mp_start_method=mp_context,
        )
    else:
        base_env = GymEnv(
            env_name=env_name,
            from_pixels=from_pixels,
            pixels_only=pixels_only,
            device=device,
            convert_actions_to_numpy=convert_actions_to_numpy,
        )

    env = TransformedEnv(
        base_env,
        Compose(
            # VecNormV2(in_keys=["observation"], decay=0.99999, eps=1e-2),
            ClipTransform(in_keys=["observation"], low=-10, high=10),
            RewardSum(),
            StepCounter(),  # to count the steps of each trajectory
            DoubleToFloat(in_keys=["observation"]),
        ),
    )

    return env
