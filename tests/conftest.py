from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable

import gymnasium as gym
import pytest
import torch
import torchrl.envs.libs.gym
from torchrl.envs import EnvCreator, ParallelEnv, TransformedEnv
from torchrl.envs.libs.gym import GymEnv as TorchRLGymEnv

from rlopt.agent.ipmd.ipmd import IPMDRLOptConfig
from rlopt.agent.l2t.l2t import L2TRLOptConfig
from rlopt.agent.ppo.ppo import PPORLOptConfig
from rlopt.agent.sac.sac import SACRLOptConfig
from rlopt.configs import (
    FeatureBlockSpec,
    LSTMBlockConfig,
    MLPBlockConfig,
    ModuleNetConfig,
    NetworkLayout,
)


def is_env_available(env_name: str) -> bool:
    try:
        # Lazy import to avoid unnecessary backend loads
        from torchrl.envs.libs.gym import GymEnv as TorchRLGymEnv

        _ = TorchRLGymEnv(env_name, device="cpu")
        return True
    except Exception:
        return False


def pytest_configure(config: pytest.Config) -> None:
    # Document markers for discoverability
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with -m 'not slow')"
    )
    config.addinivalue_line(
        "markers",
        "mujoco(name): requires a specific gymnasium mujoco env; auto-skips if unavailable",
    )
    config.addinivalue_line(
        "markers",
        "gpu: requires CUDA; auto-skips if torch.cuda.is_available() is False",
    )
    config.addinivalue_line("markers", "compile: tests that enable torch.compile")


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("compile")
    group.addoption(
        "--compile-mode",
        action="append",
        default=None,
        help=(
            "torch.compile mode(s) to test (e.g., default, reduce-overhead). "
            "Pass multiple times or as a comma-separated list."
        ),
    )
    group.addoption(
        "--compile-warmup",
        action="store",
        default=None,
        help="Warmup iterations for compile (int)",
    )
    group.addoption(
        "--cudagraphs",
        action="store_true",
        default=False,
        help="Enable CUDA graphs in compile tests",
    )


def _parse_compile_modes(config: pytest.Config) -> list[str]:
    modes_opt = config.getoption("--compile-mode")
    if not modes_opt:
        # default set when not provided
        return ["default", "reduce-overhead"]
    modes: list[str] = []
    for opt in modes_opt:
        for item in str(opt).split(","):
            val = item.strip()
            if val:
                modes.append(val)
    # Deduplicate preserving order
    seen: set[str] = set()
    result: list[str] = []
    for m in modes:
        if m not in seen:
            seen.add(m)
            result.append(m)
    return result


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "compile_mode" in metafunc.fixturenames:
        modes = _parse_compile_modes(metafunc.config)
        metafunc.parametrize("compile_mode", modes, ids=modes)


@pytest.fixture
def compile_warmup(request: pytest.FixtureRequest) -> int:
    val = request.config.getoption("--compile-warmup")
    try:
        return int(val) if val is not None else 1
    except Exception:
        return 1


@pytest.fixture
def compile_cudagraphs(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--cudagraphs"))


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    for item in items:
        mujoco_mark: pytest.Mark | None = item.get_closest_marker("mujoco")
        if mujoco_mark is not None:
            env_name = mujoco_mark.args[0] if mujoco_mark.args else "HalfCheetah-v5"
            if not is_env_available(env_name):
                item.add_marker(pytest.mark.skip(reason=f"{env_name} unavailable"))

        if item.get_closest_marker("gpu") is not None and not torch.cuda.is_available():
            item.add_marker(pytest.mark.skip(reason="CUDA not available"))


@pytest.fixture(scope="session", autouse=True)
def _set_fork_start_method() -> None:
    """Use 'fork' so subprocess workers inherit patches and RNG state.

    This mirrors the pattern used in some tests to avoid isaaclab import issues
    and keeps behavior consistent across platforms that support fork.
    """
    if mp.get_start_method(allow_none=True) != "fork":
        try:
            mp.set_start_method("fork", force=True)
        except Exception:
            # On platforms without 'fork' (e.g., Windows), ignore.
            pass


@pytest.fixture(scope="session", autouse=True)
def _patch_gymenv_vecenv_batched() -> None:
    """Ensure TorchRL GymEnv treats VecEnv as batched to avoid isaaclab side effects."""
    from stable_baselines3.common.vec_env.base_vec_env import VecEnv

    torchrl.envs.libs.gym.GymEnv._is_batched = property(  # type: ignore[attr-defined]
        lambda self: isinstance(self._env, VecEnv)
    )


# ---------------------
# Config factory fixtures
# ---------------------


@pytest.fixture
def ppo_cfg_factory() -> Callable[..., PPORLOptConfig]:
    def _make(
        *,
        env_name: str = "Pendulum-v1",
        num_envs: int = 8,
        frames_per_batch: int = 1024,
        total_frames: int = 2048,
        feature_dim: int = 64,
        lr: float = 3e-4,
        mini_batch_size: int = 256,
        epochs: int = 2,
    ) -> PPORLOptConfig:
        cfg = PPORLOptConfig()
        # env
        cfg.env.env_name = env_name
        cfg.env.device = "cpu"
        cfg.env.num_envs = num_envs
        # collector
        cfg.collector.frames_per_batch = frames_per_batch
        cfg.collector.total_frames = total_frames
        cfg.collector.set_truncated = False
        # optimization
        cfg.optim.lr = lr
        # loss
        cfg.loss.mini_batch_size = mini_batch_size
        cfg.loss.epochs = epochs
        # feature extractor + io keys
        cfg.use_feature_extractor = True
        cfg.feature_extractor.output_dim = feature_dim
        cfg.policy.num_cells = [64, 64]
        cfg.value_net.num_cells = [64, 64]
        cfg.policy_in_keys = ["hidden"]
        cfg.value_net_in_keys = ["hidden"]
        cfg.total_input_keys = ["observation"]
        # logger
        cfg.logger.backend = None
        # device
        cfg.device = "cpu"
        # compile
        cfg.compile.compile = False
        return cfg

    return _make


@pytest.fixture
def sac_cfg_factory() -> Callable[..., SACRLOptConfig]:
    def _make(
        *,
        env_name: str = "Pendulum-v1",
        num_envs: int = 8,
        frames_per_batch: int = 1024,
        total_frames: int = 2048,
        feature_dim: int = 64,
        lr: float = 3e-4,
        mini_batch_size: int = 256,
        utd_ratio: float = 0.25,
        init_random_frames: int = 256,
    ) -> SACRLOptConfig:
        cfg = SACRLOptConfig()
        # env
        cfg.env.env_name = env_name
        cfg.env.device = "cpu"
        cfg.env.num_envs = num_envs
        # collector
        cfg.collector.frames_per_batch = frames_per_batch
        cfg.collector.total_frames = total_frames
        cfg.collector.set_truncated = False
        cfg.collector.init_random_frames = init_random_frames
        # optimization
        cfg.optim.lr = lr
        # loss
        cfg.loss.mini_batch_size = mini_batch_size
        # feature extractor + io keys
        cfg.use_feature_extractor = True
        cfg.feature_extractor.output_dim = feature_dim
        cfg.policy.num_cells = [64, 64]
        cfg.value_net.num_cells = [64, 64]
        cfg.policy_in_keys = ["hidden"]
        cfg.value_net_in_keys = ["hidden"]
        cfg.total_input_keys = ["observation"]
        # logger
        cfg.logger.backend = None
        # device
        cfg.device = "cpu"
        # compile
        cfg.compile.compile = False
        # sac-specific
        cfg.sac.utd_ratio = utd_ratio
        # q net cells expected by SAC implementation
        cfg.action_value_net.num_cells = [64, 64]
        cfg.use_value_function = False
        return cfg

    return _make


@pytest.fixture
def ipmd_cfg_factory() -> Callable[..., IPMDRLOptConfig]:
    def _make(
        *,
        env_name: str = "Pendulum-v1",
        num_envs: int = 8,
        frames_per_batch: int = 1024,
        total_frames: int = 2048,
        feature_dim: int = 64,
        lr: float = 3e-4,
        mini_batch_size: int = 256,
        utd_ratio: float = 0.25,
        init_random_frames: int = 256,
    ) -> IPMDRLOptConfig:
        cfg = IPMDRLOptConfig()
        # env
        cfg.env.env_name = env_name
        cfg.env.device = "cpu"
        cfg.env.num_envs = num_envs
        # collector
        cfg.collector.frames_per_batch = frames_per_batch
        cfg.collector.total_frames = total_frames
        cfg.collector.set_truncated = False
        cfg.collector.init_random_frames = init_random_frames
        # optimization
        cfg.optim.lr = lr
        # loss
        cfg.loss.mini_batch_size = mini_batch_size
        # feature extractor + io keys
        cfg.use_feature_extractor = True
        cfg.feature_extractor.output_dim = feature_dim
        cfg.policy.num_cells = [64, 64]
        cfg.value_net.num_cells = [64, 64]
        cfg.policy_in_keys = ["hidden"]
        cfg.value_net_in_keys = ["hidden"]
        cfg.total_input_keys = ["observation"]
        # logger
        cfg.logger.backend = None
        # device
        cfg.device = "cpu"
        # compile
        cfg.compile.compile = False
        # ipmd-specific
        cfg.ipmd.utd_ratio = float(utd_ratio)
        # q net cells expected by off-policy implementation
        cfg.action_value_net.num_cells = [64, 64]
        cfg.use_value_function = False
        return cfg

    return _make


@pytest.fixture
def l2t_cfg_factory() -> Callable[..., L2TRLOptConfig]:
    def _make(
        *,
        env_name: str = "Pendulum-v1",
        num_envs: int = 4,
        frames_per_batch: int = 128,
        total_frames: int = 128,
        feature_dim: int = 64,
        lr: float = 3e-4,
        mini_batch_size: int = 64,
        epochs: int = 1,
        student_recurrent: bool = False,
        imitation: str = "l2",
        mixture_coeff: float = 0.2,
    ) -> L2TRLOptConfig:
        cfg = L2TRLOptConfig()
        # env
        cfg.env.env_name = env_name
        cfg.env.device = "cpu"
        cfg.env.num_envs = num_envs
        # collector
        cfg.collector.frames_per_batch = frames_per_batch
        cfg.collector.total_frames = total_frames
        cfg.collector.set_truncated = False
        # optimization
        cfg.optim.lr = lr
        # loss
        cfg.loss.mini_batch_size = mini_batch_size
        cfg.loss.epochs = epochs
        cfg.loss.gamma = 0.99
        # use feature extractor
        cfg.use_feature_extractor = True
        cfg.feature_extractor.output_dim = feature_dim
        # io keys
        cfg.policy_in_keys = ["hidden"]
        cfg.value_net_in_keys = ["hidden"]
        cfg.total_input_keys = ["observation"]
        # logger
        cfg.logger.backend = None
        cfg.device = "cpu"
        cfg.compile.compile = False

        # Teacher layout (MLP shared features)
        teacher = NetworkLayout()
        teacher.shared.features["shared_mlp"] = FeatureBlockSpec(
            type="mlp",
            mlp=MLPBlockConfig(num_cells=[64, 64], activation="elu"),
            output_dim=feature_dim,
        )
        teacher.policy.feature_ref = "shared_mlp"
        teacher.policy.head = MLPBlockConfig(num_cells=[64], activation="elu")
        # Value
        teacher.value = teacher.value or ModuleNetConfig()
        teacher.value.feature_ref = "shared_mlp"  # type: ignore[attr-defined]
        teacher.value.head = MLPBlockConfig(num_cells=[64], activation="elu")  # type: ignore[attr-defined]
        cfg.network = teacher

        # Student layout
        student = NetworkLayout()
        if student_recurrent:
            student.shared.features["student_lstm"] = FeatureBlockSpec(
                type="lstm",
                lstm=LSTMBlockConfig(hidden_size=feature_dim),
                output_dim=feature_dim,
            )
            student.policy.feature_ref = "student_lstm"
            student.value = student.value or ModuleNetConfig()
            student.value.feature_ref = "student_lstm"  # type: ignore[attr-defined]
        else:
            student.shared.features["student_mlp"] = FeatureBlockSpec(
                type="mlp",
                mlp=MLPBlockConfig(num_cells=[64, 64], activation="elu"),
                output_dim=feature_dim,
            )
            student.policy.feature_ref = "student_mlp"
            student.value = student.value or ModuleNetConfig()
            student.value.feature_ref = "student_mlp"  # type: ignore[attr-defined]
        student.policy.head = MLPBlockConfig(num_cells=[64], activation="elu")
        student.value.head = MLPBlockConfig(num_cells=[64], activation="elu")  # type: ignore[attr-defined]
        cfg.l2t.student = student

        # L2T specifics
        cfg.l2t.imitation_type = imitation
        cfg.l2t.imitation_coeff = 1.0
        cfg.l2t.mixture_coeff = mixture_coeff
        cfg.l2t.clip_epsilon = 0.2
        cfg.l2t.critic_coeff = 1.0
        cfg.l2t.entropy_coeff = 0.0

        return cfg

    return _make


# ---------------------
# Env factory fixtures
# ---------------------


@pytest.fixture
def make_env() -> Callable[[str, str], TransformedEnv]:
    def _make(env_name: str, device: str = "cpu") -> TransformedEnv:
        base = TorchRLGymEnv(env_name, device=device)
        env = TransformedEnv(base)
        from torchrl.envs import ClipTransform, RewardSum, StepCounter

        env.append_transform(ClipTransform(in_keys=["observation"], low=-10, high=10))
        env.append_transform(RewardSum())
        env.append_transform(StepCounter())
        # Cast to float32 only when observations are float64 (e.g., MuJoCo)
        try:
            obs_dtype = env.observation_spec["observation"].dtype
        except Exception:
            obs_dtype = None
        if obs_dtype is not None and obs_dtype == torch.float64:
            from torchrl.envs import DoubleToFloat

            env.append_transform(DoubleToFloat(in_keys=["observation"]))
        return env

    return _make


@pytest.fixture
def make_env_parallel() -> Callable[[str, int, str], TransformedEnv]:
    def _make(env_name: str, num_workers: int, device: str = "cpu") -> TransformedEnv:
        def maker():
            env = gym.make(env_name)
            return TorchRLGymEnv(env, device=device)

        base = ParallelEnv(
            num_workers,
            EnvCreator(maker),
            serial_for_single=True,
            mp_start_method="fork",
        )
        env = TransformedEnv(base)
        from torchrl.envs import ClipTransform, RewardSum, StepCounter

        env.append_transform(ClipTransform(in_keys=["observation"], low=-10, high=10))
        env.append_transform(RewardSum())
        env.append_transform(StepCounter())
        # Cast to float32 only when observations are float64 (e.g., MuJoCo)
        try:
            obs_dtype = env.observation_spec["observation"].dtype
        except Exception:
            obs_dtype = None
        if obs_dtype is not None and obs_dtype == torch.float64:
            from torchrl.envs import DoubleToFloat

            env.append_transform(DoubleToFloat(in_keys=["observation"]))
        return env

    return _make
