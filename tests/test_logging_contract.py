# ruff: noqa: E402 -- filter upstream import deprecations before importing TorchRL.
"""The training-log contract: one write path, one row shape, nothing dropped.

Each algorithm that can run here trains for a few iterations against a backend
double that applies wandb's step rule, with the progress bar on and
``trainer.log_interval`` far beyond the run. That is the configuration of job
6808096, whose PPO run kept wandb empty while ``rlopt.log`` filled. Algorithms
that cannot run here are covered by the source scan at the end of the file.
"""

from __future__ import annotations

import ast
import logging
import warnings
from collections import defaultdict, deque
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

warnings.filterwarnings(
    "ignore",
    message="`torch.jit.script_method` is deprecated.*",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message="Creating .* which inherits from WeightUpdaterBase is deprecated.*",
    category=DeprecationWarning,
)

import pytest
import torch
from tensordict import TensorDict
from torchrl.envs import (
    Compose,
    GymEnv,
    RenameTransform,
    RewardSum,
    SerialEnv,
    StepCounter,
    Transform,
    TransformedEnv,
)

import rlopt
from rlopt.agent import (
    AMP,
    ASE,
    GAIL,
    IPMD,
    IPMDL2T,
    IPMDSR,
    PPO,
    SAC,
    AMPRLOptConfig,
    FastSAC,
    FastSACRLOptConfig,
    FastTD3,
    FastTD3RLOptConfig,
    FlashSAC,
    FlashSACRLOptConfig,
    GAILRLOptConfig,
    IPMDBilinear,
    IPMDRLOptConfig,
    IPMDSRRLOptConfig,
    PPORLOptConfig,
    SACRLOptConfig,
)
from rlopt.agent.ipmd.ipmd_dagger import IPMDDagger
from rlopt.agent.wm_bptt.agent import WMBPTT
from rlopt.base_class import BaseAlgorithm, TrainingMetadata
from rlopt.config_base import RLOptConfig
from rlopt.logging_utils import (
    REQUIRED_TRAINING_ENTRIES,
    ROOT_LOGGER_NAME,
    LoggingManager,
    MetricReporter,
    missing_training_entries,
)

pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:The env passed to Collector is missing transforms.*:FutureWarning"
    ),
]

ENV_ENTRIES = (
    "Metrics/reference/mpjpe_l_mm",
    "Episode_Termination/tracking_failed",
)
AGENT_DIR = Path(rlopt.__file__).parent / "agent"


class _WandbRun:
    """wandb's step rule: a row below the current step is discarded."""

    settings = None

    def __init__(self) -> None:
        self.step = 0
        self.rows: list[tuple[dict, int]] = []
        self.discarded: list[dict] = []

    def define_metric(self, *_args, **_kwargs) -> None:
        return

    def log(self, data, step=None, commit=True) -> None:  # noqa: ARG002
        if step < self.step:
            self.discarded.append(dict(data))
            return
        self.rows.append((dict(data), step))
        self.step = step + 1


class _WandbBackend:
    def __init__(self) -> None:
        self.experiment = _WandbRun()


class _PublishEnvLog(Transform):
    """Replace the newest log payload on every step, as IsaacLabWrapper does."""

    def __init__(self, channel: deque) -> None:
        super().__init__()
        self.channel = channel

    def _step(self, tensordict, next_tensordict):
        self.channel.append({key: torch.tensor(0.5) for key in ENV_ENTRIES})
        return next_tensordict


def _logging_env(base=None, *transforms) -> TransformedEnv:
    channel: deque = deque(maxlen=1)
    env = TransformedEnv(
        GymEnv("Pendulum-v1") if base is None else base,
        Compose(StepCounter(), RewardSum(), *transforms, _PublishEnvLog(channel)),
    )
    env.log_infos = channel
    env.required_log_entries = ENV_ENTRIES
    return env


def _base_cfg(cfg, tmp_path: Path):
    cfg.env.env_name = "Pendulum-v1"
    cfg.env.device = cfg.device = "cpu"
    cfg.env.num_envs = 1
    cfg.collector.frames_per_batch = 16
    cfg.collector.total_frames = 64
    cfg.replay_buffer.size = 64
    cfg.loss.mini_batch_size = 16
    cfg.loss.epochs = 1
    cfg.compile.compile = False
    cfg.logger.backend = ""
    cfg.logger.log_dir = str(tmp_path)
    cfg.logger.log_to_console = False
    cfg.save_interval = 0
    # Job 6808096: the bar on, the text cadence longer than the run.
    cfg.trainer.progress_bar = True
    cfg.trainer.log_interval = 10**9
    for network in (cfg.policy, cfg.value_function, cfg.q_function):
        if network is not None:
            network.input_keys = ["observation"]
            network.num_cells = [8]
    return cfg


def _no_reward_estimation(cfg) -> None:
    ipmd = cfg.ipmd
    ipmd.use_latent_command = False
    ipmd.reward_input_keys = ["observation"]
    ipmd.reward_loss_coeff = ipmd.reward_l2_coeff = 0
    ipmd.reward_grad_penalty_coeff = ipmd.reward_logit_reg_coeff = 0
    ipmd.reward_param_weight_decay_coeff = ipmd.bc_coef = 0
    ipmd.rollout_bc_coef = ipmd.diversity_bonus_coeff = 0


def _flashsac(cfg) -> None:
    cfg.collector.frames_per_batch = 1
    cfg.collector.total_frames = 96
    fs = cfg.flashsac
    fs.batch_size = fs.warmup_transitions = 16
    fs.rho = 2.0
    fs.replay.capacity_steps = 128
    fs.log_interval_steps = 32
    fs.learner.actor_dim = fs.learner.critic_dim = 16
    fs.learner.amp_dtype = "fp32"


def _fasttd3(cfg) -> None:
    cfg.env.num_envs = cfg.collector.frames_per_batch = 2
    cfg.collector.init_random_frames = cfg.fasttd3.batch_size = 8
    cfg.policy.input_keys = cfg.q_function.input_keys = ["policy"]


def _fasttd3_env() -> TransformedEnv:
    return _logging_env(
        SerialEnv(2, lambda: GymEnv("Pendulum-v1")),
        RenameTransform(["observation"], ["policy"], create_copy=False),
    )


def _install_expert(agent) -> None:
    states, actions = torch.randn(64, 3), torch.randn(64, 1)

    def sample(batch_size, required_keys):
        batch = TensorDict(
            {"observation": states, "action": actions, ("next", "observation"): states},
            batch_size=[64],
        )
        return batch[:batch_size].select(*required_keys).clone()

    agent._set_test_expert_batch_sampler(sample)


# (class, config, config patch, env factory, setup, expected training rows)
RUNNABLE = {
    "PPO": (PPO, PPORLOptConfig, None, _logging_env, None, 4),
    "IPMD": (IPMD, IPMDRLOptConfig, _no_reward_estimation, _logging_env, None, 4),
    "IPMDSR": (IPMDSR, IPMDSRRLOptConfig, _no_reward_estimation, _logging_env, None, 4),
    "SAC": (SAC, SACRLOptConfig, None, _logging_env, None, 4),
    "FastSAC": (FastSAC, FastSACRLOptConfig, None, _logging_env, None, 4),
    "FlashSAC": (FlashSAC, FlashSACRLOptConfig, _flashsac, _logging_env, None, 3),
    "FastTD3": (FastTD3, FastTD3RLOptConfig, _fasttd3, _fasttd3_env, None, 3),
    "GAIL": (GAIL, GAILRLOptConfig, None, _logging_env, _install_expert, 4),
    "AMP": (AMP, AMPRLOptConfig, None, _logging_env, _install_expert, 4),
}


def _train(name: str, tmp_path: Path) -> tuple[_WandbRun, int]:
    agent_cls, config_cls, patch, make_env, setup, rows = RUNNABLE[name]
    cfg = _base_cfg(config_cls(), tmp_path)
    if patch is not None:
        patch(cfg)
    agent = agent_cls(make_env(), cfg)
    backend = _WandbBackend()
    # Several constructors take no ``logger``; the reporter is what they write to.
    agent.metrics = MetricReporter(backend, agent.log)
    if setup is not None:
        setup(agent)
    agent.train()
    return backend.experiment, rows


@pytest.mark.parametrize("name", sorted(RUNNABLE))
def test_every_row_meets_the_contract_on_the_frame_axis(name, tmp_path):
    run, expected_rows = _train(name, tmp_path)
    rows = [row for row, _ in run.rows]

    assert not run.discarded
    assert len(rows) == expected_rows, "a training row was gated or dropped"
    for row in rows:
        assert not missing_training_entries(row, (*REQUIRED_TRAINING_ENTRIES, *ENV_ENTRIES))

    frames = [row["train/cumulative_env_frames"] for row in rows]
    assert frames == sorted(set(frames))
    assert [row["env_frames"] for row in rows] == frames
    assert [step for _, step in run.rows] == frames
    assert [row["train/iteration"] for row in rows] == sorted(
        {row["train/iteration"] for row in rows}
    )
    assert rows[-1]["time/train_elapsed_s"] > 0.0
    assert all(row["Metrics/reference/mpjpe_l_mm"] == 0.5 for row in rows)


def test_a_file_backend_receives_the_same_rows(tmp_path):
    cfg = _base_cfg(PPORLOptConfig(), tmp_path)
    cfg.logger.backend = "csv"
    PPO(_logging_env(), cfg).train()
    written = {}
    for entry in (*REQUIRED_TRAINING_ENTRIES, *ENV_ENTRIES):
        name = entry.rsplit("/", 1)[1]
        (path,) = tmp_path.rglob(f"{name}.csv")
        written[entry] = len(path.read_text().splitlines())
    assert set(written.values()) == {4}


def test_a_second_row_at_one_frame_count_is_kept():
    backend = _WandbBackend()
    reporter = MetricReporter(backend, None)
    reporter.log_scalars({"train/loss": 1.0}, step=100)
    reporter.log_scalars({"train/stopped_by_time_budget": 1.0}, step=100)
    reporter.log_scalars({"train/loss": 1.0}, step=200)

    run = backend.experiment
    assert not run.discarded
    assert [row["env_frames"] for row, _ in run.rows] == [100, 100, 200]
    assert [step for _, step in run.rows] == [100, 101, 200]
    assert reporter.rows_written == 3


def test_a_resumed_segment_below_the_run_step_is_kept():
    backend = _WandbBackend()
    backend.experiment.step = 500  # wandb restored the run's step on resume
    reporter = MetricReporter(backend, None)
    reporter.log_scalars({"train/loss": 1.0}, step=100)

    run = backend.experiment
    assert not run.discarded
    assert run.rows == [({"train/loss": 1.0, "env_frames": 100}, 500)]


@contextmanager
def _captured_warnings(caplog):
    """RLOpt's root logger does not propagate, so hand it pytest's handler."""
    root = logging.getLogger(ROOT_LOGGER_NAME)
    root.addHandler(caplog.handler)
    try:
        yield
    finally:
        root.removeHandler(caplog.handler)


def test_a_row_without_env_payload_reports_the_env_gap_once(tmp_path, caplog):
    cfg = _base_cfg(PPORLOptConfig(), tmp_path)
    env = TransformedEnv(GymEnv("Pendulum-v1"), Compose(StepCounter(), RewardSum()))
    env.required_log_entries = ENV_ENTRIES  # declared but never published
    agent = PPO(env, cfg)
    with _captured_warnings(caplog):
        agent.train()
    warned = [r.getMessage() for r in caplog.records if "omits required" in r.getMessage()]
    assert len(warned) == 1
    assert "Episode_Termination/tracking_failed" in warned[0]


def _bare_agent(backend: str = "") -> SimpleNamespace:
    """The attributes ``BaseAlgorithm``'s logging methods read."""
    config = RLOptConfig()
    config.logger.backend = backend
    return SimpleNamespace(config=config, metrics=None, log=logging.getLogger("t"))


def test_a_configured_backend_without_a_reporter_fails_loudly():
    with pytest.raises(RuntimeError, match="no metric reporter"):
        BaseAlgorithm.log_metrics(_bare_agent("wandb"), {"a": 1.0}, step=1)


def test_no_backend_without_a_reporter_stays_quiet():
    BaseAlgorithm.log_metrics(_bare_agent(""), {"a": 1.0}, step=1)


def test_logging_manager_refuses_a_backend_that_built_no_logger(monkeypatch, tmp_path):
    config = RLOptConfig()
    config.logger.backend = "wandb"
    config.logger.log_dir = str(tmp_path)
    config.logger.log_to_file = False
    monkeypatch.setattr(LoggingManager, "_build_metrics_logger", lambda self: None)
    with pytest.raises(RuntimeError, match="produced no metrics logger"):
        LoggingManager(config=config, component="PPO")


def test_file_summary_flags_a_backend_with_no_rows(tmp_path, caplog):
    agent = PPO(_logging_env(), _base_cfg(PPORLOptConfig(), tmp_path))
    agent.config.logger.backend = "wandb"
    agent.metrics = MetricReporter(_WandbBackend(), agent.log)
    metadata = TrainingMetadata(
        total_iterations=1, frames_processed=16, progress_bar_enabled=True
    )
    iteration = SimpleNamespace(iteration_idx=0, metrics={})
    with _captured_warnings(caplog):
        agent._log_iteration_file_summary(metadata, iteration)
    assert "has received no metric rows" in caplog.text


def test_wmbptt_flush_carries_the_common_entries():
    agent = WMBPTT.__new__(WMBPTT)
    agent.config = SimpleNamespace(device="cpu")
    agent._frames_processed = 8
    agent._episode_length = torch.zeros(2)
    agent._episode_return = torch.zeros(2)
    agent._episode_sums = torch.zeros(3)
    agent._wm_eval = dict
    agent._lr_factor = lambda: 1.0
    done = torch.tensor([False, True])
    for _ in range(3):
        agent._track_episodes(torch.tensor([1.0, 2.0]), done)

    stats = defaultdict(list)
    stats["env_reward"].append(torch.tensor(1.5))
    row = agent._flush(defaultdict(list), stats, 0.1)

    assert row["train/step_reward_mean"] == row["env/env_reward"] == 1.5
    # Only env 1 ended: three one-step episodes of reward 2.
    assert row["episode/length"] == 1.0
    assert row["episode/return"] == 2.0
    assert agent._episode_length.tolist() == [3.0, 0.0]
    assert agent._flush(defaultdict(list), defaultdict(list), 0.1).get("episode/length") is None


def test_wmbptt_reads_live_env_scalars_not_the_log_channel():
    agent = WMBPTT.__new__(WMBPTT)
    agent.env = SimpleNamespace(
        log_infos=deque([{"Metrics/reference/mpjpe_l_mm": 0.0}]),
        live_log_metrics=lambda: {"TrainHealth/x": torch.tensor(2.0)},
    )
    # The collected window average must not be replaced by a stale reset payload.
    assert agent._env_log_metrics() == {"TrainHealth/x": 2.0}


# ------------------------------------------------------------- source scan
# Includes the loops that cannot run here: ``l2t`` and ``infogail`` import the
# removed ``rlopt.configs``, and IPMDDiffSR's diffusion constructor is broken.
AUXILIARY_LOG_METRICS = {
    ("ipmd/ipmd_bilinear.py", "_offline_pretrain_policy_bc"): "before any env frame",
    ("ipmd/ipmd_bilinear.py", "_offline_pretrain_spectral_representation"): (
        "before any env frame"
    ),
}


def _enclosing_calls(tree: ast.Module):
    """Yield ``(function name, dotted callee)`` for every call in the module."""
    for scope in ast.walk(tree):
        if not isinstance(scope, ast.FunctionDef):
            continue
        for node in ast.walk(scope):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                owner = node.func.value
                prefix = owner.id if isinstance(owner, ast.Name) else ""
                yield scope.name, f"{prefix}.{node.func.attr}"


def _scan():
    for path in sorted(AGENT_DIR.rglob("*.py")):
        rel = path.relative_to(AGENT_DIR).as_posix()
        yield rel, path.read_text(), ast.parse(path.read_text())


def test_no_algorithm_writes_around_the_one_path():
    for rel, source, tree in _scan():
        assert ".log_scalar(" not in source, rel
        assert "log_infos" not in source, rel
        assert "_should_log_iteration(" not in source, f"{rel}: console cadence gates a row"
        for function, callee in _enclosing_calls(tree):
            if callee == "self.log_metrics":
                assert (rel, function) in AUXILIARY_LOG_METRICS, (rel, function)


def test_every_training_loop_reaches_the_one_path():
    """A ``train(self)`` writes rows, runs the shared ``record``, or delegates."""
    for rel, _source, tree in _scan():
        for node in tree.body:
            # Offline ``*Trainer`` classes fit on datasets: no env, no frame axis.
            if not isinstance(node, ast.ClassDef) or node.name.endswith("Trainer"):
                continue
            for method in node.body:
                if not isinstance(method, ast.FunctionDef):
                    continue
                # ``nn.Module.train(mode)`` takes an argument; a loop does not.
                loop = method.name == "train" and len(method.args.args) == 1
                if not (loop or method.name == "record"):
                    continue
                calls = {
                    call.func.attr
                    for call in ast.walk(method)
                    if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                }
                assert calls & {"log_training_metrics", "record", "train"}, (
                    rel,
                    node.name,
                    method.name,
                )


@pytest.mark.parametrize(
    "agent_cls", [ASE, IPMDBilinear, IPMDL2T, IPMDDagger, IPMDSR, AMP]
)
def test_algorithms_not_run_here_inherit_a_tested_record(agent_cls):
    assert agent_cls.record in (PPO.record, IPMD.record) or not hasattr(agent_cls, "record")
