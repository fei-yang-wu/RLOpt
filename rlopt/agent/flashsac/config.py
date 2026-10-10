"""Configuration of the FlashSAC agent shell.

The nested ``FlashSACConfig`` holds the settings of the collect-train loop.
The learner hyper-parameters live in ``FlashSACLearnerConfig`` (``learner.py``).
A few learner fields duplicate settings that the shell owns; the agent
overwrites them so one value has one source:

==========================  =====================================
learner field               set from
==========================  =====================================
``batch_size``              ``flashsac.batch_size``
``gamma``                   ``loss.gamma``
``n_step``                  ``flashsac.replay.n_step``
``seed``                    ``seed``
``action_bound``            the resolved action bound (see below)
==========================  =====================================

Action bound. The policy returns ``bound * tanh(u)``. The bound comes from
``flashsac.action_bound`` (a scalar or one value per action dimension). When it
is ``None`` the bound is read from a *bounded, symmetric* env action spec. An
unbounded spec (the G1 envs) without ``flashsac.action_bound`` is an error.

Update schedule. With ``N`` envs, batch size ``B`` and replay ratio ``rho``
(sampled transitions per collected transition) the agent owes
``U = rho * N / B`` updates per env step. It runs exactly
``floor((t + 1) * U) - floor(t * U)`` updates at env step ``t`` (integer
arithmetic on a ``Fraction``), so the realised count never drifts from ``U``.
``updates_per_step`` overrides ``U`` directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from rlopt.agent.flashsac.learner import FlashSACLearnerConfig
from rlopt.config_base import RLOptConfig
from rlopt.config_utils import ObsKey

__all__ = [
    "FlashSACConfig",
    "FlashSACRLOptConfig",
    "FlashSACReplayConfig",
]

_OBS_DTYPES = ("float32", "bfloat16", "float16")
_WARMUP_MODES = ("uniform", "normal")
_TERMINAL_MODES = ("auto", "final", "next")


@dataclass
class FlashSACReplayConfig:
    """Time-major GPU replay ring (``TimeMajorReplay``)."""

    capacity_steps: int = 2048
    """Ring rows. One row is one vector step (all envs). Slots = rows x num_envs."""

    n_step: int = 3
    """n-step return window, applied at sample time."""

    obs_dtype: str = "float32"
    """Storage dtype of observations: ``float32``, ``bfloat16`` or ``float16``."""

    pool_fraction: float = 0.05
    """Size of the terminal-observation pool as a fraction of the ring slots.

    Only time-out rows take an entry (a termination never bootstraps), so it
    must exceed the time-out share of the rows. ``health/replay_pool_miss``
    counts sampled windows whose entry was overwritten."""

    terminal_obs: str = "auto"
    """Where the observation after an episode end comes from.

    ``final``: the ``("next", "final", ...)`` group written by the terminal
    observation reader, for an autoreset env whose ``next`` observation is the
    reset one. ``next``: the ``("next", ...)`` observation itself. ``auto``:
    ``final`` when the first collected step has the ``final`` group, else ``next``.

    Isaac Lab through ``IsaacLabWrapper`` + ``attach_terminal_obs_reader``
    resolves to ``next``: TorchRL's ``VecGymEnvTransform`` swaps the reader's
    ``final`` group into ``next`` on done rows and pops the group, so ``next``
    already holds the terminal observation. ``ImitationRLEnvV2`` publishes the
    pre-step observation buffer as the terminal observation, i.e. the row's
    own observation ``s_t`` ("stale" mode of wiki/flashsac-ipmd-plan.md), not
    a post-physics ``s_{t+1}``. Without the wrapper fix ``next`` on done rows
    was the reset observation; ``health/final_obs_missing`` counts all-zero
    terminal observations as a guard.
    """

    final_group: str = "final"
    """Name of the terminal observation group."""


@dataclass
class FlashSACConfig:
    """Collect-train loop settings of the FlashSAC agent."""

    learner: FlashSACLearnerConfig = field(default_factory=FlashSACLearnerConfig)
    """Learner hyper-parameters (some fields are overwritten, see module doc)."""

    replay: FlashSACReplayConfig = field(default_factory=FlashSACReplayConfig)
    """Replay ring settings."""

    critic_input_keys: list[ObsKey] | None = None
    """Critic observation keys. ``None`` means the same keys as the actor."""

    action_bound: float | list[float] | None = None
    """Symmetric action bound(s) in env units; see module doc."""

    batch_size: int = 2048
    """Transitions per learner update."""

    rho: float = 4.0
    """Replay ratio: sampled transitions per collected transition."""

    updates_per_step: float | None = None
    """Updates per env step. ``None`` means ``rho * num_envs / batch_size``."""

    warmup_transitions: int = 100_000
    """Transitions collected with random actions before the first update.

    On resume the ring is empty: the agent collects this many transitions with
    the *loaded policy* (not random actions) before updates restart.
    """

    warmup_mode: str = "uniform"
    """Random warm-up action law.

    ``uniform`` (the reference): tanh-space action uniform in [-1, 1], i.e.
    uniform in ``[-B_j, B_j]`` env units. ``normal`` (clipped normal): N(0, 1)
    in env units per joint, clipped to ``[-B_j, B_j]``; the stored tanh-space
    action is ``a / B_j``. With wide per-joint bounds (G1 wrist yaw 24.5) the
    uniform law commands extreme targets; the clipped normal keeps the
    warm-up at the scale a PPO policy starts at (std 1). See DV3 of
    wiki/flashsac-ipmd-plan.md."""

    noise_repeat: bool = True
    """Hold exploration noise for a zeta-distributed number of steps per env."""

    noise_zeta_s: float = 2.0
    """Exponent of the truncated zeta law of the noise repeat length."""

    noise_max_repeat: int = 16
    """Longest noise repeat."""

    log_interval_steps: int = 500
    """Env steps between metric logs (one device-to-host copy per log)."""

    checkpoint_interval_s: float = 0.0
    """Wall-clock seconds between extra checkpoints. ``0`` disables."""

    health_checks: bool = True
    """Compute the HC metrics (edge mass, saturation, replay age, BN gap) at each log."""

    fail_on_nonfinite: bool = True
    """Raise at the next log when a loss is not finite."""

    def updates_per_env_step(self, num_envs: int) -> float:
        """Return ``U``, the (possibly fractional) number of updates per env step."""
        if self.updates_per_step is not None:
            return float(self.updates_per_step)
        return float(self.rho) * float(num_envs) / float(self.batch_size)

    def validate(self, num_envs: int) -> None:
        """Raise ``ValueError`` for a setting the agent cannot run."""
        errors: list[str] = []
        rep = self.replay
        if num_envs < 1:
            errors.append(f"num_envs must be >= 1, got {num_envs}")
        if self.batch_size < 1:
            errors.append(f"batch_size must be >= 1, got {self.batch_size}")
        if self.updates_per_step is None:
            if not (math.isfinite(self.rho) and self.rho > 0.0):
                errors.append(f"rho must be positive and finite, got {self.rho}")
        elif not (
            math.isfinite(self.updates_per_step) and self.updates_per_step >= 0.0
        ):
            errors.append(
                f"updates_per_step must be >= 0 and finite, got {self.updates_per_step}"
            )
        if rep.n_step < 1:
            errors.append(f"replay.n_step must be >= 1, got {rep.n_step}")
        min_rows = 2 * (max(rep.n_step, 1) + 1)
        if rep.capacity_steps < min_rows:
            errors.append(
                f"replay.capacity_steps={rep.capacity_steps} is below 2*(n_step+1)="
                f"{min_rows}: the n-step windows and the guard rows would leave no "
                "valid start row."
            )
        slots = rep.capacity_steps * max(num_envs, 1)
        if slots < self.batch_size:
            errors.append(
                f"replay holds {slots} slots (capacity_steps*num_envs) but "
                f"batch_size={self.batch_size}; raise replay.capacity_steps to at "
                f"least {math.ceil(self.batch_size / max(num_envs, 1))}."
            )
        if rep.obs_dtype not in _OBS_DTYPES:
            errors.append(f"replay.obs_dtype must be one of {_OBS_DTYPES}")
        if not 0.0 < rep.pool_fraction <= 1.0:
            errors.append("replay.pool_fraction must be in (0, 1]")
        if rep.terminal_obs not in _TERMINAL_MODES:
            errors.append(f"replay.terminal_obs must be one of {_TERMINAL_MODES}")
        if self.warmup_transitions < 0:
            errors.append("warmup_transitions must be >= 0")
        if self.warmup_mode not in _WARMUP_MODES:
            errors.append(f"warmup_mode must be one of {_WARMUP_MODES}")
        if self.noise_max_repeat < 1:
            errors.append("noise_max_repeat must be >= 1")
        if self.log_interval_steps < 1:
            errors.append("log_interval_steps must be >= 1")
        if self.checkpoint_interval_s < 0:
            errors.append("checkpoint_interval_s must be >= 0")
        if self.action_bound is not None:
            values = (
                [self.action_bound]
                if isinstance(self.action_bound, (int, float))
                else list(self.action_bound)
            )
            if not values or any(
                not (math.isfinite(float(v)) and float(v) > 0.0) for v in values
            ):
                errors.append("action_bound entries must be positive and finite")
        if errors:
            raise ValueError("Invalid FlashSAC config:\n  - " + "\n  - ".join(errors))


@dataclass
class FlashSACRLOptConfig(RLOptConfig):
    """FlashSAC configuration that extends ``RLOptConfig``.

    The actor input keys are ``policy.input_keys``. The critic input keys are
    ``flashsac.critic_input_keys`` (``None`` means the actor keys).
    ``collector.frames_per_batch`` is the number of envs: the entry point
    multiplies the per-env value by ``num_envs``, so the default is 1.
    """

    flashsac: FlashSACConfig = field(default_factory=FlashSACConfig)
    """FlashSAC settings."""

    def __post_init__(self) -> None:
        # The shell owns warm-up; the TorchRL collector must not add random frames.
        self.collector.init_random_frames = 0
        self.collector.frames_per_batch = 1
        self.collector.no_cuda_sync = True
        self.compile.compile = False
        self.save_interval = 5_000_000
