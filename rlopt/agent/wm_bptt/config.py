"""Configuration of the world-model BPTT agent (``WMBPTT``)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rlopt.config_base import RLOptConfig


@dataclass
class WMBPTTConfig:
    """SuperMimic-style model-based actor-critic (arXiv 2610.28480, Table A.5).

    Each iteration collects a few vectorized simulator steps with the
    deterministic actor plus Gaussian action noise, fits the dynamics model on
    k-step autoregressive replay windows, then improves the actor by
    backpropagating an H-step imagined return through the frozen model, with a
    critic bootstrapping past H.
    """

    # -- task model -------------------------------------------------------
    task_model: str = "imitation_experiments.world_model.g1_task:G1TrackingTask"
    """``module:Class`` of the task model (state, features, reward, termination)."""
    task_kwargs: dict[str, Any] = field(default_factory=dict)
    """Extra keyword arguments for the task model constructor."""
    goal: str = "explicit"
    """Policy goal input, the task model's ``goal_mode`` (G1: "explicit"
    reference frames or "z", a frozen skill encoder's command). Passed only
    when not "explicit"; a plain field because Hydra cannot override entries
    of ``task_kwargs``."""
    goal_encoder: str = ""
    """Frozen skill-encoder checkpoint for ``goal="z"`` (task kwarg
    ``skill_encoder``)."""
    sensor_noise: bool = True
    """Add the task's sensor noise to policy features (collection, imagination
    and the deployed policy alike)."""

    # -- data collection ----------------------------------------------------
    env_steps_per_iter: float = 2.5
    """Vectorized env steps per iteration; the fraction carries over."""
    action_noise_std: float = 0.1
    """Std of the Gaussian noise added to collected actions (model-fit only)."""
    action_clip: float = 10.0
    """Clamp on executed actions."""
    replay_steps: int = 1000
    """Ring length per env."""

    # -- world model --------------------------------------------------------
    wm_hidden: list[int] = field(default_factory=lambda: [2048, 2048, 2048])
    wm_window: int = 8
    wm_batches_per_iter: int = 10
    wm_batch_size: int = 2048
    wm_lr: float = 2.0e-4
    wm_weight_decay: float = 0.0
    wm_grad_clip_norm: float = 1.0
    wm_warmup_iters: int = 500
    """World-model-only iterations before the first policy update."""
    min_replay_steps: int = 40
    """Steps per env in the replay before the first world-model update."""
    norm_freeze_samples: float = 2.0e8
    """Stop updating the input/increment normalizers after this many samples."""

    # -- imagination / actor --------------------------------------------------
    horizon: int = 32
    imagine_batch: int = 4096
    gamma: float = 0.99
    gae_lambda: float = 0.95
    imagine_termination: bool = True
    """Cut the imagined return (and the critic's TD targets) with the task's
    soft tracking termination. False keeps only the clip end: an ablation of
    the differentiable terminal cost; the env's real terminations still shape
    the replay the way they do for PPO."""
    termination_gradient: bool = True
    """Let the soft termination carry gradient to the actor. False detaches
    it: the imagined return is still cut at a failure, but the actor learns
    nothing from the threshold itself (ablation)."""
    critic_bootstrap: bool = True
    """Add gamma^H * alive_H * V(s_H) to the imagined return. False trains the
    actor on the H-step return alone (ablation; the critic is still fit)."""
    actor_horizon: int = 0
    """Steps of the imagined rollout the actor's objective (and so its BPTT
    gradient) covers, bootstrapped with V at that step; 0 means ``horizon``.
    The rollout still runs ``horizon`` steps for the critic's TD(lambda)
    targets, so a smaller value shortens only the actor's gradient path
    (ablation: ``horizon`` alone shortens both)."""
    actor_hidden: list[int] = field(default_factory=lambda: [1024, 1024, 512, 256])
    actor_activation: str = "elu"
    """MLP activation by name (elu, silu, relu, gelu, tanh)."""
    actor_lr: float = 2.0e-4
    actor_betas: list[float] = field(default_factory=lambda: [0.9, 0.95])
    actor_grad_clip_norm: float = 1.0
    actor_grad_clip_value: float = 0.1
    actor_output_scale: float = 0.01
    """Gain on the actor's last-layer initialization."""

    # -- critic -----------------------------------------------------------------
    critic_hidden: list[int] = field(default_factory=lambda: [512, 256])
    critic_activation: str = "elu"
    critic_lr: float = 2.0e-4
    critic_epochs: int = 5
    critic_minibatches: int = 4
    critic_update_every: int = 1
    """Fit the critic every N policy iterations (on that iteration's imagined
    rollout). With ``critic_epochs`` / ``critic_minibatches`` this sets the
    critic's gradient steps per env frame; 10 / 3 / 1 reproduces PPO
    combo50b's 3 full-batch steps per 24-step rollout (ablation)."""
    critic_grad_clip_norm: float = 1.0

    # -- schedule ---------------------------------------------------------------
    lr_cosine_iters: int = 250_000
    lr_final_ratio: float = 0.03
    bf16: bool = True
    """bf16 autocast for the MLP matmuls (integration, FK, reward stay fp32)."""
    compile: bool = False
    """``torch.compile`` the imagined control step (actor, model, reward)."""
    log_interval_iters: int = 50
    eval_window: int = 32
    """Open-loop window used for the logged k-step prediction error."""
    eval_batch: int = 1024


@dataclass
class WMBPTTRLOptConfig(RLOptConfig):
    """RLOpt config wrapper; the agent's hyperparameters live under ``wm``."""

    wm: WMBPTTConfig = field(default_factory=WMBPTTConfig)

    def __post_init__(self) -> None:
        # The agent builds its own actor, critic and dynamics model from
        # ``wm``; the generic value-function slot stays empty.
        self.value_function = None
