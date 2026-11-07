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
    wm_opt: str = "adam"
    """``adam``: AdamW on every parameter. ``muon``: Muon on the hidden weight
    matrices of the MLP (``dynamics_model=mlp``), Adam with betas (0.9, 0.999)
    on the rest, both at ``wm_lr``."""
    wm_weight_decay: float = 0.0
    wm_grad_clip_norm: float = 1.0
    wm_warmup_iters: int = 500
    """World-model-only iterations before the first policy update."""
    dynamics_model: str = "mlp"
    """``mlp``: SuperMimic's ``DynamicsModel``. ``diffsr``: ``DiffSRDynamics``
    with its mean read-out W phi + b as f, phi trained by the denoising loss on
    every window transition plus the k-step loss on the read-out (the
    ``diffsr_k8`` arm of ``dynamics_compare``)."""
    diffsr_feature_dim: int = 128
    diffsr_embed_dim: int = 512
    diffsr_num_noises: int = 16
    diffsr_denoise_coeff: float = 1.0
    """Weight of the denoising loss beside the k-step read-out loss."""
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
    critic_fit: bool = True
    """Fit the critic each policy iteration. False skips the fit and, with
    ``critic_bootstrap=False``, every critic forward: no value function at all,
    so the iteration time carries no critic cost (ablation)."""
    actor_horizon: int = 0
    """Steps of the imagined rollout the actor's objective (and so its BPTT
    gradient) covers, bootstrapped with V at that step; 0 means ``horizon``.
    The rollout still runs ``horizon`` steps for the critic's TD(lambda)
    targets, so a smaller value shortens only the actor's gradient path
    (ablation: ``horizon`` alone shortens both)."""
    actor_estimator: str = "pathwise"
    """``pathwise``: the actor's gradient flows through the model's Jacobians
    (SuperMimic, SVG). ``score_function``: the same imagined rollouts run on
    sampled actions ``a_t = mu(o_t) + sigma eps_t`` (``sf_action_std``) and the
    actor is updated by the likelihood-ratio estimator
    ``sum_t gamma^t c_t A_t grad log N(a_t; mu(o_t), sigma^2)`` with per-step
    advantages ``A_t = TD(lambda) target - V(s_t)`` from the same critic
    (REINFORCE with a GAE baseline, one gradient step per rollout, no
    clipping); the model is stepped without gradient. Ablation: same model,
    critic, data, horizon and compute, estimator only."""
    rebar_weight: float = 1.0
    """``actor_estimator=rebar``: scale of the score-function correction added to
    the mean-path SVG(H) gradient (``wm_bptt/rebar.py``). Needs
    ``actor_source=real``."""
    rebar_ema_rate: float = 0.05
    """Rate of the correction's running mean, about 2 / rate iterations."""
    rebar_norm_cap: float = 0.5
    """The applied correction never exceeds this multiple of the SVG gradient."""
    rebar_clip_eps: float = 0.2
    """PPO ratio clip on the score term."""
    rebar_shuffle: bool = False
    """Control: permute the residuals across rows, which keeps the correction's
    size and noise and removes its signal."""
    sf_action_std: float = 0.1
    """Sampling std of the score-function actor (the collection noise)."""
    sf_normalize_advantage: bool = True
    """Standardize the per-step advantages over the batch (PPO's practice)."""
    actor_source: str = "imagined"
    """``imagined``: the actor's H-step objective runs on the model's own
    rollout from a replay state (SuperMimic). ``real``: it runs along an
    H-step replay window re-simulated with the recorded noise held fixed
    (SVG noise inference): ``a_t = pi(o_t) + eta_t`` with the collector's
    action noise ``eta_t = a_t - mu_old(o_t)``, ``o_t`` carrying the
    collector's own sensor noise (keyed to the replay row), and
    ``x_{t+1} = integrate(x_t, f(x_t, a_t) + xi_t)`` with the model residual
    ``xi_t = increment(s_t, s_{t+1}) - f(s_t, a_t)``. At the collecting policy
    the forward pass is the recording; the gradient flows through the model's
    Jacobians. Episode ends are the env's: a termination cuts the return (with
    the env's reward for the last transition), a timeout bootstraps V at the
    last state."""
    critic_source: str = "imagined"
    """``imagined``: TD(lambda) targets on the imagined rollout. ``real``:
    targets on recorded replay windows (the behaviour policy's returns, with
    the env's episode ends as for ``actor_source=real``)."""
    real_window_recent_steps: int = 50
    """Real windows start within the newest N rows per env, so the recorded
    actions come from recent policies (50 rows = 20 iterations at 2.5 env
    steps per iteration)."""
    actor_hidden: list[int] = field(default_factory=lambda: [1024, 1024, 512, 256])
    actor_activation: str = "elu"
    """MLP activation by name (elu, silu, relu, gelu, tanh)."""
    actor_lr: float = 2.0e-4
    actor_opt: str = "adam"
    """``adam``: Adam on every parameter. ``muon``: Muon on the actor's hidden
    weight matrices, Adam with ``actor_betas`` on the rest, both at ``actor_lr``."""
    actor_betas: list[float] = field(default_factory=lambda: [0.9, 0.95])
    actor_lr_mult: float = 1.0
    """The actor's learning rate is ``actor_lr`` times the schedule times this,
    on Muon and Adam groups alike. The critic and world model do not get it."""
    actor_lr_decay_return: float | None = None
    """Once the training return reaches this at a log row, ``actor_lr_mult``
    ramps linearly to 1 over ``actor_lr_decay_len`` iterations. The return is
    an ended episode's mean step reward over the task's ``max_step_reward``,
    averaged per step, then over the steps that ended an episode."""
    actor_lr_decay_len: int = 0
    """Length of that ramp in iterations."""
    actor_lr_decay_by: int | None = None
    """The ramp starts at this iteration at the latest."""
    actor_grad_clip_norm: float = 1.0
    actor_grad_clip_value: float = 0.1
    actor_output_scale: float = 0.01
    """Gain on the actor's last-layer initialization."""
    actor_ema: float = 0.0
    """Decay of an exponential moving average of the actor, one step per actor
    update; 0 is off. Collection and imagination use the live actor. The
    checkpoint's ``policy_state_dict`` holds the average, which is what
    evaluators score; ``policy_live_state_dict`` holds the live actor."""

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
    fused_adam: bool = False
    """Fused Adam kernels for every optimizer: one launch per step. Changes
    rounding, so a resumed run must match its checkpoint."""
    lr_cosine_iters: int = 250_000
    lr_final_ratio: float = 0.03
    bf16: bool = True
    """bf16 autocast for the MLP matmuls (integration, FK, reward stay fp32)."""
    compile: bool = False
    """``torch.compile`` the imagined control step (actor, model, reward)."""
    log_interval_iters: int = 50
    ckpt_light: bool = False
    """Periodic checkpoints (``save_interval``) hold only the deployed policy and
    the counters, as ``policy_step_<frames>.pt``, written by a background
    thread. They score but cannot resume. The run's end still writes the full
    ``model_step_<frames>.pt``."""
    last_every_min: float = 0.0
    """With ``ckpt_light``: also write the full checkpoint, on the same thread,
    at the next periodic checkpoint once this many training minutes have passed
    since the last one, so a stopped run can resume. 0: only at the run's end."""
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


def validate_options(config: WMBPTTRLOptConfig, task: Any) -> None:
    """Refuse option values and combinations ``WMBPTT`` cannot honour.

    The one place for option dependencies (contract 3 of
    ``wiki/code-opt-ndm-port.md``): a combination that would do nothing, or do
    something else than it says, fails here at construction. The default
    config passes. Env-side pairs belong in ``ImitationG1V3EnvCfg``.
    """
    wm = config.wm
    if not 0.0 <= wm.actor_ema < 1.0:
        msg = f"wm.actor_ema must be in [0, 1), got {wm.actor_ema}."
        raise ValueError(msg)
    if wm.ckpt_light and int(config.save_interval) <= 0:
        msg = "wm.ckpt_light changes the periodic checkpoints; set save_interval, or there are none."
        raise ValueError(msg)
    if wm.last_every_min < 0:
        msg = f"wm.last_every_min must not be negative, got {wm.last_every_min}."
        raise ValueError(msg)
    if wm.last_every_min > 0 and not wm.ckpt_light:
        msg = "wm.last_every_min needs wm.ckpt_light: without it every periodic checkpoint is already full."
        raise ValueError(msg)
    if wm.actor_lr_mult <= 0:
        msg = f"wm.actor_lr_mult must be positive, got {wm.actor_lr_mult}."
        raise ValueError(msg)
    if wm.actor_lr_decay_return is None:
        if wm.actor_lr_decay_by is not None or wm.actor_lr_decay_len != 0:
            msg = "wm.actor_lr_decay_by and wm.actor_lr_decay_len need wm.actor_lr_decay_return."
            raise ValueError(msg)
    else:
        if wm.actor_lr_mult == 1:
            msg = "wm.actor_lr_decay_return decays wm.actor_lr_mult, which is 1."
            raise ValueError(msg)
        if wm.actor_lr_decay_return <= 0:
            msg = f"wm.actor_lr_decay_return must be positive, got {wm.actor_lr_decay_return}."
            raise ValueError(msg)
        if wm.actor_lr_decay_len < 1:
            msg = f"wm.actor_lr_decay_len must be at least 1 with a trigger, got {wm.actor_lr_decay_len}."
            raise ValueError(msg)
        if wm.actor_lr_decay_by is not None and wm.actor_lr_decay_by < 1:
            msg = f"wm.actor_lr_decay_by must be at least 1, got {wm.actor_lr_decay_by}."
            raise ValueError(msg)
        if getattr(task, "max_step_reward", None) is None:
            msg = "wm.actor_lr_decay_return needs the task's max_step_reward, which scales the training return."
            raise ValueError(msg)
    for name in ("actor_opt", "wm_opt"):
        if getattr(wm, name) not in ("adam", "muon"):
            msg = f"wm.{name} must be 'adam' or 'muon', got {getattr(wm, name)!r}."
            raise ValueError(msg)
    if wm.wm_opt == "muon":
        if wm.dynamics_model != "mlp":
            msg = f"wm.wm_opt=muon runs on the MLP world model, not dynamics_model={wm.dynamics_model!r}."
            raise ValueError(msg)
        if wm.wm_weight_decay != 0:
            msg = "wm.wm_opt=muon has no weight decay; set wm.wm_weight_decay to 0."
            raise ValueError(msg)
    # Muon skips each MLP's first and last Linear: one hidden layer leaves it nothing.
    for name, hidden, opt in (("wm_hidden", wm.wm_hidden, wm.wm_opt), ("actor_hidden", wm.actor_hidden, wm.actor_opt)):
        if opt == "muon" and len(hidden) < 2:
            msg = f"Muon needs a hidden weight matrix: wm.{name} must have at least two layers, got {list(hidden)}."
            raise ValueError(msg)
    for name in ("actor_source", "critic_source"):
        if getattr(wm, name) not in ("imagined", "real"):
            msg = f"wm.{name} must be 'imagined' or 'real', got {getattr(wm, name)!r}."
            raise ValueError(msg)
    if wm.actor_source == "real" and 0 < wm.actor_horizon < wm.horizon:
        msg = "wm.actor_source=real uses the window length wm.horizon; leave wm.actor_horizon at 0."
        raise ValueError(msg)
    if wm.actor_estimator not in ("pathwise", "score_function", "rebar"):
        msg = f"wm.actor_estimator must be 'pathwise', 'score_function' or 'rebar', got {wm.actor_estimator!r}."
        raise ValueError(msg)
    if wm.actor_estimator == "rebar" and wm.actor_source != "real":
        msg = "wm.actor_estimator=rebar corrects the real-window gradient (actor_source=real)."
        raise ValueError(msg)
    if wm.actor_estimator == "score_function" and wm.actor_source == "real":
        msg = "wm.actor_estimator=score_function runs on imagined rollouts only (actor_source=imagined)."
        raise ValueError(msg)
