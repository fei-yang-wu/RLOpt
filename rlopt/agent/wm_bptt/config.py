"""Configuration of the world-model BPTT agent (``WMBPTT``)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

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
    goal_table: bool = False
    """Compute the policy goal of every reference frame once and read its row, in
    the env and the model alike. Outside compiled code, the numbers of the
    computation it replaces, up to that computation's own rounding across batch
    shapes. 2.9 kB per frame: 1.2 GiB for LAFAN1."""
    reference_body_table: bool = False
    """The same for the reference body poses and twists, which FK would compute
    at every read. 1 kB per frame: 0.41 GiB for LAFAN1."""

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
    fit_terminal_transitions: bool = False
    """Fit each episode's last transition too, into the true state the step reached,
    as NDM Gym does. The next replay row cannot hold that state, because the env has
    reset, so the agent reads it from the env's batched ``final_obs`` into one more
    ring (284 B per env row: 1.16 GB at 1000 rows and 4096 envs). Windows stop after
    it. A non-finite or pushed final state is not fit (``wm/terminal_nonfinite``
    counts the first kind per step with an ended episode), and real windows
    (``actor_source``, ``critic_source``) keep their own episode masks."""
    norm_from_collection: bool = False
    """Fit the input and increment normalizers on every collected transition of
    every env, as NDM Gym does, not on the first transitions of the sampled windows.
    An episode's last transition counts, from the true state in ``final_obs``, with
    or without ``fit_terminal_transitions``. Pushed and non-finite rows do not count.
    ``norm_freeze_samples`` then counts collected transitions."""

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
    """The ramp starts at the first log row at or after this iteration, at the
    latest. Without ``actor_lr_decay_return`` it is the only trigger, which NDM
    Gym's ``PolicyLR`` permits (its train_muscles uses it)."""
    actor_grad_clip_norm: float = 1.0
    actor_grad_clip_value: float = 0.1
    actor_output_scale: float = 0.01
    """Gain on the actor's last-layer initialization."""
    actor_residual: bool = False
    """The actor outputs an offset about the reference pose of frame f + 1 (NDM
    Gym's ``--sm_residual``): the action is ``actor(obs) + task.reference_setpoint(obs)``,
    with the reference read from the raw policy observation's goal. That sum is the
    action everywhere: the simulator, the replay, the world model, the last-action
    observation and the action-rate term; collection adds its noise and the clamp
    after it. It applies in collection, imagination, the deployed policy and
    ``predict``, and ``policy_meta`` records it, so an inference load acts as the run
    did. Needs a task with ``reference_setpoint`` (G1-v3, explicit goal); refused
    with latent goals, ``actor_source=real`` and ``rebar``."""
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
    elementwise_3x3: bool = False
    """Float32 batched matrix products with every dimension at most 3 become
    fused elementwise products inside compiled functions, instead of a cuBLAS
    call each. Patches inductor for the whole process, before the first compile:
    any other agent in the process gets it too. Changes rounding."""
    compile_eager: bool = False
    """Also compile what runs eagerly around the compiled steps: the critic's
    input, the features of the world-model normalizers and the env's collection
    reward, each with ``dynamic=False``. Changes rounding."""
    combo_kernels: bool = False
    """Inductor launches independent kernels of a compiled graph together.
    Process-wide. On an RTX A4500 at the default sizes one measurement had it
    slower (693 against 668 ms per iteration) and a later one, with no other GPU
    client, faster (560 against 570), so measure it on the target GPU."""
    no_sync: bool = False
    """No host read in the update loop. The normalizers freeze on the device, and
    a window's valid rows are masked rather than indexed. A non-finite world-model
    or actor step is skipped on the device and counted (``wm/nonfinite``,
    ``actor/nonfinite``). Imagination starts are the first valid ones of 2B uniform
    candidates; a log row raises if an invalid one survived, which needs fewer
    than half of the replay's rows valid. Needs ``fused_adam``. Log rows still read
    the device."""
    update_graphs: bool = False
    """Capture the world-model update (every batch, from sampling to the optimizer
    step) and the actor-critic update (imagination, actor step, policy average,
    critic targets and every critic step) as one CUDA graph each, and replay them:
    two graph launches per iteration instead of thousands of kernel launches. Each
    update runs eagerly until it is ready (the replay ring full and the policy
    average started), then eagerly three more times on a side stream, then it is
    captured. The replay's write head lives
    on the device, and the optimizers are the capturable fused kernels with one
    device learning rate each, written in place every iteration, so the schedule and
    ``actor_lr_mult`` reach the graphs. The arithmetic is the ``no_sync`` loop's,
    except that a device learning rate is float32 where the plain fused kernel reads
    a float64, and Muon's step is the capturable form of ``MuonAdam``. The two graphs
    share one memory pool. A checkpoint load recaptures them. Needs ``no_sync``,
    ``fused_adam`` and a CUDA device; refused with ``critic_update_every`` > 1 (a host
    branch), the score-function estimator and the DiffSR model (not tested)."""
    graph_early: bool = False
    """With ``update_graphs``: the graphs do not wait for a full ring, so they warm up
    and capture at the first updates (NDM Gym's ``--graph_early``). The replay's fill
    level then lives on the device too, and sampling scales uniform floats by it
    (``rand * n``): the same distribution as ``torch.randint``, another random stream,
    so a run is not bit for bit the one without it."""
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


def validate_residual(task: Any, *, goal: str = "explicit", goal_encoder: str = "") -> None:
    """``wm.actor_residual``, or an inference load of a residual checkpoint: the task
    must read the reference pose from an explicit goal."""
    if goal != "explicit" or goal_encoder:
        msg = "wm.actor_residual reads the reference pose from the explicit goal; a latent goal does not hold it."
        raise ValueError(msg)
    for name in ("reference_setpoint", "bind_reference_setpoint"):
        if getattr(task, name, None) is None:
            msg = f"wm.actor_residual needs the task's {name} (G1V3TrackingTask)."
            raise ValueError(msg)


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
    if wm.actor_lr_decay_return is None and wm.actor_lr_decay_by is None:
        if wm.actor_lr_decay_len != 0:
            msg = "wm.actor_lr_decay_len needs a trigger: wm.actor_lr_decay_return or wm.actor_lr_decay_by."
            raise ValueError(msg)
    else:
        if wm.actor_lr_mult == 1:
            msg = "wm.actor_lr_decay_return and wm.actor_lr_decay_by decay wm.actor_lr_mult, which is 1."
            raise ValueError(msg)
        if wm.actor_lr_decay_len < 1:
            msg = f"wm.actor_lr_decay_len must be at least 1 with a trigger, got {wm.actor_lr_decay_len}."
            raise ValueError(msg)
        if wm.actor_lr_decay_by is not None and wm.actor_lr_decay_by < 1:
            msg = f"wm.actor_lr_decay_by must be at least 1, got {wm.actor_lr_decay_by}."
            raise ValueError(msg)
        if wm.actor_lr_decay_return is not None:
            if wm.actor_lr_decay_return <= 0:
                msg = f"wm.actor_lr_decay_return must be positive, got {wm.actor_lr_decay_return}."
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
    for name in ("elementwise_3x3", "compile_eager", "combo_kernels"):
        if getattr(wm, name) and not wm.compile:
            msg = f"wm.{name} acts inside compiled functions; set wm.compile."
            raise ValueError(msg)
    if wm.compile_eager and getattr(task, "compile_collection_reward", None) is None:
        msg = "wm.compile_eager compiles the collection reward, which this task cannot reach."
        raise ValueError(msg)
    if wm.goal_table and getattr(task, "goal_depends_on_state", True):
        msg = "wm.goal_table needs a task whose goal is a function of the reference frame alone."
        raise ValueError(msg)
    if wm.reference_body_table and getattr(task, "build_reference_tables", None) is None:
        msg = "wm.reference_body_table needs a task with a reference table (G1V3TrackingTask)."
        raise ValueError(msg)
    if wm.actor_residual:
        validate_residual(task, goal=str(wm.goal), goal_encoder=str(wm.goal_encoder))
        if wm.actor_source == "real" or wm.actor_estimator == "rebar":
            msg = "wm.actor_residual runs on imagined rollouts; real windows and rebar re-simulate the old actions without it."
            raise ValueError(msg)
    for name in ("fit_terminal_transitions", "norm_from_collection"):
        if getattr(wm, name) and not getattr(task, "final_obs_is_post_step", False):
            msg = f"wm.{name} reads the true state after a step from final_obs, which this task's env does not hold."
            raise ValueError(msg)
    if wm.no_sync:
        if not wm.fused_adam:
            msg = "wm.no_sync skips a non-finite step with the fused Adam; set wm.fused_adam."
            raise ValueError(msg)
        if "real" in (wm.actor_source, wm.critic_source):
            msg = "wm.no_sync covers imagined rollouts: real windows select rows on the host (actor_source, critic_source)."
            raise ValueError(msg)
    if wm.graph_early and not wm.update_graphs:
        msg = "wm.graph_early sets when wm.update_graphs captures; set wm.update_graphs."
        raise ValueError(msg)
    if wm.update_graphs:
        if not (wm.no_sync and wm.fused_adam):
            msg = "wm.update_graphs captures the no_sync loop with capturable fused optimizers; set wm.no_sync and wm.fused_adam."
            raise ValueError(msg)
        device = str(config.device)
        if not (device.startswith("cuda") or (device == "auto" and torch.cuda.is_available())):
            msg = f"wm.update_graphs captures CUDA graphs; the device is {device!r}."
            raise ValueError(msg)
        if wm.critic_update_every != 1:
            msg = "wm.update_graphs captures the critic fit with the actor; wm.critic_update_every must be 1."
            raise ValueError(msg)
        if wm.actor_estimator != "pathwise":
            msg = f"wm.update_graphs is tested with the pathwise actor only, not actor_estimator={wm.actor_estimator!r}."
            raise ValueError(msg)
        if wm.dynamics_model != "mlp":
            msg = f"wm.update_graphs is tested with the MLP world model only, not dynamics_model={wm.dynamics_model!r}."
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
