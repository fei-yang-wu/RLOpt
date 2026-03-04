from __future__ import annotations

import functools
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from tensordict import TensorDict
from tensordict.nn import InteractionType
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torchrl._utils import timeit
from torchrl.collectors import SyncDataCollector
from torchrl.data import (
    LazyMemmapStorage,
    LazyTensorStorage,
    ReplayBuffer,
    TensorDictReplayBuffer,
)
from torchrl.data.replay_buffers.samplers import RandomSampler
from torchrl.envs.utils import set_exploration_type
from torchrl.modules import MLP
from torchrl.record.loggers import Logger

from rlopt.agent.ppo.ppo import PPO, PPOConfig, PPORLOptConfig
from rlopt.config_utils import (
    BatchKey,
    ObsKey,
    dedupe_keys,
    epic_distance,
    flatten_feature_tensor,
    infer_batch_shape,
    mapping_get_obs_value,
    next_obs_key,
    strip_next_prefix,
)
from rlopt.type_aliases import OptimizerClass
from rlopt.utils import get_activation_class, log_info


@dataclass
class IPMDConfig(PPOConfig):
    """IPMD-specific configuration (PPO-based)."""

    # Reward estimator network and loss settings
    reward_input_type: str = "s'"
    """Input type for the reward estimator.

    Supported values:
    - ``"s"``   - current state only  r(s)
    - ``"s'"``  - next state only     r(s')
    - ``"sa"``  - state-action         r(s, a)
    - ``"sas"`` - state-action-state   r(s, a, s')  (default)
    """

    reward_input_keys: list[ObsKey] | None = None
    """Observation keys used for reward state inputs.

    If ``None`` (default), falls back to ``value_function.get_input_keys()``
    (then ``policy.get_input_keys()``).  Set this to let the reward model
    consume a different subset of observation keys than policy / value.
    """

    reward_num_cells: tuple[int, ...] = (256, 256)
    """Hidden layer sizes for the reward estimator MLP."""

    reward_activation: str = "elu"
    """Activation for reward estimator MLP."""

    reward_init: str = "orthogonal"
    """Weight init for reward estimator."""

    reward_output_activation: str = "tanh"
    """Output activation applied to the reward estimator.

    Supported values:
    - ``"none"``    - no activation (unbounded, default)
    - ``"tanh"``    - tanh scaled by ``reward_output_scale``
    - ``"sigmoid"`` - sigmoid scaled by ``reward_output_scale``
    """

    reward_output_scale: float = 1.0
    """Scale factor for bounded output activations (tanh / sigmoid)."""

    reward_loss_coeff: float = 1.0
    """Scale for (sum r_pi - sum r_expert)."""

    reward_l2_coeff: float = 0.05
    """L2 regularization weight for reward parameters."""

    reward_detach_features: bool = True
    """Detach features when computing reward loss (avoid leaking grads)."""

    use_estimated_rewards_for_ppo: bool = True
    """Whether to use estimated rewards instead of environment rewards for PPO (GAE + value target).

    Default is False. Set to True to train PPO on estimated rewards.
    """

    expert_batch_size: int | None = None
    """Batch size for expert data sampling. If None, uses the same as mini_batch_size."""

    detach_reward_when_used_for_ppo: bool = True
    """Detach the estimated reward when injecting into PPO (GAE/reward).

    Prevents PPO updates from backpropagating into the reward estimator.
    The reward network is then trained solely via the IPMD objective.
    """

    bc_loss_coeff: float = 0.0
    """Behavior cloning (MLE) loss coefficient on expert actions.

    When > 0, adds ``-bc_loss_coeff * mean(log_prob(expert_action | policy))``
    to each update step.  This regularises the policy toward expert actions
    during early training.  Set to 0 to disable (default).
    """

    reward_optimizer: str | None = None
    """Optional reward-optimizer name (defaults to ``optim.optimizer`` when None)."""

    reward_lr: float | None = None
    """Optional reward learning rate (defaults to ``optim.lr`` when None)."""

    reward_weight_decay: float | None = None
    """Optional reward weight decay (defaults to ``optim.weight_decay`` when None)."""

    reward_optimizer_kwargs: dict[str, Any] = field(default_factory=dict)
    """Extra kwargs merged into the reward optimizer configuration."""

    reward_max_grad_norm: float | None = None
    """Optional grad clip norm for reward optimizer (falls back to ``optim.max_grad_norm``)."""

    reward_update_interval: int = 100
    """Update reward estimator every N PPO updates (1 = every update)."""

    reward_updates_per_policy_update: int = 2
    """Number of reward optimizer steps per PPO optimizer step."""

    reward_update_warmup_updates: int = 500
    """Skip reward updates for the first N PPO updates."""

    reward_balance_policy_and_expert: bool = True
    """Whether to balance policy/expert samples to 1:1 in each reward update."""

    use_reward_target_network: bool = True
    """Whether to maintain a Polyak target copy of the reward estimator."""

    use_reward_target_for_ppo: bool = True
    """Use target reward network (instead of online reward net) for PPO reward replacement."""

    reward_target_polyak: float = 0.995
    """Polyak coefficient for reward-target update (higher = slower target updates)."""

    reward_target_update_interval: int = 1
    """Update reward target every N reward updates."""

    reward_margin: float = 0.05
    """Optional margin for reward-gap hinge: ``relu(mean(r_pi)-mean(r_exp)+margin)``."""

    reward_consistency_coeff: float = 0.2
    """Trust-region coefficient to keep reward close to target reward predictions."""

    reward_grad_penalty_coeff: float = 0.2
    """R1-style gradient penalty coefficient on expert reward inputs."""

    reward_logit_reg_coeff: float = 0.02
    """Regularization coefficient on reward logits to reduce saturation."""

    reward_train_on_logits: bool = True
    """Train reward objective in logit space (activation applied only for PPO reward)."""

    reward_param_weight_decay_coeff: float = 1.0e-5
    """Explicit L2 regularization coefficient over reward-estimator parameters."""

    normalize_reward_input: bool = True
    """Whether to apply running mean/std normalization to reward-model inputs."""

    reward_input_noise_std: float = 0.01
    """Optional Gaussian noise std applied to reward inputs during reward updates."""

    reward_input_dropout_prob: float = 0.05
    """Optional feature dropout probability for reward inputs during reward updates."""

    reward_input_norm_momentum: float = 0.01
    """EMA momentum for reward input running statistics."""

    reward_input_norm_eps: float = 1.0e-5
    """Numerical epsilon for reward input normalization."""

    reward_input_norm_clip: float | None = None
    """Optional absolute clip value applied after reward input normalization."""

    reward_replay_size: int = 200000
    """Size of optional policy-transition replay for reward updates (0 disables)."""

    reward_replay_ratio: float = 0.5
    """Extra replay samples per on-policy sample for reward updates."""

    reward_replay_batch_size: int | None = None
    """Fixed replay sample size for reward updates (overrides ratio when set)."""

    reward_replay_keep_prob: float = 0.25
    """Probability of keeping each on-policy transition when filling reward replay."""

    reward_replay_reset_interval_updates: int = 5000
    """Reset reward replay every N PPO updates (0 disables)."""

    reward_mix_alpha_start: float = 0.5
    """Initial blend weight for estimated reward in PPO reward mixing."""

    reward_mix_alpha_end: float = 0.5
    """Final blend weight for estimated reward in PPO reward mixing."""

    reward_mix_anneal_updates: int = 20000
    """Linear anneal duration (in updates) for reward mixing alpha."""

    reward_mix_gate_estimated_std_min: float = 0.05
    """If estimated reward std is below this threshold, reduce alpha to fallback."""

    reward_mix_alpha_when_unstable: float = 0.15
    """Fallback alpha when estimated reward variance is too low."""

    reward_mix_gate_after_updates: int = 500
    """Only activate std-gating after this many updates."""

    reward_mix_gate_abs_gap_max: float = 0.5
    """If |mean(est_reward)-mean(env_reward)| exceeds this value, clamp alpha."""

    reward_mix_alpha_when_gap_large: float = 0.1
    """Fallback alpha when absolute reward gap gate is triggered."""

    entropy_coeff_start: float | None = 0.02
    """Optional starting entropy coefficient for exploration schedule."""

    entropy_coeff_end: float | None = 0.005
    """Optional ending entropy coefficient for exploration schedule."""

    entropy_schedule_updates: int = 15000
    """Linear schedule duration (in updates) for entropy coefficient."""

    policy_random_action_prob_start: float = 0.0
    """Initial probability of random action injection into collected policy actions."""

    policy_random_action_prob_end: float = 0.0
    """Final probability of random action injection."""

    policy_random_action_schedule_updates: int = 0
    """Linear schedule duration for policy random action injection."""

    reward_scheduler: str | None = None
    """Optional reward optimizer scheduler (e.g., 'steplr', 'cosineannealinglr')."""

    reward_scheduler_kwargs: dict[str, Any] = field(default_factory=dict)
    """Keyword arguments forwarded to reward LR scheduler."""

    reward_scheduler_step: str = "update"
    """When to step reward scheduler: 'update' or 'epoch'."""

    bc_warmup_updates: int = 0
    """Linear anneal duration (in updates) for BC coefficient."""

    bc_final_coeff: float = 0.0
    """Final BC coefficient after warmup schedule."""


@dataclass
class IPMDRLOptConfig(PPORLOptConfig):
    """IPMD configuration extending PPORLOptConfig (PPO-based)."""

    ipmd: IPMDConfig = field(default_factory=IPMDConfig)
    """IPMD configuration."""


class IPMD(PPO):
    """IPMD algorithm with PPO as the base RL algorithm.

    Uses the same on-policy rollout + GAE + multiple epochs over mini-batches as PPO,
    and adds a reward estimator r(s, a, s') trained with the IPMD objective
    (policy estimated return - expert estimated return). Optionally uses estimated
    rewards for PPO updates.
    """

    _REWARD_OUTPUT_ACTIVATIONS = frozenset({"none", "tanh", "sigmoid"})

    def __init__(
        self,
        env,
        config: IPMDRLOptConfig,
        policy_net: torch.nn.Module | None = None,
        value_net: torch.nn.Module | None = None,
        q_net: torch.nn.Module | None = None,
        replay_buffer: type[ReplayBuffer] = ReplayBuffer,
        logger: Logger | None = None,
        feature_extractor_net: torch.nn.Module | None = None,
        **kwargs,
    ):
        self.config = cast(IPMDRLOptConfig, config)
        self.config: IPMDRLOptConfig
        self.env = env

        super().__init__(
            env=env,
            config=config,
            policy_net=policy_net,
            value_net=value_net,
            q_net=q_net,
            replay_buffer=replay_buffer,
            logger=logger,
            feature_extractor_net=feature_extractor_net,
            **kwargs,
        )

        self.config = cast(IPMDRLOptConfig, self.config)

        # Observation key groups can differ across policy, value, and reward model.
        self._policy_obs_keys: list[ObsKey] = self.config.policy.get_input_keys()
        value_cfg = self.config.value_function
        self._value_obs_keys: list[ObsKey] = (
            value_cfg.get_input_keys()
            if value_cfg is not None
            else list(self._policy_obs_keys)
        )
        self._reward_obs_keys: list[ObsKey] = self._resolve_reward_obs_keys()

        all_obs_keys = dedupe_keys(
            self._policy_obs_keys + self._value_obs_keys + self._reward_obs_keys
        )
        self._obs_feature_ndims: dict[ObsKey, int] = {
            key: self._obs_key_feature_ndim(key) for key in all_obs_keys
        }
        self._obs_feature_dims: dict[ObsKey, int] = {
            key: self._obs_key_feature_dim(key) for key in all_obs_keys
        }
        action_spec = getattr(self.env, "action_spec_unbatched", self.env.action_spec)
        action_shape = tuple(int(dim) for dim in action_spec.shape)
        self._action_feature_ndim = len(action_shape)
        self._action_feature_dim = int(math.prod(action_shape)) if action_shape else 1

        # Reward estimator: r(s, a, s') -> scalar
        self.reward_estimator: torch.nn.Module = self._construct_reward_estimator()
        self.reward_estimator.to(self.device)

        # Reward target network disabled for simplified IPMD.
        self.reward_target_estimator: torch.nn.Module | None = None

        # Running reward-input normalization stats (kept for compatibility but disabled).
        self._reward_input_running_mean: Tensor | None = None
        self._reward_input_running_var: Tensor | None = None
        self._reward_input_stats_initialized: bool = False

        # Expert data source
        self._expert_buffer: TensorDictReplayBuffer | None = None
        self._warned_no_expert = False
        self._reward_replay_buffer: TensorDictReplayBuffer | None = (
            self._construct_reward_replay_buffer()
        )

        # Re-create PPO optimizer (actor-critic only) after IPMD initialization.
        self.optim = self._configure_optimizers()
        self._refresh_grad_clip_params()
        self.reward_optim = self._configure_reward_optimizer()
        self._refresh_reward_grad_clip_params()
        self.reward_lr_scheduler = self._configure_reward_scheduler(self.reward_optim)
        self.reward_lr_scheduler_step = str(
            self.config.ipmd.reward_scheduler_step
        ).lower()
        # Cache once to avoid dynamic module construction inside compiled update.
        self._policy_operator = self.actor_critic.get_policy_operator()
        self._init_runtime_caches()

        # Compile only after all IPMD-specific components are initialized.
        self._compile_components()

    def _compile_components(self) -> None:
        """Compile update (fixed signature for torch.compile and CUDA graphs)."""
        cfg = self.config
        assert isinstance(cfg, IPMDRLOptConfig)
        if not hasattr(self, "adv_module"):
            return
        if not hasattr(self, "reward_estimator"):
            return
        if getattr(self, "_components_compiled", False):
            return
        if not cfg.compile.compile:
            return
        compile_mode = cfg.compile.compile_mode or (
            "default" if cfg.compile.cudagraphs else "reduce-overhead"
        )
        self.reward_estimator = torch.compile(self.reward_estimator, mode=compile_mode)  # type: ignore[assignment]
        if self.reward_target_estimator is not None:
            self.reward_target_estimator = torch.compile(
                self.reward_target_estimator, mode=compile_mode
            )  # type: ignore[assignment]
        self.update = torch.compile(self.update, mode=compile_mode)  # type: ignore[method-assign]
        self.adv_module = torch.compile(self.adv_module, mode=compile_mode)  # type: ignore[method-assign]
        self._components_compiled = True

    _REWARD_INPUT_TYPES = frozenset({"s", "s'", "sa", "sas"})

    def _init_runtime_caches(self) -> None:
        cfg = self.config
        assert isinstance(cfg, IPMDRLOptConfig)
        ipmd_cfg = cfg.ipmd

        self._reward_required_keys: tuple[BatchKey, ...] = tuple(
            self._reward_policy_required_keys()
        )
        self._expert_required_keys_cached: tuple[BatchKey, ...] = tuple(
            self._expert_required_keys()
        )

        reward_input_type = ipmd_cfg.reward_input_type
        self._reward_include_state = reward_input_type in ("s", "sa", "sas")
        self._reward_include_action = reward_input_type in ("sa", "sas")
        self._reward_include_next_state = reward_input_type in ("s'", "sas")
        self._reward_detach_default = bool(ipmd_cfg.reward_detach_features)

        out_act = ipmd_cfg.reward_output_activation
        if out_act == "tanh":
            self._reward_output_activation_mode = 1
        elif out_act == "sigmoid":
            self._reward_output_activation_mode = 2
        else:
            self._reward_output_activation_mode = 0
        self._reward_output_scale = float(ipmd_cfg.reward_output_scale)

        # Reward input normalization disabled for simplified IPMD.
        self._normalize_reward_input_enabled = False
        self._reward_input_noise_std = float(ipmd_cfg.reward_input_noise_std)
        self._reward_input_dropout_prob = float(ipmd_cfg.reward_input_dropout_prob)
        self._reward_input_dropout_keep_prob = min(
            1.0, max(1.0e-6, 1.0 - self._reward_input_dropout_prob)
        )
        self._reward_input_augmentation_enabled = (
            self._reward_input_noise_std > 0.0 or self._reward_input_dropout_prob > 0.0
        )

        self._reward_use_logits = bool(ipmd_cfg.reward_train_on_logits)
        self._reward_loss_coeff = float(ipmd_cfg.reward_loss_coeff)
        self._reward_l2_coeff = float(ipmd_cfg.reward_l2_coeff)
        self._reward_margin = float(ipmd_cfg.reward_margin)
        # Reward consistency regularization disabled (no target network).
        self._reward_consistency_coeff = 0.0
        self._reward_grad_penalty_coeff = float(ipmd_cfg.reward_grad_penalty_coeff)
        self._reward_logit_reg_coeff = float(ipmd_cfg.reward_logit_reg_coeff)
        self._reward_param_decay_coeff = float(ipmd_cfg.reward_param_weight_decay_coeff)
        self._reward_use_margin = self._reward_margin > 0.0
        self._reward_use_consistency = False
        self._reward_use_grad_penalty = self._reward_grad_penalty_coeff > 0.0
        self._reward_use_logit_reg = self._reward_logit_reg_coeff > 0.0
        self._reward_use_param_decay = self._reward_param_decay_coeff > 0.0
        self._reward_balance_batches = bool(ipmd_cfg.reward_balance_policy_and_expert)

        self._reward_updates_per_policy_update = max(
            1, int(ipmd_cfg.reward_updates_per_policy_update)
        )
        self._reward_update_interval = max(1, int(ipmd_cfg.reward_update_interval))
        self._reward_update_warmup = max(0, int(ipmd_cfg.reward_update_warmup_updates))
        self._reward_replay_keep_prob = min(
            1.0, max(0.0, float(ipmd_cfg.reward_replay_keep_prob))
        )
        self._reward_replay_ratio = max(0.0, float(ipmd_cfg.reward_replay_ratio))
        self._reward_replay_batch_size_cfg = (
            None
            if ipmd_cfg.reward_replay_batch_size is None
            else int(ipmd_cfg.reward_replay_batch_size)
        )
        self._expert_batch_size_cfg = (
            None
            if ipmd_cfg.expert_batch_size is None
            else int(ipmd_cfg.expert_batch_size)
        )
        self._reward_replay_reset_interval = max(
            0, int(ipmd_cfg.reward_replay_reset_interval_updates)
        )

        self._use_estimated_rewards_for_ppo = bool(
            ipmd_cfg.use_estimated_rewards_for_ppo
        )
        self._use_reward_target_for_ppo = bool(ipmd_cfg.use_reward_target_for_ppo)
        self._detach_reward_when_used_for_ppo = bool(
            ipmd_cfg.detach_reward_when_used_for_ppo
        )

        reward_max_grad_norm = ipmd_cfg.reward_max_grad_norm
        if reward_max_grad_norm is None:
            reward_max_grad_norm = cfg.optim.max_grad_norm
        self._reward_max_grad_norm = (
            None if reward_max_grad_norm is None else float(reward_max_grad_norm)
        )
        max_grad_norm = cfg.optim.max_grad_norm
        self._max_grad_norm = None if max_grad_norm is None else float(max_grad_norm)

        entropy_base = float(cfg.ppo.entropy_coeff)
        self._entropy_coeff_start = (
            entropy_base
            if ipmd_cfg.entropy_coeff_start is None
            else float(ipmd_cfg.entropy_coeff_start)
        )
        self._entropy_coeff_end = (
            entropy_base
            if ipmd_cfg.entropy_coeff_end is None
            else float(ipmd_cfg.entropy_coeff_end)
        )
        self._entropy_schedule_steps = max(0, int(ipmd_cfg.entropy_schedule_updates))

        # Fixed BC coefficient (no scheduling).
        self._bc_coeff = float(ipmd_cfg.bc_loss_coeff)

        # Reward mixing schedule disabled; use a fixed alpha.
        self._reward_mix_alpha = float(ipmd_cfg.reward_mix_alpha_start)

        self._adaptive_kl_scheduler = (cfg.optim.scheduler or "").lower() == "adaptive"
        self._is_isaac_env = "Isaac" in cfg.env.env_name
        self._compile_mode_enabled = bool(getattr(cfg.compile, "compile_mode", None))

        self._loss_record_keys: tuple[str, ...] = (
            "loss_critic",
            "loss_entropy",
            "loss_objective",
            "loss_reward_diff",
            "loss_reward_gap_term",
            "loss_reward_l2",
            "loss_reward_grad_penalty",
            "loss_reward_logit_reg",
            "loss_reward_param_decay",
            "loss_bc",
            "estimated_reward_mean",
            "estimated_reward_std",
            "expert_reward_mean",
            "expert_reward_std",
            "reward_update_mask",
            "reward_mix_alpha",
            "reward_updates_performed",
            "bc_coeff",
            "reward_lr",
        )
        self._loss_record_optional_keys: tuple[str, ...] = (
            "entropy",
            "explained_variance",
            "clip_fraction",
            "value_clip_fraction",
            "ESS",
            "kl_approx",
            "grad_norm",
            "reward_grad_norm",
        )

        self._zero_f32 = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        self._one_f32 = torch.tensor(1.0, device=self.device, dtype=torch.float32)
        self._false_bool = torch.tensor(False, device=self.device)

    @staticmethod
    def _counter_as_int(value: int | Tensor) -> int:
        if isinstance(value, Tensor):
            return int(value.detach().item())
        return int(value)

    @staticmethod
    def _linear_schedule(
        start: float, end: float, step: int, total_steps: int
    ) -> float:
        if total_steps <= 0:
            return end
        step_clamped = min(max(step, 0), total_steps)
        alpha = step_clamped / float(total_steps)
        return float(start + alpha * (end - start))

    def _current_entropy_coeff(self, update_idx: int) -> float:
        start = self._entropy_coeff_start
        end = self._entropy_coeff_end
        steps = self._entropy_schedule_steps
        if start == end or steps <= 0:
            return end
        return self._linear_schedule(start, end, update_idx, steps)

    def _set_loss_entropy_coeff(self, coeff: float) -> None:
        entropy_attr = getattr(self.loss_module, "entropy_coeff", None)
        if isinstance(entropy_attr, Tensor):
            entropy_attr.copy_(
                torch.tensor(
                    coeff, device=entropy_attr.device, dtype=entropy_attr.dtype
                )
            )
        elif entropy_attr is not None:
            self.loss_module.entropy_coeff = float(coeff)

    def _reward_policy_required_keys(self) -> list[BatchKey]:
        """Return policy-batch keys required by the reward estimator."""
        cached = getattr(self, "_reward_required_keys", None)
        if cached is not None:
            return list(cached)
        assert isinstance(self.config, IPMDRLOptConfig)
        rit = self.config.ipmd.reward_input_type
        required: list[BatchKey] = []
        if rit in ("s", "sa", "sas"):
            required.extend(self._reward_obs_keys)
        if rit in ("s'", "sas"):
            required.extend([next_obs_key(key) for key in self._reward_obs_keys])
        if rit in ("sa", "sas"):
            required.append("action")
        return dedupe_keys(required)

    def _construct_reward_replay_buffer(self) -> TensorDictReplayBuffer | None:
        cfg = self.config
        assert isinstance(cfg, IPMDRLOptConfig)
        replay_size = int(cfg.ipmd.reward_replay_size)
        if replay_size <= 0:
            return None
        replay_batch_size = (
            int(cfg.ipmd.reward_replay_batch_size)
            if cfg.ipmd.reward_replay_batch_size is not None
            else int(cfg.loss.mini_batch_size)
        )
        return TensorDictReplayBuffer(
            pin_memory=False,
            prefetch=cfg.collector.prefetch,
            sampler=RandomSampler(),
            storage=LazyTensorStorage(
                max_size=replay_size,
                compilable=cfg.compile.compile,
                device="cpu",
            ),
            batch_size=max(1, replay_batch_size),
            shared=cfg.collector.shared,
        )

    def _store_reward_replay_samples(self, batch: TensorDict) -> None:
        if self._reward_replay_buffer is None:
            return
        replay_batch = batch.select(*self._reward_required_keys).reshape(-1).detach()
        keep_prob = self._reward_replay_keep_prob
        if keep_prob < 1.0 and replay_batch.numel() > 0:
            keep_mask = (
                torch.rand(replay_batch.numel(), device=batch.device) < keep_prob
            )
            if not bool(keep_mask.any()):
                return
            replay_batch = replay_batch[keep_mask]
        if replay_batch.numel() > 0:
            self._reward_replay_buffer.extend(replay_batch)

    def _sample_reward_replay_batch(self, current_batch_size: int) -> TensorDict | None:
        if self._reward_replay_batch_size_cfg is not None:
            replay_batch_size = self._reward_replay_batch_size_cfg
        else:
            replay_batch_size = round(current_batch_size * self._reward_replay_ratio)
        replay_batch = cast(
            TensorDict,
            self._reward_replay_buffer.sample(batch_size=replay_batch_size),
        )
        return replay_batch.to(self.device)

    def _reward_batch_with_replay(
        self,
        batch: TensorDict,
        required_keys: tuple[BatchKey, ...] | list[BatchKey] | None = None,
    ) -> TensorDict:
        if required_keys is None:
            required_keys = self._reward_required_keys
        reward_batch = batch.select(*required_keys)
        replay_batch = self._sample_reward_replay_batch(reward_batch.numel())
        if replay_batch is None:
            return reward_batch
        # ReplayBuffer.sample can add metadata keys (e.g. "index"). Keep only reward keys.
        replay_batch = replay_batch.select(*required_keys)
        return cast(TensorDict, torch.cat([reward_batch, replay_batch], dim=0))

    def _configure_reward_optimizer(self) -> torch.optim.Optimizer:
        cfg = self.config
        assert isinstance(cfg, IPMDRLOptConfig)
        optimizer_map: dict[str, OptimizerClass] = {
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW,
            "adamax": torch.optim.Adamax,
            "sgd": torch.optim.SGD,
            "rmsprop": torch.optim.RMSprop,
        }
        optimizer_name = (cfg.ipmd.reward_optimizer or cfg.optim.optimizer).lower()
        if optimizer_name not in optimizer_map:
            available = ", ".join(sorted(optimizer_map))
            msg = f"Unknown reward optimizer '{optimizer_name}'. Choose one of: {available}."
            raise ValueError(msg)
        optimizer_cls = optimizer_map[optimizer_name]
        kwargs = {
            "lr": (
                float(cfg.optim.lr)
                if cfg.ipmd.reward_lr is None
                else float(cfg.ipmd.reward_lr)
            ),
            "weight_decay": (
                float(cfg.optim.weight_decay)
                if cfg.ipmd.reward_weight_decay is None
                else float(cfg.ipmd.reward_weight_decay)
            ),
        }
        kwargs.update(dict(cfg.optim.optimizer_kwargs))
        kwargs.update(dict(cfg.ipmd.reward_optimizer_kwargs))
        return optimizer_cls(self.reward_estimator.parameters(), **kwargs)

    def _configure_reward_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler.LRScheduler | None:
        cfg = self.config
        assert isinstance(cfg, IPMDRLOptConfig)
        scheduler_name = (cfg.ipmd.reward_scheduler or "").lower().strip()
        if not scheduler_name:
            return None
        scheduler_map: dict[str, type[torch.optim.lr_scheduler.LRScheduler]] = {
            "steplr": torch.optim.lr_scheduler.StepLR,
            "multisteplr": torch.optim.lr_scheduler.MultiStepLR,
            "exponentiallr": torch.optim.lr_scheduler.ExponentialLR,
            "cosineannealinglr": torch.optim.lr_scheduler.CosineAnnealingLR,
            "linearlr": torch.optim.lr_scheduler.LinearLR,
            "constantlr": torch.optim.lr_scheduler.ConstantLR,
        }
        scheduler_cls = scheduler_map.get(scheduler_name)
        if scheduler_cls is None:
            available = ", ".join(sorted(scheduler_map))
            msg = (
                f"Unknown reward scheduler '{scheduler_name}'. "
                f"Choose one of: {available}."
            )
            raise ValueError(msg)
        scheduler_kwargs = dict(cfg.ipmd.reward_scheduler_kwargs)
        return scheduler_cls(optimizer, **scheduler_kwargs)

    def _refresh_reward_grad_clip_params(self) -> None:
        self._reward_grad_clip_params: list[Tensor] = [
            param
            for group in self.reward_optim.param_groups
            for param in group["params"]
        ]

    def _maybe_reset_reward_replay(self, update_idx: int) -> None:
        if self._reward_replay_buffer is None:
            return
        interval = self._reward_replay_reset_interval
        if interval <= 0:
            return
        if update_idx > 0 and (update_idx % interval) == 0:
            self._reward_replay_buffer.empty()

    def _resolve_reward_obs_keys(self) -> list[ObsKey]:
        """Resolve observation keys used by reward estimator state inputs."""
        cfg = self.config
        assert isinstance(cfg, IPMDRLOptConfig)
        reward_keys = cfg.ipmd.reward_input_keys
        if not reward_keys:
            reward_keys = (
                cfg.value_function.get_input_keys()
                if cfg.value_function is not None
                else cfg.policy.get_input_keys()
            )
        return dedupe_keys(list(reward_keys))

    def _obs_key_feature_shape(self, key: ObsKey) -> tuple[int, ...]:
        """Return unbatched feature shape for an observation key."""
        shape = tuple(int(dim) for dim in self.env.observation_spec[key].shape)
        batch_prefix = tuple(int(dim) for dim in getattr(self.env, "batch_size", ()))
        if (
            len(batch_prefix) > 0
            and len(shape) >= len(batch_prefix)
            and shape[: len(batch_prefix)] == batch_prefix
        ):
            return shape[len(batch_prefix) :]
        return shape

    def _obs_key_feature_ndim(self, key: ObsKey) -> int:
        return len(self._obs_key_feature_shape(key))

    def _obs_key_feature_dim(self, key: ObsKey) -> int:
        shape = self._obs_key_feature_shape(key)
        return int(math.prod(shape)) if shape else 1

    def _obs_features_from_td(
        self,
        td: TensorDict | Any,
        keys: list[ObsKey],
        *,
        next_obs: bool,
        detach: bool,
    ) -> Tensor:
        parts: list[Tensor] = []
        for key in keys:
            td_key: BatchKey = next_obs_key(key) if next_obs else key
            obs = cast(Tensor, td.get(td_key))
            obs = flatten_feature_tensor(obs, self._obs_feature_ndims[key])
            parts.append(obs.detach() if detach else obs)
        if len(parts) == 1:
            return parts[0]
        return torch.cat(parts, dim=-1)

    def _action_features_from_td(self, td: TensorDict | Any, *, detach: bool) -> Tensor:
        action = cast(Tensor, td.get("action"))
        action = flatten_feature_tensor(action, self._action_feature_ndim)
        return action.detach() if detach else action

    def _expert_required_keys(self) -> list[BatchKey]:
        """Return expert-batch keys required by current IPMD settings."""
        cached = getattr(self, "_expert_required_keys_cached", None)
        if cached is not None:
            return list(cached)
        assert isinstance(self.config, IPMDRLOptConfig)
        rit = self.config.ipmd.reward_input_type
        bc_enabled = float(self.config.ipmd.bc_loss_coeff) > 0.0

        required: list[BatchKey] = []
        if rit in ("s", "sa", "sas"):
            required.extend(self._reward_obs_keys)
        if rit in ("s'", "sas"):
            required.extend([next_obs_key(key) for key in self._reward_obs_keys])
        if rit in ("sa", "sas") or bc_enabled:
            required.append("action")
        if bc_enabled:
            required.extend(self._policy_obs_keys)
        return dedupe_keys(required)

    def _construct_reward_estimator(self) -> torch.nn.Module:
        """Create reward network whose input depends on ``reward_input_type``."""
        cfg = self.config
        assert isinstance(cfg, IPMDRLOptConfig)
        rit = cfg.ipmd.reward_input_type
        if rit not in self._REWARD_INPUT_TYPES:
            msg = f"Unknown reward_input_type {rit!r}; expected one of {sorted(self._REWARD_INPUT_TYPES)}"
            raise ValueError(msg)
        out_act = cfg.ipmd.reward_output_activation
        if out_act not in self._REWARD_OUTPUT_ACTIVATIONS:
            msg = f"Unknown reward_output_activation {out_act!r}; expected one of {sorted(self._REWARD_OUTPUT_ACTIVATIONS)}"
            raise ValueError(msg)

        obs_dim = sum(self._obs_feature_dims[key] for key in self._reward_obs_keys)
        act_dim = self._action_feature_dim

        if rit in ("s", "s'"):
            in_dim = obs_dim
        elif rit == "sa":
            in_dim = obs_dim + act_dim
        else:  # "sas"
            in_dim = obs_dim * 2 + act_dim

        net = MLP(
            in_features=in_dim,
            out_features=1,
            num_cells=list(cfg.ipmd.reward_num_cells),
            activation_class=get_activation_class(cfg.ipmd.reward_activation),
            device=self.device,
        )
        self._initialize_weights(net, cfg.ipmd.reward_init)
        return net

    def _set_optimizers(
        self, optimizer_cls: OptimizerClass, optimizer_kwargs: dict[str, Any]
    ) -> list[torch.optim.Optimizer]:
        """Create optimizer(s) for actor-critic only.

        IPMD uses a dedicated optimizer for the reward estimator to decouple
        reward and PPO update dynamics.
        """
        return super()._set_optimizers(optimizer_cls, optimizer_kwargs)

    # -------------------------
    # Expert data API
    # -------------------------
    def set_expert_buffer(self, buffer: TensorDictReplayBuffer) -> None:
        """Attach an expert replay buffer.

        Required keys depend on ``reward_input_type``, ``reward_input_keys``, and
        whether behavior cloning is enabled.
        """
        self._expert_buffer = buffer
        self._expert_buffer.append_transform(lambda td: td.to(self.device))

    def create_expert_buffer(
        self, expert_data: TensorDict, buffer_size: int | None = None
    ) -> TensorDictReplayBuffer:
        """Create an expert replay buffer from expert demonstration data."""
        if buffer_size is None:
            buffer_size = expert_data.numel()

        cfg = self.config
        assert isinstance(cfg, IPMDRLOptConfig)
        sampler = RandomSampler()
        scratch_dir = cfg.collector.scratch_dir
        device = cfg.device
        batch_size = cfg.loss.mini_batch_size
        shared = cfg.collector.shared
        prefetch = cfg.collector.prefetch

        storage_cls = (
            functools.partial(LazyTensorStorage, device=device)
            if not scratch_dir
            else functools.partial(
                LazyMemmapStorage, device="cpu", scratch_dir=scratch_dir
            )
        )

        expert_buffer = TensorDictReplayBuffer(
            pin_memory=False,
            prefetch=prefetch,
            sampler=sampler,
            storage=storage_cls(max_size=buffer_size, compilable=cfg.compile.compile),
            batch_size=batch_size,
            shared=shared,
        )
        expert_buffer.extend(expert_data.reshape(-1))
        if scratch_dir:
            expert_buffer.append_transform(lambda td: td.to(device))  # type: ignore[arg-type]
        return expert_buffer

    def _check_expert_batch_keys(self, expert_batch: TensorDict) -> bool:
        required_keys = self._expert_required_keys()
        available_keys = expert_batch.keys(True)
        missing = [key for key in required_keys if key not in available_keys]
        if missing:
            self.log.warning("Expert batch missing required keys: %s", missing)
            return False
        return True

    def _next_expert_batch(self, batch_size: int | None = None) -> TensorDict | None:
        effective_batch_size = (
            self._expert_batch_size_cfg if batch_size is None else int(batch_size)
        )
        return self._expert_buffer.sample(batch_size=effective_batch_size)

    def _dummy_expert_batch(self, batch: TensorDict) -> TensorDict:
        """Return a single-transition expert batch with same structure as batch (for compile/CUDA graph)."""
        required_keys = self._expert_required_keys_cached
        available_keys = batch.keys(True)
        if all(key in available_keys for key in required_keys):
            return batch.select(*required_keys)[:1]

        one = batch[:1]
        dummy = TensorDict({}, batch_size=[1], device=batch.device)
        action_spec = getattr(self.env, "action_spec_unbatched", self.env.action_spec)
        action_shape = tuple(int(dim) for dim in action_spec.shape)
        action_dtype = cast(
            torch.dtype, getattr(self.env.action_spec, "dtype", torch.float32)
        )
        for key in required_keys:
            if key in available_keys:
                dummy.set(key, one.get(key))
                continue
            if key == "action":
                dummy_action = torch.zeros(
                    (1, *action_shape),
                    device=batch.device,
                    dtype=action_dtype,
                )
                dummy.set("action", dummy_action)
                continue
            obs_key = strip_next_prefix(key)
            obs_shape = self._obs_key_feature_shape(obs_key)
            obs_dtype = cast(
                torch.dtype,
                getattr(self.env.observation_spec[obs_key], "dtype", torch.float32),
            )
            dummy_obs = torch.zeros(
                (1, *obs_shape),
                device=batch.device,
                dtype=obs_dtype,
            )
            dummy.set(key, dummy_obs)
        return dummy

    def _dummy_loss_tensordict(self) -> TensorDict:
        """Dummy loss TensorDict with same keys as update output (for compile/fallback)."""
        return TensorDict(
            {
                "loss_critic": torch.tensor(0.0, device=self.device),
                "loss_objective": torch.tensor(0.0, device=self.device),
                "loss_entropy": torch.tensor(0.0, device=self.device),
                "loss_reward_diff": torch.tensor(0.0, device=self.device),
                "loss_reward_gap_term": torch.tensor(0.0, device=self.device),
                "loss_reward_l2": torch.tensor(0.0, device=self.device),
                "loss_reward_grad_penalty": torch.tensor(0.0, device=self.device),
                "loss_reward_logit_reg": torch.tensor(0.0, device=self.device),
                "loss_reward_param_decay": torch.tensor(0.0, device=self.device),
                "estimated_reward_mean": torch.tensor(0.0, device=self.device),
                "estimated_reward_std": torch.tensor(0.0, device=self.device),
                "expert_reward_mean": torch.tensor(0.0, device=self.device),
                "expert_reward_std": torch.tensor(0.0, device=self.device),
                "reward_update_mask": torch.tensor(0.0, device=self.device),
                "reward_mix_alpha": torch.tensor(1.0, device=self.device),
                "reward_updates_performed": torch.tensor(0.0, device=self.device),
                "bc_coeff": torch.tensor(0.0, device=self.device),
            },
            batch_size=[],
        )

    def _normalize_reward_inputs(
        self, reward_inputs: Tensor, *, update_stats: bool
    ) -> Tensor:
        assert isinstance(self.config, IPMDRLOptConfig)
        cfg = self.config.ipmd
        if not self._normalize_reward_input_enabled:
            return reward_inputs
        eps = float(cfg.reward_input_norm_eps)
        stats_source = reward_inputs
        squeeze_last_dim = False
        if stats_source.ndim == 1:
            # Scalar-feature case: treat as (batch, feature=1) for stable stats.
            stats_source = stats_source.unsqueeze(-1)
            squeeze_last_dim = True
        flat_stats_source = stats_source.reshape(-1, stats_source.shape[-1])
        feature_dim = int(flat_stats_source.shape[-1])
        if (
            not self._reward_input_stats_initialized
            or self._reward_input_running_mean is None
            or self._reward_input_running_var is None
            or int(self._reward_input_running_mean.shape[-1]) != feature_dim
        ):
            self._reward_input_running_mean = flat_stats_source.detach().mean(dim=0)
            self._reward_input_running_var = flat_stats_source.detach().var(
                dim=0, unbiased=False
            )
            self._reward_input_stats_initialized = True
        assert self._reward_input_running_mean is not None
        assert self._reward_input_running_var is not None
        if update_stats:
            momentum = min(1.0, max(0.0, float(cfg.reward_input_norm_momentum)))
            batch_mean = flat_stats_source.detach().mean(dim=0)
            batch_var = flat_stats_source.detach().var(dim=0, unbiased=False)
            self._reward_input_running_mean = (1.0 - momentum) * (
                self._reward_input_running_mean
            ) + momentum * batch_mean
            self._reward_input_running_var = (1.0 - momentum) * (
                self._reward_input_running_var
            ) + momentum * batch_var
        normalized = (stats_source - self._reward_input_running_mean) / torch.sqrt(
            self._reward_input_running_var + eps
        )
        clip_value = cfg.reward_input_norm_clip
        if clip_value is not None and clip_value > 0.0:
            normalized = normalized.clamp(-float(clip_value), float(clip_value))
        if squeeze_last_dim:
            normalized = normalized.squeeze(-1)
        return normalized

    def _reward_inputs_from_batch(
        self,
        td: TensorDict | Any,
        *,
        detach: bool | None = None,
        update_input_stats: bool = False,
        apply_input_augmentation: bool = False,
    ) -> Tensor:
        """Assemble reward-model inputs according to ``reward_input_type``."""
        if detach is None:
            detach = self._reward_detach_default

        parts: list[Tensor] = []
        if self._reward_include_state:
            parts.append(
                self._obs_features_from_td(
                    td,
                    self._reward_obs_keys,
                    next_obs=False,
                    detach=detach,
                )
            )
        if self._reward_include_action:
            parts.append(self._action_features_from_td(td, detach=detach))
        if self._reward_include_next_state:
            parts.append(
                self._obs_features_from_td(
                    td,
                    self._reward_obs_keys,
                    next_obs=True,
                    detach=detach,
                )
            )

        reward_inputs = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
        return reward_inputs

    def _apply_reward_output_activation(self, reward_logits: Tensor) -> Tensor:
        """Apply configured output activation to reward logits."""
        if self._reward_output_activation_mode == 1:
            return torch.tanh(reward_logits) * self._reward_output_scale
        if self._reward_output_activation_mode == 2:
            return torch.sigmoid(reward_logits) * self._reward_output_scale
        return reward_logits

    def _reward_logits_from_inputs(self, reward_inputs: Tensor) -> Tensor:
        return self.reward_estimator(reward_inputs)

    def _reward_from_inputs(self, reward_inputs: Tensor) -> Tensor:
        reward_logits = self._reward_logits_from_inputs(reward_inputs)
        return self._apply_reward_output_activation(reward_logits)

    def _reward_logits_from_batch(
        self,
        td: TensorDict | Any,
        *,
        detach: bool | None = None,
        update_input_stats: bool = False,
        apply_input_augmentation: bool = False,
    ) -> Tensor:
        reward_inputs = self._reward_inputs_from_batch(
            td,
            detach=detach,
            update_input_stats=update_input_stats,
            apply_input_augmentation=apply_input_augmentation,
        )
        return self._reward_logits_from_inputs(reward_inputs)

    def _reward_from_batch(
        self,
        td: TensorDict | Any,
        *,
        detach: bool | None = None,
        update_input_stats: bool = False,
        apply_input_augmentation: bool = False,
    ) -> Tensor:
        """Compute estimated reward for a batch of transitions."""
        reward_inputs = self._reward_inputs_from_batch(
            td,
            detach=detach,
            update_input_stats=update_input_stats,
            apply_input_augmentation=apply_input_augmentation,
        )
        return self._reward_from_inputs(reward_inputs)

    def _reward_input_grad_penalty_from_inputs(self, expert_inputs: Tensor) -> Tensor:
        expert_inputs = expert_inputs.detach()
        expert_inputs.requires_grad_(True)
        expert_rewards = self._reward_from_inputs(expert_inputs)
        gradients = torch.autograd.grad(
            outputs=expert_rewards.sum(),
            inputs=expert_inputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return gradients.pow(2).sum(dim=-1).mean()

    def _reward_input_grad_penalty(self, expert_batch: TensorDict) -> Tensor:
        expert_inputs = self._reward_inputs_from_batch(
            expert_batch,
            detach=False,
            update_input_stats=False,
        ).detach()
        return self._reward_input_grad_penalty_from_inputs(expert_inputs)

    def update(
        self,
        batch: TensorDict,
        num_network_updates: int | Tensor,
        expert_batch: TensorDict,
        has_expert: Tensor,
        reward_update_mask: Tensor | None = None,
        bc_coeff_override: float | None = None,
        reward_mix_alpha: float = 1.0,
        reward_updates_override: int | None = None,
    ) -> tuple[TensorDict, int]:
        """PPO update plus IPMD reward loss; fixed path for torch.compile and CUDA graphs."""
        self.optim.zero_grad(set_to_none=True)

        # 1) PPO + BC losses (policy/value optimizer)
        loss: TensorDict = self.loss_module(batch)
        critic_loss = loss["loss_critic"]
        actor_loss = loss["loss_objective"] + loss["loss_entropy"]
        total_ppo_loss = critic_loss + actor_loss
        total_ppo_loss.backward()

        output_loss = loss.detach()

        bc_coeff = (
            self._bc_coeff if bc_coeff_override is None else float(bc_coeff_override)
        )
        has_expert_flag = bool(has_expert.detach().item() > 0.0)
        if bc_coeff > 0.0 and has_expert_flag:
            expert_obs_td = expert_batch.select(*self._policy_obs_keys)
            expert_policy_td = self._policy_operator(expert_obs_td)
            loc = expert_policy_td.get("loc")
            scale = expert_policy_td.get("scale")
            expert_action = expert_batch.get("action")
            # Gaussian log-prob: sum over action dims, mean over batch
            log_prob = -0.5 * (
                ((expert_action - loc) / scale).pow(2)
                + 2.0 * scale.log()
                + math.log(2.0 * math.pi)
            )
            log_prob = log_prob.sum(dim=-1)  # sum over action dims
            bc_loss = -log_prob.mean() * bc_coeff
            bc_loss.backward()
            output_loss_bc = bc_loss.detach()
        else:
            output_loss_bc = self._zero_f32

        # PPO gradient step
        if self._max_grad_norm is not None and self._max_grad_norm > 0.0:
            grad_norm_tensor = clip_grad_norm_(
                self._grad_clip_params,
                self._max_grad_norm,
            )
        else:
            grad_norm_tensor = self._zero_f32

        self.optim.step()

        # 2) Reward estimator loss (dedicated optimizer)
        if reward_update_mask is None:
            reward_update_mask = self._one_f32
        reward_updates = self._reward_updates_per_policy_update
        if reward_updates_override is not None:
            reward_updates = max(1, int(reward_updates_override))
        reward_required_keys = self._reward_required_keys
        expert_reward_batch = expert_batch.select(*reward_required_keys)
        reward_update_weight = 1.0 / float(reward_updates)

        reward_diff = self._zero_f32.clone()
        reward_gap_term = self._zero_f32.clone()
        reward_l2 = self._zero_f32.clone()
        reward_grad_penalty = self._zero_f32.clone()
        reward_logit_reg = self._zero_f32.clone()
        reward_param_decay = self._zero_f32.clone()
        reward_grad_norm = self._zero_f32.clone()
        apply_reward_update = has_expert_flag and bool(
            reward_update_mask.detach().item() > 0.0
        )

        if apply_reward_update:
            for reward_update_idx in range(reward_updates):
                self.reward_optim.zero_grad(set_to_none=True)
                reward_batch = self._reward_batch_with_replay(
                    batch, reward_required_keys
                )
                cur_expert_reward_batch = expert_reward_batch
                target_expert_reward_bs = reward_batch.numel()
                if reward_update_idx > 0 or self._reward_balance_batches:
                    sampled_expert_batch = self._next_expert_batch(
                        batch_size=(
                            target_expert_reward_bs
                            if self._reward_balance_batches
                            else None
                        )
                    )
                    if sampled_expert_batch is not None:
                        cur_expert_reward_batch = sampled_expert_batch.to(
                            self.device
                        ).select(*reward_required_keys)

                reward_inputs_pi_aug = self._reward_inputs_from_batch(
                    reward_batch,
                    update_input_stats=True,
                    apply_input_augmentation=True,
                )
                reward_inputs_exp_aug = self._reward_inputs_from_batch(
                    cur_expert_reward_batch,
                    update_input_stats=True,
                    apply_input_augmentation=True,
                )
                reward_logits_pi_aug = self._reward_logits_from_inputs(
                    reward_inputs_pi_aug
                )
                reward_logits_exp_aug = self._reward_logits_from_inputs(
                    reward_inputs_exp_aug
                )
                if self._reward_use_logits:
                    reward_source_pi = reward_logits_pi_aug
                    reward_source_exp = reward_logits_exp_aug
                else:
                    reward_source_pi = self._apply_reward_output_activation(
                        reward_logits_pi_aug
                    )
                    reward_source_exp = self._apply_reward_output_activation(
                        reward_logits_exp_aug
                    )

                reward_diff_step = reward_source_pi.mean() - reward_source_exp.mean()
                if self._reward_use_margin:
                    reward_gap_term_step = torch.relu(
                        reward_diff_step + self._reward_margin
                    )
                else:
                    reward_gap_term_step = reward_diff_step
                reward_l2_step = (
                    reward_source_pi.pow(2).mean() + reward_source_exp.pow(2).mean()
                )

                need_clean_inputs = (
                    self._reward_use_consistency
                    or self._reward_use_logit_reg
                    or (self._reward_use_grad_penalty and self._reward_detach_default)
                )
                reward_inputs_pi_clean: Tensor | None = None
                reward_inputs_exp_clean: Tensor | None = None
                if need_clean_inputs:
                    reward_inputs_pi_clean = self._reward_inputs_from_batch(
                        reward_batch,
                        update_input_stats=False,
                    )
                    reward_inputs_exp_clean = self._reward_inputs_from_batch(
                        cur_expert_reward_batch,
                        update_input_stats=False,
                    )

                reward_grad_penalty_step = self._zero_f32
                if self._reward_use_grad_penalty:
                    if self._reward_detach_default:
                        assert reward_inputs_exp_clean is not None
                        reward_grad_penalty_step = (
                            self._reward_input_grad_penalty_from_inputs(
                                reward_inputs_exp_clean
                            )
                        )
                    else:
                        reward_grad_penalty_step = self._reward_input_grad_penalty(
                            cur_expert_reward_batch
                        )

                reward_logit_reg_step = self._zero_f32
                if self._reward_use_logit_reg:
                    assert reward_inputs_pi_clean is not None
                    assert reward_inputs_exp_clean is not None
                    logits_pi = self._reward_logits_from_inputs(reward_inputs_pi_clean)
                    logits_exp = self._reward_logits_from_inputs(
                        reward_inputs_exp_clean
                    )
                    reward_logit_reg_step = (
                        logits_pi.pow(2).mean() + logits_exp.pow(2).mean()
                    )

                reward_param_decay_step = self._zero_f32
                if self._reward_use_param_decay:
                    for param in self._reward_grad_clip_params:
                        reward_param_decay_step = (
                            reward_param_decay_step + param.pow(2).mean()
                        )

                total_reward_loss_step = (
                    (
                        self._reward_loss_coeff * reward_gap_term_step
                        + self._reward_l2_coeff * reward_l2_step.pow(0.5)
                        + self._reward_grad_penalty_coeff * reward_grad_penalty_step
                        + self._reward_logit_reg_coeff * reward_logit_reg_step
                        + self._reward_param_decay_coeff * reward_param_decay_step
                    )
                    * has_expert
                    * reward_update_mask
                )

                total_reward_loss_step.backward()

                if (
                    self._reward_max_grad_norm is not None
                    and self._reward_max_grad_norm > 0.0
                ):
                    reward_grad_norm_step = clip_grad_norm_(
                        self._reward_grad_clip_params,
                        self._reward_max_grad_norm,
                    )
                else:
                    reward_grad_norm_step = self._zero_f32
                self.reward_optim.step()

                reward_diff.add_(reward_diff_step.detach(), alpha=reward_update_weight)
                reward_gap_term.add_(
                    reward_gap_term_step.detach(), alpha=reward_update_weight
                )
                reward_l2.add_(reward_l2_step.detach(), alpha=reward_update_weight)
                reward_grad_penalty.add_(
                    reward_grad_penalty_step.detach(), alpha=reward_update_weight
                )
                reward_logit_reg.add_(
                    reward_logit_reg_step.detach(), alpha=reward_update_weight
                )
                reward_param_decay.add_(
                    reward_param_decay_step.detach(), alpha=reward_update_weight
                )
                reward_grad_norm.add_(
                    reward_grad_norm_step.detach(), alpha=reward_update_weight
                )
        output_loss.set("alpha", self._one_f32)
        output_loss.set("loss_reward_diff", reward_diff.detach())
        output_loss.set("loss_reward_gap_term", reward_gap_term.detach())
        output_loss.set("loss_reward_l2", reward_l2.detach())
        output_loss.set("loss_reward_grad_penalty", reward_grad_penalty.detach())
        output_loss.set("loss_reward_logit_reg", reward_logit_reg.detach())
        output_loss.set("loss_reward_param_decay", reward_param_decay.detach())
        output_loss.set("loss_bc", output_loss_bc)
        output_loss.set("grad_norm", grad_norm_tensor.detach())
        output_loss.set("reward_grad_norm", reward_grad_norm.detach())
        output_loss.set("reward_update_mask", reward_update_mask.detach())
        output_loss.set(
            "reward_mix_alpha",
            torch.tensor(
                float(reward_mix_alpha), device=self.device, dtype=torch.float32
            ),
        )
        output_loss.set(
            "reward_updates_performed",
            torch.tensor(
                float(reward_updates if apply_reward_update else 0),
                device=self.device,
                dtype=torch.float32,
            ),
        )
        output_loss.set(
            "bc_coeff",
            torch.tensor(float(bc_coeff), device=self.device, dtype=torch.float32),
        )
        output_loss.set(
            "lr",
            torch.tensor(
                self.optim.param_groups[0]["lr"],
                device=self.device,
                dtype=torch.float32,
            ),
        )
        output_loss.set(
            "reward_lr",
            torch.tensor(
                self.reward_optim.param_groups[0]["lr"],
                device=self.device,
                dtype=torch.float32,
            ),
        )
        output_loss.set("skipped_update", self._false_bool)

        with torch.no_grad():
            diag_rewards = self._reward_from_batch(batch, update_input_stats=False)
            output_loss.set("estimated_reward_mean", diag_rewards.mean())
            output_loss.set("estimated_reward_std", diag_rewards.std())
            if has_expert_flag:
                diag_expert_rewards = self._reward_from_batch(
                    expert_reward_batch, update_input_stats=False
                )
                output_loss.set(
                    "expert_reward_mean", diag_expert_rewards.mean().nan_to_num(0.0)
                )
                output_loss.set(
                    "expert_reward_std", diag_expert_rewards.std().nan_to_num(0.0)
                )
            else:
                output_loss.set("expert_reward_mean", self._zero_f32)
                output_loss.set("expert_reward_std", self._zero_f32)

        return output_loss, num_network_updates + 1

    def train(self) -> None:  # type: ignore[override]
        """On-policy train loop (PPO-style) with optional reward replacement and IPMD logging."""
        cfg = self.config
        assert isinstance(cfg, IPMDRLOptConfig)

        collected_frames = 0
        num_network_updates = torch.zeros((), dtype=torch.int64, device=self.device)
        num_network_updates_i = 0
        trainer_cfg = cfg.trainer
        show_progress_bar = (
            True if trainer_cfg is None else bool(trainer_cfg.progress_bar)
        )
        periodic_log_interval = (
            1000 if trainer_cfg is None else max(1, int(trainer_cfg.log_interval))
        )
        pbar = tqdm.tqdm(
            total=self.config.collector.total_frames, disable=not show_progress_bar
        )

        num_mini_batches = cfg.collector.frames_per_batch // cfg.loss.mini_batch_size
        if cfg.collector.frames_per_batch % cfg.loss.mini_batch_size != 0:
            num_mini_batches += 1

        self.total_network_updates = (
            (cfg.collector.total_frames // cfg.collector.frames_per_batch)
            * cfg.loss.epochs
            * num_mini_batches
        )

        cfg_loss_ppo_epochs: int = cfg.loss.epochs
        cfg_loss_anneal_clip_eps: bool = cfg.ppo.anneal_clip_epsilon
        cfg_loss_clip_epsilon: float = cfg.ppo.clip_epsilon

        self.collector: SyncDataCollector
        collector_iter = iter(self.collector)
        total_iter = len(self.collector)
        policy_op = self._policy_operator
        reward_updates_override = self._reward_updates_per_policy_update

        if self._expert_buffer is None and not self._warned_no_expert:
            logging.getLogger(__name__).warning(
                "Expert buffer not set; reward estimator updates will use dummy batch (has_expert=0)."
            )
            self._warned_no_expert = True

        for _i in range(total_iter):
            with timeit("collecting"):
                data = next(collector_iter)

            with timeit("iter_prep"):
                # Logging
                metrics_to_log: dict[str, Any] = {}
                frames_in_batch = data.numel()
                collected_frames += frames_in_batch
                if show_progress_bar:
                    pbar.update(frames_in_batch)

                if ("next", "reward") in data.keys(True):
                    step_rewards = data["next", "reward"]
                    metrics_to_log.update(
                        {
                            "train/step_reward_mean": step_rewards.mean().item(),
                            "train/step_reward_std": step_rewards.std().item(),
                            "train/step_reward_max": step_rewards.max().item(),
                            "train/step_reward_min": step_rewards.min().item(),
                        }
                    )
                if ("next", "episode_reward") in data.keys(True):
                    episode_rewards = data["next", "episode_reward"][
                        data["next", "done"]
                    ]
                    if len(episode_rewards) > 0:
                        episode_length = data["next", "step_count"][
                            data["next", "done"]
                        ]
                        episode_length_list = episode_length.cpu().tolist()
                        episode_rewards_list = episode_rewards.cpu().tolist()
                        self.episode_lengths.extend(episode_length_list)
                        self.episode_rewards.extend(episode_rewards_list)
                        episode_rewards_mean = float(np.mean(episode_rewards_list))
                        metrics_to_log.update(
                            {
                                "episode/length": np.mean(self.episode_lengths),
                                "episode/return": np.mean(self.episode_rewards),
                                "train/reward": episode_rewards_mean,
                            }
                        )

            # Reward mixing (simplified: no schedule or random action injection).
            reward_mix_alpha_for_update = 1.0

            # Optionally replace env rewards with estimated rewards for PPO (before GAE)
            if self._use_estimated_rewards_for_ppo:
                with timeit("reward_estimation_rollout"):
                    with torch.no_grad():
                        est_rew = self._reward_from_batch(
                            data,
                            update_input_stats=False,
                        )
                    if self._detach_reward_when_used_for_ppo:
                        est_rew = est_rew.detach()
                    env_rew = data.get(("next", "reward"))
                    # Use only estimated rewards for PPO (no scheduling or gating).
                    reward_mix_alpha_for_update = 1.0
                    mixed_rew = est_rew + env_rew
                    data.set(("next", "env_reward"), env_rew)
                    data.set(("next", "estimated_reward"), est_rew)
                    data.set(("next", "reward"), mixed_rew)
                    metrics_to_log["train/reward_mix_alpha"] = (
                        reward_mix_alpha_for_update
                    )
                    metrics_to_log["train/estimated_reward_std_rollout"] = (
                        est_rew.std().item()
                    )
                    metrics_to_log["train/ppo_reward_std_rollout"] = (
                        mixed_rew.std().item()
                    )

            # Training
            with timeit("training"):
                loss_sums: dict[str, Tensor] = {}
                loss_count = 0
                with timeit("rb - empty"):
                    self.data_buffer.empty()
                for j in range(cfg_loss_ppo_epochs):
                    with timeit("epoch"):
                        with torch.no_grad(), timeit("adv"):
                            data = self.adv_module(data)
                            if self._compile_mode_enabled:
                                data = data.clone()

                        with timeit("rb - extend"):
                            self.data_buffer.extend(data.reshape(-1))

                        for batch in self.data_buffer:
                            should_update_reward = False
                            kl_context = None
                            expert_batch: TensorDict
                            has_expert: Tensor
                            reward_update_mask: Tensor
                            with timeit("update_prep"):
                                # Add on-policy transitions to reward replay (if enabled).
                                self._maybe_reset_reward_replay(num_network_updates_i)
                                self._store_reward_replay_samples(batch)

                                entropy_coeff = self._current_entropy_coeff(
                                    num_network_updates_i
                                )
                                self._set_loss_entropy_coeff(entropy_coeff)
                                metrics_to_log["train/entropy_coeff_active"] = (
                                    entropy_coeff
                                )

                                should_update_reward = (
                                    num_network_updates_i >= self._reward_update_warmup
                                    and (
                                        num_network_updates_i
                                        % self._reward_update_interval
                                    )
                                    == 0
                                )
                                reward_update_mask = (
                                    self._one_f32
                                    if should_update_reward
                                    else self._zero_f32
                                )
                                need_expert_batch = should_update_reward or (
                                    self._bc_coeff > 0.0
                                )

                                if self._adaptive_kl_scheduler:
                                    kl_context = self._prepare_kl_context(
                                        batch, policy_op
                                    )
                                # Fixed inputs for torch.compile / CUDA graph
                                with timeit("expert_batch_fetch"):
                                    if need_expert_batch:
                                        expert_batch_raw = self._next_expert_batch()
                                        if expert_batch_raw is None:
                                            if not self._warned_no_expert:
                                                self.log.warning(
                                                    "No expert batch found"
                                                )
                                                self._warned_no_expert = True
                                            expert_batch = self._dummy_expert_batch(
                                                batch
                                            )
                                            has_expert = self._zero_f32
                                        else:
                                            expert_batch = expert_batch_raw.to(
                                                self.device
                                            )
                                            has_expert = self._one_f32
                                    else:
                                        expert_batch = self._dummy_expert_batch(batch)
                                        has_expert = self._zero_f32
                            with timeit("update"):
                                loss, num_network_updates = self.update(
                                    batch,
                                    num_network_updates,
                                    expert_batch,
                                    has_expert,
                                    reward_update_mask=reward_update_mask,
                                    bc_coeff_override=self._bc_coeff,
                                    reward_mix_alpha=reward_mix_alpha_for_update,
                                    reward_updates_override=reward_updates_override,
                                )
                                num_network_updates_i += 1
                            with timeit("scheduler_step_update"):
                                if (
                                    self.lr_scheduler
                                    and self.lr_scheduler_step == "update"
                                ):
                                    self.lr_scheduler.step()
                                if (
                                    self.reward_lr_scheduler is not None
                                    and self.reward_lr_scheduler_step == "update"
                                    and should_update_reward
                                ):
                                    self.reward_lr_scheduler.step()
                            if kl_context is not None:
                                with timeit("adaptive_kl"):
                                    kl_approx = self._compute_kl_after_update(
                                        kl_context, policy_op
                                    )
                                    if kl_approx is not None:
                                        loss.set("kl_approx", kl_approx.detach())
                                        self._maybe_adjust_lr(kl_approx, cfg.optim)
                            with timeit("loss_record"):
                                loss_keys: list[str] = [
                                    key for key in self._loss_record_keys if key in loss
                                ]
                                for key in self._loss_record_optional_keys:
                                    if key in loss:
                                        loss_keys.append(key)
                                for key in loss_keys:
                                    val = cast(Tensor, loss.get(key)).detach()
                                    if key in loss_sums:
                                        loss_sums[key].add_(val)
                                    else:
                                        loss_sums[key] = val.clone()
                                loss_count += 1

                        with timeit("scheduler_step_epoch"):
                            if self.lr_scheduler and self.lr_scheduler_step == "epoch":
                                self.lr_scheduler.step()
                            if (
                                self.reward_lr_scheduler is not None
                                and self.reward_lr_scheduler_step == "epoch"
                            ):
                                self.reward_lr_scheduler.step()

            # EPIC distance between estimated and true env reward (invariant to
            # potential-based shaping and positive rescaling).
            with timeit("reward_diagnostics"):
                with torch.no_grad():
                    if self._use_estimated_rewards_for_ppo and (
                        "next",
                        "env_reward",
                    ) in data.keys(True):
                        env_rew = data["next", "env_reward"]
                        if ("next", "estimated_reward") in data.keys(True):
                            est_rew_diag = data["next", "estimated_reward"]
                        else:
                            est_rew_diag = data["next", "reward"]
                        metrics_to_log["train/env_reward_mean"] = env_rew.mean().item()
                        metrics_to_log["train/env_reward_std"] = env_rew.std().item()
                    elif ("next", "reward") in data.keys(True):
                        env_rew = data["next", "reward"]
                        est_rew_diag = self._reward_from_batch(
                            data, update_input_stats=False
                        )
                    else:
                        env_rew = None
                        est_rew_diag = None
                    if env_rew is not None and est_rew_diag is not None:
                        pearson_corr, epic_dist = epic_distance(est_rew_diag, env_rew)
                        metrics_to_log["reward/pearson_corr"] = pearson_corr.item()
                        metrics_to_log["reward/epic_distance"] = epic_dist.item()

            # Aggregate and log losses
            with timeit("log_aggregate"):
                if loss_count > 0:
                    inv_loss_count = 1.0 / float(loss_count)
                    for key, value in loss_sums.items():
                        metrics_to_log[f"train/{key}"] = (
                            value.float() * inv_loss_count
                        ).item()
                metrics_to_log["train/lr"] = self.optim.param_groups[0]["lr"]
                metrics_to_log["train/reward_lr"] = self.reward_optim.param_groups[0][
                    "lr"
                ]
                clip_attr = getattr(self.loss_module, "clip_epsilon", None)
                if cfg_loss_anneal_clip_eps and isinstance(clip_attr, torch.Tensor):
                    clip_epsilon_value = clip_attr.detach()
                else:
                    clip_epsilon_value = torch.tensor(
                        cfg_loss_clip_epsilon,
                        device=self.device,
                        dtype=torch.float32,
                    )
                metrics_to_log["train/clip_epsilon"] = clip_epsilon_value

            with timeit("env_log_info"):
                if self._is_isaac_env and hasattr(self.env, "log_infos"):
                    log_info_dict: dict[str, Tensor] = self.env.log_infos.popleft()
                    log_info(log_info_dict, metrics_to_log)

            with timeit("metrics_flush"):
                metrics_to_log.update(timeit.todict(prefix="time"))  # type: ignore
                rate = pbar.format_dict.get("rate") if show_progress_bar else None
                if rate is not None:
                    metrics_to_log["time/speed"] = rate
                self.log_metrics(metrics_to_log, step=collected_frames)
                self.collector.update_policy_weights_()

            postfix = {}
            if "train/step_reward_mean" in metrics_to_log:
                postfix["r_step"] = f"{metrics_to_log['train/step_reward_mean']:.2f}"
            if "episode/return" in metrics_to_log:
                postfix["r_ep"] = f"{metrics_to_log['episode/return']:.1f}"
            if "train/loss_objective" in metrics_to_log:
                postfix["pi_loss"] = f"{metrics_to_log['train/loss_objective']:.3f}"
            if "train/loss_reward_diff" in metrics_to_log:
                postfix["reward_diff"] = (
                    f"{metrics_to_log['train/loss_reward_diff']:.3f}"
                )
            if show_progress_bar and postfix:
                pbar.set_postfix(postfix)
            elif not show_progress_bar and (
                (_i + 1) % periodic_log_interval == 0 or (_i + 1) == total_iter
            ):
                status_parts = [
                    f"iter={_i + 1}/{total_iter}",
                    f"frames={collected_frames}/{cfg.collector.total_frames}",
                ]
                summary_metrics: tuple[tuple[str, str], ...] = (
                    ("train/step_reward_mean", "r_step"),
                    ("episode/return", "r_ep"),
                    ("train/loss_objective", "pi_loss"),
                    ("train/loss_reward_diff", "reward_diff"),
                    ("time/speed", "fps"),
                )
                for metric_key, alias in summary_metrics:
                    metric_value = metrics_to_log.get(metric_key)
                    if isinstance(metric_value, (int, float, np.floating)):
                        status_parts.append(f"{alias}={float(metric_value):.4f}")
                self.log.info(" | ".join(status_parts))

            if (
                self.config.save_interval > 0
                and (num_network_updates_i % self.config.save_interval) == 0
            ):
                with timeit("checkpoint_save"):
                    self.save_model(
                        path=self.log_dir / self.config.logger.save_path,
                        step=collected_frames,
                    )

        pbar.close()
        self.collector.shutdown()

    def validate_ipmd_loss(
        self, test_batch: TensorDict, expert_batch: TensorDict
    ) -> dict[str, float]:
        """Validate IPMD loss computation (reward diff + L2) and PPO loss on test data."""
        for m in (self.actor_critic, self.reward_estimator):
            if hasattr(m, "eval"):
                m.eval()
        with torch.no_grad():
            estimated_rewards = self._reward_from_batch(test_batch)
            expert_rewards = self._reward_from_batch(expert_batch)
            reward_diff = (estimated_rewards.sum() - expert_rewards.sum()).item()
            l2_reg = sum(
                p.pow(2).sum().item() for p in self.reward_estimator.parameters()
            )
            ppo_loss_td = self.loss_module(test_batch)
            return {
                "reward_diff": reward_diff,
                "reward_l2": l2_reg,
                "estimated_reward_mean": estimated_rewards.mean().item(),
                "estimated_reward_std": estimated_rewards.std().item(),
                "expert_reward_mean": expert_rewards.mean().item(),
                "expert_reward_std": expert_rewards.std().item(),
                "ppo_loss_critic": ppo_loss_td["loss_critic"].item(),
                "ppo_loss_objective": ppo_loss_td["loss_objective"].item(),
                "ppo_loss_entropy": ppo_loss_td["loss_entropy"].item(),
            }

    def predict(self, obs: Tensor | np.ndarray | Mapping[Any, Any]) -> Tensor:  # type: ignore[override]
        """Predict action given observation (deterministic)."""
        policy_op = self._policy_operator
        policy_op.eval()
        with torch.no_grad(), set_exploration_type(InteractionType.DETERMINISTIC):
            input_keys = list(self._policy_obs_keys)
            td_data: dict[BatchKey, Tensor] = {}
            batch_shape: tuple[int, ...] | None = None

            if isinstance(obs, Mapping):
                for key in input_keys:
                    value = torch.as_tensor(
                        mapping_get_obs_value(obs, key),
                        device=self.device,
                    )
                    feature_ndim = self._obs_feature_ndims[key]
                    if (feature_ndim == 0 and value.ndim == 0) or (
                        feature_ndim > 0 and value.ndim == feature_ndim
                    ):
                        value = value.unsqueeze(0)
                    current_batch_shape = infer_batch_shape(value, feature_ndim)
                    if batch_shape is None:
                        batch_shape = current_batch_shape
                    elif current_batch_shape != batch_shape:
                        msg = (
                            "All observation tensors passed to predict() must have the "
                            f"same batch shape, got {batch_shape} and {current_batch_shape}."
                        )
                        raise ValueError(msg)
                    td_data[key] = value
            else:
                if len(input_keys) != 1:
                    msg = (
                        "predict() received a single tensor observation, but policy "
                        f"expects multiple keys {input_keys}. Pass a dict instead."
                    )
                    raise ValueError(msg)
                key = input_keys[0]
                value = torch.as_tensor(obs, device=self.device)
                feature_ndim = self._obs_feature_ndims[key]
                if (feature_ndim == 0 and value.ndim == 0) or (
                    feature_ndim > 0 and value.ndim == feature_ndim
                ):
                    value = value.unsqueeze(0)
                td_data[key] = value
                batch_shape = infer_batch_shape(value, feature_ndim)

            td = TensorDict(
                td_data, batch_size=list(batch_shape or [1]), device=self.device
            )
            td = policy_op(td)
            return td.get("action")
