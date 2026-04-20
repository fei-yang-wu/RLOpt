from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict
from tensordict.nn import InteractionType
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torchrl._utils import timeit
from torchrl.data import ReplayBuffer
from torchrl.envs.utils import set_exploration_type
from torchrl.modules import MLP
from torchrl.record.loggers import Logger

from rlopt.agent.imitation.latent_commands import LatentCommandController
from rlopt.agent.imitation.latent_learning import (
    BaseLatentLearner,
    build_latent_learner,
)
from rlopt.agent.ipmd.utils import (
    IPMD_COMMAND_SOURCES,
    IPMD_REWARD_INPUT_TYPES,
    IPMD_REWARD_OUTPUT_ACTIVATIONS,
    RewardInputBlock,
    build_reward_input_blocks,
    normalize_ipmd_command_source,
    normalize_ipmd_reward_input_type,
    normalize_ipmd_reward_output_activation,
    require_non_negative,
    require_positive_int,
    required_batch_keys_from_reward_blocks,
    reward_alignment_metrics,
    reward_grad_penalty_from_input,
)
from rlopt.agent.ppo.ppo import (
    PPO,
    PPOConfig,
    PPOIterationData,
    PPORLOptConfig,
    PPOTrainingMetadata,
)
from rlopt.config_utils import (
    BatchKey,
    ObsKey,
    dedupe_keys,
    flatten_action_features_from_td,
    flatten_obs_features_from_td,
    infer_batch_shape,
    mapping_get_obs_value,
    obs_key_feature_dim,
    obs_key_feature_ndim,
)
from rlopt.type_aliases import OptimizerClass
from rlopt.utils import get_activation_class, log_info


@dataclass
class IPMDLatentLearningConfig:
    """Nested latent-learning configuration used by IsaacLab imitation tasks."""

    method: str = "patch_autoencoder"
    """Latent learning method to use."""

    posterior_input_keys: list[ObsKey] = field(default_factory=list)
    """Keys for posterior."""

    prior_input_keys: list[ObsKey] = field(default_factory=list)
    """Keys for prior. """

    encoder_hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    """Encoder hidden dimension if use NN. """

    encoder_activation: str = "elu"
    """Encoder activation if use NN. """

    decoder_hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    """Decoder hidden dimension if use NN with a decoder attached, either for computing reconstructions or generating samples. """

    decoder_activation: str = "elu"
    """Decoder activation if use NN with a decoder attached, either for computing reconstructions or generating samples. """

    prior_hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    """Prior network hidden dimension if use NN for the prior. E.g. for state-dependent Gaussian prior. """

    prior_activation: str = "elu"
    """Prior network activation if use NN for the prior. """

    patch_past_steps: int = 0
    patch_future_steps: int = 0
    """A patch is a sliding window over the expert trajectory segment. The past and future steps are defined relative to the current time step. For example, with patch_past_steps=2 and patch_future_steps=3, the patch at time t will include observations from time steps [t-2, t-1, t, t+1, t+2, t+3]. Will pad if not enough past or future steps are available. """

    lr: float = 3e-4
    """Learning rate for the latent learner. """

    grad_clip_norm: float = 1.0
    """Gradient clipping norm for the latent learner. """

    recon_coeff: float = 0.0
    """Reconstruction loss coefficient. Only used if the latent learning method involves a decoder and reconstruction loss. If recon_coeff = 0, we will not update encoder and decoder with the reconstruction loss, but let the encoder update through policy improvements. """

    weight_decay_coeff: float = 0.0
    """Weight decay coefficient for the latent learner parameters. """

    kl_coeff: float = 0.0
    """KL regularization penalty. """

    probe_enabled: bool = False
    probe_condition_on_state: bool = False
    probe_target_keys: list[ObsKey] = field(default_factory=list)
    probe_hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    probe_activation: str = "elu"
    probe_lr: float = 3e-4
    probe_grad_clip_norm: float = 1.0
    probe_batch_size: int = 8192
    """Probe for analyzing the latent learner. """


@dataclass
class IPMDConfig(PPOConfig):
    """IPMD-specific configuration (PPO-based)."""

    use_latent_command: bool = True
    """Whether to use agent-managed latent commands."""

    latent_dim: int = 16
    """Dimension of the latent skill space."""

    latent_key: ObsKey = "latent_command"
    """Key for the latent skill."""

    latent_steps_min: int = 30
    """Minimum steps before resampling latent."""

    latent_steps_max: int = 120
    """Maximum steps before resampling latent."""

    command_source: str = "random"
    """Source used to generate rollout latent commands.

    Supported values:
    - ``"random"`` - sample unit latents uniformly at random
    - ``"posterior"`` - encode the current rollout posterior inputs and
      publish the latent state as the latent command
    """

    latent_learning: IPMDLatentLearningConfig = field(
        default_factory=IPMDLatentLearningConfig
    )
    """Nested latent-learning configuration used by IsaacLab imitation tasks."""

    diversity_bonus_coeff: float = 0.05
    """Coefficient for the diversity bonus."""

    diversity_target: float = 1.0
    """Target diversity level."""

    latent_uniformity_temperature: float = 2.0
    """Temperature for the latent uniformity loss."""

    # Reward estimator network and loss settings
    reward_input_type: str = "sas"
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

    reward_grad_penalty_coeff: float = 0.5
    """Gradient penalty weight for reward parameters."""

    reward_detach_features: bool = True
    """Detach features when computing reward loss (avoid leaking grads)."""

    use_estimated_rewards_for_ppo: bool = False
    """Whether to use estimated rewards instead of environment rewards for PPO (GAE + value target).

    Default is False. Set to True to train PPO on estimated rewards.
    """

    estimated_reward_clamp_min: float | None = -np.inf
    """Optional lower bound applied to estimated rewards before PPO reward mixing.

    Set to ``None`` to disable lower clipping.
    """

    estimated_reward_clamp_max: float | None = np.inf
    """Optional upper bound applied to estimated rewards before PPO reward mixing.

    Set to ``None`` to disable upper clipping.
    """

    est_reward_weight: float = 0.3
    """Linear mixing coefficient for estimated rewards in PPO.
    """

    env_reward_weight: float = 1.0
    """Linear mixing coefficient for environment rewards in PPO.
    """

    estimated_reward_done_penalty: float = 0.0
    """Penalty subtracted from estimated reward on terminal (non-truncated) steps.

    This is applied only when ``next.done`` is true and ``next.truncated`` is false.
    Set to 0 to disable.
    """

    expert_batch_size: int | None = None
    """Batch size for expert data sampling. If None, uses the same as mini_batch_size."""

    bc_coef: float = 0.0
    """Behavior cloning (MLE) loss coefficient on expert actions.

    When > 0, adds ``-bc_coef * mean(log_prob(expert_action | policy))``
    to each update step.  This regularises the policy toward expert actions
    during early training.  Set to 0 to disable (default).
    """

    def validate(self) -> None:
        self.command_source = normalize_ipmd_command_source(self.command_source)
        self.reward_input_type = normalize_ipmd_reward_input_type(
            self.reward_input_type
        )
        self.reward_output_activation = normalize_ipmd_reward_output_activation(
            self.reward_output_activation
        )
        self.latent_dim = require_positive_int("ipmd.latent_dim", self.latent_dim)
        self.latent_steps_min = require_positive_int(
            "ipmd.latent_steps_min", self.latent_steps_min
        )
        self.latent_steps_max = require_positive_int(
            "ipmd.latent_steps_max", self.latent_steps_max
        )
        if self.latent_steps_min > self.latent_steps_max:
            msg = (
                "ipmd.latent_steps_min must be <= ipmd.latent_steps_max, got "
                f"{self.latent_steps_min} > {self.latent_steps_max}."
            )
            raise ValueError(msg)
        require_non_negative("ipmd.diversity_bonus_coeff", self.diversity_bonus_coeff)
        require_non_negative("ipmd.diversity_target", self.diversity_target)
        if float(self.latent_uniformity_temperature) <= 0.0:
            msg = (
                "ipmd.latent_uniformity_temperature must be > 0, got "
                f"{self.latent_uniformity_temperature!r}."
            )
            raise ValueError(msg)
        if any(int(width) <= 0 for width in self.reward_num_cells):
            msg = (
                "ipmd.reward_num_cells must contain only positive widths, got "
                f"{self.reward_num_cells!r}."
            )
            raise ValueError(msg)
        if (
            self.reward_output_activation != "none"
            and float(self.reward_output_scale) <= 0.0
        ):
            msg = (
                "ipmd.reward_output_scale must be > 0 when reward_output_activation "
                f"is {self.reward_output_activation!r}, got {self.reward_output_scale!r}."
            )
            raise ValueError(msg)
        require_non_negative("ipmd.reward_loss_coeff", self.reward_loss_coeff)
        require_non_negative("ipmd.reward_l2_coeff", self.reward_l2_coeff)
        require_non_negative(
            "ipmd.reward_grad_penalty_coeff", self.reward_grad_penalty_coeff
        )
        require_non_negative("ipmd.est_reward_weight", self.est_reward_weight)
        require_non_negative("ipmd.env_reward_weight", self.env_reward_weight)
        require_non_negative("ipmd.bc_coef", self.bc_coef)
        if self.expert_batch_size is not None:
            self.expert_batch_size = require_positive_int(
                "ipmd.expert_batch_size", self.expert_batch_size
            )
        clamp_min = self.estimated_reward_clamp_min
        clamp_max = self.estimated_reward_clamp_max
        if clamp_min is not None and clamp_max is not None and clamp_min > clamp_max:
            msg = (
                "ipmd.estimated_reward_clamp_min must be <= "
                f"ipmd.estimated_reward_clamp_max, got {clamp_min!r} > {clamp_max!r}."
            )
            raise ValueError(msg)

    def __post_init__(self) -> None:
        self.validate()


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

    _REWARD_INPUT_TYPES = IPMD_REWARD_INPUT_TYPES
    _REWARD_OUTPUT_ACTIVATIONS = IPMD_REWARD_OUTPUT_ACTIVATIONS
    _COMMAND_SOURCES = IPMD_COMMAND_SOURCES

    @staticmethod
    def _normalize_command_source(command_source: str) -> str:
        return normalize_ipmd_command_source(command_source)

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
        self.config: IPMDRLOptConfig = config
        self.config.ipmd.validate()
        self.env = env
        self._use_latent_command = bool(self.config.ipmd.use_latent_command)
        self._latent_key = self.config.ipmd.latent_key
        self._latent_dim = int(self.config.ipmd.latent_dim)
        self._latent_command_controller: LatentCommandController | None = None
        self._command_source = self._normalize_command_source(
            self.config.ipmd.command_source
        )

        if self._use_latent_command:
            self._latent_command_controller = LatentCommandController(
                env=env,
                latent_key=self._latent_key,
                latent_dim=self._latent_dim,
                latent_steps_min=int(self.config.ipmd.latent_steps_min),
                latent_steps_max=int(self.config.ipmd.latent_steps_max),
                discover_env_method=self._discover_env_method,
            )

        # Observation key groups can differ across policy, value, and reward model.
        self._policy_obs_keys: list[ObsKey] = self.config.policy.get_input_keys()
        value_cfg = self.config.value_function
        self._value_obs_keys: list[ObsKey] = (
            value_cfg.get_input_keys()
            if value_cfg is not None
            else list(self._policy_obs_keys)
        )
        self._policy_obs_keys_without_latent: list[ObsKey] = [
            key for key in self._policy_obs_keys if key != self._latent_key
        ]
        reward_keys = self.config.ipmd.reward_input_keys
        if not reward_keys:
            reward_keys = (
                value_cfg.get_input_keys()
                if value_cfg is not None
                else self._policy_obs_keys_without_latent
            )
        self._reward_obs_keys: list[ObsKey] = dedupe_keys(list(reward_keys))
        latent_cfg = self.config.ipmd.latent_learning
        posterior_keys = latent_cfg.posterior_input_keys or self._reward_obs_keys
        self._posterior_obs_keys: list[ObsKey] = dedupe_keys(list(posterior_keys))
        self._prior_obs_keys: list[ObsKey] = dedupe_keys(
            list(latent_cfg.prior_input_keys)
        )
        self._current_reference_obs_getter = None

        available_keys = set(env.observation_spec.keys(True))
        latent_present = self._latent_key in available_keys
        if self._use_latent_command != latent_present:
            if self._use_latent_command:
                msg = (
                    "IPMD use_latent_command=True requires the environment to expose "
                    f"the latent observation key {self._latent_key!r}."
                )
            else:
                msg = (
                    "IPMD use_latent_command=False requires the environment to omit "
                    f"the latent observation key {self._latent_key!r}."
                )
            msg += self._latent_mode_hint()
            raise ValueError(msg)
        if self._use_latent_command and self._latent_key not in self._policy_obs_keys:
            msg = (
                "IPMD use_latent_command=True requires the policy input keys to "
                f"contain {self._latent_key!r}."
            )
            msg += self._latent_mode_hint()
            raise ValueError(msg)
        if not self._use_latent_command and any(
            self._latent_key in keys
            for keys in (
                self._policy_obs_keys,
                self._value_obs_keys,
                self._reward_obs_keys,
            )
        ):
            msg = (
                "IPMD use_latent_command=False requires policy/value/reward input "
                "keys to exclude the latent command."
            )
            msg += self._latent_mode_hint()
            raise ValueError(msg)

        all_obs_keys = dedupe_keys(
            self._policy_obs_keys
            + self._value_obs_keys
            + self._reward_obs_keys
            + self._posterior_obs_keys
            + self._prior_obs_keys
        )
        for key in all_obs_keys:
            env.observation_spec[key]
        self._obs_feature_ndims: dict[ObsKey, int] = {
            key: obs_key_feature_ndim(self.env, key) for key in all_obs_keys
        }
        self._obs_feature_dims: dict[ObsKey, int] = {
            key: obs_key_feature_dim(self.env, key) for key in all_obs_keys
        }
        action_spec = getattr(self.env, "action_spec_unbatched", self.env.action_spec)
        action_shape = tuple(int(dim) for dim in action_spec.shape)
        self._action_feature_ndim = len(action_shape)
        self._action_feature_dim = int(math.prod(action_shape)) if action_shape else 1
        self._reward_input_blocks: tuple[RewardInputBlock, ...] = (
            build_reward_input_blocks(
                reward_input_type=self.config.ipmd.reward_input_type,
                reward_obs_keys=self._reward_obs_keys,
                obs_feature_dims=self._obs_feature_dims,
                action_feature_dim=self._action_feature_dim,
                use_latent_command=self._use_latent_command,
                latent_dim=self._latent_dim,
            )
        )
        self._reward_input_dim: int = sum(
            block.dim for block in self._reward_input_blocks
        )

        # Reward estimator: r(s, a, s') -> scalar
        self.reward_estimator: torch.nn.Module = self._construct_reward_estimator()
        self.reward_estimator.to(self.device)

        self._expert_batch_sampler: (
            Callable[[int, list[BatchKey]], TensorDict | None] | None
        ) = None

        self._latent_learner: BaseLatentLearner | None = None

        cfg = self.config.ipmd
        # Scalar caches
        self._reward_loss_coeff: float = float(cfg.reward_loss_coeff)
        self._reward_l2_coeff: float = float(cfg.reward_l2_coeff)
        self._reward_grad_penalty_coeff: float = float(cfg.reward_grad_penalty_coeff)
        self._use_estimated_rewards_for_ppo: bool = bool(
            cfg.use_estimated_rewards_for_ppo
        )
        self._bc_coeff: float = float(cfg.bc_coef)
        self._reward_update_enabled: bool = any(
            coeff > 0.0
            for coeff in (
                self._reward_loss_coeff,
                self._reward_l2_coeff,
                self._reward_grad_penalty_coeff,
            )
        )
        self._expert_minibatch_update_enabled: bool = bool(
            self._bc_coeff > 0.0 or self._reward_update_enabled
        )
        self._reward_detach_features: bool = bool(cfg.reward_detach_features)
        max_grad = getattr(self.config.optim, "max_grad_norm", None)
        self._max_grad_norm: float = float(max_grad) if max_grad else 1e10
        # Output activation as a callable (eliminates string dispatch at call time)
        out_act = cfg.reward_output_activation
        scale = float(cfg.reward_output_scale)
        if out_act == "tanh":
            self._reward_out_fn: Callable[[Tensor], Tensor] = lambda r: (
                torch.tanh(r) * scale
            )
        elif out_act == "sigmoid":
            self._reward_out_fn = lambda r: torch.sigmoid(r) * scale
        else:
            self._reward_out_fn = lambda r: r

        if self._use_latent_command:
            method = str(self.config.ipmd.latent_learning.method)
            self._latent_learner = build_latent_learner(method)
            self._latent_learner.initialize(self)

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
        self._auto_attach_env_expert_sampler()
        self._refresh_grad_clip_params()
        self._policy_operator = self.actor_critic.get_policy_operator()
        self._bc_debug_anomaly_prints = 0

    def _require_latent_command_controller(self) -> LatentCommandController:
        controller = self._latent_command_controller
        if controller is None:
            msg = (
                "IPMD latent-command controller is unavailable. "
                "Construct the agent with ipmd.use_latent_command=True."
            )
            raise RuntimeError(msg)
        return controller

    @property
    def collector_policy(self):
        """Return the collector policy, optionally stamping latent commands."""
        policy_operator = self.actor_critic.get_policy_operator()
        if not self._use_latent_command:
            return policy_operator
        controller = self._require_latent_command_controller()
        return controller.collector_policy(
            inject_fn=self._inject_latent_command,
            policy_module=policy_operator,
        )

    def _inject_latent_command(
        self,
        td: TensorDict,
    ) -> None:
        controller = self._require_latent_command_controller()
        if self._command_source == "random":
            controller.inject_random_latent_command(
                td,
                device=self.device,
                dtype=torch.float32,
            )
            return

        assert self._latent_learner is not None
        latents = self._latent_learner.infer_batch_latents(
            td,
            detach=True,
            context="collector posterior latent",
        )
        if latents is None:
            msg = (
                "Active latent learner does not support collector-time posterior "
                "latent inference."
            )
            raise RuntimeError(msg)
        latents = latents.to(
            device=self.device,
            dtype=torch.float32,
        )
        td.set(
            self._latent_key,
            latents.reshape(*td.batch_size, self._latent_dim),
        )
        published_latents = latents.reshape(-1, self._latent_dim)
        controller.publish_latents_to_env(published_latents)

    def _expert_latents_from_td(
        self,
        td: TensorDict,
        *,
        detach: bool,
    ) -> Tensor:
        assert self._latent_learner is not None
        return self._latent_learner.infer_expert_latents(
            td,
            detach=detach,
        ).to(self.device)

    def _expert_required_keys(self) -> list[BatchKey]:
        """Return expert-batch keys required by current IPMD settings."""
        bc_enabled = float(self.config.ipmd.bc_coef) > 0.0

        required = self._reward_required_batch_keys()
        if self._use_latent_command:
            assert self._latent_learner is not None
            required.extend(self._latent_learner.required_expert_batch_keys())
        if bc_enabled:
            required.append("expert_action")
            required.extend(
                key
                for key in self._policy_obs_keys
                if not self._use_latent_command or key != self._latent_key
            )
            if self._use_latent_command:
                assert self._latent_learner is not None
                required.extend(self._latent_learner.required_expert_batch_keys())
        return dedupe_keys(required)

    def _reward_required_batch_keys(self) -> list[BatchKey]:
        """Return the batch keys needed to materialize reward-model inputs."""
        return required_batch_keys_from_reward_blocks(self._reward_input_blocks)

    def _construct_reward_estimator(self) -> torch.nn.Module:
        """Create reward network whose input depends on ``reward_input_type``."""
        cfg = self.config
        assert isinstance(cfg, IPMDRLOptConfig)

        net = MLP(
            in_features=self._reward_input_dim,
            out_features=1,
            num_cells=list(cfg.ipmd.reward_num_cells),
            activation_class=get_activation_class(cfg.ipmd.reward_activation),
            device=self.device,
        )
        self._initialize_weights(net, cfg.ipmd.reward_init)
        return net

    def _compile_components(self) -> None:
        """Compile reward estimator and update method with torch.compile (if enabled)."""
        if not self.config.compile.compile:
            return
        super()._compile_components()
        self.reward_estimator = torch.compile(self.reward_estimator)

    def _set_optimizers(
        self, optimizer_cls: OptimizerClass, optimizer_kwargs: dict[str, Any]
    ) -> list[torch.optim.Optimizer]:
        """Create optimizers for PPO and the reward estimator."""
        if not hasattr(self, "reward_estimator"):
            return super()._set_optimizers(optimizer_cls, optimizer_kwargs)
        all_params = list(self.actor_critic.parameters()) + list(
            self.reward_estimator.parameters()
        )
        return [optimizer_cls(all_params, **optimizer_kwargs)]

    def _next_expert_batch(
        self,
        batch_size: int | None = None,
        required_keys: list[BatchKey] | None = None,
    ) -> TensorDict:
        assert isinstance(self.config, IPMDRLOptConfig)
        effective_batch_size = int(
            batch_size
            or self.config.ipmd.expert_batch_size
            or self.config.loss.mini_batch_size
        )

        if self._expert_batch_sampler is None:
            msg = (
                "IPMD training requires env.sample_expert_batch(...). "
                "Tests may install a private expert sampler override."
            )
            raise RuntimeError(msg)
        required_keys = (
            self._expert_required_keys() if required_keys is None else required_keys
        )
        expert_batch = self._expert_batch_sampler(
            effective_batch_size,
            required_keys,
        )
        if expert_batch is None:
            msg = "Expert sampler returned None."
            raise RuntimeError(msg)
        available_keys = set(expert_batch.keys(True))
        missing_keys = [key for key in required_keys if key not in available_keys]
        if missing_keys:
            msg = f"Expert sampler batch is missing required keys: {missing_keys!r}."
            raise RuntimeError(msg)
        if expert_batch.numel() != effective_batch_size:
            msg = (
                "Expert sampler must return exactly the requested batch size, got "
                f"{expert_batch.numel()} for request {effective_batch_size}."
            )
            raise RuntimeError(msg)
        return expert_batch.to(self.device)

    def _apply_estimated_reward_done_penalty(
        self,
        rollout: TensorDict,
        est_reward: Tensor,
    ) -> Tensor:
        """Subtract the configured penalty on terminal non-truncated transitions."""
        penalty = float(self.config.ipmd.estimated_reward_done_penalty)
        if penalty == 0.0:
            return est_reward
        done_key: BatchKey = ("next", "done")
        if done_key not in rollout.keys(True):
            return est_reward
        done = rollout[done_key]
        truncated_key: BatchKey = ("next", "truncated")
        truncated = (
            rollout[truncated_key] if truncated_key in rollout.keys(True) else None
        )
        done_mask = done.to(device=est_reward.device, dtype=torch.bool)
        if truncated is None:
            truncated_mask = torch.zeros_like(done_mask)
        else:
            truncated_mask = truncated.to(device=est_reward.device, dtype=torch.bool)
        terminal_mask = done_mask & ~truncated_mask
        while terminal_mask.ndim < est_reward.ndim:
            terminal_mask = terminal_mask.unsqueeze(-1)
        return torch.where(terminal_mask, est_reward - penalty, est_reward)

    def _reward_input_from_td(
        self,
        td: TensorDict,
        *,
        batch_role: str,
        detach: bool,
        requires_grad: bool,
    ) -> Tensor:
        """Materialize the canonical reward-model input tensor for one batch."""
        parts: list[Tensor] = []
        for block in self._reward_input_blocks:
            if block.kind == "obs":
                parts.append(
                    flatten_obs_features_from_td(
                        td,
                        list(block.obs_keys),
                        self._obs_feature_ndims,
                        next_obs=False,
                        detach=detach,
                    )
                )
            elif block.kind == "next_obs":
                parts.append(
                    flatten_obs_features_from_td(
                        td,
                        list(block.obs_keys),
                        self._obs_feature_ndims,
                        next_obs=True,
                        detach=detach,
                    )
                )
            elif block.kind == "action":
                parts.append(
                    flatten_action_features_from_td(
                        td,
                        self._action_feature_ndim,
                        detach=detach,
                    )
                )
            elif block.kind == "latent":
                if batch_role == "rollout":
                    controller = self._require_latent_command_controller()
                    rollout_latent = controller.latent_condition_from_td(
                        td, detach=True
                    )
                    assert rollout_latent is not None
                    parts.append(rollout_latent.to(self.device))
                else:
                    parts.append(self._expert_latents_from_td(td, detach=True))
            else:
                msg = f"Unhandled reward input block kind {block.kind!r}."
                raise RuntimeError(msg)

        if not parts:
            msg = "Reward input specification must contain at least one block."
            raise RuntimeError(msg)
        reward_input = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
        if requires_grad:
            reward_input = reward_input.detach().requires_grad_(True)
        return reward_input

    def _reward_from_td(
        self,
        td: TensorDict,
        *,
        batch_role: str,
        detach: bool | None = None,
        requires_grad: bool = False,
        return_input: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Compute estimated reward for a batch of transitions."""
        if detach is None:
            detach = self._reward_detach_features
        x = self._reward_input_from_td(
            td,
            batch_role=batch_role,
            detach=detach,
            requires_grad=requires_grad,
        )
        reward = self._reward_out_fn(self.reward_estimator(x))
        if return_input:
            return reward, x
        return reward

    def _reward_from_batch(
        self,
        td: TensorDict,
        *,
        batch_role: str,
        detach: bool | None = None,
        requires_grad: bool = False,
        return_input: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Backward-compatible alias for reward evaluation on one batch."""
        return self._reward_from_td(
            td,
            batch_role=batch_role,
            detach=detach,
            requires_grad=requires_grad,
            return_input=return_input,
        )

    def _latent_uniformity(self, latent_pred: Tensor) -> Tensor:
        """Not used. Placeholder."""
        return torch.zeros((), device=latent_pred.device, dtype=latent_pred.dtype)

    def _extra_actor_loss(self, batch: TensorDict) -> tuple[Tensor, dict[str, Tensor]]:
        del batch
        return torch.zeros((), device=self.device), {}

    def _refresh_grad_clip_params(self) -> None:
        """Refresh the cached optimizer parameter list used for grad clipping."""
        self._grad_clip_params = [  # type: ignore[attr-defined]
            param for group in self.optim.param_groups for param in group["params"]
        ]

    @property
    def _required_loss_metrics(self) -> list[str]:
        return [
            *super()._required_loss_metrics,
            "loss_reward_diff",
            "loss_reward_l2",
            "loss_reward_grad_penalty",
            "loss_reward_grad_penalty_batch",
            "loss_reward_grad_penalty_expert",
        ]

    @property
    def _optional_loss_metrics(self) -> list[str]:
        return [
            *super()._optional_loss_metrics,
            "loss_diversity",
            "loss_bc",
            "bc_nll",
            "bc_has_expert",
            "bc_log_prob_mean",
            "bc_log_prob_nan_frac",
            "bc_expert_action_abs_mean",
            "bc_expert_action_zero_frac",
            "bc_expert_action_nan_frac",
            "bc_policy_action_abs_mean",
            "bc_policy_action_mae",
            "bc_policy_action_rmse",
            "bc_actor_grad_norm",
            "bc_policy_scale_mean",
            "estimated_reward_mean",
            "estimated_reward_std",
            "expert_reward_mean",
            "expert_reward_std",
        ]

    def _select_reported_loss_metrics(self, loss: TensorDict) -> TensorDict:
        """Filter IPMD update outputs down to the metrics recorded per minibatch."""
        loss_keys = [
            key
            for key in [*self._required_loss_metrics, *self._optional_loss_metrics]
            if key in loss
        ]
        return loss.select(*loss_keys)

    def _prepare_rollout_rewards(self, rollout: TensorDict) -> dict[str, float]:
        """Attach reward-model diagnostics and PPO reward mixing to one rollout."""
        metrics: dict[str, float] = {}
        reward_key = ("next", "reward")

        with torch.no_grad():
            env_reward = rollout[reward_key]
            est_reward: Tensor | None = None
            if self.config.ipmd.use_estimated_rewards_for_ppo:
                est_reward = (
                    self._reward_from_td(
                        rollout,
                        batch_role="rollout",
                    )
                    .detach()  # type: ignore[attr-defined]
                    .clamp(
                        min=self.config.ipmd.estimated_reward_clamp_min,
                        max=self.config.ipmd.estimated_reward_clamp_max,
                    )
                )  # type: ignore[attr-defined]
                est_reward = self._apply_estimated_reward_done_penalty(
                    rollout,
                    est_reward,
                )

            if self._use_estimated_rewards_for_ppo:
                assert est_reward is not None
                mixed_reward = (
                    self.config.ipmd.env_reward_weight * env_reward
                    + self.config.ipmd.est_reward_weight * est_reward
                )
            else:
                mixed_reward = env_reward

        rollout.set(("next", "env_reward"), env_reward)
        rollout.set(reward_key, mixed_reward)

        metrics.update(
            {
                "train/env_reward_mean": env_reward.mean().item(),
            }
        )
        if est_reward is not None:
            rollout.set(("next", "est_reward"), est_reward)
            metrics["train/est_reward_mean"] = est_reward.mean().item()
            metrics.update(
                reward_alignment_metrics(
                    "reward/env_vs_est",
                    env_reward,
                    est_reward,
                )
            )

        return metrics

    def _expert_batch_for_update(self, batch: TensorDict) -> tuple[TensorDict, Tensor]:
        """Return an expert batch aligned with one PPO minibatch update."""
        del batch
        expert_batch = self._next_expert_batch()
        return (
            expert_batch,
            torch.ones((), device=self.device, dtype=torch.float32),
        )

    def _backward_ppo_terms(self, batch: TensorDict) -> TensorDict:
        """Backpropagate the PPO objective and return detached update metrics."""
        loss: TensorDict = self.loss_module(batch)
        extra_actor_loss, extra_actor_metrics = self._extra_actor_loss(batch)
        (
            loss["loss_critic"]
            + loss["loss_objective"]
            + loss["loss_entropy"]
            + extra_actor_loss
        ).backward()
        output_loss = loss.clone().detach_()
        for key, value in extra_actor_metrics.items():
            output_loss.set(key, value.detach())
        return output_loss

    def _backward_bc_terms(
        self,
        expert_batch: TensorDict,
        has_expert: Tensor,
    ) -> dict[str, Tensor]:
        """Backpropagate behavior-cloning terms when enabled."""
        metrics: dict[str, Tensor] = {}
        if self._bc_coeff <= 0.0:
            return metrics

        expert_action = expert_batch["expert_action"]
        if not isinstance(expert_action, Tensor):
            msg = "BC update requires expert_batch['expert_action'] to be a Tensor."
            raise RuntimeError(msg)
        expert_obs_td = expert_batch.clone(False)
        if self._latent_key not in expert_obs_td.keys(True):
            expert_latents = self._expert_latents_from_td(
                expert_batch,
                detach=True,
            ).reshape(*expert_batch.batch_size, self._latent_dim)
            expert_obs_td.set(self._latent_key, expert_latents)
        expert_obs_td = expert_obs_td.select(*self._policy_obs_keys)
        dist = self._policy_operator.get_dist(expert_obs_td)
        log_prob = dist.log_prob(expert_action)
        log_prob = self._reduce_log_prob(log_prob, expert_action)
        has_expert_float = has_expert.to(dtype=log_prob.dtype)
        bc_nll = -log_prob.mean() * has_expert_float
        bc_loss = bc_nll * self._bc_coeff
        bc_loss.backward()
        metrics["loss_bc"] = bc_loss.detach()
        metrics["bc_nll"] = bc_nll.detach()
        return metrics

    def _backward_reward_terms(
        self,
        batch: TensorDict,
        expert_batch: TensorDict,
    ) -> dict[str, Tensor]:
        """Backpropagate the IPMD reward objective and return detached metrics."""
        metrics = {
            "loss_reward_diff": torch.zeros((), device=self.device),
            "loss_reward_l2": torch.zeros((), device=self.device),
            "loss_reward_grad_penalty": torch.zeros((), device=self.device),
            "loss_reward_grad_penalty_batch": torch.zeros((), device=self.device),
            "loss_reward_grad_penalty_expert": torch.zeros((), device=self.device),
        }
        if not self._reward_update_enabled:
            return metrics

        if self._reward_grad_penalty_coeff > 0.0:
            r_pi_with_input = self._reward_from_td(
                batch,
                batch_role="rollout",
                requires_grad=True,
                return_input=True,
            )
            r_exp_with_input = self._reward_from_td(
                expert_batch,
                batch_role="expert",
                requires_grad=True,
                return_input=True,
            )
            assert isinstance(r_pi_with_input, tuple)
            assert isinstance(r_exp_with_input, tuple)
            r_pi, r_pi_input = r_pi_with_input
            r_exp, r_exp_input = r_exp_with_input
            reward_grad_penalty_batch = reward_grad_penalty_from_input(r_pi, r_pi_input)
            reward_grad_penalty_expert = reward_grad_penalty_from_input(
                r_exp, r_exp_input
            )
        else:
            r_pi = self._reward_from_td(batch, batch_role="rollout")  # type: ignore[attr-defined]
            r_exp = self._reward_from_td(expert_batch, batch_role="expert")  # type: ignore[attr-defined]
            reward_grad_penalty_batch = torch.zeros((), device=self.device)
            reward_grad_penalty_expert = torch.zeros((), device=self.device)

        diff = r_pi.mean() - r_exp.mean()
        l2 = r_pi.pow(2).mean() + r_exp.pow(2).mean()
        reward_grad_penalty = reward_grad_penalty_batch + reward_grad_penalty_expert
        (
            self._reward_loss_coeff * diff
            + self._reward_l2_coeff * l2.pow(0.5)
            + self._reward_grad_penalty_coeff * reward_grad_penalty
        ).backward()

        metrics["loss_reward_diff"] = diff.detach()
        metrics["loss_reward_l2"] = l2.detach()
        metrics["loss_reward_grad_penalty"] = reward_grad_penalty.detach()
        metrics["loss_reward_grad_penalty_batch"] = reward_grad_penalty_batch.detach()
        metrics["loss_reward_grad_penalty_expert"] = reward_grad_penalty_expert.detach()
        return metrics

    def _record_env_metrics(self, iteration: PPOIterationData) -> None:
        """Record IsaacLab env metrics while draining the env log queue."""
        if (
            "Isaac" in self.config.env.env_name
            and hasattr(self.env, "log_infos")
            and len(self.env.log_infos) > 0
        ):
            log_info_dict: dict[str, Tensor] = self.env.log_infos.pop()
            self.env.log_infos.clear()
            log_info(log_info_dict, iteration.metrics)

    def _progress_summary_fields(self) -> tuple[tuple[str, str], ...]:
        return (
            ("train/step_reward_mean", "r_step"),
            ("episode/length", "ep_len"),
            ("episode/return", "r_ep"),
            ("train/loss_objective", "pi_loss"),
            ("train/loss_reward_diff", "reward_diff"),
            ("train/expert_reward_mean", "exp_r"),
            ("time/speed", "fps"),
        )

    def record(
        self,
        iteration: PPOIterationData,
        metadata: PPOTrainingMetadata,
    ) -> None:
        """Flush IPMD diagnostics, reward metrics, and checkpoints for one rollout."""
        rollout = iteration.rollout

        if "train/step_reward_mean" not in iteration.metrics and (
            "next",
            "reward",
        ) in rollout.keys(True):
            step_rewards = rollout["next", "reward"]
            iteration.metrics.update(
                {
                    "train/step_reward_mean": step_rewards.mean().item(),
                    "train/step_reward_std": step_rewards.std().item(),
                    "train/step_reward_max": step_rewards.max().item(),
                    "train/step_reward_min": step_rewards.min().item(),
                }
            )

        episode_rewards = rollout["next", "episode_reward"][rollout["next", "done"]]
        if len(episode_rewards) > 0:
            episode_length = rollout["next", "step_count"][rollout["next", "done"]]
            episode_lengths = episode_length.cpu().tolist()
            episode_reward_values = episode_rewards.cpu().tolist()
            self.episode_lengths.extend(episode_lengths)
            self.episode_rewards.extend(episode_reward_values)
            iteration.metrics.update(
                {
                    "episode/length": float(np.mean(self.episode_lengths)),
                    "episode/return": float(np.mean(self.episode_rewards)),
                    "train/reward": float(np.mean(episode_reward_values)),
                }
            )

        iteration.metrics.update(self._build_control_metrics(metadata))
        iteration.metrics.update(self._build_timing_metrics(iteration, metadata))
        self._record_env_metrics(iteration)
        iteration.metrics.update(timeit.todict(prefix="time"))  # type: ignore[arg-type]
        # Always stream scalar metrics to the configured backend (e.g. WandB).
        # Sparse terminal/file summaries remain controlled by
        # ``trainer.log_interval`` via ``_refresh_progress_display`` below.
        self.log_metrics(
            iteration.metrics,
            step=metadata.frames_processed,
            log_python=False,
        )
        self.collector.update_policy_weights_()
        self._refresh_progress_display(metadata, iteration)

        if (
            self.config.save_interval > 0
            and metadata.frames_processed > 0
            and metadata.frames_processed % self.config.save_interval == 0
        ):
            self.save_model(
                path=self.log_dir / self.config.logger.save_path,
                step=metadata.frames_processed,
            )

    def predict(self, obs: Tensor | np.ndarray | Mapping[Any, Any]) -> Tensor:  # type: ignore[override]
        """Predict action given observation (deterministic)."""
        policy_op = self.actor_critic.get_policy_operator()
        policy_op.eval()
        with torch.no_grad(), set_exploration_type(InteractionType.DETERMINISTIC):
            input_keys = list(self._policy_obs_keys)
            td_data: dict[BatchKey, Tensor] = {}
            batch_shape: tuple[int, ...] | None = None

            if isinstance(obs, Mapping):
                for key in input_keys:
                    if key == self._latent_key:
                        try:
                            latent_value = mapping_get_obs_value(obs, key)
                        except Exception:
                            continue
                        value = torch.as_tensor(latent_value, device=self.device)
                    else:
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
            if self._use_latent_command:
                self._inject_latent_command(td)
            td = policy_op(td)
            return td["action"]

    def _latent_mode_hint(self) -> str:
        task_name = str(getattr(self.config.env, "env_name", "") or "")
        if task_name in {"Isaac-Imitation-G1-v0", "Isaac-Imitation-G1-LafanTrack-v0"}:
            return (
                " For the vanilla G1 task, pass ipmd.use_latent_command=False, "
                "or switch to Isaac-Imitation-G1-Latent-v0."
            )
        if task_name == "Isaac-Imitation-G1-Latent-v0":
            return (
                " For the latent G1 task, keep ipmd.use_latent_command=True, "
                "or switch to Isaac-Imitation-G1-v0."
            )
        return ""

    def pre_iteration_compute(self, rollout: TensorDict) -> TensorDict:
        """Compute advantages."""
        with torch.no_grad():
            rollout = self.adv_module(rollout)

            if getattr(self.config.compile, "compile", False):
                rollout = rollout.clone()

        self.data_buffer.extend(rollout.reshape(-1))
        return rollout

    def update(
        self,
        batch: TensorDict,
        num_network_updates: int,
        expert_batch: TensorDict,
        has_expert: Tensor,
    ) -> tuple[TensorDict, int]:
        """PPO update plus optional BC loss and IPMD reward loss."""
        self.optim.zero_grad(set_to_none=True)
        output_loss = self._backward_ppo_terms(batch)
        bc_metrics = self._backward_bc_terms(expert_batch, has_expert)
        reward_metrics = self._backward_reward_terms(batch, expert_batch)

        # Gradient clipping — always call for a fixed graph
        grad_norm_tensor = clip_grad_norm_(self._grad_clip_params, self._max_grad_norm)

        self.optim.step()

        output_loss.set("alpha", torch.ones((), device=self.device))
        for key, value in reward_metrics.items():
            output_loss.set(key, value)
        for key, value in bc_metrics.items():
            output_loss.set(key, value)
        output_loss.set("grad_norm", grad_norm_tensor.detach())

        return output_loss, num_network_updates + 1

    def prepare(
        self,
        iteration: PPOIterationData,
        metadata: PPOTrainingMetadata,  # noqa: ARG002
    ) -> None:
        """Attach reward-model diagnostics and replace the PPO reward for this rollout."""
        iteration.metrics.update(self._prepare_rollout_rewards(iteration.rollout))

    def iterate(
        self,
        iteration: PPOIterationData,
        metadata: PPOTrainingMetadata,
    ) -> None:
        """Run PPO-style epochs over the prepared rollout with IPMD expert updates."""
        losses = TensorDict(
            batch_size=[metadata.epochs_per_rollout, metadata.minibatches_per_epoch]
        )
        learn_start = time.perf_counter()

        self.data_buffer.empty()
        self.actor_critic.train()
        self.adv_module.train()
        self.reward_estimator.train()

        with timeit("training"):
            rollout_flat = iteration.rollout.reshape(-1)
            if self._latent_learner is not None:
                learner_metrics = self._latent_learner.update(rollout_flat)
                if learner_metrics:
                    iteration.metrics.update(
                        {f"train/{k}": v for k, v in learner_metrics.items()}
                    )

            iteration.rollout = self.pre_iteration_compute(iteration.rollout)

            for epoch_idx in range(metadata.epochs_per_rollout):
                for batch_idx, batch in enumerate(self.data_buffer):
                    kl_context = None
                    if (self.config.optim.scheduler or "").lower() == "adaptive":
                        kl_context = self._prepare_kl_context(
                            batch, metadata.policy_operator
                        )

                    if self._expert_minibatch_update_enabled:
                        expert_batch, has_expert = self._expert_batch_for_update(batch)
                    else:
                        expert_batch = TensorDict(
                            {},
                            batch_size=batch.batch_size,
                            device=self.device,
                        )
                        has_expert = torch.zeros(
                            (),
                            device=self.device,
                            dtype=torch.float32,
                        )
                    with timeit("training/update"):
                        loss, metadata.updates_completed = self.update(
                            batch,
                            metadata.updates_completed,
                            expert_batch,
                            has_expert,
                        )

                    if self.lr_scheduler and self.lr_scheduler_step == "update":
                        self.lr_scheduler.step()
                    if kl_context is not None:
                        kl_approx = self._compute_kl_after_update(
                            kl_context, metadata.policy_operator
                        )
                        if kl_approx is not None:
                            loss.set("kl_approx", kl_approx.detach())
                            self._maybe_adjust_lr(kl_approx, self.config.optim)

                    losses[epoch_idx, batch_idx] = self._select_reported_loss_metrics(
                        loss
                    )

                if self.lr_scheduler and self.lr_scheduler_step == "epoch":
                    self.lr_scheduler.step()

        iteration.learn_time = time.perf_counter() - learn_start
        losses_mean = losses.apply(lambda x: x.float().mean(), batch_size=[])
        for key, value in losses_mean.items():  # type: ignore[attr-defined]
            iteration.metrics[f"train/{key}"] = value.item()  # type: ignore[attr-defined]
