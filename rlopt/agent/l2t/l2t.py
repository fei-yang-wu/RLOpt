from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import torch
import torch.nn
import torch.optim
import tqdm
from tensordict import TensorDict, TensorDictParams
from tensordict.nn import (
    TensorDictModule,
)
from torch import Tensor
from torchrl._utils import timeit

# Import missing modules
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyMemmapStorage, ReplayBuffer, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.envs import Compose, ExplorationType, TransformedEnv
from torchrl.envs.transforms import InitTracker
from torchrl.modules import (
    ActorValueOperator,
    LSTMModule,
    MLP,
    NormalParamExtractor,
    ProbabilisticActor,
    TanhNormal,
    ValueOperator,
)
from torchrl.modules.tensordict_module.sequence import SafeSequential
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value.advantages import GAE
from torchrl.record.loggers import Logger

from rlopt.base_class import BaseAlgorithm
from rlopt.configs import (
    NetworkLayout,
    RLOptConfig,
)
from rlopt.type_aliases import OptimizerClass
from rlopt.utils import get_activation_class


@dataclass
class L2TConfig:
    """L2T-specific configuration.

    - Uses ``RLOptConfig.network`` for the teacher layout (PPO-style).
    - Provides a separate ``student`` layout and imitation loss options.
    """

    student: NetworkLayout = field(default_factory=NetworkLayout)
    """Student network layout (separate from teacher, which uses RLOptConfig.network)"""

    mixture_coeff: float = 0.2
    """Mixture coefficient for potential teacher-student action mixing (not required for training)"""

    imitation_type: Literal["l2", "asymmetric", "bc"] = "l2"
    """Imitation loss settings"""

    imitation_coeff: float = 1.0
    """Coefficient for imitation loss"""

    # PPO-style hyperparameters for teacher training
    gae_lambda: float = 0.95
    """Generalized Advantage Estimation (GAE) lambda"""

    clip_epsilon: float = 0.2
    """Clipping epsilon for PPO"""

    clip_value: bool = True
    """Whether to clip the value function"""

    anneal_clip_epsilon: bool = False
    """Whether to anneal the clipping epsilon"""

    critic_coeff: float = 1.0
    """Coefficient for critic loss"""

    entropy_coeff: float = 0.005
    """Coefficient for entropy loss"""

    normalize_advantage: bool = False
    """Whether to normalize the advantage estimates"""

    student_hidden_key: str = "student_hidden"
    """TensorDict keying for student branch"""

    student_loc_key: str = "student_loc"
    """TensorDict keying for student location"""

    student_scale_key: str = "student_scale"
    """TensorDict keying for student scale"""


@dataclass
class L2TRLOptConfig(RLOptConfig):
    """L2T-specific configuration."""

    l2t: L2TConfig = field(default_factory=L2TConfig)
    """L2T-specific configuration."""


class L2TActorValueOperatorWrapper(SafeSequential):
    def __init__(
        self,
        teacher_operator: ActorValueOperator,
        student_operator: ActorValueOperator,
        rng: torch.Generator,
        mixture_coeff: float = 0.2,
    ):
        super().__init__(teacher_operator, student_operator)

        self.mixture_coeff = mixture_coeff
        # optional external RNG (caller promised it's ready when passed)
        self._rng: torch.Generator = rng

        assert isinstance(self.module[0], TensorDictModule)
        assert isinstance(self.module[1], TensorDictModule)

    def get_teacher_operator(self):
        """Get the teacher operator of the actor-value operator."""
        return self.module[0]

    def get_student_operator(self):
        """Get the student operator of the actor-value operator."""
        return self.module[1]

    def forward(self, tensordict: TensorDict) -> TensorDict:  # type: ignore[override]
        """Route the input to teacher or student operator.

        TorchRL collectors call policies with a positional TensorDict. Accept
        that here and forward to the selected operator.
        """
        # Draw once per call to choose teacher vs student
        # Use underlying module device as source of truth
        module_device = getattr(self.get_teacher_operator(), "device", None)
        device = module_device if module_device is not None else torch.device("cpu")
        sample = torch.rand((), generator=self._rng, device=device)

        if sample.item() < float(self.mixture_coeff):
            return self.get_teacher_operator()(tensordict)
        return self.get_student_operator()(tensordict)


class L2T(BaseAlgorithm):
    def __init__(
        self,
        env: TransformedEnv,
        config: L2TRLOptConfig,
        policy_net: torch.nn.Module | None = None,
        value_net: torch.nn.Module | None = None,
        q_net: torch.nn.Module | None = None,
        replay_buffer: type[ReplayBuffer] = ReplayBuffer,
        logger: Logger | None = None,
        **kwargs,
    ):
        # Narrow the type for static checkers early
        self.config = config  # type: ignore
        self.config: L2TRLOptConfig

        self.env = env

        # construct the student actor-critic (separate layout and keys)
        self.student_feature_extractor = self._construct_student_feature_extractor()
        self.student_policy = self._construct_student_policy()
        self.student_value_function = self._construct_student_value_function()
        self.student_actor_critic = self._construct_student_actor_critic()

        super().__init__(
            env,
            config,
            policy_net,
            value_net,
            q_net,
            replay_buffer,
            logger,
            **kwargs,
        )

        # construct the advantage module
        self.adv_module = self._construct_adv_module()

        # Compile if requested
        self._compile_components()

        # Initialize total network updates
        self.total_network_updates = 0

    @property
    def collector_policy(self) -> TensorDictModule:
        """By default, the collector_policy is self.policy or self.actor_critic.policy_operator()"""
        assert isinstance(self.config, L2TRLOptConfig)
        assert isinstance(self.actor_critic, ActorValueOperator), (
            "Actor critic is not an instance of ActorValueOperator"
        )
        assert isinstance(self.student_actor_critic, ActorValueOperator), (
            "Student actor critic is not an instance of ActorValueOperator"
        )

        return L2TActorValueOperatorWrapper(
            teacher_operator=self.actor_critic,
            student_operator=self.student_actor_critic,
            rng=self.th_rng,
            mixture_coeff=self.config.l2t.mixture_coeff,
        )

    def _construct_feature_extractor(
        self, feature_extractor_net: torch.nn.Module | None = None
    ) -> TensorDictModule:
        """Teacher feature extractor (MLP by default)."""
        if feature_extractor_net is not None:
            return TensorDictModule(
                module=feature_extractor_net,
                in_keys=list(self.total_input_keys),
                out_keys=["hidden"],
            )

        if not self.config.use_feature_extractor:
            # Identity mapping if no feature extractor
            return TensorDictModule(
                module=torch.nn.Identity(),
                in_keys=list(self.total_input_keys),
                out_keys=["hidden"],
            )

        # Build MLP feature extractor for teacher
        # Use config.network layout if specified
        hidden_size = 256  # default
        num_cells = [256, 256]  # default
        activation = "relu"  # default

        if (
            self.config.network
            and self.config.network.shared
            and self.config.network.shared.features
        ):
            # Get the first feature block or one referenced by policy/value
            feature_name = None
            if self.config.network.policy and self.config.network.policy.feature_ref:
                feature_name = self.config.network.policy.feature_ref
            elif self.config.network.value and self.config.network.value.feature_ref:
                feature_name = self.config.network.value.feature_ref
            else:
                # Use first available feature
                feature_name = next(
                    iter(self.config.network.shared.features.keys()), None
                )

            if feature_name and feature_name in self.config.network.shared.features:
                feature_spec = self.config.network.shared.features[feature_name]
                hidden_size = feature_spec.output_dim
                if feature_spec.mlp:
                    num_cells = (
                        list(feature_spec.mlp.num_cells)
                        if feature_spec.mlp.num_cells
                        else [256, 256]
                    )
                    activation = feature_spec.mlp.activation or "relu"

        # Build MLP
        feature_mlp = MLP(
            in_features=self.total_input_shape,
            out_features=hidden_size,
            num_cells=num_cells,
            activation_class=get_activation_class(activation),
            device=self.device,
        )

        return TensorDictModule(
            module=feature_mlp,
            in_keys=list(self.total_input_keys),
            out_keys=["hidden"],
        )

    def _construct_policy(
        self, policy_net: torch.nn.Module | None = None
    ) -> TensorDictModule:
        """Teacher policy (PPO-style), honoring advanced network layout if provided."""
        distribution_class = TanhNormal
        distribution_kwargs = {
            "low": self.env.action_spec_unbatched.space.low.to(self.device),  # type: ignore
            "high": self.env.action_spec_unbatched.space.high.to(self.device),  # type: ignore
            "tanh_loc": False,
        }

        # Determine input dimension
        if self.config.use_feature_extractor:
            input_dim = 256  # default from feature extractor
            if (
                self.config.network
                and self.config.network.shared
                and self.config.network.shared.features
            ):
                feature_name = None
                if (
                    self.config.network.policy
                    and self.config.network.policy.feature_ref
                ):
                    feature_name = self.config.network.policy.feature_ref
                elif (
                    self.config.network.value and self.config.network.value.feature_ref
                ):
                    feature_name = self.config.network.value.feature_ref
                else:
                    feature_name = next(
                        iter(self.config.network.shared.features.keys()), None
                    )

                if feature_name and feature_name in self.config.network.shared.features:
                    input_dim = self.config.network.shared.features[
                        feature_name
                    ].output_dim
        else:
            input_dim = self.total_input_shape

        # Determine policy head configuration
        num_cells = [256, 256]  # default
        activation = "relu"  # default

        if (
            self.config.network
            and self.config.network.policy
            and self.config.network.policy.head
        ):
            head_cfg = self.config.network.policy.head
            if head_cfg.num_cells:
                num_cells = list(head_cfg.num_cells)
            if head_cfg.activation:
                activation = head_cfg.activation

        # Build policy network
        if policy_net is None:
            policy_mlp = MLP(
                in_features=input_dim,
                out_features=2 * self.env.action_spec_unbatched.shape[-1],  # type: ignore
                num_cells=num_cells,
                activation_class=get_activation_class(activation),
                device=self.device,
            )
        else:
            policy_mlp = policy_net

        # Add parameter extractor
        extractor = NormalParamExtractor(
            scale_mapping="biased_softplus_1.0",
            scale_lb=0.1,  # type: ignore
        ).to(self.device)
        net = torch.nn.Sequential(policy_mlp, extractor)

        # Wrap in TensorDictModule
        policy_td = TensorDictModule(
            module=net,
            in_keys=list(self.config.policy_in_keys),
            out_keys=["loc", "scale"],
        )

        return ProbabilisticActor(
            policy_td,
            in_keys=["loc", "scale"],
            spec=self.env.full_action_spec_unbatched.to(self.device),  # type: ignore
            distribution_class=distribution_class,
            distribution_kwargs=distribution_kwargs,
            return_log_prob=True,
            default_interaction_type=ExplorationType.DETERMINISTIC,
        )

    def _construct_value_function(
        self, value_net: torch.nn.Module | None = None
    ) -> TensorDictModule:
        """Teacher value function (PPO-style), honoring advanced layout if provided."""
        # Determine input dimension
        if self.config.use_feature_extractor:
            input_dim = 256  # default from feature extractor
            if (
                self.config.network
                and self.config.network.shared
                and self.config.network.shared.features
            ):
                feature_name = None
                if self.config.network.value and self.config.network.value.feature_ref:
                    feature_name = self.config.network.value.feature_ref
                elif (
                    self.config.network.policy
                    and self.config.network.policy.feature_ref
                ):
                    feature_name = self.config.network.policy.feature_ref
                else:
                    feature_name = next(
                        iter(self.config.network.shared.features.keys()), None
                    )

                if feature_name and feature_name in self.config.network.shared.features:
                    input_dim = self.config.network.shared.features[
                        feature_name
                    ].output_dim
        else:
            input_dim = self.total_input_shape

        # Determine value head configuration
        num_cells = [256, 256]  # default
        activation = "relu"  # default

        if (
            self.config.network
            and self.config.network.value
            and self.config.network.value.head
        ):
            head_cfg = self.config.network.value.head
            if head_cfg.num_cells:
                num_cells = list(head_cfg.num_cells)
            if head_cfg.activation:
                activation = head_cfg.activation

        # Build value network
        if value_net is None:
            value_mlp = MLP(
                in_features=input_dim,
                out_features=1,
                num_cells=num_cells,
                activation_class=get_activation_class(activation),
                device=self.device,
            )
        else:
            value_mlp = value_net

        return ValueOperator(
            module=value_mlp,
            in_keys=list(self.config.value_net_in_keys),
        )

    def _construct_actor_critic(self) -> TensorDictModule:
        """Construct actor-critic network"""
        return ActorValueOperator(
            common_operator=self.feature_extractor,
            policy_operator=self.policy,  # type: ignore[arg-type]
            value_operator=self.value_function,  # type: ignore[arg-type]
        )

    def _construct_student_actor_critic(self) -> TensorDictModule:
        """Construct student actor-critic, which contains a feature extractor, a policy head, and a value function."""
        return ActorValueOperator(
            common_operator=self.student_feature_extractor,
            policy_operator=self.student_policy,
            value_operator=self.student_value_function,
        )

    # ------------------------------
    # Student builders (MLP variant)
    # ------------------------------
    def _construct_student_feature_extractor(self) -> TensorDictModule:
        """Build student feature extractor network (MLP by default)."""
        assert isinstance(self.config, L2TRLOptConfig)

        # For L2T base class, student uses MLP feature extractor
        # Check if we should use a feature extractor
        if not self.config.use_feature_extractor:
            # Identity mapping if no feature extractor
            return TensorDictModule(
                module=torch.nn.Identity(),
                in_keys=list(self.total_input_keys),
                out_keys=[self.config.l2t.student_hidden_key],
            )

        # Build MLP feature extractor for student
        # Use student layout if specified, otherwise use default from config
        student_layout = self.config.l2t.student

        # Determine configuration for student feature extractor
        # Try to get from student's shared features first
        hidden_size = 256  # default
        num_cells = [256, 256]  # default
        activation = "relu"  # default

        if student_layout and student_layout.shared and student_layout.shared.features:
            # Get the first feature block or one referenced by policy/value
            feature_name = None
            if student_layout.policy and student_layout.policy.feature_ref:
                feature_name = student_layout.policy.feature_ref
            elif student_layout.value and student_layout.value.feature_ref:
                feature_name = student_layout.value.feature_ref
            else:
                # Use first available feature
                feature_name = next(iter(student_layout.shared.features.keys()), None)

            if feature_name and feature_name in student_layout.shared.features:
                feature_spec = student_layout.shared.features[feature_name]
                hidden_size = feature_spec.output_dim
                if feature_spec.mlp:
                    num_cells = (
                        list(feature_spec.mlp.num_cells)
                        if feature_spec.mlp.num_cells
                        else [256, 256]
                    )
                    activation = feature_spec.mlp.activation or "relu"

        # Build MLP
        feature_mlp = MLP(
            in_features=self.total_input_shape,
            out_features=hidden_size,
            num_cells=num_cells,
            activation_class=get_activation_class(activation),
            device=self.device,
        )

        return TensorDictModule(
            module=feature_mlp,
            in_keys=list(self.total_input_keys),
            out_keys=[self.config.l2t.student_hidden_key],
        )

    def _construct_student_policy(self) -> TensorDictModule:
        """Build student policy head (outputs loc and scale for TanhNormal distribution)."""
        assert isinstance(self.config, L2TRLOptConfig)

        # Determine input dimension (from feature extractor or observation)
        if self.config.use_feature_extractor:
            # Get from student feature extractor output
            student_layout = self.config.l2t.student
            input_dim = 256  # default

            if (
                student_layout
                and student_layout.shared
                and student_layout.shared.features
            ):
                feature_name = None
                if student_layout.policy and student_layout.policy.feature_ref:
                    feature_name = student_layout.policy.feature_ref
                elif student_layout.value and student_layout.value.feature_ref:
                    feature_name = student_layout.value.feature_ref
                else:
                    feature_name = next(
                        iter(student_layout.shared.features.keys()), None
                    )

                if feature_name and feature_name in student_layout.shared.features:
                    input_dim = student_layout.shared.features[feature_name].output_dim
        else:
            input_dim = self.total_input_shape

        # Determine policy head configuration
        num_cells = [256, 256]  # default
        activation = "relu"  # default

        student_layout = self.config.l2t.student
        if student_layout and student_layout.policy and student_layout.policy.head:
            head_cfg = student_layout.policy.head
            if head_cfg.num_cells:
                num_cells = list(head_cfg.num_cells)
            if head_cfg.activation:
                activation = head_cfg.activation

        # Build policy MLP
        policy_mlp = MLP(
            in_features=input_dim,
            out_features=2 * self.env.action_spec_unbatched.shape[-1],  # type: ignore
            num_cells=num_cells,
            activation_class=get_activation_class(activation),
            device=self.device,
        )

        # Add parameter extractor for normal distribution
        extractor = NormalParamExtractor(
            scale_mapping="biased_softplus_1.0",
            scale_lb=0.1,  # type: ignore
        ).to(self.device)

        net = torch.nn.Sequential(policy_mlp, extractor)

        # Wrap in TensorDictModule (NOT ProbabilisticActor, just the parameter generation)
        return TensorDictModule(
            module=net,
            in_keys=[self.config.l2t.student_hidden_key],
            out_keys=[
                self.config.l2t.student_loc_key,
                self.config.l2t.student_scale_key,
            ],
        )

    def _construct_student_value_function(self) -> TensorDictModule:
        """Build student value function head (outputs state value estimate)."""
        assert isinstance(self.config, L2TRLOptConfig)

        # Determine input dimension (same logic as policy)
        if self.config.use_feature_extractor:
            student_layout = self.config.l2t.student
            input_dim = 256  # default

            if (
                student_layout
                and student_layout.shared
                and student_layout.shared.features
            ):
                feature_name = None
                if student_layout.value and student_layout.value.feature_ref:
                    feature_name = student_layout.value.feature_ref
                elif student_layout.policy and student_layout.policy.feature_ref:
                    feature_name = student_layout.policy.feature_ref
                else:
                    feature_name = next(
                        iter(student_layout.shared.features.keys()), None
                    )

                if feature_name and feature_name in student_layout.shared.features:
                    input_dim = student_layout.shared.features[feature_name].output_dim
        else:
            input_dim = self.total_input_shape

        # Determine value head configuration
        num_cells = [256, 256]  # default
        activation = "relu"  # default

        student_layout = self.config.l2t.student
        if student_layout and student_layout.value and student_layout.value.head:
            head_cfg = student_layout.value.head
            if head_cfg.num_cells:
                num_cells = list(head_cfg.num_cells)
            if head_cfg.activation:
                activation = head_cfg.activation

        # Build value MLP
        value_mlp = MLP(
            in_features=input_dim,
            out_features=1,
            num_cells=num_cells,
            activation_class=get_activation_class(activation),
            device=self.device,
        )

        # Wrap in ValueOperator
        return ValueOperator(
            module=value_mlp,
            in_keys=[self.config.l2t.student_hidden_key],
        )

    def _construct_q_function(
        self,
        q_net: torch.nn.Module | None = None,
        in_keys: tuple[str, ...] = (),
        out_keys: tuple[str, ...] = (),
    ) -> TensorDictModule:
        """Construct Q-function module (unused in PPO-style; returns identity)."""
        in_keys = list(in_keys) if in_keys else list(self.total_input_keys)
        out_keys = list(out_keys) if out_keys else list(self.total_input_keys)
        if q_net is not None:
            return TensorDictModule(module=q_net, in_keys=in_keys, out_keys=out_keys)
        return TensorDictModule(
            module=torch.nn.Identity(), in_keys=in_keys, out_keys=out_keys
        )

    def _construct_loss_module(self) -> torch.nn.Module:
        """Construct loss module with teacher PPO and student imitation."""
        assert isinstance(self.config, L2TRLOptConfig)
        l2t_cfg = self.config.l2t

        # Initialize lazy layers by performing a forward pass with dummy data
        fake_tensordict = self.env.fake_tensordict()
        with torch.no_grad():
            _ = self.actor_critic(fake_tensordict)
            _ = self.student_actor_critic(fake_tensordict)

        return ClipL2TLoss(
            actor_network=self.actor_critic.get_policy_operator(),
            critic_network=self.actor_critic.get_value_operator(),
            student_actor_critic=self.student_actor_critic,  # Pass full actor-critic
            clip_epsilon=l2t_cfg.clip_epsilon,
            loss_critic_type=self.config.loss.loss_critic_type,
            entropy_coeff=l2t_cfg.entropy_coeff,
            critic_coeff=l2t_cfg.critic_coeff,
            normalize_advantage=l2t_cfg.normalize_advantage,
            clip_value=l2t_cfg.clip_value,
            imitation_type=l2t_cfg.imitation_type,
            imitation_coeff=l2t_cfg.imitation_coeff,
            student_loc_key=l2t_cfg.student_loc_key,
            student_scale_key=l2t_cfg.student_scale_key,
            student_hidden_key=l2t_cfg.student_hidden_key,
        )

    def _set_optimizers(
        self, optimizer_cls: OptimizerClass, optimizer_kwargs: dict[str, Any]
    ) -> list[torch.optim.Optimizer]:
        """Create optimizers for L2T teacher and student networks.

        Creates separate optimizers for:
        1. Teacher actor-critic (policy + value)
        2. Student actor-critic (policy + value + feature extractor)
        """
        optimizers = []

        # Teacher optimizer (for PPO)
        teacher_optim = optimizer_cls(
            self.actor_critic.parameters(),
            **optimizer_kwargs,
        )
        optimizers.append(teacher_optim)

        # Student optimizer (all student components together)
        # This includes feature extractor + policy + value
        if hasattr(self, "student_actor_critic"):
            student_optim = optimizer_cls(
                self.student_actor_critic.parameters(),
                **optimizer_kwargs,
            )
            optimizers.append(student_optim)

        return optimizers

    def _construct_adv_module(self) -> torch.nn.Module:
        """Construct advantage module"""
        assert isinstance(self.config, L2TRLOptConfig)
        # Create advantage module
        return GAE(
            gamma=self.config.loss.gamma,
            lmbda=self.config.l2t.gae_lambda,
            value_network=self.actor_critic.get_value_operator(),  # type: ignore
            average_gae=False,
            device=self.device,
            vectorized=not self.config.compile.compile,
        )

    def _construct_data_buffer(self) -> ReplayBuffer:
        """Construct data buffer"""
        # Create data buffer
        cfg = self.config
        sampler = (
            SamplerWithoutReplacement()
        )  # Removed True parameter to match ppo_mujoco.py
        return TensorDictReplayBuffer(
            storage=LazyMemmapStorage(
                cfg.collector.frames_per_batch,
                compilable=cfg.compile.compile,  # type: ignore
                device=self.device,
            ),
            sampler=sampler,
            batch_size=cfg.loss.mini_batch_size,
            compilable=cfg.compile.compile,
        )

    def _compile_components(self):
        """Compile components"""
        compile_mode = None
        cfg = self.config
        if cfg.compile.compile:
            compile_mode = cfg.compile.compile_mode
            if compile_mode in ("", None):
                if cfg.compile.cudagraphs:
                    compile_mode = "default"
                else:
                    compile_mode = "reduce-overhead"

            self.update = torch.compile(self.update, mode=compile_mode)  # type: ignore
            self.adv_module = torch.compile(self.adv_module, mode=compile_mode)  # type: ignore

    def update(
        self, batch: TensorDict, num_network_updates: int
    ) -> tuple[TensorDict, int]:
        """Update function"""
        stage_prefix = self._update_stage_context or "update"

        # Validate input tensordict
        if not self._validate_tensordict(
            batch, f"{stage_prefix}:pre_update_batch", raise_error=False
        ):
            self.log.debug(
                "Discarding batch due to non-finite values in input tensordict"
            )
            # Return dummy loss with zeros
            dummy_loss = TensorDict(
                {
                    "loss_critic": torch.tensor(0.0, device=self.device),
                    "loss_objective": torch.tensor(0.0, device=self.device),
                    "loss_entropy": torch.tensor(0.0, device=self.device),
                    "loss_imitation": torch.tensor(0.0, device=self.device),
                },
                batch_size=[],
            )
            alpha = torch.ones((), device=self.device)
            if self.config.optim.anneal_lr:
                alpha = 1 - (num_network_updates / self.total_network_updates)
            return dummy_loss.set("alpha", alpha), num_network_updates

        self.optim.zero_grad(set_to_none=True)

        # Linearly decrease the learning rate and clip epsilon
        alpha = torch.ones((), device=self.device)
        if self.config.optim.anneal_lr:
            alpha = 1 - (num_network_updates / self.total_network_updates)
            for group in self.optim.param_groups:
                group["lr"] = self.config.optim.lr * alpha

        if self.config.l2t.anneal_clip_epsilon:  # type: ignore
            self.loss_module.clip_epsilon.copy_(  # type: ignore
                self.config.l2t.clip_epsilon * alpha  # type: ignore
            )

        num_network_updates = num_network_updates + 1

        # Forward pass PPO loss
        loss = self.loss_module(batch)
        loss = self._sanitize_loss_tensordict(loss, f"{stage_prefix}:raw_loss")
        critic_loss = loss["loss_critic"]
        actor_loss = loss["loss_objective"] + loss["loss_entropy"]
        imitation_loss = loss.get("loss_imitation", torch.zeros_like(actor_loss))
        total_loss = critic_loss + actor_loss + imitation_loss

        # Backward pass
        total_loss.backward()

        # Validate gradients
        grads_finite = self._validate_gradients(
            f"{stage_prefix}:post_backward", raise_error=False
        )

        if not grads_finite:
            self.log.debug(
                "Skipping optimizer step due to non-finite gradients; batch discarded"
            )
            self.optim.zero_grad(set_to_none=True)
            return loss.detach().set("alpha", alpha), num_network_updates

        # Update the networks
        self.optim.step()
        return loss.detach().set("alpha", alpha), num_network_updates

    def predict(self, obs: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Predict action given observation"""
        obs = torch.as_tensor([obs], device=self.device)
        self.policy.eval()  # type: ignore
        with torch.inference_mode():
            td = TensorDict(
                dict.fromkeys(self.config.policy_in_keys, obs),
                batch_size=[1],
                device=self.policy.device,  # type: ignore
            )
            return self.policy(td).get("action")  # type: ignore

    def _construct_trainer(self) -> None:  # type: ignore
        """Override to return None since we implement custom training loop"""
        return

    def _update_policy(self, batch: TensorDict) -> dict[str, float]:
        msg = "PPO does not support _update_policy"
        raise NotImplementedError(msg)

    def _compute_action(self, tensordict: TensorDict) -> TensorDict:
        msg = "PPO does not support _compute_action"
        raise NotImplementedError(msg)

    def _compute_returns(self, rollout: Tensor) -> Tensor:
        msg = "PPO does not support _compute_returns"
        raise NotImplementedError(msg)

    def train(self) -> None:  # type: ignore
        """Train the agent"""
        assert isinstance(self.config, L2TRLOptConfig)
        cfg = self.config
        # Main loop
        collected_frames = 0
        num_network_updates = torch.zeros((), dtype=torch.int64, device=self.device)
        pbar = tqdm.tqdm(total=self.config.collector.total_frames)

        num_mini_batches = cfg.collector.frames_per_batch // cfg.loss.mini_batch_size
        if cfg.collector.frames_per_batch % cfg.loss.mini_batch_size != 0:
            num_mini_batches += 1

        self.total_network_updates = (
            (cfg.collector.total_frames // cfg.collector.frames_per_batch)
            * cfg.loss.epochs
            * num_mini_batches
        )

        # extract cfg variables
        cfg_loss_ppo_epochs: int = self.config.loss.epochs
        cfg_optim_lr: torch.Tensor = torch.tensor(
            self.config.optim.lr, device=self.device
        )
        cfg_loss_anneal_clip_eps: bool = self.config.l2t.anneal_clip_epsilon
        cfg_loss_clip_epsilon: float = self.config.l2t.clip_epsilon

        losses = TensorDict(batch_size=[cfg_loss_ppo_epochs, num_mini_batches])  # type: ignore

        self.collector: SyncDataCollector
        collector_iter = iter(self.collector)
        total_iter = len(self.collector)
        for _i in range(total_iter):
            # timeit.printevery(1000, total_iter, erase=True)

            with timeit("collecting"):
                data = next(collector_iter)
                self._validate_tensordict(
                    data, f"collector_iter:step{_i}", raise_error=False
                )

            metrics_to_log = {}
            frames_in_batch = data.numel()
            collected_frames += frames_in_batch
            pbar.update(frames_in_batch)

            # Get training rewards and episode lengths
            episode_rewards = data["next", "episode_reward"][data["next", "done"]]
            if len(episode_rewards) > 0:
                episode_length = data["next", "step_count"][data["next", "done"]]
                self.episode_lengths.extend(episode_length.cpu().tolist())
                self.episode_rewards.extend(episode_rewards.cpu().tolist())
                metrics_to_log.update(
                    {
                        "episode/length": np.mean(self.episode_lengths),
                        "episode/return": np.mean(self.episode_rewards),
                        "train/reward": episode_rewards.mean().item(),
                    }
                )
            self.data_buffer.empty()
            with timeit("training"):
                for j in range(cfg_loss_ppo_epochs):
                    # Compute GAE
                    self._validate_tensordict(data, f"epoch{j}:pre_gae_input")
                    self._validate_parameters(f"epoch{j}:pre_gae")
                    with torch.no_grad(), timeit("adv"):
                        torch.compiler.cudagraph_mark_step_begin()
                        data = self.adv_module(data)
                        if self.config.compile.compile_mode:
                            data = data.clone()
                    self._validate_parameters(f"epoch{j}:post_gae")
                    self._validate_tensordict(data, f"epoch{j}:post_gae")

                    with timeit("rb - extend"):
                        # Update the data buffer
                        data_reshape = data.reshape(-1)
                        self.data_buffer.extend(data_reshape)

                    for k, batch in enumerate(self.data_buffer):
                        if not self._validate_tensordict(
                            batch,
                            f"epoch{j}:mini_batch{k}:pre_update_batch",
                            raise_error=False,
                        ):
                            self.log.debug(
                                "Discarding mini-batch %d in epoch %d due to non-finite values",
                                k,
                                j,
                            )
                            continue
                        self._update_stage_context = f"epoch{j}:mini_batch{k}"
                        with timeit("update"):
                            torch.compiler.cudagraph_mark_step_begin()
                            loss, num_network_updates = self.update(  # type: ignore
                                batch, num_network_updates=num_network_updates
                            )
                            loss = loss.clone()
                        self._update_stage_context = ""
                        self._validate_tensordict(
                            batch, f"epoch{j}:mini_batch{k}:post_update_batch"
                        )
                        num_network_updates = num_network_updates.clone()  # type: ignore
                        losses[j, k] = loss.select(
                            "loss_critic", "loss_entropy", "loss_objective"
                        )

            # Get training losses and times
            losses_mean = losses.apply(lambda x: x.float().mean(), batch_size=[])
            for key, value in losses_mean.items():  # type: ignore
                metrics_to_log.update({f"train/{key}": value.item()})
            metrics_to_log.update(
                {
                    "train/lr": loss["alpha"] * cfg_optim_lr,
                    "train/clip_epsilon": (
                        loss["alpha"] * cfg_loss_clip_epsilon
                        if cfg_loss_anneal_clip_eps
                        else cfg_loss_clip_epsilon
                    ),
                }
            )

            # for IsaacLab, we need to log the metrics from the environment
            if "Isaac" in self.config.env.env_name and hasattr(self.env, "log_infos"):
                for _ in range(len(self.env.log_infos)):
                    log_info_dict: dict[str, Tensor] = self.env.log_infos.popleft()
                    # log all the keys
                    for key, value in log_info_dict.items():
                        if "/" in key:
                            metrics_to_log.update(
                                {
                                    key: (
                                        value.item()
                                        if isinstance(value, Tensor)
                                        else value
                                    )
                                }
                            )
                        else:
                            metrics_to_log.update(
                                {
                                    "Episode/" + key: (
                                        value.item()
                                        if isinstance(value, Tensor)
                                        else value
                                    )
                                }
                            )

            # Log metrics
            if self.logger:
                metrics_to_log.update(timeit.todict(prefix="time"))  # type: ignore
                metrics_to_log["time/speed"] = pbar.format_dict["rate"]
                for key, value in metrics_to_log.items():
                    self.logger.log_scalar(key, value, collected_frames)  # type: ignore

            self.collector.update_policy_weights_()

            # Save model periodically
            if (
                self.config.save_interval > 0
                and collected_frames % self.config.save_interval == 0
            ):
                self.save_model(step=collected_frames)

        self.collector.shutdown()


class L2TR(L2T):
    """PPO with LSTM-based feature extractor following TorchRL recurrent patterns.

    This implementation follows the TorchRL tutorial patterns from:
    https://docs.pytorch.org/rl/main/tutorials/dqn_with_rnn.html

    Key features:
    - Uses TorchRL's LSTMModule for proper recurrent state management
    - Automatically handles recurrent states through TensorDict
    - Adds TensorDictPrimer for proper state initialization
    - Compatible with episode boundaries and InitTracker transform

    For optimal performance, ensure your environment includes:
    - InitTracker() transform to handle episode boundaries
    - Proper episode termination handling

    Example configuration:
        feature_extractor:
            lstm:
                hidden_size: 256
                num_layers: 1
                dropout: 0.0
                bidirectional: false
    """

    def __init__(
        self,
        env: TransformedEnv,
        config: L2TRLOptConfig,
        policy_net: torch.nn.Module | None = None,
        value_net: torch.nn.Module | None = None,
        q_net: torch.nn.Module | None = None,
        replay_buffer: type[ReplayBuffer] = ReplayBuffer,
        logger: Logger | None = None,
        **kwargs,
    ):
        # Store LSTM module reference for primer creation
        self.lstm_module: LSTMModule | None = None

        # Ensure required transforms (InitTracker) are present
        env = self.add_required_transforms(env)

        super().__init__(
            env,
            config,
            policy_net,
            value_net,
            q_net,
            replay_buffer,
            logger,
            **kwargs,
        )

        # Add recurrent state primer to environment if LSTM is used
        if self.lstm_module is not None:
            primer = self.lstm_module.make_tensordict_primer()
            if hasattr(self.env, "append_transform"):
                self.env.append_transform(primer)
            # For environments that don't support append_transform
            elif hasattr(self.env, "transform") and self.env.transform is not None:
                if isinstance(self.env.transform, Compose):
                    self.env.transform.append(primer)
                else:
                    self.env.transform = Compose(self.env.transform, primer)

    def _construct_student_feature_extractor(self) -> TensorDictModule:
        """Student feature extractor using LSTM (recurrent student)."""
        assert isinstance(self.config, L2TRLOptConfig)
        out_key = self.config.l2t.student_hidden_key
        if self.config.use_feature_extractor:
            # Use LSTMModule for student features
            hidden_size = int(getattr(self.config.feature_extractor, "output_dim", 256))
            # Optional per-student LSTM config via student's shared.features
            student_cfg = self.config.l2t.student
            shared_name: str | None = None
            if student_cfg.policy and student_cfg.policy.feature_ref:
                shared_name = student_cfg.policy.feature_ref
            elif student_cfg.value and student_cfg.value.feature_ref:
                shared_name = student_cfg.value.feature_ref
            if shared_name and shared_name in student_cfg.shared.features:
                spec = student_cfg.shared.features[shared_name]
                if spec.type == "lstm" and spec.lstm is not None:
                    hidden_size = int(spec.lstm.hidden_size)
                    num_layers = int(spec.lstm.num_layers)
                    dropout = float(spec.lstm.dropout)
                    bidirectional = bool(spec.lstm.bidirectional)
                else:
                    num_layers = 1
                    dropout = 0.0
                    bidirectional = False
            else:
                num_layers = 1
                dropout = 0.0
                bidirectional = False

            self.lstm_module = LSTMModule(
                input_size=self.total_input_shape,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=bidirectional,
                batch_first=True,
                device=self.device,
                in_key=self.total_input_keys[0],
                out_key=out_key,
            )
            # Return the LSTMModule directly (it is a TensorDictModule)
            return self.lstm_module  # type: ignore[return-value]
        # Identity if not using FE
        return TensorDictModule(
            module=torch.nn.Identity(),
            in_keys=list(self.total_input_keys),
            out_keys=list(self.total_input_keys),
        )

    def _student_policy_input_dim(self) -> int:
        # After FE, default student in_keys=['student_hidden']
        if self.config.use_feature_extractor:
            # Prefer student's shared feature spec output_dim if present
            student_cfg = self.config.l2t.student  # type: ignore
            shared_name: str | None = None
            if student_cfg.policy and student_cfg.policy.feature_ref:
                shared_name = student_cfg.policy.feature_ref
            elif student_cfg.value and student_cfg.value.feature_ref:
                shared_name = student_cfg.value.feature_ref
            if shared_name and shared_name in student_cfg.shared.features:
                return int(student_cfg.shared.features[shared_name].output_dim)
            return int(self.config.feature_extractor.output_dim)
        # If no FE, sum dims over configured keys
        keys = (
            self.config.l2t.student.policy.in_keys  # type: ignore
            if self.config.l2t.student and self.config.l2t.student.policy  # type: ignore
            else [self.config.l2t.student_hidden_key]  # type: ignore
        )
        return int(
            torch.tensor(
                [self.env.observation_spec[k].shape[-1] for k in keys]  # type: ignore[index]
            )
            .sum()
            .item()
        )

    def _construct_student_policy(self) -> TensorDictModule:
        """Student policy head producing loc/scale under student keys (for L2TR with LSTM)."""
        assert isinstance(self.config, L2TRLOptConfig)

        # Get input dimension from LSTM output
        input_dim = self._student_policy_input_dim()

        # Determine policy head configuration
        num_cells = [256, 256]  # default
        activation = "relu"  # default

        student_layout = self.config.l2t.student
        if student_layout and student_layout.policy and student_layout.policy.head:
            head_cfg = student_layout.policy.head
            if head_cfg.num_cells:
                num_cells = list(head_cfg.num_cells)
            if head_cfg.activation:
                activation = head_cfg.activation

        # Build policy MLP
        policy_mlp = MLP(
            in_features=input_dim,
            out_features=2 * self.env.action_spec_unbatched.shape[-1],  # type: ignore
            num_cells=num_cells,
            activation_class=get_activation_class(activation),
            device=self.device,
        )

        # Add parameter extractor for normal distribution
        extractor = NormalParamExtractor(
            scale_mapping="biased_softplus_1.0",
            scale_lb=0.1,  # type: ignore
        ).to(self.device)

        net = torch.nn.Sequential(policy_mlp, extractor)

        # Wrap in TensorDictModule
        return TensorDictModule(
            module=net,
            in_keys=[self.config.l2t.student_hidden_key],
            out_keys=[
                self.config.l2t.student_loc_key,
                self.config.l2t.student_scale_key,
            ],
        )

    def _student_value_input_dim(self) -> int:
        # Same logic as policy
        return self._student_policy_input_dim()

    def _construct_student_value_function(self) -> TensorDictModule:
        """Student value function using student layout if provided (for L2TR with LSTM)."""
        assert isinstance(self.config, L2TRLOptConfig)

        # Get input dimension from LSTM output
        input_dim = self._student_value_input_dim()

        # Determine value head configuration
        num_cells = [256, 256]  # default
        activation = "relu"  # default

        student_layout = self.config.l2t.student
        if student_layout and student_layout.value and student_layout.value.head:
            head_cfg = student_layout.value.head
            if head_cfg.num_cells:
                num_cells = list(head_cfg.num_cells)
            if head_cfg.activation:
                activation = head_cfg.activation

        # Build value MLP
        value_mlp = MLP(
            in_features=input_dim,
            out_features=1,
            num_cells=num_cells,
            activation_class=get_activation_class(activation),
            device=self.device,
        )

        # Wrap in ValueOperator
        return ValueOperator(
            module=value_mlp,
            in_keys=[self.config.l2t.student_hidden_key],
        )

    def _construct_adv_module(self) -> torch.nn.Module:
        """Construct advantage module for recurrent L2T (teacher branch)."""
        assert isinstance(self.config, L2TRLOptConfig)
        return GAE(
            gamma=self.config.loss.gamma,
            lmbda=self.config.l2t.gae_lambda,
            value_network=self.actor_critic.get_value_operator(),  # type: ignore
            average_gae=False,
            device=self.device,
            vectorized=not self.config.compile.compile,
            deactivate_vmap=True,  # to be compatible with lstm
            shifted=True,  # to be compatible with lstm
        )

    def predict(self, obs: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Predict action given observation with LSTM state management"""
        obs = torch.as_tensor([obs], device=self.device)
        self.policy.eval()  # type: ignore

        with torch.inference_mode():
            td = TensorDict(
                dict.fromkeys(self.config.policy_in_keys, obs),
                batch_size=[1],
                device=self.policy.device,  # type: ignore
            )

            # The LSTMModule automatically handles recurrent states through TensorDict
            # No need for manual state management - TorchRL handles this
            return self.policy(td).get("action")  # type: ignore

    def reset_recurrent_states(self):
        """Reset LSTM recurrent states - called at episode boundaries"""
        # With TorchRL's LSTMModule, states are automatically managed
        # This method is provided for compatibility but may not be needed
        # as the InitTracker transform and proper episode boundaries handle this

    @classmethod
    def check_environment_compatibility(
        cls, env: TransformedEnv
    ) -> tuple[bool, list[str]]:
        """Check if environment is properly configured for recurrent policies.

        Args:
            env: The environment to check

        Returns:
            tuple: (is_compatible, list_of_missing_transforms)
        """
        missing_transforms = []

        # Check for InitTracker
        has_init_tracker = False
        if hasattr(env, "transform") and env.transform is not None:
            if isinstance(env.transform, Compose):
                transforms = env.transform._modules
            else:
                transforms = [env.transform]

            for transform in transforms:
                if isinstance(transform, InitTracker):
                    has_init_tracker = True
                    break

        if not has_init_tracker:
            missing_transforms.append(
                "InitTracker() - needed for episode boundary tracking"
            )

        is_compatible = len(missing_transforms) == 0
        return is_compatible, missing_transforms

    @classmethod
    def add_required_transforms(cls, env: TransformedEnv) -> TransformedEnv:
        """Add required transforms for recurrent policy compatibility.

        Args:
            env: The environment to modify

        Returns:
            Modified environment with required transforms
        """
        is_compatible, missing_transforms = cls.check_environment_compatibility(env)

        if not is_compatible:
            new_transforms = []

            # Add InitTracker if missing
            if any("InitTracker" in msg for msg in missing_transforms):
                new_transforms.append(InitTracker())

            # Add new transforms to environment
            if new_transforms:
                if hasattr(env, "transform") and env.transform is not None:
                    if isinstance(env.transform, Compose):
                        for transform in new_transforms:
                            env.transform.append(transform)
                    else:
                        env.transform = Compose(env.transform, *new_transforms)
                else:
                    env.transform = Compose(*new_transforms)

        return env


class ClipL2TLoss(ClipPPOLoss):
    """PPO loss for the teacher plus an imitation term for the student.

    The imitation term can be one of:
    - "l2": MSE between student deterministic action (tanh(loc)) and teacher action
    - "asymmetric": MSE weighted by positive teacher advantage
    - "bc": negative log-likelihood of teacher action under student's TanhNormal
    """

    def __init__(
        self,
        *,
        actor_network: TensorDictModule,
        critic_network: TensorDictModule,
        student_actor_critic: ActorValueOperator,
        clip_epsilon: float,
        loss_critic_type: str = "l2",
        entropy_coeff: float = 0.0,
        critic_coeff: float = 1.0,
        normalize_advantage: bool = False,
        clip_value: bool = True,
        imitation_type: str = "l2",
        imitation_coeff: float = 1.0,
        student_loc_key: str = "student_loc",
        student_scale_key: str = "student_scale",
        student_hidden_key: str = "student_hidden",
    ) -> None:
        super().__init__(
            actor_network=actor_network,  # type: ignore[arg-type]
            critic_network=critic_network,
            clip_epsilon=clip_epsilon,
            loss_critic_type=loss_critic_type,
            entropy_coeff=entropy_coeff,
            critic_coeff=critic_coeff,
            normalize_advantage=normalize_advantage,
            clip_value=clip_value,
        )
        self.student_actor_critic = student_actor_critic
        self.imitation_type = imitation_type
        self.imitation_coeff = imitation_coeff
        self.student_loc_key = student_loc_key
        self.student_scale_key = student_scale_key
        self.student_hidden_key = student_hidden_key

    def forward(self, tensordict: TensorDict) -> TensorDict:  # type: ignore[override]
        # Compute teacher PPO losses
        loss_td = super().forward(tensordict)

        # Compute student outputs via full actor-critic (feature extractor + policy head)
        # Clone tensordict to avoid modifying the original
        td = tensordict.clone(False)

        # Pass through student actor-critic to get student_loc and student_scale
        # The student actor-critic should:
        # 1. Extract features from observations (via common_operator)
        # 2. Generate policy parameters (loc, scale) via policy_operator
        td = self.student_actor_critic(td)

        imitation_loss = td.new_zeros(())
        if self.imitation_coeff > 0.0:
            teacher_action = tensordict.get("action")
            student_loc = td.get(self.student_loc_key)
            student_scale = td.get(self.student_scale_key)

            if self.imitation_type == "bc":
                # Behavior cloning with TanhNormal log-likelihood
                dist = TanhNormal(student_loc, student_scale, event_dims=1)
                nll = -dist.log_prob(teacher_action)
                imitation_loss = nll.mean()
            else:
                # Deterministic action via tanh(loc)
                student_action = torch.tanh(student_loc)
                sq_error = (student_action - teacher_action).pow(2).sum(-1)
                if self.imitation_type == "asymmetric":
                    adv = tensordict.get("advantage", None)
                    if adv is not None:
                        weight = torch.relu(adv).squeeze(-1)
                        sq_error = weight * sq_error
                imitation_loss = sq_error.mean()

            imitation_loss = self.imitation_coeff * imitation_loss

        # Add imitation loss to the loss tensordict
        loss_td.set("loss_imitation", imitation_loss)
        return loss_td

    # Annotations to avoid TorchRL functionalization warnings
    actor_network: TensorDictModule
    critic_network: TensorDictModule
    actor_network_params: TensorDictParams
    critic_network_params: TensorDictParams
    target_actor_network_params: TensorDictParams
    target_critic_network_params: TensorDictParams

    # Student actor-critic (not converted to functional by base class)
    student_actor_critic: ActorValueOperator
