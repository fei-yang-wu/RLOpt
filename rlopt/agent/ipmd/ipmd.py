from __future__ import annotations

import functools
import logging
import warnings
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import torch
import tqdm

# Suppress torch.compile CUDA graph diagnostic messages
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
logging.getLogger("torch._inductor").setLevel(logging.ERROR)

# Suppress CUDA graph skip messages
import os

os.environ.setdefault("TORCH_LOGS", "-all")
from tensordict import TensorDict
from tensordict.nn import CudaGraphModule, TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import Tensor, nn
from torchrl._utils import compile_with_warmup, timeit
from torchrl.data import (
    LazyMemmapStorage,
    LazyTensorStorage,
    ReplayBuffer,
    TensorDictReplayBuffer,
)
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import (
    MLP,
    ActorCriticOperator,
    ProbabilisticActor,
    TanhNormal,
    ValueOperator,
)
from torchrl.objectives import SoftUpdate, group_optimizers
from torchrl.objectives.sac import SACLoss
from torchrl.record.loggers import Logger

from rlopt.base_class import BaseAlgorithm
from rlopt.configs import NetworkConfig, RLOptConfig
from rlopt.imitation import ExpertReplayBuffer
from rlopt.type_aliases import OptimizerClass
from rlopt.utils import get_activation_class, log_info


@dataclass
class IPMDConfig:
    """IPMD-specific configuration."""

    alpha_init: float = 1.0
    """Initial alpha value."""

    min_alpha: float | None = None
    """Minimum alpha value."""

    max_alpha: float | None = None
    """Maximum alpha value."""

    num_qvalue_nets: int = 2
    """Number of Q-value networks."""

    fixed_alpha: bool = False
    """Whether to fix alpha."""

    target_entropy: float | str = "auto"
    """Target entropy."""

    delay_actor: bool = False
    """Whether to delay the actor network."""

    delay_qvalue: bool = True
    """Whether to delay the Q-value network."""

    delay_value: bool = True
    """Whether to delay the value network."""

    priority_key: str = "td_error"
    """Priority key."""

    separate_losses: bool = False
    """Whether to separate losses."""

    reduction: str = "mean"
    """Reduction."""

    skip_done_states: bool = False
    """Whether to skip done states."""

    deactivate_vmap: bool = False
    """Whether to deactivate vmap."""

    utd_ratio: float = 1.0
    """Number of updates per batch."""

    # Reward estimator network and loss settings
    reward_num_cells: tuple[int, ...] = (256, 256)
    """Hidden layer sizes for the reward estimator MLP."""

    reward_activation: str = "elu"
    """Activation for reward estimator MLP."""

    reward_init: str = "orthogonal"
    """Weight init for reward estimator."""

    reward_loss_coeff: float = 1.0
    """Scale for (sum r_pi - sum r_expert)."""

    reward_l2_coeff: float = 1e-4
    """L2 regularization weight for reward parameters."""

    reward_detach_features: bool = True
    """Detach features when computing reward loss (avoid leaking grads)."""

    use_estimated_rewards_for_sac: bool = True
    """Whether to use estimated rewards instead of environment rewards for SAC loss."""

    expert_batch_size: int | None = None
    """Batch size for expert data sampling. If None, uses the same as mini_batch_size."""

    detach_reward_when_used_for_sac: bool = True
    """Detach the estimated reward tensor when injecting it into the SAC loss.

    This prevents the SAC updates from backpropagating into the reward estimator.
    The reward network is then trained solely via the IPMD objective.
    """


@dataclass
class IPMDRLOptConfig(RLOptConfig):
    """IPMD configuration that extends RLOptConfig."""

    ipmd: IPMDConfig = field(default_factory=IPMDConfig)
    """IPMD configuration."""

    def __post_init__(self):
        self.use_value_function = False
        # Initialize q_function config if not set (IPMD uses Q-learning)
        if self.q_function is None:
            self.q_function = NetworkConfig(
                num_cells=[256, 256],
                activation_fn="relu",
                output_dim=1,
                input_keys=["observation"],
            )


class IPMD(BaseAlgorithm):
    """IPMD algorithm.

    The class mirrors the PPO structure (custom train loop) while adapting
    to IPMD's off-policy setting with a replay buffer.
    """

    def __init__(
        self,
        env,
        config: IPMDRLOptConfig,
        policy_net: nn.Module | None = None,
        q_net: nn.Module | None = None,  # optional external Q-value module
        replay_buffer: type[ReplayBuffer] = ReplayBuffer,
        logger: Logger | None = None,
        feature_extractor_net: nn.Module | None = None,
        **kwargs,
    ):
        # Narrow the type for static checkers
        self.config = cast(IPMDRLOptConfig, config)
        self.config: IPMDRLOptConfig
        self.env = env

        # Reward estimator: r(s, a, s') -> scalar
        self.reward_estimator: nn.Module = self._construct_reward_estimator()
        self.reward_estimator.to(self.device)

        super().__init__(
            env=env,
            config=config,
            policy_net=policy_net,
            value_net=None,
            q_net=q_net,
            replay_buffer=replay_buffer,
            logger=logger,
            feature_extractor_net=feature_extractor_net,
            **kwargs,
        )

        # Expert data sources (iterator and/or replay buffer)
        self._expert_iterator: Iterator[TensorDict] | None = None
        self._expert_buffer: ExpertReplayBuffer | None = None
        self._warned_no_expert = False

        # Compile if requested
        self._compile_components()

        # Initialize total network updates
        self.total_network_updates = 0

        # construct the target net updater
        self.target_net_updater = self._construct_target_net_updater()

    def _construct_feature_extractor(
        self, feature_extractor_net: torch.nn.Module | None = None
    ) -> TensorDictModule:
        """Build feature extractor (optional for IPMD)."""
        msg = "IPMD does not require a feature extractor by default."
        raise NotImplementedError(msg)

    def _construct_policy(
        self, policy_net: torch.nn.Module | None = None
    ) -> TensorDictModule:
        """Construct policy"""
        # for IPMD, we use a probabilistic actor

        # Define policy output distribution class
        distribution_class = TanhNormal
        distribution_kwargs = {
            "low": self.env.action_spec_unbatched.space.low.to(self.device),  # type: ignore
            "high": self.env.action_spec_unbatched.space.high.to(self.device),  # type: ignore
            "tanh_loc": False,
        }

        # Build policy network
        if policy_net is None:
            policy_mlp = MLP(
                in_features=self.config.policy.input_dim,
                activation_class=get_activation_class(self.config.policy.activation_fn),
                out_features=2 * self.env.action_spec_unbatched.shape[-1],  # type: ignore
                num_cells=list(self.config.policy.num_cells),
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
            in_keys=list(self.config.policy.input_keys),
            out_keys=["loc", "scale"],
        )

        return ProbabilisticActor(
            policy_td,
            in_keys=["loc", "scale"],
            spec=self.env.full_action_spec_unbatched.to(self.device),  # type: ignore
            distribution_class=distribution_class,
            distribution_kwargs=distribution_kwargs,
            return_log_prob=False,
            default_interaction_type=ExplorationType.RANDOM,
        )

    def _construct_value_function(
        self, value_net: nn.Module | None = None
    ) -> TensorDictModule:  # type: ignore[override]
        """IPMD does not use a state-value function explicitly."""
        msg = "IPMD does not require a value function."
        raise NotImplementedError(msg)

    def _construct_q_function(self, q_net: nn.Module | None = None) -> TensorDictModule:
        """Construct Q-function for IPMD (used in SAC loss)."""
        if self.config.q_function is None:
            msg = "IPMD requires a Q-function configuration."
            raise ValueError(msg)

        # Build Q-network
        if q_net is None:
            q_mlp = MLP(
                in_features=self.config.q_function.input_dim,
                activation_class=get_activation_class(
                    self.config.q_function.activation_fn
                ),
                out_features=1,
                num_cells=list(self.config.q_function.num_cells),
                device=self.device,
            )
        else:
            q_mlp = q_net

        # Q-function takes both observation and action as inputs
        in_keys = list(self.config.q_function.input_keys)
        if "action" not in in_keys:
            in_keys.append("action")

        return ValueOperator(
            module=q_mlp,
            in_keys=in_keys,
            out_keys=["state_action_value"],
        )

    def _construct_reward_estimator(self) -> nn.Module:
        """Create reward network mapping [phi(s), a, phi(s')] to scalar."""
        cfg = self.config
        assert isinstance(cfg, IPMDRLOptConfig)
        # Get observation dimension from environment
        obs_dim = self.env.observation_spec["observation"].shape[-1]
        act_dim = self.env.action_spec.shape[-1]
        # Reward network takes [obs, action, next_obs]
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

    def _construct_actor_critic(self) -> TensorDictModule:
        """Construct actor-critic network for IPMD."""
        assert isinstance(self.q_function, TensorDictModule), (
            "Q-function must be a TensorDictModule"
        )
        assert isinstance(self.policy, TensorDictModule), (
            "Policy must be a TensorDictModule"
        )

        # Use feature extractor if available, otherwise use identity
        if self.feature_extractor:
            common_operator = self.feature_extractor
        else:
            # Create a dummy identity module when no feature extractor
            class IdentityModule(torch.nn.Module):
                def forward(self, x):
                    return x

            common_operator = TensorDictModule(
                module=IdentityModule(),
                in_keys=["observation"],
                out_keys=["observation"],
            )

        return ActorCriticOperator(
            common_operator=common_operator,
            policy_operator=self.policy,
            value_operator=self.q_function,
        )

    def _construct_loss_module(self) -> nn.Module:
        assert isinstance(self.config, IPMDRLOptConfig)
        sac_cfg = self.config.ipmd
        # Read num critics from network layout if provided
        if (
            hasattr(self.config, "network")
            and self.config.network
            and hasattr(self.config.network, "critic")
        ):
            sac_cfg.num_qvalue_nets = self.config.network.critic.num_nets

        assert isinstance(self.actor_critic, ActorCriticOperator), (
            "actor_critic must be an ActorCriticOperator"
        )

        # Initialize lazy layers by performing a forward pass with dummy data
        fake_tensordict = self.env.fake_tensordict()
        with torch.no_grad():
            _ = self.actor_critic(fake_tensordict)

        loss_module = SACLoss(
            actor_network=self.actor_critic.get_policy_operator(),  # type: ignore[arg-type]
            qvalue_network=self.actor_critic.get_critic_operator(),  # type: ignore[arg-type]
            num_qvalue_nets=sac_cfg.num_qvalue_nets,
            alpha_init=sac_cfg.alpha_init,
            loss_function=self.config.loss.loss_critic_type,
            min_alpha=sac_cfg.min_alpha,
            max_alpha=sac_cfg.max_alpha,
            action_spec=None,
            fixed_alpha=sac_cfg.fixed_alpha,
            target_entropy=sac_cfg.target_entropy,
            delay_actor=sac_cfg.delay_actor,
            delay_qvalue=sac_cfg.delay_qvalue,
            delay_value=sac_cfg.delay_value,
            separate_losses=sac_cfg.separate_losses,
            reduction=sac_cfg.reduction,
            skip_done_states=sac_cfg.skip_done_states,
            deactivate_vmap=sac_cfg.deactivate_vmap,
        )
        loss_module.make_value_estimator(gamma=self.config.loss.gamma)

        return loss_module

    def _construct_target_net_updater(self) -> SoftUpdate:
        # Prefer polyak parameter from network layout critic if present
        eps = self.config.optim.target_update_polyak
        if (
            hasattr(self.config, "network")
            and self.config.network
            and hasattr(self.config.network, "critic")
        ):
            eps = self.config.network.critic.polyak_eps
        return SoftUpdate(self.loss_module, eps=eps)

    # _get_activation_class and _initialize_weights now provided by BaseAlgorithm

    def _set_optimizers(
        self, optimizer_cls: OptimizerClass, optimizer_kwargs: dict[str, Any]
    ) -> list[torch.optim.Optimizer]:
        """Create optimizers for IPMD components: actor, critic, alpha, and reward estimator."""
        # Critic/Q-function parameters
        critic_params = list(
            self.loss_module.qvalue_network_params.flatten_keys().values()  # type: ignore[attr-defined]
        )
        # Actor/Policy parameters
        actor_params = list(
            self.loss_module.actor_network_params.flatten_keys().values()  # type: ignore[attr-defined]
        )

        optimizers = [
            optimizer_cls(actor_params, **optimizer_kwargs),
            optimizer_cls(critic_params, **optimizer_kwargs),
        ]

        # Alpha optimizer for SAC entropy temperature
        if hasattr(self.loss_module, "log_alpha"):
            param = self.loss_module.log_alpha
            if isinstance(param, torch.nn.Parameter):
                optimizers.append(torch.optim.Adam([param], lr=3.0e-4))

        # Reward estimator optimizer for IPMD objective
        if hasattr(self, "reward_estimator"):
            optimizers.append(
                optimizer_cls(self.reward_estimator.parameters(), **optimizer_kwargs)
            )

        return optimizers

    def _construct_data_buffer(self) -> ReplayBuffer:
        """Construct data buffer"""
        # Create data buffer
        cfg = self.config
        sampler = SamplerWithoutReplacement()
        scratch_dir = cfg.collector.scratch_dir
        device = cfg.device
        buffer_size = cfg.replay_buffer.size
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
        if cfg.replay_buffer.prb:
            self.log.warning(
                "Prioritized replay buffer (prb) is not supported; using uniform replay."
            )
        replay_buffer = TensorDictReplayBuffer(
            pin_memory=False,
            prefetch=prefetch,
            sampler=sampler,
            storage=storage_cls(max_size=buffer_size, compilable=cfg.compile.compile),
            batch_size=batch_size,
            shared=shared,
        )
        if scratch_dir:
            replay_buffer.append_transform(lambda td: td.to(device))  # type: ignore[arg-type]
        return replay_buffer

    def _construct_trainer(self):  # type: ignore[override]
        return None

    # -------------------------
    # Expert data API
    # -------------------------
    def set_expert_iterator(self, iterator: Iterator[TensorDict]) -> None:
        """Attach an expert data iterator.

        The iterator must yield TensorDict batches with keys:
        - 'observation' (s_t)
        - 'action' (a_t)
        - ('next', 'observation') (s_{t+1})
        """
        self._expert_iterator = iterator

    def set_expert_buffer(
        self, buffer: TensorDictReplayBuffer | ExpertReplayBuffer
    ) -> None:
        """Attach an expert replay buffer.

        This can be used to avoid Python-level iteration overhead and to
        leverage TorchRL's device-aware sampling. The buffer is expected to
        contain the keys: 'observation', 'action', ('next', 'observation').

        Args:
            buffer: Either a TensorDictReplayBuffer or ExpertReplayBuffer instance
        """
        if isinstance(buffer, ExpertReplayBuffer):
            self._expert_buffer = buffer
        else:
            # Wrap TensorDictReplayBuffer in ExpertReplayBuffer for consistency
            self._expert_buffer = ExpertReplayBuffer(buffer)

    def set_expert_source(self, source: Any) -> None:
        """Attach an expert data source.

        Accepts:
        - ImitationLearningTools ExpertReplayManager (primary)
        - TorchRL TensorDictReplayBuffer
        - Any object with a .sample() method

        The source will be wrapped in ExpertReplayBuffer for consistent interface.

        Example with ILTools:
            >>> from iltools_datasets.replay_manager import ExpertReplayManager
            >>> mgr = ExpertReplayManager(spec)
            >>> ipmd.set_expert_source(mgr)

        Example with TorchRL buffer:
            >>> buffer = TensorDictReplayBuffer(...)
            >>> ipmd.set_expert_source(buffer)
        """
        self._expert_buffer = ExpertReplayBuffer(source)

    def create_expert_buffer(
        self, expert_data: TensorDict, buffer_size: int | None = None
    ) -> ExpertReplayBuffer:
        """Create an expert replay buffer from expert demonstration data.

        Args:
            expert_data: TensorDict containing expert demonstrations with keys:
                - 'observation' (s_t)
                - 'action' (a_t)
                - ('next', 'observation') (s_{t+1})
            buffer_size: Maximum buffer size. If None, uses the size of expert_data.

        Returns:
            ExpertReplayBuffer wrapping a TensorDictReplayBuffer with expert data
        """
        if buffer_size is None:
            buffer_size = expert_data.numel()

        cfg = self.config
        sampler = SamplerWithoutReplacement()
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

        # Add expert data to buffer
        expert_buffer.extend(expert_data.reshape(-1))

        if scratch_dir:
            expert_buffer.append_transform(lambda td: td.to(device))  # type: ignore[arg-type]

        # Wrap in ExpertReplayBuffer for consistent interface
        return ExpertReplayBuffer(expert_buffer)

    def _next_expert_batch(self) -> TensorDict | None:
        # Priority: sample from expert buffer if available
        if self._expert_buffer is not None:
            try:
                expert_batch = cast(TensorDict, self._expert_buffer.sample())
                # Ensure expert batch has the right batch size if specified
                assert isinstance(self.config, IPMDRLOptConfig)
                if self.config.ipmd.expert_batch_size is not None:
                    # If expert batch is larger than desired, take a subset
                    if expert_batch.numel() > self.config.ipmd.expert_batch_size:
                        expert_batch = cast(
                            TensorDict,
                            expert_batch[: self.config.ipmd.expert_batch_size],
                        )
                return expert_batch
            except Exception:
                pass

        # Fall back to iterator
        if self._expert_iterator is not None:
            try:
                expert_batch = cast(TensorDict, next(self._expert_iterator))
                # Ensure expert batch has the right batch size if specified
                assert isinstance(self.config, IPMDRLOptConfig)
                if self.config.ipmd.expert_batch_size is not None:
                    # If expert batch is larger than desired, take a subset
                    if expert_batch.numel() > self.config.ipmd.expert_batch_size:
                        expert_batch = cast(
                            TensorDict,
                            expert_batch[: self.config.ipmd.expert_batch_size],
                        )
                return expert_batch
            except StopIteration:
                return None

        return None

    def _compute_features(self, obs: Tensor) -> Tensor:
        """Compute phi(obs) - currently just returns the observation."""
        # For IPMD without feature extractor, just return obs
        return obs

    def _reward_from_batch(self, td: TensorDict) -> Tensor:
        """Compute estimated reward for a batch of transitions."""
        obs = cast(Tensor, td.get("observation"))
        act = cast(Tensor, td.get("action"))
        next_obs = cast(Tensor, td.get(("next", "observation")))

        phi_s = self._compute_features(obs)
        phi_next = self._compute_features(next_obs)

        assert isinstance(self.config, IPMDRLOptConfig)
        if self.config.ipmd.reward_detach_features:
            phi_s = phi_s.detach()
            phi_next = phi_next.detach()
            act = act.detach()

        x = torch.cat([phi_s, act, phi_next], dim=-1)
        return self.reward_estimator(x).squeeze(-1)

    def update(self, sampled_tensordict: TensorDict) -> TensorDict:
        # Validate input tensordict
        if not self._validate_tensordict(
            sampled_tensordict, "update:input", raise_error=False
        ):
            self.log.debug(
                "Discarding batch due to non-finite values in input tensordict"
            )
            # Return dummy loss with zeros
            return TensorDict(
                {
                    "loss_actor": torch.tensor(0.0, device=self.device),
                    "loss_qvalue": torch.tensor(0.0, device=self.device),
                    "loss_alpha": torch.tensor(0.0, device=self.device),
                    "loss_reward_diff": torch.tensor(0.0, device=self.device),
                    "loss_reward_l2": torch.tensor(0.0, device=self.device),
                    "estimated_reward_mean": torch.tensor(0.0, device=self.device),
                    "estimated_reward_std": torch.tensor(0.0, device=self.device),
                    "expert_reward_mean": torch.tensor(0.0, device=self.device),
                    "expert_reward_std": torch.tensor(0.0, device=self.device),
                },
                batch_size=[],
            )

        # Zero all grads
        self.optim.zero_grad(set_to_none=True)

        # 1) Compute estimated rewards for current policy data
        estimated_rewards = self._reward_from_batch(sampled_tensordict)

        # Use estimated rewards for SAC loss if configured to do so
        assert isinstance(self.config, IPMDRLOptConfig)
        if self.config.ipmd.use_estimated_rewards_for_sac:
            sampled_tensordict_for_sac = sampled_tensordict.clone()
            # Prevent SAC from backpropagating into the reward estimator unless explicitly allowed
            reward_for_sac = (
                estimated_rewards.detach()
                if self.config.ipmd.detach_reward_when_used_for_sac
                else estimated_rewards
            )
            sampled_tensordict_for_sac["reward"] = reward_for_sac
        else:
            sampled_tensordict_for_sac = sampled_tensordict

        # 2) SAC objective using estimated rewards (actor/critic/alpha)
        loss_td = cast(TensorDict, self.loss_module(sampled_tensordict_for_sac))
        loss_td = self._sanitize_loss_tensordict(loss_td, "update:raw_sac_loss")

        actor_loss = cast(Tensor, loss_td["loss_actor"])  # ensure Tensor type
        q_loss = cast(Tensor, loss_td["loss_qvalue"])  # ensure Tensor type
        alpha_loss = cast(Tensor, loss_td["loss_alpha"])  # ensure Tensor type
        total_sac_loss = (actor_loss + q_loss + alpha_loss).sum()
        total_sac_loss.backward()

        # 3) Reward estimator objective: IPMD Loss
        reward_diff = torch.zeros((), device=self.device)
        reward_l2 = torch.zeros((), device=self.device)
        expert_td = self._next_expert_batch()
        expert_rewards_cached: Tensor | None = None
        if expert_td is not None:
            # Validate expert batch
            if not self._validate_tensordict(
                expert_td, "update:expert_batch", raise_error=False
            ):
                logging.getLogger(__name__).warning(
                    "Expert batch contains non-finite values, skipping reward update"
                )
            else:
                # Recompute estimated rewards for current policy data
                # (need fresh computation graph after SAC backward)
                r_pi = self._reward_from_batch(sampled_tensordict)

                # Compute estimated rewards for expert data
                r_exp = self._reward_from_batch(expert_td.to(self.device))
                expert_rewards_cached = r_exp

                # IPMD Loss: difference between estimated returns
                # Current policy estimated return - Expert policy estimated return
                diff = r_pi.sum() - r_exp.sum()

                # L2 regularization for reward estimator
                l2 = torch.zeros((), device=self.device)
                for p in self.reward_estimator.parameters():
                    l2 = l2 + p.pow(2).sum()

                assert isinstance(self.config, IPMDRLOptConfig)
                total_reward_loss = (
                    float(self.config.ipmd.reward_loss_coeff) * diff
                    + float(self.config.ipmd.reward_l2_coeff) * l2
                )
                total_reward_loss.backward()
                reward_diff = diff.detach()
                reward_l2 = l2.detach()
        elif not self._warned_no_expert:
            logging.getLogger(__name__).warning(
                "Expert iterator not set; skipping reward estimator updates."
            )
            self._warned_no_expert = True

        # Validate gradients before optimizer step
        grads_finite = self._validate_gradients(
            "update:post_backward", raise_error=False
        )

        if not grads_finite:
            logging.getLogger(__name__).debug(
                "Skipping optimizer step due to non-finite gradients; batch discarded"
            )
            self.optim.zero_grad(set_to_none=True)
            # Return sanitized loss
            out = cast(TensorDict, loss_td.clone())
            out["loss_reward_diff"] = reward_diff
            out["loss_reward_l2"] = reward_l2
            return out

        # Step all parameter groups
        self.optim.step()
        self.optim.zero_grad(set_to_none=True)

        # Update target network
        self.target_net_updater.step()

        # Attach diagnostics
        out = cast(TensorDict, loss_td.clone())
        out["loss_reward_diff"] = reward_diff
        out["loss_reward_l2"] = reward_l2

        # Add additional diagnostics for IPMD
        # Recompute for diagnostics (no grad needed, just for logging)
        with torch.no_grad():
            diag_rewards = self._reward_from_batch(sampled_tensordict)
            out["estimated_reward_mean"] = diag_rewards.mean()
            out["estimated_reward_std"] = diag_rewards.std()

        if expert_td is not None:
            rewards_for_diag = (
                expert_rewards_cached
                if expert_rewards_cached is not None
                else self._reward_from_batch(expert_td.to(self.device))
            )
            out["expert_reward_mean"] = rewards_for_diag.mean().detach()
            out["expert_reward_std"] = rewards_for_diag.std().detach()
        else:
            out["expert_reward_mean"] = torch.tensor(0.0, device=self.device)
            out["expert_reward_std"] = torch.tensor(0.0, device=self.device)

        return out

    def train(self) -> None:  # type: ignore[override]
        assert isinstance(self.config, IPMDRLOptConfig)
        cfg = self.config
        frames_per_batch = cfg.collector.frames_per_batch
        total_frames = cfg.collector.total_frames
        utd_ratio = float(cfg.ipmd.utd_ratio)
        init_random_frames = int(self.config.collector.init_random_frames)
        num_updates = int(frames_per_batch * utd_ratio)

        # Compile the update function if requested
        compile_mode = None
        if cfg.compile.compile:
            compile_mode = cfg.compile.compile_mode
            if compile_mode in ("", None):
                if cfg.compile.cudagraphs:
                    compile_mode = "default"
                else:
                    compile_mode = "reduce-overhead"

            self.log.info(f"Compiling update function with mode: {compile_mode}")
            self.update = compile_with_warmup(self.update, mode=compile_mode, warmup=1)  # type: ignore[method-assign]

        # Only use CUDAGraphs on CUDA devices
        if cfg.compile.cudagraphs:
            if self.device.type == "cuda":
                warnings.warn(
                    "CudaGraphModule is experimental and may lead to silently wrong results. Use with caution.",
                    category=UserWarning,
                )
                self.log.warning("Wrapping update with CudaGraphModule (experimental)")
                self.update = CudaGraphModule(
                    self.update, in_keys=[], out_keys=[], warmup=5
                )  # type: ignore[method-assign]
            else:
                self.log.warning(
                    f"CUDAGraphs requested but device is {self.device.type}, not CUDA. Skipping CUDAGraphs."
                )

        collected_frames = 0
        collector_iter = iter(self.collector)
        pbar = tqdm.tqdm(total=total_frames)
        # Prioritized replay buffer is currently not supported/dormant

        while collected_frames < total_frames:
            with timeit("collect"):
                data = next(collector_iter)

            metrics_to_log: dict[str, Any] = {}
            frames_in_batch = data.numel()
            collected_frames += frames_in_batch
            pbar.update(frames_in_batch)

            self.collector.update_policy_weights_()

            # Get training rewards and episode lengths (if available)
            if ("next", "episode_reward") in data.keys(True):
                episode_rewards = data["next", "episode_reward"][data["next", "done"]]
                if len(episode_rewards) > 0:
                    episode_length = data["next", "step_count"][data["next", "done"]]
                    self.episode_lengths.extend(episode_length.cpu().tolist())
                    self.episode_rewards.extend(episode_rewards.cpu().tolist())
                    metrics_to_log.update(
                        {
                            "episode/length": float(np.mean(self.episode_lengths)),
                            "episode/return": float(np.mean(self.episode_rewards)),
                            "train/reward": float(
                                episode_rewards.float().mean().item()
                            ),
                        }
                    )

            # Don't empty the buffer in off-policy setting
            with timeit("replay_extend"):
                self.data_buffer.extend(data.reshape(-1))  # type: ignore[arg-type]

            # Initialize losses variable
            losses = TensorDict(batch_size=[num_updates])

            with timeit("train"):
                if collected_frames >= init_random_frames:
                    losses = TensorDict(batch_size=[num_updates])
                    for i in range(num_updates):
                        with timeit("rb - sample"):
                            # Sample from replay buffer
                            sampled_tensordict = self.data_buffer.sample()

                        # Validate sampled batch before update
                        if not self._validate_tensordict(
                            sampled_tensordict,
                            f"train:mini_batch{i}:pre_update",
                            raise_error=False,
                        ):
                            logging.getLogger(__name__).debug(
                                "Discarding mini-batch %d due to non-finite values", i
                            )
                            # Create dummy loss entry
                            losses[i] = TensorDict(
                                {
                                    "loss_actor": torch.tensor(0.0, device=self.device),
                                    "loss_qvalue": torch.tensor(
                                        0.0, device=self.device
                                    ),
                                    "loss_alpha": torch.tensor(0.0, device=self.device),
                                    "loss_reward_diff": torch.tensor(
                                        0.0, device=self.device
                                    ),
                                    "loss_reward_l2": torch.tensor(
                                        0.0, device=self.device
                                    ),
                                    "estimated_reward_mean": torch.tensor(
                                        0.0, device=self.device
                                    ),
                                    "estimated_reward_std": torch.tensor(
                                        0.0, device=self.device
                                    ),
                                    "expert_reward_mean": torch.tensor(
                                        0.0, device=self.device
                                    ),
                                    "expert_reward_std": torch.tensor(
                                        0.0, device=self.device
                                    ),
                                },
                                batch_size=[],
                            ).select(
                                "loss_actor",
                                "loss_qvalue",
                                "loss_alpha",
                                "loss_reward_diff",
                                "loss_reward_l2",
                                "estimated_reward_mean",
                                "estimated_reward_std",
                                "expert_reward_mean",
                                "expert_reward_std",
                            )
                            continue

                        with timeit("update"):
                            torch.compiler.cudagraph_mark_step_begin()
                            # Compute loss
                            loss = self.update(sampled_tensordict).clone()
                        losses[i] = loss.select(
                            "loss_actor",
                            "loss_qvalue",
                            "loss_alpha",
                            "loss_reward_diff",
                            "loss_reward_l2",
                            "estimated_reward_mean",
                            "estimated_reward_std",
                            "expert_reward_mean",
                            "expert_reward_std",
                        )

                    # PRB updates disabled

            # Get training losses and times
            if losses is not None:
                losses_mean = losses.apply(lambda x: x.float().mean(), batch_size=[])
                for key, value in list(losses_mean.items()):  # type: ignore
                    if isinstance(value, Tensor):
                        metrics_to_log[f"train/{key}"] = float(value.item())

            # Merge timing metrics and emit via shared logger interface
            metrics_to_log.update(timeit.todict(prefix="time"))  # type: ignore[arg-type]
            rate = pbar.format_dict.get("rate")
            if rate is not None:
                metrics_to_log["time/speed"] = rate
            if metrics_to_log:
                self.log_metrics(metrics_to_log, step=collected_frames)

            # for IsaacLab, we need to log the metrics from the environment
            if "Isaac" in self.config.env.env_name and hasattr(self.env, "log_infos"):
                log_info_dict: dict[str, Tensor] = self.env.log_infos.popleft()
                log_info(log_info_dict, metrics_to_log)

            # Save model periodically
            if (
                self.config.save_interval > 0
                and collected_frames % self.config.save_interval == 0
            ):
                self.save_model(
                    path=self.log_dir / self.config.logger.save_path,
                    step=collected_frames,
                )

        pbar.close()
        self.collector.shutdown()

    def validate_ipmd_loss(
        self, test_batch: TensorDict, expert_batch: TensorDict
    ) -> dict[str, float]:
        """Validate the IPMD loss computation with test data.

        Args:
            test_batch: Current policy batch for testing
            expert_batch: Expert demonstration batch for testing

        Returns:
            Dictionary containing loss components and diagnostics
        """
        # switch to eval mode for deterministic diagnostics
        for m in (self.actor_critic, self.reward_estimator):
            if hasattr(m, "eval"):
                m.eval()
        with torch.no_grad():
            # Compute estimated rewards
            estimated_rewards = self._reward_from_batch(test_batch)
            expert_rewards = self._reward_from_batch(expert_batch)

            # Compute IPMD loss components
            reward_diff = estimated_rewards.sum() - expert_rewards.sum()
            l2_reg = torch.zeros((), device=self.device)
            for p in self.reward_estimator.parameters():
                l2_reg = l2_reg + p.pow(2).sum()

            # Compute SAC loss with estimated rewards
            test_batch_with_estimated_rewards = test_batch.clone()
            test_batch_with_estimated_rewards.set("reward", estimated_rewards)
            sac_loss_td = self.loss_module(test_batch_with_estimated_rewards)

            return {
                "reward_diff": reward_diff.item(),
                "reward_l2": l2_reg.item(),
                "estimated_reward_mean": estimated_rewards.mean().item(),
                "estimated_reward_std": estimated_rewards.std().item(),
                "expert_reward_mean": expert_rewards.mean().item(),
                "expert_reward_std": expert_rewards.std().item(),
                "sac_loss_actor": sac_loss_td["loss_actor"].item(),
                "sac_loss_qvalue": sac_loss_td["loss_qvalue"].item(),
                "sac_loss_alpha": sac_loss_td["loss_alpha"].item(),
            }

    def predict(self, obs: Tensor | np.ndarray) -> Tensor:  # type: ignore[override]
        obs = torch.as_tensor([obs], device=self.device)
        policy_op = self.actor_critic.get_policy_operator()
        policy_op.eval()
        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
            td = TensorDict(
                dict.fromkeys(self.total_input_keys, obs),
                batch_size=[1],
                device=self.device,
            )
            td = policy_op(td)
            return td.get("action")
