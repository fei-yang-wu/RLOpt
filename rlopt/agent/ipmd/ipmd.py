from __future__ import annotations

import functools
import logging
from dataclasses import dataclass, field
from typing import Any, cast
from collections.abc import Iterator

import numpy as np
import torch
import tqdm
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch import Tensor, nn
from torchrl._utils import timeit
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
    ActorValueOperator,
    ProbabilisticActor,
    TanhNormal,
)
from torchrl.objectives import SoftUpdate, group_optimizers
from torchrl.objectives.sac import SACLoss
from torchrl.record.loggers import Logger

from rlopt.base_class import BaseAlgorithm
from rlopt.configs import RLOptConfig
from rlopt.utils import log_info


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

    def _post_init(self):
        self.use_value_function = False


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
        self._expert_buffer: TensorDictReplayBuffer | None = None
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
        """Build shared feature extractor via base helper."""
        return self._build_feature_extractor_module(
            feature_extractor_net=feature_extractor_net,
            in_keys=list(self.total_input_keys),
            out_key="hidden",
            layout=self.config.network,
        )

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

        # Build policy head via base helper and wrap
        policy_td = self._build_policy_head_module(
            policy_net=policy_net,
            in_keys=list(self.config.policy_in_keys),
            out_keys=["loc", "scale"],
            layout=self.config.network,
        )
        return ProbabilisticActor(
            policy_td,
            in_keys=["loc", "scale"],
            spec=self.env.full_action_spec_unbatched.to(self.device),  # type: ignore
            distribution_class=distribution_class,
            distribution_kwargs=distribution_kwargs,
            return_log_prob=True,
            default_interaction_type=ExplorationType.RANDOM,
        )

    def _construct_value_function(
        self, value_net: nn.Module | None = None
    ) -> TensorDictModule:  # type: ignore[override]
        """IPMD does not use a state-value function explicitly.
        We return a minimal dummy ValueOperator (not used in loss) to satisfy
        ActorValueOperator structure.
        """

    def _construct_q_function(self, q_net: nn.Module | None = None) -> TensorDictModule:
        # Delegate to base helper
        return self._build_qvalue_module(q_net=q_net, layout=self.config.network)

    def _construct_reward_estimator(self) -> nn.Module:
        """Create reward network mapping [phi(s), a, phi(s')] to scalar."""
        cfg = self.config
        assert isinstance(cfg, IPMDRLOptConfig)
        if cfg.use_feature_extractor:
            feat_dim = int(cfg.feature_extractor.output_dim)
        else:
            feat_dim = int(self.total_input_shape)
        act_dim = int(self.policy_output_shape)
        in_dim = feat_dim * 2 + act_dim

        net = MLP(
            in_features=in_dim,
            out_features=1,
            num_cells=list(cfg.ipmd.reward_num_cells),
            activation_class=self._get_activation_class(cfg.ipmd.reward_activation),
            device=self.device,
        )
        self._initialize_weights(net, cfg.ipmd.reward_init)
        return net

    def _construct_actor_critic(self) -> TensorDictModule:
        assert isinstance(
            self.q_function, TensorDictModule
        ), "Q-function must be a TensorDictModule"
        assert isinstance(
            self.policy, TensorDictModule
        ), "Policy must be a TensorDictModule"

        return ActorValueOperator(
            common_operator=self.feature_extractor,
            policy_operator=self.policy,
            value_operator=self.q_function,  # dummy / unused in IPMD
        )

    def _construct_loss_module(self) -> nn.Module:
        assert isinstance(self.config, IPMDRLOptConfig)
        sac_cfg = self.config.ipmd
        # Read num critics from network layout if provided
        if self.config.network and self.config.network.critic:
            sac_cfg.num_qvalue_nets = self.config.network.critic.num_nets

        loss_module = SACLoss(
            actor_network=self.actor_critic.get_policy_operator(),
            qvalue_network=self.actor_critic.get_value_operator(),
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
        if self.config.network and self.config.network.critic:
            eps = self.config.network.critic.polyak_eps
        return SoftUpdate(self.loss_module, eps=eps)

    # _get_activation_class and _initialize_weights now provided by BaseAlgorithm

    def _configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizers"""
        actor_optim = torch.optim.AdamW(
            self.actor_critic.get_policy_head().parameters(),
            lr=torch.tensor(self.config.optim.lr, device=self.device),
            # eps=1e-5,
        )
        critic_optim = torch.optim.AdamW(
            self.actor_critic.get_value_head().parameters(),
            lr=torch.tensor(self.config.optim.lr, device=self.device),
            # eps=1e-5,
        )
        alpha_optim = torch.optim.AdamW(
            [self.loss_module.log_alpha],
            lr=torch.tensor(self.config.optim.lr, device=self.device),
        )
        reward_optim = torch.optim.AdamW(
            self.reward_estimator.parameters(),
            lr=torch.tensor(self.config.optim.lr, device=self.device),
        )
        if self.config.use_feature_extractor:
            feature_optim = torch.optim.AdamW(
                self.feature_extractor.parameters(),
                lr=torch.tensor(self.config.optim.lr, device=self.device),
                # eps=1e-5,
            )
            return group_optimizers(
                actor_optim, critic_optim, feature_optim, alpha_optim, reward_optim
            )

        return group_optimizers(actor_optim, critic_optim, alpha_optim, reward_optim)

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

    def set_expert_buffer(self, buffer: TensorDictReplayBuffer) -> None:
        """Attach an expert replay buffer.

        This can be used to avoid Python-level iteration overhead and to
        leverage TorchRL's device-aware sampling. The buffer is expected to
        contain the keys: 'observation', 'action', ('next', 'observation').
        """
        self._expert_buffer = buffer

    def set_expert_source(self, source: Any) -> None:
        """Attach an expert data source.

        Accepts either:
        - A TensorDict replay buffer object exposing ``sample()`` (e.g., TorchRL's
          ``TensorDictReplayBuffer``), or
        - An ImitationLearningTools replay manager exposing a ``buffer`` attribute
          whose ``sample()`` method returns a TensorDict (see ILTools tests for usage).

        Example with ILTools:
            mgr = build_replay_from_zarr(...)
            ipmd.set_expert_source(mgr)  # same as ipmd.set_expert_buffer(mgr.buffer)
        """
        # Direct replay buffer with sample()
        if hasattr(source, "sample") and callable(getattr(source, "sample")):
            self._expert_buffer = source  # type: ignore[assignment]
            return
        # Manager-like object exposing a `.buffer.sample()` API
        buf = getattr(source, "buffer", None)
        if (
            buf is not None
            and hasattr(buf, "sample")
            and callable(getattr(buf, "sample"))
        ):
            self._expert_buffer = buf  # type: ignore[assignment]
            return
        msg = (
            "Unsupported expert source: expected an object with 'sample()' or a 'buffer'"
            " exposing 'sample()'."
        )
        raise TypeError(msg)

    def create_expert_buffer(
        self, expert_data: TensorDict, buffer_size: int | None = None
    ) -> TensorDictReplayBuffer:
        """Create an expert replay buffer from expert demonstration data.

        Args:
            expert_data: TensorDict containing expert demonstrations with keys:
                - 'observation' (s_t)
                - 'action' (a_t)
                - ('next', 'observation') (s_{t+1})
            buffer_size: Maximum buffer size. If None, uses the size of expert_data.

        Returns:
            TensorDictReplayBuffer containing expert data
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

        return expert_buffer

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
        """Compute phi(obs) using the shared feature extractor or identity."""
        if self.config.use_feature_extractor:
            module = getattr(self.feature_extractor, "module", None)
            if module is None:
                td = TensorDict({self.total_input_keys[0]: obs}, obs.shape[:-1])
                return self.feature_extractor(td).get("hidden")
            return module(obs)
        return obs

    def _reward_from_batch(self, td: TensorDict) -> Tensor:
        obs_key = self.total_input_keys[0]
        obs = cast(Tensor, td.get(obs_key))
        act = cast(Tensor, td.get("action"))
        next_obs = cast(Tensor, td.get(("next", obs_key)))

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
                # Compute estimated rewards for current policy data
                r_pi = estimated_rewards

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
        out["estimated_reward_mean"] = estimated_rewards.mean().detach()
        out["estimated_reward_std"] = estimated_rewards.std().detach()
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

            # Get training rewards and episode lengths
            episode_rewards = data["next", "episode_reward"][data["next", "done"]]
            if len(episode_rewards) > 0:
                episode_length = data["next", "step_count"][data["next", "done"]]
                self.episode_lengths.extend(episode_length.cpu().tolist())
                self.episode_rewards.extend(episode_rewards.cpu().tolist())
                metrics_to_log.update(
                    {
                        "episode/length": float(np.mean(self.episode_lengths)),
                        "episode/return": float(np.mean(self.episode_rewards)),
                        "train/reward": float(episode_rewards.float().mean().item()),
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
