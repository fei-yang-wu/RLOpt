from __future__ import annotations

import functools
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional, cast

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
        self._expert_iterator: Optional[Iterator[TensorDict]] = None
        self._expert_buffer: Optional[TensorDictReplayBuffer] = None
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

        # Define policy architecture
        if policy_net is None:
            # Use NetworkLayout if provided, otherwise fall back to legacy config
            if self.config.network and self.config.network.policy:
                policy_config = self.config.network.policy
                if policy_config.head:
                    num_cells = list(policy_config.head.num_cells)
                    activation_class = self._get_activation_class(
                        policy_config.head.activation
                    )
                else:
                    # Fallback to legacy config
                    num_cells = self.config.policy.num_cells
                    activation_class = torch.nn.ELU
            else:
                # Fallback to legacy config
                num_cells = self.config.policy.num_cells
                activation_class = torch.nn.ELU

            policy_mlp = MLP(
                in_features=self.policy_input_shape,
                activation_class=activation_class,
                out_features=self.policy_output_shape,
                num_cells=num_cells,
                device=self.device,
            )
            # Initialize policy weights
            if (
                self.config.network
                and self.config.network.policy
                and self.config.network.policy.head
            ):
                self._initialize_weights(
                    policy_mlp, self.config.network.policy.head.init
                )
            else:
                self._initialize_weights(policy_mlp, "orthogonal")
        else:
            policy_mlp = policy_net

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
        buffer_size = cfg.collector.frames_per_batch
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
            logging.getLogger(__name__).warning(
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

        # Removed True parameter to match ppo_mujoco.py
        return TensorDictReplayBuffer(
            storage=LazyTensorStorage(
                cfg.collector.frames_per_batch,
                compilable=cfg.compile.compile,  # type: ignore
                device=self.device,
            ),
            sampler=sampler,
            batch_size=cfg.loss.mini_batch_size,
            compilable=cfg.compile.compile,
        )

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

    def _next_expert_batch(self) -> Optional[TensorDict]:
        # Priority: sample from expert buffer if available
        if self._expert_buffer is not None:
            try:
                return self._expert_buffer.sample()
            except Exception:
                pass

        # Fall back to iterator
        if self._expert_iterator is not None:
            try:
                return next(self._expert_iterator)
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
        obs = td.get(obs_key)
        act = td.get("action")
        next_obs = td.get(("next", obs_key))

        phi_s = self._compute_features(obs)
        phi_next = self._compute_features(next_obs)

        if self.config.ipmd.reward_detach_features:
            phi_s = phi_s.detach()
            phi_next = phi_next.detach()
            act = act.detach()

        x = torch.cat([phi_s, act, phi_next], dim=-1)
        r = self.reward_estimator(x).squeeze(-1)
        return r

    def update(self, sampled_tensordict: TensorDict) -> TensorDict:
        # Zero all grads
        self.optim.zero_grad(set_to_none=True)

        # 1) SAC objective (actor/critic/alpha)
        loss_td = self.loss_module(sampled_tensordict)
        sac_total = (
            loss_td["loss_actor"] + loss_td["loss_qvalue"] + loss_td["loss_alpha"]
        ).sum()
        sac_total.backward()

        # 2) Reward estimator objective
        reward_diff = torch.zeros((), device=self.device)
        reward_l2 = torch.zeros((), device=self.device)
        expert_td = self._next_expert_batch()
        if expert_td is not None:
            r_pi = self._reward_from_batch(sampled_tensordict)
            r_exp = self._reward_from_batch(expert_td.to(self.device))
            # Use sum as MC surrogate per spec
            diff = r_pi.sum() - r_exp.sum()
            l2 = sum((p.pow(2).sum() for p in self.reward_estimator.parameters()))
            total_reward_loss = (
                float(self.config.ipmd.reward_loss_coeff) * diff
                + float(self.config.ipmd.reward_l2_coeff) * l2
            )
            total_reward_loss.backward()
            reward_diff = diff.detach()
            reward_l2 = l2.detach()
        else:
            if not self._warned_no_expert:
                logging.getLogger(__name__).warning(
                    "Expert iterator not set; skipping reward estimator updates."
                )
                self._warned_no_expert = True

        # Step all parameter groups
        self.optim.step()
        self.optim.zero_grad(set_to_none=True)

        # Update target network
        self.target_net_updater.step()

        # Attach diagnostics
        out = loss_td.detach()
        out.set("loss_reward_diff", reward_diff)
        out.set("loss_reward_l2", reward_l2)
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
                        "episode/length": np.mean(self.episode_lengths),
                        "episode/return": np.mean(self.episode_rewards),
                        "train/reward": episode_rewards.mean().item(),
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
                        )

                    # PRB updates disabled

            # Get training losses and times
            if losses is not None:
                losses_mean = losses.apply(lambda x: x.float().mean(), batch_size=[])
                for key, value in losses_mean.items():  # type: ignore
                    metrics_to_log.update({f"train/{key}": value.item()})

            if self.logger is not None and metrics_to_log:
                for k, v in metrics_to_log.items():
                    if isinstance(v, Tensor):
                        self.logger.log_scalar(k, float(v.item()), collected_frames)
                    else:
                        self.logger.log_scalar(k, v, collected_frames)

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
                    path=Path(self.config.logger.log_dir)
                    / self.config.logger.save_path,
                    step=collected_frames,
                )

        pbar.close()
        self.collector.shutdown()

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
