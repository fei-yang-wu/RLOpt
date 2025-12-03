from __future__ import annotations

import functools
import logging
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import torch
from tensordict import TensorDict
from tensordict.nn import CudaGraphModule, InteractionType, TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import Tensor, nn
from torchrl._utils import compile_with_warmup, timeit
from torchrl.data import (
    LazyMemmapStorage,
    LazyTensorStorage,
    ReplayBuffer,
    TensorDictReplayBuffer,
)
from torchrl.data.replay_buffers.samplers import RandomSampler
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import (
    MLP,
    ActorCriticOperator,
    ProbabilisticActor,
    TanhNormal,
    ValueOperator,
)
from torchrl.objectives import SoftUpdate
from torchrl.objectives.sac import SACLoss
from torchrl.record.loggers import Logger
from tqdm.std import tqdm as Tqdm

from rlopt.base_class import BaseAlgorithm
from rlopt.configs import RLOptConfig
from rlopt.type_aliases import OptimizerClass
from rlopt.utils import get_activation_class, log_info


@dataclass
class SACConfig:
    """SAC-specific configuration."""

    alpha_init: float = 1.0  # Match TorchRL default
    """Initial alpha value."""

    min_alpha: float | None = 1e-4
    """Minimum alpha value."""

    max_alpha: float | None = 10.0
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


@dataclass
class SACRLOptConfig(RLOptConfig):
    """SAC configuration that extends RLOptConfig."""

    sac: SACConfig = field(default_factory=SACConfig)
    """SAC configuration."""


class SAC(BaseAlgorithm):
    """Soft Actor-Critic algorithm.

    The class mirrors the PPO structure (custom train loop) while adapting
    to SAC's off-policy setting with a replay buffer.
    """

    def __init__(
        self,
        env,
        config: SACRLOptConfig,
        logger: Logger | None = None,
        **kwargs,
    ):
        super().__init__(
            env=env,
            config=config,
            logger=logger,
            **kwargs,
        )

        # Narrow the type for static checkers
        self.config = cast(SACRLOptConfig, self.config)
        self.config: SACRLOptConfig

        assert self.q_function, "SAC requires a Q-function configuration."

        # Compile if requested
        self._compile_components()

        # Initialize total network updates
        self.total_network_updates = 0

        # construct the target net updater
        self.target_net_updater = self._construct_target_net_updater()

    def _construct_policy(self) -> TensorDictModule:
        """Construct policy network."""
        policy_mlp = MLP(
            in_features=self.config.policy.input_dim,
            activation_class=get_activation_class(self.config.policy.activation_fn),
            out_features=2 * self.env.action_spec_unbatched.shape[-1],  # type: ignore
            num_cells=list(self.config.policy.num_cells),
            device=self.device,
        )
        extractor = NormalParamExtractor(
            scale_mapping="biased_softplus_1.0",
            scale_lb=0.1,  # type: ignore
        ).to(self.device)
        net = torch.nn.Sequential(policy_mlp, extractor)
        # SAC policy outputs loc and scale for the Normal distribution
        policy_td = TensorDictModule(
            module=net,
            in_keys=list(self.config.policy.input_keys),
            out_keys=["loc", "scale"],
        )
        distribution_class = TanhNormal
        distribution_kwargs = {
            "low": self.env.action_spec_unbatched.space.low,  # type: ignore
            "high": self.env.action_spec_unbatched.space.high,  # type: ignore
            "tanh_loc": False,
        }
        return ProbabilisticActor(
            policy_td,
            in_keys=["loc", "scale"],
            spec=self.env.full_action_spec_unbatched.to(self.device),  # type: ignore
            distribution_class=distribution_class,
            distribution_kwargs=distribution_kwargs,
            return_log_prob=False,
            default_interaction_type=ExplorationType.RANDOM,
        )

    @property
    def collector_policy(self) -> TensorDictModule:
        """By default, the collector_policy is self.policy or self.actor_critic.policy_operator()"""
        return self.actor_critic.get_policy_operator()  # type: ignore[override]

    def _construct_q_function(self) -> TensorDictModule:
        if self.config.q_function is None:
            msg = "SAC requires a Q-function configuration."
            raise ValueError(msg)

        q_function = MLP(
            activation_class=get_activation_class(self.config.q_function.activation_fn),
            num_cells=list(self.config.q_function.num_cells),
            out_features=1,
            device=self.device,
        )

        # SAC Q-function takes both observation and action as inputs
        in_keys = list(self.config.q_function.input_keys)
        if "action" not in in_keys:
            in_keys.append("action")

        return ValueOperator(
            module=q_function,
            in_keys=in_keys,
            out_keys=["state_action_value"],
        )

    def _construct_actor_critic(self) -> TensorDictModule:
        # Pass a dummy common_operator as required by ActorCriticOperator

        if self.q_function is None or self.policy is None:
            msg = "SAC requires a Q-function and policy configuration."
            raise ValueError(msg)

        # use the feature extractor if available
        if self.feature_extractor:
            return ActorCriticOperator(
                common_operator=self.feature_extractor,
                policy_operator=self.policy,
                value_operator=self.q_function,
            )

        class IdentityModule(torch.nn.Module):
            def forward(self, x):
                return x

        dummy = TensorDictModule(
            module=IdentityModule(),
            in_keys=["observation"],
            out_keys=["observation"],
        )
        return ActorCriticOperator(
            common_operator=dummy,
            policy_operator=self.policy,
            value_operator=self.q_function,
        )

    def _construct_loss_module(self) -> nn.Module:
        sac_cfg = getattr(self.config, "sac", None)
        if sac_cfg is None:
            msg = "SAC config missing 'sac' attribute"
            raise AttributeError(msg)
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

    def _set_optimizers(
        self, optimizer_cls: OptimizerClass, optimizer_kwargs: dict[str, Any]
    ) -> list[torch.optim.Optimizer]:
        """Create optimizers for actor, critic, and alpha parameters."""
        critic_params = list(
            self.loss_module.qvalue_network_params.flatten_keys().values()  # type: ignore[attr-defined]
        )
        actor_params = list(
            self.loss_module.actor_network_params.flatten_keys().values()  # type: ignore[attr-defined]
        )
        optimizers = [
            optimizer_cls(actor_params, **optimizer_kwargs),
            optimizer_cls(critic_params, **optimizer_kwargs),
        ]
        # Alpha optimizer for entropy temperature
        if hasattr(self.loss_module, "log_alpha"):
            param = self.loss_module.log_alpha
            if isinstance(param, torch.nn.Parameter):
                optimizers.append(torch.optim.Adam([param], lr=3.0e-4))
        return optimizers

    def _construct_target_net_updater(self) -> SoftUpdate:
        # Prefer polyak parameter from network layout critic if present
        eps = self.config.optim.target_update_polyak
        return SoftUpdate(self.loss_module, eps=eps)

    def _get_additional_optimizers(
        self, optimizer_cls: OptimizerClass, optimizer_kwargs: dict[str, Any]
    ) -> list[torch.optim.Optimizer]:
        """Get additional optimizers for SAC-specific components."""
        additional_optimizers = []

        # Alpha optimizer for SAC
        if hasattr(self, "loss_module") and hasattr(self.loss_module, "log_alpha"):
            alpha_optim = optimizer_cls(
                [self.loss_module.log_alpha],
                **optimizer_kwargs,
            )
            additional_optimizers.append(alpha_optim)

        return additional_optimizers

    def _construct_data_buffer(self) -> ReplayBuffer:
        cfg = self.config
        sampler = RandomSampler()
        scratch_dir = cfg.replay_buffer.scratch_dir or cfg.collector.scratch_dir
        device = cfg.device
        buffer_size = cfg.replay_buffer.size
        batch_size = cfg.loss.mini_batch_size
        shared = cfg.collector.shared
        prefetch = cfg.replay_buffer.prefetch
        storage_cls = (
            functools.partial(LazyTensorStorage, device=device)
            if not scratch_dir
            else functools.partial(
                LazyMemmapStorage, device="cpu", scratch_dir=scratch_dir
            )
        )
        if getattr(cfg.replay_buffer, "prb", False):
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

    def _construct_feature_extractor(self) -> TensorDictModule:
        """Construct feature extractor (optional for SAC)."""
        msg = "SAC does not require a feature extractor by default."
        raise NotImplementedError(msg)

    def _construct_trainer(self):  # type: ignore[override]
        return None

    def update(self, sampled_tensordict: TensorDict) -> TensorDict:
        # Compute loss
        loss_td = self.loss_module(sampled_tensordict)
        loss_td = self._sanitize_loss_tensordict(loss_td, "update:raw_loss")

        actor_loss = loss_td["loss_actor"]
        q_loss = loss_td["loss_qvalue"]
        alpha_loss = loss_td["loss_alpha"]

        total_loss = (actor_loss + q_loss + alpha_loss).sum()  # type: ignore[operator]

        # Backward pass
        total_loss.backward()  # type: ignore[operator]

        # Update networks
        self.optim.step()

        # Zero gradients
        self.optim.zero_grad(set_to_none=True)

        # Update qnet_target params
        self.target_net_updater.step()

        return loss_td.detach_()

    def train(self) -> None:  # type: ignore[override]
        assert isinstance(self.config, SACRLOptConfig)
        cfg = self.config
        frames_per_batch = cfg.collector.frames_per_batch
        num_envs = cfg.env.num_envs
        total_frames = cfg.collector.total_frames
        utd_ratio = float(cfg.sac.utd_ratio)
        init_random_frames = int(self.config.collector.init_random_frames)
        # Compute updates per collector iteration based on UTD ratio and batch sizes
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
        pbar: Tqdm = Tqdm(total=total_frames)  # type: ignore[assignment]

        while collected_frames < total_frames:
            timeit.printevery(num_prints=1000, total_count=total_frames, erase=True)

            with timeit("collect"):
                data = next(collector_iter)

            metrics_to_log: dict[str, Any] = {}
            frames_in_batch = data.numel()
            collected_frames += frames_in_batch
            pbar.update(frames_in_batch)

            self.collector.update_policy_weights_()

            # Log step rewards (always available)
            if ("next", "reward") in data.keys(True):
                step_rewards = data["next", "reward"]
                metrics_to_log["train/step_reward_mean"] = step_rewards.mean().item()
                metrics_to_log["train/step_reward_std"] = step_rewards.std().item()
                metrics_to_log["train/step_reward_max"] = step_rewards.max().item()
                metrics_to_log["train/step_reward_min"] = step_rewards.min().item()

            # Get training rewards and episode lengths (if available)
            if ("next", "episode_reward") in data.keys(True):
                episode_rewards = data["next", "episode_reward"][data["next", "done"]]
                if len(episode_rewards) > 0:
                    episode_length = data["next", "step_count"][data["next", "done"]]
                    self.episode_lengths.extend(episode_length.cpu().tolist())
                    self.episode_rewards.extend(episode_rewards.cpu().tolist())
                    episode_rewards_mean = np.mean(episode_rewards.cpu().tolist())

                    metrics_to_log.update(
                        {
                            "episode/length": np.mean(self.episode_lengths),
                            "episode/return": np.mean(self.episode_rewards),
                            "episode/return_latest": episode_rewards_mean,
                            "episode/num_completed": len(self.episode_rewards),
                        }
                    )

            # Don't empty the buffer in off-policy setting
            with timeit("replay_extend"):
                self.data_buffer.extend(data.reshape(-1))  # type: ignore[arg-type]

            losses = None
            with timeit("train"):
                if collected_frames >= init_random_frames:
                    losses_list = []
                    for _i in range(num_updates):
                        with timeit("rb - sample"):
                            # Sample from replay buffer
                            sampled_tensordict = self.data_buffer.sample()

                        with timeit("update"):
                            torch.compiler.cudagraph_mark_step_begin()
                            # Compute loss
                            loss = self.update(sampled_tensordict).clone()
                        losses_list.append(
                            loss.select("loss_actor", "loss_qvalue", "loss_alpha")
                        )

                    # Stack collected losses into a TensorDict
                    if len(losses_list) > 0:
                        # gather keys and stack per-key tensors along new batch dim
                        keys = list(losses_list[0].keys())
                        stacked = {}
                        for key in keys:
                            stacked[key] = torch.stack(
                                [ld.get(key) for ld in losses_list]
                            )
                        losses = TensorDict(stacked, batch_size=[num_updates])

                    # PRB updates disabled

            # Get training losses and times
            if losses is not None:
                losses_mean = losses.apply(lambda x: x.float().mean(), batch_size=[])
                for key, value in losses_mean.items():  # type: ignore
                    try:
                        scalar = float(value)  # type: ignore[arg-type]
                    except Exception:
                        scalar = value.detach().cpu().float().item()  # type: ignore[attr-defined]
                    metrics_to_log.update({f"train/{key}": scalar})

                # Log SAC-specific metrics
                if hasattr(self.loss_module, "log_alpha"):
                    alpha = self.loss_module.log_alpha.exp().detach().cpu().item()
                    metrics_to_log["train/alpha"] = alpha

            # for IsaacLab, we need to log the metrics from the environment
            if "Isaac" in self.config.env.env_name and hasattr(self.env, "log_infos"):
                log_info_dict: dict[str, Tensor] = self.env.log_infos.popleft()
                log_info(log_info_dict, metrics_to_log)

            metrics_to_log.update(timeit.todict(prefix="time"))  # type: ignore[arg-type]
            rate = pbar.format_dict.get("rate")
            if rate is not None:
                metrics_to_log["time/speed"] = rate

            if metrics_to_log:
                # Use the shared base-class helper for consistent metrics emission
                self.log_metrics(metrics_to_log, step=collected_frames)

                # Update progress bar with latest metrics
                postfix = {}
                if "train/step_reward_mean" in metrics_to_log:
                    postfix["r_step"] = (
                        f"{metrics_to_log['train/step_reward_mean']:.2f}"
                    )
                if "episode/return" in metrics_to_log:
                    postfix["r_ep"] = f"{metrics_to_log['episode/return']:.1f}"
                    postfix["n_ep"] = metrics_to_log["episode/num_completed"]
                if "train/loss_actor" in metrics_to_log:
                    postfix["π_loss"] = f"{metrics_to_log['train/loss_actor']:.3f}"
                if "train/alpha" in metrics_to_log:
                    postfix["α"] = f"{metrics_to_log['train/alpha']:.3f}"
                if postfix:
                    pbar.set_postfix(postfix)

            # Save model periodically
            if (
                self.config.save_interval > 0
                and collected_frames % (self.config.save_interval * num_envs) == 0
            ):
                self.save_model(
                    path=self.log_dir / self.config.logger.save_path,
                    step=collected_frames,
                )

        pbar.close()
        self.collector.shutdown()

    def predict(self, td: TensorDict) -> Tensor:  # type: ignore[override]
        policy_op = self.actor_critic.get_policy_operator()
        policy_op.eval()
        with torch.no_grad(), set_exploration_type(InteractionType.DETERMINISTIC):
            td = policy_op(td)
            return td.get("action")
