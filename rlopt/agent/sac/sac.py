from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import torch
from tensordict import TensorDict
from tensordict.nn import (
    CudaGraphModule,
    InteractionType,
    TensorDictModule,
)
from torch import Tensor, nn
from torchrl._utils import compile_with_warmup, timeit
from torchrl.data import (
    Bounded,
    LazyMemmapStorage,
    LazyTensorStorage,
    ReplayBuffer,
    TensorDictPrioritizedReplayBuffer,
    TensorDictReplayBuffer,
)
from torchrl.data.replay_buffers.samplers import RandomSampler
from torchrl.envs.utils import set_exploration_type
from torchrl.modules import (
    MLP,
    ActorCriticOperator,
    IndependentNormal,
    ProbabilisticActor,
    TanhNormal,
    ValueOperator,
)
from torchrl.objectives import SoftUpdate
from torchrl.objectives.sac import SACLoss
from torchrl.objectives.utils import ValueEstimators
from torchrl.record.loggers import Logger
from tqdm.std import tqdm as Tqdm

from rlopt.base_class import BaseAlgorithm
from rlopt.config_base import NetworkConfig, RLOptConfig
from rlopt.models import GaussianPolicyHead
from rlopt.type_aliases import OptimizerClass
from rlopt.utils import get_activation_class, log_info


@dataclass
class SACConfig:
    """SAC-specific configuration."""

    alpha_init: float = 1.0
    """Initial alpha value."""

    min_alpha: float | None = 1e-4
    """Minimum alpha value."""

    max_alpha: float | None = 10
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

    separate_losses: bool = True
    """Whether to separate losses. Must be True for sequential updates."""

    reduction: str = "mean"
    """Reduction."""

    skip_done_states: bool = False
    """Whether to skip done states."""

    deactivate_vmap: bool = False
    """Whether to deactivate vmap."""

    utd_ratio: float = 1.0
    """Number of updates per batch."""

    # Policy log_std parameters (same as PPO for consistency)
    log_std_init: float = 0.0
    """Initial value for the policy log standard deviation (log std)."""

    clip_log_std: bool = False
    """Whether to clip the learned log standard deviation."""

    log_std_min: float = -20.0
    """Minimum log standard deviation (when clipping is enabled)."""

    log_std_max: float = 2.0
    """Maximum log standard deviation (when clipping is enabled)."""


@dataclass
class SACRLOptConfig(RLOptConfig):
    """SAC configuration that extends RLOptConfig."""

    sac: SACConfig = field(default_factory=SACConfig)
    """SAC configuration."""

    def __post_init__(self):
        """Post-initialization setup."""
        assert self.value_function is None, "SAC does not use a value function."

        self.q_function = NetworkConfig(
            num_cells=[256, 128, 128],
            activation_fn="elu",
            output_dim=1,
            input_keys=["observation"],
        )


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

        if self.compile_mode:
            self.update = compile_with_warmup(
                self.update, mode=self.compile_mode, warmup=1
            )

        if self.config.compile.cudagraphs:
            # self.logger.warn(
            #     "CudaGraphModule is experimental and may lead to silently wrong results. Use with caution.",
            #     category=UserWarning,
            # )
            self.update = CudaGraphModule(
                self.update, in_keys=[], out_keys=[], warmup=5
            )

    def _construct_policy(self) -> TensorDictModule:
        """Construct policy network with learned log-std (same as PPO)."""
        action_spec = self.env.action_spec_unbatched  # type: ignore
        action_dim = int(action_spec.shape[-1])

        # Build the base MLP that outputs action means
        policy_mlp = MLP(
            in_features=self.config.policy.input_dim,
            activation_class=get_activation_class(self.config.policy.activation_fn),
            out_features=action_dim,
            num_cells=list(self.config.policy.num_cells),
            device=self.device,
        )

        # Wrap with GaussianPolicyHead for learned log-std (consistent with PPO)
        sac_cfg = self.config.sac
        net = GaussianPolicyHead(
            base=policy_mlp,
            action_dim=action_dim,
            log_std_init=sac_cfg.log_std_init,
            log_std_min=sac_cfg.log_std_min,
            log_std_max=sac_cfg.log_std_max,
            clip_log_std=sac_cfg.clip_log_std,
            device=self.device,
        )

        # Wrap in TensorDictModule
        policy_td = TensorDictModule(
            module=net,
            in_keys=list(self.config.policy.input_keys),
            out_keys=["loc", "scale"],
        )

        # Configure distribution based on action space bounds
        if isinstance(action_spec, Bounded):
            distribution_class = TanhNormal
            distribution_kwargs = {
                "low": action_spec.space.low,  # type: ignore[union-attr]
                "high": action_spec.space.high,  # type: ignore[union-attr]
                "tanh_loc": False,
            }
        else:
            distribution_class = IndependentNormal
            distribution_kwargs = {}

        return ProbabilisticActor(
            policy_td,
            in_keys=["loc", "scale"],
            spec=self.env.full_action_spec_unbatched.to(self.device),  # type: ignore
            distribution_class=distribution_class,
            distribution_kwargs=distribution_kwargs,
            return_log_prob=False,
            default_interaction_type=InteractionType.RANDOM,
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
            in_features=self.config.q_function.input_dim,
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
            in_keys=list(self.config.policy.input_keys),
            out_keys=list(self.config.policy.input_keys),
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
            actor_network=self.actor_critic.get_policy_head(),  # type: ignore[arg-type]
            qvalue_network=self.actor_critic.get_value_head(),  # type: ignore[arg-type]
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
        loss_module.make_value_estimator(
            value_type=ValueEstimators.TD1,
            gamma=self.config.loss.gamma,
            # lmbda=0.95,
            average_rewards=False,
        )
        return loss_module

    def _set_optimizers(
        self, optimizer_cls: OptimizerClass, optimizer_kwargs: dict[str, Any]
    ) -> list[torch.optim.Optimizer]:
        """Create optimizers for actor, critic, and alpha parameters."""
        if hasattr(self.loss_module, "qvalue_network_params"):
            critic_params = list(
                self.loss_module.qvalue_network_params.flatten_keys().values()  # type: ignore[attr-defined]
            )
        else:
            critic_params = list(self.loss_module.qvalue_networks.parameters())  # type: ignore[attr-defined]
        if hasattr(self.loss_module, "actor_network_params"):
            actor_params = list(
                self.loss_module.actor_network_params.flatten_keys().values()  # type: ignore[attr-defined]
            )
        else:
            actor_params = list(self.loss_module.actor_network.parameters())  # type: ignore[attr-defined]
        optimizers = [
            optimizer_cls(actor_params, **optimizer_kwargs),
            optimizer_cls(critic_params, **optimizer_kwargs),
        ]
        # Alpha optimizer for entropy temperature
        if hasattr(self.loss_module, "log_alpha"):
            param = self.loss_module.log_alpha
            if isinstance(param, torch.nn.Parameter):
                optimizers.append(optimizer_cls([param], **optimizer_kwargs))
        return optimizers

    def _construct_target_net_updater(self) -> SoftUpdate:
        # Prefer polyak parameter from network layout critic if present
        eps = self.config.optim.target_update_polyak
        return SoftUpdate(self.loss_module, eps=eps)

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
        if cfg.replay_buffer.prb:
            replay_buffer = TensorDictPrioritizedReplayBuffer(
                alpha=0.7,
                beta=0.5,
                pin_memory=True,
                prefetch=prefetch,
                storage=storage_cls(
                    max_size=buffer_size, compilable=cfg.compile.compile
                ),
                batch_size=batch_size,
                priority_key=cfg.sac.priority_key,
                shared=shared,
                compilable=cfg.compile.compile,
            )
        else:
            replay_buffer = TensorDictReplayBuffer(
                pin_memory=True,
                prefetch=prefetch,
                sampler=sampler,
                storage=storage_cls(
                    max_size=buffer_size, compilable=cfg.compile.compile
                ),
                batch_size=batch_size,
                shared=shared,
                compilable=cfg.compile.compile,
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
        assert isinstance(self.config, SACRLOptConfig)
        policy_op = self.actor_critic.get_policy_operator()
        kl_context = None
        if (self.config.optim.scheduler or "").lower() == "adaptive":
            kl_context = self._prepare_kl_context(sampled_tensordict, policy_op)

        # Compute loss
        loss_td = self.loss_module(sampled_tensordict)

        actor_loss = loss_td["loss_actor"]
        q_loss = loss_td["loss_qvalue"]
        alpha_loss = loss_td["loss_alpha"]

        (actor_loss + q_loss + alpha_loss).sum().backward()  # type: ignore[operator]

        # clip gradients
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_norm=1.0)

        # Update networks
        self.optim.step()

        kl_approx = None
        if kl_context is not None:
            kl_approx = self._compute_kl_after_update(kl_context, policy_op)
            if kl_approx is not None:
                loss_td.set("kl_approx", kl_approx.detach())
                self._maybe_adjust_lr(kl_approx, self.config.optim)

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

        # # Compile the update function if requested
        # compile_mode = None
        # if cfg.compile.compile:
        #     compile_mode = cfg.compile.compile_mode
        #     if compile_mode in ("", None):
        #         if cfg.compile.cudagraphs:
        #             compile_mode = "default"
        #         else:
        #             compile_mode = "reduce-overhead"

        #     self.log.info(f"Compiling update function with mode: {compile_mode}")
        #     self.update = compile_with_warmup(self.update, mode=compile_mode, warmup=1)  # type: ignore[method-assign]

        # # Only use CUDAGraphs on CUDA devices
        # if cfg.compile.cudagraphs:
        #     if self.device.type == "cuda":
        #         warnings.warn(
        #             "CudaGraphModule is experimental and may lead to silently wrong results. Use with caution.",
        #             category=UserWarning,
        #         )
        #         self.log.warning("Wrapping update with CudaGraphModule (experimental)")
        #         self.update = CudaGraphModule(
        #             self.update, in_keys=[], out_keys=[], warmup=5
        #         )  # type: ignore[method-assign]
        #     else:
        #         self.log.warning(
        #             f"CUDAGraphs requested but device is {self.device.type}, not CUDA. Skipping CUDAGraphs."
        #         )

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

                    metrics_to_log.update(
                        {
                            "episode/length": np.mean(self.episode_lengths),
                            "episode/return": np.mean(self.episode_rewards),
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
                            # Compute loss
                            loss = self.update(sampled_tensordict).clone()
                        loss_keys = ["loss_actor", "loss_qvalue", "loss_alpha"]
                        if "kl_approx" in loss:
                            loss_keys.append("kl_approx")
                        losses_list.append(loss.select(*loss_keys))
                        if self.config.replay_buffer.prb:
                            self.data_buffer.update_tensordict_priority(
                                sampled_tensordict
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

            # Get training losses and times
            if losses is not None:
                losses_mean = losses.apply(lambda x: x.float().mean(), batch_size=[])
                for key, value in losses_mean.items():  # type: ignore
                    try:
                        scalar = float(value)  # type: ignore[arg-type]
                    except Exception:
                        scalar = value.detach().cpu().float().item()  # type: ignore[attr-defined]
                    metrics_to_log.update({f"train/{key}": scalar})
                metrics_to_log["train/lr"] = self.optim.param_groups[0]["lr"]

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
                if "episode/return" in metrics_to_log:
                    postfix["ep_ret"] = f"{metrics_to_log['episode/return']:.1f}"
                if "episode/length" in metrics_to_log:
                    postfix["ep_len"] = f"{metrics_to_log['episode/length']:.0f}"
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
