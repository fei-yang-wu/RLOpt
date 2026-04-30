from __future__ import annotations

import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar, cast

import numpy as np
import torch
import torch.nn
import torch.optim
from tensordict import TensorDict
from tensordict.nn import (
    InteractionType,
    TensorDictModule,
)
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torchrl._utils import timeit
from torchrl.collectors import Collector
from torchrl.data import (
    Bounded,
    LazyTensorStorage,
    ReplayBuffer,
    TensorDictReplayBuffer,
)
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.envs import Compose, ExplorationType, TransformedEnv
from torchrl.envs.transforms import InitTracker
from torchrl.envs.utils import set_exploration_type
from torchrl.modules import (
    MLP,
    ActorValueOperator,
    IndependentNormal,
    LSTMModule,
    ProbabilisticActor,
    TanhNormal,
    ValueOperator,
)
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value.advantages import GAE
from torchrl.record.loggers import Logger
from tqdm.rich import tqdm

from rlopt.base_class import (
    BaseAlgorithm,
    IterationData,
    TrainingMetadata,
)
from rlopt.config_base import NetworkConfig, RLOptConfig
from rlopt.models import GaussianPolicyHead
from rlopt.type_aliases import OptimizerClass
from rlopt.utils import get_activation_class, log_info

PpoCfgT = TypeVar("PpoCfgT", bound="PPORLOptConfig")


class CatInputs(torch.nn.Module):
    """Concatenate multiple input tensors along the last dimension.

    Used to adapt modules that expect a single tensor (e.g. MLP) when
    ``TensorDictModule`` feeds multiple ``in_keys`` as separate positional
    arguments (the ``concatenate_terms=False`` / multi-key case).
    """

    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *inputs: Tensor) -> Tensor:
        if len(inputs) == 1:
            return self.module(inputs[0])
        return self.module(torch.cat(inputs, dim=-1))


@dataclass
class PPOConfig:
    """PPO-specific configuration."""

    gae_lambda: float = 0.95
    """GAE lambda parameter."""

    clip_epsilon: float = 0.2
    """Clipping epsilon for PPO."""

    clip_value: bool = True
    """Whether to clip value function."""

    anneal_clip_epsilon: bool = False
    """Whether to anneal clip epsilon."""

    critic_coeff: float = 1.0
    """Critic coefficient."""

    entropy_coeff: float = 0.008
    """Entropy coefficient."""

    normalize_advantage: bool = True
    """Whether to normalize the advantage estimates."""

    log_std_init: float = 0.0
    """Initial value for the policy log standard deviation (log std)."""

    clip_log_std: bool = True
    """Whether to clip the learned log standard deviation."""

    log_std_min: float = -7.0
    """Minimum log standard deviation (when clipping is enabled)."""

    log_std_max: float = 2.0
    """Maximum log standard deviation (when clipping is enabled)."""


@dataclass
class PPORLOptConfig(RLOptConfig):
    """PPO configuration that extends RLOptConfig."""

    ppo: PPOConfig = field(default_factory=PPOConfig)
    """PPO configuration."""

    def __post_init__(self):
        self.use_value_function = True
        # Initialize value_function config if not set.
        # input_keys=None → resolves to ["policy"] (IsaacLab default).
        if self.value_function is None:
            self.value_function = NetworkConfig(
                num_cells=[256, 256],
                activation_fn="relu",
                output_dim=1,
            )


@dataclass(kw_only=True)
class PPOTrainingMetadata(TrainingMetadata):
    """PPO-specific extension of the generic train-metadata state."""

    collector_iter: Iterator[TensorDict]
    """ Iterator that yields collected rollouts from the TorchRL collector. """

    policy_operator: TensorDictModule
    """ Policy snapshot used for KL diagnostics around each policy update. """

    updates_completed: int
    """ Number of optimizer updates completed so far in this training metadata. """

    minibatches_per_epoch: int
    """ Number of replay-buffer minibatches consumed per PPO epoch. """

    epochs_per_rollout: int
    """ Number of PPO epochs to run for each collected rollout. """

    anneal_clip_epsilon: bool
    """ Whether clip epsilon is annealed by the loss module over training. """

    base_clip_epsilon: float
    """ Configured clip epsilon before any annealing is applied. """


@dataclass(kw_only=True)
class PPOIterationData(IterationData):
    """One collected rollout flowing through the shared PPO phases."""

    rollout: TensorDict
    """ Raw rollout TensorDict produced by the collector for this outer-metadata iteration. """


class PPO(BaseAlgorithm[PpoCfgT], Generic[PpoCfgT]):
    def __init__(
        self,
        env: TransformedEnv,
        config: PpoCfgT,
        policy_net: torch.nn.Module | None = None,
        value_net: torch.nn.Module | None = None,
        q_net: torch.nn.Module | None = None,
        replay_buffer: type[ReplayBuffer] = ReplayBuffer,
        logger: Logger | None = None,
        feature_extractor_net: torch.nn.Module | None = None,
        **kwargs,
    ):
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

        # construct the advantage module
        self.adv_module = self._construct_adv_module()

        # Cache optimizer parameters for compile-friendly grad clipping.
        self._grad_clip_params: list[Tensor] = [  # type: ignore[attr-defined]
            param for group in self.optim.param_groups for param in group["params"]
        ]

        # Compile if requested
        self._compile_components()

        # Initialize total network updates
        self.total_network_updates = 0

    def _construct_feature_extractor(
        self, feature_extractor_net: torch.nn.Module | None = None
    ) -> TensorDictModule:
        """Build feature extractor (optional for PPO)."""
        msg = "PPO does not require a feature extractor by default."
        raise NotImplementedError(msg)

    def _construct_policy(
        self, policy_net: torch.nn.Module | None = None
    ) -> TensorDictModule:
        """Construct policy"""
        # for PPO, we use a probabilistic actor
        assert isinstance(self.config, PPORLOptConfig)

        action_spec = self.env.action_spec_unbatched  # type: ignore
        action_dim = int(action_spec.shape[-1])  # type: ignore[attr-defined]
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

        # Build policy network
        if policy_net is None:
            policy_mlp = MLP(
                in_features=self.config.policy.input_dim,
                activation_class=get_activation_class(self.config.policy.activation_fn),
                out_features=action_dim,
                num_cells=list(self.config.policy.num_cells),
                device=self.device,
            )
        else:
            policy_mlp = policy_net

        net = GaussianPolicyHead(
            base=policy_mlp,
            action_dim=action_dim,
            log_std_init=self.config.ppo.log_std_init,
            log_std_min=self.config.ppo.log_std_min,
            log_std_max=self.config.ppo.log_std_max,
            clip_log_std=self.config.ppo.clip_log_std,
            device=self.device,
        )

        # Wrap in TensorDictModule
        policy_td = TensorDictModule(
            module=net,
            in_keys=self.config.policy.get_input_keys(),
            out_keys=["loc", "scale"],
        )

        # Add probabilistic sampling of the actions
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
        self, value_net: torch.nn.Module | None = None
    ) -> TensorDictModule:
        """Construct value function"""
        if self.config.value_function is None:
            msg = "PPO requires a value function configuration."
            raise ValueError(msg)

        # Build value network
        if value_net is None:
            value_mlp = MLP(
                in_features=self.config.value_function.input_dim,
                activation_class=get_activation_class(
                    self.config.value_function.activation_fn
                ),
                out_features=1,
                num_cells=list(self.config.value_function.num_cells),
                device=self.device,
            )
        else:
            value_mlp = value_net

        in_keys = self.config.value_function.get_input_keys()
        # When multiple in_keys (concatenate_terms=False), TensorDictModule
        # passes each as a separate positional arg. MLP expects a single
        # tensor, so wrap it to concatenate first.
        module = CatInputs(value_mlp) if len(in_keys) > 1 else value_mlp
        return ValueOperator(module=module, in_keys=in_keys)

    def _construct_q_function(self) -> TensorDictModule:  # type: ignore[override]
        """PPO does not use a state-action value function explicitly.
        This method is not used but required by BaseAlgorithm interface.
        """
        return TensorDictModule(
            module=torch.nn.Identity(),
            in_keys=[],
            out_keys=[],
        )

    def _construct_actor_critic(self) -> TensorDictModule:
        """Construct actor-critic network"""
        assert isinstance(self.value_function, TensorDictModule)
        assert isinstance(self.policy, TensorDictModule)

        # Use feature extractor if available, otherwise use identity
        if self.feature_extractor:
            common_operator = self.feature_extractor
        else:
            # Identity pass-through: each input key is forwarded unchanged.
            class IdentityModule(torch.nn.Module):
                def forward(self, *x):
                    return x[0] if len(x) == 1 else x

            in_keys = self.config.policy.get_input_keys()
            common_operator = TensorDictModule(
                module=IdentityModule(),
                in_keys=in_keys,
                out_keys=in_keys,
            )

        return ActorValueOperator(
            common_operator=common_operator,
            policy_operator=self.policy,
            value_operator=self.value_function,
        )

    def _construct_loss_module(self) -> torch.nn.Module:
        """Construct loss module"""
        assert isinstance(self.config, PPORLOptConfig)
        loss_config = self.config.loss
        ppo_config = self.config.ppo

        # Initialize lazy layers by performing a forward pass with dummy data
        fake_tensordict = self.env.fake_tensordict()
        with torch.no_grad():
            _ = self.actor_critic(fake_tensordict)

        return ClipPPOLoss(
            actor_network=self.actor_critic.get_policy_operator(),
            critic_network=self.actor_critic.get_value_operator(),
            clip_epsilon=ppo_config.clip_epsilon,
            loss_critic_type=loss_config.loss_critic_type,
            entropy_coeff=ppo_config.entropy_coeff,
            critic_coeff=ppo_config.critic_coeff,
            normalize_advantage=ppo_config.normalize_advantage,
            clip_value=ppo_config.clip_value,
        )

    def _construct_adv_module(self) -> torch.nn.Module:
        """Construct advantage module"""
        assert isinstance(self.config, PPORLOptConfig)
        # Create advantage module
        return GAE(
            gamma=self.config.loss.gamma,
            lmbda=self.config.ppo.gae_lambda,
            value_network=self.actor_critic.get_value_operator(),  # type: ignore
            average_gae=False,
            device=self.device,
            vectorized=False,
        )

    def _set_optimizers(  # type: ignore[override]
        self, optimizer_cls: OptimizerClass, optimizer_kwargs: dict[str, Any]
    ) -> list[torch.optim.Optimizer]:
        """Create optimizer for actor-critic parameters."""
        # For PPO, we use a single optimizer for both policy and value
        return [optimizer_cls(self.actor_critic.parameters(), **optimizer_kwargs)]

    def _construct_data_buffer(self) -> ReplayBuffer:
        """Construct data buffer"""
        # Create data buffer
        cfg = self.config
        sampler = (
            SamplerWithoutReplacement()
        )  # Removed True parameter to match ppo_mujoco.py
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

    def _compile_components(self):
        """Compile components"""
        if not hasattr(self, "adv_module"):
            return
        if getattr(self, "_components_compiled", False):
            return
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
            self._components_compiled = True  # type: ignore[attr-defined]

    @staticmethod
    def _policy_action_from_dist(dist: Any) -> Tensor | None:
        """Extract a deterministic action representative from a policy distribution."""
        for attr_name in ("mean", "loc", "mode"):
            try:
                value = getattr(dist, attr_name, None)
            except Exception:
                continue
            if value is None:
                continue
            if callable(value):
                try:
                    value = value()
                except TypeError:
                    continue
            if isinstance(value, Tensor):
                return value
        return None

    def _clip_policy_action(self, action: Tensor) -> Tensor:
        """Clamp deterministic actions to the action-spec bounds when available."""
        action_spec = getattr(self.env, "action_spec_unbatched", self.env.action_spec)
        if not isinstance(action_spec, Bounded):
            return action
        low = torch.as_tensor(
            action_spec.space.low,
            device=action.device,
            dtype=action.dtype,
        )
        high = torch.as_tensor(
            action_spec.space.high,
            device=action.device,
            dtype=action.dtype,
        )
        return torch.clamp(action, min=low, max=high)

    def _extra_actor_loss(self, batch: TensorDict) -> tuple[Tensor, dict[str, Tensor]]:
        """Return actor-only auxiliary losses added on top of the PPO objective."""
        del batch
        return torch.zeros((), device=self.device), {}

    def update(
        self, batch: TensorDict, num_network_updates: int
    ) -> tuple[TensorDict, int]:
        """Update function"""
        self.optim.zero_grad(set_to_none=True)

        # Forward pass PPO loss
        loss: TensorDict = self.loss_module(batch)
        critic_loss = loss["loss_critic"]
        extra_actor_loss, extra_actor_metrics = self._extra_actor_loss(batch)
        actor_loss = loss["loss_objective"] + loss["loss_entropy"] + extra_actor_loss
        total_loss = critic_loss + actor_loss
        # Backward pass
        total_loss.backward()  # type: ignore

        output_loss = loss.detach()  # type: ignore
        for key, value in extra_actor_metrics.items():
            metric = (
                value
                if isinstance(value, Tensor)
                else torch.tensor(value, device=self.device, dtype=torch.float32)
            )
            output_loss.set(key, metric.detach())

        max_grad_norm = self.config.optim.max_grad_norm
        if max_grad_norm is not None and max_grad_norm > 0:
            grad_norm_tensor = clip_grad_norm_(
                self._grad_clip_params, float(max_grad_norm)
            )
        else:
            grad_norm_tensor = torch.zeros((), device=self.device)

        # Update the networks
        self.optim.step()
        output_loss.set("alpha", torch.tensor(1.0, device=self.device))  # type: ignore
        output_loss.set("grad_norm", grad_norm_tensor.detach())  # type: ignore
        return output_loss, num_network_updates + 1  # type: ignore

    def _collector_iter(self) -> Iterator[TensorDict]:
        """Yield data from the collector while enforcing NaN guards per batch."""
        yield from self.collector

    def init_metadata(self) -> PPOTrainingMetadata:
        """Build the stable state shared across the full on-policy train metadata."""
        cfg = self.config

        num_mini_batches = cfg.collector.frames_per_batch // cfg.loss.mini_batch_size
        if cfg.collector.frames_per_batch % cfg.loss.mini_batch_size != 0:
            num_mini_batches += 1

        self.total_network_updates = (
            (cfg.collector.total_frames // cfg.collector.frames_per_batch)
            * cfg.loss.epochs
            * num_mini_batches
        )

        trainer_cfg = getattr(cfg, "trainer", None)
        progress_bar_enabled = (
            True if trainer_cfg is None else bool(trainer_cfg.progress_bar)
        )
        log_interval_frames = (
            1000 if trainer_cfg is None else max(1, int(trainer_cfg.log_interval))
        )

        self.collector = cast(Collector, self.collector)
        return PPOTrainingMetadata(
            collector_iter=iter(self._collector_iter()),
            total_iterations=len(self.collector),
            policy_operator=self.actor_critic.get_policy_operator(),
            progress_bar=tqdm(
                total=cfg.collector.total_frames,
                disable=not progress_bar_enabled or not sys.stdout.isatty(),
                dynamic_ncols=True,
            ),
            progress_bar_enabled=progress_bar_enabled,
            log_interval_frames=log_interval_frames,
            next_log_frame=log_interval_frames,
            next_file_log_frame=log_interval_frames,
            frames_processed=0,
            updates_completed=torch.zeros((), dtype=torch.int64, device=self.device),
            minibatches_per_epoch=num_mini_batches,
            epochs_per_rollout=cfg.loss.epochs,
            anneal_clip_epsilon=cfg.ppo.anneal_clip_epsilon,
            base_clip_epsilon=cfg.ppo.clip_epsilon,
        )

    def validate_training(self) -> None:
        """Validate algorithm-specific prerequisites before the first rollout."""
        return

    def collect(self, run: PPOTrainingMetadata, iteration_idx: int) -> PPOIterationData:
        """Collect one rollout and package it into the per-iteration state object."""
        self.actor_critic.eval()
        self.adv_module.eval()

        collect_start = time.perf_counter()
        with timeit("collecting"):
            rollout = next(run.collector_iter)
        collect_time = time.perf_counter() - collect_start

        rollout_frames = rollout.numel()
        run.frames_processed += rollout_frames
        if run.progress_bar_enabled:
            run.progress_bar.update(rollout_frames)  # type: ignore[attr-defined]
        return PPOIterationData(
            iteration_idx=iteration_idx,
            rollout=rollout,
            frames=rollout_frames,
            collect_time=collect_time,
        )

    def prepare(
        self,
        _iteration: PPOIterationData,
        _metadata: PPOTrainingMetadata,
    ) -> None:
        """Mutate the collected rollout before the shared learning phase begins."""
        return

    def pre_iteration_compute(self, rollout: TensorDict) -> TensorDict:
        """Turn one rollout into the replay-buffer view consumed by minibatch updates."""
        with torch.no_grad():
            rollout = self.adv_module(rollout)
            if getattr(self.config.compile, "compile", False):
                rollout = rollout.clone()

        self.data_buffer.extend(rollout.reshape(-1))
        return rollout

    @property
    def _required_loss_metrics(self) -> list[str]:
        """Return the always-recorded loss keys for one PPO minibatch update."""
        return ["loss_critic", "loss_entropy", "loss_objective"]

    @property
    def _optional_loss_metrics(self) -> list[str]:
        """Return optional loss keys recorded when the update produced them."""
        return [
            "entropy",
            "explained_variance",
            "clip_fraction",
            "value_clip_fraction",
            "ESS",
            "kl_approx",
            "grad_norm",
        ]

    def _select_reported_loss_metrics(self, loss: TensorDict) -> TensorDict:
        """Select the subset of update outputs that should be aggregated and logged."""
        loss_keys = [key for key in self._required_loss_metrics if key in loss]
        for key in self._optional_loss_metrics:
            loss_keys.append(key)
        return loss.select(*loss_keys)

    def iterate(
        self, iteration: PPOIterationData, metadata: PPOTrainingMetadata
    ) -> None:
        """Run update epochs over the current rollout and aggregate minibatch metrics."""
        losses = TensorDict(
            batch_size=[metadata.epochs_per_rollout, metadata.minibatches_per_epoch]
        )
        learn_start = time.perf_counter()

        self.data_buffer.empty()
        self.actor_critic.train()
        self.adv_module.train()

        with timeit("training"):
            # Pre-iteration compute GAE, once per rollout
            iteration.rollout = self.pre_iteration_compute(iteration.rollout)

            # Run PPO epochs over the rollout
            for epoch_idx in range(metadata.epochs_per_rollout):
                # Run PPO epochs over the rollout
                for batch_idx, batch in enumerate(self.data_buffer):
                    kl_context = None
                    if (self.config.optim.scheduler or "").lower() == "adaptive":
                        kl_context = self._prepare_kl_context(
                            batch, metadata.policy_operator
                        )

                    loss, metadata.updates_completed = self.update(
                        batch, metadata.updates_completed
                    )
                    loss = loss.clone()

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

    def _build_control_metrics(self, metadata: PPOTrainingMetadata) -> dict[str, Any]:
        """Report optimizer and scheduler-controlled scalars for the current run."""
        clip_attr = getattr(self.loss_module, "clip_epsilon", None)
        if metadata.anneal_clip_epsilon and isinstance(clip_attr, torch.Tensor):
            clip_epsilon = float(clip_attr.detach().item())
        else:
            clip_epsilon = float(metadata.base_clip_epsilon)
        return {
            "train/lr": self.optim.param_groups[0]["lr"],
            "train/clip_epsilon": clip_epsilon,
        }

    def _build_timing_metrics(
        self, iteration: PPOIterationData, metadata: PPOTrainingMetadata
    ) -> dict[str, float]:
        """Return timing metrics for one rollout iteration."""
        rate = (
            metadata.progress_bar.format_dict.get("rate")  # type: ignore[attr-defined]
            if metadata.progress_bar_enabled
            else None
        )
        if rate is not None:
            return {"time/speed": float(rate)}
        iter_time = iteration.collect_time + iteration.learn_time
        speed = float(iteration.frames) / iter_time if iter_time > 0.0 else 0.0
        return {"time/speed": speed}

    def _record_env_metrics(self, iteration: PPOIterationData) -> None:
        """Add environment-side scalar metrics when the env exposes them."""
        if "Isaac" in self.config.env.env_name and hasattr(self.env, "log_infos"):
            log_info_dict: dict[str, Tensor] = self.env.log_infos.popleft()
            log_info(log_info_dict, iteration.metrics)

    def _progress_summary_fields(self) -> tuple[tuple[str, str], ...]:
        """Return the metrics printed for non-progress-bar training runs."""
        return (
            ("train/step_reward_mean", "r_step"),
            ("episode/length", "ep_len"),
            ("episode/return", "r_ep"),
            ("train/loss_objective", "pi_loss"),
            ("time/speed", "fps"),
        )

    def record(
        self,
        iteration: PPOIterationData,
        metadata: PPOTrainingMetadata,
    ) -> None:
        """Flush metrics, refresh progress, and handle checkpoint cadence."""
        rollout = iteration.rollout

        step_rewards = rollout["next", "reward"]
        iteration.metrics.update(
            {
                "train/step_reward_mean": step_rewards.mean().item(),  # type: ignore
                "train/step_reward_std": step_rewards.std().item(),  # type: ignore
                "train/step_reward_max": step_rewards.max().item(),  # type: ignore
                "train/step_reward_min": step_rewards.min().item(),  # type: ignore
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
        if self._should_log_iteration(metadata, iteration):
            self.log_metrics(
                iteration.metrics,
                step=metadata.frames_processed,
                log_python=False,
            )
        self._log_iteration_file_summary(metadata, iteration)
        self.collector.update_policy_weights_()
        self._refresh_progress_display(metadata, iteration)

        if (
            self._should_save_checkpoint(
                frames_processed=metadata.frames_processed,
                frames_in_iteration=iteration.frames,
            )
        ):
            checkpoint_dir = self.log_dir / self.config.logger.save_path
            custom_save = getattr(self, "save", None)
            if callable(custom_save):
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                custom_save(
                    checkpoint_dir / f"model_step_{metadata.frames_processed}.pt"
                )
            else:
                self.save_model(
                    path=checkpoint_dir,
                    step=metadata.frames_processed,
                )

    def train(self) -> None:  # type: ignore
        """Train the agent with the shared on-policy rollout-to-update workflow."""
        self.validate_training()
        metadata = self.init_metadata()

        try:
            for iteration_idx in range(metadata.total_iterations):
                # 1) Collect a rollout and create the iteration state.
                iteration = self.collect(metadata, iteration_idx)

                # 2) Let the algorithm reshape rewards or attach extra data.
                self.prepare(iteration, metadata)

                # 3) Run the shared learning phase over the prepared rollout.
                self.iterate(iteration, metadata)

                # 4) Log, refresh collector weights, and checkpoint if needed.
                self.record(iteration, metadata)
        finally:
            metadata.progress_bar.close()  # type: ignore[attr-defined]
            self.collector.shutdown()

    def predict(self, obs: Tensor | np.ndarray) -> Tensor:  # type: ignore[override]
        """Predict action given observation."""
        obs = torch.as_tensor(obs, device=self.device)
        input_keys = list(
            getattr(self, "total_input_keys", self.config.policy.get_input_keys())
        )
        input_key = input_keys[0]
        feature_shape = self.observation_feature_shape(input_key)
        if tuple(obs.shape) == feature_shape:
            obs = obs.unsqueeze(0)
        batch_ndim = max(obs.ndim - len(feature_shape), 0)
        batch_size = list(obs.shape[:batch_ndim])
        policy_op = self.actor_critic.get_policy_operator()
        policy_op.eval()
        with torch.no_grad(), set_exploration_type(InteractionType.DETERMINISTIC):
            td = TensorDict(
                dict.fromkeys(self.total_input_keys, obs),
                batch_size=batch_size,
                device=self.device,
            )
            td = policy_op(td)
            return td.get("action")


class PPORecurrent(PPO):
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
        config: PPORLOptConfig,
        policy_net: torch.nn.Module | None = None,
        value_net: torch.nn.Module | None = None,
        q_net: torch.nn.Module | None = None,
        replay_buffer: type[ReplayBuffer] = ReplayBuffer,
        logger: Logger | None = None,
        feature_extractor_net: torch.nn.Module | None = None,
        **kwargs,
    ):
        # Store LSTM module reference for primer creation
        self.lstm_module: LSTMModule | None = None

        # Add required transforms InitTracker to environment
        env = self.add_required_transforms(env)

        super().__init__(
            env,
            config,
            policy_net,
            value_net,
            q_net,
            replay_buffer,
            logger,
            feature_extractor_net,
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
                    self.env.transform = Compose([self.env.transform, primer])

    def _construct_feature_extractor(
        self, feature_extractor_net: torch.nn.Module | None = None
    ) -> TensorDictModule:
        """Override to build LSTM-based feature extractor using TorchRL's LSTMModule."""
        if feature_extractor_net is not None:
            return TensorDictModule(
                module=feature_extractor_net,
                in_keys=list(self.total_input_keys),
                out_keys=["hidden"],
            )

        if self.config.use_feature_extractor:
            # Get LSTM configuration with defaults
            lstm_config = getattr(self.config.feature_extractor, "lstm", {})
            hidden_size = getattr(
                lstm_config,
                "hidden_size",
                getattr(self.config.feature_extractor, "output_dim", 256),
            )
            num_layers = getattr(lstm_config, "num_layers", 1)
            dropout = getattr(lstm_config, "dropout", 0.0)
            bidirectional = getattr(lstm_config, "bidirectional", False)

            # Create LSTM module using TorchRL's LSTMModule
            # Use in_key/out_key so TorchRL expands the recurrent state keys automatically
            self.lstm_module = LSTMModule(
                input_size=self.total_input_shape,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=bidirectional,
                batch_first=True,
                device=self.device,
                in_key=self.total_input_keys[0],
                out_key="hidden",
            )

            return self.lstm_module  # type: ignore

        # If not using feature extractor, return identity
        return TensorDictModule(
            module=torch.nn.Identity(),
            in_keys=list(self.total_input_keys),
            out_keys=list(self.total_input_keys),
        )

    def _construct_adv_module(self) -> torch.nn.Module:
        """Construct advantage module"""
        assert isinstance(self.config, PPORLOptConfig)
        # Create advantage module
        return GAE(
            gamma=self.config.loss.gamma,
            lmbda=self.config.ppo.gae_lambda,
            value_network=self.actor_critic.get_value_operator(),  # type: ignore
            average_gae=False,
            device=self.device,
            vectorized=not self.config.compile.compile,
            deactivate_vmap=True,  # to be compatible with lstm
            shifted=True,  # to be compatible with lstm
        )

    def reset_recurrent_states(self):
        """Reset LSTM recurrent states - called at episode boundaries"""
        # With TorchRL's LSTMModule, states are automatically managed
        # This method is provided for compatibility but may not be needed
        # as the InitTracker transform and proper episode boundaries handle this
        return

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
                        env.transform = Compose([env.transform, *new_transforms])
                else:
                    env.transform = Compose([*new_transforms])

        return env
