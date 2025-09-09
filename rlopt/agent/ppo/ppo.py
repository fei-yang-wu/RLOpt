"""
Only supports MuJoCo environments for now
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn
import torch.optim
import tqdm
from omegaconf import DictConfig
from tensordict import TensorDict
from tensordict.nn import (
    AddStateIndependentNormalScale,
    TensorDictModule,
    TensorDictModuleBase,
)
from torch import Tensor
from torchrl._utils import timeit

# Import missing modules
from torchrl.collectors import MultiSyncDataCollector, SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.envs import Compose, ExplorationType, TransformedEnv
from torchrl.envs.transforms import InitTracker
from torchrl.modules import (
    MLP,
    ActorValueOperator,
    LSTMModule,
    ProbabilisticActor,
    TanhNormal,
    ValueOperator,
)
from torchrl.objectives import ClipPPOLoss, group_optimizers
from torchrl.objectives.value.advantages import GAE
from torchrl.record.loggers import Logger

from rlopt.common.base_class import BaseAlgorithm


class PPO(BaseAlgorithm):
    def __init__(
        self,
        env: TransformedEnv,
        config: DictConfig,
        policy_net: torch.nn.Module | None = None,
        value_net: torch.nn.Module | None = None,
        q_net: torch.nn.Module | None = None,
        reward_estimator_net: torch.nn.Module | None = None,
        replay_buffer: type[ReplayBuffer] = ReplayBuffer,
        logger: Logger | None = None,
        **kwargs,
    ):
        super().__init__(
            env,
            config,
            policy_net,
            value_net,
            q_net,
            reward_estimator_net,
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

    def _construct_feature_extractor(
        self, feature_extractor_net: torch.nn.Module | None = None
    ) -> TensorDictModule:
        """Override to build your feature extractor network from config.
        Note that if you don't want to use a shared feature extractor,
        you can set `use_feature_extractor` to False, then this will be an identity map. Then do feature extraction separately in policy and value functions.
        """
        if feature_extractor_net is not None:
            return TensorDictModule(
                module=feature_extractor_net,
                in_keys=self.total_input_keys,
                out_keys=["hidden"],
            )

        if self.config.use_feature_extractor:
            feature_extractor_mlp = MLP(
                in_features=self.total_input_shape,
                out_features=self.config.feature_extractor.output_dim,
                num_cells=self.config.feature_extractor.num_cells,
                activation_class=torch.nn.ELU,
                device=self.device,
            )

            for layer in feature_extractor_mlp.modules():
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.orthogonal_(layer.weight, 1.0)  # type: ignore
                    layer.bias.data.zero_()

            return TensorDictModule(
                module=feature_extractor_mlp,
                in_keys=self.total_input_keys,
                out_keys=["hidden"],
            )
        return TensorDictModule(
            module=torch.nn.Identity(),
            in_keys=self.total_input_keys,
            out_keys=self.total_input_keys,
        )

    def _construct_policy(
        self, policy_net: torch.nn.Module | None = None
    ) -> TensorDictModule:
        """Construct policy"""
        # for PPO, we use a probabilistic actor

        # Define policy output distribution class
        distribution_class = TanhNormal
        distribution_kwargs = {
            "low": self.env.action_spec_unbatched.space.low.to(self.device),  # type: ignore
            "high": self.env.action_spec_unbatched.space.high.to(self.device),  # type: ignore
            "tanh_loc": False,
        }

        # Define policy architecture
        if policy_net is None:
            policy_mlp = MLP(
                in_features=self.policy_input_shape,
                activation_class=torch.nn.ELU,
                out_features=self.policy_output_shape,
                num_cells=self.config.policy.num_cells,
                device=self.device,
            )
            # Initialize policy weights
            for layer in policy_mlp.modules():
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.orthogonal_(layer.weight, 1.0)  # type: ignore
                    layer.bias.data.zero_()
        else:
            policy_mlp = policy_net

        # Add state-independent normal scale
        policy_mlp = torch.nn.Sequential(
            policy_mlp,
            AddStateIndependentNormalScale(
                self.policy_output_shape,
                # scale_lb=1e-8,  # type: ignore
            ).to(self.device),
        )

        # Add probabilistic sampling of the actions
        return ProbabilisticActor(
            TensorDictModule(
                module=policy_mlp,
                in_keys=self.config.policy_in_keys,
                out_keys=["loc", "scale"],
            ),
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
        # Define value architecture
        if value_net is None:
            value_mlp = MLP(
                in_features=self.value_input_shape,
                activation_class=torch.nn.ELU,
                out_features=self.value_output_shape,
                num_cells=self.config.value_net.num_cells,
                device=self.device,
            )
            # Initialize value weights
            for layer in value_mlp.modules():
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.orthogonal_(layer.weight, 0.01)  # type: ignore
                    layer.bias.data.zero_()
        else:
            value_mlp = value_net

        # Define value module
        return ValueOperator(
            value_mlp,
            in_keys=self.config.value_net_in_keys,
        )

    def _construct_actor_critic(self) -> TensorDictModule:
        """Construct actor-critic network"""
        return ActorValueOperator(
            common_operator=self.feature_extractor,
            policy_operator=self.policy,
            value_operator=self.value_function,
        )

    def _construct_q_function(self, q_net: torch.nn.Module | None = None):
        pass

    def _construct_loss_module(self) -> torch.nn.Module:
        """Construct loss module"""
        loss_config = self.config.loss
        return ClipPPOLoss(
            actor_network=self.actor_critic.get_policy_operator(),
            critic_network=self.actor_critic.get_value_operator(),
            clip_epsilon=loss_config.clip_epsilon,
            loss_critic_type=loss_config.loss_critic_type,
            entropy_coeff=loss_config.entropy_coeff,
            critic_coeff=loss_config.critic_coeff,
            normalize_advantage=False,
            clip_value=loss_config.clip_value,
        )

    def _configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizers"""
        # Create optimizers
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
        if self.config.use_feature_extractor:
            feature_optim = torch.optim.AdamW(
                self.feature_extractor.parameters(),
                lr=torch.tensor(self.config.optim.lr, device=self.device),
                # eps=1e-5,
            )
            return group_optimizers(actor_optim, critic_optim, feature_optim)

        return group_optimizers(actor_optim, critic_optim)

    def _construct_adv_module(self) -> torch.nn.Module:
        """Construct advantage module"""
        # Create advantage module
        return GAE(
            gamma=self.config.loss.gamma,
            lmbda=self.config.loss.gae_lambda,
            value_network=self.actor_critic.get_value_operator(),  # type: ignore
            average_gae=False,
            device=self.device,
            vectorized=False,
        )

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
        self.optim.zero_grad(set_to_none=True)

        # Linearly decrease the learning rate and clip epsilon
        alpha = torch.ones((), device=self.device)
        if self.config.optim.anneal_lr:
            alpha = 1 - (num_network_updates / self.total_network_updates)
            for group in self.optim.param_groups:
                group["lr"] = self.config.optim.lr * alpha

        if self.config.loss.anneal_clip_epsilon:
            self.loss_module.clip_epsilon.copy_(self.config.loss.clip_epsilon * alpha)  # type: ignore

        num_network_updates = num_network_updates + 1

        # Forward pass PPO loss
        loss = self.loss_module(batch)
        critic_loss = loss["loss_critic"]
        actor_loss = loss["loss_objective"] + loss["loss_entropy"]
        total_loss = critic_loss + actor_loss
        # Backward pass
        total_loss.backward()

        # Update the networks
        self.optim.step()
        return loss.detach().set("alpha", alpha), num_network_updates

    def predict(self, obs: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Predict action given observation"""
        obs = torch.as_tensor([obs], device=self.device)
        self.policy.eval()
        with torch.inference_mode():
            td = TensorDict(
                dict.fromkeys(self.config.policy_in_keys, obs),
                batch_size=[1],
                device=self.policy.device,  # type: ignore
            )
            return self.policy(td).get("action")

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
        cfg_loss_anneal_clip_eps: bool = self.config.loss.anneal_clip_epsilon
        cfg_loss_clip_epsilon: float = self.config.loss.clip_epsilon

        losses = TensorDict(batch_size=[cfg_loss_ppo_epochs, num_mini_batches])  # type: ignore

        self.collector: SyncDataCollector | MultiSyncDataCollector
        collector_iter = iter(self.collector)
        total_iter = len(self.collector)
        for _i in range(total_iter):
            # timeit.printevery(1000, total_iter, erase=True)

            with timeit("collecting"):
                data = next(collector_iter)

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

                    # Check for NaNs in data_reshape tensordict
                    for key in data.keys(include_nested=True):
                        value = data.get(key)
                        if isinstance(value, Tensor):
                            if torch.isnan(value).any():
                                print("Before GAE")
                                print(f"[WARNING] NaNs detected in data at key: {key}")

                    # Compute GAE
                    with torch.no_grad(), timeit("adv"):
                        torch.compiler.cudagraph_mark_step_begin()
                        data = self.adv_module(data)
                        if self.config.compile.compile_mode:
                            data = data.clone()

                    with timeit("rb - extend"):
                        # Update the data buffer
                        data_reshape = data.reshape(-1)
                        self.data_buffer.extend(data_reshape)

                    # Check for NaNs in data_reshape tensordict
                    for key in data_reshape.keys(include_nested=True):
                        value = data_reshape.get(key)
                        if isinstance(value, Tensor):
                            if torch.isnan(value).any():
                                print("after GAE")
                                print(
                                    f"[WARNING] NaNs detected in data_reshape at key: {key}"
                                )

                    for k, batch in enumerate(self.data_buffer):
                        with timeit("update"):
                            torch.compiler.cudagraph_mark_step_begin()
                            loss, num_network_updates = self.update(  # type: ignore
                                batch, num_network_updates=num_network_updates
                            )
                            loss = loss.clone()
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
                                    "Episode/"
                                    + key: (
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
                    self.logger.log_scalar(key, value, collected_frames)

            self.collector.update_policy_weights_()

            # Save model periodically
            if (
                self.config.save_interval > 0
                and collected_frames % self.config.save_interval == 0
            ):
                self.save_model(
                    path=Path(self.config.logger.get("log_dir", "logs"))
                    / self.config.get("save_path", "models"),
                    step=collected_frames,
                )

        self.collector.shutdown()


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
        config: DictConfig,
        policy_net: torch.nn.Module | None = None,
        value_net: torch.nn.Module | None = None,
        q_net: torch.nn.Module | None = None,
        reward_estimator_net: torch.nn.Module | None = None,
        replay_buffer: type[ReplayBuffer] = ReplayBuffer,
        logger: Logger | None = None,
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
            reward_estimator_net,
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

    def _construct_feature_extractor(
        self, feature_extractor_net: torch.nn.Module | None = None
    ) -> TensorDictModuleBase:
        """Override to build LSTM-based feature extractor using TorchRL's LSTMModule."""
        if feature_extractor_net is not None:
            return feature_extractor_net

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

            return self.lstm_module

        # If not using feature extractor, return identity
        return TensorDictModule(
            module=torch.nn.Identity(),
            in_keys=self.total_input_keys,
            out_keys=self.total_input_keys,
        )

    def _construct_adv_module(self) -> torch.nn.Module:
        """Construct advantage module"""
        # Create advantage module
        return GAE(
            gamma=self.config.loss.gamma,
            lmbda=self.config.loss.gae_lambda,
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
        self.policy.eval()

        with torch.inference_mode():
            td = TensorDict(
                dict.fromkeys(self.config.policy_in_keys, obs),
                batch_size=[1],
                device=self.policy.device,  # type: ignore
            )

            # The LSTMModule automatically handles recurrent states through TensorDict
            # No need for manual state management - TorchRL handles this
            return self.policy(td).get("action")

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
