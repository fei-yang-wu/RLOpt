"""
Only supports MuJoCo environments for now
"""

from __future__ import annotations

from collections import deque

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
)
from torch import Tensor
from torchrl._utils import timeit

# Import missing modules
from torchrl.collectors import MultiSyncDataCollector, SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.envs import ExplorationType, TransformedEnv
from torchrl.modules import (
    MLP,
    ProbabilisticActor,
    TanhNormal,
    TruncatedNormal,
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
        policy: torch.nn.Module | None = None,
        value_net: torch.nn.Module | None = None,
        q_net: torch.nn.Module | None = None,
        reward_estimator: torch.nn.Module | None = None,
        replay_buffer: type[ReplayBuffer] = ReplayBuffer,
        logger: Logger | None = None,
        **kwargs,
    ):
        super().__init__(
            env,
            config,
            policy,
            value_net,
            q_net,
            reward_estimator,
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

    def _construct_policy(self) -> torch.nn.Module:
        """Construct policy following utils_mujoco.py pattern"""
        # for PPO, we use a probabilistic actor
        # Define input shape
        input_shape = self.env.observation_spec["policy"].shape  # type: ignore

        # Define policy output distribution class
        num_outputs = self.env.action_spec_unbatched.shape[-1]  # type: ignore
        distribution_class = TanhNormal
        distribution_kwargs = {
            "low": self.env.action_spec_unbatched.space.low.to(self.device),  # type: ignore
            "high": self.env.action_spec_unbatched.space.high.to(self.device),  # type: ignore
            "tanh_loc": False,
        }

        # Define policy architecture
        policy_mlp = MLP(
            in_features=input_shape[-1],
            activation_class=torch.nn.ELU,
            out_features=num_outputs,
            num_cells=self.config.policy.num_cells,
            device=self.device,
        )

        # Initialize policy weights
        for layer in policy_mlp.modules():
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.orthogonal_(layer.weight, 1.0)  # type: ignore
                layer.bias.data.zero_()

        # Add state-independent normal scale
        policy_mlp = torch.nn.Sequential(
            policy_mlp,
            AddStateIndependentNormalScale(
                self.env.action_spec_unbatched.shape[-1],  # type: ignore
                scale_lb=1e-8,  # type: ignore
            ).to(self.device),
        )

        # Add probabilistic sampling of the actions
        return ProbabilisticActor(
            TensorDictModule(
                module=policy_mlp,
                in_keys=["policy"],
                out_keys=["loc", "scale"],
            ),
            in_keys=["loc", "scale"],
            spec=self.env.full_action_spec_unbatched.to(self.device),  # type: ignore
            distribution_class=distribution_class,
            distribution_kwargs=distribution_kwargs,
            return_log_prob=True,
            default_interaction_type=ExplorationType.RANDOM,
        )

    def _construct_value_function(self) -> torch.nn.Module:
        """Construct value function following utils_mujoco.py pattern"""
        # Define input shape
        input_shape = self.env.observation_spec["policy"].shape  # type: ignore
        # Define value architecture
        value_mlp = MLP(
            in_features=input_shape[-1],
            activation_class=torch.nn.Tanh,
            out_features=1,
            num_cells=self.config.policy.num_cells,
            device=self.device,
        )

        # Initialize value weights
        for layer in value_mlp.modules():
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.orthogonal_(layer.weight, 0.01)  # type: ignore
                layer.bias.data.zero_()

        # Define value module
        return ValueOperator(
            value_mlp,
            in_keys=["policy"],
        )

    def _construct_loss_module(self) -> torch.nn.Module:
        """Construct loss module following ppo_mujoco.py pattern"""
        loss_config = self.config.loss
        return ClipPPOLoss(
            actor_network=self.policy,
            critic_network=self.value_net,
            clip_epsilon=loss_config.clip_epsilon,
            loss_critic_type=loss_config.loss_critic_type,
            entropy_coef=loss_config.entropy_coef,
            critic_coef=loss_config.critic_coef,
            normalize_advantage=True,
        )

    def _configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizers following ppo_mujoco.py pattern"""
        # Create optimizers
        actor_optim = torch.optim.Adam(
            self.policy.parameters(),
            lr=torch.tensor(self.config.optim.lr, device=self.device),
            eps=1e-5,
        )
        critic_optim = torch.optim.Adam(
            self.value_net.parameters(),  # type: ignore
            lr=torch.tensor(self.config.optim.lr, device=self.device),
            eps=1e-5,
        )
        return group_optimizers(actor_optim, critic_optim)

    def _construct_adv_module(self) -> torch.nn.Module:
        """Construct advantage module following ppo_mujoco.py pattern"""
        # Create advantage module
        return GAE(
            gamma=self.config.loss.gamma,
            lmbda=self.config.loss.gae_lambda,
            value_network=self.value_net,  # type: ignore
            average_gae=False,
            device=self.device,
            vectorized=not self.config.compile.compile,
        )

    def _construct_data_buffer(self) -> ReplayBuffer:
        """Construct data buffer following ppo_mujoco.py pattern"""
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
        """Compile components following ppo_mujoco.py pattern"""
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
        """Update function following ppo_mujoco.py pattern"""
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
                {"policy": obs},
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
        """Train the agent following ppo_mujoco.py pattern"""
        cfg = self.config
        # Main loop
        collected_frames = 0
        num_network_updates = torch.zeros((), dtype=torch.int64, device=self.device)
        pbar = tqdm.tqdm(total=self.config.collector.total_frames)

        num_mini_batches = cfg.collector.frames_per_batch // cfg.loss.mini_batch_size
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
        episode_lengths = deque(maxlen=100)
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
                episode_lengths.extend(episode_length.cpu().tolist())

                metrics_to_log.update(
                    {
                        "train/reward": episode_rewards.mean().item(),
                        "train/episode_length": np.mean(episode_lengths),
                    }
                )

            with timeit("training"):
                for j in range(cfg_loss_ppo_epochs):
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
            if hasattr(self.env, "log_infos"):
                for _ in range(len(self.env.log_infos)):
                    log_info_dict: dict[str, Tensor] = self.env.log_infos.popleft()
                    # log all the keys
                    for key, value in log_info_dict.items():
                        if "/" in key:
                            metrics_to_log.update(
                                {
                                    key: value.item()
                                    if isinstance(value, Tensor)
                                    else value
                                }
                            )
                        else:
                            metrics_to_log.update(
                                {
                                    "Episode/" + key: value.item()
                                    if isinstance(value, Tensor)
                                    else value
                                }
                            )

            # Log metrics
            if self.logger:
                metrics_to_log.update(timeit.todict(prefix="time"))  # type: ignore
                metrics_to_log["time/speed"] = pbar.format_dict["rate"]
                for key, value in metrics_to_log.items():
                    self.logger.log_scalar(key, value, collected_frames)

            self.collector.update_policy_weights_()

        self.collector.shutdown()
