"""
Only supports MuJoCo environments for now
"""

from __future__ import annotations

from typing import Optional, Union, Tuple

import numpy as np
import torch
import torch.nn
import torch.optim
import tqdm
from omegaconf import DictConfig
from tensordict import TensorDict
from tensordict.nn import (
    AddStateIndependentNormalScale,
    CudaGraphModule,
    TensorDictModule,
)

from rlopt.common.base_class import BaseAlgorithm

# Import missing modules
from torchrl.collectors import MultiSyncDataCollector, SyncDataCollector
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer, ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.envs import EnvBase, ExplorationType, TransformedEnv
from torchrl.modules import MLP, ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss, group_optimizers
from torchrl.objectives.value.advantages import GAE
from torchrl.record.loggers import Logger
from torchrl._utils import timeit


class PPO(BaseAlgorithm):
    def __init__(
        self,
        env: TransformedEnv,
        config: DictConfig,
        policy: Optional[torch.nn.Module] = None,
        value_net: Optional[torch.nn.Module] = None,
        q_net: Optional[torch.nn.Module] = None,
        reward_estimator: Optional[torch.nn.Module] = None,
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
        input_shape = self.env.observation_spec['observation'].shape # type: ignore

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
            activation_class=torch.nn.Tanh,
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
                self.env.action_spec_unbatched.shape[-1], # type: ignore
                scale_lb=1e-8
            ).to(self.device),
        )

        # Add probabilistic sampling of the actions
        policy_module = ProbabilisticActor(
            TensorDictModule(
                module=policy_mlp,
                in_keys=["observation"],
                out_keys=["loc", "scale"],
            ),
            in_keys=["loc", "scale"],
            spec=self.env.full_action_spec_unbatched.to(self.device), # type: ignore
            distribution_class=distribution_class,
            distribution_kwargs=distribution_kwargs,
            return_log_prob=True,
            default_interaction_type=ExplorationType.RANDOM,  # Changed to RANDOM to match utils_mujoco.py
        )

        return policy_module

    def _construct_value_function(self) -> torch.nn.Module:
        """Construct value function following utils_mujoco.py pattern"""
        # Define input shape
        input_shape = self.env.observation_spec["observation"].shape # type: ignore
        # Define value architecture - following utils_mujoco.py pattern
        value_mlp = MLP(
            in_features=input_shape[-1],
            activation_class=torch.nn.Tanh,  # Changed from ELU to Tanh to match utils_mujoco.py
            out_features=1,
            num_cells=[64, 64],  # Use fixed architecture like utils_mujoco.py
            device=self.device,
        )

        # Initialize value weights
        for layer in value_mlp.modules():
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.orthogonal_(layer.weight, 0.01)  # type: ignore
                layer.bias.data.zero_()

        # Define value module
        value_module = ValueOperator(
            value_mlp,
            in_keys=["observation"],
        )

        return value_module

    def _construct_loss_module(self) -> torch.nn.Module:
        """Construct loss module following ppo_mujoco.py pattern"""
        loss_config = self.config.loss
        loss_module = ClipPPOLoss(
            actor_network=self.policy,
            critic_network=self.value_net,
            clip_epsilon=loss_config.clip_epsilon,
            loss_critic_type=loss_config.loss_critic_type,
            entropy_coef=loss_config.entropy_coef,
            critic_coef=loss_config.critic_coef,
            normalize_advantage=True,
        )
        return loss_module

    def _configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizers following ppo_mujoco.py pattern"""
        # Create optimizers
        actor_optim = torch.optim.Adam(
            self.policy.parameters(),
            lr=torch.tensor(self.config.optim.lr, device=self.device),
            eps=1e-5,
        )
        critic_optim = torch.optim.Adam(
            self.value_net.parameters(), # type: ignore
            lr=torch.tensor(self.config.optim.lr, device=self.device),
            eps=1e-5,
        )
        return group_optimizers(actor_optim, critic_optim)

    def _construct_adv_module(self) -> torch.nn.Module:
        """Construct advantage module following ppo_mujoco.py pattern"""
        # Create advantage module
        adv_module = GAE(
            gamma=self.config.loss.gamma,
            lmbda=self.config.loss.gae_lambda,
            value_network=self.value_net, # type: ignore
            average_gae=False,
            device=self.device,
            vectorized=not self.config.compile.compile,
        )
        return adv_module

    def _construct_data_buffer(self) -> ReplayBuffer:
        """Construct data buffer following ppo_mujoco.py pattern"""
        # Create data buffer
        cfg = self.config
        sampler = (
            SamplerWithoutReplacement()
        )  # Removed True parameter to match ppo_mujoco.py
        data_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(
                cfg.collector.frames_per_batch,
                compilable=cfg.compile.compile,  # type: ignore
                device=self.device,
            ),
            sampler=sampler,
            batch_size=cfg.loss.mini_batch_size,
            compilable=cfg.compile.compile,
        )
        return data_buffer

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

            self.update = torch.compile(self.update, mode=compile_mode) # type: ignore
            self.adv_module = torch.compile(self.adv_module, mode=compile_mode) # type: ignore

    def update(self, batch, num_network_updates) -> Tuple[TensorDict, int]:
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

    def predict(self, obs: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Predict action given observation"""
        obs = torch.as_tensor([obs], device=self.device)
        self.policy.eval()
        with torch.inference_mode():
            td = TensorDict(
                {"observation": obs},
                batch_size=[1],
                device=self.policy.device,
            )
            output = self.policy(td).get("action")

        return output

    def _construct_trainer(self) -> None: # type: ignore
        """Override to return None since we implement custom training loop"""
        return None

    def train(self) -> None: # type: ignore
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
        cfg_loss_ppo_epochs = self.config.loss.epochs
        cfg_optim_lr = torch.tensor(self.config.optim.lr, device=self.device)
        cfg_loss_anneal_clip_eps = self.config.loss.anneal_clip_epsilon
        cfg_loss_clip_epsilon = self.config.loss.clip_epsilon

        losses = TensorDict(batch_size=[cfg_loss_ppo_epochs, num_mini_batches])  # type: ignore

        self.collector: SyncDataCollector | MultiSyncDataCollector
        collector_iter = iter(self.collector)
        total_iter = len(self.collector)
        for i in range(total_iter):
            timeit.printevery(
                1000, total_iter, erase=True
            )  # Changed from 1 to 1000 to match ppo_mujoco.py

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
                metrics_to_log.update(
                    {
                        "train/reward": episode_rewards.mean().item(),
                        "train/episode_length": episode_length.sum().item()
                        / len(episode_length),
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
                            loss, num_network_updates = self.update( # type: ignore
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

            # Log metrics
            if self.logger:
                metrics_to_log.update(timeit.todict(prefix="time"))  # type: ignore
                metrics_to_log["time/speed"] = pbar.format_dict["rate"]
                for key, value in metrics_to_log.items():
                    self.logger.log_scalar(key, value, collected_frames)

            self.collector.update_policy_weights_()

        self.collector.shutdown()
