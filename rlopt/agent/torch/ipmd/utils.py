# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import functools

import torch
from torch import nn
from torchrl.envs import (
    CatTensors,
    Compose,
    DMControlEnv,
    DoubleToFloat,
    EnvCreator,
    ParallelEnv,
    TransformedEnv,
)

from torchrl.envs.libs.gym import GymEnv, set_gym_backend
from torchrl.envs.transforms import InitTracker, RewardSum, StepCounter

from torchrl.record import VideoRecorder

from torchrl.objectives import SACLoss
import torch
import torch.nn.functional as F
from tensordict.tensordict import TensorDict



# ====================================================================
# SAC Class
# -----------------


class SACLossWithRewardEstimation(SACLoss):
    """SAC loss module with IRL reward estimation and behavioral cloning."""
    
    def __init__(
        self,
        actor_network,
        qvalue_network,
        reward_network,
        num_qvalue_nets=2,
        loss_function="l2",
        delay_actor=False,
        delay_qvalue=True,
        alpha_init=1.0,
        reward_regularizer=0.01,
        bc_lambda=0.1
    ):
        super().__init__(
            actor_network=actor_network,
            qvalue_network=qvalue_network,
            num_qvalue_nets=num_qvalue_nets,
            loss_function=loss_function,
            delay_actor=delay_actor,
            delay_qvalue=delay_qvalue,
            alpha_init=alpha_init,
        )
        self.reward_network = reward_network
        self.reward_regularizer = reward_regularizer
        self.bc_lambda = bc_lambda

    def forward(self, tensordict: TensorDict, expert_tensordict: TensorDict = None):
        """
        Compute the loss with IRL reward estimation and behavioral cloning.
        
        Args:
            tensordict (TensorDict): Current policy tensordict
            expert_tensordict (TensorDict, optional): Expert demonstrations tensordict
        """
        # Estimate rewards for current batch and use them directly
        policy_estimated_reward = self.estimate_rewards(tensordict)
        tensordict.set("estimated_reward", policy_estimated_reward)  # Changed key to estimated_reward
        
        # Compute standard SAC loss with estimated rewards
        loss_dict = super().forward(tensordict)
        
        # If expert data is provided, compute reward estimation and BC losses
        if expert_tensordict is not None:
            # Compute reward estimation loss
            expert_estimated_reward = self.estimate_rewards(expert_tensordict)
            reward_loss = self.compute_reward_loss(policy_estimated_reward, expert_estimated_reward)
            loss_dict["reward_estimation_loss"] = reward_loss  # Renamed for clarity
            
            # Compute BC loss
            expert_states = expert_tensordict.get("observation")
            expert_actions = expert_tensordict.get("action")
            
            # Get policy distribution for expert states
            policy_dist = self.actor_network(expert_states)
            
            # Compute BC loss based on policy type
            if hasattr(policy_dist, "log_prob"):
                # For stochastic policies
                bc_loss = -policy_dist.log_prob(expert_actions).mean()
            else:
                # For deterministic policies, use MSE
                pred_actions = policy_dist
                bc_loss = F.mse_loss(pred_actions, expert_actions)
            
            loss_dict["bc_loss"] = bc_loss
            
            # Add both losses to total loss
            loss_dict["loss"] = loss_dict["loss"] + reward_loss + self.bc_lambda * bc_loss
            
        return loss_dict
    
    def estimate_rewards(self, tensordict: TensorDict) -> torch.Tensor:
        """
        Estimate rewards using IRL reward network.
        
        Args:
            tensordict (TensorDict): Input tensordict containing state-action pairs
            
        Returns:
            torch.Tensor: Estimated rewards from IRL
        """
        states = tensordict.get("observation")
        actions = tensordict.get("action")
        
        # Concatenate states and actions
        input_features = torch.cat([states, actions], dim=-1)
        
        # Get reward estimates from IRL
        with torch.no_grad():
            estimated_rewards = self.reward_network(input_features)
        
        return estimated_rewards

    def compute_reward_loss(self, policy_estimated_reward: torch.Tensor, expert_estimated_reward: torch.Tensor) -> torch.Tensor:
        """
        Compute the IRL reward estimation loss.
        
        Args:
            policy_estimated_reward (torch.Tensor): Estimated rewards for current policy
            expert_estimated_reward (torch.Tensor): Estimated rewards for expert demonstrations
            
        Returns:
            torch.Tensor: IRL reward estimation loss
        """
        # Compute mean rewards
        mean_policy_reward = policy_estimated_reward.mean()
        mean_expert_reward = expert_estimated_reward.mean()
        
        # Main loss: difference between policy and expert estimated rewards
        reward_diff_loss = F.mse_loss(mean_policy_reward, mean_expert_reward)
        
        # Add regularization term to prevent reward exploitation
        regularization = self.reward_regularizer * (
            torch.abs(policy_estimated_reward).mean() + torch.abs(expert_estimated_reward).mean()
        )
        
        return reward_diff_loss + regularization


# ====================================================================
# Environment utils
# -----------------


def env_maker(cfg, device="cpu", from_pixels=False):
    lib = cfg.env.library
    if lib in ("gym", "gymnasium"):
        with set_gym_backend(lib):
            return GymEnv(
                cfg.env.name,
                device=device,
                from_pixels=from_pixels,
                pixels_only=False,
            )
    elif lib == "dm_control":
        env = DMControlEnv(
            cfg.env.name, cfg.env.task, from_pixels=from_pixels, pixels_only=False
        )
        return TransformedEnv(
            env, CatTensors(in_keys=env.observation_spec.keys(), out_key="observation")  # type: ignore
        )
    else:
        raise NotImplementedError(f"Unknown lib {lib}.")


def apply_env_transforms(env, max_episode_steps=1000):
    transformed_env = TransformedEnv(
        env,
        Compose(
            InitTracker(),
            StepCounter(max_episode_steps),
            DoubleToFloat(),
            RewardSum(),
        ),
    )
    return transformed_env


def make_environment(cfg, logger=None):
    """Make environments for training and evaluation."""
    partial = functools.partial(env_maker, cfg=cfg)
    parallel_env = ParallelEnv(
        cfg.collector.env_per_collector,
        EnvCreator(partial),
        serial_for_single=True,
    )
    parallel_env.set_seed(cfg.env.seed)

    train_env = apply_env_transforms(parallel_env, cfg.env.max_episode_steps)

    partial = functools.partial(env_maker, cfg=cfg, from_pixels=cfg.logger.video)
    trsf_clone = train_env.transform.clone()
    if cfg.logger.video:
        trsf_clone.insert(
            0, VideoRecorder(logger, tag="rendering/test", in_keys=["pixels"])  # type: ignore
        )
    eval_env = TransformedEnv(
        ParallelEnv(
            1,
            EnvCreator(partial),
            serial_for_single=True,
        ),
        trsf_clone,
    )
    return train_env, eval_env


# ====================================================================
# General utils
# ---------


### Saving/loading models

def save_model(model, path):
    torch.save({
        "actor": model[0].state_dict(),
        "critic": model[1].state_dict(),
        "reward_estimate": model[2].state_dict(),
    }, path)


def load_model(model, path, device):
    states = torch.load(path, map_location=device)
    model[0].load_state_dict(states["actor"])
    model[1].load_state_dict(states["critic"])
    model[2].load_state_dict(states["reward_estimate"])




def log_metrics(logger, metrics, step):
    for metric_name, metric_value in metrics.items():
        logger.log_scalar(metric_name, metric_value, step)


def get_activation(cfg):
    if cfg.network.activation == "relu":
        return nn.ReLU
    elif cfg.network.activation == "tanh":
        return nn.Tanh
    elif cfg.network.activation == "leaky_relu":
        return nn.LeakyReLU
    else:
        raise NotImplementedError


def dump_video(module):
    if isinstance(module, VideoRecorder):
        module.dump()
