from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union
import time

from gymnasium import RewardWrapper
import hydra
import numpy as np
from scipy import optimize
import torch
import tqdm


import torch.cuda
from tensordict import TensorDict
from tensordict.utils import expand_as_right, NestedKey
from torchrl._utils import logger as torchrl_logger
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.record.loggers import generate_exp_name, get_logger
from tensordict.nn import InteractionType, TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn, optim
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictPrioritizedReplayBuffer, TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import LazyMemmapStorage
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator
from torchrl.modules.distributions import TanhNormal
from torchrl.objectives import SoftUpdate
from torchrl.objectives.sac import SACLoss


from utils import (
    dump_video,
    log_metrics,
    make_environment,
    get_activation,
    save_model,
    load_model,
)

# ====================================================================
# Collector and replay buffer
# ---------------------------


def make_collector(cfg, train_env, actor_model_explore):
    """Make collector."""
    collector = SyncDataCollector(
        train_env,
        actor_model_explore,
        init_random_frames=cfg.collector.init_random_frames,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        device=cfg.collector.device,
    )
    collector.set_seed(cfg.env.seed)
    return collector


def make_replay_buffer(
    batch_size,
    prb=False,
    buffer_size=1000000,
    scratch_dir=None,
    device="cpu",
    prefetch=3,
):
    
    ## million steps half cheetah, load trained agent and collect data using replay buffer, load replay buffer for demonstration
    if prb:
        replay_buffer = TensorDictPrioritizedReplayBuffer(
            alpha=0.7,
            beta=0.5,
            pin_memory=False,
            prefetch=prefetch,
            storage=LazyMemmapStorage(
                buffer_size,
                scratch_dir=scratch_dir,
                device=device,  # type: ignore
            ),
            batch_size=batch_size,
        )
    else:
        replay_buffer = TensorDictReplayBuffer(
            pin_memory=False,
            prefetch=prefetch,
            storage=LazyMemmapStorage(
                buffer_size,
                scratch_dir=scratch_dir,
                device=device,  # type: ignore
            ),
            batch_size=batch_size,
        )
    return replay_buffer


####### Expert Buffer


def load_expert_data(data_path, device="cpu", batch_size=None) -> TensorDict:
    """
    Load expert demonstrations from a file into a TensorDict.
    """
    expert_data = torch.load(data_path, map_location=device)

    # Create Tensor Dict
    if not isinstance(expert_data, TensorDict):
        if isinstance(expert_data, dict):
            expert_data = TensorDict(
                expert_data, batch_size=[len(next(iter(expert_data.values())))]
            )

    if batch_size is not None:
        expert_data = expert_data.reshape(-1)  #
        if len(expert_data) < batch_size:
            # Repeat data if needed
            num_repeats = (batch_size + len(expert_data) - 1) // len(expert_data)
            expert_data = expert_data.repeat(num_repeats)
        expert_data = expert_data[:batch_size]  # Trim to exact batch size

    return expert_data


def create_expert_replay_buffer(
    expert_data: TensorDict, buffer_size: int, batch_size: int, device
) -> TensorDictReplayBuffer:
    """
    Create a replay buffer from expert demonstrations.

    TensorDictReplayBuffer containing expert demonstrations
    """
    expert_buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(
            buffer_size,
            device=device,
        ),
        batch_size=batch_size,
    )

    expert_buffer.extend(expert_data)

    return expert_buffer


class DeterministicRewardEstimator(TensorDictModule):
    """General class for value functions in RL.

    The DeterministicRewardEstimator class comes with default values for the in_keys and
    out_keys arguments (["observation"] and ["state_value"] or
    ["state_action_value"], respectively and depending on whether the "action"
    key is part of the in_keys list).

    Args:
        module (nn.Module): a :class:`torch.nn.Module` used to map the input to
            the output parameter space.
        in_keys (iterable of str, optional): keys to be read from input
            tensordict and passed to the module. If it
            contains more than one element, the values will be passed in the
            order given by the in_keys iterable.
            Defaults to ``["observation"]``.
        out_keys (iterable of str): keys to be written to the input tensordict.
            The length of out_keys must match the
            number of tensors returned by the embedded module. Using "_" as a
            key avoid writing tensor to output.
            Defaults to ``["state_value"]`` or
            ``["state_action_value"]`` if ``"action"`` is part of the ``in_keys``.

    """

    def __init__(
        self,
        module: nn.Module,
        in_keys: Optional[Sequence[NestedKey]] = None,  # type: ignore
        out_keys: Optional[Sequence[NestedKey]] = None,  # type: ignore
    ) -> None:
        if in_keys is None:
            in_keys = ["observation"]
        if out_keys is None:
            out_keys = (
                ["state_reward"] if "action" not in in_keys else ["state_action_reward"]
            )
        super().__init__(
            module=module,
            in_keys=in_keys,
            out_keys=out_keys,
        )


# ====================================================================
# Model
# -----


def make_ipmd_agent(cfg, train_env, eval_env, device):
    """Make SAC agent."""
    # Define Actor Network
    in_keys = ["observation"]
    action_spec = train_env.action_spec
    if train_env.batch_size:
        action_spec = action_spec[(0,) * len(train_env.batch_size)]
    actor_net_kwargs = {
        "num_cells": cfg.network.hidden_sizes,
        "out_features": 2 * action_spec.shape[-1],
        "activation_class": get_activation(cfg),
    }

    actor_net = MLP(**actor_net_kwargs)

    dist_class = TanhNormal
    dist_kwargs = {
        "low": action_spec.space.low,
        "high": action_spec.space.high,
        "tanh_loc": False,
    }

    actor_extractor = NormalParamExtractor(
        scale_mapping=f"biased_softplus_{cfg.network.default_policy_scale}",
        scale_lb=cfg.network.scale_lb,
    )
    actor_net = nn.Sequential(actor_net, actor_extractor)

    in_keys_actor = in_keys
    actor_module = TensorDictModule(
        actor_net,
        in_keys=in_keys_actor,
        out_keys=[
            "loc",
            "scale",
        ],
    )
    actor = ProbabilisticActor(
        spec=action_spec,
        in_keys=["loc", "scale"],
        module=actor_module,
        distribution_class=dist_class,
        distribution_kwargs=dist_kwargs,
        default_interaction_type=InteractionType.RANDOM,
        return_log_prob=False,
    )

    # Define Critic Network
    qvalue_net_kwargs = {
        "num_cells": cfg.network.hidden_sizes,
        "out_features": 1,
        "activation_class": get_activation(cfg),
    }

    qvalue_net = MLP(
        **qvalue_net_kwargs,
    )

    qvalue = ValueOperator(
        in_keys=["action"] + in_keys,
        module=qvalue_net,
    )

    # Define Reward Network
    reward_net_kwargs = {
        "num_cells": cfg.network.hidden_sizes,
        "out_features": 1,
        "activation_class": get_activation(cfg),
    }

    reward_net = MLP(
        **reward_net_kwargs,
    )

    reward_estimator = DeterministicRewardEstimator(
        in_keys=["action"] + in_keys, module=reward_net, out_keys=["estimated_reward"]
    )

    model = nn.ModuleList([actor, qvalue, reward_estimator]).to(device)

    # init nets
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):  # type: ignore
        td = eval_env.fake_tensordict()
        td = td.to(device)
        for net in model:
            net(td)
    return model, model[0]


# ====================================================================
# SAC Loss
# ---------

from utils import SACLossWithRewardEstimation


def make_loss_module(cfg, model):
    """Make loss module and target network updater."""
    # Create SAC loss
    # TODO: rewrite the loss module to take reward estimation.
    # 1. Add reward estimation to the loss module (inherit the SACLoss and modify some functions, put it in the utils.py)
    # 2. When the loss module takes in data (sampled tensordict), replace the reward with the estimated reward
    # 3. Takes expoert's sampled tensordict data and compute the rewards for the sampled data
    # 4. Compute the loss with the estimated reward, i.e., average of the estimated rewards of the current batch minus the estimated rewards of the expert batch, with some regularizer.
    loss_module = SACLossWithRewardEstimation(
        actor_network=model[0],
        qvalue_network=model[1],
        reward_network=model[2],
        num_qvalue_nets=2,
        loss_function=cfg.optim.loss_function,
        delay_actor=False,
        delay_qvalue=True,
        alpha_init=cfg.optim.alpha_init,
    )
    loss_module.make_value_estimator(gamma=cfg.optim.gamma)

    # Define Target Network Updater
    target_net_updater = SoftUpdate(loss_module, eps=cfg.optim.target_update_polyak)
    return loss_module, target_net_updater


def split_critic_params(critic_params):
    critic1_params = []
    critic2_params = []

    for param in critic_params:
        data1, data2 = param.data.chunk(2, dim=0)
        critic1_params.append(nn.Parameter(data1))
        critic2_params.append(nn.Parameter(data2))
    return critic1_params, critic2_params


def make_ipmd_optimizer(cfg, loss_module):
    critic_params = list(loss_module.qvalue_network_params.flatten_keys().values())
    actor_params = list(loss_module.actor_network_params.flatten_keys().values())
    reward_params = list(loss_module.reward_network_params.flatten_keys().values())

    optimizer_actor = optim.Adam(
        actor_params,
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        eps=cfg.optim.adam_eps,
    )
    optimizer_critic = optim.Adam(
        critic_params,
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        eps=cfg.optim.adam_eps,
    )
    optimizer_alpha = optim.Adam(
        [loss_module.log_alpha],
        lr=3.0e-4,
    )
    optimizer_reward = optim.Adam(
        reward_params,
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        eps=cfg.optim.adam_eps,
    )
    return optimizer_actor, optimizer_critic, optimizer_alpha, optimizer_reward


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
        bc_lambda=0.1,
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
        # TODO: do th.no_grad() here for critic update
        # Estimate rewards for current batch and use them directly

        with torch.no_grad():
            estimated_rewards = self.estimate_rewards(tensordict)
            tensordict.set("reward", estimated_rewards)

        # Compute standard SAC loss with estimated rewards
        loss_dict = super().forward(tensordict)

        # If expert data is provided, compute reward estimation and BC losses
        if expert_tensordict is not None:
            # Compute reward estimation loss
            expert_estimated_reward = self.estimate_rewards(expert_tensordict)
            reward_loss = self.compute_reward_loss(
                estimated_rewards, expert_estimated_reward
            )
            loss_dict["reward_estimation_loss"] = reward_loss  # Renamed for clarity

            # Compute BC loss
            expert_states = expert_tensordict.get("observation")
            expert_actions = expert_tensordict.get("action")

            # Get policy distribution for expert states
            policy_dist = self.actor_network(expert_states)

            # TODO: get rid of MSE
            # Compute BC loss based on policy type
            if hasattr(policy_dist, "log_prob"):
                # For stochastic policies
                bc_loss = -policy_dist.log_prob(expert_actions).mean()

            loss_dict["bc_loss"] = bc_loss

            # Add both losses to total loss
            loss_dict["loss"] = (
                loss_dict["loss"] + reward_loss + self.bc_lambda * bc_loss
            )

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
        estimated_rewards = self.reward_network(input_features)

        return estimated_rewards

    def compute_reward_loss(
        self,
        policy_estimated_reward: torch.Tensor,
        expert_estimated_reward: torch.Tensor,
    ) -> torch.Tensor:
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

        # TODO: subtraction instead of MSE
        # Main loss: difference between policy and expert estimated rewards
        reward_diff_loss = mean_policy_reward - mean_expert_reward
    

        # Add regularization term to prevent reward exploitation
        regularization = self.reward_regularizer * (
            torch.abs(policy_estimated_reward).mean()
            + torch.abs(expert_estimated_reward).mean()
        )

        return reward_diff_loss + regularization


@hydra.main(version_base=None, config_path="", config_name="config")
def train_ipmd(cfg: "DictConfig"):  # noqa: F821 # type: ignore
    device = cfg.network.device
    if device in ("", None):
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        elif torch.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    device = torch.device(device)

    # Create logger
    exp_name = generate_exp_name("IPMD", cfg.logger.exp_name)
    logger = None
    if cfg.logger.backend:
        logger = get_logger(
            logger_type=cfg.logger.backend,
            logger_name="sac_logging",
            experiment_name=exp_name,
            wandb_kwargs={
                "mode": cfg.logger.mode,
                "config": dict(cfg),
                "project": cfg.logger.project_name,
                "group": cfg.logger.group_name,
            },
        )

    torch.manual_seed(cfg.env.seed)
    np.random.seed(cfg.env.seed)

    # Create environments
    train_env, eval_env = make_environment(cfg, logger=logger)

    # Create agent
    model, exploration_policy = make_ipmd_agent(cfg, train_env, eval_env, device)

    # Create IPMD loss
    loss_module, target_net_updater = make_loss_module(cfg, model)

    # Create off-policy collector
    collector = make_collector(cfg, train_env, exploration_policy)

    # Create replay buffer
    replay_buffer = make_replay_buffer(
        batch_size=cfg.optim.batch_size,
        prb=cfg.replay_buffer.prb,
        buffer_size=cfg.replay_buffer.size,
        scratch_dir=cfg.replay_buffer.scratch_dir,
        device="cpu",
    )

    # Create optimizers
    (
        optimizer_actor,
        optimizer_critic,
        optimizer_alpha,
        optimizer_reward,
    ) = make_ipmd_optimizer(cfg, loss_module)

    # Main loop
    start_time = time.time()
    collected_frames = 0
    pbar = tqdm.tqdm(total=cfg.collector.total_frames)

    init_random_frames = cfg.collector.init_random_frames
    num_updates = int(
        cfg.collector.env_per_collector
        * cfg.collector.frames_per_batch
        * cfg.optim.utd_ratio
    )
    # prb = cfg.replay_buffer.prb
    eval_iter = cfg.logger.eval_iter
    frames_per_batch = cfg.collector.frames_per_batch
    eval_rollout_steps = cfg.env.max_episode_steps

    sampling_start = time.time()
    for i, tensordict in enumerate(collector):
        sampling_time = time.time() - sampling_start

        # Update weights of the inference policy
        collector.update_policy_weights_()

        pbar.update(tensordict.numel())

        tensordict = tensordict.reshape(-1)
        current_frames = tensordict.numel()
        # Add to replay buffer
        replay_buffer.extend(tensordict.cpu())

        collected_frames += current_frames

        # Optimization steps
        training_start = time.time()
        if collected_frames >= init_random_frames:
            losses = TensorDict({}, batch_size=[num_updates])
            for i in range(num_updates):
                # Sample from replay buffer
                sampled_tensordict = replay_buffer.sample()
                if sampled_tensordict.device != device:
                    sampled_tensordict = sampled_tensordict.to(
                        device, non_blocking=True
                    )
                else:
                    sampled_tensordict = sampled_tensordict.clone()

                # Compute loss
                loss_td = loss_module(sampled_tensordict)

                actor_loss = loss_td["loss_actor"]
                q_loss = loss_td["loss_qvalue"]
                alpha_loss = loss_td["loss_alpha"]
                reward_loss = loss_td["loss_reward"]

                # Update actor
                optimizer_actor.zero_grad()
                actor_loss.backward()
                optimizer_actor.step()

                # Update critic
                optimizer_critic.zero_grad()
                q_loss.backward()
                optimizer_critic.step()

                # Update alpha
                optimizer_alpha.zero_grad()
                alpha_loss.backward()
                optimizer_alpha.step()

                # Update reward

                optimizer_reward.zero_grad()
                reward_loss.backward()
                optimizer_reward.step()

                losses[i] = loss_td.select(
                    "loss_actor", "loss_qvalue", "loss_alpha", "loss_reward"
                ).detach()

                # Update qnet_target params
                target_net_updater.step()

        training_time = time.time() - training_start
        episode_end = (
            tensordict["next", "done"]
            if tensordict["next", "done"].any()
            else tensordict["next", "truncated"]
        )
        episode_rewards = tensordict["next", "episode_reward"][episode_end]

        # Logging
        metrics_to_log = {}
        if len(episode_rewards) > 0:
            episode_length = tensordict["next", "step_count"][episode_end]
            metrics_to_log["train/reward"] = episode_rewards.mean().item()
            metrics_to_log["train/episode_length"] = episode_length.sum().item() / len(
                episode_length
            )
        if collected_frames >= init_random_frames:
            metrics_to_log["train/q_loss"] = losses.get("loss_qvalue").mean().item()
            metrics_to_log["train/actor_loss"] = losses.get("loss_actor").mean().item()
            metrics_to_log["train/alpha_loss"] = losses.get("loss_alpha").mean().item()
            metrics_to_log["train/reward_loss"] = (
                losses.get("loss_reward").mean().item(),
            )
            metrics_to_log["train/alpha"] = loss_td["alpha"].item()
            metrics_to_log["train/entropy"] = loss_td["entropy"].item()
            metrics_to_log["train/sampling_time"] = sampling_time
            metrics_to_log["train/training_time"] = training_time

        # Evaluation
        if abs(collected_frames % eval_iter) < frames_per_batch:
            with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():  # type: ignore
                eval_start = time.time()
                eval_rollout = eval_env.rollout(
                    eval_rollout_steps,
                    model[0],
                    auto_cast_to_device=True,
                    break_when_any_done=True,
                )
                eval_env.apply(dump_video)
                eval_time = time.time() - eval_start
                eval_reward = eval_rollout["next", "reward"].sum(-2).mean().item()  # type: ignore
                metrics_to_log["eval/reward"] = eval_reward
                metrics_to_log["eval/time"] = eval_time
        if logger is not None:
            log_metrics(logger, metrics_to_log, collected_frames)
        sampling_start = time.time()

    collector.shutdown()
    if not eval_env.is_closed:
        eval_env.close()
    if not train_env.is_closed:
        train_env.close()
    end_time = time.time()
    execution_time = end_time - start_time
    torchrl_logger.info(f"Training took {execution_time:.2f} seconds to finish")


if __name__ == "__main__":
    train_ipmd()
