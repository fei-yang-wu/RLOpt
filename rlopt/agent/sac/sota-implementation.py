# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import functools

import torch
from tensordict.nn import InteractionType, TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn, optim
from torchrl.collectors import aSyncDataCollector, SyncDataCollector
from torchrl.data import (
    LazyMemmapStorage,
    LazyTensorStorage,
    TensorDictPrioritizedReplayBuffer,
    TensorDictReplayBuffer,
)
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
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator
from torchrl.modules.distributions import TanhNormal
from torchrl.objectives import SoftUpdate
from torchrl.objectives.sac import SACLoss
from torchrl.record import VideoRecorder

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
            env, CatTensors(in_keys=env.observation_spec.keys(), out_key="observation")
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
            0, VideoRecorder(logger, tag="rendering/test", in_keys=["pixels"])
        )
    eval_env = TransformedEnv(
        ParallelEnv(
            cfg.collector.env_per_collector,
            EnvCreator(partial),
            serial_for_single=True,
        ),
        trsf_clone,
    )
    return train_env, eval_env


def make_train_environment(cfg):
    """Make environments for training and evaluation."""
    partial = functools.partial(env_maker, cfg=cfg)
    parallel_env = ParallelEnv(
        cfg.collector.env_per_collector,
        EnvCreator(partial),
        serial_for_single=True,
    )
    parallel_env.set_seed(cfg.env.seed)

    train_env = apply_env_transforms(parallel_env, cfg.env.max_episode_steps)

    return train_env


# ====================================================================
# Collector and replay buffer
# ---------------------------


def make_collector(cfg, train_env, actor_model_explore, compile_mode):
    """Make collector."""
    device = cfg.collector.device
    if device in ("", None):
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    collector = SyncDataCollector(
        train_env,
        actor_model_explore,
        init_random_frames=cfg.collector.init_random_frames,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        device=device,
        compile_policy={"mode": compile_mode} if compile_mode else False,
        cudagraph_policy={"warmup": 10} if cfg.compile.cudagraphs else False,
    )
    collector.set_seed(cfg.env.seed)
    return collector


def flatten(td):
    return td.reshape(-1)


def make_collector_async(
    cfg, train_env_make, actor_model_explore, compile_mode, replay_buffer
):
    """Make async collector."""
    device = cfg.collector.device
    if device in ("", None):
        if torch.cuda.is_available():
            if torch.cuda.device_count() < 2:
                raise RuntimeError("Requires >= 2 GPUs")
            device = torch.device("cuda:1")
        else:
            device = torch.device("cpu")

    collector = aSyncDataCollector(
        train_env_make,
        actor_model_explore,
        init_random_frames=0,  # Currently not supported, but accounted for in script: cfg.collector.init_random_frames,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        device=device,
        env_device=torch.device("cpu"),
        compile_policy={"mode": compile_mode, "warmup": 5} if compile_mode else False,
        cudagraph_policy={"warmup": 20} if cfg.compile.cudagraphs else False,
        replay_buffer=replay_buffer,
        extend_buffer=True,
        postproc=flatten,
        no_cuda_sync=True,
    )
    collector.set_seed(cfg.env.seed)
    collector.start()
    return collector


def make_replay_buffer(
    batch_size,
    prb=False,
    buffer_size=1000000,
    scratch_dir=None,
    device="cpu",
    prefetch=3,
    shared: bool = False,
):
    storage_cls = (
        functools.partial(LazyTensorStorage, device=device)
        if not scratch_dir
        else functools.partial(LazyMemmapStorage, device="cpu", scratch_dir=scratch_dir)
    )
    if prb:
        replay_buffer = TensorDictPrioritizedReplayBuffer(
            alpha=0.7,
            beta=0.5,
            pin_memory=False,
            prefetch=prefetch,
            storage=storage_cls(
                buffer_size,
            ),
            batch_size=batch_size,
            shared=shared,
        )
    else:
        replay_buffer = TensorDictReplayBuffer(
            pin_memory=False,
            prefetch=prefetch,
            storage=storage_cls(
                buffer_size,
            ),
            batch_size=batch_size,
            shared=shared,
        )
    if scratch_dir:
        replay_buffer.append_transform(lambda td: td.to(device))
    return replay_buffer


# ====================================================================
# Model
# -----


def make_sac_agent(cfg, train_env, eval_env, device):
    """Make SAC agent."""
    # Define Actor Network
    in_keys = ["observation"]
    action_spec = train_env.action_spec_unbatched.to(device)

    actor_net = MLP(
        num_cells=cfg.network.hidden_sizes,
        out_features=2 * action_spec.shape[-1],
        activation_class=get_activation(cfg),
        device=device,
    )

    dist_class = TanhNormal
    dist_kwargs = {
        "low": action_spec.space.low,
        "high": action_spec.space.high,
        "tanh_loc": False,
    }

    actor_extractor = NormalParamExtractor(
        scale_mapping=f"biased_softplus_{cfg.network.default_policy_scale}",
        scale_lb=cfg.network.scale_lb,
    ).to(device)
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
    qvalue_net = MLP(
        num_cells=cfg.network.hidden_sizes,
        out_features=1,
        activation_class=get_activation(cfg),
        device=device,
    )

    qvalue = ValueOperator(
        in_keys=["action"] + in_keys,
        module=qvalue_net,
    )

    model = nn.ModuleList([actor, qvalue])

    # init nets
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        td = eval_env.fake_tensordict()
        td = td.to(device)
        for net in model:
            net(td)
    return model, model[0]


# ====================================================================
# SAC Loss
# ---------


def make_loss_module(cfg, model):
    """Make loss module and target network updater."""
    # Create SAC loss
    loss_module = SACLoss(
        actor_network=model[0],
        qvalue_network=model[1],
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


def make_sac_optimizer(cfg, loss_module):
    critic_params = list(loss_module.qvalue_network_params.flatten_keys().values())
    actor_params = list(loss_module.actor_network_params.flatten_keys().values())

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
    return optimizer_actor, optimizer_critic, optimizer_alpha


# ====================================================================
# General utils
# ---------


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


# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""SAC Example.

This is a simple self-contained example of a SAC training script.

It supports state environments like MuJoCo.

The helper functions are coded in the utils.py associated with this script.
"""
from __future__ import annotations

import warnings

import hydra
import numpy as np
import torch
import torch.cuda
import tqdm
from tensordict import TensorDict
from tensordict.nn import CudaGraphModule
from torchrl._utils import compile_with_warmup, timeit
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.objectives import group_optimizers
from torchrl.record.loggers import generate_exp_name, get_logger
from utils import (
    dump_video,
    log_metrics,
    make_collector,
    make_environment,
    make_loss_module,
    make_replay_buffer,
    make_sac_agent,
    make_sac_optimizer,
)

torch.set_float32_matmul_precision("high")


@hydra.main(version_base="1.1", config_path="", config_name="config")
def main(cfg: DictConfig):  # noqa: F821
    device = cfg.network.device
    if device in ("", None):
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    device = torch.device(device)

    # Create logger
    exp_name = generate_exp_name("SAC", cfg.logger.exp_name)
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
    model, exploration_policy = make_sac_agent(cfg, train_env, eval_env, device)

    # Create SAC loss
    loss_module, target_net_updater = make_loss_module(cfg, model)

    compile_mode = None
    if cfg.compile.compile:
        compile_mode = cfg.compile.compile_mode
        if compile_mode in ("", None):
            if cfg.compile.cudagraphs:
                compile_mode = "default"
            else:
                compile_mode = "reduce-overhead"

    # Create off-policy collector
    collector = make_collector(
        cfg, train_env, exploration_policy, compile_mode=compile_mode
    )

    # Create replay buffer
    replay_buffer = make_replay_buffer(
        batch_size=cfg.optim.batch_size,
        prb=cfg.replay_buffer.prb,
        buffer_size=cfg.replay_buffer.size,
        scratch_dir=cfg.replay_buffer.scratch_dir,
        device=device,
    )

    # Create optimizers
    (
        optimizer_actor,
        optimizer_critic,
        optimizer_alpha,
    ) = make_sac_optimizer(cfg, loss_module)
    optimizer = group_optimizers(optimizer_actor, optimizer_critic, optimizer_alpha)
    del optimizer_actor, optimizer_critic, optimizer_alpha

    def update(sampled_tensordict):
        # Compute loss
        loss_td = loss_module(sampled_tensordict)

        actor_loss = loss_td["loss_actor"]
        q_loss = loss_td["loss_qvalue"]
        alpha_loss = loss_td["loss_alpha"]

        (actor_loss + q_loss + alpha_loss).sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Update qnet_target params
        target_net_updater.step()
        return loss_td.detach()

    if cfg.compile.compile:
        update = compile_with_warmup(update, mode=compile_mode, warmup=1)

    if cfg.compile.cudagraphs:
        warnings.warn(
            "CudaGraphModule is experimental and may lead to silently wrong results. Use with caution.",
            category=UserWarning,
        )
        update = CudaGraphModule(update, in_keys=[], out_keys=[], warmup=5)

    # Main loop
    collected_frames = 0
    pbar = tqdm.tqdm(total=cfg.collector.total_frames)

    init_random_frames = cfg.collector.init_random_frames
    num_updates = int(cfg.collector.frames_per_batch * cfg.optim.utd_ratio)
    prb = cfg.replay_buffer.prb
    eval_iter = cfg.logger.eval_iter
    frames_per_batch = cfg.collector.frames_per_batch
    eval_rollout_steps = cfg.env.max_episode_steps

    collector_iter = iter(collector)
    total_iter = len(collector)

    for i in range(total_iter):
        timeit.printevery(num_prints=1000, total_count=total_iter, erase=True)

        with timeit("collect"):
            tensordict = next(collector_iter)

        # Update weights of the inference policy
        collector.update_policy_weights_()

        current_frames = tensordict.numel()
        pbar.update(current_frames)

        with timeit("rb - extend"):
            # Add to replay buffer
            tensordict = tensordict.reshape(-1)
            replay_buffer.extend(tensordict)

        collected_frames += current_frames

        # Optimization steps
        with timeit("train"):
            if collected_frames >= init_random_frames:
                losses = TensorDict(batch_size=[num_updates])
                for i in range(num_updates):
                    with timeit("rb - sample"):
                        # Sample from replay buffer
                        sampled_tensordict = replay_buffer.sample()

                    with timeit("update"):
                        torch.compiler.cudagraph_mark_step_begin()
                        loss_td = update(sampled_tensordict).clone()
                    losses[i] = loss_td.select(
                        "loss_actor", "loss_qvalue", "loss_alpha"
                    )

                    # Update priority
                    if prb:
                        replay_buffer.update_priority(sampled_tensordict)

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
            metrics_to_log["train/reward"] = episode_rewards
            metrics_to_log["train/episode_length"] = episode_length.sum() / len(
                episode_length
            )
        if collected_frames >= init_random_frames:
            losses = losses.mean()
            metrics_to_log["train/q_loss"] = losses.get("loss_qvalue")
            metrics_to_log["train/actor_loss"] = losses.get("loss_actor")
            metrics_to_log["train/alpha_loss"] = losses.get("loss_alpha")
            metrics_to_log["train/alpha"] = loss_td["alpha"]
            metrics_to_log["train/entropy"] = loss_td["entropy"]

        # Evaluation
        if abs(collected_frames % eval_iter) < frames_per_batch:
            with (
                set_exploration_type(ExplorationType.DETERMINISTIC),
                torch.no_grad(),
                timeit("eval"),
            ):
                eval_rollout = eval_env.rollout(
                    eval_rollout_steps,
                    model[0],
                    auto_cast_to_device=True,
                    break_when_any_done=True,
                )
                eval_env.apply(dump_video)
                eval_reward = eval_rollout["next", "reward"].sum(-2).mean().item()
                metrics_to_log["eval/reward"] = eval_reward
        if logger is not None:
            metrics_to_log.update(timeit.todict(prefix="time"))
            metrics_to_log["time/speed"] = pbar.format_dict["rate"]
            log_metrics(logger, metrics_to_log, collected_frames)

    collector.shutdown()
    if not eval_env.is_closed:
        eval_env.close()
    if not train_env.is_closed:
        train_env.close()


if __name__ == "__main__":
    main()
