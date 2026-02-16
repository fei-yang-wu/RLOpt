from __future__ import annotations

import copy
import functools
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from tensordict.nn import (
    CudaGraphModule,
    TensorDictModule,
)
from torch import Tensor
from torchrl._utils import compile_with_warmup, timeit
from torchrl.data import (
    LazyMemmapStorage,
    LazyTensorStorage,
    ReplayBuffer,
    TensorDictReplayBuffer,
)
from torchrl.data.replay_buffers.samplers import RandomSampler
from torchrl.modules import ActorCriticOperator
from torchrl.record.loggers import Logger
from tqdm.std import tqdm as Tqdm

from rlopt.base_class import BaseAlgorithm
from rlopt.config_base import NetworkConfig, RLOptConfig
from rlopt.type_aliases import OptimizerClass
from rlopt.utils import log_info
from rlopt.agent.fast_td3.utils import EmpiricalNormalization


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class FastTD3Config:
    """FastTD3-specific configuration."""

    actor_hidden_dim: int = 512
    """Hidden dimension of the actor network."""

    critic_hidden_dim: int = 1024
    """Hidden dimension of the critic network."""

    num_steps: int = 1
    """Number of steps for multi-step return"""

    num_updates: int = 2
    """Number of updates per batch."""

    init_scale: float = 0.01
    """Scale of the initial actor output layer parameters."""

    num_atoms: int = 251
    """Number of atoms for the distributional critic (C51)."""

    v_min: float = -10.0
    """Minimum value of the distributional support."""

    v_max: float = 10.0
    """Maximum value of the distributional support."""

    use_cdq: bool = True
    """Whether to use Clipped Double Q-learning."""

    std_min: float = 0.001
    """Minimum scale of per-environment exploration noise."""

    std_max: float = 0.4
    """Maximum scale of per-environment exploration noise."""

    policy_noise: float = 0.001
    """Scale of target policy smoothing noise."""

    noise_clip: float = 0.5
    """Clipping range for target policy smoothing noise."""

    policy_frequency: int = 2
    """Frequency of delayed policy updates."""

    tau: float = 0.1
    """Soft target update coefficient."""

    gamma: float = 0.99
    """Discount factor."""

    disable_bootstrap: bool = False
    """Whether to disable bootstrap from truncated episodes."""

    batch_size: int = 8
    """Batch size per parallel environment."""

    norm_obs: bool = True


@dataclass
class FastTD3RLOptConfig(RLOptConfig):
    """FastTD3 configuration that extends RLOptConfig."""

    fasttd3: FastTD3Config = field(default_factory=FastTD3Config)
    """FastTD3-specific configuration."""

    def __post_init__(self):
        self.q_function = NetworkConfig(
            num_cells=[],
            activation_fn="elu",
            output_dim=1,
            input_keys=["policy"],
        )
        self.policy = NetworkConfig(
            num_cells=[],
            activation_fn="elu",
            output_dim=1,
            input_keys=["policy"],
        )


# ---------------------------------------------------------------------------
# Network modules
# ---------------------------------------------------------------------------


class FastTD3ActorNet(nn.Module):
    """Deterministic actor network for FastTD3.

    Architecture: Linear -> ReLU -> Linear -> ReLU -> Linear -> ReLU -> Linear -> Tanh
    """

    def __init__(
        self,
        n_obs: int,
        n_act: int,
        init_scale: float,
        hidden_dim: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.n_act = n_act
        self.net = nn.Sequential(
            nn.Linear(n_obs, hidden_dim, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4, device=device),
            nn.ReLU(),
        )
        self.fc_mu = nn.Sequential(
            nn.Linear(hidden_dim // 4, n_act, device=device),
            nn.Tanh(),
        )
        nn.init.normal_(self.fc_mu[0].weight, 0.0, init_scale)
        nn.init.constant_(self.fc_mu[0].bias, 0.0)

    def forward(self, obs: Tensor) -> Tensor:
        x = self.net(obs)
        return self.fc_mu(x)


class DistributionalQNetwork(nn.Module):
    """Single distributional Q-network (C51-style)."""

    def __init__(
        self,
        n_obs: int,
        n_act: int,
        num_atoms: int,
        v_min: float,
        v_max: float,
        hidden_dim: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs + n_act, hidden_dim, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, num_atoms, device=device),
        )
        self.v_min = v_min
        self.v_max = v_max
        self.num_atoms = num_atoms

    def forward(self, obs: Tensor, actions: Tensor) -> Tensor:
        x = torch.cat([obs, actions], 1)
        return self.net(x)

    def projection(
        self,
        obs: Tensor,
        actions: Tensor,
        rewards: Tensor,
        bootstrap: Tensor,
        discount: float,
        q_support: Tensor,
        device: torch.device,
    ) -> Tensor:
        delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        batch_size = rewards.shape[0]

        target_z = (
            rewards.unsqueeze(1)
            + bootstrap.unsqueeze(1) * discount.unsqueeze(1) * q_support
        )
        target_z = target_z.clamp(self.v_min, self.v_max)
        b = (target_z - self.v_min) / delta_z
        lower = torch.floor(b).long()
        upper = torch.ceil(b).long()

        is_int = lower == upper
        l_mask = is_int & (lower > 0)
        u_mask = is_int & (lower == 0)

        lower = torch.where(l_mask, lower - 1, lower)
        upper = torch.where(u_mask, upper + 1, upper)

        next_dist = F.softmax(self.forward(obs, actions), dim=1)
        proj_dist = torch.zeros_like(next_dist)
        offset = (
            torch.linspace(
                0, (batch_size - 1) * self.num_atoms, batch_size, device=device
            )
            .unsqueeze(1)
            .expand(batch_size, self.num_atoms)
            .long()
        )
        proj_dist.view(-1).index_add_(
            0, (lower + offset).view(-1), (next_dist * (upper.float() - b)).view(-1)
        )
        proj_dist.view(-1).index_add_(
            0, (upper + offset).view(-1), (next_dist * (b - lower.float())).view(-1)
        )
        return proj_dist


class DistributionalCritic(nn.Module):
    """Twin distributional Q-networks for clipped double Q-learning."""

    def __init__(
        self,
        n_obs: int,
        n_act: int,
        num_atoms: int,
        v_min: float,
        v_max: float,
        hidden_dim: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.qnet1 = DistributionalQNetwork(
            n_obs=n_obs,
            n_act=n_act,
            num_atoms=num_atoms,
            v_min=v_min,
            v_max=v_max,
            hidden_dim=hidden_dim,
            device=device,
        )
        self.qnet2 = DistributionalQNetwork(
            n_obs=n_obs,
            n_act=n_act,
            num_atoms=num_atoms,
            v_min=v_min,
            v_max=v_max,
            hidden_dim=hidden_dim,
            device=device,
        )
        self.register_buffer(
            "q_support", torch.linspace(v_min, v_max, num_atoms, device=device)
        )
        self._device = device

    def forward(
        self, obs: Tensor, actions: Tensor
    ) -> tuple[Tensor, Tensor]:
        return self.qnet1(obs, actions), self.qnet2(obs, actions)

    def projection(
        self,
        obs: Tensor,
        actions: Tensor,
        rewards: Tensor,
        bootstrap: Tensor,
        discount: float,
    ) -> tuple[Tensor, Tensor]:
        q1_proj = self.qnet1.projection(
            obs, actions, rewards, bootstrap, discount,
            self.q_support, self.q_support.device,
        )
        q2_proj = self.qnet2.projection(
            obs, actions, rewards, bootstrap, discount,
            self.q_support, self.q_support.device,
        )
        return q1_proj, q2_proj

    def get_value(self, probs: Tensor) -> Tensor:
        """Calculate expected value from probability distribution."""
        return torch.sum(probs * self.q_support, dim=1)


# ---------------------------------------------------------------------------
# Loss module
# ---------------------------------------------------------------------------


class FastTD3Loss(nn.Module):
    """Custom loss module for FastTD3 with distributional critics."""

    def __init__(
        self,
        actor: FastTD3ActorNet,
        critic: DistributionalCritic,
        target_critic: DistributionalCritic,
        gamma: float,
        policy_noise: float,
        noise_clip: float,
        use_cdq: bool,
        disable_bootstrap: bool,
    ):
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.target_critic = target_critic
        self.gamma = gamma
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.use_cdq = use_cdq
        self.disable_bootstrap = disable_bootstrap

    def critic_loss(
        self,
        obs: Tensor,
        actions: Tensor,
        next_obs: Tensor,
        rewards: Tensor,
        dones: Tensor,
        truncations: Tensor,
        effective_n_steps: Tensor, 
    ) -> Tensor:
        if self.disable_bootstrap:
            bootstrap = (~dones.bool()).float()
        else:
            bootstrap = (truncations.bool() | ~dones.bool()).float()

        # Target policy smoothing noise
        clipped_noise = (
            torch.randn_like(actions)
            .mul(self.policy_noise)
            .clamp(-self.noise_clip, self.noise_clip)
        )
        next_state_actions = (self.actor(next_obs) + clipped_noise).clamp(-1.0, 1.0)

        discount = self.gamma ** effective_n_steps # CHECK: shape

        with torch.no_grad():
            qf1_proj, qf2_proj = self.target_critic.projection(
                next_obs, next_state_actions, rewards, bootstrap, discount,
            )
            qf1_target_value = self.target_critic.get_value(qf1_proj)
            qf2_target_value = self.target_critic.get_value(qf2_proj)
            if self.use_cdq:
                qf_target_dist = torch.where(
                    qf1_target_value.unsqueeze(1) < qf2_target_value.unsqueeze(1),
                    qf1_proj,
                    qf2_proj,
                )
                qf1_target_dist = qf2_target_dist = qf_target_dist
            else:
                qf1_target_dist, qf2_target_dist = qf1_proj, qf2_proj

        qf1, qf2 = self.critic(obs, actions)
        qf1_loss = -torch.sum(
            qf1_target_dist * F.log_softmax(qf1, dim=1), dim=1
        ).mean()
        qf2_loss = -torch.sum(
            qf2_target_dist * F.log_softmax(qf2, dim=1), dim=1
        ).mean()
        return qf1_loss + qf2_loss

    def actor_loss(self, obs: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        actions = self.actor(obs)
        qf1, qf2 = self.critic(obs, actions)
        qf1_value = self.critic.get_value(F.softmax(qf1, dim=1))
        qf2_value = self.critic.get_value(F.softmax(qf2, dim=1))
        if self.use_cdq:
            qf_value = torch.minimum(qf1_value, qf2_value)
        else:
            qf_value = (qf1_value + qf2_value) / 2.0
        loss = -qf_value.mean()
        info = {
            "qf_value_mean": qf_value.mean().detach(),
            "qf_value_max": qf_value.max().detach(),
            "qf_value_min": qf_value.min().detach(),
        }
        return loss, info


# ---------------------------------------------------------------------------
# Main algorithm class
# ---------------------------------------------------------------------------


class FastTD3(BaseAlgorithm):
    """FastTD3: High-performance TD3 with distributional critics.

    Uses distributional Q-learning (C51-style), per-environment exploration
    noise, and clipped double Q-learning for robust off-policy training.
    """

    def __init__(
        self,
        env,
        config: FastTD3RLOptConfig,
        logger: Logger | None = None,
        **kwargs,
    ):
        # setup observation normalization
        if config.fasttd3.norm_obs:
            self.obs_normalizer = EmpiricalNormalization(
                shape=env.observation_spec[config.policy.input_keys[0]].shape[1:], 
                device=config.device, 
            )
        else:
            self.obs_normalizer = nn.Identity().to(config.device)
        self.obs_normalizer.train()
        
        super().__init__(
            env=env,
            config=config,
            logger=logger,
            **kwargs,
        )

        self.config = cast(FastTD3RLOptConfig, self.config)
        self.config: FastTD3RLOptConfig
        assert isinstance(self.config, FastTD3RLOptConfig)
        assert self.q_function, "FastTD3 requires a Q-function configuration."
        
        # Build target critic (deep copy of online critic)
        self.target_critic: DistributionalCritic = copy.deepcopy(self._critic_net)
        self.target_critic.requires_grad_(False)

        # Track update count for delayed policy updates
        self.total_updates = 0

        # Compile if requested
        if self.compile_mode:
            self.update = compile_with_warmup(
                self.update, mode=self.compile_mode, warmup=1
            )

        if self.config.compile.cudagraphs:
            self.update = CudaGraphModule(
                self.update,
                in_keys=[],
                out_keys=[],
                warmup=5,
            )

    def _construct_policy(self) -> TensorDictModule:
        action_spec = self.env.action_spec_unbatched
        key = self.config.policy.input_keys[0]
        action_dim = int(action_spec.shape[-1])
        obs_dim = int(self.env.observation_spec[key].shape[-1])
        cfg = self.config.fasttd3 if hasattr(self.config, "fasttd3") else FastTD3Config()

        self._actor_net = FastTD3ActorNet(
            n_obs=obs_dim,
            n_act=action_dim,
            init_scale=cfg.init_scale,
            hidden_dim=cfg.actor_hidden_dim,
            device=self.device,
        )

        return TensorDictModule(
            module=self._actor_net,
            in_keys=[key],
            out_keys=["action"],
        )

    def _init_exploration_noise(self) -> None:
        """Initialize per-environment exploration noise (called once on first collect step)."""
        cfg = self.config.fasttd3
        num_envs = self.config.env.num_envs
        self._noise_scales = torch.rand(
            num_envs, 1, device=self.device
        ) * (cfg.std_max - cfg.std_min) + cfg.std_min
        self._std_min = cfg.std_min
        self._std_max = cfg.std_max
        self._num_envs = num_envs

    def _resample_noise(self, dones: Tensor | None) -> None:
        """Resample exploration noise scales for environments that are done."""
        if dones is not None and dones.sum() > 0:
            new_scales = (
                torch.rand(self._num_envs, 1, device=self.device)
                * (self._std_max - self._std_min)
                + self._std_min
            )
            dones_view = dones.view(-1, 1) > 0
            self._noise_scales = torch.where(
                dones_view, new_scales, self._noise_scales
            )

    def _collect_step(self, obs: Tensor, step_count: Tensor) -> Tensor:
        """Collect action: normalize obs, run actor, add exploration noise."""
        if not hasattr(self, "_noise_scales"):
            self._init_exploration_noise()

        obs = self.obs_normalizer(obs)

        # Resample noise for done environments
        dones = (step_count == 0).bool() if step_count is not None else None
        self._resample_noise(dones)

        # Deterministic action + exploration noise
        action = self._actor_net(obs)
        noise = torch.randn_like(action) * self._noise_scales
        return action + noise

    @property
    def collector_policy(self) -> TensorDictModule:
        """Return exploration policy that adds per-env noise."""
        key = self.config.policy.input_keys[0]
        return TensorDictModule(
            module=self._collect_step,
            in_keys=[key, "step_count"],
            out_keys=["action"],
        )

    def _construct_q_function(self) -> TensorDictModule:
        key = self.config.q_function.input_keys[0]
        action_spec = self.env.action_spec_unbatched
        action_dim = int(action_spec.shape[-1])
        obs_dim = int(self.env.observation_spec[key].shape[-1])
        cfg = self.config.fasttd3 if hasattr(self.config, "fasttd3") else FastTD3Config()

        self._critic_net = DistributionalCritic(
            n_obs=obs_dim,
            n_act=action_dim,
            num_atoms=cfg.num_atoms,
            v_min=cfg.v_min,
            v_max=cfg.v_max,
            hidden_dim=cfg.critic_hidden_dim,
            device=self.device,
        )

        # Wrap in TensorDictModule for compatibility with BaseAlgorithm
        def _critic_forward(obs: Tensor, action: Tensor) -> Tensor:  # CHECK
            q1, _q2 = self._critic_net(obs, action)
            # Return expected value from first Q-network for compatibility
            return self._critic_net.get_value(F.softmax(q1, dim=1)).unsqueeze(-1)

        return TensorDictModule(
            module=_critic_forward,
            in_keys=[key, "action"],
            out_keys=["state_action_value"],
        )

    def _construct_actor_critic(self) -> TensorDictModule:
        if self.q_function is None or self.policy is None:
            msg = "FastTD3 requires both policy and Q-function."
            raise ValueError(msg)

        class IdentityModule(nn.Module):
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
        cfg = self.config.fasttd3

        # Initialize lazy layers by performing a forward pass with dummy data
        fake_td = self.env.fake_tensordict()
        with torch.no_grad():
            _ = self.actor_critic(fake_td)

        # target_critic is built in __init__ after super().__init__(),
        # so we use a placeholder here and set it later
        loss_module = FastTD3Loss(
            actor=self._actor_net,
            critic=self._critic_net,
            target_critic=self._critic_net,  # placeholder, replaced in __init__
            gamma=cfg.gamma,
            policy_noise=cfg.policy_noise,
            noise_clip=cfg.noise_clip,
            use_cdq=cfg.use_cdq,
            disable_bootstrap=cfg.disable_bootstrap,
        )
        return loss_module

    def _construct_data_buffer(self) -> ReplayBuffer:
        cfg = self.config
        actor_key = self.config.policy.input_keys[0]
        critic_key = self.config.q_function.input_keys[0]
        from rlopt.agent.fast_td3.utils import SimpleReplayBuffer
        data_buffer = SimpleReplayBuffer(
            n_env=cfg.env.num_envs, 
            buffer_size=cfg.replay_buffer.size, 
            n_obs=self.env.observation_spec[actor_key].shape[-1],
            n_act=self.env.action_spec.shape[-1],
            n_critic_obs=self.env.observation_spec[critic_key].shape[-1],
            asymmetric_obs=False,
            playground_mode=False,
            n_steps=cfg.fasttd3.num_steps,
            gamma=cfg.fasttd3.gamma,
            device=self.device,
        )

        return data_buffer

    def _construct_feature_extractor(self) -> TensorDictModule:
        msg = "FastTD3 does not require a feature extractor by default."
        raise NotImplementedError(msg)

    def _construct_trainer(self):
        return None

    def _set_optimizers(
        self, optimizer_cls: OptimizerClass, optimizer_kwargs: dict[str, Any]
    ) -> list[torch.optim.Optimizer]:
        """Create separate optimizers for actor and critic."""
        self._actor_optim = optimizer_cls(self._actor_net.parameters(), **optimizer_kwargs)
        self._critic_optim = optimizer_cls(self._critic_net.parameters(), **optimizer_kwargs)
        return [self._actor_optim, self._critic_optim]

    @staticmethod
    @torch.no_grad()
    def _soft_update(src: nn.Module, tgt: nn.Module, tau: float) -> None:
        """Efficient soft update using torch._foreach ops."""
        src_ps = [p.data for p in src.parameters()]
        tgt_ps = [p.data for p in tgt.parameters()]
        torch._foreach_mul_(tgt_ps, 1.0 - tau)
        torch._foreach_add_(tgt_ps, src_ps, alpha=tau)

    def update(self, sampled_tensordict: TensorDict) -> TensorDict:
        assert isinstance(self.config, FastTD3RLOptConfig)
        cfg = self.config.fasttd3
        loss_module = cast(FastTD3Loss, self.loss_module)

        # Ensure target_critic is correct (set once after __init__)
        if loss_module.target_critic is not self.target_critic:
            loss_module.target_critic = self.target_critic

        # Extract tensors
        obs = sampled_tensordict["observations"]
        actions = sampled_tensordict["actions"]
        next_obs = sampled_tensordict["next", "observations"]
        rewards = sampled_tensordict["next", "rewards"]
        dones = sampled_tensordict["next", "dones"]
        truncations = sampled_tensordict["next", "truncations"]
        effective_n_steps = sampled_tensordict["next", "effective_n_steps"]

        # --- Critic update ---
        qf_loss = loss_module.critic_loss(
            obs, actions, next_obs, rewards, dones, truncations, effective_n_steps,
        )

        self._critic_optim.zero_grad(set_to_none=True)
        qf_loss.backward()

        # Clip gradients and capture norm
        max_grad_norm = self.config.optim.max_grad_norm
        if max_grad_norm is not None:
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                self._critic_net.parameters(), max_norm=max_grad_norm
            )
        else:
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                self._critic_net.parameters(), max_norm=float("inf")
            )

        self._critic_optim.step()

        self.total_updates += 1

        loss_td = TensorDict(
            {
                "loss_critic": qf_loss.detach(),
                "critic_grad_norm": critic_grad_norm.detach(),
            },
            batch_size=[],
        )

        # --- Actor update (delayed) ---
        if self.total_updates % cfg.policy_frequency == 0:
            self._actor_optim.zero_grad(set_to_none=True)
            actor_loss, actor_info = loss_module.actor_loss(obs)
            actor_loss.backward()

            if max_grad_norm is not None:
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self._actor_net.parameters(), max_norm=max_grad_norm
                )
            else:
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self._actor_net.parameters(), max_norm=float("inf")
                )

            self._actor_optim.step()
            loss_td.set("loss_actor", actor_loss.detach())
            loss_td.set("actor_grad_norm", actor_grad_norm.detach())
            for key, value in actor_info.items():
                loss_td.set(key, value)

        # --- Soft update target critic ---
        self._soft_update(self._critic_net, self.target_critic, cfg.tau)

        return loss_td.detach_()

    def train(self) -> None:
        assert isinstance(self.config, FastTD3RLOptConfig)
        cfg = self.config
        num_envs = cfg.env.num_envs

        collected_frames = 0
        collector_iter = iter(self.collector)
        pbar: Tqdm = Tqdm(total=cfg.collector.total_frames)

        while collected_frames < cfg.collector.total_frames:
            timeit.printevery(num_prints=1000, total_count=cfg.collector.total_frames, erase=True)

            with timeit("collect"):
                data = next(collector_iter)

            metrics_to_log: dict[str, Any] = {}
            frames_in_batch = data.numel()
            collected_frames += frames_in_batch
            pbar.update(frames_in_batch)

            self.collector.update_policy_weights_()

            # Log step rewards
            if ("next", "reward") in data.keys(True):
                step_rewards = data["next", "reward"]
                metrics_to_log["train/step_reward_mean"] = step_rewards.mean().item()
                metrics_to_log["train/step_reward_std"] = step_rewards.std().item()
                metrics_to_log["train/step_reward_max"] = step_rewards.max().item()
                metrics_to_log["train/step_reward_min"] = step_rewards.min().item()

            # Log episode rewards/lengths
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

            # Extend replay buffer
            with timeit("replay_extend"):
                transition = TensorDict(
                    {
                        "observations": data["policy"].squeeze(1), 
                        "actions": data["action"].squeeze(1),
                        "next": {
                            "observations": data["next", "policy"].squeeze(1),
                            "rewards": data["next", "reward"].squeeze(),
                            "dones": data["next", "done"].squeeze().long(),
                            "truncations": data["next", "truncated"].squeeze().long(),
                        }
                    }, 
                    batch_size=[num_envs],
                    device=self.device,
                )
                self.data_buffer.extend(transition)

            losses = None
            with timeit("train"):
                if collected_frames >= cfg.collector.init_random_frames:
                    losses_list = []
                    for _ in range(cfg.fasttd3.num_updates):
                        with timeit("rb - sample"):
                            sampled_tensordict = self.data_buffer.sample(cfg.fasttd3.batch_size)
                        
                        # according to fast td3, we will update the normalizer during training
                        sampled_tensordict["observations"] = \
                            self.obs_normalizer(sampled_tensordict["observations"])
                        sampled_tensordict["next", "observations"] = \
                            self.obs_normalizer(sampled_tensordict["next", "observations"])

                        with timeit("update"):
                            loss = self.update(sampled_tensordict).clone()

                        losses_list.append(loss)

                    if losses_list:
                        all_keys: set[str] = set()
                        for ld in losses_list:
                            all_keys.update(ld.keys())
                        stacked = {}
                        for key in all_keys:
                            vals = [ld.get(key) for ld in losses_list if key in ld.keys()]
                            if vals:
                                stacked[key] = torch.stack(vals)
                        losses = {key: value.mean() for key, value in stacked.items()}

            # Log losses
            if losses is not None:
                # losses_mean = losses.apply(lambda x: x.float().mean(), batch_size=[])
                for key, value in losses.items():
                    try:
                        scalar = float(value)
                    except Exception:
                        scalar = value.detach().cpu().float().item()
                    metrics_to_log[f"train/{key}"] = scalar
                metrics_to_log["train/actor_lr"] = self._actor_optim.param_groups[0]["lr"]
                metrics_to_log["train/critic_lr"] = self._critic_optim.param_groups[0]["lr"]
                metrics_to_log["train/noise_scale"] = self._noise_scales[0].item()
                if self.config.fasttd3.norm_obs:
                    metrics_to_log["train/obs_mean"] = self.obs_normalizer.mean.mean().item()
                    metrics_to_log["train/obs_std"] = self.obs_normalizer.std.mean().item()
                    metrics_to_log["train/obs_count"] = self.obs_normalizer.count.item()

            # IsaacLab-specific logging
            if "Isaac" in self.config.env.env_name and hasattr(self.env, "log_infos"):
                log_info_dict: dict[str, Tensor] = self.env.log_infos.popleft()
                log_info(log_info_dict, metrics_to_log)

            metrics_to_log.update(timeit.todict(prefix="time"))
            rate = pbar.format_dict.get("rate")
            if rate is not None:
                metrics_to_log["time/speed"] = rate

            if collected_frames % (10 * num_envs) == 0:
                self.log_metrics(metrics_to_log, step=collected_frames)

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

    def predict(self, td: TensorDict) -> Tensor:
        self._actor_net.eval()
        with torch.no_grad():
            obs = td.get("observation")
            obs = self.obs_normalizer(obs)
            action = self._actor_net(obs)
            td.set("action", action)
            return action
