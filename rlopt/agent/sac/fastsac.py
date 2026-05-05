from __future__ import annotations

import copy
import math
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch import Tensor
from torchrl._utils import timeit
from torchrl.data import Bounded
from torchrl.modules import ActorCriticOperator, IndependentNormal, TanhNormal
from torchrl.record.loggers import Logger
from tqdm.rich import tqdm

from rlopt.agent.fast_td3.utils import EmpiricalNormalization, SimpleReplayBuffer
from rlopt.base_class import BaseAlgorithm, IterationData, TrainingMetadata
from rlopt.config_base import NetworkConfig, RLOptConfig
from rlopt.type_aliases import OptimizerClass
from rlopt.utils import log_info


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class FastSACConfig:
    """FastSAC-specific hyperparameters.

    FastSAC is an efficient SAC variant with a distributional C51 critic,
    LayerNorm+SiLU networks, observation normalization, and batched sampling.
    """

    actor_hidden_dim: int = 512
    """Hidden dimension of the actor network (narrows to dim//2//4)."""

    critic_hidden_dim: int = 768
    """Hidden dimension of the distributional critic networks."""

    num_atoms: int = 101
    """Number of atoms for the C51 distributional critic."""

    v_min: float = -20.0
    """Minimum value of the distributional support."""

    v_max: float = 20.0
    """Maximum value of the distributional support."""

    tau: float = 0.125
    """Soft target update coefficient (aggressive, vs 0.005 in standard SAC)."""

    gamma: float = 0.97
    """Discount factor."""

    batch_size: int = 8
    """Per-environment batch size for sampling (total = batch_size * num_envs)."""

    num_updates: int = 8
    """Number of gradient updates per collected step."""

    policy_frequency: int = 4
    """Delayed actor update: update actor every policy_frequency critic steps."""

    learning_starts: int = 10
    """Number of iterations to collect before starting gradient updates."""

    target_entropy_ratio: float = 0.0
    """Target entropy = -n_act * target_entropy_ratio."""

    alpha_init: float = 0.001
    """Initial entropy temperature."""

    use_autotune: bool = True
    """Whether to auto-tune the entropy temperature alpha."""

    log_std_min: float = -5.0
    """Minimum log-std for the actor (after tanh clamping)."""

    log_std_max: float = 0.0
    """Maximum log-std for the actor (after tanh clamping)."""

    use_layer_norm: bool = True
    """Whether to use LayerNorm in actor and critic."""

    num_q_networks: int = 2
    """Number of Q-networks in the critic ensemble."""

    norm_obs: bool = True
    """Whether to normalize observations with running statistics."""

    max_grad_norm: float = 0.0
    """Maximum gradient norm for clipping (0 = disabled)."""

    num_steps: int = 1
    """Number of steps for multi-step returns in replay buffer."""

    amp: bool = True
    """Enable automatic mixed precision (bfloat16) for forward/backward passes."""

    compile_updates: bool = False
    """Compile critic and actor update functions with torch.compile (warm-up cost)."""

    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    """AdamW beta coefficients (beta2=0.95 is faster-decaying than default 0.999)."""


@dataclass
class FastSACRLOptConfig(RLOptConfig):
    """FastSAC configuration extending RLOptConfig."""

    fastsac: FastSACConfig = field(default_factory=FastSACConfig)
    """FastSAC-specific configuration."""

    def __post_init__(self) -> None:
        # NetworkConfig stubs – actual network dims come from FastSACConfig.
        # The input_keys must be set by task-specific subclasses.
        self.policy = NetworkConfig(
            num_cells=[],
            activation_fn="silu",
            output_dim=1,
        )
        self.q_function = NetworkConfig(
            num_cells=[],
            activation_fn="silu",
            output_dim=1,
        )
        self.value_function = None


# ---------------------------------------------------------------------------
# Network modules
# ---------------------------------------------------------------------------


class FastSACActorNet(nn.Module):
    """Stochastic actor with env-aware action distribution."""

    def __init__(
        self,
        n_obs: int,
        n_act: int,
        hidden_dim: int,
        log_std_min: float,
        log_std_max: float,
        action_low: Tensor | None = None,
        action_high: Tensor | None = None,
        use_layer_norm: bool = True,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.n_act = n_act
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.has_action_bounds = action_low is not None and action_high is not None
        if self.has_action_bounds:
            self.register_buffer(
                "action_low",
                torch.as_tensor(action_low, device=device),
            )
            self.register_buffer(
                "action_high",
                torch.as_tensor(action_high, device=device),
            )
        else:
            self.register_buffer(
                "action_low",
                torch.empty(0, device=device),
                persistent=False,
            )
            self.register_buffer(
                "action_high",
                torch.empty(0, device=device),
                persistent=False,
            )

        H = hidden_dim
        LN = nn.LayerNorm if use_layer_norm else lambda d: nn.Identity()

        self.net = nn.Sequential(
            nn.Linear(n_obs, H, device=device),
            LN(H, device=device) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(H, H // 2, device=device),
            LN(H // 2, device=device) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(H // 2, H // 4, device=device),
            LN(H // 4, device=device) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
        )
        self.fc_mu = nn.Linear(H // 4, n_act, device=device)
        self.fc_logstd = nn.Linear(H // 4, n_act, device=device)
        # Zero-initialize output layers for stable initial policy
        nn.init.constant_(self.fc_mu.weight, 0.0)
        nn.init.constant_(self.fc_mu.bias, 0.0)
        nn.init.constant_(self.fc_logstd.weight, 0.0)
        nn.init.constant_(self.fc_logstd.bias, 0.0)

    def forward(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        """Return (mean, log_std) for the Gaussian policy."""
        x = self.net(obs)
        mean = self.fc_mu(x)
        log_std = self.fc_logstd(x)
        # Clamp log_std via tanh (SpinningUp / Denis Yarats style)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1.0)
        return mean, log_std

    def _make_action_dist(self, mean: Tensor, std: Tensor) -> IndependentNormal | TanhNormal:
        if self.has_action_bounds:
            return TanhNormal(
                mean,
                std,
                low=self.action_low,
                high=self.action_high,
                event_dims=1,
                tanh_loc=False,
            )
        return IndependentNormal(
            mean,
            std,
            event_dim=1,
            tanh_loc=False,
        )

    def _deterministic_action(self, mean: Tensor) -> Tensor:
        if not self.has_action_bounds:
            return mean
        action = torch.tanh(mean)
        action_scale = 0.5 * (self.action_high - self.action_low)
        action_bias = 0.5 * (self.action_high + self.action_low)
        return action * action_scale + action_bias

    def get_actions_and_log_probs(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        """Sample actions using the configured action distribution."""
        mean, log_std = self(obs)
        std = log_std.exp()
        dist = self._make_action_dist(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    @torch.no_grad()
    def explore(self, obs: Tensor, deterministic: bool = False) -> Tensor:
        """Sample an action (stochastic or deterministic) for environment interaction."""
        mean, log_std = self(obs)
        if deterministic:
            return self._deterministic_action(mean)
        std = log_std.exp()
        dist = self._make_action_dist(mean, std)
        return dist.rsample()


class FastSACDistributionalQNet(nn.Module):
    """Single C51 distributional Q-network with LayerNorm+SiLU architecture."""

    def __init__(
        self,
        n_obs: int,
        n_act: int,
        num_atoms: int,
        v_min: float,
        v_max: float,
        hidden_dim: int,
        use_layer_norm: bool = True,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.v_min = v_min
        self.v_max = v_max
        self.num_atoms = num_atoms

        H = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(n_obs + n_act, H, device=device),
            nn.LayerNorm(H, device=device) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(H, H // 2, device=device),
            nn.LayerNorm(H // 2, device=device) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(H // 2, H // 4, device=device),
            nn.LayerNorm(H // 4, device=device) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(H // 4, num_atoms, device=device),
        )

    def forward(self, obs: Tensor, actions: Tensor) -> Tensor:
        x = torch.cat([obs, actions], dim=-1)
        return self.net(x)

    def projection(
        self,
        obs: Tensor,
        actions: Tensor,
        rewards: Tensor,
        bootstrap: Tensor,
        discount: Tensor,
        q_support: Tensor,
    ) -> Tensor:
        """Distributional Bellman projection (C51-style)."""
        delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        batch_size = rewards.shape[0]
        device = rewards.device

        target_z = rewards.unsqueeze(1) + bootstrap.unsqueeze(1) * discount.unsqueeze(1) * q_support
        target_z = target_z.clamp(self.v_min, self.v_max)
        b = (target_z - self.v_min) / delta_z
        lower = b.floor().long()
        upper = b.ceil().long()

        is_integer = upper == lower
        lower_mask = is_integer & (lower > 0)
        upper_mask = is_integer & (lower == 0)
        lower = torch.where(lower_mask, lower - 1, lower)
        upper = torch.where(upper_mask, upper + 1, upper)

        next_dist = F.softmax(self.forward(obs, actions), dim=1)
        proj_dist = torch.zeros_like(next_dist)
        offset = (
            torch.linspace(0, (batch_size - 1) * self.num_atoms, batch_size, device=device)
            .unsqueeze(1)
            .expand(batch_size, self.num_atoms)
            .long()
        )
        lower_idx = (lower + offset).view(-1).clamp(0, proj_dist.numel() - 1)
        upper_idx = (upper + offset).view(-1).clamp(0, proj_dist.numel() - 1)
        proj_dist.view(-1).index_add_(0, lower_idx, (next_dist * (upper.float() - b)).view(-1))
        proj_dist.view(-1).index_add_(0, upper_idx, (next_dist * (b - lower.float())).view(-1))
        return proj_dist


class FastSACCriticNet(nn.Module):
    """Ensemble of distributional Q-networks (C51) for FastSAC."""

    def __init__(
        self,
        n_obs: int,
        n_act: int,
        num_atoms: int,
        v_min: float,
        v_max: float,
        hidden_dim: int,
        use_layer_norm: bool = True,
        num_q_networks: int = 2,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.num_q_networks = num_q_networks
        self.qnets = nn.ModuleList([
            FastSACDistributionalQNet(
                n_obs=n_obs,
                n_act=n_act,
                num_atoms=num_atoms,
                v_min=v_min,
                v_max=v_max,
                hidden_dim=hidden_dim,
                use_layer_norm=use_layer_norm,
                device=device,
            )
            for _ in range(num_q_networks)
        ])
        self.register_buffer("q_support", torch.linspace(v_min, v_max, num_atoms, device=device))

    def forward(self, obs: Tensor, actions: Tensor) -> Tensor:
        """Return stacked logits: shape [num_q, batch, num_atoms]."""
        outputs = [qnet(obs, actions) for qnet in self.qnets]
        return torch.stack(outputs, dim=0)

    def projection(
        self,
        obs: Tensor,
        actions: Tensor,
        rewards: Tensor,
        bootstrap: Tensor,
        discount: Tensor,
    ) -> Tensor:
        """Return stacked projected distributions: [num_q, batch, num_atoms]."""
        projections = [
            qnet.projection(obs, actions, rewards, bootstrap, discount, self.q_support)
            for qnet in self.qnets
        ]
        return torch.stack(projections, dim=0)

    def get_value(self, probs: Tensor) -> Tensor:
        """Compute expected value from probability distributions.

        Args:
            probs: shape [num_q, batch, num_atoms] (softmax probabilities)
        Returns:
            shape [num_q, batch]
        """
        return torch.sum(probs * self.q_support, dim=-1)


# ---------------------------------------------------------------------------
# Training metadata
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class FastSACTrainingMetadata(TrainingMetadata):
    """Per-run metadata for FastSAC's training loop."""

    collector_iter: Iterator
    """Active iterator over the TorchRL collector."""

    frames_per_batch: int
    """Environment frames collected per outer-loop iteration."""

    num_updates: int
    """Gradient updates to perform per outer-loop iteration."""

    learning_starts_frames: int
    """Minimum collected frames before gradient updates begin."""


@dataclass(kw_only=True)
class FastSACIterationData(IterationData):
    """Per-iteration state for one outer-loop cycle."""

    rollout: TensorDict | None = None
    """Raw rollout TensorDict from the collector."""


# ---------------------------------------------------------------------------
# FastSAC algorithm
# ---------------------------------------------------------------------------


class FastSAC(BaseAlgorithm[FastSACRLOptConfig]):
    """Fast Soft Actor-Critic with distributional C51 critic.

    Key differences from standard SAC (``rlopt.agent.sac.SAC``):
    - Distributional Q-networks (C51) instead of scalar Q-values
    - LayerNorm + SiLU in both actor and critic (vs ELU in SAC)
    - Running observation normalization (EmpiricalNormalization)
    - Batched sampling: sample once, normalize once, split into mini-batches
    - Aggressive target update (tau=0.125 default vs 0.005)
    - Delayed actor updates (every ``policy_frequency`` critic steps)

    Algorithm structure mirrors PPO's clean ``train()`` loop:
    ``collect()`` → ``iterate()`` → ``record()``

    References:
        Holosoma FastSAC implementation.
        Bellemare et al., 2017 (C51 distributional RL).
    """

    def __init__(
        self,
        env,
        config: FastSACRLOptConfig,
        logger: Logger | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(env=env, config=config, logger=logger, **kwargs)

        # Build target critic (deep copy, no grad)
        self._target_critic = copy.deepcopy(self._critic_net)
        for p in self._target_critic.parameters():
            p.requires_grad_(False)
        self._target_critic.load_state_dict(self._critic_net.state_dict())

        # Observation normalizers
        if config.fastsac.norm_obs:
            self.obs_normalizer = EmpiricalNormalization(
                shape=self._obs_dim, device=self.device
            )
            self.critic_obs_normalizer = EmpiricalNormalization(
                shape=self._critic_obs_dim, device=self.device
            )
        else:
            self.obs_normalizer = nn.Identity()
            self.critic_obs_normalizer = nn.Identity()

        # Alpha optimizer (created separately, not via base class grouping)
        self._alpha_optim = torch.optim.AdamW(
            [self.log_alpha],
            lr=config.optim.lr,
            weight_decay=0.0,
            fused=True,
            betas=config.fastsac.optimizer_betas,
        )

        # AMP autocast context (bfloat16 — no GradScaler needed, unlike float16)
        self._autocast = (
            torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
            if config.fastsac.amp and self.device.type == "cuda"
            else torch.amp.autocast(device_type="cuda", enabled=False)
        )

        # Optionally compile the hot update functions for extra throughput
        if config.fastsac.compile_updates:
            self._update_critic = torch.compile(self._update_critic)  # type: ignore[method-assign]
            self._update_actor = torch.compile(self._update_actor)  # type: ignore[method-assign]

        # Total network update counter for delayed policy updates
        self._update_count = 0

    # ------------------------------------------------------------------
    # Abstract method implementations required by BaseAlgorithm
    # ------------------------------------------------------------------

    def _construct_policy(self) -> TensorDictModule:
        """Build the stochastic actor and wrap it for the TorchRL collector."""
        cfg = self.config
        fsac = cfg.fastsac
        action_spec = self.env.action_spec_unbatched  # type: ignore[attr-defined]

        # Compute obs dims and cache input keys
        self._actor_input_keys: list = cfg.policy.get_input_keys()
        self._obs_dim: int = sum(
            self.observation_feature_size(k) for k in self._actor_input_keys
        )
        cfg.policy.input_dim = self._obs_dim

        action_dim = self.action_feature_size()

        self._actor_net = FastSACActorNet(
            n_obs=self._obs_dim,
            n_act=action_dim,
            hidden_dim=fsac.actor_hidden_dim,
            log_std_min=fsac.log_std_min,
            log_std_max=fsac.log_std_max,
            action_low=(
                torch.as_tensor(action_spec.space.low, device=self.device)
                if isinstance(action_spec, Bounded)
                else None
            ),
            action_high=(
                torch.as_tensor(action_spec.space.high, device=self.device)
                if isinstance(action_spec, Bounded)
                else None
            ),
            use_layer_norm=fsac.use_layer_norm,
            device=self.device,
        )

        # Wrap actor.explore as a TensorDictModule for the collector.
        # Normalization is applied lazily at call time via the collector_policy property.
        actor_net = self._actor_net

        class _ConcatAndExplore(nn.Module):
            """Concatenate multi-key observations and sample an action (no normalization)."""

            def __init__(self, actor: FastSACActorNet) -> None:
                super().__init__()
                self.actor = actor

            def forward(self, *obs_tensors: Tensor) -> Tensor:
                return self.actor.explore(torch.cat(obs_tensors, dim=-1))

        return TensorDictModule(
            module=_ConcatAndExplore(actor_net),
            in_keys=self._actor_input_keys,
            out_keys=["action"],
        )

    def _construct_q_function(self) -> TensorDictModule:
        """Build the distributional critic ensemble."""
        cfg = self.config
        fsac = cfg.fastsac

        self._critic_input_keys: list = cfg.q_function.get_input_keys()
        self._critic_obs_dim: int = sum(
            self.observation_feature_size(k) for k in self._critic_input_keys
        )
        cfg.q_function.input_dim = self._critic_obs_dim

        action_dim = self.action_feature_size()

        self._critic_net = FastSACCriticNet(
            n_obs=self._critic_obs_dim,
            n_act=action_dim,
            num_atoms=fsac.num_atoms,
            v_min=fsac.v_min,
            v_max=fsac.v_max,
            hidden_dim=fsac.critic_hidden_dim,
            use_layer_norm=fsac.use_layer_norm,
            num_q_networks=fsac.num_q_networks,
            device=self.device,
        )

        # Wrap as TensorDictModule stub (actual critic calls are done directly)
        critic_net = self._critic_net

        class _CriticStub(nn.Module):
            def forward(self, *obs_tensors: Tensor, action: Tensor) -> Tensor:
                obs = torch.cat(obs_tensors, dim=-1)
                q_logits = critic_net(obs, action)  # [num_q, batch, atoms]
                q_probs = F.softmax(q_logits, dim=-1)
                q_values = critic_net.get_value(q_probs)  # [num_q, batch]
                return q_values.mean(dim=0).unsqueeze(-1)  # [batch, 1]

        in_keys = self._critic_input_keys + ["action"]
        return TensorDictModule(
            module=_CriticStub(),
            in_keys=in_keys,
            out_keys=["state_action_value"],
        )

    def _construct_value_function(self) -> TensorDictModule:
        raise NotImplementedError("FastSAC uses Q-functions only; no separate value function.")

    def _construct_feature_extractor(self) -> TensorDictModule:
        raise NotImplementedError("FastSAC does not use a separate feature extractor.")

    def _construct_actor_critic(self) -> TensorDictModule:
        """Build a dummy ActorCriticOperator to satisfy BaseAlgorithm."""
        assert self.policy is not None
        assert self.q_function is not None

        class _Identity(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return x

        dummy = TensorDictModule(
            module=_Identity(),
            in_keys=self._actor_input_keys[:1],
            out_keys=self._actor_input_keys[:1],
        )
        return ActorCriticOperator(
            common_operator=dummy,
            policy_operator=self.policy,
            value_operator=self.q_function,
        )

    def _construct_loss_module(self) -> nn.Module:
        """Create log_alpha parameter and return a stub loss module."""
        self.log_alpha = nn.Parameter(
            torch.tensor(
                [math.log(self.config.fastsac.alpha_init)], device=self.device
            )
        )
        self.target_entropy = (
            -self.action_feature_size() * self.config.fastsac.target_entropy_ratio
        )
        return nn.Module()  # stub; updates are performed directly in iterate()

    def _construct_data_buffer(self) -> SimpleReplayBuffer:
        """Create a SimpleReplayBuffer that supports asymmetric observations."""
        cfg = self.config
        fsac = cfg.fastsac

        # _obs_dim / _critic_obs_dim set in _construct_policy / _construct_q_function
        obs_dim = sum(self.observation_feature_size(k) for k in cfg.policy.get_input_keys())
        critic_obs_dim = sum(self.observation_feature_size(k) for k in cfg.q_function.get_input_keys())
        action_dim = self.action_feature_size()
        asymmetric = obs_dim != critic_obs_dim

        # cfg.replay_buffer.size is the *total* transitions budget; SimpleReplayBuffer
        # allocates [n_env, per_env_size, ...] tensors, so divide by num_envs.
        per_env_buffer_size = max(64, cfg.replay_buffer.size // cfg.env.num_envs)
        return SimpleReplayBuffer(
            n_env=cfg.env.num_envs,
            buffer_size=per_env_buffer_size,
            n_obs=obs_dim,
            n_act=action_dim,
            n_critic_obs=critic_obs_dim,
            asymmetric_obs=asymmetric,
            playground_mode=False,
            n_steps=fsac.num_steps,
            gamma=fsac.gamma,
            device=self.device,
        )

    def _construct_trainer(self):  # type: ignore[override]
        return None

    def _set_optimizers(
        self, optimizer_cls: OptimizerClass, optimizer_kwargs: dict[str, Any]
    ) -> list[torch.optim.Optimizer]:
        """Create separate actor and critic optimizers with fused CUDA kernels."""
        fsac = self.config.fastsac
        # fused=True fuses all parameter updates into a single CUDA kernel per group.
        # betas=(0.9, 0.95) uses a faster-decaying beta2 than the default 0.999.
        extra = {"fused": True, "betas": fsac.optimizer_betas}
        kw = {**optimizer_kwargs, **extra}
        self._actor_optim = optimizer_cls(self._actor_net.parameters(), **kw)
        self._critic_optim = optimizer_cls(self._critic_net.parameters(), **kw)
        # Alpha optimizer is created in __init__ after super(); return placeholder here.
        # We register both real optimizers so the base class LR scheduler can manage them.
        return [self._actor_optim, self._critic_optim]

    # ------------------------------------------------------------------
    # Collector policy (stochastic exploration)
    # ------------------------------------------------------------------

    @property
    def collector_policy(self) -> TensorDictModule:
        """Return a normalization-aware policy for environment interaction.

        This is called by the collector after ``__init__`` completes, so
        ``self.obs_normalizer`` is guaranteed to exist.  We build the
        TensorDictModule here (lazily) so that any forward call automatically
        uses the *current* normalizer state without capturing a stale reference.
        """
        agent = self
        actor_net = self._actor_net
        actor_input_keys = self._actor_input_keys
        norm_obs = self.config.fastsac.norm_obs

        class _NormAndExplore(nn.Module):
            """Concatenate observations, apply running normalizer, then explore."""

            def forward(self, *obs_tensors: Tensor) -> Tensor:
                obs = torch.cat(obs_tensors, dim=-1)
                if norm_obs:
                    normalizer = getattr(agent, "obs_normalizer", None)
                    if normalizer is not None:
                        obs = normalizer(obs, update=False)
                return actor_net.explore(obs)

        return TensorDictModule(
            module=_NormAndExplore(),
            in_keys=actor_input_keys,
            out_keys=["action"],
        )

    # ------------------------------------------------------------------
    # Internal update methods
    # ------------------------------------------------------------------

    def _extract_obs(self, data: TensorDict, keys: list) -> Tensor:
        """Concatenate multiple observation keys from a TensorDict.

        Handles both flat keys (e.g. "obs") and nested tuple keys
        (e.g. ("policy", "reference_motion")).  The last dimension is
        treated as the feature dimension and the result is 2-D:
        [batch, feature_dim].
        """
        tensors = [data[k] for k in keys]
        # Flatten any leading time/batch dims beyond the first one
        flat = [t.reshape(t.shape[0], -1) if t.dim() > 2 else t for t in tensors]
        return torch.cat(flat, dim=-1)

    def _update_critic(
        self,
        obs: Tensor,
        actions: Tensor,
        next_obs: Tensor,
        rewards: Tensor,
        dones: Tensor,
        truncations: Tensor,
        n_steps: Tensor,
        critic_obs: Tensor,
        next_critic_obs: Tensor,
    ) -> Tensor:
        """Distributional Bellman update.

        Each Q-network in the ensemble is trained against its own projected
        distribution (holosoma-style), eliminating the need for a second target
        forward pass to select the min-Q index.
        """
        cfg = self.config.fastsac
        bootstrap = (truncations.bool() | ~dones.bool()).float()
        discount = cfg.gamma ** n_steps.float()

        with torch.no_grad():
            with self._autocast:
                next_actions, next_log_probs = self._actor_net.get_actions_and_log_probs(next_obs)
            adjusted_rewards = rewards - discount * bootstrap * self.log_alpha.exp() * next_log_probs
            # Single forward pass: project each Q-network's own distribution [num_q, batch, atoms]
            target_dists = self._target_critic.projection(
                next_critic_obs, next_actions, adjusted_rewards, bootstrap, discount
            )

        with self._autocast:
            q_outputs = self._critic_net(critic_obs, actions)  # [num_q, batch, atoms]
            critic_log_probs = F.log_softmax(q_outputs, dim=-1)
            # Each Q-network trained against its own projected target (no min-Q selection needed)
            critic_losses = -torch.sum(target_dists * critic_log_probs, dim=-1)  # [num_q, batch]
            qf_loss = critic_losses.mean(dim=1).sum(dim=0)

        self._critic_optim.zero_grad(set_to_none=True)
        qf_loss.backward()
        if cfg.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self._critic_net.parameters(), cfg.max_grad_norm)
        self._critic_optim.step()
        return qf_loss.detach()

    def _update_actor(
        self,
        obs: Tensor,
        critic_obs: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Actor and entropy temperature (alpha) update."""
        cfg = self.config.fastsac

        with self._autocast:
            actions, log_probs = self._actor_net.get_actions_and_log_probs(obs)
            q_logits = self._critic_net(critic_obs, actions)  # [num_q, batch, atoms]
            q_probs = F.softmax(q_logits, dim=-1)
            q_values = self._critic_net.get_value(q_probs)  # [num_q, batch]
            actor_loss = (self.log_alpha.exp().detach() * log_probs - q_values.mean(dim=0)).mean()

        self._actor_optim.zero_grad(set_to_none=True)
        actor_loss.backward()
        if cfg.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self._actor_net.parameters(), cfg.max_grad_norm)
        self._actor_optim.step()

        alpha_loss = torch.tensor(0.0, device=self.device)
        if cfg.use_autotune:
            self._alpha_optim.zero_grad(set_to_none=True)
            alpha_loss = (-self.log_alpha.exp() * (log_probs.detach() + self.target_entropy)).mean()
            alpha_loss.backward()
            self._alpha_optim.step()

        return actor_loss.detach(), alpha_loss.detach(), (-log_probs.detach().mean())

    @staticmethod
    @torch.no_grad()
    def _soft_update(src: nn.Module, tgt: nn.Module, tau: float) -> None:
        """Efficient polyak soft update using torch._foreach operations."""
        src_ps = [p.data for p in src.parameters()]
        tgt_ps = [p.data for p in tgt.parameters()]
        torch._foreach_mul_(tgt_ps, 1.0 - tau)
        torch._foreach_add_(tgt_ps, src_ps, alpha=tau)

    # ------------------------------------------------------------------
    # PPO-style training loop phases
    # ------------------------------------------------------------------

    def validate_training(self) -> None:
        """Validate prerequisites before the first rollout."""
        assert self._actor_net is not None, "Actor network not initialized."
        assert self._critic_net is not None, "Critic network not initialized."
        assert self._target_critic is not None, "Target critic not initialized."

    def init_metadata(self) -> FastSACTrainingMetadata:
        """Build the stable state shared across the full training loop."""
        cfg = self.config
        fsac = cfg.fastsac
        frames_per_batch = cfg.collector.frames_per_batch
        total_frames = cfg.collector.total_frames
        total_iterations = max(1, total_frames // frames_per_batch)
        learning_starts_frames = fsac.learning_starts * frames_per_batch

        # Scale per-env batch to absolute num_updates gradient steps
        num_envs = cfg.env.num_envs
        num_updates = fsac.num_updates

        trainer_cfg = getattr(cfg, "trainer", None)
        progress_bar_enabled = True if trainer_cfg is None else bool(trainer_cfg.progress_bar)
        pbar = tqdm(
            total=total_frames,
            disable=not progress_bar_enabled or not sys.stdout.isatty(),
            dynamic_ncols=True,
        )

        log_interval_frames = (
            getattr(trainer_cfg, "log_interval", total_frames) if trainer_cfg else total_frames
        )

        return FastSACTrainingMetadata(
            total_iterations=total_iterations,
            frames_processed=0,
            progress_bar=pbar,
            progress_bar_enabled=progress_bar_enabled,
            log_interval_frames=max(1, log_interval_frames),
            next_log_frame=max(1, log_interval_frames),
            collector_iter=iter(self.collector),
            frames_per_batch=frames_per_batch,
            num_updates=num_updates,
            learning_starts_frames=learning_starts_frames,
        )

    def _extract_next_obs(self, data: TensorDict, keys: list) -> Tensor:
        """Extract and concatenate "next" observation keys from the collector TensorDict.

        The collector stores next observations under ("next", <key>).  The keys
        may themselves be nested tuples, e.g. ("policy", "reference_motion"),
        so the full accessor is ("next", "policy", "reference_motion").
        """
        tensors = []
        for k in keys:
            full_key = ("next", *k) if isinstance(k, tuple) else ("next", k)
            t = data[full_key]
            tensors.append(t.reshape(t.shape[0], -1) if t.dim() > 2 else t)
        return torch.cat(tensors, dim=-1)

    def collect(
        self, metadata: FastSACTrainingMetadata, iteration_idx: int
    ) -> FastSACIterationData:
        """Collect one step from the environment, update normalizer, extend replay buffer."""
        cfg = self.config
        fsac = cfg.fastsac
        num_envs = cfg.env.num_envs

        with timeit("collect"):
            data = next(metadata.collector_iter)

        self.collector.update_policy_weights_()

        frames = data.numel()
        metadata.frames_processed += frames
        metadata.progress_bar.update(frames)

        # The collector with frames_per_batch = num_envs returns [num_envs, ...]
        # If frames_per_batch > num_envs we have multiple time steps; process them all.
        steps_per_env = max(1, frames // num_envs)
        flat_data = data.reshape(-1) if steps_per_env > 1 else data

        # Extract actor observations (concatenate all policy input keys)
        actor_obs = self._extract_obs(flat_data, self._actor_input_keys)
        next_actor_obs = self._extract_next_obs(flat_data, self._actor_input_keys)

        # Extract critic observations
        critic_obs = self._extract_obs(flat_data, self._critic_input_keys)
        next_critic_obs = self._extract_next_obs(flat_data, self._critic_input_keys)

        # Update running normalizer with the new observations
        if fsac.norm_obs and isinstance(self.obs_normalizer, EmpiricalNormalization):
            self.obs_normalizer(actor_obs, update=True)
            self.critic_obs_normalizer(critic_obs, update=True)

        # Handle truncation: replace next obs with true terminal obs when available.
        # IsaacLabTerminalObsReader stores terminal obs in ("next", "obs_unbatched").
        truncations = flat_data["next", "truncated"].reshape(frames)
        if ("next", "obs_unbatched") in flat_data.keys(True):
            term_td = flat_data["next", "obs_unbatched"]
            term_actor_obs = self._extract_obs(term_td, self._actor_input_keys)
            term_critic_obs = self._extract_obs(term_td, self._critic_input_keys)
            mask = truncations.bool().unsqueeze(-1)
            next_actor_obs = torch.where(mask, term_actor_obs, next_actor_obs)
            next_critic_obs = torch.where(mask, term_critic_obs, next_critic_obs)

        actions = flat_data["action"].reshape(frames, -1)
        rewards = flat_data["next", "reward"].reshape(frames)
        dones = flat_data["next", "done"].reshape(frames).long()
        trunc_long = truncations.long()

        # Extend the replay buffer in chunks of num_envs (one step per env at a time)
        with timeit("replay_extend"):
            for t in range(steps_per_env):
                s, e = t * num_envs, (t + 1) * num_envs
                transition = TensorDict(
                    {
                        "observations": actor_obs[s:e],
                        "actions": actions[s:e],
                        "next": {
                            "observations": next_actor_obs[s:e],
                            "rewards": rewards[s:e],
                            "dones": dones[s:e],
                            "truncations": trunc_long[s:e],
                        },
                    },
                    batch_size=[num_envs],
                    device=self.device,
                )
                if self._obs_dim != self._critic_obs_dim:
                    transition["critic_observations"] = critic_obs[s:e]
                    transition["next"]["critic_observations"] = next_critic_obs[s:e]
                self.data_buffer.extend(transition)

        return FastSACIterationData(
            iteration_idx=iteration_idx,
            frames=frames,
            rollout=data,
        )

    def iterate(
        self, iteration: FastSACIterationData, metadata: FastSACTrainingMetadata
    ) -> None:
        """Sample mini-batches from the replay buffer and update networks."""
        if metadata.frames_processed < metadata.learning_starts_frames:
            return

        cfg = self.config
        fsac = cfg.fastsac
        num_envs = cfg.env.num_envs

        # Global batch size: per-env * num_envs
        per_env_batch = max(fsac.batch_size, 1)
        global_batch = per_env_batch * num_envs

        with timeit("train"):
            # Sample one large batch, normalize once, split into mini-batches.
            large_batch = self.data_buffer.sample(per_env_batch * metadata.num_updates)

            # Normalize observations — extract as plain tensors immediately.
            def _norm(normalizer, x: Tensor) -> Tensor:
                if isinstance(normalizer, EmpiricalNormalization):
                    return normalizer(x, update=False)
                return normalizer(x)

            obs_norm = self.obs_normalizer
            critic_norm = self.critic_obs_normalizer
            obs       = _norm(obs_norm,    large_batch["observations"])
            next_obs  = _norm(obs_norm,    large_batch["next"]["observations"])
            asymmetric = "critic_observations" in large_batch
            crit_obs      = _norm(critic_norm, large_batch["critic_observations"]) if asymmetric else obs
            next_crit_obs = _norm(critic_norm, large_batch["next"]["critic_observations"]) if asymmetric else next_obs

            actions    = large_batch["actions"]
            rewards    = large_batch["next"]["rewards"]
            dones      = large_batch["next"]["dones"]
            truncs     = large_batch["next"]["truncations"]
            n_steps    = large_batch["next"]["effective_n_steps"]

            total_samples = obs.shape[0]
            spu = total_samples // metadata.num_updates  # samples_per_update

            qf_loss_acc   = torch.zeros((), device=self.device)
            actor_loss_acc = torch.zeros((), device=self.device)
            alpha_loss_acc = torch.zeros((), device=self.device)
            entropy_acc   = torch.zeros((), device=self.device)
            actor_updates = 0

            for i in range(metadata.num_updates):
                s, e = i * spu, (i + 1) * spu

                qf_loss = self._update_critic(
                    obs[s:e], actions[s:e], next_obs[s:e],
                    rewards[s:e], dones[s:e], truncs[s:e], n_steps[s:e],
                    crit_obs[s:e], next_crit_obs[s:e],
                )
                qf_loss_acc += qf_loss

                # Delayed actor update
                do_actor = (
                    (metadata.num_updates > 1 and i % fsac.policy_frequency == 1)
                    or (metadata.num_updates == 1 and self._update_count % fsac.policy_frequency == 0)
                )
                if do_actor:
                    a_loss, al_loss, entropy = self._update_actor(obs[s:e], crit_obs[s:e])
                    actor_loss_acc += a_loss
                    alpha_loss_acc += al_loss
                    entropy_acc    += entropy
                    actor_updates  += 1

                self._soft_update(self._critic_net, self._target_critic, fsac.tau)
                self._update_count += 1

        # Store averaged metrics in iteration for record()
        n = metadata.num_updates
        iteration.metrics["train/loss_qvalue"] = (qf_loss_acc / n).item()
        if actor_updates > 0:
            iteration.metrics["train/loss_actor"] = (actor_loss_acc / actor_updates).item()
            iteration.metrics["train/loss_alpha"] = (alpha_loss_acc / actor_updates).item()
            iteration.metrics["train/policy_entropy"] = (entropy_acc / actor_updates).item()
        iteration.metrics["train/alpha"] = self.log_alpha.exp().detach().item()
        iteration.metrics["train/lr"] = self._actor_optim.param_groups[0]["lr"]

    def record(
        self, iteration: FastSACIterationData, metadata: FastSACTrainingMetadata
    ) -> None:
        """Log metrics and save checkpoints."""
        cfg = self.config
        data = iteration.rollout
        metrics = iteration.metrics

        if data is not None:
            if ("next", "reward") in data.keys(True):
                step_rewards = data["next", "reward"]
                metrics["train/step_reward_mean"] = step_rewards.mean().item()
                metrics["train/step_reward_std"] = step_rewards.std().item()
                metrics["train/step_reward_max"] = step_rewards.max().item()
                metrics["train/step_reward_min"] = step_rewards.min().item()

            if ("next", "episode_reward") in data.keys(True):
                episode_rewards = data["next", "episode_reward"][data["next", "done"]]
                if len(episode_rewards) > 0:
                    episode_lengths = data["next", "step_count"][data["next", "done"]]
                    self.episode_lengths.extend(episode_lengths.cpu().tolist())
                    self.episode_rewards.extend(episode_rewards.cpu().tolist())
                    metrics["episode/length"] = float(np.mean(self.episode_lengths))
                    metrics["episode/return"] = float(np.mean(self.episode_rewards))

        # IsaacLab environment-specific metrics
        if "Isaac" in cfg.env.env_name and hasattr(self.env, "log_infos"):
            log_info_dict = self.env.log_infos.popleft()
            log_info(log_info_dict, metrics)

        metrics.update(timeit.todict(prefix="time"))
        rate = getattr(metadata.progress_bar, "format_dict", {}).get("rate")
        if rate is not None:
            metrics["time/speed"] = rate

        # Log all metrics to W&B / file without printing to console.
        # _refresh_progress_display below handles the filtered console summary.
        if self._should_log_iteration(metadata, iteration):
            self.log_metrics(
                metrics,
                step=metadata.frames_processed,
                log_python=False,
            )
        self._refresh_progress_display(metadata, iteration)

        # save_interval is in *samples* (same unit as log_interval and total_frames).
        if (
            self._should_save_checkpoint(
                frames_processed=metadata.frames_processed,
                frames_in_iteration=iteration.frames,
            )
        ):
            self.save_model(
                path=self.log_dir / cfg.logger.save_path,
                step=metadata.frames_processed,
            )

    def train(self) -> None:  # type: ignore[override]
        """Train the agent using the PPO-style collect → iterate → record loop."""
        self.validate_training()
        metadata = self.init_metadata()

        try:
            for iteration_idx in range(metadata.total_iterations):
                iteration = self.collect(metadata, iteration_idx)
                self.iterate(iteration, metadata)
                self.record(iteration, metadata)
        finally:
            metadata.progress_bar.close()
            self.collector.shutdown()

    def predict(self, td: TensorDict) -> Tensor:  # type: ignore[override]
        """Deterministic policy inference."""
        self._actor_net.eval()
        with torch.no_grad():
            obs = self._extract_obs(td, self._actor_input_keys)
            if isinstance(self.obs_normalizer, EmpiricalNormalization):
                obs = self.obs_normalizer(obs, update=False)
            return self._actor_net.explore(obs, deterministic=True)

    def _progress_summary_fields(self) -> tuple[tuple[str, str], ...]:
        return (
            ("train/step_reward_mean", "r_step"),
            ("episode/length", "ep_len"),
            ("episode/return", "r_ep"),
            ("train/loss_actor", "pi_loss"),
            ("train/loss_qvalue", "q_loss"),
            ("train/alpha", "alpha"),
            ("train/policy_entropy", "entropy"),
            ("time/collect", "t_collect"),
            ("time/train", "t_train"),
            ("time/speed", "fps"),
        )
