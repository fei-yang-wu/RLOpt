"""IPMD with bilinear spectral representation learning.

Implements the bottom-up skill composition approach where spectral components
are parameterized as phi_i(s,a) = f(s)^T W_i g(a) with low-rank factorization
W_i = u_i v_i^T. The policy uses f(s) @ W(z) as input to a neural network
that outputs a Gaussian action distribution (PPO on top of the representation).

The spectral representation objective is configurable via ``sr_type``:
    - ``"diffsr"``: Diffusion-based noise prediction (default).
    - ``"speder"``: Spectral contrastive loss.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch import Tensor
from torchrl.data import (
    Bounded,
    LazyTensorStorage,
    ReplayBuffer,
    TensorDictReplayBuffer,
)
from torchrl.data.replay_buffers.samplers import RandomSampler
from torchrl.envs.utils import ExplorationType
from torchrl.modules import MLP, IndependentNormal, ProbabilisticActor, TanhNormal
from torchrl.record.loggers import Logger

from rlopt.agent.ipmd.ipmd import IPMD, IPMDRLOptConfig
from rlopt.agent.ipmd.module import BilinearSR, build_bilinear_sr
from rlopt.agent.ipmd.utils import require_positive_int
from rlopt.agent.ppo.ppo import PPOIterationData, PPOTrainingMetadata
from rlopt.config_utils import (
    BatchKey,
    ObsKey,
    dedupe_keys,
    next_obs_key,
    normalize_batch_key,
    obs_key_feature_dim,
)
from rlopt.models.gaussian_policy import GaussianPolicyHead
from rlopt.utils import get_activation_class

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


DEFAULT_BILINEAR_OBS_KEYS: list[ObsKey] = [
    ("policy", "base_ang_vel"),
    ("policy", "joint_pos_rel"),
    ("policy", "joint_vel_rel"),
    ("policy", "last_action"),
]

DEFAULT_BILINEAR_NEXT_OBS_KEYS: list[ObsKey] = [
    ("policy", "base_ang_vel"),
    ("policy", "joint_pos_rel"),
    ("policy", "joint_vel_rel"),
]


def _require_non_negative_int(name: str, value: int) -> int:
    normalized = int(value)
    if normalized < 0:
        msg = f"{name} must be >= 0, got {value!r}."
        raise ValueError(msg)
    return normalized


@dataclass
class BilinearOfflinePretrainConfig:
    """Offline spectral-representation pretraining configuration."""

    enabled: bool = False
    """Run offline SR pretraining before online rollout collection."""

    num_updates: int = 2000
    """Number of offline SR gradient updates."""

    batch_size: int = 8192
    """Expert transition batch size sampled per offline SR update."""

    log_interval: int = 100
    """Log offline SR metrics every N offline updates."""

    policy_bc_updates: int = 0
    """Number of offline policy behavior-cloning updates after SR pretraining."""

    policy_bc_batch_size: int = 8192
    """Expert action batch size for offline policy BC."""

    policy_bc_train_latent: bool = False
    """Allow offline policy BC gradients to update inferred latent-command modules."""

    def requires_offline_data(self) -> bool:
        return bool(self.enabled or int(self.policy_bc_updates) > 0)

    def validate(self) -> None:
        self.policy_bc_updates = _require_non_negative_int(
            "bilinear.offline_pretrain.policy_bc_updates",
            self.policy_bc_updates,
        )
        if not self.requires_offline_data():
            return
        self.log_interval = require_positive_int(
            "bilinear.offline_pretrain.log_interval", self.log_interval
        )
        if self.enabled:
            self.num_updates = require_positive_int(
                "bilinear.offline_pretrain.num_updates", self.num_updates
            )
            self.batch_size = require_positive_int(
                "bilinear.offline_pretrain.batch_size", self.batch_size
            )
        if self.policy_bc_updates > 0:
            self.policy_bc_batch_size = require_positive_int(
                "bilinear.offline_pretrain.policy_bc_batch_size",
                self.policy_bc_batch_size,
            )


@dataclass
class BilinearSRConfig:
    """Bilinear spectral representation configuration."""

    sr_type: str = "diffsr"
    """Spectral representation type: "diffsr" (diffusion) or "speder" (contrastive) or "ctrlsr" (contrastive)"""

    feature_dim: int = 512
    """Number of spectral components (d)."""

    embed_dim: int = 512
    """Shared embedding dimension for state and action encoders."""

    f_hidden_dims: tuple[int, ...] = (512, 512)
    """Hidden layers for state encoder f(s)."""

    g_hidden_dims: tuple[int, ...] = (512,)
    """Hidden layers for action encoder g(a)."""

    mu_hidden_dims: tuple[int, ...] = (512,)
    """Hidden layers for next-state encoder mu(s')."""

    feature_lr: float = 1e-4
    """Learning rate for the bilinear SR model."""

    grad_clip_norm: float | None = None
    """Max-norm gradient clip applied to the SR optimizer. None disables clipping (default)."""

    update_steps: int = 8
    """Number of SR gradient steps per IPMD update."""

    history_buffer_size: int = 10_000_000
    """Max transitions in the SR history replay buffer."""

    sr_batch_size: int = 4096
    """Batch size for SR training mini-batches."""

    num_noises: int = 8
    """Number of diffusion/noise-perturbation steps."""

    x_min: float = -10.0
    """Clamp min for reverse diffusion (DiffSR only)."""

    x_max: float = 10.0
    """Clamp max for reverse diffusion (DiffSR only)."""

    lam: float = 1024.0
    """Negative-loss weighting lambda for spectral contrastive loss (Speder only)."""

    sample_eval_interval: int = 50
    """Run sampling check every N SR updates (only when SR supports sampling)."""

    detach_features_for_policy: bool = False
    """Detach the final policy representation before feeding it to the policy MLP."""

    policy_include_raw_state: bool = False
    """Append raw SR state to F(s)z before the policy MLP."""

    ema_tau: float = 0.01
    """EMA update rate for the policy target state_net. Lower = slower tracking."""

    use_ema_for_policy: bool = True
    """Use a slow-moving EMA copy of state_net for policy input (stabilizes PPO)."""

    obs_keys: list[ObsKey] = field(
        default_factory=lambda: list(DEFAULT_BILINEAR_OBS_KEYS)
    )
    """Observation keys concatenated into the SR state input s."""

    next_obs_keys: list[ObsKey] = field(
        default_factory=lambda: list(DEFAULT_BILINEAR_NEXT_OBS_KEYS)
    )
    """Observation keys concatenated into the SR next-state input s'."""

    offline_pretrain: BilinearOfflinePretrainConfig = field(
        default_factory=BilinearOfflinePretrainConfig
    )
    """Offline SR pretraining settings."""

    def validate(self) -> None:
        self.sr_type = str(self.sr_type).strip().lower()
        self.feature_dim = require_positive_int(
            "bilinear.feature_dim", self.feature_dim
        )
        self.embed_dim = require_positive_int("bilinear.embed_dim", self.embed_dim)
        self.update_steps = _require_non_negative_int(
            "bilinear.update_steps", self.update_steps
        )
        self.history_buffer_size = require_positive_int(
            "bilinear.history_buffer_size", self.history_buffer_size
        )
        self.sr_batch_size = require_positive_int(
            "bilinear.sr_batch_size", self.sr_batch_size
        )
        self.num_noises = _require_non_negative_int(
            "bilinear.num_noises", self.num_noises
        )
        self.obs_keys = cast(list[ObsKey], dedupe_keys(list(self.obs_keys)))
        self.next_obs_keys = cast(list[ObsKey], dedupe_keys(list(self.next_obs_keys)))
        if len(self.obs_keys) == 0:
            msg = "bilinear.obs_keys must be non-empty."
            raise ValueError(msg)
        if self.offline_pretrain.enabled and len(self.next_obs_keys) == 0:
            msg = (
                "bilinear.offline_pretrain.enabled=True requires non-empty "
                "bilinear.next_obs_keys."
            )
            raise ValueError(msg)
        self.offline_pretrain.validate()

    def get_obs_keys(self) -> list[ObsKey]:
        return list(self.obs_keys)

    def get_next_obs_keys(self) -> list[ObsKey]:
        return list(self.next_obs_keys)


@dataclass
class IPMDBilinearRLOptConfig(IPMDRLOptConfig):
    """IPMD + Bilinear spectral representation configuration."""

    bilinear: BilinearSRConfig = field(default_factory=BilinearSRConfig)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.bilinear.validate()


# ---------------------------------------------------------------------------
# Policy Head
# ---------------------------------------------------------------------------


class BilinearPolicyHead(GaussianPolicyHead):
    """Policy head using bilinear spectral representation.

    Inherits from GaussianPolicyHead. The only difference is that it first
    computes f(s) @ W(z) from the bilinear representation, then passes
    the result through the base MLP (inherited from GaussianPolicyHead)
    to produce (loc, scale) for a Gaussian action distribution.

    The bilinear representation's features are detached by default so PPO
    gradients do not flow into the representation (trained separately).
    """

    def __init__(
        self,
        bilinear_rep: BilinearSR,
        num_cells: list[int],
        activation_fn: str,
        action_dim: int,
        num_command_inputs: int,
        command_dim: int | None,
        log_std_init: float = 0.0,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        clip_log_std: bool = False,
        detach_features: bool = False,
        include_raw_state: bool = False,
        device: str | torch.device = "cpu",
    ) -> None:
        base_mlp = MLP(
            in_features=None,  # lazy initialization
            out_features=action_dim,
            num_cells=num_cells,
            activation_class=get_activation_class(activation_fn),
            device=device,
        )

        super().__init__(
            base=base_mlp,
            action_dim=action_dim,
            log_std_init=log_std_init,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            clip_log_std=clip_log_std,
            device=device,
        )

        # Store bilinear_rep as submodule (shared with agent).
        # Detach prevents PPO gradients flowing into the representation.
        self.bilinear_rep = bilinear_rep
        self.detach_features = detach_features
        self.include_raw_state = bool(include_raw_state)
        self.num_command_inputs = int(num_command_inputs)
        if self.num_command_inputs < 0:
            msg = "num_command_inputs must be >= 0."
            raise ValueError(msg)
        if self.num_command_inputs == 0:
            if command_dim is not None:
                msg = "command_dim must be None when num_command_inputs is 0."
                raise ValueError(msg)
            self.command_projector = None
        else:
            if command_dim is None or command_dim <= 0:
                msg = "command_dim must be positive when command inputs are present."
                raise ValueError(msg)
            self.command_projector = (
                None
                if command_dim == bilinear_rep.feature_dim
                else torch.nn.Linear(
                    command_dim, bilinear_rep.feature_dim, device=device
                )
            )

    def forward(self, *obs: Tensor) -> tuple[Tensor, Tensor]:
        """Compute (loc, scale) from observations via bilinear representation.

        1. Split command observations from representation-state observations.
        2. Project command observations into the spectral coefficient space.
        3. Compute policy representation f(s) @ W(z).
        3. (Optionally) detach from SR computation graph.
        4. Feed through base MLP -> loc, then log_std_module -> scale.
        """
        if len(obs) <= self.num_command_inputs:
            msg = (
                "BilinearPolicyHead.forward() expected representation-state "
                "observations after the configured command inputs."
            )
            raise ValueError(msg)
        command_obs = obs[: self.num_command_inputs]
        state_obs = obs[self.num_command_inputs :]
        state = state_obs[0] if len(state_obs) == 1 else torch.cat(state_obs, dim=-1)
        if len(command_obs) == 0:
            z = state.new_ones((*state.shape[:-1], self.bilinear_rep.feature_dim))
        else:
            command = (
                command_obs[0]
                if len(command_obs) == 1
                else torch.cat(command_obs, dim=-1)
            )
            if command.shape[-1] == self.bilinear_rep.feature_dim:
                z = command
            else:
                assert self.command_projector is not None
                z = self.command_projector(command)
        rep = self.bilinear_rep.compute_policy_representation(
            state,
            z,
            include_raw_state=self.include_raw_state,
        )
        # if self.detach_features:
        #     rep = rep.detach()
        # Delegate to GaussianPolicyHead: base MLP + log_std_module
        loc = self.base(rep)
        scale = self.log_std_module(loc)
        return loc, scale


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class IPMDBilinear(IPMD):
    """IPMD with bilinear spectral representation learning.

    Diffusion-based SR training on (s, a, s') via a history buffer,
    bilinear policy f(s) @ W(z) -> MLP -> Gaussian, plus standard
    IPMD reward estimator and BC loss.
    """

    def __init__(
        self,
        env,
        config: IPMDBilinearRLOptConfig,
        policy_net: torch.nn.Module | None = None,
        value_net: torch.nn.Module | None = None,
        q_net: torch.nn.Module | None = None,
        replay_buffer: type[ReplayBuffer] = ReplayBuffer,
        logger: Logger | None = None,
        feature_extractor_net: torch.nn.Module | None = None,
        **kwargs,
    ) -> None:
        self.config = cast(IPMDBilinearRLOptConfig, config)
        self.config.bilinear.validate()
        self.env = env
        self._provided_bilinear_policy_net = policy_net
        offline_dataset_enabled = bool(
            getattr(self.config.offline_dataset, "enabled", False)
        )
        offline_cfg = self.config.bilinear.offline_pretrain
        if (
            offline_cfg.requires_offline_data()
            and not offline_dataset_enabled
            and self._discover_env_method(env, "sample_expert_batch") is None
        ):
            msg = (
                "bilinear offline pretraining or policy BC requires "
                "env.sample_expert_batch(...) or offline_dataset.enabled=True."
            )
            raise ValueError(msg)

        # Build before super().__init__ so _construct_policy can use it.
        self._bilinear_obs_keys = self.config.bilinear.get_obs_keys()
        self._bilinear_next_obs_keys = self.config.bilinear.get_next_obs_keys()
        self.bilinear_rep = self._construct_bilinear_model()
        self.bilinear_rep.to(self.device)

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

        self.sr_optim = torch.optim.Adam(
            self.bilinear_rep.parameters(), lr=self.config.bilinear.feature_lr
        )

        self._sr_update_count = 0
        self._pending_sr_metrics: dict[str, list[float]] = {}
        self._sr_history_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(
                max_size=self.config.bilinear.history_buffer_size,
                device=self.device,
            ),
            sampler=RandomSampler(),
            batch_size=self.config.bilinear.sr_batch_size,
        )
        self._validate_offline_pretrain_setup()

    # -- Construction --

    @staticmethod
    def _get_batch_key(td: TensorDict, key: BatchKey) -> Tensor:
        return cast(Tensor, td.get(key))

    @staticmethod
    def _offline_next_key(key: ObsKey) -> tuple[str, ...]:
        return next_obs_key(key)

    def _concat_bilinear_obs_from_td(self, td: TensorDict) -> Tensor:
        """Concatenate all policy obs keys from a TensorDict."""
        parts = [
            self._get_batch_key(td, k).flatten(-1) for k in self._bilinear_obs_keys
        ]
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)

    def _concat_bilinear_next_obs_from_td(self, td: TensorDict) -> Tensor:
        """Concatenate all next policy obs keys from a TensorDict."""
        parts = [
            self._get_batch_key(td, self._offline_next_key(k)).flatten(-1)
            for k in self._bilinear_next_obs_keys
        ]
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)

    def _construct_bilinear_model(self) -> BilinearSR:
        cfg = self.config.bilinear
        action_spec = getattr(self.env, "action_spec_unbatched", self.env.action_spec)
        bilinear_obs_dim = sum(
            int(self.env.observation_spec[k].shape[-1]) for k in self._bilinear_obs_keys
        )
        bilinear_next_obs_dim = sum(
            int(self.env.observation_spec[k].shape[-1])
            for k in self._bilinear_next_obs_keys
        )

        # Common kwargs shared by all SR types
        common = {
            "obs_dim": bilinear_obs_dim,
            "next_obs_dim": bilinear_next_obs_dim,
            "action_dim": action_spec.shape[-1],
            "feature_dim": cfg.feature_dim,
            "embed_dim": cfg.embed_dim,
            "f_hidden_dims": cfg.f_hidden_dims,
            "g_hidden_dims": cfg.g_hidden_dims,
            "mu_hidden_dims": cfg.mu_hidden_dims,
            "num_noises": cfg.num_noises,
            "use_ema_for_policy": cfg.use_ema_for_policy,
            "device": self.device,
        }

        # Type-specific kwargs
        if cfg.sr_type == "diffsr":
            common.update(x_min=cfg.x_min, x_max=cfg.x_max)
        elif cfg.sr_type == "speder":
            common.update(lam=cfg.lam)

        return build_bilinear_sr(cfg.sr_type, **common)

    def _validate_offline_pretrain_setup(self) -> None:
        offline_cfg = self.config.bilinear.offline_pretrain
        if not offline_cfg.requires_offline_data():
            return
        if (
            self._offline_expert_batch_sampler is None
            and self._expert_batch_sampler is None
        ):
            msg = (
                "bilinear offline pretraining or policy BC requires "
                "env.sample_expert_batch(...) or offline_dataset.enabled=True."
            )
            raise ValueError(msg)
        offline_cfg.validate()
        self._preflight_offline_pretrain_batch()

    def _preflight_offline_pretrain_batch(self) -> None:
        offline_cfg = self.config.bilinear.offline_pretrain
        preflight_batch_size = min(
            int(
                offline_cfg.batch_size
                if offline_cfg.enabled
                else offline_cfg.policy_bc_batch_size
            ),
            8,
        )
        expert_batch = self._next_offline_expert_batch(
            batch_size=preflight_batch_size,
            required_keys=self._offline_pretrain_required_keys(),
        )
        if offline_cfg.policy_bc_updates > 0:
            expert_action = cast(Tensor, expert_batch.get("expert_action"))
            self._validate_offline_pretrain_tensor_shape(
                "expert_action",
                expert_action,
                batch_size=preflight_batch_size,
                feature_dim=int(self.bilinear_rep.action_dim),
            )
        if not offline_cfg.enabled:
            return
        transitions = self._sr_transition_batch_from_td(
            expert_batch,
            action_key="expert_action",
        )

        obs = cast(Tensor, transitions.get("obs"))
        action = cast(Tensor, transitions.get("action"))
        next_obs = cast(Tensor, transitions.get(("next", "obs")))
        self._validate_offline_pretrain_tensor_shape(
            "obs",
            obs,
            batch_size=preflight_batch_size,
            feature_dim=int(self.bilinear_rep.obs_dim),
        )
        self._validate_offline_pretrain_tensor_shape(
            "action",
            action,
            batch_size=preflight_batch_size,
            feature_dim=int(self.bilinear_rep.action_dim),
        )
        self._validate_offline_pretrain_tensor_shape(
            "next_obs",
            next_obs,
            batch_size=preflight_batch_size,
            feature_dim=int(self.bilinear_rep.next_obs_dim),
        )

    @staticmethod
    def _validate_offline_pretrain_tensor_shape(
        name: str,
        tensor: Tensor,
        *,
        batch_size: int,
        feature_dim: int,
    ) -> None:
        expected = (int(batch_size), int(feature_dim))
        actual = tuple(tensor.shape)
        if actual != expected:
            msg = (
                "Offline bilinear pretrain expert batch shape mismatch for "
                f"{name}: expected {expected}, got {actual}."
            )
            raise ValueError(msg)

    def _bilinear_policy_command_keys(self) -> list[ObsKey]:
        bilinear_state_keys = set(self._bilinear_obs_keys)
        return [
            cast(ObsKey, normalize_batch_key(key))
            for key in self.config.policy.get_input_keys()
            if normalize_batch_key(key) not in bilinear_state_keys
        ]

    def _construct_policy(
        self, policy_net: torch.nn.Module | None = None
    ) -> TensorDictModule:
        action_spec = self.env.action_spec_unbatched
        action_dim = int(action_spec.shape[-1])

        if isinstance(action_spec, Bounded):
            dist_cls, dist_kw = (
                TanhNormal,
                {
                    "low": action_spec.space.low,
                    "high": action_spec.space.high,
                    "tanh_loc": False,
                },
            )
        else:
            dist_cls, dist_kw = IndependentNormal, {}

        provided_policy_head = policy_net or self._provided_bilinear_policy_net
        if provided_policy_head is not None:
            policy_head = provided_policy_head
            policy_in_keys = self.config.policy.get_input_keys()
        else:
            command_keys = self._bilinear_policy_command_keys()
            command_dim = (
                sum(obs_key_feature_dim(self.env, key) for key in command_keys)
                if command_keys
                else None
            )
            policy_head = BilinearPolicyHead(
                bilinear_rep=self.bilinear_rep,
                num_cells=list(self.config.policy.num_cells),
                activation_fn=self.config.policy.activation_fn,
                action_dim=action_dim,
                num_command_inputs=len(command_keys),
                command_dim=command_dim,
                log_std_init=self.config.ppo.log_std_init,
                log_std_min=self.config.ppo.log_std_min,
                log_std_max=self.config.ppo.log_std_max,
                clip_log_std=self.config.ppo.clip_log_std,
                detach_features=self.config.bilinear.detach_features_for_policy,
                include_raw_state=self.config.bilinear.policy_include_raw_state,
                device=self.device,
            )
            policy_in_keys = command_keys + list(self._bilinear_obs_keys)

        return ProbabilisticActor(
            TensorDictModule(
                policy_head,
                in_keys=policy_in_keys,
                out_keys=["loc", "scale"],
            ),
            in_keys=["loc", "scale"],
            spec=self.env.full_action_spec_unbatched.to(self.device),
            distribution_class=dist_cls,
            distribution_kwargs=dist_kw,
            return_log_prob=True,
            default_interaction_type=ExplorationType.RANDOM,
        )

    # -- SR training --

    def _sr_transition_batch_from_td(
        self,
        batch: TensorDict,
        *,
        action_key: BatchKey,
    ) -> TensorDict:
        obs = self._concat_bilinear_obs_from_td(batch)
        next_obs = self._concat_bilinear_next_obs_from_td(batch)
        action = self._get_batch_key(batch, action_key)

        return TensorDict(
            {
                "obs": obs,
                "action": action,
                ("next", "obs"): next_obs,
            },
            batch_size=batch.batch_size,
        )

    def _store_transitions(
        self,
        batch: TensorDict,
        *,
        action_key: BatchKey = "action",
    ) -> None:
        transitions = self._sr_transition_batch_from_td(
            batch,
            action_key=action_key,
        )
        self._sr_history_buffer.extend(transitions.reshape(-1).detach())

        # Recompute normalization from all effective samples in the buffer
        n = len(self._sr_history_buffer)
        all_next_obs = self._sr_history_buffer._storage._storage["next", "obs"][:n]
        self.bilinear_rep.update_obs_norm(all_next_obs)

    def _offline_sr_required_keys(self) -> list[BatchKey]:
        required: list[BatchKey] = list(self._bilinear_obs_keys)
        required.extend(
            self._offline_next_key(key) for key in self._bilinear_next_obs_keys
        )
        required.append("expert_action")
        return dedupe_keys(required)

    def _offline_policy_bc_required_keys(self) -> list[BatchKey]:
        required: list[BatchKey] = ["expert_action"]
        required.extend(self._policy_obs_keys_without_latent)
        if self._use_latent_command:
            assert self._latent_learner is not None
            required.extend(self._latent_learner.required_expert_batch_keys())
        return dedupe_keys(required)

    def _offline_pretrain_required_keys(self) -> list[BatchKey]:
        required: list[BatchKey] = []
        if self.config.bilinear.offline_pretrain.enabled:
            required.extend(self._offline_sr_required_keys())
        if self.config.bilinear.offline_pretrain.policy_bc_updates > 0:
            required.extend(self._offline_policy_bc_required_keys())
        return dedupe_keys(required)

    def _sr_loss_from_batch(self, batch: TensorDict) -> tuple[dict[str, float], Tensor]:
        batch = batch.to(self.device)
        obs = cast(Tensor, batch.get("obs"))
        action = cast(Tensor, batch.get("action"))
        next_obs = cast(Tensor, batch.get(("next", "obs")))
        reward = torch.zeros(obs.shape[0], 1, device=self.device)
        metrics, loss, _ = self.bilinear_rep.compute_loss(obs, action, next_obs, reward)
        return metrics, loss

    def _train_sr_steps(self, steps: int) -> dict[str, float]:
        metrics_accum: dict[str, float] = {}
        if steps <= 0 or len(self._sr_history_buffer) == 0:
            return metrics_accum

        self.bilinear_rep.train()
        clip_norm = self.config.bilinear.grad_clip_norm
        for _ in range(steps):
            metrics, loss = self._sr_loss_from_batch(self._sr_history_buffer.sample())
            self.sr_optim.zero_grad(set_to_none=True)
            loss.backward()
            if clip_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.bilinear_rep.parameters(), max_norm=clip_norm
                )
                metrics["misc/sr_grad_norm"] = float(grad_norm)
            self.sr_optim.step()
            for k, v in metrics.items():
                metrics_accum[k] = metrics_accum.get(k, 0.0) + float(v)

        self._sr_update_count += 1
        for k in metrics_accum:
            metrics_accum[k] /= float(steps)

        # Periodic sampling evaluation (only when SR supports it, e.g. DiffSR)
        interval = self.config.bilinear.sample_eval_interval
        if (
            self.bilinear_rep.supports_sampling
            and interval > 0
            and self._sr_update_count % interval == 0
        ):
            sample_batch = self._sr_history_buffer.sample()
            sample_metrics = self._sr_sample_eval(sample_batch)
            metrics_accum.update(sample_metrics)
        return metrics_accum

    @torch.no_grad()
    def _sr_sample_eval(self, batch: TensorDict) -> dict[str, float]:
        """Run diffusion sampling to reconstruct s from s', measure distance."""
        self.bilinear_rep.eval()
        batch = batch.to(self.device)
        obs = cast(Tensor, batch.get("obs"))
        action = cast(Tensor, batch.get("action"))
        next_obs = cast(Tensor, batch.get(("next", "obs")))
        s_recon, _ = self.bilinear_rep.sample(s=obs, a=action, preserve_history=True)
        mse = (s_recon - next_obs).pow(2).sum(-1).mean()
        l1 = (s_recon - next_obs).abs().sum(-1).mean()
        self.bilinear_rep.train()
        return {
            "sample/recon_mse": mse.item(),
            "sample/recon_l1": l1.item(),
        }

    def _offline_policy_bc_step(self, expert_batch: TensorDict) -> dict[str, float]:
        offline_cfg = self.config.bilinear.offline_pretrain
        expert_batch = expert_batch.to(self.device)
        expert_action = cast(Tensor, expert_batch.get("expert_action"))
        expert_obs_td = expert_batch.clone(False)
        train_latent = bool(offline_cfg.policy_bc_train_latent)
        bc_latent: Tensor | None = None
        if self._use_latent_command and self._latent_key not in expert_obs_td.keys(
            True
        ):
            bc_latent = self._expert_latents_from_td(
                expert_batch,
                detach=not train_latent,
            ).reshape(*expert_batch.batch_size, self._latent_dim)
            expert_obs_td.set(self._latent_key, bc_latent)
        expert_obs_td = expert_obs_td.select(*self._policy_operator.in_keys)

        self.optim.zero_grad(set_to_none=True)
        dist = self._policy_operator.get_dist(expert_obs_td)
        log_prob = dist.log_prob(expert_action)
        log_prob = self._reduce_log_prob(log_prob, expert_action)
        bc_nll = -log_prob.mean()
        bc_nll.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self._grad_clip_params,
            self._max_grad_norm,
        )
        self.optim.step()

        with torch.no_grad():
            metrics = {
                "bc_nll": bc_nll.item(),
                "bc_log_prob_mean": log_prob.mean().item(),
                "bc_expert_action_abs_mean": expert_action.abs().mean().item(),
                "bc_actor_grad_norm": float(grad_norm),
                "bc_train_latent": float(train_latent),
            }
            if bc_latent is not None:
                metrics["bc_latent_abs_mean"] = bc_latent.abs().mean().item()
                metrics["bc_latent_std"] = bc_latent.std(unbiased=False).item()
            scale = getattr(dist, "scale", None)
            if isinstance(scale, Tensor):
                metrics["bc_policy_scale_mean"] = scale.mean().item()
        return metrics

    def _offline_pretrain_policy_bc(self) -> None:
        offline_cfg = self.config.bilinear.offline_pretrain
        if offline_cfg.policy_bc_updates <= 0:
            return

        required_keys = self._offline_policy_bc_required_keys()
        total_updates = int(offline_cfg.policy_bc_updates)
        batch_size = int(offline_cfg.policy_bc_batch_size)
        start_time = time.perf_counter()
        self.actor_critic.train()
        self.log.info(
            "offline_policy_bc start | updates=%d | batch_size=%d | log_interval=%d",
            total_updates,
            batch_size,
            int(offline_cfg.log_interval),
        )
        for update_idx in range(1, total_updates + 1):
            expert_batch = self._next_offline_expert_batch(
                batch_size=batch_size,
                required_keys=required_keys,
            )
            bc_metrics = self._offline_policy_bc_step(expert_batch)

            if update_idx % offline_cfg.log_interval == 0 or update_idx in {
                1,
                total_updates,
            }:
                log_metrics = {
                    f"offline/policy_bc/{key}": value
                    for key, value in bc_metrics.items()
                }
                self.log_metrics(
                    log_metrics,
                    step=update_idx,
                    log_python=bool(self.config.logger.log_to_console),
                )
                elapsed = time.perf_counter() - start_time
                updates_per_sec = update_idx / elapsed
                remaining_updates = total_updates - update_idx
                eta_seconds = remaining_updates / updates_per_sec
                self.log.info(
                    "offline_policy_bc progress | update=%d/%d | %.1f%% | "
                    "elapsed=%.1fs | eta=%.1fs | updates_per_sec=%.3f",
                    update_idx,
                    total_updates,
                    100.0 * update_idx / total_updates,
                    elapsed,
                    eta_seconds,
                    updates_per_sec,
                )

        self.log.info(
            "offline_policy_bc complete | updates=%d | elapsed=%.1fs",
            total_updates,
            time.perf_counter() - start_time,
        )

    def _offline_pretrain_latent_step(self) -> dict[str, float]:
        if self._latent_learner is None:
            return {}
        if (
            str(self.config.ipmd.latent_learning.method).strip().lower()
            != "patch_vqvae"
        ):
            return {}
        return self._latent_learner.update(
            TensorDict({}, batch_size=[], device=self.device)
        )

    def _offline_pretrain_spectral_representation(self) -> None:
        offline_cfg = self.config.bilinear.offline_pretrain
        if not offline_cfg.requires_offline_data():
            return
        if offline_cfg.enabled:
            required_keys = self._offline_sr_required_keys()
            total_updates = int(offline_cfg.num_updates)
            start_time = time.perf_counter()
            self.log.info(
                "offline_pretrain start | updates=%d | batch_size=%d | log_interval=%d",
                total_updates,
                int(offline_cfg.batch_size),
                int(offline_cfg.log_interval),
            )
            for update_idx in range(1, offline_cfg.num_updates + 1):
                expert_batch = self._next_offline_expert_batch(
                    batch_size=offline_cfg.batch_size,
                    required_keys=required_keys,
                )
                latent_metrics = self._offline_pretrain_latent_step()
                self._store_transitions(expert_batch, action_key="expert_action")
                sr_metrics = self._train_sr_steps(1)

                if update_idx % offline_cfg.log_interval == 0 or update_idx in {
                    1,
                    offline_cfg.num_updates,
                }:
                    log_metrics = {
                        f"offline/sr/{key}": value for key, value in sr_metrics.items()
                    }
                    log_metrics.update(
                        {
                            f"offline/latent/{key}": value
                            for key, value in latent_metrics.items()
                        }
                    )
                    log_metrics["offline/sr/history_buffer_size"] = float(
                        len(self._sr_history_buffer)
                    )
                    self.log_metrics(
                        log_metrics,
                        step=update_idx,
                        log_python=bool(self.config.logger.log_to_console),
                    )
                    elapsed = time.perf_counter() - start_time
                    updates_per_sec = update_idx / elapsed
                    remaining_updates = total_updates - update_idx
                    eta_seconds = remaining_updates / updates_per_sec
                    self.log.info(
                        "offline_pretrain progress | update=%d/%d | %.1f%% | "
                        "elapsed=%.1fs | eta=%.1fs | updates_per_sec=%.3f | "
                        "history=%d",
                        update_idx,
                        total_updates,
                        100.0 * update_idx / total_updates,
                        elapsed,
                        eta_seconds,
                        updates_per_sec,
                        len(self._sr_history_buffer),
                    )

            self.bilinear_rep.update_ema(1.0)
            self.log.info(
                "offline_pretrain complete | updates=%d | elapsed=%.1fs | history=%d",
                total_updates,
                time.perf_counter() - start_time,
                len(self._sr_history_buffer),
            )
        self._offline_pretrain_policy_bc()

    def train(self) -> None:  # type: ignore[override]
        """Run offline SR pretraining, then the normal online IPMD loop."""
        self.validate_training()
        self._offline_pretrain_spectral_representation()
        metadata = self.init_metadata()

        try:
            for iteration_idx in range(metadata.total_iterations):
                iteration = self.collect(metadata, iteration_idx)
                self.prepare(iteration, metadata)
                self.iterate(iteration, metadata)
                self.record(iteration, metadata)
        finally:
            metadata.progress_bar.close()  # type: ignore[attr-defined]
            self.collector.shutdown()

    # -- Update / logging overrides --
    def prepare(
        self, iteration: PPOIterationData, metadata: PPOTrainingMetadata
    ) -> None:
        super().prepare(iteration, metadata)
        self._store_transitions(iteration.rollout)

    def iterate(
        self, iteration: PPOIterationData, metadata: PPOTrainingMetadata
    ) -> None:
        super().iterate(iteration, metadata)

        sr_metrics = self._train_sr_steps(self.config.bilinear.update_steps)
        self.bilinear_rep.update_ema(self.config.bilinear.ema_tau)

        for k, v in sr_metrics.items():
            self._pending_sr_metrics.setdefault(k, []).append(v)

        if self._use_latent_command:
            latent = cast(Tensor, iteration.rollout.get(self._latent_key))
            self._pending_sr_metrics.setdefault("z_l1_mean", []).append(
                latent.reshape(-1, self._latent_dim).abs().mean().item()
            )
            self._pending_sr_metrics.setdefault("z_l1_std", []).append(
                latent.reshape(-1, self._latent_dim).std(dim=0).mean().item()
            )
        self._pending_sr_metrics.setdefault("history_buffer_size", []).append(
            float(len(self._sr_history_buffer))
        )

    def log_metrics(self, metrics: Mapping[str, Any], **kwargs) -> None:
        if self._pending_sr_metrics:
            merged = dict(metrics)
            for k, vals in self._pending_sr_metrics.items():
                merged[f"sr/{k}"] = sum(vals) / len(vals)
            self._pending_sr_metrics.clear()
            super().log_metrics(merged, **kwargs)
        else:
            super().log_metrics(metrics, **kwargs)
