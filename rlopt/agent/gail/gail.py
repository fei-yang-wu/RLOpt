"""PPO-based GAIL and AMP implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import torch
import tqdm
from tensordict import TensorDict
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torchrl._utils import timeit
from torchrl.collectors import SyncDataCollector

from rlopt.agent.ppo import PPO, PPOConfig, PPORLOptConfig
from rlopt.config_utils import (
    BatchKey,
    ObsKey,
    dedupe_keys,
    flatten_feature_tensor,
)

from .discriminator import Discriminator


class ExpertReplayBuffer(Protocol):
    def sample(self) -> TensorDict: ...

    def __len__(self) -> int: ...


@dataclass
class GAILConfig:
    """Configuration for PPO-based GAIL."""

    discriminator_hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    discriminator_activation: str = "relu"
    discriminator_lr: float = 3e-4
    discriminator_steps: int = 1
    discriminator_grad_clip_norm: float = 1.0
    discriminator_loss_coeff: float = 1.0
    expert_batch_size: int = 256
    use_gail_reward: bool = True
    gail_reward_coeff: float = 1.0
    proportion_env_reward: float = 0.0
    amp_reward_clip: bool = True
    discriminator_input_group: ObsKey | None = None
    """Optional observation group key for discriminator inputs (e.g. ``"policy"``)."""
    discriminator_input_keys: list[ObsKey] | None = None
    """Optional subset of observation keys for discriminator/reward input."""
    discriminator_use_action: bool = False
    """Whether discriminator additionally consumes ``action`` features."""


@dataclass
class GAILRLOptConfig(PPORLOptConfig):
    """PPO config extended with GAIL parameters."""

    ppo: PPOConfig = field(default_factory=PPOConfig)
    gail: GAILConfig = field(default_factory=GAILConfig)


@dataclass
class AMPRLOptConfig(GAILRLOptConfig):
    """Configuration alias for AMP (same fields as GAIL)."""


class GAIL(PPO):
    """GAIL with PPO policy optimization and discriminator reward."""

    def __init__(self, env, config: GAILRLOptConfig):
        self.config = cast(GAILRLOptConfig, config)
        self._expert_buffer: ExpertReplayBuffer | None = None
        self._warned_no_expert = False
        self._disc_obs_keys = self._resolve_discriminator_obs_keys(env, self.config)
        self._disc_obs_feature_ndims = {
            key: self._obs_key_feature_ndim(env, key) for key in self._disc_obs_keys
        }
        self._disc_obs_dim = sum(
            self._obs_key_feature_dim(env, key) for key in self._disc_obs_keys
        )
        action_spec = getattr(env, "action_spec_unbatched", env.action_spec)
        action_shape = tuple(int(dim) for dim in action_spec.shape)
        self._action_feature_ndim = len(action_shape)
        self._action_dim = int(np.prod(action_shape)) if action_shape else 1
        self.discriminator: Discriminator | None = None
        self.discriminator_optim: torch.optim.Optimizer | None = None
        super().__init__(env, config)
        self.log.info(
            "Initialized PPO-based GAIL (policy params=%d, discriminator params=%d, disc_keys=%s, use_action=%s)",
            sum(p.numel() for p in self.actor_critic.get_policy_operator().parameters()),
            sum(p.numel() for p in self.discriminator.parameters()) if self.discriminator else 0,
            self._disc_obs_keys,
            self.config.gail.discriminator_use_action,
        )

    @staticmethod
    def _compose_grouped_key(group: ObsKey, key: ObsKey) -> ObsKey:
        if isinstance(group, tuple):
            if isinstance(key, tuple):
                return (*group, *key)
            return (*group, key)
        if isinstance(key, tuple):
            return (group, *key)
        return (group, key)

    @classmethod
    def _resolve_discriminator_obs_keys(
        cls, env, cfg: GAILRLOptConfig
    ) -> list[ObsKey]:
        gail_cfg = cfg.gail
        group = gail_cfg.discriminator_input_group
        keys = gail_cfg.discriminator_input_keys
        if keys:
            base = list(keys)
            if group is not None:
                base = [cls._compose_grouped_key(group, key) for key in base]
        elif group is not None:
            base = [group]
        else:
            value_cfg = cfg.value_function
            base = (
                value_cfg.get_input_keys()
                if value_cfg is not None
                else cfg.policy.get_input_keys()
            )

        # Keep only keys that exist in current observation spec.
        available = set(env.observation_spec.keys(True))
        resolved: list[ObsKey] = []
        for key in dedupe_keys(list(base)):
            if key in available:
                resolved.append(cast(ObsKey, key))
                continue
            # If a grouped key was requested but observations are flattened, fallback.
            if isinstance(key, tuple) and len(key) > 0:
                flat_candidate = cast(ObsKey, key[-1])
                if flat_candidate in available:
                    resolved.append(flat_candidate)
                    continue
            msg = f"Discriminator observation key '{key}' not found in env.observation_spec."
            raise KeyError(msg)
        return resolved

    @staticmethod
    def _obs_key_feature_shape(env, key: ObsKey) -> tuple[int, ...]:
        shape = tuple(int(dim) for dim in env.observation_spec[key].shape)
        batch_prefix = tuple(int(dim) for dim in getattr(env, "batch_size", ()))
        if (
            len(batch_prefix) > 0
            and len(shape) >= len(batch_prefix)
            and shape[: len(batch_prefix)] == batch_prefix
        ):
            return shape[len(batch_prefix) :]
        return shape

    @classmethod
    def _obs_key_feature_ndim(cls, env, key: ObsKey) -> int:
        return len(cls._obs_key_feature_shape(env, key))

    @classmethod
    def _obs_key_feature_dim(cls, env, key: ObsKey) -> int:
        shape = cls._obs_key_feature_shape(env, key)
        return int(np.prod(shape)) if shape else 1

    def _set_optimizers(
        self, optimizer_cls: type[torch.optim.Optimizer], optimizer_kwargs: dict[str, Any]
    ) -> list[torch.optim.Optimizer]:
        optimizers = super()._set_optimizers(optimizer_cls, optimizer_kwargs)
        if self.discriminator is None:
            self.discriminator = Discriminator(
                observation_dim=self._disc_obs_dim,
                action_dim=(
                    self._action_dim if self.config.gail.discriminator_use_action else 0
                ),
                hidden_dims=self.config.gail.discriminator_hidden_dims,
                activation=self.config.gail.discriminator_activation,
            ).to(self.device)
        self.discriminator_optim = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.config.gail.discriminator_lr,
        )
        return optimizers

    def set_expert_buffer(self, expert_buffer: ExpertReplayBuffer) -> None:
        self._expert_buffer = expert_buffer
        self.log.info("Expert buffer attached: %d samples", len(expert_buffer))

    def _next_expert_batch(self) -> TensorDict | None:
        if self._expert_buffer is None:
            return None
        try:
            return self._expert_buffer.sample().to(self.device)
        except Exception as exc:  # pragma: no cover - defensive
            self.log.warning("Failed to sample expert batch: %s", exc)
            return None

    def _obs_features_from_td(self, td: TensorDict, *, detach: bool) -> Tensor:
        parts: list[Tensor] = []
        for key in self._disc_obs_keys:
            obs = cast(Tensor, td.get(cast(BatchKey, key)))
            obs = flatten_feature_tensor(obs, self._disc_obs_feature_ndims[key])
            parts.append(obs.detach() if detach else obs)
        if len(parts) == 1:
            return parts[0]
        return torch.cat(parts, dim=-1)

    def _action_features_from_td(self, td: TensorDict, *, detach: bool) -> Tensor:
        action = cast(Tensor, td.get("action"))
        action = flatten_feature_tensor(action, self._action_feature_ndim)
        return action.detach() if detach else action

    def _gail_discriminator_loss(
        self,
        expert_obs: Tensor,
        expert_action: Tensor | None,
        policy_obs: Tensor,
        policy_action: Tensor | None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        assert self.discriminator is not None
        if expert_action is None:
            expert_action = torch.zeros(
                expert_obs.shape[0], 0, device=expert_obs.device, dtype=expert_obs.dtype
            )
        if policy_action is None:
            policy_action = torch.zeros(
                policy_obs.shape[0], 0, device=policy_obs.device, dtype=policy_obs.dtype
            )
        return self.discriminator.compute_loss(
            expert_obs=expert_obs,
            expert_action=expert_action,
            policy_obs=policy_obs,
            policy_action=policy_action,
        )

    def _gail_reward(self, obs: Tensor, action: Tensor | None) -> Tensor:
        assert self.discriminator is not None
        if action is None:
            action = torch.zeros(obs.shape[0], 0, device=obs.device, dtype=obs.dtype)
        return self.discriminator.compute_reward(obs, action)

    def _update_discriminator(self, rollout_flat: TensorDict) -> dict[str, float]:
        if self._expert_buffer is None:
            if not self._warned_no_expert:
                self.log.warning(
                    "No expert buffer set. Training falls back to environment reward."
                )
                self._warned_no_expert = True
            return {}
        assert self.discriminator is not None
        assert self.discriminator_optim is not None

        policy_obs = self._obs_features_from_td(rollout_flat, detach=False).to(self.device)
        policy_action: Tensor | None = None
        if self.config.gail.discriminator_use_action:
            policy_action = self._action_features_from_td(rollout_flat, detach=False).to(
                self.device
            )
        if policy_obs.shape[0] == 0:
            return {}

        stats_accum: dict[str, float] = {}
        steps = max(1, int(self.config.gail.discriminator_steps))
        for _ in range(steps):
            expert_td = self._next_expert_batch()
            if expert_td is None:
                continue
            expert_obs = self._obs_features_from_td(expert_td, detach=False).to(self.device)
            expert_action: Tensor | None = None
            if self.config.gail.discriminator_use_action:
                expert_action = self._action_features_from_td(expert_td, detach=False).to(
                    self.device
                )
            if expert_obs.shape[0] == 0:
                continue

            batch_size = int(
                min(
                    self.config.gail.expert_batch_size,
                    expert_obs.shape[0],
                    policy_obs.shape[0],
                )
            )
            if batch_size <= 0:
                continue
            p_idx = torch.randint(
                policy_obs.shape[0], (batch_size,), device=self.device
            )
            e_idx = torch.randint(
                expert_obs.shape[0], (batch_size,), device=self.device
            )

            loss, info = self._gail_discriminator_loss(
                expert_obs[e_idx],
                expert_action[e_idx] if expert_action is not None else None,
                policy_obs[p_idx].detach(),
                policy_action[p_idx].detach() if policy_action is not None else None,
            )
            loss = loss * self.config.gail.discriminator_loss_coeff
            self.discriminator_optim.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(
                self.discriminator.parameters(),
                self.config.gail.discriminator_grad_clip_norm,
            )
            self.discriminator_optim.step()

            for key, value in info.items():
                stats_accum[key] = stats_accum.get(key, 0.0) + float(value.detach().item())

        if not stats_accum:
            return {}
        return {k: v / steps for k, v in stats_accum.items()}

    def _replace_rewards_with_discriminator(self, data: TensorDict) -> dict[str, float]:
        if not self.config.gail.use_gail_reward or self.discriminator is None:
            return {}

        with torch.no_grad():
            obs = self._obs_features_from_td(data, detach=True).to(self.device)
            action: Tensor | None = None
            if self.config.gail.discriminator_use_action:
                action = self._action_features_from_td(data, detach=True).to(self.device)
            disc_reward = self._gail_reward(obs, action)
            disc_reward = self.config.gail.gail_reward_coeff * disc_reward

            env_reward = data["next", "reward"].to(self.device)
            squeeze_last = env_reward.ndim == disc_reward.ndim + 1 and env_reward.shape[-1] == 1
            env_reward_base = env_reward.squeeze(-1) if squeeze_last else env_reward
            p_env = float(np.clip(self.config.gail.proportion_env_reward, 0.0, 1.0))
            mixed_reward = p_env * env_reward_base + (1.0 - p_env) * disc_reward
            if squeeze_last:
                mixed_reward = mixed_reward.unsqueeze(-1)
            data.set(("next", "reward"), mixed_reward)

        return {
            "gail/reward_mean": float(disc_reward.mean().item()),
            "gail/reward_std": float(disc_reward.std().item()),
        }

    def train(self) -> None:  # type: ignore[override]
        cfg = self.config
        assert isinstance(cfg, GAILRLOptConfig)

        collected_frames = 0
        num_network_updates = torch.zeros((), dtype=torch.int64, device=self.device)
        pbar = tqdm.tqdm(total=cfg.collector.total_frames)

        num_mini_batches = cfg.collector.frames_per_batch // cfg.loss.mini_batch_size
        if cfg.collector.frames_per_batch % cfg.loss.mini_batch_size != 0:
            num_mini_batches += 1
        self.total_network_updates = (
            (cfg.collector.total_frames // cfg.collector.frames_per_batch)
            * cfg.loss.epochs
            * num_mini_batches
        )

        cfg_loss_ppo_epochs = cfg.loss.epochs
        cfg_loss_anneal_clip_eps = cfg.ppo.anneal_clip_epsilon
        cfg_loss_clip_epsilon = cfg.ppo.clip_epsilon
        losses = TensorDict(batch_size=[cfg_loss_ppo_epochs, num_mini_batches])

        self.collector = cast(SyncDataCollector, self.collector)
        collector_iter = iter(self.collector)
        total_iter = len(self.collector)
        policy_op = self.actor_critic.get_policy_operator()

        for _i in range(total_iter):
            self.actor_critic.eval()
            self.adv_module.eval()
            with timeit("collecting"):
                data = next(collector_iter)

            metrics_to_log: dict[str, Any] = {}
            frames_in_batch = data.numel()
            collected_frames += frames_in_batch
            pbar.update(frames_in_batch)

            if ("next", "reward") in data.keys(True):
                step_rewards = data["next", "reward"]
                metrics_to_log.update(
                    {
                        "train/step_reward_mean": step_rewards.mean().item(),
                        "train/step_reward_std": step_rewards.std().item(),
                        "train/step_reward_max": step_rewards.max().item(),
                        "train/step_reward_min": step_rewards.min().item(),
                    }
                )
            if ("next", "episode_reward") in data.keys(True):
                episode_rewards = data["next", "episode_reward"][data["next", "done"]]
                if len(episode_rewards) > 0:
                    episode_length = data["next", "step_count"][data["next", "done"]]
                    self.episode_lengths.extend(episode_length.cpu().tolist())
                    self.episode_rewards.extend(episode_rewards.cpu().tolist())
                    metrics_to_log.update(
                        {
                            "episode/length": float(np.mean(self.episode_lengths)),
                            "episode/return": float(np.mean(self.episode_rewards)),
                            "train/reward": float(np.mean(episode_rewards.cpu().tolist())),
                        }
                    )

            discriminator_metrics = self._update_discriminator(data.reshape(-1))
            if discriminator_metrics:
                metrics_to_log.update(
                    {f"train/{k}": v for k, v in discriminator_metrics.items()}
                )
            reward_metrics = self._replace_rewards_with_discriminator(data)
            if reward_metrics:
                metrics_to_log.update(
                    {f"train/{k}": v for k, v in reward_metrics.items()}
                )

            self.data_buffer.empty()
            self.actor_critic.train()
            self.adv_module.train()
            with timeit("training"):
                for j in range(cfg_loss_ppo_epochs):
                    with torch.no_grad(), timeit("adv"):
                        data = self.adv_module(data)
                        if self.config.compile.compile_mode:
                            data = data.clone()

                    with timeit("rb - extend"):
                        data_reshape = data.reshape(-1)
                        self.data_buffer.extend(data_reshape)

                    for k, batch in enumerate(self.data_buffer):
                        kl_context = None
                        if (self.config.optim.scheduler or "").lower() == "adaptive":
                            kl_context = self._prepare_kl_context(batch, policy_op)
                        with timeit("update"):
                            loss, num_network_updates = self.update(  # type: ignore[misc]
                                batch, num_network_updates=num_network_updates
                            )
                            loss = loss.clone()
                        if kl_context is not None:
                            kl_approx = self._compute_kl_after_update(kl_context, policy_op)
                            if kl_approx is not None:
                                loss.set("kl_approx", kl_approx.detach())
                                self._maybe_adjust_lr(kl_approx, self.config.optim)

                        num_network_updates = num_network_updates.clone()
                        loss_keys = ["loss_critic", "loss_entropy", "loss_objective"]
                        optional_keys = [
                            "entropy",
                            "explained_variance",
                            "clip_fraction",
                            "value_clip_fraction",
                            "ESS",
                            "kl_approx",
                            "grad_norm",
                        ]
                        for key in optional_keys:
                            if key in loss:
                                loss_keys.append(key)
                        losses[j, k] = loss.select(*loss_keys)

                    if self.lr_scheduler and self.lr_scheduler_step == "epoch":
                        self.lr_scheduler.step()

            losses_mean = losses.apply(lambda x: x.float().mean(), batch_size=[])
            for key, value in losses_mean.items():
                metrics_to_log[f"train/{key}"] = value.item()
            metrics_to_log["train/lr"] = self.optim.param_groups[0]["lr"]

            clip_attr = getattr(self.loss_module, "clip_epsilon", None)
            if cfg_loss_anneal_clip_eps and isinstance(clip_attr, torch.Tensor):
                clip_epsilon_value = clip_attr.detach()
            else:
                clip_epsilon_value = torch.tensor(
                    cfg_loss_clip_epsilon,
                    device=self.device,
                    dtype=torch.float32,
                )
            metrics_to_log["train/clip_epsilon"] = clip_epsilon_value

            if "Isaac" in self.config.env.env_name and hasattr(self.env, "log_infos"):
                log_info_dict: dict[str, Tensor] = self.env.log_infos.popleft()
                for key, value in log_info_dict.items():
                    metrics_to_log[f"env/{key}"] = (
                        value.mean().item() if isinstance(value, Tensor) else value
                    )

            metrics_to_log.update(timeit.todict(prefix="time"))
            rate = pbar.format_dict.get("rate")
            if rate is not None:
                metrics_to_log["time/speed"] = rate

            self.log_metrics(metrics_to_log, step=collected_frames)
            self.collector.update_policy_weights_()

            if (
                self.config.save_interval > 0
                and num_network_updates % self.config.save_interval == 0
            ):
                self.save_model(
                    path=self.log_dir / self.config.logger.save_path,
                    step=int(collected_frames),
                )

        self.collector.shutdown()

    def save(self, path: str | Path) -> None:
        path = Path(path)
        torch.save(
            {
                "actor_critic": self.actor_critic.state_dict(),
                "discriminator": self.discriminator.state_dict() if self.discriminator else {},
                "config": self.config,
            },
            path,
        )
        self.log.info("Model saved to %s", path)

    def load(self, path: str | Path) -> None:
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor_critic.load_state_dict(checkpoint["actor_critic"])
        if self.discriminator is not None and "discriminator" in checkpoint:
            self.discriminator.load_state_dict(checkpoint["discriminator"])
        self.log.info("Model loaded from %s", path)


class AMP(GAIL):
    """Adversarial Motion Prior using PPO + least-squares discriminator."""

    def _gail_discriminator_loss(
        self,
        expert_obs: Tensor,
        expert_action: Tensor | None,
        policy_obs: Tensor,
        policy_action: Tensor | None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        assert self.discriminator is not None
        if expert_action is None:
            expert_action = torch.zeros(
                expert_obs.shape[0], 0, device=expert_obs.device, dtype=expert_obs.dtype
            )
        if policy_action is None:
            policy_action = torch.zeros(
                policy_obs.shape[0], 0, device=policy_obs.device, dtype=policy_obs.dtype
            )
        expert_logits = self.discriminator(expert_obs, expert_action).squeeze(-1)
        policy_logits = self.discriminator(policy_obs, policy_action).squeeze(-1)
        expert_target = torch.ones_like(expert_logits)
        policy_target = -torch.ones_like(policy_logits)
        loss = torch.mean((expert_logits - expert_target) ** 2) + torch.mean(
            (policy_logits - policy_target) ** 2
        )
        info = {
            "discriminator_loss": loss.detach(),
            "expert_d_mean": expert_logits.mean().detach(),
            "policy_d_mean": policy_logits.mean().detach(),
        }
        return loss, info

    def _gail_reward(self, obs: Tensor, action: Tensor | None) -> Tensor:
        assert self.discriminator is not None
        if action is None:
            action = torch.zeros(obs.shape[0], 0, device=obs.device, dtype=obs.dtype)
        logits = self.discriminator(obs, action).squeeze(-1)
        reward = 1.0 - 0.25 * (logits - 1.0) ** 2
        if self.config.gail.amp_reward_clip:
            reward = torch.clamp_min(reward, 0.0)
        return reward
