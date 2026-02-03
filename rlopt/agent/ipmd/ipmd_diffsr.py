from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import torch
import tqdm
from tensordict import TensorDict
from tensordict.nn import CudaGraphModule, TensorDictModule
from torch import Tensor, nn
from torchrl._utils import compile_with_warmup, timeit
from torchrl.data import ReplayBuffer
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal
from torchrl.record.loggers import Logger

from rlopt.agent.ipmd.ipmd import IPMD, IPMDRLOptConfig
from rlopt.agent.ipmd.network import DDPM
from rlopt.utils import log_info


@dataclass
class DiffSRConfig:
    """Diffusion model configuration for linear MDP feature learning."""

    embed_dim: int = 128
    feature_dim: int = 64
    phi_hidden_dims: tuple[int, ...] = (256, 256)
    mu_hidden_dims: tuple[int, ...] = (256, 256)
    reward_hidden_dims: tuple[int, ...] = (256, 256)
    rff_dim: int | None = None
    sample_steps: int = 64
    feature_lr: float = 3e-4
    reward_loss_coeff: float = 1.0
    pretrain_steps: int = 1000
    pretrain_min_samples: int | None = None
    update_steps_per_batch: int = 0
    detach_features_for_rl: bool = True
    freeze_diffusion: bool = True


@dataclass
class IPMDDiffSRConfig(IPMDRLOptConfig):
    """IPMD + DiffSR configuration."""

    diffsr: DiffSRConfig = field(default_factory=DiffSRConfig)

    def __post_init__(self) -> None:
        super().__post_init__()


class DiffusionLinearPolicy(nn.Module):
    """Linear Gaussian policy head driven by diffusion state features."""

    def __init__(
        self,
        diffusion: DDPM,
        action_dim: int,
        detach_features: bool,
    ) -> None:
        super().__init__()
        self.diffusion = diffusion
        self.detach_features = detach_features
        self.linear = nn.Linear(diffusion.embed_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.diffusion.encode_state(obs)
        if self.detach_features:
            features = features.detach()
        loc = self.linear(features)
        scale = torch.exp(self.log_std).clamp(min=1e-4)
        return loc, scale.expand_as(loc)


class DiffusionLinearQ(nn.Module):
    """Linear Q head over diffusion state-action features."""

    def __init__(
        self,
        diffusion: DDPM,
        feature_dim: int,
        detach_features: bool,
    ) -> None:
        super().__init__()
        self.diffusion = diffusion
        self.detach_features = detach_features
        self.linear = nn.Linear(feature_dim, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        features = self.diffusion.compute_feature(obs, action)
        if self.detach_features:
            features = features.detach()
        return self.linear(features)


class IPMDDiffSR(IPMD):
    """IPMD variant with diffusion-based linear MDP feature learning."""

    def __init__(
        self,
        env,
        config: IPMDDiffSRConfig,
        policy_net: nn.Module | None = None,
        q_net: nn.Module | None = None,
        replay_buffer: type[ReplayBuffer] = ReplayBuffer,
        logger: Logger | None = None,
        feature_extractor_net: nn.Module | None = None,
        **kwargs,
    ) -> None:
        self.config = cast(IPMDDiffSRConfig, config)
        self.env = env

        self.diffusion = self._construct_diffusion_model()
        self.diffusion.to(self.device)
        self._diffusion_ready = False
        self._diffusion_frozen = False
        self._warned_no_diffusion = False

        super().__init__(
            env=env,
            config=config,
            policy_net=policy_net,
            q_net=q_net,
            replay_buffer=replay_buffer,
            logger=logger,
            feature_extractor_net=feature_extractor_net,
            **kwargs,
        )

        self.diffusion_optim = torch.optim.Adam(
            self.diffusion.parameters(), lr=self.config.diffsr.feature_lr
        )

    def _construct_diffusion_model(self) -> DDPM:
        cfg = self.config.diffsr
        obs_dim = self.env.observation_spec["observation"].shape[-1]
        action_dim = self.env.action_spec_unbatched.shape[-1]
        return DDPM(
            obs_dim=obs_dim,
            action_dim=action_dim,
            embed_dim=cfg.embed_dim,
            feature_dim=cfg.feature_dim,
            phi_hidden_dims=cfg.phi_hidden_dims,
            mu_hidden_dims=cfg.mu_hidden_dims,
            reward_hidden_dims=cfg.reward_hidden_dims,
            rff_dim=cfg.rff_dim,
            sample_steps=cfg.sample_steps,
            device=self.device,
        )

    def _construct_policy(
        self, policy_net: torch.nn.Module | None = None
    ) -> TensorDictModule:
        """Construct linear policy with diffusion state features."""
        if policy_net is None:
            policy_head = DiffusionLinearPolicy(
                diffusion=self.diffusion,
                action_dim=self.env.action_spec_unbatched.shape[-1],  # type: ignore[arg-type]
                detach_features=self.config.diffsr.detach_features_for_rl,
            )
        else:
            policy_head = policy_net

        policy_td = TensorDictModule(
            module=policy_head,
            in_keys=list(self.config.policy.input_keys),
            out_keys=["loc", "scale"],
        )
        distribution_kwargs = {
            "low": self.env.action_spec_unbatched.space.low.to(self.device),  # type: ignore
            "high": self.env.action_spec_unbatched.space.high.to(self.device),  # type: ignore
            "tanh_loc": False,
        }
        return ProbabilisticActor(
            policy_td,
            in_keys=["loc", "scale"],
            spec=self.env.full_action_spec_unbatched.to(self.device),  # type: ignore
            distribution_class=TanhNormal,
            distribution_kwargs=distribution_kwargs,
            return_log_prob=False,
            default_interaction_type=ExplorationType.RANDOM,
        )

    def _construct_q_function(self, q_net: nn.Module | None = None) -> TensorDictModule:
        """Construct linear Q-function over diffusion features."""
        if q_net is None:
            q_head = DiffusionLinearQ(
                diffusion=self.diffusion,
                feature_dim=self.config.diffsr.feature_dim,
                detach_features=self.config.diffsr.detach_features_for_rl,
            )
        else:
            q_head = q_net

        in_keys = list(self.config.q_function.input_keys)
        if "action" not in in_keys:
            in_keys.append("action")

        return TensorDictModule(
            module=q_head,
            in_keys=in_keys,
            out_keys=["state_action_value"],
        )

    def _construct_reward_estimator(self) -> nn.Module:
        """Linear reward model over diffusion features."""
        cfg = self.config
        assert isinstance(cfg, IPMDDiffSRConfig)
        net = nn.Linear(cfg.diffsr.feature_dim, 1)
        self._initialize_weights(net, cfg.ipmd.reward_init)
        return net

    def _reward_from_batch(self, td: TensorDict) -> Tensor:
        obs = cast(Tensor, td.get("observation"))
        act = cast(Tensor, td.get("action"))
        phi_sa = self.diffusion.compute_feature(obs, act)
        assert isinstance(self.config, IPMDDiffSRConfig)
        if self.config.ipmd.reward_detach_features:
            phi_sa = phi_sa.detach()
        return self.reward_estimator(phi_sa).squeeze(-1)

    def _maybe_freeze_diffusion(self) -> None:
        if self._diffusion_frozen:
            return
        for param in self.diffusion.parameters():
            param.requires_grad = False
        self.diffusion.eval()
        self._diffusion_frozen = True

    def _diffusion_loss_from_batch(
        self, batch: TensorDict
    ) -> tuple[dict[str, float], torch.Tensor]:
        batch = batch.to(self.device)
        obs = cast(Tensor, batch.get("observation"))
        action = cast(Tensor, batch.get("action"))
        next_obs = cast(Tensor, batch.get(("next", "observation")))
        reward = batch.get("reward", None)
        if reward is None:
            reward = batch.get(("next", "reward"))
        if reward is None:
            reward = torch.zeros(obs.shape[0], 1, device=self.device)
        reward = cast(Tensor, reward).view(-1, 1)
        metrics, diffusion_loss, reward_loss = self.diffusion.compute_loss(
            obs, action, next_obs, reward
        )
        loss = diffusion_loss + self.config.diffsr.reward_loss_coeff * reward_loss
        return metrics, loss

    def _train_diffusion_steps(self, steps: int) -> dict[str, float]:
        metrics_accum: dict[str, float] = {}
        if steps <= 0:
            return metrics_accum

        self.diffusion.train()
        for _ in range(steps):
            batch = self.data_buffer.sample()
            if not self._validate_tensordict(
                batch, "diffusion:batch", raise_error=False
            ):
                continue
            metrics, loss = self._diffusion_loss_from_batch(batch)
            self.diffusion_optim.zero_grad(set_to_none=True)
            loss.backward()
            self.diffusion_optim.step()
            for key, value in metrics.items():
                metrics_accum[key] = metrics_accum.get(key, 0.0) + float(value)

        for key in list(metrics_accum.keys()):
            metrics_accum[key] /= float(steps)
        return metrics_accum

    def update(
        self,
        batch: TensorDict,
        num_network_updates: int,
        expert_batch: TensorDict,
        has_expert: Tensor,
    ) -> tuple[TensorDict, int]:
        if not self._diffusion_ready:
            if not self._warned_no_diffusion:
                self.log.warning("Diffusion model not trained; skipping IPMD updates.")
                self._warned_no_diffusion = True
            num_up = (
                num_network_updates + 1
                if isinstance(num_network_updates, int)
                else num_network_updates + 1
            )
            return self._dummy_loss_tensordict(), num_up
        return super().update(batch, num_network_updates, expert_batch, has_expert)

    def train(self) -> None:  # type: ignore[override]
        assert isinstance(self.config, IPMDDiffSRConfig)
        cfg = self.config
        frames_per_batch = cfg.collector.frames_per_batch
        total_frames = cfg.collector.total_frames
        utd_ratio = float(cfg.ipmd.utd_ratio)
        init_random_frames = int(self.config.collector.init_random_frames)
        num_updates = int(frames_per_batch * utd_ratio)
        pretrain_min_samples = (
            cfg.diffsr.pretrain_min_samples or cfg.loss.mini_batch_size
        )

        if cfg.diffsr.pretrain_steps == 0:
            self._diffusion_ready = True
            if cfg.diffsr.freeze_diffusion:
                self._maybe_freeze_diffusion()

        # Compile the update function if requested
        compile_mode = None
        if cfg.compile.compile:
            compile_mode = cfg.compile.compile_mode
            if compile_mode in ("", None):
                if cfg.compile.cudagraphs:
                    compile_mode = "default"
                else:
                    compile_mode = "reduce-overhead"

            self.log.info("Compiling update function with mode: %s", compile_mode)
            self.update = compile_with_warmup(self.update, mode=compile_mode, warmup=1)  # type: ignore[method-assign]

        # Only use CUDAGraphs on CUDA devices
        if cfg.compile.cudagraphs:
            if self.device.type == "cuda":
                warnings.warn(
                    "CudaGraphModule is experimental and may lead to silently wrong results. Use with caution.",
                    stacklevel=2,
                    category=UserWarning,
                )
                self.log.warning("Wrapping update with CudaGraphModule (experimental)")
                self.update = CudaGraphModule(
                    self.update, in_keys=[], out_keys=[], warmup=5
                )  # type: ignore[method-assign]
            else:
                self.log.warning(
                    "CUDAGraphs requested but device is %s, not CUDA. Skipping CUDAGraphs.",
                    self.device.type,
                )

        collected_frames = 0
        collector_iter = iter(self.collector)
        pbar = tqdm.tqdm(total=total_frames)

        while collected_frames < total_frames:
            timeit.printevery(num_prints=1000, total_count=total_frames, erase=True)
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
                            "train/reward": float(
                                episode_rewards.float().mean().item()
                            ),
                        }
                    )

            with timeit("replay_extend"):
                self.data_buffer.extend(data.reshape(-1))  # type: ignore[arg-type]

            if not self._diffusion_ready:
                if len(self.data_buffer) >= pretrain_min_samples:
                    with timeit("model_pretrain"):
                        model_metrics = self._train_diffusion_steps(
                            cfg.diffsr.pretrain_steps
                        )
                    for key, value in model_metrics.items():
                        metrics_to_log[f"model/{key}"] = value
                    self._diffusion_ready = True
                    if cfg.diffsr.freeze_diffusion:
                        self._maybe_freeze_diffusion()
                else:
                    metrics_to_log["model/pretrain_waiting"] = float(
                        len(self.data_buffer)
                    )

            if self._diffusion_ready and cfg.diffsr.update_steps_per_batch > 0:
                with timeit("model_update"):
                    model_metrics = self._train_diffusion_steps(
                        cfg.diffsr.update_steps_per_batch
                    )
                for key, value in model_metrics.items():
                    metrics_to_log[f"model/{key}"] = value

            with timeit("train"):
                losses = None
                if collected_frames >= init_random_frames and self._diffusion_ready:
                    losses_list: list[TensorDict] = []
                    num_network_updates = torch.zeros(
                        (), dtype=torch.int64, device=self.device
                    )
                    loss_keys = [
                        "loss_critic",
                        "loss_objective",
                        "loss_entropy",
                        "loss_reward_diff",
                        "loss_reward_l2",
                        "estimated_reward_mean",
                        "estimated_reward_std",
                        "expert_reward_mean",
                        "expert_reward_std",
                    ]
                    for i in range(num_updates):
                        with timeit("rb - sample"):
                            sampled_tensordict = self.data_buffer.sample()

                        expert_batch_raw = self._next_expert_batch()
                        if (
                            expert_batch_raw is None
                            or not self._check_expert_batch_keys(expert_batch_raw)
                        ):
                            expert_batch = self._dummy_expert_batch(sampled_tensordict)
                            has_expert = torch.tensor(
                                0.0, device=self.device, dtype=torch.float32
                            )
                        else:
                            expert_batch = expert_batch_raw.to(self.device)
                            has_expert = torch.tensor(
                                1.0, device=self.device, dtype=torch.float32
                            )

                        with timeit("update"):
                            torch.compiler.cudagraph_mark_step_begin()
                            loss, num_network_updates = self.update(
                                sampled_tensordict,
                                num_network_updates,
                                expert_batch,
                                has_expert,
                            )
                            loss = loss.clone()
                        losses_list.append(loss.select(*loss_keys))

                    if len(losses_list) > 0:
                        keys = list(losses_list[0].keys())
                        stacked = {}
                        for key in keys:
                            stacked[key] = torch.stack(
                                [ld.get(key) for ld in losses_list]
                            )
                        losses = TensorDict(stacked, batch_size=[len(losses_list)])

            if losses is not None:
                losses_mean = losses.apply(lambda x: x.float().mean(), batch_size=[])
                for key, value in list(losses_mean.items()):  # type: ignore
                    if isinstance(value, Tensor):
                        metrics_to_log[f"train/{key}"] = float(value.item())

                if hasattr(self.loss_module, "log_alpha"):
                    alpha = self.loss_module.log_alpha.exp().detach().cpu().item()
                    metrics_to_log["train/alpha"] = alpha

            metrics_to_log.update(timeit.todict(prefix="time"))  # type: ignore[arg-type]
            rate = pbar.format_dict.get("rate")
            if rate is not None:
                metrics_to_log["time/speed"] = rate
            if metrics_to_log:
                self.log_metrics(metrics_to_log, step=collected_frames)

            if "Isaac" in self.config.env.env_name and hasattr(self.env, "log_infos"):
                log_info_dict: dict[str, Tensor] = self.env.log_infos.popleft()
                log_info(log_info_dict, metrics_to_log)

            if metrics_to_log:
                postfix = {}
                if "train/step_reward_mean" in metrics_to_log:
                    postfix["r_step"] = (
                        f"{metrics_to_log['train/step_reward_mean']:.2f}"
                    )
                if "episode/return" in metrics_to_log:
                    postfix["r_ep"] = f"{metrics_to_log['episode/return']:.1f}"
                if "train/loss_actor" in metrics_to_log:
                    postfix["pi_loss"] = f"{metrics_to_log['train/loss_actor']:.3f}"
                if "train/alpha" in metrics_to_log:
                    postfix["alpha"] = f"{metrics_to_log['train/alpha']:.3f}"
                if postfix:
                    pbar.set_postfix(postfix)

            if (
                self.config.save_interval > 0
                and collected_frames
                % (self.config.save_interval * self.config.env.num_envs)
                == 0
            ):
                self.save_model(
                    path=self.log_dir / self.config.logger.save_path,
                    step=collected_frames,
                )

        pbar.close()
        self.collector.shutdown()

    def predict(self, obs: Tensor | np.ndarray) -> Tensor:  # type: ignore[override]
        obs = torch.as_tensor([obs], device=self.device)
        policy_op = self.actor_critic.get_policy_operator()
        policy_op.eval()
        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
            td = TensorDict(
                dict.fromkeys(self.config.policy.input_keys, obs),
                batch_size=[1],
                device=self.device,
            )
            td = policy_op(td)
            return td.get("action")
