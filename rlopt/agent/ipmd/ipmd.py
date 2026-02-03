from __future__ import annotations

import functools
import logging
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import torch
import tqdm
from tensordict import TensorDict
from tensordict.nn import InteractionType
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torchrl._utils import timeit
from torchrl.collectors import SyncDataCollector
from torchrl.data import (
    LazyMemmapStorage,
    LazyTensorStorage,
    ReplayBuffer,
    TensorDictReplayBuffer,
)
from torchrl.data.replay_buffers.samplers import RandomSampler
from torchrl.envs.utils import set_exploration_type
from torchrl.modules import MLP
from torchrl.record.loggers import Logger

from rlopt.agent.ppo.ppo import PPO, PPORLOptConfig
from rlopt.type_aliases import OptimizerClass
from rlopt.utils import get_activation_class, log_info

# Suppress torch.compile CUDA graph diagnostic messages
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
logging.getLogger("torch._inductor").setLevel(logging.ERROR)


@dataclass
class IPMDConfig:
    """IPMD-specific configuration (PPO-based)."""

    # Reward estimator network and loss settings
    reward_num_cells: tuple[int, ...] = (256, 256)
    """Hidden layer sizes for the reward estimator MLP."""

    reward_activation: str = "elu"
    """Activation for reward estimator MLP."""

    reward_init: str = "orthogonal"
    """Weight init for reward estimator."""

    reward_loss_coeff: float = 1.0
    """Scale for (sum r_pi - sum r_expert)."""

    reward_l2_coeff: float = 1e-4
    """L2 regularization weight for reward parameters."""

    reward_detach_features: bool = True
    """Detach features when computing reward loss (avoid leaking grads)."""

    use_estimated_rewards_for_ppo: bool = False
    """Whether to use estimated rewards instead of environment rewards for PPO (GAE + value target).

    Default is False. Set to True to train PPO on estimated rewards.
    """

    expert_batch_size: int | None = None
    """Batch size for expert data sampling. If None, uses the same as mini_batch_size."""

    detach_reward_when_used_for_ppo: bool = True
    """Detach the estimated reward when injecting into PPO (GAE/reward).

    Prevents PPO updates from backpropagating into the reward estimator.
    The reward network is then trained solely via the IPMD objective.
    """


@dataclass
class IPMDRLOptConfig(PPORLOptConfig):
    """IPMD configuration extending PPORLOptConfig (PPO-based)."""

    ipmd: IPMDConfig = field(default_factory=IPMDConfig)
    """IPMD configuration."""


class IPMD(PPO):
    """IPMD algorithm with PPO as the base RL algorithm.

    Uses the same on-policy rollout + GAE + multiple epochs over mini-batches as PPO,
    and adds a reward estimator r(s, a, s') trained with the IPMD objective
    (policy estimated return - expert estimated return). Optionally uses estimated
    rewards for PPO updates.
    """

    def __init__(
        self,
        env,
        config: IPMDRLOptConfig,
        policy_net: torch.nn.Module | None = None,
        value_net: torch.nn.Module | None = None,
        q_net: torch.nn.Module | None = None,
        replay_buffer: type[ReplayBuffer] = ReplayBuffer,
        logger: Logger | None = None,
        feature_extractor_net: torch.nn.Module | None = None,
        **kwargs,
    ):
        self.config = cast(IPMDRLOptConfig, config)
        self.config: IPMDRLOptConfig
        self.env = env

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

        self.config = cast(IPMDRLOptConfig, self.config)

        # Reward estimator: r(s, a, s') -> scalar
        self.reward_estimator: torch.nn.Module = self._construct_reward_estimator()
        self.reward_estimator.to(self.device)

        # Expert data source
        self._expert_buffer: TensorDictReplayBuffer | None = None
        self._warned_no_expert = False

        # Re-create optimizer to include reward_estimator parameters
        self.optim = self._configure_optimizers()

    def _compile_components(self) -> None:
        """Compile update (fixed signature for torch.compile and CUDA graphs)."""
        cfg = self.config
        assert isinstance(cfg, IPMDRLOptConfig)
        if not cfg.compile.compile:
            return
        compile_mode = cfg.compile.compile_mode or (
            "default" if cfg.compile.cudagraphs else "reduce-overhead"
        )
        self.update = torch.compile(self.update, mode=compile_mode)  # type: ignore[method-assign]
        self.adv_module = torch.compile(self.adv_module, mode=compile_mode)  # type: ignore[method-assign]

    def _construct_reward_estimator(self) -> torch.nn.Module:
        """Create reward network mapping [phi(s), a, phi(s')] to scalar."""
        cfg = self.config
        assert isinstance(cfg, IPMDRLOptConfig)
        obs_dim = int(self.env.observation_spec["observation"].shape[-1])
        act_dim = self.env.action_spec.shape[-1]
        in_dim = obs_dim * 2 + act_dim

        net = MLP(
            in_features=in_dim,
            out_features=1,
            num_cells=list(cfg.ipmd.reward_num_cells),
            activation_class=get_activation_class(cfg.ipmd.reward_activation),
            device=self.device,
        )
        self._initialize_weights(net, cfg.ipmd.reward_init)
        return net

    def _set_optimizers(
        self, optimizer_cls: OptimizerClass, optimizer_kwargs: dict[str, Any]
    ) -> list[torch.optim.Optimizer]:
        """Single optimizer for actor-critic and reward estimator (PPO + IPMD)."""
        base_optimizers = super()._set_optimizers(optimizer_cls, optimizer_kwargs)
        # PPO uses one optimizer for actor_critic; add reward_estimator to the same group
        assert len(base_optimizers) == 1
        all_params = list(self.actor_critic.parameters()) + list(
            self.reward_estimator.parameters()
        )
        return [optimizer_cls(all_params, **optimizer_kwargs)]

    # -------------------------
    # Expert data API
    # -------------------------
    def set_expert_buffer(self, buffer: TensorDictReplayBuffer) -> None:
        """Attach an expert replay buffer with keys observation, action, ('next', 'observation')."""
        self._expert_buffer = buffer

    def create_expert_buffer(
        self, expert_data: TensorDict, buffer_size: int | None = None
    ) -> TensorDictReplayBuffer:
        """Create an expert replay buffer from expert demonstration data."""
        if buffer_size is None:
            buffer_size = expert_data.numel()

        cfg = self.config
        assert isinstance(cfg, IPMDRLOptConfig)
        sampler = RandomSampler()
        scratch_dir = cfg.collector.scratch_dir
        device = cfg.device
        batch_size = cfg.loss.mini_batch_size
        shared = cfg.collector.shared
        prefetch = cfg.collector.prefetch

        storage_cls = (
            functools.partial(LazyTensorStorage, device=device)
            if not scratch_dir
            else functools.partial(
                LazyMemmapStorage, device="cpu", scratch_dir=scratch_dir
            )
        )

        expert_buffer = TensorDictReplayBuffer(
            pin_memory=False,
            prefetch=prefetch,
            sampler=sampler,
            storage=storage_cls(max_size=buffer_size, compilable=cfg.compile.compile),
            batch_size=batch_size,
            shared=shared,
        )
        expert_buffer.extend(expert_data.reshape(-1))
        if scratch_dir:
            expert_buffer.append_transform(lambda td: td.to(device))  # type: ignore[arg-type]
        return expert_buffer

    def _check_expert_batch_keys(self, expert_batch: TensorDict) -> bool:
        required_keys = ["observation", "action", ("next", "observation")]
        available_keys = expert_batch.keys(True)
        missing = [key for key in required_keys if key not in available_keys]
        if missing:
            self.log.warning("Expert batch missing required keys: %s", missing)
            return False
        return True

    def _next_expert_batch(self) -> TensorDict | None:
        if self._expert_buffer is None:
            return None
        try:
            expert_batch = cast(TensorDict, self._expert_buffer.sample())
            assert isinstance(self.config, IPMDRLOptConfig)
            if (
                self.config.ipmd.expert_batch_size is not None
                and expert_batch.numel() > self.config.ipmd.expert_batch_size
            ):
                expert_batch = cast(
                    TensorDict,
                    expert_batch[: self.config.ipmd.expert_batch_size],
                )
            return expert_batch
        except Exception:
            return None

    def _dummy_expert_batch(self, batch: TensorDict) -> TensorDict:
        """Return a single-transition expert batch with same structure as batch (for compile/CUDA graph)."""
        return batch.select("observation", "action", ("next", "observation")).clone()[
            :1
        ]

    def _dummy_loss_tensordict(self) -> TensorDict:
        """Dummy loss TensorDict with same keys as update output (for compile/fallback)."""
        return TensorDict(
            {
                "loss_critic": torch.tensor(0.0, device=self.device),
                "loss_objective": torch.tensor(0.0, device=self.device),
                "loss_entropy": torch.tensor(0.0, device=self.device),
                "loss_reward_diff": torch.tensor(0.0, device=self.device),
                "loss_reward_l2": torch.tensor(0.0, device=self.device),
                "estimated_reward_mean": torch.tensor(0.0, device=self.device),
                "estimated_reward_std": torch.tensor(0.0, device=self.device),
                "expert_reward_mean": torch.tensor(0.0, device=self.device),
                "expert_reward_std": torch.tensor(0.0, device=self.device),
            },
            batch_size=[],
        )

    def _reward_from_batch(self, td: TensorDict | Any) -> Tensor:
        """Compute estimated reward for a batch of transitions (obs, action, next_obs)."""
        obs = cast(Tensor, td.get("observation"))
        act = cast(Tensor, td.get("action"))
        next_obs = cast(Tensor, td.get(("next", "observation")))

        assert isinstance(self.config, IPMDRLOptConfig)
        if self.config.ipmd.reward_detach_features:
            obs = obs.detach()
            next_obs = next_obs.detach()
            act = act.detach()

        x = torch.cat([obs, act, next_obs], dim=-1)
        return self.reward_estimator(x).squeeze(-1)

    def update(
        self,
        batch: TensorDict,
        num_network_updates: int,
        expert_batch: TensorDict,
        has_expert: Tensor,
    ) -> tuple[TensorDict, int]:
        """PPO update plus IPMD reward loss; fixed path for torch.compile and CUDA graphs."""
        self.optim.zero_grad(set_to_none=True)
        assert isinstance(self.config, IPMDRLOptConfig)

        # 1) PPO loss
        loss: TensorDict = self.loss_module(batch)
        critic_loss = loss["loss_critic"]
        actor_loss = loss["loss_objective"] + loss["loss_entropy"]
        total_ppo_loss = critic_loss + actor_loss
        total_ppo_loss.backward()

        output_loss = loss.clone().detach_()

        # 2) IPMD reward loss (always computed; scaled by has_expert for fixed graph)
        r_pi = self._reward_from_batch(batch)
        r_exp = self._reward_from_batch(expert_batch.to(self.device))
        diff = r_pi.sum() - r_exp.sum()
        l2 = torch.zeros((), device=self.device)
        for p in self.reward_estimator.parameters():
            l2 = l2 + p.pow(2).sum()
        total_reward_loss = (
            float(self.config.ipmd.reward_loss_coeff) * diff
            + float(self.config.ipmd.reward_l2_coeff) * l2
        ) * has_expert
        total_reward_loss.backward()

        reward_diff = diff.detach()
        reward_l2 = l2.detach()

        # Gradient clipping (always call for fixed graph)
        grad_params = list(self.actor_critic.parameters()) + list(
            self.reward_estimator.parameters()
        )
        max_grad_norm = getattr(self.config.optim, "max_grad_norm", None) or 1e10
        grad_norm_tensor = clip_grad_norm_(grad_params, max_grad_norm)

        self.optim.step()
        if self.lr_scheduler and self.lr_scheduler_step == "update":
            self.lr_scheduler.step()

        output_loss.set("alpha", torch.tensor(1.0, device=self.device))
        output_loss.set("loss_reward_diff", reward_diff)
        output_loss.set("loss_reward_l2", reward_l2)
        output_loss.set("grad_norm", grad_norm_tensor.detach())
        output_loss.set(
            "lr",
            torch.tensor(
                self.optim.param_groups[0]["lr"],
                device=self.device,
                dtype=torch.float32,
            ),
        )
        output_loss.set("skipped_update", torch.tensor(False, device=self.device))

        with torch.no_grad():
            diag_rewards = self._reward_from_batch(batch)
            output_loss.set("estimated_reward_mean", diag_rewards.mean())
            output_loss.set("estimated_reward_std", diag_rewards.std())
            output_loss.set("expert_reward_mean", r_exp.mean().nan_to_num(0.0))
            output_loss.set("expert_reward_std", r_exp.std().nan_to_num(0.0))

        return output_loss, num_network_updates + 1

    def train(self) -> None:  # type: ignore[override]
        """On-policy train loop (PPO-style) with optional reward replacement and IPMD logging."""
        cfg = self.config
        assert isinstance(cfg, IPMDRLOptConfig)

        collected_frames = 0
        num_network_updates = torch.zeros((), dtype=torch.int64, device=self.device)
        pbar = tqdm.tqdm(total=self.config.collector.total_frames)

        num_mini_batches = cfg.collector.frames_per_batch // cfg.loss.mini_batch_size
        if cfg.collector.frames_per_batch % cfg.loss.mini_batch_size != 0:
            num_mini_batches += 1

        self.total_network_updates = (
            (cfg.collector.total_frames // cfg.collector.frames_per_batch)
            * cfg.loss.epochs
            * num_mini_batches
        )

        cfg_loss_ppo_epochs: int = cfg.loss.epochs
        cfg_loss_anneal_clip_eps: bool = cfg.ppo.anneal_clip_epsilon
        cfg_loss_clip_epsilon: float = cfg.ppo.clip_epsilon

        losses = TensorDict(batch_size=[cfg_loss_ppo_epochs, num_mini_batches])  # type: ignore

        self.collector: SyncDataCollector
        collector_iter = iter(self.collector)
        total_iter = len(self.collector)
        policy_op = self.actor_critic.get_policy_operator()

        if self._expert_buffer is None and not self._warned_no_expert:
            logging.getLogger(__name__).warning(
                "Expert buffer not set; reward estimator updates will use dummy batch (has_expert=0)."
            )
            self._warned_no_expert = True

        for _i in range(total_iter):
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
                    episode_rewards_mean = np.mean(episode_rewards.cpu().tolist())
                    metrics_to_log.update(
                        {
                            "episode/length": np.mean(self.episode_lengths),
                            "episode/return": np.mean(self.episode_rewards),
                            "train/reward": episode_rewards_mean,
                        }
                    )

            # Optionally replace env rewards with estimated rewards for PPO (before GAE)
            if cfg.ipmd.use_estimated_rewards_for_ppo:
                with torch.no_grad():
                    est_rew = self._reward_from_batch(data)
                if cfg.ipmd.detach_reward_when_used_for_ppo:
                    est_rew = est_rew.detach()
                # Match shape of ("next", "reward")
                data = data.clone()
                data.set(("next", "reward"), est_rew)

            self.data_buffer.empty()
            with timeit("training"):
                for j in range(cfg_loss_ppo_epochs):
                    with torch.no_grad(), timeit("adv"):
                        data = self.adv_module(data)
                        if getattr(self.config.compile, "compile_mode", None):
                            data = data.clone()

                    with timeit("rb - extend"):
                        self.data_buffer.extend(data.reshape(-1))

                    for k, batch in enumerate(self.data_buffer):
                        kl_context = None
                        if (cfg.optim.scheduler or "").lower() == "adaptive":
                            kl_context = self._prepare_kl_context(batch, policy_op)
                        # Fixed inputs for torch.compile / CUDA graph
                        expert_batch_raw = self._next_expert_batch()
                        if (
                            expert_batch_raw is None
                            or not self._check_expert_batch_keys(expert_batch_raw)
                        ):
                            expert_batch = self._dummy_expert_batch(batch)
                            has_expert = torch.tensor(
                                0.0, device=self.device, dtype=torch.float32
                            )
                        else:
                            expert_batch = expert_batch_raw.to(self.device)
                            has_expert = torch.tensor(
                                1.0, device=self.device, dtype=torch.float32
                            )
                        with timeit("update"):
                            loss, num_network_updates = self.update(
                                batch,
                                num_network_updates,
                                expert_batch,
                                has_expert,
                            )
                            loss = loss.clone()
                        if kl_context is not None:
                            kl_approx = self._compute_kl_after_update(
                                kl_context, policy_op
                            )
                            if kl_approx is not None:
                                loss.set("kl_approx", kl_approx.detach())
                                self._maybe_adjust_lr(kl_approx, cfg.optim)
                        num_network_updates = num_network_updates.clone()  # type: ignore
                        loss_keys = [
                            "loss_critic",
                            "loss_entropy",
                            "loss_objective",
                            "loss_reward_diff",
                            "loss_reward_l2",
                            "estimated_reward_mean",
                            "estimated_reward_std",
                            "expert_reward_mean",
                            "expert_reward_std",
                        ]
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
                        losses[j, k] = loss.select(
                            *[lk for lk in loss_keys if lk in loss]
                        )

                    if self.lr_scheduler and self.lr_scheduler_step == "epoch":
                        self.lr_scheduler.step()

            # Aggregate and log losses
            losses_mean = losses.apply(lambda x: x.float().mean(), batch_size=[])
            for key, value in losses_mean.items():  # type: ignore
                metrics_to_log[f"train/{key}"] = value.item()  # type: ignore
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

            if "Isaac" in cfg.env.env_name and hasattr(self.env, "log_infos"):
                log_info_dict: dict[str, Tensor] = self.env.log_infos.popleft()
                log_info(log_info_dict, metrics_to_log)

            metrics_to_log.update(timeit.todict(prefix="time"))  # type: ignore
            rate = pbar.format_dict.get("rate")
            if rate is not None:
                metrics_to_log["time/speed"] = rate
            self.log_metrics(metrics_to_log, step=collected_frames)
            self.collector.update_policy_weights_()

            postfix = {}
            if "train/step_reward_mean" in metrics_to_log:
                postfix["r_step"] = f"{metrics_to_log['train/step_reward_mean']:.2f}"
            if "episode/return" in metrics_to_log:
                postfix["r_ep"] = f"{metrics_to_log['episode/return']:.1f}"
            if "train/loss_objective" in metrics_to_log:
                postfix["pi_loss"] = f"{metrics_to_log['train/loss_objective']:.3f}"
            if "train/loss_reward_diff" in metrics_to_log:
                postfix["reward_diff"] = (
                    f"{metrics_to_log['train/loss_reward_diff']:.3f}"
                )
            if postfix:
                pbar.set_postfix(postfix)

            if (
                self.config.save_interval > 0
                and num_network_updates % self.config.save_interval == 0
            ):
                self.save_model(
                    path=self.log_dir / self.config.logger.save_path,
                    step=collected_frames,
                )

        pbar.close()
        self.collector.shutdown()

    def validate_ipmd_loss(
        self, test_batch: TensorDict, expert_batch: TensorDict
    ) -> dict[str, float]:
        """Validate IPMD loss computation (reward diff + L2) and PPO loss on test data."""
        for m in (self.actor_critic, self.reward_estimator):
            if hasattr(m, "eval"):
                m.eval()
        with torch.no_grad():
            estimated_rewards = self._reward_from_batch(test_batch)
            expert_rewards = self._reward_from_batch(expert_batch)
            reward_diff = (estimated_rewards.sum() - expert_rewards.sum()).item()
            l2_reg = sum(
                p.pow(2).sum().item() for p in self.reward_estimator.parameters()
            )
            ppo_loss_td = self.loss_module(test_batch)
            return {
                "reward_diff": reward_diff,
                "reward_l2": l2_reg,
                "estimated_reward_mean": estimated_rewards.mean().item(),
                "estimated_reward_std": estimated_rewards.std().item(),
                "expert_reward_mean": expert_rewards.mean().item(),
                "expert_reward_std": expert_rewards.std().item(),
                "ppo_loss_critic": ppo_loss_td["loss_critic"].item(),
                "ppo_loss_objective": ppo_loss_td["loss_objective"].item(),
                "ppo_loss_entropy": ppo_loss_td["loss_entropy"].item(),
            }

    def predict(self, obs: Tensor | np.ndarray) -> Tensor:  # type: ignore[override]
        """Predict action given observation (deterministic)."""
        obs = torch.as_tensor([obs], device=self.device)
        policy_op = self.actor_critic.get_policy_operator()
        policy_op.eval()
        with torch.no_grad(), set_exploration_type(InteractionType.DETERMINISTIC):
            input_keys = list(self.config.policy.input_keys)
            td = TensorDict(
                dict.fromkeys(input_keys, obs),
                batch_size=[1],
                device=self.device,
            )
            td = policy_op(td)
            return td.get("action")
