"""Soft Actor-Critic implementation following the structural pattern of PPO.

This class adapts the SOTA TorchRL SAC reference implementation to the
BaseAlgorithm / PPO style used in this repository (feature extractor +
ActorValueOperator + custom training loop with collector + replay buffer).

Key design notes:
 - Reuses BaseAlgorithm for env / collector / logging wiring.
 - Uses an optional shared feature extractor (config.use_feature_extractor).
 - Builds a ProbabilisticActor (Gaussian Tanh) and a Q-value network.
 - Leverages torchrl.objectives.sac.SACLoss with automatic entropy tuning.
 - Maintains a replay buffer and performs UTD (update-to-data) ratio updates
   per collected batch.
 - Provides Polyak target network updates through SoftUpdate helper.

Expected (additional) config keys (defaults provided if missing):
   replay_buffer:
       size: 1000000
       prb: False                # prioritized replay (not implemented here)
   optim:
       batch_size: 256           # replay sample batch size
       utd_ratio: 1.0            # updates per frames_per_batch * utd_ratio
       target_update_polyak: 0.005
       alpha_init: 1.0
       adam_eps: 1e-8
   collector:
       init_random_frames: 0     # initial random exploration frames

If keys are absent they will be inferred with safe defaults.
"""

from __future__ import annotations

from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn, Tensor
from torchrl.collectors import SyncDataCollector, MultiSyncDataCollector  # noqa: F401 (BaseAlgorithm may expect types)
from torchrl.data import (
    LazyTensorStorage,
    ReplayBuffer,
    TensorDictReplayBuffer,
)
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import (
    MLP,
    ProbabilisticActor,
    TanhNormal,
    ValueOperator,
    ActorValueOperator,
)
from torchrl.objectives.sac import SACLoss
from torchrl.objectives import SoftUpdate, group_optimizers
from torchrl._utils import timeit
import numpy as np  # noqa: F401
import tqdm
from torchrl.record.loggers import Logger

from rlopt.common.base_class import BaseAlgorithm


def _get_cfg(cfg: DictConfig, path: str, default):  # robust nested select
    try:
        val = OmegaConf.select(cfg, path)
    except Exception:  # pragma: no cover
        val = None
    return default if val in (None, "") else val


class SAC(BaseAlgorithm):
    """Soft Actor-Critic algorithm.

    The class mirrors the PPO structure (custom train loop) while adapting
    to SAC's off-policy setting with a replay buffer.
    """

    def __init__(
        self,
        env,
        config: DictConfig,
        policy_net: nn.Module | None = None,
        value_net: nn.Module
        | None = None,  # (unused; placeholder for ActorValueOperator)
        q_net: nn.Module | None = None,  # optional external Q-value module
        reward_estimator_net: nn.Module | None = None,
        replay_buffer: type[ReplayBuffer] = ReplayBuffer,
        logger: Logger | None = None,
        feature_extractor_net: nn.Module | None = None,
        **kwargs,
    ):
        super().__init__(
            env=env,
            config=config,
            policy_net=policy_net,
            value_net=value_net,
            q_net=q_net,
            reward_estimator_net=reward_estimator_net,
            replay_buffer=replay_buffer,
            logger=logger,
            feature_extractor_net=feature_extractor_net,
            **kwargs,
        )

        # SAC-specific components
        self.target_net_updater: SoftUpdate | None = None
        if isinstance(self.loss_module, SACLoss):
            tau_cfg = _get_cfg(self.config, "optim.target_update_polyak", 0.005)
            tau = float(tau_cfg) if not isinstance(tau_cfg, DictConfig) else 0.005
            self.target_net_updater = SoftUpdate(self.loss_module, eps=tau)

        # Track number of updates
        self.total_network_updates = 0

    # ---------------------------------------------------------------------
    # Network construction
    # ---------------------------------------------------------------------
    def _construct_feature_extractor(
        self, feature_extractor_net: nn.Module | None = None
    ) -> TensorDictModule:
        if feature_extractor_net is not None:
            return TensorDictModule(  # type: ignore[arg-type]
                module=feature_extractor_net,
                in_keys=list(self.total_input_keys),
                out_keys=["hidden"],
            )

        if self.config.use_feature_extractor:
            feat = MLP(
                in_features=self.total_input_shape,
                out_features=self.config.feature_extractor.output_dim,
                num_cells=self.config.feature_extractor.num_cells,
                activation_class=nn.ELU,
                device=self.device,
            )
            # Simple weight init
            for m in feat.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, a=0.01)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            return TensorDictModule(
                feat, in_keys=list(self.total_input_keys), out_keys=["hidden"]
            )  # type: ignore[arg-type]
        return TensorDictModule(  # type: ignore[arg-type]
            nn.Identity(),
            in_keys=list(self.total_input_keys),
            out_keys=list(self.total_input_keys),
        )

    def _construct_policy(
        self, policy_net: nn.Module | None = None
    ) -> TensorDictModule:
        # Actor: outputs loc, scale for TanhNormal distribution
        in_keys = (
            self.config.policy_in_keys
            if self.config.use_feature_extractor
            else list(self.total_input_keys)
        )
        input_dim = self.policy_input_shape
        action_dim = self.policy_output_shape

        if policy_net is None:
            net = MLP(
                in_features=input_dim,
                out_features=2 * action_dim,
                num_cells=self.config.policy.num_cells,
                activation_class=nn.ELU,
                device=self.device,
            )
        else:
            net = policy_net

        net = nn.Sequential(
            net, NormalParamExtractor(scale_mapping="biased_softplus_1.0")
        )
        module = TensorDictModule(  # type: ignore[arg-type]
            module=net,
            in_keys=list(in_keys),
            out_keys=["loc", "scale"],
        )
        dist_kwargs = {
            "low": self.env.action_spec_unbatched.space.low.to(self.device),  # type: ignore
            "high": self.env.action_spec_unbatched.space.high.to(self.device),  # type: ignore
            "tanh_loc": False,
        }
        return ProbabilisticActor(
            module=module,
            spec=self.env.full_action_spec_unbatched.to(self.device),  # type: ignore
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            distribution_kwargs=dist_kwargs,
            default_interaction_type=ExplorationType.RANDOM,
            return_log_prob=False,
        )

    def _construct_value_function(
        self, value_net: nn.Module | None = None
    ) -> TensorDictModule:
        """SAC does not use a state-value function explicitly.
        We return a minimal dummy ValueOperator (not used in loss) to satisfy
        ActorValueOperator structure.
        """
        in_keys = (
            self.config.value_net_in_keys
            if self.config.use_feature_extractor
            else list(self.total_input_keys)
        )
        if value_net is None:
            net = MLP(
                in_features=self.value_input_shape,
                out_features=1,
                num_cells=[64],
                activation_class=nn.ELU,
                device=self.device,
            )
        else:
            net = value_net
        return ValueOperator(net, in_keys=in_keys)

    def _construct_q_function(self, q_net: nn.Module | None = None) -> TensorDictModule:
        # Q-value network taking (obs, action) -> value
        in_keys = ["action"] + (
            self.config.policy_in_keys
            if self.config.use_feature_extractor
            else list(self.total_input_keys)
        )
        input_dim = self.policy_output_shape + self.policy_input_shape
        if q_net is None:
            net = MLP(
                in_features=input_dim,
                out_features=1,
                num_cells=self.config.value_net.num_cells,
                activation_class=nn.ELU,
                device=self.device,
            )
        else:
            net = q_net
        return ValueOperator(module=net, in_keys=in_keys)

    def _construct_actor_critic(self) -> TensorDictModule:
        return ActorValueOperator(
            common_operator=self.feature_extractor,
            policy_operator=self.policy,
            value_operator=self.value_function,  # dummy / unused in SAC
        )

    # ------------------------------------------------------------------
    # Loss / Optim / Buffer
    # ------------------------------------------------------------------
    def _construct_loss_module(self) -> nn.Module:
        alpha_init_cfg = _get_cfg(self.config, "optim.alpha_init", 1.0)
        alpha_init = (
            float(alpha_init_cfg) if not isinstance(alpha_init_cfg, DictConfig) else 1.0
        )
        gamma_cfg = _get_cfg(self.config, "loss.gamma", 0.99)
        gamma = float(gamma_cfg) if not isinstance(gamma_cfg, DictConfig) else 0.99
        loss = SACLoss(
            actor_network=self.actor_critic.get_policy_operator(),
            qvalue_network=self.q_function,
            num_qvalue_nets=2,
            alpha_init=alpha_init,
            loss_function=_get_cfg(self.config, "optim.loss_function", "l2"),
            delay_actor=False,
            delay_qvalue=True,
        )
        loss.make_value_estimator(gamma=gamma)
        return loss

    def _configure_optimizers(self) -> torch.optim.Optimizer:
        lr_cfg = _get_cfg(self.config, "optim.lr", 3e-4)
        lr = float(lr_cfg) if not isinstance(lr_cfg, DictConfig) else 3e-4
        wd_cfg = _get_cfg(self.config, "optim.weight_decay", 0.0)
        weight_decay = float(wd_cfg) if not isinstance(wd_cfg, DictConfig) else 0.0
        eps_cfg = _get_cfg(self.config, "optim.adam_eps", 1e-8)
        adam_eps = float(eps_cfg) if not isinstance(eps_cfg, DictConfig) else 1e-8

        actor_params = list(self.actor_critic.get_policy_operator().parameters())
        q_params = list(self.q_function.parameters()) if self.q_function else []
        alpha_params: list[torch.nn.Parameter] = []
        if isinstance(getattr(self, "loss_module", None), SACLoss):
            alpha_params = [self.loss_module.log_alpha]  # type: ignore[attr-defined]

        actor_optim = torch.optim.Adam(
            actor_params, lr=lr, weight_decay=weight_decay, eps=adam_eps
        )
        critic_optim = torch.optim.Adam(
            q_params, lr=lr, weight_decay=weight_decay, eps=adam_eps
        )
        if alpha_params:
            alpha_optim = torch.optim.Adam(alpha_params, lr=3e-4)
            return group_optimizers(actor_optim, critic_optim, alpha_optim)
        return group_optimizers(actor_optim, critic_optim)

    def _construct_data_buffer(self) -> ReplayBuffer:
        cap_cfg = _get_cfg(self.config, "replay_buffer.size", 1_000_000)
        bs_cfg = _get_cfg(self.config, "optim.batch_size", 256)
        capacity = int(cap_cfg) if not isinstance(cap_cfg, DictConfig) else 1_000_000
        batch_size = int(bs_cfg) if not isinstance(bs_cfg, DictConfig) else 256
        storage = LazyTensorStorage(capacity, device=self.device)
        return TensorDictReplayBuffer(storage=storage, batch_size=batch_size)

    def _construct_trainer(self):  # type: ignore[override]
        return None

    # ------------------------------------------------------------------
    # Training Loop
    # ------------------------------------------------------------------
    def train(self) -> None:  # type: ignore[override]
        cfg = self.config
        frames_per_batch = cfg.collector.frames_per_batch
        total_frames = cfg.collector.total_frames
        utd_ratio = float(_get_cfg(cfg, "optim.utd_ratio", 1.0))
        init_random_frames = int(_get_cfg(cfg, "collector.init_random_frames", 0))
        batch_size = int(_get_cfg(cfg, "optim.batch_size", 256))

        collected_frames = 0
        collector_iter = iter(self.collector)
        pbar = tqdm.tqdm(total=total_frames)
        updates_per_batch = max(1, int(frames_per_batch * utd_ratio / batch_size))

        while collected_frames < total_frames:
            with timeit("collect"):
                tensordict = next(collector_iter)

            frames = tensordict.numel()
            collected_frames += frames
            pbar.update(frames)

            with timeit("replay_extend"):
                self.data_buffer.extend(tensordict.reshape(-1))  # type: ignore[arg-type]

            self.collector.update_policy_weights_()

            metrics: dict[str, Any] = {}

            if collected_frames >= init_random_frames:
                with timeit("train"):
                    loss_list: list[TensorDict] = []
                    for _ in range(updates_per_batch):
                        batch = self.data_buffer.sample()
                        loss_td = self.loss_module(batch)
                        combined = (
                            loss_td["loss_actor"]
                            + loss_td["loss_qvalue"]
                            + loss_td.get(
                                "loss_alpha", torch.zeros_like(loss_td["loss_actor"])
                            )
                        )
                        combined.backward()
                        self.optim.step()
                        self.optim.zero_grad(set_to_none=True)
                        if self.target_net_updater is not None:
                            self.target_net_updater.step()
                        loss_list.append(loss_td.detach())
                actor_stack = torch.stack(
                    [td_loss["loss_actor"] for td_loss in loss_list]
                )
                metrics["train/actor_loss"] = actor_stack.mean()
                q_stack = torch.stack([td_loss["loss_qvalue"] for td_loss in loss_list])
                metrics["train/q_loss"] = q_stack.mean()
                if "loss_alpha" in loss_list[0]:
                    alpha_stack = torch.stack(
                        [td_loss["loss_alpha"] for td_loss in loss_list]
                    )
                    metrics["train/alpha_loss"] = alpha_stack.mean()
                    metrics["train/alpha"] = self.loss_module.alpha  # type: ignore[attr-defined]
                    metrics["train/entropy"] = self.loss_module.target_entropy  # type: ignore[attr-defined]

            done_mask = (
                tensordict["next", "done"]
                if tensordict["next", "done"].any()
                else tensordict["next", "truncated"]
            )
            ep_rewards = tensordict["next", "episode_reward"][done_mask]
            if len(ep_rewards) > 0:
                metrics["train/reward"] = ep_rewards.mean()
                if "step_count" in tensordict["next"]:
                    ep_len = tensordict["next", "step_count"][done_mask]
                    metrics["train/episode_length"] = ep_len.float().mean()

            if self.logger is not None and metrics:
                for k, v in metrics.items():
                    if isinstance(v, Tensor):
                        self.logger.log_scalar(k, float(v.item()), collected_frames)
                    else:
                        self.logger.log_scalar(k, v, collected_frames)

        pbar.close()
        self.collector.shutdown()

    # ------------------------------------------------------------------
    # Unsupported base methods for SAC in this structure
    # ------------------------------------------------------------------
    def _update_policy(self, batch: TensorDict) -> dict[str, float]:  # pragma: no cover
        msg = "SAC uses custom update inside train()."  # noqa: TRY003
        raise NotImplementedError(msg)

    def _compute_action(self, tensordict: TensorDict) -> TensorDict:  # pragma: no cover
        msg = "Use actor_critic.get_policy_operator() instead."  # noqa: TRY003
        raise NotImplementedError(msg)

    def _compute_returns(self, rollout: Tensor) -> Tensor:  # pragma: no cover
        msg = "Return computation handled by SACLoss value estimator."  # noqa: TRY003
        raise NotImplementedError(msg)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict(self, obs: Tensor | np.ndarray) -> Tensor:  # type: ignore[override]
        obs = torch.as_tensor([obs], device=self.device)
        policy_op = self.actor_critic.get_policy_operator()
        policy_op.eval()
        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
            td = TensorDict(
                dict.fromkeys(self.total_input_keys, obs),
                batch_size=[1],
                device=self.device,
            )
            td = policy_op(td)
            return td.get("action")


__all__ = ["SAC"]
