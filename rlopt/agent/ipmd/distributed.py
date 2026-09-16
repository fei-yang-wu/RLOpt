"""Distributed IPMD with a frozen command encoder and environment rewards."""

from __future__ import annotations

from rlopt.agent.ipmd.ipmd import IPMD
from rlopt.agent.ppo.distributed import DistributedPPO


class DistributedIPMD(DistributedPPO, IPMD):
    """Reuse IPMD commands/checkpoints and PPO's distributed synchronization."""

    def __init__(self, env, config, **kwargs):
        cfg = config.ipmd
        if cfg.use_latent_command and cfg.command_source != "hl_skill":
            msg = "Distributed IPMD requires a frozen hl_skill command encoder"
            raise ValueError(msg)
        if cfg.use_estimated_rewards_for_ppo:
            msg = "Distributed IPMD requires environment rewards"
            raise ValueError(msg)
        if cfg.hl_skill_finetune_enabled:
            msg = "Distributed IPMD does not support encoder fine-tuning"
            raise ValueError(msg)
        if any(
            float(getattr(cfg, name)) != 0
            for name in (
                "reward_loss_coeff",
                "reward_l2_coeff",
                "reward_grad_penalty_coeff",
                "reward_logit_reg_coeff",
                "reward_param_weight_decay_coeff",
                "bc_coef",
                "diversity_bonus_coeff",
                "rollout_bc_coef",
            )
        ):
            msg = "Distributed IPMD currently requires environment rewards without auxiliary updates"
            raise ValueError(msg)
        super().__init__(env, config, **kwargs)

    def _extra_checkpoint_state_dict(self):
        return {
            **super()._extra_checkpoint_state_dict(),
            **self._extra_model_state_dict(),
        }

    def _load_extra_checkpoint_state_dict(self, checkpoint):
        super()._load_extra_checkpoint_state_dict(checkpoint)
        self._load_extra_model_state_dict(checkpoint)
