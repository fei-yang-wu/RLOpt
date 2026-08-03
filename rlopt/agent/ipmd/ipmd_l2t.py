"""IPMD with a privileged teacher and an online-distilled student actor."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torchrl.data import ReplayBuffer
from torchrl.record.loggers import Logger

from rlopt.agent.ipmd.ipmd import IPMD, IPMDRLOptConfig
from rlopt.config_base import NetworkConfig
from rlopt.config_utils import ObsKey, normalize_batch_key
from rlopt.type_aliases import OptimizerClass


@dataclass
class IPMDL2TConfig:
    """Teacher-to-student distillation settings for :class:`IPMDL2T`."""

    student_policy: NetworkConfig = field(default_factory=NetworkConfig)
    """Deployable student actor layout and observation keys."""

    imitation_coeff: float = 1.0
    """Weight on executed-teacher-action mean-squared error."""

    student_learning_rate: float | None = None
    """Student LR; ``None`` reuses the configured teacher actor LR."""

    student_max_grad_norm: float | None = None
    """Student gradient clip; ``None`` reuses ``optim.max_grad_norm``."""

    student_latent_key: ObsKey = ("policy", "latent_command")
    """Deployable observation key receiving the mirrored teacher command."""

    def validate(self) -> None:
        """Validate the fixed v1 distillation contract."""
        if float(self.imitation_coeff) <= 0.0:
            msg = "ipmd_l2t.imitation_coeff must be positive."
            raise ValueError(msg)
        if (
            self.student_learning_rate is not None
            and float(self.student_learning_rate) <= 0.0
        ):
            msg = "ipmd_l2t.student_learning_rate must be positive."
            raise ValueError(msg)
        if (
            self.student_max_grad_norm is not None
            and float(self.student_max_grad_norm) <= 0.0
        ):
            msg = "ipmd_l2t.student_max_grad_norm must be positive."
            raise ValueError(msg)
        self.student_latent_key = normalize_batch_key(self.student_latent_key)


@dataclass
class IPMDL2TRLOptConfig(IPMDRLOptConfig):
    """RLOpt configuration for privileged-teacher IPMD distillation."""

    ipmd_l2t: IPMDL2TConfig = field(default_factory=IPMDL2TConfig)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.ipmd_l2t.validate()


class IPMDL2T(IPMD):
    """Train privileged IPMD and a deployable actor on the same rollouts.

    ``config.policy`` is the privileged teacher. It must consume the exact
    value-function observation keys. ``config.ipmd_l2t.student_policy`` is the
    deployable student and is optimized only by behavior distillation from the
    teacher action that actually controlled the environment.
    """

    config: IPMDL2TRLOptConfig
    student_policy: TensorDictModule

    def __init__(
        self,
        env,
        config: IPMDL2TRLOptConfig,
        policy_net: torch.nn.Module | None = None,
        value_net: torch.nn.Module | None = None,
        q_net: torch.nn.Module | None = None,
        replay_buffer: type[ReplayBuffer] = ReplayBuffer,
        logger: Logger | None = None,
        feature_extractor_net: torch.nn.Module | None = None,
        **kwargs,
    ) -> None:
        config.ipmd_l2t.validate()
        if config.value_function is None:
            msg = "IPMDL2T requires a value-function configuration."
            raise ValueError(msg)

        teacher_keys = [
            normalize_batch_key(key) for key in config.policy.get_input_keys()
        ]
        critic_keys = [
            normalize_batch_key(key) for key in config.value_function.get_input_keys()
        ]
        if teacher_keys != critic_keys:
            msg = (
                "IPMDL2T teacher policy inputs must exactly match value-function "
                f"inputs, got teacher={teacher_keys!r}, critic={critic_keys!r}."
            )
            raise ValueError(msg)

        architecture_fields = (
            "num_cells",
            "output_dim",
            "activation_fn",
            "normalize_input",
            "normalization_epsilon",
            "normalization_clip",
        )
        mismatched_architecture = [
            field_name
            for field_name in architecture_fields
            if getattr(config.policy, field_name)
            != getattr(config.ipmd_l2t.student_policy, field_name)
        ]
        if mismatched_architecture:
            msg = "IPMDL2T student must match the teacher actor architecture; "
            msg += f"mismatched fields: {mismatched_architecture!r}."
            raise ValueError(msg)

        self._student_obs_keys = [
            normalize_batch_key(key)
            for key in config.ipmd_l2t.student_policy.get_input_keys()
        ]
        self._student_latent_key = normalize_batch_key(
            config.ipmd_l2t.student_latent_key
        )
        self._student_imitation_coeff = float(config.ipmd_l2t.imitation_coeff)
        configured_student_clip = config.ipmd_l2t.student_max_grad_norm
        configured_default_clip = getattr(config.optim, "max_grad_norm", None)
        self._student_max_grad_norm = float(
            configured_student_clip or configured_default_clip or 1.0e10
        )

        available_keys = set(env.observation_spec.keys(True))
        missing_student_keys = [
            key for key in self._student_obs_keys if key not in available_keys
        ]
        if missing_student_keys:
            msg = (
                "IPMDL2T student observation keys are missing from the environment: "
                f"{missing_student_keys!r}."
            )
            raise ValueError(msg)
        if config.ipmd.use_latent_command and (
            self._student_latent_key not in self._student_obs_keys
        ):
            msg = (
                "IPMDL2T latent mode requires student_policy.input_keys to contain "
                f"{self._student_latent_key!r}."
            )
            raise ValueError(msg)

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

    @property
    def teacher_policy(self) -> TensorDictModule:
        """Return the privileged policy used for rollout collection."""
        assert self.policy is not None
        return self.policy

    @property
    def deployment_policy(self):
        """Return the deployable student, including latent-command injection."""
        if not self._use_latent_command:
            return self.student_policy
        controller = self._require_latent_command_controller()
        return controller.collector_policy(
            inject_fn=self._inject_latent_command,
            policy_module=self.student_policy,
        )

    def _set_optimizers(
        self, optimizer_cls: OptimizerClass, optimizer_kwargs: dict[str, Any]
    ) -> list[torch.optim.Optimizer]:
        """Add an independent, non-KL-adaptive student optimizer."""
        self.student_policy = self._construct_policy_from_config(
            self.config.ipmd_l2t.student_policy
        )
        with torch.no_grad():
            self.student_policy(self.env.fake_tensordict())

        teacher_optimizers = super()._set_optimizers(optimizer_cls, optimizer_kwargs)
        student_lr = self.config.ipmd_l2t.student_learning_rate
        if student_lr is None:
            student_lr = self.config.ipmd.actor_learning_rate
        if student_lr is None:
            student_lr = float(optimizer_kwargs["lr"])
        student_params = {
            "params": list(self.student_policy.parameters()),
            "lr": float(student_lr),
            "adaptive_lr": False,
            "name": "student",
        }
        student_optimizer = optimizer_cls([student_params], **optimizer_kwargs)
        return [*teacher_optimizers, student_optimizer]

    def _refresh_grad_clip_params(self) -> None:
        """Keep teacher and student gradient clipping fully independent."""
        self._student_grad_clip_params = list(self.student_policy.parameters())
        student_param_ids = {id(param) for param in self._student_grad_clip_params}
        self._grad_clip_params = [
            param
            for group in self.optim.param_groups
            for param in group["params"]
            if id(param) not in student_param_ids
        ]

    def _refresh_parameter_monitor(self) -> None:
        """Include the deployable actor in the normal finite-value checks."""
        super()._refresh_parameter_monitor()
        student_policy = getattr(self, "student_policy", None)
        if student_policy is None:
            return
        seen = {id(param) for _, param in self._parameter_monitor}
        for name, param in student_policy.named_parameters(recurse=True):
            if id(param) in seen or not torch.is_floating_point(param):
                continue
            seen.add(id(param))
            self._parameter_monitor.append((f"student_policy.{name}", param))

    def _inject_latent_command(self, td: TensorDict) -> None:
        """Generate one teacher command and mirror it exactly to the student."""
        super()._inject_latent_command(td)
        if not self._use_latent_command:
            return
        teacher_latent = td.get(self._latent_key)
        if not isinstance(teacher_latent, Tensor):
            msg = (
                f"IPMDL2T expected a Tensor at teacher latent key {self._latent_key!r}."
            )
            raise RuntimeError(msg)
        td.set(self._student_latent_key, teacher_latent)
        student_latent = td.get(self._student_latent_key)
        if not torch.equal(teacher_latent, student_latent):
            msg = (
                "IPMDL2T teacher and student latent commands must be tensor-identical."
            )
            raise RuntimeError(msg)

    def _backward_student_imitation(self, batch: TensorDict) -> dict[str, Tensor]:
        """Regress the student mean action to the detached executed action."""
        teacher_action = batch.get("action")
        if not isinstance(teacher_action, Tensor):
            msg = "IPMDL2T rollout batch requires Tensor key 'action'."
            raise RuntimeError(msg)
        target = teacher_action.detach()
        student_obs = batch.select(*self._student_obs_keys).detach()
        student_dist = self.student_policy.get_dist(student_obs)  # type: ignore[attr-defined]
        student_action = student_dist.deterministic_sample
        error = student_action - target
        imitation_mse = error.square().mean()
        imitation_loss = imitation_mse * self._student_imitation_coeff
        imitation_loss.backward()
        return {
            "loss_student_imitation": imitation_loss.detach(),
            "student_action_mse": imitation_mse.detach(),
            "student_action_mae": error.detach().abs().mean(),
            "student_action_rmse": error.detach().square().mean().sqrt(),
            "student_action_abs_mean": student_action.detach().abs().mean(),
            "teacher_action_abs_mean": target.abs().mean(),
        }

    def update(  # ty: ignore[invalid-method-override]
        self,
        batch: TensorDict,
        num_network_updates: int,
        expert_batch: TensorDict,
        has_expert: Tensor,
    ) -> tuple[TensorDict, int]:
        """Run unchanged IPMD teacher losses plus isolated student imitation."""
        update_idx = (
            int(num_network_updates.item())
            if isinstance(num_network_updates, Tensor)
            else int(num_network_updates)
        )
        self.optim.zero_grad(set_to_none=True)
        bc_pretrain_active = update_idx < self._bc_pretrain_updates
        output_loss = (
            self._ppo_metrics_without_backward(batch)
            if bc_pretrain_active
            else self._backward_ppo_terms(batch)
        )
        bc_metrics = self._backward_bc_terms(expert_batch, has_expert)
        rollout_bc_metrics = self._backward_rollout_bc_terms(batch)
        student_metrics = self._backward_student_imitation(batch)

        teacher_grad_norm = clip_grad_norm_(self._grad_clip_params, self._max_grad_norm)
        student_grad_norm = clip_grad_norm_(
            self._student_grad_clip_params, self._student_max_grad_norm
        )

        self.optim.step()
        hl_skill_metrics = self._run_hl_skill_online_update(
            batch,
            update_idx=update_idx,
        )

        output_loss.set("alpha", torch.ones((), device=self.device))
        for metrics in (bc_metrics, rollout_bc_metrics, student_metrics):
            for key, value in metrics.items():
                output_loss.set(key, value)
        output_loss.set(
            "bc_pretrain_active",
            torch.tensor(
                float(bc_pretrain_active),
                device=self.device,
                dtype=torch.float32,
            ),
        )
        for key, value in hl_skill_metrics.items():
            output_loss.set(key, value.detach())
        output_loss.set("grad_norm", teacher_grad_norm.detach())
        output_loss.set("student_grad_norm", student_grad_norm.detach())
        return output_loss, num_network_updates + 1

    @property
    def _optional_loss_metrics(self) -> list[str]:
        return [
            *super()._optional_loss_metrics,
            "loss_student_imitation",
            "student_action_mse",
            "student_action_mae",
            "student_action_rmse",
            "student_action_abs_mean",
            "teacher_action_abs_mean",
            "student_grad_norm",
        ]

    def _progress_summary_fields(self) -> tuple[tuple[str, str], ...]:
        return (
            *super()._progress_summary_fields(),
            ("train/loss_student_imitation", "student_loss"),
            ("train/student_action_rmse", "student_rmse"),
        )

    def _file_summary_fields(self) -> tuple[tuple[str, str], ...]:
        return (
            *super()._file_summary_fields(),
            ("train/loss_student_imitation", "student_loss"),
            ("train/student_action_rmse", "student_rmse"),
            ("train/student_grad_norm", "student_grad"),
        )

    def _checkpoint_policy_state_dict(self) -> Mapping[str, Any]:
        """Make the deployable student the checkpoint's primary policy."""
        return self.student_policy.state_dict()

    def _extra_checkpoint_state_dict(self) -> dict[str, Any]:
        """Retain the privileged teacher for exact training resume."""
        return {
            "teacher_policy_state_dict": self.teacher_policy.state_dict(),
            "checkpoint_metadata": {
                "algorithm": "IPMD_L2T",
                "primary_policy_role": "student",
            },
        }

    def _restore_training_state_from_checkpoint(
        self, checkpoint: Mapping[str, Any]
    ) -> bool:
        """Restore the full two-role training state for exact L2T resume."""
        del checkpoint
        return True

    def _load_checkpoint_policy_state_dict(self, checkpoint: Mapping[str, Any]) -> None:
        """Restore both roles and reject ambiguous ordinary-IPMD checkpoints."""
        if "teacher_policy_state_dict" not in checkpoint:
            msg = (
                "IPMDL2T resume requires teacher_policy_state_dict; ordinary IPMD "
                "checkpoints can only initialize a deployment IPMD agent."
            )
            raise ValueError(msg)
        if "policy_state_dict" not in checkpoint:
            msg = "IPMDL2T checkpoint is missing student policy_state_dict."
            raise ValueError(msg)
        self.student_policy.load_state_dict(checkpoint["policy_state_dict"])
        self.teacher_policy.load_state_dict(checkpoint["teacher_policy_state_dict"])


__all__ = ["IPMDL2T", "IPMDL2TConfig", "IPMDL2TRLOptConfig"]
