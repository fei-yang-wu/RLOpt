"""Synchronous, equal-shard PPO. The caller owns the process group.

Collector batches and minibatches are per worker; total_frames, resume offsets,
save intervals and logged steps are global. No simulator dependencies.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.distributed as dist

from rlopt.agent.ppo.ppo import PPO, RunningMeanStdCatInputs


@torch.no_grad()
def global_moments(value):
    """Stable pooled sample variance, including between-worker variation."""
    flat = value.detach().reshape(-1, value.shape[-1]).double()
    count = flat.new_tensor(float(flat.shape[0]))
    total = flat.sum(0)
    packed = torch.cat((total, count[None]))
    dist.all_reduce(packed)
    mean = packed[:-1] / packed[-1]
    squared = (flat - mean).square().sum(0)
    dist.all_reduce(squared)
    variance = squared / (packed[-1] - 1).clamp_min(1)
    return mean.to(value.dtype), variance.to(value.dtype), packed[-1].to(value.dtype)


def normalize_global(value):
    mean, variance, _ = global_moments(value)
    return (value - mean) / (variance.sqrt() + 1.0e-8)


class DistributedPPO(PPO):
    """Feed-forward PPO with explicit gradient averaging for TorchRL losses."""

    def __init__(self, env, config, *, max_training_seconds=None, **kwargs):
        if not dist.is_initialized():
            msg = "DistributedPPO requires an initialized process group"
            raise RuntimeError(msg)
        if config.compile.compile or config.compile.cudagraphs:
            msg = "Distributed PPO qualification requires compile/cudagraphs=false"
            raise ValueError(msg)
        if config.ppo.rnn_hidden_size:
            msg = "Distributed PPO currently supports feed-forward policies"
            raise ValueError(msg)
        if not config.ppo.update_normalizers_after_rollout and not config.ppo.freeze_normalizers:
            msg = "Distributed PPO requires update_normalizers_after_rollout=true"
            raise ValueError(msg)
        if config.collector.frames_per_batch % config.loss.mini_batch_size:
            msg = "Distributed PPO requires equal, complete minibatches"
            raise ValueError(msg)
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.max_training_seconds = max_training_seconds
        self.stop_requested = False
        self.stop_reason = "frame_budget"
        self._global_frames = 0
        self._training_started = 0.0
        if self.rank:
            config.logger.backend = ""
            config.logger.log_to_console = False
            config.trainer.progress_bar = False
        super().__init__(env, config, **kwargs)
        # TorchRL's local minibatch normalization would alter the global loss.
        self.loss_module.normalize_advantage = False
        for tensor in self.actor_critic.state_dict().values():
            dist.broadcast(tensor, src=0)
        self.collector.update_policy_weights_()

    def _synchronize_gradients(self):
        # PPO's actor and critic have a static graph. Fail uniformly if a new
        # recipe introduces unused parameters rather than silently desyncing.
        valid = torch.tensor(
            int(all(p.grad is not None for p in self._grad_clip_params)),
            device=self.device,
        )
        dist.all_reduce(valid, op=dist.ReduceOp.MIN)
        if not valid.item():
            msg = "Distributed PPO found an unused optimizer parameter"
            raise RuntimeError(msg)
        flat = torch.cat([p.grad.reshape(-1) for p in self._grad_clip_params])
        dist.all_reduce(flat)
        flat.div_(self.world_size)
        offset = 0
        for param in self._grad_clip_params:
            param.grad.copy_(flat[offset : offset + param.numel()].view_as(param))
            offset += param.numel()

    def _normalize_rollout_advantage(self, advantage):
        return normalize_global(advantage)

    def update(self, batch, num_network_updates, *args, **kwargs):
        if (
            self.config.ppo.normalize_advantage
            and not self.config.ppo.normalize_advantage_global
        ):
            batch = batch.clone(False)
            batch.set("advantage", normalize_global(batch.get("advantage")))
        return super().update(batch, num_network_updates, *args, **kwargs)

    def _record_kl_for_lr_adaptation(self, kl_approx, schedule_cfg):
        pooled = kl_approx.detach().mean().clone()
        dist.all_reduce(pooled)
        pooled.div_(self.world_size)
        super()._record_kl_for_lr_adaptation(pooled, schedule_cfg)

    def init_metadata(self):
        metadata = super().init_metadata()
        global_batch = self.config.collector.frames_per_batch * self.world_size
        remaining = max(
            0, self.config.collector.total_frames - metadata.frames_processed
        )
        metadata.total_iterations = (remaining + global_batch - 1) // global_batch
        self.total_network_updates = (
            self.config.collector.total_frames
            // global_batch
            * metadata.epochs_per_rollout
            * metadata.minibatches_per_epoch
        )
        initial_updates = int(getattr(self, "_resume_updates", 0))
        metadata.updates_completed.fill_(initial_updates)
        self._initial_updates = initial_updates
        self._updates_completed = initial_updates
        self._global_frames = metadata.frames_processed
        self._initial_frames = metadata.frames_processed
        return metadata

    def collect(self, metadata, iteration_idx):
        # Simulator allocators cannot reuse PyTorch's cached update workspaces.
        # Release unused blocks at the rollout boundary, never in env.step().
        if torch.device(self.device).type == "cuda":
            torch.cuda.empty_cache()
            if iteration_idx < 10 or iteration_idx % 100 == 0:
                free, total = torch.cuda.mem_get_info(self.device)
                print(
                    f"[DISTRIBUTED MEMORY] rank={self.rank} iteration={iteration_idx} "
                    f"allocated={torch.cuda.memory_allocated(self.device)} "
                    f"reserved={torch.cuda.memory_reserved(self.device)} "
                    f"free={free} total={total}",
                    flush=True,
                )
        iteration = super().collect(metadata, iteration_idx)
        extra = iteration.frames * (self.world_size - 1)
        metadata.frames_processed += extra
        if metadata.progress_bar_enabled:
            metadata.progress_bar.update(extra)
        iteration.frames *= self.world_size
        return iteration

    @torch.no_grad()
    def _update_rollout_normalizers(self, rollout):
        if self.config.ppo.freeze_normalizers:
            return
        for network, cfg in (
            (self.policy, self.config.policy),
            (self.value_function, self.config.value_function),
        ):
            norms = [
                m for m in network.modules() if isinstance(m, RunningMeanStdCatInputs)
            ]
            if norms:
                values = [rollout.get(key) for key in cfg.get_input_keys()]
                value = values[0] if len(values) == 1 else torch.cat(values, dim=-1)
                moments = global_moments(value)
                for norm in norms:
                    norm._update(value, moments=moments)

    def record(self, iteration, metadata):
        # Commit only complete iterations. Shared checkpoint writes happen on
        # rank zero; rank-specific diagnostic logs have their own directories.
        self._global_frames = metadata.frames_processed
        self._updates_completed = int(metadata.updates_completed)
        # PPO optimization metrics are equal-shard means. Environment/episode
        # summaries below are explicitly rank-local (no misleading pooled SR).
        keys = sorted(iteration.metrics)
        if keys:
            values = torch.tensor(
                [iteration.metrics[k] for k in keys], device=self.device
            )
            dist.all_reduce(values)
            values /= self.world_size
            iteration.metrics.update(zip(keys, values.cpu().tolist(), strict=True))
        super().record(iteration, metadata)

    def _record_env_metrics(self, iteration):
        before = set(iteration.metrics)
        super()._record_env_metrics(iteration)
        for key in set(iteration.metrics) - before:
            iteration.metrics[f"rank0_env/{key}"] = iteration.metrics.pop(key)

    def log_metrics(self, metrics, **kwargs):
        if self.rank == 0:
            local_keys = {
                "episode/length",
                "episode/return",
                "train/reward",
                "train/step_reward_mean",
                "train/step_reward_std",
                "train/step_reward_min",
                "train/step_reward_max",
            }
            metrics = {
                f"rank0/{k}" if k in local_keys else k: v for k, v in metrics.items()
            }
            super().log_metrics(metrics, **kwargs)

    def save_model(self, path=None, step=None):
        if self.rank == 0:
            root = Path(path) if path is not None else self.log_dir
            if root.suffix and not root.is_dir():
                target = (
                    root
                    if step is None
                    else root.with_name(f"{root.stem}_step_{step}{root.suffix}")
                )
            else:
                target = root / (
                    f"model_step_{step}.pt" if step is not None else "model.pt"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".pending.pt")
            super().save_model(temporary)
            temporary.replace(target)

    def _extra_model_state_dict(self):
        return {
            "cumulative_env_frames": self._global_frames,
            "distributed_world_size": self.world_size,
            "optimizer_updates": getattr(self, "_updates_completed", 0),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict()
            if self.lr_scheduler
            else None,
        }

    def load_model(self, path):
        super().load_model(path)
        self.collector.update_policy_weights_()

    def _load_extra_model_state_dict(self, checkpoint):
        self._resume_frame_offset = int(checkpoint.get("cumulative_env_frames", 0))
        self._resume_updates = int(checkpoint.get("optimizer_updates", 0))
        state = checkpoint.get("lr_scheduler_state_dict")
        if self.lr_scheduler and state is not None:
            self.lr_scheduler.load_state_dict(state)

    def _should_stop_training(self, _metadata):
        expired = (
            self.max_training_seconds is not None
            and time.monotonic() - self._training_started >= self.max_training_seconds
        )
        stop = torch.tensor(
            [int(self.stop_requested), int(expired)], device=self.device
        )
        dist.all_reduce(stop, op=dist.ReduceOp.MAX)
        signaled, expired = stop.tolist()
        if signaled or expired:
            self.stop_reason = "signal" if signaled else "time_budget"
            return True
        return False

    @torch.no_grad()
    def _write_health(self):
        difference = torch.zeros((), device=self.device)
        for tensor in self.actor_critic.state_dict().values():
            reference = tensor.clone()
            dist.broadcast(reference, src=0)
            difference = torch.maximum(
                difference, (tensor.double() - reference.double()).abs().max().float()
            )
        dist.all_reduce(difference, op=dist.ReduceOp.MAX)
        if not torch.isfinite(difference) or difference.item() != 0:
            msg = f"Distributed model state diverged: {difference.item()}"
            raise RuntimeError(msg)
        health = {
            "rank": self.rank,
            "world_size": self.world_size,
            "device": str(self.device),
            "initial_frames": self._initial_frames,
            "cumulative_env_frames": self._global_frames,
            "max_model_difference": difference.item(),
            "stop_reason": self.stop_reason,
            "optimizer_updates_this_segment": getattr(self, "_updates_completed", 0)
            - self._initial_updates,
            "elapsed_training_seconds": time.monotonic() - self._training_started,
            "learning_rates": [group["lr"] for group in self.optim.param_groups],
        }
        Path(self.log_dir, "distributed_health.json").write_text(
            json.dumps(health, indent=2) + "\n"
        )
        self.log.info("Distributed health: %s", json.dumps(health))

    def train(self):
        self._training_started = time.monotonic()
        super().train()
        self._write_health()
        self.save_model(
            self.log_dir / self.config.logger.save_path, self._global_frames
        )
        dist.barrier()
