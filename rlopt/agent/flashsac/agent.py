"""FlashSAC agent shell: collect-train loop on a time-major GPU replay ring.

One host thread runs the loop (architecture C of the plan). Per env step:

``collect``  one vector step of the TorchRL ``Collector`` (``frames_per_batch``
             equals ``num_envs``). The policy is a ``TensorDictModule`` around
             ``FlashSACPolicyNet``. During warm-up it writes random actions.
``prepare``  inserts the step into ``TimeMajorReplay`` and shows the raw rewards
             to the learner's return normaliser.
``iterate``  runs exactly ``floor((t + 1) * U) - floor(t * U)`` learner updates,
             ``U = rho * num_envs / batch_size`` (integer arithmetic).
``record``   every ``log_interval_steps`` env steps: ONE device-to-host copy of a
             stacked metrics tensor, then logging and checkpoints.

Host synchronisation. The hot loop reads no device value. Counters that drive
control flow (frames, rows, update credit) are Python ints. The only device to
host copies are the metric copy at each log and the checkpoint write.

Checkpoint. One file ``model_step_<cumulative_env_frames>.pt`` holds:

``policy_state_dict``   state of ``self.policy`` (the actor, its BatchNorm running
                        buffers and ``action_bound``). Strict-loadable by
                        ``load_frozen_low_level_tracker``.
``policy_meta``         architecture tag, ordered input keys, squash, bound, dims.
``learner_state_dict``  networks, optimizers, GradScaler, LR steps, reward statistics.
``agent_state``         update counter, noise-repeat state, host counters.
``cumulative_env_frames`` global frame count for walltime-segmented resume.

There is no ``vec_norm_msg`` (the actor normalises its input with BatchNorm).
The replay ring is not saved: a resume refills it with the loaded policy.

Open item. The deployed policy is ``bound * tanh(mean)`` with BatchNorm running
statistics. The PPO-style ``policy_state_dict`` of the dev evaluators is loaded
strictly and works unchanged; the exporter must dispatch on
``policy_meta["squash"]`` (see wiki/flashsac-ipmd-plan.md section 3.10).
"""

from __future__ import annotations

import copy
import dataclasses
import math
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Generic, TypeVar

import numpy as np
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch import Tensor, nn
from torchrl.data import Bounded
from torchrl.envs import TransformedEnv
from torchrl.envs.utils import ExplorationType, exploration_type
from torchrl.record.loggers.common import Logger

from rlopt.agent.flashsac.config import FlashSACRLOptConfig
from rlopt.agent.flashsac.learner import FlashSACLearner
from rlopt.agent.flashsac.noise import NoiseRepeat
from rlopt.agent.flashsac.replay import TimeMajorReplay
from rlopt.base_class import BaseAlgorithm, IterationData, TrainingMetadata
from rlopt.config_utils import ObsKey

__all__ = [
    "ARCH_TAG",
    "FlashSAC",
    "FlashSACPolicyNet",
    "update_rate",
    "updates_due",
]

ARCH_TAG = "flashsac_actor_v1"

_DETERMINISTIC_MODES = (
    ExplorationType.DETERMINISTIC,
    ExplorationType.MODE,
    ExplorationType.MEAN,
    ExplorationType.MEDIAN,
)
_OBS_DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}
# Learner info keys that are only fresh on an actor update (CUDA-graph replays
# return the stale value of the last actor update for these).
_ACTOR_ONLY_PREFIXES = ("actor/", "temperature/")
_ENV_ACC_NAMES = (
    "reward_sum",
    "done_count",
    "episode_return_sum",
    "episode_length_sum",
    "terminated_count",
    "truncated_count",
    "final_obs_missing",
)

CfgT = TypeVar("CfgT", bound=FlashSACRLOptConfig)


# --------------------------------------------------------------------- schedule
def update_rate(rho: float, num_envs: int, batch_size: int) -> Fraction:
    """Return ``U = rho * num_envs / batch_size`` as an exact fraction."""
    return Fraction(rho).limit_denominator(1_000_000) * num_envs / batch_size


def updates_due(step: int, rate: Fraction) -> int:
    """Updates owed at env step ``step`` (0-based): ``floor((t+1)U) - floor(tU)``."""
    return ((step + 1) * rate.numerator) // rate.denominator - (
        step * rate.numerator
    ) // rate.denominator


def _key_tuple(key: ObsKey) -> tuple[str, ...]:
    return (key,) if isinstance(key, str) else tuple(key)


# ----------------------------------------------------------------------- policy
class FlashSACPolicyNet(nn.Module):
    """Actor wrapper that returns ``(bound * tanh(raw), tanh(raw))``.

    The actor and the ``action_bound`` buffer are the learner's own tensors, so
    the module always acts with the current weights and BatchNorm statistics.
    ``state_dict`` keys are ``actor.*`` and ``action_bound``.

    Exploration type RANDOM draws ``raw = mean + std * noise``; with
    ``collecting`` the noise is the per-env repeated noise, otherwise it is
    fresh N(0, I). DETERMINISTIC/MODE/MEAN returns ``bound * tanh(mean)``.
    With ``random_actions`` (warm-up while ``collecting``) the actor is skipped.
    """

    def __init__(
        self,
        actor: nn.Module,
        action_bound: Tensor,
        noise: NoiseRepeat | None,
        warmup_mode: str = "uniform",
    ) -> None:
        super().__init__()
        self.actor = actor
        self.register_buffer("action_bound", action_bound)
        self.noise = noise
        self.warmup_mode = warmup_mode
        self.collecting = False
        self.random_actions = False
        self.reset_mask: Tensor | None = None

    def forward(self, *inputs: Tensor) -> tuple[Tensor, Tensor]:
        obs = inputs[0] if len(inputs) == 1 else torch.cat(inputs, dim=-1)
        lead = obs.shape[:-1]
        flat = obs.reshape(-1, obs.shape[-1])
        mode = exploration_type()
        deterministic = mode in _DETERMINISTIC_MODES
        with torch.no_grad():
            if self.collecting and self.random_actions and not deterministic:
                act_dim = self.action_bound.shape[-1]
                if self.warmup_mode == "uniform":
                    tanh_a = (
                        torch.rand(
                            flat.shape[0],
                            act_dim,
                            device=flat.device,
                            dtype=torch.float32,
                        )
                        .mul_(2.0)
                        .sub_(1.0)
                    )
                    action = self.action_bound * tanh_a
                else:
                    action = torch.randn(
                        flat.shape[0], act_dim, device=flat.device, dtype=torch.float32
                    )
                    action = torch.maximum(
                        torch.minimum(action, self.action_bound), -self.action_bound
                    )
                    tanh_a = action / self.action_bound
            else:
                mean, std = self.actor.mean_and_std(flat, training=False)
                if deterministic:
                    raw = mean
                else:
                    if self.collecting and self.noise is not None:
                        noise = self.noise(self.reset_mask)
                    else:
                        noise = torch.randn_like(mean)
                    raw = mean + std * noise
                tanh_a = torch.tanh(raw)
                action = self.action_bound * tanh_a
        return action.reshape(*lead, -1), tanh_a.reshape(*lead, -1)


# --------------------------------------------------------------------- metadata
@dataclass(kw_only=True)
class FlashSACTrainingMetadata(TrainingMetadata):
    """Run-level state of the FlashSAC loop."""

    collector_iter: Iterator[TensorDict]
    """Iterator over single-step collector batches."""

    rate: Fraction
    """Updates per env step, ``U``, as an exact fraction."""

    resume_offset: int = 0
    """Global frame count at the start of this segment."""

    interval_start_time: float = 0.0
    """Wall time of the last metric log."""

    interval_start_frames: int = 0
    """Frame count at the last metric log."""

    interval_start_updates: int = 0
    """Updates run at the last metric log."""

    last_checkpoint_time: float = 0.0
    """Wall time of the last checkpoint."""


@dataclass(kw_only=True)
class FlashSACIterationData(IterationData):
    """One collected vector step."""

    rollout: TensorDict
    """Collector output for this step, shape ``[num_envs]`` after reshape."""

    updates: int = 0
    """Learner updates run in this iteration."""


# ------------------------------------------------------------------------ agent
class FlashSAC(BaseAlgorithm[CfgT], Generic[CfgT]):
    """FlashSAC off-policy agent (see the module docstring)."""

    def __init__(
        self,
        env: TransformedEnv,
        config: CfgT,
        logger: Logger | None = None,
        **kwargs: Any,
    ) -> None:
        self._num_envs = int(env.batch_size.numel()) if len(env.batch_size) else 1
        config.flashsac.validate(self._num_envs)
        self._resume_frame_offset = 0
        self._loaded_policy = False
        self._frames_processed = 0
        self._collected_run = 0
        self._train_steps = 0
        self._updates_run = 0
        self._gate_open = False
        self._graph_ready = False
        self._acc: dict[str, Tensor] = {}
        self._acc_n: dict[str, int] = {}
        self._env_acc: Tensor | None = None
        self._terminal_mode = ""  # resolved at the first prepare()
        super().__init__(env=env, config=config, logger=logger, **kwargs)
        fs = self.config.flashsac
        self._rate = (
            Fraction(fs.updates_per_step).limit_denominator(1_000_000)
            if fs.updates_per_step is not None
            else update_rate(fs.rho, self._num_envs, fs.batch_size)
        )

    # ------------------------------------------------------------ construction
    def _action_spec(self) -> Any:
        return self.env.action_spec_unbatched  # type: ignore[attr-defined]

    def _resolve_action_bound(self) -> Tensor:
        """Return the symmetric action bound ``(act_dim,)`` or raise."""
        cfg = self.config.flashsac
        act_dim = self.action_feature_size()
        if cfg.action_bound is not None:
            bound = torch.as_tensor(cfg.action_bound, dtype=torch.float32)
            if bound.ndim == 0:
                bound = bound.expand(act_dim).clone()
            if tuple(bound.shape) != (act_dim,):
                msg = (
                    f"flashsac.action_bound has shape {tuple(bound.shape)}; "
                    f"the action has {act_dim} dimensions. Give a scalar or "
                    f"{act_dim} values."
                )
                raise ValueError(msg)
            return bound
        spec = self._action_spec()
        if isinstance(spec, Bounded):
            low = spec.space.low.detach().float().reshape(-1)
            high = spec.space.high.detach().float().reshape(-1)
            if (
                torch.isfinite(low).all()
                and torch.isfinite(high).all()
                and torch.allclose(low, -high)
                and bool((high > 0).all())
            ):
                return high.clone()
            msg = (
                "The env action spec is bounded but not symmetric about zero; "
                "set flashsac.action_bound explicitly."
            )
            raise ValueError(msg)
        msg = (
            f"The env action spec is {type(spec).__name__} (unbounded) and "
            "flashsac.action_bound is None. A tanh policy needs an explicit "
            "bound: set agent.flashsac.action_bound to a scalar or one value "
            "per action dimension."
        )
        raise ValueError(msg)

    def _obs_keys(self) -> tuple[list[ObsKey], list[ObsKey]]:
        actor_keys = list(self.config.policy.get_input_keys())
        critic_keys = self.config.flashsac.critic_input_keys
        critic = list(critic_keys) if critic_keys else list(actor_keys)
        return actor_keys, critic

    def _missing_obs_keys(self) -> list[ObsKey]:
        spec_keys = {_key_tuple(k) for k in self.env.observation_spec.keys(True, True)}
        actor_keys, critic_keys = self._obs_keys()
        missing = []
        for key in [*actor_keys, *critic_keys]:
            if _key_tuple(key) not in spec_keys and key not in missing:
                missing.append(key)
        return missing

    def _construct_policy(self) -> TensorDictModule:
        cfg = self.config.flashsac
        missing = self._missing_obs_keys()
        if missing:
            msg = (
                f"Input keys {missing} are not in the env observation spec. "
                "Fix policy.input_keys / flashsac.critic_input_keys."
            )
            raise ValueError(msg)
        bound = self._resolve_action_bound()
        actor_keys, critic_keys = self._obs_keys()
        actor_dim = sum(self.observation_feature_size(k) for k in actor_keys)
        critic_dim = sum(self.observation_feature_size(k) for k in critic_keys)
        act_dim = self.action_feature_size()
        learner_cfg = dataclasses.replace(
            cfg.learner,
            batch_size=cfg.batch_size,
            gamma=self.config.loss.gamma,
            n_step=cfg.replay.n_step,
            seed=self.config.seed,
            action_bound=[float(x) for x in bound.tolist()],
        )
        cfg.learner = learner_cfg  # record the values that ran
        self.learner = FlashSACLearner(
            learner_cfg, actor_dim, critic_dim, act_dim, self.device
        )
        self._dims = {"actor": actor_dim, "critic": critic_dim, "act": act_dim}
        noise = (
            NoiseRepeat(
                self._num_envs,
                act_dim,
                self.device,
                zeta_s=cfg.noise_zeta_s,
                max_repeat=cfg.noise_max_repeat,
            )
            if cfg.noise_repeat
            else None
        )
        self.policy_net = FlashSACPolicyNet(
            self.learner.actor, self.learner.action_bound, noise, cfg.warmup_mode
        )
        self._actor_keys = actor_keys
        self._critic_keys = critic_keys
        return TensorDictModule(
            self.policy_net,
            in_keys=list(actor_keys),
            out_keys=["action", "action_tanh"],
        )

    def _construct_feature_extractor(self) -> TensorDictModule:
        msg = "FlashSAC has no feature extractor."
        raise NotImplementedError(msg)

    def _construct_actor_critic(self) -> TensorDictModule:
        assert self.policy is not None
        return self.policy

    @property
    def collector_policy(self) -> TensorDictModule:
        """The policy module the collector and the evaluators call."""
        assert self.policy is not None
        return self.policy

    def _construct_loss_module(self) -> nn.Module:
        return self.learner

    def _configure_optimizers(self):  # type: ignore[override]
        """The learner owns its optimizers; ``self.optim`` stays ``None``."""
        self.lr_scheduler = None

    def _compile_components(self) -> None:
        # Compile and CUDA graphs are owned by the learner config.
        self.compile_mode = None

    def _resolve_terminal_mode(self, has: set[tuple[str, ...]]) -> str:
        """Pick the terminal-observation source from the first collector output.

        The ``final`` group is an info-reader output, so it is not always in the
        env observation spec; the collected tensordict is the reliable witness.
        """
        rep = self.config.flashsac.replay
        has_final = any(
            len(k) > 2 and k[0] == "next" and k[1] == rep.final_group for k in has
        )
        if rep.terminal_obs == "final" and not has_final:
            msg = (
                f"replay.terminal_obs='final' but the collected step has no "
                f"('next', '{rep.final_group}', ...) group. Attach the terminal "
                "observation reader or use terminal_obs='next'."
            )
            raise ValueError(msg)
        if rep.terminal_obs != "auto":
            return rep.terminal_obs
        return "final" if has_final else "next"

    def _construct_data_buffer(self) -> TimeMajorReplay:  # type: ignore[override]
        cfg = self.config.flashsac
        rep = cfg.replay
        dtype = _OBS_DTYPES[rep.obs_dtype]
        dims = self._dims
        slots = rep.capacity_steps * self._num_envs
        est = int(
            slots
            * TimeMajorReplay.bytes_per_transition(
                dims["actor"],
                dims["critic"],
                dims["act"],
                obs_bytes=torch.empty((), dtype=dtype).element_size(),
            )
            * (1.0 + rep.pool_fraction)
        )
        if self.device.type == "cuda":
            free, _ = torch.cuda.mem_get_info(self.device)
            if est > 0.95 * free:
                msg = (
                    f"The replay ring needs about {est / 2**30:.1f} GiB but "
                    f"{free / 2**30:.1f} GiB are free on {self.device}. Lower "
                    "flashsac.replay.capacity_steps, use obs_dtype=bfloat16, or "
                    "free GPU memory."
                )
                raise MemoryError(msg)
        return TimeMajorReplay(
            num_envs=self._num_envs,
            capacity_steps=rep.capacity_steps,
            actor_obs_dim=dims["actor"],
            critic_obs_dim=dims["critic"],
            act_dim=dims["act"],
            n_step=rep.n_step,
            gamma=self.config.loss.gamma,
            device=self.device,
            obs_dtype=dtype,
            pool_fraction=rep.pool_fraction,
        )

    # ------------------------------------------------------------- validation
    def validate_training(self) -> None:
        """Reject settings the loop cannot run. Each message names the fix."""
        cfg = self.config
        fs = cfg.flashsac
        errors: list[str] = []
        try:
            fs.validate(self._num_envs)
        except ValueError as exc:
            errors.append(str(exc))
        if int(cfg.collector.frames_per_batch) != self._num_envs:
            errors.append(
                f"collector.frames_per_batch={cfg.collector.frames_per_batch} but "
                f"the env has {self._num_envs} envs: FlashSAC collects one vector "
                "step per call. Set collector.frames_per_batch to the number of "
                "envs (the entry point multiplies the per-env value by num_envs)."
            )
        if int(cfg.collector.init_random_frames) != 0:
            errors.append(
                "collector.init_random_frames must be 0; use "
                "flashsac.warmup_transitions."
            )
        if cfg.compile.compile or cfg.compile.cudagraphs:
            errors.append(
                "compile.compile / compile.cudagraphs are not used by FlashSAC; set "
                "flashsac.learner.compile_mode and flashsac.learner.use_cuda_graph."
            )
        if fs.learner.compile_mode != "none" and os.environ.get("TORCHDYNAMO_DISABLE"):
            errors.append(
                "flashsac.learner.compile_mode='default' but TORCHDYNAMO_DISABLE is "
                "set (the smoke tasks set it). Unset it or use compile_mode='none'."
            )
        if fs.learner.use_cuda_graph and self.device.type != "cuda":
            errors.append("flashsac.learner.use_cuda_graph needs a CUDA device.")
        missing = self._missing_obs_keys()
        if missing:
            errors.append(f"input keys {missing} are not in the env observation spec.")
        try:
            self._resolve_action_bound()
        except ValueError as exc:
            errors.append(str(exc))
        total = int(cfg.collector.total_frames)
        if total <= fs.warmup_transitions:
            errors.append(
                f"collector.total_frames={total} does not exceed "
                f"flashsac.warmup_transitions={fs.warmup_transitions}: no update "
                "would run. Raise total_frames or lower the warm-up."
            )
        if errors:
            raise ValueError("FlashSAC cannot start:\n  - " + "\n  - ".join(errors))

    # ------------------------------------------------------------- run set-up
    def init_metadata(self) -> FlashSACTrainingMetadata:
        cfg = self.config
        resume = int(getattr(self, "_resume_frame_offset", 0))
        if resume <= 0:
            resume = int(getattr(cfg.collector, "initial_frame_offset", 0) or 0)
        resume = max(0, resume)
        self._resume_frame_offset = resume
        frames_per_batch = int(cfg.collector.frames_per_batch)
        remaining = max(0, int(cfg.collector.total_frames) - resume)
        iterations = min(len(self.collector), -(-remaining // frames_per_batch))
        if resume:
            self.log.info(
                "Resuming the frame budget at %d/%d (%d iterations remain).",
                resume,
                int(cfg.collector.total_frames),
                iterations,
            )
        self._frames_processed = resume
        self._collected_run = 0
        self._train_steps = 0
        self._gate_open = False
        self._env_acc = torch.zeros(
            len(_ENV_ACC_NAMES), dtype=torch.float32, device=self.device
        )
        self._acc, self._acc_n = {}, {}
        rate = self._rate
        log_every = int(cfg.flashsac.log_interval_steps) * frames_per_batch
        now = time.monotonic()
        self.log.info(
            "FlashSAC: N=%d envs, B=%d, rho=%s, U=%s updates/env step, "
            "warmup=%d transitions.",
            self._num_envs,
            cfg.flashsac.batch_size,
            cfg.flashsac.rho,
            f"{float(rate):.4f}",
            cfg.flashsac.warmup_transitions,
        )
        return FlashSACTrainingMetadata(
            total_iterations=iterations,
            frames_processed=resume,
            progress_bar=None,
            progress_bar_enabled=False,
            log_interval_frames=log_every,
            next_log_frame=resume + log_every,
            next_file_log_frame=resume + log_every,
            collector_iter=iter(self.collector),
            rate=rate,
            resume_offset=resume,
            interval_start_time=now,
            interval_start_frames=resume,
            interval_start_updates=self.learner.update_count,
            last_checkpoint_time=now,
        )

    # ------------------------------------------------------------------ phases
    def _random_phase(self) -> bool:
        """True while warm-up actions are random (not on a resume)."""
        if self._resume_frame_offset > 0 or self._loaded_policy:
            return False
        return self._collected_run < self.config.flashsac.warmup_transitions

    def collect(
        self, run: FlashSACTrainingMetadata, iteration_idx: int
    ) -> FlashSACIterationData:
        """Collect ONE vector step."""
        net = self.policy_net
        net.collecting = True
        net.random_actions = self._random_phase()
        phase_times: dict[str, float] = {}
        start = time.perf_counter()
        try:
            with self._profile_iteration_phase(phase_times, "collect"):
                rollout = next(run.collector_iter)
        finally:
            net.collecting = False
            net.random_actions = False
        collect_time = phase_times.get("collect", time.perf_counter() - start)
        rollout = rollout.reshape(-1)
        if rollout.numel() != self._num_envs:
            msg = (
                f"The collector returned {rollout.numel()} transitions; expected "
                f"{self._num_envs}."
            )
            raise RuntimeError(msg)
        run.frames_processed += self._num_envs
        self._frames_processed = run.frames_processed
        return FlashSACIterationData(
            iteration_idx=iteration_idx,
            frames=self._num_envs,
            rollout=rollout,
            collect_time=collect_time,
            phase_times=phase_times,
        )

    def _flat(
        self, td: TensorDict, prefix: tuple[str, ...], keys: list[ObsKey]
    ) -> Tensor:
        parts = [td.get(prefix + _key_tuple(k)) for k in keys]
        flat = [p.reshape(self._num_envs, -1) for p in parts]
        return flat[0] if len(flat) == 1 else torch.cat(flat, dim=-1)

    def _present_keys(self, td: TensorDict) -> set[tuple[str, ...]]:
        """Leaf keys of the collector output, listed once (the layout is fixed)."""
        cached = getattr(self, "_leaf_keys", None)
        if cached is None:
            cached = {_key_tuple(k) for k in td.keys(True, True)}
            self._leaf_keys = cached
        return cached

    @torch.no_grad()
    def prepare(
        self,
        iteration: FlashSACIterationData,
        _metadata: FlashSACTrainingMetadata,
    ) -> None:
        """Insert the step into the replay and update the reward statistics."""
        rep = self.config.flashsac.replay
        r = iteration.rollout
        n = self._num_envs
        reward = r.get(("next", "reward")).reshape(n)
        done = r.get(("next", "done")).reshape(n).to(torch.bool)
        has = self._present_keys(r)
        if not self._terminal_mode:
            self._terminal_mode = self._resolve_terminal_mode(has)
            self.log.info("Terminal observation source: %s.", self._terminal_mode)
        if ("next", "truncated") in has:
            truncated = r.get(("next", "truncated")).reshape(n).to(torch.bool)
        else:
            truncated = torch.zeros_like(done)
        if ("next", "terminated") in has:
            terminated = r.get(("next", "terminated")).reshape(n).to(torch.bool)
        else:
            terminated = done & ~truncated
        next_obs = self._flat(r, ("next",), self._actor_keys)
        next_critic = self._flat(r, ("next",), self._critic_keys)
        env_acc = self._env_acc
        assert env_acc is not None
        final_missing = torch.zeros((), device=self.device)
        if self._terminal_mode == "final":
            group = (rep.final_group,)
            fin = self._flat(r, ("next", *group), self._actor_keys)
            fin_c = self._flat(r, ("next", *group), self._critic_keys)
            final_missing = (done & (fin.abs().sum(-1) == 0)).sum().float()
            sel = done.unsqueeze(-1)
            next_obs = torch.where(sel, fin, next_obs)
            next_critic = torch.where(sel, fin_c, next_critic)
        step = {
            "obs": self._flat(r, (), self._actor_keys),
            "critic_obs": self._flat(r, (), self._critic_keys),
            "action": r.get("action_tanh").reshape(n, -1),
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "next_obs": next_obs,
            "next_critic_obs": next_critic,
        }
        self.data_buffer.add(step)
        self.learner.observe_rewards(reward, terminated, truncated)
        self.policy_net.reset_mask = done

        done_f = done.float()
        ep_ret = (
            r.get(("next", "episode_reward")).reshape(n).float()
            if ("next", "episode_reward") in has
            else torch.zeros_like(reward)
        )
        ep_len = (
            r.get(("next", "step_count")).reshape(n).float()
            if ("next", "step_count") in has
            else torch.zeros_like(reward)
        )
        env_acc += torch.stack(
            [
                reward.float().mean(),
                done_f.sum(),
                (ep_ret * done_f).sum(),
                (ep_len * done_f).sum(),
                terminated.float().sum(),
                truncated.float().sum(),
                final_missing,
            ]
        )
        self._collected_run += n
        if (
            not self._gate_open
            and self._collected_run >= self.config.flashsac.warmup_transitions
            and self.data_buffer.valid_rows >= 1
        ):
            self._gate_open = True
            self.log.info(
                "Warm-up done after %d transitions; updates start (replay rows=%d).",
                self._collected_run,
                self.data_buffer.filled_rows,
            )

    def _accumulate(self, info: dict[str, Tensor]) -> None:
        keys = sorted(info)
        vals = []
        for k in keys:
            v = info[k].detach()
            v = v if v.dtype == torch.float32 else v.float()
            if k not in self._acc:
                self._acc[k] = torch.zeros_like(v)
                self._acc_n[k] = 0
            vals.append(v)
        torch._foreach_add_([self._acc[k] for k in keys], vals)
        for k in keys:
            self._acc_n[k] += 1

    def _sample_batch(self, batch_size: int) -> dict[str, Tensor]:
        batch = self.data_buffer.sample(batch_size)
        batch["reward"] = batch["reward_n"]
        return batch

    def _one_update(self) -> None:
        learner = self.learner
        batch = self._sample_batch(self.config.flashsac.batch_size)
        if learner.cfg.use_cuda_graph:
            if not self._graph_ready:
                learner.capture_graphs(batch)
                self._graph_ready = True
            do_actor = learner._do_actor()
            info = learner.step_graph(batch)
            if not do_actor:
                info = {
                    k: v
                    for k, v in info.items()
                    if not k.startswith(_ACTOR_ONLY_PREFIXES)
                }
        else:
            info = learner.update(batch)
        self._accumulate(info)
        self._updates_run += 1

    def iterate(
        self,
        iteration: FlashSACIterationData,
        metadata: FlashSACTrainingMetadata,
    ) -> None:
        """Run exactly ``floor((t+1)U) - floor(tU)`` updates; no host sync."""
        if not self._gate_open:
            return
        due = updates_due(self._train_steps, metadata.rate)
        for _ in range(due):
            self._one_update()
        self._train_steps += 1
        iteration.updates = due

    # ----------------------------------------------------------------- logging
    def _progress_summary_fields(self) -> tuple[tuple[str, str], ...]:
        return (
            ("episode/return", "r_ep"),
            ("episode/length", "ep_len"),
            ("train/critic_loss", "q_loss"),
            ("train/actor_loss", "pi_loss"),
            ("time/speed", "fps"),
        )

    @torch.no_grad()
    def _health_tensor(self) -> tuple[list[str], Tensor]:
        """Device tensor of HC metrics from one sampled batch. No host copy."""
        learner = self.learner
        replay = self.data_buffer
        names: list[str] = []
        vals: list[Tensor] = []
        if replay.valid_rows >= 1:
            b = replay.sample(min(self.config.flashsac.batch_size, 1024))
            rows = replay.capacity_steps
            newest = (replay._total - 1) % rows
            age = ((newest - b["row"]) % rows).float().mean()
            names.append("health/replay_age_rows")
            vals.append(age)
            names.append("health/action_saturation")
            vals.append((b["action"].abs() > 0.99).float().mean())
            q, logp = learner.critic(b["critic_obs"], b["action"], False)
            probs = logp.float().exp()
            names.append("health/c51_edge_mass")
            vals.append((probs[..., 0] + probs[..., -1]).mean())
            scale = learner.return_scale.scale()
            names.append("health/q_eval_mean")
            vals.append(q.float().mean())
            names.append("health/reward_n_normalised_mean")
            vals.append((b["reward_n"] / scale).mean())
            actor_copy = copy.deepcopy(learner.actor)
            mean_train, _ = actor_copy.mean_and_std(b["obs"], training=True)
            mean_eval, _ = learner.actor.mean_and_std(b["obs"], training=False)
            names.append("health/bn_gap")
            vals.append((mean_train - mean_eval).abs().mean())
        for label, module in (("actor", learner.actor), ("critic", learner.critic)):
            for suffix in ("running_mean", "running_var"):
                bufs = [
                    buf.float()
                    for name, buf in module.named_buffers()
                    if name.endswith(suffix)
                ]
                names.append(f"health/{label}_bn_{suffix}_norm")
                vals.append(
                    torch.sqrt(sum(x.square().sum() for x in bufs))
                    if bufs
                    else torch.zeros((), device=self.device)
                )
        return names, torch.stack([v.reshape(()).float() for v in vals])

    def _flush_metrics(self, run: FlashSACTrainingMetadata) -> dict[str, float]:
        """ONE device-to-host copy of every accumulated metric."""
        assert self._env_acc is not None
        keys = sorted(self._acc)
        parts = [self._env_acc]
        parts.extend(self._acc[k].reshape(-1).mean().reshape(1) for k in keys)
        health_names: list[str] = []
        if self.config.flashsac.health_checks:
            health_names, health = self._health_tensor()
            parts.append(health)
        host = torch.cat([p.reshape(-1).float() for p in parts]).cpu().numpy()
        env = dict(zip(_ENV_ACC_NAMES, host[: len(_ENV_ACC_NAMES)], strict=True))
        offset = len(_ENV_ACC_NAMES)
        metrics: dict[str, float] = {}
        steps = max(1, int(self.config.flashsac.log_interval_steps))
        metrics["train/step_reward_mean"] = float(env["reward_sum"]) / steps
        done_count = float(env["done_count"])
        metrics["episode/done_count"] = done_count
        if done_count > 0:
            metrics["episode/return"] = float(env["episode_return_sum"]) / done_count
            metrics["episode/length"] = float(env["episode_length_sum"]) / done_count
            metrics["health/time_out_share"] = (
                float(env["truncated_count"]) / done_count
            )
        metrics["health/final_obs_missing"] = float(env["final_obs_missing"])
        nonfinite: list[str] = []
        for i, k in enumerate(keys):
            value = float(host[offset + i]) / max(1, self._acc_n[k])
            metrics[f"train/{k}"] = value
            if not math.isfinite(value):
                nonfinite.append(k)
        offset += len(keys)
        for i, name in enumerate(health_names):
            metrics[name] = float(host[offset + i])
        if "train/critic/loss" in metrics:
            metrics["train/critic_loss"] = metrics["train/critic/loss"]
        if "train/actor/loss" in metrics:
            metrics["train/actor_loss"] = metrics["train/actor/loss"]
        if "train/actor/entropy" in metrics:
            target = self.learner.target_entropy
            metrics["health/entropy_gap"] = abs(
                metrics["train/actor/entropy"] - target
            ) / max(abs(target), 1e-8)
        now = time.monotonic()
        dt = max(now - run.interval_start_time, 1e-9)
        d_frames = run.frames_processed - run.interval_start_frames
        d_updates = self.learner.update_count - run.interval_start_updates
        metrics["time/speed"] = d_frames / dt
        metrics["time/updates_per_s"] = d_updates / dt
        env_steps = max(1, d_frames // self._num_envs)
        metrics["health/updates_per_env_step"] = d_updates / env_steps
        metrics["health/updates_per_env_step_target"] = float(run.rate)
        metrics["replay/rows"] = float(self.data_buffer.filled_rows)
        metrics["replay/transitions"] = float(
            self.data_buffer.filled_rows * self._num_envs
        )
        metrics["train/update_count"] = float(self.learner.update_count)
        metrics["health/nonfinite_metrics"] = float(len(nonfinite))
        run.interval_start_time = now
        run.interval_start_frames = run.frames_processed
        run.interval_start_updates = self.learner.update_count
        self._env_acc.zero_()
        self._acc, self._acc_n = {}, {}
        if nonfinite and self.config.flashsac.fail_on_nonfinite:
            msg = (
                f"Non-finite learner metrics {nonfinite} at frame "
                f"{run.frames_processed}. Training is diverged."
            )
            raise FloatingPointError(msg)
        return metrics

    def _checkpoint_dir(self) -> Path:
        return self.log_dir / self.config.logger.save_path

    def record(
        self,
        iteration: FlashSACIterationData,
        metadata: FlashSACTrainingMetadata,
    ) -> None:
        """Log every ``log_interval_steps`` env steps and write checkpoints."""
        fs = self.config.flashsac
        end = iteration.iteration_idx + 1 == metadata.total_iterations
        if (iteration.iteration_idx + 1) % fs.log_interval_steps == 0 or end:
            steps_in_window = (iteration.iteration_idx + 1) % fs.log_interval_steps
            metrics = self._flush_metrics(metadata)
            if steps_in_window and end:
                # Short last window: rescale the per-step reward mean.
                metrics["train/step_reward_mean"] *= (
                    fs.log_interval_steps / steps_in_window
                )
            iteration.metrics.update(metrics)
            self.log_metrics(metrics, step=metadata.frames_processed, log_python=False)
            self.log.info(
                self._format_iteration_summary(
                    metadata, iteration, self._progress_summary_fields()
                )
            )
        save_now = self._should_save_checkpoint(
            frames_processed=metadata.frames_processed,
            frames_in_iteration=iteration.frames,
        )
        if fs.checkpoint_interval_s > 0 and (
            time.monotonic() - metadata.last_checkpoint_time >= fs.checkpoint_interval_s
        ):
            save_now = True
        if save_now:
            self.save_model(path=self._checkpoint_dir(), step=metadata.frames_processed)
            metadata.last_checkpoint_time = time.monotonic()

    # ------------------------------------------------------------------- train
    def train(self) -> None:  # type: ignore[override]
        """Collect, insert, update, record. Thin orchestration only."""
        self.validate_training()
        metadata = self.init_metadata()
        saved_at = -1
        try:
            for iteration_idx in range(metadata.total_iterations):
                iteration = self.collect(metadata, iteration_idx)
                with self._profile_iteration_phase(iteration.phase_times, "prepare"):
                    self.prepare(iteration, metadata)
                with self._profile_iteration_phase(iteration.phase_times, "learn"):
                    self.iterate(iteration, metadata)
                if "learn" in iteration.phase_times:
                    iteration.learn_time = iteration.phase_times["learn"]
                with self._profile_iteration_phase(iteration.phase_times, "record"):
                    self.record(iteration, metadata)
                self._finish_iteration_profile(metadata, iteration)
            if int(self.config.save_interval) > 0:
                saved_at = metadata.frames_processed
                self.save_model(
                    path=self._checkpoint_dir(), step=metadata.frames_processed
                )
        except KeyboardInterrupt:
            # Walltime/SIGTERM or Ctrl+C: persist the state so a chained segment
            # resumes from HERE (same naming as periodic checkpoints).
            if (
                int(self.config.save_interval) > 0
                and saved_at != metadata.frames_processed
            ):
                self.log.info(
                    "Interrupted at frame %d; writing a final resume checkpoint.",
                    int(metadata.frames_processed),
                )
                self.save_model(
                    path=self._checkpoint_dir(), step=int(metadata.frames_processed)
                )
            raise
        finally:
            self.collector.shutdown()

    # ------------------------------------------------------------- inference
    @torch.no_grad()
    def predict(self, obs: Tensor | np.ndarray) -> Tensor:  # type: ignore[override]
        """Deterministic action ``bound * tanh(mean)`` for concatenated actor obs."""
        x = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        single = x.ndim == 1
        if single:
            x = x.unsqueeze(0)
        mean, _ = self.learner.actor.mean_and_std(
            x.reshape(-1, x.shape[-1]), training=False
        )
        action = self.learner.action_bound * torch.tanh(mean)
        action = action.reshape(*x.shape[:-1], -1)
        return action[0] if single else action

    # -------------------------------------------------------------- checkpoint
    def policy_meta(self) -> dict[str, Any]:
        """Deployment facts that a loader must verify."""
        lc = self.learner.cfg
        return {
            "arch": ARCH_TAG,
            "in_keys": [list(_key_tuple(k)) for k in self._actor_keys],
            "critic_in_keys": [list(_key_tuple(k)) for k in self._critic_keys],
            "squash": "tanh",
            "action_bound": [float(x) for x in self.learner.action_bound.tolist()],
            "actor_obs_dim": self._dims["actor"],
            "critic_obs_dim": self._dims["critic"],
            "act_dim": self._dims["act"],
            "actor_dim": lc.actor_dim,
            "actor_blocks": lc.actor_blocks,
            "batch_norm": "running_stats_in_policy_state_dict",
        }

    def _check_policy_meta(self, meta: dict[str, Any]) -> None:
        mine = self.policy_meta()
        for key in (
            "arch",
            "in_keys",
            "squash",
            "actor_obs_dim",
            "act_dim",
            "actor_dim",
            "actor_blocks",
        ):
            if meta.get(key) != mine[key]:
                msg = (
                    f"Checkpoint policy_meta[{key!r}]={meta.get(key)!r} does not match "
                    f"this agent ({mine[key]!r})."
                )
                raise ValueError(msg)

    def _checkpoint_target(self, path: str | Path | None, step: int | None) -> Path:
        base = Path(path).expanduser() if path else self.log_dir
        is_file = base.is_file() or (bool(base.suffix) and not base.is_dir())
        if step:
            if is_file:
                stem = base.with_suffix("")
                return stem.with_name(stem.name + f"_step_{step}{base.suffix}")
            return base / f"model_step_{step}.pt"
        return base if is_file else base / "model.pt"

    def save_model(
        self, path: str | Path | None = None, step: int | None = None
    ) -> None:
        """Write ONE checkpoint file (see the module docstring)."""
        target = self._checkpoint_target(path, step)
        target.parent.mkdir(parents=True, exist_ok=True)
        assert self.policy is not None
        frames = int(step) if step is not None else int(self._frames_processed)
        noise = self.policy_net.noise
        payload: dict[str, Any] = {
            "policy_state_dict": self.policy.state_dict(),
            "policy_meta": self.policy_meta(),
            "learner_state_dict": self.learner.state_dict(),
            "agent_state": {
                "updates_run": int(self._updates_run),
                "train_steps": int(self._train_steps),
                "noise_repeat": noise.state_dict() if noise is not None else None,
            },
            "flashsac_config": dataclasses.asdict(self.config.flashsac),
            "cumulative_env_frames": frames,
        }
        tmp = target.with_name(target.name + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(target)

    def load_model(self, path: str) -> None:
        """Strict restore of the policy and, when present, the learner state."""
        data = torch.load(path, map_location=self.device, weights_only=False)
        if "vec_norm_msg" in data:
            msg = "FlashSAC checkpoints carry no vec_norm_msg; this file is not one."
            raise ValueError(msg)
        meta = data.get("policy_meta")
        if meta is None:
            msg = f"{path} has no policy_meta: it is not a FlashSAC checkpoint."
            raise ValueError(msg)
        self._check_policy_meta(meta)
        assert self.policy is not None
        learner_state = data.get("learner_state_dict")
        if learner_state is not None:
            self.learner.load_state_dict(learner_state)
        self.policy.load_state_dict(data["policy_state_dict"], strict=True)
        agent_state = data.get("agent_state") or {}
        noise = self.policy_net.noise
        if noise is not None and agent_state.get("noise_repeat") is not None:
            noise.load_state_dict(agent_state["noise_repeat"])
        self._updates_run = int(agent_state.get("updates_run", 0))
        frames = data.get("cumulative_env_frames")
        if frames is not None:
            self._resume_frame_offset = int(frames)
        self._loaded_policy = True
        if learner_state is None:
            self.log.info("Loaded policy-only FlashSAC checkpoint (no learner state).")
