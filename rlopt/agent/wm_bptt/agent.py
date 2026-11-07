"""World-model BPTT actor-critic (SuperMimic, arXiv 2610.28480).

One iteration:

1. **Collect** ``env_steps_per_iter`` vectorized simulator steps with the
   deterministic actor plus N(0, sigma^2) action noise into a per-env replay.
2. **Fit** the dynamics model on ``wm_batches_per_iter`` batches of k-step
   windows, rolled out autoregressively on the logged actions with gradients
   through every step; the error of each step is expressed in the true state's
   frames and divided by the increment scale.
3. **Improve the actor**: from replay states (history rebuilt as the env's
   buffer holds it), imagine H steps through the frozen model with fresh sensor
   noise and backpropagate J = sum_t gamma^t c_t r_t + gamma^H c_H V(s_H),
   c_{t+1} = c_t (1 - d_{t+1}), through the model's Jacobians (or, with
   ``actor_estimator=score_function``, estimate the same gradient by the
   likelihood ratio of sampled actions on the same rollouts, model detached).
4. **Fit the critic** to GAE(lambda) targets of the detached imagined rollout.

The task model (``task.WorldModelTask``) supplies states, features, reward and
termination; this module is task-agnostic. The simulator is stepped directly
(the Isaac env under the TorchRL wrapper); the TorchRL policy module
``collector_policy`` maps the env's observation TensorDict to the deterministic
action for evaluators.
"""

from __future__ import annotations

import dataclasses
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch._dynamo
from tensordict.nn import TensorDictModule
from torch import Tensor, nn
from torchrl.envs import TransformedEnv

from rlopt.agent.wm_bptt.config import WMBPTTRLOptConfig, validate_options
from rlopt.agent.wm_bptt.diffsr_dynamics import DiffSRDynamics
from rlopt.agent.wm_bptt.networks import Actor, Critic, DynamicsModel
from rlopt.agent.wm_bptt.rebar import (
    CorrectionAverage,
    clipped_score_surrogate,
    linearized_return,
    masked_center,
    shuffle_rows,
)
from rlopt.agent.wm_bptt.replay import SequenceReplay
from rlopt.agent.wm_bptt.speed_options import ActorAverage
from rlopt.agent.wm_bptt.task import (
    WorldModelTask,
    history_proprio,
    load_task,
    task_proprio,
)
from rlopt.base_class import BaseAlgorithm, TrainingMetadata, honours_train_time_budget

ARCH_TAG = "wmbptt-v1"
RESUME_MATCH: dict[str, Any] = {"actor_ema": 0.0}
"""``wm`` fields a resumed run must share with its checkpoint, with the value a
checkpoint from before the field implies: they set which optimizers and which
policy entries the checkpoint holds."""


def _find_isaac_env(env: Any) -> Any:
    stack, seen = [env], set()
    while stack:
        cur = stack.pop()
        if id(cur) in seen:
            continue
        seen.add(id(cur))
        if hasattr(cur, "step") and hasattr(cur, "reward_manager"):
            return cur
        for attr in ("base_env", "_env", "env", "unwrapped"):
            nxt = getattr(cur, attr, None)
            if nxt is not None:
                stack.append(nxt)
    msg = "WMBPTT needs an Isaac Lab manager-based env under the TorchRL wrapper."
    raise RuntimeError(msg)


def _plain(value: Any) -> Any:
    """Builtin copy of a config value (configclass wraps some strings), so a
    checkpoint loads without the config's own packages."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, str):
        return str(value)
    return value


def lambda_returns(rewards: Tensor, values: Tensor, notdone: Tensor, gamma: float, lam: float) -> Tensor:
    """TD(lambda) targets over an imagined rollout (SuperMimic Eq. 3).

    ``rewards`` and ``notdone`` are (B, H): the reward of step t and the
    probability the episode continues past it; ``values`` is (B, H + 1).
    R_H = V_H and R_t = r_t + gamma c_t ((1 - lam) V_{t+1} + lam R_{t+1}).
    """
    H = rewards.shape[1]
    targets = torch.zeros_like(rewards)
    nxt = values[:, H]
    for i in range(H - 1, -1, -1):
        nxt = rewards[:, i] + gamma * notdone[:, i] * ((1 - lam) * values[:, i + 1] + lam * nxt)
        targets[:, i] = nxt
    return targets


def score_function_surrogate(
    actions: Tensor,
    mus: Tensor,
    rewards: Tensor,
    values: Tensor,
    notdone: Tensor,
    alive: Tensor,
    std: float,
    gamma: float,
    lam: float,
    *,
    normalize: bool = True,
    actor_steps: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Likelihood-ratio surrogate of an imagined rollout, (B,).

    ``actions`` (B, H, A) were sampled as ``mu + std * eps``; ``mus`` carry the
    actor's gradient, everything else is treated as constant. Its gradient is
    ``sum_t gamma^t c_t A_t grad log N(a_t; mu_t, std^2)`` with ``c_t`` =
    ``alive`` (B, H), the probability the imagined episode is still running
    when step t is taken, and ``A_t`` = TD(lambda) target - V(s_t) from
    ``values`` (B, H + 1). ``actor_steps`` limits the sum to the first N steps
    (the actor-horizon ablation). Returns the surrogate and the advantages.
    """
    H = rewards.shape[1]
    with torch.no_grad():
        targets = lambda_returns(rewards, values, notdone, gamma, lam)
        adv = targets - values[:, :H]
        if normalize:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        disc = gamma ** torch.arange(H, device=rewards.device, dtype=rewards.dtype)
        w = disc * alive
        if actor_steps is not None and actor_steps < H:
            w = w * (torch.arange(H, device=w.device) < actor_steps)
    logp = -((actions.detach() - mus).square().sum(-1)) / (2 * std * std)
    return (w * adv * logp).sum(1), adv


def real_window_masks(same: Tensor, terminated: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Episode bookkeeping of recorded windows.

    ``same`` (B, H): transition t stays in one episode; ``terminated`` (B, H):
    the env terminated (not timed out) on transition t. Returns ``inep``
    (B, H + 1), 1 while state t belongs to the window's first episode, and
    ``term`` / ``trunc`` (B, H), 1 on the transition that ends that episode by
    a termination / any other reset (timeout).
    """
    H = same.shape[1]
    s = same.float()
    inep = torch.cat((torch.ones_like(s[:, :1]), torch.cumprod(s, 1)), 1)
    end = inep[:, :H] * (1 - s)
    term = end * terminated.float()
    return inep, term, end - term


def real_window_targets(
    rewards: Tensor, env_reward: Tensor, values: Tensor, inep: Tensor, term: Tensor, trunc: Tensor, gamma: float, lam: float
) -> Tensor:
    """TD(lambda) targets (B, H) over recorded windows (``real_window_masks``).

    Transitions inside the episode use ``rewards``; a terminating transition
    contributes the env's reward and no continuation; a timed-out transition
    is unknown (its next state is the reset row), so its state's value V(s_t)
    stands in for it. Entries past the episode are 0 and must not be fit.
    """
    H = rewards.shape[1]
    r = inep[:, 1:] * rewards + term * env_reward + trunc * values[:, :H]
    return lambda_returns(r, values, inep[:, 1:], gamma, lam)


class WMPolicy(nn.Module):
    """Deployed policy: env observation tensors -> deterministic action."""

    def __init__(self, task: WorldModelTask, actor: Actor, bf16: bool) -> None:
        super().__init__()
        self.task = task
        self.actor = actor
        self.bf16 = bf16

    def forward(self, *tensors: Tensor) -> Tensor:
        obs = self.task.policy_obs_from_env(*tensors)
        return self.actor(obs, bf16=self.bf16)


class WMBPTT(BaseAlgorithm[WMBPTTRLOptConfig]):
    """Model-based BPTT tracker (see the module docstring)."""

    def __init__(self, env: TransformedEnv, config: WMBPTTRLOptConfig, logger: Any = None, **kwargs: Any) -> None:
        validate_options(config)
        self._num_envs = int(env.batch_size.numel()) if len(env.batch_size) else 1
        device = torch.device(config.device)
        wm = config.wm
        task_kwargs = dict(wm.task_kwargs)
        if wm.goal != "explicit" or wm.goal_encoder:
            task_kwargs.update(goal_mode=str(wm.goal), skill_encoder=str(wm.goal_encoder) or None)
        self.task: WorldModelTask = load_task(
            wm.task_model, env, device, sensor_noise=wm.sensor_noise, **task_kwargs
        )
        self._isaac = _find_isaac_env(env)
        self._frames_processed = 0
        self._resume_frame_offset = 0
        self._iteration = 0
        self._step_acc = 0.0
        self.replay: SequenceReplay | None = None
        self._inference_loaded = False
        self.actor_avg: ActorAverage | None = None
        self.policy_avg: TensorDictModule | None = None
        self._rebar = CorrectionAverage(wm.rebar_ema_rate, wm.rebar_norm_cap) if wm.actor_estimator == "rebar" else None
        super().__init__(env=env, config=config, logger=logger, **kwargs)

    # ------------------------------------------------------------ construction
    def _construct_policy(self) -> TensorDictModule:
        wm = self.config.wm
        t = self.task
        self.dynamics = self._construct_dynamics().to(self.device)
        self.actor = Actor(
            t.policy_obs_dim, t.action_dim, wm.actor_hidden, wm.actor_output_scale, wm.actor_activation
        ).to(self.device)
        self.critic = Critic(t.critic_obs_dim, wm.critic_hidden, wm.critic_activation).to(self.device)
        self.policy_net = WMPolicy(t, self.actor, wm.bf16)
        if wm.actor_ema > 0:
            self.actor_avg = ActorAverage(self.actor, wm.actor_ema)
            self.policy_avg = TensorDictModule(
                WMPolicy(t, self.actor_avg.module, wm.bf16), in_keys=list(t.policy_in_keys), out_keys=["action"]
            )
        return TensorDictModule(self.policy_net, in_keys=list(t.policy_in_keys), out_keys=["action"])

    def _construct_dynamics(self) -> nn.Module:
        wm, t = self.config.wm, self.task
        if wm.dynamics_model == "mlp":
            return DynamicsModel(t.wm_input_dim, t.increment_dim, wm.wm_hidden)
        if wm.dynamics_model != "diffsr":
            msg = f"wm.dynamics_model must be 'mlp' or 'diffsr', got {wm.dynamics_model!r}."
            raise ValueError(msg)
        # The world-model input is (state features, action, sim context).
        hi = t.wm_input_dim - t.context_dim
        return DiffSRDynamics(
            t.wm_input_dim,
            t.increment_dim,
            (hi - t.action_dim, hi),
            feature_dim=wm.diffsr_feature_dim,
            embed_dim=wm.diffsr_embed_dim,
            num_noises=wm.diffsr_num_noises,
            readout_detach_phi=False,  # the read-out is f: gradients must reach (s, a)
        )

    def _dynamics_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.dynamics.parameters() if p.requires_grad]

    def _construct_feature_extractor(self) -> TensorDictModule:
        msg = "WMBPTT has no feature extractor."
        raise NotImplementedError(msg)

    def _construct_actor_critic(self) -> TensorDictModule:
        assert self.policy is not None
        return self.policy

    @property
    def _deploys_average(self) -> bool:
        """The average exists once the actor has updated; before that it would
        hold stale normalizers, so the live actor is the deployed one."""
        return self.actor_avg is not None and self.actor_avg.updates > 0

    @property
    def collector_policy(self) -> TensorDictModule:
        """The policy evaluators run. Collection during training uses ``actor``."""
        assert self.policy is not None
        return self.policy_avg if self._deploys_average else self.policy  # type: ignore[return-value]

    def _construct_data_buffer(self) -> None:  # type: ignore[override]
        return None  # allocated lazily in train(): evaluation never needs it

    def _construct_collector(self, env: TransformedEnv, policy: TensorDictModule) -> None:  # type: ignore[override]
        del env, policy  # the simulator is stepped directly

    def _construct_loss_module(self) -> nn.Module:
        return nn.ModuleDict({"dynamics": self.dynamics, "critic": self.critic})

    def _configure_optimizers(self):  # type: ignore[override]
        wm = self.config.wm
        self.wm_optim = torch.optim.AdamW(self._dynamics_parameters(), lr=wm.wm_lr, weight_decay=wm.wm_weight_decay)
        self.actor_optim = torch.optim.Adam(self.actor.net.parameters(), lr=wm.actor_lr, betas=tuple(wm.actor_betas))
        self.critic_optim = torch.optim.Adam(self.critic.net.parameters(), lr=wm.critic_lr)
        self.lr_scheduler = None

    def _compile_components(self) -> None:
        self.compile_mode = None
        if self.config.wm.compile:
            # The two compiled steps have a handful of guard variants (train vs
            # eval mode, first call); a low limit silently falls back to eager.
            torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 64)

    def _progress_summary_fields(self) -> tuple[tuple[str, str], ...]:
        return (("wm/loss", "wm"), ("actor/objective", "J"), ("imagine/alive_H", "alive"))

    def predict(self, obs: Tensor | np.ndarray) -> Tensor:  # type: ignore[override]
        x = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        actor = self.actor_avg.module if self._deploys_average else self.actor  # type: ignore[union-attr]
        with torch.no_grad():
            return actor(x, bf16=self.policy_net.bf16)

    # ------------------------------------------------------------ schedule
    def _lr_factor(self) -> float:
        wm = self.config.wm
        p = min(1.0, self._iteration / max(1, wm.lr_cosine_iters))
        return wm.lr_final_ratio + (1 - wm.lr_final_ratio) * 0.5 * (1 + math.cos(math.pi * p))

    def _set_lrs(self) -> None:
        f = self._lr_factor()
        wm = self.config.wm
        for opt, base in ((self.wm_optim, wm.wm_lr), (self.actor_optim, wm.actor_lr), (self.critic_optim, wm.critic_lr)):
            for g in opt.param_groups:
                g["lr"] = base * f

    # ------------------------------------------------------------ collection
    def _policy_features(self, rows: dict[str, Tensor]) -> Tensor:
        """The collector's policy input; its sensor noise is keyed to the
        replay row this step writes (``_real_rollout`` regenerates it)."""
        assert self.replay is not None
        step = torch.full_like(self._env_ids, self.replay.steps)
        kwargs = {}
        if getattr(self.task, "proprio_at", None) is not None:
            kwargs["episode_step"] = rows["episode_step"]
        return self.task.policy_obs(
            rows["hist_states"], rows["hist_actions"], rows["context"], rows["rank"], rows["frame"],
            noise_key=(self._noise_seed, self._env_ids, step), **kwargs,
        )

    @torch.no_grad()
    def _collect_step(self, obs: dict[str, Any], stats: dict[str, list]) -> dict[str, Any]:
        wm = self.config.wm
        rows = self.task.read_obs(obs[self.task.obs_group])
        feat = self._policy_features(rows)
        self.actor.obs_norm.update(feat)
        mean_action = self.actor(feat, bf16=wm.bf16)
        action = mean_action
        if wm.action_noise_std > 0:
            action = action + wm.action_noise_std * torch.randn_like(action)
        action = action.clamp(-wm.action_clip, wm.action_clip)
        assert self.replay is not None
        self.replay.add(
            rows["state"], action, rows["hist_actions"][:, -1], rows["context"],
            rows["rank"], rows["frame"], rows["episode_step"], rows["push"],
            policy_action=mean_action,
        )
        obs, reward, terminated, truncated, extras = self._isaac.step(action)
        self.replay.set_outcome(reward.reshape(-1), terminated.reshape(-1).bool())
        self._track_episodes(reward.reshape(-1), (terminated | truncated).reshape(-1).bool())
        stats["env_reward"].append(reward.mean())
        log = extras.get("log") if isinstance(extras, dict) else None
        if log:
            for k, value in log.items():
                if isinstance(value, torch.Tensor):
                    value = value.float().mean()  # noqa: PLW2901
                stats[k].append(torch.as_tensor(value, device=self.device, dtype=torch.float32))
        self._frames_processed += self._num_envs
        return obs

    @torch.no_grad()
    def _track_episodes(self, reward: Tensor, done: Tensor) -> None:
        """Device-side sums behind ``episode/length`` and ``episode/return``."""
        self._episode_length += 1
        self._episode_return += reward
        ended = (self._episode_length, self._episode_return)
        self._episode_sums += torch.stack(
            (done.sum(), *(torch.where(done, x, 0.0).sum() for x in ended))
        )
        for x in ended:
            x.masked_fill_(done, 0.0)

    # ------------------------------------------------------------ world model
    def _wm_step(self, state: Tensor, action: Tensor, ctx: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        inc = self.dynamics(self.task.wm_input(state, action, ctx), bf16=self.config.wm.bf16)
        nxt = self.task.integrate(state, inc)
        return nxt, self.task.state_error(nxt, target)

    def _wm_step_fn(self):
        fn = getattr(self, "_compiled_wm_step", None)
        if fn is None:
            fn = torch.compile(self._wm_step, dynamic=False) if self.config.wm.compile else self._wm_step
            self._compiled_wm_step = fn
        return fn

    def _wm_rollout(self, win: dict[str, Tensor], k: int, *, grad: bool) -> tuple[Tensor, Tensor]:
        """Open-loop rollout on logged actions; per-step errors (B, k, D) and
        the window mask (B, k) (all transitions up to j valid)."""
        state = win["state"][:, 0]
        ctx = win["context"][:, 0]
        if grad:
            state = state.detach().contiguous().requires_grad_(True)  # one compiled variant
        errs = []
        step = self._wm_step_fn() if grad else self._wm_step
        for j in range(k):
            state, err = step(state, win["action"][:, j], ctx, win["state"][:, j + 1])
            errs.append(err)
            if not grad:
                state = state.detach()
        mask = torch.cumprod(win["valid"][:, :k].float(), dim=1)
        return torch.stack(errs, 1), mask

    def _update_world_model(self, metrics: dict[str, list]) -> None:
        wm = self.config.wm
        k = wm.wm_window
        assert self.replay is not None
        scale = torch.maximum(self.dynamics.out_norm.std(), self.task.increment_floor)
        for _ in range(wm.wm_batches_per_iter):
            tau, env = self.replay.sample_starts(wm.wm_batch_size, k)
            win = self.replay.window(tau, env, k + 1)
            if self.dynamics.out_norm.count < wm.norm_freeze_samples:
                v = win["valid"][:, 0]
                if v.any():
                    s0, s1 = win["state"][v, 0], win["state"][v, 1]
                    self.dynamics.in_norm.update(self.task.wm_input(s0, win["action"][v, 0], win["context"][v, 0]))
                    self.dynamics.out_norm.update(self.task.increment(s0, s1))
                scale = torch.maximum(self.dynamics.out_norm.std(), self.task.increment_floor)
            errs, mask = self._wm_rollout(win, k, grad=True)
            per = (errs / scale).square().mean(-1)  # (B, k)
            loss = (per * mask).sum() / mask[:, 0].sum().clamp_min(1.0)
            if wm.dynamics_model == "diffsr":
                denoise = self._denoising_loss(win, k)
                metrics["wm/readout_loss"].append(loss.detach())
                metrics["wm/denoise_loss"].append(denoise.detach())
                loss = loss + wm.diffsr_denoise_coeff * denoise
            self.wm_optim.zero_grad(set_to_none=True)
            loss.backward()
            gn = nn.utils.clip_grad_norm_(self._dynamics_parameters(), wm.wm_grad_clip_norm)
            if torch.isfinite(gn):
                self.wm_optim.step()
            metrics["wm/loss"].append(loss.detach())
            metrics["wm/grad_norm"].append(gn.detach())
            metrics["wm/valid_window_frac"].append(mask[:, -1].mean())
        if self.dynamics.out_norm.count >= wm.norm_freeze_samples:
            self.dynamics.in_norm.frozen.fill_(True)
            self.dynamics.out_norm.frozen.fill_(True)

    def _denoising_loss(self, win: dict[str, Tensor], k: int) -> Tensor:
        """DiffSR's one-step loss on every valid transition of the windows,
        teacher-forced on the logged states."""
        t, wm = self.task, self.config.wm
        B = win["state"].shape[0]
        s0 = win["state"][:, :k].reshape(B * k, -1)
        s1 = win["state"][:, 1 : k + 1].reshape(B * k, -1)
        ctx = win["context"][:, :1].expand(B, k, -1).reshape(B * k, -1)
        x = t.wm_input(s0, win["action"][:, :k].reshape(B * k, -1), ctx)
        loss, _ = self.dynamics.denoising_loss(x, t.increment(s0, s1), win["valid"][:, :k].reshape(-1), bf16=wm.bf16)
        return loss

    @torch.no_grad()
    def _wm_eval(self) -> dict[str, float]:
        """Open-loop k-step error on replay windows in physical units."""
        wm = self.config.wm
        assert self.replay is not None
        L = wm.eval_window
        if self.replay.steps - self.replay.oldest < L + 2:
            return {}
        tau, env = self.replay.sample_starts(wm.eval_batch, L)
        win = self.replay.window(tau, env, L + 1)
        errs, mask = self._wm_rollout(win, L, grad=False)
        out: dict[str, float] = {}
        scale = torch.maximum(self.dynamics.out_norm.std(), self.task.increment_floor)
        for k in (1, 8, 32):
            if k > L:
                continue
            m = mask[:, k - 1] > 0
            if not m.any():
                continue
            e = errs[m, k - 1]
            out[f"wm_eval/k{k}_root_pos_mm"] = float(e[:, 0:3].norm(dim=-1).mean() * 1000)
            out[f"wm_eval/k{k}_root_rot_deg"] = float(e[:, 3:6].norm(dim=-1).mean() * 180 / math.pi)
            out[f"wm_eval/k{k}_root_lin_vel"] = float(e[:, 6:9].norm(dim=-1).mean())
            out[f"wm_eval/k{k}_joint_pos_deg"] = float(e[:, 12:41].abs().mean() * 180 / math.pi)
            out[f"wm_eval/k{k}_joint_vel"] = float(e[:, 41:70].abs().mean())
            out[f"wm_eval/k{k}_normalized"] = float((e / scale).square().mean())
            out[f"wm_eval/k{k}_windows"] = float(m.float().sum())
        return out

    # ------------------------------------------------------------ imagination
    def _imagine_step(
        self,
        state: Tensor,
        prop: Tensor,
        prev_action: Tensor,
        ctx: Tensor,
        rank: Tensor,
        frame: Tensor,
        eps: Tensor,
        detach_action: bool,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """One imagined control step: act, predict, score, extend the history.

        ``eps`` (B, A) is the pre-scaled action noise (zeros for the
        deterministic actor); ``detach_action`` cuts the gradient between the
        actor's output and the model (score-function estimator). Returns the
        executed action and the actor's mean separately.
        """
        t, wm = self.task, self.config.wm
        obs = t.policy_obs_from_parts(prop, state, rank, frame)
        mu = self.actor(obs, bf16=wm.bf16)
        action = mu + eps
        if detach_action:
            action = action.detach()
        inc = self.dynamics(t.wm_input(state, action, ctx), bf16=wm.bf16)
        nxt = t.integrate(state, inc)
        r, d = t.reward_and_termination(nxt, action, prev_action, ctx, rank, frame)
        new_prop = task_proprio(t, nxt, action, ctx, rank, frame + 1, getattr(t, "sensor_noise", True))
        prop = torch.cat((prop[:, 1:], new_prop.unsqueeze(1)), 1)
        return action, mu, nxt, r, d, prop

    def _step_fn(self):
        fn = getattr(self, "_compiled_step", None)
        if fn is None:
            fn = self._imagine_step
            if self.config.wm.compile:
                fn = torch.compile(self._imagine_step, dynamic=False)
            self._compiled_step = fn
        return fn

    def _filter_imagination_starts(
        self,
        tau: Tensor,
        env: Tensor,
        horizon: int,
        *,
        resample: bool,
        generator: torch.Generator | None,
    ) -> tuple[Tensor, Tensor]:
        """Apply a task's optional ``valid_imagination_start`` capability.

        Eight rejection rounds keep the usual case small. If valid rows are
        scarce, enumerate the eligible ring once rather than loop forever
        or fail because random proposals missed them. Both paths sample
        uniformly over valid (row, env) pairs. Explicit probe starts fail
        validation instead of silently changing the requested experiment.
        """
        rp = self.replay
        assert rp is not None
        predicate = self.task.valid_imagination_start
        lo, hi = rp.oldest + self.task.history - 1, rp.steps - 1
        for attempt in range(8):
            slot = rp._slot(tau)
            valid = predicate(rp.rank[slot, env], rp.frame[slot, env], horizon)
            if valid.all():
                return tau, env
            if not resample:
                msg = f"Explicit imagination starts are invalid for horizon {horizon}."
                raise ValueError(msg)
            if attempt < 7:
                next_tau = torch.randint(lo, hi + 1, tau.shape, device=self.device, generator=generator)
                next_env = torch.randint(0, self._num_envs, env.shape, device=self.device, generator=generator)
                tau = torch.where(valid, tau, next_tau)
                env = torch.where(valid, env, next_env)
        rows = torch.arange(lo, hi + 1, device=self.device)
        slot = rp._slot(rows)
        candidates = predicate(rp.rank[slot], rp.frame[slot], horizon).nonzero()
        if candidates.shape[0] == 0:
            msg = (
                f"No valid imagination starts for horizon {horizon} in replay rows {lo}..{hi}; "
                "collect eligible reference frames or reduce the requested horizon."
            )
            raise RuntimeError(msg)
        pick = torch.randint(0, candidates.shape[0], tau.shape, device=self.device, generator=generator)
        chosen = candidates[pick]
        return torch.where(valid, tau, rows[chosen[:, 0]]), torch.where(valid, env, chosen[:, 1])

    def _imagine(
        self,
        batch: int,
        *,
        grad: bool,
        estimator: str | None = None,
        horizon: int | None = None,
        starts: tuple[Tensor, Tensor] | None = None,
        std: float | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, Tensor]:
        """Imagined H-step rollouts from replay states.

        ``estimator`` selects what ``objective`` differentiates to: ``pathwise``
        (deterministic actor, gradient through the model), ``reparam`` (actions
        ``mu + std eps``, gradient through the sample and the model) or
        ``score_function`` (same samples, gradient through ``log pi`` only;
        the model is traversed without gradient). None: the configured
        ``actor_estimator`` when ``grad``, else pathwise. ``horizon`` and
        ``starts`` (replay rows ``tau`` and env indices) override the config
        and the uniform start sampling (gradient probes); ``generator`` draws
        the action noise.
        """
        wm = self.config.wm
        t = self.task
        assert self.replay is not None
        H = int(horizon or wm.horizon)
        est = estimator or (wm.actor_estimator if grad else "pathwise")
        if est not in ("pathwise", "reparam", "score_function"):
            msg = f"unknown estimator {est!r}"
            raise ValueError(msg)
        sf = est == "score_function"
        sigma = float(wm.sf_action_std if std is None else std)
        h = t.history
        if starts is None:
            lo = self.replay.oldest + h - 1
            hi = self.replay.steps - 1
            tau = torch.randint(lo, hi + 1, (batch,), device=self.device, generator=generator)
            env = torch.randint(0, self._num_envs, (batch,), device=self.device, generator=generator)
        else:
            tau, env = starts
            batch = int(tau.shape[0])
        if getattr(t, "valid_imagination_start", None) is not None:
            tau, env = self._filter_imagination_starts(
                tau, env, H, resample=starts is None, generator=generator,
            )
        start = self.replay.history(tau, env, h)
        ctx, rank, frame = start["context"], start["rank"], start["frame"]
        noise = getattr(t, "sensor_noise", True)
        prop = history_proprio(t, start, noise)
        prev_action = start["hist_actions"][:, -1]
        state = start["hist_states"][:, -1]
        if grad:
            # Every step then sees the same requires_grad pattern; otherwise the
            # compiled step recompiles between step 0 and the rest.
            state, prop, prev_action = (
                x.detach().contiguous().requires_grad_(True) for x in (state, prop, prev_action)
            )
        zero_eps = torch.zeros(batch, t.action_dim, device=self.device)
        alive = torch.ones(batch, device=self.device)
        ret = torch.zeros(batch, device=self.device)
        states, rewards, dones, cont = [state], [], [], []
        actions, mus, alives = [], [], []
        step_fn = self._step_fn()
        actor_h = wm.actor_horizon if 0 < wm.actor_horizon < H else H
        actor_objective = None
        for step in range(H):
            if est == "pathwise":
                eps = zero_eps
            else:
                eps = sigma * torch.randn(batch, t.action_dim, device=self.device, generator=generator)
            alives.append(alive)
            action, mu, nxt, r, d, prop = step_fn(state, prop, prev_action, ctx, rank, frame, eps, sf)
            actions.append(action)
            mus.append(mu)
            end = t.clip_end(rank, frame).float()
            if not wm.imagine_termination:
                # Ablation: the imagined episode survives every tracking
                # failure; only the clip end stops it. The soft termination
                # is still logged (``dones``) but carries no gradient and
                # no return cut, for the actor and the critic alike.
                d = d.detach() * 0
            elif not wm.termination_gradient:
                d = d.detach()  # ablation: the cut stays, the gradient does not
            ret = ret + (wm.gamma**step) * alive * r
            alive = alive * (1 - d) * (1 - end)
            states.append(nxt)
            rewards.append(r)
            dones.append(d)
            cont.append(1 - end)
            state, prev_action = nxt, action
            frame = frame + 1
            if not grad:
                state, prev_action, prop = state.detach(), prev_action.detach(), prop.detach()
            elif sf or (step + 1 == actor_h and actor_h < H):
                # Score function: the model is never differentiated, so the
                # path is cut after every step. Pathwise with a shorter actor
                # horizon: the actor's objective ends here (bootstrapped with
                # V); the remaining steps only feed the critic's targets.
                # Re-wrapping keeps the compiled step's requires_grad pattern
                # unchanged (no recompile).
                if not sf and step + 1 == actor_h:
                    if wm.critic_bootstrap:
                        v_a = self.critic(t.critic_obs(state, ctx, rank, frame), bf16=wm.bf16)
                        actor_objective = ret + (wm.gamma**actor_h) * alive * v_a
                    else:
                        actor_objective = ret
                state, prop, prev_action = (
                    x.detach().contiguous().requires_grad_(True) for x in (state, prop, prev_action)
                )
        if wm.critic_bootstrap:
            v_last = self.critic(t.critic_obs(state, ctx, rank, frame), bf16=wm.bf16)
            objective = ret + (wm.gamma**H) * alive * v_last
        else:
            objective = ret
        if actor_objective is not None:
            objective = actor_objective
        out = {
            "objective": objective,
            "J": objective.detach(),
            "states": torch.stack(states, 1),
            "rewards": torch.stack(rewards, 1),
            "dones": torch.stack(dones, 1),
            "cont": torch.stack(cont, 1),
            "alive_H": alive,
            "actions": torch.stack(actions, 1),
            "mus": torch.stack(mus, 1),
            "alives": torch.stack(alives, 1),
            "ctx": ctx,
            "rank": rank,
            "frame0": start["frame"],
        }
        if sf and grad:
            with torch.no_grad():
                values = self._critic_values(out) if wm.critic_bootstrap else torch.zeros(batch, H + 1, device=self.device)
                notdone = (1 - out["dones"].detach()) * out["cont"]
            surrogate, adv = score_function_surrogate(
                out["actions"], out["mus"], out["rewards"].detach(), values, notdone, out["alives"],
                sigma, wm.gamma, wm.gae_lambda, normalize=wm.sf_normalize_advantage,
                actor_steps=actor_h if actor_h < H else None,
            )
            out["objective"] = surrogate
            out["advantages"] = adv
        return out

    # ------------------------------------------------------------ real windows
    def _real_step(
        self,
        state: Tensor,
        prop: Tensor,
        prev_action: Tensor,
        ctx: Tensor,
        rank: Tensor,
        frame: Tensor,
        eta: Tensor,
        xi: Tensor,
        prop_noise: Tensor,
        goal_noise: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """One re-simulated recorded step: the current actor on the clean
        proprio history plus the collector's sensor noise of this row
        (``prop_noise``, ``goal_noise``) and the recorded action noise ``eta``,
        the model's increment plus the recorded residual ``xi``. At the
        collecting policy this is the recorded transition."""
        t, wm = self.task, self.config.wm
        obs = t.policy_obs_from_parts(prop + prop_noise, state, rank, frame, goal_noise)
        action = self.actor(obs, bf16=wm.bf16) + eta
        inc = self.dynamics(t.wm_input(state, action, ctx), bf16=wm.bf16)
        nxt = t.integrate(state, inc + xi)
        r, _ = t.reward(nxt, action, prev_action, ctx, rank, frame)
        new_prop = task_proprio(t, nxt, action, ctx, rank, frame + 1, False)
        prop = torch.cat((prop[:, 1:], new_prop.unsqueeze(1)), 1)
        return action, nxt, r, prop, obs

    def _real_step_fn(self):
        fn = getattr(self, "_compiled_real_step", None)
        if fn is None:
            fn = self._real_step
            if self.config.wm.compile:
                fn = torch.compile(self._real_step, dynamic=False)
            self._compiled_real_step = fn
        return fn

    def _real_rollout(
        self,
        batch: int,
        *,
        grad: bool,
        track_noise: bool = False,
        zero_noise: bool = False,
        starts: tuple[Tensor, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        """H-step recorded windows (``actor_source`` / ``critic_source`` real).

        With ``grad`` the window is re-simulated under the current actor with
        the recorded noise held fixed (SVG noise inference): the action noise
        eta_t = a_t - mu_old(o_t) of the collector, and the model residual
        xi_t = increment(s_t, s_{t+1}) - f(s_t, a_t) in increment space. The
        forward pass equals the recording at the collecting policy and moves
        with the actor; the gradient runs through the model's Jacobians.
        Without ``grad`` only the recorded states and their analytic rewards
        are produced (critic data).
        """
        wm = self.config.wm
        t = self.task
        rp = self.replay
        assert rp is not None
        H, h = wm.horizon, t.history
        hi = rp.steps - 1 - H
        lo = max(rp.oldest + h - 1, hi - wm.real_window_recent_steps + 1)
        if hi < lo:
            msg = f"replay holds {rp.steps} steps; real windows of {H} need more."
            raise RuntimeError(msg)
        if starts is None:
            tau = torch.randint(lo, hi + 1, (batch,), device=self.device)
            env = torch.randint(0, self._num_envs, (batch,), device=self.device)
        else:
            tau, env = starts
        win = rp.window(tau, env, H + 1)
        es = win["episode_step"]
        same = es[:, 1:] == es[:, :-1] + 1
        inep, term, trunc = real_window_masks(same, win["terminated"][:, :H])
        env_reward = win["env_reward"][:, :H]
        ctx, rank, frame0 = win["context"][:, 0], win["rank"][:, 0], win["frame"][:, 0]
        steps = torch.arange(H + 1, device=self.device)
        frames = frame0.unsqueeze(1) + steps  # (B, H + 1), as imagination counts them
        if grad:
            with torch.no_grad():
                B, S = win["state"].shape[0], win["state"].shape[-1]
                s_now = win["state"][:, :H].reshape(B * H, S)
                s_next = win["state"][:, 1:].reshape(B * H, S)
                a_rec = win["action"][:, :H].reshape(B * H, -1)
                inc_rec = self.dynamics(t.wm_input(s_now, a_rec, ctx.repeat_interleave(H, 0)), bf16=wm.bf16)
                xi = (t.increment(s_now, s_next) - inc_rec).reshape(B, H, -1) * same.unsqueeze(-1)
                eta = win["action"][:, :H] - win["policy_action"][:, :H]
            eta_recorded = eta
            if zero_noise:
                eta = torch.zeros_like(eta)  # the policy's mean path, residuals kept
            if track_noise:
                # d objective / d eta_t is the SVG action gradient of step t.
                eta = eta.detach().contiguous().requires_grad_(True)
            start = rp.history(tau, env, h)
            # Clean history; each step adds the collector's keyed noise of its row.
            prop = history_proprio(t, start, False)
            state, prev_action = win["state"][:, 0], start["hist_actions"][:, -1]
            # One requires_grad pattern for every compiled step (see _imagine).
            state, prop, prev_action = (
                x.detach().contiguous().requires_grad_(True) for x in (state, prop, prev_action)
            )
            step_fn = self._real_step_fn()
            states, rewards, actions, actor_obs = [state], [], [], []
            for i in range(H):
                with torch.no_grad():
                    key = (self._noise_seed, env, tau + i)
                    if getattr(t, "proprio_at", None) is not None:
                        prop_noise = t.keyed_proprio_noise(*key, episode_step=start["episode_step"] + i)
                    else:
                        prop_noise = t.keyed_proprio_noise(*key)
                    goal_noise = t.keyed_goal_noise(*key)
                prev_action, state, r, prop, obs_i = step_fn(
                    state, prop, prev_action, ctx, rank, frames[:, i], eta[:, i], xi[:, i], prop_noise, goal_noise
                )
                actor_obs.append(obs_i)
                states.append(state)
                rewards.append(r)
                actions.append(prev_action)
            states_t, rewards_t = torch.stack(states, 1), torch.stack(rewards, 1)
            with torch.no_grad():
                # How far the re-simulated window leaves the recording (task
                # state starts with the pelvis position), and its first action
                # the recorded one (zero at the collecting policy).
                dev = (states_t[..., :3] - win["state"][..., :3]).norm(dim=-1)
                dev_mm = 1000 * (dev * inep).sum() / inep.sum().clamp_min(1.0)
                act_dev0 = (actions[0] - win["action"][:, 0]).norm(dim=-1).mean()
        else:
            states_t = win["state"]
            B, S = states_t.shape[0], states_t.shape[-1]
            r, _ = t.reward(
                states_t[:, 1:].reshape(B * H, S),
                win["action"][:, :H].reshape(B * H, -1),
                win["prev_action"][:, :H].reshape(B * H, -1),
                ctx.repeat_interleave(H, 0),
                rank.repeat_interleave(H, 0),
                frames[:, :H].reshape(-1),
            )
            rewards_t = r.reshape(B, H)
        out = {
            "states": states_t,
            "rewards": rewards_t,
            "env_reward": env_reward,
            "inep": inep,
            "term": term,
            "trunc": trunc,
            "ctx": ctx,
            "rank": rank,
            "frame0": frame0,
        }
        if grad:
            if track_noise:
                out["actions"] = torch.stack(actions, 1)
                out["eta"] = eta
                out["eta_recorded"] = eta_recorded
                out["actor_obs"] = torch.stack(actor_obs, 1).detach()
                out["recorded_states"] = win["state"]
                out["recorded_action"] = win["action"][:, :H]
                out["tau"], out["env"] = tau, env
            out["pelvis_dev_mm"] = dev_mm
            out["action_dev0"] = act_dev0
            gam = wm.gamma ** steps.float()
            objective = (gam[:H] * (inep[:, 1:] * rewards_t + term * env_reward)).sum(1)
            if wm.critic_bootstrap:
                values = self._critic_values(out)
                weights = torch.cat((gam[:H] * trunc, (gam[H] * inep[:, H]).unsqueeze(1)), 1)
                objective = objective + (weights * values).sum(1)
            out["objective"] = objective
        return out

    def _critic_obs_seq(self, out: dict[str, Tensor]) -> Tensor:
        """Critic inputs (B * (H + 1), D) of a rollout's states."""
        states = out["states"]
        B, H1, S = states.shape
        frames = out["frame0"].unsqueeze(1) + torch.arange(H1, device=self.device)
        rank = out["rank"].unsqueeze(1).expand(B, H1)
        ctx = out["ctx"].unsqueeze(1).expand(B, H1, out["ctx"].shape[-1])
        return self.task.critic_obs(states.reshape(B * H1, S), ctx.reshape(B * H1, -1), rank.reshape(-1), frames.reshape(-1))

    def _critic_values(self, out: dict[str, Tensor]) -> Tensor:
        B, H1 = out["states"].shape[:2]
        return self.critic(self._critic_obs_seq(out), bf16=self.config.wm.bf16).reshape(B, H1)

    @torch.no_grad()
    def _recorded_actor_obs(self, tau: Tensor, env: Tensor, H: int) -> Tensor:
        """Actor inputs (B, H, D) of the recorded rows, with the collector's keyed
        sensor noise: the observations the recorded actions were sampled from."""
        rp, t = self.replay, self.task
        assert rp is not None
        rows = []
        for i in range(H):
            hist = rp.history(tau + i, env, t.history)
            kwargs = {"episode_step": hist["episode_step"]} if getattr(t, "proprio_at", None) is not None else {}
            rows.append(
                t.policy_obs(
                    hist["hist_states"], hist["hist_actions"], hist["context"], hist["rank"], hist["frame"],
                    noise_key=(self._noise_seed, env, tau + i), **kwargs,
                )
            )
        return torch.stack(rows, 1)

    def _rebar_gradient(self, metrics: dict[str, list]) -> dict[str, Tensor]:
        """Fill the actor's ``.grad`` with mean-path SVG(H) plus the averaged
        score correction (``wm_bptt/rebar.py``); returns the rollout."""
        wm = self.config.wm
        assert self._rebar is not None
        params = [p for p in self.actor.net.parameters() if p.requires_grad]
        out = self._real_rollout(wm.imagine_batch, grad=True, track_noise=True, zero_noise=True)
        B, H = out["eta"].shape[:2]
        *svg, d_eta = torch.autograd.grad(-out["objective"].mean(), [*params, out["eta"]], allow_unused=True)
        svg_flat = torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(svg, params)])
        disc = wm.gamma ** torch.arange(H, device=self.device, dtype=torch.float32)
        with torch.no_grad():
            slopes = -d_eta * B / disc.view(1, H, 1)  # d J / d eta in the units of the advantage at t
            eta = out["eta_recorded"]
            action = out["recorded_action"]
            rec = self._real_rollout(B, grad=False, starts=(out["tau"], out["env"]))
            values = self._critic_values(rec)
            targets = real_window_targets(
                rec["rewards"], rec["env_reward"], values, rec["inep"], rec["term"], rec["trunc"], wm.gamma, wm.gae_lambda
            )
            # The env clamps: on a clipped row a - mu is not the Gaussian draw.
            unclipped = ~action.abs().ge(wm.action_clip).any(-1)
            mask = rec["inep"][:, :H] * (1 - rec["trunc"]) * unclipped.float()
            adv = masked_center(targets - values[:, :H], mask)
            lin = linearized_return(slopes, eta, mask, wm.gamma * wm.gae_lambda)
            rho = adv - lin
            if wm.rebar_shuffle:
                rho = shuffle_rows(rho, mask > 0)
            obs = self._recorded_actor_obs(out["tau"], out["env"], H)
        mu = self.actor(obs.reshape(B * H, -1), bf16=wm.bf16).reshape(B, H, -1)
        mu_old = action - eta
        surrogate, clip_frac = clipped_score_surrogate(
            mu, mu_old, action, rho, disc * mask, wm.action_noise_std, wm.rebar_clip_eps
        )
        corr = torch.autograd.grad(-surrogate, params, allow_unused=True)
        corr_flat = torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(corr, params)])
        applied, scale = self._rebar.update(corr_flat.detach(), svg_flat.detach())
        total = svg_flat + wm.rebar_weight * applied
        offset = 0
        for p in params:
            n = p.numel()
            p.grad = total[offset : offset + n].view_as(p).clone()
            offset += n
        with torch.no_grad():
            keep = mask > 0
            a_k, l_k = adv[keep], lin[keep]
            ac, lc = a_k - a_k.mean(), l_k - l_k.mean()
            metrics["rebar/corr_adv_noise_effect"].append((ac * lc).mean() / (ac.std() * lc.std()).clamp_min(1e-12))
            metrics["rebar/residual_var_ratio"].append(rho[keep].var() / a_k.var().clamp_min(1e-12))
            metrics["rebar/adv_std"].append(a_k.std())
            metrics["rebar/noise_effect_std"].append(l_k.std())
            metrics["rebar/svg_norm"].append(svg_flat.norm())
            metrics["rebar/correction_norm"].append(corr_flat.norm())
            metrics["rebar/applied_norm"].append(applied.norm())
            metrics["rebar/cap_scale"].append(scale)
            metrics["rebar/cos_applied_svg"].append(
                (applied @ svg_flat) / (applied.norm() * svg_flat.norm()).clamp_min(1e-12)
            )
            metrics["rebar/ratio_clip_frac"].append(clip_frac)
            metrics["rebar/action_clip_frac"].append(1 - unclipped.float().mean())
            metrics["rebar/mu_dev"].append((mu.detach() - mu_old).norm(dim=-1)[keep].mean())
        return out

    def _update_actor_critic(self, metrics: dict[str, list]) -> None:
        wm = self.config.wm
        real_actor = wm.actor_source == "real"
        self.dynamics.requires_grad_(False)
        self.critic.requires_grad_(False)
        try:
            self.actor_optim.zero_grad(set_to_none=True)
            if self._rebar is not None:
                out = self._rebar_gradient(metrics)
            else:
                if real_actor:
                    out = self._real_rollout(wm.imagine_batch, grad=True)
                else:
                    out = self._imagine(wm.imagine_batch, grad=True)
                loss = -out["objective"].mean()
                loss.backward()
        finally:
            self.dynamics.requires_grad_(True)
            self.critic.requires_grad_(True)
        params = list(self.actor.net.parameters())
        gn = nn.utils.clip_grad_norm_(params, wm.actor_grad_clip_norm)
        nn.utils.clip_grad_value_(params, wm.actor_grad_clip_value)
        if torch.isfinite(gn):
            self.actor_optim.step()
        if self.actor_avg is not None:
            self.actor_avg.update()
        metrics["actor/objective"].append(out.get("J", out["objective"].detach()).mean())
        metrics["actor/grad_norm"].append(gn.detach())
        if "advantages" in out:
            metrics["actor/sf_surrogate"].append(out["objective"].detach().mean())
            metrics["actor/sf_adv_std"].append(out["advantages"].std())
        if real_actor:
            metrics["real/alive_H"].append(out["inep"][:, -1].mean())
            metrics["real/term_frac"].append(out["term"].sum(1).mean())
            metrics["real/trunc_frac"].append(out["trunc"].sum(1).mean())
            # In-episode steps only: past an episode end the re-simulation
            # runs on without a recording (masked out of the objective).
            inep1 = out["inep"][:, 1:]
            metrics["real/reward_mean"].append((out["rewards"].detach() * inep1).sum() / inep1.sum().clamp_min(1.0))
            metrics["real/pelvis_dev_mm"].append(out["pelvis_dev_mm"])
            metrics["real/action_dev0"].append(out["action_dev0"])
        else:
            metrics["imagine/alive_H"].append(out["alive_H"].detach().mean())
            metrics["imagine/reward_mean"].append(out["rewards"].detach().mean())
            metrics["imagine/done_mean"].append(out["dones"].detach().mean())
        if wm.critic_fit and self._iteration % max(1, wm.critic_update_every) == 0:
            if wm.critic_source == "real":
                # The recorded states, not the actor's re-simulated ones.
                self._update_critic_real(self._real_rollout(wm.imagine_batch, grad=False), metrics)
            else:
                self._update_critic(self._imagine(wm.imagine_batch, grad=False) if real_actor else out, metrics)

    def _update_critic_real(self, out: dict[str, Tensor], metrics: dict[str, list]) -> None:
        wm = self.config.wm
        with torch.no_grad():
            states = out["states"].detach()
            B, H1 = states.shape[:2]
            H = H1 - 1
            cobs = self._critic_obs_seq({**out, "states": states})
            fit = (out["inep"][:, :H] * (1 - out["trunc"])).reshape(-1) > 0
            self.critic.obs_norm.update(cobs.reshape(B, H1, -1)[:, :H].reshape(B * H, -1)[fit])
            values = self.critic(cobs, bf16=wm.bf16).reshape(B, H1)
            targets = real_window_targets(
                out["rewards"].detach(), out["env_reward"], values, out["inep"], out["term"], out["trunc"],
                wm.gamma, wm.gae_lambda,
            )
            cobs = cobs.reshape(B, H1, -1)[:, :H].reshape(B * H, -1)[fit]
            targets = targets.reshape(-1)[fit]
        self._fit_critic(cobs, targets, metrics)

    def _update_critic(self, out: dict[str, Tensor], metrics: dict[str, list]) -> None:
        wm = self.config.wm
        t = self.task
        with torch.no_grad():
            states = out["states"].detach()  # (B, H+1, S)
            B, H1, S = states.shape
            H = H1 - 1
            frames = out["frame0"].unsqueeze(1) + torch.arange(H1, device=self.device)
            rank = out["rank"].unsqueeze(1).expand(B, H1)
            ctx = out["ctx"].unsqueeze(1).expand(B, H1, out["ctx"].shape[-1])
            cobs = t.critic_obs(states.reshape(B * H1, S), ctx.reshape(B * H1, -1), rank.reshape(-1), frames.reshape(-1))
            self.critic.obs_norm.update(cobs)
            values = self.critic(cobs, bf16=wm.bf16).reshape(B, H1)
            notdone = (1 - out["dones"].detach()) * out["cont"].detach()
            targets = lambda_returns(out["rewards"].detach(), values, notdone, wm.gamma, wm.gae_lambda)
            cobs = cobs.reshape(B, H1, -1)[:, :H].reshape(B * H, -1)
            targets = targets.reshape(-1)
        self._fit_critic(cobs, targets, metrics)

    def _fit_critic(self, cobs: Tensor, targets: Tensor, metrics: dict[str, list]) -> None:
        """``critic_epochs`` x ``critic_minibatches`` regression steps."""
        wm = self.config.wm
        n = targets.shape[0]
        if n == 0:
            return
        mb = max(1, n // wm.critic_minibatches)
        losses = []
        for _ in range(wm.critic_epochs):
            perm = torch.randperm(n, device=self.device)
            for j in range(wm.critic_minibatches):
                idx = perm[j * mb : (j + 1) * mb]
                v = self.critic(cobs[idx], bf16=wm.bf16)
                loss = (v - targets[idx]).square().mean()
                self.critic_optim.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.critic.net.parameters(), wm.critic_grad_clip_norm)
                self.critic_optim.step()
                losses.append(loss.detach())
        metrics["critic/loss"].append(torch.stack(losses).mean())
        metrics["critic/target_mean"].append(targets.mean())

    # ------------------------------------------------------------ loop
    def _flush(self, metrics: dict[str, list], stats: dict[str, list], it_time: float) -> dict[str, float]:
        out: dict[str, float] = {}
        for src in (metrics, stats):
            for k, vals in src.items():
                if not vals:
                    continue
                name = k if "/" in k else f"env/{k}"
                values = [torch.as_tensor(v, device=self.device, dtype=torch.float32) for v in vals]
                out[name] = float(torch.stack(values).mean())
            src.clear()
        out.update(self._wm_eval())
        out["train/lr_factor"] = self._lr_factor()
        out["perf/iteration_s"] = it_time
        if "env/env_reward" in out:
            out["train/step_reward_mean"] = out["env/env_reward"]
        count, length, reward = self._episode_sums.tolist()
        self._episode_sums.zero_()
        if count > 0:
            out["episode/length"] = length / count
            out["episode/return"] = reward / count
        return out

    def _env_log_metrics(self) -> dict[str, Any]:
        """The env's live scalars; ``_collect_step`` averages ``extras["log"]`` itself."""
        reader = self._env_attribute("live_log_metrics", None)
        if not callable(reader):
            return {}
        return {key: float(value) for key, value in reader().items()}

    def _checkpoint_dir(self) -> Path:
        return self.log_dir / self.config.logger.save_path

    def ensure_replay(self) -> SequenceReplay:
        """Allocate the per-env replay on first use (training, or an offline
        probe that collects with ``_collect_step``)."""
        if self.replay is None:
            wm, t = self.config.wm, self.task
            self.replay = SequenceReplay(self._num_envs, wm.replay_steps, t.state_dim, t.action_dim, t.context_dim, self.device)
            # Collection draws its sensor noise as a function of (seed, env,
            # replay row), so a real-window re-simulation sees the same noise.
            self._noise_seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
            self._env_ids = torch.arange(self._num_envs, device=self.device)
            self._episode_length = torch.zeros(self._num_envs, device=self.device)
            self._episode_return = torch.zeros(self._num_envs, device=self.device)
            self._episode_sums = torch.zeros(3, device=self.device)
        return self.replay

    @honours_train_time_budget
    def train(self) -> None:  # type: ignore[override]
        if self._inference_loaded:
            msg = "load_policy restores a policy for inference; resume training with load_model."
            raise RuntimeError(msg)
        wm = self.config.wm
        t = self.task
        self.ensure_replay()
        total_frames = int(self.config.collector.total_frames)
        self.log.info("WMBPTT task: %s", t.describe())
        obs, _ = self._isaac.reset()
        self._frames_processed = self._resume_frame_offset
        metrics: dict[str, list] = defaultdict(list)
        stats: dict[str, list] = defaultdict(list)
        meta = TrainingMetadata(total_iterations=0)
        started = False
        last_log = time.monotonic()
        iters_since_log = 0
        try:
            while True:
                self._step_acc += wm.env_steps_per_iter
                n_steps = int(self._step_acc)
                self._step_acc -= n_steps
                if not started:
                    n_steps = max(n_steps, wm.min_replay_steps)
                frames_before = self._frames_processed
                phase: dict[str, float] = {}
                with self._profile_iteration_phase(phase, "collect"):
                    for _ in range(n_steps):
                        obs = self._collect_step(obs, stats)
                if not started:
                    self._start_train_clock()
                    started = True
                self._set_lrs()
                refresh = getattr(t, "refresh", None)
                if callable(refresh):
                    refresh()  # e.g. curriculum-moved termination thresholds
                if self.replay.steps - self.replay.oldest > max(wm.wm_window, t.history) + 2:
                    with self._profile_iteration_phase(phase, "world_model"):
                        self._update_world_model(metrics)
                    if self._iteration >= wm.wm_warmup_iters:
                        with self._profile_iteration_phase(phase, "actor_critic"):
                            self._update_actor_critic(metrics)
                for name, sec in phase.items():
                    metrics[f"perf/{name}_s"].append(torch.tensor(sec))
                self._iteration += 1
                iters_since_log += 1
                meta.frames_processed = self._frames_processed
                stop = self._train_time_budget_reached() or (total_frames > 0 and self._frames_processed >= total_frames)
                if self._iteration % wm.log_interval_iters == 0 or stop:
                    now = time.monotonic()
                    flushed = self._flush(metrics, stats, (now - last_log) / max(1, iters_since_log))
                    last_log, iters_since_log = now, 0
                    self.log_training_metrics(flushed, frames=self._frames_processed, iteration=self._iteration)
                    self.log.info(
                        "it=%d frames=%d t=%.0fs wm=%.4f k8_root=%.1fmm k8_joint=%.2fdeg obj=%.3f alive_H=%.3f env_r=%.4f it_s=%.3f",
                        self._iteration, self._frames_processed, self.train_elapsed_seconds(),
                        flushed.get("wm/loss", float("nan")), flushed.get("wm_eval/k8_root_pos_mm", float("nan")),
                        flushed.get("wm_eval/k8_joint_pos_deg", float("nan")), flushed.get("actor/objective", float("nan")),
                        flushed.get("imagine/alive_H", float("nan")), flushed.get("env/env_reward", float("nan")),
                        flushed.get("perf/iteration_s", float("nan")),
                    )
                    if "real/pelvis_dev_mm" in flushed:
                        self.log.info(
                            "real windows: pelvis_dev=%.1fmm action_dev0=%.4f alive_H=%.3f reward=%.4f",
                            flushed["real/pelvis_dev_mm"], flushed.get("real/action_dev0", float("nan")),
                            flushed.get("real/alive_H", float("nan")),
                            flushed.get("real/reward_mean", float("nan")),
                        )
                    if "perf/collect_s" in flushed:
                        self.log.info(
                            "phases: collect=%.3fs world_model=%.3fs actor_critic=%.3fs",
                            flushed.get("perf/collect_s", float("nan")),
                            flushed.get("perf/world_model_s", float("nan")),
                            flushed.get("perf/actor_critic_s", float("nan")),
                        )
                if self._should_save_checkpoint(frames_processed=self._frames_processed, frames_in_iteration=self._frames_processed - frames_before):
                    self.save_model(self._checkpoint_dir(), step=self._frames_processed)
                if stop:
                    meta.stop_after_iteration = True
                    break
            if self._train_time_budget_reached():
                self._finish_train_time_budget(meta)
            else:
                self.save_model(self._checkpoint_dir(), step=self._frames_processed)
        except KeyboardInterrupt:
            self.save_model(self._checkpoint_dir(), step=self._frames_processed)
            raise

    # ------------------------------------------------------------ checkpoint
    def policy_meta(self) -> dict[str, Any]:
        t = self.task
        return {
            "arch": ARCH_TAG,
            "in_keys": [list(k) for k in t.policy_in_keys],
            "policy_obs_dim": t.policy_obs_dim,
            "action_dim": t.action_dim,
            "actor_hidden": list(self.config.wm.actor_hidden),
            "actor_activation": str(self.config.wm.actor_activation).lower(),
            "task_model": str(self.config.wm.task_model),
            "sensor_noise": bool(self.config.wm.sensor_noise),
            "bf16": bool(self.config.wm.bf16),
            "goal_binding": _plain(t.goal_binding()) if hasattr(t, "goal_binding") else None,
        }

    def save_model(self, path: str | Path | None = None, step: int | None = None) -> None:
        base = Path(path) if path else self._checkpoint_dir()
        frames = int(step if step is not None else self._frames_processed)
        target = base if base.suffix == ".pt" else base / f"model_step_{frames}.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "policy_state_dict": self.collector_policy.state_dict(),
            "policy_meta": self.policy_meta(),
            "dynamics_state_dict": self.dynamics.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "optimizers": {
                "wm": self.wm_optim.state_dict(),
                "actor": self.actor_optim.state_dict(),
                "critic": self.critic_optim.state_dict(),
            },
            "wm_config": _plain(dataclasses.asdict(self.config.wm)),
            "task": _plain(self.task.describe()),
            "iteration": int(self._iteration),
            "cumulative_env_frames": frames,
            "cumulative_train_seconds": self.train_elapsed_seconds(),
        }
        if self._rebar is not None:
            payload["rebar"] = self._rebar.state_dict()
        if self.actor_avg is not None:
            payload["policy_live_state_dict"] = self.policy.state_dict()
            payload["actor_ema_updates"] = self.actor_avg.updates
        tmp = target.with_name(target.name + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(target)

    def save(self, path: str | Path) -> None:
        self.save_model(path)

    def _read_checkpoint(self, path: str, map_location: torch.device | str) -> dict[str, Any]:
        """A checkpoint whose policy interface matches this agent."""
        data = torch.load(path, map_location=map_location, weights_only=False)
        meta = data.get("policy_meta") or {}
        if meta.get("arch") != ARCH_TAG:
            msg = f"{path} is not a {ARCH_TAG} checkpoint (arch={meta.get('arch')!r})."
            raise ValueError(msg)
        mine = self.policy_meta()
        keys = ["in_keys", "policy_obs_dim", "action_dim", "actor_hidden"]
        # Checkpoints written before the activation field were ELU.
        meta.setdefault("actor_activation", "elu")
        keys.append("actor_activation")
        # The goal input is bound by what produces it (e.g. the frozen skill
        # encoder's weights), not by its width; older checkpoints predate it.
        if meta.get("goal_binding") is not None:
            keys.append("goal_binding")
        for key in keys:
            if meta.get(key) != mine[key]:
                msg = f"checkpoint policy_meta[{key!r}]={meta.get(key)!r} != agent {mine[key]!r}"
                raise ValueError(msg)
        return data

    def load_policy(self, path: str) -> None:
        """Inference load: ``policy_state_dict`` and ``policy_meta``, nothing else.

        The checkpoint's world model, critic and optimizers do not matter here,
        so a run with other sizes or optimizers scores on an agent built from
        the task's default config. How the policy acts comes from the
        checkpoint, not from that default config. An agent loaded this way
        refuses to train: use ``load_model`` to resume.
        """
        data = self._read_checkpoint(path, "cpu")
        meta = data["policy_meta"]
        assert self.policy is not None
        self.policy.load_state_dict(data["policy_state_dict"], strict=True)
        # Checkpoints written before ``policy_meta["bf16"]`` keep it in ``wm_config``.
        self.policy_net.bf16 = bool(meta.get("bf16", (data.get("wm_config") or {}).get("bf16", self.policy_net.bf16)))
        if "sensor_noise" in meta and hasattr(self.task, "sensor_noise"):
            self.task.sensor_noise = bool(meta["sensor_noise"])
        if self.actor_avg is not None:
            self.actor_avg.updates = 0  # ``self.policy`` now holds what is deployed
        self._inference_loaded = True

    def load_model(self, path: str) -> None:
        """Resume load: every component, strictly. A checkpoint from another
        configuration or without the training state raises (see ``load_policy``)."""
        data = self._read_checkpoint(path, self.device)
        wm = self.config.wm
        required = ["dynamics_state_dict", "critic_state_dict", "optimizers"]
        if wm.actor_ema > 0:
            required += ["policy_live_state_dict", "actor_ema_updates"]
        missing = [key for key in required if data.get(key) is None]
        if missing:
            msg = f"{path} has no {missing}: it cannot resume training. Use load_policy to score it."
            raise ValueError(msg)
        saved = data.get("wm_config") or {}
        for name, before_the_field in RESUME_MATCH.items():
            if saved.get(name, before_the_field) != getattr(wm, name):
                msg = f"wm.{name}={getattr(wm, name)!r} must match the checkpoint's {saved.get(name, before_the_field)!r} to resume."
                raise ValueError(msg)
        assert self.policy is not None
        if self.actor_avg is not None:
            self.policy.load_state_dict(data["policy_live_state_dict"], strict=True)
            self.policy_avg.load_state_dict(data["policy_state_dict"], strict=True)  # type: ignore[union-attr]
            self.actor_avg.updates = int(data["actor_ema_updates"])
        else:
            self.policy.load_state_dict(data["policy_state_dict"], strict=True)
        self.dynamics.load_state_dict(data["dynamics_state_dict"])
        self.critic.load_state_dict(data["critic_state_dict"])
        for name, opt in (("wm", self.wm_optim), ("actor", self.actor_optim), ("critic", self.critic_optim)):
            opt.load_state_dict(data["optimizers"][name])
        if self._rebar is not None and "rebar" in data:
            self._rebar.load_state_dict(data["rebar"])
        self._iteration = int(data.get("iteration", 0))
        frames = data.get("cumulative_env_frames")
        if frames is not None:
            self._resume_frame_offset = int(frames)
        self._restore_train_clock(data)
