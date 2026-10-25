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
   c_{t+1} = c_t (1 - d_{t+1}), through the model's Jacobians.
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

from rlopt.agent.wm_bptt.config import WMBPTTRLOptConfig
from rlopt.agent.wm_bptt.networks import Actor, Critic, DynamicsModel
from rlopt.agent.wm_bptt.replay import SequenceReplay
from rlopt.agent.wm_bptt.task import WorldModelTask, load_task
from rlopt.base_class import BaseAlgorithm, TrainingMetadata, honours_train_time_budget

ARCH_TAG = "wmbptt-v1"


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
        super().__init__(env=env, config=config, logger=logger, **kwargs)

    # ------------------------------------------------------------ construction
    def _construct_policy(self) -> TensorDictModule:
        wm = self.config.wm
        t = self.task
        self.dynamics = DynamicsModel(t.wm_input_dim, t.increment_dim, wm.wm_hidden).to(self.device)
        self.actor = Actor(
            t.policy_obs_dim, t.action_dim, wm.actor_hidden, wm.actor_output_scale, wm.actor_activation
        ).to(self.device)
        self.critic = Critic(t.critic_obs_dim, wm.critic_hidden, wm.critic_activation).to(self.device)
        self.policy_net = WMPolicy(t, self.actor, wm.bf16)
        return TensorDictModule(self.policy_net, in_keys=list(t.policy_in_keys), out_keys=["action"])

    def _construct_feature_extractor(self) -> TensorDictModule:
        msg = "WMBPTT has no feature extractor."
        raise NotImplementedError(msg)

    def _construct_actor_critic(self) -> TensorDictModule:
        assert self.policy is not None
        return self.policy

    @property
    def collector_policy(self) -> TensorDictModule:
        assert self.policy is not None
        return self.policy

    def _construct_data_buffer(self) -> None:  # type: ignore[override]
        return None  # allocated lazily in train(): evaluation never needs it

    def _construct_collector(self, env: TransformedEnv, policy: TensorDictModule) -> None:  # type: ignore[override]
        del env, policy  # the simulator is stepped directly

    def _construct_loss_module(self) -> nn.Module:
        return nn.ModuleDict({"dynamics": self.dynamics, "critic": self.critic})

    def _configure_optimizers(self):  # type: ignore[override]
        wm = self.config.wm
        self.wm_optim = torch.optim.AdamW(self.dynamics.net.parameters(), lr=wm.wm_lr, weight_decay=wm.wm_weight_decay)
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
        with torch.no_grad():
            return self.actor(x, bf16=self.config.wm.bf16)

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
        return self.task.policy_obs(
            rows["hist_states"], rows["hist_actions"], rows["context"], rows["rank"], rows["frame"]
        )

    @torch.no_grad()
    def _collect_step(self, obs: dict[str, Any], stats: dict[str, list]) -> dict[str, Any]:
        wm = self.config.wm
        rows = self.task.read_obs(obs[self.task.obs_group])
        feat = self._policy_features(rows)
        self.actor.obs_norm.update(feat)
        action = self.actor(feat, bf16=wm.bf16)
        if wm.action_noise_std > 0:
            action = action + wm.action_noise_std * torch.randn_like(action)
        action = action.clamp(-wm.action_clip, wm.action_clip)
        assert self.replay is not None
        self.replay.add(
            rows["state"], action, rows["hist_actions"][:, -1], rows["context"],
            rows["rank"], rows["frame"], rows["episode_step"], rows["push"],
        )
        obs, reward, _, _, extras = self._isaac.step(action)
        stats["env_reward"].append(reward.mean())
        log = extras.get("log") if isinstance(extras, dict) else None
        if log:
            for k, value in log.items():
                if isinstance(value, torch.Tensor):
                    value = value.float().mean()  # noqa: PLW2901
                stats[k].append(torch.as_tensor(value, device=self.device, dtype=torch.float32))
        self._frames_processed += self._num_envs
        return obs

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
            self.wm_optim.zero_grad(set_to_none=True)
            loss.backward()
            gn = nn.utils.clip_grad_norm_(self.dynamics.net.parameters(), wm.wm_grad_clip_norm)
            if torch.isfinite(gn):
                self.wm_optim.step()
            metrics["wm/loss"].append(loss.detach())
            metrics["wm/grad_norm"].append(gn.detach())
            metrics["wm/valid_window_frac"].append(mask[:, -1].mean())
        if self.dynamics.out_norm.count >= wm.norm_freeze_samples:
            self.dynamics.in_norm.frozen.fill_(True)
            self.dynamics.out_norm.frozen.fill_(True)

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
        self, state: Tensor, prop: Tensor, prev_action: Tensor, ctx: Tensor, rank: Tensor, frame: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """One imagined control step: act, predict, score, extend the history."""
        t, wm = self.task, self.config.wm
        obs = t.policy_obs_from_parts(prop, state, rank, frame)
        action = self.actor(obs, bf16=wm.bf16)
        inc = self.dynamics(t.wm_input(state, action, ctx), bf16=wm.bf16)
        nxt = t.integrate(state, inc)
        r, d = t.reward_and_termination(nxt, action, prev_action, ctx, rank, frame)
        new_prop = t.proprio(nxt, action, ctx, t.sensor_noise if hasattr(t, "sensor_noise") else True)
        prop = torch.cat((prop[:, 1:], new_prop.unsqueeze(1)), 1)
        return action, nxt, r, d, prop

    def _step_fn(self):
        fn = getattr(self, "_compiled_step", None)
        if fn is None:
            fn = self._imagine_step
            if self.config.wm.compile:
                fn = torch.compile(self._imagine_step, dynamic=False)
            self._compiled_step = fn
        return fn

    def _imagine(self, batch: int, *, grad: bool) -> dict[str, Tensor]:
        wm = self.config.wm
        t = self.task
        assert self.replay is not None
        h = t.history
        lo = self.replay.oldest + h - 1
        hi = self.replay.steps - 1
        tau = torch.randint(lo, hi + 1, (batch,), device=self.device)
        env = torch.randint(0, self._num_envs, (batch,), device=self.device)
        start = self.replay.history(tau, env, h)
        ctx, rank, frame = start["context"], start["rank"], start["frame"]
        noise = getattr(t, "sensor_noise", True)
        prop = t.proprio(start["hist_states"], start["hist_actions"], ctx.unsqueeze(1), noise)
        prev_action = start["hist_actions"][:, -1]
        state = start["hist_states"][:, -1]
        if grad:
            # Every step then sees the same requires_grad pattern; otherwise the
            # compiled step recompiles between step 0 and the rest.
            state, prop, prev_action = (
                x.detach().contiguous().requires_grad_(True) for x in (state, prop, prev_action)
            )
        alive = torch.ones(batch, device=self.device)
        ret = torch.zeros(batch, device=self.device)
        states, rewards, dones, cont = [state], [], [], []
        step_fn = self._step_fn()
        actor_h = wm.actor_horizon if 0 < wm.actor_horizon < wm.horizon else wm.horizon
        actor_objective = None
        for step in range(wm.horizon):
            action, nxt, r, d, prop = step_fn(state, prop, prev_action, ctx, rank, frame)
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
            elif step + 1 == actor_h and actor_h < wm.horizon:
                # The actor's objective ends here (bootstrapped with V); the
                # remaining steps only feed the critic's targets, so the
                # gradient path is cut. Re-wrapping keeps the compiled step's
                # requires_grad pattern unchanged (no recompile).
                v_a = self.critic(t.critic_obs(state, ctx, rank, frame), bf16=wm.bf16)
                actor_objective = ret + (wm.gamma**actor_h) * alive * v_a if wm.critic_bootstrap else ret
                state, prop, prev_action = (
                    x.detach().contiguous().requires_grad_(True) for x in (state, prop, prev_action)
                )
        v_last = self.critic(t.critic_obs(state, ctx, rank, frame), bf16=wm.bf16)
        objective = ret + (wm.gamma**wm.horizon) * alive * v_last if wm.critic_bootstrap else ret
        if actor_objective is not None:
            objective = actor_objective
        return {
            "objective": objective,
            "states": torch.stack(states, 1),
            "rewards": torch.stack(rewards, 1),
            "dones": torch.stack(dones, 1),
            "cont": torch.stack(cont, 1),
            "alive_H": alive,
            "ctx": ctx,
            "rank": rank,
            "frame0": start["frame"],
        }

    def _update_actor_critic(self, metrics: dict[str, list]) -> None:
        wm = self.config.wm
        self.dynamics.requires_grad_(False)
        self.critic.requires_grad_(False)
        try:
            out = self._imagine(wm.imagine_batch, grad=True)
            loss = -out["objective"].mean()
            self.actor_optim.zero_grad(set_to_none=True)
            loss.backward()
        finally:
            self.dynamics.requires_grad_(True)
            self.critic.requires_grad_(True)
        params = list(self.actor.net.parameters())
        gn = nn.utils.clip_grad_norm_(params, wm.actor_grad_clip_norm)
        nn.utils.clip_grad_value_(params, wm.actor_grad_clip_value)
        if torch.isfinite(gn):
            self.actor_optim.step()
        metrics["actor/objective"].append(out["objective"].detach().mean())
        metrics["actor/grad_norm"].append(gn.detach())
        metrics["imagine/alive_H"].append(out["alive_H"].detach().mean())
        metrics["imagine/reward_mean"].append(out["rewards"].detach().mean())
        metrics["imagine/done_mean"].append(out["dones"].detach().mean())
        if self._iteration % max(1, wm.critic_update_every) == 0:
            self._update_critic(out, metrics)

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
        n = targets.shape[0]
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
        out.update(self._train_clock_metrics(self._frames_processed))
        out["train/iteration"] = float(self._iteration)
        out["train/lr_factor"] = self._lr_factor()
        out["perf/iteration_s"] = it_time
        return out

    def _checkpoint_dir(self) -> Path:
        return self.log_dir / self.config.logger.save_path

    @honours_train_time_budget
    def train(self) -> None:  # type: ignore[override]
        wm = self.config.wm
        t = self.task
        if self.replay is None:
            self.replay = SequenceReplay(self._num_envs, wm.replay_steps, t.state_dim, t.action_dim, t.context_dim, self.device)
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
                    self.log_metrics(flushed, step=self._frames_processed, log_python=False)
                    self.log.info(
                        "it=%d frames=%d t=%.0fs wm=%.4f k8_root=%.1fmm k8_joint=%.2fdeg obj=%.3f alive_H=%.3f env_r=%.4f it_s=%.3f",
                        self._iteration, self._frames_processed, self.train_elapsed_seconds(),
                        flushed.get("wm/loss", float("nan")), flushed.get("wm_eval/k8_root_pos_mm", float("nan")),
                        flushed.get("wm_eval/k8_joint_pos_deg", float("nan")), flushed.get("actor/objective", float("nan")),
                        flushed.get("imagine/alive_H", float("nan")), flushed.get("env/env_reward", float("nan")),
                        flushed.get("perf/iteration_s", float("nan")),
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
            "goal_binding": _plain(t.goal_binding()) if hasattr(t, "goal_binding") else None,
        }

    def save_model(self, path: str | Path | None = None, step: int | None = None) -> None:
        base = Path(path) if path else self._checkpoint_dir()
        frames = int(step if step is not None else self._frames_processed)
        target = base if base.suffix == ".pt" else base / f"model_step_{frames}.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "policy_state_dict": self.policy.state_dict() if self.policy is not None else None,
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
        tmp = target.with_name(target.name + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(target)

    def save(self, path: str | Path) -> None:
        self.save_model(path)

    def load_model(self, path: str) -> None:
        data = torch.load(path, map_location=self.device, weights_only=False)
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
        assert self.policy is not None
        self.policy.load_state_dict(data["policy_state_dict"], strict=True)
        if data.get("dynamics_state_dict") is not None:
            self.dynamics.load_state_dict(data["dynamics_state_dict"])
        if data.get("critic_state_dict") is not None:
            self.critic.load_state_dict(data["critic_state_dict"])
        opts = data.get("optimizers") or {}
        for name, opt in (("wm", self.wm_optim), ("actor", self.actor_optim), ("critic", self.critic_optim)):
            if name in opts:
                opt.load_state_dict(opts[name])
        self._iteration = int(data.get("iteration", 0))
        frames = data.get("cumulative_env_frames")
        if frames is not None:
            self._resume_frame_offset = int(frames)
        self._restore_train_clock(data)
