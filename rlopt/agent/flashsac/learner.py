"""FlashSAC learner: torch-only update, reward scaling, checkpointing, CUDA graphs.

Adapted from Holiday-Robot/FlashSAC (commit 87edc90).

MIT License. Copyright (c) 2026 Holiday Robotics.
See LICENSE-FlashSAC.txt in this directory for the full notice.

What the learner owns:

- the actor, the double categorical critic, the target critic and the temperature,
- fused capturable Adam with a device learning-rate tensor per network,
- the weight-norm projection after every optimizer step,
- return-based reward scaling (``observe_rewards`` at insert time),
- a complete ``state_dict`` (networks, BatchNorm buffers, optimizers, GradScaler,
  LR-schedule steps, reward statistics, update counter),
- optional ``torch.compile`` (mode ``default``) and two CUDA graphs.

Batch contract for ``update`` (all device tensors, ``B`` rows):

``obs``              actor observation at time t, ``(B, actor_obs_dim)``
``critic_obs``       critic observation at time t, ``(B, critic_obs_dim)``
``action``           executed action in tanh space, ``[-1, 1]``, ``(B, act_dim)``
``reward``           n-step discounted sum of RAW rewards, ``(B,)``
``next_obs``         actor observation at the end of the window
``next_critic_obs``  critic observation at the end of the window
``terminated_n``     1.0 if a true termination ended the window, else 0.0, ``(B,)``
``discount_n``       ``gamma ** n_eff`` per row, ``(B,)``

The learner divides ``reward`` by its current return scale inside ``update``.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
import torch._dynamo as dynamo
from torch import Tensor, nn

from rlopt.agent.flashsac.layers import collect_normalizable, safe_tanh_log_det_jacobian
from rlopt.agent.flashsac.networks import (
    FlashSACActor,
    FlashSACDoubleCritic,
    FlashSACTemperature,
)
from rlopt.agent.flashsac.noise import publish_actor_state  # noqa: F401  (re-export)

BATCH_KEYS = (
    "obs",
    "critic_obs",
    "action",
    "reward",
    "next_obs",
    "next_critic_obs",
    "terminated_n",
    "discount_n",
)
_AMP_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": None}


@dataclass
class FlashSACLearnerConfig:
    """Hyper-parameters of the FlashSAC learner. Defaults follow the reference recipe."""

    actor_dim: int = 128
    actor_blocks: int = 2
    critic_dim: int = 256
    critic_blocks: int = 2
    num_atoms: int = 101
    v_min: float = -5.0
    v_max: float = 5.0
    gamma: float = 0.99
    n_step: int = 3  # informational: the replay applies the n-step window
    tau: float = 0.01
    lr: float = 3e-4
    lr_final: float = 1.5e-4
    lr_decay_steps: int | None = None  # None: constant learning rate
    lr_clock: str = (
        "own"  # "own": actor and temperature step on actor updates only (reference)
    )
    batch_size: int = 2048
    actor_update_period: int = 2
    temp_init: float = 0.01
    target_entropy_sigma: float = 0.15
    action_bound: float | Sequence[float] | None = None
    weight_norm: bool = True
    amp_dtype: str = "fp16"
    compile_mode: str = "none"
    use_cuda_graph: bool = False
    seed: int = 0
    reward_g_max: float = 5.0


def target_entropy(act_dim: int, sigma: float) -> float:
    """Unified entropy target in tanh space: ``0.5 * |A| * ln(2 pi e sigma^2)``."""
    return 0.5 * act_dim * math.log(2.0 * math.pi * math.e * sigma**2)


def c51_project(
    next_log_probs: Tensor,
    reward: Tensor,
    terminated: Tensor,
    discount: Tensor,
    entropy_term: Tensor,
    v_min: float,
    v_max: float,
) -> Tensor:
    """Project the soft Bellman backup onto the fixed C51 support.

    ``target z' = r + discount * (z - alpha * log_pi) * (1 - terminated)`` is
    clamped to ``[v_min, v_max]`` and its mass is split between the two
    neighbouring atoms. Row offsets in the flat scatter use an int64 ``arange``.
    A float32 offset would round above about 166k rows and write mass into the
    wrong rows.

    Shapes: ``next_log_probs (B, atoms)``; the rest ``(B,)``. Returns ``(B, atoms)``.
    """
    batch, atoms = next_log_probs.shape
    probs = next_log_probs.float().exp()
    support = torch.linspace(
        v_min, v_max, atoms, device=probs.device, dtype=probs.dtype
    ).view(1, -1)
    dz = (v_max - v_min) / (atoms - 1)
    cont = 1.0 - terminated.reshape(-1, 1).float()
    tz = (
        reward.reshape(-1, 1).float()
        + discount.reshape(-1, 1).float()
        * (support - entropy_term.reshape(-1, 1).float())
        * cont
    )
    pos = (tz.clamp(v_min, v_max) - v_min) / dz
    lower = pos.floor().long().clamp_(0, atoms - 1)
    upper = (lower + 1).clamp_(max=atoms - 1)
    frac = pos - lower.to(pos.dtype)
    offset = (
        torch.arange(batch, device=probs.device, dtype=torch.int64).unsqueeze(1) * atoms
    )
    out = torch.zeros(batch * atoms, dtype=probs.dtype, device=probs.device)
    out.index_add_(0, (lower + offset).reshape(-1), (probs * (1.0 - frac)).reshape(-1))
    out.index_add_(0, (upper + offset).reshape(-1), (probs * frac).reshape(-1))
    return out.view(batch, atoms)


def _select_min_q_log_probs(next_qs: Tensor, next_log_probs: Tensor) -> Tensor:
    """Pick the log-probs of the head with the smaller expected Q. ``(2, B)``, ``(2, B, A)`` -> ``(B, A)``."""
    idx = next_qs.argmin(dim=0)
    atoms = next_log_probs.shape[-1]
    return torch.gather(next_log_probs, 0, idx[None, :, None].expand(1, -1, atoms))[0]


class ReturnScale(nn.Module):
    """Running reward scale from per-env discounted returns.

    ``G <- gamma * (1 - done) * G + r`` per env, with running mean and variance
    of ``G`` taken across envs and a never-decreasing ``max|G|``. The divisor is
    ``max(sqrt(Var_G + 1e-8), max|G| / g_max)``. Statistics are float64 scalars.
    """

    def __init__(self, gamma: float, g_max: float = 5.0, eps: float = 1e-8) -> None:
        super().__init__()
        self.gamma = gamma
        self.g_max = g_max
        self.eps = eps
        self.rms_eps = 1e-4
        self.register_buffer("mean", torch.zeros((), dtype=torch.float64))
        self.register_buffer("var", torch.ones((), dtype=torch.float64))
        self.register_buffer("count", torch.zeros((), dtype=torch.float64))
        self.register_buffer("max_abs", torch.zeros((), dtype=torch.float64))
        self.ret: Tensor | None = None  # per-env return, allocated on first observe

    @torch.no_grad()
    def observe(
        self, reward: Tensor, terminated: Tensor, truncated: Tensor | None = None
    ) -> None:
        reward = reward.reshape(-1).float()
        done = terminated.reshape(-1).float()
        if truncated is not None:
            done = torch.maximum(done, truncated.reshape(-1).float())
        if self.ret is None:
            self.ret = torch.zeros_like(reward)
        new_ret = self.gamma * (1.0 - done) * self.ret + reward
        self.ret.copy_(new_ret)
        r64 = new_ret.double()
        self.max_abs.copy_(torch.maximum(self.max_abs, r64.abs().max()))
        n = float(r64.numel())
        s_mean = r64.mean()
        s_var = r64.var(unbiased=False)
        delta = s_mean - self.mean
        total = self.count + n
        ratio = n / total
        m2 = (
            self.var * (self.count + self.rms_eps)
            + s_var * n
            + delta.square() * self.count * ratio
        )
        self.mean.copy_(self.mean + delta * ratio)
        self.var.copy_(m2 / total)
        self.count.copy_(total)

    def scale(self) -> Tensor:
        """Current divisor as a float32 device scalar."""
        std = torch.sqrt(self.var + self.eps)
        return torch.maximum(std, self.max_abs / self.g_max).float()

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Tensor]:  # type: ignore[override]
        state = {k: v.clone() for k, v in super().state_dict(*args, **kwargs).items()}
        if self.ret is not None:
            state["ret"] = self.ret.clone()
        return state

    def load_state_dict(self, state: dict[str, Tensor], strict: bool = True):  # noqa: ARG002  # type: ignore[override]
        for name in ("mean", "var", "count", "max_abs"):
            getattr(self, name).copy_(state[name])
        if "ret" in state:
            if self.ret is not None and self.ret.shape == state["ret"].shape:
                self.ret.copy_(state["ret"])
            else:
                self.ret = state["ret"].to(self.mean.device).clone()


class _LRSchedule:
    """Device-tensor learning rate stepped on the host, outside any CUDA graph."""

    def __init__(self, tensor: Tensor, fn: Callable[[int], float]) -> None:
        self.tensor = tensor
        self.fn = fn
        self.count = 0
        self.tensor.fill_(fn(0))

    def step(self) -> None:
        self.count += 1
        self.tensor.fill_(self.fn(self.count))

    def set_count(self, count: int) -> None:
        self.count = count
        self.tensor.fill_(self.fn(count))


def _cosine_schedule(
    lr: float, lr_final: float, decay_steps: int | None
) -> Callable[[int], float]:
    def fn(step: int) -> float:
        if decay_steps is None:
            return lr
        if step >= decay_steps:
            return lr_final
        progress = step / decay_steps
        return lr_final + (lr - lr_final) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return fn


class FlashSACLearner(nn.Module):
    """FlashSAC networks, optimizers and update step."""

    def __init__(
        self,
        cfg: FlashSACLearnerConfig,
        actor_obs_dim: int,
        critic_obs_dim: int,
        act_dim: int,
        device: torch.device | str,
    ) -> None:
        super().__init__()
        if cfg.amp_dtype not in _AMP_DTYPES:
            msg = f"amp_dtype must be one of {list(_AMP_DTYPES)}"
            raise ValueError(msg)
        if cfg.compile_mode not in ("none", "default"):
            msg = "compile_mode must be 'none' or 'default'"
            raise ValueError(msg)
        if cfg.lr_clock not in ("own", "shared"):
            msg = "lr_clock must be 'own' or 'shared'"
            raise ValueError(msg)
        self.cfg = cfg
        self.device_ = torch.device(device)
        self.act_dim = act_dim
        self.actor_obs_dim = actor_obs_dim
        self.critic_obs_dim = critic_obs_dim
        on_cuda = self.device_.type == "cuda"
        amp_dtype = _AMP_DTYPES[cfg.amp_dtype]
        self._amp_dtype = amp_dtype if on_cuda else None
        self.target_entropy = target_entropy(act_dim, cfg.target_entropy_sigma)

        # Build on CPU under a private RNG fork so the global RNG is not touched.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(cfg.seed)
            self.actor = FlashSACActor(
                cfg.actor_blocks, actor_obs_dim, cfg.actor_dim, act_dim
            )
            self.critic = self._make_critic(critic_obs_dim, act_dim)
            self.target_critic = self._make_critic(critic_obs_dim, act_dim)
            self.temperature = FlashSACTemperature(cfg.temp_init)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_critic.requires_grad_(False)

        bound = (
            torch.ones(act_dim)
            if cfg.action_bound is None
            else torch.as_tensor(cfg.action_bound, dtype=torch.float32)
            .expand(act_dim)
            .clone()
        )
        self.register_buffer("action_bound", bound)
        self.return_scale = ReturnScale(cfg.gamma, cfg.reward_g_max)
        self.to(self.device_)

        # Optimizers: fused capturable Adam with a device LR tensor (CPU: plain Adam).
        lr_fn = _cosine_schedule(cfg.lr, cfg.lr_final, cfg.lr_decay_steps)
        self.optimizers: dict[str, torch.optim.Optimizer] = {}
        self.schedules: dict[str, _LRSchedule] = {}
        for name, module in (
            ("actor", self.actor),
            ("critic", self.critic),
            ("temperature", self.temperature),
        ):
            lr_t = torch.tensor(cfg.lr, device=self.device_, dtype=torch.float32)
            if on_cuda:
                opt = torch.optim.Adam(
                    module.parameters(), lr=lr_t, fused=True, capturable=True
                )
            else:
                opt = torch.optim.Adam(module.parameters(), lr=lr_t, foreach=False)
            self.optimizers[name] = opt
            self.schedules[name] = _LRSchedule(lr_t, lr_fn)

        self.scaler = (
            torch.amp.GradScaler("cuda")
            if (on_cuda and cfg.amp_dtype == "fp16")
            else None
        )
        self.update_count = 0
        self._graphs: dict[bool, torch.cuda.CUDAGraph] | None = None
        self._static_batch: dict[str, Tensor] | None = None
        self._static_info: dict[bool, dict[str, Tensor]] = {}
        self._build_callables()
        if cfg.weight_norm:
            with torch.no_grad():
                self._norm_actor()
                self._norm_critic()
                self._norm_critic_target()

    # ------------------------------------------------------------------ build

    def _make_critic(self, critic_obs_dim: int, act_dim: int) -> FlashSACDoubleCritic:
        c = self.cfg
        return FlashSACDoubleCritic(
            c.critic_blocks,
            critic_obs_dim + act_dim,
            c.critic_dim,
            c.num_atoms,
            c.v_min,
            c.v_max,
        )

    def _build_callables(self) -> None:
        """Wrap networks and helpers with torch.compile when requested.

        The compiled wrappers live in a plain namespace, not as submodules, so
        they never appear in ``state_dict``.
        """
        use_compile = self.cfg.compile_mode == "default"
        if use_compile:
            dynamo.config.cache_size_limit = max(dynamo.config.cache_size_limit, 64)

        def comp(fn: Callable[..., Any]) -> Callable[..., Any]:
            return (
                torch.compile(fn, mode="default", dynamic=False) if use_compile else fn
            )

        def make_norm(module: nn.Module) -> Callable[[], None]:
            mods = collect_normalizable(module)

            def fn() -> None:
                for m in mods:
                    m.normalize_parameters()

            return fn

        tgt = list(self.target_critic.parameters())
        src = list(self.critic.parameters())
        tau = self.cfg.tau

        def ema() -> None:
            torch._foreach_lerp_(tgt, src, tau)

        self._fn = SimpleNamespace(
            actor=comp(self.actor),
            critic=comp(self.critic),
            target=comp(self.target_critic),
            norm_actor=comp(make_norm(self.actor)),
            norm_critic=comp(make_norm(self.critic)),
            norm_target=comp(make_norm(self.target_critic)),
            ema=comp(ema),
            select_min_q=comp(_select_min_q_log_probs),
            project=comp(c51_project),
        )

    @torch.no_grad()
    def _norm_actor(self) -> None:
        self._fn.norm_actor()

    @torch.no_grad()
    def _norm_critic(self) -> None:
        self._fn.norm_critic()

    @torch.no_grad()
    def _norm_critic_target(self) -> None:
        self._fn.norm_target()

    def num_parameters(self) -> dict[str, int]:
        def n(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters())

        return {
            "actor": n(self.actor),
            "critic": n(self.critic),
            "target_critic": n(self.target_critic),
            "temperature": n(self.temperature),
        }

    # -------------------------------------------------------------- inference

    @torch.no_grad()
    def act(
        self,
        obs: Tensor,
        noise: Tensor | None = None,
        deterministic: bool = False,
        return_log_prob: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Return the action in env units, ``bound * tanh(mean + std * noise)``.

        BatchNorm uses running statistics. ``deterministic=True`` uses
        ``tanh(mean)``. The optional log-prob is that of the tanh-space action
        (no bound Jacobian, same as the training entropy).
        """
        mean, std = self.actor.mean_and_std(obs, training=False)
        if deterministic:
            noise = torch.zeros_like(mean)
        elif noise is None:
            noise = torch.randn_like(mean)
        raw = mean + std * noise
        action = self.action_bound * torch.tanh(raw)
        if not return_log_prob:
            return action
        log_prob = (
            -0.5 * noise.square()
            - torch.log(std)
            - 0.5 * math.log(2.0 * math.pi)
            - safe_tanh_log_det_jacobian(raw)
        ).sum(-1)
        return action, log_prob

    @torch.no_grad()
    def observe_rewards(
        self, reward: Tensor, terminated: Tensor, truncated: Tensor | None = None
    ) -> None:
        """Update per-env return statistics. Call at insert time, once per env step."""
        self.return_scale.observe(reward, terminated, truncated)

    # ----------------------------------------------------------------- update

    def _amp(self):
        return torch.autocast(
            device_type=self.device_.type,
            dtype=self._amp_dtype if self._amp_dtype is not None else torch.bfloat16,
            enabled=self._amp_dtype is not None,
            cache_enabled=False,
        )

    def _backward_step(self, loss: Tensor, opt: torch.optim.Optimizer) -> None:
        opt.zero_grad(set_to_none=True)
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.step(opt)
            self.scaler.update()
        else:
            loss.backward()
            opt.step()

    def _update_actor(self, b: dict[str, Tensor]) -> dict[str, Tensor]:
        with self._amp():
            obs_all = torch.cat([b["obs"], b["next_obs"]], dim=0)
            actions_all, log_probs_all = self._fn.actor(obs_all, True)
            actions = torch.chunk(actions_all, 2, dim=0)[0]
            log_probs = torch.chunk(log_probs_all, 2, dim=0)[0]
            self.critic.requires_grad_(False)
            qs, _ = self._fn.critic(b["critic_obs"], actions, False)
            self.critic.requires_grad_(True)
            q = torch.minimum(qs[0], qs[1])
            alpha = self.temperature().detach()
            actor_loss = (log_probs * alpha - q).mean()
            entropy = -log_probs.mean()
        self._backward_step(actor_loss, self.optimizers["actor"])
        if self.cfg.weight_norm:
            self._norm_actor()
        return {"actor/loss": actor_loss.detach(), "actor/entropy": entropy.detach()}

    def _update_temperature(self, entropy: Tensor) -> dict[str, Tensor]:
        value = self.temperature().clone()
        loss = value * (entropy.detach() - self.target_entropy).mean()
        opt = self.optimizers["temperature"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return {"temperature/value": value.detach(), "temperature/loss": loss.detach()}

    def _update_critic(self, b: dict[str, Tensor], reward: Tensor) -> dict[str, Tensor]:
        cfg = self.cfg
        with self._amp():
            with torch.no_grad():
                next_a, next_logp = self._fn.actor(b["next_obs"], False)
                next_a = next_a.clone()
                entropy_term = self.temperature() * next_logp.clone()
                obs_all = torch.cat([b["critic_obs"], b["next_critic_obs"]], dim=0)
                act_all = torch.cat([b["action"], next_a], dim=0)
                qs_all, lp_all = self._fn.target(obs_all, act_all, True)
                next_qs = qs_all.chunk(2, dim=1)[1]
                next_lp = self._fn.select_min_q(next_qs, lp_all.chunk(2, dim=1)[1])
                target_probs = self._fn.project(
                    next_lp,
                    reward,
                    b["terminated_n"],
                    b["discount_n"],
                    entropy_term,
                    cfg.v_min,
                    cfg.v_max,
                )
                max_bonus = entropy_term.max()
            pred_qs, pred_lp_all = self._fn.critic(obs_all, act_all, True)
            pred_lp = torch.chunk(pred_lp_all, 2, dim=1)[0]
            ce = -(target_probs.unsqueeze(0) * pred_lp).sum(dim=-1)
            critic_loss = ce.mean()
        self._backward_step(critic_loss, self.optimizers["critic"])
        if cfg.weight_norm:
            self._norm_critic()
        return {
            "critic/loss": critic_loss.detach(),
            "critic/max_entropy_bonus": max_bonus,
            "critic/q_mean": pred_qs.detach().float().mean(),
        }

    def _update_body(
        self, batch: dict[str, Tensor], do_actor: bool
    ) -> dict[str, Tensor]:
        """One update without host work: this is what CUDA graphs capture."""
        reward = batch["reward"].reshape(-1) / self.return_scale.scale()
        info: dict[str, Tensor] = {"reward/scale": self.return_scale.scale()}
        if do_actor:
            a = self._update_actor(batch)
            info.update(a)
            info.update(self._update_temperature(a["actor/entropy"]))
        info.update(self._update_critic(batch, reward))
        with torch.no_grad():
            self._fn.ema()
        return info

    def _do_actor(self) -> bool:
        return self.update_count % self.cfg.actor_update_period == 0

    def _advance(self, did_actor: bool) -> None:
        if did_actor or self.cfg.lr_clock == "shared":
            self.schedules["actor"].step()
            self.schedules["temperature"].step()
        self.schedules["critic"].step()
        self.update_count += 1

    @staticmethod
    def _check_batch(batch: dict[str, Tensor]) -> None:
        missing = [k for k in BATCH_KEYS if k not in batch]
        if missing:
            msg = f"batch lacks keys: {missing}"
            raise KeyError(msg)

    def update(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Run one update (eager or compiled). Returns device scalars, no host sync."""
        self._check_batch(batch)
        do_actor = self._do_actor()
        info = self._update_body(batch, do_actor)
        self._advance(do_actor)
        return info

    # ------------------------------------------------------------ CUDA graphs

    def capture_graphs(self, example_batch: dict[str, Tensor], warmup: int = 4) -> None:
        """Capture the update as two CUDA graphs (with and without the actor step).

        Runs ``warmup`` eager updates on a side stream first (lazy optimizer and
        scaler state, compile). The learner state is restored afterwards, so
        capture does not train. The batch shape is fixed by ``example_batch``.
        """
        if self.device_.type != "cuda":
            msg = "CUDA graphs need a CUDA device"
            raise RuntimeError(msg)
        self._check_batch(example_batch)
        self._static_batch = {k: example_batch[k].detach().clone() for k in BATCH_KEYS}
        snap = self._snapshot_training_state()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for i in range(max(warmup, 2)):
                self._update_body(self._static_batch, i % 2 == 0)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graphs: dict[bool, torch.cuda.CUDAGraph] = {}
        infos: dict[bool, dict[str, Tensor]] = {}
        kinds = (True, False) if self.cfg.actor_update_period > 1 else (True,)
        for do_actor in kinds:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                infos[do_actor] = self._update_body(self._static_batch, do_actor)
            graphs[do_actor] = g
        torch.cuda.synchronize()
        self._graphs, self._static_info = graphs, infos
        self._restore_training_state(snap)

    def step_graph(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Replay the captured update on ``batch``. Same semantics as ``update``.

        The returned tensors are static graph outputs: they are overwritten by
        the next replay of the same graph.
        """
        if self._graphs is None or self._static_batch is None:
            msg = "call capture_graphs(example_batch) first"
            raise RuntimeError(msg)
        self._check_batch(batch)
        do_actor = self._do_actor()
        for k, buf in self._static_batch.items():
            buf.copy_(batch[k])
        self._graphs[do_actor].replay()
        self._advance(do_actor)
        info = dict(self._static_info[True])
        if not do_actor:
            info.update(self._static_info[False])
        return info

    def _snapshot_training_state(self) -> dict[str, Any]:
        opt_state = {}
        for name, opt in self.optimizers.items():
            opt_state[name] = {
                p: {k: v.clone() for k, v in st.items() if torch.is_tensor(v)}
                for p, st in opt.state.items()
            }
        return {
            "modules": {
                n: copy.deepcopy(getattr(self, n).state_dict())
                for n in ("actor", "critic", "target_critic", "temperature")
            },
            "opt": opt_state,
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
        }

    @torch.no_grad()
    def _restore_training_state(self, snap: dict[str, Any]) -> None:
        for n, sd in snap["modules"].items():
            getattr(self, n).load_state_dict(sd)
        for name, opt in self.optimizers.items():
            for p, st in opt.state.items():
                saved = snap["opt"][name].get(p)
                for k, v in st.items():
                    if not torch.is_tensor(v):
                        continue
                    if saved is not None and k in saved:
                        v.copy_(saved[k])
                    else:
                        v.zero_()
        if self.scaler is not None and snap["scaler"] is not None:
            self.scaler.load_state_dict(snap["scaler"])

    # ------------------------------------------------------------- checkpoint

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:  # noqa: ARG002  # type: ignore[override]
        """Return everything needed to resume training exactly.

        Networks are saved with their BatchNorm running buffers. Optimizer
        states, the GradScaler, LR-schedule steps, reward statistics and the
        update counter are included.
        """
        return {
            "version": 1,
            "actor": copy.deepcopy(self.actor.state_dict()),
            "critic": copy.deepcopy(self.critic.state_dict()),
            "target_critic": copy.deepcopy(self.target_critic.state_dict()),
            "temperature": copy.deepcopy(self.temperature.state_dict()),
            "action_bound": self.action_bound.clone(),
            "optimizers": {
                n: copy.deepcopy(o.state_dict()) for n, o in self.optimizers.items()
            },
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
            "lr_steps": {n: s.count for n, s in self.schedules.items()},
            "update_count": self.update_count,
            "reward_scale": self.return_scale.state_dict(),
        }

    @torch.no_grad()
    def load_state_dict(  # type: ignore[override]
        self,
        state: dict[str, Any],
        strict: bool = True,  # noqa: ARG002
        assign: bool = False,  # noqa: ARG002
    ):
        """Restore a state from ``state_dict``. Tensors are copied in place.

        In-place loading keeps CUDA-graph addresses valid, so captured graphs
        stay usable after a load.
        """
        for n in ("actor", "critic", "target_critic", "temperature"):
            getattr(self, n).load_state_dict(state[n], strict=True)
        self.action_bound.copy_(state["action_bound"])
        for name, opt in self.optimizers.items():
            self._load_optimizer(opt, state["optimizers"][name])
        if self.scaler is not None and state["scaler"] is not None:
            self.scaler.load_state_dict(state["scaler"])
        for n, count in state["lr_steps"].items():
            self.schedules[n].set_count(int(count))
        self.update_count = int(state["update_count"])
        self.return_scale.load_state_dict(state["reward_scale"])

    @staticmethod
    def _load_optimizer(opt: torch.optim.Optimizer, state: dict[str, Any]) -> None:
        """Load optimizer state while keeping the LR tensor and state tensors in place."""
        lr_tensors = [g["lr"] for g in opt.param_groups]
        old_state = {p: dict(st) for p, st in opt.state.items()}
        opt.load_state_dict(state)
        for g, t in zip(opt.param_groups, lr_tensors, strict=True):
            t.fill_(float(g["lr"]))
            g["lr"] = t
        for p, old in old_state.items():
            new = opt.state.get(p)
            if new is None:
                continue
            for k, v in new.items():
                if (
                    torch.is_tensor(v)
                    and torch.is_tensor(old.get(k))
                    and old[k].shape == v.shape
                ):
                    old[k].copy_(v)
                    new[k] = old[k]
