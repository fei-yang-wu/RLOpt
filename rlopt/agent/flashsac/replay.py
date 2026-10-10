# ruff: noqa: EM101, EM102
"""Time-major GPU replay ring for FlashSAC.

The buffer keeps one vector step (all ``N`` envs) per ring row. Storage is
``[T, N, F]``, indexed by time row. Each observation is stored once. The next
observation of a transition is read from row ``t + 1`` of the same env when
the episode did not end. When the episode ended by a time-out (truncation
without termination), the env wrapper gives the terminal observation in
``next_obs``, and the buffer copies it into a small dense *terminal pool* on
the GPU. A per-(row, env) slot index points to the pool entry. A termination
does not bootstrap, so its next observation is never read by the target; the
window returns the observation of the terminating row as a finite filler
(it only enters the critic's shared BatchNorm forward). Pooling terminations
too would size the pool by the termination rate, which is far above the
time-out rate early in training (a random G1 policy ends about 28 percent of
rows), and an overwritten pool entry would then corrupt the time-out rows.

The n-step window is built at sample time from per-env, time-contiguous data
(raw per-step reward and the two done flags). A learned reward can therefore
be relabelled later. The window stops at the first episode end:

* ``reward_n``   : sum_{k<n_eff} gamma^k r_{t+k}
* ``terminated_n``: 1.0 only if a true termination ended the window. A
  time-out (truncation) does not cut the bootstrap. If both flags are set on
  one step, termination wins.
* ``discount_n`` : gamma^n_eff, where n_eff is the number of steps summed.
* ``next_obs``   : the observation after n_eff steps. On a truncated window it
  is the terminal observation of the truncating step (from the pool). On a
  terminated window it is the observation of the terminating row (filler).

Guard rows. The newest ``n_step + 1`` rows are never used as window starts, so
a window never reads past the write head (the next-observation row must be
written). Once the ring is full, the oldest row is the first valid start and
every row of a window is newer than it, so a window never reads overwritten
data.

Host sync. ``add()`` and ``sample()`` use only device-side scatter, gather and
fill kernels. The write pointer and the valid-row counters are Python ints on
the host, advanced by one per ``add()``. The terminal-pool head is a device
scalar. ``add()`` and ``sample()`` must run on the same stream (or be ordered
by the caller); the buffer is not thread safe.

CUDA graphs. ``sample()`` takes its row range from a host int, so a graph
that captured it would bake the range. Use ``sample_graph_safe()``: it reads
the range from device tensors (``valid_rows_tensor``) that ``add()`` updates
with ``fill_``. Rows are drawn as ``floor(u * valid)`` with ``u`` in float64.

Terminal pool. The pool is a ring of ``pool_slots`` entries. Every ``add()``
scatters the ``next_obs`` of the time-out rows (truncated and not terminated)
into the next free entries. A count needs no host read: slots come from a
device ``cumsum``, and the other rows write to a trash entry. Each entry
carries a stamp (``step * N + env``). ``sample()`` compares the stamp of the entry it reads
with the expected one and adds mismatches to ``pool_miss`` (device counter).
A non-zero ``pool_miss_count()`` means the pool was too small for the ring
and the next-observation of some time-out rows was wrong; raise
``pool_fraction`` (it must exceed the time-out share of the rows).

Memory formula. With ``b`` bytes per obs element (2 for bf16), per transition
(one (row, env) slot)::

    bytes = (Do + Dc) * b          # obs and critic obs, stored once
          + A * 4                  # action (fp32, tanh space)
          + 4                      # reward (fp32)
          + 2                      # terminated, truncated (bool each)
          + 4                      # terminal-pool slot index (int32)
          + 4 * sum(extra dims)    # extra keys (fp32)
          + pool_fraction * (Do + Dc) * b   # terminal pool, amortised

For Do=351, Dc=544, A=29, bf16 and the default ``pool_fraction`` of 0.05 this
is 1,790 + 116 + 4 + 2 + 4 = 1,916 B plus 89.5 B of pool = 2,005.5 B per
transition. At N=20,480 (5 x 4,096) and T rows the ring takes
``20480 * T * 1,916`` B (``T=1024``: 40.2 GB, 37.4 GiB; recomputed for
N=20,480) plus the pool. See ``memory_bytes()``.

Resume. ``state_dict()`` returns the counters (and, with
``include_storage=True``, the storage tensors). ``save_storage(path)`` writes
the full storage to one file with ``torch.save``; ``load_storage(path)``
reads it back. Storage can be tens of GB; keep it on persistent disk.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

__all__ = ["TimeMajorReplay"]

_REQUIRED_KEYS = (
    "obs",
    "critic_obs",
    "action",
    "reward",
    "terminated",
    "truncated",
    "next_obs",
    "next_critic_obs",
)


class TimeMajorReplay:
    """GPU-resident time-major replay ring with sample-time n-step returns."""

    def __init__(
        self,
        num_envs: int,
        capacity_steps: int,
        actor_obs_dim: int,
        critic_obs_dim: int,
        act_dim: int,
        n_step: int = 3,
        gamma: float = 0.99,
        device: torch.device | str = "cpu",
        obs_dtype: torch.dtype = torch.bfloat16,
        extra_keys: dict[str, int] | None = None,
        pool_fraction: float = 0.05,
    ) -> None:
        if n_step < 1:
            raise ValueError("n_step must be >= 1")
        if capacity_steps < n_step + 2:
            raise ValueError("capacity_steps must be at least n_step + 2")
        if not 0.0 < pool_fraction <= 1.0:
            raise ValueError("pool_fraction must be in (0, 1]")
        self.num_envs = int(num_envs)
        self.capacity_steps = int(capacity_steps)
        self.actor_obs_dim = int(actor_obs_dim)
        self.critic_obs_dim = int(critic_obs_dim)
        self.act_dim = int(act_dim)
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self.device = torch.device(device)
        self.obs_dtype = obs_dtype
        self.extra_keys = dict(extra_keys or {})
        self.guard_rows = self.n_step + 1

        T, N = self.capacity_steps, self.num_envs
        dev = self.device
        self.obs = torch.zeros(T, N, self.actor_obs_dim, dtype=obs_dtype, device=dev)
        self.critic_obs = torch.zeros(
            T, N, self.critic_obs_dim, dtype=obs_dtype, device=dev
        )
        self.action = torch.zeros(T, N, self.act_dim, dtype=torch.float32, device=dev)
        self.reward = torch.zeros(T, N, dtype=torch.float32, device=dev)
        self.terminated = torch.zeros(T, N, dtype=torch.bool, device=dev)
        self.truncated = torch.zeros(T, N, dtype=torch.bool, device=dev)
        self.term_slot = torch.full((T, N), -1, dtype=torch.int32, device=dev)
        self.extras = {
            k: torch.zeros(T, N, d, dtype=torch.float32, device=dev)
            for k, d in self.extra_keys.items()
        }

        # Terminal pool: ``pool_slots`` real entries plus one trash entry.
        self.pool_slots = max(2 * N, math.ceil(pool_fraction * T * N))
        self.pool_obs = torch.zeros(
            self.pool_slots + 1, self.actor_obs_dim, dtype=obs_dtype, device=dev
        )
        self.pool_critic_obs = torch.zeros(
            self.pool_slots + 1, self.critic_obs_dim, dtype=obs_dtype, device=dev
        )
        self.pool_stamp = torch.full(
            (self.pool_slots + 1,), -1, dtype=torch.int64, device=dev
        )
        self.pool_head = torch.zeros((), dtype=torch.int64, device=dev)
        self.pool_miss = torch.zeros((), dtype=torch.int64, device=dev)

        # Host counters (Python ints, advanced by one per add()).
        self._total = 0  # vector steps written so far
        # Device copies read by sample_graph_safe(): [valid, oldest_phys, oldest_logical].
        self._meta = torch.zeros(3, dtype=torch.int64, device=dev)
        self._env_ar = torch.arange(N, dtype=torch.int64, device=dev)
        self._gpow = torch.tensor(
            [self.gamma**k for k in range(self.n_step + 1)],
            dtype=torch.float32,
            device=dev,
        )

    # ------------------------------------------------------------------ sizes
    @property
    def filled_rows(self) -> int:
        return min(self._total, self.capacity_steps)

    @property
    def valid_rows(self) -> int:
        """Rows usable as window starts (filled rows minus guard rows)."""
        return max(0, self.filled_rows - self.guard_rows)

    @property
    def oldest_logical(self) -> int:
        return self._total - self.filled_rows

    @property
    def valid_rows_tensor(self) -> Tensor:
        """Device int64 scalar view of ``valid_rows`` (updated by ``add``)."""
        return self._meta[0]

    @property
    def oldest_row_tensor(self) -> Tensor:
        """Device int64 scalar: physical index of the oldest filled row."""
        return self._meta[1]

    def __len__(self) -> int:
        return self.valid_rows * self.num_envs

    def memory_bytes(self) -> int:
        tensors = [
            self.obs,
            self.critic_obs,
            self.action,
            self.reward,
            self.terminated,
            self.truncated,
            self.term_slot,
            self.pool_obs,
            self.pool_critic_obs,
            self.pool_stamp,
            *self.extras.values(),
        ]
        return sum(t.numel() * t.element_size() for t in tensors)

    @staticmethod
    def bytes_per_transition(
        actor_obs_dim: int,
        critic_obs_dim: int,
        act_dim: int,
        obs_bytes: int = 2,
        extra_dim: int = 0,
    ) -> int:
        """Ring bytes per (row, env) slot, excluding the terminal pool."""
        return (
            (actor_obs_dim + critic_obs_dim) * obs_bytes
            + act_dim * 4
            + 4
            + 2
            + 4
            + 4 * extra_dim
        )

    # -------------------------------------------------------------------- add
    @torch.no_grad()
    def add(self, step: dict[str, Tensor]) -> None:
        """Write one vector step for all envs. No host sync."""
        for key in _REQUIRED_KEYS:
            if key not in step:
                raise KeyError(f"replay.add: missing key '{key}'")
        N = self.num_envs
        row = self._total % self.capacity_steps
        obs_dt = self.obs_dtype
        terminated = step["terminated"].reshape(N).to(torch.bool)
        truncated = step["truncated"].reshape(N).to(torch.bool)

        self.obs[row] = step["obs"].reshape(N, -1).to(obs_dt)
        self.critic_obs[row] = step["critic_obs"].reshape(N, -1).to(obs_dt)
        self.action[row] = step["action"].reshape(N, -1).to(torch.float32)
        self.reward[row] = step["reward"].reshape(N).to(torch.float32)
        self.terminated[row] = terminated
        self.truncated[row] = truncated
        for key, buf in self.extras.items():
            if key not in step:
                raise KeyError(f"replay.add: missing extra key '{key}'")
            buf[row] = step[key].reshape(N, -1).to(torch.float32)

        # Terminal pool: device-side compaction of the time-out rows. A
        # termination (which wins over a simultaneous time-out) never
        # bootstraps, so it takes no pool entry.
        timeout = truncated & ~terminated
        cs = timeout.to(torch.int64).cumsum(0)
        slot = (self.pool_head + cs - 1) % self.pool_slots
        slot = torch.where(timeout, slot, torch.full_like(slot, self.pool_slots))
        self.pool_obs[slot] = step["next_obs"].reshape(N, -1).to(obs_dt)
        self.pool_critic_obs[slot] = step["next_critic_obs"].reshape(N, -1).to(obs_dt)
        self.pool_stamp[slot] = self._total * N + self._env_ar
        self.term_slot[row] = torch.where(timeout, slot, torch.full_like(slot, -1)).to(
            torch.int32
        )
        self.pool_head += cs[-1]

        self._total += 1
        self._meta[0].fill_(self.valid_rows)
        self._meta[1].fill_(self.oldest_logical % self.capacity_steps)
        self._meta[2].fill_(self.oldest_logical)

    # ----------------------------------------------------------------- sample
    @torch.no_grad()
    def sample(
        self, batch_size: int, generator: torch.Generator | None = None
    ) -> dict[str, Tensor]:
        """Sample ``batch_size`` n-step transitions, uniform over (row, env)."""
        valid = self.valid_rows
        if valid <= 0:
            raise RuntimeError(
                f"replay.sample: need more than {self.guard_rows} rows, "
                f"have {self.filled_rows}"
            )
        u = torch.randint(
            0, valid, (batch_size,), device=self.device, generator=generator
        )
        env = torch.randint(
            0, self.num_envs, (batch_size,), device=self.device, generator=generator
        )
        return self._window(
            u,
            env,
            oldest_phys=self.oldest_logical % self.capacity_steps,
            oldest_logical=self.oldest_logical,
        )

    @torch.no_grad()
    def sample_graph_safe(
        self,
        batch_size: int,
        valid_rows_tensor: Tensor | None = None,
        oldest_row_tensor: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Sample with the row range read from device scalars.

        Safe to capture in a CUDA graph: the range is not baked. Pass the
        persistent tensors ``replay.valid_rows_tensor`` and
        ``replay.oldest_row_tensor`` (the defaults); ``add()`` refreshes them
        outside the graph. Uses the global RNG (graph-registered). The caller
        must make sure ``valid_rows_tensor >= 1`` before replay.
        """
        valid = self._meta[0] if valid_rows_tensor is None else valid_rows_tensor
        oldest = self._meta[1] if oldest_row_tensor is None else oldest_row_tensor
        ur = torch.rand(batch_size, dtype=torch.float64, device=self.device)
        u = (ur * valid.to(torch.float64)).floor().to(torch.int64)
        u = torch.minimum(u, (valid - 1).clamp(min=0))
        env = torch.randint(0, self.num_envs, (batch_size,), device=self.device)
        return self._window(u, env, oldest_phys=oldest, oldest_logical=self._meta[2])

    def _window(
        self,
        u: Tensor,
        env: Tensor,
        oldest_phys: Tensor | int,
        oldest_logical: Tensor | int,
    ) -> dict[str, Tensor]:
        """Build the n-step batch for window starts (``u`` rows after oldest)."""
        T, n = self.capacity_steps, self.n_step
        s = (u + oldest_phys) % T
        B = u.shape[0]
        alive = torch.ones(B, dtype=torch.bool, device=self.device)
        ended = torch.zeros_like(alive)
        term_any = torch.zeros_like(alive)
        ret = torch.zeros(B, dtype=torch.float32, device=self.device)
        n_eff = torch.zeros(B, dtype=torch.int64, device=self.device)
        end_row = s.clone()
        for k in range(n):  # fixed small unroll
            rk = (s + k) % T
            te = self.terminated[rk, env]
            done = te | self.truncated[rk, env]
            ret = ret + torch.where(
                alive, self._gpow[k] * self.reward[rk, env], torch.zeros_like(ret)
            )
            n_eff = n_eff + alive.to(torch.int64)
            now = alive & done
            term_any = term_any | (now & te)
            end_row = torch.where(now, rk, end_row)
            ended = ended | now
            alive = alive & ~done

        timed_out = ended & ~term_any
        slot = self.term_slot[end_row, env].to(torch.int64)
        slot = slot.clamp(min=0)
        # Pool consistency: the entry must still belong to the row it was made for.
        expected = (oldest_logical + u + n_eff - 1) * self.num_envs + env
        self.pool_miss += (timed_out & (self.pool_stamp[slot] != expected)).sum()

        nxt = (s + n) % T
        # Not ended: row s+n. Time-out: the pool. Termination: the terminating
        # row's own observation (filler; the target does not bootstrap).
        src = torch.where(ended, end_row, nxt)
        e = timed_out.unsqueeze(-1)
        next_obs = torch.where(e, self.pool_obs[slot], self.obs[src, env])
        next_critic = torch.where(
            e, self.pool_critic_obs[slot], self.critic_obs[src, env]
        )
        out = {
            "obs": self.obs[s, env].float(),
            "critic_obs": self.critic_obs[s, env].float(),
            "action": self.action[s, env],
            "reward_1": self.reward[s, env],
            "reward_n": ret,
            "terminated_n": term_any.to(torch.float32),
            "discount_n": self._gpow[n_eff],
            "n_eff": n_eff,
            "next_obs": next_obs.float(),
            "next_critic_obs": next_critic.float(),
            "row": s,
            "env": env,
        }
        for key, buf in self.extras.items():
            out[key] = buf[s, env]
        return out

    # ------------------------------------------------------------- diagnostics
    def pool_miss_count(self) -> int:
        """Number of sampled terminal reads whose pool entry was overwritten.

        Host sync. Call it rarely (logging).
        """
        return int(self.pool_miss.item())

    # ------------------------------------------------------------ checkpointing
    _STORAGE = (
        "obs",
        "critic_obs",
        "action",
        "reward",
        "terminated",
        "truncated",
        "term_slot",
        "pool_obs",
        "pool_critic_obs",
        "pool_stamp",
    )

    def _storage_tensors(self) -> dict[str, Tensor]:
        out = {k: getattr(self, k) for k in self._STORAGE}
        out.update({f"extra/{k}": v for k, v in self.extras.items()})
        return out

    def state_dict(self, include_storage: bool = False) -> dict[str, Any]:
        """Counters (and optionally storage) for resume. Syncs on pool_head."""
        sd: dict[str, Any] = {
            "meta": {
                "num_envs": self.num_envs,
                "capacity_steps": self.capacity_steps,
                "actor_obs_dim": self.actor_obs_dim,
                "critic_obs_dim": self.critic_obs_dim,
                "act_dim": self.act_dim,
                "n_step": self.n_step,
                "pool_slots": self.pool_slots,
                "extra_keys": dict(self.extra_keys),
            },
            "total": self._total,
            "pool_head": int(self.pool_head.item()),
            "pool_miss": int(self.pool_miss.item()),
            "has_storage": bool(include_storage),
        }
        if include_storage:
            sd["storage"] = {k: v.clone() for k, v in self._storage_tensors().items()}
        return sd

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        """Restore counters (and storage if present).

        Without storage the ring restarts empty (counters reset) but keeps
        the global step count in ``total`` out of the buffer: resume then
        refills before sampling.
        """
        m = sd["meta"]
        for k in (
            "num_envs",
            "capacity_steps",
            "actor_obs_dim",
            "critic_obs_dim",
            "act_dim",
            "n_step",
            "pool_slots",
        ):
            if m[k] != getattr(self, k):
                raise ValueError(f"replay.load_state_dict: {k} mismatch")
        if m["extra_keys"] != self.extra_keys:
            raise ValueError("replay.load_state_dict: extra_keys mismatch")
        if sd.get("has_storage"):
            for k, v in sd["storage"].items():
                dst = self._storage_tensors()[k]
                dst.copy_(v.to(dst.device))
            self._total = int(sd["total"])
            self.pool_head.fill_(int(sd["pool_head"]))
        else:
            self._total = 0
            self.pool_head.zero_()
            self.term_slot.fill_(-1)
            self.pool_stamp.fill_(-1)
        self.pool_miss.zero_()
        self._meta[0].fill_(self.valid_rows)
        self._meta[1].fill_(self.oldest_logical % self.capacity_steps)
        self._meta[2].fill_(self.oldest_logical)

    def save_storage(self, path: str) -> None:
        """Write counters and all storage tensors to one file (may be huge)."""
        torch.save(self.state_dict(include_storage=True), path)

    def load_storage(self, path: str) -> None:
        self.load_state_dict(torch.load(path, map_location=self.device))
