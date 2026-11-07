"""CUDA-graph capture of the PPO / IPMD gradient step (``ppo.capture_update``).

One graph holds a whole gradient step on static buffers:

    loss forward -> backward -> grad clip -> optimizer step -> post-update KL

Per update the host only (1) draws the minibatch indices from the replay
buffer's own sampler, so the CUDA RNG stream is consumed exactly as in the
eager loop, (2) gathers those rows of the keys the step reads into the static
batch (one ``index_select`` per key), (3) writes the step's optimizer scalars
into a small device tensor from pinned memory, and (4) replays the graph. A
learning-rate change -- the KL rule once per iteration, the critic schedule,
a torch LR scheduler -- is just different scalars, never a recapture.

Bit-exactness with the eager update (``capture_update_optimizer="exact"``):

* The forward/backward/clip run the agent's own eager functions
  (``_captured_update_backward`` / ``_captured_update_clip_gradients``) inside
  the capture, i.e. the same kernels on the same shapes.
* The optimizer step reproduces PyTorch's foreach (non-capturable) Adam/AdamW
  arithmetic with the per-step Python scalars -- ``1 - lr*wd``,
  ``sqrt(1 - beta2^t)`` and ``-lr / (1 - beta1^t)``, computed on the host in
  float64 exactly as ``torch.optim.adam._multi_tensor_adam`` does and rounded
  to float32 -- read from device memory. The final ``addcdiv`` with a scalar
  list is replayed as ``addcmul`` of the quotient with the broadcast device
  scalar, which rounds identically (both are one fused multiply-add). The
  optimizer's own state (``exp_avg``, ``exp_avg_sq``, ``step``) is updated in
  place, so checkpoints are unchanged. Step counters a resume mapped to the
  device (``torch.load(map_location=cuda)``) are incremented inside the graph
  and mirrored in float32 on the host, where eager reads each with ``.item()``.
  ``tests/test_ppo_captured_update.py`` asserts the equivalence.
* The post-update KL forward samples an action (ProbabilisticActor), as the
  eager diagnostic does; graph-safe Philox makes the replay draw the same
  numbers from the same offsets, so the RNG stream after an iteration matches.
* The finiteness check on that KL (a host sync in the eager loop) moves to the
  end of the iteration: one device-to-host copy of all the step's KL values,
  then the same ``_record_kl_for_lr_adaptation`` calls in the same order.
  Under ``optim.kl_adapt_step="update"`` the LR changes between steps, so the
  check stays per step there (one sync per step, as eager).
* TorchRL's ``PPOLoss._get_entropy`` checks ``entropy.isfinite().all()`` on the
  host. While capturing, a copy without that check is installed; it returns the
  same tensor whenever the entropy is finite (a non-finite entropy makes the
  eager loss NaN anyway).

Warm-up runs on a side stream before capture, on a snapshot of every tensor
the step mutates (parameters, optimizer state, module buffers, CUDA/CPU RNG),
which is restored afterwards, so capturing has no numerical side effect.

``capture_update_optimizer="fused"`` replaces the replica with PyTorch's fused
Adam/AdamW kernel and a device LR: fewer kernels, NOT bit-identical.

Anything else falls back to the eager loop and logs the reason once.

Cost: the graph's private memory pool keeps the step's peak (the forward
activations of both networks) reserved between iterations, where the eager
caching allocator would hand it to the collector and GAE. At 393,216 rows with
the combo50b [2048,2048,1024,1024,512,512] networks that is ~48 GB (fp32) /
~27 GB (bf16) on top of the eager footprint.
"""

from __future__ import annotations

import types
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase
from torch import Tensor
from torchrl._utils import timeit
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement

from rlopt.agent.ppo.mixed_precision import AMP_OLD_LOG_PROB_KEY

if TYPE_CHECKING:
    from rlopt.agent.ppo.ppo import PPO, PPOIterationData, PPOTrainingMetadata

CAPTURE_UPDATE_OPTIMIZERS = ("exact", "fused")

# The update entry points whose defining class must declare
# ``_capture_update_aware = True`` for the captured step to stand in for them.
_CAPTURE_ENTRY_POINTS = (
    "update",
    "iterate",
    "_backward_ppo_terms",
    "_captured_update_backward",
    "_captured_update_clip_gradients",
)


def resolve_capture_update_optimizer(name: Any) -> str:
    mode = str(name).strip().lower()
    if mode not in CAPTURE_UPDATE_OPTIMIZERS:
        msg = (
            f"ppo.capture_update_optimizer must be one of "
            f"{CAPTURE_UPDATE_OPTIMIZERS}; got {name!r}."
        )
        raise ValueError(msg)
    return mode


def exact_adam_scalars(
    lr: float, step: float, beta1: float, beta2: float, weight_decay: float
) -> tuple[float, float, float]:
    """The per-step scalars of the eager foreach Adam/AdamW step.

    ``(1 - lr * wd, sqrt(1 - beta2^t), -lr / (1 - beta1^t))`` in float64, as
    ``torch.optim.adam._multi_tensor_adam`` (``capturable=False``) computes
    them; the kernels round them to float32 exactly as the device copy does.
    """
    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step
    return (
        1 - lr * weight_decay,
        bias_correction2**0.5,
        (lr / bias_correction1) * -1,
    )


@torch.no_grad()
def exact_adam_step(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    scalars: Tensor,
    *,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    decoupled: bool,
) -> None:
    """Eager foreach Adam/AdamW arithmetic, op for op, with device scalars.

    ``scalars`` is a float32 device tensor ``[3]`` holding
    :func:`exact_adam_scalars`; reading them from memory instead of baking
    Python numbers into the kernels is what lets a captured graph follow a
    changing learning rate. Step counters are not touched here.
    """
    if weight_decay != 0:
        if decoupled:
            torch._foreach_mul_(params, scalars[0])
        else:
            grads = torch._foreach_add(grads, params, alpha=weight_decay)
    torch._foreach_lerp_(exp_avgs, grads, 1 - beta1)
    torch._foreach_mul_(exp_avg_sqs, beta2)
    torch._foreach_addcmul_(exp_avg_sqs, grads, grads, 1 - beta2)
    denom = torch._foreach_sqrt(exp_avg_sqs)
    torch._foreach_div_(denom, scalars[1])
    torch._foreach_add_(denom, eps)
    ratio = torch._foreach_div(exp_avgs, denom)
    step_size = scalars[2]
    # == _foreach_addcdiv_(params, exp_avgs, denom, step_sizes): both round
    # p + s * (m / d) with one fused multiply-add.
    torch._foreach_addcmul_(params, ratio, [step_size.expand_as(r) for r in ratio])


@dataclass
class _GroupSpec:
    """The parameters of one optimizer param group that receive gradients."""

    group_index: int
    params: list[Tensor]
    exp_avgs: list[Tensor]
    exp_avg_sqs: list[Tensor]
    steps: list[Tensor]  # the optimizer's own step counters
    beta1: float
    beta2: float
    eps: float
    weight_decay: float
    decoupled: bool
    # Step counters restored from a checkpoint with map_location=cuda live on
    # the device; the graph increments them and the host keeps a float32
    # mirror (the eager step reads them with .item(), a sync per parameter).
    device_steps: bool = False
    host_step: np.float32 = field(default_factory=lambda: np.float32(0.0))


class _HostTruthTensor(Tensor):
    """A view of a 0-dim buffer whose truth value was read once on the host.

    Arithmetic on it returns plain tensors reading the shared storage; only
    ``bool()`` is answered from the host copy (no device sync).
    """

    __torch_function__ = torch._C._disabled_torch_function_impl
    _truth: bool = False

    def __bool__(self) -> bool:
        return self._truth


def _unique(tensors) -> list[Tensor]:
    seen: set[int] = set()
    out: list[Tensor] = []
    for tensor in tensors:
        if id(tensor) not in seen:
            seen.add(id(tensor))
            out.append(tensor)
    return out


class CapturedUpdate:
    """Owns the graph, its static buffers and the per-update host bookkeeping."""

    def __init__(self, agent: PPO) -> None:
        self.agent = agent
        self.optimizer_mode = resolve_capture_update_optimizer(
            getattr(agent.config.ppo, "capture_update_optimizer", "exact")
        )
        self.warmup_steps = 2
        self.graph: torch.cuda.CUDAGraph | None = None
        self.signature: tuple | None = None
        self.keys: list[Any] = []
        self.static_batch: TensorDict | None = None
        self.specs: list[_GroupSpec] = []
        self.scalars: Tensor | None = None
        self.out_loss: TensorDict | None = None
        self.out_grad_norm: Tensor | None = None
        self.out_post_kl: Tensor | None = None
        self.policy_operator = None
        self.disabled_reason: str | None = None
        self.captures = 0
        self.replays = 0
        # Every reason the update ran eagerly, logged once each.
        self.fallback_reasons: set[str] = set()
        self._constants: dict[str, Tensor] | None = None
        self._cpu_one = torch.tensor(1.0)
        self._failure: BaseException | None = None

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(
        self,
        iteration: PPOIterationData,  # noqa: ARG002
        metadata: PPOTrainingMetadata,
        losses: TensorDict,
    ) -> bool:
        """Run this iteration's PPO epochs captured; False means "run eagerly"."""
        reason = self.disabled_reason or self._unsupported_reason(metadata)
        if reason is not None:
            self._report(reason)
            return False
        buffer = self.agent.data_buffer
        storage = buffer._storage
        batch_size = int(buffer._batch_size)
        rows = len(storage)
        if rows % batch_size != 0:
            self._report(
                f"the minibatch size {batch_size} does not divide the "
                f"{rows}-row rollout (uneven last minibatch)"
            )
            return False
        if rows // batch_size != int(metadata.minibatches_per_epoch):
            self._report(
                f"{rows // batch_size} minibatches per epoch in the buffer, "
                f"{metadata.minibatches_per_epoch} in the metadata"
            )
            return False
        adaptive = self._adaptive()
        if self.graph is not None and not self._steps_in_lockstep():
            self.disabled_reason = (
                "optimizer step counters diverged inside a param group"
            )
            self._report(self.disabled_reason)
            return False
        signature = self._signature(storage, batch_size, adaptive)
        if self.graph is None or signature != self.signature:
            try:
                self._capture(storage, batch_size, adaptive, metadata)
            except Exception as exc:  # any capture failure: run eagerly
                self._release()
                self.disabled_reason = f"capture failed: {type(exc).__name__}: {exc}"
                self.agent.log.debug(
                    "ppo.capture_update capture failure", exc_info=True
                )
                self._failure = exc
                self._report(self.disabled_reason)
                return False
            self.signature = self._signature(storage, batch_size, adaptive)
        self._run_updates(metadata, losses, storage, batch_size, adaptive)
        return True

    # ------------------------------------------------------------------
    # Support checks
    # ------------------------------------------------------------------

    def _report(self, reason: str) -> None:
        if reason in self.fallback_reasons:
            return
        self.fallback_reasons.add(reason)
        self.agent.log.warning(
            "ppo.capture_update: running the update eagerly: %s", reason
        )

    def _adaptive(self) -> bool:
        return (self.agent.config.optim.scheduler or "").lower() == "adaptive"

    def _unsupported_reason(self, metadata: PPOTrainingMetadata) -> str | None:
        agent = self.agent
        cfg = agent.config
        if not torch.cuda.is_available() or torch.device(agent.device).type != "cuda":
            return "the agent is not on a CUDA device"
        if bool(getattr(cfg.compile, "compile", False)):
            return "compile.compile=True"
        if int(getattr(cfg.ppo, "rnn_hidden_size", 0) or 0) > 0:
            return "recurrent actor (ppo.rnn_hidden_size > 0)"
        if getattr(cfg.optim, "kl_sample_size", None) is not None:
            return "optim.kl_sample_size is set"
        for name in _CAPTURE_ENTRY_POINTS:
            owner = next(
                (cls for cls in type(agent).__mro__ if name in cls.__dict__), None
            )
            if owner is not None and not owner.__dict__.get(
                "_capture_update_aware", False
            ):
                return (
                    f"{owner.__name__}.{name} is not covered by the captured "
                    "step (_capture_update_aware)"
                )
        if metadata.epochs_per_rollout <= 0:
            return "no epochs"
        reason = self._optimizer_unsupported_reason()
        if reason is None:
            reason = agent._captured_update_unsupported_reason()
        if reason is not None:
            return reason
        buffer = agent.data_buffer
        if not isinstance(buffer, TensorDictReplayBuffer):
            return f"unsupported replay buffer {type(buffer).__name__}"
        if type(buffer._sampler) is not SamplerWithoutReplacement:
            return f"unsupported sampler {type(buffer._sampler).__name__}"
        storage = buffer._storage
        if not isinstance(storage, LazyTensorStorage) or not isinstance(
            getattr(storage, "_storage", None), TensorDictBase
        ):
            return f"unsupported storage {type(storage).__name__}"
        if storage.ndim != 1:
            return "the replay storage is not flat"
        if buffer._transform is not None and len(buffer._transform):
            return "the replay buffer has transforms"
        if getattr(buffer, "_prefetch", None):
            return "the replay buffer prefetches"
        return None

    def _optimizer_unsupported_reason(self) -> str | None:
        optim = self.agent.optim
        if type(optim) not in (torch.optim.Adam, torch.optim.AdamW):
            return f"optimizer {type(optim).__name__} (Adam / AdamW only)"
        if self.optimizer_mode == "fused" and not hasattr(torch, "_fused_adamw_"):
            return "this torch build has no fused Adam kernel"
        for group in optim.param_groups:
            name = group.get("name", "?")
            for key in ("amsgrad", "maximize", "capturable", "differentiable"):
                if bool(group.get(key, False)):
                    return f"param group {name!r} sets {key}=True"
            if group.get("fused"):
                return f"param group {name!r} sets fused=True"
            if group.get("foreach") is False:
                # The eager step would take the single-tensor path, whose
                # rounding the exact replica does not reproduce.
                return f"param group {name!r} sets foreach=False"
            if any(
                isinstance(group.get(key), Tensor)
                for key in ("lr", "weight_decay", "eps")
            ) or any(isinstance(beta, Tensor) for beta in group["betas"]):
                return f"param group {name!r} holds a tensor hyperparameter"
        return None

    def _steps_in_lockstep(self, *, include_device: bool = False) -> bool:
        """All step counters of a param group are equal (host-side ones only,
        unless ``include_device``: reading device counters is a sync)."""
        for spec in self.specs:
            if spec.device_steps and not include_device:
                continue
            first = float(spec.steps[0].item())
            if any(float(step.item()) != first for step in spec.steps[1:]):
                return False
        return True

    def _check_device_step_mirrors(self) -> None:
        """Compare the device step counters with their host mirrors.

        Called after the iteration's own device-to-host copies, so it adds no
        stall. A mismatch means something stepped the optimizer behind the
        graph's back; the scalars of this iteration were then wrong.
        """
        specs = [spec for spec in self.specs if spec.device_steps]
        if not specs:
            return
        values = torch.stack([spec.steps[0].reshape(()) for spec in specs]).cpu()
        for spec, value in zip(specs, values.tolist(), strict=True):
            if np.float32(value) != spec.host_step:
                self.disabled_reason = (
                    "a device step counter moved outside the captured update "
                    f"({value} != {float(spec.host_step)})"
                )
                self._release()
                msg = f"ppo.capture_update: {self.disabled_reason}"
                raise RuntimeError(msg)

    # ------------------------------------------------------------------
    # Static batch
    # ------------------------------------------------------------------

    def _batch_keys(self, source: TensorDictBase, adaptive: bool) -> list[Any]:
        """Keys of the rollout rows the captured step reads."""
        agent = self.agent
        wanted: list[Any] = []

        def add(key) -> None:
            key = key if isinstance(key, str) else tuple(key)
            if isinstance(key, tuple) and len(key) == 1:
                key = key[0]
            if key not in wanted:
                wanted.append(key)

        for key in agent.loss_module.in_keys:
            if isinstance(key, tuple) and key[0] == "next":
                continue  # GAE already ran; the loss reads no next-step data
            add(key)
        tensor_keys = getattr(agent.loss_module, "tensor_keys", None)
        for name in ("advantage", "value_target", "value", "sample_log_prob", "action"):
            key = getattr(tensor_keys, name, None)
            if key is not None:
                add(key)
        add("shifted_valid")
        for key in agent.actor_critic.get_policy_operator().in_keys:
            add(key)
        if adaptive:
            add("loc")
            add("scale")
        if getattr(agent, "_update_amp_dtype", None) is not None:
            add(AMP_OLD_LOG_PROB_KEY)
        present = set(source.keys(include_nested=True, leaves_only=True))
        return [key for key in wanted if key in present]

    def _allocate_static_batch(
        self, source: TensorDictBase, keys: list[Any], batch_size: int
    ) -> None:
        device = torch.device(self.agent.device)
        batch = TensorDict({}, batch_size=[batch_size], device=device)
        for key in keys:
            src = source.get(key)
            batch.set(
                key,
                torch.empty(
                    (batch_size, *src.shape[1:]), dtype=src.dtype, device=src.device
                ),
            )
        self.static_batch = batch
        self.keys = keys

    def _gather(self, source: TensorDictBase, index: Tensor) -> None:
        assert self.static_batch is not None
        for key in self.keys:
            torch.index_select(
                source.get(key), 0, index, out=self.static_batch.get(key)
            )

    # ------------------------------------------------------------------
    # Signature (recapture when anything the graph baked in moved)
    # ------------------------------------------------------------------

    def _tracked_modules(self) -> list[torch.nn.Module]:
        return [self.agent.actor_critic, self.agent.loss_module]

    def _buffers(self) -> list[Tensor]:
        return _unique(
            buf for module in self._tracked_modules() for buf in module.buffers()
        )

    def _optim_params(self) -> list[Tensor]:
        return _unique(
            p for group in self.agent.optim.param_groups for p in group["params"]
        )

    def _signature(self, storage, batch_size: int, adaptive: bool) -> tuple:
        agent = self.agent
        optim = agent.optim
        source = storage._storage
        keys = self._batch_keys(source, adaptive)
        key_sig = tuple(
            (key, tuple(source.get(key).shape[1:]), source.get(key).dtype)
            for key in keys
        )
        params = self._optim_params()
        state_sig = tuple(
            (key, value.data_ptr())
            for p in params
            for key, value in sorted(optim.state.get(p, {}).items())
            if isinstance(value, Tensor)
        )
        group_sig = tuple(
            (
                tuple(id(p) for p in group["params"]),
                float(group["weight_decay"]),
                tuple(float(b) for b in group["betas"]),
                float(group["eps"]),
                bool(group.get("decoupled_weight_decay", False)),
            )
            for group in optim.param_groups
        )
        mode_sig = tuple(
            (module.training, getattr(module, "frozen", None))
            for tracked in self._tracked_modules()
            for module in tracked.modules()
        )
        return (
            batch_size,
            key_sig,
            tuple(p.data_ptr() for p in params),
            state_sig,
            tuple(buf.data_ptr() for buf in self._buffers()),
            group_sig,
            mode_sig,
            adaptive,
            getattr(agent, "_update_amp_dtype", None),
            self.optimizer_mode,
        )

    # ------------------------------------------------------------------
    # Snapshot / restore around warm-up
    # ------------------------------------------------------------------

    def _snapshot_tensors(self) -> list[tuple[Tensor, Tensor]]:
        tensors = [*self._optim_params(), *self._buffers()]
        return [(t, t.detach().clone()) for t in _unique(tensors)]

    def _snapshot_optimizer_state(self) -> list[tuple[Tensor, Tensor]]:
        optim = self.agent.optim
        pairs = []
        for p in self._optim_params():
            for value in optim.state.get(p, {}).values():
                if isinstance(value, Tensor):
                    pairs.append((value, value.detach().clone()))
        return pairs

    @staticmethod
    @torch.no_grad()
    def _restore(pairs: list[tuple[Tensor, Tensor]]) -> None:
        for target, saved in pairs:
            target.copy_(saved)

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    @contextmanager
    def _capture_safe_loss(self) -> Iterator[None]:
        """Remove the loss module's host syncs for warm-up and capture.

        TorchRL's ``PPOLoss`` checks ``entropy.isfinite().all()`` and branches
        on ``if self.clip_value:`` (a 0-dim buffer). The first gets a copy of
        ``_get_entropy`` without the check (same tensor whenever the entropy is
        finite); the second a view of the same buffer whose truth value was
        read once, here, so the arithmetic still reads the live buffer.
        """
        loss_module = self.agent.loss_module
        original = getattr(type(loss_module), "_get_entropy", None)
        patched_entropy = False
        if original is not None:

            def _get_entropy(module, dist, adv_shape):
                try:
                    entropy = dist.entropy()
                except NotImplementedError:
                    # Monte-Carlo path, as eager (it has no host sync).
                    return original(module, dist, adv_shape)
                return entropy.unsqueeze(-1)

            loss_module._get_entropy = types.MethodType(_get_entropy, loss_module)
            patched_entropy = True
        buffers = getattr(loss_module, "_buffers", {})
        clip_value = buffers.get("clip_value")
        if isinstance(clip_value, Tensor):
            view = clip_value.as_subclass(_HostTruthTensor)
            view._truth = bool(clip_value)
            buffers["clip_value"] = view
        try:
            yield
        finally:
            if isinstance(clip_value, Tensor):
                buffers["clip_value"] = clip_value
            if patched_entropy:
                del loss_module._get_entropy

    def _release(self) -> None:
        self.graph = None
        self.signature = None
        self.out_loss = self.out_grad_norm = self.out_post_kl = None

    def _build_specs(self) -> None:
        """Group the parameters that received gradients; create missing state."""
        optim = self.agent.optim
        specs = []
        for gi, group in enumerate(optim.param_groups):
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            for p in params:
                state = optim.state[p]
                if len(state) == 0:
                    # torch.optim.Adam._init_group, non-capturable non-fused.
                    state["step"] = torch.tensor(
                        0.0,
                        dtype=(
                            torch.float64
                            if torch.get_default_dtype() == torch.float64
                            else torch.float32
                        ),
                    )
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
            beta1, beta2 = group["betas"]
            specs.append(
                _GroupSpec(
                    group_index=gi,
                    params=params,
                    exp_avgs=[optim.state[p]["exp_avg"] for p in params],
                    exp_avg_sqs=[optim.state[p]["exp_avg_sq"] for p in params],
                    steps=[optim.state[p]["step"] for p in params],
                    beta1=float(beta1),
                    beta2=float(beta2),
                    eps=float(group["eps"]),
                    weight_decay=float(group["weight_decay"]),
                    decoupled=bool(group.get("decoupled_weight_decay", False)),
                )
            )
        for spec in specs:
            on_device = [step.is_cuda for step in spec.steps]
            if any(on_device) and not all(on_device):
                msg = "optimizer step counters are split between host and device"
                raise RuntimeError(msg)
            if any(step.dtype != torch.float32 for step in spec.steps):
                msg = "optimizer step counters are not float32"
                raise RuntimeError(msg)
            spec.device_steps = all(on_device)
            # One read at capture time (a sync when they live on the device).
            spec.host_step = np.float32(spec.steps[0].item())
        self.specs = specs
        if not self._steps_in_lockstep(include_device=True):
            msg = "optimizer step counters differ inside a param group"
            raise RuntimeError(msg)
        width = 3 if self.optimizer_mode == "exact" else 2
        self.scalars = torch.zeros(
            (len(specs), width), dtype=torch.float32, device=self.agent.device
        )

    def _capture(
        self,
        storage,
        batch_size: int,
        adaptive: bool,
        metadata: PPOTrainingMetadata,
    ) -> None:
        agent = self.agent
        device = torch.device(agent.device)
        recapture = self.captures > 0
        self._release()
        source = storage._storage
        keys = self._batch_keys(source, adaptive)
        self._allocate_static_batch(source, keys, batch_size)
        assert self.static_batch is not None
        # Any rows will do for warm-up and capture; no RNG is drawn here.
        for key in keys:
            self.static_batch.get(key).copy_(source.get(key)[:batch_size])
        self.policy_operator = metadata.policy_operator

        tensors = self._snapshot_tensors()
        cuda_rng = torch.cuda.get_rng_state(device)
        cpu_rng = torch.get_rng_state()
        optim_state: list[tuple[Tensor, Tensor]] = []
        try:
            side = torch.cuda.Stream(device=device)
            side.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(side), self._capture_safe_loss():
                # Discovery: which parameters receive gradients.
                agent.optim.zero_grad(set_to_none=True)
                agent._captured_update_backward(self.static_batch)
                self._build_specs()
                optim_state = self._snapshot_optimizer_state()
                for _ in range(self.warmup_steps):
                    agent.optim.zero_grad(set_to_none=True)
                    self._stage_scalars(advance=False)
                    self._body(adaptive)
            torch.cuda.current_stream(device).wait_stream(side)
            torch.cuda.synchronize(device)
        finally:
            agent.optim.zero_grad(set_to_none=True)
            self._restore(tensors)
            self._restore(optim_state)
            torch.cuda.set_rng_state(cuda_rng, device)
            torch.set_rng_state(cpu_rng)
            torch.cuda.synchronize(device)

        # Gradients must be unset when captured: AccumulateGrad then writes
        # fresh gradients into graph-owned memory on every replay, which is
        # zero_grad(set_to_none=True) + backward.
        agent.optim.zero_grad(set_to_none=True)
        graph = torch.cuda.CUDAGraph()
        torch.cuda.empty_cache()
        reserved_before = torch.cuda.memory_reserved(device)
        try:
            with (
                self._capture_safe_loss(),
                torch.cuda.graph(graph, capture_error_mode="thread_local"),
            ):
                out_loss, grad_norm, post_kl = self._body(adaptive)
        except Exception:
            # Nothing ran, but leave the generator exactly as it was.
            agent.optim.zero_grad(set_to_none=True)
            torch.cuda.set_rng_state(cuda_rng, device)
            raise
        pool_gb = (torch.cuda.memory_reserved(device) - reserved_before) / 1e9
        self.graph = graph
        self.out_loss = out_loss
        self.out_grad_norm = grad_norm
        self.out_post_kl = post_kl
        self.captures += 1
        agent.log.info(
            "ppo.capture_update: %s the gradient step (%d rows, %d keys, "
            "optimizer=%s, ~%.2f GB graph memory).",
            "recaptured" if recapture else "captured",
            batch_size,
            len(keys),
            self.optimizer_mode,
            pool_gb,
        )

    def _body(self, adaptive: bool):
        """One gradient step on the static batch (eager in warm-up, captured)."""
        agent = self.agent
        batch = self.static_batch
        policy_op = self.policy_operator
        kl_context = agent._prepare_kl_context(batch, policy_op) if adaptive else None
        out_loss = agent._captured_update_backward(batch)
        grad_norm = agent._captured_update_clip_gradients().detach()
        self._optimizer_step()
        post_kl = (
            agent._kl_after_update_tensor(kl_context, policy_op)
            if kl_context is not None
            else None
        )
        return out_loss, grad_norm, post_kl

    # ------------------------------------------------------------------
    # Optimizer step
    # ------------------------------------------------------------------

    def _stage_scalars(self, *, advance: bool) -> None:
        """Write this step's optimizer scalars to device memory (no host sync).

        ``advance=True`` first increments the optimizer's host step counters
        (or the host mirror of device counters), as the eager step does
        before reading them.
        """
        optim = self.agent.optim
        rows = []
        for spec in self.specs:
            if spec.device_steps:
                # The graph increments the device counters (float32 + 1).
                step_f32 = spec.host_step + np.float32(1.0)
                if advance:
                    spec.host_step = step_f32
                step = float(step_f32)
            elif advance:
                torch._foreach_add_(spec.steps, self._cpu_one, alpha=1.0)
                step = float(spec.steps[0].item())
            else:
                step = float(spec.steps[0].item()) + 1.0
            group = optim.param_groups[spec.group_index]
            lr = group["lr"]
            if self.optimizer_mode == "exact":
                rows.append(
                    exact_adam_scalars(
                        lr, step, spec.beta1, spec.beta2, spec.weight_decay
                    )
                )
            else:
                rows.append((lr, step))
        host = torch.tensor(rows, dtype=torch.float64).to(torch.float32)
        pinned = torch.empty(host.shape, dtype=torch.float32, pin_memory=True)
        pinned.copy_(host)
        assert self.scalars is not None
        self.scalars.copy_(pinned, non_blocking=True)

    @torch.no_grad()
    def _optimizer_step(self) -> None:
        assert self.scalars is not None
        for si, spec in enumerate(self.specs):
            grads = [p.grad for p in spec.params]
            if spec.device_steps:
                # Eager: _foreach_add_(device_state_steps, 1) before the step.
                torch._foreach_add_(spec.steps, 1)
            if self.optimizer_mode == "fused":
                fused = torch._fused_adamw_ if spec.decoupled else torch._fused_adam_
                fused(
                    spec.params,
                    grads,
                    spec.exp_avgs,
                    spec.exp_avg_sqs,
                    [],
                    [self.scalars[si, 1]] * len(spec.params),
                    lr=self.scalars[si, 0],
                    beta1=spec.beta1,
                    beta2=spec.beta2,
                    weight_decay=spec.weight_decay,
                    eps=spec.eps,
                    amsgrad=False,
                    maximize=False,
                    grad_scale=None,
                    found_inf=None,
                )
                continue
            exact_adam_step(
                spec.params,
                grads,
                spec.exp_avgs,
                spec.exp_avg_sqs,
                self.scalars[si],
                beta1=spec.beta1,
                beta2=spec.beta2,
                eps=spec.eps,
                weight_decay=spec.weight_decay,
                decoupled=spec.decoupled,
            )

    # ------------------------------------------------------------------
    # Per-iteration loop
    # ------------------------------------------------------------------

    def _output_constants(self) -> dict[str, Tensor]:
        if self._constants is None:
            self._constants = {
                key: torch.tensor(float(value), device=self.agent.device)
                for key, value in self.agent._update_output_constants().items()
            }
        return self._constants

    def _run_updates(
        self,
        metadata: PPOTrainingMetadata,
        losses: TensorDict,
        storage,
        batch_size: int,
        adaptive: bool,
    ) -> None:
        agent = self.agent
        optim_cfg = agent.config.optim
        sampler = agent.data_buffer._sampler
        source = storage._storage
        kl_per_iteration = (
            str(getattr(optim_cfg, "kl_adapt_step", "update") or "update").lower()
            == "iteration"
        )
        timeit_name = getattr(agent, "_update_timeit_name", None)
        constants = self._output_constants()
        assert self.graph is not None
        assert self.out_loss is not None
        pending: list[tuple[int, int, Tensor]] = []
        for epoch_idx in range(metadata.epochs_per_rollout):
            # ReplayBuffer.__iter__: one pass until the sampler runs out.
            if sampler.ran_out:
                sampler.ran_out = False
            batch_idx = 0
            while not sampler.ran_out:
                index, _info = sampler.sample(storage, batch_size)
                self._gather(source, index)
                self._stage_scalars(advance=True)
                if timeit_name:
                    with timeit(timeit_name):
                        self.graph.replay()
                else:
                    self.graph.replay()
                self.replays += 1
                metadata.updates_completed = metadata.updates_completed + 1

                loss = self.out_loss.clone(False)
                for key, value in constants.items():
                    loss.set(key, value)
                loss.set("grad_norm", self.out_grad_norm)
                if agent.lr_scheduler and agent.lr_scheduler_step == "update":
                    agent.lr_scheduler.step()
                if adaptive:
                    assert self.out_post_kl is not None
                    if kl_per_iteration:
                        # Provisional; non-finite values are resolved below.
                        pending.append(
                            (epoch_idx, batch_idx, self.out_loss["kl_approx"].clone())
                        )
                        loss.set("kl_approx", self.out_post_kl)
                    else:
                        kl_host = self.out_post_kl.detach().cpu()
                        if torch.isfinite(kl_host):
                            loss.set("kl_approx", self.out_post_kl)
                            agent._record_kl_for_lr_adaptation(kl_host, optim_cfg)
                losses[epoch_idx, batch_idx] = agent._select_reported_loss_metrics(loss)
                batch_idx += 1
            if agent.lr_scheduler and agent.lr_scheduler_step == "epoch":
                agent.lr_scheduler.step()
        if pending:
            kl_values = losses.get("kl_approx")
            kl_host = kl_values.detach().cpu()  # one copy for the iteration
            for epoch_idx, batch_idx, ppo_kl in pending:
                value = kl_host[epoch_idx, batch_idx]
                if torch.isfinite(value):
                    agent._record_kl_for_lr_adaptation(value, optim_cfg)
                else:
                    kl_values[epoch_idx, batch_idx] = ppo_kl
        self._check_device_step_mirrors()
