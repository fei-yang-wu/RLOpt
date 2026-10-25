"""Opt-in bf16 autocast for the PPO actor-critic update (``ppo.update_amp_dtype``).

Only the loss-module forward of the gradient step runs under
:func:`torch.autocast`; its backward follows the dtypes autocast recorded.
Everything the update must not round stays float32:

* master weights, gradients as stored on the parameters, and optimizer state
  (autocast only casts the per-forward copies of the weights);
* the running input normalizers, whose element-wise ops are not on autocast's
  lower-precision list, so they run on the float32 observations;
* the output projection of each actor/critic MLP trunk, re-run in float32
  from the bf16 hidden features (:func:`install_update_amp_boundaries`);
* everything downstream of the network outputs: the policy ``loc``/``scale``
  and the state value leave the networks as float32, so log-probabilities,
  entropy, ratios, clip fractions and the critic loss are float32 math;
* everything outside the loss forward: collection, GAE / value targets and
  the post-update KL diagnostic (``_compute_kl_after_update``), which keeps
  comparing against the stored float32 behaviour ``loc``/``scale``.

The PPO ratio's "old" log-probability is NOT the stored behaviour log-prob.
That one came from the float32 collection forward, while the current
log-prob comes from the bf16 trunk; their difference is roughly
``sqrt(action_dim) * |delta loc| / sigma`` and grows without bound as the
policy sharpens (half the rows clipped at sigma ~0.05 before any step). So
before the first optimizer step of each iteration the agent re-evaluates the
old log-prob with the same bf16 forward under the unchanged weights, in
chunks of exactly ``loss.mini_batch_size`` rows -- GEMM kernels, and with
them the rounding, depend on the row count, while each row's result does not
depend on which other rows share its minibatch -- and stores it under
:data:`AMP_OLD_LOG_PROB_KEY`. Only the autocast loss forward reads it; the
stored behaviour log-prob is left untouched for every other consumer. The
ratio is then exactly 1 at the old weights; the price is that the ratio's
denominator is the bf16 evaluation of the behaviour policy rather than the
float32 one that sampled the actions.

float16 is rejected: it needs a gradient scaler, which this path does not run.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import torch
import torch.nn.functional as F
from torch import Tensor, nn

AMP_OLD_LOG_PROB_KEY = "amp_old_log_prob"
"""Buffer key of the bf16 re-evaluation of the old log-prob (see module doc)."""

_DTYPE_NAMES: dict[str, torch.dtype | None] = {
    "none": None,
    "fp32": None,
    "float32": None,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}


def resolve_update_amp_dtype(name: str | None) -> torch.dtype | None:
    """Map the ``ppo.update_amp_dtype`` config value to an autocast dtype.

    ``None`` (the default), ``"none"``, ``"fp32"`` and ``"float32"`` disable
    autocast. ``"bf16"`` / ``"bfloat16"`` enable bf16 autocast.
    """
    if name is None:
        return None
    key = str(name).strip().lower()
    if key in ("fp16", "float16", "half"):
        msg = (
            "ppo.update_amp_dtype=fp16 is not supported: float16 autocast needs "
            "a GradScaler, which the PPO update does not run. Use bf16."
        )
        raise ValueError(msg)
    if key not in _DTYPE_NAMES:
        msg = (
            f"ppo.update_amp_dtype must be one of None, {sorted(_DTYPE_NAMES)}; "
            f"got {name!r}."
        )
        raise ValueError(msg)
    return _DTYPE_NAMES[key]


@contextlib.contextmanager
def update_autocast(
    device: torch.device | str, dtype: torch.dtype | None
) -> Iterator[None]:
    """Autocast context for the update forward; a no-op when ``dtype`` is None."""
    if dtype is None:
        yield
        return
    device_type = torch.device(device).type
    with torch.autocast(device_type=device_type, dtype=dtype):
        yield


def _float32_outputs(output):
    if isinstance(output, Tensor):
        if output.is_floating_point() and output.dtype != torch.float32:
            return output.float()
        return output
    if isinstance(output, tuple):
        return tuple(_float32_outputs(item) for item in output)
    return output


def _cast_outputs_hook(_module: nn.Module, _args, output):
    """Forward hook: return low-precision floating outputs as float32."""
    return _float32_outputs(output)


def _float32_projection_hook(module: nn.Module, args, output):
    """Forward hook: under autocast, recompute a Linear's output in float32.

    Outside autocast the layer already ran in float32 and the hook keeps its
    output. Under autocast the bf16 product is discarded and the layer is
    re-applied to the float32-cast input with autocast disabled, so the
    network output carries no bf16 rounding of its own. For a 512-wide
    output head the extra product is negligible next to the trunk.
    """
    if not isinstance(output, Tensor) or not args:
        return output
    device_type = output.device.type
    if not torch.is_autocast_enabled(device_type):
        return output
    features = args[0]
    with torch.autocast(device_type=device_type, enabled=False):
        return F.linear(features.float(), module.weight, module.bias)


def trailing_linear(module: nn.Module) -> nn.Linear | None:
    """Return the output projection of an MLP trunk, unwrapping input adapters.

    Recognises ``nn.Sequential`` trunks (TorchRL ``MLP``) whose last child is
    an ``nn.Linear``, behind any number of modules that keep the trunk in a
    ``.module`` attribute (RLOpt's ``CatInputs`` / ``RunningMeanStdCatInputs``).
    Returns ``None`` for any other structure.
    """
    seen: set[int] = set()
    while id(module) not in seen:
        seen.add(id(module))
        if isinstance(module, nn.Sequential):
            if len(module) > 0 and isinstance(module[-1], nn.Linear):
                return module[-1]
            return None
        inner = getattr(module, "module", None)
        if not isinstance(inner, nn.Module):
            return None
        module = inner
    return None


def install_update_amp_boundaries(
    output_modules: list[nn.Module],
    trunks: list[nn.Module],
) -> list[torch.utils.hooks.RemovableHandle]:
    """Install the float32 boundaries used by the autocast update.

    ``output_modules`` get a hook that returns their floating outputs as
    float32 (the policy head's ``(loc, scale)``, the critic's value). Each
    trunk's trailing ``nn.Linear`` (see :func:`trailing_linear`) is re-run in
    float32 under autocast. All hooks are inert outside autocast, so the
    collector, GAE and KL diagnostic paths see unchanged float32 modules.
    Install them only when ``ppo.update_amp_dtype`` is set.
    """
    handles: list[torch.utils.hooks.RemovableHandle] = []
    projections: list[nn.Linear] = []
    for trunk in trunks:
        linear = trailing_linear(trunk)
        if linear is not None and all(linear is not seen for seen in projections):
            projections.append(linear)
    for linear in projections:
        handles.append(linear.register_forward_hook(_float32_projection_hook))
    for module in output_modules:
        handles.append(module.register_forward_hook(_cast_outputs_hook))
    return handles
