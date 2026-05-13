from __future__ import annotations

import threading
from collections.abc import Callable, Iterator, Mapping, MutableMapping
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from typing import Any

import torch
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer

try:
    from iltools.datasets.lerobot_stream import (
        LeRobotStreamingCacheConfig,
        StreamingTensorDictReplayCache,
        UnitreeG1WBT29DofMapper,
        UnitreeG1WBT29DofMapperConfig,
    )
except ImportError:
    LeRobotStreamingCacheConfig = None  # type: ignore[assignment]
    StreamingTensorDictReplayCache = None  # type: ignore[assignment]
    UnitreeG1WBT29DofMapper = None  # type: ignore[assignment]
    UnitreeG1WBT29DofMapperConfig = None  # type: ignore[assignment]


@dataclass
class StateMapper:
    """Map raw dataset transitions into IPMD's expected keys.

    The mapper can be configured with simple key copies or custom callables.
    It operates on batched transitions and should return tensors with a
    consistent batch dimension.

    Configure using dictionaries that map destination keys to either a
    source key (str) or a callable that takes a `TensorDict` and returns
    a `torch.Tensor`.
    """

    obs_map: Mapping[str, str | Callable[[TensorDict], torch.Tensor]] = field(
        default_factory=dict
    )
    action_key: str | Callable[[TensorDict], torch.Tensor] = "action"
    next_obs_map: Mapping[str, str | Callable[[TensorDict], torch.Tensor]] = field(
        default_factory=dict
    )

    def _apply_map(
        self,
        raw: TensorDict,
        mapping: Mapping[str, str | Callable[[TensorDict], torch.Tensor]],
    ) -> TensorDict:
        out: MutableMapping[str, torch.Tensor] = {}
        for dst, spec in mapping.items():
            if isinstance(spec, str):
                out[dst] = raw.get(spec)
            else:
                out[dst] = spec(raw)
        return TensorDict(out, batch_size=raw.batch_size, device=raw.device)

    def map_transition(self, raw: TensorDict) -> TensorDict:
        """Return a TensorDict with keys: observation, action, (next, observation).

        - `observation` is built from `obs_map`. If `obs_map` is empty and
          the raw input already has `"observation"`, it is forwarded.
        - `action` can be copied from a key or computed via callable.
        - `next/observation` is built from `next_obs_map`. If empty and the raw
          input already has nested next/observation, it is forwarded.
        """
        # observation
        if self.obs_map:
            obs_td = self._apply_map(raw, self.obs_map)
        else:
            # passthrough if present
            obs_td = TensorDict({}, batch_size=raw.batch_size, device=raw.device)
            if "observation" in raw:
                obs_td.set("observation", raw.get("observation"))
            else:
                msg = "StateMapper: obs_map is empty and 'observation' not found in raw"
                raise KeyError(msg)

        # action
        if isinstance(self.action_key, str):
            action = raw.get(self.action_key)
        else:
            action = self.action_key(raw)

        # next/observation
        if self.next_obs_map:
            next_td = self._apply_map(raw, self.next_obs_map)
        else:
            next_td = TensorDict({}, batch_size=raw.batch_size, device=raw.device)
            # support nested ('next', 'observation') or flat 'next_observation'
            if ("next", "observation") in raw.keys(True, True):
                next_td.set(("next", "observation"), raw.get(("next", "observation")))
            elif "next_observation" in raw:
                next_td.set(("next", "observation"), raw.get("next_observation"))
            else:
                msg = (
                    "StateMapper: next_obs_map is empty and next observation not found"
                )
                raise KeyError(msg)

        out = TensorDict({}, batch_size=raw.batch_size, device=raw.device)
        # Merge observation fields: if obs_td has multiple fields, concatenate under 'observation'
        if "observation" in obs_td:
            out.set("observation", obs_td.get("observation"))
        else:
            # Concatenate all values in insertion order
            vals = [obs_td.get(k) for k in obs_td]
            out.set("observation", torch.cat(vals, dim=-1))

        out.set("action", action)

        # Merge next observation similarly
        if ("next", "observation") in next_td.keys(True, True):
            out.set(("next", "observation"), next_td.get(("next", "observation")))
        else:
            vals = [next_td.get(k) for k in next_td]
            out.set(("next", "observation"), torch.cat(vals, dim=-1))

        return out


def build_prefetch_iterator(
    source: Iterator[TensorDict],
    *,
    device: torch.device,
    prefetch: int = 2,
    non_blocking: bool = True,
) -> Iterator[TensorDict]:
    """Wrap an iterator of TensorDict batches with device prefetching.

    - Moves each batch to `device` using `TensorDict.to(device, non_blocking=...)`.
    - Buffers up to `prefetch` batches on a background thread.

    This is a low-overhead way to stream expert data into GPU-heavy training
    loops (e.g., IsaacLab) without stalling on H2D transfers.
    """

    q: Queue[TensorDict | None] = Queue(maxsize=max(1, prefetch))

    def _worker() -> None:
        try:
            for batch in source:
                # Push after device transfer to overlap copy and compute
                moved = batch.to(device, non_blocking=non_blocking)
                q.put(moved, block=True)
        finally:
            # Sentinel to terminate consumer
            q.put(None)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    while True:
        item = q.get()
        if item is None:
            break
        yield item


class OfflineExpertSampler:
    """Callable adapter exposing cached TensorDict batches to IPMD."""

    def __init__(
        self,
        replay_buffer: TensorDictReplayBuffer,
        *,
        required_preflight_batch_size: int = 1,
    ) -> None:
        if len(replay_buffer) < int(required_preflight_batch_size):
            msg = (
                "Offline replay buffer does not contain enough transitions for "
                f"preflight: len={len(replay_buffer)}, required={required_preflight_batch_size}."
            )
            raise ValueError(msg)
        self.replay_buffer = replay_buffer

    def __call__(
        self, batch_size: int, required_keys: list[str | tuple[str, ...]]
    ) -> TensorDict:
        batch = self.replay_buffer.sample(int(batch_size))
        return batch.select(*required_keys).clone()


class StreamingOfflineExpertSampler:
    """Callable sampler backed by a background LeRobot-to-TorchRL cache."""

    def __init__(self, cache: Any) -> None:
        self.cache = cache

    @property
    def replay_buffer(self) -> TensorDictReplayBuffer:
        return self.cache.replay_buffer

    def __call__(
        self, batch_size: int, required_keys: list[str | tuple[str, ...]]
    ) -> TensorDict:
        batch = self.cache.sample(int(batch_size))
        return batch.select(*required_keys).clone()


def _discover_offline_mapper_params(env: object) -> dict[str, Any]:
    stack: list[object] = [env]
    visited: set[int] = set()
    while stack:
        current = stack.pop()
        obj_id = id(current)
        if obj_id in visited:
            continue
        visited.add(obj_id)
        method = getattr(current, "get_offline_dataset_mapper_params", None)
        if callable(method):
            return dict(method())
        for attr_name in ("base_env", "env", "_env", "unwrapped"):
            next_obj = getattr(current, attr_name, None)
            if next_obj is None:
                continue
            if isinstance(next_obj, list | tuple):
                stack.extend(next_obj)
            else:
                stack.append(next_obj)
    return {}


def build_offline_expert_sampler(
    config: Any, env: object
) -> StreamingOfflineExpertSampler | None:
    """Build an optional offline expert sampler from ``config.offline_dataset``."""

    offline_cfg = getattr(config, "offline_dataset", None)
    if offline_cfg is None or not bool(offline_cfg.enabled):
        return None
    if str(offline_cfg.source) != "lerobot_stream":
        msg = f"Unsupported offline_dataset.source={offline_cfg.source!r}."
        raise ValueError(msg)
    if str(offline_cfg.mapper) != "unitree_g1_wbt_29dof":
        msg = f"Unsupported offline_dataset.mapper={offline_cfg.mapper!r}."
        raise ValueError(msg)
    if str(offline_cfg.cache_storage) != "torchrl_memmap":
        msg = (
            f"Unsupported offline_dataset.cache_storage={offline_cfg.cache_storage!r}."
        )
        raise ValueError(msg)

    params = _discover_offline_mapper_params(env)
    default_joint_pos_pool = list(offline_cfg.default_joint_pos_pool) or list(
        params.get("default_joint_pos_pool", [])
    )
    default_joint_pos = list(offline_cfg.default_joint_pos) or list(
        params.get("default_joint_pos", [])
    )
    action_scale = list(offline_cfg.action_scale) or list(
        params.get("action_scale", [])
    )
    dataset_joint_names = list(offline_cfg.dataset_joint_names) or list(
        params.get("dataset_joint_names", [])
    )
    target_joint_names = list(offline_cfg.target_joint_names) or list(
        params.get("target_joint_names", params.get("joint_names", []))
    )
    default_root_height = float(offline_cfg.default_root_height)
    if default_root_height == 0.0:
        default_root_height = float(params.get("default_root_height", 0.0))
    align_root_z_to_default = bool(
        offline_cfg.align_root_z_to_default and default_root_height > 0.0
    )
    cache_dir = str(offline_cfg.cache_dir)
    if not cache_dir:
        cache_dir = str(Path("offline_dataset_cache") / str(offline_cfg.mapper))

    if (
        LeRobotStreamingCacheConfig is None
        or StreamingTensorDictReplayCache is None
        or UnitreeG1WBT29DofMapper is None
        or UnitreeG1WBT29DofMapperConfig is None
    ):
        msg = (
            "RLOpt offline_dataset.source='lerobot_stream' requires "
            "ImitationLearningTools with the LeRobot streaming utilities."
        )
        raise ImportError(msg)

    mapper_cfg = UnitreeG1WBT29DofMapperConfig(
        robot_q_current_key=str(offline_cfg.robot_q_current_key),
        robot_q_desired_key=str(offline_cfg.robot_q_desired_key),
        episode_key=str(offline_cfg.episode_key),
        dt=1.0 / float(offline_cfg.fps),
        default_joint_pos=default_joint_pos_pool or default_joint_pos,
        action_scale=action_scale,
        dataset_joint_names=dataset_joint_names,
        target_joint_names=target_joint_names,
        align_root_z_to_default=align_root_z_to_default,
        default_root_height=default_root_height,
        quat_order=str(offline_cfg.quat_order),
    )
    mapper = UnitreeG1WBT29DofMapper(mapper_cfg)
    cache_cfg = LeRobotStreamingCacheConfig(
        repo_id=str(offline_cfg.repo_id),
        repo_ids=tuple(str(repo_id) for repo_id in offline_cfg.repo_ids),
        split=str(offline_cfg.split),
        cache_dir=cache_dir,
        max_cache_transitions=int(offline_cfg.max_cache_transitions),
        min_ready_transitions=int(offline_cfg.min_ready_transitions),
        low_watermark=int(offline_cfg.low_watermark),
        starvation_timeout_s=float(offline_cfg.starvation_timeout_s),
        local_sample_prefetch=int(offline_cfg.local_sample_prefetch),
        batch_size=None
        if int(offline_cfg.batch_size) <= 0
        else int(offline_cfg.batch_size),
        max_episodes=None
        if int(offline_cfg.max_episodes) <= 0
        else int(offline_cfg.max_episodes),
        max_episodes_per_repo=None
        if int(offline_cfg.max_episodes_per_repo) <= 0
        else int(offline_cfg.max_episodes_per_repo),
        mapper=mapper_cfg,
    )
    cache = StreamingTensorDictReplayCache(cache_cfg, mapper=mapper)
    cache.start()
    cache.wait_until_ready()
    return StreamingOfflineExpertSampler(cache)
