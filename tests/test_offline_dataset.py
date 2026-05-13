from __future__ import annotations

import torch
from tensordict import TensorDict
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer

from rlopt.config_base import RLOptConfig
from rlopt.expert import OfflineExpertSampler


def test_offline_dataset_config_defaults_to_unitree_wbt_lerobot() -> None:
    cfg = RLOptConfig()

    assert cfg.offline_dataset.enabled is False
    assert cfg.offline_dataset.replace_expert_sampler is False
    assert cfg.offline_dataset.source == "lerobot_stream"
    assert cfg.offline_dataset.repo_id == "unitreerobotics/G1_WBT_Brainco_Pickup_Pillow"
    assert cfg.offline_dataset.repo_ids == []
    assert cfg.offline_dataset.split == "train"
    assert cfg.offline_dataset.mapper == "unitree_g1_wbt_29dof"
    assert cfg.offline_dataset.cache_storage == "torchrl_memmap"
    assert cfg.offline_dataset.cache_dir == ""
    assert cfg.offline_dataset.batch_size == 0
    assert cfg.offline_dataset.max_episodes == 0
    assert cfg.offline_dataset.max_episodes_per_repo == 0
    assert cfg.offline_dataset.default_joint_pos_pool == []
    assert cfg.offline_dataset.dataset_joint_names == []
    assert cfg.offline_dataset.target_joint_names == []
    assert cfg.offline_dataset.align_root_z_to_default is True
    assert cfg.offline_dataset.default_root_height == 0.0


def test_offline_expert_sampler_selects_required_tensordict_keys() -> None:
    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(max_size=8),
        batch_size=2,
    )
    replay_buffer.extend(
        TensorDict(
            {
                ("policy", "joint_pos_rel"): torch.randn(4, 29),
                ("next", "policy", "joint_pos_rel"): torch.randn(4, 29),
                "expert_action": torch.randn(4, 29),
                "unused": torch.randn(4, 3),
            },
            batch_size=[4],
        )
    )
    sampler = OfflineExpertSampler(replay_buffer)

    batch = sampler(
        2,
        [
            ("policy", "joint_pos_rel"),
            ("next", "policy", "joint_pos_rel"),
            "expert_action",
        ],
    )

    assert batch.numel() == 2
    assert ("policy", "joint_pos_rel") in batch.keys(True)
    assert ("next", "policy", "joint_pos_rel") in batch.keys(True)
    assert "expert_action" in batch.keys(True)
    assert "unused" not in batch.keys(True)
