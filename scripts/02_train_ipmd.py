#!/usr/bin/env python
"""Train IPMD with expert demonstrations and reward estimation."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import LazyTensorStorage

from rlopt.agent.imitation import IPMD, IPMDRLOptConfig
from rlopt.configs import NetworkConfig
from rlopt.env_utils import make_parallel_env
from rlopt.imitation import ExpertReplayBuffer


def train_ipmd(
    expert_data_path: str,
    env_name: str | None = None,
    total_frames: int = 100_000,
    device: str = "cuda:0",
    seed: int = 43,
    expert_batch_size: int = 256,
) -> IPMD:
    """Train IPMD with expert demonstrations.

    Args:
        expert_data_path: Path to expert buffer (.pt file)
        env_name: Environment name (auto-detected from path if None)
        total_frames: Total training frames
        device: Device for training
        seed: Random seed (different from expert to avoid bias)
        expert_batch_size: Batch size for expert sampling

    Returns:
        Trained IPMD agent
    """
    print("=" * 70)
    print("IPMD Training with Expert Demonstrations")
    print("=" * 70)

    # Load expert data
    expert_path = Path(expert_data_path)
    if not expert_path.exists():
        raise FileNotFoundError(f"Expert data not found: {expert_path}")

    print(f"\nLoading expert data from: {expert_path}")
    expert_data = torch.load(expert_path, weights_only=False)

    # Load metadata if available
    metadata_path = expert_path.parent / "metadata.pt"
    if metadata_path.exists():
        metadata = torch.load(metadata_path, weights_only=False)
        print(f"Expert data info:")
        print(f"  - Environment: {metadata['env_name']}")
        print(f"  - Transitions: {metadata['num_transitions']:,}")
        print(f"  - Expert reward: {metadata.get('final_reward', 'N/A')}")
        if env_name is None:
            env_name = metadata["env_name"]

    if env_name is None:
        raise ValueError("Environment name must be provided or detected from metadata")

    # Create expert replay buffer
    print(f"\nCreating expert replay buffer with {len(expert_data):,} transitions...")
    expert_buffer_storage = LazyTensorStorage(max_size=len(expert_data))
    expert_buffer = TensorDictReplayBuffer(
        storage=expert_buffer_storage,
        batch_size=expert_batch_size,
    )
    expert_buffer.extend(expert_data)

    # Wrap in ExpertReplayBuffer
    expert_replay = ExpertReplayBuffer(expert_buffer)
    print(f"✅ Expert buffer created: {len(expert_replay):,} samples")

    # Configure IPMD
    print("\n" + "=" * 70)
    print("Configuring IPMD")
    print("=" * 70)

    cfg = IPMDRLOptConfig()
    cfg.env.env_name = env_name
    cfg.env.device = device
    cfg.device = device
    cfg.collector.frames_per_batch = 1000
    cfg.collector.total_frames = total_frames
    cfg.collector.init_random_frames = 1000
    cfg.replay_buffer.size = 100_000
    cfg.loss.mini_batch_size = 256

    # Enable compilation for faster training
    cfg.compile.compile = True
    cfg.compile.compile_mode = "reduce-overhead"
    cfg.compile.cudagraphs = False  # Keep disabled by default (experimental)

    cfg.seed = seed
    cfg.logger.log_dir = "logs"
    cfg.logger.backend = "wandb"
    cfg.logger.project_name = "RLOpt-IPMD"
    cfg.logger.entity = "fywu"
    cfg.logger.group_name = env_name
    cfg.logger.exp_name = f"IPMD_{env_name}_seed{seed}"

    # Network configurations (matching SAC expert)
    cfg.q_function = NetworkConfig(
        num_cells=[256, 256],
        activation_fn="relu",
        output_dim=1,
        input_keys=["observation"],
    )
    cfg.policy = NetworkConfig(
        num_cells=[256, 256],
        activation_fn="relu",
        input_keys=["observation"],
    )

    # IPMD-specific: Enable reward estimation
    cfg.ipmd.use_estimated_rewards_for_sac = True
    cfg.ipmd.reward_num_cells = (256, 256)
    cfg.ipmd.reward_lr = 3e-4
    cfg.ipmd.reward_detach_features = True
    cfg.ipmd.expert_batch_size = expert_batch_size

    print("\nIPMD Configuration:")
    print(f"  - Environment: {env_name}")
    print(f"  - Training frames: {total_frames:,}")
    print(f"  - Reward estimation: {cfg.ipmd.use_estimated_rewards_for_sac}")
    print(f"  - Expert batch size: {expert_batch_size}")
    print(f"  - Seed: {seed}")

    # Create IPMD agent
    print("\n" + "=" * 70)
    print("Initializing IPMD")
    print("=" * 70)

    env = make_parallel_env(cfg)
    agent = IPMD(env, cfg)

    # Set expert buffer
    agent.set_expert_buffer(expert_replay)
    print(f"✅ Expert buffer attached: {len(expert_replay):,} samples")

    print(f"\nLog directory: {agent.log_dir}")

    # Train IPMD
    print("\n" + "=" * 70)
    print("Training IPMD")
    print("=" * 70)
    print("\nIPMD will alternate between:")
    print("  1. Reward estimation (using expert demonstrations)")
    print("  2. Policy optimization (using estimated rewards)\n")

    agent.train()

    # Get final performance
    print("\n" + "=" * 70)
    print("Training Results")
    print("=" * 70)

    if len(agent.episode_rewards) > 0:
        # Convert deque to list for slicing
        rewards_list = list(agent.episode_rewards)
        final_reward = sum(rewards_list[-10:]) / min(10, len(agent.episode_rewards))
        print(f"\n✅ IPMD Training Complete!")
        print(f"   Final reward (last 10 episodes): {final_reward:.2f}")
        print(f"   Total episodes: {len(agent.episode_rewards)}")

        # Compare to expert if available
        if metadata_path.exists() and metadata.get("final_reward"):
            expert_reward = metadata["final_reward"]
            print(f"   Expert reward: {expert_reward:.2f}")
            print(f"   Performance ratio: {final_reward / expert_reward:.2%}")
    else:
        print("\n⚠️  No episodes completed during training")

    # Save the trained model
    model_path = agent.log_dir / "ipmd_model.pt"
    agent.save_model(model_path)
    print(f"\n   Model saved: {model_path}")
    print(f"   Log directory: {agent.log_dir}")

    return agent


def main():
    parser = argparse.ArgumentParser(
        description="Train IPMD with expert demonstrations"
    )
    parser.add_argument(
        "expert_data",
        type=str,
        help="Path to expert buffer file (expert_buffer.pt)",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="Environment name (auto-detected if not specified)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=100_000,
        help="Total training frames (default: 100,000)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device for training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=43,
        help="Random seed (default: 43, different from expert)",
    )
    parser.add_argument(
        "--expert-batch-size",
        type=int,
        default=256,
        help="Batch size for expert sampling (default: 256)",
    )

    args = parser.parse_args()

    train_ipmd(
        expert_data_path=args.expert_data,
        env_name=args.env,
        total_frames=args.frames,
        device=args.device,
        seed=args.seed,
        expert_batch_size=args.expert_batch_size,
    )


if __name__ == "__main__":
    main()
