#!/usr/bin/env python
"""Collect expert demonstrations by training SAC and saving trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from rlopt.agent.rl import SAC, SACRLOptConfig
from rlopt.config_loader import apply_env_config, load_env_config
from rlopt.configs import NetworkConfig
from rlopt.env_utils import make_parallel_env


def collect_expert_data(
    env_name: str = "Pendulum-v1",
    total_frames: int | None = None,
    num_expert_episodes: int = 100,
    num_parallel_envs: int | None = None,
    save_dir: str = "expert_data",
    device: str = "cuda:0",
    seed: int = 42,
    compile: bool = True,
    cudagraphs: bool = False,
) -> Path:
    """Train SAC and collect expert demonstrations.

    Args:
        env_name: Environment to train on
        total_frames: Total training frames for SAC training
        num_expert_episodes: Number of expert episodes to collect
        num_parallel_envs: Number of parallel environments for training
        save_dir: Directory to save expert data
        device: Device for training
        seed: Random seed
        compile: Enable torch.compile (default: True)
        cudagraphs: Enable CUDAGraphs (default: False)

    Returns:
        Path to saved expert data
    """
    print("=" * 70)
    print("STEP 1: Training SAC Expert")
    print("=" * 70)

    # Load environment-specific configuration from YAML
    print(f"  📋 Loading configuration for {env_name}...")
    try:
        env_config = load_env_config(env_name)
        print(f"  ✅ Loaded config: {env_config.get('type', 'unknown')} environment")
    except FileNotFoundError as e:
        print(f"  ⚠️  {e}")
        print("  ⚠️  Using default configuration")
        env_config = {}

    # Configure SAC
    cfg = SACRLOptConfig()
    cfg.env.env_name = env_name
    cfg.env.device = device
    cfg.device = device

    # Compilation settings
    cfg.compile.compile = compile
    cfg.compile.compile_mode = "reduce-overhead" if compile else ""
    cfg.compile.cudagraphs = cudagraphs

    cfg.seed = seed
    cfg.logger.log_dir = "logs"
    cfg.logger.backend = "wandb"
    cfg.logger.project_name = "RLOpt-ExpertData"
    cfg.logger.entity = "fywu"
    cfg.logger.group_name = env_name
    cfg.logger.exp_name = f"SAC_Expert_{env_name}"

    # Network configuration (default, will be overridden by env config if specified)
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

    # Apply environment-specific config from YAML (overrides defaults)
    apply_env_config(cfg, env_config)

    # Override with command-line arguments if provided (highest priority)
    if total_frames:  # User specified frames
        cfg.collector.total_frames = total_frames
    if num_parallel_envs:  # User specified num_envs
        cfg.env.num_envs = num_parallel_envs

    # Display configuration
    print("  ⚙️  Configuration:")
    print(f"     - Total frames: {cfg.collector.total_frames:,}")
    print(f"     - UTD ratio: {cfg.sac.utd_ratio}")
    print(f"     - Replay buffer: {cfg.replay_buffer.size:,}")
    print(f"     - Parallel envs: {cfg.env.num_envs}")
    print(f"     - Init random: {cfg.collector.init_random_frames:,}")
    print(f"     - Network dims: {cfg.policy.num_cells}")
    print(f"     - Compile: {cfg.compile.compile}")
    print(f"     - CUDAGraphs: {cfg.compile.cudagraphs}")

    # Warn if settings seem suboptimal
    if env_config.get("type") == "mujoco" and cfg.collector.total_frames < 1_000_000:
        print("  ⚠️  Warning: Mujoco environments typically need 2-3M frames")
        print(
            f"  ⚠️  Current: {cfg.collector.total_frames:,}. Consider: --frames 3000000"
        )

    # Train SAC with parallel environments
    env = make_parallel_env(cfg)
    agent = SAC(env, cfg)

    print(
        f"\nTraining SAC on {cfg.env.env_name} for {cfg.collector.total_frames:,} frames..."
    )
    print(f"  - Parallel environments: {cfg.env.num_envs}")
    print(f"  - Log directory: {agent.log_dir}\n")

    agent.train()

    # Get final performance
    if len(agent.episode_rewards) > 0:
        rewards_list = list(agent.episode_rewards)
        final_reward = sum(rewards_list[-10:]) / min(10, len(rewards_list))
        print("\n✅ SAC Training Complete!")
        print(f"   Final reward (last 10 episodes): {final_reward:.2f}")
        print(f"   Total episodes: {len(agent.episode_rewards)}")

    # Save the trained model
    model_path = agent.log_dir / "sac_expert.pt"
    agent.save_model(model_path)
    print(f"   Model saved: {model_path}")

    # Training environment is already closed by collector.shutdown() in train()

    # Collect expert demonstrations from single environment
    print("\n" + "=" * 70)
    print("STEP 2: Collecting Expert Demonstrations")
    print("=" * 70)

    print(f"\nCollecting {num_expert_episodes} expert episodes...")
    print("  - Using single environment (for clean sequential data)")
    print("  - Trained policy in deterministic mode\n")

    # Create single environment for data collection
    from rlopt.env_utils import env_maker

    single_env = env_maker(cfg, device=device)

    # Collect episodes
    import torch

    all_episodes = []
    episode_rewards = []
    episode_lengths = []

    for ep in range(num_expert_episodes):
        episode_data = []
        td = single_env.reset()
        done = False
        steps = 0
        max_steps = 1000  # Safety limit

        while not done and steps < max_steps:
            # Get action from trained policy (deterministic)
            with torch.no_grad():
                td = agent.policy(td)
                action = td.get("action")

            # Store transition (before stepping)
            transition = td.clone()

            # Step environment
            td_next = single_env.step(td)

            # Extract done flag from the stepped tensordict
            done_flag = td_next.get("done")
            if done_flag is None:
                done_flag = td_next.get(("next", "done"))

            # Add next state info to transition
            # Note: RewardSum transform automatically adds "episode_reward" when done
            transition.set(("next", "observation"), td_next.get("observation"))
            transition.set(("next", "reward"), td_next.get(("next", "reward")))
            transition.set(("next", "done"), done_flag)
            transition.set(
                ("next", "terminated"),
                td_next.get("terminated", torch.zeros_like(done_flag)),
            )
            transition.set(
                ("next", "truncated"),
                td_next.get("truncated", torch.zeros_like(done_flag)),
            )

            # Copy episode_reward and step_count if present (added by transforms)
            if "episode_reward" in td_next.keys():
                transition.set("episode_reward", td_next.get("episode_reward"))
            if "step_count" in td_next.keys():
                transition.set("step_count", td_next.get("step_count"))

            episode_data.append(transition)

            done = done_flag.item() if done_flag is not None else False
            steps += 1

            # Update td for next iteration
            td = single_env.step_mdp(td_next)

        # Concatenate episode transitions
        if episode_data:
            episode_td = torch.stack(episode_data, dim=0)
            all_episodes.append(episode_td)

            # Get episode reward from the last transition (added by RewardSum when done)
            if "episode_reward" in episode_td.keys():
                final_reward = episode_td[-1].get("episode_reward").item()
            else:
                # Fallback: sum rewards manually
                final_reward = episode_td.get(("next", "reward")).sum().item()

            episode_rewards.append(final_reward)
            episode_lengths.append(steps)

        if (ep + 1) % 10 == 0 or ep == 0:
            print(
                f"  Episode {ep + 1:3d}/{num_expert_episodes}: "
                f"Reward = {episode_rewards[-1]:8.2f}, Steps = {steps:4d}"
            )

    single_env.close()

    # Concatenate all episodes
    print(f"\nCombining {len(all_episodes)} episodes...")
    expert_data = torch.cat(all_episodes, dim=0)
    num_samples = len(expert_data)

    print(f"  Total transitions: {num_samples:,}")
    print(f"  Average reward: {sum(episode_rewards) / len(episode_rewards):.2f}")
    print(f"  Average length: {sum(episode_lengths) / len(episode_lengths):.1f}")

    # Create directory for expert data
    save_path = Path(save_dir) / env_name
    save_path.mkdir(parents=True, exist_ok=True)

    # Save expert data
    buffer_file = save_path / "expert_buffer.pt"

    # Save as a replay buffer that can be loaded later
    torch.save(expert_data, buffer_file)

    print("\n✅ Expert data saved!")
    print(f"   Location: {buffer_file}")
    print(f"   Size: {num_samples:,} transitions")
    print(f"   Keys: {list(expert_data.keys())}")

    # Also save metadata
    metadata = {
        "env_name": env_name,
        "total_frames": cfg.collector.total_frames,
        "num_transitions": num_samples,
        "num_expert_episodes": num_expert_episodes,
        "expert_avg_reward": sum(episode_rewards) / len(episode_rewards),
        "expert_avg_length": sum(episode_lengths) / len(episode_lengths),
        "training_episodes": len(agent.episode_rewards),
        "final_reward": final_reward if len(agent.episode_rewards) > 0 else None,
        "seed": seed,
        "num_parallel_envs": cfg.env.num_envs,
        "model_path": str(model_path),
    }

    metadata_file = save_path / "metadata.pt"
    torch.save(metadata, metadata_file)
    print(f"   Metadata: {metadata_file}")

    # Create a summary file
    summary_file = save_path / "README.txt"
    with open(summary_file, "w") as f:
        f.write(f"Expert Data for {env_name}\n")
        f.write("=" * 50 + "\n\n")
        f.write("Generated by: 01_collect_expert_data.py\n")
        f.write(f"Environment: {env_name}\n\n")
        f.write("Training Configuration:\n")
        f.write(f"  - Training frames: {total_frames:,}\n")
        f.write(f"  - Parallel environments: {cfg.env.num_envs}\n")
        f.write(f"  - Seed: {seed}\n\n")
        f.write("Expert Data Collection:\n")
        f.write(f"  - Number of episodes: {num_expert_episodes}\n")
        f.write(f"  - Total transitions: {num_samples:,}\n")
        f.write(
            f"  - Average reward: {sum(episode_rewards) / len(episode_rewards):.2f}\n"
        )
        f.write(
            f"  - Average length: {sum(episode_lengths) / len(episode_lengths):.1f}\n"
        )
        f.write("  - Collection mode: Single environment, deterministic policy\n\n")
        f.write("Files:\n")
        f.write("  - expert_buffer.pt: TensorDict of expert transitions\n")
        f.write("  - metadata.pt: Training metadata\n")
        f.write(f"  - Model: {model_path}\n")

    print(f"   Summary: {summary_file}")

    print("\n" + "=" * 70)
    print("✅ Expert Data Collection Complete!")
    print("=" * 70)
    print(f"\nUse this data with IPMD by pointing to: {buffer_file}")

    return buffer_file


def main():
    parser = argparse.ArgumentParser(
        description="Collect expert demonstrations by training SAC"
    )
    parser.add_argument(
        "--env",
        type=str,
        default="Pendulum-v1",
        help="Environment name (default: Pendulum-v1)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Total training frames (default: None)",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=100,
        help="Number of expert episodes to collect (default: 100)",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="Number of parallel environments for training (default: None)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="expert_data",
        help="Directory to save expert data (default: expert_data)",
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
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        default=False,
        help="Enable torch.compile (default: False)",
    )
    parser.add_argument(
        "--cudagraphs",
        action="store_true",
        default=False,
        help="Enable CUDAGraphs (requires CUDA, default: False)",
    )

    args = parser.parse_args()

    collect_expert_data(
        env_name=args.env,
        total_frames=args.frames,
        num_expert_episodes=args.num_episodes,
        num_parallel_envs=args.num_envs,
        save_dir=args.save_dir,
        device=args.device,
        seed=args.seed,
        compile=args.compile,
        cudagraphs=args.cudagraphs,
    )


if __name__ == "__main__":
    main()
