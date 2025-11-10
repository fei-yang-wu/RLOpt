#!/usr/bin/env python
"""Compare SAC, IPMD, and BC on the same task."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


# Import script modules directly
def import_script(script_name):
    """Import a script module by file path."""
    script_dir = Path(__file__).parent
    script_file = script_dir / f"{script_name}.py"
    spec = importlib.util.spec_from_file_location(script_name, script_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Import the script modules
collect_expert_data = import_script("01_collect_expert_data")
train_ipmd = import_script("02_train_ipmd")
train_bc = import_script("03_train_bc")


def run_full_comparison(
    env_name: str = "Pendulum-v1",
    expert_frames: int = 100_000,
    ipmd_frames: int = 100_000,
    bc_epochs: int = 100,
    num_expert_episodes: int = 100,
    num_parallel_envs: int = 4,
    device: str = "cuda:0",
    save_dir: str = "comparison_results",
    expert_data_path: str | None = None,
) -> dict:
    """Run complete comparison of SAC, IPMD, and BC.

    Args:
        env_name: Environment to evaluate on
        expert_frames: Frames for expert (SAC) training
        ipmd_frames: Frames for IPMD training
        bc_epochs: Epochs for BC training
        num_expert_episodes: Number of expert episodes to collect
        num_parallel_envs: Number of parallel environments for training
        device: Device for training
        save_dir: Directory to save results
        expert_data_path: Optional path to existing expert data. If provided,
                         skips SAC training and uses this data directly.

    Returns:
        Dictionary with results
    """
    print("\n" + "=" * 70)
    print("FULL METHOD COMPARISON")
    print("=" * 70)
    print(f"\nEnvironment: {env_name}")
    print(f"Device: {device}")
    print("\nMethods to compare:")
    print("  1. SAC (Expert)")
    print("  2. IPMD (with reward estimation)")
    print("  3. BC (Behavioral Cloning)")

    results = {}

    # Step 1: Collect expert data (trains SAC) or use provided data
    if expert_data_path is not None:
        print("\n" + "=" * 70)
        print("STEP 1: Using Existing Expert Data")
        print("=" * 70)
        expert_buffer_path = Path(expert_data_path)
        if not expert_buffer_path.exists():
            raise FileNotFoundError(f"Expert data not found: {expert_data_path}")
        print(f"\n✅ Using expert data from: {expert_buffer_path}")

        # Load metadata to get expert reward
        metadata_path = expert_buffer_path.parent / "metadata.pt"
        if metadata_path.exists():
            metadata = torch.load(metadata_path, weights_only=False)
            expert_reward = metadata.get("average_reward", None)
            if expert_reward is not None:
                results["sac_expert"] = {"reward": expert_reward}
                print(f"   Expert reward: {expert_reward:.2f}")
    else:
        print("\n" + "=" * 70)
        print("STEP 1: Collecting Expert Data (SAC)")
        print("=" * 70)

        expert_buffer_path = collect_expert_data.collect_expert_data(
            env_name=env_name,
            total_frames=expert_frames,
            num_expert_episodes=num_expert_episodes,
            num_parallel_envs=num_parallel_envs,
            save_dir="expert_data",
            device=device,
            seed=42,
        )

        # Load expert metadata
        metadata_path = expert_buffer_path.parent / "metadata.pt"
        metadata = torch.load(metadata_path, weights_only=False)
        results["sac_expert"] = {
            "reward": metadata.get("final_reward"),
            "num_episodes": metadata.get("num_episodes"),
            "frames": expert_frames,
        }

        if results["sac_expert"]["reward"] is not None:
            print(f"\n✅ SAC Expert: {results['sac_expert']['reward']:.2f} reward")
        else:
            print(f"\n✅ SAC Expert: Training completed ({expert_frames:,} frames)")
            print(
                f"   ⚠️  No episodes completed (training too short or episodes too long)"
            )

    # Step 2: Train IPMD
    print("\n" + "=" * 70)
    print("STEP 2: Training IPMD")
    print("=" * 70)

    ipmd_agent = train_ipmd.train_ipmd(
        expert_data_path=str(expert_buffer_path),
        env_name=env_name,
        total_frames=ipmd_frames,
        device=device,
        seed=43,
    )

    if len(ipmd_agent.episode_rewards) > 0:
        # Convert deque to list for slicing
        rewards_list = list(ipmd_agent.episode_rewards)
        ipmd_reward = np.mean(rewards_list[-10:])
        results["ipmd"] = {
            "reward": ipmd_reward,
            "num_episodes": len(ipmd_agent.episode_rewards),
            "frames": ipmd_frames,
        }
        print(f"\n✅ IPMD: {ipmd_reward:.2f} reward")
    else:
        results["ipmd"] = {"reward": None}
        print(f"\n⚠️  IPMD: No episodes completed")

    # Step 3: Train BC
    print("\n" + "=" * 70)
    print("STEP 3: Training BC")
    print("=" * 70)

    bc_policy, bc_reward = train_bc.train_bc(
        expert_data_path=str(expert_buffer_path),
        env_name=env_name,
        num_epochs=bc_epochs,
        device=device,
        seed=44,
    )

    results["bc"] = {
        "reward": bc_reward,
        "epochs": bc_epochs,
    }

    print(f"\n✅ BC: {bc_reward:.2f} reward")

    # Generate comparison report
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    print(f"\nEnvironment: {env_name}")
    print("\nMethod Comparison:")
    print(f"{'Method':<20} {'Reward':>12} {'Training':>20}")
    print("-" * 55)

    sac_reward = results["sac_expert"]["reward"]

    if sac_reward is not None:
        print(f"{'SAC (Expert)':<20} {sac_reward:>12.2f} {expert_frames:>15,} frames")
    else:
        print(f"{'SAC (Expert)':<20} {'N/A':>12} {expert_frames:>15,} frames")
        print(f"{'  (no episodes)':<20} {'':>12} {'':>20}")

    if results["ipmd"]["reward"]:
        ipmd_reward = results["ipmd"]["reward"]
        if sac_reward is not None and sac_reward != 0:
            ipmd_pct = ipmd_reward / sac_reward * 100
            print(f"{'IPMD':<20} {ipmd_reward:>12.2f} {ipmd_frames:>15,} frames")
            print(f"{'  (vs Expert)':<20} {ipmd_pct:>11.1f}%")
        else:
            print(f"{'IPMD':<20} {ipmd_reward:>12.2f} {ipmd_frames:>15,} frames")

    bc_reward = results["bc"]["reward"]
    if sac_reward is not None and sac_reward != 0:
        bc_pct = bc_reward / sac_reward * 100
        print(f"{'BC':<20} {bc_reward:>12.2f} {bc_epochs:>15} epochs")
        print(f"{'  (vs Expert)':<20} {bc_pct:>11.1f}%")
    else:
        print(f"{'BC':<20} {bc_reward:>12.2f} {bc_epochs:>15} epochs")

    # Create visualization
    print(f"\n" + "=" * 70)
    print("Creating Visualization")
    print("=" * 70)

    save_path = Path(save_dir) / env_name
    save_path.mkdir(parents=True, exist_ok=True)

    # Bar plot
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = ["SAC\n(Expert)", "IPMD", "BC"]
    rewards = [
        sac_reward if sac_reward is not None else 0,
        results["ipmd"]["reward"] if results["ipmd"]["reward"] else 0,
        bc_reward,
    ]
    colors = ["#2ecc71", "#3498db", "#e74c3c"]

    bars = ax.bar(methods, rewards, color=colors, alpha=0.7, edgecolor="black")

    # Add value labels on bars
    for bar, reward, method in zip(bars, rewards, methods):
        height = bar.get_height()
        if reward == 0 and "Expert" in method and sac_reward is None:
            label = "N/A"
        else:
            label = f"{reward:.2f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height if height > 0 else 0,
            label,
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    # Add percentage labels (only if sac_reward is available)
    if results["ipmd"]["reward"] and sac_reward is not None and sac_reward != 0:
        ipmd_pct = results["ipmd"]["reward"] / sac_reward * 100
        ax.text(
            bars[1].get_x() + bars[1].get_width() / 2.0,
            bars[1].get_height() / 2,
            f"{ipmd_pct:.1f}%",
            ha="center",
            va="center",
            fontsize=10,
            color="white",
            fontweight="bold",
        )

    if sac_reward is not None and sac_reward != 0:
        bc_pct = bc_reward / sac_reward * 100
        ax.text(
            bars[2].get_x() + bars[2].get_width() / 2.0,
            bars[2].get_height() / 2,
            f"{bc_pct:.1f}%",
            ha="center",
            va="center",
            fontsize=10,
            color="white",
            fontweight="bold",
        )

    ax.set_ylabel("Average Reward", fontsize=14, fontweight="bold")
    ax.set_title(f"Method Comparison on {env_name}", fontsize=16, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plot_path = save_path / "comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\n✅ Plot saved: {plot_path}")

    # Save results
    results_path = save_path / "results.txt"
    with open(results_path, "w") as f:
        f.write(f"Method Comparison on {env_name}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"{'Method':<20} {'Reward':>12} {'Training':>20}\n")
        f.write("-" * 55 + "\n")

        if sac_reward is not None:
            f.write(
                f"{'SAC (Expert)':<20} {sac_reward:>12.2f} {expert_frames:>15,} frames\n"
            )
        else:
            f.write(f"{'SAC (Expert)':<20} {'N/A':>12} {expert_frames:>15,} frames\n")
            f.write(f"{'  (no episodes)':<20}\n")

        if results["ipmd"]["reward"]:
            ipmd_reward = results["ipmd"]["reward"]
            f.write(f"{'IPMD':<20} {ipmd_reward:>12.2f} {ipmd_frames:>15,} frames\n")
            if sac_reward is not None and sac_reward != 0:
                ipmd_pct = ipmd_reward / sac_reward * 100
                f.write(f"{'  (vs Expert)':<20} {ipmd_pct:>11.1f}%\n")

        f.write(f"{'BC':<20} {bc_reward:>12.2f} {bc_epochs:>15} epochs\n")
        if sac_reward is not None and sac_reward != 0:
            bc_pct = bc_reward / sac_reward * 100
            f.write(f"{'  (vs Expert)':<20} {bc_pct:>11.1f}%\n")

    print(f"✅ Results saved: {results_path}")

    print("\n" + "=" * 70)
    print("✅ COMPARISON COMPLETE!")
    print("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare SAC, IPMD, and BC")
    parser.add_argument(
        "--env",
        type=str,
        default="Pendulum-v1",
        help="Environment name (default: Pendulum-v1)",
    )
    parser.add_argument(
        "--expert-frames",
        type=int,
        default=100_000,
        help="Frames for expert training (default: 100,000)",
    )
    parser.add_argument(
        "--ipmd-frames",
        type=int,
        default=100_000,
        help="Frames for IPMD training (default: 100,000)",
    )
    parser.add_argument(
        "--bc-epochs",
        type=int,
        default=100,
        help="Epochs for BC training (default: 100)",
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
        default=4,
        help="Number of parallel environments for training (default: 4)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device for training",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="comparison_results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--expert-data",
        type=str,
        default=None,
        help="Path to existing expert data (e.g., expert_data/Pendulum-v1/expert_buffer.pt). "
        "If provided, skips SAC training and uses this data. Use 'latest' to automatically "
        "find the most recent expert data for the environment.",
    )

    args = parser.parse_args()

    # Handle 'latest' keyword for expert data
    expert_data_path = args.expert_data
    if expert_data_path == "latest":
        # Find the most recent expert data for this environment
        expert_dir = Path("expert_data") / args.env
        if expert_dir.exists() and (expert_dir / "expert_buffer.pt").exists():
            expert_data_path = str(expert_dir / "expert_buffer.pt")
            print(f"Using latest expert data: {expert_data_path}")
        else:
            print(
                f"No existing expert data found for {args.env}, will train new expert"
            )
            expert_data_path = None

    run_full_comparison(
        env_name=args.env,
        expert_frames=args.expert_frames,
        ipmd_frames=args.ipmd_frames,
        bc_epochs=args.bc_epochs,
        num_expert_episodes=args.num_episodes,
        num_parallel_envs=args.num_envs,
        device=args.device,
        save_dir=args.save_dir,
        expert_data_path=expert_data_path,
    )


if __name__ == "__main__":
    main()
