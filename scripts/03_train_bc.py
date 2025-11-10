#!/usr/bin/env python
"""Train Behavioral Cloning baseline with expert demonstrations."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tensordict import TensorDict

from rlopt.agent.imitation import IPMD, IPMDRLOptConfig
from rlopt.configs import NetworkConfig
from rlopt.env_utils import make_parallel_env

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def train_bc(
    expert_data_path: str,
    env_name: str | None = None,
    num_epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    device: str = "cuda:0",
    seed: int = 44,
    eval_episodes: int = 10,
    use_wandb: bool = True,
) -> tuple[nn.Module, float]:
    """Train Behavioral Cloning policy.

    Args:
        expert_data_path: Path to expert buffer (.pt file)
        env_name: Environment name (auto-detected from path if None)
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        device: Device for training
        seed: Random seed
        eval_episodes: Number of episodes for evaluation
        use_wandb: Whether to log to wandb

    Returns:
        Tuple of (trained policy, evaluation reward)
    """
    print("=" * 70)
    print("Behavioral Cloning Training")
    print("=" * 70)

    # Initialize wandb
    wandb_run = None
    if use_wandb and WANDB_AVAILABLE:
        wandb_run = wandb.init(
            project="RLOpt-BC",
            entity="fywu",
            group=env_name,
            name=f"BC_{env_name}_seed{seed}",
            config={
                "env_name": env_name,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "seed": seed,
                "eval_episodes": eval_episodes,
            },
        )
        print("✅ Wandb initialized")
    elif use_wandb and not WANDB_AVAILABLE:
        print("⚠️  Wandb not available, skipping logging")

    # Load expert data
    expert_path = Path(expert_data_path)
    if not expert_path.exists():
        raise FileNotFoundError(f"Expert data not found: {expert_path}")

    print(f"\nLoading expert data from: {expert_path}")
    expert_data = torch.load(expert_path, weights_only=False)

    # Load metadata
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

    # Extract observations and actions
    print(f"\nPreparing BC dataset...")
    observations = expert_data.get("observation").to(device)
    actions = expert_data.get("action").to(device)

    print(f"  - Observations shape: {observations.shape}")
    print(f"  - Actions shape: {actions.shape}")

    # Create dataset and dataloader
    dataset = TensorDataset(observations, actions)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create environment to get action spec
    print(f"\nCreating environment: {env_name}")
    cfg = IPMDRLOptConfig()
    cfg.env.env_name = env_name
    cfg.env.device = device
    cfg.device = device
    cfg.seed = seed

    from rlopt.env_utils import env_maker

    env = env_maker(cfg, device=device)

    obs_dim = observations.shape[-1]
    action_dim = actions.shape[-1]

    # Build BC policy (simple MLP)
    print(f"\n" + "=" * 70)
    print("Building BC Policy")
    print("=" * 70)

    from torchrl.modules import MLP

    policy = MLP(
        in_features=obs_dim,
        out_features=action_dim,
        num_cells=[256, 256],
        activation_class=nn.ReLU,
        device=device,
    )

    print(f"\nPolicy architecture:")
    print(f"  - Input: {obs_dim}")
    print(f"  - Hidden: [256, 256]")
    print(f"  - Output: {action_dim}")
    print(f"  - Parameters: {sum(p.numel() for p in policy.parameters()):,}")

    # Setup optimizer
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Training loop
    print(f"\n" + "=" * 70)
    print("Training BC Policy")
    print("=" * 70)
    print(f"\nTraining for {num_epochs} epochs...")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Dataset size: {len(dataset):,}")

    policy.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for obs_batch, action_batch in dataloader:
            # Forward pass
            pred_actions = policy(obs_batch)
            loss = criterion(pred_actions, action_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:3d}/{num_epochs}: Loss = {avg_loss:.6f}")

        # Log to wandb
        if wandb_run is not None:
            wandb.log({"train/loss": avg_loss, "train/epoch": epoch + 1})

    print(f"\n✅ BC Training Complete!")
    print(f"   Final loss: {avg_loss:.6f}")

    # Evaluate the policy
    print(f"\n" + "=" * 70)
    print("Evaluating BC Policy")
    print("=" * 70)

    policy.eval()
    episode_rewards = []

    print(f"\nRunning {eval_episodes} evaluation episodes...")

    with torch.no_grad():
        for ep in range(eval_episodes):
            td = env.reset()
            episode_reward = 0.0
            done = False
            steps = 0
            max_steps = 1000

            while not done and steps < max_steps:
                obs = td.get("observation").to(device)
                action = policy(obs)

                # Clip actions to environment bounds
                action = torch.clamp(
                    action,
                    env.action_spec.space.low.to(device),
                    env.action_spec.space.high.to(device),
                )

                td.set("action", action)
                td = env.step(td)

                reward = td.get(("next", "reward")).item()
                episode_reward += reward
                done = td.get(("next", "done")).item()

                # Update for next step
                td = env.step_mdp(td)
                steps += 1

            episode_rewards.append(episode_reward)
            print(
                f"  Episode {ep + 1:2d}: Reward = {episode_reward:8.2f} ({steps} steps)"
            )

    avg_reward = sum(episode_rewards) / len(episode_rewards)
    std_reward = torch.tensor(episode_rewards).std().item()

    print(f"\n✅ Evaluation Complete!")
    print(f"   Average reward: {avg_reward:.2f} ± {std_reward:.2f}")

    # Compare to expert if available
    expert_performance_ratio = None
    if metadata_path.exists() and metadata.get("final_reward"):
        expert_reward = metadata["final_reward"]
        expert_performance_ratio = (
            avg_reward / expert_reward if expert_reward != 0 else 0
        )
        print(f"   Expert reward: {expert_reward:.2f}")
        print(f"   Performance ratio: {expert_performance_ratio:.2%}")

    # Log final results to wandb
    if wandb_run is not None:
        log_dict = {
            "eval/reward_mean": avg_reward,
            "eval/reward_std": std_reward,
            "train/final_loss": avg_loss,
        }
        if expert_performance_ratio is not None:
            log_dict["eval/performance_vs_expert"] = expert_performance_ratio
        wandb.log(log_dict)
        wandb.finish()
        print("✅ Wandb logging complete")

    # Save the policy
    save_dir = Path("logs") / "bc" / env_name
    save_dir.mkdir(parents=True, exist_ok=True)

    model_path = save_dir / f"bc_policy_seed{seed}.pt"
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "final_loss": avg_loss,
            "eval_reward": avg_reward,
            "eval_std": std_reward,
            "num_epochs": num_epochs,
            "env_name": env_name,
        },
        model_path,
    )

    print(f"\n   Model saved: {model_path}")

    env.close()

    return policy, avg_reward


def main():
    parser = argparse.ArgumentParser(description="Train Behavioral Cloning baseline")
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
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size (default: 256)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate (default: 3e-4)",
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
        default=44,
        help="Random seed (default: 44)",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes (default: 10)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging",
    )

    args = parser.parse_args()

    train_bc(
        expert_data_path=args.expert_data,
        env_name=args.env,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        seed=args.seed,
        eval_episodes=args.eval_episodes,
        use_wandb=not args.no_wandb,
    )


if __name__ == "__main__":
    main()
