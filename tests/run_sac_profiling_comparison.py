#!/usr/bin/env python3
"""
SAC Performance Profiling Comparison Script

This script runs both RLOpt and TorchRL SAC implementations with detailed profiling
to compare their performance characteristics.
"""

import os
import sys
import subprocess
import time
from pathlib import Path


def run_rlopt_sac():
    """Run RLOpt SAC implementation with profiling."""
    print("=" * 80)
    print("RUNNING RLOpt SAC IMPLEMENTATION")
    print("=" * 80)

    # Change to RLOpt directory
    rlopt_dir = Path("/home/fwu91/Documents/Research/SkillLearning/RLOpt")
    os.chdir(rlopt_dir)

    # Run the test with profiling
    cmd = [
        "python",
        "-m",
        "pytest",
        "tests/test_sac_halfcheetah_full.py::test_sac_halfcheetah_v5_full_wandb",
        "-v",
        "-s",
        "--tb=short",
        "--run-full-halfcheetah",  # Enable the full test
    ]

    start_time = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300
        )  # 5 minute timeout
        end_time = time.time()

        print(f"RLOpt SAC completed in {end_time - start_time:.2f} seconds")
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)

        return result.returncode == 0, result.stdout, result.stderr

    except subprocess.TimeoutExpired:
        print("RLOpt SAC timed out after 5 minutes")
        return False, "", "Timeout"
    except Exception as e:
        print(f"Error running RLOpt SAC: {e}")
        return False, "", str(e)


def run_torchrl_sac():
    """Run TorchRL SAC implementation with profiling."""
    print("=" * 80)
    print("RUNNING TorchRL SAC IMPLEMENTATION")
    print("=" * 80)

    # Change to TorchRL directory
    torchrl_dir = Path(
        "/home/fwu91/Documents/Research/SkillLearning/torchrl/sota-implementations/sac"
    )
    os.chdir(torchrl_dir)

    # Create a simple config for quick testing
    config_content = """
defaults:
  - _self_

# Environment
env:
  env_name: HalfCheetah-v4
  num_envs: 8
  max_episode_steps: 1000
  seed: 42

# Collector
collector:
  frames_per_batch: 1000
  total_frames: 10000  # Reduced for quick profiling
  init_random_frames: 1000
  device: cuda:0

# Network
network:
  device: cuda:0
  hidden_sizes: [256, 256]
  default_policy_scale: 1.0
  scale_lb: 1e-4

# Optimizer
optim:
  lr: 3e-4
  batch_size: 256
  utd_ratio: 1.0
  target_update_polyak: 0.995

# Replay Buffer
replay_buffer:
  size: 100000
  prb: false
  scratch_dir: null

# Loss
loss:
  gamma: 0.99

# Logger
logger:
  backend: null
  exp_name: sac_profiling
  eval_iter: 1000

# Compile
compile:
  compile: false
  compile_mode: null
  cudagraphs: false
"""

    # Write config file
    with open("config_profiling.yaml", "w") as f:
        f.write(config_content)

    # Run the TorchRL SAC
    cmd = ["python", "sac.py", "--config-name=config_profiling"]

    start_time = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300
        )  # 5 minute timeout
        end_time = time.time()

        print(f"TorchRL SAC completed in {end_time - start_time:.2f} seconds")
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)

        return result.returncode == 0, result.stdout, result.stderr

    except subprocess.TimeoutExpired:
        print("TorchRL SAC timed out after 5 minutes")
        return False, "", "Timeout"
    except Exception as e:
        print(f"Error running TorchRL SAC: {e}")
        return False, "", str(e)


def main():
    """Main function to run both implementations and compare results."""
    print("SAC Performance Profiling Comparison")
    print("=" * 80)

    # Check if we're in the right conda environment
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "unknown")
    print(f"Current conda environment: {conda_env}")

    if conda_env != "SkillLearning":
        print("WARNING: Not in SkillLearning conda environment!")
        print("Please activate the SkillLearning environment first:")
        print("conda activate SkillLearning")
        return

    results = {}

    # Run RLOpt SAC
    print("\n" + "=" * 80)
    print("STARTING RLOpt SAC PROFILING")
    print("=" * 80)
    rlopt_success, rlopt_stdout, rlopt_stderr = run_rlopt_sac()
    results["rlopt"] = {
        "success": rlopt_success,
        "stdout": rlopt_stdout,
        "stderr": rlopt_stderr,
    }

    # Run TorchRL SAC
    print("\n" + "=" * 80)
    print("STARTING TorchRL SAC PROFILING")
    print("=" * 80)
    torchrl_success, torchrl_stdout, torchrl_stderr = run_torchrl_sac()
    results["torchrl"] = {
        "success": torchrl_success,
        "stdout": torchrl_stdout,
        "stderr": torchrl_stderr,
    }

    # Summary
    print("\n" + "=" * 80)
    print("PROFILING COMPARISON SUMMARY")
    print("=" * 80)

    print(f"RLOpt SAC: {'SUCCESS' if rlopt_success else 'FAILED'}")
    print(f"TorchRL SAC: {'SUCCESS' if torchrl_success else 'FAILED'}")

    if rlopt_success and torchrl_success:
        print("\nBoth implementations completed successfully!")
        print("Check the output above for detailed timing comparisons.")
        print("Look for timing differences in:")
        print("- Data collection")
        print("- Loss computation")
        print("- Backward pass")
        print("- Optimizer step")
        print("- Buffer operations")
        print("- Total iteration time")
    else:
        print("\nOne or both implementations failed. Check the error messages above.")

    # Save results to files for later analysis
    with open("rlopt_profiling_output.txt", "w") as f:
        f.write("RLOpt SAC Profiling Output\n")
        f.write("=" * 50 + "\n")
        f.write(f"Success: {rlopt_success}\n")
        f.write("STDOUT:\n")
        f.write(rlopt_stdout)
        f.write("\nSTDERR:\n")
        f.write(rlopt_stderr)

    with open("torchrl_profiling_output.txt", "w") as f:
        f.write("TorchRL SAC Profiling Output\n")
        f.write("=" * 50 + "\n")
        f.write(f"Success: {torchrl_success}\n")
        f.write("STDOUT:\n")
        f.write(torchrl_stdout)
        f.write("\nSTDERR:\n")
        f.write(torchrl_stderr)

    print(f"\nDetailed outputs saved to:")
    print(f"- rlopt_profiling_output.txt")
    print(f"- torchrl_profiling_output.txt")


if __name__ == "__main__":
    main()
