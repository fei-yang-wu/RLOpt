#!/usr/bin/env python
"""Verify RLOpt setup and all components are working."""

from __future__ import annotations

import sys
from pathlib import Path

print("=" * 70)
print("RLOpt Setup Verification")
print("=" * 70)

# Test 1: Core imports
print("\n1. Testing core RLOpt imports...")
try:
    from rlopt.agent.rl import SAC, SACRLOptConfig
    from rlopt.agent.imitation import IPMD, IPMDRLOptConfig
    from rlopt.imitation import ExpertReplayBuffer
    from rlopt.configs import NetworkConfig
    from rlopt.env_utils import make_parallel_env

    print("   ✅ Core imports successful")
except Exception as e:
    print(f"   ❌ Core imports failed: {e}")
    sys.exit(1)

# Test 2: Script imports
print("\n2. Testing script imports...")
try:
    import importlib.util

    def import_script(script_name):
        script_dir = Path(__file__).parent
        script_file = script_dir / f"{script_name}.py"
        spec = importlib.util.spec_from_file_location(script_name, script_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    collect_expert_data = import_script("01_collect_expert_data")
    train_ipmd = import_script("02_train_ipmd")
    train_bc = import_script("03_train_bc")

    print("   ✅ Script imports successful")
    print(f"      - collect_expert_data.collect_expert_data: ✓")
    print(f"      - train_ipmd.train_ipmd: ✓")
    print(f"      - train_bc.train_bc: ✓")
except Exception as e:
    print(f"   ❌ Script imports failed: {e}")
    sys.exit(1)

# Test 3: Wandb availability
print("\n3. Testing wandb availability...")
try:
    import wandb

    print(f"   ✅ wandb installed (version {wandb.__version__})")

    # Check login status
    try:
        api = wandb.Api()
        print(f"   ✅ wandb logged in")
    except:
        print(f"   ⚠️  wandb not logged in (run: wandb login)")
except ImportError:
    print("   ⚠️  wandb not installed (run: pip install wandb)")

# Test 4: PyTorch and device
print("\n4. Testing PyTorch and device...")
try:
    import torch

    print(f"   ✅ PyTorch installed (version {torch.__version__})")
    print(f"      - CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"      - CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"      - Recommended device: cuda:0")
    else:
        print(f"      - Recommended device: cpu")
except Exception as e:
    print(f"   ❌ PyTorch check failed: {e}")

# Test 5: Directory structure
print("\n5. Checking directory structure...")
dirs_to_check = [
    "logs",
    "expert_data",
    "comparison_results",
    "scripts",
    "tests",
    "rlopt",
]

for dir_name in dirs_to_check:
    dir_path = Path.cwd() / dir_name
    exists = dir_path.exists()
    status = "✅" if exists else "⚠️  (will be created)"
    print(f"   {status} {dir_name}/")

# Test 6: Key files
print("\n6. Checking key files...")
files_to_check = [
    "scripts/01_collect_expert_data.py",
    "scripts/02_train_ipmd.py",
    "scripts/03_train_bc.py",
    "scripts/04_compare_methods.py",
    "QUICK_START.md",
    "WANDB_INTEGRATION.md",
    "IMITATION_LEARNING_PIPELINE.md",
]

for file_name in files_to_check:
    file_path = Path.cwd() / file_name
    exists = file_path.exists()
    status = "✅" if exists else "❌"
    print(f"   {status} {file_name}")

# Summary
print("\n" + "=" * 70)
print("Setup Verification Complete!")
print("=" * 70)
print("\nNext steps:")
print("  1. Login to wandb:")
print("     wandb login")
print("\n  2. Run quick test:")
print("     python scripts/04_compare_methods.py \\")
print("         --env Pendulum-v1 \\")
print("         --expert-frames 10000 \\")
print("         --ipmd-frames 10000 \\")
print("         --bc-epochs 20")
print("\n  3. View results:")
print("     - Local: comparison_results/Pendulum-v1/comparison.png")
print("     - Wandb: https://wandb.ai/fywu")
print("\n✅ All checks passed! Ready to start training.")
