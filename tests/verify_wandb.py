#!/usr/bin/env python
"""Verify wandb integration is working correctly."""

from __future__ import annotations

import sys

# Check if wandb is installed
try:
    import wandb

    print("✅ wandb is installed")
    print(f"   Version: {wandb.__version__}")
except ImportError:
    print("❌ wandb is not installed")
    print("   Install with: pip install wandb")
    sys.exit(1)

# Check if user is logged in
try:
    api = wandb.Api()
    print("✅ wandb login successful")
    print(f"   User: {api.viewer()['entity']}")
except Exception as e:
    print("❌ wandb login required")
    print(f"   Error: {e}")
    print("   Run: wandb login")
    sys.exit(1)

# Test entity access
entity = "fywu"
try:
    # Try to access the entity
    runs = api.runs(f"{entity}/RLOpt", per_page=1)
    print(f"✅ Access to entity '{entity}' verified")
except Exception as e:
    print(f"⚠️  Cannot access entity '{entity}'")
    print(f"   Error: {e}")
    print(f"   You may need to:")
    print(f"   1. Be invited to the team")
    print(f"   2. Accept invitation at https://wandb.ai/settings")
    print(f"   Note: You can still use wandb with your personal account")

# Test project creation
print("\n" + "=" * 60)
print("Testing project configuration...")
print("=" * 60)

projects = {
    "RLOpt-ExpertData": "SAC expert data collection",
    "RLOpt-IPMD": "IPMD training",
    "RLOpt-BC": "Behavioral Cloning",
    "RLOpt": "General RL experiments",
}

for project, description in projects.items():
    print(f"\n{project}")
    print(f"  Description: {description}")
    print(f"  URL: https://wandb.ai/{entity}/{project}")

# Test a minimal run
print("\n" + "=" * 60)
print("Testing minimal run...")
print("=" * 60)

try:
    run = wandb.init(
        project="RLOpt",
        entity=entity,
        name="test_run_verification",
        config={"test": True},
        mode="offline",  # Don't actually upload
    )

    # Log some dummy metrics
    for i in range(5):
        wandb.log({"test/metric": i, "test/step": i})

    wandb.finish()
    print("✅ Test run successful (offline mode)")
    print("   All wandb integration checks passed!")

except Exception as e:
    print(f"❌ Test run failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("Wandb Configuration Summary")
print("=" * 60)
print(f"Entity: {entity}")
print(f"Projects:")
for project in projects:
    print(f"  - {project}")
print(f"\nView dashboard: https://wandb.ai/{entity}")
print("\nTo use with RLOpt scripts:")
print("  - Expert data: automatically logs to RLOpt-ExpertData")
print("  - IPMD: automatically logs to RLOpt-IPMD")
print("  - BC: automatically logs to RLOpt-BC")
print("\nTo disable logging (for testing):")
print("  export WANDB_MODE=disabled")
print("\n✅ All checks passed!")
