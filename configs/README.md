# Environment Configurations

This directory contains environment-specific hyperparameters for optimal SAC training.

## Structure

```
configs/
└── env/
    ├── pendulum.yaml          # Simple environments
    ├── halfcheetah.yaml       # HalfCheetah-v5 optimized
    ├── walker2d.yaml          # Walker2d-v5 optimized  
    ├── hopper.yaml            # Hopper-v5 optimized
    ├── mujoco_base.yaml       # Base config for all Mujoco envs
    └── ...                    # Add more as needed
```

## Configuration Format

Each YAML file contains:

```yaml
name: EnvironmentName-v5
type: simple|mujoco|atari

training:
  total_frames: 1_000_000      # Total training frames
  frames_per_batch: 1000       # Frames per batch
  init_random_frames: 5000     # Random exploration steps

replay_buffer:
  size: 1_000_000              # Replay buffer capacity

loss:
  mini_batch_size: 256         # Batch size for updates

sac:
  utd_ratio: 4.0               # Updates-to-data ratio (CRITICAL!)

parallel:
  num_envs: 20                 # Parallel environments

network:
  hidden_dims: [256, 256]      # Network architecture
  activation: relu

expert_collection:
  num_episodes: 50
  expected_reward: 3000        # Expected expert performance
  min_reward: 1000             # Minimum acceptable
```

## Usage

### Automatic Loading

Scripts automatically load configs based on environment name:

```bash
# Automatically uses configs/env/halfcheetah.yaml
python scripts/01_collect_expert_data.py --env HalfCheetah-v5

# Automatically uses configs/env/pendulum.yaml
python scripts/01_collect_expert_data.py --env Pendulum-v1
```

### Override with CLI

Command-line arguments override YAML configs:

```bash
# Use HalfCheetah config but override frames
python scripts/01_collect_expert_data.py \
    --env HalfCheetah-v5 \
    --frames 5000000 \           # Override
    --num-envs 30                # Override
```

### Programmatic Usage

```python
from rlopt.config_loader import load_env_config, apply_env_config
from rlopt.agent.rl import SACRLOptConfig

# Load environment config
env_config = load_env_config("HalfCheetah-v5")

# Create base config
cfg = SACRLOptConfig()

# Apply environment-specific settings
apply_env_config(cfg, env_config)

# Now cfg has optimal settings for HalfCheetah
print(f"UTD ratio: {cfg.sac.utd_ratio}")  # 4.0
```

## Adding New Environments

1. Create `configs/env/myenv.yaml`:

```yaml
name: MyEnv-v1
type: custom

training:
  total_frames: 500_000
  frames_per_batch: 1000
  init_random_frames: 2000
  
replay_buffer:
  size: 500_000
  
sac:
  utd_ratio: 2.0  # Tune this!
  
parallel:
  num_envs: 8
```

2. Use it:

```bash
python scripts/01_collect_expert_data.py --env MyEnv-v1
```

## Key Parameters Explained

### UTD Ratio (Most Important!)
- **Updates-To-Data ratio**: gradient updates per environment step
- **Simple envs**: 1.0 (Pendulum, CartPole)
- **Mujoco envs**: 4.0-8.0 (HalfCheetah, Walker, Hopper)
- **Higher = more learning per sample** but slower wall-clock time

### Replay Buffer Size
- **Simple envs**: 100k
- **Mujoco envs**: 1M
- Larger = more diverse samples

### Init Random Frames
- **Simple envs**: 1k
- **Mujoco envs**: 5k
- More = better initial exploration

### Parallel Environments
- **Simple envs**: 4-8
- **Mujoco envs**: 16-20
- More = faster data collection

## Available Configs

List all available configs:

```python
from rlopt.config_loader import list_available_configs

configs = list_available_configs()
print(configs)  # ['halfcheetah', 'hopper', 'pendulum', 'walker2d']
```

## Environment Type Detection

The config loader automatically detects environment types:

- **simple**: Pendulum, CartPole, MountainCar
- **mujoco**: HalfCheetah, Walker, Hopper, Ant, Humanoid
- **atari**: Breakout, Pong, SpaceInvaders
- **unknown**: Falls back to mujoco_base or pendulum

## Best Practices

1. **Start with provided configs** - They're tuned based on research
2. **Override carefully** - Only change what you need
3. **Document changes** - Add comments explaining why
4. **Test thoroughly** - Verify performance matches expectations
5. **Share successful configs** - Help others by adding new YAML files

## References

- SAC paper hyperparameters
- TorchRL examples
- Empirical testing on RLOpt framework


