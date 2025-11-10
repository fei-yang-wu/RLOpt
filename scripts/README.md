# RLOpt Training Scripts

This directory contains scripts for training and comparing RL and imitation learning methods.

## Workflow

### Full Pipeline (All Methods)

Run the complete comparison with a single command:

```bash
cd scripts
python 04_compare_methods.py --env Pendulum-v1 --expert-frames 100000 --ipmd-frames 100000 --bc-epochs 100
```

This will:
1. Train SAC expert and collect demonstrations
2. Train IPMD with reward estimation
3. Train Behavioral Cloning baseline
4. Generate comparison plots and results

### Step-by-Step

#### 1. Collect Expert Demonstrations

Train a SAC expert and save trajectories:

```bash
python 01_collect_expert_data.py \
    --env Pendulum-v1 \
    --frames 100000 \
    --device cuda:0 \
    --seed 42
```

**Output:**
- `expert_data/{env_name}/expert_buffer.pt` - Expert trajectories
- `expert_data/{env_name}/metadata.pt` - Training metadata
- `logs/sac/{env_name}/{timestamp}/sac_expert.pt` - Trained model

#### 2. Train IPMD

Train IPMD with reward estimation using the expert data:

```bash
python 02_train_ipmd.py \
    expert_data/Pendulum-v1/expert_buffer.pt \
    --frames 100000 \
    --device cuda:0 \
    --expert-batch-size 256
```

**Output:**
- `logs/ipmd/{env_name}/{timestamp}/ipmd_model.pt` - Trained model
- Training logs and metrics

**Key Features:**
- Uses estimated rewards from reward network
- Alternates between reward estimation and policy optimization
- Expert data is sampled in batches during training

#### 3. Train Behavioral Cloning

Train a BC baseline using the same expert data:

```bash
python 03_train_bc.py \
    expert_data/Pendulum-v1/expert_buffer.pt \
    --epochs 100 \
    --batch-size 256 \
    --lr 3e-4 \
    --device cuda:0
```

**Output:**
- `logs/bc/{env_name}/bc_policy_seed*.pt` - Trained policy
- Evaluation results

**Key Features:**
- Simple supervised learning (MSE loss)
- No environment interaction during training
- Fast training (epochs instead of environment steps)

## Weights & Biases Integration

All scripts automatically log to Weights & Biases under the `fywu` entity:
- **Expert Data**: Project `RLOpt-ExpertData`
- **IPMD**: Project `RLOpt-IPMD`
- **BC**: Project `RLOpt-BC`

**Setup:**
```bash
wandb login
```

**Disable wandb (for testing):**
```bash
export WANDB_MODE=disabled
# OR for BC specifically:
python scripts/03_train_bc.py ... --no-wandb
```

See `WANDB_INTEGRATION.md` for full documentation.

## Scripts

### `01_collect_expert_data.py`

Train SAC and collect expert demonstrations.

**Arguments:**
- `--env`: Environment name (default: Pendulum-v1)
- `--frames`: Total training frames (default: 100,000)
- `--save-dir`: Directory to save expert data (default: expert_data)
- `--device`: Training device (default: cuda:0)
- `--seed`: Random seed (default: 42)

**Wandb:** Logs to `RLOpt-ExpertData` project

**Example:**
```bash
python 01_collect_expert_data.py --env HalfCheetah-v5 --frames 500000
```

### `02_train_ipmd.py`

Train IPMD with expert demonstrations and reward estimation.

**Arguments:**
- `expert_data`: Path to expert buffer file (required)
- `--env`: Environment name (auto-detected if not specified)
- `--frames`: Total training frames (default: 100,000)
- `--device`: Training device (default: cuda:0)
- `--seed`: Random seed (default: 43)
- `--expert-batch-size`: Batch size for expert sampling (default: 256)

**Wandb:** Logs to `RLOpt-IPMD` project

**Example:**
```bash
python 02_train_ipmd.py expert_data/HalfCheetah-v5/expert_buffer.pt --frames 500000
```

### `03_train_bc.py`

Train Behavioral Cloning baseline.

**Arguments:**
- `expert_data`: Path to expert buffer file (required)
- `--env`: Environment name (auto-detected if not specified)
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size (default: 256)
- `--lr`: Learning rate (default: 3e-4)
- `--device`: Training device (default: cuda:0)
- `--seed`: Random seed (default: 44)
- `--eval-episodes`: Number of evaluation episodes (default: 10)
- `--no-wandb`: Disable wandb logging

**Wandb:** Logs to `RLOpt-BC` project

**Example:**
```bash
python 03_train_bc.py expert_data/HalfCheetah-v5/expert_buffer.pt --epochs 200
```

### `04_compare_methods.py`

Run complete comparison of all methods.

**Arguments:**
- `--env`: Environment name (default: Pendulum-v1)
- `--expert-frames`: Frames for expert training (default: 100,000)
- `--ipmd-frames`: Frames for IPMD training (default: 100,000)
- `--bc-epochs`: Epochs for BC training (default: 100)
- `--device`: Training device (default: cuda:0)
- `--save-dir`: Directory to save results (default: comparison_results)

**Example:**
```bash
python 04_compare_methods.py --env HalfCheetah-v5 --expert-frames 500000 --ipmd-frames 500000
```

## Output Structure

```
.
├── expert_data/
│   └── {env_name}/
│       ├── expert_buffer.pt      # Expert trajectories
│       ├── metadata.pt           # Training info
│       └── README.txt            # Summary
│
├── logs/
│   ├── sac/{env_name}/{timestamp}/
│   │   ├── sac_expert.pt
│   │   └── rlopt.log
│   │
│   ├── ipmd/{env_name}/{timestamp}/
│   │   ├── ipmd_model.pt
│   │   └── rlopt.log
│   │
│   └── bc/{env_name}/
│       └── bc_policy_seed*.pt
│
└── comparison_results/
    └── {env_name}/
        ├── comparison.png        # Bar plot
        └── results.txt           # Text summary
```

## Quick Start Examples

### Quick Test (Pendulum, Fast)

```bash
# Complete pipeline (< 10 minutes on GPU)
python 04_compare_methods.py \
    --env Pendulum-v1 \
    --expert-frames 50000 \
    --ipmd-frames 50000 \
    --bc-epochs 50
```

### Full Evaluation (HalfCheetah)

```bash
# Complete pipeline (~1-2 hours on GPU)
python 04_compare_methods.py \
    --env HalfCheetah-v5 \
    --expert-frames 500000 \
    --ipmd-frames 500000 \
    --bc-epochs 200
```

### Reuse Expert Data

If you've already collected expert data:

```bash
# Train only IPMD
python 02_train_ipmd.py expert_data/Pendulum-v1/expert_buffer.pt

# Train only BC
python 03_train_bc.py expert_data/Pendulum-v1/expert_buffer.pt

# Or both:
python 02_train_ipmd.py expert_data/Pendulum-v1/expert_buffer.pt &
python 03_train_bc.py expert_data/Pendulum-v1/expert_buffer.pt &
wait
```

## Method Comparison

### SAC (Expert)
- **Type**: Off-policy RL
- **Training**: Environment interaction only
- **Pros**: Strong baseline, no demonstrations needed
- **Cons**: Requires many environment samples

### IPMD (Inverse Preference-based Model Distillation)
- **Type**: Inverse RL
- **Training**: Learns reward from expert + policy optimization
- **Pros**: Can learn beyond expert, robust to sub-optimal data
- **Cons**: More complex, requires environment interaction

### BC (Behavioral Cloning)
- **Type**: Supervised learning
- **Training**: Direct policy mimicking (no environment needed)
- **Pros**: Fast, simple, sample efficient
- **Cons**: Limited by expert quality, no improvement possible

## Expected Performance

On **Pendulum-v1** (100K frames each):
- **SAC Expert**: ~-200 to -150 reward
- **IPMD**: 80-100% of expert (after convergence)
- **BC**: 60-80% of expert (limited by distributional shift)

On **HalfCheetah-v5** (500K frames each):
- **SAC Expert**: ~5000-7000 reward
- **IPMD**: 70-90% of expert (can improve with more training)
- **BC**: 40-60% of expert (suffers from distributional shift)

## Troubleshooting

### "Expert buffer not found"
Make sure to run step 1 first to collect expert data.

### "No episodes completed"
Increase `--frames` for IPMD or use a simpler environment for testing.

### CUDA out of memory
Reduce batch sizes:
- `--expert-batch-size` for IPMD
- `--batch-size` for BC

### Poor BC performance
This is expected due to distributional shift. BC is included as a baseline to show IPMD's advantages.

## Integration with ImitationLearningTools

To use `ImitationLearningTools.ExpertReplayManager`:

```python
from ImitationLearningTools.replay import ExpertReplayManager
from rlopt.imitation import ExpertReplayBuffer

# Create manager
manager = ExpertReplayManager(
    buffer_dir="path/to/buffer",
    env=env,
    batch_size=256,
)

# Wrap in RLOpt's interface
expert_buffer = ExpertReplayBuffer(manager)

# Use with IPMD
agent.set_expert_buffer(expert_buffer)
```

## Citations

If you use IPMD, please cite:
```
@inproceedings{ipmd2024,
  title={Inverse Preference-based Model Distillation},
  author={...},
  booktitle={...},
  year={2024}
}
```

## See Also

- `../tests/test_ipmd_baseline.py` - Unit tests for IPMD
- `../tests/test_ipmd_components.py` - Component tests
- `../IPMD_BASELINE_TEST.md` - Baseline testing documentation
- `../LOGGING_STRUCTURE.md` - Logging documentation

