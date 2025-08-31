# Network Layout Refactoring

This document describes the refactoring of SAC and PPO implementations to use the new `NetworkLayout` configuration system instead of the legacy specific policy/feature configs.

## Overview

The refactoring introduces a more flexible and modular network architecture configuration system that allows:

- **Shared feature extractors** across multiple network components
- **Flexible activation functions** and **weight initialization** strategies
- **Support for different feature types** (MLP, LSTM, CNN)
- **Backward compatibility** with existing configurations

## Key Changes

### 1. SAC Implementation (`rlopt/agent/sac/sac.py`)

- **Feature Extractor**: Now uses `NetworkLayout.shared.features` when available
- **Policy Network**: Uses `NetworkLayout.policy.head` configuration
- **Q-Function**: Uses `NetworkLayout.critic.template.head` configuration
- **Critic Configuration**: Reads `num_nets` and `polyak_eps` from `NetworkLayout.critic`

### 2. PPO Implementation (`rlopt/agent/ppo/ppo.py`)

- **Feature Extractor**: Now uses `NetworkLayout.shared.features` when available
- **Policy Network**: Uses `NetworkLayout.policy.head` configuration
- **Value Network**: Uses `NetworkLayout.value.head` configuration

### 3. New Helper Methods

Both implementations now include:

- `_get_activation_class()`: Maps activation names to PyTorch classes
- `_initialize_weights()`: Applies different weight initialization strategies

## Configuration Structure

### NetworkLayout

```python
@dataclass
class NetworkLayout:
    shared: SharedFeatures          # Registry of shared feature extractors
    policy: ModuleNetConfig        # Policy network configuration
    value: Optional[ModuleNetConfig] = None      # Value network (PPO)
    critic: Optional[CriticConfig] = None       # Critic configuration (SAC)
```

### SharedFeatures

```python
@dataclass
class SharedFeatures:
    features: dict[str, FeatureBlockSpec]  # Named feature extractors
```

### ModuleNetConfig

```python
@dataclass
class ModuleNetConfig:
    feature_ref: Optional[str] = None      # Reference to shared feature
    feature: Optional[FeatureBlockSpec] = None  # Private feature extractor
    head: Optional[MLPBlockConfig] = None      # Network head configuration
    in_keys: list[str] = ["hidden"]           # Input tensor keys
    out_key: str = "hidden"                   # Output tensor key
```

### FeatureBlockSpec

```python
@dataclass
class FeatureBlockSpec:
    type: Literal["mlp", "lstm", "cnn"] = "mlp"
    mlp: Optional[MLPBlockConfig] = None
    lstm: Optional[LSTMBlockConfig] = None
    cnn: Optional[CNNBlockConfig] = None
    output_dim: int = 128
```

## Usage Examples

### Basic SAC Configuration

```python
from rlopt.configs import NetworkLayout, SharedFeatures, FeatureBlockSpec, MLPBlockConfig

# Define shared feature extractor
shared_features = SharedFeatures(
    features={
        "shared_torso": FeatureBlockSpec(
            type="mlp",
            mlp=MLPBlockConfig(
                num_cells=[512, 256],
                activation="elu",
                init="orthogonal"
            ),
            output_dim=256
        )
    }
)

# Define policy and critic modules
network_layout = NetworkLayout(
    shared=shared_features,
    policy=ModuleNetConfig(
        feature_ref="shared_torso",
        head=MLPBlockConfig(num_cells=[128, 64])
    ),
    critic=CriticConfig(
        template=ModuleNetConfig(
            feature_ref="shared_torso",
            head=MLPBlockConfig(num_cells=[128, 64])
        ),
        num_nets=2
    )
)

# Use in SAC config
sac_config = SACRLOptConfig(network=network_layout)
```

### Basic PPO Configuration

```python
network_layout = NetworkLayout(
    shared=shared_features,
    policy=ModuleNetConfig(
        feature_ref="shared_torso",
        head=MLPBlockConfig(num_cells=[128, 64])
    ),
    value=ModuleNetConfig(
        feature_ref="shared_torso",
        head=MLPBlockConfig(num_cells=[128, 64])
    )
)

ppo_config = PPORLOptConfig(network=network_layout)
```

## Backward Compatibility

The refactoring maintains full backward compatibility:

1. **Legacy configs still work**: If `config.network` is `None`, the old config structure is used
2. **Gradual migration**: You can migrate one component at a time
3. **Fallback behavior**: New system gracefully falls back to legacy configs

## Benefits

1. **Modularity**: Easier to share feature extractors between components
2. **Flexibility**: Support for different activation functions and initialization strategies
3. **Maintainability**: Centralized network configuration
4. **Extensibility**: Easy to add new feature types (LSTM, CNN)
5. **Vectorization**: Better support for vectorized operations as per user rules

## Migration Guide

### From Legacy to NetworkLayout

1. **Identify shared features**: Look for common feature extractor configurations
2. **Create SharedFeatures**: Define named feature extractors
3. **Update ModuleNetConfig**: Reference shared features or define private ones
4. **Set NetworkLayout**: Add to your config
5. **Test**: Verify behavior matches legacy configuration

### Example Migration

**Before (Legacy)**:
```python
config = SACRLOptConfig(
    feature_extractor=FeatureExtractorConfig(
        num_cells=[512, 256],
        output_dim=128
    ),
    policy=PolicyConfig(num_cells=[128, 64]),
    action_value_net=ActionValueNetConfig(num_cells=[128, 64])
)
```

**After (NetworkLayout)**:
```python
config = SACRLOptConfig(
    network=NetworkLayout(
        shared=SharedFeatures(
            features={
                "shared": FeatureBlockSpec(
                    type="mlp",
                    mlp=MLPBlockConfig(num_cells=[512, 256]),
                    output_dim=128
                )
            }
        ),
        policy=ModuleNetConfig(
            feature_ref="shared",
            head=MLPBlockConfig(num_cells=[128, 64])
        ),
        critic=CriticConfig(
            template=ModuleNetConfig(
                feature_ref="shared",
                head=MLPBlockConfig(num_cells=[128, 64])
            )
        )
    )
)
```

## Testing

The refactoring includes comprehensive testing to ensure:

- **Backward compatibility** with existing configurations
- **Correct network construction** using new layout system
- **Proper weight initialization** and activation functions
- **Feature sharing** between policy and value/critic networks

Run the test suite to verify everything works correctly:

```bash
python -m pytest tests/ -v
```
