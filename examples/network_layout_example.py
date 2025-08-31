#!/usr/bin/env python3
"""
Example configuration using the new NetworkLayout for SAC and PPO agents.

This demonstrates how to configure network architectures using the new
NetworkLayout configuration instead of the legacy specific policy/feature configs.
"""

from rlopt.configs import (
    RLOptConfig,
    NetworkLayout,
    SharedFeatures,
    FeatureBlockSpec,
    MLPBlockConfig,
    ModuleNetConfig,
    CriticConfig,
    CNNBlockConfig,
)
from rlopt.agent.sac.sac import SACRLOptConfig, SACConfig
from rlopt.agent.ppo.ppo import PPORLOptConfig, PPOConfig


def create_sac_network_layout_example():
    """Create an example SAC configuration using NetworkLayout."""

    # Define shared feature extractor
    shared_features = SharedFeatures(
        features={
            "shared_torso": FeatureBlockSpec(
                type="mlp",
                mlp=MLPBlockConfig(
                    num_cells=[512, 256],
                    activation="elu",
                    init="orthogonal",
                    layer_norm=False,
                    dropout=0.0,
                ),
                output_dim=256,
            )
        }
    )

    # Define policy module
    policy_module = ModuleNetConfig(
        feature_ref="shared_torso",  # Reference shared feature
        head=MLPBlockConfig(num_cells=[128, 64], activation="elu", init="orthogonal"),
        in_keys=["hidden"],
        out_key="hidden",
    )

    # Define critic module (Q-function ensemble)
    critic_module = CriticConfig(
        template=ModuleNetConfig(
            feature_ref="shared_torso",  # Reference shared feature
            head=MLPBlockConfig(
                num_cells=[128, 64], activation="elu", init="orthogonal"
            ),
            in_keys=["hidden"],
            out_key="hidden",
        ),
        num_nets=2,  # Twin Q-networks
        shared_feature_ref="shared_torso",
        use_target=True,
        target_update="polyak",
        polyak_eps=0.995,
    )

    # Create network layout
    network_layout = NetworkLayout(
        shared=shared_features, policy=policy_module, critic=critic_module
    )

    # Create SAC config
    sac_config = SACRLOptConfig(
        network=network_layout,
        use_feature_extractor=True,
        use_value_function=False,  # SAC uses Q-functions, not value functions
        # Other config options...
    )

    return sac_config


def create_ppo_network_layout_example():
    """Create an example PPO configuration using NetworkLayout."""

    # Define shared feature extractor
    shared_features = SharedFeatures(
        features={
            "shared_torso": FeatureBlockSpec(
                type="mlp",
                mlp=MLPBlockConfig(
                    num_cells=[512, 256],
                    activation="elu",
                    init="orthogonal",
                    layer_norm=False,
                    dropout=0.0,
                ),
                output_dim=256,
            )
        }
    )

    # Define policy module
    policy_module = ModuleNetConfig(
        feature_ref="shared_torso",  # Reference shared feature
        head=MLPBlockConfig(num_cells=[128, 64], activation="elu", init="orthogonal"),
        in_keys=["hidden"],
        out_key="hidden",
    )

    # Define value module
    value_module = ModuleNetConfig(
        feature_ref="shared_torso",  # Reference shared feature
        head=MLPBlockConfig(num_cells=[128, 64], activation="elu", init="orthogonal"),
        in_keys=["hidden"],
        out_key="hidden",
    )

    # Create network layout
    network_layout = NetworkLayout(
        shared=shared_features, policy=policy_module, value=value_module
    )

    # Create PPO config
    ppo_config = PPORLOptConfig(
        network=network_layout,
        use_feature_extractor=True,
        use_value_function=True,  # PPO uses value functions
        # Other config options...
    )

    return ppo_config


def create_advanced_network_layout_example():
    """Create an advanced example with multiple feature extractors and different architectures."""

    # Define multiple shared feature extractors
    shared_features = SharedFeatures(
        features={
            "vision_torso": FeatureBlockSpec(
                type="cnn",
                cnn=CNNBlockConfig(
                    channels=[32, 64, 128],
                    kernels=[8, 4, 3],
                    strides=[4, 2, 1],
                    paddings=[0, 1, 1],
                    activation="relu",
                ),
                output_dim=512,
            ),
            "proprio_torso": FeatureBlockSpec(
                type="mlp",
                mlp=MLPBlockConfig(
                    num_cells=[256, 128], activation="elu", init="orthogonal"
                ),
                output_dim=128,
            ),
            "shared_torso": FeatureBlockSpec(
                type="mlp",
                mlp=MLPBlockConfig(
                    num_cells=[512, 256], activation="elu", init="orthogonal"
                ),
                output_dim=256,
            ),
        }
    )

    # Define policy with private feature extractor
    policy_module = ModuleNetConfig(
        feature=FeatureBlockSpec(  # Private feature extractor
            type="mlp",
            mlp=MLPBlockConfig(
                num_cells=[256, 128], activation="elu", init="orthogonal"
            ),
            output_dim=128,
        ),
        head=MLPBlockConfig(num_cells=[64, 32], activation="elu", init="orthogonal"),
        in_keys=["hidden"],
        out_key="hidden",
    )

    # Define value with shared feature extractor
    value_module = ModuleNetConfig(
        feature_ref="shared_torso",  # Reference shared feature
        head=MLPBlockConfig(num_cells=[128, 64], activation="elu", init="orthogonal"),
        in_keys=["hidden"],
        out_key="hidden",
    )

    # Create network layout
    network_layout = NetworkLayout(
        shared=shared_features, policy=policy_module, value=value_module
    )

    return network_layout


if __name__ == "__main__":
    print("SAC Network Layout Example:")
    sac_config = create_sac_network_layout_example()
    print(f"  - Shared features: {list(sac_config.network.shared.features.keys())}")
    print(f"  - Policy feature ref: {sac_config.network.policy.feature_ref}")
    print(f"  - Critic num nets: {sac_config.network.critic.num_nets}")

    print("\nPPO Network Layout Example:")
    ppo_config = create_ppo_network_layout_example()
    print(f"  - Shared features: {list(ppo_config.network.shared.features.keys())}")
    print(f"  - Policy feature ref: {ppo_config.network.policy.feature_ref}")
    print(f"  - Value feature ref: {ppo_config.network.value.feature_ref}")

    print("\nAdvanced Network Layout Example:")
    advanced_layout = create_advanced_network_layout_example()
    print(f"  - Shared features: {list(advanced_layout.shared.features.keys())}")
    print(
        f"  - Policy has private feature: {advanced_layout.policy.feature is not None}"
    )
    print(f"  - Value uses shared feature: {advanced_layout.value.feature_ref}")
