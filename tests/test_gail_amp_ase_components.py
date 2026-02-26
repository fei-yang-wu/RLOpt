from __future__ import annotations

import torch

from rlopt.agent.ase.ase import ASEConfig
from rlopt.agent.ase.multi_discriminator import MultiDiscriminator
from rlopt.agent.gail.discriminator import Discriminator
from rlopt.agent.gail.gail import GAILConfig


def test_discriminator_reward_and_regularizer_helpers() -> None:
    discriminator = Discriminator(
        observation_dim=6,
        action_dim=3,
        hidden_dims=[32, 32],
    )

    obs = torch.randn(16, 6)
    action = torch.randn(16, 3)

    logits = discriminator.forward_logits(obs, action)
    probs = discriminator.predict_proba(obs, action)
    reward = discriminator.compute_reward(obs, action)

    assert logits.shape == (16, 1)
    assert probs.shape == (16, 1)
    assert reward.shape == (16,)
    assert torch.isfinite(reward).all()

    manual_reward = -torch.log(1.0 - torch.sigmoid(logits).squeeze(-1) + 1e-8)
    assert torch.allclose(reward, manual_reward, atol=1e-6)

    all_weights = discriminator.all_weights()
    assert len(all_weights) > 0
    assert all(weight.ndim >= 2 for weight in all_weights)


def test_multi_discriminator_batched_path() -> None:
    discriminator = MultiDiscriminator(
        num_skills=4,
        observation_dim=5,
        action_dim=2,
        hidden_dims=[16, 16],
    )

    obs = torch.randn(32, 5)
    action = torch.randn(32, 2)
    skill_idx = torch.randint(0, 4, (32,))

    logits = discriminator.forward_logits(obs, action, skill_idx)
    probs = discriminator.forward(obs, action, skill_idx)
    reward = discriminator.compute_reward(obs, action, skill_idx)

    assert logits.shape == (32, 1)
    assert probs.shape == (32, 1)
    assert reward.shape == (32, 1)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(reward).all()


def test_config_aliases_map_to_new_fields() -> None:
    gail_cfg = GAILConfig(
        discriminator_input_key="invrwd",
        discriminator_hidden_dim=128,
        discriminator_num_layers=3,
        env_reward_proportion=0.2,
    )
    assert gail_cfg.discriminator_input_keys == ["invrwd"]
    assert gail_cfg.discriminator_hidden_dims == [128, 128, 128]
    assert gail_cfg.proportion_env_reward == 0.2

    ase_cfg = ASEConfig(num_skills=12, diversity_coeff=0.15)
    assert ase_cfg.latent_dim == 12
    assert ase_cfg.diversity_bonus_coeff == 0.15
