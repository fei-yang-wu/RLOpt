"""ProtoMotions-style shared discriminator and MI encoder for ASE."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from rlopt.utils import get_activation_class

DISC_LOGIT_INIT_SCALE = 1.0
ENC_LOGIT_INIT_SCALE = 0.1


class ASEDiscriminatorEncoder(nn.Module):
    """Shared observation trunk with conditional discriminator and MI encoder heads."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        latent_dim: int,
        hidden_dims: list[int] | None = None,
        activation: str = "elu",
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.latent_dim = int(latent_dim)
        self.obs_feature_dim = self.observation_dim - self.latent_dim
        if self.obs_feature_dim <= 0:
            msg = (
                "ASEDiscriminatorEncoder expects observation_dim > latent_dim so the "
                "observation-with-condition input can be split into obs and latent parts."
            )
            raise ValueError(msg)

        act_cls = get_activation_class(activation)
        layers: list[nn.Module] = []
        input_dim = self.obs_feature_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, int(hidden_dim)))
            layers.append(act_cls())
            input_dim = int(hidden_dim)
        self.trunk = nn.Sequential(*layers) if len(layers) > 0 else nn.Identity()
        self.feature_dim = input_dim

        self.disc_head = nn.Linear(self.feature_dim + self.latent_dim + self.action_dim, 1)
        self.enc_head = nn.Linear(self.feature_dim, self.latent_dim)

        torch.nn.init.uniform_(
            self.disc_head.weight,
            -DISC_LOGIT_INIT_SCALE,
            DISC_LOGIT_INIT_SCALE,
        )
        torch.nn.init.zeros_(self.disc_head.bias)
        torch.nn.init.uniform_(
            self.enc_head.weight,
            -ENC_LOGIT_INIT_SCALE,
            ENC_LOGIT_INIT_SCALE,
        )
        torch.nn.init.zeros_(self.enc_head.bias)

    def _split_observation(self, observation: Tensor) -> tuple[Tensor, Tensor]:
        obs_features = observation[..., : self.obs_feature_dim]
        latents = observation[..., self.obs_feature_dim :]
        return obs_features, latents

    def _trunk_features_from_obs(self, obs_features: Tensor) -> Tensor:
        return self.trunk(obs_features)

    def mi_encode_from_obs(self, obs_features: Tensor) -> Tensor:
        features = self._trunk_features_from_obs(obs_features)
        return F.normalize(self.enc_head(features), dim=-1, eps=1.0e-6)

    def mi_encode(self, observation: Tensor) -> Tensor:
        obs_features, _ = self._split_observation(observation)
        return self.mi_encode_from_obs(obs_features)

    def forward_logits(self, observation: Tensor, action: Tensor | None = None) -> Tensor:
        obs_features, latents = self._split_observation(observation)
        features = self._trunk_features_from_obs(obs_features)
        if action is None:
            action = torch.zeros(
                *features.shape[:-1],
                self.action_dim,
                dtype=features.dtype,
                device=features.device,
            )
        disc_input = torch.cat([features, latents, action], dim=-1)
        return self.disc_head(disc_input)

    def forward(self, observation: Tensor, action: Tensor | None = None) -> Tensor:
        return self.forward_logits(observation, action)

    def compute_mi_reward(
        self,
        obs_features: Tensor,
        latents: Tensor,
        mi_hypersphere_reward_shift: bool,
    ) -> Tensor:
        enc_pred = self.mi_encode_from_obs(obs_features)
        neg_err = -self.calc_von_mises_fisher_enc_error(enc_pred, latents).squeeze(-1)
        if mi_hypersphere_reward_shift:
            return (neg_err + 1.0) / 2.0
        return neg_err.clamp_min(0.0)

    @staticmethod
    def calc_von_mises_fisher_enc_error(enc_pred: Tensor, latent: Tensor) -> Tensor:
        return -(enc_pred * latent).sum(dim=-1, keepdim=True)

    def get_logit_layer(self) -> nn.Linear:
        return self.disc_head

    def all_weights(self) -> list[Tensor]:
        weights: list[Tensor] = []
        for module in self.modules():
            if isinstance(module, nn.Linear):
                weights.append(module.weight)
        return weights

    def all_discriminator_weights(self) -> list[Tensor]:
        weights: list[Tensor] = []
        for module in self.trunk.modules():
            if isinstance(module, nn.Linear):
                weights.append(module.weight)
        weights.append(self.disc_head.weight)
        return weights

    def logit_weights(self) -> list[Tensor]:
        return [self.disc_head.weight]

    def all_enc_weights(self) -> list[Tensor]:
        weights: list[Tensor] = []
        for module in self.trunk.modules():
            if isinstance(module, nn.Linear):
                weights.append(module.weight)
        weights.append(self.enc_head.weight)
        return weights

    def enc_weights(self) -> list[Tensor]:
        return [self.enc_head.weight]
