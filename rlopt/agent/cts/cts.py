import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
import gymnasium as gym
from collections import deque

from rlopt.common import DictRolloutBuffer


class CTSObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, stack_size):
        super().__init__(env)
        self.stack_size = stack_size
        self.observation_space = env.observation_space
        self.stacked_obs = deque(maxlen=stack_size)
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Initialize stack with first observation repeated
        self.stacked_obs.clear()
        for _ in range(self.stack_size):
            self.stacked_obs.append(th.zeros_like(obs["student"]))
        # Add stacked observations to obs dict
        obs["stacked_obs"] = th.cat(list(self.stacked_obs), dim=-1)
        return obs, info
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        assert isinstance(obs, dict), "Observations must be a dictionary"
        assert "student" in obs, "Student observations must be present"
        
        # Update stacked observations
        self.stacked_obs.append(th.as_tensor(obs["student"]))
        # Add stacked observations to obs dict
        obs["stacked_obs"] = th.cat(list(self.stacked_obs), dim=-1)
        return obs, reward, terminated, truncated, info


class CTSPolicy(ActorCriticPolicy):
    def __init__(self, *args, latent_dim=32, stack_size=3, **kwargs):
        super().__init__(*args, **kwargs)
        # privileged encoder on obs["teacher"]
        teacher_dim = self.observation_space["teacher"].shape[0]
        self.privileged_encoder = nn.Sequential(
            nn.Linear(teacher_dim, 128), nn.ELU(),
            nn.Linear(128, latent_dim),     nn.ELU(),
        )
        # proprio encoder on obs["student"] - now handles stacked observations
        student_dim = self.observation_space["student"].shape[0]
        student_dim *= stack_size
        self.proprio_encoder   = nn.Sequential(
            nn.Linear(student_dim, 128), nn.ELU(),
            nn.Linear(128, latent_dim),  nn.ELU(),
        )
        # force SB3 to rebuild its heads on latent_dim
        self.features_dim = latent_dim
        self._build_mlp_extractor()

        self.stack_size = stack_size

    def forward(self, obs, deterministic=False):
        # Validate input
        assert isinstance(obs, dict), "Observations must be a dictionary"
        assert "teacher" in obs and "student" in obs, "Teacher and student observations must be present"
        assert "teacher_mask" in obs, "Teacher mask must be present"
        
        mask = obs["teacher_mask"].float().unsqueeze(-1)  # [batch,1]
        z_t = self.privileged_encoder(obs["teacher"])
        
        # Handle stacked observations
        if "stacked_obs" in obs:
            # Validate stacked observations shape
            expected_shape = (obs["student"].shape[0], obs["student"].shape[1] * self.stack_size)
            assert obs["stacked_obs"].shape == expected_shape, \
                f"Stacked observations shape {obs['stacked_obs'].shape} does not match expected shape {expected_shape}"
            z_s = self.proprio_encoder(obs["stacked_obs"])
        else:
            # If no stacked observations, repeat the current observation
            repeated_obs = obs["student"].repeat(1, self.stack_size)
            z_s = self.proprio_encoder(repeated_obs)
            
        z = mask * z_t + (1 - mask) * z_s

        # standard SB3 actor-critic heads
        latent_pi, latent_vf = self.mlp_extractor(z)
        dist = self._get_action_dist_from_latent(latent_pi)
        actions = dist.get_actions(deterministic=deterministic)
        log_prob = dist.log_prob(actions)
        values = self.value_net(latent_vf)
        return actions, values, log_prob

    def _predict(self, obs, deterministic=False):
        return self.forward(obs, deterministic)[0]


class CTSPPO(PPO):
    """
    PPO with Concurrent Teacher-Student updates, now with fixed 3:1 teacher:student assignment.
    """
    def __init__(
        self,
        policy,
        env,
        *,
        teacher_ratio: float = 0.75,
        **ppo_kwargs
    ):
        # Validate teacher ratio
        assert 0 < teacher_ratio < 1, "Teacher ratio must be between 0 and 1"
        # store ratio before calling super()
        self.teacher_ratio = teacher_ratio
        super().__init__(policy, env, **ppo_kwargs)

    def _setup_model(self) -> None:
        super()._setup_model()
        # Re-compute rollout buffer to use CTSRolloutBuffer
        self.rollout_buffer = DictRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            normalize_advantage=self.normalize_advantage,
        )
        # Initialize teacher mask
        self._update_teacher_mask()

    def _update_teacher_mask(self) -> None:
        """Update the teacher-student assignments."""
        n_envs = self.env.num_envs
        n_teacher = int(self.teacher_ratio * n_envs)
        mask = np.zeros(n_envs, dtype=np.bool_)
        mask[:n_teacher] = True
        np.random.shuffle(mask)  # mix which envs are teacher
        # register as a tensor on the same device as the buffer
        self._teacher_mask = th.tensor(mask, device=self.device)

    def collect_rollouts(
        self,
        env,
        callback,
        n_rollout_steps: int,
    ) -> bool:
        assert isinstance(self.rollout_buffer, DictRolloutBuffer)
        self.rollout_buffer.reset()

        # reset the env and inject mask
        obs, info = env.reset()
        # Update teacher mask periodically (every n_rollout_steps)
        self._update_teacher_mask()
        # ---- inject teacher_mask ----
        # We assume obs is a dict of np.arrays; append teacher_mask
        obs = {**obs, "teacher_mask": self._teacher_mask.cpu().numpy()}

        # get initial value
        rollout_data = self.rollout_buffer
        for step in range(n_rollout_steps):
            # convert to tensor dict
            obs_tensor = {
                k: th.as_tensor(v).to(self.device)
                for k, v in obs.items()
            }
            # Add stacked obs to tensor dict if available
            if "stacked_obs" in info:
                obs_tensor["stacked_obs"] = th.as_tensor(info["stacked_obs"]).to(self.device)
                
            # predict actions and values
            actions, values, log_probs = self.policy.forward(obs_tensor)

            # step
            next_obs, rewards, dones, infos = env.step(actions.cpu().numpy())
            # inject the SAME mask for all steps
            next_obs = {**next_obs, "teacher_mask": self._teacher_mask.cpu().numpy()}

            # add to buffer
            rollout_data.add(
                obs_tensor,
                {k: th.as_tensor(v).to(self.device) for k, v in next_obs.items()},
                actions,
                rewards,
                dones,
                values,
                log_probs,
                info=infos[0]  # Pass info dict to store stacked obs
            )

            obs = next_obs
            info = infos[0]

        # after collecting, compute last value and GAE
        # convert last obs
        with th.no_grad():
            last_obs_tensor = {
                k: th.as_tensor(v).to(self.device)
                for k, v in obs.items()
            }
            if "stacked_obs" in info:
                last_obs_tensor["stacked_obs"] = th.as_tensor(info["stacked_obs"]).to(self.device)
            last_values = self.policy.forward(last_obs_tensor)[1]

        self.rollout_buffer.compute_returns_and_advantage(last_values, dones)
        return True

    def train(self) -> None:
        self._update_learning_rate(self.policy.optimizer)
        for epoch in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                m = rollout_data.teacher_mask
                ratio = th.exp(rollout_data.old_log_prob - rollout_data.log_prob)
                adv   = rollout_data.advantages

                # split
                ratio_t, adv_t = ratio[m], adv[m]
                ratio_s, adv_s = ratio[~m], adv[~m]

                clip_range = self.clip_range()
                loss_ppo_t = th.min(ratio_t * adv_t,
                                   th.clamp(ratio_t, 1-clip_range, 1+clip_range) * adv_t
                                  ).mean()
                loss_ppo_s = th.min(ratio_s * adv_s,
                                   th.clamp(ratio_s, 1-clip_range, 1+clip_range) * adv_s
                                  ).mean()

                value_loss   = F.mse_loss(rollout_data.values, rollout_data.returns)
                entropy_loss = rollout_data.entropy.mean()

                # reconstruction between encoders
                z_t = self.policy.privileged_encoder(rollout_data.obs["teacher"])
                # Use stacked observations for proprio encoder if available
                if hasattr(rollout_data, 'stacked_obs') and rollout_data.stacked_obs is not None:
                    stacked_obs = rollout_data.obs["stacked_obs"]
                    z_s = self.policy.proprio_encoder(stacked_obs)
                else:
                    z_s = self.policy.proprio_encoder(rollout_data.obs["student"])
                loss_rec = F.mse_loss(z_s, z_t)

                # total
                loss = - (loss_ppo_t + loss_ppo_s) \
                       + self.vf_coef * value_loss \
                       - self.ent_coef * entropy_loss \
                       + 1.0 * loss_rec

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

        self._n_updates += 1