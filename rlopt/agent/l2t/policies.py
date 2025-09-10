from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.distributions import (
    Distribution,
    DiagGaussianDistribution,
)
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import zip_strict
from torch import nn

from .moe import MoE

from sb3_contrib.common.recurrent.type_aliases import RNNStates


class RecurrentActorCriticPolicy(ActorCriticPolicy):
    """
    Recurrent policy class for actor-critic algorithms (has both policy and value prediction).
    To be used with A2C, PPO and the likes.
    It assumes that both the actor and the critic LSTM
    have the same architecture.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param lstm_hidden_size: Number of hidden units for each LSTM layer.
    :param n_lstm_layers: Number of LSTM layers.
    :param shared_lstm: Whether the LSTM is shared between the actor and the critic
        (in that case, only the actor gradient is used)
        By default, the actor and the critic have two separate LSTM.
    :param enable_critic_lstm: Use a seperate LSTM for the critic.
    :param lstm_kwargs: Additional keyword arguments to pass the the LSTM
        constructor.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.AdamW,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        lstm_hidden_size: int = 256,
        n_lstm_layers: int = 1,
        shared_lstm: bool = False,
        enable_critic_lstm: bool = True,
        lstm_kwargs: Optional[Dict[str, Any]] = None,
        # MoE config
        use_moe: bool = False,
        moe_num_experts: int = 4,
        moe_top_k: int = 2,
        moe_ffn_expand: float = 4.0,
        moe_dropout: float = 0.0,
        moe_router_jitter: float = 0.0,
        moe_sparse_dispatch: bool = False,
        moe_capacity_factor: float = 1.0,
    ):
        self.lstm_output_dim = lstm_hidden_size
        # MoE params must be set before super().__init__ because _build_mlp_extractor is called inside
        self.use_moe = use_moe
        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k
        self.moe_ffn_expand = moe_ffn_expand
        self.moe_dropout = moe_dropout
        self.moe_router_jitter = moe_router_jitter
        self.moe_sparse_dispatch = moe_sparse_dispatch
        self.moe_capacity_factor = moe_capacity_factor
        self.moe_aux_loss: th.Tensor | None = None

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )

        self.lstm_kwargs = lstm_kwargs or {}
        self.shared_lstm = shared_lstm
        self.enable_critic_lstm = enable_critic_lstm
        self.lstm_actor = nn.LSTM(
            self.features_dim,
            lstm_hidden_size,
            num_layers=n_lstm_layers,
            **self.lstm_kwargs,
        )
        # For the predict() method, to initialize hidden states
        # (n_lstm_layers, batch_size, lstm_hidden_size)
        self.lstm_hidden_state_shape = (n_lstm_layers, 1, lstm_hidden_size)
        self.critic = None
        self.lstm_critic = None
        assert not (
            self.shared_lstm and self.enable_critic_lstm
        ), "You must choose between shared LSTM, seperate or no LSTM for the critic."

        assert not (
            self.shared_lstm and not self.share_features_extractor
        ), "If the features extractor is not shared, the LSTM cannot be shared."

        # No LSTM for the critic, we still need to convert
        # output of features extractor to the correct size
        # (size of the output of the actor lstm)
        if not (self.shared_lstm or self.enable_critic_lstm):
            self.critic = nn.Linear(self.features_dim, lstm_hidden_size)

        # Use a separate LSTM for the critic
        if self.enable_critic_lstm:
            self.lstm_critic = nn.LSTM(
                self.features_dim,
                lstm_hidden_size,
                num_layers=n_lstm_layers,
                **self.lstm_kwargs,
            )

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

        self.action_space_low = th.Tensor(self.action_space.low)
        self.action_space_high = th.Tensor(self.action_space.high)

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        self.mlp_extractor = MlpExtractor(
            self.lstm_output_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )
        # If MoE is enabled, build an MoE block for the actor head.
        if self.use_moe:
            # In SB3, latent_dim_pi is set by MlpExtractor.
            actor_in = self.lstm_output_dim
            actor_out = getattr(
                self.mlp_extractor, "latent_dim_pi", self.lstm_output_dim
            )
            # Use MoE that maps d_model -> d_model; if needed, add a linear to actor_out
            # To keep compatibility with action_net input, produce actor_out dim
            # by making d_model == actor_out when possible. If mismatch, add a linear.
            d_model = actor_out
            if actor_in != actor_out:
                # project to actor_out before MoE, so MoE operates at the correct width
                self._actor_proj = nn.Linear(actor_in, actor_out)
            else:
                self._actor_proj = nn.Identity()

            self._actor_moe = MoE(
                d_model=d_model,
                num_experts=self.moe_num_experts,
                k=self.moe_top_k,
                ffn_expand=self.moe_ffn_expand,
                activation=self.activation_fn,
                dropout=self.moe_dropout,
                router_jitter=self.moe_router_jitter,
                sparse_dispatch=self.moe_sparse_dispatch,
                capacity_factor=self.moe_capacity_factor,
            )
        else:
            self._actor_proj = None
            self._actor_moe = None

    def _forward_actor_head(self, latent_pi: th.Tensor) -> th.Tensor:
        """Route actor latent through MLP or MoE depending on config, and store aux loss if any."""
        if self.use_moe and self._actor_moe is not None:
            # Ensure correct width for MoE/action head
            latent_pi = (
                self._actor_proj(latent_pi)
                if self._actor_proj is not None
                else latent_pi
            )
            latent_pi = self._actor_moe(latent_pi)
            # record aux loss for training (unscaled)
            self.moe_aux_loss = self._actor_moe.last_aux_loss
            return latent_pi
        # default: standard MLP extractor
        self.moe_aux_loss = None
        return self.mlp_extractor.forward_actor(latent_pi)

    @staticmethod
    def _process_sequence(
        features: th.Tensor,
        lstm_states: Tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
        lstm: nn.LSTM,
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        Do a forward pass in the LSTM network.

        :param features: Input tensor
        :param lstm_states: previous hidden and cell states of the LSTM, respectively
        :param episode_starts: Indicates when a new episode starts,
            in that case, we need to reset LSTM states.
        :param lstm: LSTM object.
        :return: LSTM output and updated LSTM states.
        """
        # LSTM logic
        # (sequence length, batch size, features dim)
        # (batch size = n_envs for data collection or n_seq when doing gradient update)
        n_seq = lstm_states[0].shape[1]
        # Batch to sequence
        # (padded batch size, features_dim) -> (n_seq, max length, features_dim) -> (max length, n_seq, features_dim)
        # note: max length (max sequence length) is always 1 during data collection
        features_sequence = features.reshape((n_seq, -1, lstm.input_size)).swapaxes(
            0, 1
        )
        episode_starts = episode_starts.reshape((n_seq, -1)).swapaxes(0, 1)

        # If we don't have to reset the state in the middle of a sequence
        # we can avoid the for loop, which speeds up things
        if th.all(episode_starts == 0.0):
            lstm_output, lstm_states = lstm(features_sequence, lstm_states)
            lstm_output = th.flatten(
                lstm_output.transpose(0, 1), start_dim=0, end_dim=1
            )
            return lstm_output, lstm_states

        lstm_output = []
        # Iterate over the sequence
        for features, episode_start in zip_strict(features_sequence, episode_starts):
            hidden, lstm_states = lstm(
                features.unsqueeze(dim=0),
                (
                    # Reset the states at the beginning of a new episode
                    (1.0 - episode_start).view(1, n_seq, 1) * lstm_states[0],
                    (1.0 - episode_start).view(1, n_seq, 1) * lstm_states[1],
                ),
            )
            lstm_output += [hidden]
        # Sequence to batch
        # (sequence length, n_seq, lstm_out_dim) -> (batch_size, lstm_out_dim)
        lstm_output = th.flatten(
            th.cat(lstm_output).transpose(0, 1), start_dim=0, end_dim=1
        )
        return lstm_output, lstm_states

    def forward(
        self,
        obs: th.Tensor,
        lstm_states: RNNStates,
        episode_starts: th.Tensor,
        deterministic: bool = False,
    ) -> RNNStates:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation. Observation
        :param lstm_states: The last hidden and memory states for the LSTM.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the lstm states in that case).
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            pi_features = vf_features = features  # alis
        else:
            pi_features, vf_features = features
        # latent_pi, latent_vf = self.mlp_extractor(features)
        latent_pi, lstm_states_pi = self._process_sequence(
            pi_features, lstm_states.pi, episode_starts, self.lstm_actor
        )
        if self.lstm_critic is not None:
            latent_vf, lstm_states_vf = self._process_sequence(
                vf_features, lstm_states.vf, episode_starts, self.lstm_critic
            )
        elif self.shared_lstm:
            # Re-use LSTM features but do not backpropagate
            latent_vf = latent_pi.detach()
            lstm_states_vf = (lstm_states_pi[0].detach(), lstm_states_pi[1].detach())
        else:
            # Critic only has a feedforward network
            latent_vf = self.critic(vf_features)
            lstm_states_vf = lstm_states_pi

        latent_pi = self._forward_actor_head(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob, RNNStates(lstm_states_pi, lstm_states_vf)

    def forward_lstm(
        self,
        obs: th.Tensor,
        lstm_states: RNNStates,
        episode_starts: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, RNNStates]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation. Observation
        :param lstm_states: The last hidden and memory states for the LSTM.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the lstm states in that case).
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            pi_features = vf_features = features  # alis
        else:
            pi_features, vf_features = features
        # latent_pi, latent_vf = self.mlp_extractor(features)
        latent_pi, lstm_states_pi = self._process_sequence(
            pi_features, lstm_states.pi, episode_starts, self.lstm_actor
        )
        if self.lstm_critic is not None:
            latent_vf, lstm_states_vf = self._process_sequence(
                vf_features, lstm_states.vf, episode_starts, self.lstm_critic
            )
        elif self.shared_lstm:
            # Re-use LSTM features but do not backpropagate
            latent_vf = latent_pi.detach()
            lstm_states_vf = (lstm_states_pi[0].detach(), lstm_states_pi[1].detach())
        else:
            # Critic only has a feedforward network
            latent_vf = self.critic(vf_features)
            lstm_states_vf = lstm_states_pi

        latent_pi = self._forward_actor_head(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        return RNNStates(lstm_states_pi, lstm_states_vf)

    def get_distribution(
        self,
        obs: th.Tensor,
        lstm_states: Tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
    ) -> Tuple[Distribution, Tuple[th.Tensor, ...]]:
        """
        Get the current policy distribution given the observations.

        :param obs: Observation.
        :param lstm_states: The last hidden and memory states for the LSTM.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the lstm states in that case).
        :return: the action distribution and new hidden states.
        """
        # Call the method from the parent of the parent class
        features = super(ActorCriticPolicy, self).extract_features(
            obs, self.pi_features_extractor
        )
        latent_pi, lstm_states = self._process_sequence(
            features, lstm_states, episode_starts, self.lstm_actor
        )

        latent_pi = self._forward_actor_head(latent_pi)
        return self._get_action_dist_from_latent(latent_pi), lstm_states

    def predict_values(
        self,
        obs: th.Tensor,
        lstm_states: Tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
    ) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation.
        :param lstm_states: The last hidden and memory states for the LSTM.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the lstm states in that case).
        :return: the estimated values.
        """
        # Call the method from the parent of the parent class
        features = super(ActorCriticPolicy, self).extract_features(
            obs, self.vf_features_extractor
        )

        if self.lstm_critic is not None:
            latent_vf, lstm_states_vf = self._process_sequence(
                features, lstm_states, episode_starts, self.lstm_critic
            )
        elif self.shared_lstm:
            # Use LSTM from the actor
            latent_pi, _ = self._process_sequence(
                features, lstm_states, episode_starts, self.lstm_actor
            )
            latent_vf = latent_pi.detach()
        else:
            latent_vf = self.critic(features)

        latent_vf = self.mlp_extractor.forward_critic(latent_vf)
        return self.value_net(latent_vf)

    def evaluate_actions(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        lstm_states: RNNStates,
        episode_starts: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation.
        :param actions:
        :param lstm_states: The last hidden and memory states for the LSTM.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the lstm states in that case).
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            pi_features = vf_features = features  # alias
        else:
            pi_features, vf_features = features
        latent_pi, _ = self._process_sequence(
            pi_features, lstm_states.pi, episode_starts, self.lstm_actor
        )
        if self.lstm_critic is not None:
            latent_vf, _ = self._process_sequence(
                vf_features, lstm_states.vf, episode_starts, self.lstm_critic
            )
        elif self.shared_lstm:
            latent_vf = latent_pi.detach()
        else:
            latent_vf = self.critic(vf_features)

        latent_pi = self._forward_actor_head(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()

    def evaluate_actions_whole_sequence(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions of batches of whole sequences according to the current policy,
        given the observations.

        :param obs: Observation.
        :param actions:
        :param lstm_states: The last hidden and memory states for the LSTM.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the lstm states in that case).
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed

        # temporary fix to disable the flattening that stable_baselines3 feature extractors do by default
        # flattening will turn the sequences in the batch into 1 long sequence without proper resetting of lstm hidden states
        if self.features_extractor_class == FlattenExtractor:
            features = obs
        else:
            features = self.extract_features(obs)
        latent_pi, _ = self.lstm_actor(features)

        if self.lstm_critic is not None:
            latent_vf, _ = self.lstm_critic(features)
        elif self.shared_lstm:
            latent_vf = latent_pi.detach()
        else:
            latent_vf = self.critic(features)

        latent_pi = self._forward_actor_head(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        values = self.value_net(latent_vf)

        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.distribution.log_prob(actions).sum(dim=-1)
        log_prob = log_prob.reshape((*log_prob.shape, 1))

        entropy = distribution.distribution.entropy().sum(dim=-1)
        entropy = entropy.reshape((*entropy.shape, 1))

        return values, log_prob, entropy

    def _predict(
        self,
        observation: th.Tensor,
        lstm_states: Tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
        deterministic: bool = False,
    ) -> Tuple[th.Tensor, Tuple[th.Tensor, ...]]:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param lstm_states: The last hidden and memory states for the LSTM.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the lstm states in that case).
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy and hidden states of the RNN
        """
        distribution, lstm_states = self.get_distribution(
            observation, lstm_states, episode_starts
        )
        return distribution.get_actions(deterministic=deterministic), lstm_states

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param lstm_states: The last hidden and memory states for the LSTM.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the lstm states in that case).
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        observation, vectorized_env = self.obs_to_tensor(observation)

        if isinstance(observation, dict):
            n_envs = observation[next(iter(observation.keys()))].shape[0]
        else:
            n_envs = observation.shape[0]
        # state : (n_layers, n_envs, dim)
        if state is None:
            # Initialize hidden states to zeros
            state = np.concatenate(
                [np.zeros(self.lstm_hidden_state_shape) for _ in range(n_envs)], axis=1
            )
            state = (state, state)

        if episode_start is None:
            episode_start = np.array([False for _ in range(n_envs)])

        with th.no_grad():
            # Convert to PyTorch tensors
            states = th.tensor(
                state[0], dtype=th.float32, device=self.device
            ), th.tensor(state[1], dtype=th.float32, device=self.device)
            episode_starts = th.tensor(
                episode_start, dtype=th.float32, device=self.device
            )

            actions, states = self._predict(
                observation,
                lstm_states=states,
                episode_starts=episode_starts,
                deterministic=deterministic,
            )
            states = (states[0].cpu().numpy(), states[1].cpu().numpy())

        # Convert to numpy
        actions = actions.cpu().numpy()

        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(
                    actions, self.action_space.low, self.action_space.high
                )

        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions.squeeze(axis=0)

        return actions, states

    def predict_and_return_tensor(
        self,
        observation: Union[th.Tensor, Dict[str, th.Tensor]],
        state: Optional[Tuple[th.Tensor, ...]] = None,
        episode_start: Optional[th.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[th.Tensor, Optional[Tuple[th.Tensor, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param lstm_states: The last hidden and memory states for the LSTM.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the lstm states in that case).
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        if self.action_space_low.device != self.device:
            self.action_space_low.to(self.device),
            self.action_space_high.to(self.device),

        if isinstance(observation, dict):
            n_envs = observation[next(iter(observation.keys()))].shape[0]
        else:
            n_envs = observation.shape[0]

        vectorized_env = n_envs > 1

        # state : (n_layers, n_envs, dim)
        if state is None:
            # Initialize hidden states to zeros
            state = th.concatenate(
                [th.zeros(self.lstm_hidden_state_shape) for _ in range(n_envs)], dim=1
            )
            state = (state, state)

        if episode_start is None:
            episode_start = th.tensor([False for _ in range(n_envs)])

        with th.no_grad():
            # Convert to PyTorch tensors
            states = (
                state[0].type(th.float32).to(self.device),
                state[1].type(th.float32).to(self.device),
            )

            episode_starts = episode_start.to(self.device)

            actions, states = self._predict(
                observation,
                lstm_states=states,
                episode_starts=episode_starts,
                deterministic=deterministic,
            )

        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = th.clamp(
                    actions,
                    self.action_space_low.to(self.device),
                    self.action_space_high.to(self.device),
                )

        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions.squeeze(dim=0)

        return actions, states

    def predict_whole_sequence(
        self, obs: th.Tensor, deterministic: bool = False, lstm_states: RNNStates = None
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Predict actions of batches of whole sequences according to the current policy,
        given the observations.

        :param obs: Observation (n_steps, n_seq, features_dim)
        :param deterministic: Whether to sample or use deterministic actions
        :return: estimated value, actions
            and entropy of the action distribution.
        """
        self.set_training_mode(True)

        if lstm_states is None:
            raise ValueError("lstm_states must be provided for recurrent policies")

        # Preprocess the observation if needed

        # temporary fix to disable the flattening that stable_baselines3 feature extractors do by default
        # flattening will turn the sequences in the batch into 1 long sequence without proper resetting of lstm hidden states
        if self.features_extractor_class == FlattenExtractor:
            features = obs
        else:
            features = self.extract_features(obs)
        latent_pi, _ = self.lstm_actor(features, (lstm_states.pi[0], lstm_states.pi[1]))

        if self.lstm_critic is not None:
            latent_vf, _ = self.lstm_critic(
                features, (lstm_states.vf[0], lstm_states.vf[1])
            )
        elif self.shared_lstm:
            latent_vf = latent_pi.detach()
        else:
            latent_vf = self.critic(features)

        latent_pi = self._forward_actor_head(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        # value shape (n_steps, n_seq, 1)
        values = self.value_net(latent_vf)

        # need to reshape actions to (n_steps * n_seq, n_actions)
        n_steps, n_seq, _ = latent_pi.shape
        # latent_pi = latent_pi.reshape((n_steps * n_seq, latent_pi.shape[-1]))
        distribution = self._get_action_dist_from_latent(latent_pi)
        distribution: DiagGaussianDistribution

        # actions shape (n_steps * n_seq, n_actions)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.distribution.log_prob(actions).sum(-1)
        log_prob = log_prob.unsqueeze(-1)

        entropy = distribution.distribution.entropy().sum(-1)

        mu, sigma = distribution.distribution.mean, distribution.distribution.stddev

        return values, actions, entropy, log_prob, mu, sigma


class RecurrentActorCriticCnnPolicy(RecurrentActorCriticPolicy):
    """
    CNN recurrent policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param lstm_hidden_size: Number of hidden units for each LSTM layer.
    :param n_lstm_layers: Number of LSTM layers.
    :param shared_lstm: Whether the LSTM is shared between the actor and the critic.
        By default, only the actor has a recurrent network.
    :param enable_critic_lstm: Use a seperate LSTM for the critic.
    :param lstm_kwargs: Additional keyword arguments to pass the the LSTM
        constructor.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        lstm_hidden_size: int = 256,
        n_lstm_layers: int = 1,
        shared_lstm: bool = False,
        enable_critic_lstm: bool = True,
        lstm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            lstm_hidden_size,
            n_lstm_layers,
            shared_lstm,
            enable_critic_lstm,
            lstm_kwargs,
        )


class RecurrentMultiInputActorCriticPolicy(RecurrentActorCriticPolicy):
    """
    MultiInputActorClass policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param lstm_hidden_size: Number of hidden units for each LSTM layer.
    :param n_lstm_layers: Number of LSTM layers.
    :param shared_lstm: Whether the LSTM is shared between the actor and the critic.
        By default, only the actor has a recurrent network.
    :param enable_critic_lstm: Use a seperate LSTM for the critic.
    :param lstm_kwargs: Additional keyword arguments to pass the the LSTM
        constructor.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = CombinedExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        lstm_hidden_size: int = 256,
        n_lstm_layers: int = 1,
        shared_lstm: bool = False,
        enable_critic_lstm: bool = True,
        lstm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            lstm_hidden_size,
            n_lstm_layers,
            shared_lstm,
            enable_critic_lstm,
            lstm_kwargs,
        )


MlpLstmPolicy = RecurrentActorCriticPolicy
CnnLstmPolicy = RecurrentActorCriticCnnPolicy
MultiInputLstmPolicy = RecurrentMultiInputActorCriticPolicy
