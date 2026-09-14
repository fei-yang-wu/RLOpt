# RLOpt: A Research Framework for Reinforcement Learning

RLOpt is a flexible and modular framework for Reinforcement Learning (RL) research, built on PyTorch and TorchRL. It is designed to facilitate the implementation, testing, and comparison of various RL agents and optimization techniques. The framework uses dataclass-based configs for library code and Hydra for experiment scripts, allowing convenient customization.

## PPO observation normalization

`ppo.normalizer_update_mode="minibatch"` retains the existing update schedule.
The opt-in `"rollout"` mode holds actor and critic normalization statistics
fixed during collection and all optimization epochs, then updates each
normalizer once from the full rollout. The following rollout uses the new
statistics. This avoids changing the likelihood calculation through repeated
normalizer updates on the same data. Network gradients remain enabled.

KL measurement uses evaluation mode and deterministic actions, then restores
all module training flags. It must not change normalization buffers or consume
action-sampling randomness. PPO startup shape probes also run in evaluation
mode, so synthetic observations do not enter the running statistics.

These consistency changes do not bound the first gradient update or guarantee
task improvement. Learning-rate and task-outcome checks remain necessary.
