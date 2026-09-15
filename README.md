# RLOpt: A Research Framework for Reinforcement Learning

RLOpt is a flexible and modular framework for Reinforcement Learning (RL) research, built on PyTorch and TorchRL. It is designed to facilitate the implementation, testing, and comparison of various RL agents and optimization techniques. The framework uses dataclass-based configs for library code and Hydra for experiment scripts, allowing convenient customization.

### Factorized affine PoE DiffSR

`phi_parameterization="affine_poe"` retains the affine parameter shapes and
state-dict keys, but contracts `[1; z]` with factorized expert fields.
`feature_dim` remains the internal factor width; the returned phi and mu
field axis have `action_dim + 1` entries. `forward_mu` requires source `s`
for this mode even with target-only mu conditioning. This is an algebraic
reassociation, not a new capacity setting; floating-point operation order
can differ. The fixed coordinate is internal, not an extra policy command.

### Distributed PPO qualification

`rlopt.agent.ppo.distributed.DistributedPPO` uses an initialized PyTorch process
group to average gradients before clipping and pool observation/advantage
statistics and adaptive KL decisions. The caller creates one environment and
agent per worker. `collector.frames_per_batch` and `loss.mini_batch_size` are
local; `collector.total_frames`, checkpoint frame counts and save intervals are
global. Shards and minibatches must be equal. The current path supports
feed-forward PPO without compilation/CUDA graphs; moving normalizers update
once after each rollout. Only rank zero saves checkpoints. Resume preserves
optimizer/scheduler state and counters, but restarts simulator and RNG streams.

The parent IsaacLab-Imitation entrypoint exposes `--distributed` under
`torchrun`; its environment count and minibatch CLI settings are global and
are divided before constructing the worker agents. See the parent's
`2026-09-15-distributed-ppo-qualification` campaign for the bounded ICE test.
