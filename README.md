# RLOpt: A Research Framework for Reinforcement Learning

RLOpt is a flexible and modular framework for Reinforcement Learning (RL) research, built on PyTorch and TorchRL. It is designed to facilitate the implementation, testing, and comparison of various RL agents and optimization techniques. The framework uses dataclass-based configs for library code and Hydra for experiment scripts, allowing convenient customization.

### Geometric goal planner

`rlopt.agent.geometric_goal_planner` adds world-pose goals with body masks,
tolerances and absolute deadlines, plus a Transformer flow model predicting
latent sequences from causal robot history. Targets and inputs are detached;
the module never updates an encoder or PPO optimizer. Hindsight sampling stays
within trajectories and masks tokens after the goal from attention and loss.
Data, checkpoint binding, training and simulator adapters belong to the parent
`imitation_experiments` package.

### Factorized affine PoE DiffSR

`phi_parameterization="affine_poe"` retains the affine parameter shapes and
state-dict keys, but contracts `[1; z]` with factorized expert fields.
`feature_dim` remains the internal factor width; the returned phi and mu
field axis have `action_dim + 1` entries. `forward_mu` requires source `s`
for this mode even with target-only mu conditioning. This is an algebraic
reassociation, not a new capacity setting; floating-point operation order
can differ. The fixed coordinate is internal, not an extra policy command.
