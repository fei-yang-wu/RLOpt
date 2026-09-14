"""RSL-RL time-out bootstrapping for PPO advantage estimation."""

from __future__ import annotations

import warnings
from functools import lru_cache
from types import SimpleNamespace

import torch
from tensordict import TensorDict

GAMMA = 0.9
LAMBDA = 0.8


@lru_cache(maxsize=1)
def _modules() -> SimpleNamespace:
    """Import TorchRL and RLOpt lazily; their imports raise deprecation warnings."""

    warnings.filterwarnings("ignore", category=DeprecationWarning, append=False)
    from tensordict.nn import TensorDictModule
    from torchrl.objectives.value.advantages import GAE

    from rlopt.agent.ppo.ppo import apply_rsl_rl_truncation_bootstrap

    return SimpleNamespace(
        TensorDictModule=TensorDictModule,
        GAE=GAE,
        bootstrap=apply_rsl_rl_truncation_bootstrap,
    )


class _Value(torch.nn.Module):
    """A fixed linear value function so V(obs) is known in closed form."""

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return (2.0 * obs).sum(-1, keepdim=True)


def _value_operator():
    return _modules().TensorDictModule(
        _Value(), in_keys=["obs"], out_keys=["state_value"]
    )


def _rollout() -> TensorDict:
    envs, steps = 2, 6
    obs = torch.arange(envs * steps, dtype=torch.float32).reshape(envs, steps, 1)
    next_obs = obs + 100.0  # far from obs so a wrong bootstrap is visible
    reward = torch.ones(envs, steps, 1)
    done = torch.zeros(envs, steps, 1, dtype=torch.bool)
    terminated = torch.zeros(envs, steps, 1, dtype=torch.bool)
    # env 0: time-out at step 2; env 1: true termination at step 3.
    done[0, 2] = True
    done[1, 3] = True
    terminated[1, 3] = True
    return TensorDict(
        {
            "obs": obs,
            "next": TensorDict(
                {
                    "obs": next_obs,
                    "reward": reward,
                    "done": done,
                    "terminated": terminated,
                },
                batch_size=[envs, steps],
            ),
        },
        batch_size=[envs, steps],
    )


def _rsl_rl_reference(rollout: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
    """RSL-RL ``compute_returns``: bootstrap time-outs with V(s_t), mask dones."""

    value = _Value()
    values = value(rollout["obs"])
    next_values = value(rollout["next", "obs"])
    done = rollout["next", "done"].float()
    terminated = rollout["next", "terminated"].float()
    time_out = done * (1.0 - terminated)
    reward = rollout["next", "reward"] + GAMMA * values * time_out
    steps = rollout.batch_size[1]
    advantage = torch.zeros_like(values)
    last = torch.zeros_like(values[:, 0])
    for step in reversed(range(steps)):
        not_done = 1.0 - done[:, step]
        delta = (
            reward[:, step] + GAMMA * next_values[:, step] * not_done - values[:, step]
        )
        last = delta + GAMMA * LAMBDA * not_done * last
        advantage[:, step] = last
    return advantage, advantage + values


def test_rsl_rl_bootstrap_matches_the_reference_return_loop() -> None:
    rollout = _rollout()
    stored_keys = set(rollout.keys(include_nested=True, leaves_only=True))
    view = _modules().bootstrap(rollout, _value_operator(), GAMMA)
    gae = _modules().GAE(
        gamma=GAMMA, lmbda=LAMBDA, value_network=_value_operator(), average_gae=False
    )
    view = gae(view)
    advantage, value_target = _rsl_rl_reference(_rollout())
    torch.testing.assert_close(view["advantage"], advantage, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(view["value_target"], value_target, atol=1e-5, rtol=1e-5)
    # The stored rollout keeps its reward and termination flags.
    assert set(rollout.keys(include_nested=True, leaves_only=True)) == stored_keys
    torch.testing.assert_close(rollout["next", "reward"], torch.ones(2, 6, 1))
    assert not bool(rollout["next", "terminated"][0, 2])


def test_torchrl_default_bootstraps_a_time_out_from_the_reset_observation() -> None:
    rollout = _rollout()
    gae = _modules().GAE(
        gamma=GAMMA, lmbda=LAMBDA, value_network=_value_operator(), average_gae=False
    )
    default = gae(rollout.copy())
    advantage, _ = _rsl_rl_reference(_rollout())
    # Env 0 times out at step 2: the default estimate differs there.
    assert not torch.allclose(default["advantage"][0, 2], advantage[0, 2])
    # Env 1 terminates for real at step 3: both rules agree from that step on.
    torch.testing.assert_close(
        default["advantage"][1, 3:], advantage[1, 3:], atol=1e-5, rtol=1e-5
    )
