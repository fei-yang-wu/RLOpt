"""Adaptive-KL measurement must not train the policy or consume action RNG."""

from __future__ import annotations

import warnings

import torch


def setup_policy():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from tensordict import TensorDict
    from tensordict.nn import TensorDictModule
    from torchrl.envs.utils import ExplorationType
    from torchrl.modules import IndependentNormal, ProbabilisticActor

    from rlopt.agent.ppo.ppo import RunningMeanStdCatInputs
    from rlopt.base_class import BaseAlgorithm
    from rlopt.models import GaussianPolicyHead

    torch.manual_seed(42)
    norm = RunningMeanStdCatInputs(torch.nn.Linear(2, 1), 2)
    head = GaussianPolicyHead(norm, 1)
    op = ProbabilisticActor(
        TensorDictModule(head, ["obs"], ["loc", "scale"]),
        in_keys=["loc", "scale"],
        distribution_class=IndependentNormal,
        default_interaction_type=ExplorationType.RANDOM,
    )
    op.eval()
    with torch.no_grad():
        batch = op(TensorDict({"obs": torch.tensor([[4.0, 2.0], [5.0, 3.0]])}, [2]))
    op.train()
    context = BaseAlgorithm._prepare_kl_context(None, batch, op)
    return op, norm, context, BaseAlgorithm


def test_kl_measurement_preserves_buffers_and_action_rng():
    op, norm, context, base = setup_policy()
    before = {name: value.clone() for name, value in op.state_dict().items()}
    rng = torch.get_rng_state().clone()
    base._compute_kl_after_update(None, context, op)
    assert all(
        torch.equal(before[name], value) for name, value in op.state_dict().items()
    )
    assert torch.equal(rng, torch.get_rng_state())
    assert op.training
    assert norm.training


def test_unchanged_policy_has_zero_kl_on_repeated_measurement():
    op, _, context, base = setup_policy()
    for _ in range(3):
        value = base._compute_kl_after_update(None, context, op)
        torch.testing.assert_close(value, torch.tensor(0.0), atol=1e-7, rtol=0)


def test_kl_measurement_restores_mixed_module_modes():
    op, norm, context, base = setup_policy()
    norm.eval()
    modes = [module.training for module in op.modules()]
    base._compute_kl_after_update(None, context, op)
    assert [module.training for module in op.modules()] == modes
