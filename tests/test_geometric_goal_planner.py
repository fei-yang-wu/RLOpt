from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from rlopt.agent.geometric_goal_planner import (
    GeometricGoalPlanner,
    GeometricGoalPlannerConfig,
    WorldPoseGoal,
    heading_matrix,
    quaternion_matrix,
    sample_hindsight_indices,
)


def make_goal() -> WorldPoseGoal:
    return WorldPoseGoal(
        position=torch.tensor([[[3.0, 1.0, 0.8], [1.0, 2.0, 1.4]]]),
        quaternion=torch.tensor([[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]]),
        mask=torch.ones(1, 2, 2, dtype=torch.bool),
        tolerance=torch.full((1, 2, 2), 0.1),
        deadline=torch.tensor([5.0]),
        hold_seconds=torch.tensor([0.0]),
    )


def test_world_goal_stays_fixed_as_robot_moves_and_deadline_elapses():
    goal = make_goal()
    goal.validate()
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    initial = goal.features(
        torch.tensor([[0.0, 0.0, 0.8]]), identity, torch.tensor([0.0])
    )
    later = goal.features(
        torch.tensor([[2.0, 0.0, 0.8]]), identity, torch.tensor([2.0])
    )
    assert initial[0, 0] == 3
    assert later[0, 0] == 1
    assert initial[0, 2] == 0.8  # absolute height is preserved
    assert later[0, -2] == 3
    assert goal.position[0, 0, 0] == 3


def test_common_world_yaw_translation_and_quaternion_sign_do_not_change_goal():
    goal = make_goal()
    root = torch.tensor([[0.4, 0.3, 0.8]])
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    angle = torch.tensor(0.7)
    q = torch.tensor([[torch.cos(angle / 2), 0.0, 0.0, torch.sin(angle / 2)]])
    rotation = quaternion_matrix(q)
    shift = torch.tensor([[2.0, -4.0, 0.0]])
    moved = replace(
        goal,
        position=goal.position @ rotation[0].T + shift[:, None],
        quaternion=-q[:, None].expand(-1, 2, -1),
    )
    expected = goal.features(root, identity, torch.zeros(1))
    actual = moved.features(root @ rotation[0].T + shift, q, torch.zeros(1))
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_twist_heading_and_masked_values_cannot_leak():
    q = torch.tensor([[0.6, 0.4, 0.3, 0.2]])
    twist = torch.tensor([[0.6, 0.0, 0.0, 0.2]])
    torch.testing.assert_close(heading_matrix(q), quaternion_matrix(twist))
    goal = make_goal()
    goal.mask[:, 1] = False
    before = goal.features(torch.zeros(1, 3), q, torch.zeros(1))
    goal.position[:, 1] = 1000
    goal.quaternion[:, 1] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    goal.tolerance[:, 1] = 100
    torch.testing.assert_close(
        before, goal.features(torch.zeros(1, 3), q, torch.zeros(1))
    )
    assert goal.reached(goal.position, -goal.quaternion).all()


def test_hindsight_never_crosses_trajectory_boundaries():
    lengths = torch.tensor([2, 30, 70, 500])
    ranks, starts, goals = sample_hindsight_indices(
        lengths,
        5000,
        history_steps=10,
        prediction_steps=10,
        command_stride=1,
        encoder_steps=10,
        min_goal_steps=1,
        max_goal_steps=250,
        generator=torch.Generator().manual_seed(4),
    )
    assert (ranks != 0).all()
    assert (starts >= 0).all()
    assert (starts == 0).any()
    assert (starts + 18 < lengths[ranks]).all()
    assert (goals > starts).all()
    assert (goals < lengths[ranks]).all()
    assert (goals - starts).min() == 1
    assert (goals - starts).max() > 200


def small_config() -> GeometricGoalPlannerConfig:
    return GeometricGoalPlannerConfig(
        history_steps=2,
        history_frame_dim=3,
        body_names=("pelvis",),
        latent_dim=2,
        prediction_steps=3,
        d_model=16,
        num_layers=1,
        num_heads=2,
        feedforward_dim=32,
        num_inference_steps=3,
    )


def test_flow_gradients_are_isolated_and_invalid_labels_do_not_leak():
    cfg = small_config()
    planner = GeometricGoalPlanner(cfg)
    history = torch.randn(4, 2, 3, requires_grad=True)
    goal = torch.randn(4, cfg.goal_dim, requires_grad=True)
    teacher = torch.nn.Linear(3, 6)
    target = teacher(history[:, -1]).reshape(4, 3, 2)
    valid = torch.tensor([[True, False, False]]).expand(4, -1)
    noise, time = torch.randn(4, 3, 2), torch.full((4, 1), 0.4)
    loss = planner.loss(history, goal, target, valid=valid, noise=noise, time=time)
    changed = target.detach().clone()
    changed[:, 1:] += 100
    torch.testing.assert_close(
        loss, planner.loss(history, goal, changed, valid=valid, noise=noise, time=time)
    )
    loss.backward()
    assert history.grad is None
    assert goal.grad is None
    assert all(p.grad is None for p in teacher.parameters())
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in planner.parameters()
    )


def test_sampling_padding_and_checkpoint_roundtrip():
    cfg = small_config()
    model = GeometricGoalPlanner(cfg).eval()
    history = torch.randn(3, 2, 3)
    goal = torch.randn(3, cfg.goal_dim)
    goal[:, -2:] = torch.tensor([0.02, 0.0])
    actual = model(history, goal, noise_std=0)
    assert actual.shape == (3, 3, 2)
    assert (actual[:, 1:] == 0).all()
    restored = GeometricGoalPlanner.from_checkpoint(model.checkpoint()).eval()
    torch.testing.assert_close(actual, restored(history, goal, noise_std=0))


def test_bad_goal_and_empty_trajectory_selection_fail():
    goal = make_goal()
    goal.mask[:] = False
    with pytest.raises(ValueError, match="constraint"):
        goal.validate()
    with pytest.raises(ValueError, match="No trajectories"):
        sample_hindsight_indices(
            torch.tensor([4, 9]),
            2,
            history_steps=10,
            prediction_steps=10,
            command_stride=1,
            encoder_steps=10,
            min_goal_steps=1,
            max_goal_steps=20,
        )
