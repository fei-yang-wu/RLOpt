from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from rlopt.config_utils import (
    dedupe_keys,
    epic_distance,
    flatten_feature_tensor,
    flatten_obs_features_from_td,
    flatten_obs_group,
    infer_batch_shape_from_mapping,
    next_obs_key,
    normalize_batch_key,
    strip_next_prefix,
)


def test_batch_key_normalization_helpers() -> None:
    assert normalize_batch_key("observation") == "observation"
    assert normalize_batch_key(["policy", "obs"]) == ("policy", "obs")
    assert normalize_batch_key(("reward",)) == "reward"
    assert dedupe_keys(["obs", ["next", "obs"], ("next", "obs")]) == [
        "obs",
        ("next", "obs"),
    ]
    assert next_obs_key("observation") == ("next", "observation")
    assert next_obs_key(("policy", "obs")) == ("next", "policy", "obs")
    assert strip_next_prefix(("next", "policy", "obs")) == ("policy", "obs")
    assert strip_next_prefix(("next", "reward")) == "reward"

    with pytest.raises(ValueError, match="must not be empty"):
        normalize_batch_key([])
    with pytest.raises(TypeError, match="Unsupported key type"):
        normalize_batch_key(1)


def test_flatten_feature_tensor_preserves_batch_dims() -> None:
    scalar_feature = torch.arange(6).reshape(2, 3)
    image_feature = torch.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)

    torch.testing.assert_close(
        flatten_feature_tensor(scalar_feature, feature_ndim=0),
        scalar_feature.unsqueeze(-1),
    )
    torch.testing.assert_close(
        flatten_feature_tensor(image_feature, feature_ndim=1),
        image_feature,
    )
    assert flatten_feature_tensor(image_feature, feature_ndim=2).shape == (2, 3, 20)


def test_flatten_obs_features_from_tensordict() -> None:
    td = TensorDict(
        {
            "obs": torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
            "state": torch.tensor([[5.0, 6.0]]),
            ("next", "obs"): torch.tensor([[[7.0, 8.0], [9.0, 10.0]]]),
            ("next", "state"): torch.tensor([[11.0, 12.0]]),
        },
        batch_size=[1],
    )
    key_feature_ndims = {"obs": 2, "state": 1}

    current = flatten_obs_features_from_td(
        td,
        ["obs", "state"],
        key_feature_ndims,
        next_obs=False,
        detach=True,
    )
    next_features = flatten_obs_features_from_td(
        td,
        ["obs", "state"],
        key_feature_ndims,
        next_obs=True,
        detach=False,
    )

    torch.testing.assert_close(current, torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]))
    torch.testing.assert_close(
        next_features,
        torch.tensor([[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]]),
    )
    assert current.requires_grad is False

    with pytest.raises(ValueError, match="at least one key"):
        flatten_obs_features_from_td(
            td,
            [],
            key_feature_ndims,
            next_obs=False,
            detach=True,
        )


def test_infer_batch_shape_from_mapping_validates_consistency() -> None:
    batch_shape = infer_batch_shape_from_mapping(
        {
            "obs": torch.zeros(4, 3),
            ("policy", "image"): torch.zeros(4, 2, 2),
        },
        {
            "obs": 1,
            ("policy", "image"): 2,
        },
    )
    assert batch_shape == (4,)

    with pytest.raises(ValueError, match="Observation batch shape mismatch"):
        infer_batch_shape_from_mapping(
            {
                "obs": torch.zeros(4, 3),
                ("policy", "image"): torch.zeros(5, 2, 2),
            },
            {
                "obs": 1,
                ("policy", "image"): 2,
            },
        )


def test_flatten_obs_group_moves_nested_observation_keys() -> None:
    td = TensorDict(
        {
            "observation": TensorDict(
                {
                    "policy": torch.ones(2, 3),
                    "critic": torch.zeros(2, 1),
                },
                batch_size=[2],
            ),
            "next": TensorDict(
                {
                    "observation": TensorDict(
                        {"policy": torch.full((2, 3), 2.0)},
                        batch_size=[2],
                    )
                },
                batch_size=[2],
            ),
        },
        batch_size=[2],
    )

    out = flatten_obs_group(td)

    assert "observation" not in out
    assert "policy" in out
    assert "critic" in out
    assert ("next", "policy") in out.keys(True)
    torch.testing.assert_close(out["policy"], torch.ones(2, 3))
    torch.testing.assert_close(out["next", "policy"], torch.full((2, 3), 2.0))


def test_epic_distance_handles_correlated_and_constant_rewards() -> None:
    r_true = torch.tensor([1.0, 2.0, 3.0, 4.0])
    corr, distance = epic_distance(2.0 * r_true + 5.0, r_true)
    torch.testing.assert_close(corr, torch.tensor(1.0))
    torch.testing.assert_close(distance, torch.tensor(0.0))

    corr, distance = epic_distance(torch.ones(4), r_true)
    torch.testing.assert_close(corr, torch.tensor(0.0))
    torch.testing.assert_close(distance, torch.sqrt(torch.tensor(0.5)))
