from __future__ import annotations

import pytest
import torchrl.envs.libs.gym as torchrl_gym_lib


@pytest.fixture(autouse=True)
def _disable_torchrl_isaaclab_probe_for_unit_tests():
    original_has_isaaclab = getattr(torchrl_gym_lib, "_has_isaaclab", None)
    if original_has_isaaclab is not None:
        torchrl_gym_lib._has_isaaclab = False
    try:
        yield
    finally:
        if original_has_isaaclab is not None:
            torchrl_gym_lib._has_isaaclab = original_has_isaaclab
