"""``validate_options``: every option dependency fails at construction (CPU)."""

from __future__ import annotations

from typing import Any

import pytest

from rlopt.agent.wm_bptt import agent as agent_module
from rlopt.agent.wm_bptt.config import WMBPTTConfig, WMBPTTRLOptConfig, validate_options


def config(**wm: Any) -> WMBPTTRLOptConfig:
    return WMBPTTRLOptConfig(wm=WMBPTTConfig(**wm))


def test_the_default_config_passes() -> None:
    validate_options(config())


REFUSED = [
    ({"actor_source": "dream"}, "actor_source"),
    ({"critic_source": "dream"}, "critic_source"),
    ({"actor_source": "real", "actor_horizon": 4}, "actor_horizon"),
    ({"actor_estimator": "magic"}, "actor_estimator"),
    ({"actor_estimator": "rebar"}, "rebar"),
    ({"actor_estimator": "score_function", "actor_source": "real"}, "score_function"),
    ({"actor_ema": -0.1}, "actor_ema"),
    ({"actor_ema": 1.0}, "actor_ema"),
]


@pytest.mark.parametrize(("wm", "match"), REFUSED)
def test_refused_options(wm: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        validate_options(config(**wm))


def test_construction_validates_before_it_builds_anything(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("the task was built before the options were checked")

    monkeypatch.setattr(agent_module, "load_task", fail)
    with pytest.raises(ValueError, match="actor_source"):
        agent_module.WMBPTT(env=None, config=config(actor_source="dream"))  # type: ignore[arg-type]
