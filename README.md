# RLOpt

RLOpt is a PyTorch and TorchRL training library for reinforcement learning and
imitation-learning research. It provides reusable algorithm implementations,
dataclass-driven configuration, expert-data utilities, and environment glue that
can be used directly or as a backend inside larger workspaces such as IsaacLab.

The core package is intentionally backend-agnostic: algorithm code works with
TorchRL/Gymnasium-style specs and tensor keys, while IsaacLab-specific behavior
is kept at integration boundaries.

## What Is Included

- Shared training infrastructure in `rlopt/base_class.py`.
- Dataclass configuration in `rlopt/config_base.py`.
- PPO, SAC, IPMD, GAIL, ASE, FastTD3/FastSAC, and skill-commanding agents under
  `rlopt/agent/`.
- Expert replay and streaming helpers under `rlopt/expert/`.
- Environment construction helpers in `rlopt/env_utils.py`.
- Tests covering algorithm components, replay utilities, save/load behavior, and
  smoke training paths.

## Install

Use Python 3.11 or newer. For development, install the package from the repo root:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

The project dependencies are declared in `pyproject.toml`; CI installs from that
file directly instead of relying on a separate Conda environment file.

In the parent IsaacLab workspace, the expected environment is:

```bash
conda run -n SkillLearning python -m pip install -e RLOpt
```

## Quick Start

```python
from rlopt.agent import PPO, PPORLOptConfig
from rlopt.env_utils import make_parallel_env

cfg = PPORLOptConfig()
cfg.env.env_name = "Pendulum-v1"
cfg.env.num_envs = 2
cfg.device = "cpu"
cfg.collector.frames_per_batch = 128
cfg.collector.total_frames = 1024
cfg.logger.backend = None

env = make_parallel_env(cfg)
agent = PPO(env, cfg, logger=None)
agent.train()
```

## Development Checks

From this repository, the CI-parity checks are:

```bash
python -m ruff check --select E9,F63,F7,F82 rlopt tests
python -m pyrefly check rlopt/config_base.py
python -m pytest tests
```

The default pytest configuration excludes `slow` and `compile` tests. The
compile tests exercise TorchDynamo/TorchRL/Inductor integration, so they are
kept out of the default CI gate and should be run when working on compiler
integration:

```bash
python -m pytest -m compile tests/test_compile.py tests/test_compile_cpu.py
```

For stricter cleanup before larger refactors, run the full formatting and lint
suite and address the additional style findings it reports:

```bash
python -m ruff check .
python -m ruff format --check .
```

From the parent IsaacLab workspace, prefer the shared environment and equivalent
paths:

```bash
conda run -n SkillLearning ruff check --select E9,F63,F7,F82 RLOpt/rlopt RLOpt/tests
conda run -n SkillLearning pyrefly check RLOpt/rlopt/config_base.py
conda run -n SkillLearning pytest RLOpt/tests
```

For IsaacLab-facing changes, also run a small training smoke test from the parent
workspace:

```bash
conda run -n SkillLearning ./IsaacLab/isaaclab.sh -p scripts/rlopt/train.py \
  --task Isaac-Imitation-G1-LafanTrack-v0 \
  --algo PPO \
  --max_iterations 1 \
  --num_envs 32 \
  --headless
```

## Configuration Notes

RLOpt uses dataclasses for library configuration. Most users start from an
algorithm config such as `PPORLOptConfig`, `SACRLOptConfig`, or
`IPMDRLOptConfig`, then update nested fields:

```python
cfg.collector.frames_per_batch = 1024
cfg.loss.mini_batch_size = 256
cfg.policy.input_keys = ["observation"]
cfg.logger.backend = None
```

Observation inputs should be expressed through tensor keys and specs rather than
environment-name checks. This keeps the same algorithms usable across Gymnasium,
TorchRL, and IsaacLab-style vectorized environments.

## IsaacLab Integration

RLOpt is used as a training backend by IsaacLab imitation tasks in the parent
workspace. Integration behavior should stay explicit:

- Keep IsaacLab-specific assumptions in wrappers, config, or entrypoint scripts.
- Preserve support for structured observation groups and vectorized simulation.
- Avoid unnecessary host/device copies in collectors, replay buffers, and logging.
- Validate environment-facing changes with a small IsaacLab smoke run when
  possible.
