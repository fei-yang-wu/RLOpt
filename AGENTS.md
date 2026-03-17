# AGENTS.md

This file defines how coding agents should work in the `RLOpt` submodule inside the `IsaacLabImitation` workspace.

## Scope

- This guidance applies to `RLOpt/` only.
- Treat the parent workspace, `IsaacLab/`, and `ImitationLearningTools/` as integration context, not default edit targets.
- Prefer changes in:
  - `rlopt/`
  - `tests/`
  - `README.md`
  - `pyproject.toml`

## Priorities

- Optimize first for runtime performance on real training workloads.
- Preserve generality across backends and environments; do not solve an IsaacLab need by making the core library Isaac-only.
- Keep integration with IsaacLab explicit and robust, since this workspace uses `RLOpt` as a training backend for IsaacLab imitation tasks.

## Environment

- Use the parent workspace conda environment: `SkillLearning`.
- Prefer non-interactive commands run from the workspace root:

```bash
conda run -n SkillLearning ...
```

- If a command must be run from this submodule, use `RLOpt/` as the working directory but keep the same environment.

## Repo Shape

- `rlopt/base_class.py`: shared training loop and collector orchestration.
- `rlopt/config_base.py`: dataclass config surface; keep new options documented and broadly usable.
- `rlopt/agent/`: algorithm implementations such as PPO, SAC, IPMD, GAIL, and related variants.
- `rlopt/env_utils.py`: environment construction helpers and backend glue.
- `rlopt/expert/`: expert dataset and streaming utilities; pay attention to transfer and storage costs.
- `tests/`: unit and integration coverage for algorithm and pipeline behavior.

## Working Rules

- Read `README.md` and the relevant algorithm or config files before changing training or environment behavior.
- Prefer minimal edits that improve the hot path instead of broad reorganizations.
- Keep algorithm code backend-agnostic where possible; isolate IsaacLab-specific behavior behind capability checks, config, or wrappers.
- When adding features for IsaacLab, design the public API so the same feature still works for non-Isaac TorchRL or Gymnasium environments when reasonable.
- Protect collector throughput, replay-buffer efficiency, tensor device placement, and logging overhead. Avoid unnecessary host-device copies, synchronization points, or per-step Python work in training loops.
- Reuse existing config/dataclass patterns instead of adding ad hoc dictionaries or one-off flags.
- Maintain compatibility with Hydra- and dataclass-driven workflows already used by this library and by the parent workspace scripts.
- Do not introduce IsaacLab assumptions based only on env name strings if a capability-based check is possible.
- Avoid notebook-only workflows; command-line tests and scripts are the source of truth.

## Training Loop Structure

- Prefer the shared phased training lifecycle for algorithm implementations.
- New or refactored agents should split training into `validate_training()`, `init_metadata()`, `collect()`, `prepare()`, `iterate()`, and `record()`.
- Keep `train()` as thin orchestration only. Reuse a family-level implementation such as PPO's shared loop when possible instead of re-embedding a monolithic train loop inside each agent.
- Use run-level metadata dataclasses for state that spans the full training call, and iteration-level dataclasses for state tied to one collected rollout or optimization cycle.
- Put rollout mutation and auxiliary data attachment in `prepare()`, gradient updates in `iterate()`, and logging/progress/checkpoint behavior in `record()`.
- When an algorithm needs custom metrics or progress output, prefer overriding the phase hooks and progress helpers rather than copying the whole outer loop.

## Performance Guidance

- Treat the training loop, replay pipeline, and model forward passes as performance-sensitive code.
- Prefer batched tensor operations over Python loops in rollout and optimization code.
- Be careful with `.cpu()`, `.numpy()`, `.item()`, and logging inside tight loops.
- If a change adds flexibility, check that it does not silently force extra copies, shape munging, or collector stalls.
- Keep defaults practical for large IsaacLab vectorized environments, not just tiny smoke tests.

## Generality Guidance

- New config fields should have clear defaults and compose with existing algorithms rather than specializing one script.
- Favor interfaces based on observation/action keys, specs, and capabilities over task-specific assumptions.
- Keep environment-specific logic thin. Core losses, modules, and storage utilities should stay reusable outside IsaacLab.
- When behavior truly differs for IsaacLab, document why the difference exists and keep the boundary obvious in code.

## IsaacLab Integration Guidance

- Assume IsaacLab is a primary consumer of this submodule in the parent workspace.
- Preserve compatibility with the workspace entrypoints under `scripts/rlopt/` in the parent repo.
- Changes to env interaction, logging, observation keys, or config defaults should be checked against IsaacLab task usage, especially manager-based vectorized environments.
- Prefer solutions that work with IsaacLab conventions such as structured observation groups, vectorized simulation, and environment-provided metrics.

## Validation

Run the smallest relevant checks from the workspace root with `SkillLearning`.

General checks:

```bash
conda run -n SkillLearning ruff check RLOpt
conda run -n SkillLearning ruff format --check RLOpt
conda run -n SkillLearning pytest RLOpt/tests
```

If you intentionally changed formatting:

```bash
conda run -n SkillLearning ruff format RLOpt
```

For IsaacLab-facing behavior, prefer a targeted workspace smoke test in addition to submodule tests:

```bash
conda run -n SkillLearning ./IsaacLab/isaaclab.sh -p scripts/rlopt/train.py --task Isaac-Imitation-G1-LafanTrack-v0 --algo PPO --max_iterations 1 --num_envs 32 --headless
```

Use heavier training runs only when the task requires them.

## Documentation

- Keep `RLOpt/README.md` and any command examples consistent with the actual library and the parent workspace usage.
- When documenting integration behavior, be explicit about what belongs in `RLOpt` versus what is configured by the parent IsaacLab workspace.
