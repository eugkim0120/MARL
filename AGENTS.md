# AGENTS.md

Instructions for coding agents (Claude Code, OpenCode, Codex, and others) working in this repository.

## Purpose

`easy_marl` is a Gymnasium-compatible multi-agent reinforcement learning framework with an
electricity market example. PPO agents (stable-baselines3) bid into a market, trained either
sequentially (iterated best response) or in parallel (simultaneous best response). The bidding
example also includes a battery storage (BESS) sweep that measures the effect of adding a
PPO-trained battery to a generator-only market.

## Key directories

- `easy_marl/src/` - core framework: agents (`agents.py`), environment (`environment.py`), observers (`observators.py`).
- `easy_marl/examples/bidding/` - electricity market example: market clearing (`market.py`), training (`training.py`), BESS sweep (`bess_experiment.py`, `bess_analysis.py`, `bess_viz.py`).
- `tests/unit_tests/` - pytest suite, mirroring the source layout.
- `results/` - curated, committed summaries of finished sweeps (CSV, report, plots, dashboard).
- `outputs/` - raw training and sweep artifacts; ignored by git.
- `main.py` - default training entry point (`auto_train`). This is a training run, not a check.

## Setup

Requires Python 3.11+ (`.python-version` pins 3.13) and `uv`. Dependencies are locked in `uv.lock`.

```bash
uv sync --extra dev
```

## Definition of done

A change is done when `scripts/check.sh` passes. It syncs the locked environment and runs the
full pytest suite on CPU in a few minutes. Run it before declaring any change finished, and
report its actual output. Add or update tests alongside behaviour changes; test names follow
the `test_that_...` convention described in `tests/README.md`.

Known state (October 2026): on `main`, three cases of
`tests/unit_tests/test_agents.py::TestFixedActionFunction::test_that_ppo_agent_is_not_affected_by_model_changes`
fail, so `scripts/check.sh` exits non-zero. Do not count these against your change, but do not
introduce new failures.

Training runs, sweeps and long experiments are not checks. Do not start them unless asked.

## Experiment hygiene

- Experiments are configured in code: the BESS sweep uses named presets in `PRESETS` in `bess_experiment.py`, selected with `--preset` and optionally overridden with `--powers`, `--durations`, `--seeds` and `--arms`.
- Seeds are explicit. `set_all_seeds` in `training.py` seeds Python, NumPy and PyTorch; sweeps list their seeds per preset, and evaluation uses a fixed `EVAL_SEED` so configs see the same demand draws.
- Raw outputs go to `outputs/` (training `save_dir` defaults to `outputs`, the sweep to `outputs/bess_sweep/<preset>`). Never commit anything under `outputs/`, model checkpoints, or TensorBoard `logs/`.
- Only small, final summaries belong in `results/`, and only when the user asks for results to be committed.

## Prohibited actions

- Do not commit secrets, tokens or credentials, and do not write them into code, configs, tests or logs.
- Do not push to the default branch (`main`) unless explicitly asked. Work on a branch.
- Do not commit large generated artifacts (checkpoints, raw sweep outputs, logs).
- Do not modify `.github/workflows/python-publish.yml` unless asked.
- Do not weaken or delete failing tests to make `scripts/check.sh` pass; fix the cause or report it.
