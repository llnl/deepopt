# CLAUDE.md — DeepOpt repo onboarding

This file captures repo-specific context so future Claude Code sessions can work here without redoing broad exploration.

## Project summary

DeepOpt is a scientific Python package for Bayesian optimization, with user-facing API and CLI workflows for:

- training surrogate models with `deepopt learn`
- proposing candidates with `deepopt optimize`
- GP, delUQ, nnEnsemble, optional TabPFN, and RF (scikit-learn random forest) model types
- gradient (BoTorch L-BFGS, default), gradient-free scoped `sampling`, and `boltzmann` acquisition optimizers
- single- and multi-fidelity optimization
- optional risk-aware candidate generation

Primary public compatibility surfaces are the CLI contracts, config/default behavior, model initialization/data handling, checkpoint formats, and docs examples.

## Repo layout

- `deepopt/`
  - `__init__.py` — package version only (`__version__`).
  - `deepopt_cli.py` — Click CLI entrypoint and command option contracts.
  - `configuration.py` — `ConfigSettings`; loads defaults and optional YAML/JSON config files.
  - `defaults.py` — public default values (`Defaults`, `DELUQ_CONFIG`, `NNENSEMBLE_CONFIG`, `GP_CONFIG`).
  - `models.py` — high-level model wrappers: `DeepoptBaseModel`, `GPModel`, `DelUQModel`, `NNEnsembleModel`, candidate generation, optimize/save behavior.
  - `deltaenc.py` — delUQ BoTorch-compatible NN model wrapper.
  - `nn_ensemble.py` — NN ensemble BoTorch-compatible model wrapper.
  - `surrogate_utils.py` — MLP layers, Fourier features, optimizer creation.
  - `acquisition.py` — local MES/GIBBON acquisition implementations adapted from BoTorch, plus `SampledKnowledgeGradient` (gradient-free discrete KG).
  - `acq_optimizers.py` — torch-only gradient-free optimizers: scoped sampling, Boltzmann (SMC/DEFER/importance), `SearchSpace` constraint handling.
  - `surrogate_rf.py` — `RandomForestSurrogate` (sklearn; covariance from per-tree predictions; fantasize by refitting).
  - `surrogate_tabpfn.py` — optional TabPFN surrogate.
- `docs/` — MkDocs documentation and user examples.
- `requirements/requirements.txt` — runtime deps.
- `requirements/dev.txt` — dev deps; currently includes `pytest`.
- `tests/` — pytest regression suite added for common functionality and compatibility.
- `setup.py` — package metadata and console script (`deepopt = deepopt.deepopt_cli:main`).

## Environment/dependency notes

Runtime requirements include heavy scientific/BO dependencies:

- `torch`
- `gpytorch==1.8.1`
- `botorch==0.6.6`
- `ray[tune]`
- `scikit-learn`, `scipy`, `numpy`, `click`, `pyyaml`, `psutil`, etc.

On the environment checked when this file was written:

- Older environments (torch 1.x, older click) fail several pre-existing CLI/checkpoint tests (`_opt_prefixes`, `torch.load(weights_only=...)`); these are environment issues, not regressions. If another `deepopt` is installed, run with `PYTHONPATH=.`.
- Validate in an environment with BoTorch 0.6.6, a recent torch (2.x), scikit-learn, and pytest; there all tests pass.
- On HPC systems run tests on a compute/debug allocation, not a login node; `tests/conftest.py` caps threads to 1.

Do not install missing dependencies unless the user explicitly approves. Follow the global HPC safety rules: no heavy fitting, Ray tuning, multiprocessing, large benchmarks, or BO sweeps on login nodes.

## Test suite

Default command:

```bash
python3 -m pytest -q
```

In the reference BoTorch venv all tests pass (one `slow`-marked RF parallel-fantasy test included).

Pytest markers are defined in `pytest.ini`:

- `requires_botorch` — requires BoTorch/GPyTorch and sometimes Ray.
- `slow` — slower integration tests not intended for the default fast suite.

Important test files:

- `tests/test_configuration.py`
  - Config/default behavior.
  - Explicit regression for current `NNENSEMBLE_CONFIG` keys, including the misspelled `droupout_prob` key.
  - JSON config loading is supported (`test_config_settings_loads_json_overrides`).
  - `RF_CONFIG` and `SAMPLING_OPTIMIZATION_SETTINGS` (shared by all optimization profiles) regressions.
- `tests/test_cli.py`
  - CLI help, `get_deepopt_model`, conditional options, JSON CLI argument parsing.
  - Uses monkeypatching to avoid real training/optimization.
  - Skips if `botorch`, `gpytorch`, or `ray` are unavailable because importing `deepopt.models` needs them.
- `tests/test_models_data_handling.py`
  - Base model normalization, y reshaping, multi-fidelity fidelity rounding, `FidelityCostModel`, risk objective lookup.
  - Skips if BO/Ray deps unavailable.
- `tests/test_surrogate_utils.py`
  - Lightweight tests that run without BoTorch/Ray.
  - MLP shapes, Fourier features, activations, optimizer creation.
- `tests/test_nn_models.py`
  - NN wrapper initialization and checkpoint format.
  - Skips if BO deps unavailable.
- `tests/test_acq_optimizers.py`
  - Torch-only tests of scoped sampling, joint/sequential batches, constraints, Boltzmann samplers, DEFER partition.
- `tests/test_rf_surrogate.py`
  - RF posterior/covariance, fantasize, checkpoint refit, gradient guard, tiny end-to-end optimize runs (SF/MF/risk/constraints).
- `tests/test_acquisition.py`
  - Acquisition API import/signature compatibility.
  - Skips if BO deps unavailable.

When adding features, prefer adding/adjusting focused tests in these files rather than starting with broad repo exploration.

## Current compatibility gotchas

Preserve these unless intentionally breaking compatibility:

- Model type strings are exact and user-facing: `"GP"`, `"delUQ"`, `"nnEnsemble"`, `"TabPFN"`, `"RF"` (`DEEPOPT_MODEL_TYPES` in `models.py`).
- `optimization.acq_optimizer` defaults to `"gradient"` and `batch_mode` to `"default"` (MES sequential, EI/NEI/KG joint); with these defaults the `optimize_acqf` calls must stay identical to earlier releases (regression-tested). The CLI only forwards `acq_optimizer`/`batch_mode` to `optimize` when set.
- All `_optimize_acqf*` calls route through `DeepoptBaseModel._optimize_acqf`/`_optimize_acqf_mixed`, which dispatch on the settings active for the current `get_candidates` call (`_using_optimization_settings`).
- Surrogates with `supports_gradients = False` (RF) raise if `acq_optimizer == "gradient"`. RF checkpoints store no forest; `load_model` refits deterministically.
- CLI command names: `learn`, `optimize`.
- CLI bounds and list-like numeric inputs are JSON strings parsed with `json.loads`, e.g. `"[[0, 1], [0, 1]]"` and `"[1, 10]"`.
- `.npz` training files are expected to contain keys `X` and `y`.
- `DeepoptBaseModel.__post_init__` reshapes 1-D `y` to `N x 1`.
- Single-fidelity data are normalized using provided `bounds`.
- Multi-fidelity mode treats the last input column as fidelity, rounds it after normalization, and sets `target_fidelities` to the highest fidelity index.
- `Defaults` values are part of the public behavior and covered by tests.
- `NNENSEMBLE_CONFIG` currently contains `droupout_prob` rather than `dropout_prob`; this historical spelling is captured as current behavior, and `ConfigSettings` maps it to `dropout_prob` for runtime use.
- NN checkpoint dictionaries contain keys like `epoch`, `state_dict`, `B`, and `opt_state_dict`, plus scaler state and optional `deepopt_checkpoint` metadata in modern checkpoints.
- Multi-fidelity candidate saving with `integer_fidelities=True` currently produces an object dtype array by concatenating float columns with an integer fidelity column.

## Development workflow guidance

- Keep changes small and scientific-code-oriented; avoid broad product-style refactors unless asked.
- For behavior changes, first identify which public surface is affected: CLI, config, defaults, model init/data handling, acquisition behavior, checkpoint format, docs examples.
- Add regression tests for public behavior before or with fixes.
- For tests, use tiny deterministic arrays and CPU only.
- Avoid invoking `DelUQModel.train` in normal tests: it starts Ray tuning and can be expensive.
- Avoid real candidate optimization in default tests; monkeypatch `optimize_acqf`/model methods or add skipped/slow integration tests.
- Do not rely on GPU availability in tests.
- Prefer `tmp_path` for test artifacts. The global temp-file rule still applies outside pytest-managed temp dirs: use `/tmp/kur1`, never bare `/tmp`.
- Use `python3 -m pytest -q` for default validation.

## Common focused entry points

Use these starting points instead of broad scanning:

- CLI option behavior: `deepopt/deepopt_cli.py`
- Config/default behavior: `deepopt/configuration.py`, `deepopt/defaults.py`, `tests/test_configuration.py`
- Data normalization and fidelity handling: `deepopt/models.py`, `tests/test_models_data_handling.py`
- GP train/load: `GPModel` in `deepopt/models.py`
- delUQ train/load: `DelUQModel` in `deepopt/models.py`, `deepopt/deltaenc.py`
- NN ensemble train/load: `NNEnsembleModel` in `deepopt/models.py`, `deepopt/nn_ensemble.py`
- MLP architecture/optimizers: `deepopt/surrogate_utils.py`, `tests/test_surrogate_utils.py`
- Acquisition functions: `deepopt/acquisition.py`, `tests/test_acquisition.py`
- User-facing docs examples: `docs/index.md`, `docs/user_guide/tutorial.md`, `docs/user_guide/configuration.md`, `docs/user_guide/acquisition_functions.md`
