# CLI reference

DeepOpt exposes two commands:

- `deepopt learn` trains a surrogate and writes a checkpoint.
- `deepopt optimize` loads a checkpoint and proposes new candidate inputs.

Training data files must be NumPy `.npz` files with keys `X` and `y`. `X` is an `N x d` input array and `y` is an `N` or `N x 1` objective array. Bounds are JSON encoded as one `[lower, upper]` pair per input dimension.

## Train a model

```bash
bounds='[[0, 1], [0, 1], [0, 1]]'
deepopt learn -i sims.npz -o learner_GP.ckpt -b "$bounds" -m GP
```

Important options:

| Option | Meaning |
| ------ | ------- |
| `-i`, `--infile` | Training data `.npz` file containing `X` and `y`. |
| `-o`, `--outfile` | Checkpoint path to write. |
| `-b`, `--bounds` | JSON list of `[lower, upper]` bounds in original input units. |
| `-m`, `--model-type` | Surrogate type: `GP`, `delUQ`, `nnEnsemble`, optional `TabPFN`, or `RF` (random forest). |
| `-c`, `--config-file` | Optional YAML/JSON config file. |
| `-r`, `--random-seed` | Random seed for training and candidate generation. |
| `-k`, `--k-folds` | Number of folds used by delUQ training. |
| `-d`, `--device` | `auto`, `cpu`, `gpu`, or `cuda`. |
| `--multi-fidelity` | Treat the last input column as an integer fidelity index. |

## Propose candidates

Modern DeepOpt checkpoints are self-describing, so `optimize` can reload the training data, bounds, model type, configuration, and scaler state from the checkpoint:

```bash
deepopt optimize -l learner_GP.ckpt -o suggested_inputs.npy -a EI
```

Core options:

| Option | Meaning |
| ------ | ------- |
| `-l`, `--learner-file` | Checkpoint produced by `deepopt learn`. |
| `-o`, `--outfile` | NumPy `.npy` file to write proposed candidates into. |
| `-a`, `--acq-method` | Acquisition method: `EI`, `NEI`, `KG`, or `MaxValEntropy`. `TabPFN` does not currently support `KG`, and its `MaxValEntropy` path supports only one candidate at a time. |
| `-q`, `--num-candidates` | Number of candidates to propose. |
| `--acq-optimizer` | `gradient` (default), `sampling`, or `boltzmann`. `RF` requires `sampling` or `boltzmann`. See [Acquisition Optimizers](acquisition_optimizers.md). |
| `--batch-mode` | `default`, `sequential`, or `joint` construction of `q > 1` batches. |
| `-c`, `--config-file` | Optional optimize-time config overrides. |
| `-r`, `--random-seed` | Random seed for candidate generation when using legacy checkpoints. |
| `-d`, `--device` | `auto`, `cpu`, `gpu`, or `cuda`. |
| `-v`, `--verbose` | Print model evaluation and fantasy-training details. |
| `--propose-best` | Use the first candidate for the current surrogate posterior maximizer. |

Legacy checkpoints without `deepopt_checkpoint` metadata still require the original training data, bounds, and model type:

```bash
deepopt optimize \
  -l legacy_learner.ckpt \
  -i sims.npz \
  -b '[[0, 1], [0, 1], [0, 1]]' \
  -m GP \
  -o suggested_inputs.npy \
  -a EI
```

## Optimize-time configuration

For self-describing checkpoints, a config file passed to `optimize` can override candidate-generation settings without restating the training inputs:

```yaml title="optimize.yaml"
optimization:
  profile: fast
  batch_limit_high: 6
  torch_num_threads: 4
```

```bash
deepopt optimize -l learner_GP.ckpt -o suggested_inputs.npy -a EI -c optimize.yaml
```

See [Configuration Settings](configuration.md#optimization-settings) for optimization profiles, thread settings, and nonlinear constraint control settings.

## Multi-fidelity optimization

Multi-fidelity data use the last input column as an integer fidelity index from `0` to `num_fidelities - 1`. Multi-fidelity candidate generation supports `KG` and `MaxValEntropy`.

```bash
deepopt learn \
  -i sims_mf.npz \
  -o learner_GP_mf.ckpt \
  -b '[[0, 1], [0, 1], [0, 2]]' \
  -m GP \
  --multi-fidelity

deepopt optimize \
  -l learner_GP_mf.ckpt \
  -o suggested_inputs.npy \
  -a KG \
  --fidelity-cost '[1, 4, 16]' \
  --integer-fidelities
```

| Option | Meaning |
| ------ | ------- |
| `--multi-fidelity` | Legacy-checkpoint optimize flag indicating that the last input column is fidelity. |
| `--fidelity-cost` | JSON list with one cost per fidelity. |
| `--integer-fidelities` | Save the multi-fidelity fidelity column as integers. |

DeepOpt rounds the candidate fidelity column before indexing `--fidelity-cost`.

## Risk-aware optimization

Risk measures are supported with `EI`, `NEI`, and `KG`. `MaxValEntropy` does not currently support risk measures.

```bash
deepopt optimize \
  -l learner_GP.ckpt \
  -o suggested_inputs.npy \
  -a EI \
  --risk-measure CVaR \
  --risk-level 0.8 \
  --risk-n-deltas 128 \
  --X-stddev '[0.02, 0.02, 0.02]'
```

| Option | Meaning |
| ------ | ------- |
| `--risk-measure` | `VaR` or `CVaR`. |
| `--risk-level` | Risk level in `(0, 1)`. |
| `--risk-n-deltas` | Number of input perturbations sampled for the risk objective. |
| `--X-stddev` | JSON list of input standard deviations in original input units, one per input dimension. |

For multi-fidelity optimization, the fidelity-column perturbation is forced to zero.

## Constraints

Linear constraints are JSON lists of `[indices, coefficients, rhs]` entries in original input units:

```bash
deepopt optimize \
  -l learner_GP.ckpt \
  -o suggested_inputs.npy \
  -a EI \
  --inequality-constraints '[[[0, 1], [1.0, -1.0], 0.0]]' \
  --equality-constraints '[[[2], [1.0], 0.5]]'
```

Nonlinear constraints are loaded from trusted local Python files:

```bash
deepopt optimize \
  -l learner_GP.ckpt \
  -o suggested_inputs.npy \
  -a EI \
  --nonlinear-inequality-constraints constraints.py:make_constraints \
  --nonlinear-mode enforce
```

Constraint options:

| Option | Meaning |
| ------ | ------- |
| `--inequality-constraints` | JSON list of linear inequality constraints. |
| `--equality-constraints` | JSON list of linear equality constraints. |
| `--nonlinear-inequality-constraints` | Trusted Python constraint loader in `path/to/file.py:function_name` form. |
| `--nonlinear-mode` | `enforce` or `initialization-only`; omitted flags fall back to `optimization.nonlinear_mode`. |
| `--nonlinear-initial-raw-samples` | Raw samples used when searching for nonlinear-feasible initial conditions. |
| `--nonlinear-initial-max-tries` | Maximum attempts to find nonlinear-feasible initial conditions; omitted flags fall back to config. |
| `--nonlinear-optimization-retries` | Additional retries after nonlinear constrained optimizer warnings; omitted flags fall back to config. |

See [Candidate Generation](candidate_generation.md) for constraint semantics, config-key equivalents, and current limitations.
