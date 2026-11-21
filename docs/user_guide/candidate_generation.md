# Candidate generation

`deepopt optimize` loads a trained surrogate and proposes new simulation inputs. DeepOpt optimizes acquisition functions in the scaled coordinates used by the model, then saves candidates in the original input units expected by your simulation.

Bounds, linear constraints, nonlinear constraints, and input-uncertainty standard deviations are all authored in original input units. Multi-fidelity runs use the last input column as an integer fidelity index.

## Acquisition methods

Single-fidelity optimization supports:

- `EI` — Expected Improvement
- `NEI` — Noisy Expected Improvement
- `KG` — Knowledge Gradient
- `MaxValEntropy` — Max-value entropy search

Multi-fidelity optimization supports:

- `KG`
- `MaxValEntropy`

`KG` and entropy-style acquisition functions are more expensive, so DeepOpt uses the lower-budget settings from the resolved optimization profile for those paths. See [Configuration Settings](configuration.md#optimization-settings) for profile and budget settings.

By default, acquisition functions are maximized with BoTorch's gradient-based optimizer. Gradient-free scoped sampling and Boltzmann sampling are available with `--acq-optimizer sampling|boltzmann`, and `--batch-mode` selects sequential or joint batch construction. See [Acquisition Optimizers](acquisition_optimizers.md).

## Linear constraints

Linear constraints are supplied as JSON lists of `[indices, coefficients, rhs]` entries:

```bash
deepopt optimize \
  -l learner_GP.ckpt \
  -o suggested_inputs.npy \
  -a EI \
  --inequality-constraints '[[[0, 1], [1.0, -1.0], 0.0]]'
```

Inequality constraints use

```text
sum(coefficients[i] * x[indices[i]]) >= rhs
```

Equality constraints use the same left-hand side with `==`:

```bash
deepopt optimize \
  -l learner_GP.ckpt \
  -o suggested_inputs.npy \
  -a EI \
  --equality-constraints '[[[2], [1.0], 0.5]]'
```

Constraint indices must be integers. DeepOpt converts original-unit linear constraints to the scaled coordinates used by BoTorch internally. In multi-fidelity runs, the fidelity column is kept as an unscaled fidelity index during this conversion.

Equality constraints are not supported for entropy acquisition candidate sets.

## Nonlinear constraints

Nonlinear inequality constraints are loaded from trusted local Python code:

```python title="constraints.py"
def make_constraints():
    def inside_circle(X):
        return 0.25 - ((X[..., 0] - 0.5) ** 2 + (X[..., 1] - 0.5) ** 2)

    def above_floor(X):
        return X[..., 2] - 0.1

    return [inside_circle, above_floor]
```

```bash
deepopt optimize \
  -l learner_GP.ckpt \
  -o suggested_inputs.npy \
  -a EI \
  --nonlinear-inequality-constraints constraints.py:make_constraints \
  --nonlinear-mode enforce
```

The loader format is `path/to/file.py:function_name`. The named object may be a constraint callable itself or a zero-argument factory returning a callable or list of callables. Constraint callables receive candidate tensors in original input units. A point is feasible when every callable returns values `>= 0`.

### Nonlinear modes

`--nonlinear-mode enforce` passes nonlinear constraints to BoTorch. With the pinned BoTorch optimizer this requires `batch_limit=1`; for multiple candidates, DeepOpt uses sequential `q=1` optimizer calls where supported.

`--nonlinear-mode initialization-only` uses nonlinear constraints only to choose feasible initial conditions. This preserves larger batch limits but does not guarantee final candidate feasibility.

### Nonlinear optimization controls

The same nonlinear controls are available as CLI flags or under the top-level `optimization:` config section:

| CLI flag | YAML key | Meaning | Default |
| -------- | -------- | ------- | ------- |
| `--nonlinear-mode` | `nonlinear_mode` | `enforce` or `initialization_only`; YAML also accepts `initialization-only`. | `enforce` |
| `--nonlinear-initial-raw-samples` | `nonlinear_initial_raw_samples` | Raw samples per attempt when searching for nonlinear-feasible starts. | `null` |
| `--nonlinear-initial-max-tries` | `nonlinear_initial_max_tries` | Maximum attempts to find nonlinear-feasible starts. | `5` |
| `--nonlinear-optimization-retries` | `nonlinear_optimization_retries` | Additional retries after nonlinear constrained optimizer failure warnings. | `1` |

Example optimize-time config:

```yaml title="optimize.yaml"
optimization:
  profile: fast
  nonlinear_mode: initialization-only
  nonlinear_initial_raw_samples: 2048
  nonlinear_initial_max_tries: 10
  nonlinear_optimization_retries: 2
```

```bash
deepopt optimize \
  -l learner_GP.ckpt \
  -o suggested_inputs.npy \
  -a EI \
  -c optimize.yaml \
  --nonlinear-inequality-constraints constraints.py:make_constraints
```

Explicit CLI flags override config values. If a flag is omitted, DeepOpt uses the value from `optimization:` and then falls back to the default.

### Limitations

Nonlinear constraints are currently limited to single-fidelity optimization. They are not supported for multi-fidelity mixed optimization, fixed-feature subproblems, or `KG`. If DeepOpt cannot find feasible starts, increase `nonlinear_initial_raw_samples` or `nonlinear_initial_max_tries`, or relax the constraints.

## Risk-aware candidate generation

VaR and CVaR risk measures are supported with `EI`, `NEI`, and `KG`. `MaxValEntropy` does not currently support risk measures.

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

`--X-stddev` in the CLI and `x_stddev` in the Python API are specified in original input units with one value per input dimension. DeepOpt scales these values internally. For multi-fidelity optimization, the fidelity-column perturbation is forced to zero so the risk transform does not change fidelity.

## `propose_best`

`propose_best` reserves the first returned candidate for the current posterior maximizer. DeepOpt then uses the selected acquisition function for the remaining `num_candidates - 1` points.

```bash
deepopt optimize \
  -l learner_GP.ckpt \
  -o suggested_inputs.npy \
  -a EI \
  --num-candidates 4 \
  --propose-best
```

In multi-fidelity optimization, the posterior maximizer is found at the target/highest fidelity and the fidelity column is appended before candidates are saved in original input units.
