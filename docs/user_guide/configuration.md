# Configuration Settings

!!! note

    Model training hyperparameters are only configurable for `nnEnsemble` and `delUQ` models; GP model training runs as-is. Optimization settings apply to all model types.

DeepOpt supports YAML and JSON configuration files. Model-training settings are merged over model-specific defaults, and candidate-generation settings live under a top-level `optimization:` section. `TabPFN` uses optional TabPFN-specific settings such as `tabpfn_api`, `tabpfn_version`, `max_tokens_per_forward`, and `max_weight`. `RF` uses the random forest settings described below.

## The Base Configuration Options

The following options configure neural-network surrogate training for `nnEnsemble` and `delUQ` models.

| Option | Description | Default |
| ------ | ----------- | ------- |
| `n_estimators` | Number of neural networks in an `nnEnsemble` model. | `nnEnsemble`: `100` |
| `ff` | Whether to use Fourier features before the neural network layers. | `True` |
| `dist` | Initial Fourier-frequency distribution: `uniform`, `gaussian`, or `laplace`. | `uniform` |
| `mapping_size` | Number of Fourier frequencies when `ff` is enabled. | `128` |
| `variance` | Scale parameter for the Fourier-frequency distribution. | `0.001` |
| `activation` | Activation function: `relu`, `tanh`, `identity`, or `siren`. | `relu` |
| `w0` | SIREN initialization scale. | `30` |
| `n_layers` | Total number of network layers, including first and last layers. | `4` |
| `hidden_dim` | Width of each hidden layer. | `128` |
| `dropout` | Whether to use dropout regularization. | `True` |
| `dropout_prob` | Probability of dropping a neuron when dropout is enabled. | `0.2` |
| `activation_first` | When dropout is enabled, whether to apply activation before batchnorm/dropout. | `True` |
| `batchnorm` | Whether to use batch normalization. | `False` |
| `opt_type` | Optimizer type: `Adam` or `SGD`. | `Adam` |
| `learning_rate` | Optimizer learning rate. | `0.001` |
| `weight_decay` | L2 weight-decay penalty. | `0` |
| `n_epochs` | Number of training epochs. | `delUQ`: `1000`; `nnEnsemble`: `300` |
| `batch_size` | Training batch size. If larger than the dataset, the whole dataset is used as one batch. | `delUQ`: `1000`; `nnEnsemble`: `128` |

For `nnEnsemble`, older defaults used the misspelled key `droupout_prob`. DeepOpt still accepts that key for compatibility, but new config files should use `dropout_prob`.

Example:

```yaml title="config.yaml"
n_estimators: 50
hidden_dim: 256
n_epochs: 500
dropout_prob: 0.1
```

## Random Forest Settings

The `RF` surrogate wraps scikit-learn's `RandomForestRegressor`. Its posterior mean is the average tree prediction, and the joint covariance of a batch of points is the empirical covariance of the individual tree predictions. `RF` is not differentiable and must be optimized with `--acq-optimizer sampling` or `boltzmann` (see [Acquisition Optimizers](acquisition_optimizers.md)).

| Option | Description | Default |
| ------ | ----------- | ------- |
| `n_estimators` | Number of trees (at least 2). | `200` |
| `max_depth` | Maximum tree depth; `null` grows trees fully. | `null` |
| `min_samples_leaf` | Minimum training samples per leaf. | `1` |
| `max_features` | Features considered per split (scikit-learn semantics). | `1.0` |
| `bootstrap` | Fit each tree on a bootstrap sample. | `True` |
| `n_jobs` | Threads used when fitting the main forest. | `1` |
| `variance_mode` | `trees` uses the across-tree covariance only; `trees+leaf` also adds the mean within-leaf target variance on the diagonal (law of total variance, as in SMAC). | `trees` |
| `fantasy_n_jobs` | Threads used to refit fantasy forests in parallel (KG, sequential MaxValEntropy). | `1` |

Fantasy models are built by refitting a forest on the training data augmented with sampled outcomes, one forest per fantasy.

## Optimization Settings

Candidate generation can also be configured with an `optimization:` section in the same YAML/JSON config file. If this section is omitted, DeepOpt uses the `cpu_large` profile, which is intended for single-node runs with many CPU cores.

```yaml
optimization:
  profile: cpu_large
  batch_limit_high: 24
  torch_num_threads: auto
```

For optimize-only overrides, place just the candidate-generation settings in a config file:

```yaml title="optimize.yaml"
optimization:
  profile: fast
  batch_limit_high: 6
  torch_num_threads: 4
```

Then pass it when proposing candidates from a self-describing checkpoint:

```bash
deepopt optimize -l learner_GP.ckpt -o suggested_inputs.npy -a EI -c optimize.yaml
```

For self-describing checkpoints, `-c/--config-file` can override the `optimization:` section used for candidate generation without restating the original training data, bounds, or model type.

Profiles provide sensible defaults:

| Profile | Purpose |
| ------- | ------- |
| `cpu_large` | Default profile for large single-node CPU allocations. |
| `balanced` | Legacy-compatible optimization settings. |
| `fast` | Lower-cost settings for smoke tests or quick iteration. |

Any setting specified alongside `profile` overrides that profile value. If `profile` is omitted, DeepOpt starts from `cpu_large` and applies the provided overrides.

### Budget and thread settings

| Option | Description |
| ------ | ----------- |
| `num_restarts_high` / `num_restarts_low` | Number of multistart local optimizations for normal and expensive acquisition paths. |
| `raw_samples_high` / `raw_samples_low` | Number of raw Sobol samples used to initialize local optimization. |
| `batch_limit_high` / `batch_limit_low` | Number of restart batches evaluated together. Larger values can improve CPU/GPU utilization but increase memory use. |
| `maxiter` | Maximum local optimizer iterations. |
| `n_fantasies` | Number of fantasy models for KG/MES-family acquisition functions. Larger values can improve Monte Carlo accuracy but can be much slower. |
| `torch_num_threads` | PyTorch intra-op thread count. Use `auto`, an integer, or `null`. |
| `torch_num_threads_fraction` | Fraction of available CPUs to use when `torch_num_threads: auto` and more than 8 CPUs are available. Default is `0.8`. |
| `torch_num_interop_threads` | PyTorch inter-op thread count. Use an integer or `null`. |

For `torch_num_threads: auto`, DeepOpt uses all available CPUs on small machines (8 or fewer CPUs). On larger allocations it uses `floor(torch_num_threads_fraction * available_cpus)`, preferring CPU affinity and Slurm CPU variables when available. If thread environment variables such as `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, or `TORCH_NUM_THREADS` are already set, DeepOpt does not override PyTorch's intra-op thread count unless an explicit integer is provided.

### Nonlinear constraint controls

These settings control nonlinear constrained candidate generation. They can also be supplied as CLI flags; explicit CLI flags override config values.

| Option | Description | Default |
| ------ | ----------- | ------- |
| `nonlinear_mode` | `enforce` or `initialization_only`; `initialization-only` is also accepted in YAML. | `enforce` |
| `nonlinear_initial_raw_samples` | Raw samples per attempt when searching for nonlinear-feasible starts. Use `null` to reuse the acquisition optimizer raw-sample count. | `null` |
| `nonlinear_initial_max_tries` | Maximum attempts to find nonlinear-feasible starts. | `5` |
| `nonlinear_optimization_retries` | Additional retries after nonlinear constrained optimizer failure warnings. | `1` |

```yaml title="optimize_constraints.yaml"
optimization:
  profile: fast
  nonlinear_mode: initialization-only
  nonlinear_initial_raw_samples: 2048
  nonlinear_initial_max_tries: 10
  nonlinear_optimization_retries: 2
```

See [Candidate Generation](candidate_generation.md#nonlinear-constraints) for nonlinear constraint syntax and limitations.

### Acquisition optimizer settings

`acq_optimizer` (`gradient`, `sampling`, or `boltzmann`) and `batch_mode` (`default`, `sequential`, or `joint`) select how acquisition functions are optimized and how `q > 1` batches are built. The defaults (`gradient`, `default`) reproduce earlier DeepOpt behavior. The `sampling_*`, `boltzmann_*`, and `kg_*` keys configure the gradient-free optimizers; see [Acquisition Optimizers](acquisition_optimizers.md#settings-reference) for the full list. CLI flags `--acq-optimizer` and `--batch-mode` override config values.

### Validation rules

Budget settings, restart counts, batch limits, `maxiter`, `n_fantasies`, `nonlinear_initial_raw_samples`, and `nonlinear_initial_max_tries` must be positive integers when provided. `nonlinear_initial_raw_samples` may also be `null`. `nonlinear_optimization_retries` must be a non-negative integer. `torch_num_threads` may be `auto`, `null`, or a positive integer. `torch_num_threads_fraction` must be in `(0, 1]`. Unknown keys under `optimization:` raise an error.

For acquisition constraints, risk-aware candidate generation, and `propose_best`, see [Candidate Generation](candidate_generation.md).
