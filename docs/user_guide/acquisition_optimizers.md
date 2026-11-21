# Acquisition optimizers

DeepOpt finds candidates by maximizing an acquisition function over the scaled input space. Three optimizers are available. You select one with `--acq-optimizer` on the CLI, the `acq_optimizer` argument of `optimize`, or `optimization.acq_optimizer` in a config file. The CLI or argument value takes precedence.

| `acq_optimizer` | Summary | Needs gradients |
| --------------- | ------- | --------------- |
| `gradient` (default) | BoTorch multistart L-BFGS-B, seeded from Sobol raw samples. This is DeepOpt's historical behavior. | Yes |
| `sampling` | Gradient-free scoped random search around elite points. | No |
| `boltzmann` | Draws batch members i.i.d. from the Boltzmann distribution of the acquisition function. | No |

The default settings reproduce the gradient-based behavior of earlier DeepOpt releases exactly. The gradient-free optimizers evaluate the acquisition function under `torch.no_grad()`. This makes non-differentiable surrogates such as the random forest (`RF`) usable. `RF` requires `sampling` or `boltzmann`; requesting `gradient` with `RF` raises an error.

```bash
deepopt optimize -l learner_RF.ckpt -o suggested.npy -a EI -q 4 --acq-optimizer sampling
deepopt optimize -l learner_GP.ckpt -o suggested.npy -a EI -q 8 --acq-optimizer boltzmann
```

## Batch mode

`--batch-mode` (or `optimization.batch_mode`) controls how a batch of `q > 1` candidates is built. It works the same way for the `gradient` and `sampling` optimizers.

| `batch_mode` | Behavior |
| ------------ | -------- |
| `default` | Keeps the historical rules. `MaxValEntropy` is built sequentially, while `EI`, `NEI`, and `KG` optimize the q points jointly. Multi-fidelity mixed optimization is always sequential. |
| `sequential` | Greedy construction: select one point, add it as a pending point (fantasizing where the acquisition requires it), and repeat. Available for every acquisition. |
| `joint` | Optimize all `q x d` coordinates together. Available for `EI`, `NEI`, and `KG` in single-fidelity runs. `MaxValEntropy` is pointwise, so it cannot be optimized jointly. |

The `boltzmann` optimizer always draws its q batch members i.i.d. and ignores `batch_mode`, with a warning if a non-default mode is set.

## Scoped sampling (`sampling`)

Scoped sampling replaces the gradient step with repeated local random search:

1. Draw `sampling_raw_samples` scrambled-Sobol points (default `2^14 = 16384`), score them, and keep the best `sampling_n_elites` (default `128`).
2. For each of `sampling_n_rounds` rounds (default `10`):
    1. Draw `sampling_samples_per_elite` (default `128`) new points in a box around every elite.
    2. Re-select the best `sampling_n_elites` from the elites and all new points combined.
3. Return the best point. For joint batches (`batch_mode='joint'`, or `default` with `EI`, `NEI`, or `KG`), the same procedure runs over `q x d` batches and returns the best batch.

Each round, every elite's box shrinks. Its half-width, in unit-cube coordinates, is

```
h = clip( min( NN(elite), sampling_init_halfwidth * sampling_shrink^round ), sampling_min_halfwidth, . )
```

where `NN(elite)` is the Euclidean distance to the nearest point sampled so far.

- **Nearest-neighbor term:** makes the box data-adaptive. Elites in densely explored regions get tight boxes.
- **Geometric schedule:** a TuRBO-style trust-region contraction that guarantees the boxes keep shrinking. The defaults are `0.2` and `0.7`.

The geometric term matters in 10–30 dimensions. There, nearest-neighbor distances are all large and nearly equal, so a pure nearest-neighbor box barely contracts.

`sampling_perturbation` controls how points are drawn inside a box:

- **`subset` (default):** following TuRBO, only a random subset of coordinates is resampled. Each free coordinate is resampled with probability `sampling_subset_prob`, which defaults to `min(1, 20/d)`, and at least one coordinate always moves. In high dimensions this keeps the moves meaningfully local.
- **`uniform`:** all coordinates are resampled uniformly in the box.
- **`gaussian`:** Normal steps with σ = half-width.

All three modes reflect points at the bounds.

In `joint` mode, every member of an elite batch is perturbed inside its own box. Set `sampling_joint_member_prob < 1` to move only a random subset of members, which can help when `q·d` is large.

## Boltzmann sampling (`boltzmann`)

Following B3O [[Bloor et al. 2026](https://arxiv.org/abs/2606.30228)], batch members are drawn from

```
p(x) ∝ exp( λ · standardize(α(x)) )
```

Here `α` is the acquisition function, evaluated at single points, and `λ` is `boltzmann_lambda` (default `1.0`). DeepOpt standardizes `α` with the mean and standard deviation of the initial sample population, as BoTorch's `BoltzmannSampling` does. This makes `λ` dimensionless and comparable across problems and acquisition functions.

- As `λ → 0`, the draws become uniform.
- As `λ` grows, the draws concentrate near the maximizer.

DeepOpt always uses a constant `λ`. The B3O paper reports that a well-chosen constant is competitive with its `√t log t` schedule.

Three samplers are available through `boltzmann_sampler`:

| `boltzmann_sampler` | Method | Notes |
| ------------------- | ------ | ----- |
| `smc` (default) | Annealed sequential Monte Carlo | Starts from `boltzmann_population` Sobol points and raises `λ` geometrically over `boltzmann_n_rounds`. Each round it resamples (systematic resampling when the effective sample size drops below half), then applies `boltzmann_mh_steps` Metropolis moves with the same subset/box proposal used by scoped sampling. Degrades gracefully in 10–30 dimensions. |
| `defer` | Recursive partitioning (after DEFER [Bodin et al. 2021], the sampler used in the B3O paper) | Repeatedly splits the highest-mass cells at the midpoint of their longest side. It builds a piecewise-constant approximation of `p(x)` with `boltzmann_defer_budget` evaluations, then samples cells by mass and points uniformly within each cell. Recommended for `d ≲ 10`. Even one split per axis needs `2^d` cells, so in 20–30 dimensions the partition is necessarily coarse. |
| `importance` | Self-normalized importance resampling of a Sobol population | The cheapest option; the B3O paper calls it "discretized Monte Carlo". It is weak in high dimensions because few Sobol points land in high-density regions. |

Boltzmann draws are reproducible for a given `random_seed`. They are sampled with replacement, so duplicates are possible when `λ` is large.

## Constraints

The gradient-free optimizers handle constraints by rejection:

- **Feasibility checks:** infeasible draws are discarded, with oversampling to compensate. Linear inequalities and nonlinear inequalities (`>= 0` feasible) are checked directly.
- **Linear equalities:** handled by projecting each draw onto the constraint set.
- **Nonlinear modes:** nonlinear constraints are always enforced; `nonlinear_mode` only affects the gradient optimizer.
- **Boltzmann with equalities:** projection makes the proposals asymmetric, so with linear equality constraints Boltzmann draws are approximate rather than exact samples of the restricted density.
- **Infeasible problems:** if no feasible point can be found, DeepOpt raises an error asking you to relax the constraints.

## Knowledge gradient without gradients

BoTorch's `qKnowledgeGradient` is "one-shot": it optimizes the candidates and one inner maximizer per fantasy jointly with gradients. With `sampling` or `boltzmann`, DeepOpt instead uses a discrete knowledge gradient:

```
KG(X) = E_fantasy[ max over S of value_fantasy(x') ] − current best value
```

- **The inner set `S`:** about `kg_inner_set_size` points (default `1024`). It combines posterior-mean elites from a scoped search, the training inputs, and Sobol points, and is shared by every evaluation. A single batched fantasy model per evaluation is therefore enough.
- **Refinement:** after the outer search, the best `kg_refine_top` candidates (default `8`) are re-scored with a full scoped search of each fantasy model's value function.
- **Outer budget:** every KG evaluation requires fantasizing (refitting, for `RF`), so the outer search uses the smaller `kg_raw_samples`, `kg_n_elites`, `kg_samples_per_elite`, and `kg_n_rounds` budgets (Boltzmann rounds are capped by `kg_n_rounds` too).
- **RF cost:** each KG evaluation refits `n_fantasies` forests. With RF, use a small `n_fantasies` (8–16) and `fantasy_n_jobs > 1` on a compute allocation.
- **Coverage:** multi-fidelity KG, cost weighting, and risk objectives are supported.

## Settings reference

All keys go under `optimization:`, and they are shared by every optimization profile.

| Key | Default | Meaning |
| --- | ------- | ------- |
| `acq_optimizer` | `gradient` | `gradient`, `sampling`, or `boltzmann`. |
| `batch_mode` | `default` | `default`, `sequential`, or `joint`. |
| `sampling_raw_samples` | `16384` | Initial Sobol draw. |
| `sampling_n_elites` | `128` | Elites kept each round. |
| `sampling_samples_per_elite` | `128` | New points per elite per round. |
| `sampling_n_rounds` | `10` | Scoping rounds. |
| `sampling_init_halfwidth` | `0.2` | Initial box half-width (unit cube). |
| `sampling_shrink` | `0.7` | Geometric half-width factor per round. |
| `sampling_min_halfwidth` | `0.001` | Half-width floor. |
| `sampling_perturbation` | `subset` | `subset`, `uniform`, or `gaussian`. |
| `sampling_subset_prob` | `null` | Per-coordinate move probability for `subset`; `null` means `min(1, 20/d)`. |
| `sampling_eval_batch` | `2048` | Points evaluated per acquisition call (memory control). |
| `sampling_joint_member_prob` | `1.0` | Fraction of batch members moved per joint perturbation. |
| `boltzmann_sampler` | `smc` | `smc`, `defer`, or `importance`. |
| `boltzmann_lambda` | `1.0` | Inverse temperature `λ`. |
| `boltzmann_population` | `16384` | SMC / importance population size. |
| `boltzmann_n_rounds` | `10` | SMC annealing rounds. |
| `boltzmann_mh_steps` | `1` | Metropolis moves per SMC round. |
| `boltzmann_defer_budget` | `16384` | DEFER evaluation budget. |
| `kg_inner_set_size` | `1024` | Size of the discrete KG inner set. |
| `kg_refine_top` | `8` | KG candidates re-scored with full inner searches. |
| `kg_raw_samples`, `kg_n_elites`, `kg_samples_per_elite`, `kg_n_rounds` | `512`, `16`, `16`, `3` | Outer search budget for KG. |

```yaml title="config.yaml"
optimization:
  acq_optimizer: sampling
  batch_mode: sequential
  sampling_n_rounds: 12
  sampling_shrink: 0.6
```

## Credits

The gradient-free optimizers build on the work of others:

- **BoTorch.** M. Balandat, B. Karrer, D. R. Jiang, S. Daulton, B. Letham, A. G. Wilson, E. Bakshy, "BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization", *NeurIPS* 2020 ([botorch.org](https://botorch.org), MIT license).
    - DeepOpt's acquisition functions, posteriors, samplers, input transforms, fantasy models, and gradient-based optimization are all built on BoTorch.
    - The Boltzmann weighting `exp(η · standardize(α))` follows BoTorch's `botorch.generation.sampling.BoltzmannSampling`.
    - The gradient-free knowledge gradient is adapted from BoTorch's knowledge-gradient implementation.
- **B3O.** M. Bloor, L. Xu, H. Stojic, V. Picheny, "B3O: Scalable Boltzmann Batch Bayesian Optimization", [arXiv:2606.30228](https://arxiv.org/abs/2606.30228), 2026.
    - The authors implemented B3O in [trieste](https://github.com/secondmind-labs/trieste) (Secondmind Labs, Apache-2.0), a TensorFlow Bayesian optimization library.
    - DeepOpt's PyTorch implementation was written independently from the paper's description, and we thank the B3O authors and the trieste developers.
- **DEFER.** E. Bodin, Z. Dai, N. Campbell, C. H. Ek, "Black-box density function estimation using recursive partitioning", *ICML* 2021.
- **TuRBO.** D. Eriksson, M. Pearce, J. Gardner, R. Turner, M. Poloczek, "Scalable Global Optimization via Local Bayesian Optimization", *NeurIPS* 2019.
    - TuRBO is the source of the shrinking trust-region schedule and the coordinate-subset perturbation used by scoped sampling.
