# Acquisition functions
There are currently four acquisition functions available for single-fidelity optimization: Expected Improvement (EI), Noisy Expected Improvement (NEI), Knowledge Gradient (KG), and Max Value Entropy (MaxValEntropy). The last two (KG & Max Value Entropy) are also available for multi-fidelity optimization. Risk measures, including Value-at-Risk (VaR) and Conditional Value-at-Risk (CVaR), are supported with EI, NEI, and KG; MaxValEntropy does not currently support risk measures. These acquisition functions are built around the associated [BoTorch acquisition functions](https://botorch.org/api/acquisition.html#acquisitionfunction): qExpectedImprovement, qNoisyExpectedImprovement, qKnowledgeGradient, and qMaxValueEntropy. We briefly describe the strengths and weaknesses of each acquisition.

## EI
This is one of the simplest acquisition functions for Bayesian optimization. It selects points that optimize improvement over the best function value found thus far (weighted by the probability of achieving such improvement). The interpretation is straightforward, but EI tends to favor exploitation over exploration and can get stuck near local optima.

## NEI
This adapts EI to problems that are noisy (strong fluctuations in the objective function). The major change from EI is that NEI measures improvement over the best surrogate value (rather than objective function value) among points selected thus far. This allows NEI to avoid getting thrown off by noise, since the surrogate will generally be much smoother than the objective function.

## KG
Knowledge gradient attempts to reduce EI's heavy exploitation by selecting a point such that a subsequent selection would yield the best expected improvement. Effectively it's a one-step look-ahead acquisition function. Specifically, a potential selection is evaluated by "fantasizing" at the location (drawing outputs from the probability distribution at the location and fitting a separate model to each) then, for each fantasy model, identifying how much improvement is obtained when using EI as a subsequent acquisition. Improvements are averaged over the fantasy models to assign a value to each potential selection and a final selection is made based on the best value. The need to fantasize at each potential location makes KG a fairly expensive acquisition function and it can be slow to use, but in addition to getting stuck less than EI, KG can be used for multi-fidelity optimization problems.

## MaxValEntropy
Max Value Entropy selects points to minimize its uncertainty about the optimal value. This indirect approach allows it to heavily favor exploration before zooming in on promising spots in the input space. Its information-theoretic foundation also easily extends to the multi-fidelity setting (a low-fidelity candidate is selected if it helps minimize uncertainty about the high-fidelity optimum).

## Supported combinations

| Feature | EI | NEI | KG | MaxValEntropy |
| ------- | -- | --- | -- | ------------- |
| Single-fidelity optimization | Yes | Yes | Yes | Yes |
| Multi-fidelity optimization | No | No | Yes | Yes |
| VaR/CVaR risk measures | Yes | Yes | Yes | No |
| Linear constraints | Yes | Yes | Yes | Yes, with candidate-set filtering |
| Nonlinear constraints | Yes | Yes | Gradient-free optimizers only | Single-fidelity only |
| `RF` surrogate (gradient-free optimizers) | Yes | Yes | Yes (discrete KG) | Yes |
| `batch_mode: joint` | Yes | Yes | Yes | No |

`KG` and entropy-style acquisition functions use the lower-budget optimization profile settings because they are more expensive. Entropy acquisitions use a sampled candidate set before continuous optimization. Equality constraints are not supported for these candidate sets; use inequality or nonlinear constraints instead. Nonlinear constraints are not currently supported with gradient-based KG; with `--acq-optimizer sampling|boltzmann`, KG uses a gradient-free discrete knowledge gradient that supports them. GIBBON is not available with the `RF` surrogate. See [Acquisition Optimizers](acquisition_optimizers.md) for the gradient-free optimizers. For nonlinear constraints, `initialization_only` mode filters optimizer starts but does not guarantee final feasibility. See [Candidate Generation](candidate_generation.md) for constraint syntax, risk options, and `propose_best` behavior.

## Risk measures and input perturbations

VaR and CVaR wrap the acquisition objective with input perturbations. The perturbation standard deviations are specified in original input units through `x_stddev` in the Python API or `--X-stddev` in the CLI. DeepOpt scales these values internally before calling BoTorch. For multi-fidelity runs, the fidelity-column standard deviation is set to zero so the risk transform does not change fidelity.

```python
model.optimize(
    outfile="suggested_inputs.npy",
    learner_file="learner_GP.ckpt",
    acq_method="EI",
    risk_measure="CVaR",
    risk_level=0.8,
    risk_n_deltas=128,
    x_stddev=[0.02, 0.02, 0.02],
)
```

You can also evaluate risk values at existing points from a self-describing checkpoint:

```python
from deepopt.models import load_deepopt_wrapper

model = load_deepopt_wrapper("learner_GP.ckpt")
values = model.get_cvar(
    risk_level=0.8,
    x_stddev=[0.02, 0.02, 0.02],
    risk_n_deltas=128,
)
```

## `propose_best`

`propose_best=True` reserves the first returned candidate for the current posterior maximizer. DeepOpt then uses the selected acquisition function for the remaining `num_candidates - 1` points. In multi-fidelity optimization, this posterior maximizer is found at the target fidelity and the fidelity column is appended before saving candidates in original input units. See [Candidate Generation](candidate_generation.md#propose_best) for CLI usage.