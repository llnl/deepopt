![Python versions](https://img.shields.io/pypi/pyversions/deepopt)
[![License](https://img.shields.io/pypi/l/deepopt)](https://pypi.org/project/deepopt/)
![Activity](https://img.shields.io/github/commit-activity/m/LLNL/deepopt)
[![Issues](https://img.shields.io/github/issues/LLNL/deepopt)](https://github.com/LLNL/deepopt/issues)
[![Pull requests](https://img.shields.io/github/issues-pr/LLNL/deepopt)](https://github.com/LLNL/deepopt/pulls)

<!-- This page needs links once the repo is released and docs are published -->

Visit the [DeepOpt documentation](./docs/README.md) for more information on DeepOpt that's not covered in this README.

## What is DeepOpt?

DeepOpt is a simple and easy-to-use library for performing Bayesian optimization, leveraging the powerful capabilities of [BoTorch](https://botorch.org/). Its key feature is the ability to use neural networks as surrogate functions during the optimization process, allowing Bayesian optimization to work smoothly even on large datasets and in many dimensions. DeepOpt also provides simplified wrappers for BoTorch fitting and optimization routines.

### Key Commands

The DeepOpt library comes equipped with two cornerstone commands:

1. **Learn:** The `learn` command trains a machine learning model on a given set of data. Users can select a Gaussian process (`GP`), delta-UQ neural network (`delUQ`), or neural-network ensemble (`nnEnsemble`) surrogate. Uncertainty quantification (UQ) is available in all models, allowing for direct use in a Bayesian optimization workflow. The `learn` command supports multi-fidelity modeling with an arbitrary number of fidelities.

2. **Optimize:**  The `optimize` command takes the previously trained model created through the `learn` command and runs a single Bayesian optimization step, proposing a set of candidate points aimed at improving the value of the objective function (output of the learned model). The user can choose between several available acquisition methods for selecting the candidate points. Support for optimization under input uncertainty and risk is available. Modern checkpoints are self-describing, so they can be reloaded and optimized without restating the original training data and bounds.

## Why DeepOpt?

DeepOpt is a powerful and versatile Bayesian optimization framework that provides users with the flexibility to choose between Gaussian process (GP) and neural network (NN) surrogates. This flexibility empowers users to select the most suitable surrogate model for their specific optimization problem, taking into account factors such as the complexity of the objective function and the available computational resources.

## Installation

DeepOpt is available via [PyPI](https://pypi.org/) and can be easily installed with:

```bash
pip install deepopt
```

For a quick start guide, see [Getting Started with DeepOpt](./docs/index.md#getting-started-with-deepopt).

## Contributing

See the [Contributing Page](./docs/contributing.md).

## Contact Us

Email: [deepopt@llnl.gov](mailto:deepopt@llnl.gov)

Teams (LC users only): [DeepOpt Teams Page](https://teams.microsoft.com/l/team/19%3aZtbEv_dMMAmf5ObemhhCg1rwtlONspUfpOqSHyNYTQg1%40thread.tacv2/conversations?groupId=30e71349-7146-441a-befd-b938f465499a&tenantId=a722dec9-ae4e-4ae3-9d75-fd66e2680a63)

## License

DeepOpt is released under an MIT license. For more information, please see the [LICENSE](./LICENSE.md)
and the [NOTICE](./NOTICE.md).

LLNL-CODE-2006544