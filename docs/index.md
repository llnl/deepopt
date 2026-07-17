---
hide:
  - navigation
---

# DeepOpt

Easily bring Bayesian optimization to your workflows with DeepOpt!

[On GitHub :fontawesome-brands-github:](https://github.com/LLNL/deepopt){ .md-button .md-button--primary }
<!--[On GitLab :fontawesome-brands-gitlab:](https://lc.llnl.gov/gitlab/deepopt-devs/deepopt){ .md-button .md-button--primary }-->

## Why DeepOpt?

DeepOpt is a powerful and versatile Bayesian optimization framework that provides users with the flexibility to choose between Gaussian process (GP) and neural network (NN) surrogates. This flexibility empowers users to select the most suitable surrogate model for their specific optimization problem, taking into account factors such as the complexity of the objective function and the available computational resources.

## Getting Started with DeepOpt

Install deepopt with:

```bash
pip install deepopt
```

See the [Installation page](./user_guide/installation.md) for more installation instructions.

The DeepOpt library requires initial data so choose an objective function and "simulate" some initial data:

```py title="generate_simulation_inputs.py" linenums="1"
import torch
import numpy as np

input_dim = 5
num_points = 10

X = torch.rand(num_points, input_dim)
y = -(X**2).sum(axis=1)

np.savez('sims.npz', X=X, y=y)  #(1)
```

1. Save data to file `sims.npz`

We can now generate the `sims.npz` file with:

```bash
python generate_simulation_inputs.py
```

From here we can either use the DeepOpt API or we can use the DeepOpt CLI.

If you're using the DeepOpt API, you'll first need to load the appropriate DeepOpt model (for this example this will be a `GPModel`):

```py linenums="1" title="run_deepopt.py"
import torch
from deepopt.configuration import ConfigSettings
from deepopt.deepopt_cli import get_deepopt_model

input_dim = 5 #(1)
model_type = "GP"  #(2)
model_class = get_deepopt_model(model_type=model_type) #(3)
cs = ConfigSettings(model_type=model_type) #(4)
bounds = torch.FloatTensor(input_dim*[[0,1]]).T  #(5)
model = model_class(data_file="sims.npz", bounds=bounds, config_settings=cs)  #(6)
```

1. Input dimension must match data file (`sims.npz` in this case)
2. We'll use a GP model for simplicity
3. The `get_deepopt_model` function will return the appropriate class to use for your `model_type`: `GPModel` for `GP`, `DelUQModel` for `delUQ`, and `NNEnsembleModel` for `nnEnsemble`. These models include all the high-level functionality required for the `learn` and `optimize` commands.
4. This sets up the model configuration. Since we don't pass a configuration file, the default configuration will be used.
5. Learning and optimizing will take place within these input bounds
6. Initialize the GP model

Next, we'll train a GP surrogate on the data and save the model that's created to an output file called `learner_GP.ckpt`:

=== "DeepOpt API"

    ```py title="run_deepopt.py" linenums="11"
    model.learn(outfile=f"learner_{model_type}.ckpt") #(1)
    ```

    1. This will name the outfile as `learner_GP.ckpt` for this example. If you modify the `model_type` variable this name will change.

=== "DeepOpt CLI"

    ```bash
    bounds="[[0,1],[0,1],[0,1],[0,1],[0,1]]" #(1)
    deepopt learn -i sims.npz -o learner_GP.ckpt -b $bounds
    ```
    1. This defines the appropriate JSON-encoded `bounds` variable to match `input_dim`.

The checkpoint files saved by DeepOpt use `torch.save` under the hood. They are Python dictionaries and can be viewed using `torch.load`:
```py title="view_ckpt.py" linenums="1"
import torch
from deepopt.models import get_checkpoint_metadata, load_deepopt_model, load_deepopt_wrapper

model_type = 'GP'
learner_file = f'learner_{model_type}.ckpt'
ckpt = torch.load(learner_file)
print(ckpt.keys())
print(get_checkpoint_metadata(learner_file).keys())
wrapper = load_deepopt_wrapper(learner_file)
botorch_model = load_deepopt_model(learner_file)
```

Modern checkpoints are self-describing: they include the model type, training data, bounds, configuration settings, and scaler metadata needed to reload the wrapper without restating those inputs.

Now that we have a model that's trained, we'll use this model to propose new points using [Expected Improvement (EI)](./user_guide/acquisition_functions.md#ei):

=== "DeepOpt API"
    ```py title="run_deepopt.py" linenums="12"
    model.optimize(outfile="suggested_inputs.npy", 
                   learner_file=f"learner_{model_type}.ckpt", 
                   acq_method="EI") #(1)
    ```

    1. Use Expected Improvement to acquire new points based on the model saved in `learner_file` and save those points as a `numpy` array in `outfile`.


=== "DeepOpt CLI"

    ```bash
    deepopt optimize -o suggested_inputs.npy -l learner_GP.ckpt -a EI
    ```

If you're using the DeepOpt API, we can now run our script with:

```bash
python run_deepopt.py
```

The saved file `suggested_inputs.npy` is a `numpy` save file containing the array of new points with dimension Nxd (N= # of new points, d = input dimensions). We can view the file using `numpy.load`:
```bash
python -c "import numpy as np; print(np.load('suggested_inputs.npy'))"
```

## DeepOpt's Goals and Motivations

The primary motivation behind the development of DeepOpt was to address the limitations of traditional Bayesian optimization frameworks, particularly GP-based approaches, in handling large design spaces and large datasets. GP-based Bayesian optimization has proven to be a valuable tool for a wide range of optimization problems, but its performance can degrade significantly as the dimensionality of the design space or the size of the dataset increases.

A significant limitation of GP-based Bayesian optimization is its computational complexity, which grows quadratically with the number of data points. This means that fitting a GP surrogate model to a large dataset can become extremely time-consuming, and evaluating the surrogate model can also become prohibitively expensive. As a result, GP-based Bayesian optimization is often impractical for problems involving more than a few hundred data points.

To address this limitation, DeepOpt was designed with a focus on efficient optimization for high dimensionalities and large datasets. One key feature of DeepOpt is its support for NN surrogates. Unlike GPs, which require matrix inversions that scale quadratically with the number of data points, NN surrogates can be trained and evaluated efficiently even for high dimensionalities and large datasets. This makes NN surrogates a much more practical choice for Bayesian optimization problems involving large datasets.

In addition to the support for NNs, Deepopt was designed with the following goals in mind:

- **Ease of Use:** DeepOpt's intuitive interface and well-documented API make it accessible to users with varying levels of expertise, allowing them to quickly apply Bayesian optimization to their specific problems.

- **Versatility:** DeepOpt supports a wide range of acquisition functions, enabling users to select the most appropriate one for their specific optimization problem.

- **Speed:** DeepOpt seamlessly integrates with the Delta-UQ framework, providing uncertainty quantification at much faster speeds than is typical for other UQ methods used on NNs.

## Release

DeepOpt is released under an MIT license. For more information, please see the [LICENSE](../LICENSE.md)
and the [NOTICE](../NOTICE.md).

``LLNL-CODE-2006544``
