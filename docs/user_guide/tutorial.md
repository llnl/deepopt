# Tutorial
In this page we'll present several tutorials on how to use different parts of the DeepOpt library.

## Tutorial: Using neural network surrogates

One of the powerful features of DeepOpt is the ability to use neural network surrogates in place of Gaussian process surrogates during optimization. In this tutorial, we'll repeat the [Getting Started](../index.md#getting-started-with-deepopt) example, but using neural networks in place of Gaussian process. One key difference is the need for a neural network configuration file that specifies the network architecture, activation functions, and a few other parameters. We'll describe these in detail as we move along.

### Create the initial data
Just as in [Getting Started](../index.md#getting-started-with-deepopt), we'll put some initial data in a `sims.npz` file

```py title="generate_simulation_inputs.py" linenums="1"
import torch
import numpy as np

input_dim = 5
num_points = 10

X = torch.rand(num_points, input_dim)
y = -(X**2).sum(axis=1)

np.savez('sims.npz', X=X, y=y) #(1)
```

1. Save data to file `sims.npz`

We can now generate the `sims.npz` file with:

```bash
python generate_simulation_inputs.py
```

### Default neural network
DeepOpt currently supports the ensemble method for uncertainty quantification with neural networks. The naming convention of the model reflects this, so to use neural networks, we set the `model_type` to "nnEnsemble" and the mode class is called `NNEnsembleModel`.
<!-- Neural networks in DeepOpt use the ["delta-UQ"](https://arxiv.org/abs/2110.02197) method for uncertainty quantification. The naming conventions reflect this, so to use neural networks, we set the `model_type` to "delUQ" and the model class is called `DelUQModel`. -->

From here we can either use the DeepOpt API or we can use the DeepOpt CLI.

If you're using the DeepOpt API, you'll first need to load the `ConfigSettings` class:

```py linenums="1" title="run_deepopt.py"
import torch
from deepopt.configuration import ConfigSettings
from deepopt.deepopt_cli import get_deepopt_model

input_dim = 5 #(1)
model_type = 'nnEnsemble' #(2)
model_class = get_deepopt_model(model_type=model_type) #(3)
cs = ConfigSettings(model_type=model_type) #(4)
bounds = torch.FloatTensor(input_dim*[[0,1]]).T  #(5)
model = model_class(data_file='sims.npz', bounds=bounds, config_settings=cs)  #(6)
```

1. Input dimension must match data file (`sims.npz` in this case)
2. Set the model type to use throughout the script.
3. Set the model class associated with the selected model type (in this case `NNEnsembleModel`)
4. This sets up the neural network configuration (more generally the model configuration). Since we don't pass a configuration file, the default configuration will be used.
5. Learning and optimizing will take place within these input bounds
6. Model is loaded the same way as with GP, but now we are using `NNEnsembleModel`

Training and optimizing are done as in [Getting Started](../index.md#getting-started-with-deepopt), with the array of new points being recorded in 'suggested_inputs.npy':

=== "DeepOpt API"
    ```py title="run_deepopt.py" linenums="11"
    model.learn(outfile=f'learner_{model_type}.ckpt') #(1)
    ```

    1. Train the neural network and save its state to a checkpoint file.

=== "DeepOpt CLI"
    ```bash
    bounds="[[0,1],[0,1],[0,1],[0,1],[0,1]]" #(1)
    deepopt learn -i sims.npz -o learner_nnEnsemble.ckpt -m nnEnsemble -b $bounds
    ```

    1. This defines the appropriate JSON-encoded `bounds` variable to match `input_dim`.

The checkpoint files saved by DeepOpt use `torch.save` under the hood. They are Python dictionaries and can be viewed using `torch.load`:
```py title="view_ckpt.py" linenums="1"
import torch
from deepopt.models import get_checkpoint_metadata, is_self_describing_checkpoint, load_deepopt_wrapper

model_type = 'nnEnsemble'
learner_file = f'learner_{model_type}.ckpt'
ckpt = torch.load(learner_file)
print(ckpt.keys())
print(is_self_describing_checkpoint(learner_file))
print(get_checkpoint_metadata(learner_file).keys())
model = load_deepopt_wrapper(learner_file)
```

Modern checkpoints may include model-specific state such as `epoch`, `state_dict`, `B`, and `opt_state_dict`, plus scaler state (`output_scaler` and optionally `input_scaler`). Checkpoints saved by current DeepOpt wrappers also include a `deepopt_checkpoint` metadata entry with the model type, training data, bounds, and configuration settings needed to reload the wrapper.

Now that we saved the trained model, we can use it to propose new candidate points:

=== "DeepOpt API"
    ```py title="run_deepopt.py" linenums="12"
    model.optimize(outfile='suggested_inputs.npy',
                   learner_file=f'learner_{model_type}.ckpt',
                   acq_method='EI') #(1)
    ```

    1. Use [Expected Improvement](./acquisition_functions.md#ei) to acquire new points based on the model saved in `learner_file` and save those points as a `numpy` array in `outfile`.

=== "DeepOpt CLI"
    ```bash
    deepopt optimize -o suggested_inputs.npy -l learner_nnEnsemble.ckpt -a EI
    ```

    Current DeepOpt checkpoints are self-describing, so `optimize` can reload the training data, bounds, model type, and configuration from `learner_nnEnsemble.ckpt`. Legacy checkpoints without `deepopt_checkpoint` metadata still require `--infile`, `--bounds`, and `--model-type`.

If you're using the DeepOpt API, we can now run our script with:

```bash
python run_deepopt.py
```

The saved file `suggested_inputs.npy` is a `numpy` save file containing the array of new points with dimension `N X d` (`N`: # of new points, `d`: # of input dimensions). We can view the file using `numpy.load`:
```bash
python -c "import numpy as np; print(np.load('suggested_inputs.npy'))"
```

### Changing the neural network configuration
Simply create a configuration yaml file with the desired entries (available settings described [here](configuration.md)). Training settings are loaded when you construct the model configuration, and optimize-time settings can be provided through an `optimization:` section in the same file format.

=== "DeepOpt API"
    ```py title="run_deepopt.py" linenums="11"
    cs = ConfigSettings(model_type=model_type, config_file='config.yaml')
    model = model_class(data_file='sims.npz', bounds=bounds, config_settings=cs)
    model.learn(outfile=f'learner_{model_type}.ckpt') #(1)

    model.optimize(outfile='suggested_inputs.npy',
                   learner_file=f'learner_{model_type}.ckpt',
                   acq_method='EI') #(2)
    ```

    1. Train the neural network ensemble and save its state to a checkpoint file.
    2. Use [Expected Improvement](./acquisition_functions.md#ei) to acquire new points based on the model saved in `learner_file` and save those points as a `numpy` array in `outfile`.

=== "DeepOpt CLI"
    ```bash
    bounds="[[0,1],[0,1],[0,1],[0,1],[0,1]]" #(1)
    deepopt learn -i sims.npz -o learner_nnEnsemble.ckpt -m nnEnsemble -b $bounds -c config.yaml #(2)

    deepopt optimize -o suggested_inputs.npy -l learner_nnEnsemble.ckpt -a EI -c config.yaml #(3)
    ```

    1. This defines the appropriate JSON-encoded `bounds` variable to match `input_dim`.
    2. Train the neural network ensemble and save its state to a checkpoint file.
    3. Use [Expected Improvement](./acquisition_functions.md#ei) with optimize-time settings from `config.yaml`. With self-describing checkpoints, `optimize` does not need `--infile`, `--bounds`, or `--model-type`.


## Tutorial: Iterative optimization

Typical Bayesian optimization workflows will have an iterative structure, as proposed candidates from one iteration are added to the training of the surrogate in the following iteration. We demonstrate how to use the DeepOpt API to accomplish this. A similar workflow is possible in CLI, but it is best to use a workflow manager such as [Merlin](https://merlin.readthedocs.io/en/latest/).

```py title='iterative_optimization.py' linenums="1"
import torch
import numpy as np
from deepopt.configuration import ConfigSettings
from deepopt.deepopt_cli import get_deepopt_model
import os

def objective(X):
    return -(X**2).sum(axis=1)

input_dim = 5
num_initial_points = 10

X_init = torch.rand(num_initial_points, input_dim)
y_init = objective(X_init)

# Create an output directory to store the files generated by DeepOpt
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
"opt_results")
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

#We'll store all data in npz files with keys 'X' for inputs and 'y' for outputs
np.savez(f"{output_dir}/points_iter0.npz",X=X_init,y=y_init) 

model_type = "GP"
cs = ConfigSettings(model_type=model_type)   
model_class = get_deepopt_model(model_type) 
bounds = torch.FloatTensor(input_dim*[[0,1]]).T

n_iterations = 20
for i in range(n_iterations):
    print(f"---------------\n  ITERATION {i+1}  \n---------------")
    data_prev_file = f"{output_dir}/points_iter{i}.npz" # previous points
    candidates_file = f"{output_dir}/suggested_inputs_iter{i+1}.npy" # save new proposed inputs to evaluate
    ckpt_file = f"{output_dir}/learner_{model_type}_iter{i+1}.ckpt" # save model for this iteration

    # Train model on previous points, then propose new points using Expected Improvement
    model = model_class(data_file=data_prev_file,bounds=bounds,config_settings=cs)
    model.learn(outfile=ckpt_file)
    model.optimize(outfile=candidates_file, learner_file=ckpt_file, acq_method="EI")

    # Load previous input & output values
    data_prev = np.load(data_prev_file)
    X_prev = data_prev["X"]
    y_prev = data_prev["y"]

    X_new = np.load(candidates_file) # Load proposed inputs
    y_new = objective(X_new) # Evaluate proposed inputs

    # Concatenate new input & output values with previous and save
    X = np.concatenate([X_prev,X_new],axis=0)
    y = np.concatenate([y_prev,y_new],axis=0)
    np.savez(f"{output_dir}/points_iter{i+1}.npz",X=X,y=y)
```

This workflow produces `n_iterations` npy files of suggested inputs and `n_iterations + 1` npz files of points. The npy files contain just the inputs to call the objective function with, so are redundant (in this example) with the npz files that contain both. We can visualize the optimization with the following script:

```py title='plot_optimization.py' linenums="1"
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

# Make sure these match the iterative_optimization.py file
num_initial_points = 10
num_iterations = 20

#Read in results directory and directory in which to save figure from command line or default to local subdirectory
parser = argparse.ArgumentParser(description="Read output directory.")
parser.add_argument("results_directory", type=str, 
                    help="directory of optimization results", 
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "opt_results"))
parser.add_argument("figsave_directory", type=str, 
                    help="directory to save figure",
                    default=os.path.dirname(os.path.abspath(__file__)))
args = parser.parse_args()
results_dir = args.results_directory
figsave_dir = args.figsave_directory

pts = np.load(f'{results_dir}/points_iter{num_iterations}.npz') # Load last collection of points
y = pts['y'] # Extract objective values
pts_per_iteration = (len(y)-num_initial_points)//num_iterations # Work out how many points were proposed per iteration
iters = np.concatenate([np.zeros(num_initial_points),np.repeat(
    np.arange(1,num_iterations+1),pts_per_iteration)]) # Set the iteration value of all points
plt.scatter(iters,y) # Plot all proposal results vs. iteration
plt.xlabel('Iterations')
plt.xticks(np.arange(num_iterations+1))
plt.ylabel('Objective value')

#Find running max:
max_byiter = np.zeros(num_iterations+1) # This will store max for each iteration
max_byiter[0] = np.max(y[:num_initial_points]) # Max of initial points
for i in range(1,num_iterations+1):
    max_byiter[i] = np.max(y[num_initial_points+(
        i-1)*pts_per_iteration:num_initial_points+i*pts_per_iteration]) # Max of each iteration (when multiple proposals)
plt.plot(np.arange(num_iterations+1),np.maximum.accumulate(
    max_byiter),label='running max') # Add running max to plot
plt.legend()
plt.savefig(f'{figsave_dir}/optimization_plot.png')
plt.show()
```
The generated figure should look like this: ![Optimization of an inverted paraboloid using Expected Improvement](../imgs/optimization_plot.png)

The plot shows a running max that converges to the objective maximum (0 in this example), while individual proposals are a mix of high values (at or near the running max) from succesful exploitation/exploration and low values (far below the running max) from unsuccesful exploration.

## Tutorial: Multi-fidelity optimization
In this tutorial, we'll walk through how to accomplish multi-fidelity optimization with DeepOpt. We'll start by copying the `generate_simulation_inputs.py` and `run_deepopt.py` file from the [Getting Started page](../index.md#getting-started-with-deepopt):

```bash
cp generate_simulation_inputs.py generate_simulation_inputs_mf.py
cp run_deepopt.py run_deepopt_mf.py
```

Performing multi-fidelity optimization with DeepOpt requires that the data files have the last input column as a fidelity, with integer values ranging from 0 to number of fidelities - 1. We can generate this input file by changing one line in the `generate_simulation_inputs.py` file:

```py title="generate_simulation_inputs_mf.py" linenums="1", hl_lines="8"
import torch
import numpy as np

input_dim = 5 # Last dimension will be used for fidelity
num_points = 10

X = torch.rand(num_points, input_dim)
X[:,-1] = X[:,-1].round()
y = -(X**2).sum(axis=1) #(1)

np.savez('sims.npz', X=X, y=y)
```

1. This line now produces paraboloids at two fidelities, with the high fidelity paraboloid shifted one unit relative to the low fidelity paraboloid.

We run this script with 
```bash
python generate_simulation_inputs_mf.py
```

Additionally, to achieve multi-fidelity optimization with DeepOpt you must:

1. Enable multi-fidelity in your model
2. Pass a list of fidelity costs to the optimize method (the length of the list must match the number of fidelities)
3.  Select an [acquisition function](./acquisition_functions.md) that's appropriate for multi-fidelity optimization (currently only [Knowledge Gradient](./acquisition_functions.md#kg) and [Max Value Entropy](./acquisition_functions.md#maxvalentropy) are supported)

Below, we show how to modify the `optimize` call from `run_deepopt.py` to accommodate these changes, and also how to achieve this same functionality from the command line:

=== "DeepOpt API"
    ```py title="run_deepopt_mf.py" linenums="1" hl_lines="10-11 13-15"
    import torch
    from deepopt.configuration import ConfigSettings
    from deepopt.deepopt_cli import get_deepopt_model

    input_dim = 5
    model_type = "GP"
    model_class = get_deepopt_model(model_type)
    cs = ConfigSettings(model_type=model_type)
    bounds = torch.FloatTensor(input_dim*[[0,1]]).T 
    model = model_class(data_file="sims.npz", bounds=bounds, 
                        config_settings=cs, multi_fidelity=True)
    model.learn(outfile=f"learner_{model_type}.ckpt")
    model.optimize(outfile="suggested_inputs.npy",
                   learner_file=f"learner_{model_type}.ckpt",
                   acq_method="KG",fidelity_cost=[1,6]) #(1)
    ```

    1. We use the [Knowledge Gradient (KG)](./acquisition_functions.md#kg) multi-fidelity acquisition function with a 1:6 ratio of low:high fidelity costs.

=== "DeepOpt CLI"
    ```bash
    deepopt optimize -l learner_GP.ckpt -o suggested_inputs.npy \
    -a KG --fidelity-cost "[1,6]" #(1)
    ```

    1. We use the [Knowledge Gradient (KG)](./acquisition_functions.md#kg) multi-fidelity acquisition function with a 1:6 ratio of low:high fidelity costs.

The API script can be run with
```bash
python run_deepopt_mf.py
```

We can make an iterative optimization workflow similar to the single-fidelity case above:
```py title='iterative_optimization_mf.py' linenums="1" hl_lines="13 38-39 41-42"
import torch
import numpy as np
from deepopt.configuration import ConfigSettings
from deepopt.deepopt_cli import get_deepopt_model

def objective(X):
    return -(X**2).sum(axis=1)

input_dim = 5
num_initial_points = 10

X_init = torch.rand(num_initial_points, input_dim)
X_init[:,-1] = X_init[:,-1].round()
y_init = objective(X_init)

# Create an output directory to store the files generated by DeepOpt
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
"opt_results")
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

#We'll store all data in npz files with keys 'X' for inputs and 'y' for outputs
np.savez(f"{output_dir}/points_iter0.npz",X=X_init,y=y_init) 

model_type = "GP"
cs = ConfigSettings(model_type=model_type)   
model_class = get_deepopt_model(model_type) 
bounds = torch.FloatTensor(input_dim*[[0,1]]).T

n_iterations = 20
for i in range(n_iterations):
    print(f"---------------\n  ITERATION {i+1}  \n---------------")
    data_prev_file = f"{output_dir}/points_iter{i}.npz" # previous points
    candidates_file = f"{output_dir}/suggested_inputs_iter{i+1}.npy" # save new proposed inputs to evaluate
    ckpt_file = f"{output_dir}/learner_{model_type}_iter{i+1}.ckpt" # save model for this iteration

    # Train model on previous points, then propose new points using Expected Improvement
    model = model_class(data_file=data_prev_file,bounds=bounds,
                        config_settings=cs,multi_fidelity=True)
    model.learn(outfile=ckpt_file)
    model.optimize(outfile=candidates_file, learner_file=ckpt_file, 
                   acq_method="KG",fidelity_cost=[1,6])

    # Load previous input & output values
    data_prev = np.load(data_prev_file)
    X_prev = data_prev["X"]
    y_prev = data_prev["y"]

    X_new = np.load(candidates_file) # Load proposed inputs
    y_new = objective(X_new) # Evaluate proposed inputs

    # Concatenate new input & output values with previous and save
    X = np.concatenate([X_prev,X_new],axis=0)
    y = np.concatenate([y_prev,y_new],axis=0)
    np.savez(f"{output_dir}/points_iter{i+1}.npz",X=X,y=y)
```

Just as before, we have `n_iterations` npy files of suggested inputs and `n_iterations + 1` npz files of points. To visualize, we just need to distinguish between low and high fidelity candidates and make sure the running max is only over high fidelity candidates:

```py title='plot_optimization_mf.py' linenums="1" hl_lines="22 26-27 34-40 42"
import numpy as np
import matplotlib.pyplot as plt

# Make sure these match the iterative_optimization.py file
num_initial_points = 10
num_iterations = 20

#Read in results directory and directory to save figure to from command line or default to local subdirectory
parser = argparse.ArgumentParser(description="Read output directory.")
parser.add_argument("results_directory", type=str, 
                    help="directory of optimization results", 
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "opt_results"))
parser.add_argument("figsave_directory", type=str, 
                    help="directory to save figure",
                    default=os.path.dirname(os.path.abspath(__file__)))
args = parser.parse_args()
results_dir = args.results_directory
figsave_dir = args.figsave_directory

pts = np.load(f'{results_dir}/points_iter{num_iterations}.npz') # Load last collection of points
y = pts['y'] # Extract objective values
fids = pts['X'][:,-1] # Extract fidelity values
pts_per_iteration = (len(y)-num_initial_points)//num_iterations # Work out how many points were proposed per iteration
iters = np.concatenate([np.zeros(num_initial_points),np.repeat(
    np.arange(1,num_iterations+1),pts_per_iteration)]) # Set the iteration value of all points
plt.scatter(iters[fids==0],y[fids==0],'tab:blue',label='low fidelity') # Plot low fidelity points in blue
plt.scatter(iters[fids==1],y[fids==1],'tab:orange',label='high fidelity') # Plot high fidelity points in orange
plt.xlabel('Iterations')
plt.xticks(np.arange(num_iterations+1))
plt.ylabel('Objective value')

#Find running max:
max_byiter = np.zeros(num_iterations+1) # This will store max for each iteration
iters_high = iters[fids==1] # Iterations for high fidelity candidates
y_high = y[fids==1] # High fidelity candidates
max_byiter[0] = np.max(y_high[iters_high==0]) # Max of initial points
for i in range(1,num_iterations+1):
    max_byiter[i] = np.max(y_high[iters_high==i]) if i in iters_high else max_byiter[i-1] # Max of high fidelity candidates for each iteration (when none, keep previoius running max)
plt.plot(np.arange(num_iterations+1),np.maximum.accumulate(
    max_byiter),color='tab:orange',label='running max') # Add running max to plot
plt.legend()
plt.savefig(f'{figsave_dir}/mf_optimization_plot.png')
plt.show()
```

The generated figure should look like this: ![Multi-fidelity optimization of two inverted paraboloids using Knowledge Gradient](../imgs/mf_optimization_plot.png)

The plot shows a running max in orange that converges to the objective maximum (-1 in this example), while individual proposals are a mix of low and high fidelity candidates. Note that the low fidelity maximum here is higher than the high fidelity one, but we are only interested in finding the latter.

## Tutorial: Risk-averse optimization
!!! note

    Currently only [EI](./acquisition_functions.md#ei), [NEI](./acquisition_functions.md#nei), and [KG](./acquisition_functions.md#kg) [acquisition functions](./acquisition_functions.md) support risk measures.

To do risk-averse optimization, simply specify the `risk_measure` (CLI: `--risk-measure`), `risk_level` (CLI: `--risk-level`), `risk_n_deltas` (CLI: `--risk-n-deltas`), and `x_stddev` (CLI: `--X-stddev`) when calling `optimize`.

The available risk measures are VaR (Value-at-Risk) and CVaR (Conditional Value-at-Risk). The risk level is between 0 and 1 and sets the corresponding alpha value (see the BoTorch [example](https://botorch.org/tutorials/risk_averse_bo_with_environmental_variables) for more details). `MaxValEntropy` does not currently support risk measures.

`risk_n_deltas` sets the number of samples to draw for input perturbations (more accuracy and longer run time for larger values). 

`x_stddev` sets the size of the input perturbations in each dimension (can provide a list to specify dimension-by-dimension or a scalar to set the same pertrubation for all inputs).
