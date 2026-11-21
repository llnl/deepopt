# Checkpoints and loading

DeepOpt checkpoints are written with `torch.save`. Only load checkpoints from trusted sources, because PyTorch checkpoint loading can execute Python pickle payloads in older or unrestricted loading modes.

## Modern self-describing checkpoints

Current DeepOpt wrappers save a `deepopt_checkpoint` metadata entry alongside the model state. This metadata lets DeepOpt reload a wrapper for optimization without restating the original data file, bounds, model type, or configuration.

The metadata schema includes:

| Field | Meaning |
| ----- | ------- |
| `schema_version` | DeepOpt checkpoint metadata schema version. |
| `model_type` | One of `GP`, `delUQ`, `nnEnsemble`, `TabPFN`, or `RF`. |
| `training_data` | Dictionary containing `X` and `y` tensors. |
| `bounds` | Bounds tensor with shape `2 x input_dim`, in original input units. |
| `config_settings` | Saved DeepOpt configuration settings. |
| `random_seed` | Seed used by the wrapper. |
| `k_folds` | Fold count used by delUQ training. |
| `multi_fidelity` | Whether the last input column is a fidelity index. |
| `num_fidelities` | Number of fidelity levels. |
| `target` | Neural surrogate target mode, such as `dy`. |
| `data_file` | Original training data path when one was provided. |

Example inspection:

```python
from deepopt.models import get_checkpoint_metadata, is_self_describing_checkpoint

learner_file = "learner_GP.ckpt"
print(is_self_describing_checkpoint(learner_file))
metadata = get_checkpoint_metadata(learner_file)
print(metadata["model_type"])
print(metadata["training_data"].keys())
```

`get_checkpoint_metadata` returns `None` for legacy checkpoints that do not contain DeepOpt metadata. If a checkpoint contains a malformed `deepopt_checkpoint` entry, it raises `ValueError` so broken modern files are not silently treated as old checkpoints.

## Loading wrappers and models

Use `load_deepopt_wrapper` when you want the high-level DeepOpt object with `optimize`, `get_var`, `get_cvar`, and scaling utilities:

```python
from deepopt.models import load_deepopt_wrapper

model = load_deepopt_wrapper("learner_GP.ckpt")
model.optimize(
    outfile="suggested_inputs.npy",
    learner_file="learner_GP.ckpt",
    acq_method="EI",
)
```

Use `load_deepopt_model` when you only need the underlying BoTorch-compatible model:

```python
from deepopt.models import load_deepopt_model

botorch_model = load_deepopt_model("learner_GP.ckpt")
botorch_model.eval()
```

## Legacy checkpoints

Legacy checkpoints do not contain `deepopt_checkpoint` metadata. They can still be loaded by constructing the appropriate wrapper with the original training data, bounds, configuration, and multi-fidelity setting, then calling `load_model`:

```python
import torch
from deepopt.configuration import ConfigSettings
from deepopt.models import GPModel

bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
model = GPModel(
    data_file="sims.npz",
    bounds=bounds,
    config_settings=ConfigSettings("GP"),
)
botorch_model = model.load_model("legacy_learner.ckpt")
```

The CLI follows the same rule: legacy checkpoints require `--infile`, `--bounds`, and `--model-type` during `deepopt optimize`.

## Neural surrogate checkpoint contents

`delUQ` and `nnEnsemble` checkpoints include neural-model state in addition to the DeepOpt metadata. Common keys include:

- `epoch`
- `state_dict`
- `B` for Fourier feature matrices
- `opt_state_dict`
- `output_scaler`
- optional `input_scaler`
- optional `deepopt_checkpoint`

The output scaler is used to restore predictions to original output units. Current checkpoints store DeepOpt's `OutputScaler` state. Older checkpoints may contain BoTorch `Standardize` statistics; DeepOpt reconstructs a compatibility scaler when those fields are present.

## Scaling state

Input bounds and scaler state are part of the public checkpoint contract:

- Single-fidelity inputs are min/max scaled from the provided bounds.
- Multi-fidelity inputs leave the final fidelity column as a rounded integer index.
- Neural surrogate outputs are scaled during training and inverse-transformed for original-scale predictions.
- Per-fidelity output scaling is used when `multi_fidelity=True`.

Prediction helpers such as `get_prediction_with_uncertainty` use `original_scale_x=True` and `original_scale_y=True` by default where supported, so query inputs are interpreted in original input units and returned means/variances are in original output units.

## Random forest checkpoints

`RF` checkpoints store the scalers, configuration, random seed, and training data, but not the fitted forest. On load, DeepOpt refits the forest deterministically from the saved data and `random_seed`, so predictions match the trained model without pickling scikit-learn objects.
