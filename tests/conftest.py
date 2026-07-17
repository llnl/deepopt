import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def single_fidelity_data_file(tmp_path: Path) -> Path:
    X = np.array(
        [
            [0.0, 0.0],
            [0.25, 0.5],
            [0.5, 0.25],
            [0.75, 1.0],
        ],
        dtype=np.float32,
    )
    y = -(X**2).sum(axis=1)
    path = tmp_path / "single_fidelity.npz"
    np.savez(path, X=X, y=y)
    return path


@pytest.fixture
def multi_fidelity_data_file(tmp_path: Path) -> Path:
    X = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.4, 0.0],
            [0.5, 0.5, 1.0],
            [0.8, 0.1, 1.0],
        ],
        dtype=np.float32,
    )
    y = np.array([0.0, 0.2, 1.0, 1.2], dtype=np.float32)
    path = tmp_path / "multi_fidelity.npz"
    np.savez(path, X=X, y=y)
    return path


@pytest.fixture
def minimal_nn_config() -> dict:
    return {
        "n_estimators": 2,
        "ff": False,
        "dist": "uniform",
        "mapping_size": 4,
        "n_layers": 2,
        "hidden_dim": 4,
        "activation": "relu",
        "dropout": False,
        "dropout_prob": 0.0,
        "batchnorm": False,
        "w0": 30,
        "activation_first": True,
        "opt_type": "Adam",
        "learning_rate": 0.001,
        "weight_decay": 0,
        "n_epochs": 0,
        "batch_size": 4,
        "variance": 0.001,
    }


@pytest.fixture
def minimal_nn_config_file(tmp_path: Path, minimal_nn_config: dict) -> Path:
    path = tmp_path / "nn_config.json"
    path.write_text(json.dumps(minimal_nn_config))
    return path
