import json

import numpy as np
import pytest
import torch

pytest.importorskip("botorch")
pytest.importorskip("gpytorch")

from deepopt.configuration import ConfigSettings
from deepopt.deltaenc import DeltaEnc
from deepopt.nn_ensemble import NNEnsemble
from deepopt.surrogate_utils import MLP, create_optimizer

pytestmark = pytest.mark.requires_botorch


def _settings(tmp_path, model_type, config):
    path = tmp_path / f"{model_type}.json"
    path.write_text(json.dumps(config))
    return ConfigSettings(model_type, config_file=str(path))


@pytest.mark.requires_botorch
def test_deltaenc_initialization_scales_single_fidelity_outputs(tmp_path, minimal_nn_config):
    settings = _settings(tmp_path, "delUQ", minimal_nn_config)
    network = MLP(settings, unc_type="deltaenc", input_dim=2, output_dim=1, device="cpu")
    optimizer = create_optimizer(network, settings)
    X = torch.tensor([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], dtype=torch.float32)
    y = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)

    model = DeltaEnc(network, settings, optimizer, X, y)

    assert model.actual_batch_size == 3
    assert model.num_outputs == 1
    assert model.batch_shape == torch.Size([])
    torch.testing.assert_close(model.y_train_scaled, torch.tensor([[0.0], [0.5], [1.0]]))
    assert model.train_inputs == (X,)


@pytest.mark.requires_botorch
def test_nnensemble_initialization_scales_single_fidelity_outputs(tmp_path, minimal_nn_config):
    settings = _settings(tmp_path, "nnEnsemble", minimal_nn_config)
    networks = [MLP(settings, unc_type="ensemble", input_dim=2, output_dim=1, device="cpu") for _ in range(2)]
    optimizers = [create_optimizer(network, settings) for network in networks]
    X = torch.tensor([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], dtype=torch.float32)
    y = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)

    model = NNEnsemble(networks, settings, optimizers, X, y)

    assert model.n_estimators == 2
    assert model.actual_batch_size == 3
    assert model.num_outputs == 1
    assert model.batch_shape == torch.Size([])
    torch.testing.assert_close(model.y_train_scaled, torch.tensor([[0.0], [0.5], [1.0]]))
    assert model.train_inputs == (X,)


@pytest.mark.requires_botorch
def test_nnensemble_checkpoint_round_trip(tmp_path, minimal_nn_config):
    settings = _settings(tmp_path, "nnEnsemble", minimal_nn_config)
    networks = [MLP(settings, unc_type="ensemble", input_dim=2, output_dim=1, device="cpu") for _ in range(2)]
    optimizers = [create_optimizer(network, settings) for network in networks]
    X = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    y = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
    model = NNEnsemble(networks, settings, optimizers, X, y)

    model.save_ckpt(str(tmp_path), "ensemble")
    ckpt = torch.load(tmp_path / "ensemble.ckpt")

    assert ckpt["epoch"] == minimal_nn_config["n_epochs"]
    assert len(ckpt["state_dict"]) == 2
    assert len(ckpt["opt_state_dict"]) == 2
    assert len(ckpt["B"]) == 2
