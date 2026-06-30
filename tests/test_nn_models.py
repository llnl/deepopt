import numpy as np
import pytest
import torch

pytest.importorskip("botorch")
pytest.importorskip("gpytorch")

from deepopt.configuration import ConfigSettings
from deepopt.deltaenc import DeltaEnc
from deepopt.nn_ensemble import NNEnsemble
from deepopt.models import DEEPOPT_CHECKPOINT_KEY, NNEnsembleModel, load_deepopt_wrapper
from deepopt.output_scaling import OutputScaler
from deepopt.surrogate_utils import MLP, create_optimizer

pytestmark = pytest.mark.requires_botorch


def _settings(tmp_path, model_type, config):
    settings = ConfigSettings(model_type)
    settings.config_settings.update(config)
    return settings


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
def test_nnensemble_uses_provided_output_scaler(tmp_path, minimal_nn_config):
    settings = _settings(tmp_path, "nnEnsemble", minimal_nn_config)
    networks = [MLP(settings, unc_type="ensemble", input_dim=2, output_dim=1, device="cpu") for _ in range(2)]
    optimizers = [create_optimizer(network, settings) for network in networks]
    X = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32)
    y = torch.tensor([[10.0], [20.0], [100.0], [300.0]], dtype=torch.float32)
    scaler = OutputScaler(multi_fidelity=True, num_fidelities=2, fidelity_dim=1).fit(y, X)

    model = NNEnsemble(networks, settings, optimizers, X, y, multi_fidelity=True, output_scaler=scaler)

    assert model.output_scaler is scaler
    torch.testing.assert_close(model.y_train_scaled, torch.tensor([[0.0], [1.0], [0.0], [1.0]]))


@pytest.mark.requires_botorch
def test_nnensemble_load_recomputes_scaled_training_targets(tmp_path, minimal_nn_config):
    settings = _settings(tmp_path, "nnEnsemble", minimal_nn_config)
    networks = [MLP(settings, unc_type="ensemble", input_dim=2, output_dim=1, device="cpu") for _ in range(2)]
    optimizers = [create_optimizer(network, settings) for network in networks]
    X = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    y = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
    model = NNEnsemble(networks, settings, optimizers, X, y)
    replacement_scaler = OutputScaler().fit(torch.tensor([[10.0], [30.0]], dtype=torch.float32), X)
    torch.save(
        {
            "state_dict": [network.state_dict() for network in networks],
            "B": [network.B for network in networks],
            "output_scaler": replacement_scaler.state_dict(),
        },
        tmp_path / "ensemble.ckpt",
    )

    model.load_ckpt(str(tmp_path), "ensemble")

    torch.testing.assert_close(model.y_train_scaled, torch.tensor([[[-0.5], [-0.45]]]).squeeze(0))
    torch.testing.assert_close(model.y_train_nn, model.y_train_scaled.moveaxis(-2, 0).reshape(model.n_train, -1))


def test_nnensemble_checkpoint_round_trip(tmp_path, minimal_nn_config):
    settings = _settings(tmp_path, "nnEnsemble", minimal_nn_config)
    networks = [MLP(settings, unc_type="ensemble", input_dim=2, output_dim=1, device="cpu") for _ in range(2)]
    optimizers = [create_optimizer(network, settings) for network in networks]
    X = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    y = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
    model = NNEnsemble(networks, settings, optimizers, X, y)

    checkpoint_metadata = {
        "schema_version": 1,
        "model_type": "nnEnsemble",
        "training_data": {"X": X, "y": y},
        "bounds": torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32),
        "config_settings": settings.config_settings,
    }
    model.save_ckpt(str(tmp_path), "ensemble", checkpoint_metadata=checkpoint_metadata)
    ckpt = torch.load(tmp_path / "ensemble.ckpt")

    assert ckpt["epoch"] == minimal_nn_config["n_epochs"]
    assert len(ckpt["state_dict"]) == 2
    assert len(ckpt["opt_state_dict"]) == 2
    assert len(ckpt["B"]) == 2
    assert "output_scaler" in ckpt
    metadata = ckpt[DEEPOPT_CHECKPOINT_KEY]
    assert metadata["schema_version"] == 1
    assert metadata["model_type"] == "nnEnsemble"
    torch.testing.assert_close(metadata["training_data"]["X"], X)
    torch.testing.assert_close(metadata["training_data"]["y"], y)
    torch.testing.assert_close(metadata["bounds"], checkpoint_metadata["bounds"])


def test_deltaenc_checkpoint_metadata_round_trip(tmp_path, minimal_nn_config):
    settings = _settings(tmp_path, "delUQ", minimal_nn_config)
    network = MLP(settings, unc_type="deltaenc", input_dim=2, output_dim=1, device="cpu")
    optimizer = create_optimizer(network, settings)
    X = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    y = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
    model = DeltaEnc(network, settings, optimizer, X, y)
    checkpoint_metadata = {
        "schema_version": 1,
        "model_type": "delUQ",
        "training_data": {"X": X, "y": y},
        "bounds": torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32),
        "config_settings": settings.config_settings,
    }

    model.save_ckpt(str(tmp_path), "delta", checkpoint_metadata=checkpoint_metadata)
    ckpt = torch.load(tmp_path / "delta.ckpt")

    assert ckpt["epoch"] == minimal_nn_config["n_epochs"]
    assert "state_dict" in ckpt
    assert "opt_state_dict" in ckpt
    assert "B" in ckpt
    assert "output_scaler" in ckpt
    metadata = ckpt[DEEPOPT_CHECKPOINT_KEY]
    assert metadata["schema_version"] == 1
    assert metadata["model_type"] == "delUQ"
    torch.testing.assert_close(metadata["training_data"]["X"], X)
    torch.testing.assert_close(metadata["training_data"]["y"], y)
    torch.testing.assert_close(metadata["bounds"], checkpoint_metadata["bounds"])


def test_load_deepopt_wrapper_reconstructs_nnensemble(tmp_path, minimal_nn_config):
    settings = _settings(tmp_path, "nnEnsemble", minimal_nn_config)
    X = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    y = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    checkpoint = {
        "state_dict": [],
        "B": [],
        "output_scaler": OutputScaler().fit(y, X).state_dict(),
        DEEPOPT_CHECKPOINT_KEY: {
            "schema_version": 1,
            "model_type": "nnEnsemble",
            "training_data": {"X": X, "y": y},
            "bounds": bounds,
            "config_settings": settings.config_settings,
            "random_seed": 123,
            "k_folds": 2,
            "multi_fidelity": False,
            "target": "dy",
            "data_file": None,
        },
    }
    torch.save(checkpoint, tmp_path / "ensemble.ckpt")

    wrapper = load_deepopt_wrapper(str(tmp_path / "ensemble.ckpt"), device="cpu")

    assert isinstance(wrapper, NNEnsembleModel)
    assert wrapper.config_settings.get_setting("n_estimators") == minimal_nn_config["n_estimators"]
    torch.testing.assert_close(wrapper.full_train_X, X)
    torch.testing.assert_close(wrapper.full_train_Y, y)
    assert wrapper.multi_fidelity is False
