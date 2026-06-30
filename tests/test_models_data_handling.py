import numpy as np
import pytest
import torch

pytest.importorskip("botorch")
pytest.importorskip("gpytorch")
pytest.importorskip("ray")

from deepopt.configuration import ConfigSettings
from deepopt.models import (
    DEEPOPT_CHECKPOINT_KEY,
    DeepoptBaseModel,
    DeepOptSingleTaskGP,
    FidelityCostModel,
    GPModel,
    load_deepopt_model,
)

pytestmark = pytest.mark.requires_botorch


@pytest.mark.requires_botorch
def test_base_model_single_fidelity_normalizes_data(single_fidelity_data_file):
    settings = ConfigSettings("GP")
    bounds = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)

    model = GPModel(
        data_file=str(single_fidelity_data_file),
        bounds=bounds,
        config_settings=settings,
        device="cpu",
    )

    assert model.device == "cpu"
    assert model.num_fidelities == 1
    assert model.target_fidelities == {1: 0}
    assert model.full_train_Y.shape == torch.Size([4, 1])
    assert model.full_train_X.dtype == torch.float32
    assert model.full_train_Y.dtype == torch.float32
    torch.testing.assert_close(model.full_train_Y_scaled, torch.tensor([[1.0], [0.8], [0.8], [0.0]]))
    torch.testing.assert_close(model.full_train_X, model.X_orig)
    torch.testing.assert_close(model.bounds, torch.tensor(bounds, dtype=torch.float32))


@pytest.mark.requires_botorch
def test_base_model_multi_fidelity_rounds_fidelity_column(multi_fidelity_data_file):
    settings = ConfigSettings("GP")
    bounds = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)

    model = GPModel(
        data_file=str(multi_fidelity_data_file),
        bounds=bounds,
        config_settings=settings,
        multi_fidelity=True,
        device="cpu",
    )

    assert model.num_fidelities == 2
    assert model.target_fidelities == {2: 1}
    assert model.full_train_Y.shape == torch.Size([4, 1])
    torch.testing.assert_close(model.full_train_X[:, -1], torch.tensor([0.0, 0.0, 1.0, 1.0]))
    torch.testing.assert_close(model.full_train_Y_scaled, torch.tensor([[0.0], [1.0], [0.0], [1.0]]))
    torch.testing.assert_close(model.output_scaler.y_min, torch.tensor([[0.0], [1.0]]))
    torch.testing.assert_close(model.output_scaler.y_max, torch.tensor([[0.2], [1.2]]))


@pytest.mark.requires_botorch
def test_base_model_training_data_matches_file_initialization(single_fidelity_data_file):
    settings = ConfigSettings("GP")
    bounds = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    file_model = GPModel(
        data_file=str(single_fidelity_data_file),
        bounds=bounds,
        config_settings=settings,
        device="cpu",
    )
    data_model = GPModel(
        training_data={"X": file_model.X_orig, "y": file_model.Y_orig},
        bounds=bounds,
        config_settings=ConfigSettings("GP"),
        device="cpu",
    )

    torch.testing.assert_close(data_model.full_train_X, file_model.full_train_X)
    torch.testing.assert_close(data_model.full_train_Y, file_model.full_train_Y)
    torch.testing.assert_close(data_model.full_train_Y_scaled, file_model.full_train_Y_scaled)


@pytest.mark.requires_botorch
def test_gp_uses_scaled_training_outputs_and_public_prediction_units(monkeypatch, tmp_path, single_fidelity_data_file):
    settings = ConfigSettings("GP")
    bounds = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    wrapper = GPModel(
        data_file=str(single_fidelity_data_file),
        bounds=bounds,
        config_settings=settings,
        device="cpu",
    )

    def fake_fit(_mll):
        return None

    monkeypatch.setattr("deepopt.models.fit_gpytorch_model", fake_fit)
    model = wrapper.train(str(tmp_path / "gp.pt"))

    checkpoint = torch.load(tmp_path / "gp.pt")
    metadata = checkpoint[DEEPOPT_CHECKPOINT_KEY]

    assert isinstance(model, DeepOptSingleTaskGP)
    assert metadata["model_type"] == "GP"
    assert metadata["schema_version"] == 1
    torch.testing.assert_close(metadata["training_data"]["X"], wrapper.X_orig.cpu())
    torch.testing.assert_close(metadata["training_data"]["y"], wrapper.Y_orig.cpu())
    torch.testing.assert_close(metadata["bounds"], torch.tensor(bounds, dtype=torch.float32))
    assert not hasattr(model, "outcome_transform")
    torch.testing.assert_close(model.train_targets.unsqueeze(-1), wrapper.full_train_Y_scaled)
    mean_scaled, var_scaled = model.get_prediction_with_uncertainty(wrapper.full_train_X[:1], original_scale=False)
    mean_original, var_original = model.get_prediction_with_uncertainty(wrapper.full_train_X[:1], original_scale=True)
    torch.testing.assert_close(mean_original, wrapper.output_scaler.inverse_transform(mean_scaled, wrapper.full_train_X[:1]))
    torch.testing.assert_close(var_original, wrapper.output_scaler.inverse_variance(var_scaled, wrapper.full_train_X[:1]))


@pytest.mark.requires_botorch
def test_load_deepopt_model_loads_modern_gp_without_external_inputs(monkeypatch, tmp_path, single_fidelity_data_file):
    settings = ConfigSettings("GP")
    bounds = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    wrapper = GPModel(
        data_file=str(single_fidelity_data_file),
        bounds=bounds,
        config_settings=settings,
        device="cpu",
    )

    def fake_fit(_mll):
        return None

    monkeypatch.setattr("deepopt.models.fit_gpytorch_model", fake_fit)
    wrapper.train(str(tmp_path / "gp.pt"))

    model = load_deepopt_model(str(tmp_path / "gp.pt"), device="cpu")

    assert isinstance(model, DeepOptSingleTaskGP)
    assert hasattr(model, "output_scaler")


@pytest.mark.requires_botorch
def test_gp_loads_legacy_standardize_checkpoint(monkeypatch, tmp_path, single_fidelity_data_file):
    settings = ConfigSettings("GP")
    bounds = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    wrapper = GPModel(
        data_file=str(single_fidelity_data_file),
        bounds=bounds,
        config_settings=settings,
        device="cpu",
    )
    checkpoint = tmp_path / "legacy_gp.pt"
    torch.save(
        {
            "state_dict": {
                "outcome_transform.means": torch.tensor([[0.5]]),
                "outcome_transform.stdvs": torch.tensor([[0.25]]),
            }
        },
        checkpoint,
    )
    captured_state = {}

    def fake_load_state_dict(self, state_dict):
        captured_state.update(state_dict)
        return None

    monkeypatch.setattr(DeepOptSingleTaskGP, "load_state_dict", fake_load_state_dict)

    with pytest.warns(RuntimeWarning, match="legacy GP checkpoint"):
        wrapper.load_model(str(checkpoint))

    assert not any(key.startswith("outcome_transform.") for key in captured_state)
    torch.testing.assert_close(wrapper.full_train_Y_scaled, (wrapper.full_train_Y - 0.5) / 0.25)
    torch.testing.assert_close(wrapper.output_scaler.inverse_transform(wrapper.full_train_Y_scaled), wrapper.full_train_Y)


@pytest.mark.requires_botorch
def test_model_agnostic_loader_rejects_legacy_checkpoint(tmp_path):
    checkpoint = tmp_path / "legacy_gp.pt"
    torch.save({"state_dict": {}}, checkpoint)

    with pytest.raises(ValueError, match="legacy explicit path"):
        load_deepopt_model(str(checkpoint), device="cpu")


def test_fidelity_cost_model_uses_rounded_last_column():
    model = FidelityCostModel(np.array([1.0, 2.5, 4.0], dtype=np.float32))
    X = torch.tensor(
        [
            [0.0, 0.1],
            [0.0, 0.9],
            [0.0, 2.0],
        ],
        dtype=torch.float32,
    )

    cost = model(X)

    torch.testing.assert_close(cost, torch.tensor([[1.0], [2.5], [4.0]]))


@pytest.mark.requires_botorch
def test_get_risk_measure_objective_supported_and_unknown(single_fidelity_data_file):
    settings = ConfigSettings("GP")
    bounds = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    model = GPModel(
        data_file=str(single_fidelity_data_file),
        bounds=bounds,
        config_settings=settings,
        device="cpu",
    )

    assert model.get_risk_measure_objective("unknown") is None
    assert model.get_risk_measure_objective("VaR", alpha=0.5, n_w=4).__class__.__name__ == "VaR"
    assert model.get_risk_measure_objective("CVaR", alpha=0.5, n_w=4).__class__.__name__ == "CVaR"


def test_deepopt_base_model_remains_abstract():
    assert bool(getattr(DeepoptBaseModel, "__abstractmethods__"))
