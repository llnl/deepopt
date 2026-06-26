import pytest
import torch

from deepopt.configuration import ConfigSettings
from deepopt.surrogate_utils import MLP, MLPLayer, create_optimizer


def _settings_from_dict(model_type, config):
    settings = ConfigSettings(model_type)
    for key, value in config.items():
        settings.set_setting(key, value)
    return settings


def test_mlp_forward_shape_without_fourier_features(tmp_path, minimal_nn_config):
    settings = _settings_from_dict("delUQ", minimal_nn_config)
    model = MLP(settings, unc_type="ensemble", input_dim=3, output_dim=1, device="cpu")

    X = torch.ones(5, 3)
    y = model(X)

    assert y.shape == torch.Size([5, 1])
    assert model.B is None


def test_mlp_forward_shape_with_fourier_features(tmp_path, minimal_nn_config):
    config = dict(minimal_nn_config)
    config.update({"ff": True, "mapping_size": 3})
    settings = _settings_from_dict("delUQ", config)
    model = MLP(settings, unc_type="ensemble", input_dim=2, output_dim=1, device="cpu")

    X = torch.ones(4, 2)
    mapped = model.input_mapping(X)
    y = model(X)

    assert mapped.shape == torch.Size([4, 6])
    assert y.shape == torch.Size([4, 1])
    assert model.B.shape == torch.Size([3, 2])


def test_deltaenc_mlp_expects_delta_encoded_input_width(tmp_path, minimal_nn_config):
    settings = _settings_from_dict("delUQ", minimal_nn_config)
    model = MLP(settings, unc_type="deltaenc", input_dim=3, output_dim=1, device="cpu")

    y = model(torch.ones(5, 6))

    assert y.shape == torch.Size([5, 1])


@pytest.mark.parametrize("activation", ["relu", "tanh", "identity", "siren"])
def test_mlp_layer_supported_activations(tmp_path, minimal_nn_config, activation):
    config = dict(minimal_nn_config)
    config["activation"] = activation
    settings = _settings_from_dict("delUQ", config)
    layer = MLPLayer(settings, input_dim=3, output_dim=2, is_first=True, is_last=False)

    y = layer(torch.ones(4, 3))

    assert y.shape == torch.Size([4, 2])


def test_mlp_layer_rejects_unknown_activation(tmp_path, minimal_nn_config):
    config = dict(minimal_nn_config)
    config["activation"] = "bad"
    settings = _settings_from_dict("delUQ", config)

    with pytest.raises(NotImplementedError, match="only activations"):
        MLPLayer(settings, input_dim=3, output_dim=2, is_first=True, is_last=False)


def test_create_optimizer_supported_types(tmp_path, minimal_nn_config):
    settings = _settings_from_dict("delUQ", minimal_nn_config)
    model = MLP(settings, unc_type="ensemble", input_dim=2, output_dim=1, device="cpu")

    opt = create_optimizer(model, settings)
    assert opt.__class__.__name__ == "Adam"

    settings.set_setting("opt_type", "SGD")
    opt = create_optimizer(model, settings)
    assert opt.__class__.__name__ == "SGD"


def test_create_optimizer_rejects_unknown_type(tmp_path, minimal_nn_config):
    settings = _settings_from_dict("delUQ", minimal_nn_config)
    settings.set_setting("opt_type", "RMSprop")
    model = MLP(settings, unc_type="ensemble", input_dim=2, output_dim=1, device="cpu")

    with pytest.raises(NotImplementedError, match="Only Adam and SGD"):
        create_optimizer(model, settings)
