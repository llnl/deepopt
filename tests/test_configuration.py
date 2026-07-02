import json

import pytest

from deepopt.configuration import ConfigSettings
from deepopt.defaults import DELUQ_CONFIG, GP_CONFIG, NNENSEMBLE_CONFIG, OPTIMIZATION_PROFILES, Defaults


def test_config_settings_load_defaults_for_supported_models():
    cases = {
        "GP": GP_CONFIG,
        "delUQ": DELUQ_CONFIG,
        "nnEnsemble": NNENSEMBLE_CONFIG,
    }

    for model_type, defaults in cases.items():
        settings = ConfigSettings(model_type)
        assert settings.get_setting("model_type") == model_type
        for key, value in defaults.items():
            assert key in settings
            assert settings.get_setting(key) == value


def test_config_settings_loads_json_overrides(tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"n_epochs": 7, "dropout": False}))

    settings = ConfigSettings("delUQ", config_file=str(config_file))

    assert settings.get_setting("n_epochs") == 7
    assert settings.get_setting("dropout") is False


def test_config_settings_yaml_overrides_defaults(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("n_estimators: 3\nactivation: tanh\n")

    settings = ConfigSettings("nnEnsemble", config_file=str(config_file))

    assert settings.get_setting("n_estimators") == 3
    assert settings.get_setting("activation") == "tanh"
    assert settings.get_setting("hidden_dim") == NNENSEMBLE_CONFIG["hidden_dim"]


def test_config_settings_rejects_invalid_model_type():
    with pytest.raises(ValueError, match="not yet been implemented"):
        ConfigSettings("invalid")


def test_config_settings_rejects_unsupported_file_extension(tmp_path):
    config_file = tmp_path / "config.toml"
    config_file.write_text("n_epochs = 1\n")

    with pytest.raises(ValueError, match="must be either a yaml file or a json file"):
        ConfigSettings("GP", config_file=str(config_file))


def test_get_and_set_setting_and_contains():
    settings = ConfigSettings("GP")

    assert "new_key" not in settings
    settings.set_setting("new_key", 123)

    assert "new_key" in settings
    assert settings.get_setting("new_key") == 123


def test_get_setting_rejects_unknown_key():
    settings = ConfigSettings("GP")

    with pytest.raises(KeyError, match="not a valid setting"):
        settings.get_setting("missing")


def test_defaults_public_values_regression():
    assert Defaults.random_seed == 4321
    assert Defaults.k_folds == 5
    assert Defaults.model_type == "GP"
    assert Defaults.multi_fidelity is False
    assert Defaults.num_candidates == 2
    assert Defaults.fidelity_cost == "[1,10]"
    assert Defaults.optimization_profile == "cpu_large"
    assert Defaults.num_restarts_low == 8
    assert Defaults.num_restarts_high == 32
    assert Defaults.raw_samples_low == 1024
    assert Defaults.raw_samples_high == 8192
    assert Defaults.batch_limit_low == 8
    assert Defaults.batch_limit_high == 32
    assert Defaults.maxiter == 200
    assert Defaults.n_fantasies == 64
    assert Defaults.torch_num_threads == "auto"
    assert Defaults.torch_num_threads_fraction == 0.8
    assert Defaults.torch_num_interop_threads == 1


def test_optimization_profiles_include_legacy_and_large_cpu_defaults():
    assert set(OPTIMIZATION_PROFILES) == {"balanced", "cpu_large", "fast"}
    assert OPTIMIZATION_PROFILES["balanced"]["num_restarts_high"] == 15
    assert OPTIMIZATION_PROFILES["balanced"]["raw_samples_high"] == 5000
    assert OPTIMIZATION_PROFILES["balanced"]["n_fantasies"] == 128
    assert OPTIMIZATION_PROFILES["cpu_large"]["num_restarts_high"] == Defaults.num_restarts_high
    assert OPTIMIZATION_PROFILES["cpu_large"]["batch_limit_high"] == Defaults.batch_limit_high
    assert OPTIMIZATION_PROFILES["fast"]["maxiter"] == 100


def test_config_settings_loads_nested_optimization_section(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("optimization:\n  profile: fast\n  batch_limit_high: 6\n")

    settings = ConfigSettings("GP", config_file=str(config_file))

    assert settings.get_setting("optimization") == {"profile": "fast", "batch_limit_high": 6}


def test_nnensemble_default_keys_regression():
    # Preserve the current public configuration contract, including key spelling.
    assert NNENSEMBLE_CONFIG["n_estimators"] == 100
    assert NNENSEMBLE_CONFIG["ff"] is True
    assert NNENSEMBLE_CONFIG["droupout_prob"] == 0.2
    assert "dropout_prob" not in NNENSEMBLE_CONFIG
