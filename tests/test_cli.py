import numpy as np
from click.testing import CliRunner
import pytest
import torch

pytest.importorskip("botorch")
pytest.importorskip("gpytorch")

from deepopt.deepopt_cli import deepopt_cli, get_deepopt_model
from deepopt.models import (
    DEEPOPT_CHECKPOINT_KEY,
    AcquisitionOptimizationConstraints,
    DelUQModel,
    GPModel,
    NNEnsembleModel,
    RFModel,
    TabPFNModel,
)

pytestmark = pytest.mark.requires_botorch


def test_get_deepopt_model_supported_names():
    assert get_deepopt_model("GP") is GPModel
    assert get_deepopt_model("delUQ") is DelUQModel
    assert get_deepopt_model("nnEnsemble") is NNEnsembleModel
    assert get_deepopt_model("TabPFN") is TabPFNModel
    assert get_deepopt_model("RF") is RFModel


def test_get_deepopt_model_rejects_invalid_name():
    with pytest.raises(ValueError, match="not a valid DeepOpt model"):
        get_deepopt_model("bad")


def test_cli_help_lists_commands_and_core_options():
    runner = CliRunner()

    result = runner.invoke(deepopt_cli, ["--help"])
    assert result.exit_code == 0
    assert "learn" in result.output
    assert "optimize" in result.output

    result = runner.invoke(deepopt_cli, ["learn", "--help"])
    assert result.exit_code == 0
    assert "--infile" in result.output
    assert "--outfile" in result.output
    assert "--bounds" in result.output
    assert "--model-type" in result.output
    assert "TabPFN" in result.output
    assert "RF" in result.output

    result = runner.invoke(deepopt_cli, ["optimize", "--help"])
    assert result.exit_code == 0
    assert "--learner-file" in result.output
    assert "--acq-method" in result.output
    assert "--risk-measure" in result.output
    assert "--n-fantasies" not in result.output
    assert "--acq-optimizer" in result.output
    assert "--batch-mode" in result.output


def test_learn_cli_parses_bounds_and_invokes_model(monkeypatch, single_fidelity_data_file, tmp_path):
    calls = []

    def fake_learn(self, outfile):
        calls.append(self)
        assert outfile == str(tmp_path / "learner.ckpt")

    monkeypatch.setattr(GPModel, "learn", fake_learn)
    runner = CliRunner()

    result = runner.invoke(
        deepopt_cli,
        [
            "learn",
            "-i",
            str(single_fidelity_data_file),
            "-o",
            str(tmp_path / "learner.ckpt"),
            "-b",
            "[[0, 1], [0, 1]]",
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    model = calls[0]
    assert model.config_settings.get_setting("model_type") == "GP"
    np.testing.assert_allclose(model.bounds.cpu().numpy(), np.array([[0, 0], [1, 1]], dtype=np.float32))
    assert model.device == "cpu"


def test_optimize_cli_parses_conditional_multi_fidelity_values(
    monkeypatch, multi_fidelity_data_file, tmp_path
):
    learner_file = tmp_path / "learner.ckpt"
    learner_file.write_text("placeholder")
    calls = []

    def fake_optimize(self, **kwargs):
        calls.append((self, kwargs))

    monkeypatch.setattr(GPModel, "optimize", fake_optimize)
    runner = CliRunner()

    result = runner.invoke(
        deepopt_cli,
        [
            "optimize",
            "-i",
            str(multi_fidelity_data_file),
            "-o",
            str(tmp_path / "suggested.npy"),
            "-l",
            str(learner_file),
            "-b",
            "[[0, 1], [0, 1], [0, 1]]",
            "-a",
            "KG",
            "--device",
            "cpu",
            "--multi-fidelity",
            "--fidelity-cost",
            "[1.0, 3.0]",
            "--integer-fidelities",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    model, kwargs = calls[0]
    assert model.multi_fidelity is True
    np.testing.assert_allclose(kwargs["fidelity_cost"], np.array([1.0, 3.0], dtype=np.float32))
    assert kwargs["integer_fidelities"] is True


def test_optimize_cli_parses_constraints(monkeypatch, single_fidelity_data_file, tmp_path):
    learner_file = tmp_path / "learner.ckpt"
    learner_file.write_text("placeholder")
    constraint_script = tmp_path / "constraints.py"
    constraint_script.write_text(
        "def make_constraints():\n"
        "    def c(X):\n"
        "        return X[..., 0] - 0.2\n"
        "    return [c]\n"
    )
    calls = []

    def fake_optimize(self, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(GPModel, "optimize", fake_optimize)
    runner = CliRunner()

    result = runner.invoke(
        deepopt_cli,
        [
            "optimize",
            "-i",
            str(single_fidelity_data_file),
            "-o",
            str(tmp_path / "suggested.npy"),
            "-l",
            str(learner_file),
            "-b",
            "[[0, 1], [0, 1]]",
            "-a",
            "EI",
            "--device",
            "cpu",
            "--inequality-constraints",
            "[[[0], [1.0], 0.5]]",
            "--equality-constraints",
            "[[[1], [1.0], 0.25]]",
            "--nonlinear-inequality-constraints",
            f"{constraint_script}:make_constraints",
            "--nonlinear-mode",
            "initialization-only",
            "--nonlinear-initial-raw-samples",
            "32",
            "--nonlinear-optimization-retries",
            "3",
        ],
    )

    assert result.exit_code == 0, result.output
    constraints = calls[0]["optimization_constraints"]
    assert isinstance(constraints, AcquisitionOptimizationConstraints)
    assert constraints.inequality_constraints == [([0], [1.0], 0.5)]
    assert constraints.equality_constraints == [([1], [1.0], 0.25)]
    assert len(constraints.nonlinear_inequality_constraints) == 1
    assert constraints.nonlinear_mode == "initialization_only"
    assert constraints.nonlinear_initial_raw_samples == 32
    assert constraints.nonlinear_optimization_retries == 3


def test_optimize_cli_omitted_nonlinear_controls_defer_to_config(single_fidelity_data_file, tmp_path, monkeypatch):
    learner_file = tmp_path / "learner.ckpt"
    learner_file.write_text("placeholder")
    config_file = tmp_path / "optimize.yaml"
    config_file.write_text("optimization:\n  nonlinear_optimization_retries: 4\n")
    constraint_script = tmp_path / "constraints.py"
    constraint_script.write_text("def c(X):\n    return X[..., 0]\n")
    calls = []

    def fake_optimize(self, **kwargs):
        calls.append((self, kwargs))

    monkeypatch.setattr(GPModel, "optimize", fake_optimize)
    runner = CliRunner()

    result = runner.invoke(
        deepopt_cli,
        [
            "optimize",
            "-i",
            str(single_fidelity_data_file),
            "-o",
            str(tmp_path / "suggested.npy"),
            "-l",
            str(learner_file),
            "-b",
            "[[0, 1], [0, 1]]",
            "-a",
            "EI",
            "-c",
            str(config_file),
            "--device",
            "cpu",
            "--nonlinear-inequality-constraints",
            f"{constraint_script}:c",
        ],
    )

    assert result.exit_code == 0, result.output
    model, kwargs = calls[0]
    assert model.config_settings.get_setting("optimization") == {"nonlinear_optimization_retries": 4}
    constraints = kwargs["optimization_constraints"]
    assert constraints.nonlinear_mode is None
    assert constraints.nonlinear_initial_max_tries is None
    assert constraints.nonlinear_optimization_retries is None


def test_optimize_cli_rejects_non_integer_constraint_indices(single_fidelity_data_file, tmp_path):
    learner_file = tmp_path / "learner.ckpt"
    learner_file.write_text("placeholder")
    runner = CliRunner()

    result = runner.invoke(
        deepopt_cli,
        [
            "optimize",
            "-i",
            str(single_fidelity_data_file),
            "-o",
            str(tmp_path / "suggested.npy"),
            "-l",
            str(learner_file),
            "-b",
            "[[0, 1], [0, 1]]",
            "-a",
            "EI",
            "--inequality-constraints",
            "[[[1.9], [1.0], 0.5]]",
        ],
    )

    assert result.exit_code != 0
    assert "indices must be integers" in result.output


def test_optimize_cli_rejects_malformed_constraints(single_fidelity_data_file, tmp_path):
    learner_file = tmp_path / "learner.ckpt"
    learner_file.write_text("placeholder")
    runner = CliRunner()

    result = runner.invoke(
        deepopt_cli,
        [
            "optimize",
            "-i",
            str(single_fidelity_data_file),
            "-o",
            str(tmp_path / "suggested.npy"),
            "-l",
            str(learner_file),
            "-b",
            "[[0, 1], [0, 1]]",
            "-a",
            "EI",
            "--inequality-constraints",
            "[[[0], [1.0, 2.0], 0.5]]",
        ],
    )

    assert result.exit_code != 0
    assert "same length" in result.output


def test_conditional_option_is_ignored_when_dependency_missing(
    monkeypatch, single_fidelity_data_file, tmp_path
):
    learner_file = tmp_path / "learner.ckpt"
    learner_file.write_text("placeholder")
    calls = []

    def fake_optimize(self, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(GPModel, "optimize", fake_optimize)
    runner = CliRunner()

    result = runner.invoke(
        deepopt_cli,
        [
            "optimize",
            "-i",
            str(single_fidelity_data_file),
            "-o",
            str(tmp_path / "suggested.npy"),
            "-l",
            str(learner_file),
            "-b",
            "[[0, 1], [0, 1]]",
            "-a",
            "EI",
            "--device",
            "cpu",
            "--fidelity-cost",
            "[9, 9]",
            "--integer-fidelities",
            "--risk-level",
            "0.5",
            "--risk-n-deltas",
            "4",
            "--X-stddev",
            "[0.1, 0.1]",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Option risk_level will not be used" in result.output
    assert "Option risk_n_deltas will not be used" in result.output
    assert "Option x_stddev will not be used" in result.output
    assert len(calls) == 1
    kwargs = calls[0]
    assert kwargs["fidelity_cost"] is None
    assert kwargs["integer_fidelities"] is False
    assert kwargs["risk_measure"] is None
    assert kwargs["risk_level"] is None
    assert kwargs["risk_n_deltas"] is None
    assert kwargs["x_stddev"] is None


def test_modern_optimize_cli_uses_checkpoint_metadata(monkeypatch, tmp_path):
    checkpoint = tmp_path / "learner.ckpt"
    X = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    y = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
    torch_checkpoint = {
        "state_dict": {},
        DEEPOPT_CHECKPOINT_KEY: {
            "schema_version": 1,
            "model_type": "GP",
            "training_data": {"X": X, "y": y},
            "bounds": torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32),
            "config_settings": {"model_type": "GP"},
            "multi_fidelity": False,
        },
    }
    torch.save(torch_checkpoint, checkpoint)
    calls = []

    def fake_optimize(self, **kwargs):
        calls.append((self, kwargs))

    monkeypatch.setattr(GPModel, "optimize", fake_optimize)
    runner = CliRunner()

    result = runner.invoke(
        deepopt_cli,
        [
            "optimize",
            "-o",
            str(tmp_path / "suggested.npy"),
            "-l",
            str(checkpoint),
            "-a",
            "EI",
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    model, kwargs = calls[0]
    assert model.config_settings.get_setting("model_type") == "GP"
    assert kwargs["learner_file"] == str(checkpoint)


def test_modern_optimize_cli_applies_config_file_optimization_overrides(monkeypatch, tmp_path):
    checkpoint = tmp_path / "learner.ckpt"
    config_file = tmp_path / "opt.yaml"
    config_file.write_text("optimization:\n  profile: fast\n  num_restarts_high: 3\n")
    X = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    y = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
    torch.save(
        {
            "state_dict": {},
            DEEPOPT_CHECKPOINT_KEY: {
                "schema_version": 1,
                "model_type": "GP",
                "training_data": {"X": X, "y": y},
                "bounds": torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32),
                "config_settings": {"model_type": "GP", "optimization": {"profile": "cpu_large"}},
                "multi_fidelity": False,
            },
        },
        checkpoint,
    )
    calls = []

    def fake_optimize(self, **kwargs):
        calls.append(self)

    monkeypatch.setattr(GPModel, "optimize", fake_optimize)
    runner = CliRunner()

    result = runner.invoke(
        deepopt_cli,
        [
            "optimize",
            "-o",
            str(tmp_path / "suggested.npy"),
            "-l",
            str(checkpoint),
            "-a",
            "EI",
            "-c",
            str(config_file),
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert calls[0].config_settings.get_setting("optimization") == {
        "profile": "fast",
        "num_restarts_high": 3,
    }


def test_legacy_optimize_cli_requires_infile_and_bounds(tmp_path):
    learner_file = tmp_path / "legacy.ckpt"
    learner_file.write_text("placeholder")
    runner = CliRunner()

    result = runner.invoke(
        deepopt_cli,
        [
            "optimize",
            "-o",
            str(tmp_path / "suggested.npy"),
            "-l",
            str(learner_file),
            "-a",
            "EI",
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code != 0
    assert "Legacy checkpoints require --infile" in result.output


def test_risk_cli_parses_x_stddev_when_risk_measure_used(
    monkeypatch, single_fidelity_data_file, tmp_path
):
    learner_file = tmp_path / "learner.ckpt"
    learner_file.write_text("placeholder")
    calls = []

    def fake_optimize(self, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(GPModel, "optimize", fake_optimize)
    runner = CliRunner()

    result = runner.invoke(
        deepopt_cli,
        [
            "optimize",
            "-i",
            str(single_fidelity_data_file),
            "-o",
            str(tmp_path / "suggested.npy"),
            "-l",
            str(learner_file),
            "-b",
            "[[0, 1], [0, 1]]",
            "-a",
            "EI",
            "--device",
            "cpu",
            "--risk-measure",
            "VaR",
            "--risk-level",
            "0.5",
            "--risk-n-deltas",
            "4",
            "--X-stddev",
            "[0.1, 0.2]",
        ],
    )

    assert result.exit_code == 0, result.output
    kwargs = calls[0]
    assert kwargs["risk_measure"] == "VaR"
    assert kwargs["risk_level"] == 0.5
    assert kwargs["risk_n_deltas"] == 4
    np.testing.assert_allclose(kwargs["x_stddev"], np.array([0.1, 0.2], dtype=np.float32))


def _run_optimize_cli(monkeypatch, single_fidelity_data_file, tmp_path, extra_args):
    learner_file = tmp_path / "learner.ckpt"
    learner_file.write_text("placeholder")
    calls = []
    monkeypatch.setattr(GPModel, "optimize", lambda self, **kwargs: calls.append(kwargs))
    result = CliRunner().invoke(
        deepopt_cli,
        [
            "optimize",
            "-i", str(single_fidelity_data_file),
            "-o", str(tmp_path / "suggested.npy"),
            "-l", str(learner_file),
            "-b", "[[0, 1], [0, 1]]",
            "-a", "EI",
            "--device", "cpu",
            *extra_args,
        ],
    )
    assert result.exit_code == 0, result.output
    return calls[0]


def test_optimize_cli_forwards_acq_optimizer_and_batch_mode(monkeypatch, single_fidelity_data_file, tmp_path):
    kwargs = _run_optimize_cli(
        monkeypatch, single_fidelity_data_file, tmp_path, ["--acq-optimizer", "boltzmann", "--batch-mode", "sequential"]
    )
    assert kwargs["acq_optimizer"] == "boltzmann"
    assert kwargs["batch_mode"] == "sequential"


def test_optimize_cli_omits_optimizer_overrides_by_default(monkeypatch, single_fidelity_data_file, tmp_path):
    kwargs = _run_optimize_cli(monkeypatch, single_fidelity_data_file, tmp_path, [])
    assert "acq_optimizer" not in kwargs
    assert "batch_mode" not in kwargs
