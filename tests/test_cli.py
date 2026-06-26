import numpy as np
from click.testing import CliRunner
import pytest

pytest.importorskip("botorch")
pytest.importorskip("gpytorch")
pytest.importorskip("ray")

from deepopt.deepopt_cli import deepopt_cli, get_deepopt_model
from deepopt.models import DelUQModel, GPModel, NNEnsembleModel

pytestmark = pytest.mark.requires_botorch


def test_get_deepopt_model_supported_names():
    assert get_deepopt_model("GP") is GPModel
    assert get_deepopt_model("delUQ") is DelUQModel
    assert get_deepopt_model("nnEnsemble") is NNEnsembleModel


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

    result = runner.invoke(deepopt_cli, ["optimize", "--help"])
    assert result.exit_code == 0
    assert "--learner-file" in result.output
    assert "--acq-method" in result.output
    assert "--risk-measure" in result.output


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
    assert "Option fidelity_cost will not be used" in result.output
    assert "Option integer_fidelities will not be used" in result.output
    assert "Option risk_level will not be used" in result.output
    assert "Option risk_n_deltas will not be used" in result.output
    assert "Option x_stddev will not be used" in result.output
    assert len(calls) == 1
    kwargs = calls[0]
    assert kwargs["fidelity_cost"] is None
    assert kwargs["integer_fidelities"] is None
    assert kwargs["risk_measure"] is None
    assert kwargs["risk_level"] is None
    assert kwargs["risk_n_deltas"] is None
    assert kwargs["x_stddev"] is None


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
