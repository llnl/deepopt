import warnings

import numpy as np
import pytest
import torch

pytest.importorskip("botorch")
pytest.importorskip("gpytorch")
pytest.importorskip("ray")

from deepopt.configuration import ConfigSettings
from deepopt.models import (
    DEEPOPT_CHECKPOINT_KEY,
    AcquisitionOptimizationConstraints,
    AcquisitionOptimizationSettings,
    DeepoptBaseModel,
    DeepOptSingleTaskGP,
    FidelityCostModel,
    GPModel,
    load_deepopt_model,
    load_deepopt_wrapper,
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
    assert hasattr(model, "input_scaler")
    torch.testing.assert_close(model.full_train_X, model.input_scaler.transform(model.X_orig))
    torch.testing.assert_close(model.full_train_X, model.X_orig)
    torch.testing.assert_close(model.bounds, torch.tensor(bounds, dtype=torch.float32))


@pytest.mark.requires_botorch
def test_base_model_single_fidelity_normalizes_non_unit_bounds(single_fidelity_data_file):
    settings = ConfigSettings("GP")
    bounds = np.array([[0.0, 0.0], [2.0, 4.0]], dtype=np.float32)

    model = GPModel(
        data_file=str(single_fidelity_data_file),
        bounds=bounds,
        config_settings=settings,
        device="cpu",
    )

    torch.testing.assert_close(model.full_train_X, model.X_orig / torch.tensor([2.0, 4.0]))
    torch.testing.assert_close(model.full_train_X, model.input_scaler.transform(model.X_orig))


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
    assert "input_scaler" in checkpoint
    assert hasattr(model, "input_scaler")
    torch.testing.assert_close(model.train_targets.unsqueeze(-1), wrapper.full_train_Y_scaled)
    mean_scaled, var_scaled = model.get_prediction_with_uncertainty(
        wrapper.full_train_X[:1],
        original_scale_x=False,
        original_scale_y=False,
    )
    mean_original, var_original = model.get_prediction_with_uncertainty(
        wrapper.full_train_X[:1],
        original_scale_x=False,
        original_scale_y=True,
    )
    mean_original_from_raw, var_original_from_raw = model.get_prediction_with_uncertainty(wrapper.X_orig[:1])
    torch.testing.assert_close(mean_original, mean_original_from_raw)
    torch.testing.assert_close(var_original, var_original_from_raw)
    torch.testing.assert_close(mean_original, wrapper.output_scaler.inverse_transform(mean_scaled, wrapper.full_train_X[:1]))
    torch.testing.assert_close(var_original, wrapper.output_scaler.inverse_variance(var_scaled, wrapper.full_train_X[:1]))
    with pytest.raises(TypeError, match="original_scale was renamed"):
        model.get_prediction_with_uncertainty(wrapper.full_train_X[:1], original_scale=False)
    with pytest.raises(TypeError, match="original_scale was renamed"):
        model.get_prediction_with_uncertainty(wrapper.full_train_X[:1], False, False)


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
    assert hasattr(model, "input_scaler")


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


def test_optimization_settings_resolve_profile_with_overrides(single_fidelity_data_file):
    settings = ConfigSettings("GP")
    settings.set_setting("optimization", {"profile": "fast", "num_restarts_high": 11, "torch_num_threads": 3})
    bounds = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    model = GPModel(
        data_file=str(single_fidelity_data_file),
        bounds=bounds,
        config_settings=settings,
        device="cpu",
    )

    opt_settings = model._resolve_optimization_settings()

    assert opt_settings.num_restarts_high == 11
    assert opt_settings.raw_samples_high == 2048
    assert opt_settings.batch_limit_high == 8
    assert opt_settings.n_fantasies == 32
    assert opt_settings.torch_num_threads == 3


def test_optimization_settings_resolve_nonlinear_controls(single_fidelity_data_file):
    settings = ConfigSettings("GP")
    settings.set_setting(
        "optimization",
        {
            "profile": "fast",
            "nonlinear_mode": "initialization-only",
            "nonlinear_initial_raw_samples": 64,
            "nonlinear_initial_max_tries": 2,
            "nonlinear_optimization_retries": 0,
        },
    )
    bounds = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    model = GPModel(
        data_file=str(single_fidelity_data_file),
        bounds=bounds,
        config_settings=settings,
        device="cpu",
    )

    opt_settings = model._resolve_optimization_settings()

    assert opt_settings.nonlinear_mode == "initialization_only"
    assert opt_settings.nonlinear_initial_raw_samples == 64
    assert opt_settings.nonlinear_initial_max_tries == 2
    assert opt_settings.nonlinear_optimization_retries == 0


def test_auto_torch_threads_use_all_small_allocations_and_fraction_large(monkeypatch):
    monkeypatch.setattr(DeepoptBaseModel, "_available_cpu_count", staticmethod(lambda: 8))
    assert DeepoptBaseModel._resolve_auto_torch_num_threads(0.8) == 8
    monkeypatch.setattr(DeepoptBaseModel, "_available_cpu_count", staticmethod(lambda: 200))
    assert DeepoptBaseModel._resolve_auto_torch_num_threads(0.8) == 160


def test_configure_torch_threads_respects_auto_and_explicit(monkeypatch, single_fidelity_data_file):
    settings = ConfigSettings("GP")
    bounds = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    model = GPModel(
        data_file=str(single_fidelity_data_file),
        bounds=bounds,
        config_settings=settings,
        device="cpu",
    )
    calls = []
    # "auto" defers to explicit thread environment variables, which test runners commonly set.
    for env_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TORCH_NUM_THREADS"):
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setattr(DeepoptBaseModel, "_available_cpu_count", staticmethod(lambda: 100))
    monkeypatch.setattr(torch, "set_num_threads", lambda value: calls.append(("threads", value)))
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda value: calls.append(("interop", value)))

    model._configure_torch_threads(
        AcquisitionOptimizationSettings(
            num_restarts_high=1,
            num_restarts_low=1,
            raw_samples_high=1,
            raw_samples_low=1,
            batch_limit_high=1,
            batch_limit_low=1,
            maxiter=1,
            n_fantasies=1,
            torch_num_threads="auto",
            torch_num_threads_fraction=0.7,
            torch_num_interop_threads=2,
            nonlinear_mode="enforce",
            nonlinear_initial_raw_samples=None,
            nonlinear_initial_max_tries=5,
            nonlinear_optimization_retries=1,
        )
    )

    assert calls == [("threads", 70), ("interop", 2)]


def test_single_fidelity_candidate_generation_uses_resolved_optimization_settings(
    monkeypatch, single_fidelity_data_file
):
    settings = ConfigSettings("GP")
    settings.set_setting(
        "optimization",
        {
            "profile": "fast",
            "num_restarts_high": 9,
            "raw_samples_high": 33,
            "batch_limit_high": 7,
            "maxiter": 22,
        },
    )
    bounds = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    wrapper = GPModel(
        data_file=str(single_fidelity_data_file),
        bounds=bounds,
        config_settings=settings,
        device="cpu",
    )
    captured = {}

    monkeypatch.setattr("deepopt.models.qExpectedImprovement", lambda *args, **kwargs: object())

    def fake_optimize_acqf(*args, **kwargs):
        captured.update(kwargs)
        return torch.tensor([[0.1, 0.2]]), torch.tensor(1.0)

    monkeypatch.setattr("deepopt.models.optimize_acqf", fake_optimize_acqf)

    wrapper._get_candidates_sf(model=object(), acq_method="EI", q=1)

    assert captured["num_restarts"] == 9
    assert captured["raw_samples"] == 33
    assert captured["options"] == {"batch_limit": 7, "maxiter": 22, "seed": wrapper.random_seed}


def test_single_fidelity_expensive_acquisitions_use_low_restart_settings(
    monkeypatch, single_fidelity_data_file
):
    settings = ConfigSettings("GP")
    settings.set_setting(
        "optimization",
        {
            "profile": "fast",
            "num_restarts_high": 9,
            "num_restarts_low": 4,
            "raw_samples_low": 30,
            "batch_limit_low": 3,
            "maxiter": 22,
            "n_fantasies": 5,
        },
    )
    bounds = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    wrapper = GPModel(
        data_file=str(single_fidelity_data_file),
        bounds=bounds,
        config_settings=settings,
        device="cpu",
    )
    captured = {}

    def fake_mes(*args, **kwargs):
        return object()

    def fake_optimize_acqf(*args, **kwargs):
        captured.update(kwargs)
        return torch.tensor([[0.1, 0.2]]), torch.tensor(1.0)

    monkeypatch.setattr("deepopt.models.qMaxValueEntropy", fake_mes)
    monkeypatch.setattr("deepopt.models.optimize_acqf", fake_optimize_acqf)

    wrapper._get_candidates_sf(model=object(), acq_method="MaxValEntropy", q=1)

    assert captured["num_restarts"] == 4
    assert captured["raw_samples"] == 30
    assert captured["options"] == {"batch_limit": 3, "maxiter": 22, "seed": wrapper.random_seed}


def test_single_fidelity_linear_constraints_convert_and_forward(monkeypatch, single_fidelity_data_file):
    settings = ConfigSettings("GP")
    bounds = np.array([[10.0, 0.0], [20.0, 2.0]], dtype=np.float32)
    wrapper = GPModel(
        data_file=str(single_fidelity_data_file),
        bounds=bounds,
        config_settings=settings,
        device="cpu",
    )
    captured = {}

    monkeypatch.setattr("deepopt.models.qExpectedImprovement", lambda *args, **kwargs: object())

    def fake_optimize_acqf(*args, **kwargs):
        captured.update(kwargs)
        return torch.tensor([[0.1, 0.2]]), torch.tensor(1.0)

    monkeypatch.setattr("deepopt.models.optimize_acqf", fake_optimize_acqf)

    wrapper._get_candidates_sf(
        model=object(),
        acq_method="EI",
        q=1,
        optimization_constraints=wrapper._normalize_optimization_constraints(
            inequality_constraints=[([0, 1], [1.0, 2.0], 14.0)],
            equality_constraints=[([1], [1.0], 1.0)],
        ),
    )

    ineq_indices, ineq_coefficients, ineq_rhs = captured["inequality_constraints"][0]
    torch.testing.assert_close(ineq_indices.cpu(), torch.tensor([0, 1]))
    torch.testing.assert_close(ineq_coefficients.cpu(), torch.tensor([10.0, 4.0]))
    assert ineq_rhs == pytest.approx(4.0)
    eq_indices, eq_coefficients, eq_rhs = captured["equality_constraints"][0]
    torch.testing.assert_close(eq_indices.cpu(), torch.tensor([1]))
    torch.testing.assert_close(eq_coefficients.cpu(), torch.tensor([2.0]))
    assert eq_rhs == pytest.approx(1.0)


def test_multi_fidelity_linear_constraints_keep_fidelity_unscaled(monkeypatch, multi_fidelity_data_file):
    settings = ConfigSettings("GP")
    bounds = np.array([[10.0, 0.0, 0.0], [20.0, 2.0, 2.0]], dtype=np.float32)
    wrapper = GPModel(
        data_file=str(multi_fidelity_data_file),
        bounds=bounds,
        config_settings=settings,
        multi_fidelity=True,
        device="cpu",
    )
    captured = {}

    def fake_mf_mes(*args, **kwargs):
        return object()

    def fake_optimize_acqf_mixed(*args, **kwargs):
        captured.update(kwargs)
        return torch.tensor([[0.1, 0.2, 1.0]]), torch.tensor(1.0)

    monkeypatch.setattr("deepopt.models.qMultiFidelityMaxValueEntropy", fake_mf_mes)
    monkeypatch.setattr("deepopt.models.optimize_acqf_mixed", fake_optimize_acqf_mixed)

    wrapper._get_candidates_mf(
        model=object(),
        acq_method="MaxValEntropy",
        q=1,
        fidelity_cost=np.array([1.0, 3.0, 5.0], dtype=np.float32),
        optimization_constraints=wrapper._normalize_optimization_constraints(
            inequality_constraints=[([0, 2], [1.0, 1.0], 12.0)],
        ),
    )

    indices, coefficients, rhs = captured["inequality_constraints"][0]
    torch.testing.assert_close(indices.cpu(), torch.tensor([0, 2]))
    torch.testing.assert_close(coefficients.cpu(), torch.tensor([10.0, 1.0]))
    assert rhs == pytest.approx(2.0)


def test_nonlinear_constraints_force_batch_limit_and_initial_conditions(monkeypatch, single_fidelity_data_file):
    settings = ConfigSettings("GP")
    settings.set_setting("optimization", {"profile": "fast", "batch_limit_high": 4})
    bounds = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    wrapper = GPModel(
        data_file=str(single_fidelity_data_file),
        bounds=bounds,
        config_settings=settings,
        device="cpu",
    )
    captured = {}

    class FakeAcq:
        X_pending = None

        def __call__(self, X):
            return X.sum(dim=(-1, -2))

        def set_X_pending(self, X_pending):
            self.X_pending = X_pending

    initial_conditions = torch.tensor([[[0.8, 0.2]]], dtype=torch.float32)

    def fake_optimize_acqf(*args, **kwargs):
        captured.update(kwargs)
        return torch.tensor([[0.8, 0.2]]), torch.tensor(1.0)

    monkeypatch.setattr("deepopt.models.qExpectedImprovement", lambda *args, **kwargs: FakeAcq())
    monkeypatch.setattr("deepopt.models.optimize_acqf", fake_optimize_acqf)

    with pytest.warns(RuntimeWarning, match="batch_limit=1"):
        wrapper._get_candidates_sf(
            model=object(),
            acq_method="EI",
            q=1,
            optimization_constraints=wrapper._normalize_optimization_constraints(
                nonlinear_inequality_constraints=[lambda X: X[..., 0] - 0.5],
                batch_initial_conditions=initial_conditions,
            ),
        )

    assert captured["options"]["batch_limit"] == 1
    assert "nonlinear_inequality_constraints" in captured
    torch.testing.assert_close(captured["batch_initial_conditions"], initial_conditions)


def test_entropy_candidate_sets_reject_equality_constraints(single_fidelity_data_file):
    settings = ConfigSettings("GP")
    bounds = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    wrapper = GPModel(
        data_file=str(single_fidelity_data_file),
        bounds=bounds,
        config_settings=settings,
        device="cpu",
    )

    with pytest.raises(NotImplementedError, match="Equality constraints"):
        wrapper._filter_candidate_set_for_constraints(
            torch.rand(10, 2),
            wrapper._normalize_optimization_constraints(equality_constraints=[([0], [1.0], 0.5)]),
        )


def test_nonlinear_initialization_only_omits_botorch_constraints(monkeypatch, single_fidelity_data_file):
    settings = ConfigSettings("GP")
    bounds = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    wrapper = GPModel(
        data_file=str(single_fidelity_data_file),
        bounds=bounds,
        config_settings=settings,
        device="cpu",
    )
    captured = {}

    class FakeAcq:
        def __call__(self, X):
            return X.sum(dim=(-1, -2))

    def fake_optimize_acqf(*args, **kwargs):
        captured.update(kwargs)
        return torch.tensor([[0.8, 0.2]]), torch.tensor(1.0)

    monkeypatch.setattr("deepopt.models.qExpectedImprovement", lambda *args, **kwargs: FakeAcq())
    monkeypatch.setattr("deepopt.models.optimize_acqf", fake_optimize_acqf)

    wrapper._get_candidates_sf(
        model=object(),
        acq_method="EI",
        q=1,
        optimization_constraints=wrapper._normalize_optimization_constraints(
            nonlinear_inequality_constraints=[lambda X: X[..., 0] - 0.5],
            nonlinear_mode="initialization_only",
            nonlinear_initial_raw_samples=16,
        ),
    )

    assert "nonlinear_inequality_constraints" not in captured
    assert captured["options"]["batch_limit"] != 1
    assert torch.all(captured["batch_initial_conditions"][..., 0] >= 0.5)


def test_nonlinear_optimization_retries_with_regenerated_initial_conditions(monkeypatch, single_fidelity_data_file):
    settings = ConfigSettings("GP")
    settings.set_setting("optimization", {"profile": "fast", "num_restarts_high": 1})
    bounds = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    wrapper = GPModel(
        data_file=str(single_fidelity_data_file),
        bounds=bounds,
        config_settings=settings,
        device="cpu",
        random_seed=123,
    )
    calls = []

    class FakeAcq:
        def __call__(self, X):
            return X[..., 0].reshape(-1)

    def fake_draw_sobol_samples(bounds, n, q, seed):
        assert q == 1
        return torch.tensor([[[0.6 + 0.001 * seed, 0.1]]], dtype=torch.float32)

    def fake_optimize_acqf(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            warnings.warn(
                "Optimization failed within `scipy.optimize.minimize` with status 1. "
                "Because you specified `batch_initial_conditions`, optimization will not be retried.",
                RuntimeWarning,
            )
            return torch.tensor([[0.6, 0.1]]), torch.tensor(0.1)
        return torch.tensor([[0.9, 0.1]]), torch.tensor(0.9)

    monkeypatch.setattr("deepopt.models.qExpectedImprovement", lambda *args, **kwargs: FakeAcq())
    monkeypatch.setattr("deepopt.models.draw_sobol_samples", fake_draw_sobol_samples)
    monkeypatch.setattr("deepopt.models.optimize_acqf", fake_optimize_acqf)

    with pytest.warns(RuntimeWarning, match="Retrying nonlinear acquisition optimization"):
        candidates, _ = wrapper._get_candidates_sf(
            model=object(),
            acq_method="EI",
            q=1,
            optimization_constraints=wrapper._normalize_optimization_constraints(
                nonlinear_inequality_constraints=[lambda X: X[..., 0] - 0.5],
                nonlinear_mode="initialization_only",
                nonlinear_initial_raw_samples=1,
                nonlinear_initial_max_tries=1,
            ),
        )

    assert len(calls) == 2
    assert not torch.equal(calls[0]["batch_initial_conditions"], calls[1]["batch_initial_conditions"])
    torch.testing.assert_close(candidates, torch.tensor([[0.9, 0.1]]))


def test_nonlinear_optimization_retries_can_be_disabled(monkeypatch, single_fidelity_data_file):
    settings = ConfigSettings("GP")
    bounds = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    wrapper = GPModel(
        data_file=str(single_fidelity_data_file),
        bounds=bounds,
        config_settings=settings,
        device="cpu",
    )
    calls = []

    class FakeAcq:
        def __call__(self, X):
            return X[..., 0].reshape(-1)

    def fake_optimize_acqf(*args, **kwargs):
        calls.append(kwargs)
        warnings.warn(
            "Optimization failed within `scipy.optimize.minimize` with status 1. "
            "Because you specified `batch_initial_conditions`, optimization will not be retried.",
            RuntimeWarning,
        )
        return torch.tensor([[0.6, 0.1]]), torch.tensor(0.1)

    monkeypatch.setattr("deepopt.models.qExpectedImprovement", lambda *args, **kwargs: FakeAcq())
    monkeypatch.setattr("deepopt.models.optimize_acqf", fake_optimize_acqf)

    with pytest.warns(RuntimeWarning, match="will not be retried"):
        wrapper._get_candidates_sf(
            model=object(),
            acq_method="EI",
            q=1,
            optimization_constraints=wrapper._normalize_optimization_constraints(
                nonlinear_inequality_constraints=[lambda X: X[..., 0] - 0.5],
                nonlinear_mode="initialization_only",
                nonlinear_initial_raw_samples=8,
                nonlinear_optimization_retries=0,
            ),
        )

    assert len(calls) == 1


def test_nonlinear_optimization_does_not_retry_user_initial_conditions(monkeypatch, single_fidelity_data_file):
    settings = ConfigSettings("GP")
    bounds = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    wrapper = GPModel(
        data_file=str(single_fidelity_data_file),
        bounds=bounds,
        config_settings=settings,
        device="cpu",
    )
    calls = []
    initial_conditions = torch.tensor([[[0.8, 0.2]]], dtype=torch.float32)

    class FakeAcq:
        def __call__(self, X):
            return X[..., 0].reshape(-1)

    def fake_optimize_acqf(*args, **kwargs):
        calls.append(kwargs)
        warnings.warn(
            "Optimization failed within `scipy.optimize.minimize` with status 1. "
            "Because you specified `batch_initial_conditions`, optimization will not be retried.",
            RuntimeWarning,
        )
        return torch.tensor([[0.8, 0.2]]), torch.tensor(0.1)

    monkeypatch.setattr("deepopt.models.qExpectedImprovement", lambda *args, **kwargs: FakeAcq())
    monkeypatch.setattr("deepopt.models.optimize_acqf", fake_optimize_acqf)

    with pytest.warns(RuntimeWarning, match="will not be retried"):
        wrapper._get_candidates_sf(
            model=object(),
            acq_method="EI",
            q=1,
            optimization_constraints=wrapper._normalize_optimization_constraints(
                nonlinear_inequality_constraints=[lambda X: X[..., 0] - 0.5],
                batch_initial_conditions=initial_conditions,
                nonlinear_optimization_retries=1,
            ),
        )

    assert len(calls) == 1
    torch.testing.assert_close(calls[0]["batch_initial_conditions"], initial_conditions)


def test_nonlinear_optimization_retries_must_be_non_negative(single_fidelity_data_file):
    settings = ConfigSettings("GP")
    bounds = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    wrapper = GPModel(
        data_file=str(single_fidelity_data_file),
        bounds=bounds,
        config_settings=settings,
        device="cpu",
    )

    with pytest.raises(ValueError, match="nonlinear_optimization_retries"):
        wrapper._normalize_optimization_constraints(
            nonlinear_inequality_constraints=[lambda X: X[..., 0]],
            nonlinear_optimization_retries=-1,
        )


def test_optimize_uses_configured_nonlinear_controls(monkeypatch, single_fidelity_data_file, tmp_path):
    settings = ConfigSettings("GP")
    settings.set_setting(
        "optimization",
        {
            "nonlinear_mode": "initialization-only",
            "nonlinear_initial_raw_samples": 16,
            "nonlinear_initial_max_tries": 2,
            "nonlinear_optimization_retries": 0,
        },
    )
    bounds = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    wrapper = GPModel(
        data_file=str(single_fidelity_data_file),
        bounds=bounds,
        config_settings=settings,
        device="cpu",
    )
    captured = {}

    class FakeModel:
        def eval(self):
            return None

    def fake_get_candidates(**kwargs):
        captured.update(kwargs)
        return torch.tensor([[0.2, 0.3]]), torch.tensor(1.0)

    monkeypatch.setattr(wrapper, "load_model", lambda learner_file: FakeModel())
    monkeypatch.setattr(wrapper, "_configure_torch_threads", lambda settings: None)
    monkeypatch.setattr(wrapper, "get_candidates", fake_get_candidates)

    wrapper.optimize(
        outfile=str(tmp_path / "candidates.npy"),
        learner_file="learner.ckpt",
        acq_method="EI",
        nonlinear_inequality_constraints=[lambda X: X[..., 0]],
    )

    constraints = captured["optimization_constraints"]
    assert constraints.nonlinear_mode == "initialization_only"
    assert constraints.nonlinear_initial_raw_samples == 16
    assert constraints.nonlinear_initial_max_tries == 2
    assert constraints.nonlinear_optimization_retries == 0


def test_optimize_direct_nonlinear_controls_override_config(monkeypatch, single_fidelity_data_file, tmp_path):
    settings = ConfigSettings("GP")
    settings.set_setting("optimization", {"nonlinear_optimization_retries": 3})
    bounds = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    wrapper = GPModel(
        data_file=str(single_fidelity_data_file),
        bounds=bounds,
        config_settings=settings,
        device="cpu",
    )
    captured = {}

    class FakeModel:
        def eval(self):
            return None

    def fake_get_candidates(**kwargs):
        captured.update(kwargs)
        return torch.tensor([[0.2, 0.3]]), torch.tensor(1.0)

    monkeypatch.setattr(wrapper, "load_model", lambda learner_file: FakeModel())
    monkeypatch.setattr(wrapper, "_configure_torch_threads", lambda settings: None)
    monkeypatch.setattr(wrapper, "get_candidates", fake_get_candidates)

    wrapper.optimize(
        outfile=str(tmp_path / "candidates.npy"),
        learner_file="learner.ckpt",
        acq_method="EI",
        nonlinear_inequality_constraints=[lambda X: X[..., 0]],
        nonlinear_optimization_retries=0,
    )

    assert captured["optimization_constraints"].nonlinear_optimization_retries == 0


def test_nonlinear_q_batch_initial_conditions_are_assembled_from_feasible_points(
    monkeypatch, single_fidelity_data_file
):
    settings = ConfigSettings("GP")
    settings.set_setting("optimization", {"profile": "fast", "num_restarts_high": 2})
    bounds = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    wrapper = GPModel(
        data_file=str(single_fidelity_data_file),
        bounds=bounds,
        config_settings=settings,
        device="cpu",
        random_seed=123,
    )
    captured = {}

    class FakeAcq:
        def __call__(self, X):
            return X.sum(dim=(-1, -2))

    def fake_draw_sobol_samples(bounds, n, q, seed):
        assert q == 1
        return torch.tensor(
            [
                [[0.9, 0.1]],
                [[0.8, 0.1]],
                [[0.7, 0.1]],
                [[0.6, 0.1]],
                [[0.5, 0.1]],
                [[0.4, 0.1]],
            ],
            dtype=torch.float32,
        )

    def fake_optimize_acqf(*args, **kwargs):
        captured.update(kwargs)
        return torch.tensor([[0.9, 0.1], [0.8, 0.1], [0.7, 0.1]]), torch.tensor(1.0)

    monkeypatch.setattr("deepopt.models.qExpectedImprovement", lambda *args, **kwargs: FakeAcq())
    monkeypatch.setattr("deepopt.models.draw_sobol_samples", fake_draw_sobol_samples)
    monkeypatch.setattr("deepopt.models.optimize_acqf", fake_optimize_acqf)

    wrapper._get_candidates_sf(
        model=object(),
        acq_method="EI",
        q=3,
        optimization_constraints=wrapper._normalize_optimization_constraints(
            nonlinear_inequality_constraints=[lambda X: X[..., 0] - 0.35],
            nonlinear_mode="initialization_only",
            nonlinear_initial_raw_samples=6,
            nonlinear_initial_max_tries=1,
        ),
    )

    initial_conditions = captured["batch_initial_conditions"]
    assert initial_conditions.shape == torch.Size([2, 3, 2])
    assert torch.all(initial_conditions[..., 0] >= 0.35)
    assert "nonlinear_inequality_constraints" not in captured


def test_multi_fidelity_rejects_nonlinear_constraints(multi_fidelity_data_file):
    settings = ConfigSettings("GP")
    bounds = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    wrapper = GPModel(
        data_file=str(multi_fidelity_data_file),
        bounds=bounds,
        config_settings=settings,
        multi_fidelity=True,
        device="cpu",
    )

    with pytest.raises(NotImplementedError, match="single-fidelity"):
        wrapper._normalize_optimization_constraints(nonlinear_inequality_constraints=[lambda X: X[..., 0]])


def test_model_rejects_non_integer_constraint_indices(single_fidelity_data_file):
    settings = ConfigSettings("GP")
    bounds = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    wrapper = GPModel(
        data_file=str(single_fidelity_data_file),
        bounds=bounds,
        config_settings=settings,
        device="cpu",
    )

    with pytest.raises(ValueError, match="indices must be integers"):
        wrapper._normalize_optimization_constraints(inequality_constraints=[([1.9], [1.0], 0.0)])


def test_multi_fidelity_candidate_generation_uses_resolved_optimization_settings(
    monkeypatch, multi_fidelity_data_file
):
    settings = ConfigSettings("GP")
    settings.set_setting(
        "optimization",
        {
            "profile": "fast",
            "num_restarts_high": 10,
            "raw_samples_high": 34,
            "batch_limit_high": 6,
            "maxiter": 21,
            "n_fantasies": 5,
        },
    )
    bounds = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    wrapper = GPModel(
        data_file=str(multi_fidelity_data_file),
        bounds=bounds,
        config_settings=settings,
        multi_fidelity=True,
        device="cpu",
    )
    captured = {}

    def fake_mf_mes(*args, **kwargs):
        captured["n_fantasies"] = kwargs["num_fantasies"]
        return object()

    def fake_optimize_acqf_mixed(*args, **kwargs):
        captured.update(kwargs)
        return torch.tensor([[0.1, 0.2, 1.0]]), torch.tensor(1.0)

    monkeypatch.setattr("deepopt.models.qMultiFidelityMaxValueEntropy", fake_mf_mes)
    monkeypatch.setattr("deepopt.models.optimize_acqf_mixed", fake_optimize_acqf_mixed)

    wrapper._get_candidates_mf(
        model=object(),
        acq_method="MaxValEntropy",
        q=1,
        fidelity_cost=np.array([1.0, 3.0], dtype=np.float32),
    )

    assert captured["n_fantasies"] == 5
    assert captured["num_restarts"] == 10
    assert captured["raw_samples"] == 34
    assert captured["options"] == {"batch_limit": 6, "maxiter": 21, "seed": wrapper.random_seed}


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


@pytest.mark.requires_botorch
def test_loaded_wrapper_get_var_uses_checkpoint_path(monkeypatch, tmp_path, single_fidelity_data_file):
    settings = ConfigSettings("GP")
    bounds = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    wrapper = GPModel(
        data_file=str(single_fidelity_data_file),
        bounds=bounds,
        config_settings=settings,
        device="cpu",
    )

    monkeypatch.setattr("deepopt.models.fit_gpytorch_model", lambda _mll: None)
    checkpoint = tmp_path / "gp.pt"
    wrapper.train(str(checkpoint))

    loaded_wrapper = load_deepopt_wrapper(str(checkpoint), device="cpu")
    values = loaded_wrapper.get_var(risk_level=0.5, x_stddev=torch.tensor([0.0, 0.0]), risk_n_deltas=4)

    assert values.shape == torch.Size([len(loaded_wrapper.full_train_X)])


@pytest.mark.requires_botorch
def test_get_cvar_accepts_explicit_learner_file(monkeypatch, tmp_path, single_fidelity_data_file):
    settings = ConfigSettings("GP")
    bounds = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    wrapper = GPModel(
        data_file=str(single_fidelity_data_file),
        bounds=bounds,
        config_settings=settings,
        device="cpu",
    )

    monkeypatch.setattr("deepopt.models.fit_gpytorch_model", lambda _mll: None)
    checkpoint = tmp_path / "gp.pt"
    wrapper.train(str(checkpoint))

    values = wrapper.get_cvar(
        wrapper.full_train_X[:2],
        risk_level=0.5,
        x_stddev=torch.tensor([0.0, 0.0]),
        risk_n_deltas=4,
        learner_file=str(checkpoint),
    )
    single_value = wrapper.get_cvar(
        wrapper.full_train_X[:1],
        risk_level=0.5,
        x_stddev=torch.tensor([0.0, 0.0]),
        risk_n_deltas=4,
        learner_file=str(checkpoint),
    )
    one_dimensional_query_value = wrapper.get_cvar(
        wrapper.full_train_X[0],
        risk_level=0.5,
        x_stddev=torch.tensor([0.0, 0.0]),
        risk_n_deltas=4,
        learner_file=str(checkpoint),
    )

    assert values.shape == torch.Size([2])
    assert single_value.shape == torch.Size([1])
    assert one_dimensional_query_value.shape == torch.Size([1])


@pytest.mark.requires_botorch
def test_get_var_requires_checkpoint_path(single_fidelity_data_file):
    settings = ConfigSettings("GP")
    bounds = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    wrapper = GPModel(
        data_file=str(single_fidelity_data_file),
        bounds=bounds,
        config_settings=settings,
        device="cpu",
    )

    with pytest.raises(ValueError, match="No learner_file was provided"):
        wrapper.get_var(risk_level=0.5, x_stddev=torch.tensor([0.0, 0.0]), risk_n_deltas=4)


@pytest.mark.requires_botorch
def test_risk_accessor_zeroes_fidelity_stddev(monkeypatch, tmp_path, multi_fidelity_data_file):
    settings = ConfigSettings("GP")
    bounds = np.array([[0.0, 0.0, 0.0], [2.0, 4.0, 2.0]], dtype=np.float32)
    wrapper = GPModel(
        data_file=str(multi_fidelity_data_file),
        bounds=bounds,
        config_settings=settings,
        multi_fidelity=True,
        device="cpu",
    )
    captured = {}

    monkeypatch.setattr("deepopt.models.fit_gpytorch_model", lambda _mll: None)
    checkpoint = tmp_path / "gp.pt"
    wrapper.train(str(checkpoint))

    original_get_input_perturbation = wrapper.get_input_perturbation

    def fake_get_input_perturbation(risk_n_deltas, bounds, X_stddev):
        captured["bounds"] = bounds.detach().clone()
        captured["X_stddev"] = X_stddev.detach().clone()
        return original_get_input_perturbation(risk_n_deltas, bounds, X_stddev)

    monkeypatch.setattr(wrapper, "get_input_perturbation", fake_get_input_perturbation)
    monkeypatch.setattr(wrapper, "_evaluate_risk_measure", lambda model, risk_objective, X_query: torch.zeros(X_query.shape[-2]))

    wrapper.get_var(risk_level=0.5, x_stddev=torch.tensor([0.2, 0.4, 3.0]), risk_n_deltas=4, learner_file=str(checkpoint))

    torch.testing.assert_close(captured["X_stddev"], torch.tensor([0.1, 0.1, 0.0]))
    torch.testing.assert_close(captured["bounds"][:, -1], torch.tensor([0.0, 2.0]))


@pytest.mark.requires_botorch
def test_risk_accessors_select_botorch_objectives(monkeypatch, single_fidelity_data_file):
    settings = ConfigSettings("GP")
    bounds = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    wrapper = GPModel(
        data_file=str(single_fidelity_data_file),
        bounds=bounds,
        config_settings=settings,
        device="cpu",
        learner_file="checkpoint.pt",
    )
    calls = []

    def fake_load_model(learner_file):
        return type("FakeModel", (), {"input_transform": None, "eval": lambda self: None})()

    def fake_get_risk_measure_objective(risk_measure, **kwargs):
        calls.append((risk_measure, kwargs))
        return object()

    monkeypatch.setattr(wrapper, "load_model", fake_load_model)
    monkeypatch.setattr(wrapper, "get_input_perturbation", lambda risk_n_deltas, bounds, X_stddev: object())
    monkeypatch.setattr(wrapper, "get_risk_measure_objective", fake_get_risk_measure_objective)
    monkeypatch.setattr(wrapper, "_evaluate_risk_measure", lambda model, risk_objective, X_query: torch.zeros(X_query.shape[-2]))

    wrapper.get_var(risk_level=0.25, x_stddev=torch.tensor([0.0, 0.0]), risk_n_deltas=8)
    wrapper.get_cvar(risk_level=0.75, x_stddev=torch.tensor([0.0, 0.0]), risk_n_deltas=16)

    assert calls == [
        ("VaR", {"alpha": 0.25, "n_w": 8}),
        ("CVaR", {"alpha": 0.75, "n_w": 16}),
    ]


def test_deepopt_base_model_remains_abstract():
    assert bool(getattr(DeepoptBaseModel, "__abstractmethods__"))


def _gp_wrapper(data_file, optimization=None):
    settings = ConfigSettings("GP")
    if optimization is not None:
        settings.set_setting("optimization", optimization)
    bounds = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    return GPModel(data_file=str(data_file), bounds=bounds, config_settings=settings, device="cpu")


def test_default_settings_keep_historical_gradient_optimize_acqf_call(monkeypatch, single_fidelity_data_file):
    wrapper = _gp_wrapper(single_fidelity_data_file)
    captured = {}
    monkeypatch.setattr("deepopt.models.qExpectedImprovement", lambda *args, **kwargs: object())

    def fake_optimize_acqf(*args, **kwargs):
        captured.update(kwargs)
        return torch.tensor([[0.1, 0.2], [0.3, 0.4]]), torch.tensor(1.0)

    monkeypatch.setattr("deepopt.models.optimize_acqf", fake_optimize_acqf)
    monkeypatch.setattr(
        "deepopt.models.optimize_acqf_gradient_free",
        lambda *args, **kwargs: pytest.fail("default settings must not use the gradient-free optimizer"),
    )
    wrapper._get_candidates_sf(model=object(), acq_method="EI", q=2)
    assert set(captured) == {"acq_function", "bounds", "q", "num_restarts", "raw_samples", "sequential", "options"}
    assert captured["sequential"] is False
    assert captured["num_restarts"] == 32
    assert captured["raw_samples"] == 8192
    assert captured["options"] == {"batch_limit": 32, "maxiter": 200, "seed": wrapper.random_seed}


@pytest.mark.parametrize(
    "acq_method,batch_mode,expected",
    [
        ("EI", "default", False),
        ("MaxValEntropy", "default", True),
        ("EI", "sequential", True),
        ("NEI", "joint", False),
        ("MaxValEntropy", "sequential", True),
    ],
)
def test_batch_mode_controls_gradient_sequential_flag(monkeypatch, single_fidelity_data_file, acq_method, batch_mode, expected):
    wrapper = _gp_wrapper(single_fidelity_data_file, {"batch_mode": batch_mode})
    captured = {}
    for name in ("qExpectedImprovement", "qNoisyExpectedImprovement", "qMaxValueEntropy"):
        monkeypatch.setattr(f"deepopt.models.{name}", lambda *args, **kwargs: object())

    def fake_optimize_acqf(*args, **kwargs):
        captured.update(kwargs)
        return torch.tensor([[0.1, 0.2], [0.3, 0.4]]), torch.tensor(1.0)

    monkeypatch.setattr("deepopt.models.optimize_acqf", fake_optimize_acqf)
    wrapper._get_candidates_sf(model=object(), acq_method=acq_method, q=2)
    assert captured["sequential"] is expected


def test_joint_batch_mode_rejected_for_max_value_entropy(monkeypatch, single_fidelity_data_file):
    wrapper = _gp_wrapper(single_fidelity_data_file, {"batch_mode": "joint"})
    monkeypatch.setattr("deepopt.models.qMaxValueEntropy", lambda *args, **kwargs: object())
    with pytest.raises(ValueError, match="batch_mode='joint'"):
        wrapper._get_candidates_sf(model=object(), acq_method="MaxValEntropy", q=2)


def test_sampling_optimizer_routes_to_gradient_free(monkeypatch, single_fidelity_data_file):
    wrapper = _gp_wrapper(single_fidelity_data_file, {"acq_optimizer": "sampling", "sampling_n_rounds": 3})
    captured = {}
    monkeypatch.setattr("deepopt.models.qExpectedImprovement", lambda *args, **kwargs: object())
    monkeypatch.setattr("deepopt.models.optimize_acqf", lambda *a, **k: pytest.fail("gradient optimizer must not run"))

    def fake_gradient_free(**kwargs):
        captured.update(kwargs)
        return torch.tensor([[0.1, 0.2]]), torch.tensor(1.0)

    monkeypatch.setattr("deepopt.models.optimize_acqf_gradient_free", fake_gradient_free)
    wrapper._get_candidates_sf(model=object(), acq_method="EI", q=1)
    assert captured["method"] == "sampling"
    assert captured["options"].n_rounds == 3
    assert captured["seed"] == wrapper.random_seed


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"acq_optimizer": "lbfgs"}, "acq_optimizer"),
        ({"batch_mode": "parallel"}, "batch_mode"),
        ({"boltzmann_sampler": "mcmc"}, "boltzmann_sampler"),
        ({"sampling_shrink": 1.5}, "sampling_shrink"),
        ({"sampling_n_elites": 0}, "sampling_n_elites"),
        ({"boltzmann_lambda": 0}, "boltzmann_lambda"),
    ],
)
def test_invalid_sampling_settings_rejected(single_fidelity_data_file, overrides, match):
    wrapper = _gp_wrapper(single_fidelity_data_file, overrides)
    with pytest.raises(ValueError, match=match):
        wrapper._resolve_optimization_settings()


def test_optimize_overrides_take_precedence_over_config(monkeypatch, single_fidelity_data_file, tmp_path):
    wrapper = _gp_wrapper(single_fidelity_data_file, {"acq_optimizer": "boltzmann"})
    captured = {}
    monkeypatch.setattr(wrapper, "load_model", lambda learner_file: torch.nn.Module())
    monkeypatch.setattr(wrapper, "_configure_torch_threads", lambda settings: None)

    def fake_get_candidates(**kwargs):
        captured.update(kwargs)
        return torch.tensor([[0.5, 0.5]]), torch.tensor(1.0)

    monkeypatch.setattr(wrapper, "get_candidates", fake_get_candidates)
    wrapper.optimize(
        outfile=str(tmp_path / "c.npy"), learner_file="unused", acq_method="EI", num_candidates=1,
        acq_optimizer="sampling", batch_mode="sequential",
    )
    assert captured["optimization_settings"].acq_optimizer == "sampling"
    assert captured["optimization_settings"].batch_mode == "sequential"
