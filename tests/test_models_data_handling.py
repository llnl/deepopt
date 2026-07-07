import numpy as np
import pytest
import torch

pytest.importorskip("botorch")
pytest.importorskip("gpytorch")
pytest.importorskip("ray")

from deepopt.configuration import ConfigSettings
from deepopt.parallel_acq import (
    ParallelAcqSettings,
    _optimize_acqf_worker,
    parallel_optimize_acqf,
    select_best,
    split_list_by_workers,
    split_tensor_by_workers,
)
from deepopt.models import (
    DEEPOPT_CHECKPOINT_KEY,
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


def test_parallel_single_fidelity_candidate_generation_forwards_optimization_kwargs(
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

    def fake_parallel_optimize_acqf(*args, **kwargs):
        captured.update(kwargs)
        return torch.tensor([[0.1, 0.2]]), torch.tensor(1.0)

    monkeypatch.setattr("deepopt.models.parallel_optimize_acqf", fake_parallel_optimize_acqf)

    wrapper._get_candidates_sf(model=object(), acq_method="EI", q=1, parallel_acq={"enabled": True, "num_workers": 2})

    assert captured["settings"].enabled is True
    assert captured["settings"].num_workers == 2
    assert captured["num_restarts"] == 9
    assert captured["raw_samples"] == 33
    assert captured["options"] == {"batch_limit": 7, "maxiter": 22, "seed": wrapper.random_seed}


def test_parallel_multi_fidelity_candidate_generation_forwards_optimization_kwargs(
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

    def fake_parallel_optimize_acqf_mixed(*args, **kwargs):
        captured.update(kwargs)
        return torch.tensor([[0.1, 0.2, 1.0]]), torch.tensor(1.0)

    monkeypatch.setattr("deepopt.models.qMultiFidelityMaxValueEntropy", fake_mf_mes)
    monkeypatch.setattr("deepopt.models.parallel_optimize_acqf_mixed", fake_parallel_optimize_acqf_mixed)

    wrapper._get_candidates_mf(
        model=object(),
        acq_method="MaxValEntropy",
        q=1,
        fidelity_cost=np.array([1.0, 3.0], dtype=np.float32),
        parallel_acq={"enabled": True, "num_workers": 2},
    )

    assert captured["settings"].enabled is True
    assert captured["n_fantasies"] == 5
    assert captured["fixed_features_list"] == [{2: 0}, {2: 1}]
    assert captured["num_restarts"] == 10
    assert captured["raw_samples"] == 34
    assert captured["options"] == {"batch_limit": 6, "maxiter": 21, "seed": wrapper.random_seed}


def test_parallel_acq_split_and_select_helpers():
    tensor = torch.arange(15).reshape(5, 3)
    chunks = split_tensor_by_workers(tensor, 2)
    assert [chunk.shape[0] for chunk in chunks] == [3, 2]

    list_chunks = split_list_by_workers([{0: i} for i in range(5)], 2)
    assert [len(chunk) for chunk in list_chunks] == [3, 2]

    candidates = torch.tensor([[0.1], [0.9], [0.3]])
    values = torch.tensor([1.0, 5.0, 2.0])
    best_candidate, best_value = select_best(candidates, values)
    torch.testing.assert_close(best_candidate, torch.tensor([0.9]))
    torch.testing.assert_close(best_value, torch.tensor(5.0))


def test_parallel_acq_worker_accepts_legacy_payload(monkeypatch):
    captured = {}

    def fake_optimize_acqf(**kwargs):
        captured.update(kwargs)
        return torch.tensor([[0.4]]), torch.tensor([2.0])

    monkeypatch.setattr("deepopt.parallel_acq.optimize_acqf", fake_optimize_acqf)

    candidates, acq_values, warning_messages = _optimize_acqf_worker(
        {
            "settings": ParallelAcqSettings(worker_torch_num_interop_threads=None),
            "bounds": torch.tensor([[0.0], [1.0]]),
            "q": 1,
            "num_restarts": 1,
            "batch_initial_conditions": torch.tensor([[[0.4]]]),
        }
    )

    assert captured["num_restarts"] == 1
    torch.testing.assert_close(candidates, torch.tensor([[0.4]]))
    torch.testing.assert_close(acq_values, torch.tensor([2.0]))
    assert warning_messages == []


def test_parallel_acq_fork_workers_share_unpicklable_acq_function(monkeypatch):
    settings = ParallelAcqSettings(enabled=True, num_workers=2, start_method="fork", worker_torch_num_interop_threads=None)

    def fake_gen_initial_conditions(**kwargs):
        return torch.tensor([[[0.2]], [[0.8]]])

    def fake_optimize_acqf(**kwargs):
        assert kwargs["acq_function"]() == "local-object"
        return kwargs["batch_initial_conditions"].squeeze(1), torch.arange(kwargs["num_restarts"], dtype=torch.float32)

    monkeypatch.setattr("deepopt.parallel_acq._gen_initial_conditions", fake_gen_initial_conditions)
    monkeypatch.setattr("deepopt.parallel_acq.optimize_acqf", fake_optimize_acqf)

    candidates, acq_values = parallel_optimize_acqf(
        settings=settings,
        acq_function=lambda: "local-object",
        bounds=torch.tensor([[0.0], [1.0]]),
        q=1,
        num_restarts=2,
        raw_samples=2,
    )

    torch.testing.assert_close(candidates, torch.tensor([0.2]))
    torch.testing.assert_close(acq_values, torch.tensor(0.0))


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
