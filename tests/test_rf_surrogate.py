"""Tests for the random forest surrogate and its integration with gradient-free optimizers."""
import numpy as np
import pytest
import torch

pytest.importorskip("botorch")
pytest.importorskip("gpytorch")

from deepopt._compat import make_sobol_qmc_normal_sampler
from deepopt.configuration import ConfigSettings
from deepopt.models import RFModel, load_deepopt_wrapper
from deepopt.surrogate_rf import RandomForestSurrogate, tree_predictions

pytestmark = pytest.mark.requires_botorch

D = 4
TINY_OPTIMIZATION = {
    "sampling_raw_samples": 64,
    "sampling_n_elites": 4,
    "sampling_samples_per_elite": 4,
    "sampling_n_rounds": 1,
    "sampling_eval_batch": 64,
    "boltzmann_population": 64,
    "boltzmann_n_rounds": 2,
    "boltzmann_defer_budget": 64,
    "n_fantasies": 2,
    "kg_inner_set_size": 16,
    "kg_raw_samples": 8,
    "kg_n_elites": 2,
    "kg_samples_per_elite": 2,
    "kg_n_rounds": 1,
    "kg_refine_top": 1,
}


def rf_settings(**overrides):
    settings = ConfigSettings("RF")
    settings.set_setting("n_estimators", 5)
    settings.set_setting("optimization", dict(TINY_OPTIMIZATION))
    for key, value in overrides.items():
        settings.set_setting(key, value)
    return settings


@pytest.fixture
def rf_data_file(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.random((12, D)).astype(np.float32)
    y = -((X - 0.3) ** 2).sum(1)
    path = tmp_path / "rf.npz"
    np.savez(path, X=X, y=y)
    return path


@pytest.fixture
def rf_mf_data_file(tmp_path):
    rng = np.random.default_rng(1)
    X = rng.random((12, D)).astype(np.float32)
    X[:, -1] = np.array([0, 1] * 6, dtype=np.float32)
    y = -((X[:, :-1] - 0.3) ** 2).sum(1) + 0.1 * X[:, -1]
    path = tmp_path / "rf_mf.npz"
    np.savez(path, X=X, y=y)
    return path


def make_wrapper(data_file, multi_fidelity=False, **overrides):
    bounds = np.array([[0.0] * D, [1.0] * D], dtype=np.float32)
    return RFModel(
        data_file=str(data_file),
        bounds=bounds,
        config_settings=rf_settings(**overrides),
        multi_fidelity=multi_fidelity,
        device="cpu",
    )


def make_surrogate(variance_mode="trees"):
    torch.manual_seed(0)
    X = torch.rand(10, D)
    y = -((X - 0.3) ** 2).sum(-1, keepdim=True)
    return RandomForestSurrogate(config=rf_settings(variance_mode=variance_mode), X_train=X, y_train=y, seed=0)


def test_rf_config_defaults():
    settings = ConfigSettings("RF")
    assert settings.get_setting("n_estimators") == 200
    assert settings.get_setting("variance_mode") == "trees"
    assert settings.get_setting("fantasy_n_jobs") == 1
    assert settings.get_setting("n_jobs") == 1


def test_posterior_shapes_and_no_grad():
    model = make_surrogate()
    X = torch.rand(7, 3, D, requires_grad=True)
    posterior = model.posterior(X)
    assert posterior.mean.shape == (7, 3, 1)
    assert posterior.variance.shape == (7, 3, 1)
    assert not posterior.mean.requires_grad
    assert model.supports_gradients is False


def test_covariance_matches_tree_empirical_covariance():
    model = make_surrogate()
    X = torch.rand(1, 3, D)
    covariance = model.posterior(X).mvn.covariance_matrix[0]
    preds, _ = tree_predictions(model.forests[0], X[0].numpy().astype(np.float32), with_leaf_variance=False)
    expected = torch.as_tensor(np.cov(preds, rowvar=False), dtype=torch.float)
    assert torch.allclose(covariance - 1e-6 * torch.eye(3), expected, atol=1e-5)
    assert torch.linalg.eigvalsh(covariance).min() > 0
    mean = model.posterior(X).mean[0, :, 0]
    assert torch.allclose(mean, torch.as_tensor(preds.mean(0), dtype=torch.float), atol=1e-6)


def test_leaf_variance_mode_increases_diagonal():
    X = torch.rand(5, 1, D)
    trees = make_surrogate("trees").posterior(X).variance
    leaf = make_surrogate("trees+leaf").posterior(X).variance
    assert (leaf >= trees - 1e-7).all()


def test_invalid_variance_mode_rejected():
    with pytest.raises(ValueError, match="variance_mode"):
        make_surrogate("bad")


def test_fantasize_refits_batched_forests():
    model = make_surrogate()
    X = torch.rand(3, 1, D)
    fantasy = model.fantasize(X, sampler=make_sobol_qmc_normal_sampler(2, seed=0))
    assert fantasy.batch_shape == torch.Size([2, 3])
    assert len(fantasy.forests) == 6
    assert fantasy.X_train.shape == (2, 3, 11, D)
    posterior = fantasy.posterior(torch.rand(4, 2, 3, 1, D))
    assert posterior.mean.shape == (4, 2, 3, 1, 1)


@pytest.mark.slow
def test_fantasize_parallel_matches_serial():
    serial = make_surrogate()
    parallel = RandomForestSurrogate(
        config=rf_settings(fantasy_n_jobs=2), X_train=serial.X_train, y_train=serial.y_train, seed=0
    )
    X = torch.rand(2, 1, D)
    sampler = make_sobol_qmc_normal_sampler(2, seed=0)
    query = torch.rand(3, 2, 2, 1, D)
    a = serial.fantasize(X, sampler=sampler).posterior(query).mean
    b = parallel.fantasize(X, sampler=make_sobol_qmc_normal_sampler(2, seed=0)).posterior(query).mean
    assert torch.allclose(a, b)


def test_checkpoint_round_trip_refits_identically(rf_data_file, tmp_path):
    wrapper = make_wrapper(rf_data_file)
    trained = wrapper.train(str(tmp_path / "rf.ckpt"))
    loaded_wrapper = load_deepopt_wrapper(str(tmp_path / "rf.ckpt"), device="cpu")
    assert isinstance(loaded_wrapper, RFModel)
    loaded = loaded_wrapper.load_model(str(tmp_path / "rf.ckpt"))
    query = torch.rand(6, 1, D)
    assert torch.allclose(trained.posterior(query).mean, loaded.posterior(query).mean)
    mean, variance = loaded.get_prediction_with_uncertainty(torch.rand(3, D))
    assert mean.shape == (3, 1) and variance.shape == (3, 1)


def test_gradient_optimizer_rejected_for_rf(rf_data_file, tmp_path):
    wrapper = make_wrapper(rf_data_file)
    wrapper.train(str(tmp_path / "rf.ckpt"))
    with pytest.raises(ValueError, match="not differentiable"):
        wrapper.optimize(outfile=str(tmp_path / "c.npy"), learner_file=str(tmp_path / "rf.ckpt"), acq_method="EI", num_candidates=1)


@pytest.mark.parametrize(
    "acq_method,q,acq_optimizer,batch_mode",
    [
        ("EI", 2, "sampling", "default"),
        ("NEI", 2, "sampling", "joint"),
        ("EI", 2, "boltzmann", "default"),
        ("MaxValEntropy", 2, "sampling", "default"),
        ("KG", 1, "sampling", "default"),
        ("KG", 2, "sampling", "sequential"),
    ],
)
def test_rf_end_to_end_optimize(rf_data_file, tmp_path, acq_method, q, acq_optimizer, batch_mode):
    wrapper = make_wrapper(rf_data_file)
    wrapper.train(str(tmp_path / "rf.ckpt"))
    outfile = tmp_path / "candidates.npy"
    wrapper.optimize(
        outfile=str(outfile),
        learner_file=str(tmp_path / "rf.ckpt"),
        acq_method=acq_method,
        num_candidates=q,
        acq_optimizer=acq_optimizer,
        batch_mode=batch_mode,
    )
    candidates = np.load(outfile)
    assert candidates.shape == (q, D)
    assert np.all((candidates >= 0) & (candidates <= 1))


def test_rf_risk_and_constraints(rf_data_file, tmp_path):
    wrapper = make_wrapper(rf_data_file)
    wrapper.train(str(tmp_path / "rf.ckpt"))
    outfile = tmp_path / "candidates.npy"
    wrapper.optimize(
        outfile=str(outfile),
        learner_file=str(tmp_path / "rf.ckpt"),
        acq_method="EI",
        num_candidates=2,
        acq_optimizer="sampling",
        risk_measure="VaR",
        risk_level=0.8,
        risk_n_deltas=3,
        x_stddev=np.full(D, 0.01, dtype=np.float32),
        inequality_constraints=[([0, 1], [-1.0, -1.0], -0.6)],
    )
    candidates = np.load(outfile)
    assert candidates.shape == (2, D)
    assert np.all(candidates[:, 0] + candidates[:, 1] <= 0.6 + 1e-5)


@pytest.mark.parametrize("acq_method,q", [("MaxValEntropy", 2), ("KG", 1)])
def test_rf_multi_fidelity(rf_mf_data_file, tmp_path, acq_method, q):
    wrapper = make_wrapper(rf_mf_data_file, multi_fidelity=True)
    wrapper.train(str(tmp_path / "rf_mf.ckpt"))
    outfile = tmp_path / "candidates.npy"
    wrapper.optimize(
        outfile=str(outfile),
        learner_file=str(tmp_path / "rf_mf.ckpt"),
        acq_method=acq_method,
        num_candidates=q,
        fidelity_cost=np.array([1.0, 3.0], dtype=np.float32),
        acq_optimizer="sampling",
    )
    candidates = np.load(outfile)
    assert candidates.shape == (q, D)
    assert set(candidates[:, -1].tolist()) <= {0.0, 1.0}
