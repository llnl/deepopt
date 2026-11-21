"""Tests for the gradient-free acquisition optimizers (torch only, no BoTorch required)."""
import pytest
import torch

from deepopt.acq_optimizers import (
    DeferPartition,
    SamplingOptions,
    SearchSpace,
    _PointObjective,
    _scheduled_halfwidth,
    batched_max_search,
    nearest_neighbor_distance,
    optimize_acqf_gradient_free,
    perturb,
)

D = 10
CENTER = torch.full((D,), 0.3)
SMALL = SamplingOptions(
    raw_samples=512,
    n_elites=16,
    samples_per_elite=16,
    n_rounds=6,
    boltzmann_population=512,
    boltzmann_n_rounds=4,
    boltzmann_defer_budget=512,
    eval_batch=256,
)


def bump(X):
    """Quadratic bump: ``b x q x d -> b``, maximized at ``CENTER`` for every batch member."""
    return -((X - CENTER) ** 2).sum(-1).sum(-1)


def unit_bounds(d=D):
    return torch.stack([torch.zeros(d), torch.ones(d)])


class GradSpy:
    """Acquisition wrapper that records whether autograd was enabled during evaluation."""

    def __init__(self, fn):
        self.fn = fn
        self.grad_enabled = []

    def __call__(self, X):
        self.grad_enabled.append(torch.is_grad_enabled())
        return self.fn(X)


class PendingBump:
    """Bump acquisition with ``set_X_pending`` that penalizes points close to pending ones."""

    def __init__(self):
        self.X_pending = None
        self.history = []

    def set_X_pending(self, X_pending=None):
        self.X_pending = X_pending
        self.history.append(None if X_pending is None else X_pending.clone())

    def __call__(self, X):
        value = bump(X)
        if self.X_pending is not None:
            dist = torch.cdist(X.reshape(-1, D), self.X_pending).min(-1).values.reshape(X.shape[:-1]).min(-1).values
            value = value - torch.exp(-((dist / 0.05) ** 2))
        return value


def test_scoped_sampling_beats_raw_sobol_and_runs_without_grad():
    spy = GradSpy(bump)
    candidate, value = optimize_acqf_gradient_free(spy, unit_bounds(), 1, "sampling", SMALL, seed=0)
    _, sobol_value = optimize_acqf_gradient_free(bump, unit_bounds(), 1, "sampling", SamplingOptions(**{**SMALL.__dict__, "n_rounds": 0}), seed=0)
    assert candidate.shape == (1, D)
    assert value > sobol_value
    assert value > -0.05
    assert spy.grad_enabled and not any(spy.grad_enabled)


def test_scheduled_halfwidth_shrinks_monotonically():
    widths = [_scheduled_halfwidth(SMALL, k) for k in range(5)]
    assert all(later < earlier for earlier, later in zip(widths, widths[1:]))


def test_perturb_respects_bounds_and_halfwidth():
    space = SearchSpace(unit_bounds())
    generator = torch.Generator().manual_seed(0)
    parents = torch.full((4, D), 0.98)
    halfwidth = torch.full((4,), 0.1)
    for perturbation in ("subset", "uniform"):
        options = SamplingOptions(perturbation=perturbation)
        children, _ = perturb(parents, torch.zeros(4, dtype=torch.long), halfwidth, 32, space, options, generator)
        assert children.shape == (4, 32, D)
        assert (children >= 0).all() and (children <= 1).all()
        # Reflection at the boundary keeps every coordinate within one half-width of its parent.
        assert ((children - parents.unsqueeze(1)).abs() <= 0.1 + 1e-6).all()
    children, _ = perturb(parents, torch.zeros(4, dtype=torch.long), halfwidth, 32, space, SamplingOptions(subset_prob=0.05), generator)
    moved = (children != parents.unsqueeze(1)).sum(-1)
    assert (moved >= 1).all()
    assert moved.float().mean() < D / 2


def test_nearest_neighbor_distance_ignores_self():
    query = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]])
    archive = [torch.tensor([[[0.0, 0.0], [0.0, 0.5], [1.0, 1.0]]])]
    dist = nearest_neighbor_distance(query, archive)
    assert torch.allclose(dist, torch.tensor([[0.5, (0.5**2 + 1.0) ** 0.5]]))


def test_fixed_features_are_respected():
    candidates, _ = optimize_acqf_gradient_free(
        PendingBump(), unit_bounds(), 2, "sampling", SMALL, sequential=True, seed=1,
        fixed_features_list=[{D - 1: 0.0}, {D - 1: 1.0}],
    )
    assert set(candidates[:, -1].tolist()) <= {0.0, 1.0}
    # Fixed value 0.0 is closer to the bump center (0.3) than 1.0.
    assert candidates[0, -1] == 0.0


def test_linear_and_nonlinear_constraints_are_enforced():
    inequality = [(torch.tensor([0, 1]), torch.tensor([-1.0, -1.0]), -0.4)]  # x0 + x1 <= 0.4
    nonlinear = [lambda X: X[..., 2] - 0.6]  # x2 >= 0.6
    for method in ("sampling", "boltzmann"):
        candidates, _ = optimize_acqf_gradient_free(
            bump, unit_bounds(), 3, method, SMALL, sequential=False, seed=2,
            inequality_constraints=inequality, nonlinear_inequality_constraints=nonlinear,
        )
        assert (candidates[:, 0] + candidates[:, 1] <= 0.4 + 1e-6).all()
        assert (candidates[:, 2] >= 0.6 - 1e-6).all()


def test_linear_equality_constraints_are_projected():
    equality = [(torch.tensor([0, 1]), torch.tensor([1.0, 1.0]), 1.0)]
    candidates, _ = optimize_acqf_gradient_free(bump, unit_bounds(), 1, "sampling", SMALL, seed=3, equality_constraints=equality)
    assert torch.allclose(candidates[:, 0] + candidates[:, 1], torch.tensor([1.0]), atol=1e-4)


def test_sequential_and_joint_shapes_and_pending_cleanup():
    acq = PendingBump()
    candidates, values = optimize_acqf_gradient_free(acq, unit_bounds(), 3, "sampling", SMALL, sequential=True, seed=4)
    assert candidates.shape == (3, D)
    assert values.shape == (3,)
    assert acq.X_pending is None
    assert [h.shape[0] for h in acq.history[:-1]] == [1, 2, 3]
    # Pending penalty spreads sequential candidates out.
    assert torch.pdist(candidates).min() > 0.02
    candidates, value = optimize_acqf_gradient_free(bump, unit_bounds(), 3, "sampling", SMALL, sequential=False, seed=4)
    assert candidates.shape == (3, D)
    assert value.ndim == 0


def test_joint_mode_moves_every_member():
    calls = []

    def recorder(X):
        calls.append(X.clone())
        return bump(X)

    options = SamplingOptions(**{**SMALL.__dict__, "n_rounds": 1, "joint_member_prob": 1.0, "perturbation": "uniform"})
    optimize_acqf_gradient_free(recorder, unit_bounds(), 2, "sampling", options, sequential=False, seed=5)
    batch_calls = [c for c in calls if c.shape[-2] == 2]
    children = batch_calls[-1]
    assert children.ndim == 3
    # Seed batches are evaluated before the single perturbation round; the last call holds the children.
    seeds = torch.cat(batch_calls[:-1]).reshape(-1, D)
    # With joint_member_prob=1 every member is perturbed, so no child member equals any seed point exactly.
    matches = (children.reshape(-1, 1, D) == seeds.unsqueeze(0)).all(-1).any(-1)
    assert not matches.any()


@pytest.mark.parametrize("sampler", ["smc", "defer", "importance"])
def test_boltzmann_samplers_return_feasible_reproducible_draws(sampler):
    options = SamplingOptions(**{**SMALL.__dict__, "boltzmann_sampler": sampler})
    first, values = optimize_acqf_gradient_free(bump, unit_bounds(), 5, "boltzmann", options, seed=6)
    second, _ = optimize_acqf_gradient_free(bump, unit_bounds(), 5, "boltzmann", options, seed=6)
    assert first.shape == (5, D)
    assert values.shape == (5,)
    assert (first >= 0).all() and (first <= 1).all()
    assert torch.equal(first, second)


@pytest.mark.parametrize("sampler", ["smc", "importance"])
def test_boltzmann_temperature_controls_concentration(sampler):
    def mean_distance(lam):
        options = SamplingOptions(**{**SMALL.__dict__, "boltzmann_sampler": sampler, "boltzmann_lambda": lam})
        draws, _ = optimize_acqf_gradient_free(bump, unit_bounds(), 64, "boltzmann", options, seed=7)
        return (draws - CENTER).norm(dim=-1).mean()

    # Near-uniform for tiny lambda; concentrated toward the maximizer for large lambda.
    uniform_mean = mean_distance(1e-4)
    assert abs(uniform_mean - 1.0) < 0.2  # E||U - 0.3|| for U ~ Unif[0,1]^10 is ~1.03
    assert mean_distance(20.0) < uniform_mean - 0.15


def test_boltzmann_draws_are_iid_with_replacement():
    options = SamplingOptions(**{**SMALL.__dict__, "boltzmann_sampler": "importance", "boltzmann_lambda": 50.0, "boltzmann_population": 64})
    draws, _ = optimize_acqf_gradient_free(bump, unit_bounds(), 32, "boltzmann", options, seed=8)
    assert torch.unique(draws, dim=0).shape[0] < 32


def test_defer_partition_masses_and_cells():
    space = SearchSpace(unit_bounds(3))
    generator = torch.Generator().manual_seed(0)
    options = SamplingOptions(boltzmann_defer_budget=200, eval_batch=64)
    partition = DeferPartition(_PointObjective(lambda X: -((X - 0.3) ** 2).sum(-1).sum(-1), space, 64), space, options, generator, seed=0)
    probs = torch.softmax(partition.log_masses(), 0)
    assert torch.isclose(probs.sum(), torch.tensor(1.0))
    assert (partition.lo <= partition.hi).all()
    volumes = (partition.hi - partition.lo).prod(-1)
    assert torch.isclose(volumes.sum(), torch.tensor(1.0), atol=1e-5)
    # Every evaluated point lies inside the cell it is assigned to.
    lo, hi = partition.lo[partition.point_cell], partition.hi[partition.point_cell]
    assert ((partition.u >= lo - 1e-6) & (partition.u <= hi + 1e-6)).all()


def test_batched_max_search_runs_independent_searches():
    centers = torch.tensor([[0.2] * 3, [0.8] * 3])

    def value_fn(points):  # K x n x d -> K x n
        return -((points - centers.unsqueeze(1)) ** 2).sum(-1)

    best = batched_max_search(value_fn, unit_bounds(3), 2, SMALL, seed=0)
    assert best.shape == (2,)
    assert (best > -0.01).all()


def test_infeasible_space_raises():
    impossible = [(torch.tensor([0]), torch.tensor([1.0]), 2.0)]  # x0 >= 2 with x0 in [0, 1]
    with pytest.raises(ValueError, match="feasible"):
        optimize_acqf_gradient_free(bump, unit_bounds(), 1, "sampling", SMALL, seed=0, inequality_constraints=impossible)
