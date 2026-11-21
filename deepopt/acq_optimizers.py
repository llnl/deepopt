"""
Gradient-free acquisition optimizers and samplers.

This module provides alternatives to BoTorch's gradient-based ``optimize_acqf``
that only require *evaluations* of an acquisition function. They make it
possible to use non-differentiable surrogates (e.g. random forests) and to
draw batches by sampling from the acquisition function.

Two families are implemented:

``sampling`` (scoped random search)
    A large scrambled-Sobol draw is down-selected to elite points. New points
    are drawn in shrinking boxes around the elites and the best points are
    re-selected. Each box's half-width is the smaller of the Euclidean distance
    from the elite to its nearest already-sampled neighbor and a geometric
    trust-region-style schedule. Following TuRBO, the default perturbation
    resamples only a random subset of coordinates, which keeps moves local in
    10-30 dimensions.

``boltzmann`` (sampling from ``p(x) ∝ exp(λ·standardize(α(x)))``)
    Batch members are drawn i.i.d. from the Boltzmann distribution of the
    acquisition function, as proposed in B3O. Three samplers are available:
    annealed sequential Monte Carlo (``smc``, the default), a recursive
    partitioning sampler in the spirit of DEFER (``defer``), and
    self-normalized importance resampling of a Sobol population
    (``importance``).

Credits
-------
* BoTorch (Balandat et al., "BoTorch: A Framework for Efficient Monte-Carlo
  Bayesian Optimization", NeurIPS 2020; MIT license) provides the acquisition,
  posterior, sampler, transform, and fantasy-model machinery everything here
  builds on. The Boltzmann weighting ``exp(eta * standardize(acq))`` follows
  BoTorch's ``botorch.generation.sampling.BoltzmannSampling``.
* B3O: M. Bloor, L. Xu, H. Stojic, V. Picheny, "B3O: Scalable Boltzmann Batch
  Bayesian Optimization", arXiv:2606.30228, 2026. The reference
  implementation was built on trieste (Secondmind Labs, Apache-2.0).
* DEFER: E. Bodin, Z. Dai, N. Campbell, C. H. Ek, "Black-box density function
  estimation using recursive partitioning", ICML 2021.
* TuRBO: D. Eriksson, M. Pearce, J. Gardner, R. Turner, M. Poloczek,
  "Scalable Global Optimization via Local Bayesian Optimization",
  NeurIPS 2019.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor
from torch.quasirandom import SobolEngine

ACQ_OPTIMIZERS = ("gradient", "sampling", "boltzmann")
BATCH_MODES = ("default", "sequential", "joint")
SAMPLING_PERTURBATIONS = ("subset", "uniform", "gaussian")
BOLTZMANN_SAMPLERS = ("smc", "defer", "importance")



@dataclass(frozen=True)
class SamplingOptions:
    """
    Resolved options for the gradient-free optimizers.

    Built from ``AcquisitionOptimizationSettings`` with :meth:`from_settings`.
    """

    raw_samples: int = 16384
    n_elites: int = 128
    samples_per_elite: int = 128
    n_rounds: int = 10
    init_halfwidth: float = 0.2
    shrink: float = 0.7
    min_halfwidth: float = 1e-3
    perturbation: str = "subset"
    subset_prob: Optional[float] = None
    eval_batch: int = 2048
    joint_member_prob: float = 1.0
    boltzmann_sampler: str = "smc"
    boltzmann_lambda: float = 1.0
    boltzmann_population: int = 16384
    boltzmann_n_rounds: int = 10
    boltzmann_mh_steps: int = 1
    boltzmann_defer_budget: int = 16384
    kg_inner_set_size: int = 1024
    kg_refine_top: int = 8
    kg_raw_samples: int = 512
    kg_n_elites: int = 16
    kg_samples_per_elite: int = 16
    kg_n_rounds: int = 3

    @classmethod
    def from_settings(cls, settings: Any) -> "SamplingOptions":
        """
        Build options from resolved DeepOpt acquisition optimization settings.

        :param settings: An ``AcquisitionOptimizationSettings`` instance.
        :returns: The resolved sampling options.
        """
        return cls(
            raw_samples=settings.sampling_raw_samples,
            n_elites=settings.sampling_n_elites,
            samples_per_elite=settings.sampling_samples_per_elite,
            n_rounds=settings.sampling_n_rounds,
            init_halfwidth=settings.sampling_init_halfwidth,
            shrink=settings.sampling_shrink,
            min_halfwidth=settings.sampling_min_halfwidth,
            perturbation=settings.sampling_perturbation,
            subset_prob=settings.sampling_subset_prob,
            eval_batch=settings.sampling_eval_batch,
            joint_member_prob=settings.sampling_joint_member_prob,
            boltzmann_sampler=settings.boltzmann_sampler,
            boltzmann_lambda=settings.boltzmann_lambda,
            boltzmann_population=settings.boltzmann_population,
            boltzmann_n_rounds=settings.boltzmann_n_rounds,
            boltzmann_mh_steps=settings.boltzmann_mh_steps,
            boltzmann_defer_budget=settings.boltzmann_defer_budget,
            kg_inner_set_size=settings.kg_inner_set_size,
            kg_refine_top=settings.kg_refine_top,
            kg_raw_samples=settings.kg_raw_samples,
            kg_n_elites=settings.kg_n_elites,
            kg_samples_per_elite=settings.kg_samples_per_elite,
            kg_n_rounds=settings.kg_n_rounds,
        )

    def for_kg_outer(self) -> "SamplingOptions":
        """
        Options with the smaller outer budget used for fantasy-based KG.

        Every KG evaluation builds a fantasy model, so the outer scoped search and
        the Boltzmann population/budget are capped by the ``kg_*`` settings.
        """
        return replace(
            self,
            raw_samples=self.kg_raw_samples,
            n_elites=self.kg_n_elites,
            samples_per_elite=self.kg_samples_per_elite,
            n_rounds=self.kg_n_rounds,
            boltzmann_population=min(self.boltzmann_population, self.kg_raw_samples),
            boltzmann_defer_budget=min(self.boltzmann_defer_budget, self.kg_raw_samples),
            boltzmann_n_rounds=max(1, min(self.boltzmann_n_rounds, self.kg_n_rounds)),
        )


class SearchSpace:
    """
    Box-bounded search space with optional fixed features and constraints.

    Points are handled in unit-cube coordinates ``u`` and mapped to model
    coordinates with :meth:`to_x`. Each point carries an option index selecting
    one entry of ``fixed_features_list`` (used for discrete fidelity choices).

    Linear constraints use the BoTorch convention ``(indices, coefficients, rhs)``:
    inequalities mean ``sum(c_i x_i) >= rhs`` and equalities mean
    ``sum(c_i x_i) == rhs``. Nonlinear inequality callables receive ``n x 1 x d``
    model-coordinate tensors, and values ``>= 0`` count as feasible.
    """

    def __init__(
        self,
        bounds: Tensor,
        fixed_features_list: Optional[Sequence[Dict[int, float]]] = None,
        inequality_constraints: Optional[Sequence[Tuple[Tensor, Tensor, float]]] = None,
        equality_constraints: Optional[Sequence[Tuple[Tensor, Tensor, float]]] = None,
        nonlinear_inequality_constraints: Optional[Sequence[Callable[[Tensor], Tensor]]] = None,
        equality_tol: float = 1e-5,
    ):
        self.bounds = bounds
        self.device = bounds.device
        self.dtype = bounds.dtype
        self.lb = bounds[0]
        self.ub = bounds[1]
        self.d = bounds.shape[-1]
        self.span = self.ub - self.lb
        self._safe_span = torch.where(self.span > 0, self.span, torch.ones_like(self.span))
        options = list(fixed_features_list) if fixed_features_list else [{}]
        keys = set(options[0])
        if any(set(option) != keys for option in options):
            raise NotImplementedError("Gradient-free optimizers require every fixed_features entry to fix the same columns.")
        if any(index < 0 or index >= self.d for index in keys):
            raise ValueError(f"Fixed feature indices must be in [0, {self.d}).")
        self.fixed_dims = sorted(keys)
        self.free_dims = [index for index in range(self.d) if index not in keys]
        if not self.free_dims:
            raise ValueError("Gradient-free optimizers need at least one free (non-fixed) input dimension.")
        self.free_index = torch.tensor(self.free_dims, dtype=torch.long, device=self.device)
        self.fixed_index = torch.tensor(self.fixed_dims, dtype=torch.long, device=self.device)
        self.n_options = len(options)
        self.option_values = torch.tensor(
            [[float(option[index]) for index in self.fixed_dims] for option in options],
            dtype=self.dtype,
            device=self.device,
        ).reshape(self.n_options, len(self.fixed_dims))
        self.A_ineq, self.b_ineq = self._dense_constraints(inequality_constraints)
        self.A_eq, self.b_eq = self._dense_constraints(equality_constraints)
        self.nonlinear_constraints = list(nonlinear_inequality_constraints or [])
        self.equality_tol = equality_tol

    @property
    def df(self) -> int:
        """Number of free dimensions."""
        return len(self.free_dims)

    def _dense_constraints(self, constraints) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        if not constraints:
            return None, None
        A = torch.zeros(len(constraints), self.d, dtype=self.dtype, device=self.device)
        b = torch.zeros(len(constraints), dtype=self.dtype, device=self.device)
        for row, (indices, coefficients, rhs) in enumerate(constraints):
            indices = torch.as_tensor(indices, dtype=torch.long, device=self.device).reshape(-1)
            coefficients = torch.as_tensor(coefficients, dtype=self.dtype, device=self.device).reshape(-1)
            A[row].index_add_(0, indices, coefficients)
            b[row] = float(rhs)
        return A, b

    def to_x(self, u: Tensor, opt: Optional[Tensor] = None) -> Tensor:
        """
        Map unit-cube points to model coordinates and pin fixed features.

        :param u: ``... x d`` tensor of unit-cube points.
        :param opt: Optional ``...`` long tensor of fixed-feature option indices.
        :returns: ``... x d`` model-coordinate points.
        """
        x = self.lb + u * self.span
        if self.fixed_dims:
            if opt is None:
                opt = torch.zeros(u.shape[:-1], dtype=torch.long, device=u.device)
            x = x.clone()
            x[..., self.fixed_index] = self.option_values[opt]
        return x

    def to_u(self, x: Tensor) -> Tensor:
        """Map model-coordinate points to unit-cube coordinates."""
        return ((x - self.lb) / self._safe_span).clamp(0.0, 1.0)

    def option_of(self, x: Tensor) -> Tensor:
        """Return the fixed-feature option index nearest to each point of ``x``."""
        if not self.fixed_dims or self.n_options == 1:
            return torch.zeros(x.shape[:-1], dtype=torch.long, device=x.device)
        fixed = x[..., self.fixed_index]
        dist = (fixed.unsqueeze(-2) - self.option_values).abs().sum(-1)
        return dist.argmin(-1)

    def project_equality(self, x: Tensor, n_iter: int = 20) -> Tensor:
        """
        Project points onto the linear equality constraints.

        Free coordinates are moved by alternating an orthogonal projection onto
        the affine set with clipping to the bounds. Fixed coordinates never change.
        """
        if self.A_eq is None:
            return x
        A_free = self.A_eq[:, self.free_index]
        gram_pinv = torch.linalg.pinv(A_free @ A_free.T)
        lb_free, ub_free = self.lb[self.free_index], self.ub[self.free_index]
        x = x.clone()
        for _ in range(n_iter):
            residual = x @ self.A_eq.T - self.b_eq
            correction = (residual @ gram_pinv.T) @ A_free
            free = x[..., self.free_index] - correction
            x[..., self.free_index] = torch.maximum(torch.minimum(free, ub_free), lb_free)
        return x

    def feasible(self, x: Tensor) -> Tensor:
        """
        Return a boolean mask of points satisfying all constraints.

        :param x: ``... x d`` model-coordinate points.
        :returns: Boolean tensor with shape ``...``.
        """
        mask = torch.ones(x.shape[:-1], dtype=torch.bool, device=x.device)
        if self.A_ineq is not None:
            mask &= (x @ self.A_ineq.T >= self.b_ineq - 1e-7).all(-1)
        if self.A_eq is not None:
            mask &= ((x @ self.A_eq.T - self.b_eq).abs() <= self.equality_tol).all(-1)
        if self.nonlinear_constraints:
            flat = x.reshape(-1, 1, self.d)
            nonlinear_mask = torch.ones(flat.shape[0], dtype=torch.bool, device=x.device)
            for constraint in self.nonlinear_constraints:
                values = constraint(flat) >= 0
                while values.ndim > 1:
                    values = values.all(dim=-1)
                nonlinear_mask &= values.to(x.device)
            mask &= nonlinear_mask.reshape(x.shape[:-1])
        return mask

    @property
    def has_constraints(self) -> bool:
        """Whether any linear or nonlinear constraints are present."""
        return self.A_ineq is not None or self.A_eq is not None or bool(self.nonlinear_constraints)

    def draw(self, n: int, generator: torch.Generator, seed: int, max_tries: int = 20) -> Tuple[Tensor, Tensor]:
        """
        Draw ``n`` feasible points with scrambled Sobol sequences and rejection.

        :param n: Number of points requested.
        :param generator: CPU generator for option indices.
        :param seed: Base Sobol seed.
        :param max_tries: Maximum oversampled Sobol draws.
        :returns: ``(u, opt)`` with shapes ``n' x d`` and ``n'``, where ``n' <= n``.
        :raises ValueError: If no feasible point is found.
        """
        kept_u, kept_opt, total = [], [], 0
        draw_size = n
        for attempt in range(max_tries):
            engine = SobolEngine(self.d, scramble=True, seed=int(seed) + attempt)
            u = engine.draw(draw_size).to(device=self.device, dtype=self.dtype)
            opt = torch.randint(self.n_options, (draw_size,), generator=generator).to(self.device)
            u, mask = self.repair(u, opt)
            if mask.any():
                kept_u.append(u[mask])
                kept_opt.append(opt[mask])
                total += int(mask.sum())
            if total >= n:
                break
            acceptance = max(total / ((attempt + 1) * draw_size), 1e-3)
            draw_size = int(min(max(n, (n - total) / acceptance * 1.2), 64 * n))
        if total == 0:
            raise ValueError("Could not find any feasible points for gradient-free acquisition optimization; relax the constraints.")
        u, opt = torch.cat(kept_u)[:n], torch.cat(kept_opt)[:n]
        if u.shape[0] < n:
            warnings.warn(
                f"Only found {u.shape[0]} of {n} requested feasible points for gradient-free acquisition optimization.",
                RuntimeWarning,
            )
        return u, opt

    def repair(self, u: Tensor, opt: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Apply equality projection (if any) and return ``(u, feasible_mask)``.
        """
        if self.A_eq is not None:
            x = self.project_equality(self.to_x(u, opt))
            u = self.to_u(x)
        else:
            x = self.to_x(u, opt)
        return u, self.feasible(x)


def _reflect_unit(u: Tensor) -> Tensor:
    """Reflect values into ``[0, 1]`` (keeps symmetric proposals symmetric)."""
    u = torch.remainder(u, 2.0)
    return torch.where(u > 1.0, 2.0 - u, u)


def _rand(shape, generator: torch.Generator, like: Tensor) -> Tensor:
    return torch.rand(shape, generator=generator).to(device=like.device, dtype=like.dtype)


def _randn(shape, generator: torch.Generator, like: Tensor) -> Tensor:
    return torch.randn(shape, generator=generator).to(device=like.device, dtype=like.dtype)


def perturb(
    u: Tensor,
    opt: Tensor,
    halfwidth: Tensor,
    n_children: int,
    space: SearchSpace,
    options: SamplingOptions,
    generator: torch.Generator,
) -> Tuple[Tensor, Tensor]:
    """
    Draw ``n_children`` local perturbations of each parent point.

    :param u: ``... x d`` parent points in unit coordinates.
    :param opt: ``...`` parent fixed-feature option indices.
    :param halfwidth: ``...`` per-parent box half-widths in unit coordinates.
    :param n_children: Children per parent.
    :param space: Search space.
    :param options: Sampling options selecting the perturbation type.
    :param generator: CPU random generator.
    :returns: ``(u_children, opt_children)`` with shapes ``... x n_children x d`` and ``... x n_children``.
    """
    batch_shape = u.shape[:-1]
    df = space.df
    parents = u.unsqueeze(-2).expand(*batch_shape, n_children, space.d)
    free = parents[..., space.free_index]
    h = halfwidth.reshape(*batch_shape, 1, 1)
    shape = (*batch_shape, n_children, df)
    if options.perturbation == "gaussian":
        delta = _randn(shape, generator, u) * h
    else:
        delta = (2.0 * _rand(shape, generator, u) - 1.0) * h
    if options.perturbation == "subset":
        prob = options.subset_prob if options.subset_prob is not None else min(1.0, 20.0 / df)
        mask = _rand(shape, generator, u) < prob
        forced = torch.randint(df, (*batch_shape, n_children, 1), generator=generator).to(u.device)
        mask.scatter_(-1, forced, True)
        delta = delta * mask
    children = parents.clone()
    children[..., space.free_index] = _reflect_unit(free + delta)
    children_opt = opt.unsqueeze(-1).expand(*batch_shape, n_children).clone()
    if space.n_options > 1:
        # Symmetric discrete move: occasionally resample the fixed-feature option.
        switch = _rand((*batch_shape, n_children), generator, u) < 1.0 / (df + 1)
        new_opt = torch.randint(space.n_options, (*batch_shape, n_children), generator=generator).to(u.device)
        children_opt = torch.where(switch, new_opt, children_opt)
    return children, children_opt


def nearest_neighbor_distance(query: Tensor, archive: Sequence[Tensor], chunk: int = 8192) -> Tensor:
    """
    Distance from each query point to its nearest distinct archive point.

    :param query: ``K x E x df`` query points.
    :param archive: Sequence of ``K x A_i x df`` archive tensors.
    :returns: ``K x E`` nearest-neighbor distances (``inf`` if none is found).
    """
    best = torch.full(query.shape[:-1], float("inf"), dtype=query.dtype, device=query.device)
    for block in archive:
        for start in range(0, block.shape[-2], chunk):
            dist = torch.cdist(query, block[..., start : start + chunk, :])
            dist = torch.where(dist > 1e-9, dist, torch.full_like(dist, float("inf")))
            best = torch.minimum(best, dist.min(dim=-1).values)
    return best


def _scheduled_halfwidth(options: SamplingOptions, round_index: int) -> float:
    return options.init_halfwidth * options.shrink**round_index


def scoped_search(
    score_fn: Callable[[Tensor, Tensor], Tensor],
    space: SearchSpace,
    init_u: Tensor,
    init_opt: Tensor,
    init_values: Tensor,
    options: SamplingOptions,
    generator: torch.Generator,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Run ``K`` independent scoped random searches for point maximizers.

    :param score_fn: Maps ``(u, opt)`` with shapes ``K x n x d`` and ``K x n`` to ``K x n`` scores.
    :param space: Search space.
    :param init_u: ``K x n0 x d`` initial points in unit coordinates.
    :param init_opt: ``K x n0`` option indices for the initial points.
    :param init_values: ``K x n0`` scores of the initial points.
    :param options: Sampling options.
    :param generator: CPU random generator.
    :returns: ``(elite_u, elite_opt, elite_values)`` sorted by decreasing score, with shapes
        ``K x E x d``, ``K x E`` and ``K x E``.
    """
    n_elites = min(options.n_elites, init_u.shape[-2])
    values, index = torch.topk(init_values, n_elites, dim=-1)
    elite_u = torch.gather(init_u, -2, index.unsqueeze(-1).expand(*index.shape, space.d))
    elite_opt = torch.gather(init_opt, -1, index)
    archive = [init_u[..., space.free_index]]
    for round_index in range(options.n_rounds):
        nn_dist = nearest_neighbor_distance(elite_u[..., space.free_index], archive)
        halfwidth = torch.clamp(
            torch.clamp(nn_dist, max=_scheduled_halfwidth(options, round_index)),
            min=options.min_halfwidth,
        )
        child_u, child_opt = perturb(elite_u, elite_opt, halfwidth, options.samples_per_elite, space, options, generator)
        K = child_u.shape[0]
        child_u = child_u.reshape(K, -1, space.d)
        child_opt = child_opt.reshape(K, -1)
        child_u, feasible = space.repair(child_u, child_opt)
        child_values = score_fn(child_u, child_opt)
        child_values = torch.where(feasible, child_values, torch.full_like(child_values, -float("inf")))
        archive.append(child_u[..., space.free_index])
        pool_u = torch.cat([elite_u, child_u], dim=-2)
        pool_opt = torch.cat([elite_opt, child_opt], dim=-1)
        pool_values = torch.cat([values, child_values], dim=-1)
        values, index = torch.topk(pool_values, n_elites, dim=-1)
        elite_u = torch.gather(pool_u, -2, index.unsqueeze(-1).expand(*index.shape, space.d))
        elite_opt = torch.gather(pool_opt, -1, index)
    return elite_u, elite_opt, values


def _chunked_point_scores(acq_function: Callable[[Tensor], Tensor], x: Tensor, eval_batch: int) -> Tensor:
    """Evaluate an acquisition function at ``n x d`` points one point per t-batch."""
    out = []
    for start in range(0, x.shape[0], eval_batch):
        out.append(acq_function(x[start : start + eval_batch].unsqueeze(-2)).reshape(-1))
    return torch.cat(out) if out else x.new_zeros(0)


def _chunked_batch_scores(acq_function: Callable[[Tensor], Tensor], x: Tensor, eval_batch: int) -> Tensor:
    """Evaluate an acquisition function at ``n x q x d`` batches."""
    out = []
    step = max(1, eval_batch // max(1, x.shape[-2]))
    for start in range(0, x.shape[0], step):
        out.append(acq_function(x[start : start + step]).reshape(-1))
    return torch.cat(out) if out else x.new_zeros(0)


def _finite_or_neg_inf(values: Tensor) -> Tensor:
    return torch.where(torch.isfinite(values), values, torch.full_like(values, -float("inf")))


class _PointObjective:
    """Wrap an acquisition function as a ``K=1`` scoped-search score function."""

    def __init__(self, acq_function: Callable[[Tensor], Tensor], space: SearchSpace, eval_batch: int):
        self.acq_function = acq_function
        self.space = space
        self.eval_batch = eval_batch

    def __call__(self, u: Tensor, opt: Tensor) -> Tensor:
        x = self.space.to_x(u, opt).reshape(-1, self.space.d)
        values = _chunked_point_scores(self.acq_function, x, self.eval_batch)
        return _finite_or_neg_inf(values.reshape(u.shape[:-1]))


def _initial_pool(
    space: SearchSpace,
    n: int,
    generator: torch.Generator,
    seed: int,
    batch_initial_conditions: Optional[Tensor],
) -> Tuple[Tensor, Tensor]:
    u, opt = space.draw(n, generator, seed)
    if batch_initial_conditions is not None:
        x0 = batch_initial_conditions.reshape(-1, space.d).to(device=space.device, dtype=space.dtype)
        x0 = x0[space.feasible(x0)]
        if x0.shape[0] > 0:
            u = torch.cat([space.to_u(x0), u])
            opt = torch.cat([space.option_of(x0), opt])
    return u, opt


def _point_search(
    acq_function: Callable[[Tensor], Tensor],
    space: SearchSpace,
    options: SamplingOptions,
    generator: torch.Generator,
    seed: int,
    batch_initial_conditions: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Return the best single point (``1 x d``) and its acquisition value."""
    objective = _PointObjective(acq_function, space, options.eval_batch)
    u0, opt0 = _initial_pool(space, options.raw_samples, generator, seed, batch_initial_conditions)
    values0 = objective(u0.unsqueeze(0), opt0.unsqueeze(0))
    elite_u, elite_opt, values = scoped_search(objective, space, u0.unsqueeze(0), opt0.unsqueeze(0), values0, options, generator)
    elite_x = space.to_x(elite_u[0], elite_opt[0])
    values = values[0]
    refine = getattr(acq_function, "refine_scores", None)
    if refine is not None and options.kg_refine_top > 0:
        top = min(options.kg_refine_top, elite_x.shape[0])
        refined = refine(elite_x[:top].unsqueeze(-2))
        best = int(torch.argmax(refined))
        return elite_x[best : best + 1], refined[best]
    if not torch.isfinite(values[0]):
        raise ValueError("Gradient-free acquisition optimization found no feasible candidate with a finite acquisition value.")
    return elite_x[:1], values[0]


def _joint_search(
    acq_function: Callable[[Tensor], Tensor],
    space: SearchSpace,
    q: int,
    options: SamplingOptions,
    generator: torch.Generator,
    seed: int,
    batch_initial_conditions: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Search jointly over ``q x d`` batches. Returns ``(q x d, value)``."""
    if space.n_options > 1:
        raise NotImplementedError("Joint q-batch search is not supported with multiple fixed-feature choices (multi-fidelity); use batch_mode='sequential'.")
    d = space.d
    u0, _ = space.draw(options.raw_samples, generator, seed)
    x0 = space.to_x(u0)
    point_values = _finite_or_neg_inf(_chunked_point_scores(acq_function, x0, options.eval_batch))
    n_pool = min(u0.shape[0], max(options.n_elites, 4 * q))
    pool_index = torch.topk(point_values, n_pool).indices
    n_seed = max(options.n_elites, options.raw_samples // q)
    n_combo = n_seed // 2
    combo = torch.argsort(torch.rand(n_combo, n_pool, generator=generator), dim=-1)[:, :q] if n_pool >= q else torch.randint(n_pool, (n_combo, q), generator=generator)
    combo_u = u0[pool_index[combo.to(u0.device)]]
    random_u = u0[torch.randint(u0.shape[0], (n_seed - n_combo, q), generator=generator).to(u0.device)]
    seeds = [combo_u, random_u]
    if batch_initial_conditions is not None and batch_initial_conditions.shape[-2] == q:
        bic = batch_initial_conditions.to(device=space.device, dtype=space.dtype)
        bic = bic[space.feasible(bic).all(-1)]
        if bic.shape[0] > 0:
            seeds.append(space.to_u(bic))
    batch_u = torch.cat(seeds)
    values = _finite_or_neg_inf(_chunked_batch_scores(acq_function, space.to_x(batch_u), options.eval_batch))
    n_elites = min(options.n_elites, batch_u.shape[0])
    values, index = torch.topk(values, n_elites)
    elite_u = batch_u[index]
    archive = [u0[:, space.free_index].unsqueeze(0)]
    zero_opt = torch.zeros(1, dtype=torch.long, device=space.device)
    for round_index in range(options.n_rounds):
        members = elite_u.reshape(1, -1, d)
        nn_dist = nearest_neighbor_distance(members[..., space.free_index], archive)
        halfwidth = torch.clamp(
            torch.clamp(nn_dist, max=_scheduled_halfwidth(options, round_index)),
            min=options.min_halfwidth,
        ).reshape(n_elites, q)
        C = options.samples_per_elite
        # Perturb every member of each parent batch (each in its own box), optionally a random subset of members.
        moved, _ = perturb(
            elite_u, zero_opt.expand(n_elites, q), halfwidth, C, space, options, generator
        )  # n_elites x q x C x d
        moved = moved.permute(0, 2, 1, 3)  # n_elites x C x q x d
        member_move = torch.rand(n_elites, C, q, generator=generator) < options.joint_member_prob
        forced = torch.randint(q, (n_elites, C, 1), generator=generator)
        member_move.scatter_(-1, forced, True)
        member_move = member_move.to(space.device).unsqueeze(-1)
        parents = elite_u.unsqueeze(1).expand(n_elites, C, q, d)
        child_u = torch.where(member_move, moved, parents).reshape(-1, q, d)
        child_u, feasible = space.repair(child_u, torch.zeros(child_u.shape[:-1], dtype=torch.long, device=space.device))
        feasible = feasible.all(-1)
        child_values = _finite_or_neg_inf(_chunked_batch_scores(acq_function, space.to_x(child_u), options.eval_batch))
        child_values = torch.where(feasible, child_values, torch.full_like(child_values, -float("inf")))
        archive.append(child_u.reshape(1, -1, d)[..., space.free_index])
        pool_u = torch.cat([elite_u, child_u])
        pool_values = torch.cat([values, child_values])
        values, index = torch.topk(pool_values, n_elites)
        elite_u = pool_u[index]
    elite_x = space.to_x(elite_u)
    refine = getattr(acq_function, "refine_scores", None)
    if refine is not None and options.kg_refine_top > 0:
        top = min(options.kg_refine_top, elite_x.shape[0])
        refined = refine(elite_x[:top])
        best = int(torch.argmax(refined))
        return elite_x[best], refined[best]
    if not torch.isfinite(values[0]):
        raise ValueError("Gradient-free joint acquisition optimization found no feasible batch with a finite acquisition value.")
    return elite_x[0], values[0]


# ----------------------------------------------------------------------------- Boltzmann samplers


def _standardize_constants(values: Tensor) -> Tuple[Tensor, Tensor]:
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        raise ValueError("Acquisition function returned no finite values while building the Boltzmann density.")
    mean = finite.mean()
    std = finite.std() if finite.numel() > 1 else torch.ones_like(mean)
    # Match BoTorch's standardize(): a (near-)constant acquisition gets unit scale.
    std = torch.where(std < 1e-9, torch.ones_like(std), std)
    return mean, std


def _systematic_resample(weights: Tensor, n: int, generator: torch.Generator) -> Tensor:
    cdf = torch.cumsum(weights, dim=0)
    cdf = cdf / cdf[-1]
    offset = torch.rand(1, generator=generator).item()
    positions = (torch.arange(n, dtype=weights.dtype, device=weights.device) + offset) / n
    return torch.searchsorted(cdf, positions).clamp(max=weights.shape[0] - 1)


def _ess(log_weights: Tensor) -> float:
    return float(torch.exp(2 * torch.logsumexp(log_weights, 0) - torch.logsumexp(2 * log_weights, 0)))


def _lambda_schedule(lam: float, n_rounds: int) -> List[float]:
    if n_rounds <= 1:
        return [lam]
    return [lam * 10.0 ** (-2.0 * (1.0 - k / n_rounds)) for k in range(1, n_rounds + 1)]


def _mh_moves(
    u: Tensor,
    opt: Tensor,
    z: Tensor,
    raw: Tensor,
    lam: float,
    halfwidth: float,
    steps: int,
    objective: _PointObjective,
    space: SearchSpace,
    mean: Tensor,
    std: Tensor,
    options: SamplingOptions,
    generator: torch.Generator,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Random-walk Metropolis moves targeting ``exp(lam * z)`` with symmetric box proposals."""
    for _ in range(steps):
        h = torch.full(u.shape[:-1], halfwidth, dtype=u.dtype, device=u.device)
        prop_u, prop_opt = perturb(u, opt, h, 1, space, options, generator)
        prop_u, prop_opt = prop_u[:, 0], prop_opt[:, 0]
        prop_u, feasible = space.repair(prop_u, prop_opt)
        prop_raw = objective(prop_u.unsqueeze(0), prop_opt.unsqueeze(0))[0]
        prop_z = (prop_raw - mean) / std
        log_ratio = lam * (prop_z - z)
        accept = feasible & torch.isfinite(prop_z) & (torch.log(_rand(z.shape, generator, z)) < log_ratio)
        u = torch.where(accept.unsqueeze(-1), prop_u, u)
        opt = torch.where(accept, prop_opt, opt)
        z = torch.where(accept, prop_z, z)
        raw = torch.where(accept, prop_raw, raw)
    return u, opt, z, raw


def _boltzmann_importance(objective, space, q, options, generator, seed):
    u, opt = space.draw(options.boltzmann_population, generator, seed)
    raw = objective(u.unsqueeze(0), opt.unsqueeze(0))[0]
    mean, std = _standardize_constants(raw)
    logits = options.boltzmann_lambda * (raw - mean) / std
    weights = torch.softmax(_finite_or_neg_inf(logits), dim=0)
    index = torch.multinomial(weights.cpu(), q, replacement=True, generator=generator).to(u.device)
    return space.to_x(u[index], opt[index]), raw[index]


def _boltzmann_smc(objective, space, q, options, generator, seed):
    u, opt = space.draw(options.boltzmann_population, generator, seed)
    n = u.shape[0]
    raw = objective(u.unsqueeze(0), opt.unsqueeze(0))[0]
    mean, std = _standardize_constants(raw)
    z = _finite_or_neg_inf((raw - mean) / std)
    log_w = torch.zeros(n, dtype=u.dtype, device=u.device)
    lam_prev = 0.0
    lam_k = options.boltzmann_lambda
    for k, lam_k in enumerate(_lambda_schedule(options.boltzmann_lambda, options.boltzmann_n_rounds)):
        log_w = log_w + (lam_k - lam_prev) * z
        log_w = torch.where(torch.isfinite(log_w), log_w, torch.full_like(log_w, -float("inf")))
        if _ess(log_w) < n / 2:
            index = _systematic_resample(torch.softmax(log_w, 0), n, generator)
            u, opt, z, raw = u[index], opt[index], z[index], raw[index]
            log_w = torch.zeros_like(log_w)
        spread = float(u[:, space.free_index].std(dim=0).mean()) if n > 1 else options.init_halfwidth
        halfwidth = max(options.min_halfwidth, min(_scheduled_halfwidth(options, k), 2.0 * spread))
        u, opt, z, raw = _mh_moves(
            u, opt, z, raw, lam_k, halfwidth, options.boltzmann_mh_steps, objective, space, mean, std, options, generator
        )
        lam_prev = lam_k
    weights = torch.softmax(log_w, 0)
    index = torch.multinomial(weights.cpu(), q, replacement=True, generator=generator).to(u.device)
    sel_u, sel_opt, sel_z, sel_raw = u[index], opt[index], z[index], raw[index]
    # Extra independent MH moves decorrelate duplicated particles while preserving the target.
    spread = float(u[:, space.free_index].std(dim=0).mean()) if n > 1 else options.init_halfwidth
    halfwidth = max(options.min_halfwidth, min(_scheduled_halfwidth(options, options.boltzmann_n_rounds), 2.0 * spread))
    sel_u, sel_opt, sel_z, sel_raw = _mh_moves(
        sel_u, sel_opt, sel_z, sel_raw, lam_k, halfwidth, max(1, options.boltzmann_mh_steps), objective, space, mean, std, options, generator
    )
    return space.to_x(sel_u, sel_opt), sel_raw


class DeferPartition:
    """
    Recursive-partitioning approximation of ``exp(lam * z(x))`` (DEFER-style).

    Cells are axis-aligned boxes in unit coordinates, each tied to one
    fixed-feature option. The cells with the largest estimated mass
    (``volume x mean(exp(lam z))`` over the evaluated points they contain) are
    repeatedly split at the midpoint of their longest free side, and the new
    child centers are evaluated, until the evaluation budget is spent.
    """

    def __init__(self, objective: _PointObjective, space: SearchSpace, options: SamplingOptions, generator: torch.Generator, seed: int):
        self.space = space
        self.options = options
        self.generator = generator
        self.objective = objective
        self.lam = options.boltzmann_lambda
        budget = options.boltzmann_defer_budget
        n_init = max(2, min(budget // 4, 1024))
        engine = SobolEngine(space.d, scramble=True, seed=int(seed))
        u = engine.draw(n_init).to(device=space.device, dtype=space.dtype)
        opt = torch.randint(space.n_options, (n_init,), generator=generator).to(space.device)
        raw = objective(u.unsqueeze(0), opt.unsqueeze(0))[0]
        self.mean, self.std = _standardize_constants(raw)
        self.u, self.opt = u, opt
        self.logv = self.lam * _finite_or_neg_inf((raw - self.mean) / self.std)
        n_opt = space.n_options
        self.lo = torch.zeros(n_opt, space.d, dtype=space.dtype, device=space.device)
        self.hi = torch.ones(n_opt, space.d, dtype=space.dtype, device=space.device)
        self.cell_opt = torch.arange(n_opt, device=space.device)
        self.point_cell = opt.clone()
        self.used = n_init
        while self.used + 2 <= budget:
            n_split = int(min(self.lo.shape[0], max(1, (budget - self.used) // 2), 256))
            self._split(n_split)

    def log_masses(self) -> Tensor:
        """Log of the unnormalized mass of each cell."""
        n_cells = self.lo.shape[0]
        width = (self.hi - self.lo)[:, self.space.free_index].clamp_min(1e-30)
        log_volume = torch.log(width).sum(-1)
        max_v = torch.full((n_cells,), -float("inf"), dtype=self.logv.dtype, device=self.logv.device)
        max_v = max_v.scatter_reduce(0, self.point_cell, self.logv, reduce="amax", include_self=True)
        safe_max = torch.where(torch.isfinite(max_v), max_v, torch.zeros_like(max_v))
        sums = torch.zeros(n_cells, dtype=self.logv.dtype, device=self.logv.device)
        sums.index_add_(0, self.point_cell, torch.exp(self.logv - safe_max[self.point_cell]))
        counts = torch.bincount(self.point_cell, minlength=n_cells).to(self.logv.dtype)
        log_mean = safe_max + torch.log(sums.clamp_min(1e-300)) - torch.log(counts.clamp_min(1.0))
        log_mean = torch.where(counts > 0, log_mean, torch.full_like(log_mean, -float("inf")))
        return log_volume + log_mean

    def _split(self, n_split: int) -> None:
        space = self.space
        log_mass = self.log_masses()
        chosen = torch.topk(log_mass, n_split).indices
        width = (self.hi - self.lo)[chosen][:, space.free_index]
        jitter = 1e-6 * torch.rand(width.shape, generator=self.generator).to(width)
        split_dim = space.free_index[torch.argmax(width + jitter, dim=-1)]
        rows = torch.arange(n_split, device=space.device)
        parent_lo = self.lo[chosen].clone()
        parent_hi = self.hi[chosen].clone()
        mid = 0.5 * (parent_lo[rows, split_dim] + parent_hi[rows, split_dim])
        left_hi = parent_hi.clone()
        left_hi[rows, split_dim] = mid
        right_lo = parent_lo.clone()
        right_lo[rows, split_dim] = mid
        new_ids = torch.arange(self.lo.shape[0], self.lo.shape[0] + n_split, device=space.device)
        # The left child keeps the parent id; the right child is appended.
        self.hi[chosen] = left_hi
        self.lo = torch.cat([self.lo, right_lo])
        self.hi = torch.cat([self.hi, parent_hi])
        self.cell_opt = torch.cat([self.cell_opt, self.cell_opt[chosen]])
        # Reassign existing points of the split cells.
        lookup = torch.full((self.lo.shape[0],), -1, dtype=torch.long, device=space.device)
        lookup[chosen] = rows
        member = lookup[self.point_cell]
        in_split = member >= 0
        if in_split.any():
            point_rows = member[in_split]
            goes_right = self.u[in_split, split_dim[point_rows]] >= mid[point_rows]
            updated = self.point_cell[in_split]
            updated = torch.where(goes_right, new_ids[point_rows], updated)
            self.point_cell[in_split] = updated
        # Evaluate both children's centers.
        child_ids = torch.cat([chosen, new_ids])
        centers = 0.5 * (self.lo[child_ids] + self.hi[child_ids])
        child_opt = self.cell_opt[child_ids]
        raw = self.objective(centers.unsqueeze(0), child_opt.unsqueeze(0))[0]
        self.u = torch.cat([self.u, centers])
        self.opt = torch.cat([self.opt, child_opt])
        self.point_cell = torch.cat([self.point_cell, child_ids])
        self.logv = torch.cat([self.logv, self.lam * _finite_or_neg_inf((raw - self.mean) / self.std)])
        self.used += centers.shape[0]

    def sample(self, q: int, max_tries: int = 100) -> Tuple[Tensor, Tensor]:
        """Draw ``q`` feasible points from the piecewise-constant density."""
        space = self.space
        probs = torch.softmax(self.log_masses(), 0)
        kept_u, kept_opt, total = [], [], 0
        for _ in range(max_tries):
            n_draw = max(2 * (q - total), 16)
            cells = torch.multinomial(probs.cpu(), n_draw, replacement=True, generator=self.generator).to(space.device)
            u = self.lo[cells] + _rand((n_draw, space.d), self.generator, self.lo) * (self.hi[cells] - self.lo[cells])
            opt = self.cell_opt[cells]
            u, feasible = space.repair(u, opt)
            if feasible.any():
                kept_u.append(u[feasible])
                kept_opt.append(opt[feasible])
                total += int(feasible.sum())
            if total >= q:
                break
        if total < q:
            raise ValueError("DEFER Boltzmann sampler could not draw enough feasible points; relax the constraints.")
        return torch.cat(kept_u)[:q], torch.cat(kept_opt)[:q]


def _boltzmann_defer(objective, space, q, options, generator, seed):
    partition = DeferPartition(objective, space, options, generator, seed)
    u, opt = partition.sample(q)
    raw = objective(u.unsqueeze(0), opt.unsqueeze(0))[0]
    return space.to_x(u, opt), raw


_BOLTZMANN_SAMPLER_FUNCS = {
    "smc": _boltzmann_smc,
    "defer": _boltzmann_defer,
    "importance": _boltzmann_importance,
}


def sample_acqf_boltzmann(
    acq_function: Callable[[Tensor], Tensor],
    space: SearchSpace,
    q: int,
    options: SamplingOptions,
    generator: torch.Generator,
    seed: int,
) -> Tuple[Tensor, Tensor]:
    """
    Draw ``q`` i.i.d. points from ``p(x) ∝ exp(λ·standardize(α(x)))``.

    The acquisition function is evaluated one point at a time (``q=1``). Values
    are standardized with the mean and standard deviation of the initial
    population, which makes ``λ`` (``boltzmann_lambda``) dimensionless.

    :returns: ``(q x d candidates, q acquisition values)``.
    """
    if options.boltzmann_sampler not in _BOLTZMANN_SAMPLER_FUNCS:
        raise ValueError(f"Unknown boltzmann_sampler {options.boltzmann_sampler}. Options: {list(_BOLTZMANN_SAMPLER_FUNCS)}")
    objective = _PointObjective(acq_function, space, options.eval_batch)
    return _BOLTZMANN_SAMPLER_FUNCS[options.boltzmann_sampler](objective, space, q, options, generator, seed)


def optimize_acqf_gradient_free(
    acq_function: Any,
    bounds: Tensor,
    q: int,
    method: str,
    options: SamplingOptions,
    sequential: bool = False,
    seed: int = 0,
    fixed_features_list: Optional[Sequence[Dict[int, float]]] = None,
    inequality_constraints: Optional[Sequence[Tuple[Tensor, Tensor, float]]] = None,
    equality_constraints: Optional[Sequence[Tuple[Tensor, Tensor, float]]] = None,
    nonlinear_inequality_constraints: Optional[Sequence[Callable[[Tensor], Tensor]]] = None,
    post_processing_func: Optional[Callable[[Tensor], Tensor]] = None,
    batch_initial_conditions: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Gradient-free drop-in replacement for ``optimize_acqf`` / ``optimize_acqf_mixed``.

    Every acquisition evaluation runs under ``torch.no_grad()``, so surrogates
    without autograd support (for example random forests) work.

    :param acq_function: A BoTorch-style acquisition function mapping ``b x q x d`` to ``b``.
    :param bounds: ``2 x d`` bounds in model coordinates.
    :param q: Number of candidates.
    :param method: ``'sampling'`` (scoped random search) or ``'boltzmann'``.
    :param options: Sampling options.
    :param sequential: For ``'sampling'``, build the batch greedily with ``X_pending`` (``True``) or search
        jointly over ``q x d`` batches (``False``). Ignored for ``'boltzmann'``, which always draws i.i.d.
    :param seed: Random seed.
    :param fixed_features_list: Optional list of ``{column: value}`` choices (as in ``optimize_acqf_mixed``).
    :param inequality_constraints: Linear ``sum(c_i x_i) >= rhs`` constraints.
    :param equality_constraints: Linear ``sum(c_i x_i) == rhs`` constraints.
    :param nonlinear_inequality_constraints: Callables where ``>= 0`` is feasible.
    :param post_processing_func: Optional function applied to the final candidates.
    :param batch_initial_conditions: Optional initial points added to the starting pool.
    :returns: ``(candidates, acq_value)``. Candidates have shape ``q x d``. ``acq_value`` is a scalar for joint
        search and a ``q``-dim tensor for sequential search or Boltzmann sampling.
    """
    if method not in ("sampling", "boltzmann"):
        raise ValueError(f"Unknown gradient-free acquisition optimizer {method}.")
    space = SearchSpace(
        bounds=bounds,
        fixed_features_list=fixed_features_list,
        inequality_constraints=inequality_constraints,
        equality_constraints=equality_constraints,
        nonlinear_inequality_constraints=nonlinear_inequality_constraints,
    )
    generator = torch.Generator().manual_seed(int(seed))
    with torch.no_grad():
        if method == "boltzmann":
            candidates, acq_value = sample_acqf_boltzmann(acq_function, space, q, options, generator, seed)
        elif q == 1 or sequential:
            candidates, acq_value = _sequential_search(acq_function, space, q, options, generator, seed, batch_initial_conditions)
        else:
            candidates, acq_value = _joint_search(acq_function, space, q, options, generator, seed, batch_initial_conditions)
            candidates = candidates.reshape(q, -1)
        if post_processing_func is not None:
            candidates = post_processing_func(candidates)
    return candidates.detach(), acq_value.detach()


def _sequential_search(acq_function, space, q, options, generator, seed, batch_initial_conditions):
    if q == 1:
        candidate, value = _point_search(acq_function, space, options, generator, seed, batch_initial_conditions)
        return candidate, value.reshape(())
    if not hasattr(acq_function, "set_X_pending"):
        raise NotImplementedError("Sequential batch generation requires an acquisition function with set_X_pending.")
    base_X_pending = getattr(acq_function, "X_pending", None)
    candidates, values = [], []
    try:
        for step in range(q):
            candidate, value = _point_search(acq_function, space, options, generator, seed + 7919 * step, batch_initial_conditions)
            candidates.append(candidate)
            values.append(value.reshape(()))
            stacked = torch.cat(candidates, dim=-2)
            acq_function.set_X_pending(torch.cat([base_X_pending, stacked], dim=-2) if base_X_pending is not None else stacked)
    finally:
        acq_function.set_X_pending(base_X_pending)
    return torch.cat(candidates, dim=-2), torch.stack(values)


def batched_max_search(
    value_fn: Callable[[Tensor], Tensor],
    bounds: Tensor,
    K: int,
    options: SamplingOptions,
    seed: int,
    fixed_features_list: Optional[Sequence[Dict[int, float]]] = None,
    inequality_constraints: Optional[Sequence[Tuple[Tensor, Tensor, float]]] = None,
    nonlinear_inequality_constraints: Optional[Sequence[Callable[[Tensor], Tensor]]] = None,
) -> Tensor:
    """
    Run ``K`` independent scoped maximizations (e.g. one per fantasy model).

    :param value_fn: Maps ``K x n x d`` model-coordinate points to ``K x n`` values.
    :param bounds: ``2 x d`` bounds.
    :param K: Number of independent searches.
    :returns: ``K`` best values found.
    """
    space = SearchSpace(
        bounds=bounds,
        fixed_features_list=fixed_features_list,
        inequality_constraints=inequality_constraints,
        nonlinear_inequality_constraints=nonlinear_inequality_constraints,
    )
    generator = torch.Generator().manual_seed(int(seed))

    def score(u: Tensor, opt: Tensor) -> Tensor:
        x = space.to_x(u, opt)
        out = []
        step = max(1, options.eval_batch // max(1, K))
        for start in range(0, x.shape[1], step):
            out.append(value_fn(x[:, start : start + step]))
        return _finite_or_neg_inf(torch.cat(out, dim=1))

    with torch.no_grad():
        u0, opt0 = space.draw(options.raw_samples, generator, seed)
        u0 = u0.unsqueeze(0).expand(K, *u0.shape)
        opt0 = opt0.unsqueeze(0).expand(K, *opt0.shape)
        values0 = score(u0, opt0)
        _, _, values = scoped_search(score, space, u0, opt0, values0, options, generator)
    return values[:, 0]


def search_elites(
    acq_function: Callable[[Tensor], Tensor],
    bounds: Tensor,
    n: int,
    options: SamplingOptions,
    seed: int,
    fixed_features_list: Optional[Sequence[Dict[int, float]]] = None,
    inequality_constraints: Optional[Sequence[Tuple[Tensor, Tensor, float]]] = None,
    equality_constraints: Optional[Sequence[Tuple[Tensor, Tensor, float]]] = None,
    nonlinear_inequality_constraints: Optional[Sequence[Callable[[Tensor], Tensor]]] = None,
) -> Tensor:
    """
    Return the ``n`` best points (``n x d``) found by a scoped search of a point acquisition.

    Used to build the shared inner set of the gradient-free knowledge gradient.
    """
    space = SearchSpace(
        bounds=bounds,
        fixed_features_list=fixed_features_list,
        inequality_constraints=inequality_constraints,
        equality_constraints=equality_constraints,
        nonlinear_inequality_constraints=nonlinear_inequality_constraints,
    )
    generator = torch.Generator().manual_seed(int(seed))
    options = replace(options, n_elites=max(options.n_elites, n))
    with torch.no_grad():
        objective = _PointObjective(acq_function, space, options.eval_batch)
        u0, opt0 = space.draw(options.raw_samples, generator, seed)
        values0 = objective(u0.unsqueeze(0), opt0.unsqueeze(0))
        elite_u, elite_opt, values = scoped_search(objective, space, u0.unsqueeze(0), opt0.unsqueeze(0), values0, options, generator)
    keep = torch.isfinite(values[0])
    return space.to_x(elite_u[0][keep], elite_opt[0][keep])[:n]


def draw_feasible_points(
    bounds: Tensor,
    n: int,
    seed: int,
    fixed_features_list: Optional[Sequence[Dict[int, float]]] = None,
    inequality_constraints: Optional[Sequence[Tuple[Tensor, Tensor, float]]] = None,
    equality_constraints: Optional[Sequence[Tuple[Tensor, Tensor, float]]] = None,
    nonlinear_inequality_constraints: Optional[Sequence[Callable[[Tensor], Tensor]]] = None,
) -> Tensor:
    """Draw up to ``n`` feasible scrambled-Sobol points (``n' x d``) in model coordinates."""
    space = SearchSpace(
        bounds=bounds,
        fixed_features_list=fixed_features_list,
        inequality_constraints=inequality_constraints,
        equality_constraints=equality_constraints,
        nonlinear_inequality_constraints=nonlinear_inequality_constraints,
    )
    generator = torch.Generator().manual_seed(int(seed))
    with torch.no_grad():
        u, opt = space.draw(n, generator, seed)
    return space.to_x(u, opt)
