#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
!!! note
    This module was adopted from the botorch library. There was a bug
    we decided to fix on our own so we didn't have to wait for them to
    adopt our changes. You can view their library
    [here](https://github.com/pytorch/botorch).

Acquisition functions for Max-value Entropy Search (MES), General
Information-Based Bayesian Optimization (GIBBON), and
multi-fidelity MES with noisy observations and trace observations.

References

.. [Moss2021gibbon]
    Moss, H. B., et al.,
    GIBBON: General-purpose Information-Based Bayesian OptimisatioN
    arXiv:2102.03324, 2021

.. [Takeno2020mfmves]
    S. Takeno, H. Fukuoka, Y. Tsukada, T. Koyama, M. Shiga, I. Takeuchi,
    M. Karasuyama. Multi-fidelity Bayesian Optimization with Max-value Entropy
    Search and its Parallelization. Proceedings of the 37th International
    Conference on Machine Learning, 2020.

.. [Wang2017mves]
    Z. Wang, S. Jegelka, Max-value Entropy Search for Efficient
    Bayesian Optimization. Proceedings of the 37th International
    Conference on Machine Learning, 2017.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from math import log
from typing import Any, Callable, Optional

import numpy as np
import torch
from deepopt._compat import make_sobol_qmc_normal_sampler
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.cost_aware import CostAwareUtility, InverseCostWeightedUtility
from botorch.acquisition.objective import PosteriorTransform
from botorch.exceptions.errors import UnsupportedError
from botorch.models.cost import AffineFidelityCostModel
from botorch.models.model import Model
from botorch.models.utils import check_no_nans
from botorch.utils.transforms import match_batch_shape, t_batch_mode_transform
from gpytorch.functions import inv_quad
from gpytorch.utils.cholesky import psd_safe_cholesky
from scipy.optimize import brentq
from scipy.stats import norm
from torch import Tensor

CLAMP_LB = 1.0e-8


class MaxValueBase(AcquisitionFunction, ABC):
    r"""Abstract base class for acquisition functions based on Max-value Entropy Search.

    This class provides the basic building blocks for constructing max-value
    entropy-based acquisition functions along the lines of [Wang2017mves]_.

    Subclasses need to implement `_sample_max_values` and _compute_information_gain`
    methods.

    :meta private:
    """

    def __init__(
        self,
        model: Model,
        num_mv_samples: int,
        posterior_transform: Optional[PosteriorTransform] = None,
        maximize: bool = True,
        X_pending: Optional[Tensor] = None,
    ) -> None:
        r"""Single-outcome max-value entropy search-based acquisition functions.

        :param model: A fitted single-outcome model.
        :param num_mv_samples: Number of max value samples.
        :param posterior_transform: A PosteriorTransform. If using a multi-output model,
            a PosteriorTransform that transforms the multi-output posterior into a
            single-output posterior is required.
        :param maximize: If True, consider the problem a maximization problem.
        :param X_pending: A `m x d`-dim Tensor of `m` design points that have been
            submitted for function evaluation but have not yet been evaluated.
        """
        super().__init__(model=model)

        if posterior_transform is None and model.num_outputs != 1:
            raise UnsupportedError("Must specify a posterior transform when using a multi-output model.")

        # Batched GP models are not currently supported
        try:
            batch_shape = model.batch_shape
        except NotImplementedError:
            batch_shape = torch.Size()
        if len(batch_shape) > 0:
            raise NotImplementedError(
                "Batched GP models (e.g., fantasized models) are not yet " f"supported by `{self.__class__.__name__}`."
            )
        self.num_mv_samples = num_mv_samples
        self.posterior_transform = posterior_transform
        self.maximize = maximize
        self.weight = 1.0 if maximize else -1.0
        self.set_X_pending(X_pending)

    def forward(self, X: Tensor) -> Tensor:
        """
        Helper method to re-shape the tensor when requesting multiple candidate points prior to calculating
        max-value entropy at the design points `X`. Multiple candidate points causes the dimensions of this
        tensor to be `batch_shape x q x d` where q is the number of candidate points. This will break the
        q_forward method as it can only handle `batch_shape x 1 x d`-dim tensors.

        :param X: A `batch_shape x q x d`-dim Tensor of `batch_shape` t-batches
            with `q` candidate points resulting in `q x d`-dim design points each.

        :returns: A `batch_shape`-dim Tensor of MVE values at the given design points `X`.
        """
        if X.shape[-2] == 1:
            return self.q_forward(X)
        return self.q_forward(X.unsqueeze(-2))

    @t_batch_mode_transform(expected_q=1)
    def q_forward(self, X: Tensor) -> Tensor:
        r"""Compute max-value entropy at the design points `X`.

        :param X: A `batch_shape x 1 x d`-dim Tensor of `batch_shape` t-batches
            with `1` `d`-dim design points each.

        :returns: A `batch_shape`-dim Tensor of MVE values at the given design points `X`.
        """
        # Compute the posterior, posterior mean, variance and std
        posterior = self.model.posterior(
            X.unsqueeze(-3),
            observation_noise=False,
            posterior_transform=self.posterior_transform,
        )
        # batch_shape x num_fantasies x (m) x 1
        mean = self.weight * posterior.mean.squeeze(-1).squeeze(-1)
        variance = posterior.variance.clamp_min(CLAMP_LB).view_as(mean)
        ig = self._compute_information_gain(X=X, mean_M=mean, variance_M=variance, covar_mM=variance.unsqueeze(-1))
        return ig.mean(dim=0)  # average over fantasies

    def set_X_pending(self, X_pending: Optional[Tensor] = None) -> None:
        r"""Set pending design points.

        Set "pending points" to inform the acquisition function of the candidate
        points that have been generated but are pending evaluation.

        :param X_pending: `n x d` Tensor with `n` `d`-dim design points that have
            been submitted for evaluation but have not yet been evaluated.
        """
        if X_pending is not None:
            X_pending = X_pending.detach().clone()
        self._sample_max_values(num_samples=self.num_mv_samples, X_pending=X_pending)
        self.X_pending = X_pending

    # ------- Abstract methods that need to be implemented by subclasses ------- #

    @abstractmethod
    def _compute_information_gain(self, X: Tensor, **kwargs: Any) -> Tensor:
        r"""Compute the information gain at the design points `X`.

        `num_fantasies = 1` for non-fantasized models.

        :param X: A `batch_shape x 1 x d`-dim Tensor of `batch_shape` t-batches
            with `1` `d`-dim design point each.
        :param kwargs: Other keyword arguments used by subclasses.

        :returns: A `num_fantasies x batch_shape`-dim Tensor of information gains at the
            given design points `X` (`num_fantasies=1` for non-fantasized models).
        """
        pass  # pragma: no cover

    @abstractmethod
    def _sample_max_values(self, num_samples: int, X_pending: Optional[Tensor] = None) -> Tensor:
        r"""Draw samples from the posterior over maximum values.

        These samples are used to compute Monte Carlo approximations of expecations
        over the posterior over the function maximum.

        :param num_samples: The number of samples to draw.
        :param X_pending: A `m x d`-dim Tensor of `m` design points that have been
            submitted for function evaluation but have not yet been evaluated.

        :returns: A `num_samples x num_fantasies` Tensor of posterior max value samples
            (`num_fantasies=1` for non-fantasized models).
        """
        pass  # pragma: no cover


class DiscreteMaxValueBase(MaxValueBase):
    r"""Abstract base class for MES-like methods using discrete max posterior sampling.

    This class provides basic functionality for sampling posterior maximum values from
    a surrogate Gaussian process model using a discrete set of candidates. It supports
    either exact (w.r.t. the candidate set) sampling, or using a Gumbel approximation.
    """

    def __init__(
        self,
        model: Model,
        candidate_set: Tensor,
        num_mv_samples: int = 10,
        posterior_transform: Optional[PosteriorTransform] = None,
        use_gumbel: bool = True,
        maximize: bool = True,
        X_pending: Optional[Tensor] = None,
        train_inputs: Optional[Tensor] = None,
    ) -> None:
        r"""Single-outcome MES-like acquisition functions based on discrete MV sampling.

        :param model: A fitted single-outcome model.
        :param candidate_set: A `n x d` Tensor including `n` candidate points to
            discretize the design space. Max values are sampled from the
            (joint) model posterior over these points.
        :param num_mv_samples: Number of max value samples.
        :param posterior_transform: A PosteriorTransform. If using a multi-output model,
            a PosteriorTransform that transforms the multi-output posterior into a
            single-output posterior is required.
        :param use_gumbel: If True, use Gumbel approximation to sample the max values.
        :param maximize: If True, consider the problem a maximization problem.
        :param X_pending: A `m x d`-dim Tensor of `m` design points that have been
            submitted for function evaluation but have not yet been evaluated.
        :param train_inputs: A `n_train x d` Tensor that the model has been fitted on.
            Not required if the model is an instance of a GPyTorch ExactGP model.
        """
        self.use_gumbel = use_gumbel

        if train_inputs is None and hasattr(model, "train_inputs"):
            train_inputs = model.train_inputs[0]
        if train_inputs is not None:
            if train_inputs.ndim > 2:
                raise NotImplementedError(
                    "Batch GP models (e.g. fantasized models) " "are not yet supported by `MaxValueBase`"
                )
            train_inputs = match_batch_shape(train_inputs, candidate_set)
            candidate_set = torch.cat([candidate_set, train_inputs], dim=0)

        self.candidate_set = candidate_set

        super().__init__(
            model=model,
            num_mv_samples=num_mv_samples,
            posterior_transform=posterior_transform,
            maximize=maximize,
            X_pending=X_pending,
        )

    def _sample_max_values(self, num_samples: int, X_pending: Optional[Tensor] = None) -> Tensor:
        r"""Draw samples from the posterior over maximum values on a discrete set.

        These samples are used to compute Monte Carlo approximations of expecations
        over the posterior over the function maximum.

        :param num_samples: The number of samples to draw.
        :param X_pending: A `m x d`-dim Tensor of `m` design points that have been
            submitted for function evaluation but have not yet been evaluated.

        :returns: A `num_samples x num_fantasies` Tensor of posterior max value samples
            (`num_fantasies=1` for non-fantasized models).
        """
        if self.use_gumbel:
            sample_max_values = _sample_max_value_Gumbel
        else:
            sample_max_values = _sample_max_value_Thompson
        candidate_set = self.candidate_set

        with torch.no_grad():
            if X_pending is not None:
                # Append X_pending to candidate set
                X_pending = match_batch_shape(X_pending, self.candidate_set)
                candidate_set = torch.cat([self.candidate_set, X_pending], dim=0)

            # project the candidate_set to the highest fidelity,
            # which is needed for the multi-fidelity MES
            try:
                candidate_set = self.project(candidate_set)
            except AttributeError:
                pass

            self.posterior_max_values = sample_max_values(
                model=self.model,
                candidate_set=candidate_set,
                num_samples=self.num_mv_samples,
                posterior_transform=self.posterior_transform,
                maximize=self.maximize,
            )


class qMaxValueEntropy(DiscreteMaxValueBase):
    r"""The acquisition function for Max-value Entropy Search.

    This acquisition function computes the mutual information of max values and
    a candidate point X. See [Wang2017mves]_ for a detailed discussion.

    The model must be single-outcome. The batch case `q > 1` is supported
    through cyclic optimization and fantasies.

    Example:

        >>> model = SingleTaskGP(train_X, train_Y)
        >>> candidate_set = torch.rand(1000, bounds.size(1))
        >>> candidate_set = bounds[0] + (bounds[1] - bounds[0]) * candidate_set
        >>> MES = qMaxValueEntropy(model, candidate_set)
        >>> mes = MES(test_X)
    """

    def __init__(
        self,
        model: Model,
        candidate_set: Tensor,
        num_fantasies: int = 16,
        num_mv_samples: int = 10,
        num_y_samples: int = 128,
        posterior_transform: Optional[PosteriorTransform] = None,
        use_gumbel: bool = True,
        maximize: bool = True,
        X_pending: Optional[Tensor] = None,
        train_inputs: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> None:
        r"""Single-outcome max-value entropy search acquisition function.

        :param model: A fitted single-outcome model.
        :param candidate_set: A `n x d` Tensor including `n` candidate points to
            discretize the design space. Max values are sampled from the
            (joint) model posterior over these points.
        :param num_fantasies: Number of fantasies to generate. The higher this
            number the more accurate the model (at the expense of model
            complexity, wall time and memory). Ignored if `X_pending` is `None`.
        :param num_mv_samples: Number of max value samples.
        :param num_y_samples: Number of posterior samples at specific design point `X`.
        :param posterior_transform: A PosteriorTransform. If using a multi-output model,
            a PosteriorTransform that transforms the multi-output posterior into a
            single-output posterior is required.
        :param use_gumbel: If True, use Gumbel approximation to sample the max values.
        :param maximize: If True, consider the problem a maximization problem.
        :param X_pending: A `m x d`-dim Tensor of `m` design points that have been
            submitted for function evaluation but have not yet been evaluated.
        :param train_inputs: A `n_train x d` Tensor that the model has been fitted on.
            Not required if the model is an instance of a GPyTorch ExactGP model.
        """
        super().__init__(
            model=model,
            candidate_set=candidate_set,
            num_mv_samples=num_mv_samples,
            posterior_transform=posterior_transform,
            use_gumbel=use_gumbel,
            maximize=maximize,
            X_pending=X_pending,
            train_inputs=train_inputs,
        )
        self._init_model = model  # used for `fantasize()` when setting `X_pending`
        self.sampler = make_sobol_qmc_normal_sampler(num_y_samples, seed=kwargs.get("seed"))
        self.fantasies_sampler = make_sobol_qmc_normal_sampler(num_fantasies, seed=kwargs.get("seed"))
        self.num_fantasies = num_fantasies
        self.set_X_pending(X_pending)  # this did not happen in the super constructor

    def set_X_pending(self, X_pending: Optional[Tensor] = None) -> None:
        r"""Set pending points.

        Informs the acquisition function about pending design points,
        fantasizes the model on the pending points and draws max-value samples
        from the fantasized model posterior.

        :param X_pending: `m x d` Tensor with `m` `d`-dim design points that have
            been submitted for evaluation but have not yet been evaluated.
        """
        try:
            init_model = self._init_model
        except AttributeError:
            # Short-circuit (this allows calling the super constructor)
            return
        if X_pending is not None:
            # fantasize the model and use this as the new model
            self.model = init_model.fantasize(X=X_pending, sampler=self.fantasies_sampler, observation_noise=True)
        else:
            self.model = init_model
        super().set_X_pending(X_pending)

    def _compute_information_gain(self, X: Tensor, mean_M: Tensor, variance_M: Tensor, covar_mM: Tensor) -> Tensor:
        r"""Computes the information gain at the design points `X`.

        Approximately computes the information gain at the design points `X`,
        for both MES with noisy observations and multi-fidelity MES with noisy
        observation and trace observations.

        The implementation is inspired from the papers on multi-fidelity MES by
        [Takeno2020mfmves]_. The notation in the comments in this function follows
        the Appendix C of [Takeno2020mfmves]_.

        `num_fantasies = 1` for non-fantasized models.

        :param X: A `batch_shape x 1 x d`-dim Tensor of `batch_shape` t-batches
            with `1` `d`-dim design point each.
        :param mean_M: A `batch_shape x num_fantasies x (m)`-dim Tensor of means.
        :param variance_M: A `batch_shape x num_fantasies x (m)`-dim Tensor of variances.
        :param covar_mM: A `batch_shape x num_fantasies x (m) x (1 + num_trace_observations)`-dim
            Tensor of covariances.

        :returns: A `num_fantasies x batch_shape`-dim Tensor of information gains at the
            given design points `X` (`num_fantasies=1` for non-fantasized models).
        """
        # compute the std_m, variance_m with noisy observation
        posterior_m = self.model.posterior(
            X.unsqueeze(-3),
            observation_noise=True,
            posterior_transform=self.posterior_transform,
        )
        # batch_shape x num_fantasies x (m) x (1 + num_trace_observations)
        mean_m = self.weight * posterior_m.mean.squeeze(-1)
        # batch_shape x num_fantasies x (m) x (1 + num_trace_observations)
        variance_m = posterior_m.mvn.covariance_matrix
        check_no_nans(variance_m)

        # compute mean and std for fM|ym, x, Dt ~ N(u, s^2)
        samples_m = self.weight * self.sampler(posterior_m).squeeze(-1)
        # s_m x batch_shape x num_fantasies x (m) (1 + num_trace_observations)
        L = psd_safe_cholesky(variance_m)
        temp_term = torch.cholesky_solve(covar_mM.unsqueeze(-1), L).transpose(-2, -1)
        # equivalent to torch.matmul(covar_mM.unsqueeze(-2), torch.inverse(variance_m))
        # batch_shape x num_fantasies (m) x 1 x (1 + num_trace_observations)

        mean_pt1 = torch.matmul(temp_term, (samples_m - mean_m).unsqueeze(-1))
        mean_new = mean_pt1.squeeze(-1).squeeze(-1) + mean_M
        # s_m x batch_shape x num_fantasies x (m)
        variance_pt1 = torch.matmul(temp_term, covar_mM.unsqueeze(-1))
        variance_new = variance_M - variance_pt1.squeeze(-1).squeeze(-1)
        # batch_shape x num_fantasies x (m)
        stdv_new = variance_new.clamp_min(CLAMP_LB).sqrt()
        # batch_shape x num_fantasies x (m)

        # define normal distribution to compute cdf and pdf
        normal = torch.distributions.Normal(
            torch.zeros(1, device=X.device, dtype=X.dtype),
            torch.ones(1, device=X.device, dtype=X.dtype),
        )

        # Compute p(fM <= f* | ym, x, Dt)
        view_shape = torch.Size(
            [
                self.posterior_max_values.shape[0],
                # add 1s to broadcast across the batch_shape of X
                *[1 for _ in range(X.ndim - self.posterior_max_values.ndim)],
                *self.posterior_max_values.shape[1:],
            ]
        )  # s_M x batch_shape x num_fantasies x (m)
        max_vals = self.posterior_max_values.view(view_shape).unsqueeze(1)
        # s_M x 1 x batch_shape x num_fantasies x (m)
        normalized_mvs_new = (max_vals - mean_new) / stdv_new
        # s_M x s_m x batch_shape x num_fantasies x (m)  =
        #   s_M x 1 x batch_shape x num_fantasies x (m)
        #   - s_m x batch_shape x num_fantasies x (m)
        cdf_mvs_new = normal.cdf(normalized_mvs_new).clamp_min(CLAMP_LB)

        # Compute p(fM <= f* | x, Dt)
        stdv_M = variance_M.sqrt()
        normalized_mvs = (max_vals - mean_M) / stdv_M
        # s_M x 1 x batch_shape x num_fantasies  x (m) =
        # s_M x 1 x 1 x num_fantasies x (m) - batch_shape x num_fantasies x (m)
        cdf_mvs = normal.cdf(normalized_mvs).clamp_min(CLAMP_LB)
        # s_M x 1 x batch_shape x num_fantasies x (m)

        # Compute log(p(ym | x, Dt))
        log_pdf_fm = posterior_m.mvn.log_prob(self.weight * samples_m).unsqueeze(0)
        # 1 x s_m x batch_shape x num_fantasies x (m)

        # H0 = H(ym | x, Dt)
        H0 = posterior_m.mvn.entropy()  # batch_shape x num_fantasies x (m)

        # regression adjusted H1 estimation, H1_hat = H1_bar - beta * (H0_bar - H0)
        # H1 = E_{f*|x, Dt}[H(ym|f*, x, Dt)]
        Z = cdf_mvs_new / cdf_mvs  # s_M x s_m x batch_shape x num_fantasies x (m)
        # s_M x s_m x batch_shape x num_fantasies x (m)
        h1 = -Z * Z.log() - Z * log_pdf_fm
        check_no_nans(h1)
        dim = [0, 1]  # dimension of fm samples, fM samples
        H1_bar = h1.mean(dim=dim)
        h0 = -log_pdf_fm
        H0_bar = h0.mean(dim=dim)
        cov = ((h1 - H1_bar) * (h0 - H0_bar)).mean(dim=dim)
        beta = cov / (h0.var(dim=dim) * h1.var(dim=dim)).sqrt()
        H1_hat = H1_bar - beta * (H0_bar - H0)
        ig = H0 - H1_hat  # batch_shape x num_fantasies x (m)
        if self.posterior_max_values.ndim == 2:
            permute_idcs = [-1, *range(ig.ndim - 1)]
        else:
            permute_idcs = [-2, *range(ig.ndim - 2), -1]
        ig = ig.permute(*permute_idcs)  # num_fantasies x batch_shape x (m)
        return ig


class qLowerBoundMaxValueEntropy(DiscreteMaxValueBase):
    r"""The acquisition function for General-purpose Information-Based
    Bayesian Optimisation (GIBBON).

    This acquisition function provides a computationally cheap approximation of
    the mutual information between max values and a batch of candidate points `X`.
    See [Moss2021gibbon]_ for a detailed discussion.

    The model must be single-outcome, unless using a PosteriorTransform.
    q > 1 is supported through greedy batch filling.

    Example:

        >>> model = SingleTaskGP(train_X, train_Y)
        >>> candidate_set = torch.rand(1000, bounds.size(1))
        >>> candidate_set = bounds[0] + (bounds[1] - bounds[0]) * candidate_set
        >>> qGIBBON = qLowerBoundMaxValueEntropy(model, candidate_set)
        >>> candidates, _ = optimize_acqf(qGIBBON, bounds, q=5)
    """

    def _compute_information_gain(self, X: Tensor, mean_M: Tensor, variance_M: Tensor, covar_mM: Tensor) -> Tensor:
        r"""Compute GIBBON's approximation of information gain at the design points `X`.

        When using GIBBON for batch optimization (i.e `q > 1`), we calculate the
        additional information provided by adding a new candidate point to the current
        batch of design points (`X_pending`), rather than calculating the information
        provided by the whole batch. This allows a modest computational saving.

        :param X: A `batch_shape x 1 x d`-dim Tensor of `batch_shape` t-batches
            with `1` `d`-dim design point each.
        :param mean_M: A `batch_shape x 1`-dim Tensor of means.
        :param variance_M: A `batch_shape x 1`-dim Tensor of variances
            consisting of `batch_shape` t-batches with `num_fantasies` fantasies.
        :param covar_mM: A `batch_shape x num_fantasies x (1 + num_trace_observations)`
            -dim Tensor of covariances.

        :returns: A `num_fantasies x batch_shape`-dim Tensor of information gains at the
            given design points `X`.
        """
        # TODO: give the Posterior API an add_observation_noise function to avoid
        # doing posterior computations twice

        # compute the mean_m, variance_m with noisy observation
        posterior_m = self.model.posterior(X, observation_noise=True, posterior_transform=self.posterior_transform)
        mean_m = self.weight * posterior_m.mean.squeeze(-1)
        # batch_shape x 1
        variance_m = posterior_m.variance.clamp_min(CLAMP_LB).squeeze(-1)
        # batch_shape x 1
        check_no_nans(variance_m)

        # get stdv of noiseless variance
        stdv = variance_M.sqrt()
        # batch_shape x 1

        # define normal distribution to compute cdf and pdf
        normal = torch.distributions.Normal(
            torch.zeros(1, device=X.device, dtype=X.dtype),
            torch.ones(1, device=X.device, dtype=X.dtype),
        )

        # prepare max value quantities required by GIBBON
        mvs = torch.transpose(self.posterior_max_values, 0, 1)
        # 1 x s_M
        normalized_mvs = (mvs - mean_m) / stdv
        # batch_shape x s_M

        cdf_mvs = normal.cdf(normalized_mvs).clamp_min(CLAMP_LB)
        pdf_mvs = torch.exp(normal.log_prob(normalized_mvs))
        ratio = pdf_mvs / cdf_mvs
        check_no_nans(ratio)

        # prepare squared correlation between current and target fidelity
        rhos_squared = torch.pow(covar_mM.squeeze(-1), 2) / (variance_m * variance_M)
        # batch_shape x 1
        check_no_nans(rhos_squared)

        # calculate quality contribution to the GIBBON acqusition function
        inner_term = 1 - rhos_squared * ratio * (normalized_mvs + ratio)
        acq = -0.5 * inner_term.clamp_min(CLAMP_LB).log()
        # average over posterior max samples
        acq = acq.mean(dim=1).unsqueeze(0)

        if self.X_pending is None:
            # for q=1, no replusion term required
            return acq

        # for q>1 GIBBON requires repulsion terms r_i, where
        # r_i = log |C_i| for the predictive
        # correlation matricies C_i between each candidate point in X and
        # the m current batch elements in X_pending.

        # Each predictive covariance matrix can be expressed as
        # V_i = [[v_i, A_i], [A_i,B]] for a shared m x m tensor B.
        # So we can efficientely calculate |V_i| using the formula for
        # determinant of block matricies, i.e.
        # |V_i| = (v_i - A_i^T * B^{-1} * A_i) * |B|
        # As the |B| term does not depend on X and we later take its log,
        # it provides only a translation of the acqusition function surface
        # and can thus be ignored.

        if self.posterior_transform is not None:
            raise UnsupportedError(
                "qLowerBoundMaxValueEntropy does not support PosteriorTransforms" "when X_pending is not None."
            )

        X_batches = torch.cat([X, self.X_pending.unsqueeze(0).repeat(X.shape[0], 1, 1)], 1)
        # batch_shape x (1 + m) x d
        # NOTE: This is the blocker for supporting posterior transforms.
        # We would have to process this MVN, applying whatever operations
        # are typically applied for the corresponding posterior, then applying
        # the posterior transform onto the resulting object.
        V = self.model(X_batches)
        # Evaluate terms required for A
        A = V.lazy_covariance_matrix[:, 0, 1:].unsqueeze(1)
        # batch_shape x 1 x m
        # Evaluate terms required for B
        B = self.model.posterior(
            self.X_pending,
            observation_noise=True,
            posterior_transform=self.posterior_transform,
        ).mvn.covariance_matrix.unsqueeze(0)
        # 1 x m x m

        # use determinant of block matrix formula
        V_determinant = variance_m - inv_quad(B, A.transpose(1, 2)).unsqueeze(1)
        # batch_shape x 1

        # Take logs and convert covariances to correlations.
        r = V_determinant.log() - variance_m.log()
        r = 0.5 * r.transpose(0, 1)
        return acq + r


class qMultiFidelityMaxValueEntropy(qMaxValueEntropy):
    r"""Multi-fidelity max-value entropy.

    The acquisition function for multi-fidelity max-value entropy search
    with support for trace observations. See [Takeno2020mfmves]_
    for a detailed discussion of the basic ideas on multi-fidelity MES
    (note that this implementation is somewhat different).

    The model must be single-outcome, unless using a PosteriorTransform.
    The batch case `q > 1` is supported through cyclic optimization and fantasies.

    Example:

        >>> model = SingleTaskGP(train_X, train_Y)
        >>> candidate_set = torch.rand(1000, bounds.size(1))
        >>> candidate_set = bounds[0] + (bounds[1] - bounds[0]) * candidate_set
        >>> MF_MES = qMultiFidelityMaxValueEntropy(model, candidate_set)
        >>> mf_mes = MF_MES(test_X)
    """

    def __init__(
        self,
        model: Model,
        candidate_set: Tensor,
        num_fantasies: int = 16,
        num_mv_samples: int = 10,
        num_y_samples: int = 128,
        posterior_transform: Optional[PosteriorTransform] = None,
        use_gumbel: bool = True,
        maximize: bool = True,
        X_pending: Optional[Tensor] = None,
        cost_aware_utility: Optional[CostAwareUtility] = None,
        project: Callable[[Tensor], Tensor] = lambda X: X,
        expand: Callable[[Tensor], Tensor] = lambda X: X,
        **kwargs: Any,
    ) -> None:
        r"""Single-outcome max-value entropy search acquisition function.

        :param model: A fitted single-outcome model.
        :param candidate_set: A `n x d` Tensor including `n` candidate points to
            discretize the design space, which will be used to sample the
            max values from their posteriors.
        :param cost_aware_utility: A CostAwareUtility computing the cost-transformed
            utility from a candidate set and samples of increases in utility.
        :param num_fantasies: Number of fantasies to generate. The higher this
            number the more accurate the model (at the expense of model
            complexity and performance) and it's only used when `X_pending`
            is not `None`.
        :param num_mv_samples: Number of max value samples.
        :param num_y_samples: Number of posterior samples at specific design point `X`.
        :param posterior_transform: A PosteriorTransform. If using a multi-output model,
            a PosteriorTransform that transforms the multi-output posterior into a
            single-output posterior is required.
        :param use_gumbel: If True, use Gumbel approximation to sample the max values.
        :param maximize: If True, consider the problem a maximization problem.
        :param X_pending: A `m x d`-dim Tensor of `m` design points that have been
            submitted for function evaluation but have not yet been evaluated.
        :param project: A callable mapping a `batch_shape x q x d` tensor of design
            points to a tensor of the same shape projected to the desired
            target set (e.g. the target fidelities in case of multi-fidelity
            optimization).
        :param expand: A callable mapping a `batch_shape x q x d` input tensor to
            a `batch_shape x (q + q_e)' x d`-dim output tensor, where the
            `q_e` additional points in each q-batch correspond to
            additional ("trace") observations.
        """
        super().__init__(
            model=model,
            candidate_set=candidate_set,
            num_fantasies=num_fantasies,
            num_mv_samples=num_mv_samples,
            num_y_samples=num_y_samples,
            posterior_transform=posterior_transform,
            use_gumbel=use_gumbel,
            maximize=maximize,
            X_pending=X_pending,
        )

        if cost_aware_utility is None:
            cost_model = AffineFidelityCostModel(fidelity_weights={-1: 1.0})
            cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)

        self.cost_aware_utility = cost_aware_utility
        self.expand = expand
        self.project = project
        self._cost_sampler = None

        # @TODO make sure fidelity_dims align in project, expand & cost_aware_utility
        # It seems very difficult due to the current way of handling project/expand

        # resample max values after initializing self.project
        # so that the max value samples are at the highest fidelity
        self._sample_max_values(self.num_mv_samples)

    @property
    def cost_sampler(self):
        """
        Sampler used for cost-aware utility evaluation.

        The sampler is lazily deep-copied from the fantasy sampler so cost samples do not mutate the base
        fantasy sampler state when the expanded cost model has a different output or test-point shape.
        """
        if self._cost_sampler is None:
            # Note: Using the deepcopy here is essential. Removing this poses a
            # problem if the base model and the cost model have a different number
            # of outputs or test points (this would be caused by expand), as this
            # would trigger re-sampling the base samples in the fantasy sampler.
            # By cloning the sampler here, the right thing will happen if the
            # the sizes are compatible, if they are not this will result in
            # samples being drawn using different base samples, but it will at
            # least avoid changing state of the fantasy sampler.
            self._cost_sampler = deepcopy(self.fantasies_sampler)
        return self._cost_sampler

    def forward(self, X):
        """
        Helper method to re-shape the tensor when requesting multiple candidate points prior to calculating
        max-value entropy at the design points `X`. Multiple candidate points causes the dimensions of this
        tensor to be `batch_shape x q x d` where q is the number of candidate points. This will break the
        q_forward method as it can only handle `batch_shape x 1 x d`-dim tensors.

        :param X: A `batch_shape x q x d`-dim Tensor of `batch_shape` t-batches
            with `q` candidate points resulting in `q x d`-dim design points each.

        :returns: A `batch_shape`-dim Tensor of MVE values at the given design points `X`.
        """
        if X.shape[-2] == 1:
            return self.q_forward(X)
        else:
            return self.q_forward(X.unsqueeze(-2))

    @t_batch_mode_transform(expected_q=1)
    def q_forward(self, X: Tensor) -> Tensor:
        r"""Evaluates `qMultifidelityMaxValueEntropy` at the design points `X`

        :param X: A `batch_shape x 1 x d`-dim Tensor of `batch_shape` t-batches
            with `1` `d`-dim design point each.

        :returns: A `batch_shape`-dim Tensor of MF-MVES values at the design points `X`.
        """
        X_expand = self.expand(X)  # batch_shape x (1 + num_trace_observations) x d
        X_max_fidelity = self.project(X)  # batch_shape x 1 x d
        X_all = torch.cat((X_expand, X_max_fidelity), dim=-2).unsqueeze(-3)
        # batch_shape x num_fantasies x (2 + num_trace_observations) x d

        # Compute the posterior, posterior mean, variance without noise
        # `_m` and `_M` in the var names means the current and the max fidelity.
        posterior = self.model.posterior(X_all, observation_noise=False, posterior_transform=self.posterior_transform)
        mean_M = self.weight * posterior.mean[..., -1, 0]  # batch_shape x num_fantasies
        variance_M = posterior.variance[..., -1, 0].clamp_min(CLAMP_LB)
        # get the covariance between the low fidelities and max fidelity
        covar_mM = posterior.mvn.covariance_matrix[..., :-1, -1]
        # batch_shape x num_fantasies x (1 + num_trace_observations)

        check_no_nans(mean_M)
        check_no_nans(variance_M)
        check_no_nans(covar_mM)

        # compute the information gain (IG)
        ig = self._compute_information_gain(X=X_expand, mean_M=mean_M, variance_M=variance_M, covar_mM=covar_mM)
        ig = self.cost_aware_utility(X=X, deltas=ig, sampler=self.cost_sampler)
        return ig.mean(dim=0)  # average over the fantasies


class qMultiFidelityLowerBoundMaxValueEntropy(qMultiFidelityMaxValueEntropy):
    r"""Multi-fidelity acquisition function for General-purpose Information-Based
    Bayesian optimization (GIBBON).

    The acquisition function for multi-fidelity max-value entropy search
    with support for trace observations. See [Takeno2020mfmves]_
    for a detailed discussion of the basic ideas on multi-fidelity MES
    (note that this implementation is somewhat different). This acquisition function
    is similar to `qMultiFidelityMaxValueEntropy` but computes the information gain
    from the lower bound described in [Moss2021gibbon].

    The model must be single-outcome, unless using a PosteriorTransform.
    The batch case `q > 1` is supported through cyclic optimization and fantasies.

    Example:

        >>> model = SingleTaskGP(train_X, train_Y)
        >>> candidate_set = torch.rand(1000, bounds.size(1))
        >>> candidate_set = bounds[0] + (bounds[1] - bounds[0]) * candidate_set
        >>> MF_qGIBBON = qMultiFidelityLowerBoundMaxValueEntropy(model, candidate_set)
        >>> mf_gibbon = MF_qGIBBON(test_X)
    """

    def _compute_information_gain(self, X: Tensor, mean_M: Tensor, variance_M: Tensor, covar_mM: Tensor) -> Tensor:
        r"""Compute GIBBON's approximation of information gain at the design points `X`.

        When using GIBBON for batch optimization (i.e `q > 1`), we calculate the
        additional information provided by adding a new candidate point to the current
        batch of design points (`X_pending`), rather than calculating the information
        provided by the whole batch. This allows a modest computational saving.

        :param X: A `batch_shape x 1 x d`-dim Tensor of `batch_shape` t-batches
            with `1` `d`-dim design point each.
        :param mean_M: A `batch_shape x 1`-dim Tensor of means.
        :param variance_M: A `batch_shape x 1`-dim Tensor of variances
            consisting of `batch_shape` t-batches with `num_fantasies` fantasies.
        :param covar_mM: A `batch_shape x num_fantasies x (1 + num_trace_observations)`
            -dim Tensor of covariances.

        :returns: A `num_fantasies x batch_shape`-dim Tensor of information gains at the
            given design points `X`.
        """
        return qLowerBoundMaxValueEntropy._compute_information_gain(
            self, X=X, mean_M=mean_M, variance_M=variance_M, covar_mM=covar_mM
        )


def _sample_max_value_Thompson(
    model: Model,
    candidate_set: Tensor,
    num_samples: int,
    posterior_transform: Optional[PosteriorTransform] = None,
    maximize: bool = True,
) -> Tensor:
    """Samples the max values by discrete Thompson sampling.

    Should generally be called within a `with torch.no_grad()` context.

    :param model: A fitted single-outcome model.
    :param candidate_set: A `n x d` Tensor including `n` candidate points to
        discretize the design space.
    :param num_samples: Number of max value samples.
    :param posterior_transform: A PosteriorTransform. If using a multi-output model,
        a PosteriorTransform that transforms the multi-output posterior into a
        single-output posterior is required.
    :param maximize: If True, consider the problem a maximization problem.

    :returns: A `num_samples x num_fantasies` Tensor of posterior max value samples.
    """
    posterior = model.posterior(candidate_set, posterior_transform=posterior_transform)
    weight = 1.0 if maximize else -1.0
    samples = weight * posterior.rsample(torch.Size([num_samples])).squeeze(-1)
    # samples is num_samples x (num_fantasies) x n
    max_values, _ = samples.max(dim=-1)
    if len(samples.shape) == 2:
        max_values = max_values.unsqueeze(-1)  # num_samples x num_fantasies

    return max_values


def _sample_max_value_Gumbel(
    model: Model,
    candidate_set: Tensor,
    num_samples: int,
    posterior_transform: Optional[PosteriorTransform] = None,
    maximize: bool = True,
) -> Tensor:
    """Samples the max values by Gumbel approximation.

    Should generally be called within a `with torch.no_grad()` context.

    :param model: A fitted single-outcome model.
    :param candidate_set: A `n x d` Tensor including `n` candidate points to
        discretize the design space.
    :param num_samples: Number of max value samples.
    :param posterior_transform: A PosteriorTransform. If using a multi-output model,
        a PosteriorTransform that transforms the multi-output posterior into a
        single-output posterior is required.
    :param maximize: If True, consider the problem a maximization problem.

    :returns: A `num_samples x num_fantasies` Tensor of posterior max value samples.
    """
    # define the approximate CDF for the max value under the independence assumption
    posterior = model.posterior(candidate_set, posterior_transform=posterior_transform)
    weight = 1.0 if maximize else -1.0
    mu = weight * posterior.mean
    sigma = posterior.variance.clamp_min(1e-8).sqrt()
    # mu, sigma is (num_fantasies) X n X 1
    if len(mu.shape) == 3 and mu.shape[-1] == 1:
        mu = mu.squeeze(-1).T
        sigma = sigma.squeeze(-1).T
    # mu, sigma is now n X num_fantasies or n X 1

    # bisect search to find the quantiles 25, 50, 75
    lo = (mu - 3 * sigma).min(dim=0).values
    hi = (mu + 5 * sigma).max(dim=0).values
    num_fantasies = mu.shape[1]
    device = candidate_set.device
    dtype = candidate_set.dtype
    quantiles = torch.zeros(num_fantasies, 3, device=device, dtype=dtype)
    for i in range(num_fantasies):
        lo_, hi_ = lo[i], hi[i]
        N = norm(mu[:, i].cpu().numpy(), sigma[:, i].cpu().numpy())
        quantiles[i, :] = torch.tensor(
            [brentq(lambda y: np.exp(np.sum(N.logcdf(y))) - p, lo_, hi_) for p in [0.25, 0.50, 0.75]]
        )
    q25, q50, q75 = quantiles[:, 0], quantiles[:, 1], quantiles[:, 2]
    # q25, q50, q75 are 1 dimensional tensor with size of either 1 or num_fantasies

    # parameter fitting based on matching percentiles for the Gumbel distribution
    b = (q25 - q75) / (log(log(4.0 / 3.0)) - log(log(4.0)))
    a = q50 + b * log(log(2.0))

    # inverse sampling from the fitted Gumbel CDF distribution
    sample_shape = (num_samples, num_fantasies)
    eps = torch.rand(*sample_shape, device=device, dtype=dtype)
    max_values = a - b * eps.log().mul(-1.0).log()

    return max_values  # num_samples x num_fantasies


class SampledKnowledgeGradient(AcquisitionFunction):
    r"""
    Gradient-free (discrete) knowledge gradient for sampling-based optimizers.

    BoTorch's ``qKnowledgeGradient`` is one-shot: it jointly optimizes the
    candidate batch and one inner maximizer per fantasy with gradients. This
    variant instead estimates

    ``KG(X) = E_fantasy[ max_{x' in S} value_fantasy(x') ] - current_value``

    where ``S`` is a shared discrete inner set (posterior-mean elites, training
    inputs, and Sobol points), so every evaluation needs only one batched
    fantasy model and posterior evaluations. ``refine_scores`` re-scores a
    handful of outer candidates with a full scoped search of each fantasy
    model's value function, which the sampling optimizer calls on its final
    elites.

    The value function is the posterior mean, the MC expected objective
    (``qSimpleRegret``) when a risk objective is given, and optionally the
    ``project``-ed version for multi-fidelity KG. With a ``cost_aware_utility``
    the fantasy improvements are cost-weighted as in
    ``qMultiFidelityKnowledgeGradient``.

    Adapted from BoTorch's knowledge-gradient implementation (MIT license).
    """

    def __init__(
        self,
        model: Model,
        inner_set: Tensor,
        sampler: Any,
        inner_sampler: Any = None,
        current_value: Optional[Tensor] = None,
        objective: Optional[Any] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        cost_aware_utility: Optional[CostAwareUtility] = None,
        project: Optional[Callable[[Tensor], Tensor]] = None,
        inner_bounds: Optional[Tensor] = None,
        inner_search: Optional[Callable[[Callable[[Tensor], Tensor], Tensor], Tensor]] = None,
        eval_batch: int = 64,
        X_pending: Optional[Tensor] = None,
    ):
        """
        :param model: A fitted model with ``fantasize``.
        :param inner_set: ``m x d`` points over which each fantasy's value function is maximized.
        :param sampler: Sampler for fantasy outcomes (its sample count is the number of fantasies).
        :param inner_sampler: Sampler for the MC value function when ``objective`` is given.
        :param current_value: Current best value, subtracted from fantasy maxima.
        :param objective: Optional MC objective (e.g. a risk measure) for the value function.
        :param posterior_transform: Optional posterior transform for the value function.
        :param cost_aware_utility: Optional cost-aware utility (multi-fidelity).
        :param project: Optional projection applied before evaluating the value function (multi-fidelity).
        :param inner_bounds: Bounds for ``refine_scores`` inner searches.
        :param inner_search: Callable ``(value_fn, bounds, num_fantasies) -> num_fantasies`` best values, used by
            ``refine_scores``. ``value_fn`` maps ``num_fantasies x n x d`` points (one search per fantasy) to
            ``num_fantasies x n`` values.
        :param eval_batch: Number of outer t-batches fantasized at once.
        :param X_pending: Optional pending points appended to every evaluated batch.
        """
        super().__init__(model=model)
        self.inner_set = inner_set
        self.sampler = sampler
        self.inner_sampler = inner_sampler
        self.current_value = current_value
        self.objective = objective
        self.posterior_transform = posterior_transform
        self.cost_aware_utility = cost_aware_utility
        self.project = project
        self.inner_bounds = inner_bounds
        self.inner_search = inner_search
        self.eval_batch = max(1, int(eval_batch))
        self.cost_sampler = deepcopy(sampler) if cost_aware_utility is not None else None
        self.set_X_pending(X_pending)

    def _value(self, fantasy_model: Model, X: Tensor) -> Tensor:
        """Value function at ``... x 1 x d`` points of a fantasy model; returns ``...`` values."""
        if self.project is not None:
            X = self.project(X)
        posterior = fantasy_model.posterior(X, posterior_transform=self.posterior_transform)
        if self.objective is None:
            return posterior.mean.squeeze(-1).squeeze(-1)
        samples = self.inner_sampler(posterior)
        return self.objective(samples, X=X).mean(dim=0).squeeze(-1)

    def _fantasy(self, X: Tensor) -> Model:
        if self.X_pending is not None:
            X = torch.cat([X, match_batch_shape(self.X_pending, X)], dim=-2)
        return self.model.fantasize(X=X, sampler=self.sampler, observation_noise=True)

    def _finish(self, X: Tensor, values: Tensor) -> Tensor:
        """Turn ``num_fantasies x b`` fantasy maxima into ``b`` KG values."""
        if self.current_value is not None:
            values = values - self.current_value
        if self.cost_aware_utility is not None:
            values = self.cost_aware_utility(X=X, deltas=values, sampler=self.cost_sampler)
        return values.mean(dim=0)

    def forward(self, X: Tensor) -> Tensor:
        """
        :param X: ``b x q x d`` candidate batches.
        :returns: ``b`` KG estimates.
        """
        out = []
        for start in range(0, X.shape[0], self.eval_batch):
            X_chunk = X[start : start + self.eval_batch]
            fantasy_model = self._fantasy(X_chunk)  # batch: num_fantasies x b
            S = self.inner_set.unsqueeze(-2)  # m x 1 x d
            n_f, b = fantasy_model.batch_shape[0], X_chunk.shape[0]
            S_exp = S.reshape(-1, 1, 1, 1, S.shape[-1]).expand(-1, n_f, b, 1, S.shape[-1])
            values = self._value(fantasy_model, S_exp)  # m x num_fantasies x b
            out.append(self._finish(X_chunk, values.max(dim=0).values))
        return torch.cat(out)

    def refine_scores(self, X: Tensor) -> Tensor:
        """
        Re-score candidates by fully maximizing each fantasy model's value function.

        :param X: ``b x q x d`` candidate batches (``b`` is small).
        :returns: ``b`` refined KG estimates.
        """
        if self.inner_search is None:
            return self.forward(X)
        scores = []
        for index in range(X.shape[0]):
            X_one = X[index : index + 1]
            fantasy_model = self._fantasy(X_one)
            n_f = fantasy_model.batch_shape[0]

            def value_fn(points: Tensor, fantasy_model: Model = fantasy_model) -> Tensor:
                # points: num_fantasies x n x d (one search per fantasy) -> num_fantasies x n values
                expanded = points.transpose(0, 1).unsqueeze(-2).unsqueeze(-2)  # n x num_fantasies x 1 x 1 x d
                return self._value(fantasy_model, expanded).reshape(points.shape[1], -1).T

            best = self.inner_search(value_fn, self.inner_bounds, n_f)  # num_fantasies
            # The discrete inner set is a lower bound; keep the larger of the two maxima.
            S = self.inner_set.unsqueeze(0).expand(n_f, *self.inner_set.shape)
            best = torch.maximum(best, value_fn(S).max(dim=-1).values)
            scores.append(self._finish(X_one, best.reshape(-1, 1)))
        return torch.cat(scores)
