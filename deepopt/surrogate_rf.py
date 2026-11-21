"""
BoTorch-compatible random forest surrogate built on scikit-learn.

The posterior at a ``q``-batch of points is a multivariate normal whose mean is
the average of the individual tree predictions and whose ``q x q`` covariance is
the empirical covariance of the tree predictions across trees (the same
construction DeepOpt uses for NN ensembles). With ``variance_mode='trees+leaf'``
the mean within-leaf variance of the training targets is added on the diagonal
(law of total variance, as in SMAC's random forest).

Random forests are not differentiable, so this surrogate must be used with a
gradient-free acquisition optimizer (``acq_optimizer: sampling`` or
``boltzmann``). Fantasy models are built by refitting the forest on the
augmented data; refits for different fantasies run in parallel threads when
``fantasy_n_jobs > 1``.
"""
from math import prod
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import torch
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor

try:  # gpytorch < 1.9
    from gpytorch.lazy import DiagLazyTensor as _DiagOperator
    from gpytorch.lazy import RootLazyTensor as _RootOperator
except ImportError:  # pragma: no cover - gpytorch >= 1.9 uses linear_operator
    from linear_operator.operators import DiagLinearOperator as _DiagOperator
    from linear_operator.operators import RootLinearOperator as _RootOperator

from deepopt.configuration import ConfigSettings
from deepopt.input_scaling import InputScaler, reject_deprecated_original_scale
from deepopt.output_scaling import OutputScaler

RF_VARIANCE_MODES = ("trees", "trees+leaf")
_JITTER = 1e-6
_FOREST_PARAM_KEYS = ("n_estimators", "max_depth", "min_samples_leaf", "max_features", "bootstrap")


def _forest_params(config: ConfigSettings) -> dict:
    params = {key: config.get_setting(key) for key in _FOREST_PARAM_KEYS}
    params["n_estimators"] = int(params["n_estimators"])
    if params["n_estimators"] < 2:
        raise ValueError("RF n_estimators must be at least 2 to estimate a covariance across trees.")
    return params


def fit_forest(X: np.ndarray, y: np.ndarray, params: dict, seed: int, n_jobs: int = 1) -> RandomForestRegressor:
    """
    Fit a scikit-learn random forest.

    :param X: ``n x d`` float array of training inputs.
    :param y: ``n`` float array of training targets.
    :param params: Forest hyperparameters.
    :param seed: Forest ``random_state``.
    :param n_jobs: Threads used across trees.
    :returns: The fitted forest.
    """
    forest = RandomForestRegressor(random_state=int(seed), n_jobs=n_jobs, **params)
    forest.fit(X, y)
    return forest


def tree_predictions(forest: RandomForestRegressor, X: np.ndarray, with_leaf_variance: bool) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Predictions of each tree of a forest.

    :param forest: A fitted forest.
    :param X: ``n x d`` ``float32`` C-contiguous array.
    :param with_leaf_variance: Also return each tree's within-leaf target variance.
    :returns: ``(T x n predictions, T x n leaf variances or None)``.
    """
    preds = np.empty((len(forest.estimators_), X.shape[0]), dtype=np.float64)
    leaf_var = np.empty_like(preds) if with_leaf_variance else None
    for index, tree in enumerate(forest.estimators_):
        preds[index] = tree.predict(X, check_input=False)
        if with_leaf_variance:
            # For the squared-error criterion, node impurity is the (bootstrap-weighted) target variance.
            leaf_var[index] = tree.tree_.impurity[tree.apply(X, check_input=False)]
    return preds, leaf_var


class RandomForestSurrogate(Model):
    """
    BoTorch-compatible random forest model.

    A model with ``batch_shape`` ``B`` holds ``prod(B)`` independently fitted
    forests (fantasy models are batched this way). Inputs and targets are in
    DeepOpt's scaled model units.
    """

    supports_gradients = False

    def __init__(
        self,
        config: ConfigSettings,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        multi_fidelity: bool = False,
        seed: int = 0,
        device: str = "cpu",
        output_scaler: OutputScaler = None,
        input_scaler: InputScaler = None,
        forests: Optional[List[RandomForestRegressor]] = None,
    ):
        """
        :param config: Settings containing the ``RF_CONFIG`` keys.
        :param X_train: ``batch_shape x n x d`` training inputs in model units.
        :param y_train: ``batch_shape x n x 1`` (or ``n``) scaled training targets.
        :param multi_fidelity: Whether the last input column is a fidelity index.
        :param seed: Random seed for forest fitting.
        :param device: Device for returned tensors.
        :param output_scaler: DeepOpt output scaler used for original-unit predictions.
        :param input_scaler: DeepOpt input scaler used for original-unit queries.
        :param forests: Pre-fitted forests (one per batch element); fitted here when ``None``.
        """
        super().__init__()
        self.config = config
        self.device = device
        self.multi_fidelity = multi_fidelity
        self.seed = int(seed)
        self.input_scaler = input_scaler.to(device) if input_scaler is not None else None
        self.output_scaler = output_scaler.to(device) if output_scaler is not None else None
        self.forest_params = _forest_params(config)
        self.n_jobs = int(config.get_setting("n_jobs"))
        self.fantasy_n_jobs = int(config.get_setting("fantasy_n_jobs"))
        self.variance_mode = config.get_setting("variance_mode")
        if self.variance_mode not in RF_VARIANCE_MODES:
            raise ValueError(f"RF variance_mode must be one of {RF_VARIANCE_MODES}, got {self.variance_mode}.")

        X_train = torch.as_tensor(X_train, dtype=torch.float, device=device).detach()
        y_train = torch.as_tensor(y_train, dtype=torch.float, device=device).detach()
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)
        if y_train.shape[-1] != 1:
            raise ValueError("RF currently supports one scalar output.")
        self.X_train = X_train
        self.y_train = y_train
        self.input_dim = X_train.shape[-1]
        self.output_dim = 1
        self._batch_shape = X_train.shape[:-2]
        self.train_inputs = (X_train,)
        if forests is None:
            n_batch = prod(self._batch_shape)
            X_flat = X_train.reshape(n_batch, *X_train.shape[-2:]).cpu().numpy().astype(np.float32)
            y_flat = y_train.reshape(n_batch, -1).cpu().numpy().astype(np.float64)
            forests = [
                fit_forest(X_flat[index], y_flat[index], self.forest_params, self.seed + index, self.n_jobs)
                for index in range(n_batch)
            ]
        if len(forests) != prod(self._batch_shape):
            raise ValueError("Number of forests must match the model batch shape.")
        self.forests = forests

    @property
    def batch_shape(self) -> torch.Size:
        return self._batch_shape

    @property
    def num_outputs(self) -> int:
        return self.output_dim

    def _set_transformed_inputs(self) -> None:
        if hasattr(self, "input_transform") and not self._has_transformed_inputs:
            if hasattr(self, "train_inputs"):
                self._original_train_inputs = self.train_inputs[0]
                with torch.no_grad():
                    X_tf = self.input_transform.preprocess_transform(self.train_inputs[0])
                self.set_train_data(X_tf, strict=False)
                self._has_transformed_inputs = True

    def set_train_data(self, inputs: torch.Tensor = None, targets: torch.Tensor = None, strict: bool = True) -> None:
        if inputs is not None:
            if torch.is_tensor(inputs):
                inputs = (inputs,)
            self.train_inputs = inputs
        if targets is not None:
            self.train_targets = targets

    def posterior(
        self,
        X: torch.Tensor,
        posterior_transform: Callable[[GPyTorchPosterior], GPyTorchPosterior] = None,
        observation_noise: bool = False,
        **kwargs: Any,
    ) -> GPyTorchPosterior:
        """
        Posterior at ``X`` (``batch_shape x q x d``); ``observation_noise`` is ignored.
        """
        X = self.transform_inputs(X)
        posterior = GPyTorchPosterior(self.forward(X))
        if posterior_transform:
            return posterior_transform(posterior)
        return posterior

    def forward(self, X: torch.Tensor) -> MultivariateNormal:
        """
        Joint normal over the q points with a low-rank-plus-diagonal covariance.

        The covariance is kept lazy (``root @ root.T + diag``), so it is positive
        definite by construction and never forms a dense ``q x q`` matrix unless asked.
        """
        mean, root, diag = self._predict(X)
        covariance = _RootOperator(root) + _DiagOperator(diag + _JITTER)
        return MultivariateNormal(mean, covariance)

    def _dense_covariance(self, root: torch.Tensor, diag: torch.Tensor) -> torch.Tensor:
        cov = root @ root.transpose(-1, -2)
        return cov + torch.diag_embed(diag)

    def _predict(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Mean (``... x q``), covariance root (``... x q x T``), and extra diagonal variance (``... x q``).

        The empirical covariance across the ``T`` trees is ``root @ root.T``, where
        ``root`` holds the centered tree predictions scaled by ``1/sqrt(T-1)``.
        """
        if X.shape[-1] != self.input_dim:
            raise ValueError(f"Expected input_dim {self.input_dim}, found tensor of shape {X.shape}.")
        if X.ndim == 1:
            X = X.unsqueeze(0)
        batch = self._batch_shape
        n_batch = prod(batch)
        q = X.shape[-2]
        full_batch = torch.broadcast_shapes(X.shape[:-2], batch)
        lead = full_batch[: len(full_batch) - len(batch)]
        n_lead = prod(lead)
        X_exp = X.detach().expand(*full_batch, q, self.input_dim).reshape(n_lead, n_batch, q, self.input_dim)
        X_np = np.ascontiguousarray(X_exp.permute(1, 0, 2, 3).reshape(n_batch, n_lead * q, self.input_dim).cpu().numpy(), dtype=np.float32)
        with_leaf = self.variance_mode == "trees+leaf"
        means, roots, diags = [], [], []
        for index, forest in enumerate(self.forests):
            preds, leaf_var = tree_predictions(forest, X_np[index], with_leaf)
            n_trees = preds.shape[0]
            preds = preds.reshape(n_trees, n_lead, q)
            mean = preds.mean(axis=0)
            means.append(mean)
            roots.append(np.moveaxis(preds - mean, 0, -1) / np.sqrt(n_trees - 1))  # n_lead x q x T
            diags.append(leaf_var.reshape(n_trees, n_lead, q).mean(axis=0) if with_leaf else np.zeros_like(mean))
        n_trees = roots[0].shape[-1]
        mean = torch.as_tensor(np.stack(means, axis=1), dtype=X.dtype, device=X.device).reshape(*full_batch, q)
        root = torch.as_tensor(np.stack(roots, axis=1), dtype=X.dtype, device=X.device).reshape(*full_batch, q, n_trees)
        diag = torch.as_tensor(np.stack(diags, axis=1), dtype=X.dtype, device=X.device).reshape(*full_batch, q)
        return mean, root, diag

    def fantasize(self, X: torch.Tensor, sampler: Any, observation_noise: bool = True, **kwargs: Any) -> "RandomForestSurrogate":
        """
        Fantasy model obtained by refitting forests on sampled outcomes at ``X``.

        :param X: ``batch_shape x m x d`` fantasy inputs.
        :param sampler: BoTorch MC sampler producing ``num_fantasies`` outcome samples.
        :param observation_noise: Accepted for BoTorch compatibility; random forests have no noise model.
        :returns: A batched model with batch shape ``num_fantasies x broadcast(X batch, model batch)``.
        """
        kwargs.pop("propagate_grads", None)
        X = X.detach()
        with torch.no_grad():
            Y = sampler(self.posterior(X)).detach()  # n_f x batch x m' x 1
        X_tf = self.transform_inputs(X)
        new_batch = Y.shape[:-2]
        X_new = X_tf.expand(*new_batch, *X_tf.shape[-2:])
        X_all = torch.cat([self.X_train.expand(*new_batch, *self.X_train.shape[-2:]), X_new], dim=-2)
        y_all = torch.cat([self.y_train.expand(*new_batch, *self.y_train.shape[-2:]), Y], dim=-2)
        n_batch = prod(new_batch)
        X_np = X_all.reshape(n_batch, *X_all.shape[-2:]).cpu().numpy().astype(np.float32)
        y_np = y_all.reshape(n_batch, -1).cpu().numpy().astype(np.float64)
        seed = self.seed + 1_000_003
        forests = Parallel(n_jobs=self.fantasy_n_jobs, prefer="threads")(
            delayed(fit_forest)(X_np[index], y_np[index], self.forest_params, seed + index, 1) for index in range(n_batch)
        )
        fantasy = RandomForestSurrogate(
            config=self.config,
            X_train=X_all,
            y_train=y_all,
            multi_fidelity=self.multi_fidelity,
            seed=seed,
            device=self.device,
            output_scaler=self.output_scaler,
            input_scaler=self.input_scaler,
            forests=forests,
        )
        if hasattr(self, "input_transform"):
            fantasy.input_transform = self.input_transform
        return fantasy.eval()

    def get_prediction_with_uncertainty(
        self,
        q: torch.Tensor,
        get_cov: bool = False,
        *args: Any,
        original_scale_x: bool = True,
        original_scale_y: bool = True,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forest mean and variance (or ``q x q`` covariance) at query points.

        :param q: Query tensor with shape ``... x q x d``.
        :param get_cov: If ``True``, return the joint covariance; otherwise return pointwise variance.
        :param original_scale_x: If ``True``, transform queries from original units to model units.
        :param original_scale_y: If ``True``, return statistics in original output units.
        :returns: ``(mean, variance)`` with shapes ``... x q x 1``, or ``(mean, covariance)`` with shapes
            ``... x q`` and ``... x q x q``.
        """
        reject_deprecated_original_scale(args, kwargs)
        q = torch.as_tensor(q, dtype=torch.float, device=self.device)
        if original_scale_x:
            if self.input_scaler is None:
                raise RuntimeError("original_scale_x=True requires this model to have an input_scaler.")
            q = self.input_scaler.transform(q)
        mean, root, diag = self._predict(q)
        variance = ((root**2).sum(-1) + diag).unsqueeze(-1)
        cov = self._dense_covariance(root, diag) if get_cov else None
        mean = mean.unsqueeze(-1)
        if original_scale_y:
            if self.output_scaler is None:
                raise RuntimeError("original_scale_y=True requires this model to have an output_scaler.")
            if get_cov:
                cov = self.output_scaler.inverse_covariance(cov, q)
            mean = self.output_scaler.inverse_transform(mean, q)
            variance = self.output_scaler.inverse_variance(variance, q)
        if get_cov:
            return mean.squeeze(-1), cov
        return mean, variance
