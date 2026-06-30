"""
This module contains the Delta UQ models used for single-fidelity and multi-fidelity
neural networks.
"""
import os
import warnings
from copy import copy
from typing import Any, Callable, Dict, Tuple, Type, Union

import numpy as np
import torch
from botorch import settings
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.sampling.samplers import MCSampler
from gpytorch.distributions import MultivariateNormal
from torch import nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, TensorDataset

from deepopt.configuration import ConfigSettings
from deepopt.output_scaling import OutputScaler
from deepopt.surrogate_utils import MLP as Arch
from deepopt.surrogate_utils import create_optimizer

device = "cuda" if torch.cuda.is_available() else "cpu"


class DeltaEnc(Model):
    """
    The `DeltaEnc` class represents the single-fidelity Delta UQ model for neural
    networks. This model will allow us to set the training data, fit it, and get
    our prediciton values with uncertainty.
    """

    def __init__(
        self,
        network: Arch,
        config: ConfigSettings,
        optimizer: Union[Adam, SGD],
        X_train: np.ndarray,
        y_train: np.ndarray,
        target: str = "dy",  # 'y', 'dy'
        multi_fidelity: bool = False,
        output_scaler: OutputScaler = None,
    ):
        """
        Initialize an instance of the `DeltaEnc` model for further processing.

        :param network: A neural network module
        :param config: Configuration settings provided by the user
        :param optimizer: An optimizer object from torch to help train the neural network
        :param X_train: The full input dataset
        :param y_train: The full output array
        :param target: Whether to fit the neural network with the y that pairs with the x or
            to the difference y-Y. Options for this configuration are `y` and `dy`
        """
        super().__init__()
        if isinstance(network, list):
            self.multi_network = True
            self.n_epochs = [cf.get_setting("n_epochs") for cf in config]
            self.actual_batch_size = [min(cf.get_setting("batch_size"), len(X_train)) for cf in config]
        else:
            self.multi_network = False
            self.n_epochs = config.get_setting("n_epochs")
            self.actual_batch_size = min(config.get_setting("batch_size"), len(X_train))

        self.f_predictor = network
        self.f_optimizer = optimizer
        self.config = config
        self.device = network.device  # might not work for multi-networks
        self.multi_fidelity = multi_fidelity

        X_train = X_train.float()
        y_train = y_train.float()

        self.X_train = X_train
        self.y_train = y_train

        self.input_dim = X_train.shape[-1]
        self.output_dim = y_train.shape[-1]

        self.is_fantasy_model = False

        batch_shape = X_train.shape[:-2]
        self._batch_shape = batch_shape
        self.n_train = X_train.shape[-2]

        if output_scaler is None:
            output_scaler = OutputScaler(
                multi_fidelity=self.multi_fidelity,
                num_fidelities=int(X_train[..., -1].max().item()) + 1 if self.multi_fidelity else 1,
                fidelity_dim=self.input_dim - 1,
            ).fit(y_train, X_train)
        self.output_scaler = output_scaler.to(self.device)
        self.y_min = self.output_scaler.y_min
        self.y_max = self.output_scaler.y_max
        self.out_scaler = lambda y, y_min, y_max: (y - y_min) / torch.where((y_max - y_min).abs() > 0, y_max - y_min, torch.ones_like(y_max - y_min))
        self.out_scaler_inv = lambda y, y_min, y_max: y * torch.where((y_max - y_min).abs() > 0, y_max - y_min, torch.ones_like(y_max - y_min)) + y_min

        if self.multi_fidelity:
            self.train_fid_locs = [X_train[..., -1] == i for i in X_train[..., -1].unique()]

        self.X_train_scaled = X_train.clone()
        self.y_train_scaled = self.output_scaler.transform(y_train.clone(), X_train)

        self.X_train_nn = self.X_train_scaled.moveaxis(-2, 0).reshape(self.n_train, -1)
        self.y_train_nn = self.y_train_scaled.moveaxis(-2, 0).reshape(self.n_train, -1)

        self.nn_input_dim = self.X_train_nn.shape[1]
        self.nn_output_dim = self.y_train_nn.shape[1]

        self.loss_fn = nn.MSELoss()

        self.target = target

        self.train_inputs = self.X_train
        if self.train_inputs is not None and torch.is_tensor(self.train_inputs):
            self.train_inputs = (self.train_inputs,)

    @property
    def batch_shape(self):
        return self._batch_shape

    @batch_shape.setter
    def batch_shape(self, value):
        self._batch_shape = value

    @property
    def num_outputs(self):
        return self.output_dim

    # coped from model.py
    # Originally, this method accessed the first dim of self.train_inputs (i.e self.train_inputs[0])
    # because the inputs are assumed to be in batches.
    def _set_transformed_inputs(self) -> None:
        r"""Update training inputs with transformed inputs."""
        if hasattr(self, "input_transform") and not self._has_transformed_inputs:
            if hasattr(self, "train_inputs"):
                self._original_train_inputs = self.train_inputs[0]
                with torch.no_grad():
                    X_tf = self.input_transform.preprocess_transform(self.train_inputs[0])
                self.set_train_data(X_tf, strict=False)
                self._has_transformed_inputs = True
            else:
                warnings.warn(
                    "Could not update `train_inputs` with transformed inputs "
                    f"since {self.__class__.__name__} does not have a `train_inputs` "
                    "attribute. Make sure that the `input_transform` is applied to "
                    "both the train inputs and test inputs.",
                    RuntimeWarning,
                )

    # copied from exact_gp.py
    # This needs to be defined if you are passing in an input transformation.
    def set_train_data(
        self,
        inputs: torch.Tensor = None,
        targets: torch.Tensor = None,
        strict: bool = True,
    ):
        """
        Set training data (does not re-fit model hyper-parameters).

        :param inputs: The new training inputs.
        :param targets: The new training targets.
        :param strict: If `True`, the new inputs and targets must have the same
            shape, dtype, and device as the current inputs and targets. Otherwise,
            any shape/dtype/device are allowed.
        """
        if inputs is not None:
            if torch.is_tensor(inputs):
                inputs = (inputs,)
            inputs = tuple(input_.unsqueeze(-1) if input_.ndimension() == 1 else input_ for input_ in inputs)
            if strict:
                for input_, t_input in zip(inputs, self.train_inputs or (None,)):
                    for attr in ("shape", "dtype", "device"):
                        expected_attr = getattr(t_input, attr, None)
                        found_attr = getattr(input_, attr, None)
                        if expected_attr != found_attr:
                            msg = "Cannot modify {attr} of inputs (expected {e_attr}, found {f_attr})."
                            msg = msg.format(attr=attr, e_attr=expected_attr, f_attr=found_attr)
                            raise RuntimeError(msg)
            self.train_inputs = inputs
            if self.train_inputs is not None and torch.is_tensor(self.train_inputs):
                self.train_inputs = (self.train_inputs,)
        if targets is not None:
            if strict:
                for attr in ("shape", "dtype", "device"):
                    expected_attr = getattr(self.train_targets, attr, None)
                    found_attr = getattr(targets, attr, None)
                    if expected_attr != found_attr:
                        msg = "Cannot modify {attr} of targets (expected {e_attr}, found {f_attr})."
                        msg = msg.format(attr=attr, e_attr=expected_attr, f_attr=found_attr)
                        raise RuntimeError(msg)
            self.train_targets = targets
        self.prediction_strategy = None

    def fit(self):
        """
        Train the model. The results of this process will eventually be pulled
        from the predictor object and the optimizer object (either an `Adam` object
        or an `SGD` object from torch).
        """
        if self.multi_network:
            pass
        else:
            data = TensorDataset(self.X_train_nn, self.y_train_nn)
            loader = DataLoader(data, shuffle=True, batch_size=self.actual_batch_size)
            self.f_predictor.train()
            for _ in range(self.n_epochs):
                avg_loss = 0.0

                for _, (xi, yi) in enumerate(loader):
                    xi = xi.to(self.device)
                    yi = yi.to(self.device)

                    xi = self.f_predictor.input_mapping(xi)
                    flipped_x = torch.flip(xi, [0])
                    diff_x = xi - flipped_x
                    inp = torch.cat([flipped_x, diff_x], axis=1)

                    if self.target == "y":
                        out = yi
                    else:
                        flipped_y = torch.flip(yi, [0])
                        diff_y = yi - flipped_y
                        out = diff_y

                    out_hat = self.f_predictor(inp)
                    self.f_optimizer.zero_grad()
                    f_loss = self.loss_fn(out_hat.float(), out.float())
                    f_loss.backward()
                    self.f_optimizer.step()
                    avg_loss += f_loss.item() / len(loader)

    def save_ckpt(self, path: str, name: str, checkpoint_metadata: Dict[str, Any] = None):
        """
        Save a trained model to a checkpoint file

        :param path: The path to the checkpoint file
        :param name: The name of the checkpoint file
        """
        state = {"epoch": self.n_epochs}
        state["state_dict"] = self.f_predictor.state_dict()
        state["B"] = self.f_predictor.B
        state["opt_state_dict"] = self.f_optimizer.state_dict()
        state["output_scaler"] = self.output_scaler.state_dict()
        if checkpoint_metadata is not None:
            state["deepopt_checkpoint"] = checkpoint_metadata
        filename = path + "/" + name + ".ckpt"
        torch.save(state, filename)
        print("Saved Ckpts")

    def load_ckpt(self, path: str, name: str):
        """
        Load in a trained model from a checkpoint file

        :param path: The path to the checkpoint file
        :param name: The name of the checkpoint file
        """
        saved_state = torch.load(os.path.join(path, name + ".ckpt"), map_location=self.device)
        if "output_scaler" in saved_state:
            self.output_scaler = OutputScaler.from_state_dict(saved_state["output_scaler"], device=self.device)
            self.y_min = self.output_scaler.y_min
            self.y_max = self.output_scaler.y_max
        else:
            warnings.warn(
                "DeltaEnc checkpoint has no output_scaler state; using scaler fit from the current data file.",
                RuntimeWarning,
            )
        self.y_train_scaled = self.output_scaler.transform(self.y_train.clone(), self.X_train)
        self.y_train_nn = self.y_train_scaled.moveaxis(-2, 0).reshape(self.n_train, -1)
        self.f_predictor.load_state_dict(saved_state["state_dict"])
        self.f_predictor.B = saved_state["B"]

    def _map_delta_model(self, ref: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        """
        Maps the delta model based on the input reference (ref) and query tensors

        :param ref: The input reference tensor
        :param query: The query tensor

        :returns: The predicted output tensor
        """
        ref = self.f_predictor.input_mapping(ref)
        query = self.f_predictor.input_mapping(query)
        diff = query - ref
        samps = torch.cat([ref, diff], 1)
        pred = self.f_predictor(samps)
        return pred

    def posterior(
        self,
        X: torch.Tensor,
        posterior_transform: Callable[[GPyTorchPosterior], GPyTorchPosterior] = None,
        # observation_noise: bool = False,
        **kwargs,
    ) -> GPyTorchPosterior:
        """
        Computes a posterior based on GPyTorch's multi-variate Normal distributions.
        Posterior transformation is done if a `posterior_transform` function is provided.

        :param X: A batch_shape x q x d-dim Tensor, where d is the dimension of the feature
            space and q is the number of points considered jointly.
        :param posterior_transform: An optional function to transform the computed posterior
            before returning

        :returns: A GPyTorchPosterior object with information on the posterior we calculated
        """
        # Transformations are applied at evaluation time.
        # An acquisiton's objective funtion will call
        # the model's posterior.
        X = self.transform_inputs(X)
        mvn = self.forward(X, **kwargs)
        if posterior_transform:
            return posterior_transform(GPyTorchPosterior(mvn))
        return GPyTorchPosterior(mvn)

    def forward(self, X: torch.Tensor, **kwargs) -> MultivariateNormal:
        """
        Compute the model output at X with uncertainties, then use that to
        compute a multivariate normal.

        :param X: A batch_shape x q x d-dim Tensor, where d is the dimension of the feature
            space and q is the number of points considered jointly.

        :returns: A multivariate normal object computed using `X`
        """
        use_variances = kwargs.get("use_variances")
        if any([use_variances is None, use_variances is False]):
            means, covs = self.get_prediction_with_uncertainty(X, get_cov=True, original_scale=False, **kwargs)
            try:
                return MultivariateNormal(means, covs + 1e-6 * torch.eye(covs.shape[-1]))
            except Exception as exc1:
                print(exc1)
                print("Trying with stronger regularization (1e-5)")
                try:
                    return MultivariateNormal(means, covs + 1e-5 * torch.eye(covs.shape[-1]))
                except Exception as exc2:
                    print(exc2)
                    print("Trying with even stronger regularization (1e-4)")
                    try:
                        return MultivariateNormal(means, covs + 1e-4 * torch.eye(covs.shape[-1]))
                    except Exception as exc3:
                        print(exc3)
                        print("Trying with yet stronger regularization (1e-3)")
                        return MultivariateNormal(means, covs + 1e-3 * torch.eye(covs.shape[-1]))

        else:
            means, variances = self.get_prediction_with_uncertainty(X, **kwargs)
            if means.ndim in (1, 2):
                means_squeeze, variances_squeeze = means.squeeze(), variances.squeeze()
                if means_squeeze.ndim == 0:
                    means_squeeze = torch.Tensor([means_squeeze])
                if variances_squeeze.ndim == 0:
                    variances_squeeze = torch.Tensor([variances_squeeze])
                mvn = MultivariateNormal(means_squeeze, torch.diag(variances_squeeze + 1e-6))
            else:
                covar_diag = variances.squeeze(-1) + 1e-6
                covars = torch.zeros(*covar_diag.shape, covar_diag.shape[-1])
                for i in range(covar_diag.shape[-1]):
                    covars[..., i, i] = covar_diag[..., i]
                mvn = MultivariateNormal(means.squeeze(-1), covars)

            return mvn

    def get_prediction_with_uncertainty(
        self,
        q: torch.Tensor,
        get_cov: bool = False,
        original_scale: bool = True,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given a tensor calculate the prediction with uncertainty.

        :param q: A tensor with data we'll calculate prediction with uncertainty for
        :param get_cov: If True, get the covariance. Otherwise, get the variance.
        :param original_scale: If True, apply an inverse scaling transformation to get the scaled
            predictions back to the original scale. Otherwise, don't re-scale the predictions.

        :returns: A tuple containing the mean tensor and the variance (or covariance if
            `get_cov=True`) tensor
        """
        orig_input_shape = q.shape
        assert (
            q.shape[-1] == self.input_dim
        ), f"Expected tensor to have size=input_dim ({self.input_dim}) in last dimension, found tensor of shape {q.shape}"
        if q.shape[-len(self.batch_shape) - 2 : -2] != self.batch_shape:  # noqa: E203
            try:
                print(f"Need to expand input shape {orig_input_shape} to match training batch shape {self.batch_shape}.")
                if q.shape[-len(self.batch_shape) - 2 : -2] == torch.Size(len(self.batch_shape) * [1]):  # noqa: E203
                    q = q.expand(
                        *q.shape[: -len(self.batch_shape) - 2],
                        *self.batch_shape,
                        *q.shape[-2:],
                    )
                else:
                    q = q.expand(*self.batch_shape, *q.shape)
                    for _ in range(len(self.batch_shape)):
                        q = q.moveaxis(0, -3)
            except Exception as e:
                print(f"Could not make tensor of shape {orig_input_shape} match training batch shape {self.batch_shape}.")
                print(e)
        assert (
            q.shape[-len(self.batch_shape) - 2 : -2] == self.batch_shape  # noqa: E203
        ), f"Expected tensor to have batch shape matching training batch shape {self.batch_shape}, instead found "
        f"tensor of shape {q.shape}."
        input_shape = q.shape
        q_move = q.moveaxis(-2, 0)
        samples_shape = q_move.shape[: -len(self.batch_shape) - 1]

        n_ref = 100
        ref_choice = torch.randint(self.X_train_nn.shape[0], (n_ref * int(np.prod(samples_shape[1:])),))
        ref = (
            self.X_train_nn[ref_choice]
            .reshape(n_ref, 1, *samples_shape[1:], self.X_train_nn.shape[-1])
            .expand(n_ref, *samples_shape, self.X_train_nn.shape[-1])
            .reshape(-1, self.X_train_nn.shape[-1])
        )
        ref_y = (
            self.y_train_nn[ref_choice]
            .reshape(n_ref, 1, *samples_shape[1:], self.y_train_nn.shape[-1])
            .expand(n_ref, *samples_shape, self.y_train_nn.shape[-1])
            .reshape(-1, self.y_train_nn.shape[-1])
        )

        q_combine_samples = q_move.expand(n_ref, *q_move.shape).reshape(-1, *self.batch_shape, self.input_dim)
        q_reshape = q_combine_samples.reshape(q_combine_samples.shape[0], -1)
        self.f_predictor.eval()
        val = self._map_delta_model(ref, q_reshape.float())
        if self.target != "y":
            val += ref_y

        val = val.reshape(n_ref, *samples_shape, *self.batch_shape, self.output_dim).moveaxis(1, -2)
        assert val.shape[1:-1] == input_shape[:-1], "Something went wrong with reshaping."

        if original_scale:
            all_preds = self.output_scaler.inverse_transform(val, q)
        else:
            all_preds = val

        mu = all_preds.mean(axis=0)
        var = all_preds.var(axis=0)
        if get_cov:
            assert self.output_dim == 1, "Output must be 1-dimensional to compute covariances."
            all_preds = all_preds.squeeze(-1)
            mu = mu.squeeze(-1)
            var = var.squeeze(-1)
            del_pred = all_preds - all_preds.mean(axis=0)
            cov = torch.einsum("i...A,i...B->...AB", del_pred, del_pred) / (n_ref - 1)
            cov = 0.5 * (cov + cov.transpose(-1, -2))
            return mu, cov

        return mu, var

    def fantasize(
        self,
        X: torch.Tensor,
        sampler: Type[MCSampler],
        # observation_noise: bool = True,  # TODO uncomment this if we implement it
        **kwargs,
    ) -> "DeltaEnc":
        """
        Augment the dataset and return a new model with the fantasized points.

        :param X: The input tensor to augment
        :param sampler: A botorch sampling class to apply to the posterior of the input deck

        :returns: A new `DeltaEnc` model with the augmented dataset
        """
        propagate_grads = kwargs.pop("propagate_grads", False)
        with settings.propagate_grads(propagate_grads):
            # TODO uncomment this if we implement observation_noise
            # post_X = self.posterior(X,observation_noise=observation_noise,**kwargs)
            post_X = self.posterior(X, **kwargs)
            Y_fantasized = sampler(post_X)

        Y_fantasized = self.output_scaler.inverse_transform(Y_fantasized.detach().clone(), X)
        num_fantasies = Y_fantasized.shape[0]
        X_clone = X.detach().clone()

        if hasattr(self, "input_transform") and self.input_transform is not None:
            X_clone = self.transform_inputs(X_clone)
            X_train_orig = self.transform_inputs(self.X_train)
            y_train_orig = self.y_train.tile((X_train_orig.shape[-2] // self.X_train.shape[-2], 1))
        else:
            X_train_orig = self.X_train
            y_train_orig = self.y_train

        X_train_new = X_clone.expand(num_fantasies, *X_clone.shape)
        X_train = torch.cat(
            [
                X_train_orig.expand(*X_train_new.shape[:-2], *X_train_orig.shape[-2:]),
                X_train_new,
            ],
            axis=-2,
        )
        Y_train = torch.cat(
            [
                y_train_orig.expand(*Y_fantasized.shape[:-2], *y_train_orig.shape[-2:]),
                Y_fantasized,
            ],
            axis=-2,
        )

        in_size = X_train.moveaxis(-2, 0).reshape(X_train.shape[-2], -1).shape[-1]
        out_size = Y_train.moveaxis(-2, 0).reshape(Y_train.shape[-2], -1).shape[-1]

        config_fantasy = copy(self.config)
        config_fantasy.set_setting("n_epochs", 20)
        # config_fantasy = {key: self.config[key] for key in self.config}
        # config_fantasy["n_epochs"] = 20

        with torch.enable_grad():
            network = Arch(
                config=self.config,
                unc_type="deltaenc",
                input_dim=in_size,
                output_dim=out_size,
                device=self.device,
            )
            opt = create_optimizer(network, self.config)
            fantasy_model = DeltaEnc(
                network=network,
                config=config_fantasy,
                optimizer=opt,
                X_train=X_train,
                y_train=Y_train,
                target=self.target,
                multi_fidelity=self.multi_fidelity,
                output_scaler=self.output_scaler,
            )
            if hasattr(self, "input_transform"):
                fantasy_model.input_transform = self.input_transform

            state_dict_prev = self.f_predictor.state_dict()
            state_dict_new = fantasy_model.f_predictor.state_dict()
            state_dict_new = {
                key_new: state_dict_prev[key_prev].expand(state_dict_new[key_new].shape).detach().clone()
                for (key_new, key_prev) in zip(state_dict_new, state_dict_prev)
            }

            fantasy_model.f_predictor.load_state_dict(state_dict_new)
            fantasy_model.f_predictor.B = self.f_predictor.B.tile(
                (1, fantasy_model.f_predictor.B.shape[1] // self.f_predictor.B.shape[1])
            )
            fantasy_model.fit()
            fantasy_model.eval()
            fantasy_model.is_fantasy_model = True
        return fantasy_model
