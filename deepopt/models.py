"""
This module contains the high-level functionality of the DeepOpt library.
It's where we handle the configuration of each model (i.e. how we process
the `learn` and `optimize` commands for each model).
"""
import json
import random
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from os import getcwd
from os.path import basename, dirname, join
from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np
import psutil
import ray
import torch
from botorch import fit_gpytorch_model
from botorch.acquisition import PosteriorMean, qExpectedImprovement, qNoisyExpectedImprovement
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.acquisition.utils import project_to_target_fidelity
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient, qMultiFidelityKnowledgeGradient
from botorch.acquisition.objective import ExpectationPosteriorTransform
from botorch.acquisition.risk_measures import CVaR, RiskMeasureMCObjective, VaR
from botorch.models.deterministic import DeterministicModel
from botorch.models.gp_regression_fidelity import SingleTaskGP, SingleTaskMultiFidelityGP
from botorch.models.model import Model
from botorch.models.transforms.input import InputPerturbation
from botorch.optim.optimize import optimize_acqf, optimize_acqf_mixed
from botorch.sampling.qmc import MultivariateNormalQMCEngine
from botorch.sampling.samplers import SobolQMCNormalSampler
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from ray import tune
from ray.air.config import RunConfig
from ray.tune.schedulers import ASHAScheduler
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset

from deepopt.acquisition import qMaxValueEntropy, qMultiFidelityLowerBoundMaxValueEntropy, qMultiFidelityMaxValueEntropy
from deepopt.configuration import ConfigSettings
from deepopt.defaults import Defaults
from deepopt.deltaenc import DeltaEnc
from deepopt.nn_ensemble import NNEnsemble
from deepopt.output_scaling import OutputScaler, StandardizeOutputScaler
from deepopt.surrogate_utils import MLP as Arch
from deepopt.surrogate_utils import create_optimizer


DEEPOPT_CHECKPOINT_KEY = "deepopt_checkpoint"
DEEPOPT_CHECKPOINT_SCHEMA_VERSION = 1


def _torch_load(learner_file: str, map_location: str = "cpu", weights_only: bool = False):
    try:
        return torch.load(learner_file, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(learner_file, map_location=map_location)


def get_checkpoint_metadata(learner_file: str, map_location: str = "cpu") -> Optional[Dict[str, Any]]:
    try:
        checkpoint = _torch_load(learner_file, map_location=map_location, weights_only=True)
    except Exception:
        return None
    if not isinstance(checkpoint, dict) or DEEPOPT_CHECKPOINT_KEY not in checkpoint:
        return None
    metadata = checkpoint[DEEPOPT_CHECKPOINT_KEY]
    if not isinstance(metadata, dict):
        raise ValueError("DeepOpt checkpoint metadata is malformed.")
    schema_version = metadata.get("schema_version")
    if schema_version != DEEPOPT_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported DeepOpt checkpoint schema version {schema_version}. "
            f"This DeepOpt version supports schema version {DEEPOPT_CHECKPOINT_SCHEMA_VERSION}."
        )
    model_type = metadata.get("model_type")
    if model_type not in {"GP", "delUQ", "nnEnsemble"}:
        raise ValueError(f"DeepOpt checkpoint metadata has invalid model_type {model_type}.")
    required_fields = {"training_data", "bounds", "config_settings"}
    missing_fields = required_fields.difference(metadata)
    if missing_fields:
        missing = ", ".join(sorted(missing_fields))
        raise ValueError(f"DeepOpt checkpoint metadata is missing required field(s): {missing}.")
    training_data = metadata["training_data"]
    if not isinstance(training_data, dict) or "X" not in training_data or "y" not in training_data:
        raise ValueError("DeepOpt checkpoint metadata training_data must contain X and y.")
    return metadata


def is_self_describing_checkpoint(learner_file: str) -> bool:
    return get_checkpoint_metadata(learner_file) is not None


def _config_settings_from_checkpoint(metadata: Dict[str, Any]) -> ConfigSettings:
    config_settings = ConfigSettings(metadata["model_type"])
    config_settings.config_settings.update(metadata.get("config_settings", {}))
    config_settings.config_file = None
    return config_settings


def load_deepopt_wrapper(learner_file: str, device: str = "auto", verbose: bool = False) -> "DeepoptBaseModel":
    metadata = get_checkpoint_metadata(learner_file)
    if metadata is None:
        raise ValueError(
            "Checkpoint does not contain DeepOpt self-describing metadata. Load it using the legacy explicit path: "
            "construct the appropriate GPModel, DelUQModel, or NNEnsembleModel with data_file, bounds, "
            "config_settings, and multi_fidelity, then call load_model(...)."
        )
    model_classes = {"GP": GPModel, "delUQ": DelUQModel, "nnEnsemble": NNEnsembleModel}
    config_settings = _config_settings_from_checkpoint(metadata)
    return model_classes[metadata["model_type"]](
        config_settings=config_settings,
        data_file=metadata.get("data_file"),
        training_data=metadata["training_data"],
        bounds=metadata["bounds"],
        multi_fidelity=metadata.get("multi_fidelity", Defaults.multi_fidelity),
        random_seed=metadata.get("random_seed", Defaults.random_seed),
        k_folds=metadata.get("k_folds", Defaults.k_folds),
        target=metadata.get("target", "dy"),
        device=device,
        verbose=verbose,
    )


def load_deepopt_model(learner_file: str, device: str = "auto", verbose: bool = False) -> Type[Model]:
    wrapper = load_deepopt_wrapper(learner_file, device=device, verbose=verbose)
    return wrapper.load_model(learner_file)


class FidelityCostModel(DeterministicModel):
    """
    The cost model for multi-fidelity runs.
    """

    def __init__(self, fidelity_weights: np.ndarray):
        """
        Initialize the fidelity cost model with the weights for different fidelities.

        :param fidelity_weights: An ndarray of weight values for different fidelities
        """
        super().__init__()
        self._num_outputs = 1
        self.fidelity_weights = torch.tensor(fidelity_weights,dtype=torch.float)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute the fidelity cost based on the provided input tensor.

        Given an input tensor X, extract the last element along the last dimension, round it to the nearest integer,
        and use this value as an index to retrieve the corresponding fidelity weight from the pre-defined
        fidelity weights tensor. Return the retrieved weight as a tensor with an additional dimension.

        :param X: The input tensor representing the data for the computation.

        :returns: A tensor containing the fidelity weight for the provided input, expanded to have an additional dimension.
        """
        fidelity_weights = self.fidelity_weights.to(X.device)
        return fidelity_weights[X[..., -1].round().long()].unsqueeze(-1)


@dataclass
class DeepoptBaseModel(ABC):
    """
    The base model that our other models will inherit from. This class will handle
    construction of model instances.

    :cvar data_file: A .npz or .npy file containing the data to use as input
    :cvar bounds: Reasonable limits on where to do your optimization search
    :cvar config_settings: A ConfigSettings object with all of our configuration settings
    :cvar random_seed: The random seed to use when training and optimizing
    :cvar multi_fidelity: True if we're doing a multi-fidelity run, False otherwise
    :cvar num_fidelities: The number of fidelities to use if we're doing a
        multi-fidelity run. `Default: None`
    :cvar kfolds: The number of kfolds to use when training a delUQ surrogate.
        `Default: None`
    :cvar full_train_X: The full input dataset. This is read in from `data_file`.
        `Default: None`
    :cvar full_train_Y: The full output dataset. This is read in from `data_file`.
        `Default: None`
    :cvar input_dim: The dimensions of `full_train_X`. `Default: None`
    :cvar output_dim: The dimensions of `full_train_Y`. `Default: None`
    :cvar config: The configuration options read in from `config_file`. `Default: None`
    :cvar device: The device to run on. This option is read in from `config_file`.
        Options for this configuration are `cpu` and `gpu`. `Default: None`
    :cvar target: Whether to fit the neural network with the y that pairs with the x or
        to the difference y-Y. This option is read in from `config_file`. Options for this
        configuration are `y`, `dy`, and `None`. `Default: None`
    :cvar target_fidelities: Explicitly states our target (highest) fidelity. This is
        saved in a dict format since it's necessary for BoTorch. `Default: None`
    """

    data_file: str = None
    bounds: np.ndarray = None
    config_settings: ConfigSettings = None
    random_seed: int = Defaults.random_seed
    multi_fidelity: bool = Defaults.multi_fidelity
    num_fidelities: int = None
    k_folds: int = Defaults.k_folds
    full_train_X: np.ndarray = None
    full_train_Y: np.ndarray = None
    input_dim: int = None
    output_dim: int = None
    config: Dict[str, Any] = None
    device: str = "auto"
    target: str = "dy"
    target_fidelities: Dict[int, float] = None
    verbose: bool = False
    training_data: Dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.training_data is None:
            try:
                input_data = np.load(self.data_file)
                self.X_orig = torch.from_numpy(input_data["X"]).float()
                self.Y_orig = torch.from_numpy(input_data["y"]).float()
            except ValueError:
                input_data = np.load(self.data_file,allow_pickle=True)
                self.X_orig = torch.from_numpy(input_data["X"].astype(np.float32))
                self.Y_orig = torch.from_numpy(input_data["y"].astype(np.float32))
        else:
            self.X_orig = torch.as_tensor(self.training_data["X"]).float()
            self.Y_orig = torch.as_tensor(self.training_data["y"]).float()
        if len(self.Y_orig.shape) == 1:
            self.Y_orig = self.Y_orig.reshape(-1, 1)
        bounds = torch.as_tensor(self.bounds, dtype=torch.float)
        self.full_train_X = (self.X_orig - bounds[0]) / (bounds[1] - bounds[0])  # both models
        if self.multi_fidelity:
            self.full_train_X[:, -1] = self.X_orig[:, -1].round()
            self.num_fidelities = int(bounds[1, -1]) + 1
        else:
            self.num_fidelities = 1
            
        if self.device=='auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        elif self.device=='cuda' or self.device=='gpu':
            if torch.cuda.is_available():
                self.device = 'cuda'
            else:
                print('No GPU available, setting device to CPU.')
                self.device = 'cpu'

        self.full_train_X = self.full_train_X.to(self.device)
        self.full_train_Y = self.Y_orig.clone().to(self.device)
        self.bounds = bounds.to(self.device)

        self.input_dim = self.full_train_X.size(-1)
        self.output_dim = self.full_train_Y.shape[-1]
        assert self.output_dim == 1, "Multi-output models not currently supported."
        self.output_scaler = OutputScaler(
            multi_fidelity=self.multi_fidelity,
            num_fidelities=self.num_fidelities,
            fidelity_dim=self.input_dim - 1,
        ).fit(self.full_train_Y, self.full_train_X)
        self.full_train_Y_scaled = self.output_scaler.transform(self.full_train_Y, self.full_train_X)
        self.target_fidelities = {self.input_dim - 1: self.num_fidelities - 1}

        # TODO: when running single fidelity with deluq, should n_epochs be set to 1000?
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

    def _checkpoint_metadata(self) -> Dict[str, Any]:
        return {
            "schema_version": DEEPOPT_CHECKPOINT_SCHEMA_VERSION,
            "model_type": self.config_settings.get_setting("model_type"),
            "training_data": {
                "X": self.X_orig.detach().cpu(),
                "y": self.Y_orig.detach().cpu(),
            },
            "bounds": self.bounds.detach().cpu(),
            "config_settings": dict(self.config_settings.config_settings),
            "random_seed": self.random_seed,
            "k_folds": self.k_folds,
            "multi_fidelity": self.multi_fidelity,
            "num_fidelities": self.num_fidelities,
            "target": self.target,
            "data_file": self.data_file,
        }

    @abstractmethod
    def train(self, outfile: str) -> Type[Model]:
        """
        Train the surrogate model and save the resulting model to a checkpoint file.

        :param outfile: The name of the checkpoint file where we will save
            the model trained on the dataset

        :returns: The model produced by training.
        """
        raise NotImplementedError("All DeepOpt models must have a `train` method implemented.")

    @abstractmethod
    def load_model(self, learner_file: str) -> Type[Model]:
        """
        Load the surrogate model (either GP or delUQ) from the learner file.

        :param learner_file: The name of the checkpoint file where we will load
            the model in from

        :returns: The model we loaded in.
        """
        raise NotImplementedError("All DeepOpt models must have a `load_model` method implemented.")

    def learn(self, outfile: str):
        """
        The method to process the `deepopt learn` command.

        Here we'll train a model on our dataset and save the model to a checkpoint file.

        :param outfile: The name of the checkpoint file where we will save the model
            trained on the dataset
        """
        print(
            f"""
            Infile: {self.data_file}
            Outfile: {outfile}
            Config File: {self.config_settings.config_file}
            Random Seed: {self.random_seed}
            K-Folds: {self.k_folds}
            Bounds: {self.bounds}
            Model Type: {self.config_settings.get_setting("model_type")}
            Multi-Fidelity: {self.multi_fidelity}
            """
        )
        self.train(outfile=outfile)
        
    def _project(self, X):
        return project_to_target_fidelity(X=X, target_fidelities=self.target_fidelities)

    def get_risk_measure_objective(self, risk_measure: str, **kwargs) -> Type[RiskMeasureMCObjective]:
        """
        Given a risk measure, return the associated BoTorch risk measure object.

        :param risk_measure: The risk measure to use. Options are 'CVaR' (Conditional Value-at-Risk)
            and 'VaR' (Value-at-Risk).

        :returns: Either a `CVaR` or `VaR` risk measure object from BoTorch
        """
        if risk_measure == "CVaR":
            return CVaR(**kwargs)
        if risk_measure == "VaR":
            return VaR(**kwargs)
        return None

    def _multiv_normal_samples(self, n: int, std_devs: np.ndarray) -> torch.Tensor:
        """
        Create a multivariate normal and draw `n` quasi-Monte Carlo (qMC) samples from the
        multivariate normal.

        :param n: The number of qMC samples to draw from the multivariate normal we'll
            obtain from `std_devs`
        :param std_devs: The tensor we'll draw qMC samples from

        :returns: A n x d tensor of samples where d is the dimension of the samples
        """
        mean = torch.zeros_like(std_devs,device=self.device)
        cov = torch.diag(std_devs**2).to(self.device)
        engine = MultivariateNormalQMCEngine(mean, cov, seed=self.random_seed)
        samples = engine.draw(n)
        return samples

    def get_input_perturbation(self, risk_n_deltas: int, bounds: np.ndarray, X_stddev: np.ndarray) -> InputPerturbation:
        """
        Get the input perturbation.

        :param risk_n_deltas: The number of input perturbations to sample for X's uncertainty
        :param bounds: Scaled bounds for each input dimension
        :param X_stddev: Scaled uncertainity in X (stddev) in each dimension

        :returns: A transform that adds the set of perturbations to the given input
        """
        assert len(X_stddev) == len(bounds.T), f"Expected {len(bounds.T)} values for X_stddev but recieved {len(X_stddev)}."
        input_pertubation = InputPerturbation(
            perturbation_set=self._multiv_normal_samples(risk_n_deltas, X_stddev),
            bounds=bounds,
        ).eval()
        return input_pertubation

    def _get_candidates_mf(
        self,
        model: Type[Model],
        acq_method: str,
        q: int,
        fidelity_cost: np.ndarray,
        risk_objective: Optional[Type[RiskMeasureMCObjective]] = None,
        risk_n_deltas: Optional[int] = None,
        n_fantasies: Optional[int] = Defaults.n_fantasies,
        propose_best: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the candidates for a multi-fidelity run.

        The bounds will be set and the fidelity cost model will be applied here first.
        Then whatever acquisition method requested with `acq_method` will be applied.

        :param model: The model loaded in by `load_model`. This will be a `SingleTaskMultiFidelityGP`
            model if we used GP to train the model or a `DeltaEnc` model if we used delUQ.
        :param acq_method: The acquisition method. Either 'GIBBON', 'MaxValEntropy', or 'KG'
        :param q: The number of candidates provided by the user (or the default value assigned
            in Default)
        :param fidelity_cost: A list of how expensive each fidelity should be seen as
        :param risk_objective: Either a `VaR` or a `CVaR` risk objective object from BoTorch. This will
            be determined by the `risk_measure` argument given by the user to the `deepopt optimize`
            command.
        :param risk_n_deltas: The number of input perturbations to sample for X's uncertainty
        :param n_fantasies: Number of fantasies to generate. The higher this number the more accurate
            the model (at the expense of model complexity and performance).
        :param propose_best: If `True`, the first candidate is selected to maximize the surrogate posterior,
            while the rest are acquired by the specified acquisition method. If `False`, acquire all points
            with the acquisition method as usual.

        :returns: A two element tuple containing a q x d-dim tensor of generated candidates
            and an associated acquisition value.
        """
        bounds = torch.tensor(self.input_dim * [[0, 1]],dtype=torch.float,device=self.device).T
        bounds[1, -1] = self.num_fidelities - 1

        cost_model = FidelityCostModel(fidelity_weights=fidelity_cost)
        cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)

        if propose_best:
            curr_val_acqf = FixedFeatureAcquisitionFunction(
                acq_function=PosteriorMean(
                    model,
                    posterior_transform=ExpectationPosteriorTransform(n_w=risk_n_deltas) if risk_objective else None,
                ),
                d=self.input_dim,
                columns=[self.input_dim - 1],
                values=[self.num_fidelities - 1],
            )

            best_candidate, max_pmean = optimize_acqf(
                acq_function=curr_val_acqf,
                bounds=bounds[:, :-1],
                q=1,
                num_restarts=Defaults.num_restarts_high,
                raw_samples=Defaults.raw_samples_high,
                options={"batch_limit": 10, "maxiter": 200, "seed": self.random_seed},
            )
            q-=1
            if q==0:
                acq_value = max_pmean
                print(f"{acq_value = }")
                candidates = torch.concat([best_candidate.reshape(1,-1),(self.num_fidelities-1)*torch.ones(1,1)],axis=1)
                return candidates, acq_value


        if acq_method in ("GIBBON", "MaxValEntropy"):
            n_candidates = 2000 * self.num_fidelities
            candidate_set = torch.rand(n_candidates, self.input_dim,device=self.device)
            candidate_set[:, -1] *= self.num_fidelities - 1
            candidate_set[:, -1] = candidate_set[:, -1].round()
            if acq_method == "MaxValEntropy":
                q_acq = qMultiFidelityMaxValueEntropy(
                    model,
                    num_fantasies=n_fantasies,
                    cost_aware_utility=cost_aware_utility,
                    project=self._project,
                    candidate_set=candidate_set,
                    seed=self.random_seed,
                )
            else:
                q_acq = qMultiFidelityLowerBoundMaxValueEntropy(
                    model,
                    posterior_transform=ExpectationPosteriorTransform(n_w=risk_n_deltas) if risk_objective else None,
                    num_fantasies=n_fantasies,
                    cost_aware_utility=cost_aware_utility,
                    project=self._project,
                    candidate_set=candidate_set,
                    seed=self.random_seed,
                )
            candidates, acq_value = optimize_acqf_mixed(
                q_acq,
                bounds=bounds,
                fixed_features_list=[{self.input_dim - 1: i} for i in range(self.num_fidelities)],
                q=q,
                num_restarts=Defaults.num_restarts_high,
                raw_samples=Defaults.raw_samples_high,
                options={"seed": self.random_seed},
            )
        elif acq_method == "KG":
            if not propose_best:
                curr_val_acqf = FixedFeatureAcquisitionFunction(
                    acq_function=PosteriorMean(
                        model,
                        posterior_transform=ExpectationPosteriorTransform(n_w=risk_n_deltas) if risk_objective else None,
                    ),
                    d=self.input_dim,
                    columns=[self.input_dim - 1],
                    values=[self.num_fidelities - 1],
                )

                _, max_pmean = optimize_acqf(
                    acq_function=curr_val_acqf,
                    bounds=bounds[:, :-1],
                    q=1,
                    num_restarts=Defaults.num_restarts_high,
                    raw_samples=Defaults.raw_samples_high,
                    options={"batch_limit": 10, "maxiter": 200, "seed": self.random_seed},
                )
            

            mfkg_acqf = qMultiFidelityKnowledgeGradient(
                model=model,
                num_fantasies=n_fantasies,
                sampler=SobolQMCNormalSampler(n_fantasies, seed=self.random_seed),
                inner_sampler=SobolQMCNormalSampler(n_fantasies, seed=self.random_seed),
                current_value=max_pmean,
                cost_aware_utility=cost_aware_utility,
                project=self._project,
                objective=risk_objective,
            )
            candidates, acq_value = optimize_acqf_mixed(
                acq_function=mfkg_acqf,
                bounds=bounds,
                fixed_features_list=[{self.input_dim - 1: i} for i in range(self.num_fidelities)],
                q=q,
                num_restarts=Defaults.num_restarts_low,
                raw_samples=Defaults.raw_samples_low,
                options={"batch_limit": 5, "maxiter": 200, "seed": self.random_seed},
            )
        if propose_best:
            best_candidate = torch.concat([best_candidate.reshape(1,-1),(self.num_fidelities-1)*torch.ones(1,1,device=self.device)],axis=1)
            candidates = torch.concat([best_candidate,candidates],axis=0)

        print(f"{acq_value = }")
        return candidates, acq_value

    def _get_candidates_sf(
        self,
        model: Type[Model],
        acq_method: str,
        q: int,
        risk_objective: Optional[Type[RiskMeasureMCObjective]] = None,
        risk_n_deltas: Optional[int] = None,
        n_fantasies: Optional[int] = Defaults.n_fantasies,
        propose_best: Optional[bool] = False,
    ) -> Tuple[Any, Any]:
        """
        Get the candidates for a single-fidelity run.

        The bounds will be set first, then whatever acquisition method requested with `acq_method`
        will be applied.

        :param model: The model loaded in by `load_model`. This will be a `SingleTaskGP`
            model if we used GP to train the model or a `DeltaEnc` if we used delUQ.
        :param acq_method: The acquisition method. Either 'EI', 'NEI', 'MaxValEntropy', or 'KG'
        :param q: The number of candidates provided by the user (or the default value assigned
            in Default)
        :param risk_objective: Either a `VaR` or a `CVaR` risk objective object from BoTorch. This will
            be determined by the `risk_measure` argument given by the user to the `deepopt optimize`
            command.
        :param risk_n_deltas: The number of input perturbations to sample for X's uncertainty
        :param n_fantasies: Number of fantasies to generate. The higher this number the more accurate
            the model (at the expense of model complexity and performance).
        :param propose_best: If `True`, the first candidate is selected to maximize the surrogate posterior,
            while the rest are acquired by the specified acquisition method. If `False`, acquire all points
            with the acquisition method as usual.
        :returns: A two element tuple containing a q x d-dim tensor of generated candidates
            and an associated acquisition value.
        """
        bounds = torch.tensor(self.input_dim * [[0, 1]],dtype=torch.float,device=self.device).T

        if propose_best:
            best_candidate, max_pmean = optimize_acqf(
                acq_function=PosteriorMean(
                    model,
                    posterior_transform=ExpectationPosteriorTransform(n_w=risk_n_deltas) if risk_objective else None,
                ),
                bounds=bounds,
                q=1,
                num_restarts=Defaults.num_restarts_high,
                raw_samples=Defaults.raw_samples_high,
            )
            q-=1
            if q==0:
                acq_value = max_pmean
                print(f"{acq_value = }")
                candidates = best_candidate.reshape(1,-1)
                return candidates, acq_value

        if acq_method == "EI":
            max_y = self.full_train_Y_scaled.max().item()
            q_acq = qExpectedImprovement(model, max_y, objective=risk_objective)
        elif acq_method == "NEI":
            q_acq = qNoisyExpectedImprovement(model, self.full_train_X, objective=risk_objective, prune_baseline=True)
            # TODO: Verify call syntax for qNoisyExpectedImprovement (why does it need inputs?)
        elif acq_method == "MaxValEntropy":
            n_candidates = 1000
            candidate_set = torch.rand(n_candidates, self.input_dim,device=self.device)
            q_acq = qMaxValueEntropy(
                model,
                posterior_transform=ExpectationPosteriorTransform(n_w=risk_n_deltas) if risk_objective else None,
                candidate_set=candidate_set,
                num_fantasies=n_fantasies,
                seed=self.random_seed,
            )
        elif acq_method == "KG":
            if not propose_best:
                _, max_pmean = optimize_acqf(
                    acq_function=PosteriorMean(
                        model,
                        posterior_transform=ExpectationPosteriorTransform(n_w=risk_n_deltas) if risk_objective else None,
                    ),
                    bounds=bounds,
                    q=1,
                    num_restarts=Defaults.num_restarts_high,
                    raw_samples=Defaults.raw_samples_high,
                )
            
            q_acq = qKnowledgeGradient(
                model=model,
                num_fantasies=n_fantasies,
                sampler=SobolQMCNormalSampler(n_fantasies, seed=self.random_seed),
                inner_sampler=SobolQMCNormalSampler(n_fantasies, seed=self.random_seed),
                current_value=max_pmean,
                objective=risk_objective,
            )
        candidates, acq_value = optimize_acqf(
            q_acq,
            bounds=bounds,
            q=q,
            num_restarts=Defaults.num_restarts_high,
            raw_samples=Defaults.raw_samples_low if acq_method in ["MaxValEntropy", "KG"] else Defaults.raw_samples_high,
            sequential=(acq_method == "MaxValEntropy"),
            options={"seed": self.random_seed},
        )
        if propose_best:
            candidates = torch.concat([best_candidate.reshape(1,-1),candidates],axis=0)
        print(f"{acq_value=}")
        return candidates, acq_value

    def get_candidates(
        self,
        model: Type[Model],
        acq_method: str,
        q: int,
        risk_objective: Optional[Type[RiskMeasureMCObjective]] = None,
        risk_n_deltas: Optional[int] = None,
        fidelity_cost: Optional[np.ndarray] = None,
        n_fantasies: Optional[int] = Defaults.n_fantasies,
        propose_best: Optional[bool] = False,
    ) -> Tuple[Any, Any]:
        """
        Get the candidates using the model loaded in with `load_model` and the acquisition method
        requested by the user.

        :param model: The model loaded in by `load_model`.
        :param acq_method: The acquisition method. Either 'EI', 'NEI', 'GIBBON', 'MaxValEntropy', or 'KG'
        :param q: The number of candidates provided by the user (or the default value assigned
            in Default)
        :param risk_objective: Either a `VaR` or a `CVaR` risk objective object from BoTorch. This will
            be determined by the `risk_measure` argument given by the user to the `deepopt optimize`
            command.
        :param risk_n_deltas: The number of input perturbations to sample for X's uncertainty
        :param fidelity_cost: A list of how expensive each fidelity should be seen as
        :param n_fantasies: Number of fantasies to generate. The higher this number the more accurate
            the model (at the expense of model complexity and performance).
        :param propose_best: If `True`, the first candidate is selected to maximize the surrogate posterior,
            while the rest are acquired by the specified acquisition method. If `False`, acquire all points
            with the acquisition method as usual.

        :returns: A two element tuple containing a q x d-dim tensor of generated candidates
            and an associated acquisition value.
        """

        current_max = self.full_train_Y[self.full_train_X[:,-1]==(self.num_fidelities-1)].max(
            ) if self.multi_fidelity else self.full_train_Y.max()
        print(f"Number of simulations: {len(self.full_train_X)}. Current max: {current_max.item():.5f}")

        if self.multi_fidelity:
            candidates, acq_value = self._get_candidates_mf(
                model=model,
                acq_method=acq_method,
                q=q,
                fidelity_cost=fidelity_cost,
                risk_objective=risk_objective,
                risk_n_deltas=risk_n_deltas,
                n_fantasies=n_fantasies,
                propose_best=propose_best,
            )
        else:
            candidates, acq_value = self._get_candidates_sf(
                model=model,
                acq_method=acq_method,
                q=q,
                risk_objective=risk_objective,
                risk_n_deltas=risk_n_deltas,
                n_fantasies=n_fantasies,
                propose_best=propose_best,
            )
        return candidates, acq_value

    def optimize(
        self,
        outfile: str,
        learner_file: str,
        acq_method: str,
        num_candidates: int = Defaults.num_candidates,
        fidelity_cost: np.ndarray = torch.tensor(json.loads(Defaults.fidelity_cost),dtype=torch.float),
        risk_measure: str = None,
        risk_level: float = None,
        risk_n_deltas: int = None,
        x_stddev: np.ndarray = None,
        n_fantasies: int = Defaults.n_fantasies,
        propose_best: bool = False,
        integer_fidelities: bool = False
    ) -> None:
        """
        The function to process the `deepopt optimize` command.

        Here we'll use the model created by `learn` to produce new simulation points.

        :param outfile: The name of the file to save the proposed candidates in
        :param learner_file: The name of the checkpoint file produced by `learn`
        :param acq_method: The acquisiton function. Single-fidelity options:
            'KG', 'MaxValEntropy', 'EI', or 'NEI'. Multi-fidelity options: 'KG' or
            'MaxValEntropy'
        :param num_candidates: The number of candidates
        :param fidelity_cost: List of costs for each fidelity
        :param risk_measure: The risk measure to use. Options: 'CVaR' (Conditional Value-at-Risk)
                or 'VaR' (Value-at-Risk).
        :param risk_level: The risk level (a float between 0 and 1)
        :param risk_n_deltas: The number of input perturbations to sample for X's uncertainty
        :param x_stddev: Uncertainity in X (stddev) in each dimension
        :param n_fantasies: Number of fantasies to generate. The higher this number the more accurate
            the model (at the expense of model complexity and performance).
        :param propose_best: If `True`, the first candidate is selected to maximize the surrogate posterior,
            while the rest are acquired by the specified acquisition method. If `False`, acquire all points
            with the acquisition method as usual. 
        :param integer_fidelities: If `True`, converts fidelity column to integers when saving candidate .npy file.
            Saved numpy array had dtype 'object' and requires `allow_pickle=True` option in `np.load` to read.
        """
        print(
            f"""
            Infile: {self.data_file}
            Outfile: {outfile}
            Config File: {self.config_settings.config_file}
            Learner File: {learner_file}
            Random Seed: {self.random_seed}
            Bounds: {self.bounds}
            Acq Method: {acq_method}
            Model Type: {self.config_settings.get_setting("model_type")}
            Multi-Fidelity: {self.multi_fidelity}
            Fidelity Cost: {fidelity_cost}
            """
        )
        model = self.load_model(learner_file=learner_file)

        if risk_measure:
            assert acq_method != "MaxValEntropy", "Risk measure not yet supported for MaxValueEntropy acquisition"
            x_stddev = torch.tensor(x_stddev,dtype=torch.float,device=self.bounds.device)
            x_stddev_scaled = x_stddev / (self.bounds[1] - self.bounds[0])
            bounds_scaled = torch.tensor(self.input_dim * [[0, 1]],dtype=torch.float).T
            if self.multi_fidelity:
                x_stddev_scaled[-1] = 0
            risk_objective = self.get_risk_measure_objective(risk_measure=risk_measure, alpha=risk_level, n_w=risk_n_deltas)
            model.input_transform = self.get_input_perturbation(
                risk_n_deltas=risk_n_deltas,
                bounds=bounds_scaled.to(self.device),
                X_stddev=x_stddev_scaled.to(self.device),
            )
        else:
            risk_objective = None
        model.eval()

        candidates, _ = self.get_candidates(
            model=model,
            acq_method=acq_method,
            q=num_candidates,
            risk_objective=risk_objective,
            risk_n_deltas=risk_n_deltas,
            fidelity_cost=fidelity_cost,
            n_fantasies=n_fantasies,
            propose_best=propose_best,
        )
        if self.multi_fidelity:
            candidates[:, :-1] = candidates[:, :-1] * (self.bounds[1, :-1] - self.bounds[0, :-1]) + self.bounds[0, :-1]
            candidates[:, -1] = candidates[:, -1].round()
        else:
            candidates = candidates * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        candidates_npy = candidates.cpu().detach().numpy()
        if integer_fidelities and self.multi_fidelity:
            candidates_npy = np.concatenate([candidates_npy[:,:-1].astype(np.float32),candidates_npy[:,-1:].astype(int)],axis=1,dtype='object')
        np.save(outfile, candidates_npy)


class DeepOptGPMixin:
    """
    Shared prediction API for GP models trained in scaled output units.
    """

    def get_prediction_with_uncertainty(
        self,
        q: torch.Tensor,
        get_cov: bool = False,
        original_scale: bool = True,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return GP posterior mean and variance or covariance.
        """
        posterior = self.posterior(q, **kwargs)
        mean = posterior.mean
        if get_cov:
            cov = posterior.mvn.covariance_matrix
            if original_scale:
                mean = self.output_scaler.inverse_transform(mean, q)
                cov = self.output_scaler.inverse_covariance(cov, q)
            return mean.squeeze(-1), cov
        variance = posterior.variance
        if original_scale:
            mean = self.output_scaler.inverse_transform(mean, q)
            variance = self.output_scaler.inverse_variance(variance, q)
        return mean, variance


class DeepOptSingleTaskGP(DeepOptGPMixin, SingleTaskGP):
    pass


class DeepOptSingleTaskMultiFidelityGP(DeepOptGPMixin, SingleTaskMultiFidelityGP):
    pass


class GPModel(DeepoptBaseModel):
    """
    DeepOpt's GP model representation. This class is where we'll
    define how to handle `learn` and `optimize` for GP models.

    This class has the same class variables as `DeepoptBaseModel`.
    """

    def train(self, outfile: str) -> Union[SingleTaskGP, SingleTaskMultiFidelityGP]:
        """
        Train the GP surrogate and save the model produced.

        :param outfile: The name of the output file to save the model to

        :returns: The model produced by training the GP surrogate. This will be a `SingleTaskGP`
             model from BoTorch if we're doing a single-fidelity run or a `SingleTaskMultiFidelityGP`
             model from BoTorch if we're doing a multi-fidelity run.
        """

        print("Training GP Surrogate.")
        model: Union[SingleTaskGP, SingleTaskMultiFidelityGP] = None
        mll: ExactMarginalLogLikelihood = None

        if self.multi_fidelity:
            model = DeepOptSingleTaskMultiFidelityGP(
                self.full_train_X,
                self.full_train_Y_scaled,
                data_fidelity=self.input_dim - 1,
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
        else:
            model = DeepOptSingleTaskGP(self.full_train_X, self.full_train_Y_scaled)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
        model.output_scaler = self.output_scaler

        fit_gpytorch_model(mll)

        state = {
            "state_dict": model.state_dict(),
            "output_scaler": self.output_scaler.state_dict(),
            DEEPOPT_CHECKPOINT_KEY: self._checkpoint_metadata(),
        }
        torch.save(state, join(getcwd(), dirname(outfile), basename(outfile)))
        return model

    def load_model(self, learner_file: str) -> Union[SingleTaskGP, SingleTaskMultiFidelityGP]:
        """
        Load in the GP model from the learner file.

        :param learner_file: The learner file that has the model we want to load

        :returns: Either a `SingleTaskGP` model or a `SingleTaskMultiFidelityGP` model
            depending on if we're doing a single-fidelity run or a multi-fidelity run
        """

        model: Union[SingleTaskGP, SingleTaskMultiFidelityGP] = None
        state_dict = _torch_load(learner_file, map_location=self.device)
        model_state = dict(state_dict["state_dict"])
        if "output_scaler" in state_dict:
            self.output_scaler = OutputScaler.from_state_dict(state_dict["output_scaler"], device=self.device)
            self.full_train_Y_scaled = self.output_scaler.transform(self.full_train_Y, self.full_train_X)
        elif any(key.startswith("outcome_transform.") for key in model_state):
            self.output_scaler = StandardizeOutputScaler.from_botorch_state_dict(model_state, device=self.device)
            self.full_train_Y_scaled = self.output_scaler.transform(self.full_train_Y, self.full_train_X)
            model_state = {key: value for key, value in model_state.items() if not key.startswith("outcome_transform.")}
            warnings.warn(
                "Loaded legacy GP checkpoint with BoTorch Standardize output scaling; using compatibility "
                "z-score scaling for predictions.",
                RuntimeWarning,
            )
        else:
            warnings.warn(
                "GP checkpoint has no output_scaler state; using scaler fit from the current data file.",
                RuntimeWarning,
            )

        if self.multi_fidelity:
            model = DeepOptSingleTaskMultiFidelityGP(
                self.full_train_X,
                self.full_train_Y_scaled,
                data_fidelity=self.input_dim - 1,
            )
        else:
            model = DeepOptSingleTaskGP(
                self.full_train_X,
                self.full_train_Y_scaled,
            )
        model.output_scaler = self.output_scaler
        model.load_state_dict(model_state)
        return model


class DelUQModel(DeepoptBaseModel):
    """
    DeepOpt's delta UQ model representation. This class is where we'll
    define how to handle `learn` and `optimize` for delta UQ models.

    This class has the same class variables as `DeepoptBaseModel`.
    """

    def _deluq_experiment(self, ray_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Training experiment used by ray tuning.

        :param ray_config: Configurations for the tuning, i.e. hyperparmeters to tune.

        :returns: A dictionary representing the score.
        """
        self.config_settings.set_setting("variance", ray_config["variance"])  # (2**-3)**2
        self.config_settings.set_setting("learning_rate", ray_config["learning_rate"])  # 0.01

        seed = ray_config["seed"]
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        dataset = TensorDataset(self.full_train_X, self.full_train_Y)

        if self.k_folds > len(self.full_train_X):
            kfold = KFold(n_splits=len(self.full_train_X), shuffle=True)
        else:
            kfold = KFold(n_splits=self.k_folds, shuffle=True)

        cv_loss_fun = torch.nn.MSELoss()
        cv_score = 0
        for _, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
            train_subsampler = SubsetRandomSampler(train_ids)
            test_subsampler = SubsetRandomSampler(test_ids)

            train_loader = DataLoader(dataset, batch_size=len(train_ids), sampler=train_subsampler)
            test_loader = DataLoader(dataset, batch_size=len(test_ids), sampler=test_subsampler)

            net = Arch(
                config=self.config_settings,
                unc_type="deltaenc",
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                device=self.device,
            )
            opt = create_optimizer(net, self.config_settings)

            for _, (X_train, y_train) in enumerate(train_loader):
                model = DeltaEnc(
                    network=net,
                    config=self.config_settings,
                    optimizer=opt,
                    X_train=X_train,
                    y_train=y_train,
                    target=self.target,
                    multi_fidelity=self.multi_fidelity,
                    output_scaler=self.output_scaler,
                )
                model.train()
                model.fit()

            model.eval()
            with torch.no_grad():  # TODO: is this needed?
                for _, (X_test, y_test) in enumerate(test_loader):
                    y_test_scaled = self.output_scaler.transform(y_test, X_test)
                    y_pred, _ = model.get_prediction_with_uncertainty(X_test, original_scale=False)
                    cv_score += cv_loss_fun(y_test_scaled, y_pred)

        return {"score": cv_score.item()}

    def train(self, outfile: str) -> Type[Model]:
        """
        Train the delUQ surrogate and save the model produced. We use ray to
        train the surrogate here.

        :param outfile: The name of the output file to save the model to

        :returns: The DeltaEnc model produced by training the delUQ surrogate.
        """

        print("Training DelUQ Surrogate.")

        warnings.filterwarnings("ignore", category=UserWarning)
        cpu_count = max(4, psutil.cpu_count(logical=False) - 3)
        # cpu_count = 4 if os.cpu_count() == 0 else (os.cpu_count()-3)
        gpu_count = torch.cuda.device_count()  # outputs warning when gpu not found
        warnings.resetwarnings()

        ray.init(num_cpus=cpu_count, num_gpus=gpu_count)
        num_samples = 20
        search_space = {
            "variance": tune.loguniform((2**-3) ** 2, 5e-1),  # (2 ** -3) ** 2,
            "learning_rate": tune.loguniform(2e-4, 5e-1),
            "seed": tune.randint(0, 10000),
        }
        trainable_with_resources = tune.with_resources(
            trainable=self._deluq_experiment,
            resources={
                "cpu": 1 if cpu_count < num_samples else 2,
            },
        )
        tuner = tune.Tuner(
            trainable=trainable_with_resources,
            run_config=RunConfig(
                verbose=0,
            ),
            tune_config=tune.TuneConfig(
                num_samples=num_samples,
                scheduler=ASHAScheduler(
                    metric="score",
                    mode="min",
                ),
            ),
            param_space=search_space,
        )
        result = tuner.fit()
        best_result = result.get_best_result(metric="score", mode="min")
        print(best_result)

        for key, val in best_result.config.items():
            print(f"{key} {val}")
            if key in self.config_settings:
                self.config_settings.set_setting(key, val)
        net = Arch(
            config=self.config_settings,
            unc_type="deltaenc",
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            device=self.device,
        )
        opt = create_optimizer(net, self.config_settings)

        model = DeltaEnc(
            network=net,
            config=self.config_settings,
            optimizer=opt,
            X_train=self.full_train_X,
            y_train=self.full_train_Y,
            target=self.target,
            multi_fidelity=self.multi_fidelity,
            output_scaler=self.output_scaler,
        )

        model.fit()
        if basename(outfile).split(".")[-1] == "ckpt":
            fname = basename(outfile)[:-5]
        else:
            fname = basename(outfile)
        model.save_ckpt(join(getcwd(), dirname(outfile)), fname, checkpoint_metadata=self._checkpoint_metadata())
        ray.shutdown()
        return model

    def load_model(self, learner_file: str) -> Type[Model]:
        """
        Load in the delUQ model from the learner file.

        :param learner_file: The learner file that has the model we want to load

        :returns: A 'DeltaEnc' model.
        """
        net = Arch(
            config=self.config_settings,
            unc_type="deltaenc",
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            device=self.device,
        )
        opt = create_optimizer(net, self.config_settings)

        model = DeltaEnc(
            network=net,
            config=self.config_settings,
            optimizer=opt,
            X_train=self.full_train_X,
            y_train=self.full_train_Y,
            target=self.target,
            multi_fidelity=self.multi_fidelity,
            output_scaler=self.output_scaler,
        )

        # DeltaEnc model requries the parent path and file name to be separated.
        # Extension of file is also removed and assumed to be ".ckpt".
        if basename(learner_file).split(".")[-1] == "ckpt":
            file_name = basename(learner_file)[:-5]
        else:
            file_name = basename(learner_file)
        # file_name = basename(learner_file).split(".")[0]
        dir_name = dirname(learner_file)
        model.load_ckpt(dir_name, file_name)
        return model

class NNEnsembleModel(DeepoptBaseModel):
    def train(self, outfile: str) -> Type[Model]:
        """
        Train the NN Ensemble surrogate and save the model produced. 

        :param outfile: The name of the output file to save the model to

        :returns: The NNEnsemble model produced by training the NN Ensemble surrogate.
        """

        print("Training NN Ensemble Surrogate.")

        warnings.filterwarnings("ignore", category=UserWarning)
        n_estimators = self.config_settings.get_setting("n_estimators")
        nets = [Arch(
            config=self.config_settings,
            unc_type="ensemble",
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            device=self.device,
        ) for _ in range(n_estimators)]
        opts = [create_optimizer(net, self.config_settings) for net in nets]

        model = NNEnsemble(
            networks=nets,
            config=self.config_settings,
            optimizers=opts,
            X_train=self.full_train_X,
            y_train=self.full_train_Y,
            multi_fidelity=self.multi_fidelity,
            output_scaler=self.output_scaler,
            verbose=self.verbose,
        )

        model.fit()
        if basename(outfile).split(".")[-1] == "ckpt":
            fname = basename(outfile)[:-5]
        else:
            fname = basename(outfile)
        model.save_ckpt(join(getcwd(), dirname(outfile)), fname, checkpoint_metadata=self._checkpoint_metadata())
        ray.shutdown()
        return model

    def load_model(self, learner_file: str) -> Type[Model]:
        """
        Load in the nnEnsemble model from the learner file.

        :param learner_file: The learner file that has the model we want to load

        :returns: A 'NNEnsemble' model.
        """
        n_estimators = self.config_settings.get_setting("n_estimators")
        nets = [Arch(
            config=self.config_settings,
            unc_type="ensemble",
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            device=self.device,
        ) for _ in range(n_estimators)]
        opts = [create_optimizer(net, self.config_settings) for net in nets]

        model = NNEnsemble(
            networks=nets,
            config=self.config_settings,
            optimizers=opts,
            X_train=self.full_train_X,
            y_train=self.full_train_Y,
            multi_fidelity=self.multi_fidelity,
            output_scaler=self.output_scaler,
            verbose=self.verbose
        )

        # NNEnsemble model requries the parent path and file name to be separated.
        # Extension of file is also removed and assumed to be ".ckpt".
        if basename(learner_file).split(".")[-1] == "ckpt":
            file_name = basename(learner_file)[:-5]
        else:
            file_name = basename(learner_file)
        # file_name = basename(learner_file).split(".")[0]
        dir_name = dirname(learner_file)
        model.load_ckpt(dir_name, file_name)
        return model
