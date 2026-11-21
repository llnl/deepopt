"""
This module contains the high-level functionality of the DeepOpt library.
It's where we handle the configuration of each model (i.e. how we process
the `learn` and `optimize` commands for each model).
"""
import json
import os
import random
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from os import getcwd
from os.path import basename, dirname, join
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import psutil
import torch
from deepopt._compat import OptionalDependencyError, fit_gpytorch_model, make_sobol_qmc_normal_sampler, require_ray_for_deluq
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
from botorch.optim.initializers import gen_batch_initial_conditions
from botorch.optim.optimize import optimize_acqf, optimize_acqf_mixed
from botorch.utils.sampling import draw_sobol_samples
from botorch.sampling.qmc import MultivariateNormalQMCEngine
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset

from deepopt.acq_optimizers import (
    ACQ_OPTIMIZERS,
    BATCH_MODES,
    BOLTZMANN_SAMPLERS,
    SAMPLING_PERTURBATIONS,
    SamplingOptions,
    SearchSpace,
    batched_max_search,
    draw_feasible_points,
    optimize_acqf_gradient_free,
    search_elites,
)
from deepopt.acquisition import (
    SampledKnowledgeGradient,
    qMaxValueEntropy,
    qMultiFidelityLowerBoundMaxValueEntropy,
    qMultiFidelityMaxValueEntropy,
)
from deepopt.configuration import ConfigSettings
from deepopt.defaults import Defaults, OPTIMIZATION_PROFILES, SAMPLING_OPTIMIZATION_SETTINGS
from deepopt.deltaenc import DeltaEnc
from deepopt.input_scaling import InputScaler, reject_deprecated_original_scale
from deepopt.nn_ensemble import NNEnsemble
from deepopt.output_scaling import OutputScaler, StandardizeOutputScaler
from deepopt.surrogate_rf import RandomForestSurrogate
from deepopt.surrogate_tabpfn import TabPFN
from deepopt.surrogate_utils import MLP as Arch
from deepopt.surrogate_utils import create_optimizer


DEEPOPT_CHECKPOINT_KEY = "deepopt_checkpoint"
DEEPOPT_CHECKPOINT_SCHEMA_VERSION = 1
DEEPOPT_MODEL_TYPES = ("GP", "delUQ", "nnEnsemble", "TabPFN", "RF")
OPTIMIZATION_SETTING_KEYS = frozenset(next(iter(OPTIMIZATION_PROFILES.values())).keys())
OPTIMIZATION_THREAD_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TORCH_NUM_THREADS")


@dataclass(frozen=True)
class AcquisitionOptimizationSettings:
    """
    Resolved acquisition optimizer settings.

    The ``high`` settings are used for normal acquisition optimization paths, while
    the ``low`` settings are used for more expensive acquisition functions such as
    KG and MaxValEntropy. Thread settings apply only to CPU optimization.

    :cvar num_restarts_high: Multistart local optimizations for high-budget paths.
    :cvar num_restarts_low: Multistart local optimizations for low-budget paths.
    :cvar raw_samples_high: Sobol samples used to initialize high-budget paths.
    :cvar raw_samples_low: Sobol samples used to initialize low-budget paths.
    :cvar batch_limit_high: Restart batches evaluated together for high-budget paths.
    :cvar batch_limit_low: Restart batches evaluated together for low-budget paths.
    :cvar maxiter: Maximum iterations for each local optimizer run.
    :cvar n_fantasies: Number of fantasy models used by KG/MES-family acquisitions.
    :cvar torch_num_threads: PyTorch intra-op threads, ``"auto"``, or ``None``.
    :cvar torch_num_threads_fraction: Fraction of available CPUs used by ``"auto"``.
    :cvar torch_num_interop_threads: PyTorch inter-op thread count, or ``None``.
    :cvar nonlinear_mode: ``'enforce'`` or ``'initialization_only'`` for nonlinear constraints.
    :cvar nonlinear_initial_raw_samples: Optional raw sample count for nonlinear-feasible starts.
    :cvar nonlinear_initial_max_tries: Number of attempts to find nonlinear-feasible starts.
    :cvar nonlinear_optimization_retries: Additional retries after optimizer failure warnings.
    :cvar acq_optimizer: ``'gradient'`` (BoTorch multistart L-BFGS, the default), ``'sampling'``
        (gradient-free scoped random search), or ``'boltzmann'`` (i.i.d. draws from the Boltzmann
        distribution of the acquisition function).
    :cvar batch_mode: ``'default'`` (MaxValEntropy sequential; EI/NEI/KG joint), ``'sequential'``
        (greedy with pending points), or ``'joint'`` (optimize the q-batch jointly; EI/NEI/KG only).
        Ignored by ``'boltzmann'``, which always draws i.i.d. points.
    :cvar sampling_*: Scoped-search budgets and box schedule (see the acquisition optimizer docs).
    :cvar boltzmann_*: Boltzmann sampler choice (``'smc'``, ``'defer'``, ``'importance'``),
        inverse temperature, and budgets.
    :cvar kg_*: Gradient-free KG inner-set size, refinement count, and outer search budget.
    """

    num_restarts_high: int
    num_restarts_low: int
    raw_samples_high: int
    raw_samples_low: int
    batch_limit_high: int
    batch_limit_low: int
    maxiter: int
    n_fantasies: int
    torch_num_threads: Optional[Union[int, str]]
    torch_num_threads_fraction: float
    torch_num_interop_threads: Optional[int]
    nonlinear_mode: str = Defaults.nonlinear_mode
    nonlinear_initial_raw_samples: Optional[int] = Defaults.nonlinear_initial_raw_samples
    nonlinear_initial_max_tries: int = Defaults.nonlinear_initial_max_tries
    nonlinear_optimization_retries: int = Defaults.nonlinear_optimization_retries
    acq_optimizer: str = SAMPLING_OPTIMIZATION_SETTINGS["acq_optimizer"]
    batch_mode: str = SAMPLING_OPTIMIZATION_SETTINGS["batch_mode"]
    sampling_raw_samples: int = SAMPLING_OPTIMIZATION_SETTINGS["sampling_raw_samples"]
    sampling_n_elites: int = SAMPLING_OPTIMIZATION_SETTINGS["sampling_n_elites"]
    sampling_samples_per_elite: int = SAMPLING_OPTIMIZATION_SETTINGS["sampling_samples_per_elite"]
    sampling_n_rounds: int = SAMPLING_OPTIMIZATION_SETTINGS["sampling_n_rounds"]
    sampling_init_halfwidth: float = SAMPLING_OPTIMIZATION_SETTINGS["sampling_init_halfwidth"]
    sampling_shrink: float = SAMPLING_OPTIMIZATION_SETTINGS["sampling_shrink"]
    sampling_min_halfwidth: float = SAMPLING_OPTIMIZATION_SETTINGS["sampling_min_halfwidth"]
    sampling_perturbation: str = SAMPLING_OPTIMIZATION_SETTINGS["sampling_perturbation"]
    sampling_subset_prob: Optional[float] = SAMPLING_OPTIMIZATION_SETTINGS["sampling_subset_prob"]
    sampling_eval_batch: int = SAMPLING_OPTIMIZATION_SETTINGS["sampling_eval_batch"]
    sampling_joint_member_prob: float = SAMPLING_OPTIMIZATION_SETTINGS["sampling_joint_member_prob"]
    boltzmann_sampler: str = SAMPLING_OPTIMIZATION_SETTINGS["boltzmann_sampler"]
    boltzmann_lambda: float = SAMPLING_OPTIMIZATION_SETTINGS["boltzmann_lambda"]
    boltzmann_population: int = SAMPLING_OPTIMIZATION_SETTINGS["boltzmann_population"]
    boltzmann_n_rounds: int = SAMPLING_OPTIMIZATION_SETTINGS["boltzmann_n_rounds"]
    boltzmann_mh_steps: int = SAMPLING_OPTIMIZATION_SETTINGS["boltzmann_mh_steps"]
    boltzmann_defer_budget: int = SAMPLING_OPTIMIZATION_SETTINGS["boltzmann_defer_budget"]
    kg_inner_set_size: int = SAMPLING_OPTIMIZATION_SETTINGS["kg_inner_set_size"]
    kg_refine_top: int = SAMPLING_OPTIMIZATION_SETTINGS["kg_refine_top"]
    kg_raw_samples: int = SAMPLING_OPTIMIZATION_SETTINGS["kg_raw_samples"]
    kg_n_elites: int = SAMPLING_OPTIMIZATION_SETTINGS["kg_n_elites"]
    kg_samples_per_elite: int = SAMPLING_OPTIMIZATION_SETTINGS["kg_samples_per_elite"]
    kg_n_rounds: int = SAMPLING_OPTIMIZATION_SETTINGS["kg_n_rounds"]


@dataclass(frozen=True)
class AcquisitionOptimizationConstraints:
    """
    Optional constraints and initialization controls for acquisition optimization.

    Linear constraints are supplied in original input units as tuples of
    ``(indices, coefficients, rhs)``. DeepOpt converts them to the scaled model
    coordinates used by BoTorch, leaving the fidelity index unscaled in
    multi-fidelity runs. Nonlinear inequality constraints are callables that
    receive original-unit candidate tensors and return values where ``>= 0`` means
    feasible. ``nonlinear_mode='enforce'`` passes nonlinear constraints to BoTorch;
    ``'initialization_only'`` only uses them to seed feasible initial conditions.

    :cvar inequality_constraints: Linear inequalities of the form ``sum(c_i x_i) >= rhs``.
    :cvar equality_constraints: Linear equalities of the form ``sum(c_i x_i) == rhs``.
    :cvar nonlinear_inequality_constraints: Callable nonlinear feasibility tests.
    :cvar post_processing_func: Optional BoTorch post-processing function for candidates.
    :cvar batch_initial_conditions: Optional original-unit initial conditions with shape
        ``num_restarts x q x input_dim`` or ``num_restarts x input_dim``.
    :cvar nonlinear_mode: ``'enforce'`` or ``'initialization_only'``.
    :cvar nonlinear_initial_raw_samples: Optional raw sample count for nonlinear-feasible starts.
    :cvar nonlinear_initial_max_tries: Number of attempts to find nonlinear-feasible starts.
    :cvar nonlinear_optimization_retries: Additional retries after optimizer failure warnings.
    """

    inequality_constraints: Optional[List[Tuple[Sequence[int], Sequence[float], float]]] = None
    equality_constraints: Optional[List[Tuple[Sequence[int], Sequence[float], float]]] = None
    nonlinear_inequality_constraints: Optional[List[Callable[[torch.Tensor], torch.Tensor]]] = None
    post_processing_func: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    batch_initial_conditions: Optional[torch.Tensor] = None
    nonlinear_mode: Optional[str] = None
    nonlinear_initial_raw_samples: Optional[int] = None
    nonlinear_initial_max_tries: Optional[int] = None
    nonlinear_optimization_retries: Optional[int] = None



def _torch_load(learner_file: str, map_location: str = "cpu", weights_only: bool = False):
    """
    Load a PyTorch checkpoint across supported PyTorch versions.

    :param learner_file: Path to the checkpoint file.
    :param map_location: Device mapping passed to ``torch.load``.
    :param weights_only: Whether to request PyTorch's restricted checkpoint loader.
    :returns: The object loaded from ``learner_file``.
    """
    try:
        return torch.load(learner_file, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(learner_file, map_location=map_location)


def get_checkpoint_metadata(learner_file: str, map_location: str = "cpu") -> Optional[Dict[str, Any]]:
    """
    Read self-describing DeepOpt metadata from a checkpoint.

    Legacy checkpoints that do not contain ``deepopt_checkpoint`` metadata return
    ``None``. Checkpoints that contain malformed DeepOpt metadata raise
    ``ValueError`` so callers can distinguish old files from broken modern files.

    :param learner_file: Path to the checkpoint file saved with ``torch.save``.
    :param map_location: Device mapping passed to ``torch.load``.
    :returns: Metadata with schema version, model type, training data, bounds, and config settings, or ``None`` for legacy checkpoints.
    :raises ValueError: If metadata is present but has an unsupported schema, invalid model type, or missing required fields.
    """
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
    if model_type not in DEEPOPT_MODEL_TYPES:
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
    """
    Return whether a checkpoint contains valid DeepOpt metadata.

    :param learner_file: Path to the checkpoint file.
    :returns: ``True`` if ``get_checkpoint_metadata`` returns metadata, otherwise ``False``.
    """
    return get_checkpoint_metadata(learner_file) is not None


def _config_settings_from_checkpoint(metadata: Dict[str, Any]) -> ConfigSettings:
    """
    Rebuild configuration settings from checkpoint metadata.

    :param metadata: Valid self-describing checkpoint metadata.
    :returns: A ``ConfigSettings`` object populated from the saved settings.
    """
    config_settings = ConfigSettings(metadata["model_type"])
    config_settings.config_settings.update(metadata.get("config_settings", {}))
    config_settings.config_file = None
    return config_settings


def load_deepopt_wrapper(learner_file: str, device: str = "auto", verbose: bool = False) -> "DeepoptBaseModel":
    """
    Load the appropriate DeepOpt wrapper from a self-describing checkpoint.

    :param learner_file: Path to a checkpoint containing ``deepopt_checkpoint`` metadata.
    :param device: Device requested for the wrapper and underlying model.
    :param verbose: If ``True``, enable verbose model evaluation output where supported.
    :returns: A ``GPModel``, ``DelUQModel``, ``NNEnsembleModel``, ``TabPFNModel``, or ``RFModel`` initialized from checkpoint metadata.
    :raises ValueError: If the checkpoint is legacy or has invalid metadata.
    """
    metadata = get_checkpoint_metadata(learner_file)
    if metadata is None:
        raise ValueError(
            "Checkpoint does not contain DeepOpt self-describing metadata. Load it using the legacy explicit path: "
            "construct the appropriate GPModel, DelUQModel, NNEnsembleModel, TabPFNModel, or RFModel with data_file, bounds, "
            "config_settings, and multi_fidelity, then call load_model(...)."
        )
    model_classes = {"GP": GPModel, "delUQ": DelUQModel, "nnEnsemble": NNEnsembleModel, "TabPFN": TabPFNModel, "RF": RFModel}
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
        learner_file=learner_file,
    )


def load_deepopt_model(learner_file: str, device: str = "auto", verbose: bool = False) -> Type[Model]:
    """
    Load the underlying BoTorch-compatible model from a self-describing checkpoint.

    :param learner_file: Path to a checkpoint containing ``deepopt_checkpoint`` metadata.
    :param device: Device requested for the wrapper and underlying model.
    :param verbose: If ``True``, enable verbose model evaluation output where supported.
    :returns: The model returned by the reconstructed wrapper's ``load_model`` method.
    """
    wrapper = load_deepopt_wrapper(learner_file, device=device, verbose=verbose)
    return wrapper.load_model(learner_file)


class FidelityCostModel(DeterministicModel):
    """
    Deterministic cost model for multi-fidelity acquisition utilities.

    The last input column is interpreted as a fidelity index. During acquisition
    optimization the index is rounded to the nearest integer and used to select
    the corresponding entry from ``fidelity_weights``.
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
    :cvar k_folds: The number of folds to use when training a delUQ surrogate.
        `Default: Defaults.k_folds`
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
    learner_file: str = None

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
        self.input_scaler = InputScaler(bounds, multi_fidelity=self.multi_fidelity, fidelity_dim=-1)
        self.full_train_X = self.input_scaler.transform(self.X_orig)
        if self.multi_fidelity:
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
        self.input_scaler.to(self.device)

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
        """
        Build metadata needed to reload this wrapper without the original CLI inputs.

        :returns: Self-describing checkpoint metadata stored under ``deepopt_checkpoint``.
        """
        return {
            "schema_version": DEEPOPT_CHECKPOINT_SCHEMA_VERSION,
            "model_type": self.config_settings.get_setting("model_type"),
            "training_data": {
                "X": self.X_orig.detach().cpu(),
                "y": self.Y_orig.detach().cpu(),
            },
            "bounds": self.bounds.detach().cpu(),
            # Save config settings verbatim so optimize-time loading preserves user choices.
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
        Build the BoTorch input transform used by VaR and CVaR objectives.

        ``X_stddev`` must already be in the same scaled coordinates as ``bounds``.
        Public APIs such as ``optimize``, ``get_var``, and ``get_cvar`` accept
        standard deviations in original input units and scale them before calling
        this helper. Multi-fidelity callers set the fidelity-column standard
        deviation to zero so perturbations do not change fidelity.

        :param risk_n_deltas: Number of quasi-Monte Carlo input perturbations.
        :param bounds: Scaled bounds for each input dimension.
        :param X_stddev: Scaled input standard deviations, one per input dimension.
        :returns: A transform that adds sampled perturbations to query inputs.
        """
        assert len(X_stddev) == len(bounds.T), f"Expected {len(bounds.T)} values for X_stddev but recieved {len(X_stddev)}."
        input_pertubation = InputPerturbation(
            perturbation_set=self._multiv_normal_samples(risk_n_deltas, X_stddev),
            bounds=bounds,
        ).eval()
        return input_pertubation

    def _resolve_learner_file(self, learner_file: Optional[str]) -> str:
        learner_file = learner_file or self.learner_file
        if learner_file is None:
            raise ValueError(
                "No learner_file was provided and this wrapper does not have a checkpoint path. "
                "Pass learner_file=... or construct the wrapper with load_deepopt_wrapper(...)."
            )
        return learner_file

    def _prepare_risk_query_inputs(
        self,
        X_query: Optional[torch.Tensor],
        x_stddev: Union[np.ndarray, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if X_query is None:
            X_query = self.full_train_X
        X_query = torch.as_tensor(X_query, dtype=torch.float, device=self.device)
        if X_query.ndim == 1:
            X_query = X_query.reshape(1, -1)
        x_stddev = torch.as_tensor(x_stddev, dtype=torch.float, device=self.device)
        if X_query.shape[-1] != self.input_dim:
            raise ValueError(f"Expected X_query last dimension {self.input_dim}, received {X_query.shape[-1]}.")
        if x_stddev.numel() != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} values for x_stddev, received {x_stddev.numel()}.")
        return X_query, x_stddev.reshape(-1)

    def _get_scaled_x_stddev(self, x_stddev: torch.Tensor) -> torch.Tensor:
        x_stddev_scaled = x_stddev / (self.bounds[1] - self.bounds[0])
        x_stddev_scaled = x_stddev_scaled.clone()
        if self.multi_fidelity:
            x_stddev_scaled[-1] = 0
        return x_stddev_scaled

    def _prepare_risk_evaluation(
        self,
        risk_measure: str,
        risk_level: float,
        risk_n_deltas: int,
        x_stddev: torch.Tensor,
        learner_file: Optional[str],
    ) -> Tuple[Type[Model], Type[RiskMeasureMCObjective]]:
        if risk_n_deltas <= 0:
            raise ValueError("risk_n_deltas must be positive.")
        model = self.load_model(learner_file=self._resolve_learner_file(learner_file))
        x_stddev_scaled = self._get_scaled_x_stddev(x_stddev)
        bounds_scaled = torch.tensor(self.input_dim * [[0, 1]], dtype=torch.float, device=self.device).T
        if self.multi_fidelity:
            bounds_scaled[1, -1] = self.num_fidelities - 1
        risk_objective = self.get_risk_measure_objective(
            risk_measure=risk_measure,
            alpha=risk_level,
            n_w=risk_n_deltas,
        )
        if risk_objective is None:
            raise ValueError(f"Unsupported risk measure {risk_measure}.")
        model.input_transform = self.get_input_perturbation(
            risk_n_deltas=risk_n_deltas,
            bounds=bounds_scaled,
            X_stddev=x_stddev_scaled,
        )
        model.eval()
        return model, risk_objective

    def _evaluate_risk_measure(
        self,
        model: Type[Model],
        risk_objective: Type[RiskMeasureMCObjective],
        X_query: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            posterior = model.posterior(X_query.unsqueeze(-2))
            samples = posterior.rsample().squeeze(0)
            return risk_objective(samples).squeeze(-1)

    def _get_risk_measure_values(
        self,
        risk_measure: str,
        X_query: Optional[torch.Tensor],
        risk_level: float,
        x_stddev: Union[np.ndarray, torch.Tensor],
        risk_n_deltas: int,
        learner_file: Optional[str],
    ) -> torch.Tensor:
        X_query, x_stddev = self._prepare_risk_query_inputs(X_query, x_stddev)
        model, risk_objective = self._prepare_risk_evaluation(
            risk_measure=risk_measure,
            risk_level=risk_level,
            risk_n_deltas=risk_n_deltas,
            x_stddev=x_stddev,
            learner_file=learner_file,
        )
        return self._evaluate_risk_measure(model, risk_objective, X_query)

    def get_var(
        self,
        X_query: Optional[torch.Tensor] = None,
        *,
        risk_level: float,
        x_stddev: Union[np.ndarray, torch.Tensor],
        risk_n_deltas: int = 128,
        learner_file: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Evaluate Value-at-Risk at query points under input perturbations.

        :param X_query: Query inputs in model units. If ``None``, evaluate at the training inputs.
        :param risk_level: Quantile level between 0 and 1.
        :param x_stddev: Input standard deviations in original input units, one value per input dimension.
        :param risk_n_deltas: Number of input perturbations sampled for the risk objective.
        :param learner_file: Optional checkpoint path; defaults to this wrapper's checkpoint.
        :returns: VaR values with one value per query point.
        """
        return self._get_risk_measure_values("VaR", X_query, risk_level, x_stddev, risk_n_deltas, learner_file)

    def get_cvar(
        self,
        X_query: Optional[torch.Tensor] = None,
        *,
        risk_level: float,
        x_stddev: Union[np.ndarray, torch.Tensor],
        risk_n_deltas: int = 128,
        learner_file: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Evaluate Conditional Value-at-Risk at query points under input perturbations.

        :param X_query: Query inputs in model units. If ``None``, evaluate at the training inputs.
        :param risk_level: Tail level between 0 and 1.
        :param x_stddev: Input standard deviations in original input units, one value per input dimension.
        :param risk_n_deltas: Number of input perturbations sampled for the risk objective.
        :param learner_file: Optional checkpoint path; defaults to this wrapper's checkpoint.
        :returns: CVaR values with one value per query point.
        """
        return self._get_risk_measure_values("CVaR", X_query, risk_level, x_stddev, risk_n_deltas, learner_file)

    def _resolve_optimization_settings(
        self,
        optimization_settings: Optional[AcquisitionOptimizationSettings] = None,
    ) -> AcquisitionOptimizationSettings:
        if optimization_settings is not None:
            return optimization_settings
        optimization_config = self.config_settings.config_settings.get("optimization", {})
        if optimization_config is None:
            optimization_config = {}
        if not isinstance(optimization_config, dict):
            raise ValueError("The optimization config section must be a mapping.")
        profile = optimization_config.get("profile", Defaults.optimization_profile)
        if profile not in OPTIMIZATION_PROFILES:
            raise ValueError(f"Unknown optimization profile {profile}. Valid profiles: {sorted(OPTIMIZATION_PROFILES)}")
        unknown_keys = set(optimization_config).difference(OPTIMIZATION_SETTING_KEYS | {"profile"})
        if unknown_keys:
            raise ValueError(f"Unknown optimization setting(s): {sorted(unknown_keys)}")
        settings = dict(OPTIMIZATION_PROFILES[profile])
        settings.update({key: value for key, value in optimization_config.items() if key != "profile"})
        return self._validate_optimization_settings(settings)

    def _validate_optimization_settings(self, settings: Dict[str, Any]) -> AcquisitionOptimizationSettings:
        positive_int_keys = (
            "num_restarts_high",
            "num_restarts_low",
            "raw_samples_high",
            "raw_samples_low",
            "batch_limit_high",
            "batch_limit_low",
            "maxiter",
            "n_fantasies",
        )
        for key in positive_int_keys:
            if isinstance(settings[key], bool) or int(settings[key]) <= 0:
                raise ValueError(f"optimization.{key} must be a positive integer.")
            settings[key] = int(settings[key])
        nonlinear_mode = settings["nonlinear_mode"].replace("-", "_")
        if nonlinear_mode not in {"enforce", "initialization_only"}:
            raise ValueError("optimization.nonlinear_mode must be 'enforce' or 'initialization_only'.")
        settings["nonlinear_mode"] = nonlinear_mode
        nonlinear_initial_raw_samples = settings["nonlinear_initial_raw_samples"]
        if nonlinear_initial_raw_samples is not None:
            if isinstance(nonlinear_initial_raw_samples, bool) or int(nonlinear_initial_raw_samples) <= 0:
                raise ValueError("optimization.nonlinear_initial_raw_samples must be null or a positive integer.")
            settings["nonlinear_initial_raw_samples"] = int(nonlinear_initial_raw_samples)
        if isinstance(settings["nonlinear_initial_max_tries"], bool) or int(settings["nonlinear_initial_max_tries"]) <= 0:
            raise ValueError("optimization.nonlinear_initial_max_tries must be a positive integer.")
        settings["nonlinear_initial_max_tries"] = int(settings["nonlinear_initial_max_tries"])
        if isinstance(settings["nonlinear_optimization_retries"], bool) or int(settings["nonlinear_optimization_retries"]) < 0:
            raise ValueError("optimization.nonlinear_optimization_retries must be a non-negative integer.")
        settings["nonlinear_optimization_retries"] = int(settings["nonlinear_optimization_retries"])
        fraction = float(settings["torch_num_threads_fraction"])
        if not 0 < fraction <= 1:
            raise ValueError("optimization.torch_num_threads_fraction must be in (0, 1].")
        settings["torch_num_threads_fraction"] = fraction
        torch_num_threads = settings["torch_num_threads"]
        if torch_num_threads is not None and torch_num_threads != "auto":
            if isinstance(torch_num_threads, bool) or int(torch_num_threads) <= 0:
                raise ValueError("optimization.torch_num_threads must be 'auto', null, or a positive integer.")
            settings["torch_num_threads"] = int(torch_num_threads)
        torch_num_interop_threads = settings["torch_num_interop_threads"]
        if torch_num_interop_threads is not None:
            if isinstance(torch_num_interop_threads, bool) or int(torch_num_interop_threads) <= 0:
                raise ValueError("optimization.torch_num_interop_threads must be null or a positive integer.")
            settings["torch_num_interop_threads"] = int(torch_num_interop_threads)
        self._validate_sampling_settings(settings)
        return AcquisitionOptimizationSettings(**settings)

    @staticmethod
    def _validate_sampling_settings(settings: Dict[str, Any]) -> None:
        """Validate and coerce gradient-free acquisition optimizer settings in place."""
        choices = {
            "acq_optimizer": ACQ_OPTIMIZERS,
            "batch_mode": BATCH_MODES,
            "sampling_perturbation": SAMPLING_PERTURBATIONS,
            "boltzmann_sampler": BOLTZMANN_SAMPLERS,
        }
        for key, allowed in choices.items():
            if settings[key] not in allowed:
                raise ValueError(f"optimization.{key} must be one of {list(allowed)}, got {settings[key]!r}.")
        positive_int_keys = (
            "sampling_raw_samples",
            "sampling_n_elites",
            "sampling_samples_per_elite",
            "sampling_eval_batch",
            "boltzmann_population",
            "boltzmann_n_rounds",
            "boltzmann_defer_budget",
            "kg_inner_set_size",
            "kg_raw_samples",
            "kg_n_elites",
            "kg_samples_per_elite",
        )
        non_negative_int_keys = ("sampling_n_rounds", "boltzmann_mh_steps", "kg_refine_top", "kg_n_rounds")
        for key in positive_int_keys + non_negative_int_keys:
            value = settings[key]
            minimum = 1 if key in positive_int_keys else 0
            if value is None or isinstance(value, bool) or int(value) != value or int(value) < minimum:
                kind = "a positive" if minimum == 1 else "a non-negative"
                raise ValueError(f"optimization.{key} must be {kind} integer.")
            settings[key] = int(value)
        for key in ("sampling_init_halfwidth", "sampling_min_halfwidth", "boltzmann_lambda"):
            if settings[key] is None or isinstance(settings[key], bool) or float(settings[key]) <= 0:
                raise ValueError(f"optimization.{key} must be positive.")
            settings[key] = float(settings[key])
        for key in ("sampling_shrink", "sampling_joint_member_prob"):
            if settings[key] is None or isinstance(settings[key], bool) or not 0 < float(settings[key]) <= 1:
                raise ValueError(f"optimization.{key} must be in (0, 1].")
            settings[key] = float(settings[key])
        if settings["sampling_subset_prob"] is not None:
            value = settings["sampling_subset_prob"]
            if isinstance(value, bool) or not 0 < float(value) <= 1:
                raise ValueError("optimization.sampling_subset_prob must be null or in (0, 1].")
            settings["sampling_subset_prob"] = float(value)

    @staticmethod
    def _available_cpu_count() -> int:
        if hasattr(os, "sched_getaffinity"):
            return len(os.sched_getaffinity(0))
        for env_var in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
            value = os.environ.get(env_var)
            if value:
                return int(value)
        return psutil.cpu_count(logical=False) or os.cpu_count() or 1

    @classmethod
    def _resolve_auto_torch_num_threads(cls, fraction: float) -> int:
        available_cpus = cls._available_cpu_count()
        if available_cpus <= 8:
            return available_cpus
        return max(1, int(fraction * available_cpus))

    def _configure_torch_threads(self, settings: AcquisitionOptimizationSettings) -> None:
        if self.device != "cpu":
            return
        torch_num_threads = settings.torch_num_threads
        if torch_num_threads == "auto":
            if any(os.environ.get(env_var) for env_var in OPTIMIZATION_THREAD_ENV_VARS):
                torch_num_threads = None
            else:
                torch_num_threads = self._resolve_auto_torch_num_threads(settings.torch_num_threads_fraction)
        if torch_num_threads is not None:
            torch.set_num_threads(torch_num_threads)
        if settings.torch_num_interop_threads is not None:
            try:
                torch.set_num_interop_threads(settings.torch_num_interop_threads)
            except RuntimeError as exc:
                warnings.warn(f"Could not set torch interop threads: {exc}", RuntimeWarning)
        if self.verbose:
            print(
                f"Torch threads: intraop={torch.get_num_threads()}, "
                f"interop={torch.get_num_interop_threads()}"
            )

    def _make_acq_options(self, batch_limit: int, maxiter: int) -> Dict[str, int]:
        return {"batch_limit": batch_limit, "maxiter": maxiter, "seed": self.random_seed}

    def _optimize_acqf(self, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._active_acq_optimizer() == "gradient":
            return optimize_acqf(**kwargs)
        return self._optimize_acqf_gradient_free(**kwargs)

    def _optimize_acqf_mixed(self, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._active_acq_optimizer() == "gradient":
            return optimize_acqf_mixed(**kwargs)
        return self._optimize_acqf_gradient_free(**kwargs)

    @contextmanager
    def _using_optimization_settings(self, settings: AcquisitionOptimizationSettings):
        """Make ``settings`` visible to ``_optimize_acqf`` for the duration of one candidate generation call."""
        previous = getattr(self, "_active_optimization_settings", None)
        self._active_optimization_settings = settings
        try:
            yield settings
        finally:
            self._active_optimization_settings = previous

    def _active_acq_optimizer(self) -> str:
        """The acquisition optimizer for the candidate generation call in progress."""
        settings = getattr(self, "_active_optimization_settings", None)
        return settings.acq_optimizer if settings is not None else "gradient"

    def _optimize_acqf_gradient_free(
        self,
        acq_function: Any,
        bounds: torch.Tensor,
        q: int,
        num_restarts: Optional[int] = None,
        raw_samples: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
        sequential: bool = False,
        fixed_features: Optional[Dict[int, float]] = None,
        fixed_features_list: Optional[List[Dict[int, float]]] = None,
        inequality_constraints: Optional[List[Tuple[torch.Tensor, torch.Tensor, float]]] = None,
        equality_constraints: Optional[List[Tuple[torch.Tensor, torch.Tensor, float]]] = None,
        nonlinear_inequality_constraints: Optional[List[Callable[[torch.Tensor], torch.Tensor]]] = None,
        post_processing_func: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        batch_initial_conditions: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gradient-free replacement for ``optimize_acqf``/``optimize_acqf_mixed``.

        ``num_restarts``, ``raw_samples``, and the scipy ``options`` of the gradient
        path are ignored; budgets come from the ``sampling_*``/``boltzmann_*``/``kg_*``
        optimization settings.
        """
        settings = self._active_optimization_settings
        sampling_options = SamplingOptions.from_settings(settings)
        if isinstance(acq_function, SampledKnowledgeGradient):
            sampling_options = sampling_options.for_kg_outer()
        seed = (options or {}).get("seed", self.random_seed)
        if fixed_features is not None:
            if fixed_features_list is not None:
                raise ValueError("Pass fixed_features or fixed_features_list, not both.")
            fixed_features_list = [fixed_features]
        return optimize_acqf_gradient_free(
            acq_function=acq_function,
            bounds=bounds,
            q=q,
            method=settings.acq_optimizer,
            options=sampling_options,
            sequential=sequential,
            seed=seed,
            fixed_features_list=fixed_features_list,
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
            nonlinear_inequality_constraints=nonlinear_inequality_constraints,
            post_processing_func=post_processing_func,
            batch_initial_conditions=batch_initial_conditions,
        )

    def _normalize_optimization_constraints(
        self,
        optimization_constraints: Optional[AcquisitionOptimizationConstraints] = None,
        inequality_constraints: Optional[List[Tuple[Sequence[int], Sequence[float], float]]] = None,
        equality_constraints: Optional[List[Tuple[Sequence[int], Sequence[float], float]]] = None,
        nonlinear_inequality_constraints: Optional[List[Callable[[torch.Tensor], torch.Tensor]]] = None,
        post_processing_func: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        batch_initial_conditions: Optional[torch.Tensor] = None,
        nonlinear_mode: Optional[str] = None,
        nonlinear_initial_raw_samples: Optional[int] = None,
        nonlinear_initial_max_tries: Optional[int] = None,
        nonlinear_optimization_retries: Optional[int] = None,
        optimization_settings: Optional[AcquisitionOptimizationSettings] = None,
    ) -> Optional[AcquisitionOptimizationConstraints]:
        direct_constraints = any(
            value is not None
            for value in (
                inequality_constraints,
                equality_constraints,
                nonlinear_inequality_constraints,
                post_processing_func,
                batch_initial_conditions,
            )
        )
        if optimization_constraints is not None and direct_constraints:
            raise ValueError("Pass either optimization_constraints or direct constraint arguments, not both.")
        if optimization_constraints is None and not direct_constraints:
            return None
        if optimization_settings is None:
            optimization_settings = self._resolve_optimization_settings()
        constraints = optimization_constraints or AcquisitionOptimizationConstraints(
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
            nonlinear_inequality_constraints=nonlinear_inequality_constraints,
            post_processing_func=post_processing_func,
            batch_initial_conditions=batch_initial_conditions,
            nonlinear_mode=nonlinear_mode,
            nonlinear_initial_raw_samples=nonlinear_initial_raw_samples,
            nonlinear_initial_max_tries=nonlinear_initial_max_tries,
            nonlinear_optimization_retries=nonlinear_optimization_retries,
        )
        nonlinear_mode = constraints.nonlinear_mode or optimization_settings.nonlinear_mode
        nonlinear_initial_raw_samples = (
            constraints.nonlinear_initial_raw_samples
            if constraints.nonlinear_initial_raw_samples is not None
            else optimization_settings.nonlinear_initial_raw_samples
        )
        nonlinear_initial_max_tries = (
            constraints.nonlinear_initial_max_tries
            if constraints.nonlinear_initial_max_tries is not None
            else optimization_settings.nonlinear_initial_max_tries
        )
        nonlinear_optimization_retries = (
            constraints.nonlinear_optimization_retries
            if constraints.nonlinear_optimization_retries is not None
            else optimization_settings.nonlinear_optimization_retries
        )
        if nonlinear_mode not in {"enforce", "initialization_only"}:
            raise ValueError("nonlinear_mode must be 'enforce' or 'initialization_only'.")
        if nonlinear_initial_raw_samples is not None:
            if (
                isinstance(nonlinear_initial_raw_samples, (bool, np.bool_))
                or not isinstance(nonlinear_initial_raw_samples, (int, np.integer))
                or nonlinear_initial_raw_samples <= 0
            ):
                raise ValueError("nonlinear_initial_raw_samples must be null or a positive integer.")
            nonlinear_initial_raw_samples = int(nonlinear_initial_raw_samples)
        if (
            isinstance(nonlinear_initial_max_tries, (bool, np.bool_))
            or not isinstance(nonlinear_initial_max_tries, (int, np.integer))
            or nonlinear_initial_max_tries <= 0
        ):
            raise ValueError("nonlinear_initial_max_tries must be positive.")
        nonlinear_initial_max_tries = int(nonlinear_initial_max_tries)
        if (
            isinstance(nonlinear_optimization_retries, (bool, np.bool_))
            or not isinstance(nonlinear_optimization_retries, (int, np.integer))
            or nonlinear_optimization_retries < 0
        ):
            raise ValueError("nonlinear_optimization_retries must be a non-negative integer.")
        nonlinear_optimization_retries = int(nonlinear_optimization_retries)
        nonlinear_constraints = self._normalize_nonlinear_constraints(constraints.nonlinear_inequality_constraints)
        if nonlinear_constraints and self.multi_fidelity:
            raise NotImplementedError("Nonlinear inequality constraints are only supported for single-fidelity optimization.")
        return AcquisitionOptimizationConstraints(
            inequality_constraints=self._linear_constraints_to_model_units(constraints.inequality_constraints),
            equality_constraints=self._linear_constraints_to_model_units(constraints.equality_constraints),
            nonlinear_inequality_constraints=nonlinear_constraints,
            post_processing_func=constraints.post_processing_func,
            batch_initial_conditions=self._batch_initial_conditions_to_model_units(constraints.batch_initial_conditions),
            nonlinear_mode=nonlinear_mode,
            nonlinear_initial_raw_samples=nonlinear_initial_raw_samples,
            nonlinear_initial_max_tries=nonlinear_initial_max_tries,
            nonlinear_optimization_retries=nonlinear_optimization_retries,
        )

    def _normalize_nonlinear_constraints(
        self,
        nonlinear_constraints: Optional[Union[Callable[[torch.Tensor], torch.Tensor], Sequence[Callable[[torch.Tensor], torch.Tensor]]]],
    ) -> Optional[List[Callable[[torch.Tensor], torch.Tensor]]]:
        if nonlinear_constraints is None:
            return None
        if callable(nonlinear_constraints):
            nonlinear_constraints = [nonlinear_constraints]
        normalized = []
        for constraint in nonlinear_constraints:
            if not callable(constraint):
                raise ValueError("Each nonlinear inequality constraint must be callable.")

            def model_unit_constraint(X: torch.Tensor, constraint: Callable[[torch.Tensor], torch.Tensor] = constraint) -> torch.Tensor:
                return constraint(self.input_scaler.inverse_transform(X))

            normalized.append(model_unit_constraint)
        return normalized

    def _linear_constraints_to_model_units(
        self,
        constraints: Optional[List[Tuple[Sequence[int], Sequence[float], float]]],
    ) -> Optional[List[Tuple[torch.Tensor, torch.Tensor, float]]]:
        if constraints is None:
            return None
        converted = []
        for indices, coefficients, rhs in constraints:
            raw_indices = indices.tolist() if hasattr(indices, "tolist") else list(indices)
            if any(isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)) for index in raw_indices):
                raise ValueError("Linear constraint indices must be integers.")
            index_tensor = torch.as_tensor(raw_indices, dtype=torch.long, device=self.device).reshape(-1)
            coefficient_tensor = torch.as_tensor(coefficients, dtype=torch.float, device=self.device).reshape(-1)
            if index_tensor.numel() != coefficient_tensor.numel():
                raise ValueError("Linear constraint indices and coefficients must have the same length.")
            if index_tensor.numel() == 0:
                raise ValueError("Linear constraints must include at least one index.")
            if torch.any(index_tensor < 0) or torch.any(index_tensor >= self.input_dim):
                raise ValueError(f"Linear constraint indices must be in [0, {self.input_dim}).")
            rhs_value = float(rhs)
            scales = self.bounds[1, index_tensor] - self.bounds[0, index_tensor]
            offsets = self.bounds[0, index_tensor]
            # Constraint coefficients are authored in original units, while BoTorch optimizes scaled inputs.
            if self.multi_fidelity:
                fidelity_terms = index_tensor == self.input_dim - 1
                scales = torch.where(fidelity_terms, torch.ones_like(scales), scales)
                offsets = torch.where(fidelity_terms, torch.zeros_like(offsets), offsets)
            converted.append((index_tensor, coefficient_tensor * scales, rhs_value - torch.sum(coefficient_tensor * offsets).item()))
        return converted

    def _batch_initial_conditions_to_model_units(self, batch_initial_conditions: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if batch_initial_conditions is None:
            return None
        initial_conditions = torch.as_tensor(batch_initial_conditions, dtype=torch.float, device=self.device)
        if initial_conditions.shape[-1] != self.input_dim:
            raise ValueError(f"batch_initial_conditions last dimension must be {self.input_dim}.")
        if initial_conditions.ndim == 2:
            initial_conditions = initial_conditions.unsqueeze(-2)
        if initial_conditions.ndim != 3:
            raise ValueError("batch_initial_conditions must have shape num_restarts x q x input_dim or num_restarts x input_dim.")
        return self.input_scaler.transform(initial_conditions)

    @staticmethod
    def _has_nonlinear_constraints(optimization_constraints: Optional[AcquisitionOptimizationConstraints]) -> bool:
        return bool(optimization_constraints and optimization_constraints.nonlinear_inequality_constraints)

    def _optimization_constraint_kwargs(
        self,
        optimization_constraints: Optional[AcquisitionOptimizationConstraints],
        include_nonlinear: bool = True,
    ) -> Dict[str, Any]:
        if optimization_constraints is None:
            return {}
        kwargs: Dict[str, Any] = {}
        for key in ("inequality_constraints", "equality_constraints", "post_processing_func", "batch_initial_conditions"):
            value = getattr(optimization_constraints, key)
            if value is not None:
                kwargs[key] = value
        if include_nonlinear and optimization_constraints.nonlinear_inequality_constraints is not None:
            kwargs["nonlinear_inequality_constraints"] = optimization_constraints.nonlinear_inequality_constraints
        return kwargs

    def _make_constrained_acq_options(
        self,
        batch_limit: int,
        maxiter: int,
        optimization_constraints: Optional[AcquisitionOptimizationConstraints],
    ) -> Dict[str, int]:
        options = self._make_acq_options(batch_limit, maxiter)
        if (
            self._has_nonlinear_constraints(optimization_constraints)
            and optimization_constraints.nonlinear_mode == "enforce"
        ):
            if options["batch_limit"] != 1:
                warnings.warn(
                    "BoTorch requires batch_limit=1 when enforcing nonlinear inequality constraints; overriding batch_limit.",
                    RuntimeWarning,
                )
            # The BoTorch nonlinear-constraint optimizer requires one restart batch at a time.
            options["batch_limit"] = 1
        return options

    def _make_feasible_nonlinear_initial_conditions(
        self,
        acq_function: Any,
        bounds: torch.Tensor,
        q: int,
        num_restarts: int,
        raw_samples: int,
        options: Dict[str, int],
        optimization_constraints: AcquisitionOptimizationConstraints,
    ) -> torch.Tensor:
        if optimization_constraints.batch_initial_conditions is not None:
            return optimization_constraints.batch_initial_conditions
        raw_samples = optimization_constraints.nonlinear_initial_raw_samples or raw_samples
        required_points = num_restarts * q
        raw_samples = max(raw_samples, required_points)
        # Feasible starts keep initialization_only mode useful even when final optimization is unconstrained.
        seed = options.get("seed", self.random_seed)
        feasible_points = []
        for attempt in range(optimization_constraints.nonlinear_initial_max_tries):
            if optimization_constraints.inequality_constraints or optimization_constraints.equality_constraints:
                samples = gen_batch_initial_conditions(
                    acq_function=acq_function,
                    bounds=bounds,
                    q=1,
                    num_restarts=raw_samples,
                    raw_samples=raw_samples,
                    options={**options, "seed": seed + attempt},
                    inequality_constraints=optimization_constraints.inequality_constraints,
                    equality_constraints=optimization_constraints.equality_constraints,
                )
            else:
                samples = draw_sobol_samples(bounds=bounds.cpu(), n=raw_samples, q=1, seed=seed + attempt).to(self.device)
            feasible = self._nonlinear_feasible_mask(samples, optimization_constraints.nonlinear_inequality_constraints)
            if feasible.any():
                feasible_points.append(samples[feasible].squeeze(-2))
            if feasible_points and sum(points.shape[0] for points in feasible_points) >= required_points:
                break
        if not feasible_points:
            raise ValueError("Could not find nonlinear-feasible initial conditions.")
        feasible_samples = torch.cat(feasible_points, dim=0)
        if feasible_samples.shape[0] < required_points:
            raise ValueError(
                "Not enough nonlinear-feasible initial conditions found; increase nonlinear_initial_raw_samples "
                "or nonlinear_initial_max_tries, or relax the constraints."
            )
        num_batches = feasible_samples.shape[0] // q
        initial_batches = feasible_samples[: num_batches * q].reshape(num_batches, q, bounds.shape[-1])
        with torch.no_grad():
            values = acq_function(initial_batches).reshape(-1)
        top_indices = torch.topk(values, k=num_restarts).indices
        return initial_batches[top_indices]

    def _filter_candidate_set_for_constraints(
        self,
        candidate_set: torch.Tensor,
        optimization_constraints: Optional[AcquisitionOptimizationConstraints],
    ) -> torch.Tensor:
        if optimization_constraints is None:
            return candidate_set
        if optimization_constraints.equality_constraints:
            raise NotImplementedError("Equality constraints are not supported with entropy acquisition candidate sets.")
        feasible = torch.ones(candidate_set.shape[0], dtype=torch.bool, device=candidate_set.device)
        for indices, coefficients, rhs in optimization_constraints.inequality_constraints or []:
            feasible &= candidate_set[:, indices].matmul(coefficients) >= rhs
        for indices, coefficients, rhs in optimization_constraints.equality_constraints or []:
            feasible &= torch.isclose(candidate_set[:, indices].matmul(coefficients), torch.tensor(rhs, device=candidate_set.device, dtype=candidate_set.dtype))
        if optimization_constraints.nonlinear_inequality_constraints:
            feasible &= self._nonlinear_feasible_mask(
                candidate_set.unsqueeze(-2),
                optimization_constraints.nonlinear_inequality_constraints,
            )
        filtered = candidate_set[feasible]
        if filtered.numel() == 0:
            raise ValueError("No acquisition candidate_set points satisfy the optimization constraints.")
        return filtered

    @staticmethod
    def _nonlinear_feasible_mask(
        samples: torch.Tensor,
        nonlinear_constraints: Sequence[Callable[[torch.Tensor], torch.Tensor]],
    ) -> torch.Tensor:
        feasible = torch.ones(samples.shape[0], dtype=torch.bool, device=samples.device)
        for constraint in nonlinear_constraints:
            values = constraint(samples)
            constraint_feasible = values >= 0
            while constraint_feasible.ndim > 1:
                constraint_feasible = constraint_feasible.all(dim=-1)
            feasible &= constraint_feasible.to(samples.device)
        return feasible

    @staticmethod
    def _is_retry_suppressed_optimization_warning(warning_message: warnings.WarningMessage) -> bool:
        message = str(warning_message.message)
        return (
            "batch_initial_conditions" in message
            and "retried" in message
            and ("Optimization failed" in message or "scipy.optimize.minimize" in message)
        )

    @staticmethod
    def _reemit_warnings(caught_warnings: Sequence[warnings.WarningMessage]) -> None:
        for caught in caught_warnings:
            warnings.warn(caught.message, caught.category)

    def _constraint_kwargs_for_call(
        self,
        acq_function: Any,
        bounds: torch.Tensor,
        q: int,
        num_restarts: int,
        raw_samples: int,
        options: Dict[str, int],
        optimization_constraints: Optional[AcquisitionOptimizationConstraints],
    ) -> Tuple[Dict[str, Any], Dict[str, int]]:
        if optimization_constraints is None:
            return {}, options
        if not self._has_nonlinear_constraints(optimization_constraints):
            return self._optimization_constraint_kwargs(optimization_constraints), options
        seed = options.get("seed", self.random_seed)
        options = self._make_constrained_acq_options(options["batch_limit"], options["maxiter"], optimization_constraints)
        options["seed"] = seed
        batch_initial_conditions = self._make_feasible_nonlinear_initial_conditions(
            acq_function=acq_function,
            bounds=bounds,
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options=options,
            optimization_constraints=optimization_constraints,
        )
        kwargs = self._optimization_constraint_kwargs(
            optimization_constraints,
            include_nonlinear=optimization_constraints.nonlinear_mode == "enforce",
        )
        kwargs["batch_initial_conditions"] = batch_initial_conditions
        return kwargs, options

    def _optimize_acqf_constrained(
        self,
        optimization_constraints: Optional[AcquisitionOptimizationConstraints] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._active_acq_optimizer() != "gradient":
            # Gradient-free optimizers reject infeasible draws directly, including nonlinear constraints.
            return self._optimize_acqf(**kwargs, **self._optimization_constraint_kwargs(optimization_constraints))
        if kwargs.get("sequential", False) and kwargs["q"] > 1 and isinstance(kwargs["acq_function"], qKnowledgeGradient):
            # BoTorch cannot run one-shot KG sequentially; build the batch greedily with X_pending instead.
            return self._optimize_acqf_sequential_greedy(optimization_constraints, **kwargs)
        if not self._has_nonlinear_constraints(optimization_constraints):
            return self._optimize_acqf(**kwargs, **self._optimization_constraint_kwargs(optimization_constraints))
        if kwargs["q"] > 1 and (
            optimization_constraints.nonlinear_mode == "enforce" or kwargs.get("sequential", False)
        ):
            if isinstance(kwargs["acq_function"], qKnowledgeGradient):
                raise NotImplementedError("Nonlinear constraints with KG currently require num_candidates=1.")
            return self._optimize_acqf_nonlinear_sequential(optimization_constraints, **kwargs)
        attempts = 1
        if optimization_constraints.batch_initial_conditions is None:
            attempts += optimization_constraints.nonlinear_optimization_retries
        base_options = dict(kwargs["options"])
        base_seed = base_options.get("seed", self.random_seed)
        for attempt in range(attempts):
            attempt_options = dict(base_options)
            attempt_options["seed"] = base_seed + attempt * optimization_constraints.nonlinear_initial_max_tries
            constraint_kwargs, options = self._constraint_kwargs_for_call(
                acq_function=kwargs["acq_function"],
                bounds=kwargs["bounds"],
                q=kwargs["q"],
                num_restarts=kwargs["num_restarts"],
                raw_samples=kwargs["raw_samples"],
                options=attempt_options,
                optimization_constraints=optimization_constraints,
            )
            call_kwargs = dict(kwargs)
            call_kwargs["options"] = options
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                result = self._optimize_acqf(**call_kwargs, **constraint_kwargs)
            retry_warning = any(
                self._is_retry_suppressed_optimization_warning(warning_message)
                for warning_message in caught_warnings
            )
            if retry_warning and attempt < attempts - 1:
                self._reemit_warnings(
                    [
                        warning_message
                        for warning_message in caught_warnings
                        if not self._is_retry_suppressed_optimization_warning(warning_message)
                    ]
                )
                warnings.warn(
                    "Retrying nonlinear acquisition optimization with regenerated feasible initial conditions.",
                    RuntimeWarning,
                )
                continue
            self._reemit_warnings(caught_warnings)
            return result
        raise RuntimeError("Unreachable nonlinear acquisition optimization retry state.")

    def _optimize_acqf_sequential_greedy(
        self,
        optimization_constraints: Optional[AcquisitionOptimizationConstraints],
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Greedy sequential q-batch construction with ``X_pending`` for any acquisition supporting it."""
        acq_function = kwargs["acq_function"]
        base_X_pending = getattr(acq_function, "X_pending", None)
        candidate_list, acq_value_list = [], []
        try:
            for _ in range(kwargs["q"]):
                step_kwargs = dict(kwargs, q=1, sequential=False)
                candidate, acq_value = self._optimize_acqf_constrained(optimization_constraints, **step_kwargs)
                candidate_list.append(candidate)
                acq_value_list.append(acq_value.reshape(()))
                candidates = torch.cat(candidate_list, dim=-2)
                acq_function.set_X_pending(
                    torch.cat([base_X_pending, candidates], dim=-2) if base_X_pending is not None else candidates
                )
        finally:
            acq_function.set_X_pending(base_X_pending)
        return torch.cat(candidate_list, dim=-2), torch.stack(acq_value_list)

    def _optimize_acqf_nonlinear_sequential(
        self,
        optimization_constraints: AcquisitionOptimizationConstraints,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        acq_function = kwargs["acq_function"]
        if not hasattr(acq_function, "set_X_pending"):
            raise NotImplementedError("Sequential nonlinear optimization requires an acquisition function with set_X_pending.")
        q = kwargs["q"]
        base_X_pending = getattr(acq_function, "X_pending", None)
        candidate_list = []
        acq_value_list = []
        try:
            for _ in range(q):
                step_kwargs = dict(kwargs)
                step_kwargs["q"] = 1
                step_kwargs["sequential"] = False
                candidate, acq_value = self._optimize_acqf_constrained(optimization_constraints, **step_kwargs)
                candidate_list.append(candidate)
                acq_value_list.append(acq_value)
                candidates = torch.cat(candidate_list, dim=-2)
                acq_function.set_X_pending(
                    torch.cat([base_X_pending, candidates], dim=-2) if base_X_pending is not None else candidates
                )
        finally:
            acq_function.set_X_pending(base_X_pending)
        return torch.cat(candidate_list, dim=-2), torch.stack(acq_value_list)

    def _optimize_acqf_mixed_constrained(
        self,
        optimization_constraints: Optional[AcquisitionOptimizationConstraints] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._has_nonlinear_constraints(optimization_constraints):
            raise NotImplementedError("Nonlinear inequality constraints are not supported for multi-fidelity mixed optimization.")
        return self._optimize_acqf_mixed(**kwargs, **self._optimization_constraint_kwargs(optimization_constraints))

    def _resolve_sequential(self, acq_method: str, settings: AcquisitionOptimizationSettings, multi_fidelity: bool = False) -> bool:
        """
        Map ``batch_mode`` to the ``sequential`` flag for a q-batch.

        ``'default'`` keeps the historical rules: MaxValEntropy/GIBBON are sequential, EI/NEI/KG are
        joint in single-fidelity runs, and multi-fidelity mixed optimization is always sequential.
        """
        if settings.acq_optimizer == "boltzmann":
            if settings.batch_mode != "default":
                warnings.warn(
                    "batch_mode is ignored by acq_optimizer='boltzmann', which always draws i.i.d. batch members.",
                    RuntimeWarning,
                )
            return False
        if settings.batch_mode == "joint":
            if acq_method in ("MaxValEntropy", "GIBBON"):
                raise ValueError(f"batch_mode='joint' is not supported for {acq_method}, whose value is pointwise; use 'sequential'.")
            if multi_fidelity:
                raise NotImplementedError("batch_mode='joint' is not supported for multi-fidelity runs; use 'sequential'.")
            return False
        if settings.batch_mode == "sequential" or multi_fidelity:
            return True
        return acq_method in ("MaxValEntropy", "GIBBON")

    def _sampled_kg_inner_set(
        self,
        model: Type[Model],
        settings: AcquisitionOptimizationSettings,
        bounds: torch.Tensor,
        value_acqf: Any,
        fixed_features_list: Optional[List[Dict[int, float]]],
        optimization_constraints: Optional[AcquisitionOptimizationConstraints],
    ) -> torch.Tensor:
        """
        Shared discrete inner set for gradient-free KG.

        It combines posterior-mean elites from a scoped search, the (feasible) training
        inputs, and feasible Sobol points, about ``kg_inner_set_size`` points in total.
        Multi-fidelity inner sets are pinned to the target fidelity.
        """
        sampling_options = SamplingOptions.from_settings(settings)
        constraint_kwargs = self._optimization_constraint_kwargs(optimization_constraints)
        constraint_kwargs = {
            key: constraint_kwargs[key]
            for key in ("inequality_constraints", "equality_constraints", "nonlinear_inequality_constraints")
            if key in constraint_kwargs
        }
        size = settings.kg_inner_set_size
        n_elites = max(1, size // 4)
        elites = search_elites(
            value_acqf, bounds, n_elites, sampling_options, self.random_seed + 17,
            fixed_features_list=fixed_features_list, **constraint_kwargs,
        )
        space = SearchSpace(bounds=bounds, fixed_features_list=fixed_features_list, **constraint_kwargs)
        train_X = self.full_train_X.to(bounds)
        if fixed_features_list:
            train_X = space.to_x(space.to_u(train_X))
        train_X = train_X[space.feasible(train_X)]
        n_sobol = max(0, size - elites.shape[0] - train_X.shape[0])
        parts = [elites, train_X]
        if n_sobol > 0:
            parts.append(draw_feasible_points(
                bounds, n_sobol, self.random_seed + 29, fixed_features_list=fixed_features_list, **constraint_kwargs
            ))
        return torch.cat(parts).detach()

    def _sampled_kg_inner_search(
        self,
        settings: AcquisitionOptimizationSettings,
        bounds: torch.Tensor,
        fixed_features_list: Optional[List[Dict[int, float]]],
        optimization_constraints: Optional[AcquisitionOptimizationConstraints],
    ) -> Callable[..., torch.Tensor]:
        """Per-fantasy scoped maximization used to refine the final gradient-free KG candidates."""
        sampling_options = SamplingOptions.from_settings(settings).for_kg_outer()
        constraint_kwargs = self._optimization_constraint_kwargs(optimization_constraints)
        inequality_constraints = constraint_kwargs.get("inequality_constraints")
        nonlinear_constraints = constraint_kwargs.get("nonlinear_inequality_constraints")
        if constraint_kwargs.get("equality_constraints"):
            warnings.warn("Gradient-free KG refinement ignores equality constraints; the discrete inner set still respects them.", RuntimeWarning)

        def inner_search(value_fn: Callable[[torch.Tensor], torch.Tensor], inner_bounds: torch.Tensor, n_fantasies: int) -> torch.Tensor:
            return batched_max_search(
                value_fn, inner_bounds, n_fantasies, sampling_options, self.random_seed + 31,
                fixed_features_list=fixed_features_list,
                inequality_constraints=inequality_constraints,
                nonlinear_inequality_constraints=nonlinear_constraints,
            )

        return inner_search

    def _make_sampled_kg(
        self,
        model: Type[Model],
        settings: AcquisitionOptimizationSettings,
        bounds: torch.Tensor,
        current_value: torch.Tensor,
        risk_objective: Optional[Type[RiskMeasureMCObjective]],
        risk_n_deltas: Optional[int],
        optimization_constraints: Optional[AcquisitionOptimizationConstraints],
        cost_aware_utility: Optional[InverseCostWeightedUtility] = None,
    ) -> SampledKnowledgeGradient:
        """Build the gradient-free KG used when ``acq_optimizer`` is not ``'gradient'``."""
        n_fantasies = settings.n_fantasies
        if getattr(model, "fantasize_refits", False) and n_fantasies > 16:
            warnings.warn(
                f"Gradient-free KG refits {n_fantasies} fantasy models per evaluation for this surrogate; consider a "
                "smaller optimization.n_fantasies (e.g. 8-16) and/or fantasy_n_jobs > 1 to reduce run time.",
                RuntimeWarning,
            )
        fixed_features_list = None
        project = None
        if self.multi_fidelity:
            fixed_features_list = [{self.input_dim - 1: float(self.num_fidelities - 1)}]
            project = self._project
        value_acqf = PosteriorMean(
            model,
            posterior_transform=ExpectationPosteriorTransform(n_w=risk_n_deltas) if risk_objective else None,
        )
        inner_set = self._sampled_kg_inner_set(model, settings, bounds, value_acqf, fixed_features_list, optimization_constraints)
        return SampledKnowledgeGradient(
            model=model,
            inner_set=inner_set,
            sampler=make_sobol_qmc_normal_sampler(n_fantasies, seed=self.random_seed),
            inner_sampler=make_sobol_qmc_normal_sampler(n_fantasies, seed=self.random_seed),
            current_value=current_value,
            objective=risk_objective,
            cost_aware_utility=cost_aware_utility,
            project=project,
            inner_bounds=bounds,
            inner_search=self._sampled_kg_inner_search(settings, bounds, fixed_features_list, optimization_constraints),
            eval_batch=max(1, settings.sampling_eval_batch // max(1, n_fantasies)),
        )

    def _fixed_feature_subproblem_constraints(
        self,
        optimization_constraints: Optional[AcquisitionOptimizationConstraints],
        fixed_features: Dict[int, float],
    ) -> Optional[AcquisitionOptimizationConstraints]:
        if optimization_constraints is None:
            return None
        if self._has_nonlinear_constraints(optimization_constraints):
            raise NotImplementedError("Nonlinear inequality constraints are not supported for fixed-feature subproblems.")

        def reduce_constraints(constraints, equality: bool = False):
            if constraints is None:
                return None
            reduced = []
            fixed_indices = sorted(fixed_features)
            for indices, coefficients, rhs in constraints:
                kept_indices = []
                kept_coefficients = []
                adjusted_rhs = rhs
                for index, coefficient in zip(indices.tolist(), coefficients.tolist()):
                    if index in fixed_features:
                        adjusted_rhs -= coefficient * fixed_features[index]
                    else:
                        kept_indices.append(index - sum(fixed_index < index for fixed_index in fixed_indices))
                        kept_coefficients.append(coefficient)
                if kept_indices:
                    reduced.append((torch.tensor(kept_indices, dtype=torch.long, device=self.device), torch.tensor(kept_coefficients, dtype=torch.float, device=self.device), adjusted_rhs))
                elif (equality and abs(adjusted_rhs) > 1e-7) or (not equality and adjusted_rhs > 0):
                    raise ValueError("Fixed-feature subproblem cannot satisfy a linear constraint.")
            return reduced or None

        return AcquisitionOptimizationConstraints(
            inequality_constraints=reduce_constraints(optimization_constraints.inequality_constraints),
            equality_constraints=reduce_constraints(optimization_constraints.equality_constraints, equality=True),
        )

    def _get_candidates_mf(
        self,
        model: Type[Model],
        acq_method: str,
        q: int,
        fidelity_cost: np.ndarray,
        risk_objective: Optional[Type[RiskMeasureMCObjective]] = None,
        risk_n_deltas: Optional[int] = None,
        optimization_settings: Optional[AcquisitionOptimizationSettings] = None,
        propose_best: Optional[bool] = False,
        optimization_constraints: Optional[AcquisitionOptimizationConstraints] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate scaled candidates for a multi-fidelity run.

        The last column is the integer fidelity index. Public CLI usage supports
        ``KG`` and ``MaxValEntropy``; ``GIBBON`` is an internal/API entropy-style
        option handled by the same mixed-optimization path.

        :param model: The model loaded by ``load_model``.
        :param acq_method: ``KG``, ``MaxValEntropy``, or internal ``GIBBON``.
        :param q: Number of candidates to generate.
        :param fidelity_cost: Cost weights indexed by rounded fidelity.
        :param risk_objective: Optional BoTorch VaR/CVaR objective applied to posterior samples.
        :param risk_n_deltas: Number of input perturbations used by the risk objective.
        :param optimization_settings: Resolved or explicit acquisition optimizer settings.
        :param propose_best: Whether to prepend the target-fidelity posterior maximizer.
        :param optimization_constraints: Optional linear optimization constraints.
        :returns: ``(candidates, acq_value)`` in scaled model coordinates.
        """
        settings = self._resolve_optimization_settings(optimization_settings)
        self._check_gradient_support(model, settings)
        with self._using_optimization_settings(settings):
            return self._get_candidates_mf_impl(
                model, acq_method, q, fidelity_cost, risk_objective, risk_n_deltas, settings, propose_best,
                optimization_constraints,
            )

    def _get_candidates_mf_impl(
        self,
        model: Type[Model],
        acq_method: str,
        q: int,
        fidelity_cost: np.ndarray,
        risk_objective: Optional[Type[RiskMeasureMCObjective]],
        risk_n_deltas: Optional[int],
        settings: AcquisitionOptimizationSettings,
        propose_best: Optional[bool],
        optimization_constraints: Optional[AcquisitionOptimizationConstraints],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_fantasies = settings.n_fantasies
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

            subproblem_constraints = self._fixed_feature_subproblem_constraints(
                optimization_constraints,
                {self.input_dim - 1: self.num_fidelities - 1},
            )
            best_candidate, max_pmean = self._optimize_acqf_constrained(
                acq_function=curr_val_acqf,
                bounds=bounds[:, :-1],
                q=1,
                num_restarts=settings.num_restarts_high,
                raw_samples=settings.raw_samples_high,
                options=self._make_constrained_acq_options(
                    settings.batch_limit_high,
                    settings.maxiter,
                    subproblem_constraints,
                ),
                optimization_constraints=subproblem_constraints,
            )
            q-=1
            if q==0:
                acq_value = max_pmean
                print(f"{acq_value = }")
                candidates = torch.concat([best_candidate.reshape(1,-1),(self.num_fidelities-1)*torch.ones(1,1,device=self.device)],axis=1)
                return candidates, acq_value


        if acq_method in ("GIBBON", "MaxValEntropy"):
            n_candidates = 2000 * self.num_fidelities
            candidate_set = torch.rand(n_candidates, self.input_dim,device=self.device)
            candidate_set[:, -1] *= self.num_fidelities - 1
            candidate_set[:, -1] = candidate_set[:, -1].round()
            candidate_set = self._filter_candidate_set_for_constraints(candidate_set, optimization_constraints)
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
            candidates, acq_value = self._optimize_mf_batch(
                acq_function=q_acq,
                bounds=bounds,
                fixed_features_list=[{self.input_dim - 1: i} for i in range(self.num_fidelities)],
                q=q,
                acq_method=acq_method,
                settings=settings,
                num_restarts=settings.num_restarts_high,
                raw_samples=settings.raw_samples_high,
                options=self._make_constrained_acq_options(
                    settings.batch_limit_high,
                    settings.maxiter,
                    optimization_constraints,
                ),
                optimization_constraints=optimization_constraints,
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

                subproblem_constraints = self._fixed_feature_subproblem_constraints(
                    optimization_constraints,
                    {self.input_dim - 1: self.num_fidelities - 1},
                )
                _, max_pmean = self._optimize_acqf_constrained(
                    acq_function=curr_val_acqf,
                    bounds=bounds[:, :-1],
                    q=1,
                    num_restarts=settings.num_restarts_high,
                    raw_samples=settings.raw_samples_high,
                    options=self._make_constrained_acq_options(
                        settings.batch_limit_high,
                        settings.maxiter,
                        subproblem_constraints,
                    ),
                    optimization_constraints=subproblem_constraints,
                )
            

            if settings.acq_optimizer == "gradient":
                mfkg_acqf = qMultiFidelityKnowledgeGradient(
                    model=model,
                    num_fantasies=n_fantasies,
                    sampler=make_sobol_qmc_normal_sampler(n_fantasies, seed=self.random_seed),
                    inner_sampler=make_sobol_qmc_normal_sampler(n_fantasies, seed=self.random_seed),
                    current_value=max_pmean,
                    cost_aware_utility=cost_aware_utility,
                    project=self._project,
                    objective=risk_objective,
                )
            else:
                mfkg_acqf = self._make_sampled_kg(
                    model=model,
                    settings=settings,
                    bounds=bounds,
                    current_value=max_pmean,
                    risk_objective=risk_objective,
                    risk_n_deltas=risk_n_deltas,
                    optimization_constraints=optimization_constraints,
                    cost_aware_utility=cost_aware_utility,
                )
            candidates, acq_value = self._optimize_mf_batch(
                acq_function=mfkg_acqf,
                bounds=bounds,
                fixed_features_list=[{self.input_dim - 1: i} for i in range(self.num_fidelities)],
                q=q,
                acq_method=acq_method,
                settings=settings,
                num_restarts=settings.num_restarts_low,
                raw_samples=settings.raw_samples_low,
                options=self._make_constrained_acq_options(
                    settings.batch_limit_low,
                    settings.maxiter,
                    optimization_constraints,
                ),
                optimization_constraints=optimization_constraints,
            )
        else:
            raise ValueError(f"Unsupported multi-fidelity acquisition method {acq_method}. Options: 'KG', 'MaxValEntropy', or 'GIBBON'.")
        if propose_best:
            best_candidate = torch.concat([best_candidate.reshape(1,-1),(self.num_fidelities-1)*torch.ones(1,1,device=self.device)],axis=1)
            candidates = torch.concat([best_candidate,candidates],axis=0)

        print(f"{acq_value = }")
        return candidates, acq_value

    def _optimize_mf_batch(
        self,
        acq_method: str,
        settings: AcquisitionOptimizationSettings,
        optimization_constraints: Optional[AcquisitionOptimizationConstraints] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Multi-fidelity mixed optimization honoring ``batch_mode`` (sequential only) and gradient-free optimizers."""
        sequential = self._resolve_sequential(acq_method, settings, multi_fidelity=True)
        if settings.acq_optimizer != "gradient":
            # Gradient-free mixed search handles the fidelity choice inside one search space.
            return self._optimize_acqf_mixed_constrained(
                optimization_constraints=optimization_constraints, sequential=sequential, **kwargs
            )
        if kwargs["q"] > 1 and isinstance(kwargs["acq_function"], qKnowledgeGradient) and settings.batch_mode == "sequential":
            return self._optimize_mixed_sequential_greedy(optimization_constraints, **kwargs)
        return self._optimize_acqf_mixed_constrained(optimization_constraints=optimization_constraints, **kwargs)

    def _optimize_mixed_sequential_greedy(
        self,
        optimization_constraints: Optional[AcquisitionOptimizationConstraints],
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Greedy multi-fidelity batch for one-shot KG (BoTorch's mixed optimizer evaluates KG jointly otherwise)."""
        acq_function = kwargs["acq_function"]
        base_X_pending = getattr(acq_function, "X_pending", None)
        candidate_list, acq_value_list = [], []
        try:
            for _ in range(kwargs["q"]):
                candidate, acq_value = self._optimize_acqf_mixed_constrained(
                    optimization_constraints=optimization_constraints, **dict(kwargs, q=1)
                )
                candidate_list.append(candidate)
                acq_value_list.append(acq_value.reshape(()))
                candidates = torch.cat(candidate_list, dim=-2)
                acq_function.set_X_pending(
                    torch.cat([base_X_pending, candidates], dim=-2) if base_X_pending is not None else candidates
                )
        finally:
            acq_function.set_X_pending(base_X_pending)
        return torch.cat(candidate_list, dim=-2), torch.stack(acq_value_list)

    def _get_candidates_sf(
        self,
        model: Type[Model],
        acq_method: str,
        q: int,
        risk_objective: Optional[Type[RiskMeasureMCObjective]] = None,
        risk_n_deltas: Optional[int] = None,
        optimization_settings: Optional[AcquisitionOptimizationSettings] = None,
        propose_best: Optional[bool] = False,
        optimization_constraints: Optional[AcquisitionOptimizationConstraints] = None,
    ) -> Tuple[Any, Any]:
        """
        Generate scaled candidates for a single-fidelity run.

        :param model: The model loaded by ``load_model``.
        :param acq_method: ``EI``, ``NEI``, ``MaxValEntropy``, or ``KG``.
        :param q: Number of candidates to generate.
        :param risk_objective: Optional BoTorch VaR/CVaR objective applied to posterior samples.
        :param risk_n_deltas: Number of input perturbations used by the risk objective.
        :param optimization_settings: Resolved or explicit acquisition optimizer settings.
        :param propose_best: Whether to prepend the current posterior maximizer.
        :param optimization_constraints: Optional linear/nonlinear optimization constraints.
        :returns: ``(candidates, acq_value)`` in scaled model coordinates.
        """
        settings = self._resolve_optimization_settings(optimization_settings)
        self._check_gradient_support(model, settings)
        with self._using_optimization_settings(settings):
            return self._get_candidates_sf_impl(
                model, acq_method, q, risk_objective, risk_n_deltas, settings, propose_best, optimization_constraints
            )

    def _get_candidates_sf_impl(
        self,
        model: Type[Model],
        acq_method: str,
        q: int,
        risk_objective: Optional[Type[RiskMeasureMCObjective]],
        risk_n_deltas: Optional[int],
        settings: AcquisitionOptimizationSettings,
        propose_best: Optional[bool],
        optimization_constraints: Optional[AcquisitionOptimizationConstraints],
    ) -> Tuple[Any, Any]:
        n_fantasies = settings.n_fantasies
        bounds = torch.tensor(self.input_dim * [[0, 1]],dtype=torch.float,device=self.device).T
        if (
            acq_method == "KG"
            and self._has_nonlinear_constraints(optimization_constraints)
            and settings.acq_optimizer == "gradient"
        ):
            raise NotImplementedError("Nonlinear constraints are not supported with KG acquisition optimization.")

        if propose_best:
            best_candidate, max_pmean = self._optimize_acqf_constrained(
                acq_function=PosteriorMean(
                    model,
                    posterior_transform=ExpectationPosteriorTransform(n_w=risk_n_deltas) if risk_objective else None,
                ),
                bounds=bounds,
                q=1,
                num_restarts=settings.num_restarts_high,
                raw_samples=settings.raw_samples_high,
                options=self._make_constrained_acq_options(
                    settings.batch_limit_high,
                    settings.maxiter,
                    optimization_constraints,
                ),
                optimization_constraints=optimization_constraints,
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
            candidate_set = self._filter_candidate_set_for_constraints(candidate_set, optimization_constraints)
            q_acq = qMaxValueEntropy(
                model,
                posterior_transform=ExpectationPosteriorTransform(n_w=risk_n_deltas) if risk_objective else None,
                candidate_set=candidate_set,
                num_fantasies=n_fantasies,
                seed=self.random_seed,
            )
        elif acq_method == "KG":
            if not propose_best:
                _, max_pmean = self._optimize_acqf_constrained(
                    acq_function=PosteriorMean(
                        model,
                        posterior_transform=ExpectationPosteriorTransform(n_w=risk_n_deltas) if risk_objective else None,
                    ),
                    bounds=bounds,
                    q=1,
                    num_restarts=settings.num_restarts_high,
                    raw_samples=settings.raw_samples_high,
                    options=self._make_constrained_acq_options(
                        settings.batch_limit_high,
                        settings.maxiter,
                        optimization_constraints,
                    ),
                    optimization_constraints=optimization_constraints,
                )
            
            if settings.acq_optimizer == "gradient":
                q_acq = qKnowledgeGradient(
                    model=model,
                    num_fantasies=n_fantasies,
                    sampler=make_sobol_qmc_normal_sampler(n_fantasies, seed=self.random_seed),
                    inner_sampler=make_sobol_qmc_normal_sampler(n_fantasies, seed=self.random_seed),
                    current_value=max_pmean,
                    objective=risk_objective,
                )
            else:
                q_acq = self._make_sampled_kg(
                    model=model,
                    settings=settings,
                    bounds=bounds,
                    current_value=max_pmean,
                    risk_objective=risk_objective,
                    risk_n_deltas=risk_n_deltas,
                    optimization_constraints=optimization_constraints,
                )
        else:
            raise ValueError(f"Unsupported acquisition method {acq_method}. Options: 'EI', 'NEI', 'KG', or 'MaxValEntropy'.")
        is_expensive_acq = acq_method in ["MaxValEntropy", "KG"]
        candidates, acq_value = self._optimize_acqf_constrained(
            acq_function=q_acq,
            bounds=bounds,
            q=q,
            num_restarts=settings.num_restarts_low if is_expensive_acq else settings.num_restarts_high,
            raw_samples=settings.raw_samples_low if is_expensive_acq else settings.raw_samples_high,
            sequential=self._resolve_sequential(acq_method, settings),
            options=self._make_constrained_acq_options(
                settings.batch_limit_low if is_expensive_acq else settings.batch_limit_high,
                settings.maxiter,
                optimization_constraints,
            ),
            optimization_constraints=optimization_constraints,
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
        optimization_settings: Optional[AcquisitionOptimizationSettings] = None,
        propose_best: Optional[bool] = False,
        optimization_constraints: Optional[AcquisitionOptimizationConstraints] = None,
    ) -> Tuple[Any, Any]:
        """
        Generate scaled candidate tensors with the requested acquisition method.

        Single-fidelity runs support ``EI``, ``NEI``, ``KG``, and ``MaxValEntropy``.
        Multi-fidelity CLI runs support ``KG`` and ``MaxValEntropy``; ``GIBBON`` is
        available through the Python API entropy path. Candidate tensors returned
        from this method are still in model coordinates; ``optimize`` converts them
        back to original units before saving. Risk objectives may be used with EI,
        NEI, and KG, but not MaxValEntropy. ``propose_best=True`` reserves the first
        candidate for the current posterior maximizer and acquires the remaining
        ``q - 1`` points with ``acq_method``.

        :param model: The model loaded by ``load_model``.
        :param acq_method: Acquisition method name.
        :param q: Number of candidates to generate.
        :param risk_objective: Optional BoTorch VaR/CVaR objective applied to posterior samples.
        :param risk_n_deltas: Number of input perturbations used by the risk objective.
        :param fidelity_cost: Cost weights indexed by rounded fidelity for multi-fidelity runs.
        :param optimization_settings: Resolved or explicit acquisition optimizer settings.
        :param propose_best: Whether to prepend the current posterior maximizer.
        :param optimization_constraints: Optional linear/nonlinear optimization constraints.
        :returns: ``(candidates, acq_value)`` in scaled model coordinates.
        """

        current_max = self.full_train_Y[self.full_train_X[:,-1]==(self.num_fidelities-1)].max(
            ) if self.multi_fidelity else self.full_train_Y.max()
        print(f"Number of simulations: {len(self.full_train_X)}. Current max: {current_max.item():.5f}")
        settings = self._resolve_optimization_settings(optimization_settings)
        model_type = self.config_settings.get_setting("model_type")
        if model_type == "TabPFN":
            if acq_method == "KG":
                raise NotImplementedError("KG acquisition requires model fantasizing, which TabPFN does not currently support.")
            if acq_method == "MaxValEntropy" and q > 1:
                raise NotImplementedError("TabPFN MaxValEntropy currently supports only one candidate at a time.")
        self._check_gradient_support(model, settings)
        if not getattr(model, "supports_gradients", True) and acq_method == "GIBBON":
            raise NotImplementedError(f"GIBBON requires a GPyTorch model and is not supported with the {model_type} surrogate.")

        with self._using_optimization_settings(settings):
            return self._dispatch_candidates(
                model=model,
                acq_method=acq_method,
                q=q,
                risk_objective=risk_objective,
                risk_n_deltas=risk_n_deltas,
                fidelity_cost=fidelity_cost,
                optimization_settings=settings,
                propose_best=propose_best,
                optimization_constraints=optimization_constraints,
            )

    def _check_gradient_support(self, model: Any, settings: AcquisitionOptimizationSettings) -> None:
        """Reject the gradient optimizer for non-differentiable surrogates."""
        if not getattr(model, "supports_gradients", True) and settings.acq_optimizer == "gradient":
            model_type = self.config_settings.get_setting("model_type")
            raise ValueError(
                f"The {model_type} surrogate is not differentiable, so it cannot be used with the gradient-based "
                "acquisition optimizer. Choose a gradient-free optimizer with --acq-optimizer sampling or "
                "--acq-optimizer boltzmann (config: optimization.acq_optimizer)."
            )

    def _dispatch_candidates(
        self,
        model: Type[Model],
        acq_method: str,
        q: int,
        risk_objective: Optional[Type[RiskMeasureMCObjective]],
        risk_n_deltas: Optional[int],
        fidelity_cost: Optional[np.ndarray],
        optimization_settings: AcquisitionOptimizationSettings,
        propose_best: bool,
        optimization_constraints: Optional[AcquisitionOptimizationConstraints],
    ) -> Tuple[Any, Any]:
        if self.multi_fidelity:
            candidates, acq_value = self._get_candidates_mf(
                model=model,
                acq_method=acq_method,
                q=q,
                fidelity_cost=fidelity_cost,
                risk_objective=risk_objective,
                risk_n_deltas=risk_n_deltas,
                optimization_settings=optimization_settings,
                propose_best=propose_best,
                optimization_constraints=optimization_constraints,
            )
        else:
            candidates, acq_value = self._get_candidates_sf(
                model=model,
                acq_method=acq_method,
                q=q,
                risk_objective=risk_objective,
                risk_n_deltas=risk_n_deltas,
                optimization_settings=optimization_settings,
                propose_best=propose_best,
                optimization_constraints=optimization_constraints,
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
        propose_best: bool = False,
        integer_fidelities: bool = False,
        optimization_settings: Optional[AcquisitionOptimizationSettings] = None,
        optimization_constraints: Optional[AcquisitionOptimizationConstraints] = None,
        inequality_constraints: Optional[List[Tuple[Sequence[int], Sequence[float], float]]] = None,
        equality_constraints: Optional[List[Tuple[Sequence[int], Sequence[float], float]]] = None,
        nonlinear_inequality_constraints: Optional[List[Callable[[torch.Tensor], torch.Tensor]]] = None,
        post_processing_func: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        batch_initial_conditions: Optional[torch.Tensor] = None,
        nonlinear_mode: Optional[str] = None,
        nonlinear_initial_raw_samples: Optional[int] = None,
        nonlinear_initial_max_tries: Optional[int] = None,
        nonlinear_optimization_retries: Optional[int] = None,
        acq_optimizer: Optional[str] = None,
        batch_mode: Optional[str] = None,
    ) -> None:
        """
        Propose candidates from a trained checkpoint and save them as a NumPy array.

        Candidates are optimized in scaled model coordinates and saved in original
        input units. For multi-fidelity runs, the last column is the rounded fidelity
        index and ``fidelity_cost`` supplies the relative cost for each fidelity.
        Linear constraints and ``batch_initial_conditions`` are authored in original
        input units and converted internally. Nonlinear constraints receive original
        input units, use ``>= 0`` as feasible, and are limited to single-fidelity
        optimization. Nonlinear control arguments override matching values from the
        resolved ``optimization`` config section. Risk measures interpret
        ``x_stddev`` in original input units; the fidelity perturbation is forced to
        zero for multi-fidelity runs.

        :param outfile: File path for the saved candidate ``.npy`` array.
        :param learner_file: Checkpoint file produced by ``learn``.
        :param acq_method: Single-fidelity options are ``KG``, ``MaxValEntropy``, ``EI``, or ``NEI``;
            multi-fidelity options are ``KG`` or ``MaxValEntropy``.
        :param num_candidates: Number of candidates to propose.
        :param fidelity_cost: Cost weights for each fidelity.
        :param risk_measure: Optional ``'CVaR'`` or ``'VaR'`` risk measure.
        :param risk_level: Risk alpha level between 0 and 1.
        :param risk_n_deltas: Number of input perturbations for the risk objective.
        :param x_stddev: Input standard deviations in original units.
        :param propose_best: Whether to prepend the current posterior maximizer.
        :param integer_fidelities: If ``True``, save the fidelity column as integers using object dtype.
        :param optimization_settings: Optional explicit acquisition optimizer settings.
        :param optimization_constraints: Optional grouped optimization constraints.
        :param inequality_constraints: Linear inequalities in original input units.
        :param equality_constraints: Linear equalities in original input units.
        :param nonlinear_inequality_constraints: Nonlinear feasibility callables.
        :param post_processing_func: Optional BoTorch candidate post-processing function.
        :param batch_initial_conditions: Optional original-unit optimizer initial conditions.
        :param nonlinear_mode: ``'enforce'`` or ``'initialization_only'``; defaults to config.
        :param nonlinear_initial_raw_samples: Optional raw samples for nonlinear-feasible starts; defaults to config.
        :param nonlinear_initial_max_tries: Maximum attempts to find nonlinear-feasible starts; defaults to config.
        :param nonlinear_optimization_retries: Additional retries after nonlinear optimizer failure warnings; defaults to config.
        :param acq_optimizer: ``'gradient'``, ``'sampling'``, or ``'boltzmann'``; overrides ``optimization.acq_optimizer``.
        :param batch_mode: ``'default'``, ``'sequential'``, or ``'joint'``; overrides ``optimization.batch_mode``.
        """
        optimization_settings = self._resolve_optimization_settings(optimization_settings)
        overrides = {
            key: value
            for key, value in (("acq_optimizer", acq_optimizer), ("batch_mode", batch_mode))
            if value is not None
        }
        if overrides:
            merged = dict(optimization_settings.__dict__)
            merged.update(overrides)
            optimization_settings = self._validate_optimization_settings(merged)
        optimization_constraints = self._normalize_optimization_constraints(
            optimization_constraints=optimization_constraints,
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
            nonlinear_inequality_constraints=nonlinear_inequality_constraints,
            post_processing_func=post_processing_func,
            batch_initial_conditions=batch_initial_conditions,
            nonlinear_mode=nonlinear_mode,
            nonlinear_initial_raw_samples=nonlinear_initial_raw_samples,
            nonlinear_initial_max_tries=nonlinear_initial_max_tries,
            nonlinear_optimization_retries=nonlinear_optimization_retries,
            optimization_settings=optimization_settings,
        )
        self._configure_torch_threads(optimization_settings)
        print(
            f"""
            Infile: {self.data_file}
            Outfile: {outfile}
            Config File: {self.config_settings.config_file}
            Learner File: {learner_file}
            Random Seed: {self.random_seed}
            Bounds: {self.bounds}
            Acq Method: {acq_method}
            Acq Optimizer: {optimization_settings.acq_optimizer}
            Batch Mode: {optimization_settings.batch_mode}
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
                bounds_scaled[1, -1] = self.num_fidelities - 1
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
            optimization_settings=optimization_settings,
            propose_best=propose_best,
            optimization_constraints=optimization_constraints,
        )
        if self.multi_fidelity:
            candidates[:, :-1] = candidates[:, :-1] * (self.bounds[1, :-1] - self.bounds[0, :-1]) + self.bounds[0, :-1]
            candidates[:, -1] = candidates[:, -1].round()
        else:
            candidates = candidates * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        candidates_npy = candidates.cpu().detach().numpy()
        if integer_fidelities and self.multi_fidelity:
            # Mixed float inputs and integer fidelities require object dtype in the saved array.
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
        *args: Any,
        original_scale_x: bool = True,
        original_scale_y: bool = True,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return GP posterior mean and variance or covariance.

        :param q: Query tensor with shape ``... x q x d``.
        :param get_cov: If ``True``, return the joint ``q x q`` covariance; otherwise return pointwise variance.
        :param original_scale_x: If ``True``, transform query inputs from original units to model units.
        :param original_scale_y: If ``True``, transform posterior statistics back to original output units.
        :returns: ``(mean, variance)`` or ``(mean, covariance)`` depending on ``get_cov``.
        """
        reject_deprecated_original_scale(args, kwargs)
        q_model = q
        if original_scale_x:
            if not hasattr(self, "input_scaler") or self.input_scaler is None:
                raise RuntimeError("original_scale_x=True requires this model to have an input_scaler.")
            q_model = self.input_scaler.transform(q)
        posterior = self.posterior(q_model, **kwargs)
        mean = posterior.mean
        if get_cov:
            cov = posterior.mvn.covariance_matrix
            if original_scale_y:
                mean = self.output_scaler.inverse_transform(mean, q_model)
                cov = self.output_scaler.inverse_covariance(cov, q_model)
            return mean.squeeze(-1), cov
        variance = posterior.variance
        if original_scale_y:
            mean = self.output_scaler.inverse_transform(mean, q_model)
            variance = self.output_scaler.inverse_variance(variance, q_model)
        return mean, variance


class DeepOptSingleTaskGP(DeepOptGPMixin, SingleTaskGP):
    """
    Single-fidelity GP model with DeepOpt prediction scaling helpers.
    """


class DeepOptSingleTaskMultiFidelityGP(DeepOptGPMixin, SingleTaskMultiFidelityGP):
    """
    Multi-fidelity GP model with DeepOpt prediction scaling helpers.
    """


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
        model.input_scaler = self.input_scaler

        fit_gpytorch_model(mll)

        state = {
            "state_dict": model.state_dict(),
            "output_scaler": self.output_scaler.state_dict(),
            "input_scaler": self.input_scaler.state_dict(),
            DEEPOPT_CHECKPOINT_KEY: self._checkpoint_metadata(),
        }
        self.learner_file = join(getcwd(), dirname(outfile), basename(outfile))
        torch.save(state, self.learner_file)
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
        if "input_scaler" in state_dict:
            self.input_scaler = InputScaler.from_state_dict(state_dict["input_scaler"], device=self.device)
            self.full_train_X = self.input_scaler.transform(self.X_orig).to(self.device)
        else:
            self.input_scaler = InputScaler(self.bounds, multi_fidelity=self.multi_fidelity, fidelity_dim=-1).to(self.device)
            self.full_train_X = self.input_scaler.transform(self.X_orig).to(self.device)
            warnings.warn(
                "GP checkpoint has no input_scaler state; reconstructing input scaling from current bounds.",
                RuntimeWarning,
            )
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
        model.input_scaler = self.input_scaler
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
                    input_scaler=self.input_scaler,
                )
                model.train()
                model.fit()

            model.eval()
            with torch.no_grad():  # TODO: is this needed?
                for _, (X_test, y_test) in enumerate(test_loader):
                    y_test_scaled = self.output_scaler.transform(y_test, X_test)
                    y_pred, _ = model.get_prediction_with_uncertainty(
                        X_test,
                        original_scale_x=False,
                        original_scale_y=False,
                    )
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

        ray, tune, RunConfig, ASHAScheduler = require_ray_for_deluq()
        try:
            ray.init(num_cpus=cpu_count, num_gpus=gpu_count)
        except Exception as exc:
            raise OptionalDependencyError(
                "delUQ training requires Ray Tune, but Ray could not be initialized in this environment. "
                "Non-delUQ DeepOpt models do not require Ray; use model_type='GP' or "
                "model_type='nnEnsemble', or install a Ray build compatible with this platform."
            ) from exc

        try:
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
        finally:
            ray.shutdown()
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
            input_scaler=self.input_scaler,
        )

        model.fit()
        if basename(outfile).split(".")[-1] == "ckpt":
            fname = basename(outfile)[:-5]
        else:
            fname = basename(outfile)
        model.save_ckpt(join(getcwd(), dirname(outfile)), fname, checkpoint_metadata=self._checkpoint_metadata())
        self.learner_file = join(getcwd(), dirname(outfile), f"{fname}.ckpt")
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
            input_scaler=self.input_scaler,
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

class TabPFNModel(DeepoptBaseModel):
    """
    DeepOpt wrapper for optional TabPFN surrogates.
    """

    def _make_tabpfn(self) -> TabPFN:
        return TabPFN(
            config=self.config_settings,
            X_train=self.full_train_X,
            y_train=self.full_train_Y_scaled,
            multi_fidelity=self.multi_fidelity,
            seed=self.random_seed,
            device=self.device,
            output_scaler=self.output_scaler,
            input_scaler=self.input_scaler,
        )

    def train(self, outfile: str) -> Type[Model]:
        """
        Construct a TabPFN surrogate and save a lightweight DeepOpt checkpoint.
        """
        print("Training TabPFN Surrogate.")
        model = self._make_tabpfn()
        state = {
            "input_scaler": self.input_scaler.state_dict(),
            "output_scaler": self.output_scaler.state_dict(),
            "tabpfn_backend": {"api": model.backend_api},
            DEEPOPT_CHECKPOINT_KEY: self._checkpoint_metadata(),
        }
        self.learner_file = join(getcwd(), dirname(outfile), basename(outfile))
        torch.save(state, self.learner_file)
        return model

    def load_model(self, learner_file: str) -> Type[Model]:
        """
        Reconstruct a TabPFN surrogate from a DeepOpt checkpoint.
        """
        state = _torch_load(learner_file, map_location=self.device)
        if "input_scaler" in state:
            self.input_scaler = InputScaler.from_state_dict(state["input_scaler"], device=self.device)
            self.full_train_X = self.input_scaler.transform(self.X_orig).to(self.device)
        if "output_scaler" in state:
            self.output_scaler = OutputScaler.from_state_dict(state["output_scaler"], device=self.device)
            self.full_train_Y_scaled = self.output_scaler.transform(self.full_train_Y, self.full_train_X)
        return self._make_tabpfn()


class NNEnsembleModel(DeepoptBaseModel):
    """
    DeepOpt wrapper for training, loading, and optimizing NN ensemble surrogates.

    This class has the same class variables as ``DeepoptBaseModel``.
    """

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
            input_scaler=self.input_scaler,
            verbose=self.verbose,
        )

        model.fit()
        if basename(outfile).split(".")[-1] == "ckpt":
            fname = basename(outfile)[:-5]
        else:
            fname = basename(outfile)
        model.save_ckpt(join(getcwd(), dirname(outfile)), fname, checkpoint_metadata=self._checkpoint_metadata())
        self.learner_file = join(getcwd(), dirname(outfile), f"{fname}.ckpt")
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
            input_scaler=self.input_scaler,
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


class RFModel(DeepoptBaseModel):
    """
    DeepOpt wrapper for scikit-learn random forest surrogates.

    Random forests are not differentiable, so candidate generation requires a
    gradient-free acquisition optimizer (``--acq-optimizer sampling`` or
    ``boltzmann``). Checkpoints store only scalers, settings, and training data;
    the forest is refit deterministically from ``random_seed`` on load.
    """

    def _make_rf(self) -> RandomForestSurrogate:
        return RandomForestSurrogate(
            config=self.config_settings,
            X_train=self.full_train_X,
            y_train=self.full_train_Y_scaled,
            multi_fidelity=self.multi_fidelity,
            seed=self.random_seed,
            device=self.device,
            output_scaler=self.output_scaler,
            input_scaler=self.input_scaler,
        )

    def train(self, outfile: str) -> RandomForestSurrogate:
        """
        Fit the random forest surrogate and save a lightweight DeepOpt checkpoint.

        :param outfile: Checkpoint path.
        :returns: The fitted ``RandomForestSurrogate``.
        """
        print("Training Random Forest Surrogate.")
        model = self._make_rf()
        state = {
            "input_scaler": self.input_scaler.state_dict(),
            "output_scaler": self.output_scaler.state_dict(),
            "rf_backend": {"refit_on_load": True},
            DEEPOPT_CHECKPOINT_KEY: self._checkpoint_metadata(),
        }
        self.learner_file = join(getcwd(), dirname(outfile), basename(outfile))
        torch.save(state, self.learner_file)
        return model

    def load_model(self, learner_file: str) -> RandomForestSurrogate:
        """
        Rebuild the random forest surrogate from a DeepOpt checkpoint by refitting it.

        :param learner_file: Checkpoint path produced by ``train``.
        :returns: The refit ``RandomForestSurrogate``.
        """
        state = _torch_load(learner_file, map_location=self.device)
        if "input_scaler" in state:
            self.input_scaler = InputScaler.from_state_dict(state["input_scaler"], device=self.device)
            self.full_train_X = self.input_scaler.transform(self.X_orig).to(self.device)
        if "output_scaler" in state:
            self.output_scaler = OutputScaler.from_state_dict(state["output_scaler"], device=self.device)
            self.full_train_Y_scaled = self.output_scaler.transform(self.full_train_Y, self.full_train_X)
        return self._make_rf()
