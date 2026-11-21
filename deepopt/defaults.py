"""
This module contains default values used throughout the deepopt library.
"""

DELUQ_CONFIG = {
    "ff": True,
    "dist": "uniform",
    "mapping_size": 128,
    "n_layers": 4,
    "hidden_dim": 128,
    "activation": "relu",
    "dropout": True,
    "dropout_prob": 0.2,
    "batchnorm": False,
    "w0": 30,
    "activation_first": True,
    "opt_type": "Adam",
    "learning_rate": 0.001,
    "weight_decay": 0,
    "n_epochs": 1000,
    "batch_size": 1000,
    "variance": 0.001,
}

NNENSEMBLE_CONFIG = {"n_estimators": 100,
                     "ff": True,
                     "dist": "uniform",
                     "mapping_size": 128,
                     "n_layers": 4,
                     "hidden_dim": 128,
                     "activation": "relu",
                     "dropout": True,
                     "droupout_prob": 0.2,
                     "batchnorm": False,
                     "w0": 30,
                     "activation_first": True,
                     "opt_type": "Adam",
                     "learning_rate": 0.001,
                     "weight_decay": 0,
                     "n_epochs": 300,
                     "batch_size": 128,
                     "variance": 0.001}

GP_CONFIG = {}

TABPFN_CONFIG = {
    "max_weight": 1,
    "tabpfn_api": "auto",
    "tabpfn_version": "v2",
    "max_tokens_per_forward": 3000,
}

# scikit-learn RandomForestRegressor surrogate. Requires a gradient-free acquisition optimizer.
RF_CONFIG = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_leaf": 1,
    "max_features": 1.0,
    "bootstrap": True,
    "n_jobs": 1,
    "variance_mode": "trees",
    "fantasy_n_jobs": 1,
}

# Gradient-free acquisition optimizer settings shared by every optimization profile.
SAMPLING_OPTIMIZATION_SETTINGS = {
    "acq_optimizer": "gradient",
    "batch_mode": "default",
    "sampling_raw_samples": 16384,
    "sampling_n_elites": 128,
    "sampling_samples_per_elite": 128,
    "sampling_n_rounds": 10,
    "sampling_init_halfwidth": 0.2,
    "sampling_shrink": 0.7,
    "sampling_min_halfwidth": 1e-3,
    "sampling_perturbation": "subset",
    "sampling_subset_prob": None,
    "sampling_eval_batch": 2048,
    "sampling_joint_member_prob": 1.0,
    "boltzmann_sampler": "smc",
    "boltzmann_lambda": 1.0,
    "boltzmann_population": 16384,
    "boltzmann_n_rounds": 10,
    "boltzmann_mh_steps": 1,
    "boltzmann_defer_budget": 16384,
    "kg_inner_set_size": 1024,
    "kg_refine_top": 8,
    "kg_raw_samples": 512,
    "kg_n_elites": 16,
    "kg_samples_per_elite": 16,
    "kg_n_rounds": 3,
}

# Acquisition optimization profiles used by ConfigSettings['optimization'].
OPTIMIZATION_PROFILES = {
    "balanced": {
        "num_restarts_high": 15,
        "num_restarts_low": 5,
        "raw_samples_high": 5000,
        "raw_samples_low": 512,
        "batch_limit_high": 15,
        "batch_limit_low": 5,
        "maxiter": 200,
        "n_fantasies": 128,
        "torch_num_threads": None,
        "torch_num_threads_fraction": 0.8,
        "torch_num_interop_threads": None,
        "nonlinear_mode": "enforce",
        "nonlinear_initial_raw_samples": None,
        "nonlinear_initial_max_tries": 5,
        "nonlinear_optimization_retries": 1,
        **SAMPLING_OPTIMIZATION_SETTINGS,
    },
    "cpu_large": {
        "num_restarts_high": 32,
        "num_restarts_low": 8,
        "raw_samples_high": 8192,
        "raw_samples_low": 4096,
        "batch_limit_high": 32,
        "batch_limit_low": 8,
        "maxiter": 200,
        "n_fantasies": 64,
        "torch_num_threads": None,
        "torch_num_threads_fraction": 0.8,
        "torch_num_interop_threads": None,
        "nonlinear_mode": "enforce",
        "nonlinear_initial_raw_samples": None,
        "nonlinear_initial_max_tries": 5,
        "nonlinear_optimization_retries": 1,
        **SAMPLING_OPTIMIZATION_SETTINGS,
    },
    "fast": {
        "num_restarts_high": 8,
        "num_restarts_low": 4,
        "raw_samples_high": 2048,
        "raw_samples_low": 512,
        "batch_limit_high": 8,
        "batch_limit_low": 4,
        "maxiter": 100,
        "n_fantasies": 32,
        "torch_num_threads": "auto",
        "torch_num_threads_fraction": 0.8,
        "torch_num_interop_threads": 1,
        "nonlinear_mode": "enforce",
        "nonlinear_initial_raw_samples": None,
        "nonlinear_initial_max_tries": 5,
        "nonlinear_optimization_retries": 1,
        **SAMPLING_OPTIMIZATION_SETTINGS,
    },
}


class Defaults:
    """
    Default values for the DeepOpt library. This must be a class for ray tuning.

    :cvar random_seed: The default random seed. `Default value: 4321`
    :cvar k_folds: The default k-folds value. `Default value: 5`
    :cvar model_type: The default model type. Options here are 'GP', 'delUQ', 'nnEnsemble', 'TabPFN', or 'RF'.
        `Default value: 'GP'`
    :cvar multi_fidelity: The default value on whether to run multi-fidelity
        settings or not. `Default value: False`
    :cvar num_candidates: The default number of candidates. `Default value: 2`
    :cvar fidelity_cost: The default fidelity cost range. `Default value: '[1,10]'`
    :cvar num_restarts_low: The default value for the number of restarts to use (low).
        This default is used for expensive acquisition paths. `Default value: 8`
    :cvar num_restarts_high: The default value for the number of restarts to use (high).
        This default is used for most acquisition optimization calls. `Default value: 32`
    :cvar raw_samples_low: The default value for the number of raw samples to use (low).
        `Default value: 1024`
    :cvar raw_samples_high: The default value for the number of raw samples to use (high).
        `Default value: 8192`
    :cvar batch_limit_low: The default low-budget restart batch limit. `Default value: 8`
    :cvar batch_limit_high: The default high-budget restart batch limit. `Default value: 32`
    :cvar maxiter: The default maximum local optimizer iterations. `Default value: 200`
    :cvar n_fantasies: The default value for the number of fantasy models to construct. `Default value: 64`
    :cvar torch_num_threads: The default PyTorch intra-op thread setting. `Default value: 'auto'`
    :cvar torch_num_threads_fraction: Fraction of available CPUs used by ``auto`` threading. `Default value: 0.8`
    :cvar torch_num_interop_threads: The default PyTorch inter-op thread setting. `Default value: 1`
    :cvar nonlinear_mode: Default nonlinear constraint mode. `Default value: 'enforce'`
    :cvar nonlinear_initial_raw_samples: Default raw samples for nonlinear-feasible starts. `Default value: None`
    :cvar nonlinear_initial_max_tries: Default attempts to find nonlinear-feasible starts. `Default value: 5`
    :cvar nonlinear_optimization_retries: Default retries after nonlinear optimizer warnings. `Default value: 1`
    :cvar acq_optimizer: Default acquisition optimizer. `Default value: 'gradient'`
    :cvar batch_mode: Default batch construction mode. `Default value: 'default'`
    """

    random_seed: int = 4321
    k_folds: int = 5
    model_type: str = "GP"
    multi_fidelity: bool = False
    num_candidates: int = 2
    fidelity_cost: str = "[1,10]"
    optimization_profile: str = "cpu_large"
    num_restarts_low: int = 8
    num_restarts_high: int = 32
    raw_samples_low: int = 1024
    raw_samples_high: int = 8192
    batch_limit_low: int = 8
    batch_limit_high: int = 32
    maxiter: int = 200
    n_fantasies: int = 64
    torch_num_threads: str = "auto"
    torch_num_threads_fraction: float = 0.8
    torch_num_interop_threads: int = 1
    nonlinear_mode: str = "enforce"
    nonlinear_initial_raw_samples = None
    nonlinear_initial_max_tries: int = 5
    nonlinear_optimization_retries: int = 1
    acq_optimizer: str = "gradient"
    batch_mode: str = "default"
