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

# Acquisition optimization profiles used by ConfigSettings['optimization'].
OPTIMIZATION_PROFILES = {
    "balanced": {
        "num_restarts_high": 15,
        "num_restarts_low": 5,
        "raw_samples_high": 5000,
        "raw_samples_low": 512,
        "batch_limit_high": 10,
        "batch_limit_low": 5,
        "maxiter": 200,
        "n_fantasies": 128,
        "torch_num_threads": None,
        "torch_num_threads_fraction": 0.8,
        "torch_num_interop_threads": None,
    },
    "cpu_large": {
        "num_restarts_high": 32,
        "num_restarts_low": 8,
        "raw_samples_high": 8192,
        "raw_samples_low": 1024,
        "batch_limit_high": 32,
        "batch_limit_low": 8,
        "maxiter": 200,
        "n_fantasies": 64,
        "torch_num_threads": "auto",
        "torch_num_threads_fraction": 0.8,
        "torch_num_interop_threads": 1,
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
    },
}


class Defaults:
    """
    Default values for the DeepOpt library. This must be a class for ray tuning.

    :cvar random_seed: The default random seed. `Default value: 4321`
    :cvar k_folds: The default k-folds value. `Default value: 5`
    :cvar model_type: The default model type. Options here are 'GP', 'delUQ', or 'nnEnsemble'.
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
    :cvar n_fantasies: The default value for the number of fantasy models to construct. `Default value: 64`
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
