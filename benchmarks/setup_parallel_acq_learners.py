#!/usr/bin/env python3
"""
Create small learner checkpoints for benchmark_parallel_acq.py.
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from botorch.test_functions.multi_fidelity import AugmentedHartmann
from botorch.test_functions.synthetic import Hartmann

from deepopt.configuration import ConfigSettings
from deepopt.deepopt_cli import get_deepopt_model
from deepopt.models import get_checkpoint_metadata, load_deepopt_wrapper


SEED = 10
NUM_INITIAL_POINTS = 20
NNENSEMBLE_SETUP_CONFIG = {
    "n_estimators": 4,
    "n_epochs": 25,
    "batch_size": 20,
    "mapping_size": 32,
    "hidden_dim": 64,
}


Dataset = Tuple[np.ndarray, np.ndarray, torch.Tensor, bool]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory for generated .npz data and .ckpt learner files.",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num-points", type=int, default=NUM_INITIAL_POINTS)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def hartmann_dataset(num_points: int, generator: torch.Generator) -> Dataset:
    objective = Hartmann(negate=True)
    bounds = objective.bounds.float()
    X = torch.rand(num_points, objective.dim, generator=generator)
    X = X * (bounds[1] - bounds[0]) + bounds[0]
    y = objective(X).reshape(-1, 1)
    return X.numpy(), y.numpy(), bounds, False


def augmented_hartmann_dataset(num_points: int, generator: torch.Generator) -> Dataset:
    objective = AugmentedHartmann(negate=True)
    bounds = objective.bounds.float()
    X = torch.rand(num_points, objective.dim, generator=generator)
    X[:, :-1] = X[:, :-1] * (bounds[1, :-1] - bounds[0, :-1]) + bounds[0, :-1]
    X[:, -1] = X[:, -1].round()
    X_eval = X.clone()
    X_eval[X_eval[:, -1] == 0, -1] = 0.5
    y = objective(X_eval).reshape(-1, 1)
    return X.numpy(), y.numpy(), bounds, True


def config_settings(model_type: str) -> ConfigSettings:
    settings = ConfigSettings(model_type=model_type)
    if model_type == "nnEnsemble":
        for key, value in NNENSEMBLE_SETUP_CONFIG.items():
            settings.set_setting(key, value)
    return settings


def write_dataset(outdir: Path, stem: str, X: np.ndarray, y: np.ndarray) -> Path:
    data_file = outdir / f"{stem}.npz"
    np.savez(data_file, X=X, y=y)
    return data_file


def train_learner(
    outdir: Path,
    model_type: str,
    problem_name: str,
    dataset: Dataset,
    seed: int,
    device: str,
) -> Path:
    X, y, bounds, multi_fidelity = dataset
    model_label = model_type.lower() if model_type == "GP" else "nnensemble"
    stem = f"{model_label}_{problem_name}"
    data_file = write_dataset(outdir, stem, X, y)
    learner_file = outdir / f"{stem}.ckpt"
    model_class = get_deepopt_model(model_type)
    model = model_class(
        data_file=str(data_file),
        bounds=bounds,
        config_settings=config_settings(model_type),
        multi_fidelity=multi_fidelity,
        random_seed=seed,
        device=device,
        verbose=False,
    )
    model.learn(outfile=str(learner_file))
    metadata = get_checkpoint_metadata(str(learner_file))
    if metadata is None or metadata["model_type"] != model_type:
        raise RuntimeError(f"Generated checkpoint metadata did not identify {model_type}: {learner_file}")
    load_deepopt_wrapper(str(learner_file), device=device, verbose=False)
    return learner_file


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(args.seed)
    datasets: Dict[str, Dataset] = {
        "hartmann": hartmann_dataset(args.num_points, generator),
        "augmented_hartmann": augmented_hartmann_dataset(args.num_points, generator),
    }

    for problem_name, dataset in datasets.items():
        for model_type in ("GP", "nnEnsemble"):
            learner_file = train_learner(
                outdir=args.outdir,
                model_type=model_type,
                problem_name=problem_name,
                dataset=dataset,
                seed=args.seed,
                device=args.device,
            )
            print(learner_file)


if __name__ == "__main__":
    main()
