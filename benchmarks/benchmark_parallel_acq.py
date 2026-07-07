#!/usr/bin/env python3
"""
Benchmark serial and process-parallel DeepOpt acquisition optimization.
"""
import argparse
import json
import os
import sys
import time
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepopt.configuration import ConfigSettings
from deepopt.deepopt_cli import get_deepopt_model
from deepopt.models import get_checkpoint_metadata, load_deepopt_wrapper


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learner-file", required=True)
    parser.add_argument("--infile")
    parser.add_argument("--bounds")
    parser.add_argument("--config-file")
    parser.add_argument("--model-type", default="GP")
    parser.add_argument("--acq-method", default="KG")
    parser.add_argument("--num-candidates", type=int, default=5)
    parser.add_argument("--multi-fidelity", action="store_true")
    parser.add_argument("--fidelity-cost", default="[5, 6]")
    parser.add_argument("--risk-measure")
    parser.add_argument("--risk-level", type=float)
    parser.add_argument("--risk-n-deltas", type=int)
    parser.add_argument("--x-stddev")
    parser.add_argument("--propose-best", action="store_true")
    parser.add_argument("--workers", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["cpu-parallel", "cpu-parallel-from-checkpoint"],
        choices=["gpu-serial", "cpu-serial", "cpu-parallel", "cpu-parallel-from-checkpoint"],
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--worker-torch-num-threads", type=int)
    parser.add_argument("--total-worker-torch-threads", type=int, default=64)
    parser.add_argument("--worker-torch-num-interop-threads", type=int)
    parser.add_argument("--total-worker-torch-num-interop-threads", type=int, default=64)
    return parser.parse_args(argv)


def env_summary() -> Dict[str, object]:
    return {
        "host": os.uname().nodename,
        "affinity": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "TORCH_NUM_THREADS": os.environ.get("TORCH_NUM_THREADS"),
    }


def load_wrapper(args: argparse.Namespace, device: str):
    metadata = get_checkpoint_metadata(args.learner_file)
    if metadata is not None:
        wrapper = load_deepopt_wrapper(args.learner_file, device=device, verbose=False)
        if args.config_file is not None:
            optimize_config_settings = ConfigSettings(metadata["model_type"], config_file=args.config_file)
            if "optimization" in optimize_config_settings:
                wrapper.config_settings.set_setting("optimization", optimize_config_settings.get_setting("optimization"))
        return wrapper

    if args.infile is None or args.bounds is None:
        raise ValueError("Legacy checkpoints require --infile and --bounds.")
    bounds = np.array(json.loads(args.bounds), dtype=np.float32).T
    config_settings = ConfigSettings(args.model_type, config_file=args.config_file)
    model_class = get_deepopt_model(args.model_type)
    return model_class(
        config_settings=config_settings,
        data_file=args.infile,
        multi_fidelity=args.multi_fidelity,
        bounds=bounds,
        device=device,
        verbose=False,
    )


def build_risk_state(wrapper, model, args: argparse.Namespace):
    if args.risk_measure is None or args.risk_measure == "None":
        return None
    x_stddev = torch.tensor(json.loads(args.x_stddev), dtype=torch.float, device=wrapper.bounds.device)
    x_stddev_scaled = x_stddev / (wrapper.bounds[1] - wrapper.bounds[0])
    bounds_scaled = torch.tensor(wrapper.input_dim * [[0, 1]], dtype=torch.float).T
    if wrapper.multi_fidelity:
        bounds_scaled[1, -1] = wrapper.num_fidelities - 1
        x_stddev_scaled[-1] = 0
    risk_objective = wrapper.get_risk_measure_objective(
        risk_measure=args.risk_measure,
        alpha=args.risk_level,
        n_w=args.risk_n_deltas,
    )
    model.input_transform = wrapper.get_input_perturbation(
        risk_n_deltas=args.risk_n_deltas,
        bounds=bounds_scaled.to(wrapper.device),
        X_stddev=x_stddev_scaled.to(wrapper.device),
    )
    return risk_objective


def mode_device(mode: str) -> Optional[str]:
    if mode == "gpu-serial":
        return "cuda" if torch.cuda.is_available() else None
    return "cpu"


def worker_threads_for_workers(
    workers: int,
    explicit_threads: Optional[int],
    total_threads: int,
    thread_name: str,
    total_thread_name: str,
) -> int:
    if workers <= 0:
        raise ValueError("workers must be positive.")
    if explicit_threads is not None:
        if explicit_threads <= 0:
            raise ValueError(f"{thread_name} must be positive.")
        return explicit_threads
    if total_threads <= 0:
        raise ValueError(f"{total_thread_name} must be positive.")
    if total_threads % workers != 0:
        raise ValueError(f"{total_thread_name} must be divisible by workers.")
    return total_threads // workers


def worker_torch_num_threads_for_workers(workers: int, args: argparse.Namespace) -> int:
    return worker_threads_for_workers(
        workers,
        args.worker_torch_num_threads,
        args.total_worker_torch_threads,
        "worker_torch_num_threads",
        "total_worker_torch_threads",
    )


def worker_torch_num_interop_threads_for_workers(workers: int, args: argparse.Namespace) -> int:
    return worker_threads_for_workers(
        workers,
        args.worker_torch_num_interop_threads,
        args.total_worker_torch_num_interop_threads,
        "worker_torch_num_interop_threads",
        "total_worker_torch_num_interop_threads",
    )


def parallel_settings(mode: str, workers: int, args: argparse.Namespace) -> Optional[Dict[str, object]]:
    if "parallel" not in mode:
        return None
    return {
        "enabled": True,
        "num_workers": workers,
        "start_method": "fork",
        "worker_torch_num_threads": worker_torch_num_threads_for_workers(workers, args),
        "worker_torch_num_interop_threads": worker_torch_num_interop_threads_for_workers(workers, args),
    }


def configure_parent_torch_threads(
    parallel_acq_settings: Optional[Dict[str, object]], wrapper=None, optimization_settings=None
) -> None:
    if parallel_acq_settings is None:
        if wrapper is None or optimization_settings is None:
            raise ValueError("wrapper and optimization_settings are required for serial mode.")
        wrapper._configure_torch_threads(optimization_settings)
        return
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def worker_counts_for_mode(mode: str, workers: Iterable[int]) -> Iterable[int]:
    if "parallel" in mode:
        return workers
    return [1]


def run_once(args: argparse.Namespace, mode: str, workers: int) -> Dict[str, object]:
    device = mode_device(mode)
    if device is None:
        return {"mode": mode, "workers": workers, "skipped": "cuda unavailable"}

    parallel_acq_settings = parallel_settings(mode, workers, args)
    if parallel_acq_settings is not None:
        configure_parent_torch_threads(parallel_acq_settings)
    wrapper = load_wrapper(args, device=device)
    optimization_settings = wrapper._resolve_optimization_settings()
    if parallel_acq_settings is None:
        configure_parent_torch_threads(parallel_acq_settings, wrapper, optimization_settings)
    model = wrapper.load_model(args.learner_file)
    risk_objective = build_risk_state(wrapper, model, args)
    model.eval()

    fidelity_cost = None
    if wrapper.multi_fidelity:
        fidelity_cost = np.array(json.loads(args.fidelity_cost), dtype=np.float32)

    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    candidates, acq_value = wrapper.get_candidates(
        model=model,
        acq_method=args.acq_method,
        q=args.num_candidates,
        risk_objective=risk_objective,
        risk_n_deltas=args.risk_n_deltas,
        fidelity_cost=fidelity_cost,
        propose_best=args.propose_best,
        optimization_settings=optimization_settings,
        parallel_acq=parallel_acq_settings,
    )
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return {
        "mode": mode,
        "workers": workers,
        "worker_torch_num_threads": (
            parallel_acq_settings["worker_torch_num_threads"] if parallel_acq_settings is not None else None
        ),
        "worker_torch_num_interop_threads": (
            parallel_acq_settings["worker_torch_num_interop_threads"] if parallel_acq_settings is not None else None
        ),
        "device": device,
        "elapsed_seconds": elapsed,
        "candidate_shape": list(candidates.shape),
        "acq_value": acq_value.detach().cpu().reshape(-1).tolist() if torch.is_tensor(acq_value) else str(acq_value),
        "optimization_settings": optimization_settings.__dict__,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
    }


def main() -> None:
    args = parse_args()
    print(json.dumps({"event": "environment", **env_summary()}), flush=True)
    for mode in args.modes:
        for workers in worker_counts_for_mode(mode, args.workers):
            for repeat in range(args.repeat):
                try:
                    result = run_once(args, mode, workers)
                except Exception as exc:
                    result = {"mode": mode, "workers": workers, "repeat": repeat, "error": repr(exc)}
                result["repeat"] = repeat
                print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
