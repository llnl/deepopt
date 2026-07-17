#!/usr/bin/env python3
"""
Benchmark native BoTorch acquisition optimization scaling.
"""
import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepopt.configuration import ConfigSettings
from deepopt.deepopt_cli import get_deepopt_model
from deepopt.models import AcquisitionOptimizationSettings, get_checkpoint_metadata, load_deepopt_wrapper


ThreadSetting = Optional[int]
ThreadToken = Union[int, str]

FULL_TORCH_NUM_THREADS: Tuple[ThreadToken, ...] = ("default", 1, 2, 4, 8, 16, 32, 64)
FULL_TORCH_NUM_INTEROP_THREADS: Tuple[ThreadToken, ...] = ("default", 1, 2, 4, 8)
FULL_BATCH_LIMITS = (1, 2, 4, 8, 16, 32)
QUICK_TORCH_NUM_THREADS: Tuple[ThreadToken, ...] = ("default", 1, 2)
QUICK_TORCH_NUM_INTEROP_THREADS: Tuple[ThreadToken, ...] = ("default", 1)
QUICK_BATCH_LIMITS = (1, 4)
GENERATED_LEARNER_DIR = Path("benchmark_results/acq_native/learners")
DEFAULT_NUM_POINTS = 20
DEFAULT_SEED = 10
DEFAULT_MODES = ("cpu-serial",)
DEFAULT_PROBLEMS = ("hartmann", "augmented_hartmann")


Dataset = Tuple[np.ndarray, np.ndarray, torch.Tensor, bool]


def parse_thread_token(value: str) -> ThreadToken:
    if value == "default":
        return value
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("thread counts must be positive or 'default'.")
    return parsed


def default_run_name() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learner-file", action="append", default=[])
    parser.add_argument("--infile")
    parser.add_argument("--bounds")
    parser.add_argument("--config-file")
    parser.add_argument("--model-type", default="GP")
    parser.add_argument("--acq-method", default="auto")
    parser.add_argument("--num-candidates", type=int, default=1)
    parser.add_argument("--multi-fidelity", action="store_true")
    parser.add_argument("--fidelity-cost", default="[5, 6]")
    parser.add_argument("--risk-measure")
    parser.add_argument("--risk-level", type=float)
    parser.add_argument("--risk-n-deltas", type=int)
    parser.add_argument("--x-stddev")
    parser.add_argument("--propose-best", action="store_true")
    parser.add_argument("--modes", nargs="+", default=list(DEFAULT_MODES), choices=["gpu-serial", "cpu-serial"])
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--torch-num-threads", type=parse_thread_token, nargs="+", default=None)
    parser.add_argument("--torch-num-interop-threads", type=parse_thread_token, nargs="+", default=None)
    parser.add_argument("--batch-limits", type=int, nargs="+", default=None)
    parser.add_argument("--num-restarts", type=int, default=None)
    parser.add_argument("--raw-samples", type=int, default=None)
    parser.add_argument("--maxiter", type=int, default=None)
    parser.add_argument("--run-name")
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--learner-outdir", type=Path, default=GENERATED_LEARNER_DIR)
    parser.add_argument("--problems", nargs="+", default=list(DEFAULT_PROBLEMS), choices=["hartmann", "augmented_hartmann"])
    parser.add_argument("--num-points", type=int, default=DEFAULT_NUM_POINTS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--in-process", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-run", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-learner-file", help=argparse.SUPPRESS)
    parser.add_argument("--worker-mode", choices=["gpu-serial", "cpu-serial"], help=argparse.SUPPRESS)
    parser.add_argument("--worker-batch-limit", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-torch-num-threads", type=parse_thread_token, help=argparse.SUPPRESS)
    parser.add_argument("--worker-torch-num-interop-threads", type=parse_thread_token, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.run_name is None:
        args.run_name = default_run_name()
    if args.outdir is None:
        args.outdir = Path("benchmark_results/acq_native") / args.run_name
    if args.quick:
        args.torch_num_threads = list(QUICK_TORCH_NUM_THREADS) if args.torch_num_threads is None else args.torch_num_threads
        args.torch_num_interop_threads = (
            list(QUICK_TORCH_NUM_INTEROP_THREADS)
            if args.torch_num_interop_threads is None
            else args.torch_num_interop_threads
        )
        args.batch_limits = list(QUICK_BATCH_LIMITS) if args.batch_limits is None else args.batch_limits
        args.num_restarts = 4 if args.num_restarts is None else args.num_restarts
        args.raw_samples = 64 if args.raw_samples is None else args.raw_samples
        args.maxiter = 5 if args.maxiter is None else args.maxiter
    else:
        args.torch_num_threads = list(FULL_TORCH_NUM_THREADS) if args.torch_num_threads is None else args.torch_num_threads
        args.torch_num_interop_threads = (
            list(FULL_TORCH_NUM_INTEROP_THREADS)
            if args.torch_num_interop_threads is None
            else args.torch_num_interop_threads
        )
        args.batch_limits = list(FULL_BATCH_LIMITS) if args.batch_limits is None else args.batch_limits
        args.num_restarts = 32 if args.num_restarts is None else args.num_restarts
        args.raw_samples = 8192 if args.raw_samples is None else args.raw_samples
        args.maxiter = 200 if args.maxiter is None else args.maxiter

    if any(batch_limit <= 0 for batch_limit in args.batch_limits):
        parser.error("--batch-limits values must be positive.")
    if args.num_restarts <= 0 or args.raw_samples <= 0 or args.maxiter <= 0:
        parser.error("--num-restarts, --raw-samples, and --maxiter must be positive.")
    return args


def current_git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def env_summary() -> Dict[str, object]:
    return {
        "host": os.uname().nodename,
        "git_commit": current_git_commit(),
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


def hartmann_dataset(num_points: int, generator: torch.Generator) -> Dataset:
    from botorch.test_functions.synthetic import Hartmann

    objective = Hartmann(negate=True)
    bounds = objective.bounds.float()
    X = torch.rand(num_points, objective.dim, generator=generator)
    X = X * (bounds[1] - bounds[0]) + bounds[0]
    y = objective(X).reshape(-1, 1)
    return X.numpy(), y.numpy(), bounds, False


def augmented_hartmann_dataset(num_points: int, generator: torch.Generator) -> Dataset:
    from botorch.test_functions.multi_fidelity import AugmentedHartmann

    objective = AugmentedHartmann(negate=True)
    bounds = objective.bounds.float()
    X = torch.rand(num_points, objective.dim, generator=generator)
    X[:, :-1] = X[:, :-1] * (bounds[1, :-1] - bounds[0, :-1]) + bounds[0, :-1]
    X[:, -1] = X[:, -1].round()
    X_eval = X.clone()
    X_eval[X_eval[:, -1] == 0, -1] = 0.5
    y = objective(X_eval).reshape(-1, 1)
    return X.numpy(), y.numpy(), bounds, True


def write_dataset(outdir: Path, stem: str, X: np.ndarray, y: np.ndarray) -> Path:
    data_file = outdir / f"{stem}.npz"
    np.savez(data_file, X=X, y=y)
    return data_file


def train_or_reuse_gp_learner(outdir: Path, problem_name: str, dataset: Dataset, seed: int, device: str) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    X, y, bounds, multi_fidelity = dataset
    stem = f"gp_{problem_name}_seed{seed}_n{X.shape[0]}"
    learner_file = outdir / f"{stem}.ckpt"
    metadata = get_checkpoint_metadata(str(learner_file)) if learner_file.exists() else None
    if metadata is not None and metadata["model_type"] == "GP":
        return learner_file

    data_file = write_dataset(outdir, stem, X, y)
    model_class = get_deepopt_model("GP")
    model = model_class(
        data_file=str(data_file),
        bounds=bounds,
        config_settings=ConfigSettings("GP"),
        multi_fidelity=multi_fidelity,
        random_seed=seed,
        device=device,
        verbose=False,
    )
    model.learn(outfile=str(learner_file))
    return learner_file


def generated_learner_files(args: argparse.Namespace, device: str) -> List[str]:
    generator = torch.Generator().manual_seed(args.seed)
    datasets = {
        "hartmann": hartmann_dataset(args.num_points, generator),
        "augmented_hartmann": augmented_hartmann_dataset(args.num_points, generator),
    }
    return [
        str(train_or_reuse_gp_learner(args.learner_outdir, problem_name, datasets[problem_name], args.seed, device))
        for problem_name in args.problems
    ]


def learner_files_for_args(args: argparse.Namespace, device: str) -> List[str]:
    if args.learner_file:
        return args.learner_file
    return generated_learner_files(args, device)


def load_wrapper(args: argparse.Namespace, learner_file: str, device: str):
    metadata = get_checkpoint_metadata(learner_file)
    if metadata is not None:
        wrapper = load_deepopt_wrapper(learner_file, device=device, verbose=False)
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


def normalize_thread_setting(value: ThreadToken) -> ThreadSetting:
    return None if value == "default" else int(value)


def benchmark_settings(
    base_settings: AcquisitionOptimizationSettings,
    args: argparse.Namespace,
    batch_limit: int,
    torch_num_threads: ThreadToken,
    torch_num_interop_threads: ThreadToken,
) -> AcquisitionOptimizationSettings:
    values = dataclasses.asdict(base_settings)
    values.update(
        {
            "num_restarts_high": args.num_restarts,
            "num_restarts_low": args.num_restarts,
            "raw_samples_high": args.raw_samples,
            "raw_samples_low": args.raw_samples,
            "batch_limit_high": batch_limit,
            "batch_limit_low": batch_limit,
            "maxiter": args.maxiter,
            "torch_num_threads": normalize_thread_setting(torch_num_threads),
            "torch_num_interop_threads": normalize_thread_setting(torch_num_interop_threads),
        }
    )
    return AcquisitionOptimizationSettings(**values)


def acq_method_for_wrapper(args: argparse.Namespace, wrapper) -> str:
    if args.acq_method != "auto":
        return args.acq_method
    return "MaxValEntropy" if wrapper.multi_fidelity else "EI"


def run_once(
    args: argparse.Namespace,
    learner_file: str,
    mode: str,
    batch_limit: int,
    torch_num_threads: ThreadToken,
    torch_num_interop_threads: ThreadToken,
) -> Dict[str, object]:
    device = mode_device(mode)
    if device is None:
        return {
            "event": "skipped",
            "mode": mode,
            "learner_file": learner_file,
            "skipped": "cuda unavailable",
        }

    wrapper = load_wrapper(args, learner_file, device=device)
    optimization_settings = benchmark_settings(
        wrapper._resolve_optimization_settings(),
        args,
        batch_limit,
        torch_num_threads,
        torch_num_interop_threads,
    )
    wrapper._configure_torch_threads(optimization_settings)
    effective_torch_num_threads = torch.get_num_threads()
    effective_torch_num_interop_threads = torch.get_num_interop_threads()

    model = wrapper.load_model(learner_file)
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
        acq_method=acq_method_for_wrapper(args, wrapper),
        q=args.num_candidates,
        risk_objective=risk_objective,
        risk_n_deltas=args.risk_n_deltas,
        fidelity_cost=fidelity_cost,
        propose_best=args.propose_best,
        optimization_settings=optimization_settings,
    )
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return {
        "event": "result",
        "mode": mode,
        "device": device,
        "learner_file": learner_file,
        "multi_fidelity": wrapper.multi_fidelity,
        "batch_limit": batch_limit,
        "num_restarts": args.num_restarts,
        "raw_samples": args.raw_samples,
        "requested_torch_num_threads": torch_num_threads,
        "effective_torch_num_threads": effective_torch_num_threads,
        "requested_torch_num_interop_threads": torch_num_interop_threads,
        "effective_torch_num_interop_threads": effective_torch_num_interop_threads,
        "elapsed_seconds": elapsed,
        "candidates_per_second": args.num_candidates / elapsed if elapsed > 0 else None,
        "restarts_per_second": args.num_restarts / elapsed if elapsed > 0 else None,
        "candidate_shape": list(candidates.shape),
        "acq_value": acq_value.detach().cpu().reshape(-1).tolist() if torch.is_tensor(acq_value) else str(acq_value),
        "optimization_settings": dataclasses.asdict(optimization_settings),
    }


def result_path(args: argparse.Namespace) -> Path:
    return args.outdir / f"{args.run_name}.jsonl"


def write_record(handle, record: Dict[str, object]) -> None:
    handle.write(json.dumps(record) + "\n")
    handle.flush()


def worker_command(
    args: argparse.Namespace,
    output_path: Path,
    learner_file: str,
    mode: str,
    batch_limit: int,
    torch_num_threads: ThreadToken,
    torch_num_interop_threads: ThreadToken,
) -> List[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-run",
        "--worker-output",
        str(output_path),
        "--worker-learner-file",
        learner_file,
        "--worker-mode",
        mode,
        "--worker-batch-limit",
        str(batch_limit),
        "--worker-torch-num-threads",
        str(torch_num_threads),
        "--worker-torch-num-interop-threads",
        str(torch_num_interop_threads),
        "--run-name",
        args.run_name,
        "--outdir",
        str(args.outdir),
        "--num-restarts",
        str(args.num_restarts),
        "--raw-samples",
        str(args.raw_samples),
        "--maxiter",
        str(args.maxiter),
        "--acq-method",
        args.acq_method,
        "--num-candidates",
        str(args.num_candidates),
        "--fidelity-cost",
        args.fidelity_cost,
        "--model-type",
        args.model_type,
    ]
    if args.config_file is not None:
        command.extend(["--config-file", args.config_file])
    if args.infile is not None:
        command.extend(["--infile", args.infile])
    if args.bounds is not None:
        command.extend(["--bounds", args.bounds])
    if args.risk_measure is not None:
        command.extend(["--risk-measure", args.risk_measure])
    if args.risk_level is not None:
        command.extend(["--risk-level", str(args.risk_level)])
    if args.risk_n_deltas is not None:
        command.extend(["--risk-n-deltas", str(args.risk_n_deltas)])
    if args.x_stddev is not None:
        command.extend(["--x-stddev", args.x_stddev])
    if args.multi_fidelity:
        command.append("--multi-fidelity")
    if args.propose_best:
        command.append("--propose-best")
    return command


def run_worker(args: argparse.Namespace) -> None:
    result = run_once(
        args,
        args.worker_learner_file,
        args.worker_mode,
        args.worker_batch_limit,
        args.worker_torch_num_threads,
        args.worker_torch_num_interop_threads,
    )
    args.worker_output.write_text(json.dumps(result) + "\n")


def run_matrix(args: argparse.Namespace) -> Path:
    args.outdir.mkdir(parents=True, exist_ok=True)
    path = result_path(args)
    with path.open("w") as handle:
        write_record(
            handle,
            {
                "event": "environment",
                "run_name": args.run_name,
                "quick": args.quick,
                "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                **env_summary(),
            },
        )
        for mode in args.modes:
            device = mode_device(mode)
            if device is None and not args.learner_file:
                learner_files = [f"generated:{problem_name}" for problem_name in args.problems]
            else:
                learner_files = args.learner_file if device is None else learner_files_for_args(args, device)
            for learner_file in learner_files:
                for batch_limit in args.batch_limits:
                    for torch_num_threads in args.torch_num_threads:
                        for torch_num_interop_threads in args.torch_num_interop_threads:
                            for repeat in range(args.repeat):
                                try:
                                    if args.in_process:
                                        result = run_once(
                                            args,
                                            learner_file,
                                            mode,
                                            batch_limit,
                                            torch_num_threads,
                                            torch_num_interop_threads,
                                        )
                                    else:
                                        worker_output = args.outdir / f"worker-{mode}-{Path(learner_file).stem}-{batch_limit}-{torch_num_threads}-{torch_num_interop_threads}-{repeat}.jsonl"
                                        subprocess.run(
                                            worker_command(
                                                args,
                                                worker_output,
                                                learner_file,
                                                mode,
                                                batch_limit,
                                                torch_num_threads,
                                                torch_num_interop_threads,
                                            ),
                                            check=True,
                                        )
                                        result = json.loads(worker_output.read_text())
                                        worker_output.unlink()
                                except Exception as exc:
                                    result = {
                                        "event": "error",
                                        "mode": mode,
                                        "learner_file": learner_file,
                                        "batch_limit": batch_limit,
                                        "requested_torch_num_threads": torch_num_threads,
                                        "requested_torch_num_interop_threads": torch_num_interop_threads,
                                        "error": repr(exc),
                                    }
                                result["repeat"] = repeat
                                write_record(handle, result)
                                print(
                                    f"{result['event']}: mode={mode} learner={Path(learner_file).name} "
                                    f"batch_limit={batch_limit} threads={torch_num_threads} "
                                    f"interop={torch_num_interop_threads} repeat={repeat}",
                                    flush=True,
                                )
    return path


def main() -> None:
    args = parse_args()
    if args.worker_run:
        run_worker(args)
        return
    path = run_matrix(args)
    print(f"Wrote benchmark results to {path}", flush=True)


if __name__ == "__main__":
    main()
