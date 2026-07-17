import json
from pathlib import Path

import pytest

from benchmarks import benchmark_native_acq
from benchmarks.benchmark_native_acq import (
    FULL_BATCH_LIMITS,
    FULL_TORCH_NUM_INTEROP_THREADS,
    FULL_TORCH_NUM_THREADS,
    QUICK_BATCH_LIMITS,
    QUICK_TORCH_NUM_INTEROP_THREADS,
    QUICK_TORCH_NUM_THREADS,
    benchmark_settings,
    learner_files_for_args,
    parse_args,
    run_matrix,
    run_once,
    train_or_reuse_gp_learner,
    worker_command,
)
from deepopt.models import AcquisitionOptimizationSettings


def test_zero_argument_defaults_select_full_benchmark(monkeypatch):
    monkeypatch.setattr("benchmarks.benchmark_native_acq.default_run_name", lambda: "run")

    args = parse_args([])

    assert args.learner_file == []
    assert args.outdir == Path("benchmark_results/acq_native/run")
    assert args.torch_num_threads == list(FULL_TORCH_NUM_THREADS)
    assert args.torch_num_interop_threads == list(FULL_TORCH_NUM_INTEROP_THREADS)
    assert args.batch_limits == list(FULL_BATCH_LIMITS)
    assert args.num_restarts == 32
    assert args.raw_samples == 8192
    assert args.maxiter == 200


def test_quick_defaults_are_smoke_safe(monkeypatch):
    monkeypatch.setattr("benchmarks.benchmark_native_acq.default_run_name", lambda: "quick")

    args = parse_args(["--quick"])

    assert args.torch_num_threads == list(QUICK_TORCH_NUM_THREADS)
    assert args.torch_num_interop_threads == list(QUICK_TORCH_NUM_INTEROP_THREADS)
    assert args.batch_limits == list(QUICK_BATCH_LIMITS)
    assert args.num_restarts == 4
    assert args.raw_samples == 64
    assert args.maxiter == 5


def test_explicit_native_benchmark_options_parse(tmp_path):
    args = parse_args(
        [
            "--torch-num-threads",
            "default",
            "3",
            "--torch-num-interop-threads",
            "default",
            "2",
            "--batch-limits",
            "1",
            "7",
            "--num-restarts",
            "9",
            "--raw-samples",
            "33",
            "--maxiter",
            "11",
            "--outdir",
            str(tmp_path),
            "--run-name",
            "custom",
        ]
    )

    assert args.torch_num_threads == ["default", 3]
    assert args.torch_num_interop_threads == ["default", 2]
    assert args.batch_limits == [1, 7]
    assert args.num_restarts == 9
    assert args.raw_samples == 33
    assert args.maxiter == 11
    assert args.outdir == tmp_path
    assert args.run_name == "custom"


def test_default_thread_sentinels_become_none_in_settings():
    args = parse_args(["--quick"])
    base_settings = AcquisitionOptimizationSettings(
        num_restarts_high=1,
        num_restarts_low=1,
        raw_samples_high=1,
        raw_samples_low=1,
        batch_limit_high=1,
        batch_limit_low=1,
        maxiter=1,
        n_fantasies=2,
        torch_num_threads="auto",
        torch_num_threads_fraction=0.8,
        torch_num_interop_threads=1,
    )

    settings = benchmark_settings(base_settings, args, 4, "default", "default")

    assert settings.torch_num_threads is None
    assert settings.torch_num_interop_threads is None
    assert settings.batch_limit_high == 4
    assert settings.num_restarts_high == args.num_restarts


def test_omitted_learner_file_uses_generated_paths(monkeypatch):
    args = parse_args(["--quick"])

    monkeypatch.setattr("benchmarks.benchmark_native_acq.generated_learner_files", lambda args, device: ["generated.ckpt"])

    assert learner_files_for_args(args, "cpu") == ["generated.ckpt"]


def test_run_once_uses_native_get_candidates(monkeypatch, tmp_path):
    calls = []
    args = parse_args(
        [
            "--quick",
            "--learner-file",
            "model.pt",
            "--run-name",
            "run",
            "--outdir",
            str(tmp_path),
        ]
    )

    class FakeValue:
        def detach(self):
            return self

        def cpu(self):
            return self

        def reshape(self, *_shape):
            return self

        def tolist(self):
            return [0.0]

    class FakeCandidates:
        shape = (1, 2)

    class FakeModel:
        def eval(self):
            calls.append("model.eval")

    class FakeWrapper:
        multi_fidelity = False

        def _resolve_optimization_settings(self):
            return AcquisitionOptimizationSettings(
                num_restarts_high=1,
                num_restarts_low=1,
                raw_samples_high=1,
                raw_samples_low=1,
                batch_limit_high=1,
                batch_limit_low=1,
                maxiter=1,
                n_fantasies=1,
                torch_num_threads="auto",
                torch_num_threads_fraction=0.8,
                torch_num_interop_threads=1,
            )

        def _configure_torch_threads(self, settings):
            calls.append(("configure", settings.torch_num_threads, settings.torch_num_interop_threads))

        def load_model(self, learner_file):
            calls.append(("load_model", learner_file))
            return FakeModel()

        def get_candidates(self, **kwargs):
            assert "parallel" + "_acq" not in kwargs
            calls.append(("get_candidates", kwargs["optimization_settings"]))
            return FakeCandidates(), FakeValue()

    monkeypatch.setattr("benchmarks.benchmark_native_acq.load_wrapper", lambda args, learner_file, device: FakeWrapper())
    monkeypatch.setattr("benchmarks.benchmark_native_acq.torch.get_num_threads", lambda: 3)
    monkeypatch.setattr("benchmarks.benchmark_native_acq.torch.get_num_interop_threads", lambda: 1)
    monkeypatch.setattr("benchmarks.benchmark_native_acq.torch.is_tensor", lambda value: True)

    result = run_once(args, "model.pt", "cpu-serial", 4, "default", 1)

    assert ("configure", None, 1) in calls
    assert result["batch_limit"] == 4
    assert result["requested_torch_num_threads"] == "default"
    assert result["effective_torch_num_threads"] == 3
    assert result["requested_torch_num_interop_threads"] == 1
    assert result["elapsed_seconds"] >= 0
    assert result["candidates_per_second"] is not None
    assert result["restarts_per_second"] is not None


def test_generated_learner_name_includes_seed_and_size(monkeypatch, tmp_path):
    dataset = (
        benchmark_native_acq.np.zeros((7, 2), dtype=benchmark_native_acq.np.float32),
        benchmark_native_acq.np.zeros((7, 1), dtype=benchmark_native_acq.np.float32),
        benchmark_native_acq.torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
        False,
    )

    class FakeModel:
        def __init__(self, **kwargs):
            pass

        def learn(self, outfile):
            Path(outfile).write_text("checkpoint")

    monkeypatch.setattr("benchmarks.benchmark_native_acq.get_checkpoint_metadata", lambda path: None)
    monkeypatch.setattr("benchmarks.benchmark_native_acq.get_deepopt_model", lambda model_type: FakeModel)

    path = train_or_reuse_gp_learner(tmp_path, "problem", dataset, 123, "cpu")

    assert path.name == "gp_problem_seed123_n7.ckpt"


def test_worker_command_launches_fresh_process(tmp_path):
    args = parse_args(["--quick", "--outdir", str(tmp_path), "--run-name", "records"])

    command = worker_command(args, tmp_path / "worker.jsonl", "model.pt", "cpu-serial", 4, "default", 1)

    assert command[0]
    assert "--worker-run" in command
    assert "--worker-torch-num-threads" in command
    assert "default" in command
    assert "--worker-torch-num-interop-threads" in command


def test_run_matrix_writes_jsonl_records(monkeypatch, tmp_path):
    args = parse_args(
        [
            "--quick",
            "--learner-file",
            "model.pt",
            "--torch-num-threads",
            "default",
            "--torch-num-interop-threads",
            "1",
            "--batch-limits",
            "4",
            "--run-name",
            "records",
            "--outdir",
            str(tmp_path),
        ]
    )

    args.in_process = True
    monkeypatch.setattr(
        "benchmarks.benchmark_native_acq.run_once",
        lambda *args, **kwargs: {
            "event": "result",
            "mode": "cpu-serial",
            "batch_limit": 4,
            "requested_torch_num_threads": "default",
            "requested_torch_num_interop_threads": 1,
            "elapsed_seconds": 1.0,
            "candidates_per_second": 1.0,
            "restarts_per_second": 4.0,
        },
    )
    monkeypatch.setattr("benchmarks.benchmark_native_acq.env_summary", lambda: {"host": "test"})

    path = run_matrix(args)
    records = [json.loads(line) for line in path.read_text().splitlines()]

    assert records[0]["event"] == "environment"
    assert records[0]["host"] == "test"
    assert records[1]["event"] == "result"
    assert records[1]["elapsed_seconds"] == 1.0


def test_run_matrix_records_unavailable_generated_gpu_problems(monkeypatch, tmp_path):
    args = parse_args(
        [
            "--quick",
            "--modes",
            "gpu-serial",
            "--problems",
            "hartmann",
            "augmented_hartmann",
            "--torch-num-threads",
            "default",
            "--torch-num-interop-threads",
            "1",
            "--batch-limits",
            "4",
            "--run-name",
            "records",
            "--outdir",
            str(tmp_path),
        ]
    )

    args.in_process = True
    monkeypatch.setattr("benchmarks.benchmark_native_acq.torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("benchmarks.benchmark_native_acq.env_summary", lambda: {"host": "test"})

    path = run_matrix(args)
    records = [json.loads(line) for line in path.read_text().splitlines()]

    assert [record["event"] for record in records] == ["environment", "skipped", "skipped"]
    assert [record["learner_file"] for record in records[1:]] == ["generated:hartmann", "generated:augmented_hartmann"]
    assert {record["skipped"] for record in records[1:]} == {"cuda unavailable"}
