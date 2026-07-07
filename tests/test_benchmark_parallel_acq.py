import pytest

from benchmarks.benchmark_parallel_acq import (
    parallel_settings,
    parse_args,
    worker_torch_num_threads_for_workers,
)


def test_default_worker_thread_mapping_keeps_total_threads_constant():
    args = parse_args(["--learner-file", "model.pt"])

    assert [worker_torch_num_threads_for_workers(workers, args) for workers in args.workers] == [32, 16, 8, 4]


def test_parallel_settings_uses_default_thread_budget():
    args = parse_args(["--learner-file", "model.pt"])

    for workers in args.workers:
        settings = parallel_settings("cpu-parallel", workers, args)

        assert settings["num_workers"] == workers
        assert settings["worker_torch_num_threads"] * workers == 64


def test_serial_mode_has_no_parallel_settings():
    args = parse_args(["--learner-file", "model.pt"])

    assert parallel_settings("cpu-serial", 1, args) is None


def test_explicit_worker_thread_override_is_preserved():
    args = parse_args(["--learner-file", "model.pt", "--worker-torch-num-threads", "1"])

    assert [worker_torch_num_threads_for_workers(workers, args) for workers in args.workers] == [1, 1, 1, 1]


def test_nondivisible_thread_budget_raises_clear_error():
    args = parse_args(["--learner-file", "model.pt", "--workers", "3"])

    with pytest.raises(ValueError, match="total_worker_torch_threads must be divisible by workers"):
        worker_torch_num_threads_for_workers(3, args)
