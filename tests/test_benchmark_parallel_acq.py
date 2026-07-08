import pytest

from benchmarks.benchmark_parallel_acq import (
    configure_parent_torch_threads,
    parallel_settings,
    parse_args,
    restore_parent_autograd_multithreading,
    run_once,
    worker_torch_num_interop_threads_for_workers,
    worker_torch_num_threads_for_workers,
)


def test_default_worker_thread_mapping_keeps_total_threads_constant():
    args = parse_args(["--learner-file", "model.pt"])

    assert [worker_torch_num_threads_for_workers(workers, args) for workers in args.workers] == [32, 16, 8, 4]
    assert [worker_torch_num_interop_threads_for_workers(workers, args) for workers in args.workers] == [32, 16, 8, 4]


def test_parallel_settings_uses_default_thread_budget():
    args = parse_args(["--learner-file", "model.pt"])

    for workers in args.workers:
        settings = parallel_settings("cpu-parallel", workers, args)

        assert settings["num_workers"] == workers
        assert settings["start_method"] == "fork"
        assert settings["worker_torch_num_threads"] * workers == 64
        assert settings["worker_torch_num_interop_threads"] * workers == 64


def test_default_modes_run_both_parallel_paths():
    args = parse_args(["--learner-file", "model.pt"])

    assert args.modes == ["cpu-parallel", "cpu-parallel-from-checkpoint"]


def test_explicit_start_method_override_is_preserved():
    args = parse_args(["--learner-file", "model.pt", "--start-method", "fork"])

    assert parallel_settings("cpu-parallel", 2, args)["start_method"] == "fork"


def test_serial_mode_has_no_parallel_settings():
    args = parse_args(["--learner-file", "model.pt"])

    assert parallel_settings("cpu-serial", 1, args) is None


def test_explicit_worker_thread_override_is_preserved():
    args = parse_args(
        [
            "--learner-file",
            "model.pt",
            "--worker-torch-num-threads",
            "1",
            "--worker-torch-num-interop-threads",
            "2",
        ]
    )

    assert [worker_torch_num_threads_for_workers(workers, args) for workers in args.workers] == [1, 1, 1, 1]
    assert [worker_torch_num_interop_threads_for_workers(workers, args) for workers in args.workers] == [2, 2, 2, 2]


def test_nondivisible_thread_budget_raises_clear_error():
    args = parse_args(["--learner-file", "model.pt", "--workers", "3"])

    with pytest.raises(ValueError, match="total_worker_torch_threads must be divisible by workers"):
        worker_torch_num_threads_for_workers(3, args)
    with pytest.raises(ValueError, match="total_worker_torch_num_interop_threads must be divisible by workers"):
        worker_torch_num_interop_threads_for_workers(3, args)


def test_parallel_mode_constrains_parent_torch_threads(monkeypatch):
    calls = []

    monkeypatch.setattr("benchmarks.benchmark_parallel_acq.torch.set_num_threads", lambda value: calls.append(("threads", value)))
    monkeypatch.setattr(
        "benchmarks.benchmark_parallel_acq.torch.autograd.is_multithreading_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        "benchmarks.benchmark_parallel_acq.torch.autograd.set_multithreading_enabled",
        lambda value: calls.append(("autograd_threads", value)),
    )
    monkeypatch.setattr("benchmarks.benchmark_parallel_acq.torch.set_num_interop_threads", lambda value: calls.append(("interop", value)))

    restore_parent_autograd_multithreading()
    configure_parent_torch_threads({"enabled": True})

    assert calls == [("threads", 1), ("autograd_threads", False), ("interop", 1)]


def test_serial_mode_uses_configured_torch_threads(monkeypatch):
    calls = []
    wrapper = type("Wrapper", (), {"_configure_torch_threads": lambda self, settings: calls.append(("wrapper", settings))})()

    configure_parent_torch_threads(None, wrapper, "settings")

    assert calls == [("wrapper", "settings")]


def test_serial_mode_restores_parent_autograd_multithreading_after_parallel(monkeypatch):
    calls = []
    wrapper = type("Wrapper", (), {"_configure_torch_threads": lambda self, settings: calls.append(("wrapper", settings))})()

    monkeypatch.setattr(
        "benchmarks.benchmark_parallel_acq.torch.autograd.is_multithreading_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        "benchmarks.benchmark_parallel_acq.torch.autograd.set_multithreading_enabled",
        lambda value: calls.append(("autograd_threads", value)),
    )

    restore_parent_autograd_multithreading()
    configure_parent_torch_threads({"enabled": True})
    configure_parent_torch_threads(None, wrapper, "settings")

    assert calls == [("autograd_threads", False), ("autograd_threads", True), ("wrapper", "settings")]


def test_parallel_run_constrains_parent_before_loading_wrapper(monkeypatch):
    calls = []
    args = parse_args(["--learner-file", "model.pt", "--workers", "2"])

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

    class FakeSettings:
        pass

    settings = FakeSettings()

    class FakeWrapper:
        multi_fidelity = False

        def _resolve_optimization_settings(self):
            calls.append("resolve_settings")
            return settings

        def _configure_torch_threads(self, settings):
            calls.append(("wrapper_threads", settings))

        def load_model(self, learner_file):
            calls.append(("load_model", learner_file))
            return FakeModel()

        def get_candidates(self, **kwargs):
            calls.append("get_candidates")
            return FakeCandidates(), FakeValue()

    monkeypatch.setattr("benchmarks.benchmark_parallel_acq.torch.set_num_threads", lambda value: calls.append(("threads", value)))
    monkeypatch.setattr(
        "benchmarks.benchmark_parallel_acq.torch.autograd.is_multithreading_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        "benchmarks.benchmark_parallel_acq.torch.autograd.set_multithreading_enabled",
        lambda value: calls.append(("autograd_threads", value)),
    )
    monkeypatch.setattr("benchmarks.benchmark_parallel_acq.torch.set_num_interop_threads", lambda value: calls.append(("interop", value)))
    monkeypatch.setattr("benchmarks.benchmark_parallel_acq.torch.is_tensor", lambda value: True)

    def fake_load_wrapper(args, device):
        calls.append("load_wrapper")
        return FakeWrapper()

    monkeypatch.setattr("benchmarks.benchmark_parallel_acq.load_wrapper", fake_load_wrapper)

    run_once(args, "cpu-parallel", 2)

    assert calls[:4] == [("threads", 1), ("autograd_threads", False), ("interop", 1), "load_wrapper"]
    assert ("wrapper_threads", settings) not in calls
