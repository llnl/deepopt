"""
Internal helpers for process-parallel acquisition optimization.
"""
import multiprocessing as mp
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from botorch.acquisition.acquisition import OneShotAcquisitionFunction
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.exceptions.warnings import OptimizationWarning
from botorch.optim.initializers import gen_batch_initial_conditions, gen_one_shot_kg_initial_conditions
from botorch.optim.optimize import optimize_acqf


@dataclass(frozen=True)
class ParallelAcqSettings:
    enabled: bool = False
    num_workers: int = 1
    start_method: str = "spawn"
    worker_torch_num_threads: int = 1
    worker_torch_num_interop_threads: Optional[int] = 1


def resolve_parallel_acq_settings(parallel_acq: Optional[Union[Dict[str, Any], ParallelAcqSettings]]) -> ParallelAcqSettings:
    if parallel_acq is None:
        return ParallelAcqSettings()
    if isinstance(parallel_acq, ParallelAcqSettings):
        settings = parallel_acq
    else:
        if not isinstance(parallel_acq, dict):
            raise TypeError("parallel_acq must be None, a dict, or ParallelAcqSettings.")
        valid_keys = set(ParallelAcqSettings.__dataclass_fields__)
        unknown_keys = set(parallel_acq).difference(valid_keys)
        if unknown_keys:
            raise ValueError(f"Unknown parallel_acq setting(s): {sorted(unknown_keys)}")
        settings = ParallelAcqSettings(**parallel_acq)
    if settings.num_workers <= 0:
        raise ValueError("parallel_acq.num_workers must be positive.")
    if settings.worker_torch_num_threads <= 0:
        raise ValueError("parallel_acq.worker_torch_num_threads must be positive.")
    if settings.worker_torch_num_interop_threads is not None and settings.worker_torch_num_interop_threads <= 0:
        raise ValueError("parallel_acq.worker_torch_num_interop_threads must be positive or None.")
    return settings


def split_tensor_by_workers(tensor: torch.Tensor, num_workers: int) -> List[torch.Tensor]:
    if num_workers <= 1 or tensor.shape[0] <= 1:
        return [tensor]
    chunk_size = max(1, (tensor.shape[0] + num_workers - 1) // num_workers)
    return [chunk for chunk in tensor.split(chunk_size) if chunk.shape[0] > 0]


def split_list_by_workers(values: Sequence[Any], num_workers: int) -> List[List[Any]]:
    if num_workers <= 1 or len(values) <= 1:
        return [list(values)]
    chunk_size = max(1, (len(values) + num_workers - 1) // num_workers)
    return [list(values[i : i + chunk_size]) for i in range(0, len(values), chunk_size)]


def select_best(candidates: torch.Tensor, acq_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    best = torch.argmax(acq_values.view(-1), dim=0)
    return candidates[best], acq_values.view(-1)[best]


def _set_worker_torch_threads(settings: ParallelAcqSettings) -> None:
    torch.set_num_threads(settings.worker_torch_num_threads)
    if settings.worker_torch_num_interop_threads is not None:
        try:
            torch.set_num_interop_threads(settings.worker_torch_num_interop_threads)
        except RuntimeError:
            pass


_OPTIMIZE_ACQF_WORKER_SHARED_KWARGS: Optional[Dict[str, Any]] = None


def _initialize_optimize_acqf_worker(shared_kwargs: Dict[str, Any], settings: ParallelAcqSettings) -> None:
    global _OPTIMIZE_ACQF_WORKER_SHARED_KWARGS
    _OPTIMIZE_ACQF_WORKER_SHARED_KWARGS = shared_kwargs
    _set_worker_torch_threads(settings)


def _optimize_acqf_worker(payload: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    shared_kwargs = _OPTIMIZE_ACQF_WORKER_SHARED_KWARGS
    if shared_kwargs is None:
        settings = payload["settings"]
        _set_worker_torch_threads(settings)
        worker_kwargs = {key: value for key, value in payload.items() if key != "settings"}
    else:
        worker_kwargs = dict(shared_kwargs)
        worker_kwargs.update(payload)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        candidates, acq_values = optimize_acqf(**worker_kwargs)
    warning_messages = [str(w.message) for w in caught]
    return candidates.detach().cpu(), acq_values.detach().cpu(), warning_messages


def _gen_initial_conditions(
    acq_function: Any,
    bounds: torch.Tensor,
    q: int,
    num_restarts: int,
    raw_samples: Optional[int],
    fixed_features: Optional[Dict[int, float]],
    options: Optional[Dict[str, Any]],
    inequality_constraints: Optional[List[Tuple[torch.Tensor, torch.Tensor, float]]],
    equality_constraints: Optional[List[Tuple[torch.Tensor, torch.Tensor, float]]],
    seed_offset: int = 0,
) -> torch.Tensor:
    if raw_samples is None:
        raise ValueError("Must specify raw_samples when batch_initial_conditions is None.")
    init_options = dict(options or {})
    if seed_offset and "seed" in init_options and init_options["seed"] is not None:
        init_options["seed"] = int(init_options["seed"]) + seed_offset
    init_func = gen_one_shot_kg_initial_conditions if isinstance(acq_function, qKnowledgeGradient) else gen_batch_initial_conditions
    return init_func(
        acq_function=acq_function,
        bounds=bounds,
        q=q,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        fixed_features=fixed_features,
        options=init_options,
        inequality_constraints=inequality_constraints,
        equality_constraints=equality_constraints,
    )


def _should_use_serial(settings: ParallelAcqSettings, bounds: torch.Tensor, num_restarts: int) -> bool:
    return not settings.enabled or settings.num_workers <= 1 or num_restarts <= 1 or bounds.device.type != "cpu"


def parallel_optimize_acqf(
    *,
    settings: ParallelAcqSettings,
    acq_function: Any,
    bounds: torch.Tensor,
    q: int,
    num_restarts: int,
    raw_samples: Optional[int] = None,
    options: Optional[Dict[str, Any]] = None,
    inequality_constraints: Optional[List[Tuple[torch.Tensor, torch.Tensor, float]]] = None,
    equality_constraints: Optional[List[Tuple[torch.Tensor, torch.Tensor, float]]] = None,
    nonlinear_inequality_constraints: Optional[List[Any]] = None,
    fixed_features: Optional[Dict[int, float]] = None,
    post_processing_func: Optional[Any] = None,
    batch_initial_conditions: Optional[torch.Tensor] = None,
    return_best_only: bool = True,
    sequential: bool = False,
    **kwargs: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if (
        _should_use_serial(settings, bounds, num_restarts)
        or nonlinear_inequality_constraints
        or (fixed_features is not None and len(fixed_features) == bounds.shape[-1])
    ):
        return optimize_acqf(
            acq_function=acq_function,
            bounds=bounds,
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options=options,
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
            nonlinear_inequality_constraints=nonlinear_inequality_constraints,
            fixed_features=fixed_features,
            post_processing_func=post_processing_func,
            batch_initial_conditions=batch_initial_conditions,
            return_best_only=return_best_only,
            sequential=sequential,
            **kwargs,
        )

    if sequential and q > 1:
        if not return_best_only:
            raise NotImplementedError("return_best_only=False only supported for joint optimization.")
        if isinstance(acq_function, OneShotAcquisitionFunction):
            raise NotImplementedError("sequential optimization is not supported for one-shot acquisition functions.")
        candidate_list = []
        acq_value_list = []
        base_X_pending = acq_function.X_pending
        for _ in range(q):
            candidate, acq_value = parallel_optimize_acqf(
                settings=settings,
                acq_function=acq_function,
                bounds=bounds,
                q=1,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                options=options,
                inequality_constraints=inequality_constraints,
                equality_constraints=equality_constraints,
                fixed_features=fixed_features,
                post_processing_func=post_processing_func,
                return_best_only=True,
                sequential=False,
                **kwargs,
            )
            candidate_list.append(candidate)
            acq_value_list.append(acq_value)
            candidates = torch.cat(candidate_list, dim=-2)
            acq_function.set_X_pending(torch.cat([base_X_pending, candidates], dim=-2) if base_X_pending is not None else candidates)
        acq_function.set_X_pending(base_X_pending)
        return candidates, torch.stack(acq_value_list)

    initial_conditions_provided = batch_initial_conditions is not None
    if not initial_conditions_provided:
        batch_initial_conditions = _gen_initial_conditions(
            acq_function=acq_function,
            bounds=bounds,
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            fixed_features=fixed_features,
            options=options,
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
        )

    chunks = split_tensor_by_workers(batch_initial_conditions, settings.num_workers)

    def run_workers(initial_condition_chunks: List[torch.Tensor]) -> List[Tuple[torch.Tensor, torch.Tensor, List[str]]]:
        worker_payloads = [
            {
                "num_restarts": chunk.shape[0],
                "batch_initial_conditions": chunk,
            }
            for chunk in initial_condition_chunks
        ]
        shared_kwargs = {
            "acq_function": acq_function,
            "bounds": bounds,
            "q": q,
            "raw_samples": raw_samples,
            "options": options or {},
            "inequality_constraints": inequality_constraints,
            "equality_constraints": equality_constraints,
            "nonlinear_inequality_constraints": None,
            "fixed_features": fixed_features,
            "post_processing_func": post_processing_func,
            "return_best_only": False,
            "sequential": False,
            **kwargs,
        }
        if settings.start_method == "fork":
            context = mp.get_context(settings.start_method)
            with context.Pool(
                processes=min(settings.num_workers, len(worker_payloads)),
                initializer=_initialize_optimize_acqf_worker,
                initargs=(shared_kwargs, settings),
            ) as pool:
                return pool.map(_optimize_acqf_worker, worker_payloads)
        worker_payloads = [{"settings": settings, **shared_kwargs, **payload} for payload in worker_payloads]
        context = mp.get_context(settings.start_method)
        with context.Pool(processes=min(settings.num_workers, len(worker_payloads))) as pool:
            return pool.map(_optimize_acqf_worker, worker_payloads)

    worker_results = run_workers(chunks)

    warning_messages = [message for _, _, messages in worker_results for message in messages]
    if warning_messages and not initial_conditions_provided:
        batch_initial_conditions = _gen_initial_conditions(
            acq_function=acq_function,
            bounds=bounds,
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            fixed_features=fixed_features,
            options=options,
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
            seed_offset=1,
        )
        worker_results = run_workers(split_tensor_by_workers(batch_initial_conditions, settings.num_workers))
        warning_messages = [message for _, _, messages in worker_results for message in messages]
        if warning_messages:
            warnings.warn("Optimization failed on the second parallel try after generating new initial conditions.", RuntimeWarning)
    elif warning_messages:
        warnings.warn(
            "Optimization warnings were raised in parallel acquisition workers: "
            f"{warning_messages}",
            RuntimeWarning,
        )

    candidates = torch.cat([result[0] for result in worker_results]).to(device=bounds.device)
    acq_values = torch.cat([result[1].view(-1) for result in worker_results]).to(device=bounds.device)
    if return_best_only:
        candidates, acq_values = select_best(candidates, acq_values)
    return candidates, acq_values


def parallel_optimize_acqf_mixed(
    *,
    settings: ParallelAcqSettings,
    acq_function: Any,
    bounds: torch.Tensor,
    q: int,
    num_restarts: int,
    fixed_features_list: List[Dict[int, float]],
    raw_samples: Optional[int] = None,
    options: Optional[Dict[str, Any]] = None,
    inequality_constraints: Optional[List[Tuple[torch.Tensor, torch.Tensor, float]]] = None,
    equality_constraints: Optional[List[Tuple[torch.Tensor, torch.Tensor, float]]] = None,
    post_processing_func: Optional[Any] = None,
    batch_initial_conditions: Optional[torch.Tensor] = None,
    **kwargs: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not fixed_features_list:
        raise ValueError("fixed_features_list must be non-empty.")
    if _should_use_serial(settings, bounds, num_restarts):
        from botorch.optim.optimize import optimize_acqf_mixed

        return optimize_acqf_mixed(
            acq_function=acq_function,
            bounds=bounds,
            q=q,
            num_restarts=num_restarts,
            fixed_features_list=fixed_features_list,
            raw_samples=raw_samples,
            options=options,
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
            post_processing_func=post_processing_func,
            batch_initial_conditions=batch_initial_conditions,
            **kwargs,
        )

    if q > 1:
        base_X_pending = acq_function.X_pending
        candidate_list = []
        for _ in range(q):
            candidate, _ = parallel_optimize_acqf_mixed(
                settings=settings,
                acq_function=acq_function,
                bounds=bounds,
                q=1,
                num_restarts=num_restarts,
                fixed_features_list=fixed_features_list,
                raw_samples=raw_samples,
                options=options,
                inequality_constraints=inequality_constraints,
                equality_constraints=equality_constraints,
                post_processing_func=post_processing_func,
                batch_initial_conditions=batch_initial_conditions,
                **kwargs,
            )
            candidate_list.append(candidate)
            candidates = torch.cat(candidate_list, dim=-2)
            acq_function.set_X_pending(torch.cat([base_X_pending, candidates], dim=-2) if base_X_pending is not None else candidates)
        acq_function.set_X_pending(base_X_pending)
        if isinstance(acq_function, OneShotAcquisitionFunction):
            return candidates, acq_function.evaluate(X=candidates, bounds=bounds)
        return candidates, acq_function(candidates)

    candidates_list = []
    acq_values_list = []
    for fixed_features in fixed_features_list:
        candidate, acq_value = parallel_optimize_acqf(
            settings=settings,
            acq_function=acq_function,
            bounds=bounds,
            q=1,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options=options,
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
            fixed_features=fixed_features,
            post_processing_func=post_processing_func,
            batch_initial_conditions=batch_initial_conditions,
            return_best_only=True,
            sequential=False,
            **kwargs,
        )
        candidates_list.append(candidate)
        acq_values_list.append(acq_value)
    acq_values = torch.stack([value.view(-1)[0] for value in acq_values_list])
    best = torch.argmax(acq_values)
    return candidates_list[best], acq_values[best]
