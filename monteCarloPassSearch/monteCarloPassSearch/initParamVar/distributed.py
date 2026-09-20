from __future__ import annotations

import multiprocessing as mp
from typing import Any, Callable, Dict, Iterable, Iterator, List

_WORKER_STATE = None
_WORKER_TASK_FN = None


def _pool_init(
    gpu_ids: List[int],
    idx_counter,
    idx_lock,
    init_fn: Callable[..., Any],
    init_kwargs: Dict[str, Any],
    task_fn: Callable[[Any, Any], Any],
) -> None:
    global _WORKER_STATE, _WORKER_TASK_FN
    with idx_lock:
        worker_idx = int(idx_counter.value)
        idx_counter.value += 1
    gpu_id = int(gpu_ids[worker_idx % len(gpu_ids)])
    _WORKER_STATE = init_fn(gpu_id=gpu_id, **init_kwargs)
    _WORKER_TASK_FN = task_fn


def _pool_exec(task: Any) -> Any:
    global _WORKER_STATE, _WORKER_TASK_FN
    return _WORKER_TASK_FN(task, _WORKER_STATE)


def run_distributed(
    tasks: Iterable[Any],
    *,
    gpu_ids: List[int],
    workers_per_gpu: int,
    init_fn: Callable[..., Any],
    init_kwargs: Dict[str, Any],
    task_fn: Callable[[Any, Any], Any],
) -> Iterator[Any]:
    """Run tasks on a fixed worker pool with deterministic GPU pinning."""
    if not gpu_ids:
        raise ValueError("gpu_ids must be non-empty")
    n_workers = max(1, int(workers_per_gpu) * len(gpu_ids))
    ctx = mp.get_context("spawn")
    counter = ctx.Value("i", 0)
    lock = ctx.Lock()

    with ctx.Pool(
        processes=n_workers,
        initializer=_pool_init,
        initargs=(gpu_ids, counter, lock, init_fn, init_kwargs, task_fn),
        maxtasksperchild=64,
    ) as pool:
        for item in pool.imap_unordered(_pool_exec, tasks, chunksize=1):
            yield item

