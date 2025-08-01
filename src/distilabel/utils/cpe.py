from typing import Callable, Iterable, Iterator, Generator, Any
import concurrent
from tqdm import tqdm
import itertools

def _shortened_task(task: dict) -> dict:
    """Limit printing."""
    return {k: str(v)[:200] for k, v in task.items()}

def continuous_parallel_execution(  # noqa: PLR0913
    function: Callable,
    tasks: Iterable[dict],
    process_type: str,
    num_workers: int,
    max_active_tasks: int,
    task_count: int,
    tqdm_desc: str,
):
    """Create a generator for data parallel tasks.

    Is a generator which handles parallel execution of tasks,
    keeping max_active_tasks # of tasks in the parallel
    executor at all times (which avoids overloading the executor).

    Use it like so:
    tqdm_desc = "Processing Tasks"
    cpe = continuous_parallel_execution(
        function=process_task,
        tasks=tasks,
        task_count=len(tasks),
        process_type="thread",
        num_workers=128,
        max_active_tasks=1024,
        tqdm_desc=tqdm_desc,
    )
    for task, result in cpe:
        handle_result(task, result, etc)

    Params:
    function: callable
        Takes unpacked tasks and returns a result
    tasks: Iterable[dict]
        Each dictionary in the iterable is kwargs for the function
    process_type: str
        One of "thread" or "process" for which type of parallel pool to use.
    num_workers: int
        The number of workers to use in ThreadPoolExecutor
    max_active_tasks: int
        The maximum number of active tasks in the executor.
        Should be maybe around 10x the num_workers.
        This makes it robust for long lists of tasks,
        as you can't submit them all to the executor at the same time.
    task_count: int
        Estimate for the total number of tasks
    tqdm_desc: str
        The description for the tqdm bar

    Returns
    -------
        task: dict
            The task that yielded the result
        result: Any
            The return value of the function for the task

    """
    task_iter = tasks if isinstance(tasks, Iterator) else iter(tasks)
    progress_bar = tqdm(task_iter, total=task_count, desc=tqdm_desc)
    pool = (
        concurrent.futures.ThreadPoolExecutor
        if process_type == "thread"
        else concurrent.futures.ProcessPoolExecutor
    )
    with pool(max_workers=num_workers) as executor:
        futures = {
            executor.submit(function, **task): task
            for task in itertools.islice(task_iter, max_active_tasks)
        }
        while futures:
            done, _ = concurrent.futures.wait(
                futures,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                progress_bar.update(1)
                progress_bar.refresh()

                task = futures.pop(future)
                try:
                    result = future.result()
                except Exception as e:
                    print(f"Error processing task {_shortened_task(task)}: {e}")
                    result = None
                yield task, result

            for task in itertools.islice(task_iter, len(done)):
                future = executor.submit(function, **task)
                futures[future] = task

    progress_bar.close()

