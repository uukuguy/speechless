""" Utilities for running functions in parallel processes. """
"""
from multiprocess_utils import run_tasks_in_parallel
    outputs = []
    arguments = [
        (
            prompt,
            self.cache,  ## pass the cache as argument for cache check
            self.args,  ## pass the args as argument for cache check
            self._run_single,  ## pass the _run_single method as argument because of multiprocessing
        )
        for prompt in prompts
    ]
    if self.args.multiprocess > 1:
        parallel_outputs = run_tasks_in_parallel(
            self.run_single,
            arguments,
            self.args.multiprocess,
            use_progress_bar=True,
        )
        for output in parallel_outputs:
            if output.is_success():
                outputs.append(output.result)
            else:
                print("Failed to run the model for some prompts")
                print(output.status)
                print(output.exception_tb)
                outputs.extend([""] * self.args.n)
    else:
        outputs = [self.run_single(argument) for argument in tqdm(arguments)]

"""
import sys
import resource
import multiprocessing as mp
import queue
import traceback
from enum import Enum
from typing import Callable, Optional, Dict, Any, List, Iterator
from concurrent.futures import TimeoutError

import attrs
from tqdm import tqdm
from pebble import concurrent, ProcessPool, ProcessExpired


class FuncTimeoutError(TimeoutError):
    pass


def generate_queue() -> mp.Queue:
    """
    Generates a queue that can be shared amongst processes
    Returns:
        (multiprocessing.Queue): A queue instance
    """
    manager = mp.Manager()
    return manager.Queue()


QueueEmptyException = queue.Empty


def run_func_in_process(
    func: Callable,
    *args,
    _timeout: Optional[int] = None,
    _use_spawn: bool = True,
    **kwargs,
):
    """
    Runs the provided function in a separate process with the supplied args
    and kwargs. The args, kwargs, and
    return values must all be pickle-able.
    Args:
        func: The function to run.
        *args: Positional args, if any.
        _timeout: A timeout to use for the function.
        _use_spawn: The 'spawn' multiprocess context is used.'fork' otherwise.
        **kwargs: Keyword args, if any.
    Returns:
        The result of executing the function.
    """
    mode = "spawn" if _use_spawn else "fork"
    c_func = concurrent.process(timeout=_timeout, context=mp.get_context(mode))(func)
    future = c_func(*args, **kwargs)

    try:
        result = future.result()
        return result

    except TimeoutError:
        raise FuncTimeoutError


class TaskRunStatus(Enum):
    SUCCESS = 0
    EXCEPTION = 1
    TIMEOUT = 2
    PROCESS_EXPIRED = 3


@attrs.define(eq=False, repr=False)
class TaskResult:
    status: TaskRunStatus

    result: Optional[Any] = None
    exception_tb: Optional[str] = None

    def is_success(self) -> bool:
        return self.status == TaskRunStatus.SUCCESS

    def is_timeout(self) -> bool:
        return self.status == TaskRunStatus.TIMEOUT

    def is_exception(self) -> bool:
        return self.status == TaskRunStatus.EXCEPTION

    def is_process_expired(self) -> bool:
        return self.status == TaskRunStatus.PROCESS_EXPIRED


def initializer(limit):
    """Set maximum amount of memory each worker process can allocate."""
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (limit, hard))


def run_tasks_in_parallel_iter(
    func: Callable,
    tasks: List[Any],
    num_workers: int = 2,
    timeout_per_task: Optional[int] = None,
    use_progress_bar: bool = False,
    progress_bar_desc: Optional[str] = None,
    max_tasks_per_worker: Optional[int] = None,
    use_spawn: bool = True,
    max_mem: int = 1024 * 1024 * 1024 * 4,
) -> Iterator[TaskResult]:
    """
    Args:
        func: The function to run. The function must accept a single argument.
        tasks: A list of tasks i.e. arguments to func.
        num_workers: Maximum number of parallel workers.
        timeout_per_task: The timeout, in seconds, to use per task.
        use_progress_bar: Whether to use a progress bar. Default False.
        progress_bar_desc: String to display in the progress bar. Default None.
        max_tasks_per_worker: Maximum number of tasks assigned
        to a single process / worker. None means infinite.
            Use 1 to force a restart.
        use_spawn: The 'spawn' multiprocess context is used. 'fork' otherwise.
    Returns:
        A list of TaskResult objects, one per task.
    """

    mode = "spawn" if use_spawn else "fork"

    with ProcessPool(
        max_workers=num_workers,
        max_tasks=0 if max_tasks_per_worker is None else max_tasks_per_worker,
        context=mp.get_context(mode),
    ) as pool:
        future = pool.map(func, tasks, timeout=timeout_per_task)

        iterator = future.result()
        if use_progress_bar:
            pbar = tqdm(
                desc=progress_bar_desc,
                total=len(tasks),
                dynamic_ncols=True,
                file=sys.stdout,
            )
        else:
            pbar = None

        succ = timeouts = exceptions = expirations = 0

        while True:
            try:
                result = next(iterator)

            except StopIteration:
                break

            except TimeoutError as error:
                yield TaskResult(status=TaskRunStatus.TIMEOUT, )

                timeouts += 1

            except ProcessExpired as error:
                yield TaskResult(status=TaskRunStatus.PROCESS_EXPIRED, )
                expirations += 1

            except Exception as error:
                exception_tb = traceback.format_exc()

                yield TaskResult(
                    status=TaskRunStatus.EXCEPTION,
                    exception_tb=exception_tb,
                )
                exceptions += 1

            else:
                yield TaskResult(
                    status=TaskRunStatus.SUCCESS,
                    result=result,
                )

                succ += 1

            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(succ=succ, timeouts=timeouts, exc=exceptions, p_exp=expirations)
                sys.stdout.flush()
                sys.stderr.flush()


def run_tasks_in_parallel(
    func: Callable,
    tasks: List[Any],
    num_workers: int = 2,
    timeout_per_task: Optional[int] = None,
    use_progress_bar: bool = False,
    progress_bar_desc: Optional[str] = None,
    max_tasks_per_worker: Optional[int] = None,
    use_spawn: bool = True,
) -> List[TaskResult]:
    """
    Args:
        func: The function to run. The function must accept a single argument.
        tasks: A list of tasks i.e. arguments to func.
        num_workers: Maximum number of parallel workers.
        timeout_per_task: The timeout, in seconds, to use per task.
        use_progress_bar: Whether to use a progress bar. Defaults False.
        progress_bar_desc: String to display in the progress bar. Default None.
        max_tasks_per_worker: Maximum number of tasks assigned to a single
        process / worker. None means infinite.
            Use 1 to force a restart.
        use_spawn: The 'spawn' multiprocess context is used. 'fork' otherwise.
    Returns:
        A list of TaskResult objects, one per task.
    """

    task_results: List[TaskResult] = list(
        run_tasks_in_parallel_iter(
            func=func,
            tasks=tasks,
            num_workers=num_workers,
            timeout_per_task=timeout_per_task,
            use_progress_bar=use_progress_bar,
            progress_bar_desc=progress_bar_desc,
            max_tasks_per_worker=max_tasks_per_worker,
            use_spawn=use_spawn,
        )
    )

    return task_results


import os
from multiprocessing import Pool
from typing import Iterable

"""
outputs = []
params_list = [
    { 
        'prompt': prompt,
        'temperture': 0.6,
        'max_tokens': 8192,
    } 
    for prompt in prompts
]
if args.num_processes > 1:
    parallel_outputs = run_in_multiprocessing(
        run_single,
        params_list,
        num_processes=args.num_processes,
        chunk_size=args.pool_map_chunk_size,
        unordered=True,
        use_progress_bar=True,
        progress_bar_desc="Generating responses",
    )
    for output in parallel_outputs:
        if output.is_success():
            outputs.append(output.result)
        else:
            print("Failed to run the model for some prompts")
            print(output.status)
            print(output.exception_tb)
            outputs.extend([""] * self.args.n)
else:
    outputs = [run_func_in_process(run_single, params) for params in tqdm(params_list)]
    
"""


def run_func_in_multiprocessing_iterator(
    func,
    params_list: Iterable,
    num_processes: int = 4,
    chunk_size: int = 16,
    unordered=False,
    use_progress_bar: bool = False,
    progress_bar_desc: Optional[str] = None,
):
    num_cpus = os.cpu_count()
    if num_processes <= 0:
        num_processes = num_cpus
    num_processes = min(num_processes, num_cpus)

    if use_progress_bar:
        pbar = tqdm(
            desc=progress_bar_desc,
            total=len(params_list),
            dynamic_ncols=True,
            file=sys.stdout,
        )
    else:
        pbar = None


    with Pool(processes=num_processes) as pool:
        imap_func = pool.imap_unordered if unordered else pool.imap

        success, exceptions = 0, 0
        for result in imap_func(func, params_list, chunksize=chunk_size):
            yield TaskResult(
                status=TaskRunStatus.SUCCESS,
                result=result,
            )
            success += 1

            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(success=success, exceptions=exceptions)
                sys.stdout.flush()
                sys.stderr.flush()

def run_func_in_multiprocessing(*args, **kwargs):
    return list(run_func_in_multiprocessing_iterator(*args, **kwargs))


def initialize_multiprocessing(start_method="spawn"):
    import multiprocessing as mp
    mp.set_start_method(start_method)
