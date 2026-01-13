"""
GPU task scheduler for running experiments in parallel across multiple GPUs.
"""
import os
import multiprocessing as mp
from queue import Empty
import torch


class GPUScheduler:
    """
    Manages parallel execution of tasks across multiple GPUs.

    Each GPU gets a worker process that pulls tasks from a shared queue.
    """

    def __init__(self, gpu_ids=None, verbose=True):
        """
        Initialize GPU scheduler.

        Args:
            gpu_ids: List of GPU IDs to use. If None, uses all available GPUs.
            verbose: Whether to print scheduling information.
        """
        if gpu_ids is None:
            # Use all available GPUs
            if torch.cuda.is_available():
                gpu_ids = list(range(torch.cuda.device_count()))
            else:
                gpu_ids = []

        self.gpu_ids = gpu_ids
        self.verbose = verbose

        if not self.gpu_ids:
            print("WARNING: No GPUs available. Will run on CPU sequentially.")
        elif self.verbose:
            print(f"GPU Scheduler initialized with GPUs: {self.gpu_ids}")

    def _worker(self, gpu_id, task_queue, result_queue, worker_func):
        """
        Worker process that executes tasks on a specific GPU.

        Args:
            gpu_id: GPU ID to use for this worker
            task_queue: Queue containing tasks to execute
            result_queue: Queue to put results into
            worker_func: Function to execute for each task
        """
        # Set this process to use only the assigned GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        if self.verbose:
            print(f"Worker started on GPU {gpu_id} (PID: {os.getpid()})")

        while True:
            try:
                # Get task from queue with timeout
                task = task_queue.get(timeout=1)

                if task is None:  # Poison pill to stop worker
                    if self.verbose:
                        print(f"Worker on GPU {gpu_id} received stop signal")
                    break

                task_id, task_args = task

                if self.verbose:
                    print(f"GPU {gpu_id}: Processing task {task_id}")

                try:
                    # Execute the task
                    result = worker_func(*task_args)
                    result_queue.put((task_id, result, None))  # (task_id, result, error)

                    if self.verbose:
                        print(f"GPU {gpu_id}: Completed task {task_id}")

                except Exception as e:
                    # Put error in result queue
                    result_queue.put((task_id, None, str(e)))
                    print(f"GPU {gpu_id}: Error in task {task_id}: {e}")

            except Empty:
                # No task available, continue waiting
                continue
            except Exception as e:
                print(f"Worker on GPU {gpu_id} encountered error: {e}")
                break

        if self.verbose:
            print(f"Worker on GPU {gpu_id} finished")

    def run_tasks(self, tasks, worker_func):
        """
        Run tasks in parallel across available GPUs.

        Args:
            tasks: List of task arguments. Each task is a tuple of arguments to pass to worker_func.
            worker_func: Function that takes task arguments and returns a result.
                         The function will be called as: worker_func(*task_args)

        Returns:
            List of results in the same order as tasks.
        """
        if not tasks:
            return []

        num_tasks = len(tasks)

        # If no GPUs available, run sequentially on CPU
        if not self.gpu_ids:
            print("Running tasks sequentially on CPU...")
            results = []
            for i, task_args in enumerate(tasks):
                print(f"Task {i+1}/{num_tasks}")
                result = worker_func(*task_args)
                results.append(result)
            return results

        # Create queues
        task_queue = mp.Queue()
        result_queue = mp.Queue()

        # Populate task queue
        for i, task_args in enumerate(tasks):
            task_queue.put((i, task_args))

        # Add poison pills (one per worker)
        for _ in self.gpu_ids:
            task_queue.put(None)

        # Start worker processes
        processes = []
        for gpu_id in self.gpu_ids:
            p = mp.Process(
                target=self._worker,
                args=(gpu_id, task_queue, result_queue, worker_func)
            )
            p.start()
            processes.append(p)

        if self.verbose:
            print(f"\nStarted {len(processes)} workers for {num_tasks} tasks")
            print("=" * 80)

        # Collect results
        results_dict = {}
        errors = {}

        for _ in range(num_tasks):
            task_id, result, error = result_queue.get()

            if error is not None:
                errors[task_id] = error
                results_dict[task_id] = None
            else:
                results_dict[task_id] = result

        # Wait for all workers to finish
        for p in processes:
            p.join()

        if self.verbose:
            print("=" * 80)
            print(f"All workers finished. Completed {num_tasks} tasks.")

        # Report errors if any
        if errors:
            print(f"\nWarning: {len(errors)} tasks failed:")
            for task_id, error in errors.items():
                print(f"  Task {task_id}: {error}")

        # Return results in original order
        results = [results_dict[i] for i in range(num_tasks)]
        return results


def parse_gpu_ids(gpu_str):
    """
    Parse GPU ID string into list of integers.

    Examples:
        "0" -> [0]
        "0,1,2" -> [0, 1, 2]
        "0-3" -> [0, 1, 2, 3]
        "0,2-4,7" -> [0, 2, 3, 4, 7]

    Args:
        gpu_str: String specifying GPU IDs

    Returns:
        List of GPU IDs
    """
    if not gpu_str:
        return None

    gpu_ids = []
    parts = gpu_str.split(',')

    for part in parts:
        part = part.strip()
        if '-' in part:
            # Range specification
            start, end = part.split('-')
            gpu_ids.extend(range(int(start), int(end) + 1))
        else:
            # Single GPU
            gpu_ids.append(int(part))

    return sorted(list(set(gpu_ids)))  # Remove duplicates and sort


if __name__ == '__main__':
    # Test the scheduler
    def dummy_task(task_id, duration):
        """Dummy task for testing"""
        import time
        import torch
        time.sleep(duration)
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        return f"Task {task_id} completed on {gpu_name}"

    # Test with dummy tasks
    scheduler = GPUScheduler(gpu_ids=[0, 1, 2, 3])
    tasks = [(i, 2) for i in range(10)]  # 10 tasks, each taking 2 seconds

    results = scheduler.run_tasks(tasks, dummy_task)

    print("\nResults:")
    for i, result in enumerate(results):
        print(f"  {i}: {result}")
