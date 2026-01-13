"""
Test script to verify multi-GPU scheduler functionality.
Run this before starting experiments to ensure everything works.
"""
import torch
from gpu_scheduler import GPUScheduler, parse_gpu_ids


def test_gpu_detection():
    """Test GPU detection"""
    print("\n" + "="*80)
    print("GPU DETECTION TEST")
    print("="*80)

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"✓ CUDA is available")
        print(f"✓ Number of GPUs detected: {num_gpus}")

        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print("✗ CUDA is not available. Will run on CPU.")

    print("="*80)


def test_parse_gpu_ids():
    """Test GPU ID parsing"""
    print("\n" + "="*80)
    print("GPU ID PARSING TEST")
    print("="*80)

    test_cases = [
        ("0", [0]),
        ("0,1,2", [0, 1, 2]),
        ("0-3", [0, 1, 2, 3]),
        ("0,2-4,7", [0, 2, 3, 4, 7]),
    ]

    for input_str, expected in test_cases:
        result = parse_gpu_ids(input_str)
        status = "✓" if result == expected else "✗"
        print(f"{status} '{input_str}' -> {result} (expected {expected})")

    print("="*80)


def dummy_worker(task_id, duration=1):
    """Dummy worker function for testing"""
    import time
    import torch

    time.sleep(duration)

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        return f"Task {task_id} completed on {device_name}"
    else:
        return f"Task {task_id} completed on CPU"


def test_scheduler(gpu_ids=None, num_tasks=5):
    """Test the GPU scheduler with dummy tasks"""
    print("\n" + "="*80)
    print(f"SCHEDULER TEST ({num_tasks} tasks)")
    print("="*80)

    print(f"GPU IDs: {gpu_ids if gpu_ids else 'CPU mode'}")

    scheduler = GPUScheduler(gpu_ids=gpu_ids, verbose=True)

    # Create dummy tasks
    tasks = [(i, 2) for i in range(num_tasks)]

    print(f"\nRunning {num_tasks} tasks (2 seconds each)...")
    print("This should complete in ~2 seconds per GPU")

    import time
    start_time = time.time()

    results = scheduler.run_tasks(tasks, dummy_worker)

    elapsed = time.time() - start_time

    print(f"\n✓ All tasks completed in {elapsed:.1f} seconds")

    # Check results
    successful = sum(1 for r in results if r is not None)
    print(f"✓ Successful tasks: {successful}/{num_tasks}")

    if gpu_ids and len(gpu_ids) > 1:
        expected_time = (num_tasks / len(gpu_ids)) * 2
        speedup = expected_time / elapsed
        print(f"✓ Speedup: {speedup:.2f}x (theoretical: {len(gpu_ids)}x)")

    print("="*80)


def test_imports():
    """Test that all required modules can be imported"""
    print("\n" + "="*80)
    print("IMPORT TEST")
    print("="*80)

    modules = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
        ('tqdm', 'tqdm'),
    ]

    all_ok = True
    for module_name, display_name in modules:
        try:
            __import__(module_name)
            print(f"✓ {display_name} imported successfully")
        except ImportError as e:
            print(f"✗ {display_name} import failed: {e}")
            all_ok = False

    if all_ok:
        print("\n✓ All required modules are available")
    else:
        print("\n✗ Some modules are missing. Run: pip install -r requirements.txt")

    print("="*80)


def main():
    print("\n" + "="*80)
    print("MULTI-GPU FRAMEWORK TEST SUITE")
    print("="*80)

    # Test imports
    test_imports()

    # Test GPU detection
    test_gpu_detection()

    # Test GPU ID parsing
    test_parse_gpu_ids()

    # Test scheduler with available GPUs
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()

        # Test with all GPUs
        print(f"\n>>> Testing with all {num_gpus} GPUs")
        test_scheduler(gpu_ids=list(range(num_gpus)), num_tasks=8)

        # Test with single GPU
        if num_gpus > 1:
            print(f"\n>>> Testing with single GPU (GPU 0)")
            test_scheduler(gpu_ids=[0], num_tasks=4)
    else:
        # Test CPU mode
        print(f"\n>>> Testing CPU mode")
        test_scheduler(gpu_ids=[], num_tasks=3)

    print("\n" + "="*80)
    print("TEST SUITE COMPLETED")
    print("="*80)
    print("\nIf all tests passed, you're ready to run experiments!")
    print("Example: python run_experiments.py --experiment 1 --gpus all --epochs 10")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
