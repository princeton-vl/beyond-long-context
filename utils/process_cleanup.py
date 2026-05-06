"""
Process cleanup utilities for HuggingFace Transformers, Accelerate, and other multi-process frameworks.

This module provides utilities to ensure all child processes are properly terminated
when a job is killed, preempted, or exits normally. This prevents zombie processes
from holding GPU memory.
"""

import os
import signal
import time
import subprocess
import logging
from typing import Optional, List

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    logging.warning("psutil not available, process cleanup will be limited")


logger = logging.getLogger(__name__)


def cleanup_child_processes(verbose: bool = True, force_after_seconds: int = 5):
    """
    Clean up all child processes spawned by the current process.

    This is essential for frameworks like Accelerate that spawn child processes
    which may not be automatically cleaned up on job termination.

    Args:
        verbose: If True, print cleanup progress
        force_after_seconds: How long to wait before force-killing (SIGKILL) processes

    Usage:
        # Register cleanup handler at start of script
        import atexit
        import signal
        from utils.process_cleanup import cleanup_child_processes, setup_cleanup_handlers

        setup_cleanup_handlers()  # Automatically registers cleanup

        # Or manually:
        atexit.register(cleanup_child_processes)
        signal.signal(signal.SIGTERM, lambda s, f: cleanup_child_processes())
    """
    if verbose:
        print("Cleaning up child processes...")

    if not HAS_PSUTIL:
        # Fallback: use process group kill
        try:
            if verbose:
                print("  Using process group kill (psutil not available)")
            # Kill entire process group
            os.killpg(os.getpgid(0), signal.SIGTERM)
            time.sleep(2)
        except Exception as e:
            if verbose:
                print(f"  Warning: Process group kill failed: {e}")
        return

    try:
        current_pid = os.getpid()
        current_process = psutil.Process(current_pid)

        # Get all child processes (recursive to catch grandchildren)
        children = current_process.children(recursive=True)

        if not children:
            if verbose:
                print("  No child processes to clean up")
            return

        if verbose:
            print(f"  Found {len(children)} child processes")

        # Step 1: Graceful termination (SIGTERM)
        for child in children:
            try:
                if child.is_running():
                    cmdline = ' '.join(child.cmdline()[:3]) if child.cmdline() else child.name()
                    if verbose:
                        print(f"    Terminating PID {child.pid} ({child.name()}): {cmdline}")
                    child.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                if verbose:
                    print(f"    Warning: Could not terminate PID {child.pid}: {e}")

        # Step 2: Wait for graceful shutdown
        if verbose:
            print(f"  Waiting {force_after_seconds}s for graceful shutdown...")
        time.sleep(force_after_seconds)

        # Step 3: Force kill remaining processes (SIGKILL)
        remaining = [child for child in children if child.is_running()]
        if remaining:
            if verbose:
                print(f"  Force killing {len(remaining)} remaining processes...")
            for child in remaining:
                try:
                    if child.is_running():
                        if verbose:
                            print(f"    Force killing PID {child.pid} ({child.name()})")
                        child.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                    if verbose:
                        print(f"    Warning: Could not kill PID {child.pid}: {e}")

        if verbose:
            print("  Child process cleanup complete")

    except Exception as e:
        if verbose:
            print(f"  Error during cleanup: {e}")


def cleanup_gpu_processes(gpu_ids: Optional[List[int]] = None, verbose: bool = True):
    """
    Clean up processes using specific GPUs.

    Args:
        gpu_ids: List of GPU IDs to clean (e.g., [0, 1]). If None, clean all GPUs.
        verbose: If True, print cleanup progress

    Note: Only kills Python/PyTorch processes, not all GPU processes (to avoid
    killing system processes or other users' jobs).
    """
    if verbose:
        gpu_str = ','.join(map(str, gpu_ids)) if gpu_ids else "all"
        print(f"Cleaning up GPU processes on GPUs: {gpu_str}")

    try:
        # Get list of GPUs to clean
        if gpu_ids is None:
            import torch
            if torch.cuda.is_available():
                gpu_ids = list(range(torch.cuda.device_count()))
            else:
                if verbose:
                    print("  No GPUs available")
                return

        for gpu_id in gpu_ids:
            try:
                # Query processes on this GPU
                result = subprocess.run(
                    ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader', f'--id={gpu_id}'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )

                if result.returncode != 0:
                    if verbose:
                        print(f"  Warning: Could not query GPU {gpu_id}")
                    continue

                # Parse PIDs
                pids_str = result.stdout.strip()
                if not pids_str:
                    if verbose:
                        print(f"  GPU {gpu_id}: No processes found")
                    continue

                pids = [int(pid.strip()) for pid in pids_str.split('\n') if pid.strip()]

                if verbose:
                    print(f"  GPU {gpu_id}: Found {len(pids)} processes")

                # Kill only Python/PyTorch processes
                for pid in pids:
                    try:
                        if not HAS_PSUTIL:
                            # Fallback: kill all PIDs (risky!)
                            os.kill(pid, signal.SIGKILL)
                            if verbose:
                                print(f"    Killed PID {pid} on GPU {gpu_id}")
                            continue

                        proc = psutil.Process(pid)

                        # Check if it's a Python/PyTorch process
                        cmdline = ' '.join(proc.cmdline()).lower()
                        name = proc.name().lower()

                        if any(keyword in cmdline or keyword in name
                               for keyword in ['python', 'torch', 'accelerate', 'transformers']):
                            if verbose:
                                print(f"    Killing PID {pid} ({proc.name()}) on GPU {gpu_id}")
                            proc.kill()
                        else:
                            if verbose:
                                print(f"    Skipping PID {pid} ({proc.name()}) - not a Python process")

                    except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError) as e:
                        if verbose:
                            print(f"    Warning: Could not access PID {pid}: {e}")

            except Exception as e:
                if verbose:
                    print(f"  Error cleaning GPU {gpu_id}: {e}")

    except Exception as e:
        if verbose:
            print(f"  Error during GPU cleanup: {e}")


def cleanup_all(gpu_ids: Optional[List[int]] = None, verbose: bool = True):
    """
    Comprehensive cleanup: child processes + GPU processes + CUDA cache.

    Args:
        gpu_ids: List of GPU IDs to clean. If None, clean all GPUs.
        verbose: If True, print cleanup progress

    This is the recommended cleanup function to use on exit.
    """
    if verbose:
        print("Starting comprehensive cleanup...")

    # Clean up child processes
    cleanup_child_processes(verbose=verbose)

    # Clean up GPU processes
    if gpu_ids or _has_gpu():
        cleanup_gpu_processes(gpu_ids=gpu_ids, verbose=verbose)

    # Clear CUDA cache
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            if verbose:
                print("  CUDA cache cleared and synchronized")
    except Exception as e:
        if verbose:
            print(f"  Warning: CUDA cleanup failed: {e}")

    # Force garbage collection
    try:
        import gc
        gc.collect()
        if verbose:
            print("  Garbage collection completed")
    except Exception as e:
        if verbose:
            print(f"  Warning: Garbage collection failed: {e}")

    if verbose:
        print("Comprehensive cleanup complete")


def _has_gpu() -> bool:
    """Check if GPUs are available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def setup_cleanup_handlers(gpu_ids: Optional[List[int]] = None, verbose: bool = True):
    """
    Register cleanup handlers for normal exit and signal termination.

    This should be called early in your script to ensure cleanup happens
    on all exit paths (normal exit, SIGTERM, SIGINT, etc.).

    Args:
        gpu_ids: List of GPU IDs to clean on exit. If None, clean all GPUs.
        verbose: If True, print cleanup progress

    Usage:
        # At the start of your script
        from utils.process_cleanup import setup_cleanup_handlers
        setup_cleanup_handlers()

        # Rest of your code...
    """
    import atexit

    # Cleanup function that will be called
    def cleanup_handler(signum=None, frame=None):
        if signum is not None:
            print(f"\nReceived signal {signum}, cleaning up...")
        cleanup_all(gpu_ids=gpu_ids, verbose=verbose)
        if signum is not None:
            # Exit after cleanup
            exit(143 if signum == signal.SIGTERM else 130)

    # Register for normal exit
    atexit.register(lambda: cleanup_all(gpu_ids=gpu_ids, verbose=verbose))

    # Register for signal termination
    signal.signal(signal.SIGTERM, cleanup_handler)  # scancel, kill
    signal.signal(signal.SIGINT, cleanup_handler)   # Ctrl+C

    if verbose:
        print("Cleanup handlers registered (SIGTERM, SIGINT, atexit)")


# Bash-compatible wrapper for use in shell scripts
def create_bash_cleanup_wrapper(output_file: str = "cleanup_wrapper.sh"):
    """
    Create a bash script that can be sourced to add cleanup handlers.

    This is useful for wrapping Accelerate launches and other shell scripts.

    Args:
        output_file: Path to write the bash script

    Usage in bash:
        source cleanup_wrapper.sh
        trap cleanup SIGTERM SIGINT SIGQUIT
        accelerate launch --num_processes=2 train.py &
        wait $!
    """
    bash_script = '''#!/bin/bash
# Cleanup handler for Accelerate and other multi-process frameworks
# Kills all child processes on exit/signal

cleanup() {
    echo "Cleaning up processes..."
    # Kill entire process group (including all children)
    kill -TERM -$$ 2>/dev/null
    wait
    echo "Cleanup complete"
    exit 143
}

# Export function so it can be used in traps
export -f cleanup
'''

    with open(output_file, 'w') as f:
        f.write(bash_script)

    # Make executable
    os.chmod(output_file, 0o755)
    print(f"Created bash cleanup wrapper: {output_file}")
    print(f"Usage: source {output_file} && trap cleanup SIGTERM SIGINT SIGQUIT")


if __name__ == "__main__":
    # Demo/test
    print("Testing process cleanup utilities...")
    print("\nChild process cleanup:")
    cleanup_child_processes(verbose=True)
    print("\nGPU process cleanup:")
    cleanup_gpu_processes(verbose=True)
    print("\nCreating bash wrapper:")
    create_bash_cleanup_wrapper()
