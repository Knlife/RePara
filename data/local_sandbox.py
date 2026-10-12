# https://github.com/TIGER-AI-Lab/AceCoder/blob/main/data/inference/EvaluateInferencedCode.py

import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile  # Preload before restricted imports are installed in workers.
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional

import numpy as np

from utils.evaluation.worker import (
    start_worker as _start_worker,
    terminate_worker as _terminate_process,
    worker_environment,
)

cwd = os.getcwd()
cache_wd = os.environ.get("JUSTGRPO_CODE_CACHE_DIR", cwd + "/cache")
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
_PROCESS_STARTUP_GRACE = 3.0
# Keep the worker command independent from the training entry point. A
# multiprocessing ``spawn`` worker imports the parent's main module, which in
# training means importing torch/transformers and can exceed the one-second
# per-test timeout before user code starts. A small ``python -c`` worker only
# imports this module and receives the untrusted strings through stdin.
_WORKER_CODE = (
    "import io, json, sys; "
    "from data.local_sandbox import run_single_test; "
    "payload = json.load(sys.stdin); "
    "original_stdout = sys.stdout; "
    "original_dunder_stdout = sys.__stdout__; "
    "sink = io.StringIO(); "
    "sys.stdout = sys.__stdout__ = sink; "
    "result = run_single_test(payload['program'], payload['test'], "
    "payload['working_directory'], payload['timeout']); "
    "sys.stdout = original_stdout; "
    "sys.__stdout__ = original_dunder_stdout; "
    "sys.exit(42 if result else 1)"
)


def _worker_environment():
    return worker_environment(_REPO_ROOT, include_home=True)


class _ExecutionTimeout(TimeoutError):
    """Raised inside a worker when generated code exceeds its time budget."""


@contextmanager
def _time_limit(seconds: float):
    """Interrupt synchronous generated code after ``seconds`` in a worker."""
    seconds = float(seconds)
    if seconds <= 0:
        yield
        return

    def handle_timeout(signum, frame):  # pragma: no cover - frame is signal-owned
        raise _ExecutionTimeout(f"execution exceeded {seconds:.3f}s")

    previous_handler = None
    previous_timer = (0.0, 0.0)
    timer_started = False
    try:
        previous_handler = signal.getsignal(signal.SIGALRM)
        previous_timer = signal.getitimer(signal.ITIMER_REAL)
        signal.signal(signal.SIGALRM, handle_timeout)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        timer_started = True
    except (AttributeError, OSError, ValueError):
        # ``run_single_test`` is also a public helper and may be called from a
        # non-main thread/platform without SIGALRM. The subprocess deadline is
        # still enforced by the parent in that case.
        if previous_handler is not None:
            try:
                signal.signal(signal.SIGALRM, previous_handler)
            except (OSError, ValueError):
                pass
        yield
        return
    try:
        yield
    finally:
        if timer_started:
            try:
                signal.setitimer(signal.ITIMER_REAL, *previous_timer)
            except (OSError, ValueError):
                pass
            try:
                signal.signal(signal.SIGALRM, previous_handler)
            except (OSError, ValueError):
                pass


def run_single_test(
    func: str,
    test: str,
    working_directory: Optional[str] = None,
    max_execution_time: float = 1.0,
) -> int:
    """Run one test case and return 1 if passed, else 0."""
    execution_context = {"__builtins__": __builtins__}
    tempfile.gettempdir()
    try:
        with _time_limit(max_execution_time):
            with reliability_guard(working_directory=working_directory):
                exec(func, execution_context)
                exec(test, execution_context)
        return 1
    except BaseException:
        return 0
    finally:
        # Drop objects created by the generated program only after the guard has
        # restored Python's cleanup functions. Objects such as
        # TemporaryDirectory otherwise run their finalizers while rmtree is
        # intentionally disabled and flood the training logs with tracebacks.
        execution_context.clear()


def local_execute(pairs: List[tuple]) -> List[float]:
    """Execute code-test pairs and return pass rates.

    Args:
        pairs: List of (code, test_cases) tuples

    Returns:
        List of pass rates (0.0 - 1.0)
    """
    results = []
    for code, tests in pairs:
        result = get_successful_tests(code, tests)
        results.append(np.mean(result) if result else 0.0)
    return results


def get_successful_tests(
    program: str,
    tests: List[str],
    max_execution_time: float = 1.0,
    num_workers: int = 4
) -> List[int]:
    """Run a program against a list of tests.

    Args:
        program: Python program string
        tests: List of assert statements
        max_execution_time: Timeout per test in seconds
        num_workers: Number of parallel workers

    Returns:
        List of 0/1 indicating passed or not
    """
    test_length = len(tests)
    if test_length == 0:
        return []
    if not should_execute(program=program, tests=tests):
        return [0] * test_length

    worker_count = min(max(1, int(num_workers)), test_length)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                _run_test_in_subprocess,
                program,
                test,
                cache_wd,
                max_execution_time,
            )
            for test in tests
        ]
        # Iterating in submission order preserves the historical result order.
        return [future.result() for future in futures]


def _run_test_in_subprocess(
    program: str,
    test: str,
    working_directory: Optional[str],
    max_execution_time: float,
) -> int:
    """Run one generated test in an isolated worker."""
    try:
        timeout = float(max_execution_time)
        if timeout <= 0 or not math.isfinite(timeout):
            return 0
        payload = json.dumps({
            "program": program,
            "test": test,
            "working_directory": working_directory,
            "timeout": timeout,
        })
    except (TypeError, ValueError):
        return 0

    process = _start_worker(
        [sys.executable, "-c", _WORKER_CODE],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        start_new_session=True,
        text=True,
        env=_worker_environment(),
    )
    if process is None:
        return 0
    try:
        process.communicate(payload, timeout=timeout + _PROCESS_STARTUP_GRACE)
    except (subprocess.TimeoutExpired, OSError, ValueError):
        return 0
    finally:
        _terminate_process(process)
    return int(process.returncode == 42)


def should_execute(program: str, tests: List[str]) -> bool:
    """Determine if we should execute this program for safety reasons."""
    dangerous_commands = [
        "threading",
        "multiprocess",
        "multiprocessing",
        "import os",
        "from os",
        "shutil",
        "import torch",
        "from torch",
        "import sklearn",
        "from sklearn",
    ]
    for comm in dangerous_commands:
        if comm in program:
            return False
    return True


@contextmanager
def reliability_guard(
    maximum_memory_bytes: Optional[int] = None,
    working_directory: Optional[str] = None,
):
    """Temporarily disable destructive functions for code execution.

    WARNING: This is NOT a security sandbox. Use with caution.
    """
    import faulthandler
    import platform
    import sys

    missing = object()
    original_directory = os.getcwd()
    faulthandler_was_enabled = faulthandler.is_enabled()
    builtin_names = ("exit", "quit", "print", "input")
    os_names = (
        "system", "putenv", "remove", "removedirs", "rmdir", "fchdir",
        "setuid", "forkpty", "killpg", "rename", "renames", "truncate",
        "replace", "unlink", "fchmod", "fchown", "chmod", "chown",
        "chroot", "lchflags", "lchmod", "lchown", "getcwd", "chdir",
    )
    shutil_names = ("rmtree", "move", "chown")
    blocked_modules = ("ipdb", "joblib", "resource", "psutil", "tkinter")
    original_builtins = {
        name: getattr(__import__("builtins"), name, missing)
        for name in builtin_names
    }
    original_os = {name: getattr(os, name, missing) for name in os_names}
    original_shutil = {
        name: getattr(shutil, name, missing) for name in shutil_names
    }
    original_popen = subprocess.Popen
    original_modules = {
        name: sys.modules.get(name, missing) for name in blocked_modules
    }
    original_limits = []

    if maximum_memory_bytes is not None:
        import resource
        limit_names = [resource.RLIMIT_AS, resource.RLIMIT_DATA]
        if platform.uname().system != "Darwin":
            limit_names.append(resource.RLIMIT_STACK)
        for limit_name in limit_names:
            previous = resource.getrlimit(limit_name)
            original_limits.append((limit_name, previous))
            hard_limit = previous[1]
            soft_limit = maximum_memory_bytes
            if hard_limit != resource.RLIM_INFINITY:
                soft_limit = min(soft_limit, hard_limit)
            resource.setrlimit(limit_name, (soft_limit, hard_limit))

    import builtins

    def blocked_input(*args, **kwargs):
        raise EOFError("interactive input is disabled")

    worker_directory = working_directory or cache_wd
    os.makedirs(worker_directory, exist_ok=True)
    os.chdir(worker_directory)

    try:
        faulthandler.disable()
        builtins.exit = None
        builtins.quit = None
        builtins.print = lambda *args, **kwargs: None
        builtins.input = blocked_input

        for name in os_names:
            if original_os[name] is not missing:
                setattr(os, name, None)
        for name in shutil_names:
            if original_shutil[name] is not missing:
                setattr(shutil, name, None)
        subprocess.Popen = None
        for name in blocked_modules:
            sys.modules[name] = None

        # Generated programs frequently contain invalid escape sequences. They
        # should affect the reward, not fill every distributed rank's stderr.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    finally:
        for name, value in original_builtins.items():
            if value is missing:
                delattr(builtins, name)
            else:
                setattr(builtins, name, value)
        for name, value in original_os.items():
            if value is missing:
                if hasattr(os, name):
                    delattr(os, name)
            else:
                setattr(os, name, value)
        for name, value in original_shutil.items():
            if value is missing:
                if hasattr(shutil, name):
                    delattr(shutil, name)
            else:
                setattr(shutil, name, value)
        subprocess.Popen = original_popen
        for name, value in original_modules.items():
            if value is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
        os.chdir(original_directory)
        if original_limits:
            import resource
            for limit_name, previous in original_limits:
                resource.setrlimit(limit_name, previous)
        if faulthandler_was_enabled:
            faulthandler.enable()
        else:
            faulthandler.disable()
