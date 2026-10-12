"""Sandbox code execution and pass@k evaluation for HumanEval/MBPP."""

import contextlib
import faulthandler
import gzip
import io
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from utils.evaluation.worker import (
    start_worker as _start_worker,
    terminate_worker as _terminate_process,
    worker_environment,
)

CODE_BLOCK_RE = re.compile(r"```python\n(.*?)```", re.DOTALL)
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
_PROCESS_STARTUP_GRACE = 3.0
_CODE_EXEC_WORKER_CODE = (
    "import json, sys; "
    "from pathlib import Path; "
    "from utils.evaluation.code_exec import _run_test_code; "
    "payload = json.load(sys.stdin); "
    "result = _run_test_code(payload['test_code'], payload['timeout']); "
    "Path(payload['result_path']).write_text(json.dumps(result), encoding='utf-8')"
)


def _worker_environment():
    return worker_environment(_REPO_ROOT)


# ---- JSONL I/O ----

def stream_jsonl(path):
    if path.endswith(".gz"):
        with open(path, "rb") as raw:
            with gzip.open(raw, "rt", encoding="utf-8") as fp:
                for line in fp:
                    if line.strip():
                        yield json.loads(line)
    else:
        with open(path, "r", encoding="utf-8") as fp:
            for line in fp:
                if line.strip():
                    yield json.loads(line)


def read_problems(path):
    return {item["task_id"]: item for item in stream_jsonl(path)}


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as fp:
        fp.writelines(json.dumps(row) + "\n" for row in rows)


# ---- Pass@k ----

def estimate_pass_at_k(num_samples, num_correct, k):
    def _est(n, c, k):
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
    if isinstance(num_samples, int):
        num_samples = [num_samples] * len(num_correct)
    return np.array([_est(int(n), int(c), k) for n, c in zip(num_samples, num_correct)])


# ---- Test code assembly ----

def extract_completion_code(completion):
    match = CODE_BLOCK_RE.search(completion)
    return match.group(1).strip() if match else completion


def build_test_code(sample, problems, is_mbpp=False):
    tid = sample["task_id"]
    code = extract_completion_code(sample["completion"])
    p = problems[tid]
    if is_mbpp:
        return code + "\n" + "\n".join(p["test"])
    return p["prompt"] + code + "\n" + p["test"] + "\n" + f"check({p['entry_point']})"


# ---- Sandbox execution ----

class _Timeout(Exception):
    pass


class _WOStringIO(io.StringIO):
    def read(self, *a, **k): raise IOError
    def readline(self, *a, **k): raise IOError
    def readlines(self, *a, **k): raise IOError
    def readable(self, *a, **k): return False


class _StdinRedirect(contextlib._RedirectStream):
    _stream = "stdin"


@contextlib.contextmanager
def _time_limit(seconds):
    seconds = float(seconds)
    if seconds <= 0:
        yield
        return

    def handler(signum, frame):  # pragma: no cover - frame is signal-owned
        raise _Timeout()

    previous_handler = None
    previous_timer = (0.0, 0.0)
    timer_started = False
    try:
        previous_handler = signal.getsignal(signal.SIGALRM)
        previous_timer = signal.getitimer(signal.ITIMER_REAL)
        signal.signal(signal.SIGALRM, handler)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        timer_started = True
    except (AttributeError, OSError, ValueError):
        # The parent process still enforces a deadline on platforms/threads
        # where SIGALRM cannot be installed.
        if previous_handler is not None:
            with contextlib.suppress(OSError, ValueError):
                signal.signal(signal.SIGALRM, previous_handler)
        yield
        return
    try:
        yield
    finally:
        if timer_started:
            with contextlib.suppress(OSError, ValueError):
                signal.setitimer(signal.ITIMER_REAL, *previous_timer)
            with contextlib.suppress(OSError, ValueError):
                signal.signal(signal.SIGALRM, previous_handler)


@contextlib.contextmanager
def _swallow_io():
    s = _WOStringIO()
    # Redirect the dunder streams as well as sys.stdout/stderr. Generated code
    # occasionally writes through sys.__stdout__; without this, that text can
    # corrupt the worker's JSON response pipe.
    original_stdout, original_stderr = sys.stdout, sys.stderr
    original_dunder_stdout, original_dunder_stderr = sys.__stdout__, sys.__stderr__
    sys.stdout = sys.__stdout__ = s
    sys.stderr = sys.__stderr__ = s
    try:
        with _StdinRedirect(s):
            yield
    finally:
        sys.stdout, sys.stderr = original_stdout, original_stderr
        sys.__stdout__, sys.__stderr__ = (
            original_dunder_stdout,
            original_dunder_stderr,
        )


@contextlib.contextmanager
def _reliability_guard():
    """Temporarily disable destructive operations in candidate code.

    Every monkey patch is restored before the surrounding temporary directory
    is cleaned up. In particular, leaving ``os.unlink`` disabled causes noisy
    atexit failures in spawned workers.
    """
    import builtins
    import shutil
    import subprocess as sp
    import sys

    os_names = (
        "system", "kill", "setuid", "fork", "putenv", "remove", "removedirs",
        "rmdir", "fchdir", "forkpty", "killpg", "rename", "renames", "truncate",
        "replace", "unlink", "fchmod", "fchown", "chmod", "chown", "chroot",
        "lchflags", "lchmod", "lchown", "getcwd", "chdir",
    )
    builtin_names = ("exit", "quit", "help")
    shutil_names = ("rmtree", "move", "chown")
    blocked_modules = ("ipdb", "joblib", "resource", "psutil", "tkinter")
    missing = object()
    original_builtins = {
        name: getattr(builtins, name, missing) for name in builtin_names
    }
    original_os = {name: getattr(os, name, missing) for name in os_names}
    original_shutil = {
        name: getattr(shutil, name, missing) for name in shutil_names
    }
    original_popen = sp.Popen
    original_modules = {
        name: sys.modules.get(name, missing) for name in blocked_modules
    }
    original_omp = os.environ.get("OMP_NUM_THREADS", missing)
    faulthandler_was_enabled = faulthandler.is_enabled()

    def _dis(*args, **kwargs):
        raise RuntimeError("Restricted")

    try:
        faulthandler.disable()
        builtins.exit = builtins.quit = builtins.help = None
        os.environ["OMP_NUM_THREADS"] = "1"
        for attr in ("system", "kill", "setuid", "fork"):
            if original_os[attr] is not missing:
                setattr(os, attr, _dis)
        for attr in os_names[4:]:
            if original_os[attr] is not missing:
                setattr(os, attr, None)
        for attr in shutil_names:
            if original_shutil[attr] is not missing:
                setattr(shutil, attr, None)
        sp.Popen = None
        for name in blocked_modules:
            sys.modules[name] = None
        yield
    finally:
        for name, value in original_builtins.items():
            if value is missing:
                if hasattr(builtins, name):
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
        sp.Popen = original_popen
        for name, value in original_modules.items():
            if value is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
        if original_omp is missing:
            os.environ.pop("OMP_NUM_THREADS", None)
        else:
            os.environ["OMP_NUM_THREADS"] = original_omp
        if faulthandler_was_enabled:
            faulthandler.enable()


def _run_test_code(test_code, timeout):
    """Execute one candidate in an already isolated worker process."""
    outcome = "timed out"
    execution_context = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            with _reliability_guard(), _swallow_io(), _time_limit(timeout):
                exec(test_code, execution_context)
            outcome = "passed"
        except _Timeout:
            outcome = "timed out"
        except AssertionError:
            outcome = "failed: AssertionError"
        except BaseException as exc:
            outcome = f"failed: {exc}"
        finally:
            # Drop generated objects only after the guard has restored cleanup
            # functions such as os.unlink and shutil.rmtree.
            execution_context.clear()
            os.chdir(cwd)
    return outcome


def check_correctness(task_id, sample, timeout=3.0, completion_id=None):
    result = "timed out"
    try:
        timeout = float(timeout)
        test_code = sample["test_code"]
    except (KeyError, TypeError, ValueError):
        timeout = None

    if timeout is not None and timeout > 0 and math.isfinite(timeout):
        try:
            with tempfile.TemporaryDirectory(prefix="code-exec-") as tmpdir:
                result_path = Path(tmpdir) / "result.json"
                payload = json.dumps({
                    "test_code": test_code,
                    "timeout": timeout,
                    "result_path": str(result_path),
                })
                process = _start_worker(
                    [sys.executable, "-c", _CODE_EXEC_WORKER_CODE],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    close_fds=True,
                    start_new_session=True,
                    text=True,
                    env=_worker_environment(),
                )
                if process is not None:
                    try:
                        process.communicate(
                            payload, timeout=timeout + _PROCESS_STARTUP_GRACE
                        )
                    except (subprocess.TimeoutExpired, OSError, ValueError):
                        pass
                    finally:
                        _terminate_process(process)
                    if process.returncode == 0 and result_path.is_file():
                        try:
                            if result_path.stat().st_size <= 1024:
                                candidate = json.loads(
                                    result_path.read_text(encoding="utf-8")
                                )
                                if isinstance(candidate, str):
                                    result = candidate
                        except (OSError, TypeError, ValueError):
                            pass
        except (OSError, TypeError, ValueError):
            pass

    return {
        "task_id": task_id,
        "completion_id": completion_id,
        "result": result,
        "passed": result == "passed",
    }


# ---- Evaluation orchestrator ----

def evaluate_functional_correctness(input_file, problem_file, is_mbpp=False,
                                    n_workers=8, timeout=3.0, k=(1,)):
    problems = read_problems(problem_file)
    samples = list(stream_jsonl(input_file))
    completion_ids = Counter()
    results = defaultdict(list)

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for s in samples:
            tid = s["task_id"]
            cid = s.get("completion_id", completion_ids[tid])
            futures.append(executor.submit(
                check_correctness, task_id=tid,
                sample={"test_code": build_test_code(s, problems, is_mbpp=is_mbpp)},
                timeout=timeout, completion_id=cid,
            ))
            completion_ids[tid] += 1
        for f in as_completed(futures):
            r = f.result()
            results[r["task_id"]].append((r["completion_id"], r))

    totals, correct = [], []
    for tid in problems:
        task_res = results.get(tid, [])
        passed = [row[1]["passed"] for row in task_res]
        totals.append(len(passed))
        correct.append(sum(passed))
    totals, correct = np.array(totals), np.array(correct)

    return {
        f"pass@{v}": float(estimate_pass_at_k(totals, correct, v).mean())
        for v in k if len(totals) and (totals >= v).all()
    }
