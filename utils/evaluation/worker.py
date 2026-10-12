"""Common lifecycle helpers for evaluator subprocesses."""

import contextlib
import os
import signal
import subprocess
import time


_ENV_KEYS = (
    "PATH",
    "PYTHONPATH",
    "PYTHONHOME",
    "LANG",
    "LC_ALL",
    "TMPDIR",
    "TEMP",
    "TMP",
    "PYTHONHASHSEED",
)


def worker_environment(repo_root: str, *, include_home: bool = False) -> dict[str, str]:
    """Return only the environment needed to import and run a worker."""
    keys = _ENV_KEYS + (("HOME",) if include_home else ())
    env = {key: os.environ[key] for key in keys if key in os.environ}
    current_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        repo_root
        if not current_pythonpath
        else os.pathsep.join((repo_root, current_pythonpath))
    )
    env["OMP_NUM_THREADS"] = "1"
    return env


def start_worker(command, *, retries: int = 20, retry_delay: float = 0.01, **kwargs):
    """Start a worker, retrying only filelock's transient fork rejection."""
    for attempt in range(retries):
        try:
            return subprocess.Popen(command, **kwargs)
        except RuntimeError as exc:
            if "unsafe while filelock" not in str(exc).lower():
                return None
            if attempt + 1 < retries:
                time.sleep(retry_delay)
        except (OSError, TypeError, ValueError):
            return None
    return None


def terminate_worker(process) -> None:
    """Terminate a worker process group and reap its pipes."""
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (AttributeError, OSError):
        with contextlib.suppress(OSError, ValueError):
            process.kill()
    try:
        process.communicate(timeout=1.0)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(OSError, ValueError):
            process.kill()
        with contextlib.suppress(OSError, ValueError, subprocess.TimeoutExpired):
            process.communicate(timeout=1.0)
    except (OSError, ValueError):
        pass
