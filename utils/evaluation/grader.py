"""
This logic is largely copied from the Hendrycks' MATH release (math_equivalence), and borrowed from:
- https://github.com/microsoft/ProphetNet/tree/master/CRITIC
- https://github.com/openai/prm800k
- https://github.com/microsoft/ToRA/blob/main/src/eval/grader.py
- https://github.com/deepseek-ai/DeepSeek-Math/blob/main/evaluation/eval/eval_utils.py
"""

import atexit
import json
import math
import os
import re
import select
import subprocess
import sys
import threading
from math import isclose
from pathlib import Path

import regex
from latex2sympy2 import latex2sympy
from sympy import N, simplify
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr
from utils.evaluation.worker import (
    start_worker as _start_worker,
    terminate_worker as _stop_symbolic_worker,
    worker_environment,
)


_REPO_ROOT = str(Path(__file__).resolve().parents[2])
_WORKER_STARTUP_TIMEOUT = 10.0
_SYMBOLIC_WORKER_READY = "__repara_symbolic_ready__"
_SYMBOLIC_WORKER_CODE = r"""
import json
import sys
from utils.evaluation.grader import symbolic_equal

print("__repara_symbolic_ready__", flush=True)
for line in sys.stdin:
    try:
        prediction, reference = json.loads(line)
        result = bool(symbolic_equal(prediction, reference))
    except BaseException:
        result = False
    print(json.dumps(result), flush=True)
"""
_SYMBOLIC_WORKER = None
_SYMBOLIC_WORKER_LOCK = threading.Lock()


def _worker_environment():
    return worker_environment(_REPO_ROOT)


def _start_symbolic_worker():
    if os.name != "posix":
        return None
    process = _start_worker(
        [sys.executable, "-c", _SYMBOLIC_WORKER_CODE],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        start_new_session=True,
        text=True,
        bufsize=1,
        env=_worker_environment(),
    )
    if process is None:
        return None
    try:
        ready, _, _ = select.select([process.stdout], [], [], _WORKER_STARTUP_TIMEOUT)
        if ready and process.stdout.readline().strip() == _SYMBOLIC_WORKER_READY:
            return process
    except (OSError, ValueError):
        pass
    _stop_symbolic_worker(process)
    return None


def _symbolic_equal_with_timeout(prediction, reference, seconds):
    global _SYMBOLIC_WORKER
    with _SYMBOLIC_WORKER_LOCK:
        process = _SYMBOLIC_WORKER
        if process is None or process.poll() is not None:
            _stop_symbolic_worker(process)
            process = _start_symbolic_worker()
            _SYMBOLIC_WORKER = process
        if process is None:
            return False
        try:
            process.stdin.write(
                json.dumps((str(prediction), str(reference))) + "\n"
            )
            process.stdin.flush()
            ready, _, _ = select.select(
                [process.stdout], [], [], max(0.0, float(seconds))
            )
            if not ready:
                raise TimeoutError("symbolic grading timed out")
            line = process.stdout.readline()
            if not line:
                raise EOFError("symbolic evaluator exited")
            return bool(json.loads(line))
        except (EOFError, OSError, TypeError, ValueError, TimeoutError):
            _stop_symbolic_worker(process)
            _SYMBOLIC_WORKER = None
            return False


def _shutdown_symbolic_worker():
    global _SYMBOLIC_WORKER
    with _SYMBOLIC_WORKER_LOCK:
        _stop_symbolic_worker(_SYMBOLIC_WORKER)
        _SYMBOLIC_WORKER = None


atexit.register(_shutdown_symbolic_worker)


def choice_answer_clean(pred: str):
    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ").lstrip(":")
    # Clean the answer based on the dataset
    tmp = re.findall(r"\b(A|B|C|D|E)\b", pred.upper())
    if tmp:
        pred = tmp
    else:
        pred = [pred.strip().strip(".")]
    pred = pred[-1]
    # Remove the period at the end, again!
    pred = pred.rstrip(".").rstrip("/")
    return pred


def parse_digits(num):
    num = regex.sub(",", "", str(num))
    try:
        return float(num)
    except Exception:
        if num.endswith("%"):
            num = num[:-1]
            num = num.removesuffix("\\")
            try:
                return float(num) / 100
            except Exception:
                pass
    return None


def is_digit(num):
    # paired with parse_digits
    return parse_digits(num) is not None


def str_to_pmatrix(input_str):
    input_str = input_str.strip()
    matrix_str = re.findall(r"\{.*,.*\}", input_str)
    pmatrix_list = []

    for m in matrix_str:
        m = m.strip("{}")
        pmatrix = r"\begin{pmatrix}" + m.replace(",", "\\") + r"\end{pmatrix}"
        pmatrix_list.append(pmatrix)

    return ", ".join(pmatrix_list)


def _symbolic_compare(a, b, timeout: bool) -> bool:
    return (
        call_with_timeout(symbolic_equal, a, b)
        if timeout
        else symbolic_equal(a, b)
    )


def math_equal(
    prediction: bool | float | str,
    reference: float | str,
    include_percentage: bool = True,
    is_close: bool = True,
    timeout: bool = False,
) -> bool:
    """
    Exact match of math if and only if:
    1. numerical equal: both can convert to float and are equal
    2. symbolic equal: both can convert to sympy expression and are equal
    """
    # print("Judge:", prediction, reference)
    if prediction is None or reference is None:
        return False
    if str(prediction).strip().lower() == str(reference).strip().lower():
        return True
    if (
        reference in ["A", "B", "C", "D", "E"]
        and choice_answer_clean(str(prediction)) == reference
    ):
        return True

    try:  # 1. numerical equal
        if is_digit(prediction) and is_digit(reference):
            prediction = parse_digits(prediction)
            reference = parse_digits(reference)
            # number questions
            if include_percentage:
                gt_result = [reference / 100, reference, reference * 100]
            else:
                gt_result = [reference]
            for item in gt_result:
                try:
                    if is_close:
                        if numeric_equal(prediction, item):
                            return True
                    else:
                        if item == prediction:
                            return True
                except Exception:
                    continue
            return False
    except Exception:
        pass

    if not prediction and prediction not in [0, False]:
        return False

    # 2. symbolic equal
    reference = str(reference).strip()
    prediction = str(prediction).strip()

    ## pmatrix (amps)
    if "pmatrix" in prediction and not "pmatrix" in reference:
        reference = str_to_pmatrix(reference)

    ## deal with [], (), {}
    pred_str, ref_str = prediction, reference
    if (
        prediction.startswith("[")
        and prediction.endswith("]")
        and not reference.startswith("(")
    ) or (
        prediction.startswith("(")
        and prediction.endswith(")")
        and not reference.startswith("[")
    ):
        pred_str = pred_str.strip("[]()")
        ref_str = ref_str.strip("[]()")
    for s in ["{", "}", "(", ")"]:
        ref_str = ref_str.replace(s, "")
        pred_str = pred_str.replace(s, "")
    if pred_str.lower() == ref_str.lower():
        return True

    ## [a, b] vs. [c, d], return a==c and b==d
    if (
        regex.match(r"(\(|\[).+(\)|\])", prediction) is not None
        and regex.match(r"(\(|\[).+(\)|\])", reference) is not None
    ):
        pred_parts = prediction[1:-1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts) and all(
            math_equal(
                pred_parts[i],
                ref_parts[i],
                include_percentage,
                is_close,
                timeout,
            )
                for i in range(len(pred_parts))
        ):
            return True
    if (
        (
            prediction.startswith(("\\begin{pmatrix}", "\\begin{bmatrix}"))
        )
        and (
            prediction.endswith(("\\end{pmatrix}", "\\end{bmatrix}"))
        )
        and (
            reference.startswith(("\\begin{pmatrix}", "\\begin{bmatrix}"))
        )
        and (
            reference.endswith(("\\end{pmatrix}", "\\end{bmatrix}"))
        )
    ):
        pred_lines = [
            line.strip()
            for line in prediction[
                len("\\begin{pmatrix}") : -len("\\end{pmatrix}")
            ].split("\\\\")
            if line.strip()
        ]
        ref_lines = [
            line.strip()
            for line in reference[
                len("\\begin{pmatrix}") : -len("\\end{pmatrix}")
            ].split("\\\\")
            if line.strip()
        ]
        matched = True
        if len(pred_lines) == len(ref_lines):
            for pred_line, ref_line in zip(pred_lines, ref_lines):
                pred_parts = pred_line.split("&")
                ref_parts = ref_line.split("&")
                if len(pred_parts) == len(ref_parts):
                    if not all(
                        math_equal(
                            pred_parts[i],
                            ref_parts[i],
                            include_percentage,
                            is_close,
                            timeout,
                        )
                            for i in range(len(pred_parts))
                    ):
                        matched = False
                        break
                else:
                    matched = False
                if not matched:
                    break
        else:
            matched = False
        if matched:
            return True

    if prediction.count("=") == 1 and reference.count("=") == 1:
        pred = prediction.split("=")
        pred = f"{pred[0].strip()} - ({pred[1].strip()})"
        ref = reference.split("=")
        ref = f"{ref[0].strip()} - ({ref[1].strip()})"
        if _symbolic_compare(pred, ref, timeout) or _symbolic_compare(
            f"-({pred})", ref, timeout
        ):
            return True
    elif (
        prediction.count("=") == 1
        and len(prediction.split("=")[0].strip()) <= 2
        and "=" not in reference
    ):
        if math_equal(
            prediction.split("=")[1],
            reference,
            include_percentage,
            is_close,
            timeout,
        ):
            return True
    elif (
        reference.count("=") == 1
        and len(reference.split("=")[0].strip()) <= 2
        and "=" not in prediction
    ):
        if math_equal(
            prediction,
            reference.split("=")[1],
            include_percentage,
            is_close,
            timeout,
        ):
            return True

    # symbolic equal with sympy
    if timeout:
        if call_with_timeout(symbolic_equal, prediction, reference):
            return True
    else:
        if symbolic_equal(prediction, reference):
            return True

    return False


def math_equal_process(param):
    return math_equal(param[-2], param[-1])


def numeric_equal(prediction: float, reference: float):
    # Note that relative tolerance has significant impact
    # on the result of the synthesized GSM-Hard dataset
    # if reference.is_integer():
    #     return isclose(reference, round(prediction), abs_tol=1e-4)
    # else:
    # prediction = round(prediction, len(str(reference).split(".")[-1]))
    return isclose(reference, prediction, rel_tol=1e-4)


def symbolic_equal(a, b):
    def _parse(s):
        for f in [parse_latex, parse_expr, latex2sympy]:
            try:
                return f(s.replace("\\\\", "\\"))
            except Exception:
                try:
                    return f(s)
                except Exception:
                    pass
        return s

    a = _parse(a)
    b = _parse(b)

    # direct equal
    try:
        if str(a) == str(b) or a == b:
            return True
    except Exception:
        pass

    # simplify equal
    try:
        if a.equals(b) or simplify(a - b) == 0:
            return True
    except Exception:
        pass

    # equation equal
    try:
        if (abs(a.lhs - a.rhs)).equals(abs(b.lhs - b.rhs)):
            return True
    except Exception:
        pass

    try:
        if numeric_equal(float(N(a)), float(N(b))):
            return True
    except Exception:
        pass

    # matrix
    try:
        # if a and b are matrix
        if a.shape == b.shape:
            _a = a.applyfunc(lambda x: round(x, 3))
            _b = b.applyfunc(lambda x: round(x, 3))
            if _a.equals(_b):
                return True
    except Exception:
        pass

    return False


def symbolic_equal_process(a, b, result_writer):
    """Legacy process target retained for callers outside the evaluator.

    Both multiprocessing ``Pipe`` connections (``send``) and the historical
    ``Queue`` writer (``put``) are accepted so existing callers keep working.
    """
    result = symbolic_equal(a, b)
    if hasattr(result_writer, "send"):
        result_writer.send(result)
    elif hasattr(result_writer, "put"):
        result_writer.put(result)
    else:
        raise TypeError("result_writer must provide send() or put()")
    close = getattr(result_writer, "close", None)
    if callable(close):
        close()


def call_with_timeout(func, *args, timeout=1, **kwargs):
    """Run the symbolic comparator in the reusable isolated worker."""
    try:
        seconds = float(timeout)
    except (TypeError, ValueError):
        return False
    if (
        (func is not symbolic_equal and func is not symbolic_equal_process)
        or len(args) < 2
        or kwargs
        or seconds <= 0
        or not math.isfinite(seconds)
    ):
        return False
    return _symbolic_equal_with_timeout(args[0], args[1], seconds)
