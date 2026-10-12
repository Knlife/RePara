from concurrent.futures import ThreadPoolExecutor
import unittest
from unittest import mock

import filelock  # Registers the fork-safety audit hook used during evaluation.

from utils.evaluation.code_exec import check_correctness


class CodeExecutionTest(unittest.TestCase):
    def test_check_reports_candidate_outcomes(self):
        failed = check_correctness(
            task_id="failed",
            sample={"test_code": "print(\'candidate noise\'); assert False"},
            timeout=1.0,
        )
        timed_out = check_correctness(
            task_id="timed-out",
            sample={"test_code": "while True: pass"},
            timeout=0.1,
        )
        crashed = check_correctness(
            task_id="crashed",
            sample={"test_code": "__import__('os')._exit(1)"},
            timeout=1.0,
        )

        self.assertEqual(failed["result"], "failed: AssertionError")
        self.assertFalse(failed["passed"])
        self.assertEqual(timed_out["result"], "timed out")
        self.assertFalse(timed_out["passed"])
        self.assertEqual(crashed["result"], "timed out")
        self.assertFalse(crashed["passed"])

        dunder_output = check_correctness(
            task_id="dunder-output",
            sample={"test_code": "import sys; sys.__stdout__.write('noise'); assert True"},
            timeout=1.0,
        )
        self.assertTrue(dunder_output["passed"])

    def test_parallel_checks_are_safe_with_filelock(self):
        def run_check(completion_id):
            return check_correctness(
                task_id=f"task-{completion_id}",
                sample={"test_code": "assert 6 * 7 == 42"},
                timeout=1.0,
                completion_id=completion_id,
            )

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(run_check, range(16)))

        self.assertTrue(all(result["passed"] for result in results))

    def test_transient_filelock_start_failure_is_retried(self):
        import utils.evaluation.code_exec as code_exec

        real_popen = code_exec.subprocess.Popen
        attempts = {"count": 0}

        def flaky_popen(*args, **kwargs):
            if attempts["count"] == 0:
                attempts["count"] += 1
                raise RuntimeError(
                    "os.fork is unsafe while filelock is changing descriptor ownership"
                )
            return real_popen(*args, **kwargs)

        with mock.patch.object(code_exec.subprocess, "Popen", side_effect=flaky_popen):
            result = code_exec.check_correctness(
                task_id="transient-filelock",
                sample={"test_code": "assert 6 * 7 == 42"},
                timeout=1.0,
            )

        self.assertTrue(result["passed"])
        self.assertEqual(attempts["count"], 1)

    def test_worker_start_failure_is_reported_as_a_failed_sample(self):
        with mock.patch(
            "utils.evaluation.code_exec.subprocess.Popen",
            side_effect=OSError("worker unavailable"),
        ):
            result = check_correctness(
                task_id="worker-failure",
                sample={"test_code": "assert 6 * 7 == 42"},
                timeout=1.0,
            )

        self.assertEqual(result["result"], "timed out")
        self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
