import builtins
import os
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

from data import local_sandbox


class LocalSandboxIsolationTest(unittest.TestCase):
    def test_worker_guard_is_scoped_and_blocks_file_deletion(self):
        original_cwd = os.getcwd()
        originals = {
            "input": builtins.input,
            "print": builtins.print,
            "unlink": os.unlink,
            "rmtree": shutil.rmtree,
            "popen": subprocess.Popen,
        }

        with tempfile.TemporaryDirectory() as directory:
            sentinel = os.path.join(directory, "sentinel")
            with open(sentinel, "w", encoding="utf-8") as handle:
                handle.write("keep")

            result = local_sandbox.run_single_test(
                f"__import__('os').unlink({sentinel!r})",
                "assert True",
                working_directory=directory,
            )

            self.assertEqual(result, 0)
            self.assertTrue(os.path.isfile(sentinel))

        self.assertEqual(os.getcwd(), original_cwd)
        self.assertIs(builtins.input, originals["input"])
        self.assertIs(builtins.print, originals["print"])
        self.assertIs(os.unlink, originals["unlink"])
        self.assertIs(shutil.rmtree, originals["rmtree"])
        self.assertIs(subprocess.Popen, originals["popen"])

    def test_worker_guard_does_not_leak_into_pool_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(local_sandbox, "cache_wd", directory):
                results = local_sandbox.get_successful_tests(
                    "holder = __import__('tempfile').TemporaryDirectory()",
                    ["assert holder.name"],
                    num_workers=1,
                )

        self.assertEqual(results, [1])

    def test_execution_does_not_mutate_parent_process(self):
        original_cwd = os.getcwd()
        originals = {
            "exit": builtins.exit,
            "print": builtins.print,
            "putenv": os.putenv,
            "system": os.system,
            "remove": os.remove,
            "replace": os.replace,
            "rmtree": shutil.rmtree,
            "popen": subprocess.Popen,
        }

        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(local_sandbox, "cache_wd", directory):
                results = local_sandbox.get_successful_tests(
                    "value = 6 * 7",
                    ["assert value == 42", "assert value != 42"],
                    num_workers=1,
                )

        self.assertEqual(results, [1, 0])
        self.assertEqual(os.getcwd(), original_cwd)
        self.assertIs(builtins.exit, originals["exit"])
        self.assertIs(builtins.print, originals["print"])
        self.assertIs(os.putenv, originals["putenv"])
        self.assertIs(os.system, originals["system"])
        self.assertIs(os.remove, originals["remove"])
        self.assertIs(os.replace, originals["replace"])
        self.assertIs(shutil.rmtree, originals["rmtree"])
        self.assertIs(subprocess.Popen, originals["popen"])
        os.environ["JUSTGRPO_PARENT_ENV_PROBE"] = "ok"
        self.assertEqual(os.environ.pop("JUSTGRPO_PARENT_ENV_PROBE"), "ok")

    def test_worker_startup_is_not_charged_to_code_timeout(self):
        # Import the same heavy modules that are live when the reward runs.
        # The old spawn Pool re-imported this process's main module and could
        # consume the one-second budget before executing the candidate.
        import filelock  # noqa: F401
        import train  # noqa: F401
        import eval as eval_module  # noqa: F401

        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(local_sandbox, "cache_wd", directory):
                results = local_sandbox.get_successful_tests(
                    "value = 6 * 7",
                    ["assert value == 42", "assert value != 42"],
                    max_execution_time=1.0,
                    num_workers=2,
                )

        self.assertEqual(results, [1, 0])


if __name__ == "__main__":
    unittest.main()
