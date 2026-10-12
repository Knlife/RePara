import unittest

from utils.evaluation.grader import math_equal


class GraderProcessTest(unittest.TestCase):
    def test_symbolic_timeout_worker_isolated_and_returns(self):
        self.assertTrue(math_equal("x + 1", "1 + x", timeout=True))
        self.assertFalse(math_equal("x + 1", "x + 2", timeout=True))


if __name__ == "__main__":
    unittest.main()
