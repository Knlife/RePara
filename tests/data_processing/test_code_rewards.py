import unittest
from unittest import mock

import torch

from data.code import reward_code


class CodeRewardTest(unittest.TestCase):
    def test_reward_exposes_format_and_pass_rate_components(self):
        batch = {"test_cases": [["assert True"]]}
        responses = [
            "```python\nvalue = 1\n```",
            "not a code block",
        ]

        with mock.patch("data.code.local_execute", return_value=[0.75]) as execute:
            total, components = reward_code(
                batch=batch,
                responses=responses,
                num_generations=2,
                device=torch.device("cpu"),
                return_components=True,
            )

        execute.assert_called_once_with([
            ("value = 1\n", ["assert True"]),
        ])
        self.assertTrue(torch.equal(components["format_reward"], torch.tensor([1.0, 0.0])))
        self.assertTrue(torch.equal(components["pass_rate"], torch.tensor([0.75, 0.0])))
        self.assertTrue(torch.equal(total, torch.tensor([1.75, 0.0])))


if __name__ == "__main__":
    unittest.main()
