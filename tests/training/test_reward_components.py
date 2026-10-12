import unittest
from unittest.mock import patch

import torch

from data.code import reward_code
from grpo import _normalize_reward_output


class RewardComponentsTest(unittest.TestCase):
    def test_code_reward_exposes_legacy_total_and_components(self):
        fence = chr(96) * 3
        responses = [
            f"{fence}python\ndef f(x):\n    return x\n{fence}\n",
            "not a fenced completion",
        ]
        batch = {"test_cases": [["assert f(1) == 1"]]}

        with patch("data.code.local_execute", return_value=[1.0]) as execute:
            rewards, components = reward_code(
                batch,
                responses,
                num_generations=2,
                device="cpu",
                return_components=True,
            )

        torch.testing.assert_close(rewards, torch.tensor([2.0, 0.0]))
        torch.testing.assert_close(
            components["format_reward"], torch.tensor([1.0, 0.0])
        )
        torch.testing.assert_close(
            components["pass_rate"], torch.tensor([1.0, 0.0])
        )
        execute.assert_called_once()

    def test_normalizer_keeps_legacy_reward_functions_compatible(self):
        rewards, components = _normalize_reward_output(
            torch.tensor([1.0, -1.0]), expected_size=2, device="cpu"
        )

        torch.testing.assert_close(rewards, torch.tensor([1.0, -1.0]))

if __name__ == "__main__":
    unittest.main()
