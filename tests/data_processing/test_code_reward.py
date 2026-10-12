import unittest
from unittest.mock import patch

import torch

from data.code import reward_code
from utils.configuration.config import TrainConfig


class CodeRewardTest(unittest.TestCase):
    def setUp(self):
        self.batch = {"test_cases": [["assert f(1) == 1"]]}
        fence = chr(96) * 3
        self.responses = [
            f"{fence}python\ndef f(x):\n    return x\n{fence}\n",
            f"{fence}python\ndef f(x):\n    return x + 1\n{fence}\n",
            "not a fenced completion",
        ]

    def test_total_reward_preserves_weighted_format_reward(self):
        with patch("data.code.local_execute", return_value=[1.0, 0.5]):
            rewards, components = reward_code(
                self.batch,
                self.responses,
                num_generations=3,
                device="cpu",
                format_reward_alpha=0.25,
                return_components=True,
            )

        torch.testing.assert_close(rewards, torch.tensor([1.25, 0.75, 0.0]))
        torch.testing.assert_close(
            components["format_reward"], torch.tensor([1.0, 1.0, 0.0])
        )
        torch.testing.assert_close(
            components["pass_rate"], torch.tensor([1.0, 0.5, 0.0])
        )

    def test_default_alpha_matches_existing_behavior(self):
        self.assertEqual(TrainConfig().format_reward_alpha, 1.0)
        with patch("data.code.local_execute", return_value=[1.0, 0.5]):
            rewards = reward_code(
                self.batch, self.responses, 3, "cpu"
            )
        torch.testing.assert_close(rewards, torch.tensor([2.0, 1.5, 0.0]))

    def test_alpha_must_be_finite_and_non_negative(self):
        with self.assertRaisesRegex(ValueError, "format_reward_alpha"):
            reward_code(
                self.batch, self.responses, 3, "cpu", format_reward_alpha=-0.1
            )


    def test_zero_alpha_removes_format_contribution(self):
        with patch("data.code.local_execute", return_value=[1.0, 0.5]):
            rewards = reward_code(
                self.batch, self.responses, 3, "cpu", format_reward_alpha=0.0
            )
        torch.testing.assert_close(rewards, torch.tensor([1.0, 0.5, 0.0]))

    def test_train_config_validates_alpha(self):
        config = TrainConfig(
            model_path="/models/llada",
            output_dir="/runs/code",
            dataset="code",
            code_data_path="/datasets/code.jsonl",
            format_reward_alpha=-1.0,
        )
        with self.assertRaisesRegex(ValueError, "format_reward_alpha"):
            config.validate()

if __name__ == "__main__":
    unittest.main()
