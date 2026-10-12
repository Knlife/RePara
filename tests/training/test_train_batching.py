import unittest

import torch

from grpo import compute_group_advantages
from train import (
    _organize_prompt_rollouts,
    _split_prompt_batch,
    validate_global_batch_size,
)
from utils.configuration.config import TrainConfig


class TrainBatchingTest(unittest.TestCase):
    def test_global_prompt_batch_is_fixed_at_64(self):
        configurations = (
            (8, 8, 1),
            (1, 16, 4),
            (4, 2, 8),
        )
        for world_size, micro_batch, accumulation in configurations:
            with self.subTest(
                world_size=world_size,
                micro_batch=micro_batch,
                accumulation=accumulation,
            ):
                config = TrainConfig(
                    batch_size_per_device=micro_batch,
                    grad_accumulation=accumulation,
                )
                self.assertEqual(
                    validate_global_batch_size(config, world_size),
                    64,
                )

        with self.assertRaisesRegex(ValueError, "must be 64"):
            validate_global_batch_size(
                TrainConfig(batch_size_per_device=2, grad_accumulation=2),
                world_size=8,
            )

    def test_prompt_batch_is_split_without_losing_fields(self):
        batch = {
            "problems": ["problem-a", "problem-b"],
            "answers": ["answer-a", "answer-b"],
            "sample_ids": [11, 29],
        }

        result = _split_prompt_batch(batch)

        self.assertEqual(
            result,
            [
                {
                    "problems": ["problem-a"],
                    "answers": ["answer-a"],
                    "sample_ids": [11],
                },
                {
                    "problems": ["problem-b"],
                    "answers": ["answer-b"],
                    "sample_ids": [29],
                },
            ],
        )

    def test_advantages_are_normalized_per_prompt_across_repeats(self):
        prompt_rollouts = [
            [
                {"name": "repeat0-prompt0", "rewards": torch.tensor([1.0, 3.0])},
                {"name": "repeat0-prompt1", "rewards": torch.tensor([10.0, 20.0])},
            ],
            [
                {"name": "repeat1-prompt0", "rewards": torch.tensor([5.0, 7.0])},
                {"name": "repeat1-prompt1", "rewards": torch.tensor([30.0, 40.0])},
            ],
        ]
        reward_grid = torch.tensor(
            [
                [[1.0, 10.0], [3.0, 20.0]],
                [[5.0, 30.0], [7.0, 40.0]],
            ]
        )
        expected = compute_group_advantages(reward_grid.reshape(-1), group_size=4)
        expected_grid = expected.view_as(reward_grid)

        chunks, advantages = _organize_prompt_rollouts(prompt_rollouts)

        torch.testing.assert_close(advantages, expected)
        self.assertEqual(
            [chunk["name"] for chunk in chunks],
            [
                "repeat0-prompt0",
                "repeat0-prompt1",
                "repeat1-prompt0",
                "repeat1-prompt1",
            ],
        )
        for repeat_index in range(2):
            for prompt_index in range(2):
                chunk = prompt_rollouts[repeat_index][prompt_index]
                torch.testing.assert_close(
                    chunk["advantages"],
                    expected_grid[repeat_index, :, prompt_index],
                )


if __name__ == "__main__":
    unittest.main()
