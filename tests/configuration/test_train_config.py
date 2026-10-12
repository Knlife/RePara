import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from utils.configuration.config import TrainConfig
from utils.models.forward import  forward_logits





class TrainConfigTest(unittest.TestCase):
    def test_default_rollout_group_uses_one_16_sample_batch(self):
        config = TrainConfig()
        self.assertEqual(config.num_generations, 16)
        self.assertEqual(config.repeat_times, 1)

    def test_default_training_length_is_128_steps(self):
        self.assertEqual(TrainConfig().total_steps, 128)

    def test_timing_is_opt_in_with_one_warmup_step(self):
        config = TrainConfig()
        self.assertFalse(config.profile_timing)
        self.assertEqual(config.timing_warmup_steps, 1)

    def test_train_schedule_switches_on_zero_based_optimizer_step(self):
        config = TrainConfig(
            model_path="/models/llada",
            output_dir="/runs/train",
            dataset="code",
            code_data_path="/datasets/code.jsonl",
            total_steps=128,
            gen_steps=256,
            gen_length=256,
            train_schedule=[
                {"start_step": 0, "method": "justgrpo", "block_size": 1},
                {"start_step": 64, "method": "fixed_mtp", "block_size": 2},
            ],
        )
        config.validate()

        self.assertEqual(config.stage_for_step(0).method, "justgrpo")
        self.assertEqual(config.stage_for_step(63).block_size, 1)
        self.assertEqual(config.stage_for_step(64).method, "fixed_mtp")
        self.assertEqual(config.stage_for_step(127).block_size, 2)
        self.assertEqual(
            config.normalized_train_schedule(),
            [
                {"start_step": 0, "method": "justgrpo", "block_size": 1},
                {"start_step": 64, "method": "fixed_mtp", "block_size": 2},
            ],
        )

    def test_train_schedule_validates_boundaries_and_method_constraints(self):
        common = dict(
            model_path="/models/llada",
            output_dir="/runs/train",
            dataset="code",
            code_data_path="/datasets/code.jsonl",
            total_steps=128,
            gen_steps=256,
            gen_length=256,
        )
        invalid_schedules = [
            ([{"start_step": 1, "method": "justgrpo", "block_size": 1}], "start at step 0"),
            ([
                {"start_step": 0, "method": "justgrpo", "block_size": 1},
                {"start_step": 128, "method": "fixed_mtp", "block_size": 2},
            ], "less than total_steps"),
            ([{"start_step": 0, "method": "justgrpo", "block_size": 2}], "block_size=1"),
        ]
        for schedule, message in invalid_schedules:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    TrainConfig(**common, train_schedule=schedule).validate()

        with self.assertRaisesRegex(ValueError, "MTP methods require"):
            invalid_common = {**common, "gen_steps": 128}
            TrainConfig(
                **invalid_common,
                train_schedule=[
                    {"start_step": 0, "method": "justgrpo", "block_size": 1},
                    {"start_step": 64, "method": "fixed_mtp", "block_size": 2},
                ],
            ).validate()









if __name__ == "__main__":
    unittest.main()
