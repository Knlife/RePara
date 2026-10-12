import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from utils.configuration.config import TrainConfig
from utils.models.forward import _sdar_attention_mask, forward_logits


class _SdarStub(torch.nn.Module):
    def __init__(self, *, fuse_cross_entropy=False):
        super().__init__()
        self.config = SimpleNamespace(
            block_size=2, fuse_cross_entropy=fuse_cross_entropy
        )
        self.last_kwargs = None
        self.forward_training_states = []
        self.decoder = torch.nn.Dropout(p=0.0)
        self.decoder_training_states = []

    def forward(self, input_ids, **kwargs):
        self.last_kwargs = kwargs
        self.forward_training_states.append(self.training)
        self.decoder_training_states.append(self.decoder.training)
        logits = torch.zeros(*input_ids.shape, 5, dtype=torch.float32)
        return SimpleNamespace(logits=logits)


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

    def test_sdar_forward_keeps_training_mode_and_builds_masks(self):
        model = _SdarStub()
        model.train()
        input_ids = torch.tensor([
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 0],
        ])

        logits = forward_logits(model, input_ids, model_family="sdar")

        self.assertTrue(model.training)
        self.assertEqual(model.forward_training_states, [False])
        self.assertEqual(tuple(logits.shape), (2, 5, 5))
        self.assertEqual(tuple(model.last_kwargs["attention_mask"].shape), (2, 1, 5, 5))
        expected_mask = torch.tensor([
            [True, True, False, False, False],
            [True, True, False, False, False],
            [True, True, True, True, False],
            [True, True, True, True, False],
            [True, True, True, True, True],
        ])
        torch.testing.assert_close(model.last_kwargs["attention_mask"][0, 0], expected_mask)
        torch.testing.assert_close(model.last_kwargs["attention_mask"][1, 0], expected_mask)
        self.assertEqual(tuple(model.last_kwargs["position_ids"].shape), (2, 5))
        torch.testing.assert_close(
            model.last_kwargs["position_ids"],
            torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]),
        )
        self.assertFalse(model.last_kwargs["use_cache"])
        self.assertEqual(model.decoder_training_states, [True])

    def test_sdar_fused_cross_entropy_uses_logits_path_in_training(self):
        model = _SdarStub(fuse_cross_entropy=True)
        model.train()
        logits = forward_logits(model, torch.tensor([[1, 2]]), model_family="sdar")

        self.assertTrue(model.training)
        self.assertEqual(model.forward_training_states, [False])
        self.assertEqual(tuple(logits.shape), (1, 2, 5))

    def test_sdar_attention_mask_broadcasts_over_attention_heads(self):
        input_ids = torch.zeros((16, 5), dtype=torch.long)
        mask = _sdar_attention_mask(input_ids, block_size=2)
        query = torch.randn(16, 32, 5, 4)
        key = torch.randn(16, 32, 5, 4)
        value = torch.randn(16, 32, 5, 4)

        output = F.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=mask,
            is_causal=False,
        )

        self.assertEqual(tuple(output.shape), (16, 32, 5, 4))



if __name__ == "__main__":
    unittest.main()
