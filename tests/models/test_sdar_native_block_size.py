import unittest
from types import SimpleNamespace

import torch

from utils.configuration.config import TrainConfig, EvalConfig
from utils.models.forward import _sdar_attention_mask
from utils.generation.generate import _remaining_slots_in_sdar_block, _sdar_model_block_size
from utils.models.sdar_config import configure_sdar_native_block_size


class SDARNativeBlockSizeTest(unittest.TestCase):
    def test_b32_is_shared_by_rollout_mask_and_boundaries(self):
        config = SimpleNamespace()
        self.assertEqual(configure_sdar_native_block_size(config, 32), 32)
        model = SimpleNamespace(config=config)
        self.assertEqual(_sdar_model_block_size(model), 32)
        mask = _sdar_attention_mask(torch.zeros(1, 64, dtype=torch.long), 32)
        self.assertTrue(mask[0, 0, 0, 31].item())
        self.assertFalse(mask[0, 0, 0, 32].item())
        self.assertTrue(mask[0, 0, 32, 63].item())
        remaining = _remaining_slots_in_sdar_block(31, torch.tensor([0, 1, 2]), 32)
        self.assertEqual(remaining.tolist(), [1, 32, 31])

    def test_legacy_b4_is_preserved(self):
        config = SimpleNamespace()
        self.assertEqual(configure_sdar_native_block_size(config), 4)
        self.assertEqual(config.block_size, 4)

    def test_declared_b32_is_preserved_without_override(self):
        self.assertEqual(configure_sdar_native_block_size(SimpleNamespace(block_size=32)), 32)

    def test_conflicting_override_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "disagrees"):
            configure_sdar_native_block_size(SimpleNamespace(block_size=4), 32)

    def test_invalid_values_are_rejected(self):
        for value in (0, -1, True, 32.0, "32"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                configure_sdar_native_block_size(SimpleNamespace(), value)

    def config(self, **kwargs):
        return TrainConfig(
            model_path="/models/sdar-b32", model_family="sdar",
            output_dir="/runs/b32", gsm8k_data_path="/data/gsm8k",
            method="fixed_mtp", mtp_block_size=2, **kwargs,
        )

    def test_training_config_keeps_fix2_and_128_steps(self):
        config = self.config(sdar_native_block_size=32)
        config.validate()
        self.assertEqual(config.mtp_block_size, 2)
        self.assertEqual(config.total_steps, 128)

    def test_invalid_training_native_sizes_are_rejected(self):
        for value in (0, -1, True, 32.0, "32", 1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.config(sdar_native_block_size=value).validate()

    def test_evaluation_native_size_can_be_supplied_for_missing_metadata(self):
        config = EvalConfig(
            task="gsm8k", ckpt_path="/models/sdar-b32", model_family="sdar",
            gsm8k_data_files="/data/test.parquet", output="/results",
            generation_method="fixed_mtp", mtp_block_size=2,
            sdar_native_block_size=32,
        )
        config.validate()
        self.assertEqual(config.sdar_native_block_size, 32)
        config.sdar_native_block_size = 1
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            config.validate()

    def test_b32_online_math_rates_use_native_attention_size(self):
        from utils.evaluation.online_eval import build_online_eval_runs
        config = self.config(
            sdar_native_block_size=32, save_every=16, eval_every=16,
            eval_config={"gsm8k_data_files": "/data/test.parquet"},
        )
        config.validate()
        runs = build_online_eval_runs(config, "/ckpt", 16)
        self.assertEqual([r.sdar_denoising_steps for r in runs], [32, 16, 8])
        self.assertTrue(all(r.sdar_native_block_size == 32 for r in runs))


if __name__ == "__main__":
    unittest.main()
