import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn
from unittest.mock import Mock

from utils.generation.sdar_native_generate import (
    _sample_tokens,
    _select_transfer_index,
    block_diffusion_generate_with_stats,
    init_sdar_mtp_stats,
)


class ToyNativeSDAR(nn.Module):
    @property
    def device(self):
        return torch.device("cpu")

    def forward(self, input_ids, **kwargs):
        logits = torch.full((*input_ids.shape, 6), -100.0)
        logits[..., 1] = 100.0
        return SimpleNamespace(logits=logits)


class PaddedVocabSDAR(ToyNativeSDAR):
    def forward(self, input_ids, **kwargs):
        logits = torch.full((*input_ids.shape, 8), -100.0)
        logits[..., 1] = 10.0
        logits[..., 7] = 100.0
        return SimpleNamespace(logits=logits)


class SdarNativeGenerateTest(unittest.TestCase):

    def test_native_block_metadata_mismatch_is_rejected(self):
        model = ToyNativeSDAR()
        model.config = SimpleNamespace(block_size=4)

        with self.assertRaisesRegex(ValueError, "declared block_size"):
            block_diffusion_generate_with_stats(
                model=model, prompt={"input_ids": torch.tensor([[2]])},
                mask_id=5, gen_length=3, block_length=2, denoising_steps=2,
            )

    def test_transfer_quota_never_changes_observed_tokens_or_completed_rows(self):
        cur_x = torch.tensor([[2, 5, 3, 4], [2, 3, 4, 1]])
        mask_index = cur_x == 5
        token_confidence = torch.full_like(cur_x, 0.2, dtype=torch.float32)
        for strategy in (
            "sequential", "low_confidence_static", "low_confidence_dynamic",
            "entropy_bounded",
        ):
            with self.subTest(strategy=strategy):
                selected, _ = _select_transfer_index(
                    cur_x=cur_x, mask_index=mask_index,
                    token_confidence=token_confidence, transfer_count=2,
                    remasking_strategy=strategy, confidence_threshold=0.85,
                    eb_threshold=0.35, stats=init_sdar_mtp_stats(4, "cpu"),
                )
                torch.testing.assert_close(selected, mask_index)
                updated = torch.where(selected, torch.ones_like(cur_x), cur_x)
                torch.testing.assert_close(updated[~mask_index], cur_x[~mask_index])

    def test_sequential_transfer_skips_observed_positions_between_masks(self):
        cur_x = torch.tensor([[5, 3, 5, 2]])
        selected, _ = _select_transfer_index(
            cur_x=cur_x, mask_index=cur_x == 5,
            token_confidence=torch.ones_like(cur_x, dtype=torch.float32),
            transfer_count=2, remasking_strategy="sequential",
            confidence_threshold=0.85, eb_threshold=0.35,
            stats=init_sdar_mtp_stats(4, "cpu"),
        )
        torch.testing.assert_close(selected, cur_x == 5)

    def test_partial_prompt_block_keeps_its_observed_prefix(self):
        prompt = {"input_ids": torch.tensor([[2, 3, 4]])}
        for strategy in ("sequential", "low_confidence_static", "low_confidence_dynamic"):
            with self.subTest(strategy=strategy):
                generated_ids, _ = block_diffusion_generate_with_stats(
                    model=ToyNativeSDAR(), prompt=prompt, mask_id=5,
                    gen_length=1, block_length=4, denoising_steps=2,
                    temperature=0.0, remasking_strategy=strategy,
                    confidence_threshold=1.0,
                )
                self.assertEqual(generated_ids.tolist(), [[2, 3, 4, 1]])

    def test_sampling_excludes_padded_ids_for_greedy_and_stochastic_decoding(self):
        logits = torch.tensor([[[0.0, 5.0, 1.0, 2.0, 30.0, 40.0]]])
        original = logits.clone()

        def sample_fn(filtered, **kwargs):
            self.assertTrue(torch.isneginf(filtered[..., 4:]).all())
            return torch.softmax(filtered, dim=-1).max(dim=-1).indices, None

        for temperature in (0.0, 1.0):
            with self.subTest(temperature=temperature):
                token, _ = _sample_tokens(
                    logits, temperature=temperature, top_k=0, top_p=1.0,
                    sample_fn=sample_fn, mask_id=3, sampling_vocab_size=4,
                )
                self.assertEqual(token.item(), 1)
                torch.testing.assert_close(logits, original)

    def test_native_sampling_keeps_mask_token_in_its_upstream_support(self):
        logits = torch.tensor([[[0.0, 1.0, 2.0, 5.0, 30.0]]])
        token, _ = _sample_tokens(
            logits, temperature=0.0, top_k=0, top_p=1.0,
            sample_fn=Mock(), mask_id=3, sampling_vocab_size=4,
        )
        self.assertEqual(token.item(), 3)

    def test_generator_never_emits_padded_vocab_ids(self):
        model = PaddedVocabSDAR()
        model.config = SimpleNamespace(block_size=2)
        for temperature in (0.0, 1.0):
            with self.subTest(temperature=temperature):
                generated_ids, _ = block_diffusion_generate_with_stats(
                    model=model, prompt={"input_ids": torch.tensor([[2]])},
                    mask_id=5, gen_length=3, block_length=2, denoising_steps=2,
                    temperature=temperature, top_k=1, sampling_vocab_size=6,
                )
                self.assertEqual(generated_ids.tolist(), [[2, 1, 1, 1]])

    def test_block_diffusion_generate_returns_mtp_like_stats_without_submodule_patch(self):
        model = ToyNativeSDAR()
        prompt = {"input_ids": torch.tensor([[2]])}

        generated_ids, stats = block_diffusion_generate_with_stats(
            model=model,
            prompt=prompt,
            mask_id=5,
            gen_length=3,
            block_length=2,
            denoising_steps=2,
            confidence_threshold=0.0,
            mtp_like_prefix_size=2,
        )

        self.assertEqual(tuple(generated_ids.shape), (1, 4))
        self.assertFalse((generated_ids[:, 1:] == 5).any())
        self.assertEqual(stats["forward_rounds"].item(), 2)
        self.assertEqual(stats["length_counts"].tolist(), [1, 1])
        self.assertEqual(stats["confidence_count"].item(), 3)
        self.assertEqual(stats["prefix_pair_opportunities"].item(), 2)
        self.assertEqual(stats["prefix_pair_count"].item(), 1)

    def test_temperature_zero_uses_greedy_decoding(self):
        model = ToyNativeSDAR()
        prompt = {"input_ids": torch.tensor([[2]])}

        generated_ids, stats = block_diffusion_generate_with_stats(
            model=model,
            prompt=prompt,
            mask_id=5,
            gen_length=3,
            block_length=2,
            denoising_steps=2,
            temperature=0.0,
            confidence_threshold=0.0,
        )

        self.assertEqual(generated_ids.tolist(), [[2, 1, 1, 1]])
        self.assertEqual(stats["confidence_count"].item(), 3)


if __name__ == "__main__":
    unittest.main()
