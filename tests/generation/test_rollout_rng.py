from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from grpo import _should_synchronize_mtp_forward_rounds
from utils.generation.generate import (
    add_gumbel_noise,
    generate,
    generate_monotonic_blocks,
)
from utils.runtime.rng import derive_learner_seed, derive_seed, make_generators


class DeterministicModel:
    """Small model stub whose logits depend only on each row's current tokens."""

    def __init__(self, vocab_size=7):
        self.vocab_size = vocab_size

    def __call__(self, input_ids):
        batch, length = input_ids.shape
        token_ids = torch.arange(self.vocab_size, dtype=torch.float32, device=input_ids.device)
        positions = torch.arange(length, dtype=torch.float32, device=input_ids.device)
        row_bias = input_ids.float().sum(dim=1, keepdim=True).unsqueeze(-1) * 0.01
        logits = token_ids.view(1, 1, -1) * 0.07
        logits = logits + positions.view(1, -1, 1) * 0.03 + row_bias
        return SimpleNamespace(logits=logits.expand(batch, -1, -1).clone())


def _seeds(sample_ids, *, stream):
    return [
        derive_seed(
            1234,
            stream=stream,
            optimizer_step=7,
            sample_id=sample_id,
            repeat_id=1,
            generation_id=3,
        )
        for sample_id in sample_ids
    ]


class RolloutRngTest(unittest.TestCase):
    def test_learner_seed_is_stable_rank_local_and_world_size_free(self):
        self.assertEqual(
            derive_learner_seed(1234, rank=7),
            derive_learner_seed(1234, rank=7),
        )
        self.assertNotEqual(
            derive_learner_seed(1234, rank=7),
            derive_learner_seed(1234, rank=8),
        )

    def test_derive_seed_has_stable_known_answer(self):
        seed = derive_seed(
            1234,
            stream="gumbel-v1",
            optimizer_step=7,
            sample_id="gsm8k:42",
            repeat_id=1,
            generation_id=3,
        )

        self.assertEqual(seed, 1496865584815556242)
        self.assertTrue(0 <= seed < 1 << 63)

    def test_derive_seed_distinguishes_logical_key_fields(self):
        baseline = derive_seed(
            9,
            stream="gumbel-v1",
            optimizer_step=2,
            sample_id="sample-4",
            repeat_id=1,
            generation_id=5,
        )
        key = {
            "stream": "gumbel-v1",
            "optimizer_step": 2,
            "sample_id": "sample-4",
            "repeat_id": 1,
            "generation_id": 5,
        }
        variants = set()
        for field, value in (
            ("stream", "remask-v1"),
            ("optimizer_step", 3),
            ("sample_id", "sample-5"),
            ("repeat_id", 2),
            ("generation_id", 6),
        ):
            variant = dict(key)
            variant[field] = value
            variants.add(derive_seed(9, **variant))
        variants.add(derive_seed(10, **key))

        self.assertNotIn(baseline, variants)
        self.assertEqual(len(variants), 6)

    def test_explicit_gumbel_generators_are_row_order_invariant(self):
        logits = torch.linspace(-1.0, 1.0, steps=3 * 2 * 5).reshape(3, 2, 5)
        seeds = _seeds(["a", "b", "c"], stream="gumbel-v1")

        torch.manual_seed(2026)
        global_state = torch.random.get_rng_state().clone()
        noise_a = add_gumbel_noise(logits, 1.0, make_generators(seeds, "cpu"))
        self.assertTrue(torch.equal(global_state, torch.random.get_rng_state()))

        order = torch.tensor([2, 0, 1])
        reordered_seeds = [seeds[index] for index in order.tolist()]
        noise_b = add_gumbel_noise(
            logits.index_select(0, order),
            1.0,
            make_generators(reordered_seeds, "cpu"),
        )

        self.assertTrue(torch.equal(noise_a.index_select(0, order), noise_b))

    def test_generate_is_row_order_invariant_with_per_row_generators(self):
        sample_ids = ["sample-a", "sample-b", "sample-c"]
        prompt = torch.tensor([[1, 2], [1, 2], [1, 2]])
        gumbel_seeds = _seeds(sample_ids, stream="gumbel-v1")
        remask_seeds = _seeds(sample_ids, stream="random-remask-v1")
        order = torch.tensor([2, 0, 1])

        for remasking in ("low_confidence", "random"):
            with self.subTest(remasking=remasking):
                generated_a = generate(
                    DeterministicModel(),
                    prompt,
                    steps=4,
                    gen_length=4,
                    block_length=1,
                    temperature=1.0,
                    remasking=remasking,
                    mask_id=6,
                    generators=make_generators(gumbel_seeds, "cpu"),
                    remasking_generators=make_generators(remask_seeds, "cpu"),
                )

                reordered_gumbel = [gumbel_seeds[index] for index in order.tolist()]
                reordered_remask = [remask_seeds[index] for index in order.tolist()]
                generated_b = generate(
                    DeterministicModel(),
                    prompt.index_select(0, order),
                    steps=4,
                    gen_length=4,
                    block_length=1,
                    temperature=1.0,
                    remasking=remasking,
                    mask_id=6,
                    generators=make_generators(reordered_gumbel, "cpu"),
                    remasking_generators=make_generators(reordered_remask, "cpu"),
                )

                self.assertTrue(
                    torch.equal(generated_a.index_select(0, order), generated_b)
                )

    def test_generator_count_must_match_batch_rows(self):
        logits = torch.zeros(2, 3, 4)
        with self.assertRaisesRegex(ValueError, "one generator per batch row"):
            add_gumbel_noise(logits, 1.0, make_generators([1], "cpu"))

    def test_eval_style_calls_without_generators_remain_supported(self):
        generated = generate(
            DeterministicModel(),
            torch.tensor([[1, 2]]),
            steps=2,
            gen_length=2,
            block_length=1,
            temperature=0.0,
            mask_id=6,
        )

        self.assertEqual(generated.shape, (1, 4))

    def test_adaptive_forward_sync_is_opt_in_for_training(self):
        prompt = torch.tensor([[1, 2]])
        generators = make_generators([17], "cpu")
        with (
            mock.patch.object(torch.distributed, "is_available", return_value=True),
            mock.patch.object(torch.distributed, "is_initialized", return_value=True),
            mock.patch.object(torch.distributed, "all_reduce") as all_reduce,
        ):
            generate_monotonic_blocks(
                DeterministicModel(),
                prompt,
                gen_length=2,
                max_block_size=2,
                schedule="adaptive",
                confidence_threshold=0.0,
                temperature=1.0,
                mask_id=6,
                generators=generators,
            )
            all_reduce.assert_not_called()

            generate_monotonic_blocks(
                DeterministicModel(),
                prompt,
                gen_length=2,
                max_block_size=2,
                schedule="adaptive",
                confidence_threshold=0.0,
                temperature=1.0,
                mask_id=6,
                generators=make_generators([17], "cpu"),
                synchronize_forward_rounds=True,
            )
            self.assertGreater(all_reduce.call_count, 0)





if __name__ == "__main__":
    unittest.main()
