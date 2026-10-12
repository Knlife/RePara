import copy
import unittest
from unittest import mock
from types import SimpleNamespace

import torch
import torch.nn as nn

from grpo import block_logprob_loss, logprob_loss
from utils.models.forward import (
    forward_logits,
    forward_selected_logits,
    install_position_selective_output_head,
    select_logit_positions,
)
from utils.generation.generate import (

    generate,
    generate_monotonic_blocks,
)
from utils.runtime.rng import make_generators
from utils.runtime.timing import TimingRecorder


class ToyLLaDAInner(nn.Module):
    def __init__(self, vocab_size=13, hidden_size=7):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.Module()
        self.transformer.ff_out = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids):
        hidden = self.embedding(input_ids)
        hidden = hidden + torch.tanh(hidden.mean(dim=1, keepdim=True))
        return SimpleNamespace(logits=self.transformer.ff_out(hidden))


class ToyLLaDA(nn.Module):
    def __init__(self, vocab_size=13):
        super().__init__()
        self.config = SimpleNamespace(weight_tying=False)
        self.model = ToyLLaDAInner(vocab_size=vocab_size)

    def forward(self, input_ids):
        return self.model(input_ids)











class TailBiasedToyModel(nn.Module):
    def __init__(self, vocab_size=8, invalid_token_id=7, fallback_token_id=3):
        super().__init__()
        self.config = SimpleNamespace(block_size=2, fuse_cross_entropy=True)
        self.vocab_size = vocab_size
        self.invalid_token_id = invalid_token_id
        self.fallback_token_id = fallback_token_id

    def forward(self, input_ids, **kwargs):
        logits = torch.zeros(
            *input_ids.shape, self.vocab_size, device=input_ids.device
        )
        logits[..., self.invalid_token_id] = 100.0
        logits[..., self.fallback_token_id] = 50.0
        return SimpleNamespace(logits=logits)


class LocalAccelerator:
    def gather(self, value):
        return value

    def reduce(self, value, reduction="mean"):
        return value

    def backward(self, loss):
        loss.backward()


class SelectiveLogitsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_output_matches_full_logits_and_state_dict(self):
        model = ToyLLaDA()
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        positions = torch.tensor([0, 2])
        keys_before = list(model.state_dict())
        weight_before = model.model.transformer.ff_out.weight

        install_position_selective_output_head(model)
        full_logits = forward_logits(model, input_ids)
        selected_logits = forward_selected_logits(model, input_ids, positions)

        rows = torch.arange(input_ids.shape[0])
        torch.testing.assert_close(selected_logits, full_logits[rows, positions])
        self.assertEqual(keys_before, list(model.state_dict()))
        self.assertIs(weight_before, model.model.transformer.ff_out.weight)

    def test_output_matches_multiple_positions_per_row(self):
        model = ToyLLaDA()
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        positions = torch.tensor([[0, 2], [1, 1]])
        install_position_selective_output_head(model)

        full_logits = forward_logits(model, input_ids)
        selected_logits = forward_selected_logits(model, input_ids, positions)

        rows = torch.arange(input_ids.shape[0]).unsqueeze(1)
        torch.testing.assert_close(selected_logits, full_logits[rows, positions])



    def test_rollout_matches_greedy_reference(self):
        reference = ToyLLaDA(vocab_size=17)
        optimized = copy.deepcopy(reference)
        install_position_selective_output_head(optimized)
        prompt = torch.tensor([[1, 2], [1, 2]])

        expected = generate(
            reference, prompt, steps=4, gen_length=4, block_length=1,
            temperature=0.0, mask_id=16,
        )
        actual = generate(
            optimized, prompt, steps=4, gen_length=4, block_length=1,
            temperature=0.0, mask_id=16, selective_logits=True,
        )
        torch.testing.assert_close(actual, expected)

    def test_rollout_preserves_residual_mask_selection(self):
        reference = ToyLLaDA(vocab_size=17)
        with torch.no_grad():
            reference.model.embedding.weight.zero_()
            reference.model.embedding.weight[16].fill_(1.0)
            reference.model.transformer.ff_out.weight.zero_()
            reference.model.transformer.ff_out.weight[16].fill_(1.0)
        optimized = copy.deepcopy(reference)
        install_position_selective_output_head(optimized)
        prompt = torch.tensor([[1, 2], [1, 2]])

        expected = generate(
            reference, prompt, steps=4, gen_length=4, block_length=1,
            temperature=0.0, mask_id=16,
        )
        actual = generate(
            optimized, prompt, steps=4, gen_length=4, block_length=1,
            temperature=0.0, mask_id=16, selective_logits=True,
        )
        torch.testing.assert_close(actual, expected)

    def test_selective_rollout_handles_repeated_mask_with_generators(self):
        model = ToyLLaDA(vocab_size=3)
        with torch.no_grad():
            model.model.embedding.weight.fill_(1.0)
            model.model.transformer.ff_out.weight.zero_()
            model.model.transformer.ff_out.weight[2].fill_(50.0)
        install_position_selective_output_head(model)
        prompt = torch.tensor([[1, 1], [1, 1]])

        actual = generate(
            model,
            prompt,
            steps=2,
            gen_length=2,
            block_length=1,
            temperature=1.0,
            mask_id=2,
            generators=make_generators([101, 102], "cpu"),
            selective_logits=True,
        )

        expected = torch.tensor([
            [1, 1, 2, 2],
            [1, 1, 2, 2],
        ])
        torch.testing.assert_close(actual, expected)

    def test_selective_rollout_preserves_explicit_generator_streams(self):
        sample_count = 32
        optimized = ToyLLaDA(vocab_size=7)
        with torch.no_grad():
            optimized.model.transformer.ff_out.weight.zero_()
        install_position_selective_output_head(optimized)
        prompt = torch.tensor([[1, 2]]).repeat(sample_count, 1)
        seeds = list(range(100, 100 + sample_count))
        order = torch.arange(sample_count - 1, -1, -1)

        torch.manual_seed(2026)
        global_state = torch.random.get_rng_state().clone()
        optimized_tokens = generate(
            optimized, prompt, steps=1, gen_length=1, block_length=1,
            temperature=1.0, mask_id=6,
            generators=make_generators(seeds, "cpu"),
            selective_logits=True,
        )[:, -1]
        self.assertTrue(torch.equal(global_state, torch.random.get_rng_state()))

        reordered_tokens = generate(
            optimized, prompt.index_select(0, order),
            steps=1, gen_length=1, block_length=1,
            temperature=1.0, mask_id=6,
            generators=make_generators([seeds[index] for index in order.tolist()], "cpu"),
            selective_logits=True,
        )[:, -1]

        torch.testing.assert_close(optimized_tokens.index_select(0, order), reordered_tokens)

    def test_generate_masks_tokens_outside_sampling_vocab(self):
        model = TailBiasedToyModel()
        prompt = torch.tensor([[1, 2], [1, 2]])

        actual = generate(
            model,
            prompt,
            steps=2,
            gen_length=2,
            block_length=1,
            temperature=0.0,
            mask_id=4,
            sampling_vocab_size=5,
        )

        torch.testing.assert_close(actual[:, -2:], torch.full((2, 2), 3))



    def test_mtp_rollout_matches_full_logits(self):
        prompt = torch.tensor([[1, 2], [1, 2]])
        for schedule, temperature, threshold in (
            ("fixed", 0.0, 0.8),
            ("adaptive", 1.0, 0.0),
        ):
            with self.subTest(schedule=schedule):
                reference = ToyLLaDA(vocab_size=17)
                optimized = copy.deepcopy(reference)
                install_position_selective_output_head(optimized)

                torch.manual_seed(19)
                expected = generate_monotonic_blocks(
                    reference,
                    prompt,
                    gen_length=5,
                    max_block_size=2,
                    schedule=schedule,
                    confidence_threshold=threshold,
                    temperature=temperature,
                    mask_id=16,
                    generators=make_generators([101, 102], "cpu"),
                )
                actual = generate_monotonic_blocks(
                    optimized,
                    prompt,
                    gen_length=5,
                    max_block_size=2,
                    schedule=schedule,
                    confidence_threshold=threshold,
                    temperature=temperature,
                    mask_id=16,
                    generators=make_generators([101, 102], "cpu"),
                    selective_logits=True,
                )

                for key in ("generated_ids", "block_starts", "block_lengths", "num_blocks"):
                    torch.testing.assert_close(actual[key], expected[key])
                for key in (
                    "length_counts", "forced_singletons", "confidence_sum",
                    "confidence_count", "forward_rounds",
                ):
                    torch.testing.assert_close(actual["rollout_stats"][key], expected["rollout_stats"][key])

















    def test_loss_matches_k1_full_logits_gradient(self):
        reference = ToyLLaDA(vocab_size=13)
        optimized = copy.deepcopy(reference)
        install_position_selective_output_head(optimized)
        inputs = {
            "generated_ids": torch.tensor([
                [1, 2, 3, 4, 5],
                [1, 2, 6, 7, 8],
            ]),
            "prompt_len": 2,
            "advantages": torch.tensor([0.75, -0.25]),
            "rewards": torch.tensor([1.0, -1.0]),
        }
        timing = TimingRecorder(enabled=True, device="cpu")

        reference_stats = logprob_loss(
            reference, inputs, loss_scale=1.0, accelerator=LocalAccelerator(),
            gen_length=3, mask_id=12, selective_logits=False,
        )
        optimized_stats = logprob_loss(
            optimized, inputs, loss_scale=1.0, accelerator=LocalAccelerator(),
            gen_length=3, mask_id=12, selective_logits=True, timing=timing,
        )

        self.assertEqual(
            set(timing.elapsed()), {"backward", "logprob_prefill"},
        )
        self.assertAlmostEqual(
            reference_stats["raw_abs_loss_sum"].item(),
            optimized_stats["raw_abs_loss_sum"].item(),
            places=6,
        )
        for reference_parameter, optimized_parameter in zip(
            reference.parameters(), optimized.parameters(), strict=True,
        ):
            torch.testing.assert_close(
                reference_parameter.grad, optimized_parameter.grad, rtol=1e-5, atol=1e-6,
            )

    def test_mtp_block_loss_matches_full_logits_gradient(self):
        reference = ToyLLaDA(vocab_size=13)
        optimized = copy.deepcopy(reference)
        install_position_selective_output_head(optimized)
        inputs = {
            "generated_ids": torch.tensor([
                [1, 2, 3, 4, 5, 6],
                [1, 2, 7, 8, 9, 10],
            ]),
            "prompt_len": 2,
            "advantages": torch.tensor([0.75, -0.25]),
            "rewards": torch.tensor([1.0, -1.0]),
            "block_starts": torch.tensor([
                [0, 2, -1, -1],
                [0, 1, 3, -1],
            ]),
            "block_lengths": torch.tensor([
                [2, 2, 0, 0],
                [1, 2, 1, 0],
            ]),
            "num_blocks": torch.tensor([2, 3]),
        }
        timing = TimingRecorder(enabled=True, device="cpu")

        reference_stats = block_logprob_loss(
            reference,
            inputs,
            loss_scale=1.0,
            max_block_size=2,
            accelerator=LocalAccelerator(),
            gen_length=4,
            mask_id=12,
            selective_logits=False,
        )
        optimized_stats = block_logprob_loss(
            optimized,
            inputs,
            loss_scale=1.0,
            max_block_size=2,
            accelerator=LocalAccelerator(),
            gen_length=4,
            mask_id=12,
            selective_logits=True,
            timing=timing,
        )

        self.assertEqual(
            set(timing.elapsed()), {"backward", "logprob_prefill"},
        )
        self.assertAlmostEqual(
            reference_stats["raw_abs_loss_sum"].item(),
            optimized_stats["raw_abs_loss_sum"].item(),
            places=6,
        )
        self.assertEqual(reference_stats["loss_forward_rounds"], optimized_stats["loss_forward_rounds"])
        for reference_parameter, optimized_parameter in zip(
            reference.parameters(), optimized.parameters(), strict=True,
        ):
            torch.testing.assert_close(
                reference_parameter.grad, optimized_parameter.grad, rtol=1e-5, atol=1e-6,
            )


if __name__ == "__main__":
    unittest.main()
