import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from grpo import block_logprob_loss, block_token_log_probs, logprob_loss, sample


class _BackwardOnlyAccelerator:
    @staticmethod
    def backward(loss):
        loss.backward()


class _Policy(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.token_logits = torch.nn.Parameter(logits.clone())

    def forward(self, input_ids):
        batch_size, sequence_length = input_ids.shape
        logits = self.token_logits.view(1, 1, -1).expand(
            batch_size, sequence_length, -1
        )
        return SimpleNamespace(logits=logits)


class _Tokenizer:
    def __len__(self):
        return 5

    def apply_chat_template(self, conversations, **kwargs):
        return ["prompt" for _ in conversations]

    def __call__(self, prompts, **kwargs):
        return {"input_ids": torch.zeros(len(prompts), 1, dtype=torch.long)}

    def batch_decode(self, token_ids, **kwargs):
        return ["response" for _ in token_ids]


class SamplingVocabularyTest(unittest.TestCase):
    def test_rollout_records_the_tokenizer_sampling_support(self):
        model = _Policy(torch.zeros(8)).eval()
        with patch("grpo.generate", return_value=torch.tensor([[0, 1], [0, 2]])) as generate:
            rollout = sample(
                model,
                {"problems": ["question"]},
                _Tokenizer(),
                torch.device("cpu"),
                reward_fn=lambda *args: torch.tensor([1.0, -1.0]),
                num_generations=2,
                gen_length=1,
                steps=1,
                mask_id=4,
            )
        self.assertEqual(rollout["sampling_vocab_size"], 5)
        self.assertEqual(generate.call_args.kwargs["sampling_vocab_size"], 5)

    def test_baseline_scores_and_gradients_ignore_padded_logits_but_keep_mask(self):
        valid_logits = torch.tensor([0.2, -0.1, 0.4, 0.6, 1.0])
        padded = _Policy(torch.cat([valid_logits, torch.tensor([1000.0, 1100.0])]))
        reference = _Policy(valid_logits)
        inputs = {
            "generated_ids": torch.tensor([[0, 1]]),
            "prompt_len": 1,
            "advantages": torch.tensor([2.0]),
            "sampling_vocab_size": 5,
        }
        scored_log_probs = []
        original_log_softmax = torch.nn.functional.log_softmax

        def capture_log_probs(*args, **kwargs):
            result = original_log_softmax(*args, **kwargs)
            scored_log_probs.append(result.detach())
            return result

        with patch("grpo.F.log_softmax", side_effect=capture_log_probs):
            for model in (padded, reference):
                logprob_loss(
                    model, inputs, loss_scale=0.25,
                    accelerator=_BackwardOnlyAccelerator(), gen_length=1, mask_id=4,
                )

        torch.testing.assert_close(scored_log_probs[0][..., :5], scored_log_probs[1])
        self.assertTrue(torch.isfinite(scored_log_probs[0][..., 4]).all())
        torch.testing.assert_close(padded.token_logits.grad[:5], reference.token_logits.grad)
        torch.testing.assert_close(padded.token_logits.grad[5:], torch.zeros(2))
        expected_gradient = 0.5 * valid_logits.softmax(dim=-1)
        expected_gradient[1] -= 0.5
        torch.testing.assert_close(reference.token_logits.grad, expected_gradient)

    def test_block_scores_and_gradients_match_valid_support_with_inactive_padding(self):
        for selected in (False, True):
            with self.subTest(selected=selected):
                sequence_length = 2 if selected else 4
                valid_logits = torch.arange(2 * sequence_length * 5).reshape(
                    2, sequence_length, 5
                ).float() / 10
                padded_logits = torch.cat([
                    valid_logits,
                    torch.full((2, sequence_length, 3), 1000.0),
                ], dim=-1).requires_grad_()
                reference_logits = valid_logits.clone().requires_grad_()
                kwargs = {
                    "completion_ids": torch.tensor([[1, 2, 3], [1, 7, 4]]),
                    "prompt_len": 1,
                    "block_starts": torch.tensor([0, 1]),
                    "block_lengths": torch.tensor([2, 0]),
                    "max_block_size": 2,
                    "mask_id": 4,
                    "logits_are_selected": selected,
                    "sampling_vocab_size": 5,
                }
                padded_scores, accepted = block_token_log_probs(padded_logits, **kwargs)
                reference_scores, _ = block_token_log_probs(reference_logits, **kwargs)
                torch.testing.assert_close(padded_scores, reference_scores)
                first_position = 0 if selected else 1
                expected_scores = valid_logits[0, first_position:first_position + 2, :4].log_softmax(dim=-1)
                torch.testing.assert_close(padded_scores[0], expected_scores[[0, 1], [1, 2]])
                self.assertTrue(torch.isfinite(padded_scores).all())
                torch.testing.assert_close(padded_scores[~accepted], torch.zeros(2))
                padded_scores.sum().backward()
                reference_scores.sum().backward()
                torch.testing.assert_close(padded_logits.grad[..., :5], reference_logits.grad)
                torch.testing.assert_close(padded_logits.grad[..., 5:], torch.zeros_like(padded_logits.grad[..., 5:]))
                self.assertEqual(padded_logits.grad[..., 4].count_nonzero().item(), 0)
                self.assertTrue(torch.isfinite(padded_logits.grad).all())
                self.assertEqual(padded_logits.grad[1].count_nonzero().item(), 0)

    def test_block_loss_excludes_padding_slots_and_padded_vocabulary(self):
        valid_logits = torch.tensor([0.2, -0.1, 0.4, 0.6, 1.0])
        padded = _Policy(torch.cat([valid_logits, torch.tensor([1000.0, 1100.0, 1200.0])]))
        reference = _Policy(valid_logits)
        inputs = {
            "generated_ids": torch.tensor([[0, 1, 2, 3], [0, 1, 2, 7]]),
            "prompt_len": 1,
            "advantages": torch.tensor([0.75, -0.5]),
            "block_starts": torch.tensor([[0, 2], [0, 2]]),
            "block_lengths": torch.tensor([[2, 1], [2, 0]]),
            "num_blocks": torch.tensor([2, 1]),
            "sampling_vocab_size": 5,
        }
        results = []
        for model in (padded, reference):
            results.append(block_logprob_loss(
                model, inputs, loss_scale=0.25, max_block_size=2,
                accelerator=_BackwardOnlyAccelerator(), gen_length=3, mask_id=4,
            ))
        self.assertTrue(torch.isfinite(results[0]["raw_loss_sum"]))
        self.assertTrue(torch.isfinite(padded.token_logits.grad).all())
        torch.testing.assert_close(results[0]["raw_loss_sum"], results[1]["raw_loss_sum"])
        torch.testing.assert_close(padded.token_logits.grad[:5], reference.token_logits.grad)
        torch.testing.assert_close(padded.token_logits.grad[5:], torch.zeros(3))


if __name__ == "__main__":
    unittest.main()
