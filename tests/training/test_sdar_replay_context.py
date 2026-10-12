"""Cached SDAR replay must preserve the mask context of a partial native block."""

import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from grpo import block_logprob_loss, block_token_log_probs
from utils.models.forward import install_position_selective_output_head
from utils.generation.generate import generate_monotonic_blocks


class ContextSDAR(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(block_size=4)
        self.lm_head = torch.nn.Linear(1, 6, bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(torch.tensor([[0.0], [1.0], [-1.0], [0.2], [0.1], [0.0]]))
        self.calls = []

    def forward(self, input_ids, **kwargs):
        self.calls.append(input_ids.detach().clone())
        # This test uses one native block, whose positions mutually attend.
        # An omitted final mask changes both its predictions and gradients.
        context = input_ids.float().mean(dim=1) - 3.8
        hidden = context[:, None, None].expand(-1, input_ids.shape[1], 1)
        return SimpleNamespace(logits=self.lm_head(hidden))


class BackwardOnlyAccelerator:
    @staticmethod
    def backward(loss):
        loss.backward()


class SdarReplayContextTest(unittest.TestCase):
    def test_partial_native_block_replay_matches_rollout_likelihood_and_gradient(self):
        prompt = torch.tensor([[1]])
        for selective in (False, True):
            with self.subTest(selective=selective):
                model = ContextSDAR().eval()
                reference = copy.deepcopy(model)
                if selective:
                    install_position_selective_output_head(model, model_family="sdar")
                rollout = generate_monotonic_blocks(
                    model, prompt, gen_length=2, max_block_size=2,
                    schedule="fixed", temperature=1.0, mask_id=5,
                    model_family="sdar", sampling_vocab_size=5,
                    generators=[torch.Generator().manual_seed(1234)],
                )
                self.assertEqual(tuple(rollout["generated_ids"].shape), (1, 3))
                self.assertEqual(rollout["rollout_input_length"], 4)
                sampling_state = model.calls[0]
                self.assertEqual(sampling_state.tolist(), [[1, 5, 5, 5]])

                target = rollout["generated_ids"][:, 1:]
                logits = reference(sampling_state).logits[:, 1:3, :5]
                expected_log_probs = torch.log_softmax(logits, dim=-1).gather(
                    -1, target.unsqueeze(-1),
                ).squeeze(-1)
                with torch.no_grad():
                    truncated_logits = reference(sampling_state[:, :-1]).logits[:, 1:3, :5]
                    truncated_log_probs = torch.log_softmax(truncated_logits, dim=-1).gather(
                        -1, target.unsqueeze(-1),
                    ).squeeze(-1)
                self.assertFalse(torch.allclose(expected_log_probs, truncated_log_probs))
                (-expected_log_probs.sum()).backward()
                expected_gradient = reference.lm_head.weight.grad.clone()

                captured_log_probs = []

                def capture_log_probs(*args, **kwargs):
                    values = block_token_log_probs(*args, **kwargs)
                    captured_log_probs.append(values[0].detach().clone())
                    return values

                model.calls.clear()
                rollout.update(prompt_len=1, advantages=torch.ones(1), sampling_vocab_size=5)
                with patch("grpo.block_token_log_probs", side_effect=capture_log_probs):
                    block_logprob_loss(
                        model, rollout, loss_scale=1.0, max_block_size=2,
                        accelerator=BackwardOnlyAccelerator(), gen_length=2,
                        mask_id=5, model_family="sdar", selective_logits=selective,
                    )
                self.assertEqual(len(model.calls), 1)
                torch.testing.assert_close(model.calls[0], sampling_state)
                torch.testing.assert_close(captured_log_probs[0], expected_log_probs.detach())
                torch.testing.assert_close(model.lm_head.weight.grad, expected_gradient)

    def test_missing_context_metadata_preserves_unpadded_input(self):
        model = ContextSDAR()
        inputs = {
            "generated_ids": torch.tensor([[1, 2, 2]]), "prompt_len": 1,
            "advantages": torch.ones(1), "block_starts": torch.tensor([[0]]),
            "block_lengths": torch.tensor([[2]]), "num_blocks": torch.tensor([1]),
        }
        block_logprob_loss(
            model, inputs, loss_scale=1.0, max_block_size=2,
            accelerator=BackwardOnlyAccelerator(), gen_length=2,
            mask_id=5, model_family="sdar",
        )
        self.assertEqual(model.calls[0].tolist(), [[1, 5, 5]])


if __name__ == "__main__":
    unittest.main()
