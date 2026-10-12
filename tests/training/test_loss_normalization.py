import math
import unittest
from types import SimpleNamespace

import torch

from grpo import (
    block_logprob_loss,
    compute_group_advantages,
    compute_pooled_loss_scale,
    logprob_loss,
)
from utils.runtime.timing import TimingRecorder


def _distributed_average(local_numerators, *, world_size, scale):
    """Model DDP/FSDP's rank-average after local loss scaling."""
    return sum(local_numerators) * scale / world_size


class _BackwardOnlyAccelerator:
    @staticmethod
    def backward(loss):
        loss.backward()


class _TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.token_logits = torch.nn.Parameter(torch.tensor([0.2, -0.1, 0.4]))

    def forward(self, input_ids):
        batch_size, sequence_length = input_ids.shape
        logits = self.token_logits.view(1, 1, -1).expand(
            batch_size, sequence_length, -1
        )
        return SimpleNamespace(logits=logits)


def _learner_input(target, advantage):
    return {
        "generated_ids": torch.tensor([[0, target]], dtype=torch.long),
        "prompt_len": 1,
        "advantages": torch.tensor([advantage], dtype=torch.float32),
        "rewards": torch.tensor([1.0]),
    }


def _block_learner_input(target, advantage):
    inputs = _learner_input(target, advantage)
    inputs.update(
        {
            "block_starts": torch.tensor([[0]], dtype=torch.long),
            "block_lengths": torch.tensor([[1]], dtype=torch.long),
            "num_blocks": torch.tensor([1], dtype=torch.long),
        }
    )
    return inputs


def _gradient_for_topology(assignments, targets, advantages):
    world_size = len(assignments)
    global_valid = sum(advantage != 0 for advantage in advantages)
    loss_scale = compute_pooled_loss_scale(global_valid, world_size, gen_length=1)
    rank_gradients = []

    for sample_indices in assignments:
        model = _TinyPolicy()
        for sample_index in sample_indices:
            logprob_loss(
                model,
                _learner_input(targets[sample_index], advantages[sample_index]),
                loss_scale=loss_scale,
                accelerator=_BackwardOnlyAccelerator(),
                gen_length=1,
            )
        rank_gradients.append(model.token_logits.grad.detach())

    # This is the rank-average performed by DDP/FSDP.
    return torch.stack(rank_gradients).mean(dim=0)


class PooledLossScaleTest(unittest.TestCase):
    def test_loss_returns_unscaled_local_numerators(self):
        model = _TinyPolicy()
        stats = logprob_loss(
            model,
            _learner_input(target=1, advantage=2.5),
            loss_scale=0.25,
            accelerator=_BackwardOnlyAccelerator(),
            gen_length=1,
        )

        self.assertEqual(stats["raw_loss_sum"].item(), -2.5)
        self.assertEqual(stats["raw_abs_loss_sum"].item(), 2.5)

    def test_loss_reports_logprob_and_backward_timing(self):
        model = _TinyPolicy()
        timing = TimingRecorder(enabled=True, device="cpu")

        logprob_loss(
            model,
            _learner_input(target=1, advantage=2.5),
            loss_scale=0.25,
            accelerator=_BackwardOnlyAccelerator(),
            gen_length=1,
            timing=timing,
        )

        elapsed = timing.elapsed()
        self.assertGreaterEqual(elapsed["logprob_prefill"], 0.0)
        self.assertGreaterEqual(elapsed["backward"], 0.0)

    def test_block_loss_uses_the_same_timing_phases(self):
        model = _TinyPolicy()
        timing = TimingRecorder(enabled=True, device="cpu")

        block_logprob_loss(
            model,
            _block_learner_input(target=1, advantage=2.5),
            loss_scale=0.25,
            max_block_size=1,
            accelerator=_BackwardOnlyAccelerator(),
            gen_length=1,
            timing=timing,
        )

        elapsed = timing.elapsed()
        self.assertGreaterEqual(elapsed["logprob_prefill"], 0.0)
        self.assertGreaterEqual(elapsed["backward"], 0.0)

    def test_actual_loss_gradient_is_topology_invariant(self):
        targets = [0, 1, 2, 1]
        advantages = [1.0, -2.0, 0.0, 3.0]

        gradient_2x2 = _gradient_for_topology(
            [[0, 2], [1, 3]], targets, advantages
        )
        gradient_4x1 = _gradient_for_topology(
            [[0], [1], [2], [3]], targets, advantages
        )
        global_reference = _gradient_for_topology(
            [[0, 1, 2, 3]], targets, advantages
        )

        self.assertTrue(torch.allclose(gradient_2x2, global_reference, atol=1e-7, rtol=1e-7))
        self.assertTrue(torch.allclose(gradient_4x1, global_reference, atol=1e-7, rtol=1e-7))

    def test_matches_single_global_reference_with_uneven_valid_counts(self):
        gen_length = 256

        # Four logical cohorts have deliberately different valid counts. Their
        # numerator contributions are not proportional, so per-cohort averaging
        # would also change the gradient direction in a vector-valued model.
        valid_by_cohort = [2, 4, 6, 8]
        numerator_by_cohort = [3.0, -5.0, 11.0, 7.0]
        global_valid = sum(valid_by_cohort)
        reference = sum(numerator_by_cohort) / (gen_length * global_valid)

        # 2 ranks x 4 accumulations and 8 ranks x 1 accumulation both represent
        # the same logical global batch. Partitioning the same numerator sum over
        # ranks must not change the resulting averaged gradient.
        local_2x4 = [10.0, 6.0]
        local_8x1 = [3.0, -5.0, 11.0, 7.0, 2.0, 4.0, 1.0, -7.0]

        result_2x4 = _distributed_average(
            local_2x4,
            world_size=2,
            scale=compute_pooled_loss_scale(global_valid, 2, gen_length),
        )
        result_8x1 = _distributed_average(
            local_8x1,
            world_size=8,
            scale=compute_pooled_loss_scale(global_valid, 8, gen_length),
        )

        self.assertTrue(math.isclose(result_2x4, reference, rel_tol=0.0, abs_tol=1e-15))
        self.assertTrue(math.isclose(result_8x1, reference, rel_tol=0.0, abs_tol=1e-15))

    def test_has_no_extra_gradient_accumulation_divisor(self):
        scale = compute_pooled_loss_scale(global_valid=64, world_size=16, gen_length=256)
        self.assertEqual(scale, 16 / (256 * 64))

    def test_zero_valid_group_has_zero_scale(self):
        self.assertEqual(
            compute_pooled_loss_scale(0, world_size=64, gen_length=256),
            0.0,
        )

    def test_accepts_scalar_tensor_count(self):
        self.assertEqual(
            compute_pooled_loss_scale(torch.tensor(8), world_size=2, gen_length=4),
            1 / 16,
        )

    def test_group_advantages_reject_invalid_groups(self):
        cases = [
            (torch.tensor([1.0]), 1, "at least 2"),
            (torch.tensor([1.0, 2.0, 3.0]), 2, "divisible"),
            (torch.tensor([1.0, float("nan")]), 2, "non-finite"),
        ]
        for rewards, group_size, message in cases:
            with self.subTest(group_size=group_size, message=message):
                with self.assertRaisesRegex(ValueError, message):
                    compute_group_advantages(rewards, group_size)


    def test_invalid_scale_inputs_fail_fast(self):
        cases = [
            ({"global_valid": -1, "world_size": 2, "gen_length": 4}, ValueError),
            ({"global_valid": float("nan"), "world_size": 2, "gen_length": 4}, ValueError),
            ({"global_valid": 1, "world_size": 0, "gen_length": 4}, ValueError),
            ({"global_valid": 1, "world_size": 2, "gen_length": 0}, ValueError),
        ]
        for kwargs, exception in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(exception):
                compute_pooled_loss_scale(**kwargs)


if __name__ == "__main__":
    unittest.main()
