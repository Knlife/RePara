import unittest
from collections import Counter
from itertools import islice

from accelerate.data_loader import BatchSamplerShard
from torch.utils.data import BatchSampler

from data.sampler import InfiniteSampler


def _sample_ids_by_step(
    dataset_size,
    start_position,
    world_size,
    grad_accumulation,
    steps,
    batch_size_per_device=1,
):
    """Simulate Accelerate's non-split BatchSamplerShard."""
    global_samples = list(islice(
        iter(InfiniteSampler(range(dataset_size), start_position=start_position)),
        world_size * batch_size_per_device * grad_accumulation * steps,
    ))
    global_batches = [
        global_samples[offset : offset + batch_size_per_device]
        for offset in range(0, len(global_samples), batch_size_per_device)
    ]
    rank_batches = [global_batches[rank::world_size] for rank in range(world_size)]

    result = []
    for step in range(steps):
        step_ids = []
        offset = step * grad_accumulation
        for accum_idx in range(grad_accumulation):
            for rank in range(world_size):
                step_ids.extend(rank_batches[rank][offset + accum_idx])
        result.append(step_ids)
    return result


def _accelerate_sample_ids_by_step(
    dataset_size,
    start_position,
    world_size,
    grad_accumulation,
    steps,
    batch_size_per_device=1,
):
    """Exercise Accelerate's real non-split batch sharding implementation."""
    rank_iterators = []
    for rank in range(world_size):
        sampler = InfiniteSampler(range(dataset_size), start_position=start_position)
        batch_sampler = BatchSampler(
            sampler, batch_size=batch_size_per_device, drop_last=False
        )
        shard = BatchSamplerShard(
            batch_sampler,
            num_processes=world_size,
            process_index=rank,
            split_batches=False,
            even_batches=False,
        )
        rank_iterators.append(iter(shard))

    result = []
    for _ in range(steps):
        step_ids = []
        for _ in range(grad_accumulation):
            for rank_iterator in rank_iterators:
                step_ids.extend(next(rank_iterator))
        result.append(step_ids)
    return result


class InfiniteSamplerTest(unittest.TestCase):
    def test_start_position_is_absolute_and_wraps(self):
        actual = list(islice(iter(InfiniteSampler(range(5), start_position=7)), 8))
        self.assertEqual(actual, [2, 3, 4, 0, 1, 2, 3, 4])

    def test_invalid_state_is_rejected(self):
        with self.assertRaises(ValueError):
            InfiniteSampler([], start_position=0)
        with self.assertRaises(ValueError):
            InfiniteSampler(range(3), start_position=-1)
        with self.assertRaises(TypeError):
            InfiniteSampler(range(3), start_position=1.5)

    def test_equal_global_batches_across_topologies(self):
        dataset_size = 13
        start_position = 9
        steps = 5

        two_by_four = _sample_ids_by_step(
            dataset_size, start_position, world_size=2, grad_accumulation=4, steps=steps,
        )
        eight_by_one = _sample_ids_by_step(
            dataset_size, start_position, world_size=8, grad_accumulation=1, steps=steps,
        )

        for step, (left, right) in enumerate(zip(two_by_four, eight_by_one)):
            expected = [
                (start_position + step * 8 + offset) % dataset_size
                for offset in range(8)
            ]
            self.assertEqual(Counter(left), Counter(right))
            self.assertEqual(Counter(left), Counter(expected))

    def test_real_sharding_is_equal_for_8x8x1_and_1x16x4(self):
        dataset_size = 4096
        steps = 3

        eight_by_eight_by_one = _accelerate_sample_ids_by_step(
            dataset_size,
            0,
            world_size=8,
            batch_size_per_device=8,
            grad_accumulation=1,
            steps=steps,
        )
        one_by_sixteen_by_four = _accelerate_sample_ids_by_step(
            dataset_size,
            0,
            world_size=1,
            batch_size_per_device=16,
            grad_accumulation=4,
            steps=steps,
        )

        for step, (left, right) in enumerate(
            zip(eight_by_eight_by_one, one_by_sixteen_by_four)
        ):
            expected = list(range(step * 64, (step + 1) * 64))
            self.assertEqual(sorted(left), expected)
            self.assertEqual(sorted(right), expected)

    def test_resume_cursor_can_change_topology(self):
        dataset_size = 17
        initial_steps = 3
        resumed_steps = 4
        global_batch_size = 8
        cursor = initial_steps * global_batch_size

        resumed = _sample_ids_by_step(
            dataset_size,
            cursor,
            world_size=8,
            grad_accumulation=1,
            steps=resumed_steps,
        )
        reference = _sample_ids_by_step(
            dataset_size,
            0,
            world_size=2,
            grad_accumulation=4,
            steps=initial_steps + resumed_steps,
        )[initial_steps:]

        self.assertEqual([Counter(step) for step in resumed], [Counter(step) for step in reference])


if __name__ == "__main__":
    unittest.main()
