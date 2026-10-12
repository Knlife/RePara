"""CPU/Gloo regression for cached SDAR prefill with uneven prompt lengths."""

from datetime import timedelta
import tempfile
from types import SimpleNamespace
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from utils.generation.generate import generate_monotonic_blocks


class CollectiveToySDAR(torch.nn.Module):
    def __init__(self, *, collective):
        super().__init__()
        self.config = SimpleNamespace(block_size=32)
        self.collective = collective
        self.calls = []

    def forward(self, input_ids, *, position_ids, past_key_values=None,
                store_kv=False, use_cache=False, **kwargs):
        self.calls.append((input_ids.shape[1], past_key_values, store_kv, use_cache))
        if self.collective:
            # Stand in for the parameter gather in an FSDP forward. A rank
            # entering the rollout's scalar all_reduce instead must fail.
            value = torch.tensor([dist.get_rank(), len(self.calls)], dtype=torch.int64)
            gathered = [torch.empty_like(value) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, value)
            assert all(int(item[1]) == len(self.calls) for item in gathered)
        positions = position_ids.expand(input_ids.shape[0], -1)
        logits = torch.zeros((*input_ids.shape, 17), device=input_ids.device)
        logits.scatter_(-1, positions.remainder(16).unsqueeze(-1), 10.0)
        return SimpleNamespace(logits=logits)


def _rollout(model, prompt, *, schedule, temperature, synchronize):
    return generate_monotonic_blocks(
        model, prompt, gen_length=40, max_block_size=2,
        schedule=schedule, confidence_threshold=0.95,
        temperature=temperature, mask_id=16, model_family="sdar",
        generators=[torch.Generator().manual_seed(1234)],
        synchronize_forward_rounds=synchronize,
    )


def _prefill_worker(rank, directory):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{directory}/rendezvous",
        rank=rank, world_size=2, timeout=timedelta(seconds=20),
    )
    try:
        for lengths in ((20, 32), (20, 64), (31, 32), (32, 20), (20, 31), (32, 64)):
            for schedule in ("fixed", "adaptive"):
                temperatures = (0.0, 1.0) if schedule == "fixed" else (0.5, 1.0)
                for temperature in temperatures:
                    prompt = torch.full((1, lengths[rank]), 3, dtype=torch.long)
                    model = CollectiveToySDAR(collective=True)
                    result = _rollout(
                        model, prompt, schedule=schedule,
                        temperature=temperature, synchronize=True,
                    )
                    reference_model = CollectiveToySDAR(collective=False)
                    reference = _rollout(
                        reference_model, prompt, schedule=schedule,
                        temperature=temperature, synchronize=False,
                    )
                    for key in ("generated_ids", "block_starts", "block_lengths", "num_blocks"):
                        assert torch.equal(result[key], reference[key]), (lengths, schedule, temperature, key)

                    prefill_length = (lengths[rank] // 32) * 32
                    first_length, first_cache, first_store, first_use = model.calls[0]
                    if prefill_length:
                        assert first_length == prefill_length
                        assert first_cache is not None and first_store and first_use
                    elif max(lengths) >= 32:
                        assert first_length == 1
                        assert first_cache is None and not first_store and not first_use
                        # The dummy prefill did not consume/cache the first
                        # native block; generation still starts at position 0.
                        assert model.calls[1][0] == 32
                    else:
                        assert first_length == 32 and not first_store

                    # Real prefill/cache commits retain their local order.
                    # A finished rank may make additional cache commits while
                    # another rank crosses a native-block boundary.
                    writes = [call[0] for call in model.calls if call[2]]
                    reference_writes = [call[0] for call in reference_model.calls if call[2]]
                    assert writes[:len(reference_writes)] == reference_writes
                    assert all(call[1] is not None and call[3] for call in model.calls if call[2])
                    counts = [None, None]
                    dist.all_gather_object(counts, len(model.calls))
                    assert counts[0] == counts[1], counts
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "requires Gloo")
class SDARPrefillDistributedTest(unittest.TestCase):
    def test_mixed_native_block_prefill_matches_local_rollout(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_prefill_worker, args=(directory,), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
