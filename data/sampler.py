from itertools import islice

import torch

import utils.runtime.distributed as dist


class InfiniteSampler(torch.utils.data.Sampler):
    """Yield a topology-independent, infinitely repeating dataset order.

    ``start_position`` is an absolute position in the global sample stream. It
    is intentionally not mutated while iterating: the training loop owns the
    committed global cursor, so DataLoader prefetch cannot advance checkpoint
    state past a completed optimizer step.
    """

    def __init__(self, dataset, start_position=0):
        if len(dataset) == 0:
            raise ValueError("InfiniteSampler requires a non-empty dataset")
        if not isinstance(start_position, int):
            raise TypeError("start_position must be an integer")
        if start_position < 0:
            raise ValueError("start_position must be non-negative")

        super().__init__()
        self.dataset = dataset
        self.start_position = start_position

    def __iter__(self):
        dataset_size = len(self.dataset)
        position = self.start_position
        while True:
            yield position % dataset_size
            position += 1


def split_by_node(src, group=None):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size > 1:
        yield from islice(src, rank, None, world_size)
    else:
        yield from src
        
def nosplit_by_node(src, group=None):
    yield from src
