"""Topology-independent random seeds for JustGRPO rollouts."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any, Iterable, List, Tuple, Union

if TYPE_CHECKING:
    import torch


ROLLOUT_RNG_SEMANTICS_VERSION = "justgrpo-rollout-rng-v1"
LEARNER_RNG_SEMANTICS_VERSION = "justgrpo-learner-rank-rng-v1"
_MAX_TORCH_SEED = (1 << 63) - 1


def _canonical_component(value: Any) -> List[Any]:
    """Encode supported key components without Python hash or repr semantics."""
    if value is None:
        return ["none", None]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        # JSON numbers can lose precision in other runtimes, so encode as decimal text.
        return ["int", str(value)]
    if isinstance(value, str):
        return ["str", value]
    if isinstance(value, bytes):
        return ["bytes", value.hex()]
    if isinstance(value, tuple):
        return ["tuple", [_canonical_component(item) for item in value]]
    if isinstance(value, list):
        return ["list", [_canonical_component(item) for item in value]]

    # Accept scalar tensor/NumPy-like IDs without importing those libraries here.
    item = getattr(value, "item", None)
    if callable(item):
        scalar = item()
        if scalar is value:
            raise TypeError(f"Unsupported seed component type: {type(value).__name__}")
        return _canonical_component(scalar)

    raise TypeError(
        "Seed key components must be None, bool, int, str, bytes, tuples/lists "
        f"of those values, or scalar objects with item(); got {type(value).__name__}"
    )


def stable_hash_seed(*components: Any, namespace: str) -> int:
    """Hash logical RNG key components into a portable non-negative 63-bit seed."""
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("namespace must be a non-empty string")

    payload = {
        "namespace": namespace,
        "components": [_canonical_component(component) for component in components],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.blake2b(encoded, digest_size=8, person=b"JustGRPO-RNG-v1").digest()
    return int.from_bytes(digest, byteorder="little", signed=False) & _MAX_TORCH_SEED


def derive_seed(
    base_seed: int,
    *,
    stream: str,
    optimizer_step: int,
    sample_id: Any,
    repeat_id: int,
    generation_id: int,
    semantics_version: str = ROLLOUT_RNG_SEMANTICS_VERSION,
) -> int:
    """Derive one completion's seed without using rank or world-size topology."""
    for name, value in (
        ("base_seed", base_seed),
        ("optimizer_step", optimizer_step),
        ("repeat_id", repeat_id),
        ("generation_id", generation_id),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an int")
    for name, value in (
        ("optimizer_step", optimizer_step),
        ("repeat_id", repeat_id),
        ("generation_id", generation_id),
    ):
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
    if not isinstance(stream, str) or not stream:
        raise ValueError("stream must be a non-empty string")

    return stable_hash_seed(
        base_seed,
        stream,
        optimizer_step,
        sample_id,
        repeat_id,
        generation_id,
        namespace=semantics_version,
    )


def derive_learner_seed(
    base_seed: int,
    *,
    rank: int,
    semantics_version: str = LEARNER_RNG_SEMANTICS_VERSION,
) -> int:
    """Derive an independent rank-local stream for learner stochasticity.

    Unlike rollout seeds, learner dropout is intentionally treated as ordinary
    stochastic-training noise rather than a sample-keyed, pathwise invariant
    stream. The world size is omitted so a run's RNG policy is not itself keyed
    by the selected topology.
    """
    for name, value in (("base_seed", base_seed), ("rank", rank)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an int")
    if rank < 0:
        raise ValueError("rank must be non-negative")
    return stable_hash_seed(base_seed, rank, namespace=semantics_version)


def make_generators(
    seeds: Iterable[int],
    device: Union[str, "torch.device"],
) -> Tuple["torch.Generator", ...]:
    """Create explicit PyTorch generators, one per batch row."""
    import torch

    generators = []
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("Every seed must be an int")
        if not 0 <= seed <= _MAX_TORCH_SEED:
            raise ValueError(f"Seed must be in [0, {_MAX_TORCH_SEED}]")
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        generators.append(generator)
    return tuple(generators)
