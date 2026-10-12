import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from utils.models.forward import (
    _get_model_config,
    forward_logits,
    forward_masked_logits,
    forward_selected_logits,
)


def _materialize_generators(generators):
    if generators is None or isinstance(generators, torch.Generator):
        return generators
    return tuple(generators)


def _rand_by_row(shape, *, device, dtype, generators=None):
    """Sample uniform values from either the global RNG or explicit row RNGs."""
    if generators is None:
        return torch.rand(shape, device=device, dtype=dtype)
    if isinstance(generators, torch.Generator):
        return torch.rand(shape, device=device, dtype=dtype, generator=generators)

    generators = _materialize_generators(generators)
    if len(shape) == 0:
        raise ValueError("Per-row generators require a tensor with a batch dimension")
    if len(generators) != shape[0]:
        raise ValueError(
            f"Expected one generator per batch row ({shape[0]}), got {len(generators)}"
        )
    if not all(isinstance(generator, torch.Generator) for generator in generators):
        raise TypeError(
            "generators must be a torch.Generator or a sequence of "
            "torch.Generator objects"
        )

    random_values = torch.empty(shape, device=device, dtype=dtype)
    for row, generator in enumerate(generators):
        random_values[row].uniform_(generator=generator)
    return random_values


def add_gumbel_noise(logits, temperature, generators=None):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = _rand_by_row(
        logits.shape,
        device=logits.device,
        dtype=torch.float64,
        generators=generators,
    )
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def _mask_invalid_sampling_logits(
    logits, *, mask_id, sampling_vocab_size=None, mask_mask_id=False
):
    """Mask token ids that should never be sampled from model logits."""
    if sampling_vocab_size is not None:
        sampling_vocab_size = int(sampling_vocab_size)
        if sampling_vocab_size < 1:
            raise ValueError("sampling_vocab_size must be positive")
        if sampling_vocab_size < logits.shape[-1]:
            logits[..., sampling_vocab_size:] = -torch.inf
    if mask_mask_id and 0 <= mask_id < logits.shape[-1]:
        logits[..., mask_id] = -torch.inf
    return logits


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


def update_diffusion_mtp_like_stats(stats, x, transfer_index, prompt_len, gen_length,
                                    prefix_size=2, mask_id=126336):
    if stats is None:
        return

    completion_mask = x[:, prompt_len:prompt_len + gen_length] == mask_id
    selected = transfer_index[:, prompt_len:prompt_len + gen_length]
    active = completion_mask.any(dim=1)
    if not active.any():
        return

    prefix_size = min(prefix_size, gen_length)
    offsets = torch.arange(prefix_size, device=x.device).unsqueeze(0)
    frontier = completion_mask.to(torch.int64).argmax(dim=1)
    positions = (frontier.unsqueeze(1) + offsets).clamp(max=gen_length - 1)
    valid_prefix = active.unsqueeze(1) & (frontier.unsqueeze(1) + offsets < gen_length)
    valid_prefix &= completion_mask.gather(1, positions)
    selected_prefix = selected.gather(1, positions) & valid_prefix

    prefix_lengths = selected_prefix.to(torch.int64).cumprod(dim=1).sum(dim=1)
    opportunities = valid_prefix.all(dim=1)
    stats["prefix_pair_opportunities"] += opportunities.sum()
    stats["prefix_pair_count"] += (opportunities & (prefix_lengths >= prefix_size)).sum()
    stats["prefix_transfer_sum"] += prefix_lengths[opportunities].sum()
    stats["prefix_transfer_count"] += opportunities.sum()
    stats["forward_rounds"] += active.sum()


def compute_mtp_confidence(logits, metric="normalized_entropy", temperature=1.0):
    if temperature < 0:
        raise ValueError("Adaptive MTP confidence requires a non-negative temperature")
    # Greedy token selection still measures confidence in the model distribution.
    temperature = temperature or 1.0

    scaled_logits = logits.float() / temperature
    log_probs = F.log_softmax(scaled_logits, dim=-1)
    probs = log_probs.exp()

    if metric == "normalized_entropy":
        entropy_terms = torch.where(torch.isfinite(log_probs), probs * log_probs, torch.zeros_like(probs))
        entropy = -entropy_terms.sum(dim=-1)
        support_size = torch.isfinite(scaled_logits).sum(dim=-1).clamp_min(2)
        return 1.0 - entropy / support_size.float().log()
    if metric == "max_probability":
        return probs.amax(dim=-1)
    raise ValueError(f"Unknown MTP confidence metric: {metric}")


def select_contiguous_prefix_length(confidence, valid_slots, threshold):
    if confidence.ndim != 2 or valid_slots.ndim != 1:
        raise ValueError("confidence must be [batch, slots] and valid_slots must be [batch]")

    slots = torch.arange(confidence.shape[1], device=confidence.device).unsqueeze(0)
    valid = slots < valid_slots.unsqueeze(1)
    passed = (confidence >= threshold) & valid
    prefix = passed.to(torch.int64).cumprod(dim=1).sum(dim=1)
    return torch.maximum(prefix, torch.ones_like(prefix)).minimum(valid_slots)








def _init_mtp_rollout_tracking(batch_size, gen_length, max_block_size, device):
    return {
        "positions": torch.zeros(batch_size, dtype=torch.long, device=device),
        "block_starts": torch.full(
            (batch_size, gen_length), -1, dtype=torch.long, device=device
        ),
        "block_lengths": torch.zeros(
            (batch_size, gen_length), dtype=torch.long, device=device
        ),
        "num_blocks": torch.zeros(batch_size, dtype=torch.long, device=device),
        "length_counts": torch.zeros(max_block_size, dtype=torch.long, device=device),
        "forced_singletons": torch.zeros((), dtype=torch.long, device=device),
        "confidence_sum": torch.zeros((), dtype=torch.float64, device=device),
        "confidence_count": torch.zeros((), dtype=torch.long, device=device),
        "forward_rounds": 0,
    }


def _mtp_rollout_result(x, tracking, total_length=None):
    if total_length is None:
        generated_ids = x
    else:
        generated_ids = x[:, :total_length]
    return {
        "generated_ids": generated_ids,
        "rollout_input_length": x.shape[1],
        "block_starts": tracking["block_starts"],
        "block_lengths": tracking["block_lengths"],
        "num_blocks": tracking["num_blocks"],
        "rollout_stats": {
            "length_counts": tracking["length_counts"],
            "forced_singletons": tracking["forced_singletons"],
            "confidence_sum": tracking["confidence_sum"],
            "confidence_count": tracking["confidence_count"],
            "forward_rounds": tracking["forward_rounds"],
        },
    }


def _is_globally_true(local_value, synchronize_forward_rounds, device=None):
    if torch.is_tensor(local_value):
        global_value = local_value.to(dtype=torch.int64)
    else:
        if device is None:
            raise ValueError("device is required when synchronizing a Python bool")
        global_value = torch.tensor(int(bool(local_value)), dtype=torch.int64, device=device)
    if (
        synchronize_forward_rounds
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        torch.distributed.all_reduce(global_value, op=torch.distributed.ReduceOp.MAX)
    return bool(global_value.item())


def _distributed_is_initialized():
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _distributed_rank():
    if not _distributed_is_initialized():
        return 0
    return torch.distributed.get_rank()


def _has_globally_active_rows(local_active, synchronize_forward_rounds):
    return _is_globally_true(local_active.any(), synchronize_forward_rounds)


def _global_position_range(positions, synchronize_forward_rounds):
    position_min = positions.min().to(dtype=torch.int64)
    position_max = positions.max().to(dtype=torch.int64)
    if synchronize_forward_rounds and _distributed_is_initialized():
        torch.distributed.all_reduce(position_min, op=torch.distributed.ReduceOp.MIN)
        torch.distributed.all_reduce(position_max, op=torch.distributed.ReduceOp.MAX)
    return int(position_min.item()), int(position_max.item())





def _apply_mtp_acceptance(
    x,
    *,
    prompt_len,
    tracking,
    local_active,
    candidate_positions,
    accepted_lengths,
    sampling_logits,
    offsets,
    temperature,
    generators,
    candidate_confidence=None,
    sample_active_only=False,
):
    positions = tracking["positions"]
    accepted_lengths = torch.where(
        local_active, accepted_lengths, torch.zeros_like(accepted_lengths)
    )
    if sample_active_only:
        active_rows = local_active.nonzero(as_tuple=False).squeeze(1)
        sampled_tokens = torch.zeros(
            sampling_logits.shape[:-1],
            dtype=torch.long,
            device=sampling_logits.device,
        )
        if active_rows.numel() > 0:
            active_generators = generators
            if generators is not None and not isinstance(generators, torch.Generator):
                generators = _materialize_generators(generators)
                active_generators = tuple(
                    generators[row] for row in active_rows.cpu().tolist()
                )
            sampled_tokens[active_rows] = torch.argmax(
                add_gumbel_noise(
                    sampling_logits[active_rows],
                    temperature,
                    generators=active_generators,
                ),
                dim=-1,
            )
    else:
        sampled_tokens = torch.argmax(
            add_gumbel_noise(
                sampling_logits,
                temperature,
                generators=generators,
            ),
            dim=-1,
        )
    accepted_mask = offsets < accepted_lengths.unsqueeze(1)
    accepted_rows, accepted_offsets = accepted_mask.nonzero(as_tuple=True)
    x[
        accepted_rows,
        prompt_len + candidate_positions[accepted_rows, accepted_offsets],
    ] = sampled_tokens[accepted_rows, accepted_offsets]

    active_rows = local_active.nonzero(as_tuple=False).squeeze(1)
    active_blocks = tracking["num_blocks"][active_rows]
    tracking["block_starts"][active_rows, active_blocks] = positions[active_rows]
    tracking["block_lengths"][active_rows, active_blocks] = accepted_lengths[active_rows]
    tracking["num_blocks"][active_rows] += 1

    for length in range(1, tracking["length_counts"].shape[0] + 1):
        tracking["length_counts"][length - 1] += (accepted_lengths == length).sum()

    if candidate_confidence is not None:
        tracking["confidence_sum"] += candidate_confidence[accepted_mask].double().sum()
        tracking["confidence_count"] += accepted_mask.sum()

    positions += accepted_lengths
    tracking["forward_rounds"] += 1
    return accepted_lengths














@torch.no_grad()
def generate_monotonic_blocks(model, prompt, gen_length=128, max_block_size=4, schedule="fixed",
                              confidence_metric="normalized_entropy", confidence_threshold=0.8,
                              temperature=1.0, mask_id=126336, model_family="llada",
                              generators=None, synchronize_forward_rounds=False,
                              selective_logits=False,
                              sampling_vocab_size=None):
    if max_block_size < 1:
        raise ValueError("max_block_size must be at least 1")
    if max_block_size > gen_length:
        raise ValueError("max_block_size cannot exceed gen_length")
    if schedule not in ("fixed", "adaptive"):
        raise ValueError(f"Unknown MTP schedule: {schedule}")
    if prompt.shape[0] > 1 and not (prompt[0] == prompt).all():
        raise ValueError("MTP rollout requires all prompt rows to be identical")

    batch_size, prompt_len = prompt.shape
    device = prompt.device


    # Materialize iterables once because this stream is reused at every round.
    generators = _materialize_generators(generators)
    x = torch.full((batch_size, prompt_len + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = prompt

    tracking = _init_mtp_rollout_tracking(
        batch_size, gen_length, max_block_size, device
    )

    while True:
        positions = tracking["positions"]
        local_active = positions < gen_length
        if not _has_globally_active_rows(local_active, synchronize_forward_rounds):
            break

        offsets = torch.arange(max_block_size, device=device).unsqueeze(0)
        candidate_positions = positions.unsqueeze(1) + offsets
        valid_slots = (gen_length - positions).clamp(min=0, max=max_block_size)

        safe_positions = candidate_positions.clamp(max=gen_length - 1)
        if selective_logits:
            candidate_logits = forward_selected_logits(
                model,
                x,
                positions=prompt_len + safe_positions,
                model_family=model_family,
            )
        else:
            logits = forward_logits(model, x, model_family=model_family)
            batch_index = torch.arange(batch_size, device=device).unsqueeze(1)
            candidate_logits = logits[batch_index, prompt_len + safe_positions]

        sampling_logits = candidate_logits.clone()
        _mask_invalid_sampling_logits(
            sampling_logits,
            mask_id=mask_id,
            sampling_vocab_size=sampling_vocab_size,
            mask_mask_id=True,
        )

        if schedule == "fixed":
            accepted_lengths = valid_slots
            candidate_confidence = None
        else:
            candidate_confidence = compute_mtp_confidence(
                sampling_logits, metric=confidence_metric, temperature=temperature,
            )
            accepted_lengths = select_contiguous_prefix_length(
                candidate_confidence, valid_slots, confidence_threshold,
            )
            first_failed = local_active & (candidate_confidence[:, 0] < confidence_threshold)
            tracking["forced_singletons"] += first_failed.sum()

        _apply_mtp_acceptance(
            x,
            prompt_len=prompt_len,
            tracking=tracking,
            local_active=local_active,
            candidate_positions=candidate_positions,
            accepted_lengths=accepted_lengths,
            sampling_logits=sampling_logits,
            offsets=offsets,
            temperature=temperature,
            generators=generators,
            candidate_confidence=candidate_confidence,
        )

    return _mtp_rollout_result(x, tracking)


@torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336,
             generators=None, remasking_generators=None, return_stats=False,
             mtp_like_prefix_size=2, model_family="llada",
             selective_logits=False, sampling_vocab_size=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The token id of [MASK]. LLaDA uses 126336 by default.
        generators: Optional torch.Generator or one Generator per batch row for
            Gumbel sampling. Per-row generators make samples independent of row
            ordering and distributed topology.
        remasking_generators: Optional independent generator(s) for random
            remasking. Defaults to ``generators`` when omitted.
        return_stats: Whether to return diffusion MTP-like transfer statistics.
        mtp_like_prefix_size: Prefix width used by the optional transfer statistics.
        model_family: Forward adapter family passed to ``forward_logits``.
        selective_logits: Project vocabulary logits only at positions consumed by rollout.
    '''
    assert prompt.shape[0] == 1 or (prompt[0] == prompt).all(), \
        "generate() requires all prompts in the batch to be identical (use prompt.repeat(n, 1)). " \
        "Variable-length prompts with mask_id padding are not supported."
    if selective_logits and (block_length != 1 or steps != gen_length):
        raise ValueError("Selective rollout logits require block_length == 1 and steps == gen_length")
    if selective_logits and cfg_scale != 0:
        raise ValueError("Selective rollout logits do not support classifier-free guidance")
    if selective_logits and return_stats:
        raise ValueError("Selective rollout logits do not support diffusion MTP-like statistics")
    if selective_logits and remasking != 'low_confidence':
        raise ValueError("Selective rollout logits currently require remasking='low_confidence'")

    # Materialize iterables once because these streams are reused at every step.
    generators = _materialize_generators(generators)
    remasking_generators = _materialize_generators(remasking_generators)

    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(prompt.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    stats = None
    if return_stats:
        stats = {
            "prefix_pair_count": torch.zeros((), device=prompt.device, dtype=torch.long),
            "prefix_pair_opportunities": torch.zeros((), device=prompt.device, dtype=torch.long),
            "prefix_transfer_sum": torch.zeros((), device=prompt.device, dtype=torch.long),
            "prefix_transfer_count": torch.zeros((), device=prompt.device, dtype=torch.long),
            "forward_rounds": torch.zeros((), device=prompt.device, dtype=torch.long),
        }

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if selective_logits:
                position = prompt.shape[1] + num_block
                candidate_mask = mask_index.clone()
                candidate_mask[:, position + 1:] = False
                candidate_logits, candidate_rows, candidate_positions = forward_masked_logits(
                    model, x, selection_mask=candidate_mask, model_family=model_family,
                )
                # Residual masks add candidates but still belong to their logical row's RNG stream.
                candidate_generators = generators
                if generators is not None and not isinstance(generators, torch.Generator):
                    if len(generators) != x.shape[0]:
                        raise ValueError(
                            f"Expected one generator per batch row ({x.shape[0]}), "
                            f"got {len(generators)}"
                        )
                    if candidate_logits.shape[0] != x.shape[0]:
                        candidate_generators = tuple(
                            generators[row] for row in candidate_rows.cpu().tolist()
                        )
                sampling_logits = candidate_logits.clone()
                _mask_invalid_sampling_logits(
                    sampling_logits,
                    mask_id=mask_id,
                    sampling_vocab_size=sampling_vocab_size,
                )
                sampled_candidates = torch.argmax(
                    add_gumbel_noise(
                        sampling_logits,
                        temperature=temperature,
                        generators=candidate_generators,
                    ),
                    dim=-1,
                )
                candidate_probs = F.softmax(sampling_logits, dim=-1)
                candidate_confidence = candidate_probs.gather(
                    -1, sampled_candidates.unsqueeze(-1),
                ).squeeze(-1)

                confidence = torch.full(
                    x.shape, -torch.inf, dtype=candidate_confidence.dtype, device=x.device,
                )
                sampled_tokens = torch.full_like(x, mask_id)
                confidence[candidate_rows, candidate_positions] = candidate_confidence
                sampled_tokens[candidate_rows, candidate_positions] = sampled_candidates
                _, transfer_positions = torch.topk(confidence, k=1, dim=1)
                transfer_positions = transfer_positions.squeeze(1)
                rows = torch.arange(x.shape[0], device=x.device)
                x[rows, transfer_positions] = sampled_tokens[rows, transfer_positions]
                continue

            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = forward_logits(model, x_, model_family=model_family)
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = forward_logits(model, x, model_family=model_family)

            logits = _mask_invalid_sampling_logits(
                logits.clone(),
                mask_id=mask_id,
                sampling_vocab_size=sampling_vocab_size,
            )
            logits_with_noise = add_gumbel_noise(
                logits,
                temperature=temperature,
                generators=generators,
            )
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = _rand_by_row(
                    x0.shape,
                    device=x0.device,
                    dtype=torch.get_default_dtype(),
                    generators=(
                        remasking_generators
                        if remasking_generators is not None
                        else generators
                    ),
                )
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -float("inf")

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -float("inf"))

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            update_diffusion_mtp_like_stats(
                stats, x, transfer_index, prompt.shape[1], gen_length,
                prefix_size=mtp_like_prefix_size, mask_id=mask_id,
            )
            x[transfer_index] = x0[transfer_index]

    if return_stats:
        return x, stats
    return x


def main():
    from transformers import AutoModel, AutoTokenizer

    device = 'cuda'

    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    out = generate(model, input_ids, steps=128, gen_length=128, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
    print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0])


if __name__ == '__main__':
    main()
