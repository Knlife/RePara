import importlib.util
from pathlib import Path

import torch
from transformers.cache_utils import DynamicCache

from utils.models.forward import _get_model_config
from utils.generation.generate import _mask_invalid_sampling_logits


_SDAR_GENERATE_MODULE = None


def _load_sdar_generate_module():
    global _SDAR_GENERATE_MODULE
    if _SDAR_GENERATE_MODULE is not None:
        return _SDAR_GENERATE_MODULE

    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "third-party" / "sdar" / "generate.py"
    spec = importlib.util.spec_from_file_location("sdar_native_generate", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load SDAR native generate module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _SDAR_GENERATE_MODULE = module
    return module


def init_sdar_mtp_stats(block_length, device):
    return {
        "length_counts": torch.zeros(block_length, dtype=torch.long, device=device),
        "forced_singletons": torch.zeros((), dtype=torch.long, device=device),
        "confidence_sum": torch.zeros((), dtype=torch.float64, device=device),
        "confidence_count": torch.zeros((), dtype=torch.long, device=device),
        "forward_rounds": torch.zeros((), dtype=torch.long, device=device),
        "prefix_pair_count": torch.zeros((), dtype=torch.long, device=device),
        "prefix_pair_opportunities": torch.zeros((), dtype=torch.long, device=device),
        "prefix_transfer_sum": torch.zeros((), dtype=torch.long, device=device),
        "prefix_transfer_count": torch.zeros((), dtype=torch.long, device=device),
    }


def _update_prefix_stats(stats, x, transfer_index, prompt_len, gen_length,
                         prefix_size=2, mask_id=126336):
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


def _update_transfer_stats(stats, transfer_index, confidence, block_length):
    transferred = transfer_index.sum(dim=1)
    active = transferred > 0
    for length in range(1, block_length + 1):
        stats["length_counts"][length - 1] += (transferred[active] == length).sum()
    if confidence is not None and active.any():
        selected_confidence = confidence[transfer_index]
        stats["confidence_sum"] += selected_confidence.double().sum()
        stats["confidence_count"] += selected_confidence.numel()


def _stopping_criteria_met(x, prompt_len, gen_length, stopping_criteria_idx):
    if stopping_criteria_idx is None:
        return False
    stop_ids = torch.as_tensor(stopping_criteria_idx, device=x.device)
    completion = x[:, prompt_len:prompt_len + gen_length]
    return torch.isin(completion, stop_ids).any().item()


def _select_transfer_index(
        *,
        cur_x,
        mask_index,
        token_confidence,
        transfer_count,
        remasking_strategy,
        confidence_threshold,
        eb_threshold,
        stats,
    ):
    confidence = None
    transfer_index = torch.zeros_like(cur_x, dtype=torch.bool)
    if remasking_strategy == "sequential":
        for row in range(cur_x.shape[0]):
            masked = mask_index[row].nonzero(as_tuple=True)[0]
            count = min(transfer_count, masked.numel())
            transfer_index[row, masked[:count]] = True
    elif remasking_strategy == "low_confidence_static":
        confidence = torch.where(mask_index, token_confidence, -torch.inf)
        for row in range(confidence.shape[0]):
            masked = mask_index[row].nonzero(as_tuple=True)[0]
            count = min(transfer_count, masked.numel())
            idx = torch.topk(confidence[row, masked], count).indices
            transfer_index[row, masked[idx]] = True
    elif remasking_strategy == "low_confidence_dynamic":
        confidence = torch.where(mask_index, token_confidence, -torch.inf)
        for row in range(confidence.shape[0]):
            masked = mask_index[row].nonzero(as_tuple=True)[0]
            if masked.numel() == 0:
                continue
            count = min(transfer_count, masked.numel())
            high_conf_mask = confidence[row] > confidence_threshold
            if high_conf_mask.sum() >= count:
                transfer_index[row] = high_conf_mask
            else:
                if count == 1:
                    stats["forced_singletons"] += 1
                idx = torch.topk(confidence[row, masked], count).indices
                transfer_index[row, masked[idx]] = True
    elif remasking_strategy == "entropy_bounded":
        eps = 1e-12
        token_entropy = -(
            token_confidence.clamp_min(eps) * token_confidence.clamp_min(eps).log()
        )
        entropies = torch.where(mask_index, token_entropy, torch.inf)
        ent_sorted, order = torch.sort(entropies, dim=1, descending=False)
        cumsum = torch.cumsum(ent_sorted, dim=1)
        for row in range(token_confidence.shape[0]):
            remaining = int(mask_index[row].sum().item())
            if remaining == 0:
                continue
            k = torch.searchsorted(
                cumsum[row],
                torch.tensor(eb_threshold, device=token_confidence.device),
                right=False,
            ).item()
            k = max(1, min(k, remaining))
            transfer_index[row, order[row, :k]] = True
    else:
        raise ValueError(f"Unknown remasking strategy: {remasking_strategy}")
    return transfer_index, confidence


def _sample_tokens(logits, *, temperature, top_k, top_p, sample_fn,
                   mask_id=None, sampling_vocab_size=None):
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if sampling_vocab_size is not None:
        logits = _mask_invalid_sampling_logits(
            logits.clone(), mask_id=mask_id, sampling_vocab_size=sampling_vocab_size,
        )
    if temperature == 0:
        probs = torch.softmax(logits, dim=-1)
        token_prob, token = probs.max(dim=-1)
        return token, token_prob
    return sample_fn(
        logits,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )


@torch.no_grad()
def block_diffusion_generate_with_stats(
        model,
        prompt,
        mask_id,
        gen_length=128,
        block_length=8,
        denoising_steps=8,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        remasking_strategy='low_confidence_dynamic',
        confidence_threshold=0.85,
        eb_threshold=None,
        stopping_criteria_idx=None,
        mtp_like_prefix_size=2,
        sampling_vocab_size=None,
    ):
    """Run upstream SDAR native decoding and collect RePara evaluation stats.

    This mirrors the pinned upstream `third-party/sdar/generate.py` loop in the
    main repository so the submodule can stay at an official, fetchable commit.
    """
    declared_block_size = getattr(_get_model_config(model), "block_size", None)
    if declared_block_size is not None and block_length != declared_block_size:
        raise ValueError(
            "SDAR decoding block_length must equal the checkpoint's declared "
            f"block_size ({declared_block_size}), got {block_length}"
        )
    sdar_generate = _load_sdar_generate_module()
    sample_with_temperature_topk_topp = sdar_generate.sample_with_temperature_topk_topp
    get_num_transfer_tokens = sdar_generate.get_num_transfer_tokens

    model.eval()
    input_ids = prompt["input_ids"]
    batch_size, prompt_length = input_ids.shape
    device = input_ids.device
    past_key_values = DynamicCache()
    stats = init_sdar_mtp_stats(block_length, device)

    num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
    total_length = num_blocks * block_length

    block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=device))
    block_diffusion_attention_mask = (
        block_mask.repeat_interleave(block_length, dim=0)
        .repeat_interleave(block_length, dim=1)
        .unsqueeze(0)
    )
    position_ids = torch.arange(total_length, device=device).unsqueeze(0)

    x = torch.full(
        (batch_size, total_length), mask_id, dtype=torch.long, device=device
    )
    x[:, :prompt_length] = input_ids
    prefill_blocks = prompt_length // block_length
    prefill_length = prefill_blocks * block_length

    if prefill_length > 0:
        cur_x = x[:, :prefill_length]
        cur_attn_mask = block_diffusion_attention_mask[
            :, :prefill_length, :prefill_length
        ]
        cur_position_ids = position_ids[:, :prefill_length]
        model(
            cur_x,
            attention_mask=cur_attn_mask,
            position_ids=cur_position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            store_kv=True,
        )

    num_transfer_tokens = get_num_transfer_tokens(block_length, denoising_steps).to(device)
    absolute_positions = torch.arange(total_length, device=device)

    for num_block in range(prefill_blocks, num_blocks):
        block_start = num_block * block_length
        block_end = (num_block + 1) * block_length
        cur_x = x[:, block_start:block_end].clone()
        cur_attn_mask = block_diffusion_attention_mask[
            :, block_start:block_end, :block_end
        ]
        cur_position_ids = position_ids[:, block_start:block_end]

        for step in range(denoising_steps + 1):
            mask_index = cur_x == mask_id
            if mask_index.sum() == 0:
                model(
                    cur_x,
                    attention_mask=cur_attn_mask,
                    position_ids=cur_position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    store_kv=True,
                )
                break

            logits = model(
                cur_x,
                attention_mask=cur_attn_mask,
                position_ids=cur_position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                store_kv=False,
            ).logits
            stats["forward_rounds"] += mask_index.any(dim=1).sum()

            x0, x0_p = _sample_tokens(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                sample_fn=sample_with_temperature_topk_topp,
                mask_id=mask_id,
                sampling_vocab_size=sampling_vocab_size,
            )

            transfer_count = int(num_transfer_tokens[step].item())
            transfer_index, confidence = _select_transfer_index(
                cur_x=cur_x,
                mask_index=mask_index,
                token_confidence=x0_p,
                transfer_count=transfer_count,
                remasking_strategy=remasking_strategy,
                confidence_threshold=confidence_threshold,
                eb_threshold=eb_threshold,
                stats=stats,
            )

            real_completion_positions = (
                (absolute_positions[block_start:block_end] >= prompt_length)
                & (absolute_positions[block_start:block_end] < prompt_length + gen_length)
            )
            real_transfer_index = transfer_index & real_completion_positions.unsqueeze(0)
            _update_transfer_stats(
                stats, real_transfer_index, confidence, block_length
            )

            full_transfer_index = torch.zeros_like(x, dtype=torch.bool)
            full_transfer_index[:, block_start:block_end] = real_transfer_index
            current_x = x.clone()
            current_x[:, block_start:block_end] = cur_x
            _update_prefix_stats(
                stats,
                current_x,
                full_transfer_index,
                prompt_length,
                gen_length,
                prefix_size=mtp_like_prefix_size,
                mask_id=mask_id,
            )

            cur_x[transfer_index] = x0[transfer_index]

        x[:, block_start:block_end] = cur_x
        if _stopping_criteria_met(
            x, prompt_length, gen_length, stopping_criteria_idx
        ):
            break

    return x[:, :prompt_length + gen_length], stats
