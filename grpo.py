import inspect
import math
from collections.abc import Mapping
from contextlib import nullcontext
from numbers import Integral, Real

import torch
import torch.nn.functional as F

from utils.models.forward import forward_logits, forward_selected_logits
from utils.generation.generate import (
    _mask_invalid_sampling_logits,
    generate,
    generate_monotonic_blocks,
)
from utils.runtime.rng import derive_seed, make_generators


def _record_timing(timing, name, *, replacing=None):
    if timing is None:
        return nullcontext()
    return timing.record(name, replacing=replacing)


def _should_synchronize_mtp_forward_rounds(schedule, model_family):
    return schedule == "adaptive" or model_family == "sdar"


def _logical_completion_generators(
    batch,
    prompt_ids,
    num_generations,
    *,
    base_seed,
    optimizer_step,
    repeat_id,
):
    """Build topology-independent RNG streams for every rollout row."""
    rng_context = (base_seed, optimizer_step, repeat_id)
    if not any(value is not None for value in rng_context):
        return None, None
    if any(value is None for value in rng_context):
        raise ValueError(
            "base_seed, optimizer_step, and repeat_id must be provided together"
        )

    sample_ids = batch.get("sample_ids")
    if sample_ids is None or len(sample_ids) != prompt_ids.shape[0]:
        raise ValueError("A stable sample_id is required for every rollout prompt")

    # Tensor.repeat(num_generations, 1) is generation-major: every prompt
    # appears once for generation 0, then once for generation 1, and so on.
    logical_rows = [
        (sample_id, generation_id)
        for generation_id in range(num_generations)
        for sample_id in sample_ids
    ]
    gumbel_seeds = [
        derive_seed(
            base_seed,
            stream="gumbel-v1",
            optimizer_step=optimizer_step,
            sample_id=sample_id,
            repeat_id=repeat_id,
            generation_id=generation_id,
        )
        for sample_id, generation_id in logical_rows
    ]
    remasking_seeds = [
        derive_seed(
            base_seed,
            stream="random-remask-v1",
            optimizer_step=optimizer_step,
            sample_id=sample_id,
            repeat_id=repeat_id,
            generation_id=generation_id,
        )
        for sample_id, generation_id in logical_rows
    ]
    return (
        make_generators(gumbel_seeds, prompt_ids.device),
        make_generators(remasking_seeds, prompt_ids.device),
    )


def _normalize_reward_output(reward_output, expected_size, device):
    """Normalize a reward tensor and optional component tensors."""
    components = None
    if isinstance(reward_output, tuple):
        if len(reward_output) != 2:
            raise ValueError(
                "Reward function tuple output must be (rewards, components)"
            )
        rewards, components = reward_output
    else:
        rewards = reward_output

    def normalize(value, name):
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value, device=device)
        value = value.detach().to(device=device, dtype=torch.float32).reshape(-1)
        if value.numel() != expected_size:
            raise ValueError(
                f"Reward component {name!r} has {value.numel()} values; "
                f"expected {expected_size}"
            )
        return value

    rewards = normalize(rewards, "total_reward")
    if components is None:
        components = {}
    elif not isinstance(components, Mapping):
        raise TypeError("Reward components must be a mapping")

    normalized = {}
    for name, value in components.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Reward component names must be non-empty strings")
        normalized[name] = normalize(value, name)
    normalized.setdefault("total_reward", rewards)
    return rewards, normalized


def _reward_function_accepts_components(reward_fn):
    try:
        signature = inspect.signature(reward_fn)
    except (TypeError, ValueError):
        return True
    return "return_components" in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


@torch.no_grad()
def sample(
    model,
    batch,
    tokenizer,
    device,
    reward_fn=None,
    num_generations=1,
    temperature=1.0,
    steps=256,
    gen_length=256,
    method="justgrpo",
    mtp_block_size=4,
    mtp_confidence="normalized_entropy",
    mtp_threshold=0.8,
    mask_id=126336,
    model_family="llada",
    base_seed=None,
    optimizer_step=None,
    repeat_id=None,
    selective_logits=False,
    timing=None,
):
    if model.training:
        raise ValueError("Rollout sampling requires model.eval()")
    if method not in (
        "justgrpo",
        "fixed_mtp",
        "ar_mtp",
    ):
        raise ValueError(f"Unknown sampling method: {method}")
    if method in ("fixed_mtp", "ar_mtp") and steps != gen_length:
        raise ValueError("MTP rollout requires steps == gen_length")

    prompts = tokenizer.apply_chat_template(
        [[{"role": "user", "content": problem}] for problem in batch["problems"]],
        add_generation_prompt=True,
        tokenize=False,
    )
    prompt_ids = tokenizer(prompts, return_tensors="pt", padding=True)["input_ids"].to(device)
    rollout_prompts = prompt_ids.repeat(num_generations, 1)
    sampling_vocab_size = len(tokenizer)
    generators, remasking_generators = _logical_completion_generators(
        batch,
        prompt_ids,
        num_generations,
        base_seed=base_seed,
        optimizer_step=optimizer_step,
        repeat_id=repeat_id,
    )

    with _record_timing(timing, "rollout_generation"):
        if method == "justgrpo":
            generated_ids = generate(
                model=model,
                prompt=rollout_prompts,
                steps=steps,
                gen_length=gen_length,
                temperature=temperature,
                block_length=1,
                mask_id=mask_id,
                generators=generators,
                remasking_generators=remasking_generators,
                model_family=model_family,
                selective_logits=selective_logits,
                sampling_vocab_size=sampling_vocab_size,
            )
            rollout = {"generated_ids": generated_ids}
        else:
            schedule = "fixed" if method == "fixed_mtp" else "adaptive"
            rollout = generate_monotonic_blocks(
                model=model,
                prompt=rollout_prompts,
                gen_length=gen_length,
                max_block_size=mtp_block_size,
                schedule=schedule,
                confidence_metric=mtp_confidence,
                confidence_threshold=mtp_threshold,
                temperature=temperature,
                mask_id=mask_id,
                model_family=model_family,
                selective_logits=selective_logits,
                generators=generators,
                synchronize_forward_rounds=_should_synchronize_mtp_forward_rounds(
                    schedule, model_family
                ),
                sampling_vocab_size=sampling_vocab_size,
            )
            generated_ids = rollout["generated_ids"]

    responses = tokenizer.batch_decode(
        generated_ids[:, prompt_ids.shape[1] :], skip_special_tokens=True
    )
    if reward_fn is None:
        raise ValueError("sample requires a reward_fn")
    reward_kwargs = (
        {"return_components": True}
        if _reward_function_accepts_components(reward_fn)
        else {}
    )
    reward_output = reward_fn(
        batch, responses, num_generations, device, **reward_kwargs
    )
    rewards, reward_components = _normalize_reward_output(
        reward_output, len(responses), device
    )
    rollout.update(
        {
            "prompt_len": prompt_ids.shape[1],
            "sampling_vocab_size": sampling_vocab_size,
            "rewards": rewards,
            "reward_components": reward_components,
        }
    )
    return rollout


def _as_nonnegative_finite_float(value, name):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must be a scalar")
        value = value.detach().item()
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def compute_pooled_loss_scale(global_valid, world_size, gen_length, gain=1.0):
    """Return the local scale for a globally pooled valid-completion loss.

    DDP and FSDP average gradients across ranks, so multiplying every rank's
    local loss sum by world_size / global_valid recovers a global sum divided
    by the global valid count. global_valid must already include every rank
    and every micro-batch in the optimizer step.
    """
    if isinstance(world_size, bool) or not isinstance(world_size, Integral):
        raise TypeError("world_size must be an integer")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if isinstance(gen_length, bool) or not isinstance(gen_length, Integral):
        raise TypeError("gen_length must be an integer")
    if gen_length <= 0:
        raise ValueError("gen_length must be positive")

    global_valid = _as_nonnegative_finite_float(global_valid, "global_valid")
    gain = _as_nonnegative_finite_float(gain, "gain")
    if global_valid == 0:
        return 0.0

    scale = gain * int(world_size) / (int(gen_length) * global_valid)
    if not math.isfinite(scale):
        raise ValueError("computed loss scale must be finite")
    return scale


def logprob_loss(
    model,
    inputs,
    loss_scale,
    eps=0.2,
    temperature=1.0,
    accelerator=None,
    gen_length=256,
    mask_id=126336,
    model_family="llada",
    selective_logits=False,
    timing=None,
):
    """Backpropagate local AR loss with one optimizer-step-level scale.

    loss_scale already includes gain, generation length, world size, and the
    optimizer-step pooled valid-completion denominator. No distributed
    normalization or metric collective is performed here.
    """
    advantages = inputs["advantages"]
    generated_ids = inputs["generated_ids"]
    prompt_len = inputs["prompt_len"]
    batch_size, device = advantages.shape[0], generated_ids.device
    prompt_ids = generated_ids[:, :prompt_len]
    completion_ids = generated_ids[:, prompt_len:]

    loss_scale = _as_nonnegative_finite_float(loss_scale, "loss_scale")
    raw_loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    raw_abs_loss_sum = torch.zeros((), dtype=torch.float64, device=device)

    with _record_timing(timing, "logprob_prefill"):
        for token_index in range(gen_length):
            # AR masking: the past is observed and the current/future suffix is masked.
            x = torch.cat(
                [
                    prompt_ids,
                    completion_ids[:, :token_index],
                    torch.full(
                        (batch_size, gen_length - token_index),
                        mask_id,
                        device=device,
                        dtype=generated_ids.dtype,
                    ),
                ],
                dim=1,
            )

            with torch.autocast(
                device_type=device.type,
                enabled=generated_ids.is_cuda,
                dtype=torch.bfloat16,
            ):
                if selective_logits:
                    token_logits = forward_selected_logits(
                        model,
                        x,
                        positions=prompt_len + token_index,
                        model_family=model_family,
                    ) / temperature
                else:
                    logits = (
                        forward_logits(model, x, model_family=model_family) / temperature
                    )
                    token_logits = logits[:, prompt_len + token_index, :]

            token_logits = _mask_invalid_sampling_logits(
                token_logits.float().clone(),
                mask_id=mask_id,
                sampling_vocab_size=inputs.get("sampling_vocab_size"),
                mask_mask_id=False,
            )
            log_probs = F.log_softmax(token_logits, dim=-1)
            token_log_probs = log_probs.gather(
                -1, completion_ids[:, token_index : token_index + 1]
            ).squeeze(-1)

            ratio = (token_log_probs - token_log_probs.detach()).exp()
            clipped_ratio = ratio.clamp(1 - eps, 1 + eps)
            loss = -torch.min(ratio * advantages, clipped_ratio * advantages)
            loss_sum = loss.sum()

            raw_loss_sum += loss_sum.detach().double()
            raw_abs_loss_sum += loss.detach().abs().sum().double()
            scaled_loss = loss_sum.mul(loss_scale)
            with _record_timing(
                timing,
                "backward",
                replacing="logprob_prefill",
            ):
                accelerator.backward(scaled_loss)

    return {
        "raw_loss_sum": raw_loss_sum,
        "raw_abs_loss_sum": raw_abs_loss_sum,
    }


def reconstruct_block_states(generated_ids, prompt_len, block_starts, mask_id=126336):
    completion_length = generated_ids.shape[1] - prompt_len
    completion_positions = torch.arange(
        completion_length, device=generated_ids.device
    ).unsqueeze(0)
    observed = completion_positions < block_starts.unsqueeze(1)
    completion = generated_ids[:, prompt_len:]
    masked_completion = torch.where(
        observed, completion, torch.full_like(completion, mask_id)
    )
    return torch.cat([generated_ids[:, :prompt_len], masked_completion], dim=1)


def block_token_log_probs(
    logits,
    completion_ids,
    prompt_len,
    block_starts,
    block_lengths,
    max_block_size,
    temperature=1.0,
    mask_id=126336,
    logits_are_selected=False,
    sampling_vocab_size=None,
):
    offsets = torch.arange(max_block_size, device=logits.device).unsqueeze(0)
    safe_starts = block_starts.clamp(min=0)
    positions = safe_starts.unsqueeze(1) + offsets
    safe_positions = positions.clamp(max=completion_ids.shape[1] - 1)
    batch_index = torch.arange(logits.shape[0], device=logits.device).unsqueeze(1)
    if logits_are_selected:
        selected_logits = (logits / temperature).clone()
    else:
        selected_logits = (
            logits[batch_index, prompt_len + safe_positions] / temperature
        ).clone()
    selected_logits = _mask_invalid_sampling_logits(
        selected_logits,
        mask_id=mask_id,
        sampling_vocab_size=sampling_vocab_size,
        mask_mask_id=True,
    )
    accepted = offsets < block_lengths.unsqueeze(1)
    target_tokens = completion_ids[batch_index, safe_positions]
    # Padding slots need no token score and can hold ids outside sampling support.
    target_tokens = torch.where(accepted, target_tokens, torch.zeros_like(target_tokens))
    log_probs = F.log_softmax(selected_logits.float(), dim=-1)
    token_log_probs = log_probs.gather(-1, target_tokens.unsqueeze(-1)).squeeze(-1)
    token_log_probs = torch.where(accepted, token_log_probs, torch.zeros_like(token_log_probs))
    return token_log_probs, accepted


def block_logprob_loss(
    model,
    inputs,
    loss_scale,
    max_block_size,
    eps=0.2,
    temperature=1.0,
    accelerator=None,
    gen_length=256,
    mask_id=126336,
    model_family="llada",
    selective_logits=False,
    timing=None,
):
    """Backpropagate block/MTP loss with the pooled optimizer-step scale."""
    advantages = inputs["advantages"]
    generated_ids = inputs["generated_ids"]
    prompt_len = inputs["prompt_len"]
    completion_ids = generated_ids[:, prompt_len:]
    block_starts = inputs["block_starts"]
    block_lengths = inputs["block_lengths"]
    rollout_input_length = inputs.get("rollout_input_length", generated_ids.shape[1])
    if (
        isinstance(rollout_input_length, bool)
        or not isinstance(rollout_input_length, Integral)
        or rollout_input_length < generated_ids.shape[1]
    ):
        raise ValueError("rollout_input_length must cover the generated sequence")
    padding_length = int(rollout_input_length) - generated_ids.shape[1]

    loss_scale = _as_nonnegative_finite_float(loss_scale, "loss_scale")
    raw_loss_sum = torch.zeros((), dtype=torch.float64, device=generated_ids.device)
    raw_abs_loss_sum = torch.zeros((), dtype=torch.float64, device=generated_ids.device)

    # All ranks must execute the same number of model forwards for FSDP/DDP.
    # This MAX is synchronization only; loss normalization stays exclusively at
    # the optimizer-step level in the caller.
    global_rounds = inputs["num_blocks"].max().clone()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(global_rounds, op=torch.distributed.ReduceOp.MAX)
    global_rounds = int(global_rounds.item())

    prefill_timing = timing if global_rounds > 0 else None
    with _record_timing(prefill_timing, "logprob_prefill"):
        for block_index in range(global_rounds):
            starts = block_starts[:, block_index]
            lengths = block_lengths[:, block_index]
            x = reconstruct_block_states(
                generated_ids, prompt_len, starts, mask_id=mask_id
            )
            # Cached SDAR attends masked padding in its last native block.
            # Retain that context while scoring the real completion tokens.
            if padding_length:
                x = F.pad(x, (0, padding_length), value=mask_id)

            with torch.autocast(
                device_type=generated_ids.device.type,
                enabled=generated_ids.is_cuda,
                dtype=torch.bfloat16,
            ):
                if selective_logits:
                    offsets = torch.arange(
                        max_block_size, device=generated_ids.device
                    ).unsqueeze(0)
                    safe_starts = starts.clamp(min=0)
                    safe_positions = (
                        safe_starts.unsqueeze(1) + offsets
                    ).clamp(max=completion_ids.shape[1] - 1)
                    logits = forward_selected_logits(
                        model,
                        x,
                        positions=prompt_len + safe_positions,
                        model_family=model_family,
                    )
                else:
                    logits = forward_logits(model, x, model_family=model_family)

            token_log_probs, accepted = block_token_log_probs(
                logits=logits,
                completion_ids=completion_ids,
                prompt_len=prompt_len,
                block_starts=starts,
                block_lengths=lengths,
                max_block_size=max_block_size,
                temperature=temperature,
                mask_id=mask_id,
                logits_are_selected=selective_logits,
                sampling_vocab_size=inputs.get("sampling_vocab_size"),
            )
            block_log_probs = (token_log_probs * accepted).sum(dim=1)
            ratio = (block_log_probs - block_log_probs.detach()).exp()
            clipped_ratio = ratio.clamp(1 - eps, 1 + eps)
            loss = -torch.min(ratio * advantages, clipped_ratio * advantages)

            # Inactive padding rows only exist to align distributed forward counts.
            # Excluding them keeps both the scalar loss and detached numerators tied
            # to real logical completions.
            active = lengths > 0
            loss = loss * active
            loss_sum = loss.sum()
            raw_loss_sum += loss_sum.detach().double()
            raw_abs_loss_sum += loss.detach().abs().sum().double()
            scaled_loss = loss_sum.mul(loss_scale)
            with _record_timing(
                timing,
                "backward",
                replacing="logprob_prefill",
            ):
                accelerator.backward(scaled_loss)

    return {
        "raw_loss_sum": raw_loss_sum,
        "raw_abs_loss_sum": raw_abs_loss_sum,
        "loss_forward_rounds": global_rounds,
    }




def compute_group_advantages(rewards, group_size):
    if group_size < 2:
        raise ValueError("group_size must be at least 2")
    if rewards.numel() % group_size != 0:
        raise ValueError("reward count must be divisible by group_size")
    mean = rewards.view(group_size, -1).mean(dim=0).repeat(group_size)
    std = rewards.view(group_size, -1).std(dim=0).repeat(group_size)
    advantages = (rewards - mean) / (std + 1e-4)
    if not torch.isfinite(advantages).all():
        raise ValueError("group advantages contain non-finite values")
    return advantages
