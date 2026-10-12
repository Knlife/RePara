import argparse
import ast
import glob
import json
import os
from dataclasses import asdict, replace
from datetime import timedelta

import torch
import torch.distributed as torch_dist
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from data.math import collate_fn_gsm8k, collate_fn_math, extract_answer_gsm8k
from datasets import load_dataset
from utils.configuration.config import EvalConfig, load_config, snapshot_config
import utils.runtime.distributed as train_dist
from utils.models.forward import install_position_selective_output_head
from utils.models.sdar_config import configure_sdar_native_block_size
from utils.generation.generate import generate, generate_monotonic_blocks
from utils.evaluation.grader import math_equal
from utils.evaluation.parser import extract_answer

from utils.generation.sdar_native_generate import (
    block_diffusion_generate_with_stats,
    init_sdar_mtp_stats,
)


def _raw_dist_ready():
    return torch_dist.is_available() and torch_dist.is_initialized()


def _accelerator():
    return train_dist.get_accelerator()


def _dist_rank():
    if _raw_dist_ready():
        return torch_dist.get_rank()
    accelerator = _accelerator()
    return accelerator.process_index if accelerator is not None else 0


def _dist_world_size():
    if _raw_dist_ready():
        return torch_dist.get_world_size()
    accelerator = _accelerator()
    return accelerator.num_processes if accelerator is not None else 1


def _dist_barrier():
    if _raw_dist_ready():
        torch_dist.barrier()
    else:
        accelerator = _accelerator()
        if accelerator is not None:
            accelerator.wait_for_everyone()


def _dist_all_reduce_(value):
    if _raw_dist_ready():
        torch_dist.all_reduce(value, op=torch_dist.ReduceOp.SUM)
    else:
        accelerator = _accelerator()
        if accelerator is not None and accelerator.num_processes > 1:
            value.copy_(accelerator.reduce(value, reduction="sum"))
    return value


def _dist_all_gather_object(value):
    if _raw_dist_ready():
        gathered = [None] * torch_dist.get_world_size()
        torch_dist.all_gather_object(gathered, value)
        return gathered

    accelerator = _accelerator()
    if accelerator is None or accelerator.num_processes == 1:
        return [value]

    # Wrapping the value keeps one list per rank with Accelerate's flattened
    # gather_object implementation.
    from accelerate.utils import gather_object
    return gather_object([value])


def _synchronize_rank_error(error, context):
    """Make a rank-local failure visible on every rank before continuing."""
    errors = _dist_all_gather_object(error)
    first_error = next((item for item in errors if item), None)
    if first_error is not None:
        raise RuntimeError(f"{context}: {first_error}")


def _sweep_values(value, name):
    values = value if isinstance(value, list) else [value]
    if not values:
        raise ValueError(f"{name} sweep must not be empty")
    if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in values):
        raise ValueError(f"{name} values must be positive integers")
    return values


def expand_eval_configs(config: EvalConfig) -> list[EvalConfig]:
    gen_lengths = _sweep_values(config.gen_length, "gen_length")
    mtp_sizes = _sweep_values(config.mtp_size, "mtp_size")
    if config.generation_method != "diffusion" and len(mtp_sizes) != 1:
        raise ValueError("mtp_size sweeps currently require generation_method=diffusion")

    runs = []
    for gen_length in gen_lengths:
        for mtp_size in mtp_sizes:
            if gen_length % mtp_size != 0:
                raise ValueError(
                    f"gen_length={gen_length} must be divisible by mtp_size={mtp_size}"
                )
            if config.generation_method == "diffusion":
                if gen_length % config.block_length != 0:
                    raise ValueError(
                        f"gen_length={gen_length} must be divisible by "
                        f"block_length={config.block_length}"
                    )
                if config.block_length % mtp_size != 0:
                    raise ValueError(
                        f"block_length={config.block_length} must be divisible by "
                        f"mtp_size={mtp_size}"
                    )
            runs.append(
                replace(config, gen_length=gen_length, mtp_size=mtp_size)
            )
    return runs


# ---- Initialization ----

def init_dist():
    if not _raw_dist_ready():
        torch_dist.init_process_group(backend="nccl", timeout=timedelta(hours=1))
    local_rank_value = os.environ.get("LOCAL_RANK")
    if local_rank_value is None:
        device_count = torch.cuda.device_count()
        if device_count <= 0:
            raise RuntimeError("Evaluation requires at least one CUDA device")
        local_rank = _dist_rank() % device_count
    else:
        local_rank = int(local_rank_value)
    torch.cuda.set_device(local_rank)
    return torch.device(f"cuda:{local_rank}")


def eval_setting_name(args):
    if args.generation_method == "diffusion":
        return f"{args.task}-diffusion-len{args.gen_length}-blk{args.block_length}-step{args.steps}"
    if args.generation_method == "sdar_native":
        threshold = str(args.sdar_confidence_threshold).replace(".", "p")
        return (
            f"{args.task}-sdar-native-len{args.gen_length}"
            f"-blk{args.sdar_block_length}-denoise{args.sdar_denoising_steps}"
            f"-{args.sdar_remasking_strategy}-thr{threshold}"
        )
    threshold = str(args.mtp_threshold).replace(".", "p")
    return (
        f"{args.task}-{args.generation_method}-len{args.gen_length}"
        f"-mtpblk{args.mtp_block_size}-{args.mtp_confidence}-thr{threshold}"
    )


def save_log(args, metrics):
    out_dir = os.path.join(args.output, eval_setting_name(args))
    os.makedirs(out_dir, exist_ok=True)
    effective_config = asdict(args)
    effective_config["steps"] = args.steps
    config_path = getattr(args, "_config_path", None)
    if config_path is not None:
        snapshot_config(
            out_dir,
            source_path=config_path,
            section="eval",
            effective_config=effective_config,
            overrides=getattr(args, "_config_overrides", {}),
        )
    log = {
        **effective_config,
        "local_files_only": not args.allow_remote_files,
        **getattr(args, "_online_metadata", {}),
        **metrics,
    }
    with open(
        os.path.join(out_dir, "evaluation_log.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(log, f, indent=2)
    print(f"Log saved to {out_dir}/evaluation_log.json")


# ---- LoRA ----

def resolve_adapter_base_path(adapter_dir):
    config_path = os.path.join(adapter_dir, "adapter_config.json")
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    base_path = config.get("base_model_name_or_path")
    if not isinstance(base_path, str) or not base_path:
        raise ValueError(
            f"LoRA adapter does not record base_model_name_or_path: {config_path}"
        )
    return base_path


def resolve_mask_id(args, model, tokenizer):
    if args.mask_id is not None:
        return args.mask_id

    for value in (
        getattr(getattr(model, "config", None), "mask_token_id", None),
        getattr(tokenizer, "mask_token_id", None),
    ):
        if value is not None:
            return int(value)

    if args.model_family == "llada":
        return 126336
    if args.model_family == "sdar":
        return 151669

    raise ValueError(
        f"Could not resolve mask token id for model_family={args.model_family!r}. "
        "Set --mask_id explicitly or add mask_token_id to the model config/tokenizer."
    )


def merge_lora(model, adapter_dir):
    # Fold the LoRA update (scaling * B @ A) into the base weights, computing the
    # update in fp32 for numerical accuracy.
    from safetensors.torch import load_file

    with open(os.path.join(adapter_dir, "adapter_config.json")) as f:
        cfg = json.load(f)
    scaling = cfg["lora_alpha"] / cfg["r"]
    state = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
    for key in [k for k in state if ".lora_A." in k]:
        target = key.split(".lora_A.")[0].removeprefix("base_model.model.")
        weight = model.get_submodule(target).weight
        delta = (state[key.replace(".lora_A.", ".lora_B.")].float() @ state[key].float()) * scaling
        weight.data += delta.to(weight.dtype)


def generate_eval(model, prompt_ids, args):
    if args.generation_method == "sdar_native":
        return block_diffusion_generate_with_stats(
            model=model,
            prompt={"input_ids": prompt_ids},
            mask_id=args.mask_id,
            gen_length=args.gen_length,
            block_length=args.sdar_block_length,
            denoising_steps=args.sdar_denoising_steps,
            temperature=args.sdar_temperature,
            top_k=args.sdar_top_k,
            top_p=args.sdar_top_p,
            remasking_strategy=args.sdar_remasking_strategy,
            confidence_threshold=args.sdar_confidence_threshold,
            eb_threshold=args.sdar_eb_threshold,
            stopping_criteria_idx=args.sdar_stopping_criteria_idx,
            mtp_like_prefix_size=args.mtp_like_prefix_size,
            sampling_vocab_size=getattr(args, "_sampling_vocab_size", None),
        )

    if args.generation_method == "diffusion":
        if args.report_diffusion_mtp_like_stats:
            return generate(
                model=model, prompt=prompt_ids, steps=args.steps,
                gen_length=args.gen_length, block_length=args.block_length,
                mask_id=args.mask_id, return_stats=True,
                temperature=args.temperature,
                mtp_like_prefix_size=args.mtp_like_prefix_size,
                model_family=args.model_family,
                sampling_vocab_size=getattr(args, "_sampling_vocab_size", None),
            )
        return generate(
            model=model, prompt=prompt_ids, steps=args.steps,
            gen_length=args.gen_length, block_length=args.block_length,
            mask_id=args.mask_id, model_family=args.model_family,
            temperature=args.temperature,
            sampling_vocab_size=getattr(args, "_sampling_vocab_size", None),
        ), None
    # force decode conseciutive k tokens
    schedule = "fixed" if args.generation_method == "fixed_mtp" else "adaptive"
    rollout = generate_monotonic_blocks(
        model=model,
        prompt=prompt_ids,
        gen_length=args.gen_length,
        max_block_size=args.mtp_block_size,
        schedule=schedule,
        confidence_metric=args.mtp_confidence,
        confidence_threshold=args.mtp_threshold,
        temperature=getattr(args, "temperature", 0.0),
        mask_id=args.mask_id,
        model_family=args.model_family,
        selective_logits=args.selective_logits,
        synchronize_forward_rounds=True,
        sampling_vocab_size=getattr(args, "_sampling_vocab_size", None),
    )
    return rollout["generated_ids"], rollout["rollout_stats"]


def init_mtp_stats(args, device):
    if args.generation_method == "sdar_native":
        return init_sdar_mtp_stats(args.sdar_block_length, device)
    if args.generation_method == "diffusion":
        if not args.report_diffusion_mtp_like_stats:
            return None
        return {
            "prefix_pair_count": torch.zeros((), device=device, dtype=torch.long),
            "prefix_pair_opportunities": torch.zeros((), device=device, dtype=torch.long),
            "prefix_transfer_sum": torch.zeros((), device=device, dtype=torch.long),
            "prefix_transfer_count": torch.zeros((), device=device, dtype=torch.long),
            "forward_rounds": torch.zeros((), device=device, dtype=torch.long),
        }
    return {
        "length_counts": torch.zeros(args.mtp_block_size, device=device, dtype=torch.long),
        "forced_singletons": torch.zeros((), device=device, dtype=torch.long),
        "confidence_sum": torch.zeros((), device=device, dtype=torch.float64),
        "confidence_count": torch.zeros((), device=device, dtype=torch.long),
        "forward_rounds": torch.zeros((), device=device, dtype=torch.long),
    }


def update_mtp_stats(total_stats, batch_stats):
    if total_stats is None or batch_stats is None:
        return
    for key in total_stats:
        total_stats[key] += batch_stats[key]


def finalize_mtp_stats(args, stats, total_samples):
    if stats is None:
        return {}

    for value in stats.values():
        _dist_all_reduce_(value)

    if args.generation_method == "diffusion":
        opportunities = stats["prefix_pair_opportunities"].clamp_min(1)
        transfer_count = stats["prefix_transfer_count"].clamp_min(1)
        total_samples = max(total_samples, 1)
        return {
            "diffusion_mtp_like/prefix_size": args.mtp_like_prefix_size,
            "diffusion_mtp_like/consecutive_prefix_count": int(stats["prefix_pair_count"].item()),
            "diffusion_mtp_like/consecutive_prefix_opportunities": int(stats["prefix_pair_opportunities"].item()),
            "diffusion_mtp_like/consecutive_prefix_fraction": (stats["prefix_pair_count"].float() / opportunities).item(),
            "diffusion_mtp_like/mean_prefix_transfer": (stats["prefix_transfer_sum"].float() / transfer_count).item(),
            "diffusion_mtp_like/forward_rounds_total": int(stats["forward_rounds"].item()),
            "diffusion_mtp_like/forward_rounds_per_sample": stats["forward_rounds"].item() / total_samples,
        }

    if args.generation_method == "sdar_native":
        opportunities = stats["prefix_pair_opportunities"].clamp_min(1)
        transfer_count = stats["prefix_transfer_count"].clamp_min(1)
        total_samples = max(total_samples, 1)
        prefix_metrics = {
            "sdar_mtp_like/prefix_size": args.mtp_like_prefix_size,
            "sdar_mtp_like/consecutive_prefix_count": int(stats["prefix_pair_count"].item()),
            "sdar_mtp_like/consecutive_prefix_opportunities": int(stats["prefix_pair_opportunities"].item()),
            "sdar_mtp_like/consecutive_prefix_fraction": (stats["prefix_pair_count"].float() / opportunities).item(),
            "sdar_mtp_like/mean_prefix_transfer": (stats["prefix_transfer_sum"].float() / transfer_count).item(),
            "sdar_mtp_like/forward_rounds_total": int(stats["forward_rounds"].item()),
            "sdar_mtp_like/forward_rounds_per_sample": stats["forward_rounds"].item() / total_samples,
        }
    else:
        prefix_metrics = {}

    length_counts = stats["length_counts"]
    total_blocks = length_counts.sum().clamp_min(1)
    max_block_size = args.sdar_block_length if args.generation_method == "sdar_native" else args.mtp_block_size
    block_lengths = torch.arange(1, max_block_size + 1, device=length_counts.device)
    mean_block_length = (length_counts * block_lengths).sum().float() / total_blocks
    singleton_fraction = length_counts[0].float() / total_blocks
    confidence_mean = stats["confidence_sum"] / stats["confidence_count"].clamp_min(1)
    total_samples = max(total_samples, 1)

    metrics = {
        "mtp/mean_block_length": mean_block_length.item(),
        "mtp/singleton_fraction": singleton_fraction.item(),
        "mtp/forward_rounds_total": int(stats["forward_rounds"].item()),
        "mtp/forward_rounds_per_sample": stats["forward_rounds"].item() / total_samples,
        "mtp/forced_singletons": int(stats["forced_singletons"].item()),
        "mtp/forced_singletons_per_sample": stats["forced_singletons"].item() / total_samples,
    }
    metrics.update(prefix_metrics)
    if args.generation_method in ("ar_mtp", "sdar_native"):
        metrics["mtp/confidence_mean"] = confidence_mean.item()
    for idx, value in enumerate(length_counts, start=1):
        metrics[f"mtp/block_length_{idx}_count"] = int(value.item())
    return metrics


# ---- Math evaluation (GSM8K / MATH500) ----

def load_eval_dataset(
    pattern, *, dataset_name, dataset_config=None, allow_remote=False, split_name="test"
):
    matches = sorted(glob.glob(pattern))
    if matches:
        dataset = load_dataset(
            "parquet", data_files={split_name: matches}, split=split_name
        )
        return dataset.with_format("torch")

    if not allow_remote:
        raise FileNotFoundError(
            f"No local parquet files matched: {pattern}. "
            "Pass --allow_remote_files to load the dataset from the Hub."
        )
    dataset_args = [dataset_name]
    if dataset_config is not None:
        dataset_args.append(dataset_config)
    return load_dataset(*dataset_args, split=split_name).with_format("torch")


def eval_math(model, tokenizer, device, args):
    load_error = None
    ds = None
    collate_fn = None
    try:
        if args.task == "gsm8k":
            ds = load_eval_dataset(
                args.gsm8k_data_files,
                dataset_name="openai/gsm8k",
                dataset_config="main",
                allow_remote=args.allow_remote_files,
            )
            collate_fn = collate_fn_gsm8k
        else:
            ds = load_eval_dataset(
                args.math500_data_files,
                dataset_name="ankner/math-500",
                allow_remote=args.allow_remote_files,
            )
            collate_fn = collate_fn_math
    except BaseException as exc:
        load_error = f"{type(exc).__name__}: {exc}"
    _synchronize_rank_error(load_error, f"{args.task} dataset loading failed")

    rank = _dist_rank()
    world_size = _dist_world_size()
    setup_error = None
    local_batches = None
    mtp_stats = None
    try:
        ds = ds.select(range(rank, len(ds), world_size))
        # Evaluation is already distributed across ranks. Keeping the local
        # loader single-process avoids an implicit fork from a CUDA/filelock-
        # heavy evaluator and makes rank-local failures deterministic.
        dl = DataLoader(
            ds,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True,
            shuffle=False,
        )
        # Materialize the small benchmark batches so every rank can execute
        # the same number of synchronization rounds even when sharding leaves
        # one rank with one fewer batch.
        local_batches = list(dl)
        mtp_stats = init_mtp_stats(args, device)
    except BaseException as exc:
        setup_error = f"{type(exc).__name__}: {exc}"
    _synchronize_rank_error(setup_error, f"{args.task} evaluator setup failed")

    counts = torch.tensor([0, 0], device=device)
    batch_count = max(_dist_all_gather_object(len(local_batches)))
    # A shorter shard still performs a real model forward for every collective
    # round.  Its result is discarded, preventing FSDP/DDP from waiting on a
    # rank that skipped the batch.
    dummy_size = int(args.batch_size)
    dummy_batch = {
        "problems": ["Return the number zero."] * dummy_size,
        "answers": ["0"] * dummy_size,
    }
    pbar = tqdm(total=batch_count, disable=rank != 0)
    for batch_index in range(batch_count):
        is_padding = batch_index >= len(local_batches)
        batch = dummy_batch if is_padding else local_batches[batch_index]
        batch_error = None
        try:
            msgs = [[{"role": "user", "content": p}] for p in batch["problems"]]
            prompts = tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False,
            )
            prompt_ids = tokenizer(
                prompts, return_tensors="pt", padding=True,
            )["input_ids"].to(device)

            gen_ids, batch_mtp_stats = generate_eval(model, prompt_ids, args)
            if not is_padding:
                update_mtp_stats(mtp_stats, batch_mtp_stats)
                responses = tokenizer.batch_decode(
                    gen_ids[:, prompt_ids.shape[1]:], skip_special_tokens=True,
                )

                for ans, res in zip(batch["answers"], responses):
                    counts[1] += 1
                    if args.task == "gsm8k":
                        correct = math_equal(
                            extract_answer_gsm8k(ans), extract_answer(res)
                        )
                    else:
                        correct = math_equal(
                            extract_answer(ans), extract_answer(res), timeout=True,
                        )
                    if correct:
                        counts[0] += 1

                if rank == 0:
                    acc = counts[0] / max(counts[1], 1)
                    pbar.set_description(f"acc: {acc.item() * 100:.2f}%")
        except BaseException as exc:
            batch_error = f"{type(exc).__name__}: {exc}"
        _synchronize_rank_error(
            batch_error, f"{args.task} batch {batch_index} failed"
        )
        if rank == 0:
            pbar.update(1)
    pbar.close()

    _dist_all_reduce_(counts)
    mtp_metrics = finalize_mtp_stats(args, mtp_stats, int(counts[1].item()))
    metrics = None
    postprocess_error = None
    if rank == 0:
        try:
            total = int(counts[1].item())
            correct = int(counts[0].item())
            acc = correct / max(total, 1)
            print(f"\n{args.task} Accuracy: {correct}/{total} = {acc * 100:.2f}%")
            metrics = {
                "accuracy": acc,
                "correct": correct,
                "total": total,
            }
            metrics.update(mtp_metrics)
            save_log(args, metrics)
        except BaseException as exc:
            postprocess_error = f"{type(exc).__name__}: {exc}"
    _synchronize_rank_error(postprocess_error, f"{args.task} evaluation failed")
    return metrics


# ---- Code evaluation (HumanEval / MBPP) ----

def _format_humaneval(problems):
    formatted = {}
    for tid, p in problems.items():
        formatted[tid] = dict(p)
        formatted[tid]["prompt"] = (
            "You are an expert Python programmer. Your task is to complete the "
            f"implementation of a function named `{p['entry_point']}`.\n\n"
            f"Here is the function to complete:\n```python\n{p['prompt'].rstrip()}\n```\n"
        )
    return formatted


def _format_mbpp_prompt(ex):
    func_name = ex["test_list"][0].split(" ")[1].split("(")[0]
    tests_str = "\n".join(f"    {t}" for t in ex["test_list"])
    try:
        tree = ast.parse(ex["test_list"][0].strip())
        n_args = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == func_name:
                n_args = len(node.args)
                break
    except Exception:
        n_args = 2
    params = ", ".join(f"input_param_{i + 1}" for i in range(n_args))
    return (
        "You are an expert Python programmer. Your task is to complete the "
        f"implementation of a function named `{func_name}`.\n\n"
        f"** TARGET FUNCTION **\n{ex['text']}\n\n"
        "** UNIT TESTS **\n"
        f"Your code should pass unit tests like:\n{tests_str}\n\n"
        "Here is the function to complete:\n"
        f"```python\ndef {func_name}({params}):\n"
        f"    \"\"\"{ex['text']}\n    \"\"\"\n```\n"
    )


def _load_code_tasks(args):
    from utils.evaluation.code_exec import read_problems, stream_jsonl
    if args.task == "humaneval":
        return _format_humaneval(read_problems(args.humaneval_data_file))
    if args.task == "mbpp":
        examples = list(stream_jsonl(args.mbpp_data_file))
        return {
            ex["task_id"]: {"task_id": ex["task_id"], "prompt": _format_mbpp_prompt(ex)}
            for ex in examples[10:510]
        }
    raise ValueError(f"Unsupported benchmark code task: {args.task}")


def eval_code(model, tokenizer, device, args):
    from utils.evaluation.code_exec import write_jsonl, evaluate_functional_correctness

    load_error = None
    problems = None
    try:
        problems = _load_code_tasks(args)
    except BaseException as exc:
        load_error = f"{type(exc).__name__}: {exc}"
    _synchronize_rank_error(load_error, f"{args.task} dataset loading failed")

    world_size = _dist_world_size()
    rank = _dist_rank()
    setup_error = None
    out_dir = os.path.join(args.output, eval_setting_name(args))
    mtp_stats = None
    try:
        task_ids = sorted(problems.keys())
        while len(task_ids) % world_size != 0:
            task_ids.append("[PAD]")
        task_ids = task_ids[rank::world_size]
        os.makedirs(out_dir, exist_ok=True)
        mtp_stats = init_mtp_stats(args, device)
    except BaseException as exc:
        setup_error = f"{type(exc).__name__}: {exc}"
    _synchronize_rank_error(setup_error, f"{args.task} evaluator setup failed")

    samples = []
    for task_index, tid in enumerate(tqdm(task_ids, disable=rank != 0)):
        sample_error = None
        try:
            prompt = (
                "Please write a valid Python function."
                if tid == "[PAD]"
                else problems[tid]["prompt"]
            )
            msgs = [[{"role": "user", "content": prompt}]]
            prompt_text = tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False,
            )
            prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(device)

            gen_ids, batch_mtp_stats = generate_eval(model, prompt_ids, args)
            if tid != "[PAD]":
                update_mtp_stats(mtp_stats, batch_mtp_stats)
                completion = tokenizer.batch_decode(
                    gen_ids[:, prompt_ids.shape[1]:], skip_special_tokens=True,
                )[0]
                row = {"task_id": tid, "completion": completion}
                if args.task == "mbpp":
                    row["prompt"] = prompt
                samples.append(row)
        except BaseException as exc:
            sample_error = f"{type(exc).__name__}: {exc}"
        _synchronize_rank_error(
            sample_error, f"{args.task} sample {task_index} failed"
        )

    _dist_barrier()
    gathered = _dist_all_gather_object(samples)
    total_samples = sum(len(part) for part in gathered)
    mtp_metrics = finalize_mtp_stats(args, mtp_stats, total_samples)

    result = None
    postprocess_error = None
    if rank == 0:
        try:
            merged = [s for part in gathered for s in part]
            merged_path = os.path.join(out_dir, f"{args.task}_samples_merged.jsonl")
            write_jsonl(merged_path, merged)

            problem_file = (
                args.humaneval_data_file
                if args.task == "humaneval"
                else args.mbpp_test_file
            )
            result = evaluate_functional_correctness(
                input_file=merged_path, problem_file=problem_file,
                is_mbpp=(args.task == "mbpp"), n_workers=8, timeout=3.0, k=(1,),
            )
            print(f"\n{args.task}: {result}")
            result.update(mtp_metrics)
            save_log(args, {
                "metrics": result, "pass@1": result.get("pass@1"),
                "total": total_samples, **mtp_metrics,
            })
        except BaseException as exc:
            postprocess_error = f"{type(exc).__name__}: {exc}"
    _synchronize_rank_error(postprocess_error, f"{args.task} evaluation failed")
    return result


def _format_train_code_prompt(example):
    question = example["question"]
    test_cases = example["test_cases"]
    if "\n```\n" not in question and test_cases:
        question = question + "\nTest cases: " + test_cases[0]
    return question


def _load_train_code_tasks(path):
    from utils.evaluation.code_exec import stream_jsonl

    tasks = []
    for sample_id, example in enumerate(stream_jsonl(path)):
        question = example.get("question")
        test_cases = example.get("test_cases")
        if not isinstance(question, str) or not isinstance(test_cases, list):
            raise ValueError(
                f"Invalid code example at line {sample_id + 1}: "
                "expected string question and list test_cases"
            )
        tasks.append({
            "sample_id": sample_id,
            "prompt": _format_train_code_prompt(example),
            "test_cases": test_cases,
        })
    if not tasks:
        raise ValueError(f"No code examples found in {path}")
    return tasks


def eval_train_code(model, tokenizer, device, args):
    """Evaluate train-format code prompts with the training reward."""
    from data.code import reward_code
    from utils.evaluation.code_exec import write_jsonl

    load_error = None
    tasks = None
    try:
        tasks = _load_train_code_tasks(args.code_data_file)
    except BaseException as exc:
        load_error = f"{type(exc).__name__}: {exc}"
    _synchronize_rank_error(load_error, "Code dataset loading failed")

    world_size = _dist_world_size()
    rank = _dist_rank()
    out_dir = os.path.join(args.output, eval_setting_name(args))
    setup_error = None
    mtp_stats = None
    try:
        task_indices = list(range(len(tasks)))
        while len(task_indices) % world_size != 0:
            task_indices.append(None)
        task_indices = task_indices[rank::world_size]
        os.makedirs(out_dir, exist_ok=True)
        mtp_stats = init_mtp_stats(args, device)
    except BaseException as exc:
        setup_error = f"{type(exc).__name__}: {exc}"
    _synchronize_rank_error(setup_error, "Code evaluator setup failed")

    samples = []
    for sample_index, task_index in enumerate(tqdm(task_indices, disable=rank != 0)):
        sample_error = None
        try:
            prompt = (
                "Please write a valid Python function."
                if task_index is None else tasks[task_index]["prompt"]
            )
            prompt_text = tokenizer.apply_chat_template(
                [[{"role": "user", "content": prompt}]],
                add_generation_prompt=True, tokenize=False,
            )
            prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(device)
            gen_ids, batch_mtp_stats = generate_eval(model, prompt_ids, args)
            if task_index is not None:
                update_mtp_stats(mtp_stats, batch_mtp_stats)
                completion = tokenizer.batch_decode(
                    gen_ids[:, prompt_ids.shape[1]:], skip_special_tokens=True,
                )[0]
                samples.append({
                    "sample_id": tasks[task_index]["sample_id"],
                    "completion": completion,
                    "test_cases": tasks[task_index]["test_cases"],
                })
        except BaseException as exc:
            sample_error = f"{type(exc).__name__}: {exc}"
        _synchronize_rank_error(sample_error, f"Code sample {sample_index} failed")

    _dist_barrier()
    gathered = _dist_all_gather_object(samples)
    total_samples = sum(len(part) for part in gathered)
    mtp_metrics = finalize_mtp_stats(args, mtp_stats, total_samples)
    metrics = None
    postprocess_error = None
    if rank == 0:
        try:
            merged = sorted(
                (sample for part in gathered for sample in part),
                key=lambda sample: sample["sample_id"],
            )
            merged_path = os.path.join(out_dir, "code_samples_merged.jsonl")
            write_jsonl(merged_path, merged)
            batch = {"test_cases": [sample["test_cases"] for sample in merged]}
            responses = [sample["completion"] for sample in merged]
            rewards, components = reward_code(
                batch, responses, 1, device, return_components=True,
            )
            metrics = {
                "num_tasks": len(merged),
                "format": components["format_reward"].mean().item(),
                "pass_rate": components["pass_rate"].mean().item(),
                "reward": rewards.mean().item(),
                "pass@1": components["pass_rate"].mean().item(),
                **mtp_metrics,
            }
            print(f"\ncode: {metrics}")
            save_log(args, {"metrics": metrics, **metrics})
        except BaseException as exc:
            postprocess_error = f"{type(exc).__name__}: {exc}"
    _synchronize_rank_error(postprocess_error, "Code evaluation failed")
    return metrics


# ---- Main ----

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="RePara Evaluation")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="YAML experiment configuration containing one top-level eval section",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Override eval.ckpt_path for this run",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override eval.output for this run",
    )
    cli = parser.parse_args(argv)

    overrides = {}
    if cli.ckpt_path is not None:
        overrides["ckpt_path"] = cli.ckpt_path
    if cli.output is not None:
        overrides["output"] = cli.output
    args, config_path = load_config(
        cli.config,
        section="eval",
        config_class=EvalConfig,
        overrides=overrides,
    )
    return args, config_path, overrides


def main():
    args, config_path, overrides = parse_args()

    eval_runs = expand_eval_configs(args)
    torch.manual_seed(args.seed)
    raw_dist_was_ready = _raw_dist_ready()
    try:
        # Load the model independently on each rank, but synchronize errors
        # before any dataset collective so one bad checkpoint cannot leave the
        # other ranks waiting indefinitely.
        device = None
        model = None
        tokenizer = None
        setup_error = None
        try:
            device = init_dist()
            local_files_only = not args.allow_remote_files
            tokenizer = AutoTokenizer.from_pretrained(
                args.ckpt_path,
                trust_remote_code=(args.model_family != "sdar"),
                local_files_only=local_files_only,
            )
            if args.sdar_stopping_criteria_idx is None and tokenizer.eos_token_id is not None:
                args.sdar_stopping_criteria_idx = [tokenizer.eos_token_id]
            model_cls = AutoModelForCausalLM if args.model_family == "sdar" else AutoModel

            if os.path.exists(os.path.join(args.ckpt_path, "adapter_config.json")):
                adapter_base_path = resolve_adapter_base_path(args.ckpt_path)
                model = model_cls.from_pretrained(
                    adapter_base_path,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    local_files_only=local_files_only,
                )
                merge_lora(model, args.ckpt_path)
            else:
                model = model_cls.from_pretrained(
                    args.ckpt_path,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    local_files_only=local_files_only,
                )
            if args.model_family == "sdar":
                native_size = args.sdar_native_block_size
                if native_size is None and getattr(model.config, "block_size", None) is None:
                    if args.generation_method == "sdar_native":
                        native_size = args.sdar_block_length
                configure_sdar_native_block_size(model.config, native_size)
            if args.selective_logits:
                install_position_selective_output_head(model, model_family=args.model_family)
            model.eval().requires_grad_(False).to(device)
            args.mask_id = resolve_mask_id(args, model, tokenizer)
            tokenizer.pad_token_id = args.mask_id
        except BaseException as exc:
            setup_error = f"{type(exc).__name__}: {exc}"
        _synchronize_rank_error(setup_error, "evaluation model setup failed")

        rank = _dist_rank()
        for run_index, run_args in enumerate(eval_runs, start=1):
            run_args.mask_id = args.mask_id
            run_args._sampling_vocab_size = len(tokenizer)
            run_args.sdar_stopping_criteria_idx = args.sdar_stopping_criteria_idx
            run_args._config_path = config_path
            run_args._config_overrides = overrides
            torch.manual_seed(run_args.seed)

            if rank == 0:
                print(f"Evaluation {run_index}/{len(eval_runs)}")
                print(f"Model family: {run_args.model_family}")
                print(f"Mask token id: {run_args.mask_id}")
                print(f"Generation method: {run_args.generation_method}")
                print(
                    f"Generation length: {run_args.gen_length}; "
                    f"MTP size: {run_args.mtp_size}; steps: {run_args.steps}"
                )

            if run_args.task in ("gsm8k", "math500"):
                eval_math(model, tokenizer, device, run_args)
            elif run_args.task in ("humaneval", "mbpp"):
                eval_code(model, tokenizer, device, run_args)
            else:
                eval_train_code(model, tokenizer, device, run_args)
            _dist_barrier()
    finally:
        # Do not tear down a process group owned by an embedding caller.
        if not raw_dist_was_ready and _raw_dist_ready():
            torch_dist.destroy_process_group()


if __name__ == "__main__":
    main()
