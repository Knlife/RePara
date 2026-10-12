'LLaDA model and evaluation support.'

from contextlib import contextmanager
from dataclasses import asdict, fields
import gc
import glob
import json
import os
from pathlib import Path
import random
import traceback

import numpy as np
import torch
import yaml

from utils.configuration.config import EvalConfig, TrainConfig, _atomic_write, resolve_online_eval_config


def online_evaluation_due(config: TrainConfig, completed_step: int) -> bool:
    return config.eval_every > 0 and completed_step > 0 and (
        completed_step % config.eval_every == 0 or completed_step == config.total_steps
    )


def validate_online_eval_inputs(config: TrainConfig) -> None:
    """Catch missing test data before the first training optimizer step."""
    if config.eval_every <= 0:
        return
    spec = resolve_online_eval_config(config)
    key = "math500_data_files" if config.dataset == "math" else "gsm8k_data_files"
    matches = [Path(p).resolve() for p in glob.glob(spec[key]) if Path(p).is_file()]
    if not matches:
        raise FileNotFoundError(f"No online evaluation test files matched: {spec[key]}")
    if Path(config.data_path).resolve() in matches:
        raise ValueError("Online evaluation must use held-out test data, not the training file")


def build_online_eval_runs(config, checkpoint_path, completed_step, *, mask_id=None, eos_token_id=None):
    spec = resolve_online_eval_config(config)
    task = spec["tasks"][0]
    step_dir = Path(spec["output_dir"]) / task / f"step-{completed_step:06d}"
    allowed = {field.name for field in fields(EvalConfig)}
    runs = []
    for rate in spec["mtp_size"]:
        values = {key: value for key, value in spec.items() if key in allowed}
        values.update(
            task=task, ckpt_path=str(checkpoint_path), model_family=config.model_family,
            mask_id=mask_id, mtp_size=rate, output=str(step_dir / f"diff{rate}"),
        )

        args = EvalConfig(**values)
        args.validate()
        args._online_metadata = {
            "completed_step": completed_step,
            "decoding_rate": rate,
            "decoding_rate_is_nominal": False,
            "temperature": 0.0,
        }
        runs.append(args)
    return runs


@contextmanager
def isolated_eval_rng(seed, device):
    """Checkpoint evaluation must not advance learner or dataloader RNGs."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    device = torch.device(device)
    devices = [torch.cuda.current_device() if device.index is None else device.index] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed % (1 << 32))
            torch.random.default_generator.manual_seed(seed)
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


@contextmanager
def _replicated_model_loading():
    # Transformers otherwise sees the training launcher's FSDP environment
    # and can instantiate empty weights on nonzero ranks. This model is NOT
    # passed to accelerator.prepare and every rank needs its own full weights.
    key = "ACCELERATE_USE_FSDP"
    previous = os.environ.get(key)
    os.environ[key] = "false"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


@contextmanager
def _llada_allocator_compat(model_family):
    'LLaDA model and evaluation support.'
    if model_family != "llada":
        yield
        return
    from transformers import modeling_utils

    original_warmup = modeling_utils.caching_allocator_warmup

    def compatible_warmup(model, *args, **kwargs):
        if getattr(model, "_tp_plan", None) is None:
            return
        return original_warmup(model, *args, **kwargs)

    modeling_utils.caching_allocator_warmup = compatible_warmup
    try:
        yield
    finally:
        modeling_utils.caching_allocator_warmup = original_warmup


def _load_checkpoint_for_eval(checkpoint_path, model_family, device):
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path, trust_remote_code=True, local_files_only=True,
    )
    model_cls = AutoModel
    with _replicated_model_loading(), _llada_allocator_compat(model_family):
        model = model_cls.from_pretrained(
            checkpoint_path, trust_remote_code=True, local_files_only=True,
            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            device_map={"": str(device)},
        )
    model.eval().requires_grad_(False)
    return model, tokenizer


def run_online_evaluation(*, device, config, checkpoint_path, completed_step):
    'LLaDA model and evaluation support.'
    if not online_evaluation_due(config, completed_step):
        return {}
    from eval import (
        _dist_barrier, _dist_rank, _synchronize_rank_error,
        eval_math, eval_setting_name, resolve_mask_id,
    )

    device = torch.device(device)
    spec = resolve_online_eval_config(config)
    model = tokenizer = None
    result = {}
    try:
        with isolated_eval_rng(spec["seed"], device):
            load_error = None
            try:
                validate_online_eval_inputs(config)
                if not Path(checkpoint_path).is_dir():
                    raise FileNotFoundError(f"Missing saved checkpoint: {checkpoint_path}")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                model, tokenizer = _load_checkpoint_for_eval(
                    str(checkpoint_path), config.model_family, device,
                )
                mask_id = resolve_mask_id(config, model, tokenizer)
                tokenizer.pad_token_id = mask_id
                sampling_vocab_size = len(tokenizer)
                runs = build_online_eval_runs(
                    config, checkpoint_path, completed_step, mask_id=mask_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            except Exception as exc:
                load_error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            _synchronize_rank_error(load_error, "Online checkpoint evaluation setup failed")

            for args in runs:
                args._sampling_vocab_size = sampling_vocab_size
                setup_error = None
                try:
                    args._config_path = Path(args.output) / "eval_config.yaml"
                    args._config_overrides = {}
                    if _dist_rank() == 0:
                        args._config_path.parent.mkdir(parents=True, exist_ok=True)
                        _atomic_write(
                            args._config_path,
                            yaml.safe_dump({"eval": asdict(args)}, sort_keys=False).encode("utf-8"),
                        )
                        print(
                            f"Online eval: step={completed_step} task={args.task} "
                            f"diff#{args.mtp_size} length={args.gen_length} temperature=0"
                        )
                except Exception as exc:
                    setup_error = f"{type(exc).__name__}: {exc}"
                _synchronize_rank_error(setup_error, "Online evaluation output setup failed")
                with isolated_eval_rng(args.seed, device), torch.no_grad():
                    metrics = eval_math(model, tokenizer, device, args)
                if _dist_rank() == 0:
                    result[f"{args.task}/diff{args.mtp_size}"] = {
                        **metrics, **args._online_metadata,
                        "evaluation_log": str(Path(args.output) / eval_setting_name(args) / "evaluation_log.json"),
                    }
                _dist_barrier()

            save_error = None
            if _dist_rank() == 0:
                try:
                    summary_path = Path(runs[0].output).parent / "summary.json"
                    _atomic_write(summary_path, (json.dumps({
                        "checkpoint_path": str(checkpoint_path),
                        "completed_step": completed_step,
                        "model_family": config.model_family,
                        "gen_length": spec["gen_length"],
                        "results": result,
                    }, indent=2, sort_keys=True) + "\n").encode("utf-8"))
                except Exception as exc:
                    save_error = f"{type(exc).__name__}: {exc}"
            _synchronize_rank_error(save_error, "Online evaluation summary save failed")
    finally:
        del model, tokenizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return result
