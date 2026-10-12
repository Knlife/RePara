"""Reusable training and evaluation configuration."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, fields
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Mapping, Optional

import yaml


ACTIVATION_CHECKPOINTING_STRATEGIES = ("whole_layer", "none")
_MODEL_FAMILIES = ("llada", "sdar")
_MTP_METHODS = ("fixed_mtp", "ar_mtp")
_TRAIN_METHODS = ("justgrpo", *_MTP_METHODS)
_TRAIN_SCHEDULE_FIELDS = {"start_step", "method", "block_size"}
_EVAL_GENERATION_METHODS = ("diffusion", *_MTP_METHODS, "sdar_native")
_ONLINE_EVAL_GENERATION_METHODS = ("diffusion", *_MTP_METHODS, "sdar_native")
_MTP_CONFIDENCE_METRICS = ("normalized_entropy", "max_probability")
_SDAR_REMASKING_STRATEGIES = (
    "low_confidence_dynamic",
    "low_confidence_static",
    "sequential",
    "entropy_bounded",
)
_TRAIN_DATA_PATH_FIELDS = {
    "gsm8k": "gsm8k_data_path",
    "math": "math_data_path",
    "code": "code_data_path",
}
_EVAL_DATA_FIELDS = {
    "gsm8k": ("gsm8k_data_files",),
    "math500": ("math500_data_files",),
    "humaneval": ("humaneval_data_file",),
    "mbpp": ("mbpp_data_file", "mbpp_test_file"),
    "code": ("code_data_file",),
}
ONLINE_EVAL_FIELDS = (
    "gen_length",
    "mtp_size",
    "block_length",
    "seed",
    "temperature",
    "generation_method",
    "mtp_block_size",
    "mtp_confidence",
    "mtp_threshold",
    "selective_logits",
    "allow_remote_files",
    "gsm8k_data_files",
    "math500_data_files",
    "humaneval_data_file",
    "mbpp_data_file",
    "mbpp_test_file",
    "code_data_file",
    "sdar_block_length",
    "sdar_native_block_size",
    "sdar_denoising_steps",
    "sdar_remasking_strategy",
    "sdar_confidence_threshold",
    "sdar_eb_threshold",
    "sdar_temperature",
    "sdar_top_k",
    "sdar_top_p",
    "sdar_stopping_criteria_idx",
    "batch_size",
    "mtp_like_prefix_size",
    "report_diffusion_mtp_like_stats",
)
_ONLINE_EVAL_KEYS = {"tasks", "output_dir", *ONLINE_EVAL_FIELDS}
_ENV_REFERENCE = re.compile(
    r"\$(?:\{[A-Za-z_][A-Za-z0-9_]*(?::-[^}]*)?\}|[A-Za-z_][A-Za-z0-9_]*)"
)
_ENV_DEFAULT_REFERENCE = re.compile(
    r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*):-"
    r"(?P<default>[^}]*)\}"
)


def _expand_environment(text: str) -> str:
    """Expand ``$NAME`` and shell-style ``${NAME:-default}`` references."""
    def replace_default(match: re.Match[str]) -> str:
        value = os.environ.get(match.group("name"))
        return value if value else match.group("default")

    return os.path.expandvars(_ENV_DEFAULT_REFERENCE.sub(replace_default, text))


def _contains_environment_reference(value) -> bool:
    if isinstance(value, str):
        return _ENV_REFERENCE.search(value) is not None
    if isinstance(value, (list, tuple)):
        return any(_contains_environment_reference(item) for item in value)
    if isinstance(value, Mapping):
        return any(_contains_environment_reference(item) for item in value.values())
    return False


def _validate_environment(config, section: str) -> None:
    unresolved = [
        field.name
        for field in fields(config)
        if _contains_environment_reference(getattr(config, field.name))
    ]
    if unresolved:
        raise ValueError(
            f"Unresolved environment variable(s) in {section} config field(s): "
            f"{', '.join(unresolved)}"
        )


def _require_fields(config, section: str, names) -> None:
    missing = []
    invalid = []
    for name in names:
        value = getattr(config, name)
        if value is None or (isinstance(value, str) and not value.strip()):
            missing.append(name)
        elif not isinstance(value, (str, os.PathLike)):
            invalid.append(name)
    if missing:
        raise ValueError(
            f"Missing required {section} config field(s): {', '.join(missing)}"
        )
    if invalid:
        raise ValueError(
            f"{section} config path field(s) must be strings: {', '.join(invalid)}"
        )


def _validate_online_eval(value, *, model_family="llada") -> None:
    """Validate the resolved online-evaluation specification."""
    if not isinstance(value, Mapping):
        raise ValueError("eval_config must be a mapping")

    unknown = sorted(set(value) - _ONLINE_EVAL_KEYS)
    if unknown:
        raise ValueError(f"Unknown eval_config field(s): {', '.join(unknown)}")

    output_dir = value.get("output_dir")
    if not isinstance(output_dir, (str, os.PathLike)):
        raise ValueError("eval_config.output_dir must be a path string")
    for name in (
        "gsm8k_data_files",
        "math500_data_files",
        "humaneval_data_file",
        "mbpp_data_file",
        "mbpp_test_file",
    ):
        item = value.get(name)
        if item is not None and not isinstance(item, (str, os.PathLike)):
            raise ValueError(f"eval_config.{name} must be a path string")

    tasks = value.get("tasks", ("mbpp", "humaneval"))
    if isinstance(tasks, (str, bytes)) or not isinstance(tasks, (list, tuple)):
        raise ValueError("eval_config.tasks must be a non-empty list")
    if not tasks or any(
        not isinstance(task, str) or task not in _EVAL_DATA_FIELDS
        for task in tasks
    ):
        raise ValueError(
            "eval_config.tasks must contain only gsm8k, math500, humaneval, mbpp, or code"
        )
    if len(set(tasks)) != len(tasks):
        raise ValueError("eval_config.tasks must not contain duplicates")

    for name in (
        "gen_length",
        "mtp_size",
        "block_length",
        "mtp_block_size",
        "mtp_like_prefix_size",
    ):
        item = value.get(name)
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise ValueError(f"eval_config.{name} must be a positive integer")

    seed = value.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("eval_config.seed must be a non-negative integer")

    batch_size = value.get("batch_size")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise ValueError("eval_config.batch_size must be an integer")
    if batch_size != 1:
        raise ValueError("eval_config.batch_size must be 1")

    for name in (
        "selective_logits",
        "allow_remote_files",
        "report_diffusion_mtp_like_stats",
    ):
        if not isinstance(value.get(name), bool):
            raise ValueError(f"eval_config.{name} must be boolean")

    threshold = value.get("mtp_threshold")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, Real)
        or not isfinite(float(threshold))
        or not 0 <= threshold <= 1
    ):
        raise ValueError("eval_config.mtp_threshold must be a finite number in [0, 1]")

    gen_length = value["gen_length"]
    mtp_size = value["mtp_size"]
    block_length = value["block_length"]
    generation_method = value["generation_method"]
    if generation_method not in _ONLINE_EVAL_GENERATION_METHODS:
        raise ValueError(
            f"Unsupported online eval generation method: {generation_method}"
        )
    if gen_length % mtp_size != 0:
        raise ValueError("eval_config.gen_length must be divisible by mtp_size")
    if generation_method == "diffusion":
        if gen_length % block_length != 0:
            raise ValueError(
                "eval_config.gen_length must be divisible by block_length"
            )
        if block_length % mtp_size != 0:
            raise ValueError(
                "eval_config.block_length must be divisible by mtp_size"
            )
    elif value["mtp_block_size"] > gen_length:
        raise ValueError("eval_config.mtp_block_size cannot exceed gen_length")
    if (
        value["report_diffusion_mtp_like_stats"]
        and value["mtp_like_prefix_size"] > gen_length
    ):
        raise ValueError("eval_config.mtp_like_prefix_size cannot exceed gen_length")

    defaults = EvalConfig()
    eval_field_names = {field.name for field in fields(EvalConfig)} - {"task"}
    eval_values = {name: getattr(defaults, name) for name in eval_field_names}
    eval_values.update({name: value[name] for name in eval_field_names & value.keys()})
    eval_values.update(
        {
            "ckpt_path": "__online_eval__",
            "model_family": model_family,
            "mask_id": None,
            "output": "__online_eval__",
        }
    )
    for task in tasks:
        EvalConfig(task=task, **eval_values).validate()


@dataclass(frozen=True)
class TrainStage:
    """One immutable training stage selected by optimizer step."""

    start_step: int
    method: str
    block_size: int

    def as_dict(self) -> dict[str, object]:
        return {
            "start_step": self.start_step,
            "method": self.method,
            "block_size": self.block_size,
        }


def _parse_train_schedule(value, *, total_steps: int, gen_length: int):
    """Validate and normalize an explicit zero-based training schedule."""
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError("train_schedule must be a non-empty list of mappings")
    if not value:
        raise ValueError("train_schedule must not be empty")

    stages = []
    previous_start = -1
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"train_schedule[{index}] must be a mapping")
        unknown = [key for key in item if key not in _TRAIN_SCHEDULE_FIELDS]
        if unknown:
            names = ", ".join(str(key) for key in unknown)
            raise ValueError(f"Unknown train_schedule field(s): {names}")
        missing = [key for key in _TRAIN_SCHEDULE_FIELDS if key not in item]
        if missing:
            raise ValueError(
                f"train_schedule[{index}] missing field(s): {', '.join(sorted(missing))}"
            )

        start_step = item["start_step"]
        if isinstance(start_step, bool) or not isinstance(start_step, int):
            raise ValueError(f"train_schedule[{index}].start_step must be an integer")
        if start_step < 0:
            raise ValueError(f"train_schedule[{index}].start_step must be non-negative")
        if start_step >= total_steps:
            raise ValueError(
                f"train_schedule[{index}].start_step must be less than total_steps"
            )
        if index == 0 and start_step != 0:
            raise ValueError("train_schedule must start at step 0")
        if start_step <= previous_start:
            raise ValueError("train_schedule start_step values must be strictly increasing")

        method = item["method"]
        if not isinstance(method, str) or method not in _TRAIN_METHODS:
            raise ValueError(
                f"Unknown train_schedule method at index {index}: {method!r}"
            )

        block_size = item["block_size"]
        if isinstance(block_size, bool) or not isinstance(block_size, int):
            raise ValueError(f"train_schedule[{index}].block_size must be an integer")
        if not 1 <= block_size <= gen_length:
            raise ValueError(
                f"train_schedule[{index}].block_size must be in [1, gen_length]"
            )
        if method == "justgrpo" and block_size != 1:
            raise ValueError(
                f"train_schedule[{index}] JustGRPO stages must use block_size=1"
            )

        stages.append(TrainStage(start_step, method, block_size))
        previous_start = start_step
    return tuple(stages)


@dataclass
class TrainConfig:
    model_path: str = ""
    model_family: str = "llada"
    mask_id: Optional[int] = None
    # Model-native attention/KV-cache block size, independent of RePara's MTP size.
    sdar_native_block_size: Optional[int] = None

    batch_size_per_device: int = 1
    grad_accumulation: int = 8
    total_steps: int = 128
    learning_rate: float = 5e-6
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    seed: int = 1234
    num_generations: int = 16
    repeat_times: int = 1
    gen_steps: int = 256
    gen_length: int = 256

    method: str = "justgrpo"
    mtp_block_size: int = 4
    mtp_confidence: str = "normalized_entropy"
    mtp_threshold: float = 0.8
    format_reward_alpha: float = 1.0
    train_schedule: Optional[list[dict]] = None

    selective_logits: bool = False
    activation_checkpointing_strategy: str = "none"


    output_dir: str = ""
    log_every: int = 1
    save_every: int = 10
    eval_every: int = 0
    eval_config: Optional[dict] = None
    resume_ckpt: Optional[str] = None
    dataset: str = "gsm8k"
    save_model_only: bool = False
    keep_latest_state_only: bool = False
    gsm8k_data_path: Optional[str] = None
    math_data_path: Optional[str] = None
    code_data_path: Optional[str] = None

    wandb: bool = False
    wandb_project: str = "RePara"
    wandb_run_name: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_mode: str = "offline"
    wandb_dir: Optional[str] = None
    profile_timing: bool = False
    timing_warmup_steps: int = 1

    def resolved_train_schedule(self) -> tuple[TrainStage, ...]:
        """Return the explicit schedule, or a single legacy stage."""
        if self.train_schedule is None:
            return (TrainStage(0, self.method, self.mtp_block_size),)
        return _parse_train_schedule(
            self.train_schedule,
            total_steps=self.total_steps,
            gen_length=self.gen_length,
        )

    def normalized_train_schedule(self) -> Optional[list[dict[str, object]]]:
        """Return a JSON/YAML-stable schedule representation when configured."""
        if self.train_schedule is None:
            return None
        return [stage.as_dict() for stage in self.resolved_train_schedule()]

    def _stage_selection_for_step(self, step: int) -> tuple[int, TrainStage]:
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("step must be a non-negative integer")
        for index, stage in reversed(tuple(enumerate(self.resolved_train_schedule()))):
            if step >= stage.start_step:
                return index, stage
        raise ValueError(f"No training stage covers step {step}")

    def stage_for_step(self, step: int) -> TrainStage:
        """Select the last stage whose zero-based start step has been reached."""
        return self._stage_selection_for_step(step)[1]

    def stage_index_for_step(self, step: int) -> int:
        return self._stage_selection_for_step(step)[0]

    def validate(self) -> None:
        _validate_environment(self, "train")
        if self.dataset not in _TRAIN_DATA_PATH_FIELDS:
            raise ValueError(f"Unknown training dataset: {self.dataset}")
        _require_fields(
            self,
            "train",
            ("model_path", "output_dir", _TRAIN_DATA_PATH_FIELDS[self.dataset]),
        )

        for name in (
            "batch_size_per_device",
            "grad_accumulation",
            "total_steps",
            "num_generations",
            "repeat_times",
            "gen_steps",
            "gen_length",
            "log_every",
            "save_every",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.num_generations * self.repeat_times < 2:
            raise ValueError(
                "num_generations * repeat_times must be at least 2 "
                "for group normalization"
            )
        if self.model_family not in _MODEL_FAMILIES:
            raise ValueError(f"Unknown model family: {self.model_family}")
        if self.method not in _TRAIN_METHODS:
            raise ValueError(f"Unknown training method: {self.method}")
        if self.sdar_native_block_size is not None:
            if self.model_family != "sdar":
                raise ValueError("sdar_native_block_size requires model_family=sdar")
            if (
                isinstance(self.sdar_native_block_size, bool)
                or not isinstance(self.sdar_native_block_size, int)
                or self.sdar_native_block_size < 1
            ):
                raise ValueError("sdar_native_block_size must be a positive integer")
            if self.method in _MTP_METHODS and self.mtp_block_size > self.sdar_native_block_size:
                raise ValueError("mtp_block_size must not exceed sdar_native_block_size")
        if not 1 <= self.mtp_block_size <= self.gen_length:
            raise ValueError("mtp_block_size must be in [1, gen_length]")
        if self.train_schedule is None:
            scheduled_methods = {self.method}
        else:
            scheduled_methods = {stage.method for stage in self.resolved_train_schedule()}
        if self.sdar_native_block_size is not None and any(
            stage.method in _MTP_METHODS
            and stage.block_size > self.sdar_native_block_size
            for stage in self.resolved_train_schedule()
        ):
            raise ValueError("Scheduled MTP block_size must not exceed sdar_native_block_size")
        if scheduled_methods & set(_MTP_METHODS) and self.gen_steps != self.gen_length:
            raise ValueError("MTP methods require gen_steps == gen_length")
        if "justgrpo" in scheduled_methods and self.gen_steps % self.gen_length != 0:
            raise ValueError(
                "JustGRPO requires gen_steps to be divisible by gen_length"
            )
        if self.mtp_confidence not in _MTP_CONFIDENCE_METRICS:
            raise ValueError(
                f"Unknown MTP confidence metric: {self.mtp_confidence}"
            )
        if not 0 <= self.mtp_threshold <= 1:
            raise ValueError("mtp_threshold must be between 0 and 1")
        if (
            isinstance(self.format_reward_alpha, bool)
            or not isinstance(self.format_reward_alpha, (int, float))
            or not isfinite(float(self.format_reward_alpha))
            or self.format_reward_alpha < 0
        ):
            raise ValueError("format_reward_alpha must be finite and non-negative")
        if self.selective_logits and self.model_family not in ("llada", "sdar"):
            raise ValueError(
                "selective_logits currently supports model_family=llada/sdar only"
            )
        if (
            self.activation_checkpointing_strategy
            not in ACTIVATION_CHECKPOINTING_STRATEGIES
        ):
            raise ValueError(
                "Unknown activation checkpointing strategy: "
                f"{self.activation_checkpointing_strategy}"
            )
        if self.save_model_only and self.keep_latest_state_only:
            raise ValueError("keep_latest_state_only requires saving training state")
        if self.timing_warmup_steps < 0:
            raise ValueError("timing_warmup_steps must be non-negative")
        if (
            isinstance(self.eval_every, bool)
            or not isinstance(self.eval_every, int)
            or self.eval_every < 0
        ):
            raise ValueError("eval_every must be a non-negative integer")
        if self.eval_every > 0:
            if self.eval_config is None:
                raise ValueError("eval_config is required when eval_every is enabled")
            if self.dataset in ("gsm8k", "math"):
                if self.eval_every % self.save_every:
                    raise ValueError(
                        "eval_every must be a multiple of save_every; remove stale SAVE_EVERY overrides"
                    )
                resolve_online_eval_config(self)
            else:
                spec = resolve_online_eval_config(self)
                if spec["generation_method"] == "sdar_native":
                    raise ValueError(
                        "Native SDAR online code evaluation requires checkpoint replicas; "
                        "run standalone evaluation instead"
                    )
                _validate_online_eval(
                    spec, model_family=self.model_family,
                )
        elif self.eval_config is not None and not isinstance(self.eval_config, Mapping):
            raise ValueError("eval_config must be a mapping")

    @property
    def data_path(self) -> Optional[str]:
        return getattr(self, _TRAIN_DATA_PATH_FIELDS[self.dataset])


@dataclass
class EvalConfig:
    task: str = "gsm8k"
    ckpt_path: str = ""
    model_family: str = "llada"
    mask_id: Optional[int] = None
    sdar_native_block_size: Optional[int] = None
    gsm8k_data_files: str = ""
    math500_data_files: str = ""
    humaneval_data_file: str = ""
    mbpp_data_file: str = ""
    mbpp_test_file: str = ""
    code_data_file: str = ""
    allow_remote_files: bool = False
    output: str = ""
    batch_size: int = 1
    gen_length: int | list[int] = 256
    mtp_size: int | list[int] = 1
    block_length: int = 32
    seed: int = 113
    temperature: float = 0.0
    generation_method: str = "diffusion"
    mtp_block_size: int = 2
    mtp_confidence: str = "normalized_entropy"
    mtp_threshold: float = 0.8
    selective_logits: bool = False
    mtp_like_prefix_size: int = 2
    report_diffusion_mtp_like_stats: bool = False
    sdar_block_length: int = 4
    sdar_denoising_steps: int = 4
    sdar_remasking_strategy: str = "low_confidence_dynamic"
    sdar_confidence_threshold: float = 0.85
    sdar_eb_threshold: float = 0.35
    sdar_temperature: float = 1.0
    sdar_top_k: int = 0
    sdar_top_p: float = 1.0
    sdar_stopping_criteria_idx: Optional[list[int]] = None

    def validate(self) -> None:
        _validate_environment(self, "eval")
        if self.task not in _EVAL_DATA_FIELDS:
            raise ValueError(f"Unknown evaluation task: {self.task}")
        if not isinstance(self.allow_remote_files, bool):
            raise ValueError("allow_remote_files must be boolean")
        required_fields = ("ckpt_path", "output")
        if not self.allow_remote_files or self.task in ("humaneval", "mbpp", "code"):
            required_fields += _EVAL_DATA_FIELDS[self.task]
        _require_fields(self, "eval", required_fields)
        if self.model_family not in _MODEL_FAMILIES:
            raise ValueError(f"Unknown model family: {self.model_family}")
        if self.sdar_native_block_size is not None:
            if self.model_family != "sdar":
                raise ValueError("sdar_native_block_size requires model_family=sdar")
            if (
                isinstance(self.sdar_native_block_size, bool)
                or not isinstance(self.sdar_native_block_size, int)
                or self.sdar_native_block_size < 1
            ):
                raise ValueError("sdar_native_block_size must be a positive integer")
            if self.generation_method in _MTP_METHODS and self.mtp_block_size > self.sdar_native_block_size:
                raise ValueError("mtp_block_size must not exceed sdar_native_block_size")
            if self.generation_method == "sdar_native" and self.sdar_block_length != self.sdar_native_block_size:
                raise ValueError("sdar_block_length must match sdar_native_block_size")
        if self.generation_method not in _EVAL_GENERATION_METHODS:
            raise ValueError(
                f"Unknown generation method: {self.generation_method}"
            )
        if self.mtp_confidence not in _MTP_CONFIDENCE_METRICS:
            raise ValueError(
                f"Unknown MTP confidence metric: {self.mtp_confidence}"
            )
        if self.sdar_remasking_strategy not in _SDAR_REMASKING_STRATEGIES:
            raise ValueError(
                "Unknown SDAR remasking strategy: "
                f"{self.sdar_remasking_strategy}"
            )
        if isinstance(self.batch_size, bool) or self.batch_size != 1:
            raise ValueError("RePara evaluation requires batch_size=1")
        for name in ("temperature", "sdar_temperature"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(float(value)) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in ("block_length", "mtp_block_size", "mtp_like_prefix_size", "sdar_block_length", "sdar_denoising_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("mtp_threshold", "sdar_confidence_threshold", "sdar_eb_threshold"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(float(value)) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be a finite number in [0, 1]")
        if self.generation_method == "sdar_native":
            if self.model_family != "sdar":
                raise ValueError("sdar_native evaluation requires model_family=sdar")
            if self.sdar_denoising_steps > self.sdar_block_length:
                raise ValueError("sdar_denoising_steps cannot exceed sdar_block_length")
        if self.selective_logits and self.model_family not in ("llada", "sdar"):
            raise ValueError(
                "selective_logits currently supports model_family=llada/sdar only"
            )
        if self.selective_logits and self.generation_method not in _MTP_METHODS:
            raise ValueError(
                "selective_logits evaluation currently supports "
                "MTP generation methods only"
            )

    @property
    def steps(self) -> int:
        if not isinstance(self.gen_length, int) or isinstance(self.gen_length, bool):
            raise ValueError("gen_length must be scalar after expanding the eval sweep")
        if not isinstance(self.mtp_size, int) or isinstance(self.mtp_size, bool):
            raise ValueError("mtp_size must be scalar after expanding the eval sweep")
        return self.gen_length // self.mtp_size


def resolve_online_eval_config(config: TrainConfig) -> dict:
    """Return one complete, runtime-ready online evaluation specification.

    The nested train ``eval_config`` is an override layer over ``EvalConfig``.
    Rollout-critical fields inherit the training protocol when omitted; the
    evaluator still owns its benchmark-specific defaults (MTP size, block
    length, generation method, and batch size).  Every explicit nested value
    wins over both layers.
    """
    if config.dataset in ("gsm8k", "math"):
        return _resolve_math_online_eval_config(config)
    if config.eval_config is None:
        return {}
    if not isinstance(config.eval_config, Mapping):
        raise ValueError("eval_config must be a mapping")

    eval_defaults = EvalConfig()
    resolved = {name: getattr(eval_defaults, name) for name in ONLINE_EVAL_FIELDS}
    resolved.update(
        {
            "tasks": ("mbpp", "humaneval"),
            "output_dir": str(Path(config.output_dir) / "benchmark"),
            "gen_length": config.gen_length,
            "seed": config.seed,
            "mtp_block_size": config.mtp_block_size,
            "mtp_confidence": config.mtp_confidence,
            "mtp_threshold": config.mtp_threshold,
            "sdar_native_block_size": config.sdar_native_block_size,
        }
    )
    resolved.update(dict(config.eval_config))
    output_dir = resolved.get("output_dir")
    if output_dir is None or output_dir == "":
        resolved["output_dir"] = str(Path(config.output_dir) / "benchmark")
    elif isinstance(output_dir, (str, os.PathLike)):
        resolved["output_dir"] = os.fspath(output_dir)
    # Leave other types untouched so _validate_online_eval reports a clear
    # configuration error instead of silently falling back to the default.
    return resolved


def _resolve_math_online_eval_config(config: TrainConfig) -> dict:
    """Resolve the train-monitor eval spec, extended to SDAR and rate sweeps.

    Ported selectively from origin/feat/train-monitor (c0101b9). For SDAR,
    mtp_size denotes the nominal rate: denoising_steps = native block / rate.
    """
    if not isinstance(config.eval_config, Mapping):
        raise ValueError("eval_config is required and must be a mapping")
    if config.dataset not in ("gsm8k", "math"):
        raise ValueError("Online checkpoint evaluation currently supports gsm8k/math training")
    task = "math500" if config.dataset == "math" else "gsm8k"
    excluded = {"task", "ckpt_path", "model_family", "mask_id", "output", "sdar_denoising_steps"}
    defaults = EvalConfig()
    allowed = {field.name for field in fields(EvalConfig)} - excluded
    unknown = set(config.eval_config) - allowed - {"tasks", "output_dir"}
    if unknown:
        raise ValueError(f"Unknown eval_config field(s): {', '.join(sorted(unknown))}")
    spec = {key: getattr(defaults, key) for key in allowed}
    spec.update(
        tasks=[task], output_dir=str(Path(config.output_dir) / "online-eval"),
        gen_length=config.gen_length, mtp_size=[1, 2, 4],
        generation_method="sdar_native" if config.model_family == "sdar" else "diffusion",
        sdar_temperature=0.0,
        sdar_native_block_size=config.sdar_native_block_size,
        sdar_block_length=config.sdar_native_block_size or 4,
    )
    spec.update(config.eval_config)
    if spec["tasks"] != [task]:
        raise ValueError(f"eval_config.tasks must be [{task!r}] for dataset={config.dataset}")
    if not isinstance(spec["output_dir"], str) or not spec["output_dir"].strip():
        raise ValueError("eval_config.output_dir must be a non-empty path string")
    expected_method = "sdar_native" if config.model_family == "sdar" else "diffusion"
    if spec["generation_method"] != expected_method:
        raise ValueError(f"Online {config.model_family} evaluation requires {expected_method}")
    for key in ("gen_length", "block_length", "sdar_block_length"):
        if isinstance(spec[key], bool) or not isinstance(spec[key], int) or spec[key] <= 0:
            raise ValueError(f"eval_config.{key} must be a positive integer")
    rates = spec["mtp_size"]
    rates = rates if isinstance(rates, list) else [rates]
    if not rates or any(isinstance(r, bool) or not isinstance(r, int) or r <= 0 for r in rates):
        raise ValueError("eval_config.mtp_size must contain positive integer rates")
    if len(set(rates)) != len(rates):
        raise ValueError("eval_config.mtp_size must not contain duplicates")
    spec["mtp_size"] = rates
    if isinstance(spec["seed"], bool) or not isinstance(spec["seed"], int) or spec["seed"] < 0:
        raise ValueError("eval_config.seed must be a non-negative integer")
    if spec["batch_size"] != 1 or isinstance(spec["batch_size"], bool):
        raise ValueError("Online evaluation requires batch_size=1")
    if spec["sdar_temperature"] != 0 or spec["temperature"] != 0:
        raise ValueError("Online math evaluation uses temperature=0")
    if spec["selective_logits"] or spec["allow_remote_files"]:
        raise ValueError("Online diffusion evaluation requires local files and selective_logits=false")
    for key in ("sdar_confidence_threshold", "sdar_eb_threshold"):
        if isinstance(spec[key], bool) or not isinstance(spec[key], (int, float)) or not 0 <= spec[key] <= 1:
            raise ValueError(f"eval_config.{key} must be a number in [0, 1]")
    block = spec["sdar_block_length"] if config.model_family == "sdar" else spec["block_length"]
    if spec["gen_length"] % block:
        raise ValueError("eval_config.gen_length must be divisible by the decoding block length")
    for rate in rates:
        if block % rate:
            raise ValueError("Decoding block length must be divisible by every mtp_size rate")
        values = {k: spec[k] for k in allowed}
        values["mtp_size"] = rate
        args = EvalConfig(
            **values, task=task, ckpt_path=config.model_path,
            model_family=config.model_family, output=spec["output_dir"],
            sdar_denoising_steps=block // rate if config.model_family == "sdar" else 4,
        )
        args.validate()
    return spec


def load_config(config_path, *, section, config_class, overrides=None):
    """Load one strict top-level section from an experiment YAML file."""
    path = Path(config_path).expanduser().resolve()
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(_expand_environment(handle.read()))

    if not isinstance(payload, dict) or set(payload) != {section}:
        raise ValueError(
            f"Experiment config must contain only the top-level {section!r} section"
        )
    values = payload[section]
    if not isinstance(values, dict):
        raise ValueError(f"Experiment config section {section!r} must be a mapping")

    values = dict(values)
    values.update(overrides or {})
    allowed_keys = {field.name for field in fields(config_class)}
    unknown = sorted(set(values) - allowed_keys)
    if unknown:
        raise ValueError(
            f"Unknown {section} config field(s): {', '.join(unknown)}"
        )

    config = config_class(**values)
    config.validate()
    return config, path


def snapshot_config(
    output_dir: str | Path,
    *,
    source_path: Path,
    section: str,
    effective_config: Mapping[str, object],
    overrides: Mapping[str, object],
) -> None:
    """Store the submitted YAML and the resolved configuration in a run directory."""
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    source_destination = output / "source_config.yaml"
    if source_path.resolve() != source_destination:
        _atomic_write(source_destination, source_path.read_bytes())

    effective_payload = {
        "source_config": str(source_path),
        "overrides": dict(overrides),
        section: dict(effective_config),
    }
    serialized = yaml.safe_dump(
        effective_payload,
        sort_keys=False,
        allow_unicode=True,
    ).encode("utf-8")
    _atomic_write(output / "effective_config.yaml", serialized)


def _atomic_write(path: Path, content: bytes) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)
