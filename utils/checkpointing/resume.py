"""Validate checkpoint completeness and training semantics for recovery."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Dict, Optional

from utils.checkpointing.manifest import (
    MANIFEST_FILENAME, MANIFEST_SCHEMA_VERSION, sha256_file, training_source_files,
)


_ROLLOUT_RNG_KNOWN_ANSWER = 1496865584815556242
_RESUME_DEPENDENCY_VERSIONS = {
    "torch": "2.8.0+cu128",
    "accelerate": "1.14.0",
    "datasets": "3.1.0",
    "transformers": "4.52.4",
    "PyYAML": "6.0.3",
}


def _require_file_sha256(path: Path, expected: str, label: str) -> None:
    if not isinstance(expected, str) or not expected:
        raise ValueError(f"Resume manifest lacks {label} hash")
    if not path.is_file():
        raise ValueError(f"Resume {label} is missing: {path}")
    if sha256_file(path) != expected:
        raise ValueError(f"Resume {label} hash mismatch")


@dataclass(frozen=True)
class ResumeCheckpoint:
    path: str
    model_path: str
    completed_steps: int
    global_data_cursor: int
    source_run_dir: str


_CHECKPOINT_STATE_RE = re.compile(r"training-state-(\d{6})")


# Keep this in sync with utils.checkpointing.checkpoint._MODEL_PENDING_MARKER.
_MODEL_PENDING_MARKER = ".model-pending"


def _regular_names(directory: Path, pattern: str) -> set[str]:
    return {
        path.name
        for path in directory.glob(pattern)
        if path.is_file() and not path.is_symlink()
    }


def _checkpoint_pair_is_complete(
    output_dir: Path, state_path: Path, world_size: Optional[int] = None
) -> bool:
    """Return whether a numbered state has a published model counterpart."""
    if world_size is not None and world_size <= 0:
        return False
    match = _CHECKPOINT_STATE_RE.fullmatch(state_path.name)
    if (
        match is None
        or state_path.is_symlink()
        or not state_path.is_dir()
        or Path(f"{state_path}.incomplete").exists()
    ):
        return False

    model_path = output_dir / f"ckpt-{match.group(1)}"
    if (
        model_path.is_symlink()
        or not model_path.is_dir()
        or Path(f"{model_path}.incomplete").exists()
    ):
        return False
    if not (model_path / "config.json").is_file():
        return False
    if not any(
        (model_path / name).is_file()
        for name in ("model.safetensors.index.json", "model.safetensors")
    ):
        return False
    if not (model_path / "tokenizer_config.json").is_file():
        return False
    if not any(
        (model_path / name).is_file()
        for name in (
            "tokenizer.json",
            "tokenizer.model",
            "spiece.model",
            "vocab.json",
            "vocab.txt",
        )
    ):
        return False

    for directory_name in ("pytorch_model_fsdp_0", "optimizer_0"):
        shard_dir = state_path / directory_name
        metadata = shard_dir / ".metadata"
        if not metadata.is_file() or metadata.is_symlink():
            return False
        actual_shards = _regular_names(shard_dir, "__*_0.distcp")
        if not actual_shards:
            return False
        if world_size is not None:
            expected = {f"__{rank}_0.distcp" for rank in range(world_size)}
            if actual_shards != expected:
                return False

    actual_rng = _regular_names(state_path, "random_states_*.pkl")
    if not actual_rng:
        return False
    return (
        world_size is None
        or actual_rng == {
            f"random_states_{rank}.pkl" for rank in range(world_size)
        }
    )


def _latest_marker_target(output_dir: Path) -> Optional[Path]:
    """Read the optional atomic ``latest`` marker used by checkpoint saves."""
    marker = output_dir / "latest"
    if marker.is_symlink():
        try:
            candidate = marker.resolve()
        except OSError:
            return None
    elif marker.is_file():
        try:
            value = marker.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        if not value:
            return None
        candidate = (output_dir / value).resolve()
    else:
        return None
    if candidate.parent != output_dir:
        return None
    return candidate


def _remove_pending_checkpoint_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _cleanup_published_pending_states(
    output_dir: Path, entries: tuple[Path, ...]
) -> None:
    """Remove only state directories explicitly marked as model-pending."""
    latest_marker = output_dir / "latest"
    latest_target = _latest_marker_target(output_dir)
    for state_path in entries:
        match = _CHECKPOINT_STATE_RE.fullmatch(state_path.name)
        if (
            match is None
            or state_path.is_symlink()
            or not state_path.is_dir()
        ):
            continue
        pending_marker = state_path / _MODEL_PENDING_MARKER
        if pending_marker.is_symlink() or not pending_marker.is_file():
            continue
        model_path = output_dir / f"ckpt-{match.group(1)}"
        if model_path.exists() or model_path.is_symlink():
            continue

        state_target = state_path.resolve()
        if latest_target == state_target and (
            latest_marker.is_symlink() or latest_marker.is_file()
        ):
            latest_marker.unlink()
        _remove_pending_checkpoint_path(state_path)
        for stale_path in (
            Path(f"{state_path}.incomplete"),
            Path(f"{model_path}.incomplete"),
        ):
            if stale_path.is_symlink() or stale_path.exists():
                _remove_pending_checkpoint_path(stale_path)


def find_latest_training_state(
    output_dir: str | Path, *, world_size: Optional[int] = None
) -> Optional[str]:
    """Find the highest complete state in one run directory.

    The numeric scan is deliberate: a stale/missing ``latest`` marker or an
    interrupted save must never make an older run look newer than a complete
    checkpoint.
    """
    root = Path(output_dir).expanduser().resolve()
    if not root.is_dir():
        return None

    if world_size is not None and world_size <= 0:
        raise ValueError("world_size must be positive")
    candidates = []
    entries = tuple(root.iterdir())
    _cleanup_published_pending_states(root, entries)
    entries = tuple(root.iterdir())
    marker_target = _latest_marker_target(root)
    if marker_target is not None and _checkpoint_pair_is_complete(root, marker_target, world_size):
        candidates.append((int(marker_target.name.rsplit("-", 1)[-1]), marker_target))
    for candidate in entries:
        match = _CHECKPOINT_STATE_RE.fullmatch(candidate.name)
        if match is None or not _checkpoint_pair_is_complete(root, candidate, world_size):
            continue
        candidates.append((int(match.group(1)), candidate.resolve()))
    if candidates:
        return str(max(candidates, key=lambda item: item[0])[1])

    has_checkpoint_artifact = any(
        re.fullmatch(r"(?:training-state|ckpt)-\d{6}(?:\.incomplete)?", item.name)
        or item.name == "latest"
        for item in entries
    )
    if has_checkpoint_artifact:
        raise ValueError(
            f"Run directory contains checkpoint artifacts but no complete state/model pair: {root}"
        )
    return None


def _resolve_resume_state_path(
    resume_ckpt: str, world_size: Optional[int] = None
) -> Path:
    """Resolve a state path, ``latest`` marker, or run directory."""
    path = Path(resume_ckpt).expanduser()
    # Resolve the marker before following a symlink to its state directory.
    if path.name == "latest":
        latest = find_latest_training_state(
            path.parent.resolve(), world_size=world_size
        )
        if latest is not None:
            return Path(latest)
    resolved = path.resolve()
    if resolved.is_dir() and _CHECKPOINT_STATE_RE.fullmatch(resolved.name) is None:
        latest = find_latest_training_state(resolved, world_size=world_size)
        if latest is None:
            raise ValueError(f"No complete training state found in run directory: {resolved}")
        return Path(latest)
    return resolved


def _manifest_integer(environment: Dict[str, str], key: str) -> int:
    try:
        return int(environment[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Resume manifest has invalid {key}") from exc


def _require_manifest_value(environment: Dict[str, str], key: str, expected: int) -> None:
    actual = _manifest_integer(environment, key)
    if actual != expected:
        raise ValueError(f"Resume {key} mismatch: checkpoint={actual}, current={expected}")


def _validate_model_checkpoint(path: Path) -> None:
    """Require the model/tokenizer pair consumed by a strict resume."""
    if Path(f"{path}.incomplete").exists():
        raise ValueError(f"Resume model checkpoint is incomplete: {path}.incomplete")
    if not path.is_dir():
        raise ValueError(f"Resume model checkpoint does not exist: {path}")

    config_path = path / "config.json"
    if not config_path.is_file():
        raise ValueError(f"Resume model checkpoint lacks config.json: {path}")
    if not any(
        (path / name).is_file()
        for name in ("model.safetensors.index.json", "model.safetensors")
    ):
        raise ValueError(f"Resume model checkpoint lacks model weights: {path}")
    if not (path / "tokenizer_config.json").is_file():
        raise ValueError(f"Resume model checkpoint lacks tokenizer_config.json: {path}")
    if not any(
        (path / name).is_file()
        for name in (
            "tokenizer.json",
            "tokenizer.model",
            "spiece.model",
            "vocab.json",
            "vocab.txt",
        )
    ):
        raise ValueError(f"Resume model checkpoint lacks tokenizer vocabulary: {path}")

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Resume model checkpoint has invalid config.json: {path}") from exc
    auto_map = config.get("auto_map", {})
    if not isinstance(auto_map, dict):
        raise ValueError(f"Resume model checkpoint has invalid config auto_map: {path}")
    for value in auto_map.values():
        references = value if isinstance(value, list) else [value]
        for reference in references:
            if not isinstance(reference, str):
                continue
            module_name = reference.split("--")[-1].rsplit(".", 1)[0]
            code_path = path / Path(*module_name.split(".")).with_suffix(".py")
            if not code_path.is_file():
                raise ValueError(
                    f"Resume model checkpoint lacks custom code required by auto_map: "
                    f"{code_path}"
                )


def _validate_resume_output_dir(
    output_dir: str,
    state_path: Path,
    *,
    world_size: int,
) -> None:
    target = Path(output_dir).expanduser().resolve()
    source = state_path.parent.resolve()
    if target == source:
        latest = find_latest_training_state(target, world_size=world_size)
        if latest is not None and Path(latest).resolve() != state_path.resolve():
            raise ValueError(
                "Same-run resume must use the latest checkpoint; use a new "
                "run directory to branch from an older state"
            )
        return

    if not target.exists():
        return
    if (target / MANIFEST_FILENAME).is_file():
        raise ValueError(
            "Cannot resume into an output directory that already has a "
            "different run manifest; choose a new run directory"
        )
    existing_entries = tuple(entry for entry in target.iterdir() if entry.name != ".cache")
    if existing_entries:
        raise ValueError("External resume requires a new, empty output directory")


def validate_resume_checkpoint(
    resume_ckpt: str,
    *,
    world_size: int,
    nnodes: int,
    nproc_per_node: int,
    batch_size_per_device: int,
    grad_accumulation: int,
    total_steps: int,
    save_every: int,
    seed: int,
    learning_rate: float,
    weight_decay: float,
    max_grad_norm: float,
    dataset: str,
    dataset_path: Optional[str],
    model_family: str,
    mask_id: Optional[int],
    method: str,
    mtp_block_size: int,
    mtp_confidence: str,
    mtp_threshold: float,
    config_path: str,
    num_generations: int,
    repeat_times: int,
    gen_steps: int,
    gen_length: int,
    selective_logits: bool,
    activation_checkpointing_strategy: str,
    output_dir: str,
    format_reward_alpha: float = 1.0,
    save_model_only: bool = False,
    train_schedule: Optional[list[dict]] = None,
    sdar_native_block_size: Optional[int] = None,
) -> ResumeCheckpoint:
    """Validate a complete state and derive its data cursor.

    ``resume_ckpt`` may be a numbered state, the run's ``latest`` marker, or
    the run directory itself.  A resumed run may keep the same output directory;
    its immutable manifest is validated before any new files are written.
    """
    state_path = _resolve_resume_state_path(resume_ckpt, world_size=world_size)
    match = _CHECKPOINT_STATE_RE.fullmatch(state_path.name)
    if match is None:
        raise ValueError(
            "--resume_ckpt must point to training-state-NNNNNN, not model-only ckpt-NNNNNN"
        )
    if not state_path.is_dir():
        raise ValueError(f"Resume training state does not exist: {state_path}")
    if Path(f"{state_path}.incomplete").exists():
        raise ValueError(f"Resume training state is incomplete: {state_path}.incomplete")
    _validate_resume_output_dir(output_dir, state_path, world_size=world_size)
    completed_steps = int(match.group(1))
    if completed_steps <= 0 or completed_steps > total_steps:
        raise ValueError(
            f"Resume completed_steps must be in [1, {total_steps}], got {completed_steps}"
        )

    manifest_path = state_path.parent / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise ValueError(f"Resume manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Resume manifest is invalid: {manifest_path}") from exc
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported resume manifest schema")

    environment = manifest.get("environment")
    external_inputs = manifest.get("external_inputs")
    if not all(isinstance(item, dict) for item in (environment, external_inputs)):
        raise ValueError("Resume manifest lacks environment or inputs")
    if dataset not in ("code", "gsm8k", "math"):
        raise ValueError(f"Resume only supports code/gsm8k/math, got {dataset}")
    effective_config = manifest.get("effective_config")
    if effective_config is not None:
        if not isinstance(effective_config, dict):
            raise ValueError("Resume manifest has invalid effective_config")
        if effective_config.get("lora", False):
            raise ValueError("Resume requires a full-parameter training checkpoint")
        expected_config = {
            "model_family": model_family,
            "mask_id": mask_id,
            "method": method,
            "mtp_block_size": mtp_block_size,
            "mtp_confidence": mtp_confidence,
            "mtp_threshold": mtp_threshold,
            "batch_size_per_device": batch_size_per_device,
            "grad_accumulation": grad_accumulation,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "max_grad_norm": max_grad_norm,
            "seed": seed,
            "num_generations": num_generations,
            "repeat_times": repeat_times,
            "gen_steps": gen_steps,
            "gen_length": gen_length,
            "selective_logits": selective_logits,
            "activation_checkpointing_strategy": activation_checkpointing_strategy,
            "dataset": dataset,
            "train_schedule": train_schedule,
            "sdar_native_block_size": sdar_native_block_size,
            "save_model_only": save_model_only,
        }
        if dataset == "code":
            expected_config["format_reward_alpha"] = format_reward_alpha
        for key, expected in expected_config.items():
            actual = effective_config.get(key)
            if key in ("format_reward_alpha", "save_model_only") and key not in effective_config:
                actual = 1.0 if key == "format_reward_alpha" else False
            if actual != expected:
                raise ValueError(
                    f"Resume config mismatch for {key}: checkpoint={actual}, current={expected}"
                )
    source_hashes = manifest.get("source_sha256")
    if not isinstance(source_hashes, dict):
        raise ValueError("Resume manifest lacks source hashes")
    repo_root = Path(__file__).resolve().parents[2]
    for relative in training_source_files(dataset):
        _require_file_sha256(repo_root / relative, source_hashes.get(relative), relative)
    recorded_versions = manifest.get("dependency_versions")
    for package, legacy_version in _RESUME_DEPENDENCY_VERSIONS.items():
        expected_version = (
            recorded_versions.get(package)
            if isinstance(recorded_versions, dict)
            else legacy_version
        )
        if not isinstance(expected_version, str) or not expected_version:
            raise ValueError(f"Resume manifest has invalid dependency version: {package}")
        try:
            actual_version = version(package)
        except PackageNotFoundError as exc:
            raise ValueError(f"Resume dependency is missing: {package}") from exc
        if actual_version != expected_version:
            raise ValueError(
                f"Resume {package} mismatch: checkpoint={expected_version}, current={actual_version}"
            )

    from utils.runtime.rng import derive_seed

    rng_answer = derive_seed(
        1234,
        stream="gumbel-v1",
        optimizer_step=7,
        sample_id="gsm8k:42",
        repeat_id=1,
        generation_id=3,
    )
    if rng_answer != _ROLLOUT_RNG_KNOWN_ANSWER:
        raise ValueError("Current rollout RNG known-answer is incompatible with resume state")

    recorded_total_steps = _manifest_integer(environment, "TOTAL_STEPS")
    if total_steps < recorded_total_steps:
        raise ValueError(
            "Resume TOTAL_STEPS cannot be reduced: "
            f"checkpoint={recorded_total_steps}, current={total_steps}"
        )

    expected_values = {
        "NNODES": nnodes,
        "NPROC_PER_NODE": nproc_per_node,
        "GRAD_ACCUM": grad_accumulation,
        "SEED": seed,
        "NUM_GENERATIONS": num_generations,
        "REPEAT_TIMES": repeat_times,
        "GEN_STEPS": gen_steps,
        "GEN_LENGTH": gen_length,
    }
    for key, expected in expected_values.items():
        _require_manifest_value(environment, key, expected)
    if nnodes * nproc_per_node != world_size:
        raise ValueError(f"Current topology {nnodes}x{nproc_per_node} != world_size {world_size}")

    global_prompt_batch = world_size * grad_accumulation * batch_size_per_device
    global_completion_batch = global_prompt_batch * num_generations * repeat_times
    _require_manifest_value(environment, "EXPECTED_GLOBAL_PROMPTS", global_prompt_batch)
    _require_manifest_value(environment, "EXPECTED_GLOBAL_BSZ", global_completion_batch)

    try:
        recorded_lr = Decimal(str(environment["LR"]))
        current_lr = Decimal(str(learning_rate))
    except (KeyError, InvalidOperation) as exc:
        raise ValueError("Resume manifest has invalid LR") from exc
    if recorded_lr != current_lr:
        raise ValueError(f"Resume LR mismatch: checkpoint={recorded_lr}, current={current_lr}")

    recorded_dataset = environment.get("DATASET", "")
    if not recorded_dataset:
        if "code_dataset" in external_inputs:
            recorded_dataset = "code"
        elif "gsm8k_dataset" in external_inputs:
            recorded_dataset = "gsm8k"
        elif "math_dataset" in external_inputs:
            recorded_dataset = "math"
    if recorded_dataset != dataset:
        raise ValueError(
            f"Resume dataset mismatch: checkpoint={recorded_dataset}, current={dataset}"
        )

    dataset_key = {
        "code": "code_dataset",
        "gsm8k": "gsm8k_dataset",
        "math": "math_dataset",
    }[dataset]
    recorded_dataset_input = external_inputs.get(dataset_key)
    if not isinstance(recorded_dataset_input, dict) or not dataset_path:
        raise ValueError(f"Resume manifest lacks {dataset_key} identity")
    _require_file_sha256(
        Path(dataset_path).expanduser().resolve(),
        recorded_dataset_input.get("sha256"), dataset_key,
    )

    recorded_config = external_inputs.get("accelerate_config")
    if not isinstance(recorded_config, dict):
        raise ValueError("Resume manifest lacks accelerate_config identity")
    _require_file_sha256(
        Path(config_path).expanduser().resolve(),
        recorded_config.get("sha256"), "accelerate_config",
    )

    model_path = state_path.parent / f"ckpt-{completed_steps:06d}"
    _validate_model_checkpoint(model_path)

    for directory_name in ("pytorch_model_fsdp_0", "optimizer_0"):
        shard_dir = state_path / directory_name
        metadata = shard_dir / ".metadata"
        if not metadata.is_file() or metadata.is_symlink():
            raise ValueError(f"Resume state lacks {directory_name}/.metadata")
        expected_shards = {f"__{rank}_0.distcp" for rank in range(world_size)}
        actual_shards = _regular_names(shard_dir, "__*_0.distcp")
        if actual_shards != expected_shards:
            raise ValueError(
                f"Resume {directory_name} shards are incomplete for world_size={world_size}"
            )
    expected_rng = {f"random_states_{rank}.pkl" for rank in range(world_size)}
    actual_rng = _regular_names(state_path, "random_states_*.pkl")
    if actual_rng != expected_rng:
        raise ValueError(f"Resume RNG states are incomplete for world_size={world_size}")

    return ResumeCheckpoint(
        path=str(state_path),
        model_path=str(model_path),
        completed_steps=completed_steps,
        global_data_cursor=completed_steps * global_prompt_batch,
        source_run_dir=str(state_path.parent),
    )
