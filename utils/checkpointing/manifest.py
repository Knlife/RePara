"""Record the source, inputs, and runtime needed to reproduce training."""

from __future__ import annotations

import json
import hashlib
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping, Optional


MANIFEST_FILENAME = "manifest.json"
MANIFEST_SCHEMA_VERSION = "repara-training-manifest-v1"

TRAINING_SOURCE_FILES = (
    "train.py",
    "grpo.py",
    "utils/configuration/config.py",
    "utils/models/forward.py",
    "utils/generation/generate.py",
    "utils/runtime/rng.py",
    "utils/runtime/distributed.py",
    "utils/checkpointing/checkpoint.py",
    "utils/models/sdar_config.py",
    "utils/models/model_paths.py",
    "data/sampler.py",
)


def training_source_files(dataset: str) -> tuple[str, ...]:
    task_sources = (
        ("data/code.py", "data/local_sandbox.py", "utils/evaluation/worker.py")
        if dataset == "code"
        else ("data/math.py", "utils/evaluation/grader.py", "utils/evaluation/parser.py")
    )
    return TRAINING_SOURCE_FILES + task_sources


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

DEPENDENCY_PACKAGES = ("torch", "accelerate", "datasets", "transformers", "PyYAML")


def _git_output(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.rstrip("\n")


def _dependency_versions() -> dict[str, Optional[str]]:
    versions = {}
    for package in DEPENDENCY_PACKAGES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def _file_identity(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _model_identity(model_path: str) -> dict[str, Any]:
    path = Path(model_path).expanduser()
    if not path.is_dir():
        return {"kind": "model_identifier", "identifier": model_path}

    path = path.resolve()
    identity = {"kind": "local_directory", "path": str(path)}
    config = path / "config.json"
    if config.is_file():
        identity["config"] = _file_identity(config)
    return identity


def _dataset_identity(dataset: str, dataset_path: Optional[str]) -> dict[str, Any]:
    if dataset_path:
        path = Path(dataset_path).expanduser()
        if path.is_file():
            return _file_identity(path)
    return {"kind": "dataset_identifier", "identifier": dataset_path or dataset}


def _manifest_environment(
    config: Mapping[str, Any],
    *,
    world_size: int,
    nnodes: int,
    nproc_per_node: int,
    dataset_path: Optional[str],
    accelerate_config_path: Optional[str],
    environ: Mapping[str, str],
) -> dict[str, str]:
    global_prompts = (
        world_size
        * int(config["batch_size_per_device"])
        * int(config["grad_accumulation"])
    )
    global_completions = (
        global_prompts
        * int(config["num_generations"])
        * int(config["repeat_times"])
    )
    dataset = str(config["dataset"])
    values = {
        "TOPOLOGY": environ.get(
            "TOPOLOGY",
            (
                f"{nnodes}x{nproc_per_node}x"
                f"{config['batch_size_per_device']}x{config['grad_accumulation']}"
            ),
        ),
        "RUN_NAME": environ.get("RUN_NAME", ""),
        "RUN_DIR": str(Path(str(config["output_dir"])).expanduser().resolve()),
        "RESUME_CKPT": str(config.get("resume_ckpt") or ""),
        "MODEL_PATH": str(config["model_path"]),
        "DATASET": dataset,
        "CODE_DATA_PATH": dataset_path if dataset == "code" and dataset_path else "",
        "GSM8K_DATA_PATH": dataset_path if dataset == "gsm8k" and dataset_path else "",
        "MATH_DATA_PATH": dataset_path if dataset == "math" and dataset_path else "",
        "CONFIG_FILE": accelerate_config_path or "",
        "NNODES": str(nnodes),
        "NPROC_PER_NODE": str(nproc_per_node),
        "BATCH_SIZE_PER_DEVICE": str(config["batch_size_per_device"]),
        "GRAD_ACCUM": str(config["grad_accumulation"]),
        "TOTAL_STEPS": str(config["total_steps"]),
        "SAVE_EVERY": str(config["save_every"]),
        "NUM_GENERATIONS": str(config["num_generations"]),
        "REPEAT_TIMES": str(config["repeat_times"]),
        "GEN_STEPS": str(config["gen_steps"]),
        "GEN_LENGTH": str(config["gen_length"]),
        "EXPECTED_GLOBAL_PROMPTS": str(global_prompts),
        "EXPECTED_GLOBAL_BSZ": str(global_completions),
        "SEED": str(config["seed"]),
        "LR": str(config["learning_rate"]),
        "CONDA_ENV": str(Path(sys.prefix).resolve()),
        "HF_HUB_OFFLINE": environ.get("HF_HUB_OFFLINE", ""),
        "TRANSFORMERS_OFFLINE": environ.get("TRANSFORMERS_OFFLINE", ""),
        "CUBLAS_WORKSPACE_CONFIG": environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
        "FLASH_ATTENTION_DETERMINISTIC": environ.get(
            "FLASH_ATTENTION_DETERMINISTIC", ""
        ),
    }
    return values


def build_manifest(
    *,
    repo_root: Path,
    config: Mapping[str, Any],
    world_size: int,
    nnodes: int,
    nproc_per_node: int,
    dataset_path: Optional[str],
    accelerate_config_path: Optional[str],
    environ: Optional[Mapping[str, str]] = None,
) -> dict[str, Any]:
    """Build a manifest from the effective trainer config and runtime state."""
    repo_root = repo_root.expanduser().resolve()
    environ = os.environ if environ is None else environ

    dataset = str(config["dataset"])
    external_inputs: dict[str, Any] = {
        "model": _model_identity(str(config["model_path"])),
    }
    dataset_key = {
        "code": "code_dataset",
        "gsm8k": "gsm8k_dataset",
        "math": "math_dataset",
    }.get(dataset, "dataset")
    external_inputs[dataset_key] = _dataset_identity(dataset, dataset_path)
    if accelerate_config_path:
        config_path = Path(accelerate_config_path).expanduser()
        if config_path.is_file():
            external_inputs["accelerate_config"] = _file_identity(config_path)

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "python_prefix": str(Path(sys.prefix).resolve()),
        "dependency_versions": _dependency_versions(),
        "git_head": _git_output(repo_root, "rev-parse", "HEAD"),
        "git_status": _git_output(repo_root, "status", "--short"),
        "source_sha256": {
            relative: sha256_file(repo_root / relative)
            for relative in training_source_files(dataset)
        },
        "external_inputs": external_inputs,
        "effective_config": dict(config),
        "environment": _manifest_environment(
            config,
            world_size=world_size,
            nnodes=nnodes,
            nproc_per_node=nproc_per_node,
            dataset_path=dataset_path,
            accelerate_config_path=accelerate_config_path,
            environ=environ,
        ),
    }


def write_manifest(
    output: Path,
    **build_kwargs: Any,
) -> dict[str, Any]:
    """Atomically write one training manifest and return its payload."""
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = build_manifest(**build_kwargs)
    temporary = output.with_suffix(output.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return payload
