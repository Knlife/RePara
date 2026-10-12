"""Fail clearly when a local model or one of its symlink targets is unmounted."""

import json
from pathlib import Path


def _require_readable_file(path):
    try:
        with path.open("rb") as handle:
            if not handle.read(1):
                raise ValueError(f"Empty model file: {path}")
    except OSError as exc:
        raise FileNotFoundError(
            f"Local model file is unavailable in this job: {path}; "
            f"resolved target: {path.resolve()}. Check container mounts and symlinks."
        ) from exc


def validate_local_model_path(model_path):
    """Check a local model without loading weights or allocating CUDA memory.

    Hub model identifiers keep their existing behavior. Resume runs are checked
    separately by the trainer against the resolved checkpoint, not the old base.
    """
    path = Path(model_path).expanduser()
    explicit_local = path.is_absolute() or str(model_path).startswith(("./", "../", "~"))
    if not explicit_local and not path.exists() and not path.is_symlink():
        return
    if not path.is_dir():
        raise FileNotFoundError(
            f"Local model directory is unavailable in this job: {path}; "
            f"resolved target: {path.resolve()}. Check container mounts and symlinks; "
            "this is a filesystem path, not a Hugging Face repo ID."
        )

    for name in ("config.json", "tokenizer_config.json"):
        _require_readable_file(path / name)
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    for reference in config.get("auto_map", {}).values():
        for item in reference if isinstance(reference, (list, tuple)) else (reference,):
            if isinstance(item, str):
                module = item.split("--")[-1].rsplit(".", 1)[0]
                _require_readable_file(path / (module.replace(".", "/") + ".py"))

    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = path / index_name
        if index_path.is_file() or index_path.is_symlink():
            _require_readable_file(index_path)
            index = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = index.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError(f"Missing weight_map in model index: {index_path}")
            for shard in set(weight_map.values()):
                _require_readable_file(path / shard)
            break
    else:
        weights = next((path / name for name in ("model.safetensors", "pytorch_model.bin")
                        if (path / name).is_file() or (path / name).is_symlink()), None)
        if weights is None:
            raise FileNotFoundError(f"No model weights found under {path}")
        _require_readable_file(weights)

    print(f"Local model preflight passed: {path} (resolved: {path.resolve()})")
