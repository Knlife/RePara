"""Memory-conscious, atomic checkpoint helpers for distributed training."""

import gc
import inspect
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch


_MODEL_PENDING_MARKER = ".model-pending"


def _auto_map_references(model: Any) -> Iterable[str]:
    auto_map = getattr(getattr(model, "config", None), "auto_map", None)
    if not isinstance(auto_map, dict):
        return ()

    references = []
    for value in auto_map.values():
        values = value if isinstance(value, (list, tuple)) else (value,)
        references.extend(item for item in values if isinstance(item, str))
    return references


def _copy_custom_model_code(
    model: Any,
    output_dir: Path,
    source_model_path: Optional[str],
) -> None:
    """Keep trust_remote_code checkpoints loadable without a separate base path."""
    references = tuple(_auto_map_references(model))
    if not references:
        return

    source_root = None
    if source_model_path is not None:
        candidate = Path(source_model_path).expanduser()
        if candidate.is_dir():
            source_root = candidate.resolve()

    model_values = [model, getattr(model, "config", None)]
    get_base_model = getattr(model, "get_base_model", None)
    if callable(get_base_model):
        base_model = get_base_model()
        model_values.extend((base_model, getattr(base_model, "config", None)))

    class_sources = {}
    for value in model_values:
        if value is None:
            continue
        try:
            source = Path(inspect.getfile(type(value))).resolve()
        except (OSError, TypeError):
            continue
        class_sources[source.name] = source

    for reference in references:
        module_name = reference.split("--")[-1].rsplit(".", 1)[0]
        relative_path = Path(*module_name.split(".")).with_suffix(".py")
        destination = output_dir / relative_path
        if destination.is_file():
            continue

        candidates = []
        if source_root is not None:
            candidates.extend(
                (source_root / relative_path, source_root / relative_path.name)
            )
        class_source = class_sources.get(relative_path.name)
        if class_source is not None:
            candidates.append(class_source)

        source = next((path for path in candidates if path.is_file()), None)
        if source is None:
            raise FileNotFoundError(
                f"Could not package custom model code required by auto_map: "
                f"{relative_path}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _checkpoint_paths(output_dir: str, completed_steps: int) -> Dict[str, Path]:
    if completed_steps <= 0:
        raise ValueError("completed_steps must be positive")

    output_path = Path(output_dir)
    state_path = output_path / f"training-state-{completed_steps:06d}"
    model_path = output_path / f"ckpt-{completed_steps:06d}"
    return {
        "state": state_path,
        "state_staging": Path(f"{state_path}.incomplete"),
        "state_pending_marker": state_path / _MODEL_PENDING_MARKER,
        "model": model_path,
        "model_staging": Path(f"{model_path}.incomplete"),
    }


def _remove_checkpoint_path(path: Path) -> None:
    """Remove one known checkpoint path, including a broken symlink."""
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _cleanup_stale_checkpoint_paths(
    paths: Dict[str, Path], *, save_model_only: bool
) -> None:
    """Drop staging/orphan paths left by an interrupted save of this step."""
    model_staging_exists = os.path.lexists(paths["model_staging"])
    for key in ("state_staging", "model_staging"):
        if os.path.lexists(paths[key]):
            _remove_checkpoint_path(paths[key])
    # A state published before model serialization is not resumable.  The marker
    # covers failures before model staging exists; the staging check keeps older
    # interrupted saves retryable.  A lone pre-existing state remains protected.
    if (
        not save_model_only
        and os.path.lexists(paths["state"])
        and not os.path.lexists(paths["model"])
        and (model_staging_exists or os.path.lexists(paths["state_pending_marker"]))
    ):
        _remove_checkpoint_path(paths["state"])


def _prune_old_training_states(output_dir: Path, keep: Path) -> None:
    """Remove only published state/model pairs."""
    for candidate in output_dir.iterdir():
        if candidate == keep or candidate.is_symlink() or not candidate.is_dir():
            continue
        match = re.fullmatch(r"training-state-(\d{6})", candidate.name)
        if match is None or Path(f"{candidate}.incomplete").exists():
            continue
        if (candidate / _MODEL_PENDING_MARKER).exists():
            continue
        model_path = output_dir / f"ckpt-{match.group(1)}"
        if (
            model_path.is_symlink()
            or not model_path.is_dir()
            or Path(f"{model_path}.incomplete").exists()
        ):
            # Keep an orphaned state: it may still contain the only usable
            # optimizer/RNG data and can be inspected or recovered manually.
            continue
        shutil.rmtree(candidate)
        shutil.rmtree(model_path)


def _update_latest_marker(output_dir: Path, state_path: Path) -> None:
    """Atomically point ``latest`` at a fully published training state."""
    marker = output_dir / "latest"
    temporary = output_dir / f".latest.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        os.symlink(state_path.name, temporary, target_is_directory=True)
        os.replace(temporary, marker)
    finally:
        if os.path.lexists(temporary):
            os.unlink(temporary)


def _raise_if_any_process_failed(
    accelerator: Any,
    error: Optional[Exception],
    operation: str,
) -> None:
    """Propagate a failure observed by any rank to every distributed rank."""
    failed = torch.tensor(
        int(error is not None),
        dtype=torch.int32,
        device=accelerator.device,
    )
    failed = accelerator.reduce(failed, reduction="max")
    if int(failed.item()) == 0:
        return
    if error is not None:
        raise error
    raise RuntimeError(f"A distributed rank failed while {operation}")


def _raise_if_main_process_failed(
    accelerator: Any,
    error: Optional[Exception],
    operation: str,
) -> None:
    """Propagate a rank-zero filesystem failure to every distributed rank."""
    _raise_if_any_process_failed(accelerator, error, operation)


def _abort_distributed_collectives() -> None:
    """Stop a multi-rank communicator after a collective failure."""
    dist = torch.distributed
    if not dist.is_available() or not dist.is_initialized():
        return
    try:
        if dist.get_world_size() > 1:
            dist.destroy_process_group()
    except Exception:
        pass


def save_training_checkpoint(
    accelerator: Any,
    model: Any,
    tokenizer: Any,
    output_dir: str,
    completed_steps: int,
    *,
    model_source_path: Optional[str] = None,
    save_model_only: bool = False,
    keep_latest_state_only: bool = False,
) -> Dict[str, Optional[str]]:
    """Save Accelerator and Hugging Face checkpoints without duplicate full states.

    FSDP full-state collection can require tens of GiB on rank 0. Saving the
    Accelerator state first and collecting the Hugging Face state only after it
    has been released avoids retaining two full model state dicts at once.
    Staging directories are atomically renamed so a failed save cannot look like
    a complete checkpoint.
    """

    if save_model_only and keep_latest_state_only:
        raise ValueError("keep_latest_state_only requires saving training state")

    paths = _checkpoint_paths(output_dir, completed_steps)
    cleanup_error = None
    if accelerator.is_main_process:
        try:
            _cleanup_stale_checkpoint_paths(paths, save_model_only=save_model_only)
        except Exception as exc:
            cleanup_error = exc
    _raise_if_main_process_failed(
        accelerator, cleanup_error, "cleaning interrupted checkpoint paths"
    )
    accelerator.wait_for_everyone()

    active_path_keys = ["model", "model_staging"]
    if not save_model_only:
        active_path_keys.extend(("state", "state_staging"))
    precheck_error = None
    for key in active_path_keys:
        path = paths[key]
        if os.path.lexists(path):
            precheck_error = FileExistsError(
                f"Refusing to overwrite checkpoint path: {path}"
            )
            break
    _raise_if_any_process_failed(
        accelerator, precheck_error, "checking checkpoint paths"
    )

    # Ensure every rank has completed the existence check before any rank starts
    # creating a staging directory on the shared filesystem.
    accelerator.wait_for_everyone()

    if not save_model_only:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        try:
            # save_state is collective under FSDP.  A rank-local exception
            # cannot safely be reported by a later reduce; abort the group and
            # let every launcher rank terminate instead of waiting indefinitely.
            accelerator.save_state(str(paths["state_staging"]))
        except BaseException:
            _abort_distributed_collectives()
            raise
        accelerator.wait_for_everyone()
        state_error = None
        if accelerator.is_main_process:
            try:
                paths["state_staging"].joinpath(_MODEL_PENDING_MARKER).touch(
                    exist_ok=False
                )
                os.replace(paths["state_staging"], paths["state"])
            except Exception as exc:
                state_error = exc
        _raise_if_main_process_failed(
            accelerator, state_error, "publishing Accelerator state"
        )

    # Drop temporary Python references held by checkpointing before gathering a
    # second full state dict for save_pretrained().
    gc.collect()
    try:
        # Full-state collection is also collective under FSDP.
        state_dict = accelerator.get_state_dict(model)
    except BaseException:
        _abort_distributed_collectives()
        raise
    save_error = None
    try:
        if accelerator.is_main_process:
            try:
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    str(paths["model_staging"]),
                    state_dict=state_dict,
                    safe_serialization=True,
                )
                _copy_custom_model_code(
                    unwrapped_model,
                    paths["model_staging"],
                    model_source_path,
                )
                tokenizer.save_pretrained(str(paths["model_staging"]))
                os.replace(paths["model_staging"], paths["model"])
                if not save_model_only:
                    # A complete model/state pair no longer needs the pending marker.
                    try:
                        paths["state_pending_marker"].unlink()
                    except FileNotFoundError:
                        pass
                    except OSError:
                        pass
            except Exception as exc:
                save_error = exc
    finally:
        del state_dict
        gc.collect()

    _raise_if_main_process_failed(
        accelerator, save_error, "publishing model checkpoint"
    )
    accelerator.wait_for_everyone()
    if not save_model_only:
        latest_error = None
        if accelerator.is_main_process:
            try:
                _update_latest_marker(Path(output_dir), paths["state"])
            except Exception as exc:
                latest_error = exc
        _raise_if_main_process_failed(
            accelerator, latest_error, "publishing latest checkpoint marker"
        )
    if keep_latest_state_only:
        prune_error = None
        if accelerator.is_main_process:
            try:
                _prune_old_training_states(Path(output_dir), paths["state"])
            except Exception as exc:
                prune_error = exc
        _raise_if_main_process_failed(
            accelerator, prune_error, "pruning old training states"
        )

    return {
        "state": None if save_model_only else str(paths["state"]),
        "model": str(paths["model"]),
    }
