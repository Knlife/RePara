import os
import argparse
import json
import re
import uuid
import weakref
from datetime import datetime, timezone
import numpy as np
import torch
from dataclasses import asdict
from numbers import Real
from pathlib import Path
from tqdm import tqdm

import utils.runtime.distributed as dist
from grpo import (
    block_logprob_loss,
    compute_group_advantages,
    compute_pooled_loss_scale,
    logprob_loss,
    sample,
)
from utils.checkpointing.checkpoint import save_training_checkpoint
from utils.configuration.config import (
    ACTIVATION_CHECKPOINTING_STRATEGIES,
    EvalConfig,
    ONLINE_EVAL_FIELDS,
    TrainConfig,
    resolve_online_eval_config,
    load_config,
    snapshot_config,
)
from utils.models.forward import install_position_selective_output_head
from utils.models.sdar_config import configure_sdar_native_block_size
from utils.models.model_paths import validate_local_model_path
from utils.checkpointing.resume import (
    ResumeCheckpoint,
    find_latest_training_state,
    validate_resume_checkpoint,
)
from utils.runtime.rng import derive_learner_seed
from utils.checkpointing.manifest import MANIFEST_FILENAME, write_manifest
from utils.runtime.timing import TimingRecorder, reduce_max_timings
from utils.evaluation.online_eval import (
    online_evaluation_due,
    run_online_evaluation as run_checkpoint_evaluation,
    validate_online_eval_inputs,
)


REQUIRED_GLOBAL_BATCH_SIZE = 64


def validate_global_batch_size(config: TrainConfig, world_size: int) -> int:
    global_batch_size = (
        world_size * config.batch_size_per_device * config.grad_accumulation
    )
    if global_batch_size != REQUIRED_GLOBAL_BATCH_SIZE:
        raise ValueError(
            "Global prompt batch size must be 64: "
            f"world_size({world_size}) * "
            f"batch_size_per_device({config.batch_size_per_device}) * "
            f"grad_accumulation({config.grad_accumulation}) = {global_batch_size}"
        )
    return global_batch_size


def _split_prompt_batch(batch):
    prompt_count = len(batch.get("problems", ()))
    if prompt_count == 0:
        raise ValueError("Rollout batch must contain at least one prompt")

    result = []
    for prompt_index in range(prompt_count):
        prompt_batch = {}
        for key, values in batch.items():
            if len(values) != prompt_count:
                raise ValueError(
                    f"Batch field {key!r} has {len(values)} rows; expected {prompt_count}"
                )
            prompt_batch[key] = values[prompt_index : prompt_index + 1]
        result.append(prompt_batch)
    return result


def _organize_prompt_rollouts(prompt_rollouts):
    if not prompt_rollouts or not prompt_rollouts[0]:
        raise ValueError("Prompt rollouts must contain at least one repeat and prompt")

    prompt_count = len(prompt_rollouts[0])
    if any(len(repeat) != prompt_count for repeat in prompt_rollouts):
        raise ValueError("Every rollout repeat must contain the same prompts")

    reward_grid = torch.stack(
        [
            torch.stack([rollout["rewards"] for rollout in repeat], dim=1)
            for repeat in prompt_rollouts
        ],
        dim=0,
    )
    repeat_count, generation_count, _ = reward_grid.shape
    advantages = compute_group_advantages(
        reward_grid.reshape(-1), repeat_count * generation_count
    )
    advantage_grid = advantages.view_as(reward_grid)

    chunks = []
    for repeat_index, repeat in enumerate(prompt_rollouts):
        for prompt_index, rollout in enumerate(repeat):
            rollout["advantages"] = advantage_grid[
                repeat_index, :, prompt_index
            ]
            chunks.append(rollout)
    return chunks, advantages


def set_activation_checkpointing(model, strategy, *, model_family="llada"):
    """Configure the model's activation-checkpointing controllers."""
    if model_family == "sdar":
        action = getattr(
            model,
            "gradient_checkpointing_enable" if strategy else "gradient_checkpointing_disable",
            None,
        )
        if callable(action):
            action()
            return True
        return False
    changed = False
    for module in model.modules():
        setter = getattr(module, "set_activation_checkpointing", None)
        if callable(setter):
            setter(strategy)
            changed = True
        elif hasattr(module, "activation_checkpointing_strategy"):
            module.activation_checkpointing_strategy = strategy
            changed = True
    return changed


def resolve_activation_checkpointing_strategy(strategy):
    """Map the CLI/config spelling to the value expected by LLaDA."""
    if strategy not in ACTIVATION_CHECKPOINTING_STRATEGIES:
        choices = ", ".join(ACTIVATION_CHECKPOINTING_STRATEGIES)
        raise ValueError(
            f"Unknown activation checkpointing strategy: {strategy!r}; choose from {choices}"
        )
    return None if strategy == "none" else strategy


def _raise_if_main_process_failed(accelerator, error, operation):
    """Make a failure observed by any rank visible on every rank."""
    failed = torch.tensor(
        int(error is not None), dtype=torch.int32, device=accelerator.device
    )
    failed = accelerator.reduce(failed, reduction="max")
    if int(failed.item()) == 0:
        return
    if error is not None:
        raise error
    raise RuntimeError(f"A distributed process failed while {operation}")


def _broadcast_main_value(value, rank):
    """Broadcast one rank-zero value, including structured errors."""
    raw_ready = (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
    )
    if not raw_ready:
        try:
            accelerator_world_size = dist.get_world_size()
        except (AttributeError, RuntimeError):
            accelerator_world_size = 1
        if accelerator_world_size > 1:
            raise RuntimeError(
                "Distributed process group is required for multi-rank resume"
            )
        return value
    if torch.distributed.get_world_size() == 1:
        return value
    # Use the process-group rank as the source of truth.  Accelerator and
    # torch.distributed ranks normally match, but custom launchers can expose
    # different logical rank values.
    source_rank = torch.distributed.get_rank()
    values = [value if source_rank == 0 else None]
    backend = str(torch.distributed.get_backend()).lower()
    if backend == "nccl":
        torch.distributed.broadcast_object_list(
            values, src=0, device=torch.device("cuda", torch.cuda.current_device())
        )
    else:
        torch.distributed.broadcast_object_list(values, src=0)
    return values[0]


def _validate_resume_output_dir(
    output_dir: str, checkpoint, *, explicit: bool, world_size: int
) -> None:
    """Reject accidental cross-run writes and same-run rewind branches."""
    target = Path(output_dir).expanduser().resolve()
    source = Path(checkpoint.source_run_dir).expanduser().resolve()
    if target == source:
        if explicit:
            latest = find_latest_training_state(target, world_size=world_size)
            if latest and Path(latest).resolve() != Path(checkpoint.path).resolve():
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
    existing_entries = tuple(
        entry for entry in target.iterdir() if entry.name != ".cache"
    )
    if existing_entries:
        raise ValueError("External resume requires a new, empty output directory")


def _append_resume_history(path: Path, checkpoint, target_total_steps: int) -> None:
    record = {
        "resumed_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": checkpoint.path,
        "completed_steps": checkpoint.completed_steps,
        "target_total_steps": target_total_steps,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _finish_wandb_after_error(run) -> None:
    """Best-effort close for an unhandled training exception."""
    try:
        run.finish(exit_code=1)
    except Exception:
        pass


_WANDB_IDENTITY_FILENAME = "wandb_identity.json"
_WANDB_IDENTITY_SCHEMA_VERSION = 1
_WANDB_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


_WANDB_MODES = frozenset(("online", "offline", "disabled", "shared"))


def _normalize_wandb_mode(value) -> str:
    mode = str(value or "").strip().lower()
    return {"dryrun": "offline", "run": "online"}.get(mode, mode)


def _requested_wandb_mode(config: TrainConfig) -> str:
    value = os.environ.get("WANDB_MODE")
    if value is None or not value.strip():
        value = config.wandb_mode
    return _normalize_wandb_mode(value)


def _wandb_effective_mode(run, requested_mode: str) -> str:
    """Use the SDK-resolved mode, falling back to the requested mode."""
    requested = _normalize_wandb_mode(requested_mode)
    try:
        settings = getattr(run, "settings", None)
        actual = _normalize_wandb_mode(getattr(settings, "mode", None))
    except Exception:
        actual = ""
    return actual if actual in _WANDB_MODES else requested


# These knobs describe the optimizer/data protocol and must not drift while
# one logical W&B run is resumed.  Step budget and observability settings are
# intentionally mutable so a continuation can extend or reconfigure logging.
_WANDB_PATH_CONFIG_FIELDS = frozenset(
    {"model_path", "gsm8k_data_path", "math_data_path", "code_data_path"}
)
_WANDB_MUTABLE_CONFIG_FIELDS = frozenset(
    {
        "output_dir",
        "total_steps",
        "log_every",
        "save_every",
        "eval_every",
        "eval_config",
        "resume_ckpt",
        "keep_latest_state_only",
        "wandb",
        "wandb_project",
        "wandb_run_name",
        "wandb_group",
        "wandb_mode",
        "wandb_dir",
        "profile_timing",
        "timing_warmup_steps",
    }
)


def _wandb_run_id(wandb_dir: str) -> str:
    """Read or create the legacy ``wandb/run_id`` file.

    The helper remains for backwards compatibility with old run directories;
    new runs persist their identity in the output root instead.
    """
    path = Path(wandb_dir) / "run_id"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        value = path.read_text(encoding="utf-8").strip()
        if value:
            return value
    run_id = uuid.uuid4().hex
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(run_id + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            value = path.read_text(encoding="utf-8").strip()
            if value:
                return value
            raise ValueError(f"W&B legacy run id is empty: {path}")
        return run_id
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _wandb_identity_path(output_dir: str | Path) -> Path:
    return Path(output_dir).expanduser().resolve() / _WANDB_IDENTITY_FILENAME


def _wandb_name(config: TrainConfig, output_root: Path) -> str:
    if config.wandb_run_name is not None and str(config.wandb_run_name).strip():
        return str(config.wandb_run_name).strip()
    # A deterministic fallback avoids W&B's random name changing on resume.
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", output_root.name).strip("-_.")
    return (name or "train")[:64]


def _wandb_config(config: TrainConfig) -> dict:
    values = asdict(config)
    if config.train_schedule is not None:
        values["train_schedule"] = config.normalized_train_schedule()
    if config.eval_config is not None:
        values["eval_config"] = resolve_online_eval_config(config)
    # Ensure the sidecar is JSON even if a caller supplied a Path-like eval
    # override. W&B receives the same normalized snapshot on every restart.
    return json.loads(json.dumps(values, default=str))


def _validate_wandb_run_id(value, path: Path) -> str:
    if not isinstance(value, str) or _WANDB_RUN_ID_PATTERN.fullmatch(value) is None:
        raise ValueError(f"Invalid W&B run id in {path}: {value!r}")
    return value


def _read_wandb_identity(path: Path) -> dict:
    try:
        identity = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"W&B identity is invalid: {path}") from exc
    if not isinstance(identity, dict):
        raise ValueError(f"W&B identity must be an object: {path}")
    if identity.get("schema_version") != _WANDB_IDENTITY_SCHEMA_VERSION:
        raise ValueError(f"Unsupported W&B identity schema: {path}")
    _validate_wandb_run_id(identity.get("run_id"), path)
    for field in ("project", "name"):
        if not isinstance(identity.get(field), str) or not identity[field].strip():
            raise ValueError(f"W&B identity has invalid {field}: {path}")
    if identity.get("group") is not None and not isinstance(identity["group"], str):
        raise ValueError(f"W&B identity has invalid group: {path}")
    if identity.get("entity") is not None and not isinstance(identity["entity"], str):
        raise ValueError(f"W&B identity has invalid entity: {path}")
    status = identity.get("status", "active")
    if status not in ("pending", "active"):
        raise ValueError(f"W&B identity has invalid status: {path}")
    if not isinstance(identity.get("config"), dict):
        raise ValueError(f"W&B identity lacks its initial config: {path}")
    wandb_dirs = identity.get("wandb_dirs", [])
    if not isinstance(wandb_dirs, list) or any(
        not isinstance(item, str) or not item.strip() for item in wandb_dirs
    ):
        raise ValueError(f"W&B identity has invalid wandb_dirs: {path}")
    fragments = identity.get("wandb_fragments", [])
    if not isinstance(fragments, list) or any(
        not isinstance(item, str) or not item.strip() for item in fragments
    ):
        raise ValueError(f"W&B identity has invalid wandb_fragments: {path}")
    return identity


def _wandb_config_value(key, value):
    if key in _WANDB_PATH_CONFIG_FIELDS and isinstance(value, str) and value:
        candidate = Path(value).expanduser()
        # Resolve explicit/recognizable local paths, while preserving Hub IDs
        # such as ``org/model`` as opaque strings.
        if value.startswith(("/", "~", ".")) or candidate.exists():
            return str(candidate.resolve())
    return value


def _validate_wandb_config(identity: dict, requested: dict, path: Path) -> None:
    stored = identity.get("config", {})
    current = requested.get("config", {})
    keys = (set(stored) | set(current)) - _WANDB_MUTABLE_CONFIG_FIELDS
    for key in sorted(keys):
        if _wandb_config_value(key, stored.get(key)) != _wandb_config_value(
            key, current.get(key)
        ):
            raise ValueError(
                f"W&B initial config mismatch for {key}: "
                f"stored={stored.get(key)!r}, current={current.get(key)!r}; "
                "use a new output directory for a changed training protocol"
            )


def _write_wandb_identity(path: Path, identity: dict, *, exclusive: bool) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(identity, indent=2, sort_keys=True, default=str) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive:
            try:
                # A hard link gives us create-if-absent semantics while the
                # fully written temporary file remains invisible to readers.
                os.link(temporary, path)
            except FileExistsError:
                return False
        else:
            temporary.replace(path)
        return True
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _resume_stage_index(output_root: Path) -> int:
    """Derive a stable continuation number from committed local history."""
    history = output_root / "resume_history.jsonl"
    try:
        return sum(1 for line in history.open(encoding="utf-8") if line.strip()) + 1
    except OSError:
        return 1


def _wandb_fragment_path(run) -> str | None:
    """Return the concrete local W&B fragment directory, if available."""
    try:
        value = getattr(run, "dir", None)
        if not value or not isinstance(value, (str, os.PathLike)):
            return None
        path = Path(value).expanduser().resolve()
    except (OSError, TypeError, ValueError):
        return None
    # W&B exposes ``<fragment>/files`` through Run.dir.  The parent is what
    # ``wandb beta sync`` accepts as one resumable fragment.
    return str(path.parent if path.name == "files" else path)


def _truncate_jsonl_after(path: Path, max_step: int) -> None:
    """Drop speculative metric/timing rows written after the last checkpoint."""
    if not path.is_file():
        return
    try:
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    except OSError:
        return
    kept = []
    changed = False
    for line in lines:
        try:
            row = json.loads(line)
            row_step = row.get("step") if isinstance(row, dict) else None
        except (TypeError, json.JSONDecodeError):
            row_step = None
        if isinstance(row_step, int) and row_step > max_step:
            changed = True
            continue
        kept.append(line)
    if not changed:
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text("".join(kept), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


_WANDB_STEP_WATERMARK_ATTR = "_repara_last_logged_step"
# W&B Run deliberately rejects arbitrary attributes. Keep the logical-step
# watermark outside the SDK object so real runs and lightweight test doubles
# follow the same fast path.
_WANDB_STEP_WATERMARKS = {}


def _drop_wandb_step_watermark(reference, key):
    item = _WANDB_STEP_WATERMARKS.get(key)
    if item is not None and item[0] is reference:
        _WANDB_STEP_WATERMARKS.pop(key, None)


def _set_wandb_step_watermark(run, step: int) -> None:
    """Cache the last logical step without querying W&B's blocking property."""
    key = id(run)
    reference = None
    owner = None
    try:
        current = _WANDB_STEP_WATERMARKS.get(key)
        if current is not None and current[0] is not None and current[0]() is run:
            reference = current[0]
        else:
            reference = weakref.ref(
                run, lambda ref, key=key: _drop_wandb_step_watermark(ref, key)
            )
    except TypeError:
        # A few lightweight objects are not weak-referenceable. Retain a
        # strong owner in that rare case so Python cannot reuse the id for a
        # different run while its watermark is still present.
        owner = run
    _WANDB_STEP_WATERMARKS[key] = (reference, owner, int(step))
    # Preserve compatibility for simple objects that permit attributes, but do
    # not rely on this: the official W&B Run rejects setattr.
    try:
        setattr(run, _WANDB_STEP_WATERMARK_ATTR, int(step))
    except Exception:
        pass


def _wandb_step_watermark(run):
    key = id(run)
    item = _WANDB_STEP_WATERMARKS.get(key)
    if item is not None:
        reference, owner, value = item
        if (reference is not None and reference() is run) or owner is run:
            return value
        _WANDB_STEP_WATERMARKS.pop(key, None)
    # Keep old lightweight test doubles working if they provide the legacy
    # attribute directly.
    try:
        value = getattr(run, _WANDB_STEP_WATERMARK_ATTR)
    except Exception:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _wandb_should_log_step(run, mode: str, step: int) -> bool:
    """Avoid replaying lower/duplicate steps using a local watermark."""
    if mode == "shared":
        return True
    current = _wandb_step_watermark(run)
    # The watermark is initialized once from the selected checkpoint and the
    # SDK's current history position, then updated only after a successful log.
    return current is None or int(step) > current


def _initialize_wandb_step_watermark(
    run, mode: str, resume_step: int | None
) -> None:
    """Read the SDK history step once, never once per training iteration."""
    watermark = int(resume_step) - 1 if resume_step is not None else -1
    if mode != "shared":
        try:
            current = getattr(run, "step")
        except Exception:
            current = None
        if isinstance(current, int) and not isinstance(current, bool):
            # Run.step is the *next* SDK history step.  Store the last
            # materialized step so a checkpoint at N can resume logging at N+1.
            watermark = max(watermark, current - 1)
    _set_wandb_step_watermark(run, watermark)


def _mark_wandb_step_logged(run, step: int) -> None:
    current = _wandb_step_watermark(run)
    if current is None or int(step) > current:
        _set_wandb_step_watermark(run, int(step))


def _configure_wandb_step_metric(run, mode: str) -> None:
    """Give shared-mode charts a stable optimizer-step x-axis."""
    if mode != "shared":
        return
    define_metric = getattr(run, "define_metric", None)
    if callable(define_metric):
        try:
            define_metric("*", step_metric="train/step")
        except Exception:
            # Logging remains useful on older SDKs; the explicit train/step
            # field is still available as a manual chart axis.
            pass


def _log_wandb(
    run, mode: str, payload: dict, step: int,
) -> bool:
    """Log one point with mode-correct step semantics and local de-duplication."""
    step = int(step)
    if mode != "shared" and not _wandb_should_log_step(run, mode, step):
        return False
    if mode == "shared":
        # Shared mode assigns its own monotonically increasing history step;
        # W&B ignores an explicit ``step`` there.
        run.log(payload, commit=True)
    else:
        run.log(payload, step=step, commit=True)
    _mark_wandb_step_logged(run, step)
    return True


def _log_wandb_stage(
    run, mode: str, output_root: Path, resume_step: int | None, target_total_steps: int,
    *, stage_index: int | None = None,
) -> None:
    """Record a continuation boundary after local metadata is committed."""
    if resume_step is None:
        return
    if stage_index is None:
        stage_index = _resume_stage_index(output_root)
    payload = {
        "train/step": int(resume_step),
        "run/stage": int(stage_index),
        "run/stage_start_step": int(resume_step),
        "run/target_total_steps": int(target_total_steps),
    }
    if mode == "shared" or _wandb_should_log_step(run, mode, int(resume_step)):
        _log_wandb(run, mode, payload, int(resume_step))
    # If the cloud history is already ahead of the checkpoint (for example, a
    # worker crashed after logging speculative metrics), do not consume another
    # SDK step for this marker: doing so would make the first resumed optimizer
    # metric at N+1 collide with the implicit history position.  The same fields
    # are always written to run.summary by init_wandb.


def _activate_wandb_identity(output_dir: str | Path) -> None:
    """Mark a successfully initialized run active after local metadata is safe."""
    path = _wandb_identity_path(output_dir)
    identity = _read_wandb_identity(path)
    if identity.get("status", "active") != "pending":
        return
    identity["status"] = "active"
    identity["activated_at_utc"] = datetime.now(timezone.utc).isoformat()
    if not _write_wandb_identity(path, identity, exclusive=False):
        raise RuntimeError(f"Could not activate W&B identity: {path}")


def _requested_wandb_identity(
    config: TrainConfig, output_root: Path, wandb_mode: str, wandb_dir: Path
) -> dict:
    # On a brand-new directory, standard WANDB_* identity variables are valid
    # initial values.  Once the sidecar exists, the validator below makes them
    # strict so a stale launcher environment cannot fork the run.
    def env_value(name):
        value = os.environ.get(name)
        return value.strip() if value and value.strip() else None

    project = env_value("WANDB_PROJECT") or config.wandb_project
    if not isinstance(project, str) or not project.strip():
        raise ValueError("wandb_project must be a non-empty string")
    name = env_value("WANDB_NAME") or _wandb_name(config, output_root)
    group = env_value("WANDB_RUN_GROUP")
    if group is None:
        group = config.wandb_group
    group = str(group).strip() if group and str(group).strip() else None
    entity = env_value("WANDB_ENTITY")
    run_id = env_value("WANDB_RUN_ID") or uuid.uuid4().hex
    _validate_wandb_run_id(run_id, _wandb_identity_path(output_root))
    return {
        "schema_version": _WANDB_IDENTITY_SCHEMA_VERSION,
        "run_id": run_id,
        "output_dir": str(output_root),
        "project": project.strip(),
        "entity": entity,
        "group": group,
        "name": name,
        "job_type": "train",
        "created_mode": wandb_mode,
        "status": "pending",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        # Keep every configured parent location discoverable when a
        # continuation changes WANDB_DIR (for example node-local scratch).
        "wandb_dirs": [str(wandb_dir)],
        "wandb_fragments": [],
        "config": _wandb_config(config),
    }


def _validate_wandb_identity_env(identity: dict) -> None:
    """Reject launch-time identity overrides that would split one logical run."""
    fields = {
        "WANDB_PROJECT": "project",
        "WANDB_NAME": "name",
        "WANDB_RUN_GROUP": "group",
        "WANDB_RUN_ID": "run_id",
        "WANDB_ENTITY": "entity",
    }
    for variable, field in fields.items():
        value = os.environ.get(variable)
        if value is None or not value.strip():
            continue
        expected = identity.get(field)
        if expected is None or value.strip() != str(expected).strip():
            raise ValueError(
                f"{variable} conflicts with the persisted W&B {field}; "
                "unset it or use a new output directory"
            )


def _validate_wandb_identity(identity: dict, requested: dict, path: Path) -> bool:
    """Validate identity fields and record mutable local fragment metadata."""
    _validate_wandb_config(identity, requested, path)
    stored_config = identity.get("config", {})
    requested_config = requested.get("config", {})
    identity_config_fields = {"project": "wandb_project", "name": "wandb_run_name"}
    for field in ("output_dir", "project", "name"):
        if identity.get(field) == requested.get(field):
            continue
        config_field = identity_config_fields.get(field)
        env_name = {
            "project": "WANDB_PROJECT",
            "name": "WANDB_NAME",
        }.get(field)
        configured_value = requested_config.get(config_field) if config_field else None
        stored_configured_value = stored_config.get(config_field) if config_field else None
        # An environment value may intentionally override the first launch's
        # YAML. If it is absent on a continuation and the YAML layer itself has
        # not changed, reuse the persisted identity instead of reverting to the
        # config default. Explicit changes still fail closed.
        can_inherit_env_identity = (
            config_field is not None
            and not (env_name and os.environ.get(env_name, "").strip())
            and (
                configured_value is None
                or configured_value == stored_configured_value
            )
        )
        if can_inherit_env_identity:
            requested[field] = identity[field]
            continue
        raise ValueError(
            f"W&B identity mismatch for {field}: "
            f"stored={identity.get(field)!r}, current={requested.get(field)!r}; "
            "use a new output directory for a new run"
        )
    # If the first launch resolved an account entity (from WANDB_ENTITY or the
    # server default), an omitted entity on continuation must keep that account.
    if requested.get("entity") is None and identity.get("entity") is not None:
        requested["entity"] = identity["entity"]
    elif (
        requested.get("entity") is not None
        and identity.get("entity") != requested["entity"]
    ):
        raise ValueError(
            "W&B identity mismatch for entity: "
            f"stored={identity.get('entity')!r}, current={requested['entity']!r}; "
            "use a new output directory for a new run"
        )

    stored_group = identity.get("group") or None
    requested_group = requested.get("group") or None
    changed = False
    if requested_group != stored_group and stored_group is not None:
        configured_group = requested_config.get("wandb_group")
        stored_configured_group = stored_config.get("wandb_group")
        env_group = os.environ.get("WANDB_RUN_GROUP", "").strip()
        if not env_group and (
            configured_group is None or configured_group == stored_configured_group
        ):
            # A first-launch WANDB_RUN_GROUP may override YAML. Reusing the
            # same YAML without repeating that environment value must preserve
            # the sidecar group rather than reverting to the config value.
            requested_group = stored_group
            requested["group"] = stored_group
    elif requested_group is None and stored_group is not None:
        # Omitting a group on a later launch means “keep the persisted group”,
        # not “remove it”; removal would split the run in the W&B UI.
        requested_group = stored_group
        requested["group"] = stored_group
    current_wandb_dir = (requested.get("wandb_dirs") or [None])[0]
    stored_wandb_dirs = identity.setdefault("wandb_dirs", [])
    if current_wandb_dir is not None and current_wandb_dir not in stored_wandb_dirs:
        stored_wandb_dirs.append(current_wandb_dir)
        changed = True
    if stored_group is None and requested_group is not None:
        # W&B groups are mutable run metadata. This supports adding a group to
        # an existing ungrouped run without changing its logical run id.
        identity["group"] = requested_group
        identity["group_updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        changed = True
    elif stored_group != requested_group:
        raise ValueError(
            f"W&B identity mismatch for group: stored={stored_group!r}, "
            f"current={requested_group!r}; use a new output directory for a new run"
        )
    return changed


def _legacy_wandb_run_id(config: TrainConfig, output_root: Path) -> str | None:
    # A custom W&B directory may belong to another run.  Migrate an external
    # sidecar only when the immutable manifest records that same directory.
    candidates = [output_root / "wandb" / "run_id"]
    if config.wandb_dir:
        custom_dir = Path(config.wandb_dir).expanduser().resolve()
        try:
            custom_dir.relative_to(output_root)
        except ValueError:
            try:
                manifest = json.loads(
                    (output_root / MANIFEST_FILENAME).read_text(encoding="utf-8")
                )
                recorded = manifest.get("effective_config", {}).get("wandb_dir")
                if recorded and Path(recorded).expanduser().resolve() == custom_dir:
                    candidates.append(custom_dir / "run_id")
            except (OSError, json.JSONDecodeError, AttributeError):
                pass
        else:
            candidates.append(custom_dir / "run_id")
    values = []
    for path in dict.fromkeys(candidates):
        if path.is_file():
            value = path.read_text(encoding="utf-8").strip()
            values.append(_validate_wandb_run_id(value, path))
    if len(set(values)) > 1:
        raise ValueError("W&B legacy run-id files disagree; remove the ambiguity")
    return values[0] if values else None


def _has_existing_run_artifacts(config: TrainConfig, output_root: Path) -> bool:
    """Return whether an output root already contains run progress.

    A missing identity in a progressed directory is ambiguous: silently
    creating a new id would split one training run into multiple W&B runs.
    Empty W&B directories and a sidecar-only pending launch are intentionally
    ignored so a failed initialization can be retried.
    """
    for name in (
        MANIFEST_FILENAME,
        "source_config.yaml",
        "effective_config.yaml",
        "resume_history.jsonl",
        "training_metrics.jsonl",
        "latest",
    ):
        path = output_root / name
        if path.is_file() and (name != "training_metrics.jsonl" or path.stat().st_size):
            return True

    try:
        entries = tuple(output_root.iterdir())
    except OSError:
        return True
    for entry in entries:
        if re.fullmatch(r"(?:training-state|ckpt)-\d{6}(?:\.incomplete)?", entry.name):
            return True

    wandb_dirs = [output_root / "wandb"]
    if config.wandb_dir:
        custom_dir = Path(config.wandb_dir).expanduser().resolve()
        try:
            custom_dir.relative_to(output_root)
        except ValueError:
            custom_dir = None
        if custom_dir is not None:
            wandb_dirs.append(custom_dir)
    for wandb_dir in dict.fromkeys(wandb_dirs):
        try:
            if wandb_dir.is_dir() and next(wandb_dir.iterdir(), None) is not None:
                return True
        except OSError:
            return True
    return False


def _manifest_disabled_wandb(output_root: Path) -> bool:
    """Whether an existing run explicitly disabled W&B at creation time."""
    try:
        manifest = json.loads(
            (output_root / MANIFEST_FILENAME).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(manifest.get("effective_config"), dict):
        return False
    effective = manifest["effective_config"]
    return effective.get("wandb") is False or str(
        effective.get("wandb_mode_resolved", "")
    ).strip().lower() == "disabled"


def _load_or_create_wandb_identity(
    config: TrainConfig, wandb_mode: str, wandb_dir: Path | None = None
) -> tuple[dict, bool, bool]:
    output_root = Path(config.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    path = _wandb_identity_path(output_root)
    wandb_dir = (
        wandb_dir
        if wandb_dir is not None
        else Path(config.wandb_dir or output_root / "wandb").expanduser().resolve()
    )
    requested = _requested_wandb_identity(
        config, output_root, wandb_mode, wandb_dir
    )

    if path.exists():
        identity = _read_wandb_identity(path)
        legacy_id = _legacy_wandb_run_id(config, output_root)
        if legacy_id is not None and legacy_id != identity["run_id"]:
            raise ValueError(
                "W&B identity disagrees with the legacy run-id file; "
                "remove the ambiguity"
            )
        # Apply a permitted group backfill before checking environment
        # overrides, so WANDB_RUN_GROUP can represent that same backfill.
        changed = _validate_wandb_identity(identity, requested, path)
        _validate_wandb_identity_env(identity)
        return identity, True, changed

    legacy_id = _legacy_wandb_run_id(config, output_root)
    if legacy_id is not None:
        requested["run_id"] = legacy_id
        requested["migrated_from"] = "legacy-wandb-run-id"
    elif _has_existing_run_artifacts(config, output_root):
        if _manifest_disabled_wandb(output_root):
            requested["started_after_wandb_disabled"] = True
        else:
            raise ValueError(
                "W&B identity is missing from an existing run directory; "
                "restore wandb_identity.json or choose a new output directory"
            )
    # Validate before creating the sidecar: a bad launcher environment must
    # not leave a half-initialized run identity behind.
    _validate_wandb_identity_env(requested)
    if _write_wandb_identity(path, requested, exclusive=True):
        return requested, legacy_id is not None, False

    # Another launcher won the exclusive create. Re-read and validate rather
    # than silently replacing its logical run id.
    identity = _read_wandb_identity(path)
    legacy_id = _legacy_wandb_run_id(config, output_root)
    if legacy_id is not None and legacy_id != identity["run_id"]:
        raise ValueError(
            "W&B identity disagrees with the legacy run-id file; "
            "remove the ambiguity"
        )
    changed = _validate_wandb_identity(identity, requested, path)
    _validate_wandb_identity_env(identity)
    return identity, True, changed


def _has_pending_checkpoint_artifact(output_dir: str | Path) -> bool:
    """Detect a state published with the explicit model-pending marker."""
    root = Path(output_dir).expanduser().resolve()
    if not root.is_dir():
        return False
    for state_path in root.iterdir():
        match = re.fullmatch(r"training-state-(\d{6})", state_path.name)
        if match is None or state_path.is_symlink() or not state_path.is_dir():
            continue
        marker = state_path / ".model-pending"
        model_path = root / f"ckpt-{match.group(1)}"
        if marker.is_file() and not marker.is_symlink() and not model_path.exists() and not model_path.is_symlink():
            return True
    return False


def _validate_run_resume_progress(
    output_dir: str | Path,
    has_checkpoint: bool,
    *,
    allow_pending_checkpoint_retry: bool = False,
) -> None:
    """Reject an accidental step-zero restart in an existing run directory."""
    if has_checkpoint:
        return
    root = Path(output_dir).expanduser().resolve()

    # A pending identity denotes a launch that has not committed run metadata
    # yet.  It is intentionally retryable; an active identity proves that the
    # previous launch reached the training boundary and must not restart at 0.
    identity_pending = False
    identity_path = root / _WANDB_IDENTITY_FILENAME
    identity_present = identity_path.is_file()
    if identity_present:
        try:
            identity_pending = (
                json.loads(identity_path.read_text(encoding="utf-8")).get("status")
                == "pending"
            )
        except (OSError, json.JSONDecodeError, AttributeError):
            identity_pending = False
    retryable_orphan = (
        allow_pending_checkpoint_retry
        and identity_present
        and not identity_pending
    )
    if identity_present and not identity_pending and not retryable_orphan:
        raise ValueError(
            "Run directory has an active W&B identity but no resumable checkpoint; "
            "resume from a complete state or choose a new output directory"
        )

    metrics_path = root / "training_metrics.jsonl"
    if metrics_path.is_file() and metrics_path.stat().st_size > 0:
        raise ValueError(
            "Run directory has prior training metrics but no resumable checkpoint; "
            "resume from a complete state or choose a new output directory"
        )

    # A pending launch may have written the immutable pre-training snapshots
    # before it failed.  Continuation/history/timing files, however, prove that
    # an optimizer step already ran and are never retryable without a state.
    if any(
        (root / name).is_file()
        for name in ("resume_history.jsonl", "timing.jsonl", "timing_summary.json")
    ):
        raise ValueError(
            "Run directory has prior run metadata but no resumable checkpoint; "
            "resume from a complete state or choose a new output directory"
        )
    if not identity_pending and not retryable_orphan and (root / MANIFEST_FILENAME).is_file():
        raise ValueError(
            "Run directory has a manifest but no resumable checkpoint; "
            "resume from a complete state or choose a new output directory"
        )
    if not identity_pending and not retryable_orphan and any(
        (root / name).is_file()
        for name in ("source_config.yaml", "effective_config.yaml")
    ):
        raise ValueError(
            "Run directory has prior run metadata but no resumable checkpoint; "
            "resume from a complete state or choose a new output directory"
        )

    try:
        wandb_root_has_files = (
            (root / "wandb").is_dir()
            and next((root / "wandb").iterdir(), None) is not None
        )
    except OSError:
        wandb_root_has_files = True
    if wandb_root_has_files and not identity_pending and not retryable_orphan:
        raise ValueError(
            "Run directory has W&B artifacts but no resumable checkpoint; "
            "resume from a complete state or choose a new output directory"
        )


def init_wandb(
    config: TrainConfig,
    *,
    resume: bool = False,
    resume_step: int | None = None,
    activate: bool = True,
):
    if not config.wandb:
        return None
    wandb_mode = _requested_wandb_mode(config)
    if wandb_mode not in _WANDB_MODES:
        raise ValueError(
            "WANDB_MODE must be one of online, offline, disabled, or shared"
        )
    # Disabled means no W&B run or local identity should be created.
    if wandb_mode == "disabled":
        return None
    if (
        (not isinstance(config.wandb_project, str) or not config.wandb_project.strip())
        and not os.environ.get("WANDB_PROJECT", "").strip()
    ):
        raise ValueError("wandb_project must be a non-empty string")

    wandb_dir = Path(
        config.wandb_dir or os.path.join(config.output_dir, "wandb")
    ).expanduser().resolve()
    # Point the SDK at the directory before initialization, but do not create
    # an empty W&B tree until identity validation has succeeded.
    os.environ["WANDB_MODE"] = wandb_mode
    os.environ["WANDB_DIR"] = str(wandb_dir)
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("wandb logging requested but wandb is not installed") from exc

    identity, identity_exists, identity_dirty = _load_or_create_wandb_identity(
        config, wandb_mode, wandb_dir
    )
    wandb_dir.mkdir(parents=True, exist_ok=True)

    wandb_kwargs = {
        "project": identity["project"],
        "name": identity["name"],
        "group": identity.get("group"),
        "job_type": identity.get("job_type", "train"),
        "config": identity["config"],
        "dir": str(wandb_dir),
        "mode": wandb_mode,
        "id": identity["run_id"],
    }
    if identity.get("entity") is not None:
        wandb_kwargs["entity"] = identity["entity"]
    # Only an identity that already belongs to this output directory can be
    # resumed.  Importing a checkpoint into a new directory starts a new W&B
    # run (the checkpoint lineage is tracked by the local manifest).  W&B
    # ignores ``resume`` in offline mode, so do not pass it there.
    if wandb_mode in ("online", "shared") and identity_exists:
        wandb_kwargs["resume"] = "allow"
    # Keep a stale WANDB_RESUME value from overriding this explicit lifecycle
    # decision, while restoring the caller's environment after initialization.
    previous_resume_env = os.environ.pop("WANDB_RESUME", None)
    try:
        run = wandb.init(**wandb_kwargs)
    finally:
        if previous_resume_env is not None:
            os.environ["WANDB_RESUME"] = previous_resume_env

    effective_mode = _wandb_effective_mode(run, wandb_mode)
    _configure_wandb_step_metric(run, effective_mode)
    expected_values = {
        "id": identity["run_id"],
        "project": identity["project"],
        "name": identity["name"],
        "group": identity.get("group"),
        "entity": identity.get("entity"),
    }
    mismatches = {}
    for field, expected in expected_values.items():
        actual = getattr(run, field, None)
        if field in ("id", "project", "name"):
            if not isinstance(actual, str) or not actual.strip() or actual != expected:
                mismatches[field] = (actual, expected)
            continue
        if field == "entity" and expected is None:
            # W&B may resolve the default entity only after init.
            if actual is not None and not isinstance(actual, str):
                mismatches[field] = (actual, expected)
            continue
        if field == "entity" and effective_mode == "offline" and actual in (None, ""):
            continue
        if field == "group" and actual in (None, "") and expected is None:
            continue
        if not isinstance(actual, str) or actual != expected:
            mismatches[field] = (actual, expected)
    if mismatches:
        try:
            run.finish(exit_code=1)
        except Exception:
            pass
        raise RuntimeError(
            "W&B returned metadata that differs from the persisted identity: "
            f"{mismatches}"
        )

    # When no entity was configured, W&B resolves the account server-side.
    # Persist that resolution so a later restart cannot silently use a
    # different account's default entity.
    resolved_entity = getattr(run, "entity", None)
    if (
        effective_mode in ("online", "shared")
        and identity.get("entity") is None
        and isinstance(resolved_entity, str)
        and resolved_entity.strip()
    ):
        identity["entity"] = resolved_entity.strip()
        identity["entity_resolved_at_utc"] = datetime.now(timezone.utc).isoformat()
        identity_dirty = True

    # Start one logical continuation just before its checkpoint step.  This
    # lets the stage marker use that step and makes subsequent logging strict-
    # monotonic without reading Run.step from the SDK on every call.
    _initialize_wandb_step_watermark(run, effective_mode, resume_step)

    fragment = _wandb_fragment_path(run)
    if fragment is not None:
        fragments = identity.setdefault("wandb_fragments", [])
        if fragment not in fragments:
            fragments.append(fragment)
            identity_dirty = True

    # Keep a newly reserved identity pending until the caller has committed
    # its local run metadata.  Standalone callers retain the old convenient
    # behavior through activate=True; the training entry point activates only
    # after its manifest/config write succeeds.
    if activate and identity.get("status", "active") == "pending":
        identity["status"] = "active"
        identity["activated_at_utc"] = datetime.now(timezone.utc).isoformat()
        identity_dirty = True

    if identity_dirty:
        try:
            _write_wandb_identity(
                _wandb_identity_path(config.output_dir),
                identity,
                exclusive=False,
            )
        except Exception as exc:
            try:
                run.finish(exit_code=1)
            except Exception:
                pass
            raise RuntimeError(
                "Could not persist the updated W&B identity"
            ) from exc

    run_metadata = {
        "run/identity_reused": int(identity_exists),
        "run/checkpoint_resumed": int(resume or resume_step is not None),
        "run/target_total_steps": config.total_steps,
        **(
            {
                "run/resume_from_step": int(resume_step),
                "run/stage": int(_resume_stage_index(Path(config.output_dir).expanduser().resolve())),
                "run/stage_start_step": int(resume_step),
            }
            if resume_step is not None
            else {}
        ),
    }
    summary = getattr(run, "summary", None)
    summary_update = getattr(summary, "update", None)
    try:
        if callable(summary_update):
            summary_update(run_metadata)
        # Summary values are overwritten on each continuation.  Standalone
        # callers can still request an immediate stage point; training defers
        # it until its manifest/history write has succeeded below.
        if activate and callable(getattr(run, "log", None)):
            _log_wandb_stage(
                run,
                effective_mode,
                Path(config.output_dir).expanduser().resolve(),
                resume_step,
                config.total_steps,
            )
    except Exception as exc:
        try:
            run.finish(exit_code=1)
        except Exception:
            pass
        raise RuntimeError("Could not write W&B run metadata") from exc
    return run


def resolve_mask_id(config: TrainConfig, model, tokenizer):
    if config.mask_id is not None:
        return config.mask_id

    for value in (
        getattr(getattr(model, "config", None), "mask_token_id", None),
        getattr(tokenizer, "mask_token_id", None),
    ):
        if value is not None:
            return int(value)

    if config.model_family == "llada":
        return 126336
    if config.model_family == "sdar":
        return 151669

    raise ValueError(
        f"Could not resolve mask token id for model_family={config.model_family!r}. "
        "Set --mask_id explicitly or add mask_token_id to the model config/tokenizer."
    )


def _build_online_eval_args(
    config: TrainConfig,
    spec: dict,
    task: str,
    output_dir: Path,
    checkpoint_path: str,
    mask_id: int,
) -> EvalConfig:
    values = {name: spec[name] for name in ONLINE_EVAL_FIELDS if name in spec}
    args = EvalConfig(
        **values,
        task=task,
        ckpt_path=str(checkpoint_path),
        model_family=config.model_family,
        mask_id=mask_id,
        output=str(output_dir),
    )
    args.validate()
    return args


def run_online_evaluation(
    model,
    tokenizer,
    device,
    config: TrainConfig,
    mask_id: int,
    checkpoint_path: str,
    completed_step: int,
) -> dict:
    """Evaluate the current model at a configured optimizer step."""
    if config.eval_every <= 0:
        return {}
    if not online_evaluation_due(config, completed_step):
        raise ValueError(
            f"Online evaluation called at invalid completed_step={completed_step}"
        )
    if config.eval_config is None:
        raise ValueError("eval_config is required when online evaluation is enabled")

    from eval import (
        eval_code, eval_math, eval_train_code, _dist_barrier, _synchronize_rank_error,
    )

    spec = resolve_online_eval_config(config)
    tasks = tuple(spec["tasks"])
    output_root = Path(spec["output_dir"])
    setup_error = None
    try:
        output_root.mkdir(parents=True, exist_ok=True)
    except BaseException as exc:
        setup_error = f"{type(exc).__name__}: {exc}"
    _synchronize_rank_error(setup_error, "online evaluation setup failed")

    benchmark_metrics = {}
    was_training = model.training
    model_error = None
    try:
        model.eval()
    except BaseException as exc:
        model_error = f"{type(exc).__name__}: {exc}"
    _synchronize_rank_error(model_error, "online evaluation model setup failed")
    try:
        cuda_devices = [torch.cuda.current_device()] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=cuda_devices):
            for task_index, task in enumerate(tasks):
                task_output = output_root / task / f"step-{completed_step:06d}"
                task_error = None
                args = None
                try:
                    task_output.mkdir(parents=True, exist_ok=True)
                    args = _build_online_eval_args(
                        config, spec, task, task_output, checkpoint_path, mask_id
                    )
                except BaseException as exc:
                    task_error = f"{type(exc).__name__}: {exc}"
                _synchronize_rank_error(
                    task_error, f"online {task} setup failed"
                )

                task_error = None
                task_metrics = None
                try:
                    # The online resolver intentionally produces one scalar
                    # protocol per task; standalone sweep lists are rejected
                    # at train-config validation time.
                    args._sampling_vocab_size = len(tokenizer)
                    task_seed = args.seed + task_index
                    torch.manual_seed(task_seed)
                    if device.type == "cuda":
                        torch.cuda.manual_seed_all(task_seed)
                    if dist.get_rank() == 0:
                        print(
                            f"Online evaluation: step={completed_step}, task={task}, "
                            f"gen_length={args.gen_length}, mtp_size={args.mtp_size}"
                        )
                    with torch.no_grad():
                        if task in ("gsm8k", "math500"):
                            task_metrics = eval_math(model, tokenizer, device, args)
                        elif task == "code":
                            task_metrics = eval_train_code(model, tokenizer, device, args)
                        else:
                            task_metrics = eval_code(model, tokenizer, device, args)
                except BaseException as exc:
                    task_error = f"{type(exc).__name__}: {exc}"
                _synchronize_rank_error(
                    task_error, f"online {task} execution failed"
                )
                if dist.get_rank() == 0 and task_metrics is not None:
                    benchmark_metrics[task] = task_metrics
                _dist_barrier()
    finally:
        if was_training:
            model.train()
    return benchmark_metrics


def _replay_checkpoint_evaluation(*, device, config, checkpoint_path, completed_step):
    """Recover an evaluation interrupted after its training checkpoint was saved."""
    if not online_evaluation_due(config, completed_step):
        return {}
    if config.dataset != "code":
        return run_checkpoint_evaluation(
            device=device, config=config, checkpoint_path=checkpoint_path,
            completed_step=completed_step,
        )
    from eval import _synchronize_rank_error
    from utils.evaluation.online_eval import _load_checkpoint_for_eval, isolated_eval_rng

    model = tokenizer = None
    try:
        with isolated_eval_rng(resolve_online_eval_config(config)["seed"], device):
            error = None
            try:
                model, tokenizer = _load_checkpoint_for_eval(
                    checkpoint_path, config.model_family, device,
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            _synchronize_rank_error(error, "Online checkpoint replay setup failed")
            return run_online_evaluation(
                model, tokenizer, torch.device(device), config,
                resolve_mask_id(config, model, tokenizer), checkpoint_path, completed_step,
            )
    finally:
        del model, tokenizer
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()


def _complete_step_timings(timings):
    names = (
        "rollout_generation",
        "rollout_phase",
        "logprob_prefill",
        "backward",
        "learner_phase",
        "optimizer",
        "train_step",
        "metrics",
        "checkpoint",
        "online_evaluation",
        "end_to_end",
    )
    completed = {name: float(timings.get(name, 0.0)) for name in names}
    completed["rollout_other"] = max(
        completed["rollout_phase"] - completed["rollout_generation"], 0.0
    )
    completed["learner_other"] = max(
        completed["learner_phase"]
        - completed["logprob_prefill"]
        - completed["backward"],
        0.0,
    )
    completed["train_other"] = max(
        completed["train_step"]
        - completed["rollout_phase"]
        - completed["learner_phase"]
        - completed["optimizer"],
        0.0,
    )
    return completed


def _append_timing_record(path, payload):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _append_training_metrics(path, step, metrics):
    payload = {"step": step, **metrics}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _flatten_benchmark_metrics(benchmark_metrics):
    flattened = {}
    for task, metrics in benchmark_metrics.items():
        if not isinstance(metrics, dict):
            continue
        normalized = {}
        for name, value in metrics.items():
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    continue
                value = value.item()
            if isinstance(value, bool):
                value = int(value)
            elif isinstance(value, Real) and not isinstance(value, (int, float)):
                value = float(value)
            if isinstance(value, Real):
                normalized[name] = value
                flattened[f"benchmark/{task}/{name}"] = value
        primary = normalized.get("pass@1", normalized.get("accuracy"))
        if primary is not None:
            flattened[f"benchmark/{task}"] = primary
    return flattened


def _write_timing_summary(path, payload):
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def train(
    config: TrainConfig,
    *,
    source_config_path: Path | None = None,
    config_overrides: dict | None = None,
):
    """Main GRPO training loop."""

    config.validate()
    if config.dataset != "code":
        validate_online_eval_inputs(config)
    train_schedule = config.resolved_train_schedule()
    loss_checkpoint_strategy = resolve_activation_checkpointing_strategy(
        config.activation_checkpointing_strategy,
    )

    # --- Initialize distributed ---
    dist.init()
    rank = dist.get_rank()
    device = torch.device('cuda')

    print("=" * 60)
    print("RePara Training")
    print("=" * 60)

    world_size = dist.get_world_size()
    global_prompt_batch = validate_global_batch_size(config, world_size)
    dataset_path = config.data_path

    try:
        nnodes = int(os.environ.get("NNODES", "1"))
        default_nproc = world_size // nnodes if nnodes > 0 else 0
        nproc_per_node = int(os.environ.get("NPROC_PER_NODE", str(default_nproc)))
    except (TypeError, ValueError) as exc:
        raise ValueError("NNODES and NPROC_PER_NODE must be integers") from exc
    if nnodes <= 0 or nproc_per_node <= 0 or nnodes * nproc_per_node != world_size:
        raise ValueError(
            f"Runtime topology {nnodes}x{nproc_per_node} does not match world_size={world_size}"
        )
    accelerate_config_path = os.environ.get("CONFIG_FILE")

    resume_checkpoint = None
    data_start_position = 0
    resume_payload = {"checkpoint": None, "error": None}
    if rank == 0:
        try:
            pending_checkpoint_retry = _has_pending_checkpoint_artifact(
                config.output_dir
            )
            resume_path = config.resume_ckpt or find_latest_training_state(
                config.output_dir, world_size=world_size
            )
            # Never restart a run directory from step zero after it has
            # published metadata or metrics.  This preflight runs before the
            # 8B model is loaded and is independent of W&B being enabled.
            _validate_run_resume_progress(
                config.output_dir,
                resume_path is not None,
                allow_pending_checkpoint_retry=pending_checkpoint_retry,
            )
            if resume_path is not None:
                if config.resume_ckpt is None:
                    print(
                        f"Found latest complete checkpoint in {config.output_dir}: "
                        f"{resume_path}"
                    )
                if accelerate_config_path is None:
                    raise ValueError("Resume requires CONFIG_FILE from the launcher")
                checkpoint = validate_resume_checkpoint(
                    resume_path,
                    world_size=world_size,
                    nnodes=nnodes,
                    nproc_per_node=nproc_per_node,
                    batch_size_per_device=config.batch_size_per_device,
                    grad_accumulation=config.grad_accumulation,
                    total_steps=config.total_steps,
                    save_every=config.save_every,
                    seed=config.seed,
                    learning_rate=config.learning_rate,
                    weight_decay=config.weight_decay,
                    max_grad_norm=config.max_grad_norm,
                    dataset=config.dataset,
                    dataset_path=dataset_path,
                    model_family=config.model_family,
                    mask_id=config.mask_id,
                    method=config.method,
                    mtp_block_size=config.mtp_block_size,
                    mtp_confidence=config.mtp_confidence,
                    mtp_threshold=config.mtp_threshold,
                    config_path=accelerate_config_path,
                    num_generations=config.num_generations,
                    repeat_times=config.repeat_times,
                    gen_steps=config.gen_steps,
                    gen_length=config.gen_length,
                    selective_logits=config.selective_logits,
                    activation_checkpointing_strategy=config.activation_checkpointing_strategy,
                    output_dir=config.output_dir,
                    format_reward_alpha=config.format_reward_alpha,
                    save_model_only=config.save_model_only,
                    train_schedule=config.normalized_train_schedule(),
                    sdar_native_block_size=config.sdar_native_block_size,
                )
                _validate_resume_output_dir(
                    config.output_dir,
                    checkpoint,
                    explicit=config.resume_ckpt is not None,
                    world_size=world_size,
                )
                same_run = (
                    Path(config.output_dir).expanduser().resolve()
                    == Path(checkpoint.source_run_dir).expanduser().resolve()
                )
                if checkpoint.completed_steps == config.total_steps and not same_run:
                    raise ValueError(
                        "A completed checkpoint can only be resumed in its source "
                        "run directory; increase total_steps to import it elsewhere"
                    )
                resume_payload["checkpoint"] = asdict(checkpoint)
        except Exception as exc:
            resume_payload["error"] = str(exc)

    resume_payload = _broadcast_main_value(resume_payload, rank)
    if resume_payload["error"] is not None:
        raise RuntimeError(
            "Could not prepare checkpoint resume: " + resume_payload["error"]
        )
    if resume_payload["checkpoint"] is not None:
        resume_checkpoint = ResumeCheckpoint(**resume_payload["checkpoint"])
        data_start_position = resume_checkpoint.global_data_cursor
        if resume_checkpoint.completed_steps == config.total_steps:
            _replay_checkpoint_evaluation(
                device=device, config=config,
                checkpoint_path=resume_checkpoint.model_path,
                completed_step=resume_checkpoint.completed_steps,
            )
            if rank == 0:
                print(
                    f"Run already completed {config.total_steps} steps; "
                    "nothing to resume."
                )
            return

    model_path = (
        resume_checkpoint.model_path
        if resume_checkpoint is not None
        else config.model_path
    )

    # Rollout sampling uses completion-keyed generators rather than this fallback stream.
    np.random.seed(config.seed % (1 << 32))
    torch.manual_seed(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # --- Load model ---
    print(f"Loading model and tokenizer from {model_path}...")
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    model_cls = AutoModelForCausalLM if config.model_family == "sdar" else AutoModel
    model_path_error = None
    if rank == 0:
        try:
            validate_local_model_path(model_path)
        except Exception as exc:
            model_path_error = str(exc)
    model_path_error = _broadcast_main_value(model_path_error, rank)
    if model_path_error is not None:
        raise RuntimeError("Could not validate local model files: " + model_path_error)
    model = model_cls.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    if config.model_family == "sdar":
        native_block_size = configure_sdar_native_block_size(
            model.config, config.sdar_native_block_size
        )
        if rank == 0:
            print(
                f"SDAR native block size: {native_block_size}; "
                f"RePara MTP block size: {config.mtp_block_size}",
                flush=True,
            )
    model.eval().to(device)

    if config.model_family == "sdar":
        if hasattr(model.config, "fuse_cross_entropy"):
            model.config.fuse_cross_entropy = False
    activation_checkpointing_supported = set_activation_checkpointing(
        model, loss_checkpoint_strategy, model_family=config.model_family,
    )
    if not activation_checkpointing_supported and rank == 0:
        print(
            f"Warning: the loaded {config.model_family} model exposes no activation-checkpointing "
            f"controller; requested strategy {config.activation_checkpointing_strategy!r} "
            "will have no effect"
        )

    if config.selective_logits:
        install_position_selective_output_head(model, model_family=config.model_family)
        print(f"Installed position-selective {config.model_family} output head")

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    mask_id = resolve_mask_id(config, model, tokenizer)
    tokenizer.pad_token_id = mask_id
    print(f"Model family: {config.model_family}")
    print(f"Mask token id: {mask_id}")

    # --- Load dataset ---
    if config.dataset == "gsm8k":
        print("Loading GSM8K dataset...")
        from data.math import load_gsm8k_dataset_and_reward
        dataloader, reward_fn = load_gsm8k_dataset_and_reward(
            local_path=dataset_path,
            batch_size=config.batch_size_per_device,
            num_workers=4,
            start_position=data_start_position,
        )
    elif config.dataset == "math":
        print("Loading MATH dataset...")
        from data.math import load_math_dataset_and_reward
        dataloader, reward_fn = load_math_dataset_and_reward(
            local_path=dataset_path,
            batch_size=config.batch_size_per_device,
            num_workers=4,
            start_position=data_start_position,
        )
    elif config.dataset == "code":
        print(f"Loading code dataset from {dataset_path}...")
        from data.code import load_code_dataset_and_reward
        dataloader, reward_fn = load_code_dataset_and_reward(
            local_path=dataset_path,
            batch_size=config.batch_size_per_device,
            num_workers=4,
            start_position=data_start_position,
            format_reward_alpha=config.format_reward_alpha,
        )
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")

    if len(dataloader.dataset) < global_prompt_batch:
        raise ValueError(
            f"Dataset has {len(dataloader.dataset)} rows but one optimizer step "
            f"requires {global_prompt_batch} unique prompts."
        )

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        params=[p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=config.weight_decay,
    )

    # --- Accelerator setup ---
    accelerator = dist.get_accelerator()
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    if accelerator.gradient_accumulation_steps != 1:
        raise ValueError(
            "JustGRPO performs manual gradient accumulation; Accelerator gradient_accumulation_steps must remain 1."
        )

    # Keep learner dropout independent across ranks without keying it by world size.
    learner_seed = derive_learner_seed(config.seed, rank=rank)
    np.random.seed(learner_seed % (1 << 32))
    torch.manual_seed(learner_seed)

    start_step = 0
    if resume_checkpoint is not None:
        print(f"Loading complete training state from {resume_checkpoint.path}...")
        load_error = None
        try:
            accelerator.load_state(resume_checkpoint.path)
        except Exception as exc:
            load_error = exc
        _raise_if_main_process_failed(
            accelerator, load_error, "loading the training state"
        )
        accelerator.wait_for_everyone()
        start_step = resume_checkpoint.completed_steps
        print(
            f"Resuming at optimizer step {start_step + 1}; "
            f"global data cursor={resume_checkpoint.global_data_cursor}"
        )
        same_resume_run = (
            Path(config.output_dir).expanduser().resolve()
            == Path(resume_checkpoint.source_run_dir).expanduser().resolve()
        )
        if rank == 0 and same_resume_run:
            _truncate_jsonl_after(
                Path(config.output_dir) / "training_metrics.jsonl", start_step
            )
            _truncate_jsonl_after(
                Path(config.output_dir) / "timing.jsonl", start_step
            )

    # Initialize W&B only after model, data, and (when applicable) checkpoint
    # state are ready.  This prevents a failed launch from leaving a seemingly
    # resumable identity before any training progress exists.  Rank zero owns
    # the SDK; every rank receives the same initialization error.
    wandb_run = None
    wandb_payload = {
        "error": None,
        "mode": _requested_wandb_mode(config),
    }
    if rank == 0:
        try:
            wandb_run = init_wandb(
                config,
                resume=resume_checkpoint is not None,
                resume_step=(
                    resume_checkpoint.completed_steps
                    if resume_checkpoint is not None
                    else None
                ),
                activate=False,
            )
        except Exception as exc:
            wandb_payload["error"] = str(exc)
        else:
            wandb_payload["mode"] = (
                _wandb_effective_mode(
                    wandb_run,
                    _requested_wandb_mode(config),
                )
                if wandb_run is not None
                else _requested_wandb_mode(config)
            )
    try:
        wandb_payload = _broadcast_main_value(wandb_payload, rank)
    except BaseException:
        if rank == 0 and wandb_run is not None:
            _finish_wandb_after_error(wandb_run)
        raise
    if wandb_payload["error"] is not None:
        raise RuntimeError("Could not initialize W&B: " + wandb_payload["error"])
    wandb_log_mode = _normalize_wandb_mode(wandb_payload.get("mode"))

    try:
        dataloader_iter = iter(dataloader)
    except BaseException:
        if rank == 0 and wandb_run is not None:
            _finish_wandb_after_error(wandb_run)
        raise

    # --- Output directory ---
    resume_stage_index = (
        _resume_stage_index(Path(config.output_dir).expanduser().resolve())
        if resume_checkpoint is not None
        else None
    )
    metadata_error = None
    if rank == 0:
        try:
            os.makedirs(config.output_dir, exist_ok=True)
            manifest_path = Path(config.output_dir) / MANIFEST_FILENAME
            manifest_exists = manifest_path.is_file()
            effective_config = asdict(config)
            if config.train_schedule is not None:
                effective_config["train_schedule"] = config.normalized_train_schedule()
            if config.eval_config is not None:
                effective_config["eval_config"] = resolve_online_eval_config(config)
            effective_config["model_path"] = model_path
            effective_config["resolved_mask_id"] = mask_id
            effective_config["wandb_mode_resolved"] = wandb_log_mode

            # The first manifest/config snapshot is immutable.  This is what
            # lets an in-place continuation validate the original run.
            if not manifest_exists:
                if source_config_path is not None:
                    snapshot_config(
                        config.output_dir,
                        source_path=source_config_path,
                        section="train",
                        effective_config=effective_config,
                        overrides=config_overrides or {},
                    )
                write_manifest(
                    manifest_path,
                    repo_root=Path(__file__).resolve().parent,
                    config=effective_config,
                    world_size=world_size,
                    nnodes=nnodes,
                    nproc_per_node=nproc_per_node,
                    dataset_path=dataset_path,
                    accelerate_config_path=accelerate_config_path,
                )
            elif resume_checkpoint is not None:
                _append_resume_history(
                    Path(config.output_dir) / "resume_history.jsonl",
                    resume_checkpoint,
                    config.total_steps,
                )
        except Exception as exc:
            metadata_error = exc
    if metadata_error is not None and wandb_run is not None:
        try:
            wandb_run.finish(exit_code=1)
        except Exception:
            pass
    _raise_if_main_process_failed(
        accelerator, metadata_error, "writing run metadata"
    )

    activation_error = None
    if rank == 0 and wandb_run is not None:
        try:
            _activate_wandb_identity(config.output_dir)
            _log_wandb_stage(
                wandb_run,
                wandb_log_mode,
                Path(config.output_dir).expanduser().resolve(),
                (
                    resume_checkpoint.completed_steps
                    if resume_checkpoint is not None
                    else None
                ),
                config.total_steps,
                stage_index=resume_stage_index,
            )
        except Exception as exc:
            activation_error = exc
            try:
                wandb_run.finish(exit_code=1)
            except Exception:
                pass
    _raise_if_main_process_failed(
        accelerator, activation_error, "activating W&B identity"
    )
    try:
        accelerator.wait_for_everyone()
    except BaseException:
        if rank == 0 and wandb_run is not None:
            _finish_wandb_after_error(wandb_run)
        raise

    # A state checkpoint is published before its evaluation. Replay that
    # evaluation on resume so an interruption cannot silently omit a result.
    if (
        resume_checkpoint is not None
        and online_evaluation_due(config, start_step)
    ):
        _replay_checkpoint_evaluation(
            device=device, config=config,
            checkpoint_path=resume_checkpoint.model_path, completed_step=start_step,
        )

    # --- Training loop ---
    print(f"Starting training for {config.total_steps} steps...")
    print(f"Group size: {config.num_generations * config.repeat_times}")
    print(f"Grad accumulation: {config.grad_accumulation}")
    print(f"Effective batch: {global_prompt_batch}")
    print(f"Learning rate: {config.learning_rate}")
    schedule_summary = ", ".join(
        f"{index}@{stage.start_step}:{stage.method}/block{stage.block_size}"
        for index, stage in enumerate(train_schedule)
    )
    print(f"Training schedule: {schedule_summary}")
    print(f"Selective logits: {config.selective_logits}")
    print(f"Activation checkpointing: {config.activation_checkpointing_strategy}")
    if any(stage.method == "ar_mtp" for stage in train_schedule):
        print(f"MTP confidence: {config.mtp_confidence} >= {config.mtp_threshold}")
    if config.profile_timing:
        print(
            "Timing: enabled "
            f"(first {config.timing_warmup_steps} executed step(s) excluded from averages)"
        )

    timing_path = Path(config.output_dir) / "timing.jsonl"
    timing_summary_path = Path(config.output_dir) / "timing_summary.json"
    training_metrics_path = Path(config.output_dir) / "training_metrics.jsonl"
    timing_totals = {}
    measured_timing_steps = 0
    expected_global_completions = (
        global_prompt_batch * config.num_generations * config.repeat_times
    )

    try:
        progress_bar = tqdm(
            total=config.total_steps,
            initial=start_step,
            desc="Training",
            unit="step",
            dynamic_ncols=True,
            disable=(rank != 0),
        )
    except BaseException:
        if rank == 0 and wandb_run is not None:
            _finish_wandb_after_error(wandb_run)
        raise

    try:
        for step in range(start_step, config.total_steps):
            stage_index = config.stage_index_for_step(step)
            stage = config.stage_for_step(step)
            stage_method = stage.method
            stage_block_size = stage.block_size
            timing = TimingRecorder(enabled=config.profile_timing, device=device)
            operation_timing = timing if config.profile_timing else None
            timing.start("end_to_end")
            timing.start("train_step")
            optimizer.zero_grad(set_to_none=True)

            all_rewards = []
            all_reward_components = {}
            rollout_batches = []
            local_valid_total = torch.zeros((), dtype=torch.int64, device=device)
            local_raw_loss_sum = torch.zeros((), dtype=torch.float64, device=device)
            local_raw_abs_loss_sum = torch.zeros((), dtype=torch.float64, device=device)
            local_mtp_length_counts = torch.zeros(
                stage_block_size, dtype=torch.int64, device=device
            )
            local_mtp_forced_singletons = torch.zeros((), dtype=torch.int64, device=device)
            local_mtp_confidence_sum = torch.zeros((), dtype=torch.float64, device=device)
            local_mtp_confidence_count = torch.zeros((), dtype=torch.int64, device=device)
            local_rollout_forward_rounds = torch.zeros((), dtype=torch.int64, device=device)
            local_loss_forward_rounds = torch.zeros((), dtype=torch.int64, device=device)

            # Phase A: collect every rollout in this optimizer step first so the
            # global valid-completion denominator is known before any backward.
            timing.start("rollout_phase")
            model.eval()
            for accum_idx in range(config.grad_accumulation):
                print(
                    f"[Step {step + 1}/{config.total_steps}] "
                    f"[Accum {accum_idx + 1}/{config.grad_accumulation}] Sampling..."
                )
                with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                    batch = next(dataloader_iter)
                    prompt_batches = _split_prompt_batch(batch)
                    if len(prompt_batches) != config.batch_size_per_device:
                        raise RuntimeError(
                            f"Expected {config.batch_size_per_device} prompts, "
                            f"got {len(prompt_batches)}"
                        )

                    prompt_rollouts = []
                    for repeat_id in range(config.repeat_times):
                        repeat_rollouts = []
                        for prompt_batch in prompt_batches:
                            inputs = sample(
                                model=model,
                                batch=prompt_batch,
                                tokenizer=tokenizer,
                                device=device,
                                reward_fn=reward_fn,
                                num_generations=config.num_generations,
                                steps=config.gen_steps,
                                gen_length=config.gen_length,
                                method=stage_method,
                                mtp_block_size=stage_block_size,
                                mtp_confidence=config.mtp_confidence,
                                mtp_threshold=config.mtp_threshold,
                                mask_id=mask_id,
                                model_family=config.model_family,
                                base_seed=config.seed,
                                optimizer_step=step,
                                repeat_id=repeat_id,
                                selective_logits=config.selective_logits,
                                timing=operation_timing,
                            )
                            repeat_rollouts.append(inputs)
                            if stage_method != "justgrpo":
                                rollout_stats = inputs["rollout_stats"]
                                local_mtp_length_counts += rollout_stats["length_counts"]
                                local_mtp_forced_singletons += rollout_stats[
                                    "forced_singletons"
                                ]
                                local_mtp_confidence_sum += rollout_stats["confidence_sum"]
                                local_mtp_confidence_count += rollout_stats[
                                    "confidence_count"
                                ]
                                local_rollout_forward_rounds += int(
                                    rollout_stats["forward_rounds"]
                                )
                        prompt_rollouts.append(repeat_rollouts)

                    inputs_chunks, advantages = _organize_prompt_rollouts(
                        prompt_rollouts
                    )
                    local_valid_total += (advantages != 0).sum().to(torch.int64)
                    for chunk in inputs_chunks:
                        all_rewards.append(chunk["rewards"].detach())
                        for name, values in chunk.get("reward_components", {}).items():
                            all_reward_components.setdefault(name, []).append(values.detach())
                    rollout_batches.append(inputs_chunks)

            global_valid_total = accelerator.reduce(local_valid_total, reduction="sum")
            global_valid_count = int(global_valid_total.item())
            if not 0 <= global_valid_count <= expected_global_completions:
                raise RuntimeError(
                    f"Invalid global valid count {global_valid_count}; expected a value "
                    f"in [0, {expected_global_completions}]"
                )
            loss_scale = compute_pooled_loss_scale(
                global_valid=global_valid_count,
                world_size=dist.get_world_size(),
                gen_length=config.gen_length,
            )
            timing.stop("rollout_phase")

            # Phase B: replay learner forwards with one optimizer-step-level scale.
            # DDP/FSDP averages rank gradients, hence the world-size factor in the
            # scale. There is intentionally no extra gradient-accumulation divisor.
            timing.start("learner_phase")
            model.train()
            for accum_idx, inputs_chunks in enumerate(rollout_batches):
                print(
                    f"[Step {step + 1}/{config.total_steps}] "
                    f"[Accum {accum_idx + 1}/{config.grad_accumulation}] Computing loss..."
                )
                for chunk_index, inputs in enumerate(inputs_chunks):
                    sync = (
                        accum_idx == config.grad_accumulation - 1
                        and chunk_index == len(inputs_chunks) - 1
                    )
                    with dist.ddp_sync(model, sync=sync):
                        if stage_method in ("fixed_mtp", "ar_mtp"):
                            loss_stats = block_logprob_loss(
                                model=model,
                                inputs=inputs,
                                loss_scale=loss_scale,
                                max_block_size=stage_block_size,
                                accelerator=accelerator,
                                gen_length=config.gen_length,
                                mask_id=mask_id,
                                model_family=config.model_family,
                                selective_logits=config.selective_logits,
                                timing=operation_timing,
                            )
                            local_loss_forward_rounds += loss_stats["loss_forward_rounds"]
                        else:
                            loss_stats = logprob_loss(
                                model=model,
                                inputs=inputs,
                                loss_scale=loss_scale,
                                accelerator=accelerator,
                                gen_length=config.gen_length,
                                mask_id=mask_id,
                                model_family=config.model_family,
                                selective_logits=config.selective_logits,
                                timing=operation_timing,
                            )
                            local_loss_forward_rounds += config.gen_length
                        local_raw_loss_sum += loss_stats["raw_loss_sum"]
                        local_raw_abs_loss_sum += loss_stats["raw_abs_loss_sum"]
                    inputs.clear()

            del rollout_batches

            global_raw_loss_sum = accelerator.reduce(local_raw_loss_sum, reduction="sum")
            global_raw_abs_loss_sum = accelerator.reduce(local_raw_abs_loss_sum, reduction="sum")
            timing.stop("learner_phase")

            # --- Grad Clip & Optimizer Step ---
            timing.start("optimizer")
            for param in model.parameters():
                if param.grad is not None:
                    torch.nan_to_num(param.grad, nan=0, posinf=0, neginf=0, out=param.grad)

            grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            if hasattr(grad_norm, "item"):
                grad_norm = grad_norm.item()

            optimizer.step()
            # Gradients are not part of AdamW's checkpoint state. Release them
            # before checkpointing to reduce the FSDP save-time memory peak.
            optimizer.zero_grad(set_to_none=True)
            timing.stop("optimizer")
            timing.stop("train_step")

            # --- Metric preparation ---
            log_data = None
            log_message = None
            progress_postfix = None
            timing.start("metrics")
            if (step + 1) % config.log_every == 0 or step == stage.start_step:
                all_rewards_tensor = torch.cat(all_rewards, dim=0)
                gathered_rewards = accelerator.gather(all_rewards_tensor)
                mean_reward = gathered_rewards.mean().item()
                objective_denominator = config.gen_length * max(global_valid_count, 1)
                pooled_loss = (
                    global_raw_loss_sum.item() / objective_denominator if global_valid_count > 0 else 0.0
                )
                pooled_abs_loss = (
                    global_raw_abs_loss_sum.item() / objective_denominator if global_valid_count > 0 else 0.0
                )
                memory_allocated_gb = torch.cuda.memory_allocated(device) / 1024**3
                memory_reserved_gb = torch.cuda.memory_reserved(device) / 1024**3
                log_data = {
                    "train/reward": mean_reward,
                    "train/reward_total": mean_reward,
                    "train/loss": pooled_loss,
                    "train/loss_abs": pooled_abs_loss,
                    "train/global_valid": global_valid_count,
                    "train/valid_fraction": global_valid_count / expected_global_completions,
                    "train/grad_norm_preclip": grad_norm,
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/stage": stage_index,
                    "train/method": stage_method,
                    "train/block_size": stage_block_size,
                    "system/mem_alloc_gb": memory_allocated_gb,
                    "system/mem_reserved_gb": memory_reserved_gb,
                }
                reward_component_means = {}
                for name in sorted(all_reward_components):
                    component_tensor = torch.cat(all_reward_components[name], dim=0)
                    gathered_component = accelerator.gather(component_tensor)
                    component_mean = gathered_component.mean().item()
                    component_std = gathered_component.std(unbiased=False).item()
                    reward_component_means[name] = component_mean
                    log_data[f"train/{name}"] = component_mean
                    log_data[f"train/{name}_std"] = component_std

                log_message = (
                    f"[Step {step + 1}/{config.total_steps}] loss={pooled_loss:.8f}, "
                    f"loss_abs={pooled_abs_loss:.8f}, reward={mean_reward:.6f}, "
                    f"valid={global_valid_count}/{expected_global_completions}, "
                    f"grad_preclip={grad_norm:.8f}, "
                    f"mem={memory_allocated_gb:.1f}/{memory_reserved_gb:.1f}GB"
                )
                component_message = ", ".join(
                    f"{name}={value:.4f}"
                    for name, value in sorted(reward_component_means.items())
                    if name != "total_reward"
                )
                if component_message:
                    log_message += f", rewards[{component_message}]"

                if stage_method != "justgrpo":
                    mtp_length_counts = accelerator.reduce(
                        local_mtp_length_counts, reduction="sum"
                    )
                    mtp_forced_singletons = accelerator.reduce(
                        local_mtp_forced_singletons, reduction="sum"
                    )
                    mtp_confidence_sum = accelerator.reduce(
                        local_mtp_confidence_sum, reduction="sum"
                    )
                    mtp_confidence_count = accelerator.reduce(
                        local_mtp_confidence_count, reduction="sum"
                    )
                    rollout_forward_rounds = accelerator.reduce(
                        local_rollout_forward_rounds, reduction="sum"
                    )
                    loss_forward_rounds = accelerator.reduce(
                        local_loss_forward_rounds, reduction="sum"
                    )

                    total_blocks = mtp_length_counts.sum().clamp_min(1)
                    block_lengths = torch.arange(
                        1, stage_block_size + 1, device=device
                    )
                    mean_block_length = (
                        mtp_length_counts * block_lengths
                    ).sum().float() / total_blocks
                    singleton_fraction = mtp_length_counts[0].float() / total_blocks
                    histogram = ",".join(str(value.item()) for value in mtp_length_counts)
                    log_data.update(
                        {
                            "mtp/mean_block_length": mean_block_length.item(),
                            "mtp/singleton_fraction": singleton_fraction.item(),
                            "mtp/rollout_forward_rounds": rollout_forward_rounds.item(),
                            "mtp/loss_forward_rounds": loss_forward_rounds.item(),
                            "mtp/forced_singletons": mtp_forced_singletons.item(),
                        }
                    )
                    for index, value in enumerate(mtp_length_counts, start=1):
                        log_data[f"mtp/block_length_{index}_count"] = value.item()
                    log_message += (
                        f", block={mean_block_length.item():.2f}, "
                        f"singleton={singleton_fraction.item():.3f}, hist=[{histogram}], "
                        f"rollout_fw={rollout_forward_rounds.item()}, "
                        f"loss_fw={loss_forward_rounds.item()}"
                    )
                    if stage_method == "ar_mtp":
                        confidence_mean = (
                            mtp_confidence_sum / mtp_confidence_count.clamp_min(1)
                        )
                        log_data["mtp/confidence_mean"] = confidence_mean.item()
                        log_message += (
                            f", forced={mtp_forced_singletons.item()}, "
                            f"conf={confidence_mean.item():.3f}"
                        )
                progress_postfix = {
                    "reward": f"{mean_reward:.3f}",
                    "loss_abs": f"{pooled_abs_loss:.3f}",
                    "valid": str(global_valid_count),
                }
            timing.stop("metrics")

            # --- Save checkpoint and run optional online evaluation ---
            evaluation_due = online_evaluation_due(config, step + 1)
            should_save = (step + 1) % config.save_every == 0 or (
                step + 1 == config.total_steps
            ) or evaluation_due
            saved_paths = None
            if should_save:
                timing.start("checkpoint")
                saved_paths = save_training_checkpoint(
                    accelerator=accelerator,
                    model=model,
                    tokenizer=tokenizer,
                    output_dir=config.output_dir,
                    completed_steps=step + 1,
                    model_source_path=model_path,
                    save_model_only=config.save_model_only,
                    keep_latest_state_only=config.keep_latest_state_only,
                )
                if rank == 0:
                    saved_parts = [f"model={saved_paths['model']}"]
                    if saved_paths["state"] is not None:
                        saved_parts.append(f"state={saved_paths['state']}")
                    print(f"Saved checkpoint: {', '.join(saved_parts)}")
                timing.stop("checkpoint")
            if evaluation_due:
                if saved_paths is None:
                    raise RuntimeError("Online evaluation requires a saved checkpoint")
                timing.start("online_evaluation")
                if config.dataset == "code":
                    benchmark_metrics = run_online_evaluation(
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        config=config,
                        mask_id=mask_id,
                        checkpoint_path=saved_paths["model"],
                        completed_step=step + 1,
                    )
                else:
                    optimizer.zero_grad(set_to_none=True)
                    benchmark_metrics = run_checkpoint_evaluation(
                        device=device,
                        config=config,
                        checkpoint_path=saved_paths["model"],
                        completed_step=step + 1,
                    )
                timing.stop("online_evaluation")
                if rank == 0:
                    benchmark_log = _flatten_benchmark_metrics(benchmark_metrics)
                    if log_data is None:
                        log_data = {
                            "train/stage": stage_index,
                            "train/method": stage_method,
                            "train/block_size": stage_block_size,
                        }
                        log_message = (
                            f"[Step {step + 1}/{config.total_steps}] "
                            f"stage={stage_index} method={stage_method} "
                            f"block={stage_block_size}, benchmark"
                        )
                        progress_postfix = {"benchmark": "done"}
                    log_data["eval/completed_step"] = step + 1
                    log_data.update(benchmark_log)
                    benchmark_message = ", ".join(
                        f"{key.removeprefix('benchmark/')}={value:.4f}"
                        for key, value in sorted(benchmark_log.items())
                        if key.count("/") == 1
                    )
                    if benchmark_message:
                        log_message += f", benchmark[{benchmark_message}]"
            accelerator.wait_for_everyone()
            timing.stop("end_to_end")

            timing_message = None
            step_log_error = None
            if config.profile_timing:
                step_timings = _complete_step_timings(
                    reduce_max_timings(timing.elapsed(), device)
                )
                is_warmup = (step - start_step) < config.timing_warmup_steps
                if not is_warmup:
                    measured_timing_steps += 1
                    for name, seconds in step_timings.items():
                        timing_totals[name] = timing_totals.get(name, 0.0) + seconds

                timing_averages = {
                    name: total / measured_timing_steps
                    for name, total in timing_totals.items()
                } if measured_timing_steps else {}
                global_tokens = expected_global_completions * config.gen_length
                train_step_seconds = max(step_timings["train_step"], 1e-12)
                rollout_seconds = max(step_timings["rollout_phase"], 1e-12)
                throughput = {
                    "train_completions_per_s": (
                        expected_global_completions / train_step_seconds
                    ),
                    "train_tokens_per_s": global_tokens / train_step_seconds,
                    "rollout_tokens_per_s": global_tokens / rollout_seconds,
                }

                timing_record = {
                    "type": "step",
                    "step": step + 1,
                    "stage": stage_index,
                    "method": stage_method,
                    "block_size": stage_block_size,
                    "aggregation": "max_per_metric_across_ranks",
                    "is_warmup": is_warmup,
                    "world_size": world_size,
                    "global_prompts": global_prompt_batch,
                    "global_completions": expected_global_completions,
                    "seconds": step_timings,
                    "cumulative_average_seconds": timing_averages,
                    "throughput": throughput,
                }
                if rank == 0:
                    try:
                        _append_timing_record(timing_path, timing_record)
                    except Exception as exc:
                        step_log_error = exc

                if log_data is not None:
                    log_data.update(
                        {
                            f"timing/{name}_s": seconds
                            for name, seconds in step_timings.items()
                        }
                    )
                    log_data.update(
                        {
                            f"timing_avg/{name}_s": seconds
                            for name, seconds in timing_averages.items()
                        }
                    )
                    log_data.update(
                        {
                            f"throughput/{name}": value
                            for name, value in throughput.items()
                        }
                    )
                    log_data["timing/is_warmup"] = int(is_warmup)

                timing_message = (
                    f"[Timing {step + 1}/{config.total_steps}] "
                    f"train={step_timings['train_step']:.3f}s, "
                    f"rollout={step_timings['rollout_phase']:.3f}s "
                    f"(generation={step_timings['rollout_generation']:.3f}s), "
                    f"learner={step_timings['learner_phase']:.3f}s "
                    f"(logprob/prefill={step_timings['logprob_prefill']:.3f}s, "
                    f"backward={step_timings['backward']:.3f}s), "
                    f"optimizer={step_timings['optimizer']:.3f}s, "
                    f"checkpoint={step_timings['checkpoint']:.3f}s, "
                    f"end_to_end={step_timings['end_to_end']:.3f}s, "
                    f"tokens/s={throughput['train_tokens_per_s']:.1f}"
                )

            if log_data is not None:
                # In shared mode W&B owns the internal history step; this field
                # keeps optimizer-step alignment available for all charts.
                log_data["train/step"] = step + 1
            if log_data is not None and rank == 0 and step_log_error is None:
                try:
                    _append_training_metrics(training_metrics_path, step + 1, log_data)
                    if wandb_run is not None:
                        _log_wandb(
                            wandb_run,
                            wandb_log_mode,
                            log_data,
                            step + 1,
                        )
                except Exception as exc:
                    step_log_error = exc

            # A rank-zero filesystem/W&B failure must be communicated before the
            # next barrier; otherwise the other ranks can wait forever.
            _raise_if_main_process_failed(accelerator, step_log_error, "writing training logs")

            if log_data is not None:
                print(log_message)
                progress_bar.set_postfix(progress_postfix, refresh=False)
            if timing_message is not None:
                print(timing_message)

            accelerator.wait_for_everyone()
            progress_bar.update(1)

        progress_bar.close()
    except BaseException:
        if rank == 0 and wandb_run is not None:
            _finish_wandb_after_error(wandb_run)
        raise

    finalize_error = None
    if rank == 0:
        try:
            if config.profile_timing:
                timing_averages = {
                    name: total / measured_timing_steps
                    for name, total in timing_totals.items()
                } if measured_timing_steps else {}
                summary = {
                    "type": "summary",
                    "method": train_schedule[0].method if len(train_schedule) == 1 else "scheduled",
                    "schedule": [stage.as_dict() for stage in train_schedule],
                    "aggregation": "max_per_metric_across_ranks",
                    "world_size": world_size,
                    "global_prompts": global_prompt_batch,
                    "global_completions": expected_global_completions,
                    "warmup_steps": config.timing_warmup_steps,
                    "measured_steps": measured_timing_steps,
                    "total_seconds": timing_totals,
                    "average_seconds": timing_averages,
                }
                _write_timing_summary(timing_summary_path, summary)
                print(
                    f"Timing summary: {timing_summary_path} "
                    f"({measured_timing_steps} measured step(s))"
                )
        except Exception as exc:
            finalize_error = exc
        if wandb_run is not None:
            try:
                wandb_run.finish()
            except Exception as exc:
                finalize_error = finalize_error or exc
    _raise_if_main_process_failed(accelerator, finalize_error, "finalizing the run")
    print("\nTraining complete!")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="RePara Training")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="YAML experiment configuration containing one top-level train section",
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Override train.output_dir for this run",
    )
    parser.add_argument(
        "--resume_ckpt",
        type=str,
        default=None,
        help="Override train.resume_ckpt for this run",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=None,
        help="Override train.save_every for this run",
    )
    parser.add_argument(
        "--wandb_group",
        "--wandb-group",
        dest="wandb_group",
        default=None,
        help="Override train.wandb_group for this run",
    )
    parser.add_argument(
        "--wandb_run_name",
        "--wandb-run-name",
        dest="wandb_run_name",
        default=None,
        help="Override train.wandb_run_name for this run",
    )
    args = parser.parse_args(argv)

    overrides = {}
    if args.run_dir is not None:
        overrides["output_dir"] = args.run_dir
    if args.resume_ckpt is not None:
        overrides["resume_ckpt"] = args.resume_ckpt
    if args.save_every is not None:
        overrides["save_every"] = args.save_every
    if args.wandb_group is not None:
        overrides["wandb_group"] = args.wandb_group
    if args.wandb_run_name is not None:
        overrides["wandb_run_name"] = args.wandb_run_name
    config, config_path = load_config(
        args.config,
        section="train",
        config_class=TrainConfig,
        overrides=overrides,
    )
    return config, config_path, overrides


if __name__ == "__main__":
    config, source_config_path, config_overrides = parse_args()
    try:
        train(
            config,
            source_config_path=source_config_path,
            config_overrides=config_overrides,
        )
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
