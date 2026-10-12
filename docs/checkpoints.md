# Checkpoints and continuation

RePara saves model exports separately from complete training state. Use a model export for evaluation or a new experiment; use a paired training state to continue the same experiment.

## Saved artifacts

After optimizer update 128, a full save contains:

```text
run/
  ckpt-000128/              # Model weights, config, tokenizer, and custom model code
  training-state-000128/    # Distributed model/optimizer/scheduler/RNG state
  latest                   # Marker for the latest complete training-state save
  manifest.json
  effective_config.yaml
```

Checkpoint export first writes to `.incomplete` staging directories. The final directories and `latest` marker become available only after the paired save completes. An interrupted staging directory is not a resumable checkpoint.

`save_model_only: true` omits the training-state directory and therefore cannot support strict resume. `keep_latest_state_only: true` requires saving training state and removes older complete training-state/model pairs according to the checkpoint policy.

## Restart the same run

Keep the output directory and protocol unchanged and launch the same configuration. The trainer automatically selects the highest complete checkpoint pair. It restores the optimizer, RNG state, and deterministic global data cursor and continues at the next optimizer update.

To choose a training state explicitly:

```bash
CONFIG_FILE=configs/runtime/fsdp.yaml accelerate launch --config_file configs/runtime/fsdp.yaml --num_processes 8 train.py \
  --config configs/train/llada/gsm8k/repara-llada-gsm8k-fix2.yaml \
  --resume_ckpt outputs/repara-llada-gsm8k-fix2/training-state-000128
```

Increase `total_steps` in the YAML when extending a completed run. The total step target cannot be reduced. In the same output directory, resume must use its latest complete state; importing an older state requires a new empty output directory via `--run_dir`.

Run one training job per output directory. Existing run metadata without a complete checkpoint is rejected instead of silently starting again from step zero.

## Compatibility requirements

Strict resume supports full-parameter GSM8K, MATH, and code training with the FSDP sharded-state layout. It requires:

- A complete training state and its paired model export, including tokenizer files.
- The same distributed topology and per-device batch/accumulation settings.
- Matching training semantics: model family, method, block/confidence settings, schedule, optimizer hyperparameters, reward settings, generation budget, and seeds.
- Matching recorded package versions, data-file fingerprint, FSDP configuration, and training-semantic source hashes.

The manifest schema is `repara-training-manifest-v1`. Training-semantic sources, the local data file, and the accelerator configuration are verified by SHA-256. The source checkout, Python environment, data file, and accelerator configuration may be relocated when their contents, recorded package versions, and topology match. A full-state checkpoint with different training-semantic source contents is rejected.

Moving modules into a different source layout changes the recorded source identities. Use the matching code revision for full-state continuation; a model export can initialize a new run after a source reorganization.

Strict continuation uses a single local training data file to verify its content. For initial runs using a directory of shards, create and use a stable local file if you need strict resume. Keep the original model, data, and configuration provenance with the run.

Research training states created before this release's manifest schema are not accepted as strict continuation states. Their self-contained model exports can initialize a new run.

## Start a new stage from model weights

For a new method or block size, set `model_path` to a model export, choose a new `output_dir`, and leave `resume_ckpt: null`:

```yaml
train:
  model_path: outputs/repara-llada-math-fix2/ckpt-000128
  output_dir: outputs/repara-llada-math-fix2-to-fix4
  method: fixed_mtp
  mtp_block_size: 4
  resume_ckpt: null
  total_steps: 128
```

The existing `fix2-to-fix4` recipes follow this model-initialization protocol. The new run starts with fresh optimizer/RNG/data state. A new 128-update stage after a 128-update Fix2 run therefore represents 256 model-training updates in total, but its local step counter starts at zero.

To preserve optimizer state across a planned transition, configure the whole `train_schedule` before starting the original run. A schedule is immutable under strict resume. See [fade and staged training](protocol.md#fade-and-staged-training).
