# Training and evaluation protocols

This page describes the code and recipes shipped in RePara. Use the resolved configuration and run manifest when comparing experiments: model attention blocks, rollout blocks, and evaluation quotas have different meanings.

## Data and checkpoints

Training requires a model directory or model identifier with its matching tokenizer. The recipes use local paths under `models/` and `datasets/`; replace them with your own locations.

| Task | Configuration field | Accepted local data |
| --- | --- | --- |
| GSM8K training | `gsm8k_data_path` | A Parquet file, or a directory containing `main/train-*.parquet` or `train-*.parquet`; fields `question`, `answer` |
| MATH training | `math_data_path` | A Parquet file, or a directory containing `data/train-*.parquet` or `train-*.parquet`; fields `problem`, `solution` |
| Code training | `code_data_path` | JSON/JSONL with `question` and a list of `test_cases` |
| GSM8K evaluation | `gsm8k_data_files` | Test Parquet file or glob |
| MATH-500 evaluation | `math500_data_files` | Test Parquet file or glob |
| HumanEval evaluation | `humaneval_data_file` | Benchmark JSONL, including gzip-compressed input |
| MBPP evaluation | `mbpp_data_file`, `mbpp_test_file` | Benchmark JSONL and its test split |
| Code evaluation | `code_data_file` | JSON/JSONL containing `question` and `test_cases` |

Use the intended MATH training corpus independently of the MATH-500 evaluation split. The MATH loader filters out empty problems and problems with at least 1,500 characters and appends a request for a boxed final answer. Data loaders assign stable sample IDs before shuffling with seed 112.

The recipes use single training files: `datasets/gsm8k/main/train-00000-of-00001.parquet`, `datasets/math/train-00000-of-00001.parquet`, and `datasets/code/train.jsonl`. Replace these paths with the files for your intended corpus. Strict resume fingerprints the local training file. A directory or a remote dataset identifier is accepted for a fresh training run but does not provide the single-file identity required for strict continuation.

GSM8K and MATH rewards are +1 for a correct extracted answer and −1 otherwise. Code reward is the test pass fraction plus `format_reward_alpha` times the format score. Test execution is gated on a valid Python code fence and a parsable program, including when `format_reward_alpha=0`. Code rewards execute Python test cases in local worker processes.

## Training methods

| YAML value | Rollout and learner behavior |
| --- | --- |
| `justgrpo` | Commit one token at a time in left-to-right order and optimize the corresponding token log probabilities. |
| `fixed_mtp` | Sample a consecutive block of up to `mtp_block_size` tokens from the same masked state; reconstruct that state and optimize the sum of the sampled token log probabilities. |
| `ar_mtp` | Propose a consecutive block and accept its longest confident prefix, with a one-token fallback; learn from the actual recorded block boundaries. |

For Adaptive, `mtp_confidence=normalized_entropy` computes `1 − H / log(V)`, where `V` is the sampleable support size. `max_probability` uses the largest token probability. `mtp_threshold` determines acceptance. Mask IDs and logits beyond the tokenizer's vocabulary are excluded from MTP sampling. A block stays masked during its joint likelihood reconstruction; later tokens within the block are not conditioned on earlier sampled tokens from that block.

At positive sampling temperature, confidence uses the temperature-scaled distribution. At temperature zero, token selection is greedy and confidence uses the unscaled model distribution; the confidence values do not become one-hot merely because selection is greedy.

These are full-parameter, on-policy recipes with a single update per rollout group and no KL penalty. Advantages are normalized within each prompt's completion group. Loss normalization accounts for the response budget and the global number of completions with a nonzero advantage.

Training is CUDA/BF16 with FSDP. The fixed global prompt batch is 64. The example recipes use 16 generations per prompt, response budget 256, training seed 1234, learning rate `5e-6`, and 128 optimizer updates. Hyperparameters, including weight decay, are explicit in each YAML; changing a recipe creates a different experiment protocol.

### SDAR blocks and cache

SDAR's native block size defines its block-causal attention and KV-cache boundaries. RePara's `mtp_block_size` defines the proposed consecutive output block. For example, a native attention block of 32 tokens can use Fix2 rollout blocks. Set `sdar_native_block_size` to match the intended SDAR model protocol; it must not be smaller than the proposed MTP block.

SDAR fixed/adaptive rollout reuses the cache of committed native blocks. Proposals stop at the next native block boundary, so a configured Fix2 block can become a singleton at a boundary. The rollout uses a temporary cache for its active block and records the actual commit lengths. Learner log-probability reconstruction does not reuse the rollout KV cache.

The SDAR mathematical JustGRPO-named recipes use `fixed_mtp` with block size one. This preserves the SDAR baseline sampler's native-boundary and sampleable-vocabulary behavior while committing one token per block. The SDAR code baseline uses `justgrpo`; inspect the method field when comparing these recipes.

Adaptive rollout and SDAR forwards synchronize execution across distributed ranks. Mean accepted block length is a property of committed blocks; it is not the same as executed forward rounds or a measured speedup.

### Memory and projection options

`selective_logits=true` projects only the positions needed for rollout or likelihood computation. `activation_checkpointing_strategy=whole_layer` recomputes transformer-layer activations during backward; `none` disables this recomputation. The recipes select these explicitly. FSDP profiles are in `configs/runtime/fsdp.yaml` and `configs/runtime/fsdp_sdar.yaml`; `configs/runtime/fsdp_llada_h200.yaml` preserves a separate H200-oriented sharding profile.

## Fade and staged training

`train_schedule` selects the method and block size by zero-based optimizer step. The stage with `start_step: 32` begins on the 33rd optimizer update:

```yaml
train_schedule:
  - start_step: 0
    method: justgrpo
    block_size: 1
  - start_step: 32
    method: fixed_mtp
    block_size: 2
```

The supplied code fade recipes switch after 32, 64, or 96 JustGRPO updates. The schedule changes sampling and likelihood computation together and remains part of the checkpoint's training protocol. It does not interpolate model weights or gradually change the learning rate.

The `fix2-to-fix4` recipes start a new 128-update run from `ckpt-000128` model weights. They initialize a new optimizer and data stream. Use a predefined schedule in one run when the intended experiment requires preserving optimizer state through a method/block transition. [Checkpoint continuation](checkpoints.md) explains the difference from strict resume.

## Evaluation protocols

| `generation_method` | Important parameters |
| --- | --- |
| `diffusion` | `gen_length`, `block_length`, `mtp_size`; denoising steps are `gen_length / mtp_size` |
| `fixed_mtp` | `gen_length`, `mtp_block_size` |
| `ar_mtp` | `gen_length`, `mtp_block_size`, `mtp_confidence`, `mtp_threshold` |
| `sdar_native` | `sdar_block_length`, `sdar_denoising_steps`, remasking and confidence settings |

For an original SDAR-b32 model whose checkpoint does not declare `config.block_size`, set `sdar_native_block_size: 32` in the evaluation YAML. The b32 consecutive-block evaluation recipes supply this override. A conflicting size in a checkpoint with declared metadata is rejected.

The ordinary diffusion evaluator uses a confidence-selection window (`block_length`), which is independent of SDAR's model-native attention block. Select `sdar_native` explicitly when evaluating SDAR's native generator.

For native SDAR with four-token blocks, the supplied `denoise1`, `denoise2`, and `denoise4` configurations perform one, two, and four denoising steps per block. Their nominal token quotas are respectively four, two, and one. With `low_confidence_dynamic`, confidence-based selection can commit more than the nominal quota; `sdar_confidence_threshold` controls that behavior.

The standalone recipes use temperature zero, one completion per task, and batch size one. Ordinary diffusion and SDAR native generation have separate temperature fields, `temperature` and `sdar_temperature`. List-valued `gen_length` and `mtp_size` expand into a Cartesian sweep while loading the model once. For consecutive-block decoders, `mtp_block_size` sets the block cap; `mtp_size` is not a replacement for it.

The CLI permits checkpoint/output overrides for evaluation and run-directory/resume, checkpoint cadence, and W&B identity overrides for training. The submitted YAML is saved as `source_config.yaml`; `effective_config.yaml` records expanded environment variables and accepted overrides.

## Logs and timing

Local logs work with `wandb: false`, which is the default in the recipes:

| Run artifact | Contents |
| --- | --- |
| `manifest.json` | Source provenance, semantic source identities, package versions, topology, effective configuration, and input identities |
| `source_config.yaml`, `effective_config.yaml` | Submitted and resolved configurations |
| `training_metrics.jsonl` | Reward components, gradient/loss summaries, optimizer step, method/stage, and MTP block statistics |
| `timing.jsonl` | Per-step phase timings and throughput when `profile_timing=true` |
| `timing_summary.json` | Cumulative timing totals and averages excluding configured warmup steps |
| `resume_history.jsonl` | Full-state continuation boundaries |

Timing uses CUDA events and reports the maximum rank time independently for each metric. `rollout_generation` covers generation; `rollout_phase` includes preparation and rewards; `logprob_prefill` excludes backward; `learner_phase` includes it. `train_step` covers rollout, learner, and optimizer work; `end_to_end` also includes checkpointing and reporting preparation. Maxima from different ranks mean nested metrics need not sum exactly.

All per-step measurements are retained. `timing_warmup_steps` excludes the first executed steps of each process invocation from cumulative averages. For an efficiency comparison, keep hardware, topology, model, input data, response budget, checkpoint cadence, instrumentation, and aggregation rules fixed. The paper's historical timing aggregation is not recreated by this release's cumulative summary alone.

W&B is optional; enable `wandb` and select its mode in the YAML. `wandb_identity.json` preserves one logical run identity within an output directory. Offline restarts create separate local SDK fragments. The local JSONL logs remain the source of truth and rows beyond a resumed checkpoint are removed before replay.

Online evaluation is optional and disabled with `eval_every: 0`. Enabling it requires benchmark paths under `eval_config`; mathematical online evaluation runs from saved checkpoints and requires `eval_every` to be a multiple of `save_every`. Its output is separate from training rollouts and adds evaluation work to the run.

Native SDAR code evaluation runs through the standalone evaluator. The optional code online evaluator forwards through the distributed learner and accepts diffusion or consecutive-block decoding; it rejects native decoding because native per-rank forward counts can differ.
