# Release validation

The RePara release was checked on October 12, 2026. Validation covered the complete project test suite, public LLaDA/SDAR model weights, and distributed checkpoint recovery. The findings and resulting corrections are recorded in [release review](release-review.md).

## Test environment

GPU checks used NVIDIA H200 GPUs, Python 3.12.13, PyTorch 2.8.0+cu128, Transformers 4.52.4, Accelerate 1.14.0, and Datasets 3.1.0. SDAR model checks exercised FlashAttention. The dependency under `third-party/sdar` was pinned at commit `4c2749ba103448f45520e8411533710a1e66574d`.

The model fixtures used the public [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) and [SDAR-8B-Chat](https://huggingface.co/JetLM/SDAR-8B-Chat) checkpoints. Every weight shard was checked against its official Hugging Face LFS SHA-256 before use; model configuration, tokenizer, and custom code came from the matching official revision.

## Project suite and source checks

- All **245 tests in 32 test files passed** in the GPU environment, including CUDA timing, configuration, generation, learner, reward, distributed, checkpoint, and resume checks.
- Added regressions cover missing training paths, prepared data-loader RNG isolation, SDAR activation checkpointing, final-step code evaluation, completed-run evaluation replay, and training-code evaluator dispatch.
- All 38 training and 36 evaluation recipes pass configuration validation. Together with four runtime profiles, the 78 YAML files retain their byte-identical contents through directory reorganization.
- Project Python sources parse, Bash launchers pass syntax checking, and the Git diff passes whitespace checking. Documentation links and named configuration paths resolve.
- Publication checks exclude local model weights, test logs, machine paths, credentials, and private development history.

Run the repository suite after installation and submodule initialization:

```bash
python -m pip install -r requirements-llada.txt
OMP_NUM_THREADS=1 GLOO_SOCKET_IFNAME=lo python -m pytest -q
```

The GPU validation ran the same test methods through `python -m unittest discover -s tests -t .`. A CPU-only environment skips the CUDA timing check.

## Checks with pretrained 8B models

| Model | Native SDAR block | Checks passed | Parameters receiving nonzero gradients |
| --- | --- | --- | --- |
| LLaDA-8B-Instruct | N/A | 18 | 291 parameter tensors |
| SDAR-8B-Chat | 4 | 24 | 399 parameter tensors |
| SDAR-8B-Chat | 32 | 24 | 399 parameter tensors |

All model parameters were trainable. These bounded checks exercised fixed/adaptive rollout, both confidence metrics at threshold boundaries, JustGRPO, valid vocabulary support, and learner backpropagation with finite nonzero gradients in the transformer and output head. Learner checks used short sampled completions and controlled positive/negative advantages.

Selective and full output projection produced matching greedy tokens and matching logits at the checked positions. Sampled output-head gradient fingerprints also matched; this is not a claim of bitwise equality for every gradient in the full model. LLaDA layer activation checkpointing produced matching gradient fingerprints.

SDAR checks additionally covered native decoding with one, two, and four denoising steps, cached/uncached fixed and adaptive rollout equivalence at aligned block boundaries, native blocks 4 and 32, and replay of the padded masked context for a partial final block. Its activation-checkpointing controller was checked in both enabled and disabled states.

## Two-process FSDP recovery

Both model families passed separate two-process FSDP checks using `configs/runtime/fsdp.yaml` and `configs/runtime/fsdp_sdar.yaml`. These fixtures used the actual model implementations and tokenizers with **two randomly initialized transformer layers and a reduced hidden size**.

Each check performed an optimizer update, saved paired model/training-state checkpoints, continued one update, then restored the saved state and repeated that update with dropout enabled. Sampled continuations, gradient norms, and every local parameter shard matched the uninterrupted run; the maximum parameter difference was **0.0** for both families. The exported model and tokenizer also reloaded independently and generated valid tokens.

## Validation boundary

These checks establish bounded execution and recovery behavior. They do not rerun the paper's 128-update training protocol, benchmark accuracy evaluation, or historical throughput measurements. Full-size 8B models were used for generation and gradient checks; the distributed save/restart checks used the reduced two-layer fixtures described above. Model weights and raw validation logs are not included in the release.
