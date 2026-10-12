# Release scope and provenance

This release provides the RePara training and evaluation framework for LLaDA and SDAR. It includes fixed/adaptive consecutive-block training, native SDAR decoding, reproducibility tooling, and method/block schedules.

Editable local launchers and YAML recipes configure the experiments. Configuration names use RePara for fixed/adaptive methods, represented by the method identifiers `fixed_mtp` and `ar_mtp`, respectively.

## Included capabilities

| Selected capability | Public implementation |
| --- | --- |
| JustGRPO, Fix, Adaptive | One-token baseline and consecutive fixed/adaptive block rollout and likelihood reconstruction |
| Two confidence metrics | Normalized entropy and maximum probability |
| LLaDA and SDAR | Model forward adapters and matching tokenizer/model loading; SDAR native blocks 4 and 32 |
| Math and code | GSM8K/MATH training rewards; Python code formatting and test rewards; benchmark evaluation |
| Full-parameter training | BF16/FSDP with a fixed global prompt batch and editable launch topology |
| Multiple evaluation decoders | Confidence diffusion, fixed/adaptive blocks, and explicit SDAR native generation |
| Memory/performance options | Selective logits, layer activation checkpointing, SDAR rollout KV cache |
| Checkpoints and reproducibility | Self-contained model export, paired training-state saves, semantic resume validation, sample IDs and RNG streams |
| Monitoring | Local reward/block/timing logs and optional W&B/online evaluation |
| Fade | A predeclared schedule of optimizer-step method/block transitions |

LLaDA2.1-mini is planned support, with no implementation bundled in this release. LoRA training, mismatch objectives, Ascend support, Dream integration, cluster submission tooling, and detailed coding diagnosis/intervention/calibration experiments are outside the selected scope. Evaluation can load compatible external adapter checkpoints for baseline comparisons; this does not add adapter training.

## Paper reproduction boundary

The code includes training and evaluation recipes for the selected LLaDA/SDAR experiments and mathematical threshold/block-size variations. The configuration files are runnable starting points whose model and dataset paths must be supplied by the user. Their availability does not imply that a fresh run on arbitrary hardware or data revisions will exactly match a published score.

The README includes the paper's method overview and LLaDA/SDAR result plots, rasterized from their original PDFs, plus a snapshot of reported table values. Their sources and protocols are recorded in [paper results](paper-results.md).

The following experiment implementations live outside this core release and have not been copied into it:

- Four-model comparison integrations for Dream and LLaDA2.1-mini.
- Nemotron single-forward probability/entropy probes and their plotting pipeline.
- External baseline model preparation and consolidated paper figure/table reconstruction.
- Historical raw timing measurements and their multi-continuation aggregation.
- Detailed coding analysis and intervention pipelines.

The showcased scores and timings are the paper's reported results. Release checks do not establish new GPU benchmark scores or new hardware timing measurements. Model weights and the historical raw training/evaluation logs are not bundled; no checkpoint-download location or public paper URL is introduced by this release.

## Review the release

Review method behavior in `grpo.py` and `utils/generation/generate.py`, model integration in `utils/models/forward.py` and the SDAR native generator, and continuation validation in `utils/checkpointing/resume.py`. Inspect configuration defaults for the intended experiment, especially SDAR native attention size, MTP block size, temperature, data splits, and staged initialization.

Run the repository checks in the chosen Python environment:

```bash
python -m pip install -r requirements-llada.txt
python -m pytest -q
```

The release passed GPU checks with public LLaDA/SDAR 8B weights and two-process FSDP save/restart checks with reduced architecture fixtures. The exact coverage and limits are recorded in [validation](validation.md); other GPU topologies and the full paper training protocol require their own validation.

## Attribution

RePara builds on [JustGRPO](https://github.com/LeapLabTHU/JustGRPO). The JustGRPO MIT copyright notice remains in [LICENSE](../LICENSE). Existing source comments retain attribution for task, generation, and mathematical evaluation utilities. The pinned SDAR dependency carries its own source and license information under `third-party/sdar`.
