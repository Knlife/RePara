# Release review

The October 12, 2026 review covered the complete public source tree against the selected release scope, documented data/evaluation behavior, and reproducibility requirements. Corrections include focused regression tests; GPU verification is recorded in [validation](validation.md).

## Standards

Two actionable findings were corrected:

- **Training input handling:** a missing local path could silently select another dataset, including an evaluation corpus for MATH training. Local training paths now fail explicitly when absent, while valid Hugging Face dataset identifiers remain supported. Regressions cover file, directory, missing-path, and Hub inputs.
- **Configuration navigation:** the README previously described MATH/code recipes as sharing the GSM8K directory. It now points to the categorized `configs/train/<model>/<task>/` layout.

Some task loaders retain similar setup code. This is a maintenance consideration, with no demonstrated correctness defect requiring further changes for this release.

## Spec

Six behavior findings were corrected:

| Finding | Resulting behavior |
| --- | --- |
| Missing training paths silently selected other data | Explicit local-path failure; no fallback to held-out data |
| Preparing a resumed data-loader iterator consumed learner RNG | Dedicated loader generators isolate iterator seeding from the restored learner stream |
| SDAR ignored the activation-checkpointing configuration | The controller enables or disables checkpointing according to the requested strategy |
| Code evaluation omitted a final step outside the periodic cadence | The shared evaluation schedule includes the final training step |
| Restarting a completed run omitted checkpoint evaluation | Due evaluation is replayed from the exported checkpoint for both math and code tasks |
| Online evaluation accepted task `code` but selected the MBPP evaluator | Training-code evaluation uses its corresponding evaluator |

Checkpoint evaluation loads a separate model replica and isolates evaluation RNG. It can rerun an already completed evaluation when a completed run is restarted; users should account for that evaluation time.

After correction and verification, no release-blocking finding remained on either axis. There were two actionable Standards findings and six Spec findings; the training-data fallback was shared by both reviews.
