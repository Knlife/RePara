import os
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path
from unittest.mock import patch

import yaml

from eval import (
    eval_setting_name,
    expand_eval_configs,
    parse_args as parse_eval_args,
)
from train import parse_args as parse_train_args
from utils.configuration.config import EvalConfig, TrainConfig, load_config, snapshot_config


REPOSITORY_ENVIRONMENT = {
    "MODEL_PATH": "/models/llada",
    "OUTPUT_DIR": "/runs/experiment",
    "BATCH_SIZE_PER_DEVICE": "8",
    "GRAD_ACCUMULATION": "1",
    "GSM8K_DATA_PATH": "/datasets/gsm8k",
    "MATH_DATA_PATH": "/datasets/math",
    "CODE_DATA_PATH": "/datasets/code.jsonl",
    "CODE_EVAL_DATA_FILE": "/datasets/code-eval.jsonl",
    "GSM8K_EVAL_DATA_FILES": "/datasets/gsm8k/test-*.parquet",
    "MATH500_EVAL_DATA_FILES": "/datasets/math/test-*.parquet",
    "HUMANEVAL_DATA_FILE": "/datasets/HumanEval.jsonl.gz",
    "MBPP_DATA_FILE": "/datasets/mbpp.jsonl",
    "MBPP_TEST_FILE": "/datasets/mbpp_test.jsonl",
    "WANDB_GROUP": "test-group",
    "WANDB_RUN_NAME": "test-run",
    "WANDB_MODE": "offline",
}

class ConfigTest(unittest.TestCase):
    def test_repository_train_matrix_matches_runtime_schema(self):
        config_dir = Path(__file__).resolve().parents[2] / "configs/train"
        expected_names = {
            f"justgrpo-{family}-{dataset}.yaml"
            for family in ("llada", "sdar")
            for dataset in ("gsm8k", "math", "code")
        } | {
            f"repara-{family}-{dataset}-{variant}.yaml"
            for family in ("llada", "sdar")
            for dataset in ("gsm8k", "math", "code")
            for variant in ("fix2", "adaptive095")
        }
        self.assertTrue(expected_names.issubset({p.name for p in config_dir.rglob("*.yaml")}))
        allowed_keys = {field.name for field in fields(TrainConfig)}
        self.assertFalse(any(name.startswith("lora") for name in allowed_keys))
        with patch.dict(os.environ, REPOSITORY_ENVIRONMENT, clear=True):
            for path in sorted(config_dir.rglob("*.yaml")):
                with self.subTest(config=str(path.relative_to(config_dir))):
                    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
                    self.assertLessEqual(set(payload["train"]), allowed_keys)
                    config, _, _ = parse_train_args(["--config", str(path)])
                    self.assertEqual(
                        path.relative_to(config_dir).parts[:2],
                        (config.model_family, config.dataset),
                    )
                    self.assertIn(config.method, ("justgrpo", "fixed_mtp", "ar_mtp"))
                    self.assertIn(config.model_family, ("llada", "sdar"))
                    self.assertEqual(config.eval_every, 0)
                    for stage in config.resolved_train_schedule():
                        self.assertIn(stage.method, ("justgrpo", "fixed_mtp", "ar_mtp"))

    def test_repository_eval_matrix_covers_all_model_task_decoders(self):
        config_dir = Path(__file__).resolve().parents[2] / "configs/eval"
        allowed_keys = {field.name for field in fields(EvalConfig)}
        protocols = set()
        with patch.dict(os.environ, REPOSITORY_ENVIRONMENT, clear=True):
            for path in sorted(config_dir.rglob("*.yaml")):
                with self.subTest(config=str(path.relative_to(config_dir))):
                    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
                    self.assertLessEqual(set(payload["eval"]), allowed_keys)
                    config, _, _ = parse_eval_args(["--config", str(path)])
                    self.assertEqual(
                        path.relative_to(config_dir).parts[:2],
                        (config.model_family, config.task),
                    )
                    runs = expand_eval_configs(config)
                    protocols.add((config.model_family, config.task, config.generation_method))
                    self.assertEqual({run.gen_length for run in runs}, {128, 256, 512})
                    self.assertEqual(len({eval_setting_name(run) for run in runs}), len(runs))
                    self.assertEqual(config.temperature, 0)
                    if config.generation_method == "diffusion":
                        self.assertEqual({run.mtp_size for run in runs}, {1, 2, 4, 8, 16})
                    if config.generation_method == "sdar_native":
                        self.assertEqual(config.sdar_temperature, 0)
                        self.assertEqual(config.sdar_stopping_criteria_idx, [])
        expected = {
            (family, task, method)
            for family in ("llada", "sdar")
            for task in ("gsm8k", "math500", "humaneval", "mbpp")
            for method in (("diffusion", "fixed_mtp", "ar_mtp") if family == "llada" else ("sdar_native", "fixed_mtp", "ar_mtp"))
        }
        self.assertEqual(protocols, expected)

    def test_full_parameter_training_rejects_excluded_methods_and_fields(self):
        for method in ("ar_mtp_mismatch", "sdar_native"):
            config = TrainConfig(model_path="/model", output_dir="/run", gsm8k_data_path="/data", method=method)
            with self.subTest(method=method), self.assertRaisesRegex(ValueError, "Unknown training method"):
                config.validate()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.yaml"
            path.write_text("train:\n  model_path: /model\n  output_dir: /run\n  gsm8k_data_path: /data\n  lora: true\n")
            with self.assertRaisesRegex(ValueError, "Unknown train config field"):
                load_config(path, section="train", config_class=TrainConfig)

    def test_loads_one_strict_section(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.yaml"
            path.write_text(
                "train:\n"
                "  model_path: /models/step-0\n"
                "  output_dir: /runs/experiment\n"
                "  gsm8k_data_path: /datasets/gsm8k\n"
                "  total_steps: 10\n",
                encoding="utf-8",
            )

            config, resolved = load_config(
                str(path),
                section="train",
                config_class=TrainConfig,
            )

            self.assertEqual(config.model_path, "/models/step-0")
            self.assertEqual(config.total_steps, 10)
            self.assertEqual(resolved, path.resolve())

    def test_environment_expansion_preserves_scalar_types(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.yaml"
            path.write_text(
                "train:\n"
                "  model_path: ${MODEL_PATH}\n"
                "  batch_size_per_device: ${MBS}\n"
                "  output_dir: ${RUN_ROOT}/trial\n"
                "  dataset: code\n"
                "  code_data_path: /datasets/code.jsonl\n"
                "  format_reward_alpha: ${FORMAT_REWARD_ALPHA:-1.0}\n",
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {
                    "MODEL_PATH": "/models/step-0",
                    "MBS": "16",
                    "RUN_ROOT": "/runs",
                    "FORMAT_REWARD_ALPHA": "0.25",
                },
                clear=True,
            ):
                config, _ = load_config(
                    str(path),
                    section="train",
                    config_class=TrainConfig,
                )

            self.assertEqual(config.model_path, "/models/step-0")
            self.assertEqual(config.batch_size_per_device, 16)
            self.assertIsInstance(config.batch_size_per_device, int)
            self.assertEqual(config.output_dir, "/runs/trial")
            self.assertEqual(config.format_reward_alpha, 0.25)

    def test_environment_default_expansion_is_used_when_variable_is_unset(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.yaml"
            path.write_text(
                "train:\n"
                "  model_path: /models/step-0\n"
                "  output_dir: /runs/experiment\n"
                "  dataset: code\n"
                "  code_data_path: /datasets/code.jsonl\n"
                "  format_reward_alpha: ${FORMAT_REWARD_ALPHA:-1.0}\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                config, _ = load_config(
                    str(path), section="train", config_class=TrainConfig
                )

            self.assertEqual(config.format_reward_alpha, 1.0)

    def test_final_config_validation_rejects_missing_and_unresolved_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.yaml"
            path.write_text(
                "train:\n"
                "  dataset: code\n"
                "  code_data_path: /datasets/code.jsonl\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "Missing required train config field.*model_path, output_dir",
            ):
                load_config(
                    str(path),
                    section="train",
                    config_class=TrainConfig,
                )

            path.write_text(
                "train:\n"
                "  model_path: ${MODEL_PATH}\n"
                "  output_dir: /runs/experiment\n"
                "  dataset: gsm8k\n"
                "  gsm8k_data_path: /datasets/gsm8k\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(
                    ValueError,
                    "Unresolved environment variable.*model_path",
                ):
                    load_config(
                        str(path),
                        section="train",
                        config_class=TrainConfig,
                    )

    def test_overrides_are_applied_before_final_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "eval.yaml"
            path.write_text(
                "eval:\n"
                "  task: gsm8k\n"
                "  ckpt_path: ${MODEL_PATH}\n"
                "  output: ${OUTPUT_DIR}\n"
                "  gsm8k_data_files: /datasets/gsm8k/test.parquet\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {}, clear=True):
                config, _ = load_config(
                    str(path),
                    section="eval",
                    config_class=EvalConfig,
                    overrides={
                        "ckpt_path": "/models/override",
                        "output": "/eval/override",
                    },
                )

            self.assertEqual(config.ckpt_path, "/models/override")
            self.assertEqual(config.output, "/eval/override")

    def test_algorithm_constraints_are_validated_with_the_config(self):
        train_config = TrainConfig(
            model_path="/models/llada",
            output_dir="/runs/train",
            gsm8k_data_path="/datasets/gsm8k",
            method="unsupported",
        )
        with self.assertRaisesRegex(ValueError, "Unknown training method"):
            train_config.validate()

        TrainConfig(
            model_path="/models/llada",
            output_dir="/runs/train",
            dataset="math",
            math_data_path="/datasets/math",
            resume_ckpt="/runs/source/training-state-000010",
        ).validate()

        eval_config = EvalConfig(
            ckpt_path="/models/llada",
            output="/runs/eval",
            gsm8k_data_files="/datasets/gsm8k/test.parquet",
            batch_size=2,
        )
        with self.assertRaisesRegex(ValueError, "batch_size=1"):
            eval_config.validate()

    def test_rejects_unknown_fields_and_mixed_sections(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text("train:\n  typo: true\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unknown train config"):
                load_config(str(path), section="train", config_class=TrainConfig)

            path.write_text("train: {}\neval: {}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "top-level 'train'"):
                load_config(str(path), section="train", config_class=TrainConfig)

            path.write_text(
                "train:\n"
                "  model_path: /models/step-0\n"
                "  output_dir: /runs/experiment\n"
                "  gsm8k_data_path: /datasets/gsm8k\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Unknown train config.*typo"):
                load_config(
                    str(path),
                    section="train",
                    config_class=TrainConfig,
                    overrides={"typo": True},
                )

    def test_snapshots_source_effective_values_and_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.yaml"
            source.write_text("train:\n  total_steps: 10\n", encoding="utf-8")
            output = root / "run"

            snapshot_config(
                output,
                source_path=source,
                section="train",
                effective_config={"total_steps": 12},
                overrides={"output_dir": str(output)},
            )

            self.assertEqual(
                (output / "source_config.yaml").read_text(encoding="utf-8"),
                source.read_text(encoding="utf-8"),
            )
            effective = yaml.safe_load(
                (output / "effective_config.yaml").read_text(encoding="utf-8")
            )
            self.assertEqual(effective["train"]["total_steps"], 12)
            self.assertEqual(effective["overrides"]["output_dir"], str(output))

    def test_train_cli_overrides_paths_and_wandb_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.yaml"
            path.write_text(
                "train:\n"
                "  model_path: /models/step-0\n"
                "  output_dir: /runs/from-yaml\n"
                "  gsm8k_data_path: /datasets/gsm8k\n",
                encoding="utf-8",
            )

            config, resolved, overrides = parse_train_args(
                [
                    "--config",
                    str(path),
                    "--run_dir",
                    "/runs/override",
                    "--resume_ckpt",
                    "/runs/source/training-state-000010",
                    "--wandb_group",
                    "repara-code",
                    "--wandb_run_name",
                    "stage-20260904",
                ]
            )

            self.assertEqual(config.model_path, "/models/step-0")
            self.assertEqual(config.output_dir, "/runs/override")
            self.assertEqual(
                config.resume_ckpt, "/runs/source/training-state-000010"
            )
            self.assertEqual(resolved, path.resolve())
            self.assertEqual(config.wandb_group, "repara-code")
            self.assertEqual(config.wandb_run_name, "stage-20260904")
            self.assertEqual(
                overrides,
                {
                    "output_dir": "/runs/override",
                    "resume_ckpt": "/runs/source/training-state-000010",
                    "wandb_group": "repara-code",
                    "wandb_run_name": "stage-20260904",
                },
            )

    def test_eval_cli_only_overrides_checkpoint_and_output_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "eval.yaml"
            path.write_text(
                "eval:\n"
                "  ckpt_path: /models/from-yaml\n"
                "  task: gsm8k\n"
                "  gsm8k_data_files: /datasets/gsm8k/test.parquet\n"
                "  output: /eval/from-yaml\n",
                encoding="utf-8",
            )

            config, resolved, overrides = parse_eval_args(
                [
                    "--config",
                    str(path),
                    "--ckpt_path",
                    "/models/override",
                    "--output",
                    "/eval/output",
                ]
            )

            self.assertEqual(config.ckpt_path, "/models/override")
            self.assertEqual(config.output, "/eval/output")
            self.assertEqual(resolved, path.resolve())
            self.assertEqual(
                overrides,
                {"ckpt_path": "/models/override", "output": "/eval/output"},
            )


if __name__ == "__main__":
    unittest.main()
