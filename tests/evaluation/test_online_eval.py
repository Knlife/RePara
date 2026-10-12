import json
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from utils.configuration.config import TrainConfig, resolve_online_eval_config
from utils.evaluation.online_eval import (
    _replicated_model_loading, build_online_eval_runs, isolated_eval_rng,
    online_evaluation_due, run_online_evaluation, validate_online_eval_inputs,
)


class TokenizerStub(SimpleNamespace):
    def __len__(self):
        return 128


def config_for(directory, *, family="llada", dataset="gsm8k"):
    key = "math500_data_files" if dataset == "math" else "gsm8k_data_files"
    return TrainConfig(
        model_path=str(Path(directory) / "ckpt-000016"), model_family=family,
        output_dir=directory, dataset=dataset,
        gsm8k_data_path=str(Path(directory) / "train.parquet"),
        math_data_path=str(Path(directory) / "train.parquet"),
        save_every=16, eval_every=16,
        eval_config={key: str(Path(directory) / "test.parquet")},
    )


class OnlineEvalConfigTest(unittest.TestCase):
    def test_all_four_model_dataset_combinations(self):
        for family in ("llada", "sdar"):
            for dataset in ("gsm8k", "math"):
                with self.subTest(family=family, dataset=dataset):
                    cfg = config_for("/runs/test", family=family, dataset=dataset)
                    cfg.validate()
                    runs = build_online_eval_runs(cfg, "/ckpt/16", 16, mask_id=99, eos_token_id=2)
                    self.assertEqual([r.mtp_size for r in runs], [1, 2, 4])
                    self.assertEqual([r.steps for r in runs], [256, 128, 64])
                    self.assertEqual(len({r.output for r in runs}), 3)
                    for r in runs:
                        self.assertEqual(r.ckpt_path, "/ckpt/16")
                        self.assertEqual(r.gen_length, 256)
                        self.assertEqual(r._online_metadata["temperature"], 0)
                        self.assertEqual(r.task, "math500" if dataset == "math" else "gsm8k")
                    if family == "sdar":
                        self.assertEqual([r.sdar_denoising_steps for r in runs], [4, 2, 1])
                        self.assertTrue(all(r.sdar_remasking_strategy == "low_confidence_dynamic" for r in runs))
                        self.assertTrue(all(r.sdar_confidence_threshold == 0.85 for r in runs))
                        self.assertTrue(all(r.sdar_stopping_criteria_idx == [2] for r in runs))

    def test_schedule_and_final_checkpoint(self):
        cfg = config_for("/runs/test")
        self.assertEqual([s for s in range(129) if online_evaluation_due(cfg, s)], list(range(16, 129, 16)))
        cfg.total_steps = 18
        self.assertTrue(online_evaluation_due(cfg, 18))
        cfg.eval_every = 0
        self.assertFalse(online_evaluation_due(cfg, 16))

    def test_old_configs_remain_opt_in(self):
        cfg = TrainConfig(model_path="/model", output_dir="/out", gsm8k_data_path="/train")
        cfg.validate()
        self.assertFalse(online_evaluation_due(cfg, 128))

    def test_stale_save_every_override_rejected(self):
        cfg = config_for("/runs/test")
        cfg.save_every = 6
        with self.assertRaisesRegex(ValueError, "SAVE_EVERY"):
            cfg.validate()

    def test_missing_spec_or_wrong_task_rejected(self):
        cfg = config_for("/runs/test")
        cfg.eval_config = None
        with self.assertRaisesRegex(ValueError, "eval_config"):
            cfg.validate()
        cfg = config_for("/runs/test")
        cfg.eval_config["tasks"] = ["math500"]
        with self.assertRaisesRegex(ValueError, "tasks"):
            cfg.validate()

    def test_invalid_rates_rejected(self):
        for rates in ([], [1, 1], [True], [3], [0], [1.5], "1"):
            with self.subTest(rates=rates):
                cfg = config_for("/runs/test", family="sdar")
                cfg.eval_config["mtp_size"] = rates
                with self.assertRaises(ValueError):
                    cfg.validate()

    def test_unresolved_nested_environment_rejected(self):
        cfg = config_for("/runs/test")
        cfg.eval_config["gsm8k_data_files"] = "${MISSING_EVAL_DATA}"
        with self.assertRaisesRegex(ValueError, "Unresolved"):
            cfg.validate()

    def test_no_remote_or_nonzero_temperature(self):
        for key, value in (("allow_remote_files", True), ("sdar_temperature", 1), ("unknown", 1)):
            cfg = config_for("/runs/test")
            cfg.eval_config[key] = value
            with self.assertRaises(ValueError):
                cfg.validate()

    def test_missing_or_training_data_fails_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = config_for(directory)
            with self.assertRaises(FileNotFoundError):
                validate_online_eval_inputs(cfg)
            Path(cfg.data_path).touch()
            cfg.eval_config["gsm8k_data_files"] = cfg.data_path
            with self.assertRaisesRegex(ValueError, "held-out"):
                validate_online_eval_inputs(cfg)


class OnlineEvalRuntimeTest(unittest.TestCase):
    def test_rng_restored_on_success_and_exception(self):
        for fail in (False, True):
            random.seed(10)
            np.random.seed(10)
            torch.manual_seed(10)
            before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
            try:
                with isolated_eval_rng(113, "cpu"):
                    random.random()
                    np.random.rand()
                    torch.rand(4)
                    if fail:
                        raise RuntimeError("test")
            except RuntimeError:
                pass
            self.assertEqual(random.getstate(), before[0])
            np.testing.assert_equal(np.random.get_state(), before[1])
            torch.testing.assert_close(torch.get_rng_state(), before[2])

    def test_fsdp_loading_flag_restored_on_exception(self):
        with patch.dict(os.environ, {"ACCELERATE_USE_FSDP": "true"}):
            with self.assertRaises(RuntimeError):
                with _replicated_model_loading():
                    self.assertEqual(os.environ["ACCELERATE_USE_FSDP"], "false")
                    raise RuntimeError("test")
            self.assertEqual(os.environ["ACCELERATE_USE_FSDP"], "true")

    def test_one_checkpoint_load_three_results_and_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = config_for(directory, family="sdar")
            Path(cfg.model_path).mkdir()
            (Path(directory) / "test.parquet").touch()
            model = SimpleNamespace(config=SimpleNamespace(mask_token_id=99))
            tokenizer = TokenizerStub(eos_token_id=2)
            calls = []

            def evaluate(model, tokenizer, device, args):
                from eval import save_log
                self.assertFalse(torch.is_grad_enabled())
                calls.append(args.sdar_denoising_steps)
                result = {"accuracy": 0.5, "correct": 2, "total": 4}
                save_log(args, result)
                return result

            with patch("utils.evaluation.online_eval._load_checkpoint_for_eval", return_value=(model, tokenizer)) as loader, patch("eval.eval_math", side_effect=evaluate):
                results = run_online_evaluation(device="cpu", config=cfg, checkpoint_path=cfg.model_path, completed_step=16)
            loader.assert_called_once()
            self.assertEqual(calls, [4, 2, 1])
            self.assertEqual(len(results), 3)
            root = Path(directory) / "online-eval/gsm8k/step-000016"
            summary = json.loads((root / "summary.json").read_text())
            self.assertEqual(summary["checkpoint_path"], cfg.model_path)
            logs = list(root.glob("diff*/*/evaluation_log.json"))
            self.assertEqual(len(logs), 3)
            for path in logs:
                log = json.loads(path.read_text())
                self.assertEqual(log["completed_step"], 16)
                self.assertEqual(log["ckpt_path"], cfg.model_path)
                self.assertEqual(log["temperature"], 0)
                self.assertTrue((path.parent / "effective_config.yaml").exists())

    def test_loading_failure_is_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = config_for(directory)
            Path(cfg.model_path).mkdir()
            (Path(directory) / "test.parquet").touch()
            with patch("utils.evaluation.online_eval._load_checkpoint_for_eval", side_effect=RuntimeError("out of memory")):
                with self.assertRaisesRegex(RuntimeError, "out of memory"):
                    run_online_evaluation(device="cpu", config=cfg, checkpoint_path=cfg.model_path, completed_step=16)
            self.assertFalse(list(Path(directory).rglob("summary.json")))


if __name__ == "__main__":
    unittest.main()
