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



    def test_unresolved_nested_environment_rejected(self):
        cfg = config_for("/runs/test")
        cfg.eval_config["gsm8k_data_files"] = "${MISSING_EVAL_DATA}"
        with self.assertRaisesRegex(ValueError, "Unresolved"):
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
