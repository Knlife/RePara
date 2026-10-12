"""CPU/Gloo regression: uneven shards, native SDAR early exits, rank failures."""

from datetime import timedelta
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from utils.configuration.config import TrainConfig
from utils.evaluation.online_eval import run_online_evaluation


class ToyTokenizer:
    eos_token_id = 2

    def __len__(self):
        return 6

    mask_token_id = 5

    def apply_chat_template(self, messages, **kwargs):
        return [item[0]["content"] for item in messages]

    def __call__(self, prompts, **kwargs):
        return {"input_ids": torch.full((1, len(prompts[0]) % 5 + 1), 3, dtype=torch.long)}

    def batch_decode(self, ids, **kwargs):
        return ["1"] * len(ids)


class ToyPredictor(torch.nn.Module):
    def __init__(self, rank):
        super().__init__()
        self.config = SimpleNamespace(mask_token_id=5, block_size=4)
        self.rank = rank

    def forward(self, input_ids, **kwargs):
        logits = torch.zeros((*input_ids.shape, 6))
        # Rank zero accepts all confident tokens early; rank one consumes
        # its full denoising budget. No FSDP collectives may run in forward.
        logits[..., 1] = 20 if self.rank == 0 else 0.1
        return SimpleNamespace(logits=logits)


class ToyDataset:
    def __init__(self, rows=None):
        self.rows = rows if rows is not None else [
            {"question": "a", "answer": "#### 1"},
            {"question": "longer", "answer": "#### 1"},
            {"question": "three", "answer": "#### 1"},
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]

    def select(self, indices):
        return ToyDataset([self.rows[i] for i in indices])


def _worker(rank, directory, mode):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{directory}/rendezvous", rank=rank,
        world_size=2, timeout=timedelta(seconds=45),
    )
    try:
        for family in ("llada", "sdar"):
            cfg = TrainConfig(
                model_path=f"{directory}/ckpt-000016", output_dir=f"{directory}/{family}",
                model_family=family, gsm8k_data_path=f"{directory}/train.parquet",
                save_every=16, eval_every=16,
                eval_config={"gsm8k_data_files": f"{directory}/test.parquet", "gen_length": 8, "block_length": 4},
            )

            def loader(*args):
                if mode == "load_failure" and rank == 1:
                    raise RuntimeError("injected rank-one load failure")
                return ToyPredictor(rank), ToyTokenizer()

            import eval as evaluator
            real_save = evaluator.save_log

            def save(args, metrics):
                if mode == "write_failure":
                    raise OSError("injected result write failure")
                return real_save(args, metrics)

            with patch("utils.evaluation.online_eval._load_checkpoint_for_eval", side_effect=loader), patch("eval.load_eval_dataset", return_value=ToyDataset()), patch("eval.save_log", side_effect=save):
                if mode == "success":
                    result = run_online_evaluation(device="cpu", config=cfg, checkpoint_path=cfg.model_path, completed_step=16)
                    if rank == 0:
                        assert len(result) == 3
                        assert all(m["total"] == 3 and m["correct"] == 3 for m in result.values())
                        for m in result.values():
                            payload = json.loads(Path(m["evaluation_log"]).read_text())
                            assert payload["ckpt_path"] == cfg.model_path
                            assert payload["gen_length"] == 8
                else:
                    try:
                        run_online_evaluation(device="cpu", config=cfg, checkpoint_path=cfg.model_path, completed_step=16)
                    except RuntimeError as exc:
                        assert "injected" in str(exc)
                    else:
                        raise AssertionError("All ranks must receive the evaluation failure")
            dist.barrier()
    finally:
        dist.destroy_process_group()


class OnlineEvalDistributedTest(unittest.TestCase):
    def _run(self, mode):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "test.parquet").touch()
            (Path(directory) / "ckpt-000016").mkdir()
            mp.spawn(_worker, args=(directory, mode), nprocs=2, join=True)
            if mode != "success":
                self.assertFalse(list(Path(directory).rglob("summary.json")))

    def test_uneven_shards_and_sdar_dynamic_early_exits(self):
        self._run("success")

    def test_rank_local_load_failure_propagates(self):
        self._run("load_failure")

    def test_rank_zero_result_write_failure_propagates(self):
        self._run("write_failure")


if __name__ == "__main__":
    unittest.main()
