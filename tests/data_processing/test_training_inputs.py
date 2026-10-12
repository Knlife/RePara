import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from accelerate import Accelerator
from datasets import Dataset
import torch

from data.code import load_code_dataset_and_reward
from data.math import load_gsm8k_dataset_and_reward, load_math_dataset_and_reward


class TrainingInputsTest(unittest.TestCase):
    def test_missing_local_paths_fail_with_other_datasets_present(self):
        with tempfile.TemporaryDirectory() as directory:
            previous = Path.cwd()
            try:
                os.chdir(directory)
                Path("datasets/gsm8k/main").mkdir(parents=True)
                Path("datasets/math500/data").mkdir(parents=True)
                for loader in (load_gsm8k_dataset_and_reward, load_math_dataset_and_reward):
                    for path in ("datasets/missing/train.parquet", str(Path(directory) / "missing.parquet")):
                        with self.subTest(loader=loader.__name__, path=path), patch("data.math.load_dataset") as load:
                            with self.assertRaisesRegex(FileNotFoundError, "Training data path"):
                                loader(path, num_workers=0)
                            load.assert_not_called()
            finally:
                os.chdir(previous)

    def test_hub_identifier_is_preserved(self):
        cases = (
            (load_gsm8k_dataset_and_reward, "research/gsm8k", {"question": ["q"], "answer": ["#### 1"]}, ("main",)),
            (load_math_dataset_and_reward, "research/math", {"problem": ["p"], "solution": ["1"]}, ()),
        )
        for loader, identifier, data, config in cases:
            with self.subTest(identifier=identifier), patch("data.math.load_dataset", return_value=Dataset.from_dict(data)) as load:
                dataloader, _ = loader(identifier, num_workers=0)
                load.assert_called_once_with(identifier, *config, split="train")
                self.assertEqual(len(dataloader.dataset), 1)

    def test_local_parquet_directory_and_file_use_configured_rows(self):
        cases = (
            (load_gsm8k_dataset_and_reward, "main", {"question": ["configured"], "answer": ["#### 1"]}, "problems"),
            (load_math_dataset_and_reward, "data", {"problem": ["configured"], "solution": ["1"]}, "problems"),
        )
        with tempfile.TemporaryDirectory() as directory, patch(
            "datasets.config.HF_DATASETS_CACHE", Path(directory) / "cache",
        ):
            for loader, subdir, rows, field in cases:
                source = Path(directory) / loader.__name__ / subdir
                source.mkdir(parents=True)
                path = source / "train-00000.parquet"
                Dataset.from_dict(rows).to_parquet(path)
                for configured in (str(path), str(source.parent)):
                    with self.subTest(configured=configured):
                        dataloader, _ = loader(configured, num_workers=0)
                        self.assertTrue(next(iter(dataloader))[field][0].startswith("configured"))

    def test_resumed_loader_preserves_dropout_updates(self):
        accelerator = Accelerator(cpu=True)
        cases = (
            (load_gsm8k_dataset_and_reward, "data.math.load_dataset", {"question": ["q"] * 8, "answer": ["#### 1"] * 8}),
            (load_math_dataset_and_reward, "data.math.load_dataset", {"problem": ["p"] * 8, "solution": ["1"] * 8}),
            (load_code_dataset_and_reward, "data.code.load_dataset", {"question": ["q"] * 8, "test_cases": [[]] * 8}),
        )
        for loader, target, rows in cases:
            with self.subTest(loader=loader.__name__), patch(target, return_value=Dataset.from_dict(rows)):
                dataloader, _ = loader("research/training", batch_size=2, num_workers=0)
                iterator = iter(accelerator.prepare_data_loader(dataloader))
                first = next(iterator)
                torch.manual_seed(77)
                model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Dropout(0.4))
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

                def update(batch, learner, optim):
                    ids = torch.tensor(batch["sample_ids"], dtype=torch.float32)
                    inputs = (ids[:, None] + torch.arange(8)[None, :]) / 8
                    optim.zero_grad()
                    loss = (learner(inputs) - (ids % 2)[:, None]).square().mean()
                    loss.backward()
                    optim.step()

                update(first, model, optimizer)
                saved_model = copy.deepcopy(model)
                saved_optimizer = copy.deepcopy(optimizer.state_dict())
                saved_rng = torch.get_rng_state().clone()
                uninterrupted_batch = next(iterator)
                update(uninterrupted_batch, model, optimizer)

                resumed_loader, _ = loader("research/training", batch_size=2, num_workers=0, start_position=2)
                resumed_optimizer = torch.optim.AdamW(saved_model.parameters(), lr=0.01)
                resumed_optimizer.load_state_dict(saved_optimizer)
                torch.set_rng_state(saved_rng)
                resumed_batch = next(iter(accelerator.prepare_data_loader(resumed_loader)))
                self.assertTrue(torch.equal(torch.get_rng_state(), saved_rng))
                self.assertEqual(resumed_batch["sample_ids"], uninterrupted_batch["sample_ids"])
                update(resumed_batch, saved_model, resumed_optimizer)
                for actual, expected in zip(saved_model.parameters(), model.parameters()):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
