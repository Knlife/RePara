import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from utils.checkpointing.checkpoint import save_training_checkpoint


class _FakeModel:
    def __init__(self, events):
        self.events = events
        self.config = SimpleNamespace(auto_map={})

    def save_pretrained(self, path, *, state_dict, safe_serialization):
        self.events.append(("save_pretrained", state_dict, safe_serialization))
        output = Path(path)
        output.mkdir()
        (output / "model.safetensors").write_text("model", encoding="utf-8")


class _FakeTokenizer:
    def __init__(self, events):
        self.events = events

    def save_pretrained(self, path):
        self.events.append(("save_tokenizer",))
        output = Path(path)
        (output / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
        (output / "tokenizer.json").write_text("{}\n", encoding="utf-8")


class _FakeAccelerator:
    is_main_process = True
    device = torch.device("cpu")

    def __init__(self):
        self.events = []
        self.model = _FakeModel(self.events)

    def wait_for_everyone(self):
        self.events.append(("barrier",))

    def save_state(self, path):
        self.events.append(("save_state",))
        output = Path(path)
        output.mkdir()
        (output / "optimizer.bin").write_text("optimizer", encoding="utf-8")

    def get_state_dict(self, model):
        self.events.append(("get_state_dict",))
        return {"weight": "state"}

    def unwrap_model(self, model):
        self.events.append(("unwrap_model",))
        return self.model

    def reduce(self, tensor, reduction):
        self.events.append(("reduce", reduction))
        return tensor


class _FailingModel(_FakeModel):
    def save_pretrained(self, path, *, state_dict, safe_serialization):
        Path(path).mkdir()
        raise RuntimeError("injected model save failure")


class _FailingTokenizer(_FakeTokenizer):
    def save_pretrained(self, path):
        self.events.append(("save_tokenizer",))
        raise RuntimeError("injected tokenizer save failure")


class _FailingStateDictAccelerator(_FakeAccelerator):
    def get_state_dict(self, model):
        self.events.append(("get_state_dict",))
        raise RuntimeError("injected state dict failure")


class CheckpointTest(unittest.TestCase):
    @staticmethod
    def tokenizer(accelerator):
        return _FakeTokenizer(accelerator.events)

    def test_state_is_saved_before_hf_state_collection_and_renamed_atomically(self):
        accelerator = _FakeAccelerator()
        with tempfile.TemporaryDirectory() as directory:
            paths = save_training_checkpoint(
                accelerator=accelerator,
                model=object(),
                tokenizer=self.tokenizer(accelerator),
                output_dir=directory,
                completed_steps=5,
            )

            self.assertTrue(Path(paths["state"]).is_dir())
            self.assertTrue(Path(paths["model"]).is_dir())
            self.assertTrue((Path(paths["model"]) / "tokenizer_config.json").is_file())
            self.assertTrue((Path(paths["model"]) / "tokenizer.json").is_file())
            self.assertFalse(Path(f"{paths['state']}.incomplete").exists())
            self.assertFalse(Path(f"{paths['model']}.incomplete").exists())
            latest = Path(directory) / "latest"
            self.assertTrue(latest.is_symlink())
            self.assertEqual(latest.resolve(), Path(paths["state"]).resolve())

        event_names = [event[0] for event in accelerator.events]
        self.assertLess(event_names.index("save_state"), event_names.index("get_state_dict"))
        self.assertLess(event_names.index("get_state_dict"), event_names.index("save_pretrained"))
        self.assertLess(event_names.index("save_pretrained"), event_names.index("save_tokenizer"))

    def test_latest_marker_tracks_newest_complete_state(self):
        accelerator = _FakeAccelerator()
        with tempfile.TemporaryDirectory() as directory:
            first = save_training_checkpoint(
                accelerator=accelerator,
                model=object(),
                tokenizer=self.tokenizer(accelerator),
                output_dir=directory,
                completed_steps=5,
            )
            second = save_training_checkpoint(
                accelerator=accelerator,
                model=object(),
                tokenizer=self.tokenizer(accelerator),
                output_dir=directory,
                completed_steps=10,
            )

            latest = Path(directory) / "latest"
            self.assertEqual(latest.resolve(), Path(second["state"]).resolve())
            self.assertNotEqual(latest.resolve(), Path(first["state"]).resolve())


    def test_existing_checkpoint_is_not_overwritten(self):
        accelerator = _FakeAccelerator()
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "training-state-000005").mkdir()
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                save_training_checkpoint(
                    accelerator=accelerator,
                    model=object(),
                    tokenizer=self.tokenizer(accelerator),
                    output_dir=directory,
                    completed_steps=5,
                )
        event_names = [event[0] for event in accelerator.events]
        self.assertNotIn("save_state", event_names)
        self.assertNotIn("get_state_dict", event_names)

    def test_failed_model_save_is_not_published_as_complete(self):
        accelerator = _FakeAccelerator()
        accelerator.model = _FailingModel(accelerator.events)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "injected model save failure"):
                save_training_checkpoint(
                    accelerator=accelerator,
                    model=object(),
                    tokenizer=self.tokenizer(accelerator),
                    output_dir=directory,
                    completed_steps=5,
                )

            self.assertTrue((Path(directory) / "training-state-000005").is_dir())
            self.assertFalse((Path(directory) / "ckpt-000005").exists())
            self.assertTrue((Path(directory) / "ckpt-000005.incomplete").is_dir())
            self.assertFalse((Path(directory) / "latest").exists())

    def test_interrupted_model_save_can_retry_same_step(self):
        failing = _FakeAccelerator()
        failing.model = _FailingModel(failing.events)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "injected model save failure"):
                save_training_checkpoint(
                    accelerator=failing,
                    model=object(),
                    tokenizer=self.tokenizer(failing),
                    output_dir=directory,
                    completed_steps=5,
                )

            retry = _FakeAccelerator()
            paths = save_training_checkpoint(
                accelerator=retry,
                model=object(),
                tokenizer=self.tokenizer(retry),
                output_dir=directory,
                completed_steps=5,
            )
            self.assertTrue(Path(paths["state"]).is_dir())
            self.assertTrue(Path(paths["model"]).is_dir())
            self.assertEqual(
                (Path(directory) / "latest").resolve(), Path(paths["state"]).resolve()
            )

    def test_orphan_state_after_state_dict_failure_can_retry_same_step(self):
        failing = _FailingStateDictAccelerator()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "injected state dict failure"):
                save_training_checkpoint(
                    accelerator=failing,
                    model=object(),
                    tokenizer=self.tokenizer(failing),
                    output_dir=directory,
                    completed_steps=5,
                )

            state = Path(directory) / "training-state-000005"
            self.assertTrue(state.is_dir())
            self.assertTrue((state / ".model-pending").is_file())
            self.assertFalse((Path(directory) / "ckpt-000005").exists())

            retry = _FakeAccelerator()
            paths = save_training_checkpoint(
                accelerator=retry,
                model=object(),
                tokenizer=self.tokenizer(retry),
                output_dir=directory,
                completed_steps=5,
            )
            self.assertTrue(Path(paths["state"]).is_dir())
            self.assertTrue(Path(paths["model"]).is_dir())
            self.assertFalse((Path(paths["state"]) / ".model-pending").exists())

    def test_failed_tokenizer_save_is_not_published_as_complete(self):
        accelerator = _FakeAccelerator()
        tokenizer = _FailingTokenizer(accelerator.events)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "injected tokenizer save failure"):
                save_training_checkpoint(
                    accelerator=accelerator,
                    model=object(),
                    tokenizer=tokenizer,
                    output_dir=directory,
                    completed_steps=5,
                )

            self.assertTrue((Path(directory) / "training-state-000005").is_dir())
            self.assertFalse((Path(directory) / "ckpt-000005").exists())
            self.assertTrue((Path(directory) / "ckpt-000005.incomplete").is_dir())

    def test_custom_model_code_is_packaged_with_checkpoint(self):
        accelerator = _FakeAccelerator()
        accelerator.model.config.auto_map = {
            "AutoModel": "modeling_custom.CustomModel",
            "AutoConfig": "configuration_custom.CustomConfig",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "modeling_custom.py").write_text("# model\n", encoding="utf-8")
            (source / "configuration_custom.py").write_text(
                "# config\n", encoding="utf-8"
            )
            output = root / "output"
            output.mkdir()

            paths = save_training_checkpoint(
                accelerator=accelerator,
                model=object(),
                tokenizer=self.tokenizer(accelerator),
                output_dir=str(output),
                completed_steps=5,
                model_source_path=str(source),
                save_model_only=True,
            )

            checkpoint = Path(paths["model"])
            self.assertTrue((checkpoint / "modeling_custom.py").is_file())
            self.assertTrue((checkpoint / "configuration_custom.py").is_file())

    def test_model_only_skips_accelerator_state(self):
        accelerator = _FakeAccelerator()
        with tempfile.TemporaryDirectory() as directory:
            paths = save_training_checkpoint(
                accelerator=accelerator,
                model=object(),
                tokenizer=self.tokenizer(accelerator),
                output_dir=directory,
                completed_steps=5,
                save_model_only=True,
            )

            self.assertIsNone(paths["state"])
            self.assertTrue(Path(paths["model"]).is_dir())
            self.assertFalse((Path(directory) / "training-state-000005").exists())
            self.assertNotIn("save_state", [event[0] for event in accelerator.events])

    def test_keep_latest_prunes_only_completed_training_state_pairs(self):
        accelerator = _FakeAccelerator()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            old_state = output / "training-state-000001"
            old_model = output / "ckpt-000001"
            incomplete = output / "training-state-000002.incomplete"
            unrelated = output / "training-state-backup"
            old_state.mkdir()
            old_model.mkdir()
            incomplete.mkdir()
            unrelated.mkdir()

            paths = save_training_checkpoint(
                accelerator=accelerator,
                model=object(),
                tokenizer=self.tokenizer(accelerator),
                output_dir=directory,
                completed_steps=5,
                keep_latest_state_only=True,
            )

            self.assertFalse(old_state.exists())
            self.assertFalse(old_model.exists())
            self.assertTrue(incomplete.exists())
            self.assertTrue(unrelated.exists())
            self.assertTrue(Path(paths["state"]).is_dir())
            self.assertEqual(
                (output / "latest").resolve(), Path(paths["state"]).resolve()
            )

    def test_model_only_and_keep_latest_are_mutually_exclusive(self):
        accelerator = _FakeAccelerator()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "requires saving training state"):
                save_training_checkpoint(
                    accelerator=accelerator,
                    model=object(),
                    tokenizer=self.tokenizer(accelerator),
                    output_dir=directory,
                    completed_steps=5,
                    save_model_only=True,
                    keep_latest_state_only=True,
                )


if __name__ == "__main__":
    unittest.main()
