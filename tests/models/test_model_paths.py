import json
from pathlib import Path
import tempfile
import unittest

from utils.models.model_paths import validate_local_model_path


class LocalModelPathTest(unittest.TestCase):
    def fixture(self, root):
        (root / "config.json").write_text("{}")
        (root / "tokenizer_config.json").write_text("{}")
        (root / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {"weight": "model-00001-of-00001.safetensors"},
        }))
        (root / "model-00001-of-00001.safetensors").write_bytes(b"weights")

    def test_local_sharded_model(self):
        with tempfile.TemporaryDirectory() as directory:
            self.fixture(Path(directory))
            validate_local_model_path(directory)

    def test_hub_id_is_not_treated_as_a_local_directory(self):
        validate_local_model_path("example-org/example-model")

    def test_missing_absolute_directory_has_mount_diagnostic(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "Check container mounts"):
                validate_local_model_path(str(Path(directory) / "missing"))

    def test_dangling_parent_symlink_has_real_target_diagnostic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "models").symlink_to(root / "unmounted", target_is_directory=True)
            with self.assertRaisesRegex(FileNotFoundError, "unmounted/model"):
                validate_local_model_path(str(root / "models/model"))

    def test_dangling_weight_symlink_is_caught(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            weight = root / "model-00001-of-00001.safetensors"
            weight.unlink()
            weight.symlink_to(root / "unmounted/weights")
            with self.assertRaisesRegex(FileNotFoundError, "unmounted/weights"):
                validate_local_model_path(directory)

    def test_missing_custom_model_code_is_caught(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            (root / "config.json").write_text(json.dumps({
                "auto_map": {"AutoModel": "modeling_custom.Model"},
            }))
            with self.assertRaisesRegex(FileNotFoundError, "modeling_custom.py"):
                validate_local_model_path(directory)


if __name__ == "__main__":
    unittest.main()
