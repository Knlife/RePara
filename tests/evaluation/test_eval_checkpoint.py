import json
import tempfile
import unittest
from pathlib import Path

from eval import resolve_adapter_base_path


class EvalCheckpointTest(unittest.TestCase):
    def test_adapter_base_path_comes_from_adapter_config(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter_dir = Path(directory)
            (adapter_dir / "adapter_config.json").write_text(
                json.dumps({"base_model_name_or_path": "/models/step-0"}),
                encoding="utf-8",
            )

            self.assertEqual(resolve_adapter_base_path(adapter_dir), "/models/step-0")

    def test_adapter_base_path_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter_dir = Path(directory)
            (adapter_dir / "adapter_config.json").write_text("{}\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "base_model_name_or_path"):
                resolve_adapter_base_path(adapter_dir)


if __name__ == "__main__":
    unittest.main()
