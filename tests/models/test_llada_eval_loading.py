import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from transformers import modeling_utils

from utils.evaluation.online_eval import _llada_allocator_compat, _load_checkpoint_for_eval


class LLaDAEvalLoadingTest(unittest.TestCase):
    def test_real_transformers_failure_and_guard(self):
        # Exercise the actual installed CUDA/distributed warmup branch without
        # allocating GPU memory: the legacy bug occurs before any allocation.
        legacy_model = SimpleNamespace(_tp_plan=None)
        device_map = {"weight": "cuda:0"}
        original = modeling_utils.caching_allocator_warmup
        with patch("torch.distributed.is_initialized", return_value=True):
            with self.assertRaises(TypeError):
                original(legacy_model, device_map, None)
            with _llada_allocator_compat("llada"):
                self.assertTrue(torch.distributed.is_initialized())
                modeling_utils.caching_allocator_warmup(legacy_model, device_map, None)
            self.assertIs(modeling_utils.caching_allocator_warmup, original)

    def test_valid_tp_plan_is_forwarded(self):
        model = SimpleNamespace(_tp_plan={"layer.*": "colwise"})
        with patch.object(modeling_utils, "caching_allocator_warmup", return_value="called") as original:
            with _llada_allocator_compat("llada"):
                self.assertEqual(modeling_utils.caching_allocator_warmup(model, {}, None), "called")
            original.assert_called_once_with(model, {}, None)

    def test_sdar_hook_is_unchanged(self):
        original = modeling_utils.caching_allocator_warmup
        with _llada_allocator_compat("sdar"):
            self.assertIs(modeling_utils.caching_allocator_warmup, original)

    def test_hook_restored_after_exception(self):
        original = modeling_utils.caching_allocator_warmup
        with self.assertRaisesRegex(RuntimeError, "load failed"):
            with _llada_allocator_compat("llada"):
                raise RuntimeError("load failed")
        self.assertIs(modeling_utils.caching_allocator_warmup, original)

    def test_checkpoint_loader_uses_guard_and_preserves_device_map(self):
        original = modeling_utils.caching_allocator_warmup
        model = Mock()
        model.eval.return_value = model
        tokenizer = Mock()

        def load(*args, **kwargs):
            self.assertEqual(os.environ["ACCELERATE_USE_FSDP"], "false")
            self.assertEqual(kwargs["device_map"], {"": "cuda:3"})
            self.assertEqual(kwargs["torch_dtype"], torch.bfloat16)
            self.assertTrue(kwargs["local_files_only"])
            self.assertTrue(torch.distributed.is_initialized())
            modeling_utils.caching_allocator_warmup(SimpleNamespace(_tp_plan=None), {"weight": "cuda:3"}, None)
            return model

        with patch.dict(os.environ, {"ACCELERATE_USE_FSDP": "true"}), patch("torch.distributed.is_initialized", return_value=True), patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer), patch("transformers.AutoModel.from_pretrained", side_effect=load):
            loaded, tok = _load_checkpoint_for_eval("/checkpoint", "llada", torch.device("cuda:3"))
            self.assertEqual(os.environ["ACCELERATE_USE_FSDP"], "true")
        self.assertIs(loaded, model)
        self.assertIs(tok, tokenizer)
        self.assertIs(modeling_utils.caching_allocator_warmup, original)
        model.requires_grad_.assert_called_once_with(False)

    def test_failed_checkpoint_load_restores_fsdp_and_warmup(self):
        original = modeling_utils.caching_allocator_warmup
        with patch.dict(os.environ, {"ACCELERATE_USE_FSDP": "true"}), patch("transformers.AutoTokenizer.from_pretrained"), patch("transformers.AutoModel.from_pretrained", side_effect=RuntimeError("load failed")):
            with self.assertRaisesRegex(RuntimeError, "load failed"):
                _load_checkpoint_for_eval("/checkpoint", "llada", torch.device("cuda:0"))
            self.assertEqual(os.environ["ACCELERATE_USE_FSDP"], "true")
        self.assertIs(modeling_utils.caching_allocator_warmup, original)


if __name__ == "__main__":
    unittest.main()
