"""Code benchmarks share all decoders and count only real benchmark samples."""
import tempfile
from unittest import TestCase, mock

import torch
from torch import nn
from types import SimpleNamespace

import eval as evaluator
from utils.configuration.config import EvalConfig


class TokenizerStub:
    def apply_chat_template(self, messages, **kwargs):
        return messages[0][0]["content"]

    def __call__(self, text, **kwargs):
        return {"input_ids": torch.tensor([[1, 2]])}

    def batch_decode(self, ids, **kwargs):
        return ["def answer(): return 1"] * len(ids)


class ToyDiffusionModel(nn.Module):
    def forward(self, input_ids, **kwargs):
        logits = torch.zeros((*input_ids.shape, 6))
        logits[..., 3] = 2
        return SimpleNamespace(logits=logits)


class CodeEvaluationTest(TestCase):
    def config(self, directory, **overrides):
        values = dict(
            task="humaneval", ckpt_path="/model", output=directory,
            humaneval_data_file="/HumanEval.jsonl", gen_length=4,
            generation_method="fixed_mtp", mtp_block_size=2, mask_id=5,
        )
        values.update(overrides)
        return EvalConfig(**values)

    def stats(self):
        return {
            "length_counts": torch.tensor([0, 2]),
            "forced_singletons": torch.tensor(0),
            "confidence_sum": torch.tensor(0., dtype=torch.float64),
            "confidence_count": torch.tensor(0),
            "forward_rounds": torch.tensor(2),
        }

    def test_code_tasks_accept_fixed_and_adaptive_with_both_confidences(self):
        for family in ("llada", "sdar"):
            for task in ("humaneval", "mbpp", "code"):
                for method in ("fixed_mtp", "ar_mtp"):
                    for metric in ("normalized_entropy", "max_probability"):
                        with self.subTest(family=family, task=task, method=method, metric=metric):
                            self.config(
                                "/run", task=task, model_family=family,
                                generation_method=method, mtp_confidence=metric,
                                mbpp_data_file="/mbpp.jsonl", mbpp_test_file="/mbpp_test.jsonl",
                                code_data_file="/code.jsonl", selective_logits=True,
                            ).validate()

    def test_padding_keeps_forward_calls_and_is_excluded_from_statistics(self):
        problems = {key: {"prompt": key} for key in ("a", "b", "c")}
        with tempfile.TemporaryDirectory() as directory:
            args = self.config(directory)
            with mock.patch.object(evaluator, "_load_code_tasks", return_value=problems), \
                    mock.patch.object(evaluator, "_dist_world_size", return_value=2), \
                    mock.patch.object(evaluator, "_dist_rank", return_value=1), \
                    mock.patch.object(evaluator, "_synchronize_rank_error"), \
                    mock.patch.object(evaluator, "_dist_barrier"), \
                    mock.patch.object(evaluator, "_dist_all_gather_object", side_effect=lambda value: [[], value]), \
                    mock.patch.object(evaluator, "generate_eval", return_value=(torch.tensor([[1, 2, 3, 3, 3, 3]]), self.stats())) as generate, \
                    mock.patch.object(evaluator, "finalize_mtp_stats", return_value={}) as finalize:
                result = evaluator.eval_code(object(), TokenizerStub(), torch.device("cpu"), args)
            self.assertIsNone(result)
            self.assertEqual(generate.call_count, 2)
            _, stats, sample_count = finalize.call_args.args
            self.assertEqual(stats["length_counts"].tolist(), [0, 2])
            self.assertEqual(stats["forward_rounds"].item(), 2)
            self.assertEqual(sample_count, 1)

    def test_code_log_contains_score_and_decoder_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.config(directory)
            with mock.patch.object(evaluator, "_load_code_tasks", return_value={"a": {"prompt": "complete"}}), \
                    mock.patch.object(evaluator, "generate_eval", return_value=(torch.tensor([[1, 2, 3, 3, 3, 3]]), self.stats())), \
                    mock.patch("utils.evaluation.code_exec.evaluate_functional_correctness", return_value={"pass@1": 0.5}), \
                    mock.patch.object(evaluator, "save_log") as save:
                result = evaluator.eval_code(object(), TokenizerStub(), torch.device("cpu"), args)
            self.assertEqual(result["pass@1"], 0.5)
            self.assertEqual(result["mtp/mean_block_length"], 2)
            self.assertEqual(result["mtp/forward_rounds_per_sample"], 2)
            self.assertEqual(save.call_args.args[1]["total"], 1)

    def test_greedy_adaptive_decode_uses_unscaled_confidence(self):
        model = ToyDiffusionModel()
        for metric in ("normalized_entropy", "max_probability"):
            args = self.config(
                "/run", generation_method="ar_mtp", temperature=0,
                mtp_confidence=metric, mtp_threshold=0.99,
            )
            with self.subTest(metric=metric):
                tokens, stats = evaluator.generate_eval(model, torch.tensor([[1, 2]]), args)
                self.assertEqual(tokens.tolist(), [[1, 2, 3, 3, 3, 3]])
                self.assertEqual(stats["forced_singletons"].item(), 4)

    def test_fixed_sdar_decode_requests_synchronized_forward_rounds(self):
        args = self.config("/run", model_family="sdar")
        rollout = {"generated_ids": torch.tensor([[1, 2]]), "rollout_stats": None}
        with mock.patch.object(evaluator, "generate_monotonic_blocks", return_value=rollout) as generate:
            evaluator.generate_eval(object(), torch.tensor([[1, 2]]), args)
        self.assertTrue(generate.call_args.kwargs["synchronize_forward_rounds"])
        self.assertEqual(generate.call_args.kwargs["temperature"], 0)

    def test_invalid_sampling_temperature_is_rejected(self):
        for value in (-1, float("nan"), float("inf"), True):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "temperature"):
                self.config("/run", temperature=value).validate()
