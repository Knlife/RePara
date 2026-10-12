import unittest

from utils.configuration.config import TrainConfig


class OnlineEvalConfigTest(unittest.TestCase):
    def test_sdar_code_online_eval_accepts_all_repara_decoders(self):
        for method in ("diffusion", "fixed_mtp", "ar_mtp"):
            with self.subTest(method=method):
                config = TrainConfig(
                    model_path="/models/sdar", model_family="sdar",
                    output_dir="/runs/train", dataset="code",
                    code_data_path="/datasets/code.jsonl", eval_every=16,
                    eval_config={
                        "tasks": ["mbpp"], "generation_method": method,
                        "mbpp_data_file": "/datasets/mbpp.jsonl",
                        "mbpp_test_file": "/datasets/mbpp_test.jsonl",
                    },
                )
                config.validate()

    def test_native_sdar_code_online_evaluation_requires_replicas(self):
        config = TrainConfig(
            model_path="/models/sdar", model_family="sdar",
            output_dir="/runs/train", dataset="code",
            code_data_path="/datasets/code.jsonl", eval_every=16,
            eval_config={
                "tasks": ["mbpp"], "generation_method": "sdar_native",
                "mbpp_data_file": "/datasets/mbpp.jsonl",
                "mbpp_test_file": "/datasets/mbpp_test.jsonl",
            },
        )
        with self.assertRaisesRegex(ValueError, "checkpoint replicas"):
            config.validate()

    def test_code_online_eval_config_is_valid(self):
        config = TrainConfig(
            model_path="/models/llada",
            output_dir="/runs/train",
            dataset="code",
            code_data_path="/datasets/code.jsonl",
            eval_every=16,
            eval_config={
                "tasks": ["mbpp", "humaneval"],
                "output_dir": "/runs/train/online-eval",
                "mbpp_data_file": "/datasets/mbpp.jsonl",
                "mbpp_test_file": "/datasets/mbpp_test.jsonl",
                "humaneval_data_file": "/datasets/HumanEval.jsonl.gz",
                "gen_length": 256,
                "mtp_size": 1,
                "block_length": 32,
            },
        )

        config.validate()

    def test_online_eval_requires_a_spec(self):
        config = TrainConfig(
            model_path="/models/llada",
            output_dir="/runs/train",
            dataset="code",
            code_data_path="/datasets/code.jsonl",
            eval_every=16,
        )

        with self.assertRaisesRegex(ValueError, "eval_config is required"):
            config.validate()


if __name__ == "__main__":
    unittest.main()
