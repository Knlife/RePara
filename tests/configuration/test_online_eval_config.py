import unittest

from utils.configuration.config import TrainConfig


class OnlineEvalConfigTest(unittest.TestCase):




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
