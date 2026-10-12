import unittest

import torch.nn as nn

from train import (
    resolve_activation_checkpointing_strategy,
    set_activation_checkpointing,
)
from utils.configuration.config import TrainConfig


class CheckpointBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.strategy = "unset"

    def set_activation_checkpointing(self, strategy):
        self.strategy = strategy


class CheckpointController(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation_checkpointing_strategy = "unset"
        self.setter_calls = []
        self.block = CheckpointBlock()

    def set_activation_checkpointing(self, strategy):
        self.activation_checkpointing_strategy = strategy
        self.setter_calls.append(strategy)


class ActivationCheckpointingTest(unittest.TestCase):


    def test_none_reaches_wrapped_controllers(self):
        wrapper = nn.Module()
        wrapper.module = CheckpointController()

        self.assertEqual(TrainConfig().activation_checkpointing_strategy, "none")
        self.assertTrue(set_activation_checkpointing(wrapper, "whole_layer"))
        self.assertEqual(wrapper.module.activation_checkpointing_strategy, "whole_layer")
        self.assertEqual(wrapper.module.setter_calls, ["whole_layer"])
        self.assertEqual(wrapper.module.block.strategy, "whole_layer")
        self.assertTrue(set_activation_checkpointing(wrapper, None))
        self.assertIsNone(wrapper.module.activation_checkpointing_strategy)
        self.assertEqual(wrapper.module.setter_calls, ["whole_layer", None])
        self.assertIsNone(wrapper.module.block.strategy)
        self.assertEqual(resolve_activation_checkpointing_strategy("whole_layer"), "whole_layer")
        self.assertIsNone(resolve_activation_checkpointing_strategy("none"))
        with self.assertRaisesRegex(ValueError, "Unknown activation checkpointing strategy"):
            resolve_activation_checkpointing_strategy("unsupported")


if __name__ == "__main__":
    unittest.main()
