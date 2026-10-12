import unittest
from unittest.mock import patch

import torch

from utils.runtime.timing import TimingRecorder, reduce_max_timings


class TimingRecorderTest(unittest.TestCase):
    def test_cpu_phases_accumulate_repeated_spans(self):
        recorder = TimingRecorder(enabled=True, device="cpu")

        with patch(
            "utils.runtime.timing.time.perf_counter",
            side_effect=[1.0, 1.25, 2.0, 2.75],
        ):
            with recorder.record("phase"):
                pass
            with recorder.record("phase"):
                pass

        self.assertEqual(recorder.elapsed(), {"phase": 1.0})

    def test_nested_different_phases_are_supported(self):
        recorder = TimingRecorder(enabled=True, device="cpu")

        with patch(
            "utils.runtime.timing.time.perf_counter",
            side_effect=[1.0, 2.0, 3.0, 4.0],
        ):
            with recorder.record("outer"):
                with recorder.record("inner"):
                    pass

        self.assertEqual(recorder.elapsed(), {"inner": 1.0, "outer": 3.0})

    def test_switch_shares_one_boundary_between_phases(self):
        recorder = TimingRecorder(enabled=True, device="cpu")

        with patch(
            "utils.runtime.timing.time.perf_counter",
            side_effect=[1.0, 2.0, 4.0],
        ):
            recorder.start("prefill")
            recorder.switch("prefill", "backward")
            recorder.stop("backward")

        self.assertEqual(
            recorder.elapsed(), {"prefill": 1.0, "backward": 2.0}
        )

    def test_record_can_temporarily_replace_an_active_phase(self):
        recorder = TimingRecorder(enabled=True, device="cpu")

        with patch(
            "utils.runtime.timing.time.perf_counter",
            side_effect=[1.0, 2.0, 4.0, 5.0],
        ):
            with recorder.record("prefill"):
                with recorder.record("backward", replacing="prefill"):
                    pass

        self.assertEqual(
            recorder.elapsed(), {"prefill": 2.0, "backward": 2.0}
        )

    def test_disabled_recorder_is_a_noop(self):
        recorder = TimingRecorder(enabled=False, device="cuda")
        with recorder.record("phase"):
            pass
        self.assertEqual(recorder.elapsed(), {})

    def test_elapsed_rejects_active_phases(self):
        recorder = TimingRecorder(enabled=True, device="cpu")
        recorder.start("phase")
        with self.assertRaisesRegex(RuntimeError, "active phases: phase"):
            recorder.elapsed()

    def test_single_process_reduction_preserves_values(self):
        reduced = reduce_max_timings(
            {"rollout": 1.25, "learner": 2.5}, torch.device("cpu")
        )
        self.assertEqual(reduced, {"learner": 2.5, "rollout": 1.25})

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_events_measure_gpu_work(self):
        recorder = TimingRecorder(enabled=True, device="cuda")
        values = torch.ones((128, 128), device="cuda")

        with recorder.record("matmul"):
            product = torch.mm(values, values)
            with recorder.record("activation", replacing="matmul"):
                torch.relu_(product)

        elapsed = recorder.elapsed()
        self.assertGreater(elapsed["matmul"], 0.0)
        self.assertGreater(elapsed["activation"], 0.0)


if __name__ == "__main__":
    unittest.main()
