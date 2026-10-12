import unittest

from data.code import collate_fn_code
from data.math import collate_fn_gsm8k, collate_fn_math


class DataSampleIdTest(unittest.TestCase):
    def test_gsm8k_collator_preserves_source_ids(self):
        batch = [
            {"question": "q1", "answer": "a1", "__sample_id__": 17},
            {"question": "q2", "answer": "a2", "__sample_id__": 3},
        ]
        self.assertEqual(collate_fn_gsm8k(batch)["sample_ids"], [17, 3])

    def test_math_eval_collators_remain_compatible_without_source_ids(self):
        gsm8k = collate_fn_gsm8k([{"question": "q", "answer": "a"}])
        math = collate_fn_math([{"problem": "p", "solution": "s"}])
        self.assertNotIn("sample_ids", gsm8k)
        self.assertNotIn("sample_ids", math)

    def test_math_collator_preserves_source_ids(self):
        batch = [
            {"problem": "p1", "solution": "s1", "__sample_id__": 11},
            {"problem": "p2", "solution": "s2", "__sample_id__": 29},
        ]
        self.assertEqual(collate_fn_math(batch)["sample_ids"], [11, 29])

    def test_code_collator_preserves_source_ids(self):
        batch = [
            {"question": "c1", "test_cases": [], "__sample_id__": 5},
            {"question": "c2", "test_cases": [], "__sample_id__": 8},
        ]
        self.assertEqual(collate_fn_code(batch)["sample_ids"], [5, 8])


if __name__ == "__main__":
    unittest.main()
