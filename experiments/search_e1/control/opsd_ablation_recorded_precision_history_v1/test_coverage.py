"""One reporting-boundary fixture; no scientific/model/runtime test."""
import unittest
from derive import coverage


class CoverageBoundary(unittest.TestCase):
    def test_missing_rank_and_omitted_invocation_are_not_complete_zero_overflow(self):
        complete = {(0, 'attempt-one'): dict(precision={i: {} for i in range(8)}, source_load=True)}
        self.assertEqual(coverage(complete, 1, 0), [])
        incomplete = {(0, 'attempt-one'): dict(precision={i: {} for i in range(7)}, source_load=True)}
        gaps = coverage(incomplete, 2, 0)
        self.assertEqual(gaps[0]['missing_precision_ranks'], [7])
        self.assertEqual(gaps[1]['training_invocation_terminals'], 2)
        self.assertTrue(coverage(complete, 1, 1))


if __name__ == '__main__':
    unittest.main()
