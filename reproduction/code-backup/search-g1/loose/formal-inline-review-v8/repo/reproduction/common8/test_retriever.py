"""Small CPU vectors test parity rejection; never load the large corpus or GPU."""
import unittest
import numpy as np
from reproduction.common8.retriever import compare_search


class RetrievalParityTests(unittest.TestCase):
    def fixture(self, metric):
        q = np.zeros((64, 768), dtype=np.float32); q[:, 0] = 1
        v = np.zeros((64, 3, 768), dtype=np.float32)
        v[:, :, 0] = [1, .9, .8]
        score = v[:, :, 0].copy() if metric == 'inner_product' else ((q[:, None] - v)**2).sum(-1)
        ids = np.tile(np.arange(3), (64, 1))
        return q, score, ids, score.copy(), ids.copy(), v, v.copy(), metric

    def test_identical_actual_vector_scores_for_both_metrics(self):
        for metric in ('inner_product', 'l2'):
            result = compare_search(*self.fixture(metric))
            self.assertEqual(result['top3_set_recall'], 1)
            self.assertLess(result['gpu_vector_score_max_abs'], 1e-6)

    def test_wrong_gpu_scores_ids_or_nonfinite_vectors_rejected(self):
        for case in ('scores', 'ids', 'nan', 'duplicates'):
            with self.subTest(case=case):
                x = self.fixture('inner_product')
                if case == 'scores': x[3][0, 0] -= .3
                elif case == 'ids': x[4][:] = [10, 11, 12]
                elif case == 'nan': x[6][0, 0, 0] = float('nan')
                else: x[4][0] = [0, 0, 0]
                with self.assertRaises(AssertionError): compare_search(*x)

    def test_wrong_cpu_reference_scores_rejected(self):
        x = self.fixture('inner_product'); x[1][0, 0] += .1
        with self.assertRaises(AssertionError): compare_search(*x)

    def test_same_ids_do_not_hide_a_bad_selected_vector(self):
        x = self.fixture('inner_product'); x[6][0, 2, 0] = .1; x[3][0, 2] = .1
        with self.assertRaises(AssertionError): compare_search(*x)


if __name__ == '__main__': unittest.main()
