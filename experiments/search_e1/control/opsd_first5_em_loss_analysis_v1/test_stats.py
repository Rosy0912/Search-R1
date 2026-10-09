"""Small necessary CPU checks; no real input, model, GPU or network."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
from scipy.stats import rankdata

spec = importlib.util.spec_from_file_location('isolated_first5_stats', Path(__file__).with_name('stats.py'))
s = importlib.util.module_from_spec(spec); spec.loader.exec_module(s)


def row(q, group='C0', value=1.):
    r = dict(question_id=q, group=group, outer_index=0, data_source='nq', valid_tokens=10,
             student_search_calls=1, reference_search_calls=2, reference_response_characters=100,
             reference_token_count=50, normalized_update_progress=0.,
             selected_student_strict_em=0, sampling_success_fraction=.4)
    r['regions'] = {k: dict(means={name: value for name in ('teacher_kl', 'target_post_loss', 'outer_old_kl',
        'student_entropy', 'student_observed_logp', 'student_teacher_observed_logp_gap')}) for k in s.REGIONS}
    return r


class StatsTests(unittest.TestCase):
    def test_native_extractor_feature_SHA_identity_guard(self):
        audit = dict(schema='opsd-first5-em-loss-extraction-summary-v1', accepted=True,
                     record_SHA256_before_summary={'pair_feature': 'a' * 64, 'cohort_feature': 'b' * 64})
        s.bind_extractor_outputs({'sha256': 'a' * 64}, {'sha256': 'b' * 64}, audit)
        with self.assertRaisesRegex(ValueError, 'Pair-feature bytes'):
            s.bind_extractor_outputs({'sha256': 'c' * 64}, {'sha256': 'b' * 64}, audit)
        with self.assertRaisesRegex(ValueError, 'Cohort-feature bytes'):
            s.bind_extractor_outputs({'sha256': 'a' * 64}, {'sha256': 'c' * 64}, audit)

    def test_auc_ties_and_weighted_sampling(self):
        x = np.array([0., 1., 1., 3.]); y = np.array([0., 0., 1., 1.]); w = np.array([2., 1., 3., 1.])
        brute = sum(w[i] * w[j] * (float(x[i] > x[j]) + .5 * float(x[i] == x[j]))
                    for i in range(4) for j in range(4) if y[i] == 1 and y[j] == 0)
        brute /= np.dot(w, y) * np.dot(w, 1 - y)
        self.assertAlmostEqual(s.Ranking(x).auc(y, w), brute)

    def test_weighted_spearman_matches_repeated_rows(self):
        x = np.array([2., 2., 3., 7.]); y = np.array([0., 1., 0., 1.]); w = np.array([1, 3, 2, 4])
        actual = s.weighted_corr(s.Ranking(x).ranks(w), s.Ranking(y).ranks(w), w)
        repeated = np.corrcoef(rankdata(np.repeat(x, w)), rankdata(np.repeat(y, w)))[0, 1]
        self.assertAlmostEqual(actual, repeated)

    def test_train_only_imputation_and_scaling(self):
        train = [row('a', value=1.), row('b', value=3.)]
        test = [row('c', value=10000.)]
        a, b, proof = s.design(train, test, ('teacher_kl',))
        self.assertEqual(proof['feature_transforms']['teacher_kl']['mean'], 2.)
        self.assertEqual(proof['feature_transforms']['teacher_kl']['sd'], 1.)
        self.assertTrue(np.isfinite(a).all() and np.isfinite(b).all())

    def test_question_block_and_joint_bootstrap(self):
        identity = {str(i): (i // 10, 'nq') for i in range(50)}
        folds = s.fold_questions(identity, 5, 42)
        self.assertEqual(set(folds.values()), set(range(5)))
        self.assertEqual(folds, s.fold_questions(identity, 5, 42))
        q, index, weights = s.boot_weights(identity, 8, 42)
        self.assertTrue(np.all(weights.sum(1) == 50))
        # Multiple groups of q must use this identical column, not independent draws.
        self.assertTrue(np.array_equal(weights[:, index['7']], weights[:, index['7']]))

    def test_absent_answer_is_not_logp_zero(self):
        r = row('a'); r['regions']['final_answer']['means']['student_observed_logp'] = None
        self.assertTrue(np.isnan(s.feature(r, 'answer_student_logp')))

    def test_regularized_logistic_and_metric_direction(self):
        x = np.array([[1., v] for v in (-3., -2., -1., 1., 2., 3.)]); y = np.array([0., 0., 0., 1., 1., 1.])
        beta, proof = s.fit_logistic(x, y)
        self.assertTrue(proof['success'] and beta[1] > 0)
        result = s.prediction_metrics(y, s.expit(x @ beta), np.ones(6))
        self.assertEqual(result['auc'], 1.)
        self.assertLess(result['brier'], .25)


if __name__ == '__main__':
    unittest.main()
