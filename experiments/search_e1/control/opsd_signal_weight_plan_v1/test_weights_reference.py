import math
import random
import unittest

from weights_reference import weights, selected_batch_summary


class WeightMath(unittest.TestCase):
    def test_linear_example(self):
        actual = weights([0, 1, 2, 3, 4], 1)['weights']
        for a, b in zip(actual, [.1, .55, 1., 1.45, 1.9]):
            self.assertAlmostEqual(a, b)

    def test_opposite_directions_are_complements(self):
        x = [-2, -1, 0, 4, 8]
        for a, b in zip(weights(x, 1)['weights'], weights(x, -1)['weights']):
            self.assertAlmostEqual(a + b, 2.)

    def test_missing_is_neutral_not_zero(self):
        r = weights([None, -4., -2., 0., None], 1)
        self.assertEqual(r['available_siblings'], [1, 2, 3])
        self.assertEqual(r['weights'][0], 1.)
        self.assertEqual(r['weights'][4], 1.)
        self.assertNotEqual(r['weights'], weights([0., -4., -2., 0., 0.], 1)['weights'])
        self.assertAlmostEqual(sum(r['weights']), 5.)

    def test_all_equal(self):
        self.assertEqual(weights([2.] * 5, 1)['weights'], [1.] * 5)

    def test_zero_or_one_available(self):
        for x in ([None] * 5, [None, 10., None, None, None]):
            self.assertEqual(weights(x, 1)['weights'], [1.] * 5)

    def test_tiny_difference_not_amplified(self):
        x = [0., 1e-10, 2e-10, 3e-10, 4e-10]
        self.assertLess(max(abs(w - 1.) for w in weights(x, 1)['weights']), .001)

    def test_permutation_equivariance(self):
        x = [None, -3., 1., 2., 8.]
        permutation = [3, 0, 4, 1, 2]
        expected = weights(x, -1)['weights']
        actual = weights([x[j] for j in permutation], -1)['weights']
        for a, j in zip(actual, permutation):
            self.assertAlmostEqual(a, expected[j])

    def test_common_offset(self):
        x = [-3., 1., 2., 4., 8.]
        self.assertEqual(weights(x, 1)['weights'], weights([v + 1e8 for v in x], 1)['weights'])

    def test_unscorable_group_is_neutral(self):
        self.assertEqual(weights([1, 2, 3, 4, 5], -1,
                                 neutral_reason='sibling_joint_context_limit')['weights'], [1.] * 5)

    def test_nonfinite_always_fails(self):
        for bad in (math.nan, math.inf, -math.inf, True, '1'):
            with self.assertRaises(ValueError):
                weights([1, 2, 3, 4, bad], 1, neutral_reason='context')

    def test_input_schema(self):
        for direction in (0, True, 1., 'positive'):
            with self.assertRaises(ValueError):
                weights([1] * 5, direction)
        with self.assertRaises(ValueError):
            weights([1] * 4, 1)

    def test_many_signed_groups(self):
        rng = random.Random(42)
        for _ in range(1000):
            x = [rng.uniform(-50, 50) if rng.random() > .15 else None for _ in range(5)]
            for direction in (-1, 1):
                r = weights(x, direction)['weights']
                self.assertAlmostEqual(math.fsum(r), 5.)
                self.assertTrue(all(.1 - 1e-12 <= w <= 1.9 + 1e-12 for w in r))

    def test_selected_mean_need_not_be_one(self):
        chosen = [weights([0, 1, 2, 3, 4], 1)['weights'][-1]] * 8
        r = selected_batch_summary(chosen)
        self.assertAlmostEqual(r['selected_weight_mean'], 1.9)
        self.assertAlmostEqual(r['effective_sample_size'], 8.)

    def test_selected_summary_invalid(self):
        for x in ([], [0], [-1], [math.nan]):
            with self.assertRaises(ValueError):
                selected_batch_summary(x)


if __name__ == '__main__':
    unittest.main()
