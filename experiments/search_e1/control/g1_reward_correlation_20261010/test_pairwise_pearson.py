"""Synthetic CPU fixtures only; no original questions, trajectories or tensors."""
import copy
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('pairwise_pearson', HERE / 'pairwise_pearson.py')
code = importlib.util.module_from_spec(spec)
spec.loader.exec_module(code)
RUN = 'a' * 32


def moments(C, R):
    result = code.ExactBinaryMoments()
    for c, r in zip(C, R):
        result.add(int(c), float(r))
    return result


def complete_fixture(steps=1):
    rows, updates = [], []
    for step in range(1, steps + 1):
        attempt = format(step, '032x')
        identities = []
        for index in range(2560):
            parent, sibling = divmod(index, 5)
            group = hashlib.sha256(('synthetic-parent-' + str(parent)).encode()).hexdigest()
            identity = RUN + '-' + str(step) + '-attempt-' + attempt + '-' + str(parent) + '-' + group + '-sample-' + str(sibling)
            identities.append(identity)
            C = int((index + step) % 3 == 0)
            reward = 1.0 + (index % 11) / 20 if C else float(np.float32(.05))
            rows.append(dict(run_id=RUN, step=step, attempt_id=attempt, trajectory_id=identity,
                             C_raw=C, R_reconstructed=reward, original_online_reward=False))
        updates.append(dict(step=step, attempt_id=attempt,
                            trajectory_ids_sha256=code.trajectory_ids_sha256(identities)))
    coverage = dict(run_id=RUN, start_step=1, end_step=steps, trajectories_per_step=2560,
                    trajectory_ids_hash_mode=code.HASH_MODE,
                    reward_semantics='reconstructed_frozen_historical_input', updates=updates)
    return coverage, rows


class ExactMathFixtures(unittest.TestCase):
    def test_numpy_fullvector_and_point_biserial_agree(self):
        rng = np.random.default_rng(42)
        C = rng.integers(0, 2, 20481)
        R = rng.normal(size=len(C)) + 1.3 * C
        result = moments(C, R).report()
        self.assertAlmostEqual(result['Pearson']['value'], float(np.corrcoef(C, R)[0, 1]), places=14)
        p = C.mean()
        point = (R[C == 1].mean() - R[C == 0].mean()) * np.sqrt(p * (1 - p)) / np.std(R, ddof=0)
        self.assertAlmostEqual(result['point_biserial']['value'], float(point), places=14)
        for key, expected in (('sumC', C.sum()), ('sumC2', (C * C).sum()),
                              ('sumR', sum(map(Fraction.from_float, R.tolist()))),
                              ('sumR2', sum(Fraction.from_float(r) ** 2 for r in R.tolist())),
                              ('sumCR', sum(int(c) * Fraction.from_float(r) for c, r in zip(C, R)))):
            self.assertEqual(result[key], float(expected) if key.startswith('sumR') or key == 'sumCR' else int(expected))

    def test_large_offset_does_not_lose_reward_variance(self):
        C = np.array([0, 0, 1, 1, 0, 1])
        R = (2.0 ** 40) + np.array([0., 1., 2., 3., 4., 5.])
        result = moments(C, R).report()
        self.assertTrue(result['Pearson']['defined'])
        self.assertAlmostEqual(result['Pearson']['value'], float(np.corrcoef(C, R)[0, 1]), places=14)
        self.assertAlmostEqual(result['R_variance_population'], float(np.var(R)), places=14)

    def test_subnormal_squared_values_do_not_erase_correlation(self):
        smallest = float.fromhex('0x0.0000000000001p-1022')
        result = moments([0, 1], [0, smallest]).report()
        self.assertEqual(result['Pearson']['value'], 1.0)
        self.assertEqual(result['point_biserial']['value'], 1.0)

    def test_negative_perfect_correlation(self):
        self.assertEqual(moments([0, 1, 0, 1], [1, -1, 1, -1]).report()['Pearson']['value'], -1.0)

    def test_constant_C_undefined(self):
        report = moments([1, 1, 1], [1, 2, 3]).report()
        self.assertIsNone(report['Pearson']['value'])
        self.assertEqual(report['Pearson']['reason'], 'zero_C_variance')

    def test_constant_R_undefined(self):
        report = moments([0, 1, 0], [1, 1, 1]).report()
        self.assertIsNone(report['Pearson']['value'])
        self.assertEqual(report['Pearson']['reason'], 'zero_R_variance')

    def test_both_constant_undefined(self):
        self.assertEqual(moments([0, 0], [.05, .05]).report()['Pearson']['reason'], 'zero_both_variances')

    def test_single_row_undefined(self):
        self.assertFalse(moments([1], [1.0]).report()['Pearson']['defined'])

    def test_empty_moments_HOLD(self):
        with self.assertRaisesRegex(code.Hold, 'empty_pair_moments'):
            code.ExactBinaryMoments().report()

    def test_nonfinite_rewards_HOLD(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            with self.subTest(value=value), self.assertRaisesRegex(code.Hold, 'nonfinite_reward'):
                code.ExactBinaryMoments().add(0, value)

    def test_nonbinary_or_boolean_C_HOLD(self):
        for value in (2, -1, 0.5, True, '1'):
            with self.subTest(value=value), self.assertRaisesRegex(code.Hold, 'strict_binary_integer'):
                code.ExactBinaryMoments().add(value, 1.0)

    def test_boolean_or_text_reward_HOLD(self):
        for value in (True, '1.0', None):
            with self.subTest(value=value), self.assertRaisesRegex(code.Hold, 'non_numeric_or_boolean_reward'):
                code.ExactBinaryMoments().add(0, value)

    def test_Simpson_pooled_rows_differ_from_within_and_mean_correlations(self):
        # Deliberately generic numerical data, not a proposed G1 reward model.
        C1, R1 = [0, 0, 0, 1], [11, 11, 11, 10]
        C2, R2 = [0, 1, 1, 1], [21, 20, 20, 20]
        report = code.summarize_steps([moments(C1, R1), moments(C2, R2)])
        self.assertEqual(report['within_step_centered_Pearson']['value'], -1.0)
        self.assertEqual(report['between_step_means_Pearson']['value'], 1.0)
        self.assertGreater(report['pooled']['Pearson']['value'], 0)
        self.assertAlmostEqual(report['pooled']['Pearson']['value'],
                               float(np.corrcoef(C1 + C2, R1 + R2)[0, 1]), places=14)
        decomp = report['covariance_decomposition']
        self.assertAlmostEqual(decomp['pooled_covariance_population'],
                               decomp['within_step_covariance_population'] + decomp['between_step_covariance_population'], places=14)

    def test_allcorrect_group_variation_affects_pooled_denominator(self):
        report = code.summarize_steps([moments([1, 1, 1], [1, 1.3, 1.6]),
                                      moments([0, 0, 0], [.05, .05, .05])])
        self.assertFalse(report['within_step_centered_Pearson']['defined'])
        self.assertTrue(report['pooled']['Pearson']['defined'])
        self.assertLess(report['pooled']['Pearson']['value'], 1.0)

    def test_merge_matches_single_fullvector(self):
        one, two = moments([0, 1], [.05, 1.2]), moments([1, 0], [1.4, -.1])
        one.merge(two)
        self.assertEqual(one.report(), moments([0, 1, 1, 0], [.05, 1.2, 1.4, -.1]).report())

    def test_unequal_step_size_decomposition_HOLD(self):
        with self.assertRaisesRegex(code.Hold, 'unequal_step_sizes'):
            code.summarize_steps([moments([0, 1], [0, 1]), moments([0], [0])])


class CoverageFixtures(unittest.TestCase):
    def test_complete_2560_and_all_row_pooled_numpy_match(self):
        coverage, rows = complete_fixture(2)
        ledger = code.CompletePairLedger(coverage, 'R_reconstructed')
        # Interleaving does not change identity pairing or sufficient statistics.
        for row in rows[::2] + rows[1::2]:
            ledger.add(row)
        result = ledger.finish()
        self.assertEqual(result['total_pairs'], 5120)
        C = np.array([row['C_raw'] for row in rows])
        R = np.array([row['R_reconstructed'] for row in rows])
        self.assertAlmostEqual(result['pooled']['Pearson']['value'], float(np.corrcoef(C, R)[0, 1]), places=14)
        self.assertFalse(result['historical_original_online_R_claimed'])
        self.assertEqual([row['N'] for row in result['steps']], [2560, 2560])

    def test_missing_pair_HOLD(self):
        coverage, rows = complete_fixture()
        ledger = code.CompletePairLedger(coverage, 'R_reconstructed')
        for row in rows[:-1]:
            ledger.add(row)
        with self.assertRaisesRegex(code.Hold, 'missing_individual_pairs'):
            ledger.finish()

    def test_duplicate_pair_HOLD(self):
        coverage, rows = complete_fixture()
        ledger = code.CompletePairLedger(coverage, 'R_reconstructed')
        ledger.add(rows[0])
        with self.assertRaisesRegex(code.Hold, 'duplicate_pair_identity'):
            ledger.add(rows[0])

    def test_wrong_run_attempt_or_step_HOLD(self):
        coverage, rows = complete_fixture()
        for field, value in [('run_id', 'f' * 32), ('attempt_id', 'f' * 32), ('step', 0), ('step', 2), ('step', True)]:
            ledger = code.CompletePairLedger(coverage, 'R_reconstructed')
            row = dict(rows[0]); row[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(code.Hold):
                ledger.add(row)

    def test_wrong_trajectory_embedded_binding_HOLD(self):
        coverage, rows = complete_fixture()
        row = dict(rows[0]); row['trajectory_id'] = row['trajectory_id'].replace('-1-attempt-', '-2-attempt-')
        with self.assertRaisesRegex(code.Hold, 'trajectory_run_step_attempt_or_sibling_mismatch'):
            code.CompletePairLedger(coverage, 'R_reconstructed').add(row)

    def test_parent_group_changes_between_siblings_HOLD(self):
        coverage, rows = complete_fixture()
        ledger = code.CompletePairLedger(coverage, 'R_reconstructed'); ledger.add(rows[0])
        row = dict(rows[1]); pieces = row['trajectory_id'].split('-'); pieces[-3] = 'f' * 64
        row['trajectory_id'] = '-'.join(pieces)
        with self.assertRaisesRegex(code.Hold, 'parent_hash_changed'):
            ledger.add(row)

    def test_unknown_complete_ID_set_HOLD(self):
        coverage, rows = complete_fixture()
        coverage['updates'][0]['trajectory_ids_sha256'] = 'f' * 64
        ledger = code.CompletePairLedger(coverage, 'R_reconstructed')
        for row in rows:
            ledger.add(row)
        with self.assertRaisesRegex(code.Hold, 'complete_original_trajectory_ID_set_sha_mismatch'):
            ledger.finish()

    def test_scope_missing_updates_and_attempt_reuse_HOLD(self):
        coverage, _ = complete_fixture(2)
        for mutation in ('missing', 'reuse', 'step0', 'step151', 'wrong_size', 'missing_hash'):
            changed = copy.deepcopy(coverage)
            if mutation == 'missing': changed['updates'].pop()
            if mutation == 'reuse': changed['updates'][1]['attempt_id'] = changed['updates'][0]['attempt_id']
            if mutation == 'step0': changed['start_step'] = 0
            if mutation == 'step151': changed['end_step'] = 151
            if mutation == 'wrong_size': changed['trajectories_per_step'] = 2559
            if mutation == 'missing_hash': changed['updates'][0].pop('trajectory_ids_sha256')
            with self.subTest(mutation=mutation), self.assertRaises(code.Hold):
                code.CompletePairLedger(changed, 'R_reconstructed')

    def test_mixed_online_provenance_HOLD(self):
        coverage, rows = complete_fixture()
        row = dict(rows[0]); row['original_online_reward'] = True
        with self.assertRaisesRegex(code.Hold, 'reward_online_provenance_flag_mismatch'):
            code.CompletePairLedger(coverage, 'R_reconstructed').add(row)

    def test_boolean_coverage_step_or_float_coverage_size_HOLD(self):
        coverage, _ = complete_fixture()
        changed = copy.deepcopy(coverage); changed['updates'][0]['step'] = True
        with self.assertRaisesRegex(code.Hold, 'noninteger_coverage_step'):
            code.CompletePairLedger(changed, 'R_reconstructed')
        changed = copy.deepcopy(coverage); changed['trajectories_per_step'] = 2560.0
        with self.assertRaisesRegex(code.Hold, 'unsupported_complete_coverage_contract'):
            code.CompletePairLedger(changed, 'R_reconstructed')

    def test_mixed_semantics_HOLD(self):
        coverage, rows = complete_fixture()
        row = dict(rows[0]); row['reward_semantics'] = 'original_online_record'
        with self.assertRaisesRegex(code.Hold, 'mixed_reward_semantics'):
            code.CompletePairLedger(coverage, 'R_reconstructed').add(row)

    def test_missing_C_or_R_HOLD(self):
        coverage, rows = complete_fixture()
        for field in ('C_raw', 'R_reconstructed', 'trajectory_id'):
            row = dict(rows[0]); row.pop(field)
            with self.subTest(field=field), self.assertRaisesRegex(code.Hold, 'missing_pair_identity_or_value'):
                code.CompletePairLedger(coverage, 'R_reconstructed').add(row)

    def test_aggregate_does_not_export_IDs_or_per_row_values(self):
        coverage, rows = complete_fixture()
        ledger = code.CompletePairLedger(coverage, 'R_reconstructed')
        for row in rows:
            ledger.add(row)
        document = json.dumps(ledger.finish())
        self.assertNotIn(rows[0]['trajectory_id'], document)
        self.assertNotIn('trajectory_id"', document)


class StreamFixtures(unittest.TestCase):
    def write(self, directory, rows, name='synthetic.jsonl'):
        path = Path(directory).resolve() / name
        raw = b''.join(code.canonical(row) + b'\n' for row in rows)
        path.write_bytes(raw)
        return path, hashlib.sha256(raw).hexdigest()

    def test_streamed_one_or_split_files_same_complete_result(self):
        coverage, rows = complete_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            full, fullsha = self.write(tmp, rows)
            first, firstsha = self.write(tmp, rows[:1280], 'first.jsonl')
            second, secondsha = self.write(tmp, rows[1280:], 'second.jsonl')
            a = code.analyze_jsonl([full], [fullsha], coverage, 'R_reconstructed')
            b = code.analyze_jsonl([first, second], [firstsha, secondsha], coverage, 'R_reconstructed')
            self.assertEqual(a['pooled'], b['pooled'])
            self.assertEqual(a['total_pairs'], 2560)

    def test_input_SHA_mismatch_HOLD(self):
        coverage, rows = complete_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = self.write(tmp, rows)
            with self.assertRaisesRegex(code.Hold, 'frozen_pair_file_sha_mismatch'):
                code.analyze_jsonl([path], ['f' * 64], coverage, 'R_reconstructed')

    def test_truncated_JSONL_HOLD(self):
        coverage, rows = complete_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = self.write(tmp, rows)
            raw = path.read_bytes()[:-1]; path.write_bytes(raw)
            with self.assertRaisesRegex(code.Hold, 'incomplete_or_blank_pair_JSONL_record'):
                code.analyze_jsonl([path], [hashlib.sha256(raw).hexdigest()], coverage, 'R_reconstructed')

    def test_duplicate_JSON_keys_HOLD(self):
        with self.assertRaisesRegex(code.Hold, 'duplicate_JSON_object_key'):
            code.parse_json(b'{"C_raw":0,"C_raw":1}')

    def test_oversized_JSONL_record_HOLD(self):
        coverage, _ = complete_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp).resolve() / 'synthetic.jsonl'
            raw = b'X' * (code.MAX_PAIR_JSONL_BYTES + 1) + b'\n'; path.write_bytes(raw)
            with self.assertRaisesRegex(code.Hold, 'oversized_pair_JSONL_record'):
                code.analyze_jsonl([path], [hashlib.sha256(raw).hexdigest()], coverage, 'R_reconstructed')

    def test_nonobject_or_invalid_JSON_HOLD(self):
        for raw in (b'[]', b'{broken'):
            with self.subTest(raw=raw), self.assertRaises(code.Hold):
                code.parse_json(raw)

    def test_symlink_input_HOLD(self):
        coverage, rows = complete_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            path, sha = self.write(tmp, rows)
            link = Path(tmp).resolve() / 'link.jsonl'; link.symlink_to(path)
            with self.assertRaisesRegex(code.Hold, 'invalid_frozen_pair_file_path'):
                code.analyze_jsonl([link], [sha], coverage, 'R_reconstructed')

    def test_no_DL_or_network_dependency(self):
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('requests', sys.modules)


if __name__ == '__main__':
    unittest.main(verbosity=2)
