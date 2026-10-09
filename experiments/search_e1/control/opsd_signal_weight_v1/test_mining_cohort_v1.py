"""Small CPU selection/weight contracts; synthetic questions only."""
import copy
import hashlib
import os
from pathlib import Path
import random
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cohort_v1 as cohort
import miner_v1 as miner
import weights_v1 as weights


def frozen_core():
    runtime = Path(os.environ.get('OPSD_ONLINE_TEST_RUNTIME') or str(HERE.parent / 'opsd_debug_frozen_runtime_20261008/runtime'))
    # Existing retained local CPU dependency, not a production fallback/version.
    deps = HERE.parents[4] / 'search-e1-analysis-output/opsd-selection-audit-local-testdeps'
    sys.path.insert(0, str(deps))
    core_path = runtime / 'search_e1/core.py'; qa_path = runtime / 'upstream/Search-R1/search_r1/qa_protocol.py'
    core_raw = core_path.read_bytes(); qa_raw = qa_path.read_bytes()
    assert hashlib.sha256(core_raw).hexdigest() == '0fbfbe63d0aafb41f79db702e08c9a0efc5f9bbf29a2964310594ec2ec0a83fc'
    assert hashlib.sha256(qa_raw).hexdigest() == '9a825a7b222b5a3fd2c1de7f63ed38e157258fcf858e773ff64c963a4b6daddd'
    package = types.ModuleType('search_r1'); package.__path__ = [str(qa_path.parent)]
    qa = types.ModuleType('search_r1.qa_protocol'); qa.__file__ = str(qa_path)
    exec(compile(qa_raw, str(qa_path), 'exec'), qa.__dict__)
    sys.modules['search_r1'] = package; sys.modules['search_r1.qa_protocol'] = qa
    core = types.ModuleType('actual_frozen_core'); core.__file__ = str(core_path)
    exec(compile(core_raw, str(core_path), 'exec'), core.__dict__)
    return core


def siblings(core, qid='synthetic-q', rewards=(1, 0, 0, 0, 0)):
    rows = []
    for index, reward in enumerate(rewards):
        response = '<think>reason%s</think><answer>%s</answer>' % (index, 'gold' if reward else 'wrong-' + str(index))
        spans = [[0, len(response)]]
        row = dict(question_id=qid, question='synthetic private fixture question', answers=['gold'],
                   sibling=index, response=response, policy_spans=spans, search_calls=0,
                   token_count=len(response), reward=core.reward(response, ['gold'], spans))
        assert row['reward'] == reward
        rows.append(row)
    return rows


class MiningAndWeights(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.core = frozen_core()

    def test_success_B_original_first_tie_and_uniform_D_four_groups(self):
        rows = siblings(self.core, rewards=(1, 1, 0, 0, 0))
        rows[0]['search_calls'] = 1; rows[1]['token_count'] = 1000
        selected, draw = miner.uniform_sibling(rows[0]['question_id'], 0)
        for variant in miner.VARIANTS:
            pairs, stats = miner.mine_pairs(rows, variant, 0, self.core)
            self.assertEqual(pairs[0]['mining']['reference_sibling'], 1)
            self.assertEqual(pairs[0]['mining']['student_sibling'], selected)
            self.assertEqual(pairs[0]['mining']['random_draw'], draw)
            self.assertEqual(pairs[0]['pair_weight'], 1.0)
            self.assertFalse(pairs[0]['mining']['weights_applied'])
            self.assertFalse(stats['native_selection_shadow_checked'])

    def test_unbiased_rejection_word_and_rng_independence(self):
        words = [(2 ** 64 - 1).to_bytes(8, 'big') + bytes(24), (22).to_bytes(8, 'big') + bytes(24)]
        with patch.object(miner, '_digest', side_effect=words):
            index, proof = miner.uniform_sibling('synthetic', 2)
        self.assertEqual(index, 2); self.assertEqual(proof['rejected_attempts'], 1)
        before = random.getstate()
        a = miner.uniform_sibling('synthetic-unicode-题', 3)
        self.assertEqual(a, miner.uniform_sibling('synthetic-unicode-题', 3))
        self.assertEqual(before, random.getstate())
        self.assertEqual(a[1]['accepted_counter'], 0)

    def test_random_self_pair_and_text_equal_are_separate(self):
        rows = siblings(self.core, rewards=(1, 1, 1, 1, 1))
        rows[1]['response'] = rows[0]['response']; rows[1]['policy_spans'] = copy.deepcopy(rows[0]['policy_spans'])
        rows[1]['token_count'] = rows[0]['token_count']
        for selected in (0, 1):
            with patch.object(miner, 'uniform_sibling', return_value=(selected, {'fixture_selected_sibling': selected})):
                pair = miner.mine_pairs(rows, 'G_POS', 0, self.core)[0][0]
            self.assertTrue(pair['mining']['response_text_equal'])
            self.assertEqual(pair['mining']['self_pair_by_sibling'], selected == 0)
            self.assertEqual(pair['pair_weight'], 1.0)

    def test_sample_std_positive_weight_for_all_reward_patterns(self):
        for correct in range(6):
            proof = weights.correctness_weights([1] * correct + [0] * (5 - correct))
            self.assertEqual(proof['correction'], 1)
            self.assertAlmostEqual(proof['sigma'] ** 2, sum((r - proof['mu']) ** 2 for r in proof['rewards']) / 4)
            self.assertAlmostEqual(sum(proof['weights']), 5)
            self.assertTrue(all(w > 0 for w in proof['weights']))
            if correct in (0, 5): self.assertEqual(proof['weights'], [1.0] * 5)
            else: self.assertGreater(proof['weights'][0], proof['weights'][-1])
        proof = weights.correctness_weights([1, 0, 0, 0, 0])
        self.assertAlmostEqual(proof['weights'][0], 3.503, places=2)
        self.assertAlmostEqual(proof['weights'][1], 0.374, places=2)

    def test_four_groups_same_B_D_no_correctness_weight_applied(self):
        rows = siblings(self.core)
        results = [miner.mine_pairs(rows, v, 4, self.core)[0][0] for v in miner.VARIANTS]
        self.assertEqual(len({x['mining']['student_sibling'] for x in results}), 1)
        self.assertTrue(all(x['pair_weight'] == 1 for x in results))
        self.assertTrue(all(x['mining']['random_draw'] == results[0]['mining']['random_draw'] for x in results))

    def test_all_wrong_skipped_all_correct_kept_and_strict_laundering_refused(self):
        for variant in miner.VARIANTS:
            pairs, stats = miner.mine_pairs(siblings(self.core, rewards=(0, 0, 0, 0, 0)), variant, 0, self.core)
            self.assertEqual(pairs, []); self.assertEqual(stats['all_wrong_skipped'], 1)
            pairs, _ = miner.mine_pairs(siblings(self.core, rewards=(1, 1, 1, 1, 1)), variant, 0, self.core)
            self.assertEqual(len(pairs), 1); self.assertEqual(pairs[0]['pair_weight'], 1.0)
        rows = siblings(self.core)
        rows[0]['response'] = '<information><answer>gold</answer></information>'
        rows[0]['policy_spans'] = []
        with self.assertRaises(ValueError): miner.mine_pairs(rows, 'G_NEG', 0, self.core)

    def test_resume_pair_version_weight_and_slot_mutation_refused(self):
        rows = siblings(self.core); pairs, stats = miner.mine_pairs(rows, 'G_NEG', 1, self.core)
        self.assertEqual(miner.verify_pairs(rows, pairs, 'G_NEG', 1, self.core), stats)
        wrong = copy.deepcopy(pairs); wrong[0]['pair_weight'] += 0.1
        with self.assertRaises(ValueError): miner.verify_pairs(rows, wrong, 'G_NEG', 1, self.core)
        with self.assertRaises(ValueError): miner.verify_pairs(rows, pairs, 'G_NEG', 1, self.core, expected_version='legacy')
        rows[-1]['sibling'] = 0
        with self.assertRaises(ValueError): miner.mine_pairs(rows, 'G_POS', 0, self.core)

    def test_production_full_cohort_binding_rejects_incomplete_or_wrong_question(self):
        rows = [r for i in range(512) for r in siblings(self.core, qid='synthetic-%d' % i)]
        questions = [{k: r[k] for k in ('question_id', 'question', 'answers')} for r in rows[::5]]
        pairs, stats = miner.verify_rollouts(rows, questions, 'G_POS', 0, self.core)
        self.assertEqual(len(pairs), 512); self.assertEqual(stats['generated_trajectories'], 2560)
        with self.assertRaises(ValueError): miner.verify_rollouts(rows[:-1], questions, 'G_POS', 0, self.core)
        changed = copy.deepcopy(rows); changed[0]['question'] = 'changed cohort source'
        with self.assertRaises(ValueError): miner.verify_rollouts(changed, questions, 'G_NEG', 0, self.core)


class Cohorts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = [dict(question_id='train-%06d' % i, question='synthetic-train-text-%06d' % i, answers=['synthetic']) for i in range(169103)]
        cls.validation = [dict(question_id='heldout-%03d' % i, question='synthetic-heldout-text-%03d' % i, answers=['synthetic']) for i in range(512)]
        cls.train_ref = dict(path='/fixture/train.jsonl', sha256='a' * 64)
        cls.validation_ref = dict(path='/fixture/validation.jsonl', sha256='b' * 64)

    def freeze(self, train=None, validation=None):
        return cohort.freeze_cohorts(train or self.train, validation or self.validation, self.train_ref, self.validation_ref)

    def test_25_original_seed42_cohorts_no_global_rng_change(self):
        before = random.getstate(); frozen = self.freeze()
        self.assertEqual(before, random.getstate())
        expected = list(range(169103)); random.Random(42).shuffle(expected)
        self.assertEqual([i for c in frozen['cohorts'] for i in c['train_indices']], expected[:12800])
        self.assertEqual(frozen['selected_unique_question_count'], 12800)
        self.assertTrue(all(len(c['questions']) == 512 and c['before']['offset'] == c['outer_index'] * 512 for c in frozen['cohorts']))
        self.assertTrue(cohort.verify_frozen(frozen, self.train, self.validation, self.train_ref, self.validation_ref)['accepted'])

    def test_exact_text_and_id_overlap_fail_without_replacing_selected(self):
        selected = self.freeze()['cohorts'][0]['train_indices'][0]
        for field in ('question_id', 'question'):
            validation = list(self.validation); validation[0] = dict(validation[0], **{field: self.train[selected][field]})
            with self.assertRaisesRegex(ValueError, 'no replacement/resampling'): self.freeze(validation=validation)

    def test_saved_cohort_order_body_source_version_mutation_rejected(self):
        frozen = self.freeze()
        for mutate in (lambda x: x.update(version='legacy'),
                       lambda x: x['cohorts'][0]['questions'][0].update(question='changed'),
                       lambda x: x['train_identity'].update(sha256='c' * 64)):
            wrong = copy.deepcopy(frozen); mutate(wrong)
            with self.assertRaises(ValueError): cohort.verify_frozen(wrong, self.train, self.validation, self.train_ref, self.validation_ref)

    def test_bound_jsonl_sha_and_partial_line_are_not_trusted(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'source.jsonl'; raw = b'{"question_id":"synthetic"}\n'; path.write_bytes(raw)
            ref = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
            self.assertEqual(cohort._read_bound_jsonl(ref), [{'question_id': 'synthetic'}])
            path.write_bytes(b'{"question_id":"changed"}\n')
            with self.assertRaises(ValueError): cohort._read_bound_jsonl(ref)
            raw = b'{"question_id":"synthetic"}'; path.write_bytes(raw)
            ref = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest())
            with self.assertRaises(ValueError): cohort._read_bound_jsonl(ref)


if __name__ == '__main__': unittest.main()
