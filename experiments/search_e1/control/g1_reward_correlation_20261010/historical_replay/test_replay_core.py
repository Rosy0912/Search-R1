import hashlib
import os
import importlib.util
from pathlib import Path
import tempfile
import types
import unittest

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location('replay_core', HERE / 'replay_core.py')
core = importlib.util.module_from_spec(spec); spec.loader.exec_module(core)
REPO = Path(os.environ.get('G1_REPLAY_FIXTURE_REPO', 'work/code-publication-20261008T023352Z/search-g1'))
CONST = dict(PENALTY=.1, MAX_SEARCH=2, wrong_floor=.05, bal_ground=.6,
    bal_nosearch_bonus=.5, w_cost=.02, wrong_ground_scale=0., probe_base=0.,
    probe_gate=True, probe_alpha_cost=False)

class NativeMechanisms(unittest.TestCase):
    def setUp(self):
        self.balance_file = REPO / 'verl/utils/seqlen_balancing.py'
        self.balance = core.load_balance(self.balance_file, core.sha_file(self.balance_file))
        self.reward_file = REPO / 'verl/trainer/main_ppo_eco.py'
        self.score = core.load_reward(self.reward_file, core.sha_file(self.reward_file), CONST)
    def test_equal_ties_exact_KK(self):
        self.assertEqual(core.balance_order([10] * 8, self.balance), [0, 7, 1, 6, 2, 5, 3, 4])
    def test_crossed_ties_exact_KK(self):
        self.assertEqual(core.balance_order([2, 2, 4, 4, 2, 2, 4, 4], self.balance), [0, 7, 1, 6, 3, 4, 2, 5])
    def test_complete_2560_and_permutation(self):
        order = core.balance_order([(i % 257) + 1 for i in range(2560)], self.balance)
        self.assertEqual(sorted(order), list(range(2560)))
        self.assertNotEqual(order, list(range(2560)))
    def test_missing_balance_source(self):
        with self.assertRaises(core.Hold): core.load_balance(self.balance_file, '0' * 64)
    def test_original_balance_method(self):
        p = REPO / 'verl/trainer/ppo/ray_trainer.py'
        self.assertEqual(len(core.verify_balance_method(p, core.sha_file(p))), 64)
    def test_no_world8_replacement(self):
        with self.assertRaises(core.Hold): core.balance_order([10] * 8, self.balance, 8)
    def test_invalid_not_rawzero(self):
        x = core.scalar_reward(self.score, False, 0, 0, .5, .5)
        self.assertEqual(x['R_reconstructed'], -.10000000149011612)
        self.assertEqual(x['R_python_before_FP32'], -.1)
    def test_wrong_searched_floor(self):
        x = core.scalar_reward(self.score, True, 0, 2, 0., 1.)
        self.assertEqual(x['R_reconstructed'], .05000000074505806)
    def test_wrong_direct_floor(self):
        self.assertEqual(core.scalar_reward(self.score, True, 0, 0, 1., 1.)['R_python_before_FP32'], .05)
    def test_correct_direct_b(self):
        self.assertEqual(core.scalar_reward(self.score, True, 1, 0, .4, .8)['R_python_before_FP32'], 1.2)
    def test_correct_search_dependent_ground(self):
        self.assertEqual(core.scalar_reward(self.score, True, 1, 1, .5, .5)['R_python_before_FP32'], 1.15)
    def test_authored_cost_second_and_saturation(self):
        x = core.scalar_reward(self.score, True, 1, 2, 0., 1.)
        y = core.scalar_reward(self.score, True, 1, 8, 0., 1.)
        self.assertEqual(x, y); self.assertAlmostEqual(x['R_python_before_FP32'], 1.58)
    def test_reward_parameters_drift_rejected(self):
        constants = dict(CONST, wrong_ground_scale=.15)
        with self.assertRaises(core.Hold): core.load_reward(self.reward_file, core.sha_file(self.reward_file), constants)
    def test_probability_invalid_nan_rejected(self):
        with self.assertRaises(core.Hold): core.scalar_reward(self.score, True, 1, 1, float('nan'), .5)
    def test_outcome_invalid_correct_rejected(self):
        with self.assertRaises(core.Hold): core.scalar_reward(self.score, False, 1, 0, .5, .5)
    def test_perpair_stats_no_means_substitute(self):
        rows = [dict(trajectory_id=str(i), C_raw=c, R_reconstructed=r)
                for i, (c, r) in enumerate([(0, -.1), (0, .05), (1, 1.1), (1, 1.6)])]
        stats = core.compact_sufficient_statistics(rows)
        self.assertEqual(stats['N'], 4); self.assertEqual(stats['sum_C'], 2)
        self.assertGreater(stats['individual_pair_Pearson'], .95)
        self.assertFalse(stats['original_online_row_rewards_available'])
    def test_duplicate_pairs_rejected(self):
        r = dict(trajectory_id='same', C_raw=1, R_reconstructed=1.1)
        with self.assertRaises(core.Hold): core.compact_sufficient_statistics([r, r])
    def test_constant_pairs_undefined(self):
        rows = [dict(trajectory_id=str(i), C_raw=1, R_reconstructed=1.) for i in range(4)]
        self.assertIsNone(core.compact_sufficient_statistics(rows)['individual_pair_Pearson'])
    def test_request_hc_hb_original_code_structure(self):
        text = (REPO / 'probe_server.py').read_text()
        self.assertIn('micro=4, max_len=4096', text)
        self.assertIn('padding=True, truncation=True, max_length=max_len', text)
        self.assertLess(text.index('Hc = _last_hidden'), text.index('Hb = _last_hidden'))
        self.assertIn('[:, -1, :].float().cpu().numpy()', text)

if __name__ == '__main__': unittest.main()
