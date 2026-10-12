"""UNEXECUTED exact source live-goal binding regression; no GPU acceptance.

Only the original binding/initialization bodies and constants are AST extracted.
No project module is imported and no training, Ray or remote action is invoked.
"""
import ast
import copy
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LATEST_HUMAN_GOAL_UPDATED_AT = 1791299045
SUPERSEDED_SOURCE10_GOAL_UPDATED_AT = 1791295951


class StripImports(ast.NodeTransformer):
    def visit_Import(self, node):
        return None

    def visit_ImportFrom(self, node):
        return None


def actual_source_namespace():
    namespace = dict(re=re)
    selections = (
        ('search_r1/baselines/fresh200.py', {'calibration_initialization'}),
        ('search_r1/baselines/fresh200_live.py',
         {'require', 'live_initialization', 'validate_live_initialization', 'validate_live_binding'}),
    )
    for relative, names in selections:
        path = ROOT / relative
        tree = ast.parse(path.read_text(), filename=str(path))
        body, found = [], set()
        for node in tree.body:
            if isinstance(node, ast.Assign):
                body.append(copy.deepcopy(node))
            elif isinstance(node, ast.FunctionDef) and node.name in names:
                body.append(StripImports().visit(copy.deepcopy(node)))
                found.add(node.name)
        if found != names:
            raise AssertionError('Missing original live binding body: ' + repr(names - found))
        selected = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
        exec(compile(selected, str(path), 'exec'), namespace)
    return namespace


def candidate_binding(namespace):
    """Synthetic initial candidate, never an actual execution/review receipt."""
    return dict(namespace['live_initialization'](),
        status='REVIEWED_FOR_FRESH200_LIVE_TRAINING_ONLY', validation_scope=namespace['LIVE_SCOPE'],
        calibration_scope=namespace['CALIBRATION_SCOPE'],
        live_goal_updated_at=LATEST_HUMAN_GOAL_UPDATED_AT,
        live_authorization_goal_updated_at=1791293201, official_training_outer_target=200,
        initial_calibration_counted_outer_limit=5, first_full_acceptance_step=5,
        first5_count_toward_goal=True, restart_after_calibration=False,
        first5_known_token_kl_enforced=True, first5_tail_observation_only=True,
        production_tail_required_before_outer6=True, retained_training_updates_contributed=0,
        candidate_old_policy_kl=.2, target_old_policy_kl=.2,
        token_kl_threshold_source='human', token_kl_human_goal_updated_at=1791282908,
        outer_counting_authorization_source='human', outer_counting_human_goal_updated_at=1791283983,
        allowed_safety_modes=['calibration', 'production'], code_commit='a' * 40,
        source_commit='a' * 40, source_native_acceptance=None, negative_tail_limit=None,
        actual_gpu_acceptance=None, actual_probability_acceptance=None)


class LatestLiveGoalBinding(unittest.TestCase):
    def test_actual_validator_accepts_latest_candidate_clock(self):
        namespace = actual_source_namespace()
        self.assertEqual(namespace['LIVE_GOAL_UPDATED_AT'], LATEST_HUMAN_GOAL_UPDATED_AT)
        binding = candidate_binding(namespace)
        namespace['validate_live_binding'](binding, 'a' * 40)
        self.assertEqual(binding['official_training_outer_target'], 200)
        self.assertTrue(binding['first5_count_toward_goal'])
        self.assertIsNone(binding['actual_gpu_acceptance'])
        self.assertIsNone(binding['actual_probability_acceptance'])

    def test_actual_validator_rejects_superseded_source10_clock(self):
        namespace = actual_source_namespace()
        binding = candidate_binding(namespace)
        binding['live_goal_updated_at'] = SUPERSEDED_SOURCE10_GOAL_UPDATED_AT
        with self.assertRaisesRegex(ValueError, 'live_goal_updated_at'):
            namespace['validate_live_binding'](binding, 'a' * 40)

    def test_latest_clock_does_not_relax_source_or_counted_first5_identity(self):
        namespace = actual_source_namespace()
        for change in ({'source_commit': 'b' * 40}, {'first5_count_toward_goal': False},
                       {'restart_after_calibration': True}, {'official_training_outer_target': 201}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                namespace['validate_live_binding'](dict(candidate_binding(namespace), **change), 'a' * 40)
