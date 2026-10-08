"""CPU orchestration regressions. Fixture logs are not actual GPU evidence."""
import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent / 'repo'
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location('confirmation_orchestration', HERE / 'submit_policy_confirmation_native.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
from reproduction.common8.policy_calibration_plan import canonical_bytes, freeze_rule
from reproduction.common8.production_safety_admission import evaluate_health, AdmissionError
from reproduction.common8.safety_calibration_evidence import verify_segment, EvidenceError
from reproduction.common8.test_safety_calibration_evidence import fabricated_segment


def fixture():
    """Explicit CPU fixture; no native execution/acceptance is manufactured."""
    method_bytes = (HERE / module.METHOD_NAME).read_bytes()
    method = module.strict_json(method_bytes)
    launch = dict(base='/mnt/data/shaoshitong/test-discovery', parent_base='/mnt/data/shaoshitong/test-parent',
                  source_commit=module.FROZEN_COMMIT, source_sha256={'fixture.py': 'a' * 64}, job_id='fixture3274')
    health = freeze_rule(method['health_rule_document'])
    selection = dict(schema_version=1, role='policy_safety_confirmation_orchestration_selection',
                     selected_by='root_explicit_selection', selection_provenance={'scope': 'CPU fixture, not real selected GPU run'},
                     parent_base=launch['parent_base'], discovery_base=launch['base'],
                     parent_production_run_id=method['parent_production_run_id'], discovery_run_id=method['discovery_run_id'],
                     discovery_slurm_job_id=launch['job_id'], source_commit=module.FROZEN_COMMIT,
                     source_map_sha256=module.sha(module.canonical(launch['source_sha256'])),
                     scientific_identity_sha256='b' * 64, discovery_manifest_sha256='c' * 64,
                     discovery_plan_file_sha256='d' * 64, discovery_plan_sha256='e' * 64,
                     discovery_evidence_sha256='f' * 64, selection_method_sha256=module.FROZEN_METHOD_SHA256,
                     negative_tail_limit=0.25, health_rule=health)
    thresholds = dict(method['fixed_candidates'], negative_tail_limit=selection['negative_tail_limit'])
    selection['threshold_selection'] = freeze_rule(dict(role='policy_safety_threshold_selection',
        provenance={'scope': 'CPU fixture, not real selected GPU run'}, criteria=dict(
            discovery_evidence_sha256=selection['discovery_evidence_sha256'],
            health_rule_sha256=health['sha256'], candidate_thresholds=thresholds)))
    return selection, launch, method_bytes


def reconstructed_fixture(selection):
    logs, binding = fabricated_segment(role=module.DISCOVERY_ROLE)
    binding['run_id'] = selection['discovery_run_id']
    binding['calibration_sha256'] = selection['discovery_plan_sha256']
    for rows in logs.values():
        for row in rows:
            row['run_id'] = binding['run_id']
            if 'calibration_sha256' in row:
                row['calibration_sha256'] = binding['calibration_sha256']
    return logs, binding, dict(structural=verify_segment(logs, binding))


class ConfirmationPreparationTests(unittest.TestCase):
    def test_unicode_hash_matches_frozen_runtime(self):
        value = dict(provenance='仅结构测试，无实际GPU健康证明')
        self.assertEqual(module.canonical(value), canonical_bytes(value))

    def test_selection_explicit_roles_and_method_hash(self):
        selection, launch, method_bytes = fixture()
        self.assertEqual(module.selection_schema(selection, launch), selection)
        self.assertEqual(module.method_binding(method_bytes, selection)['negative_tail_limit'], None)
        for key, value in [('schema_version', True), ('selected_by', 'automatic_default'),
                           ('negative_tail_limit', None), ('negative_tail_limit', True),
                           ('negative_tail_limit', float('nan')), ('selection_method_sha256', '0' * 64)]:
            changed = copy.deepcopy(selection); changed[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                module.selection_schema(changed, launch)

    def test_reject_cross_run_source_and_raw_hash(self):
        selection, launch, _ = fixture()
        for key, value in [('source_commit', '2ac5d5c' + '0' * 33),
                           ('source_map_sha256', '0' * 64), ('discovery_base', launch['parent_base']),
                           ('discovery_slurm_job_id', '3275'),
                           ('parent_production_run_id', selection['discovery_run_id'])]:
            changed = copy.deepcopy(selection); changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                module.selection_schema(changed, launch)
        changed = copy.deepcopy(selection)
        changed['threshold_selection']['document']['criteria']['discovery_evidence_sha256'] = '0' * 64
        changed['threshold_selection'] = freeze_rule(changed['threshold_selection']['document'])
        with self.assertRaises(ValueError): module.selection_schema(changed, launch)

    def test_frozen_original_method_rejects_relaxed_health_even_with_correct_wrapper_sha(self):
        selection, launch, method_bytes = fixture()
        selection['health_rule']['document']['criteria']['allowed_drop']['weighted_em'] = 0.5
        selection['health_rule'] = freeze_rule(selection['health_rule']['document'])
        choice = selection['threshold_selection']['document']
        choice['criteria']['health_rule_sha256'] = selection['health_rule']['sha256']
        selection['threshold_selection'] = freeze_rule(choice)
        module.selection_schema(selection, launch)  # Valid shape is not root method consent.
        with self.assertRaises(ValueError): module.method_binding(method_bytes, selection)
        with self.assertRaises(ValueError): module.method_binding(method_bytes + b' ', fixture()[0])

    def test_real_health_api_compares_before_after_when_same_segment_passed_twice(self):
        selection, _, _ = fixture()
        ids = ['cpu-fixture-' + str(i) for i in range(512)]
        before = dict(question_count=512, question_ids=ids, data_sources=['nq'] * 256 + ['hotpotqa'] * 256,
                      weighted_em=0.25, weighted_f1=0.5, valid_action_rate=1.0,
                      valid_action_rate_definition='CPU fixture typed valid action')
        segment = dict(evaluation_before=before, evaluation_after=copy.deepcopy(before))
        result = evaluate_health(selection['health_rule'], segment, segment)
        self.assertTrue(result['accepted'])
        for key in ('weighted_em', 'weighted_f1', 'valid_action_rate'):
            changed = copy.deepcopy(segment); changed['evaluation_after'][key] -= 0.01
            with self.subTest(key=key), self.assertRaises(AdmissionError):
                evaluate_health(selection['health_rule'], changed, changed)

    def test_exact_40_global_max_no_floor_or_margin(self):
        selection, _, _ = fixture()
        _, _, segment = reconstructed_fixture(selection)
        observations = segment['structural']['global_monitors']
        maximum = max(row['negative_tail_weight_fraction'] for row in observations)
        selection['negative_tail_limit'] = maximum
        self.assertEqual(module.measured_selection(selection, segment), maximum)
        self.assertEqual(len(observations), 40)  # Real reader coalesces eight replicas.
        selection['negative_tail_limit'] = 1.0
        with self.assertRaises(ValueError): module.measured_selection(selection, segment)

    def test_reader_rejects_global_rank_disagreement_before_method(self):
        selection, _, _ = fixture()
        logs, binding, _ = reconstructed_fixture(selection)
        logs[7][0]['negative_tail_weight_fraction'] += 0.01
        with self.assertRaises(EvidenceError): verify_segment(logs, binding)

    def test_incomplete_duplicate_and_reordered_coalesced_observations_reject(self):
        selection, _, _ = fixture()
        _, _, segment = reconstructed_fixture(selection)
        selection['negative_tail_limit'] = max(row['negative_tail_weight_fraction'] for row in segment['structural']['global_monitors'])
        for variant in ('missing', 'duplicate', 'reorder', 'role', 'bool_context'):
            changed = copy.deepcopy(segment); rows = changed['structural']['global_monitors']
            if variant == 'missing': rows.pop()
            elif variant == 'duplicate': rows[1] = copy.deepcopy(rows[0])
            elif variant == 'reorder': rows[0], rows[1] = rows[1], rows[0]
            elif variant == 'role': rows[0]['execution_role'] = module.CONFIRMATION_ROLE
            else: rows[0]['context']['logical_update'] = True
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                module.measured_selection(selection, changed)

    def test_missing_selection_has_no_remote_or_intent(self):
        with tempfile.TemporaryDirectory() as directory:
            prep = Path(directory)
            with patch.object(module, 'remote', side_effect=AssertionError('Must not reach network')) as call:
                with self.assertRaises(FileNotFoundError): module.submit(prep)
                call.assert_not_called()
            self.assertFalse((prep / 'policy-confirmation-native-submit-intent.json').exists())

    def test_partial_intent_is_not_overwritten_or_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            prep = Path(directory); claim = prep / 'policy-confirmation-native-submit-intent.json'
            claim.write_bytes(b'{partial')
            with patch.object(module, 'remote') as call:
                with self.assertRaises(ValueError): module.submit(prep)
                call.assert_not_called()
            self.assertEqual(claim.read_bytes(), b'{partial')

    def test_source_failure_rejects_before_remote_mutation(self):
        selection, launch, _ = fixture()
        with tempfile.TemporaryDirectory() as directory:
            prep = Path(directory)
            with patch.object(module, 'load_inputs', return_value=(json.dumps(selection).encode(), selection, launch)), \
                 patch.object(module, 'frozen_sources', return_value={}), \
                 patch.object(module, 'remote', side_effect=ValueError('Actual source not complete')) as call:
                with self.assertRaises(ValueError): module.submit(prep)
                self.assertEqual(call.call_count, 1)
                self.assertIn('target_evidence', call.call_args.args[1])
            self.assertFalse((prep / 'policy-confirmation-native-submit-intent.json').exists())

    def test_unknown_target_copy_keeps_one_claim_and_cannot_resubmit(self):
        selection, launch, _ = fixture()
        files = {'reproduction/common8/launch_calibration_native.sh':
                 b'#!/bin/bash\npython -m reproduction.common8.prepare_calibration_run "${args[@]}"\nexec python -m reproduction.common8.run_pipeline --base "$BASE"\n',
                 'reproduction/common8/assets_export.py':
                 b"BASE=Path('/dataset/shaoshitong/g1-common8-original-retrieval-export')\n"}
        with tempfile.TemporaryDirectory() as directory:
            prep = Path(directory); (prep / module.METHOD_NAME).write_bytes(fixture()[2])
            with patch.object(module, 'load_inputs', return_value=(json.dumps(selection).encode(), selection, launch)), \
                 patch.object(module, 'frozen_sources', return_value=files), \
                 patch.object(module, 'remote', side_effect=[{'fixture_prerequisite': True}, subprocess.TimeoutExpired('fixture', 1)]) as call:
                with self.assertRaises(subprocess.TimeoutExpired): module.submit(prep)
                self.assertEqual(call.call_count, 2)
                self.assertTrue((prep / 'policy-confirmation-native-submit-intent.json').exists())
                with self.assertRaises(ValueError): module.submit(prep)
                self.assertEqual(call.call_count, 2)

    def test_wrapper_binds_only_after_native_prepare_before_runtime(self):
        original = (REPO / 'reproduction/common8/launch_calibration_native.sh').read_bytes()
        wrapper = module.native_wrapper({'reproduction/common8/launch_calibration_native.sh': original}).decode()
        self.assertLess(wrapper.index('python -m reproduction.common8.prepare_calibration_run'), wrapper.index('--native-bind'))
        self.assertLess(wrapper.index('--native-bind'), wrapper.index('exec python -m reproduction.common8.run_pipeline'))
        result = subprocess.run(['bash', '-n'], input=wrapper, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_frozen_commit_source_blobs_actual_344_files(self):
        launch = module.strict_json((HERE / 'policy-discovery-native-launch.json').read_bytes())
        files = module.frozen_sources(launch)
        self.assertEqual({name: module.sha(content) for name, content in files.items()}, launch['source_sha256'])
        self.assertEqual(len(files), 344)
        with self.assertRaises(ValueError): module.frozen_sources(dict(launch, source_commit='2ac5d5c' + '0' * 33))

    def test_short_base_ray_budget_and_long_unix_path_reject(self):
        proof = module.ray_socket_budget('/mnt/data/shaoshitong/g1c-12345678')
        self.assertLessEqual(proof['worst_plasma_socket_bytes'], 107)
        self.assertIs(proof['actual_ray_started'], False)
        with self.assertRaises(ValueError): module.ray_socket_budget('/mnt/data/shaoshitong/Search-G1-common8-policy-confirmation-' + 'a' * 12)

    def test_cleanup_is_never_inferred_from_unknown_submission(self):
        self.assertFalse(module.cleanup_eligible(dict(actual_active_namespace_processes=[], submission_outcome_known=False)))
        self.assertFalse(module.cleanup_eligible(dict(job_returncode=0, job_stdout='JobState=RUNNING', actual_active_namespace_processes=[])))
        self.assertFalse(module.cleanup_eligible(dict(job_returncode=0, job_stdout='JobState=FAILED', actual_active_namespace_processes=[{'pid': 1}])))
        self.assertTrue(module.cleanup_eligible(dict(job_returncode=0, job_stdout='JobState=FAILED', actual_active_namespace_processes=[])))
        self.assertTrue(module.cleanup_eligible(dict(actual_active_namespace_processes=[{'pid': 1}],
                        **{'retrieval-staging-status.json': {'stage': 'COMPLETE', 'accepted': True}})))

    def test_embedded_scripts_compile_and_submit_resource_contract(self):
        for name in ('PREPARE_CODE', 'START_SOURCE_CODE', 'SUBMIT_CODE', 'OBSERVE_CODE', 'SOURCE_HANDLE_CODE', 'CLEANUP_SOURCE_CODE'):
            compile(getattr(module, name), name, 'exec')
        compile(module.verification_code(), 'actual-source-verification', 'exec')
        for flag in ('--gres=gpu:h100:8', '--mem=512G', '--exclusive', '--nodelist=h100-8'):
            self.assertIn(flag, module.SUBMIT_CODE)
        self.assertLess(module.CLEANUP_SOURCE_CODE.index('os.pidfd_open'), module.CLEANUP_SOURCE_CODE.index("(path/'stat')"))
        self.assertIn('signal.pidfd_send_signal(descriptor', module.CLEANUP_SOURCE_CODE)
        self.assertNotIn('os.kill(', module.CLEANUP_SOURCE_CODE)


if __name__ == '__main__':
    unittest.main()
