"""External v2 draft CPU checks; fixtures never certify native execution."""
import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import submit_policy_discovery_v2_native as module
common = module.common


def fixtures():
    old = dict(base='/mnt/data/shaoshitong/oldfixture', parent_base='/mnt/data/shaoshitong/oldparentfixture',
               job_id='3274', source_commit=module.OLD_COMMIT, source_sha256={'fixture.py': 'a' * 64})
    old_proof = dict(source_sha256=old['source_sha256'], checked_unix=1., modules=['original_suite'])
    new_proof = dict(base='/mnt/data/shaoshitong/newfixture', source_sha256={'fixture.py': 'b' * 64},
                     checked_unix=2., modules=['original_suite', module.REQUIRED_CACHE_TEST],
                     returncode=0, shell_syntax_returncode=0,
                     stdout=json.dumps(dict(successful=True, cuda_available=False, tests_run=15,
                                            skipped=[], failures=0, errors=0)))
    cleanup = dict(base=old['base'], job_id=old['job_id'], run_id='c' * 32, actual_completed_updates=0,
                   owned_namespace_empty=True, owned_shm_root_absent=True, cleaned=True,
                   owned_temporary_cleanup_complete=True, failed_training_run_remains_failed=True,
                   production_admission=False, scope='CPU fixture, no actual owned cleanup claim')
    return old, old_proof, new_proof, cleanup


class DiscoveryRetryTests(unittest.TestCase):
    def test_explicit_new_commit_rejects_old_or_missing_alias(self):
        for value in (None, '', module.OLD_COMMIT, 'HEAD', 'main', 'a' * 39, 'z' * 40):
            with self.subTest(value=value), self.assertRaises(ValueError): module.check_commit(value)
        module.check_commit('a' * 40)

    def test_new_actual_target_suite_required_not_old132(self):
        _, old, new, _ = fixtures()
        self.assertEqual(module.validate_cpu(new, old), new['source_sha256'])
        for key, value in [('source_sha256', old['source_sha256']), ('checked_unix', 1.),
                           ('modules', ['original_suite']), ('returncode', True),
                           ('shell_syntax_returncode', 1), ('base', '/Users/fixture')]:
            changed = copy.deepcopy(new); changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): module.validate_cpu(changed, old)
        for key, value in [('successful', False), ('cuda_available', True), ('skipped', ['cache-test']),
                           ('tests_run', True), ('failures', 1), ('errors', 1)]:
            changed = copy.deepcopy(new); summary = json.loads(changed['stdout']); summary[key] = value
            changed['stdout'] = json.dumps(summary)
            with self.subTest(key=key), self.assertRaises(ValueError): module.validate_cpu(changed, old)

    def test_cleanup_proof_identity_counter_and_failure_preservation(self):
        old, _, _, cleanup = fixtures()
        self.assertEqual(module.cleanup_schema(cleanup, old), cleanup)
        for key, value in [('base', old['parent_base']), ('job_id', '3275'),
                           ('actual_completed_updates', True), ('actual_completed_updates', 1),
                           ('owned_namespace_empty', False), ('owned_shm_root_absent', False),
                           ('failed_training_run_remains_failed', False), ('production_admission', True)]:
            changed = copy.deepcopy(cleanup); changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): module.cleanup_schema(changed, old)

    def test_missing_new_proof_has_no_remote_or_reserved_v2_intent(self):
        with tempfile.TemporaryDirectory() as directory:
            prep = Path(directory)
            with patch.object(common, 'remote', side_effect=AssertionError('No remote calls allowed')) as call, \
                 patch.object(module, 'original_inputs', return_value=fixtures()[:2]):
                with self.assertRaises(FileNotFoundError):
                    module.submit('a' * 40, prep / 'missing-new-cpu.json', prep / 'missing-cleanup.json', prep=prep)
                call.assert_not_called()
            self.assertFalse((prep / module.NAMES['intent']).exists())

    def test_unknown_partial_v2_intent_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            prep = Path(directory); path = prep / module.NAMES['intent']; path.write_bytes(b'{partial')
            with patch.object(common, 'remote') as call:
                with self.assertRaises(ValueError): module.submit('a' * 40, 'missing', 'missing', prep=prep)
                call.assert_not_called()
            self.assertEqual(path.read_bytes(), b'{partial')

    def test_failed_live_previous_reread_prevents_any_mutation(self):
        old, _, new, cleanup = fixtures()
        inputs = dict(old_launch=old, previous_cleanup=cleanup, source_commit='a' * 40,
                      source_sha256=new['source_sha256'])
        with tempfile.TemporaryDirectory() as directory:
            prep = Path(directory)
            with patch.object(module, 'load_inputs', return_value=(inputs, {})), \
                 patch.object(common, 'remote', side_effect=ValueError('Live original process remains')) as call:
                with self.assertRaises(ValueError): module.submit('a' * 40, 'fixture', 'fixture', prep=prep)
                self.assertEqual(call.call_count, 1)
                self.assertIn('previous_readonly', call.call_args.args[1])
            self.assertFalse((prep / module.NAMES['intent']).exists())

    def test_unknown_new_copy_reserves_exactly_one_v2_and_preserves_original(self):
        old, _, new, cleanup = fixtures()
        inputs = dict(old_launch=old, previous_cleanup=cleanup, source_commit='a' * 40,
                      source_sha256=new['source_sha256'], target_cpu_proof_sha256='d' * 64,
                      previous_cleanup_proof_sha256='e' * 64)
        files = {'reproduction/common8/assets_export.py':
                 b"BASE=Path('/dataset/shaoshitong/g1-common8-original-retrieval-export')\n"}
        with tempfile.TemporaryDirectory() as directory:
            prep = Path(directory); original = prep / 'policy-discovery-native-launch.json'
            original.write_bytes(b'original failed handle bytes')
            with patch.object(module, 'load_inputs', return_value=(inputs, files)), \
                 patch.object(common, 'remote', side_effect=[{'fixture_readonly': True}, subprocess.TimeoutExpired('fixture', 1)]) as call:
                with self.assertRaises(subprocess.TimeoutExpired): module.submit('a' * 40, 'fixture', 'fixture', prep=prep)
                claim = common.strict_json((prep / module.NAMES['intent']).read_bytes())
                self.assertRegex(claim['base'], r'^/mnt/data/shaoshitong/g1d-[0-9a-f]{8}$')
                self.assertRegex(claim['parent_base'], r'^/mnt/data/shaoshitong/g1p-[0-9a-f]{8}$')
                self.assertNotEqual(claim['base'], claim['parent_base'])
                self.assertEqual(claim['ray_socket_budgets']['discovery']['worst_plasma_socket_bytes'], 105)
                with self.assertRaises(ValueError): module.submit('a' * 40, 'fixture', 'fixture', prep=prep)
                self.assertEqual(call.call_count, 2)
            self.assertEqual(original.read_bytes(), b'original failed handle bytes')

    def test_complete_commit_tree_and_each_actual_blob_match(self):
        # Evidence-only HEAD has the same old runtime: this validates only the
        # Git reader. load_inputs still independently rejects the old CPU map.
        commit = subprocess.check_output(['git', 'rev-parse', '2ac5d5c'], cwd=module.REPO, text=True).strip()
        mapping = common.strict_json((HERE / 'policy-discovery-native-launch.json').read_bytes())['source_sha256']
        files = module.committed_sources(commit, mapping)
        self.assertEqual({key: common.sha(value) for key, value in files.items()}, mapping)
        missing = dict(mapping); missing.pop(next(iter(missing)))
        with self.assertRaises(ValueError): module.committed_sources(commit, missing)
        changed = dict(mapping); changed[next(iter(changed))] = '0' * 64
        with self.assertRaises(ValueError): module.committed_sources(commit, changed)

    def test_actual_installed_ray_archived_method_boundary_and_worst_session(self):
        proof = common.strict_json((HERE / 'policy-discovery-ray-socket-source.json').read_bytes())
        namespace = {'sys': SimpleNamespace(platform='linux')}
        exec(proof['functions'][0], namespace)
        function = namespace['_prepare_socket_file']
        dummy = SimpleNamespace(_sockets_dir='/unused', _make_inc_temp=lambda **kwargs: '/' + 'a' * 106)
        self.assertEqual(len(function(dummy, None, 'plasma_store').encode()), 107)
        dummy._make_inc_temp = lambda **kwargs: '/' + 'a' * 107
        with self.assertRaises(OSError): function(dummy, None, 'plasma_store')
        for name in ('g1p-', 'g1d-'):
            self.assertEqual(common.ray_socket_budget('/mnt/data/shaoshitong/' + name + '12345678')['worst_plasma_socket_bytes'], 105)

    def test_exact_actual_collected_failed_history_binds_purged_fallback(self):
        old = common.strict_json((HERE / 'policy-discovery-native-launch.json').read_bytes())
        observation = common.strict_json((HERE / 'policy-discovery-native-observation.json').read_bytes())
        run_id = observation['run-manifest.json']['run_id']
        self.assertEqual(module.previous_terminal_schema(observation, old, run_id)['JobState'], 'FAILED')
        for mutation in ('jobid', 'runid', 'command', 'active', 'notfailed'):
            changed = copy.deepcopy(observation)
            if mutation == 'jobid': changed['job_id'] = '3275'
            elif mutation == 'runid': changed['run-manifest.json']['run_id'] = 'f' * 32
            elif mutation == 'command': changed['job_stdout'] = changed['job_stdout'].replace('/repo/reproduction/common8/launch_calibration_native.sh', '/different-command')
            elif mutation == 'active': changed['actual_namespace_processes'] = [{'state': 'R'}]
            else: changed['job_stdout'] = changed['job_stdout'].replace('JobState=FAILED', 'JobState=RUNNING')
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                module.previous_terminal_schema(changed, old, run_id)

    def test_purged_native_route_requires_exact_history_and_current_empty_queue(self):
        old = common.strict_json((HERE / 'policy-discovery-native-launch.json').read_bytes())
        observation = common.strict_json((HERE / 'policy-discovery-native-observation.json').read_bytes())
        run_id = observation['run-manifest.json']['run_id']
        command = SimpleNamespace(returncode=1, stdout='', stderr='slurm_load_jobs error: Invalid job id specified')
        with patch.object(module.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout='', stderr='')) as call:
            self.assertEqual(module.native_terminal_route(old, observation, run_id, command),
                             'exact_historical_failed_command_and_actual_slurm_purged')
            self.assertEqual(call.call_args.args[0][:4], ['squeue', '-h', '-j', '3274'])
        for result in (SimpleNamespace(returncode=0, stdout='3274|RUNNING|foreign', stderr=''),
                       SimpleNamespace(returncode=1, stdout='', stderr='network unavailable')):
            with patch.object(module.subprocess, 'run', return_value=result), self.assertRaises(ValueError):
                module.native_terminal_route(old, observation, run_id, command)
        with patch.object(module.subprocess, 'run') as call, self.assertRaises(ValueError):
            module.native_terminal_route(old, observation, run_id,
                SimpleNamespace(returncode=1, stdout='', stderr='connection failed'))
        call.assert_not_called()
        changed = copy.deepcopy(observation); changed['job_stdout'] = changed['job_stdout'].replace('JobState=FAILED', 'JobState=RUNNING')
        with patch.object(module.subprocess, 'run') as call, self.assertRaises(ValueError):
            module.native_terminal_route(old, changed, run_id, command)
        call.assert_not_called()

    def test_compile_resource_and_prior_actual_reread_contracts(self):
        for name in ('PREPARE_CODE', 'SUBMIT_CODE'):
            compile(getattr(module, name), name, 'exec')
        compile(module.previous_code(), 'actual-old-failed-readonly', 'exec')
        for token in ('--gres=gpu:h100:8', '--mem=512G', '--exclusive', '--nodelist=h100-8'):
            self.assertIn(token, module.SUBMIT_CODE)
        self.assertIn('Node._prepare_socket_file', module.PREPARE_CODE)
        self.assertIn('installed_limit_tested_bytes=[107,108]', module.PREPARE_CODE)
        self.assertNotIn('"107" in s', module.PREPARE_CODE)
        self.assertIn('previous_readonly', module.previous_code())


if __name__ == '__main__':
    unittest.main()
