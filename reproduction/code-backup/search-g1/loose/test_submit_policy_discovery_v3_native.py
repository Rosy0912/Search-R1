"""Stdlib-only external v3 protocol tests; no remote/native/model execution.

Actual historical JSON bytes are read locally. New CPU/source and route outputs
are explicitly unit fixtures where noted; no fixture authenticates a GPU job.
"""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as S
from unittest.mock import patch

import submit_policy_discovery_v3_native as v


def old_inputs():
    old, cpu = v.original_inputs()
    failure = v.common.strict_json((v.PREP / v.FAILURE_NAME).read_bytes())
    observed = v.common.strict_json((v.PREP / v.OBSERVATION_NAME).read_bytes())
    running = v.common.strict_json((v.PREP / v.RUNNING_NAME).read_bytes())
    return old, cpu, failure, observed, running


def cpu_unit_fixture(old):
    # Explicit CPU reader fixture, not a claimed real target test result.
    proof = copy.deepcopy(old)
    summary = v.common.strict_json(proof['stdout'])
    summary.update(tests_run=218, successful=True, failures=0, errors=0, skipped=[], cuda_available=False)
    proof.update(stdout=json.dumps(summary), checked_unix=old['checked_unix'] + 1,
                 modules=old['modules'] + sorted(v.NEW_TEST_MODULES),
                 source_sha256=old['source_sha256'] | {'cpu_only_fixture.py': 'a' * 64})
    return proof


class DiscoveryV3Tests(unittest.TestCase):
    def test_actual_historical_bytes_are_partial_one_not_fresh_zero(self):
        old, cpu, failure, observed, running = old_inputs()
        v.partial_schema(failure, old); v.history_schema(running, observed, old)
        self.assertEqual(v.common.strict_json(cpu['stdout'])['tests_run'], 147)
        self.assertEqual(old['source_commit'], v.OLD_COMMIT)
        self.assertEqual(observed['actual_updates'], [])
        for rank in range(8):
            rows = failure['rank_evidence'][str(rank)]['rows']
            self.assertEqual([r['kind'] for r in rows], ['POLICY_MONITOR', 'OPTIMIZER_WRITE', 'POLICY_MONITOR'])
            self.assertEqual(rows[1]['write_count'], 1)
            self.assertEqual(rows[2]['optimizer_writes'], 1)
            self.assertFalse(rows[2]['allowed'])

    def test_partial_zero_two_missing_rank_and_completed_budget_reject(self):
        old, _, original, _, _ = old_inputs()
        changes = [lambda f: f['rank_evidence'].pop('7'),
                   lambda f: f['rank_evidence']['7']['rows'][1].update(write_count=0),
                   lambda f: f['rank_evidence']['7']['rows'][1].update(write_count=2),
                   lambda f: f['rank_evidence']['7']['rows'][1].update(write_count=True),
                   lambda f: f['rank_evidence']['7']['rows'][1].update(learning_rates=[1e-5]),
                   lambda f: f['rank_evidence']['7']['rows'].append(dict(kind='OUTER_BUDGET')),
                   lambda f: f['rank_evidence']['7']['rows'][2].update(allowed=True),
                   lambda f: f.update(owned_staging_root_exists=True)]
        for change in changes:
            bad = copy.deepcopy(original); change(bad)
            with self.assertRaises(ValueError): v.partial_schema(bad, old)

    def test_old_run_source_exit_or_namespace_drift_reject(self):
        old, _, _, original, running = old_inputs()
        changes = [lambda o: o['run-manifest.json'].update(source_commit='0' * 40),
                   lambda o: o['run-manifest.json'].update(run_id='0' * 32),
                   lambda o: o['process-ledger.json']['jobs']['trainer'].update(exit_code=0),
                   lambda o: o.update(actual_updates=[dict(step=1)]),
                   lambda o: o.update(active_owned_pids=[123]),
                   lambda o: o['cleanup-status.json'].update(cleaned=False),
                   lambda o: o['process-ledger.json']['jobs']['trainer'].update(state='RUNNING')]
        for change in changes:
            bad = copy.deepcopy(original); change(bad)
            with self.assertRaises(ValueError): v.history_schema(running, bad, old)

    def test_actual_running_history_is_required_not_fabricated_failed(self):
        old, _, _, observed, running = old_inputs()
        for old_value, new_value in [('JobState=RUNNING', 'JobState=FAILED'), ('JobId=3278', 'JobId=3274'),
                                     ('MinMemoryNode=512G', 'MinMemoryNode=256G')]:
            bad = copy.deepcopy(running); bad['job_stdout'] = bad['job_stdout'].replace(old_value, new_value)
            with self.assertRaises(ValueError): v.history_schema(bad, observed, old)

    def test_exact_purged_route_preserves_unknown_slurm_terminal_state(self):
        old, *_ = old_inputs()
        missing = S(returncode=1, stdout='', stderr='slurm_load_jobs error: Invalid job id specified\n')
        for queue in (S(returncode=0, stdout='', stderr=''), missing):
            with patch.object(v.subprocess, 'run', return_value=queue) as call:
                route = v.native_terminal_route(old, missing)
            self.assertIsNone(route['slurm_terminal_state'])
            self.assertIn('historical_running_actual_workflow_failed', route['route'])
            self.assertEqual(call.call_args.args[0][3], '3278')

    def test_other_query_errors_active_queue_and_unknown_job_do_not_pass(self):
        old, *_ = old_inputs()
        missing = S(returncode=1, stdout='', stderr='slurm_load_jobs error: Invalid job id specified\n')
        for command in (S(returncode=1, stdout='', stderr='connection refused'),
                        S(returncode=1, stdout='unexpected', stderr=missing.stderr)):
            with self.assertRaises(ValueError): v.native_terminal_route(old, command)
        for queue in (S(returncode=0, stdout='3278|RUNNING', stderr=''),
                      S(returncode=1, stdout='', stderr='socket error')):
            with patch.object(v.subprocess, 'run', return_value=queue), self.assertRaises(ValueError):
                v.native_terminal_route(old, missing)

    def test_old147_or_missing_new_suite_cannot_be_new_target_proof(self):
        _, old, *_ = old_inputs(); new = cpu_unit_fixture(old)
        self.assertEqual(len(new['modules']), 16)
        v.validate_cpu(new, old)
        for change in (lambda p: p.update(source_sha256=old['source_sha256']),
                       lambda p: p.update(checked_unix=old['checked_unix']),
                       lambda p: p['modules'].pop(),
                       lambda p: p.update(stdout=p['stdout'].replace('218', '147')),
                       lambda p: p.update(stdout=p['stdout'].replace('"skipped": []', '"skipped": ["fixture"]'))):
            bad = copy.deepcopy(new); change(bad)
            with self.assertRaises(ValueError): v.validate_cpu(bad, old)

    def test_committed_reader_uses_actual_git_blobs_and_exact_full_map(self):
        # Actual existing historical Git blob reading only, no Git mutation.
        # This is source-reader coverage, not permission to use the old recipe.
        import subprocess
        commit = '61f6e1da532e08553410666b430d1933a0e4a49a'
        names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', commit], cwd=v.REPO, text=True).splitlines()
        names = [name for name in names if (Path(name).suffix in ('.py','.sh','.yaml','.yml','.json') or
                 name in ('verl/version/version','verl/single_controller/version/version')) and
                 not {'.git','__pycache__','evidence'}.intersection(Path(name).parts)]
        mapping = {name:v.common.sha(subprocess.check_output(['git','cat-file','blob',commit+':'+name],cwd=v.REPO))
                   for name in names}
        files = v.committed_sources(commit, mapping)
        self.assertEqual(set(files), set(mapping))
        for bad in (mapping | {'missing.py':'0'*64}, mapping | {names[0]:'0'*64}):
            with self.assertRaises(ValueError): v.committed_sources(commit,bad)
        for bad in (v.OLD_COMMIT, 'HEAD', '130d17a'):
            with self.assertRaises(ValueError): v.check_commit(bad)

    def test_missing_new_proof_or_source_failure_has_no_remote_or_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            prep = Path(directory)
            with (patch.object(v, 'load_inputs', side_effect=ValueError('explicit unit missing proof')),
                  patch.object(v.common, 'remote') as remote):
                with self.assertRaises(ValueError): v.submit('a'*40, prep/'missing.json', prep=prep)
                remote.assert_not_called(); self.assertFalse((prep/v.NAMES['intent']).exists())

    def test_partial_intent_never_retries_or_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            prep=Path(directory); path=prep/v.NAMES['intent']; path.write_bytes(b'{partial')
            with patch.object(v, 'load_inputs') as load, patch.object(v.common, 'remote') as remote:
                with self.assertRaises(ValueError): v.submit('a'*40, 'unused', prep=prep)
                load.assert_not_called(); remote.assert_not_called(); self.assertEqual(path.read_bytes(),b'{partial')

    def test_unknown_copy_keeps_one_intent_and_blocks_all_retries(self):
        with tempfile.TemporaryDirectory() as directory:
            prep=Path(directory)
            old,*_=old_inputs()
            inputs=dict(old_launch=old,source_commit='a'*40,source_sha256={'cpu-only.py':'b'*64})
            files={'reproduction/common8/assets_export.py':b"BASE=Path('/dataset/shaoshitong/g1-common8-original-retrieval-export')"}
            with patch.object(v,'load_inputs',return_value=(inputs,files)), patch.object(v.common,'remote',
                    side_effect=[dict(scope='cpu-only old reread fixture'),RuntimeError('unknown copy')]) as remote:
                with self.assertRaises(RuntimeError):v.submit('a'*40,'unit',prep=prep)
                self.assertEqual(remote.call_count,2)
                original=(prep/v.NAMES['intent']).read_bytes()
                with self.assertRaises(ValueError):v.submit('a'*40,'unit',prep=prep)
                self.assertEqual((prep/v.NAMES['intent']).read_bytes(),original)
                self.assertEqual(remote.call_count,2)

    def test_resource_shortbase_and_distinct_claim_contract(self):
        for code in (v.previous_code(),v.PREPARE_CODE,v.SUBMIT_CODE):compile(code,'<cpu-only-embedded>','exec')
        self.assertNotIn('optimizer evidence exists; do not pretend fresh0',v.previous_code())
        for token in ('--gres=gpu:h100:8','--mem=512G','--exclusive','--time=08:00:00','search-g1-policy-discovery-v3'):
            self.assertIn(token,v.SUBMIT_CODE)
        self.assertLess(v.SUBMIT_CODE.index('os.fsync(fd)'),v.SUBMIT_CODE.index('r=subprocess.run(args'))
        for base in ('/mnt/data/shaoshitong/g1p-12345678','/mnt/data/shaoshitong/g1d-12345678'):
            self.assertLessEqual(v.common.ray_socket_budget(base)['worst_plasma_socket_bytes'],107)
        self.assertTrue(all('v3' in value for value in v.NAMES.values()))
        self.assertNotIn('METHOD_SHA',v.__dict__)

    def test_actual_new_recipe_rejects_old_top_p_and_duplicate_override(self):
        # Current local recipe bytes are configuration fixture data only.
        files = {'reproduction/common8/'+name+'.py':b'cpu-only presence fixture' for name in
                 ('behavior_trace','behavior_probability','behavior_runtime','generation_behavior')}
        recipe = v.common.strict_json((v.REPO/'reproduction/common8/recipe.json').read_bytes())
        files['reproduction/common8/recipe.json'] = v.common.canonical(recipe)
        self.assertEqual(v.new_recipe(files)['actor_rollout_ref.rollout.top_p'],'1.0')
        for bad in ([value.replace('rollout.top_p=1.0','rollout.top_p=0.95') for value in recipe],
                    recipe + ['actor_rollout_ref.rollout.top_p=0.95'],
                    [value.replace('ppo_micro_batch_size=16','ppo_micro_batch_size=8') for value in recipe]):
            with self.assertRaises(ValueError):v.new_recipe(files | {'reproduction/common8/recipe.json':v.common.canonical(bad)})

    def test_full_manifest_extra_fields_match_saved_trimmed_subset_and_raw_sha(self):
        _, _, _, observed, _ = old_inputs()
        # Real old poll's seven fields, with explicit CPU-only additional full
        # manifest fields. This fixture does not invent native original bytes.
        full = dict(observed['run-manifest.json'], profile='common8',
                    policy_calibration_slurm_job_id='3278', cpu_only_extra_field='not-a-native-proof')
        raw = v.common.canonical(full)
        fixture = copy.deepcopy(observed); fixture['child_manifest_sha256'] = v.common.sha(raw)
        self.assertNotEqual(full, fixture['run-manifest.json'])
        self.assertEqual(v.verify_old_manifest(raw, fixture), full)
        with self.assertRaises(ValueError):v.verify_old_manifest(raw + b'\n', fixture)
        bad = copy.deepcopy(fixture); bad['run-manifest.json'].pop('train_log')
        with self.assertRaises(ValueError):v.verify_old_manifest(raw, bad)
        bad = copy.deepcopy(fixture); bad['run-manifest.json']['train_log'] = 'different'
        with self.assertRaises(ValueError):v.verify_old_manifest(raw, bad)
        for name, value in (('policy_calibration_slurm_job_id','3274'), ('source_commit','0'*40),
                            ('completed_updates',True), ('completed_updates',1)):
            changed = dict(full); changed[name] = value
            changed_raw = v.common.canonical(changed)
            bad = copy.deepcopy(fixture); bad['child_manifest_sha256'] = v.common.sha(changed_raw)
            if name in bad['run-manifest.json']:bad['run-manifest.json'][name] = value
            with self.assertRaises(ValueError):v.verify_old_manifest(changed_raw,bad)


if __name__ == '__main__':
    unittest.main()
