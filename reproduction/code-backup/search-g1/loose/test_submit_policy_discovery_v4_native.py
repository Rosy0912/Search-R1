"""Stdlib external v4 protocol tests, never GPU/native/healthy admission.

Historical 3283 observations are real local original bytes. New CPU proofs,
scheduler command results and successful-terminal examples are explicitly unit
fixtures. Git tests use a temporary real repository, never edit project source.
"""
import base64
import copy
import inspect
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as S
from unittest.mock import patch

import submit_policy_discovery_v4_native as v


def terminal_arguments():
    return dict(expected_sha256=v.TERMINAL_SHA256,
                launch_bytes=(v.PREP / v.OLD_LAUNCH_NAME).read_bytes(),
                expected_launch_sha256=v.OLD_LAUNCH_SHA256, expected_run_id=v.OLD_RUN,
                expected_parent_run_id=v.OLD_PARENT_RUN, expected_state='FAILED', expected_exit_code='1:0',
                birth_observation_bytes=(v.PREP / v.BIRTH_NAME).read_bytes(), expected_birth_sha256=v.BIRTH_SHA256)


def actual_terminal():
    raw = (v.PREP / v.TERMINAL_NAME).read_bytes()
    return raw, v.read_terminal_observation(raw, **terminal_arguments())


def cpu_unit_fixture(old):
    # CPU reader fixture only: this is not an actual cluster regression proof.
    proof = copy.deepcopy(old)
    summary = v.common.strict_json(proof['stdout'])
    summary.update(tests_run=223, successful=True, failures=0, errors=0, skipped=[], cuda_available=False)
    mapping = copy.deepcopy(old['source_sha256']); mapping[next(iter(mapping))] = 'a' * 64
    proof.update(stdout=json.dumps(summary), checked_unix=old['checked_unix'] + 1,
                 modules=list(v.CPU_MODULES), source_sha256=mapping)
    return proof


class TerminalReaderTests(unittest.TestCase):
    def test_actual_original_failed_bytes_are_failure_not_completion(self):
        raw, terminal = actual_terminal()
        self.assertEqual(v.common.sha(raw), v.TERMINAL_SHA256)
        self.assertEqual((terminal['slurm_terminal_state'], terminal['exit_code']), ('FAILED', '1:0'))
        self.assertEqual(terminal['job_id'], '3283')
        self.assertEqual(terminal['run_id'], v.OLD_RUN)
        self.assertFalse(terminal['production_admission'])
        self.assertFalse(terminal['healthy_checkpoint_accepted'])
        args = terminal_arguments() | dict(expected_state='COMPLETED', expected_exit_code='0:0')
        with self.assertRaises(ValueError): v.read_terminal_observation(raw, **args)

    def test_original_raw_hash_launch_and_birth_hash_cannot_be_relabelled(self):
        raw, _ = actual_terminal()
        for key, value in (('expected_sha256', '0' * 64), ('expected_launch_sha256', '0' * 64),
                           ('expected_birth_sha256', '0' * 64), ('expected_sha256', True)):
            with self.assertRaises(ValueError): v.read_terminal_observation(raw, **(terminal_arguments() | {key: value}))
        with self.assertRaises(ValueError): v.read_terminal_observation(raw + b'\n', **terminal_arguments())

    def test_running_exit_zero_and_purged_never_become_terminal(self):
        raw, _ = actual_terminal(); original = v.common.strict_json(raw)
        for change in (lambda o: o.update(job_stdout=o['job_stdout'].replace('JobState=FAILED', 'JobState=RUNNING')),
                       lambda o: o.update(job_stdout=o['job_stdout'].replace('ExitCode=1:0', 'ExitCode=0:0')),
                       lambda o: o.update(job_stdout='', job_returncode=1, job_stderr=v.INVALID_JOB),
                       lambda o: o.update(job_returncode=True)):
            bad = copy.deepcopy(original); change(bad); content = v.common.canonical(bad)
            with self.assertRaises(ValueError):
                v.read_terminal_observation(content, **(terminal_arguments() | dict(expected_sha256=v.common.sha(content))))
        for state, exit_code in (('RUNNING','0:0'), ('COMPLETED','1:0'), ('FAILED','0:0'), ('CANCELLED','0:15')):
            with self.assertRaises(ValueError):
                v.read_terminal_observation(raw, **(terminal_arguments() | dict(expected_state=state, expected_exit_code=exit_code)))

    def test_run_source_job_command_workdir_resources_and_birth_drift_reject(self):
        raw, _ = actual_terminal(); original = v.common.strict_json(raw)
        mutations = [lambda o: o['run-manifest.json'].update(run_id='0' * 32),
                     lambda o: o['run-manifest.json'].update(source_commit='0' * 40),
                     lambda o: o.update(base='/mnt/data/shaoshitong/elsewhere'),
                     lambda o: o.update(source_hashes_match=False)]
        for before, after in [('JobId=3283','JobId=3284'), ('MinMemoryNode=512G','MinMemoryNode=256G'),
                              ('NodeList=h100-8','NodeList=h100-9'), ('Restarts=0','Restarts=1'),
                              ('SubmitTime=2026-10-06T08:09:32','SubmitTime=2026-10-06T08:09:31'),
                              ('StartTime=2026-10-06T08:09:33','StartTime=2026-10-06T08:09:34'),
                              ('WorkDir='+v.OLD_BASE+'/repo','WorkDir=/wrong'),
                              ('Command='+v.OLD_BASE+'/repo/reproduction/common8/launch_calibration_native.sh','Command=/wrong'),
                              ('EndTime=2026-10-06T08:33:44','EndTime=2026-10-06T08:09:01')]:
            mutations.append(lambda o, before=before, after=after: o.update(job_stdout=o['job_stdout'].replace(before, after)))
        for change in mutations:
            bad = copy.deepcopy(original); change(bad); content = v.common.canonical(bad)
            with self.assertRaises(ValueError):
                v.read_terminal_observation(content, **(terminal_arguments() | dict(expected_sha256=v.common.sha(content))))

    def test_capture_timeline_and_duplicate_scheduler_keys_reject(self):
        raw, _ = actual_terminal(); original = v.common.strict_json(raw)
        for change in (lambda o: o.update(checked_unix=1), lambda o: o.update(checked_unix=float('nan')),
                       lambda o: o.update(observation_finished_unix=o['checked_unix'] - 1),
                       lambda o: o.update(job_stdout=o['job_stdout'] + 'JobId=3283')):
            bad = copy.deepcopy(original); change(bad)
            # allow_nan is solely malformed JSON input fixture, never a proof.
            content = json.dumps(bad, allow_nan=True).encode()
            with self.assertRaises(ValueError):
                v.read_terminal_observation(content, **(terminal_arguments() | dict(expected_sha256=v.common.sha(content))))

    def test_explicit_completed_protocol_fixture_requires_zero_exit_and_stays_nonhealth(self):
        raw, _ = actual_terminal(); value = v.common.strict_json(raw)
        # Explicit scheduler reader fixture; 3283 really FAILED and this altered
        # content is never a native artifact nor a complete healthy fixture.
        value['job_stdout'] = value['job_stdout'].replace('JobState=FAILED','JobState=COMPLETED').replace('ExitCode=1:0','ExitCode=0:0')
        content = v.common.canonical(value)
        args = terminal_arguments() | dict(expected_state='COMPLETED', expected_exit_code='0:0',
                                           expected_sha256=v.common.sha(content))
        out = v.read_terminal_observation(content, **args)
        self.assertEqual(out['slurm_terminal_state'], 'COMPLETED')
        self.assertFalse(out['production_admission']); self.assertFalse(out['healthy_checkpoint_accepted'])

    def test_matching_current_terminal_or_exact_scoped_purge_preserves_original_fact(self):
        raw, terminal = actual_terminal(); observed = v.common.strict_json(raw)
        current = S(returncode=0, stdout=observed['job_stdout'], stderr='')
        self.assertIn('current_terminal_matches', v.verify_terminal_current(terminal, current)['route'])
        missing = S(returncode=1, stdout='', stderr=v.INVALID_JOB + '\n')
        for queue in (S(returncode=0, stdout='', stderr=''), missing):
            out = v.verify_terminal_current(terminal, missing, queue)
            self.assertIn('original_actual_terminal_bytes', out['route'])
            self.assertEqual((out['slurm_terminal_state'],out['exit_code']), ('FAILED','1:0'))
            self.assertEqual(out['original_terminal_sha256'], v.TERMINAL_SHA256)

    def test_other_errors_active_queue_recycled_job_and_changed_terminal_reject(self):
        raw, terminal = actual_terminal(); observed = v.common.strict_json(raw)
        missing = S(returncode=1, stdout='', stderr=v.INVALID_JOB + '\n')
        for query in (S(returncode=1, stdout='', stderr='connection refused'),
                      S(returncode=1, stdout='unexpected', stderr=missing.stderr),
                      S(returncode=True, stdout='', stderr=missing.stderr)):
            with self.assertRaises(ValueError): v.verify_terminal_current(terminal, query, S(returncode=0,stdout='',stderr=''))
        for queue in (None, S(returncode=0,stdout='3283|RUNNING',stderr=''),
                      S(returncode=1,stdout='',stderr='connection refused'),
                      S(returncode=0,stdout='',stderr='unexpected warning')):
            with self.assertRaises(ValueError): v.verify_terminal_current(terminal, missing, queue)
        for before, after in (('JobState=FAILED','JobState=RUNNING'), ('ExitCode=1:0','ExitCode=0:0'),
                              ('EndTime=2026-10-06T08:33:44','EndTime=2026-10-06T08:33:45'),
                              ('SubmitTime=2026-10-06T08:09:32','SubmitTime=2026-10-06T08:09:31')):
            changed = S(returncode=0,stdout=observed['job_stdout'].replace(before,after),stderr='')
            with self.assertRaises(ValueError): v.verify_terminal_current(terminal, changed)


class DiscoveryV4Tests(unittest.TestCase):
    def test_actual_original_failure_has_claim_but_zero_recorded_adam_and_outer(self):
        old, cpu, history = v.original_inputs()
        observed = v.common.strict_json(base64.b64decode(history['previous_terminal_bytes']))
        v.previous_schema(observed, old)
        self.assertEqual(v.common.strict_json(cpu['stdout'])['tests_run'],222)
        self.assertTrue(observed['child_actual_unstarted']['training_claim_exists'])
        self.assertEqual(observed['child_actual_unstarted']['optimizer_evidence_paths'],[])
        self.assertEqual(observed['actual_updates'],[])
        self.assertEqual(observed['process-ledger.json']['jobs']['trainer']['exit_code'],1)

    def test_previous_writes_outer_checkpoint_claim_exit_cleanup_or_parent_drift_reject(self):
        old, _, history = v.original_inputs(); original = v.common.strict_json(base64.b64decode(history['previous_terminal_bytes']))
        mutations = [lambda o:o.update(actual_updates=[dict(step=1)]), lambda o:o.update(actual_behavior_gates=[{}]),
                     lambda o:o['child_actual_unstarted'].update(optimizer_evidence_paths=['optimizer-rank-0.jsonl']),
                     lambda o:o['child_actual_unstarted'].update(checkpoint_entries=['actor']),
                     lambda o:o['child_actual_unstarted'].update(training_claim_exists=False),
                     lambda o:o['child_actual_unstarted'].update(segment_exists=True),
                     lambda o:o['cleanup-status.json'].update(cleaned=False),
                     lambda o:o['process-ledger.json']['jobs']['trainer'].update(exit_code=0),
                     lambda o:o['process-ledger.json']['jobs']['trainer'].update(state='RUNNING'),
                     lambda o:o['parent_actual_unstarted'].update(training_claim_exists=True),
                     lambda o:o.update(actual_namespace_processes=[dict(pid=1)]),
                     lambda o:o['run-manifest.json'].update(completed_updates=True)]
        for change in mutations:
            bad=copy.deepcopy(original);change(bad)
            with self.assertRaises(ValueError):v.previous_schema(bad,old)

    def test_cpu_count_modules_map_time_flags_and_old_proof_drift_reject(self):
        _, old, _=v.original_inputs(); proof=cpu_unit_fixture(old)
        self.assertEqual(v.validate_cpu(proof,old,223),proof['source_sha256'])
        for count in (None, True, 0, -1, 222):
            with self.assertRaises(ValueError):v.validate_cpu(proof,old,count)
        mutations=[lambda p:p.update(source_sha256=old['source_sha256']),
                   lambda p:p.update(checked_unix=old['checked_unix']), lambda p:p['modules'].pop(),
                   lambda p:p['modules'].reverse(), lambda p:p['source_sha256'].pop(next(iter(p['source_sha256']))),
                   lambda p:p.update(returncode=True), lambda p:p.update(base='/local/fixture')]
        for field,value in [('skipped',['unit']),('failures',1),('errors',1),('successful',False),
                            ('cuda_available',True),('tests_run',True)]:
            mutations.append(lambda p,field=field,value=value:p.update(stdout=json.dumps(v.common.strict_json(p['stdout'])|{field:value})))
        for change in mutations:
            bad=copy.deepcopy(proof);change(bad)
            with self.assertRaises(ValueError):v.validate_cpu(bad,old,223)

    def test_explicit_new_commit_and_missing_input_fail_before_any_remote_or_claim(self):
        for commit in (None,'HEAD','a09a553',v.OLD_COMMIT,'130d17a0d6f0a3ab4b4aee730928f461e25105f0'):
            with self.assertRaises(ValueError):v.check_commit(commit)
        with tempfile.TemporaryDirectory() as directory:
            prep=Path(directory)
            for args in ((),('a'*40,),('a'*40,prep/'missing'),('a'*40,None,223)):
                with patch.object(v.common,'remote') as remote, self.assertRaises(ValueError):v.submit(*args,prep=prep)
                remote.assert_not_called();self.assertFalse((prep/v.NAMES['intent']).exists())

    def test_real_temporary_git353_blobs_and_exact_map_are_read(self):
        # Real Git objects in an explicitly tiny CPU source-reader repository.
        with tempfile.TemporaryDirectory() as directory:
            repo=Path(directory)
            for i in range(351):(repo/f'f{i:03}.py').write_text('# CPU source reader fixture '+str(i)+'\n')
            for name in ('verl/version/version','verl/single_controller/version/version'):
                path=repo/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('CPU-fixture\n')
            (repo/'ignored.md').write_text('not runtime\n')
            def git(*args):return subprocess.check_output(['git',*args],cwd=repo,stderr=subprocess.DEVNULL,text=True).strip()
            git('init');git('add','.');git('-c','user.name=CPU fixture','-c','user.email=fixture@example.invalid','commit','-m','CPU only source reader')
            commit=git('rev-parse','HEAD')
            mapping={str(p.relative_to(repo)):v.common.sha(p.read_bytes()) for p in repo.rglob('*') if p.is_file() and
                     (p.suffix=='.py' or str(p.relative_to(repo)) in ('verl/version/version','verl/single_controller/version/version'))}
            files=v.committed_sources(commit,mapping,repo)
            self.assertEqual(len(files),353)
            self.assertEqual(set(files),set(mapping))
            for bad in (mapping|{'extra.py':'0'*64},mapping|{'f000.py':'0'*64},dict(list(mapping.items())[1:])):
                with self.assertRaises(ValueError):v.committed_sources(commit,bad,repo)

    def test_actual_old_cpu_bytes_cannot_be_new_source_even_when_count_matches(self):
        with patch.object(v,'committed_sources') as git, patch.object(v.common,'remote') as remote:
            with self.assertRaises(ValueError):v.load_inputs('a'*40,v.PREP/v.CPU_OLD_NAME,222)
            git.assert_not_called();remote.assert_not_called()

    def test_actual_root_approved_new226_bytes_and_git353_bind_without_remote(self):
        commit='3fdf92f71cf624f78611cf5ab5f03dba64ce4fec'
        path=v.PREP/'behavior-consistency-target-cpu-v2-result.json'
        self.assertEqual(v.common.sha(path.read_bytes()),
                         '5399795ab3c6b50400d378493aecede6307663672e32ebc38a13fc201ac64e35')
        with patch.object(v.common,'remote') as remote:
            inputs,files=v.load_inputs(commit,path,226)
            remote.assert_not_called()
        self.assertEqual(inputs['source_commit'],commit)
        self.assertEqual(inputs['actual_target_cpu_tests'],226)
        self.assertEqual(len(files),353)
        self.assertEqual(v.common.sha(v.common.canonical(inputs['source_sha256'])),
                         '91174372029ae4be7d82195de95158261b7a56ae9fd1b440cd63b00141cea25f')
        self.assertEqual(inputs['previous_terminal']['slurm_terminal_state'],'FAILED')
        self.assertFalse(inputs['previous_terminal']['production_admission'])

    def test_partial_intent_is_never_replayed(self):
        with tempfile.TemporaryDirectory() as directory:
            prep=Path(directory);path=prep/v.NAMES['intent'];path.write_bytes(b'{partial')
            with patch.object(v,'load_inputs') as load,patch.object(v.common,'remote') as remote:
                with self.assertRaises(ValueError):v.submit('a'*40,'unused',223,prep=prep)
                load.assert_not_called();remote.assert_not_called();self.assertEqual(path.read_bytes(),b'{partial')

    def test_unknown_copy_retains_one_exclusive_intent_and_no_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            prep=Path(directory);old,_,history=v.original_inputs()
            inputs=dict(old_launch=old,**history,source_commit='a'*40,source_sha256={'cpu-only.py':'b'*64},actual_target_cpu_tests=223)
            files={'reproduction/common8/assets_export.py':b"BASE=Path('/dataset/shaoshitong/g1-common8-original-retrieval-export')"}
            with patch.object(v,'load_inputs',return_value=(inputs,files)),patch.object(v.common,'remote',
                     side_effect=[dict(scope='CPU-only old reread response fixture'),RuntimeError('unknown copy')]) as remote:
                with self.assertRaises(RuntimeError):v.submit('a'*40,'unit',223,prep=prep)
                self.assertEqual(remote.call_count,2);raw=(prep/v.NAMES['intent']).read_bytes()
                with self.assertRaises(ValueError):v.submit('a'*40,'unit',223,prep=prep)
                self.assertEqual(remote.call_count,2);self.assertEqual((prep/v.NAMES['intent']).read_bytes(),raw)

    def test_full_manifest_extra_fields_and_original_byte_hash(self):
        _,_,history=v.original_inputs();observed=v.common.strict_json(base64.b64decode(history['previous_terminal_bytes']))
        # Explicit full-manifest CPU fixture extending the real trimmed fields.
        full=dict(observed['run-manifest.json'],profile='common8',policy_calibration_slurm_job_id=v.OLD_JOB,
                  cpu_fixture_only=True)
        raw=v.common.canonical(full);fixture=copy.deepcopy(observed);fixture['child_manifest_sha256']=v.common.sha(raw)
        self.assertEqual(v.verify_old_manifest(raw,fixture),full)
        with self.assertRaises(ValueError):v.verify_old_manifest(raw+b'\n',fixture)
        for key,value in [('completed_updates',True),('source_commit','0'*40),('policy_calibration_slurm_job_id','3278')]:
            bad=dict(full)|{key:value};content=v.common.canonical(bad);other=copy.deepcopy(fixture)
            other['child_manifest_sha256']=v.common.sha(content)
            if key in other['run-manifest.json']:other['run-manifest.json'][key]=value
            with self.assertRaises(ValueError):v.verify_old_manifest(content,other)

    def test_embedded_code_capture_and_resource_contract_are_exact_v4(self):
        for code in (v.previous_code(),v.PREPARE_CODE,v.SUBMIT_CODE,v.OBSERVE_CODE):compile(code,'<CPU-only-embedded>','exec')
        for token in ('--gres=gpu:h100:8','--mem=512G','--exclusive','--time=08:00:00','search-g1-policy-discovery-v4'):
            self.assertIn(token,v.SUBMIT_CODE)
        self.assertLess(v.SUBMIT_CODE.index('os.fsync(fd)'),v.SUBMIT_CODE.index('r=subprocess.run(args'))
        self.assertTrue(all('v4' in value for value in v.NAMES.values()))
        self.assertIn('source_hashes_match',v.OBSERVE_CODE);self.assertIn('observation_finished_unix',v.OBSERVE_CODE)
        self.assertNotIn('historical_running_actual_workflow_failed',v.previous_code())
        self.assertNotIn('METHOD_SHA',v.__dict__)
        for base in ('/mnt/data/shaoshitong/g1p-12345678','/mnt/data/shaoshitong/g1d-12345678'):
            self.assertLessEqual(v.common.ray_socket_budget(base)['worst_plasma_socket_bytes'],107)
        native=inspect.getsource(v.previous_readonly)
        self.assertIn("['squeue', '-h', '-j', OLD_JOB",native)
        self.assertIn("behavior-probability-outer-*.json",native)
        self.assertLess(native.index('Original complete a09 source file set changed'),
                        native.index('from reproduction.a800.owned_processes'))

    def test_current_common_recipe_keeps_science_and_neutral_sampling(self):
        files={'reproduction/common8/'+name+'.py':b'CPU-only presence fixture' for name in
               ('behavior_trace','behavior_probability','behavior_runtime','generation_behavior')}
        recipe=v.common.strict_json((v.REPO/'reproduction/common8/recipe.json').read_bytes())
        files['reproduction/common8/recipe.json']=v.common.canonical(recipe)
        self.assertEqual(v.new_recipe(files)['actor_rollout_ref.rollout.top_p'],'1.0')
        for bad in (recipe+['actor_rollout_ref.rollout.top_p=0.95'],
                    [x.replace('rollout.top_p=1.0','rollout.top_p=0.95') for x in recipe],
                    [x.replace('train_batch_size=512','train_batch_size=256') for x in recipe]):
            with self.assertRaises(ValueError):v.new_recipe(files|{'reproduction/common8/recipe.json':v.common.canonical(bad)})


if __name__=='__main__':unittest.main()
