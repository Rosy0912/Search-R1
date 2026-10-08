"""Local external v6 protocol tests; no remote, Slurm, GPU or health claim.

3294 terminal/birth/launch and scalar diagnostic are actual saved local bytes.
Future source, CPU proof/count, Hydra descriptor, scheduler query and boundary
responses below are explicitly CPU fixtures, never approved native inputs.
Temporary Git and file tests do not edit the Search-G1 runtime repository.
"""
import base64
import copy
import hashlib
import inspect
import json
import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as S
from unittest.mock import patch

import submit_policy_discovery_v6_native as v


def terminal_arguments():
    return dict(expected_sha256=v.TERMINAL_SHA256,
        launch_bytes=(v.PREP / v.OLD_LAUNCH_NAME).read_bytes(),
        expected_launch_sha256=v.OLD_LAUNCH_SHA256, expected_run_id=v.OLD_RUN,
        expected_parent_run_id=v.OLD_PARENT_RUN, expected_state='FAILED', expected_exit_code='1:0',
        birth_observation_bytes=(v.PREP / v.BIRTH_NAME).read_bytes(), expected_birth_sha256=v.BIRTH_SHA256)


def actual_terminal():
    raw = (v.PREP / v.TERMINAL_NAME).read_bytes()
    return raw, v.read_terminal_observation(raw, **terminal_arguments())


def original_launch():
    return v.common.strict_json((v.PREP / v.OLD_LAUNCH_NAME).read_bytes())


def actual_old_cpu():
    return v.common.strict_json((v.PREP / v.CPU_OLD_NAME).read_bytes())


def future_cpu_fixture(old):
    # Deliberately arbitrary count: actual future count is required from root.
    value = copy.deepcopy(old)
    summary = v.common.strict_json(value['stdout'])
    summary.update(tests_run=999, successful=True, failures=0, errors=0, skipped=[], cuda_available=False)
    mapping = copy.deepcopy(old['source_sha256']); mapping[next(iter(mapping))] = 'a' * 64
    mapping['reproduction/common8/sampling_capture.py'] = 'b' * 64
    value.update(stdout=json.dumps(summary), checked_unix=old['checked_unix'] + 1,
                 modules=list(v.CPU_MODULES), source_sha256=mapping)
    return value


def source_set_fixture():
    # Unknown future old-snapshot descriptor is never guessed from this fixture.
    folder = 'outputs/2026-10-06/12-34-56/.hydra/'
    rows = [dict(path=folder + name, bytes=3, sha256='a' * 64, mtime=10.0, symlink=False)
            for name in ('config.yaml', 'hydra.yaml', 'overrides.yaml')]
    return dict(checked_unix=20.0, namespaces=[
        dict(base=v.OLD_BASE, expected_count=353, actual_count=356, extra=rows, missing=[], changed=[]),
        dict(base=v.OLD_PARENT_BASE, expected_count=353, actual_count=353, extra=[], missing=[], changed=[])])


def real_mapping(namespace):
    repo = namespace / 'repo'; repo.mkdir(parents=True)
    for index in range(351):
        (repo / ('f%03d.py' % index)).write_bytes(b'' if index == 0 else ('# CPU fixture %s\n' % index).encode())
    for name in ('verl/version/version', 'verl/single_controller/version/version'):
        path = repo / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'CPU-fixture\n')
    return {str(path.relative_to(repo)): v.common.sha(path.read_bytes()) for path in repo.rglob('*') if path.is_file()}


def new_mapping(namespace):
    mapping = real_mapping(namespace)
    path = namespace / 'repo/reproduction/common8/sampling_capture.py'
    path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'# CPU-only new capture fixture\n')
    mapping[str(path.relative_to(namespace / 'repo'))] = v.common.sha(path.read_bytes())
    return mapping


def metadata(path):
    value = path.stat()
    return dict(size_bytes=value.st_size, mtime_ns=value.st_mtime_ns,
                mode=oct(stat.S_IMODE(value.st_mode)), sha256=v.common.sha(path.read_bytes()))


class TerminalReaderTests(unittest.TestCase):
    def test_actual_3294_failed_bytes_never_completed(self):
        raw, terminal = actual_terminal()
        self.assertEqual(v.common.sha(raw), v.TERMINAL_SHA256)
        self.assertEqual((terminal['job_id'], terminal['slurm_terminal_state'], terminal['exit_code']), ('3294', 'FAILED', '1:0'))
        self.assertFalse(terminal['production_admission']); self.assertFalse(terminal['healthy_checkpoint_accepted'])
        with self.assertRaises(ValueError):
            v.read_terminal_observation(raw, **(terminal_arguments() | dict(expected_state='COMPLETED', expected_exit_code='0:0')))

    def test_original_hashes_cannot_be_relabelled(self):
        raw, _ = actual_terminal()
        for key, value in [('expected_sha256','0'*64), ('expected_launch_sha256','0'*64),
                           ('expected_birth_sha256','0'*64), ('expected_sha256',True)]:
            with self.assertRaises(ValueError): v.read_terminal_observation(raw, **(terminal_arguments() | {key:value}))
        with self.assertRaises(ValueError): v.read_terminal_observation(raw + b'\n', **terminal_arguments())

    def test_running_purged_exit_zero_cannot_create_terminal(self):
        raw, _ = actual_terminal(); original = v.common.strict_json(raw)
        changes = [lambda o:o.update(job_stdout=o['job_stdout'].replace('JobState=FAILED','JobState=RUNNING')),
                   lambda o:o.update(job_stdout=o['job_stdout'].replace('ExitCode=1:0','ExitCode=0:0')),
                   lambda o:o.update(job_stdout='',job_returncode=1,job_stderr=v.INVALID_JOB),
                   lambda o:o.update(job_returncode=True)]
        for change in changes:
            value = copy.deepcopy(original); change(value); content = v.common.canonical(value)
            with self.assertRaises(ValueError):
                v.read_terminal_observation(content, **(terminal_arguments() | dict(expected_sha256=v.common.sha(content))))
        for state, exit_code in [('RUNNING','0:0'),('CANCELLED','0:15'),('FAILED','0:0'),('COMPLETED','1:0')]:
            with self.assertRaises(ValueError):
                v.read_terminal_observation(raw, **(terminal_arguments() | dict(expected_state=state,expected_exit_code=exit_code)))

    def test_source_command_resource_birth_and_times_drift_reject(self):
        raw, _ = actual_terminal(); original = v.common.strict_json(raw)
        fields = v._job_fields(original['job_stdout'])
        changes = [lambda o:o['run-manifest.json'].update(run_id='0'*32),
                   lambda o:o['run-manifest.json'].update(source_commit='0'*40),
                   lambda o:o.update(source_hashes_match=False), lambda o:o.update(checked_unix=1),
                   lambda o:o.update(observation_finished_unix=o['checked_unix']-1),
                   lambda o:o.update(job_stdout=o['job_stdout']+' JobId=3294')]
        for key, value in [('JobId','3295'),('Command','/wrong'),('WorkDir','/wrong'),('MinMemoryNode','256G'),
                           ('TresPerNode','gres:gpu:h100:4'),('NodeList','h100-9'),('Restarts','1'),
                           ('SubmitTime','2026-10-01T00:00:00'),('EndTime','2026-10-06T00:00:00')]:
            changes.append(lambda o,key=key,value=value:o.update(job_stdout=o['job_stdout'].replace(key+'='+fields[key],key+'='+value)))
        for change in changes:
            value=copy.deepcopy(original); change(value); content=v.common.canonical(value)
            with self.assertRaises(ValueError):
                v.read_terminal_observation(content, **(terminal_arguments() | dict(expected_sha256=v.common.sha(content))))

    def test_current_matching_record_or_exact_purge_preserves_only_saved_failure(self):
        raw, terminal=actual_terminal(); original=v.common.strict_json(raw)
        record=S(returncode=0,stdout=original['job_stdout'],stderr='')
        self.assertIn('current_terminal_matches',v.verify_terminal_current(terminal,record)['route'])
        missing=S(returncode=1,stdout='',stderr=v.INVALID_JOB+'\n')
        for queue in [S(returncode=0,stdout='',stderr=''),missing]:
            out=v.verify_terminal_current(terminal,missing,queue)
            self.assertIn('original_actual_terminal_bytes',out['route'])
            self.assertEqual((out['slurm_terminal_state'],out['exit_code']),('FAILED','1:0'))
        for queue in [None,S(returncode=0,stdout='3294|RUNNING',stderr=''),
                      S(returncode=1,stdout='',stderr='connection refused')]:
            with self.assertRaises(ValueError):v.verify_terminal_current(terminal,missing,queue)
        for query in [S(returncode=True,stdout='',stderr=v.INVALID_JOB),
                      S(returncode=1,stdout='',stderr='connection refused'),
                      S(returncode=0,stdout=original['job_stdout'].replace('Restarts=0','Restarts=1'),stderr='')]:
            with self.assertRaises(ValueError):v.verify_terminal_current(terminal,query,S(returncode=0,stdout='',stderr=''))

    def test_reader_body_is_frozen_v4_pure_reader(self):
        import submit_policy_discovery_v4_hydra_native as original
        for name in ('_job_fields','_scheduler_time','_snapshot_binding','read_terminal_observation','verify_terminal_current'):
            self.assertEqual(inspect.getsource(getattr(v,name)),inspect.getsource(getattr(original,name)))


class PreviousFailureTests(unittest.TestCase):
    def test_actual_failure_has_attempt_claim_without_recorded_writes(self):
        raw,_=actual_terminal(); observed=v.common.strict_json(raw)
        v.previous_schema(observed,original_launch())
        self.assertTrue(observed['child_actual_unstarted']['training_claim_exists'])
        self.assertFalse(observed['child_actual_unstarted_verified'])
        self.assertEqual(observed['actual_updates'],[])
        self.assertEqual(observed['child_actual_unstarted']['optimizer_evidence_paths'],[])
        self.assertEqual(v.common.strict_json(actual_old_cpu()['stdout'])['tests_run'],239)

    def test_recorded_writes_checkpoint_claim_cleanup_or_parent_drift_reject(self):
        raw,_=actual_terminal(); original=v.common.strict_json(raw); old=original_launch()
        changes=[lambda o:o.update(actual_updates=[dict(step=1)]),lambda o:o.update(actual_behavior_gates=[{}]),
                 lambda o:o['child_actual_unstarted'].update(optimizer_evidence_paths=['optimizer-rank-0.jsonl']),
                 lambda o:o['child_actual_unstarted'].update(checkpoint_entries=['actor']),
                 lambda o:o['child_actual_unstarted'].update(training_claim_exists=False),
                 lambda o:o.update(child_actual_unstarted_verified=True),
                 lambda o:o['cleanup-status.json'].update(cleaned=False),
                 lambda o:o['process-ledger.json']['jobs']['trainer'].update(exit_code=0),
                 lambda o:o['parent_actual_unstarted'].update(training_claim_exists=True),
                 lambda o:o.update(actual_namespace_processes=[dict(pid=1)]),
                 lambda o:o['run-manifest.json'].update(completed_updates=True),
                 lambda o:o.update(**{'trainer.log_tail':'old-policy KL failure'})]
        for change in changes:
            value=copy.deepcopy(original);change(value)
            with self.assertRaises(ValueError):v.previous_schema(value,old)

    def test_full_manifest_preserves_rawsha_and_saved_subset(self):
        raw,_=actual_terminal(); observed=v.common.strict_json(raw)
        full=dict(observed['run-manifest.json'],profile='common8',policy_calibration_slurm_job_id=v.OLD_JOB,cpu_fixture_only=True)
        content=v.common.canonical(full); fixture=copy.deepcopy(observed);fixture['child_manifest_sha256']=v.common.sha(content)
        self.assertEqual(v.verify_old_manifest(content,fixture),full)
        with self.assertRaises(ValueError):v.verify_old_manifest(content+b'\n',fixture)
        for key,value in [('completed_updates',True),('source_commit','0'*40),('policy_calibration_slurm_job_id','3278')]:
            changed=full|{key:value};newbytes=v.common.canonical(changed);other=copy.deepcopy(fixture)
            other['child_manifest_sha256']=v.common.sha(newbytes)
            if key in other['run-manifest.json']:other['run-manifest.json'][key]=value
            with self.assertRaises(ValueError):v.verify_old_manifest(newbytes,other)


class ExactSourceAndMetadataTests(unittest.TestCase):
    def test_explicit_sealed_exactthree_descriptor(self):
        doc=source_set_fixture();raw=v.common.canonical(doc)
        self.assertEqual(v.read_source_set_observation(raw,v.common.sha(raw)),doc)
        for digest in [None,True,'0'*64]:
            with self.assertRaises(ValueError):v.read_source_set_observation(raw,digest)
        for change in [lambda o:o['namespaces'][0]['extra'].append(copy.deepcopy(o['namespaces'][0]['extra'][0])),
                       lambda o:o['namespaces'][0]['extra'].pop(),
                       lambda o:o['namespaces'][0]['extra'][0].update(path='outputs/arbitrary.py'),
                       lambda o:o['namespaces'][0]['extra'][0].update(path='outputs/2026-10-06/12-34-57/.hydra/config.yaml'),
                       lambda o:o['namespaces'][0]['extra'][0].update(bytes=True),
                       lambda o:o['namespaces'][0]['extra'][0].update(bytes=131073),
                       lambda o:o['namespaces'][0]['extra'][0].update(sha256='unknown'),
                       lambda o:o['namespaces'][0]['extra'][0].update(symlink=True),
                       lambda o:o['namespaces'][0].update(missing=['f000.py']),
                       lambda o:o['namespaces'][1].update(actual_count=356),
                       lambda o:o['namespaces'][0]['extra'][0].update(mtime=21.0)]:
            value=copy.deepcopy(doc);change(value)
            with self.assertRaises(ValueError):v.source_set_schema(value)
        self.assertNotIn('SOURCE_SET_EXTRA_PATHS',v.__dict__)
        self.assertNotIn('SOURCE_SET_SHA256',v.__dict__)

    def test_real_declared353_and_empty_source_file(self):
        with tempfile.TemporaryDirectory() as directory:
            namespace=Path(directory).resolve();mapping=real_mapping(namespace)
            self.assertEqual(len(mapping),353);v._verify_original_mapping(namespace,mapping)
            (namespace/'repo/f000.py').write_bytes(b'changed')
            with self.assertRaises(ValueError):v._verify_original_mapping(namespace,mapping)

    def test_real_exact_three_generated_bytes_not_blanket_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            namespace=Path(directory).resolve();mapping=real_mapping(namespace);repo=namespace/'repo'
            extras=[]
            for name in ('config.yaml','hydra.yaml','overrides.yaml'):
                path=repo/'outputs/2026-10-06/12-34-56/.hydra'/name
                path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'fixture\n')
                extras.append(dict(path=str(path.relative_to(repo)),bytes=path.stat().st_size,
                    sha256=v.common.sha(path.read_bytes()),mtime=path.stat().st_mtime,symlink=False))
            captured=dict(base=str(namespace),expected_count=353,actual_count=356,extra=extras)
            v._verify_original_source_set(namespace,mapping,captured)
            for name in ('outputs/unknown.py','outputs/elsewhere/.hydra/config.yaml'):
                extra=repo/name;extra.parent.mkdir(parents=True,exist_ok=True);extra.write_bytes(b'unknown')
                with self.assertRaises(ValueError):v._verify_original_source_set(namespace,mapping,captured)
                extra.unlink()
            path=repo/extras[0]['path'];path.write_bytes(b'changed\n')
            with self.assertRaises(ValueError):v._verify_original_source_set(namespace,mapping,captured)

    def test_generated_symlink_and_missing_file_reject(self):
        with tempfile.TemporaryDirectory() as directory:
            namespace=Path(directory).resolve();mapping=real_mapping(namespace)
            path=namespace/'repo/outputs/2026-10-06/12-34-56/.hydra/config.yaml'
            path.parent.mkdir(parents=True);path.symlink_to(namespace/'repo/f001.py')
            captured=dict(base=str(namespace),expected_count=353,actual_count=354,
                extra=[dict(path=str(path.relative_to(namespace/'repo')),bytes=path.stat().st_size,
                            sha256=v.common.sha(path.read_bytes()),mtime=path.stat().st_mtime,symlink=False)])
            with self.assertRaises(ValueError):v._verify_original_source_set(namespace,mapping,captured)
            path.unlink()
            with self.assertRaises(ValueError):v._verify_original_source_set(namespace,mapping,captured)

    def test_private_raw_fixture_is_streamed_no_path_read_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory).resolve()/'private.raw';path.write_bytes(b'CPU-private-journal\x00'*70000);path.chmod(0o600)
            sealed=metadata(path)
            with patch.object(Path,'read_bytes',side_effect=AssertionError('Do not load raw private bytes')):
                v._verify_metadata_file(path,sealed)
            for key,value in [('size_bytes',sealed['size_bytes']+1),('mode','0o644'),('sha256','0'*64),('mtime_ns',0)]:
                with self.assertRaises(ValueError):v._verify_metadata_file(path,sealed|{key:value})
            linked=path.parent/'linked';linked.symlink_to(path)
            with self.assertRaises(ValueError):v._verify_metadata_file(linked,sealed)



class FutureInputAndOneSubmitTests(unittest.TestCase):
    def test_future_count_flags_source_map_modules_and_time_are_explicit(self):
        old=actual_old_cpu();proof=future_cpu_fixture(old)
        self.assertEqual(v.validate_cpu(proof,old,999),proof['source_sha256'])
        for count in [None,True,0,-1,239,998]:
            with self.assertRaises(ValueError):v.validate_cpu(proof,old,count)
        changes=[lambda o:o.update(source_sha256=old['source_sha256']),lambda o:o.update(checked_unix=old['checked_unix']),
                 lambda o:o['modules'].reverse(),lambda o:o['source_sha256'].pop(next(iter(o['source_sha256']))),
                 lambda o:o.update(returncode=True),lambda o:o.update(base='/local/fixture')]
        for field,value in [('skipped',['unit']),('failures',1),('errors',1),('successful',False),('cuda_available',True),('tests_run',True)]:
            changes.append(lambda o,field=field,value=value:o.update(stdout=json.dumps(v.common.strict_json(o['stdout'])|{field:value})))
        for change in changes:
            value=copy.deepcopy(proof);change(value)
            with self.assertRaises(ValueError):v.validate_cpu(value,old,999)

    def test_no_future_identity_defaults_and_missing_inputs_reject_before_remote(self):
        for commit in [None,'HEAD','3fdf',v.OLD_COMMIT,'130d17a0d6f0a3ab4b4aee730928f461e25105f0']:
            with self.assertRaises(ValueError):v.check_commit(commit)
        with tempfile.TemporaryDirectory() as directory:
            prep=Path(directory)
            for args in [(),('a'*40,),('a'*40,prep/'missing'),('a'*40,None,999)]:
                with patch.object(v.common,'remote') as remote,patch.object(v,'original_inputs') as original:
                    with self.assertRaises(ValueError):v.submit(*args,prep=prep)
                    remote.assert_not_called();original.assert_not_called();self.assertFalse((prep/v.NAMES['intent']).exists())

    def test_missing_each_explicit_sha_path_count_precondition_rejects(self):
        kwargs=dict(target_cpu_proof_sha256='a'*64,previous_source_set_path='fixture-source',
            previous_source_set_sha256='b'*64,previous_source_cleanup_path='fixture-cleanup',previous_source_cleanup_sha256='c'*64)
        for key in kwargs:
            with patch.object(v,'original_inputs') as original,patch.object(v.common,'remote') as remote:
                with self.assertRaises(ValueError):v.load_inputs('a'*40,'fixture-cpu',999,**(kwargs|{key:None}))
                original.assert_not_called();remote.assert_not_called()

    def test_actual_old_cpu_cannot_admit_even_when_given_actual239(self):
        old=original_launch();cpu=actual_old_cpu();source=source_set_fixture();source['checked_unix']=1792000000
        source_raw=v.common.canonical(source)
        history=dict(previous_terminal=dict(checked_unix=1791250913.0))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'source.json';path.write_bytes(source_raw)
            with patch.object(v,'original_inputs',return_value=(old,cpu,history)),patch.object(v,'committed_sources') as git:
                with self.assertRaises(ValueError):v.load_inputs('a'*40,v.PREP/v.CPU_OLD_NAME,239,
                    target_cpu_proof_sha256=old['target_cpu_proof_sha256'],previous_source_set_path=path,
                    previous_source_set_sha256=v.common.sha(source_raw),previous_source_cleanup_path='fixture',
                    previous_source_cleanup_sha256='c'*64)
                git.assert_not_called()

    def test_real_temporary_git353_blobs_bind_explicit_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            namespace=Path(directory).resolve();mapping=new_mapping(namespace);repo=namespace/'repo'
            (repo/'ignored.md').write_bytes(b'not runtime\n')
            def git(*args):return subprocess.check_output(['git',*args],cwd=repo,stderr=subprocess.DEVNULL,text=True).strip()
            git('init');git('add','.');git('-c','user.name=CPU fixture','-c','user.email=fixture@example.invalid','commit','-m','CPU-only source reader')
            commit=git('rev-parse','HEAD');files=v.committed_sources(commit,mapping,repo)
            self.assertEqual(set(files),set(mapping));self.assertEqual(len(files),354)
            for bad in [mapping|{'extra.py':'0'*64},mapping|{'f001.py':'0'*64},dict(list(mapping.items())[1:])]:
                with self.assertRaises(ValueError):v.committed_sources(commit,bad,repo)

    def test_repeated_partial_intent_does_not_read_new_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            prep=Path(directory);path=prep/v.NAMES['intent'];path.write_bytes(b'{partial')
            with patch.object(v,'load_inputs') as load,patch.object(v.common,'remote') as remote:
                with self.assertRaises(ValueError):v.submit('a'*40,'fixture',999,prep=prep)
                load.assert_not_called();remote.assert_not_called();self.assertEqual(path.read_bytes(),b'{partial')

    def test_old_readonly_failure_precedes_all_local_intents(self):
        inputs=dict(old_launch=original_launch(),source_commit='a'*40,source_sha256={'fixture.py':'b'*64},actual_target_cpu_tests=999)
        files={'reproduction/common8/assets_export.py':b"BASE=Path('/dataset/shaoshitong/g1-common8-original-retrieval-export')"}
        with tempfile.TemporaryDirectory() as directory:
            prep=Path(directory)
            with patch.object(v,'load_inputs',return_value=(inputs,files)),patch.object(v.common,'remote',side_effect=ValueError('old cleanup unknown')) as remote:
                with self.assertRaises(ValueError):v.submit('a'*40,'fixture',999,prep=prep)
                self.assertEqual(remote.call_count,1);self.assertEqual(list(prep.iterdir()),[])

    def test_unknown_submission_preserves_intent_and_never_retries(self):
        # Mock only orchestration boundaries. Replies are CPU control fixtures;
        # they do not stand in for source/GPU/health authentication.
        inputs=dict(old_launch=original_launch(),source_commit='a'*40,source_sha256={'fixture.py':'b'*64},
            actual_target_cpu_tests=999,target_cpu_proof_sha256='c'*64)
        files={'reproduction/common8/assets_export.py':b"BASE=Path('/dataset/shaoshitong/g1-common8-original-retrieval-export')"}
        replies=[dict(scope='CPU old readonly fixture'),dict(scope='CPU prepare fixture'),
                 dict(scope='CPU source handle fixture'),dict(scope='CPU second old readonly fixture'),RuntimeError('unknown sbatch outcome')]
        with tempfile.TemporaryDirectory() as directory:
            prep=Path(directory)
            with patch.object(v,'load_inputs',return_value=(inputs,files)),patch.object(v.common,'remote',side_effect=replies) as remote:
                with self.assertRaises(RuntimeError):v.submit('a'*40,'fixture',999,prep=prep)
                self.assertEqual(remote.call_count,5);raw=(prep/v.NAMES['intent']).read_bytes()
                self.assertTrue((prep/v.NAMES['package']).exists());self.assertTrue((prep/v.NAMES['handle']).exists())
                self.assertFalse((prep/v.NAMES['launch']).exists())
                with self.assertRaises(ValueError):v.submit('a'*40,'fixture',999,prep=prep)
                self.assertEqual(remote.call_count,5);self.assertEqual((prep/v.NAMES['intent']).read_bytes(),raw)

    def test_embedded_compile_new_namespace_single_native_claim_and_ray_boundary(self):
        for code in [v.previous_code(),v.PREPARE_CODE,v.SUBMIT_CODE,v.OBSERVE_CODE]:compile(code,'<CPU-only-embedded>','exec')
        for token in ['--gres=gpu:h100:8','--mem=512G','--exclusive','--time=08:00:00','search-g1-policy-discovery-v6']:
            self.assertIn(token,v.SUBMIT_CODE)
        self.assertLess(v.SUBMIT_CODE.index('os.fsync(fd)'),v.SUBMIT_CODE.index('r=subprocess.run(args'))
        self.assertEqual(v.SUBMIT_CODE.count('r=subprocess.run(args'),1)
        self.assertTrue(all('v6' in value for value in v.NAMES.values()))
        for name in v.NAMES.values():self.assertFalse((v.PREP/name).exists())
        for base in ['/mnt/data/shaoshitong/g1p-12345678','/mnt/data/shaoshitong/g1d-12345678']:
            self.assertEqual(v.common.ray_socket_budget(base)['worst_plasma_socket_bytes'],105)
        with self.assertRaises(ValueError):v.common.ray_socket_budget('/mnt/data/shaoshitong/g1d-123456789abc')
        self.assertIn('Node._prepare_socket_file',v.PREPARE_CODE)
        self.assertIn('installed_limit_tested_bytes=[107,108]',v.PREPARE_CODE)
        src=inspect.getsource(v.previous_readonly)
        self.assertLess(src.index('_verify_original_mapping(namespace'),src.index('from reproduction.a800.owned_processes'))
        self.assertIn('recorded_optimizer_write_marker_count=0',src)
        self.assertNotIn('full_checkpoint_verification',src)

    def test_scientific_recipe_and_neutral_sampling_remain_literal(self):
        files={'reproduction/common8/'+name+'.py':b'CPU-only presence fixture' for name in
            ['sampling_capture','behavior_trace','behavior_probability','behavior_runtime','generation_behavior']}
        recipe=v.common.strict_json((v.REPO/'reproduction/common8/recipe.json').read_bytes())
        files['reproduction/common8/recipe.json']=v.common.canonical(recipe)
        self.assertEqual(v.new_recipe(files)['actor_rollout_ref.rollout.top_p'],'1.0')
        for altered in [recipe+['actor_rollout_ref.rollout.top_p=1.0'],
                        [x.replace('rollout.top_p=1.0','rollout.top_p=0.95') for x in recipe],
                        [x.replace('train_batch_size=512','train_batch_size=256') for x in recipe]]:
            with self.assertRaises(ValueError):v.new_recipe(files|{'reproduction/common8/recipe.json':v.common.canonical(altered)})


if __name__ == '__main__':
    unittest.main()
