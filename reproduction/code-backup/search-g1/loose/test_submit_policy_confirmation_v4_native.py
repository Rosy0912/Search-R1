"""Stdlib CPU protocol fixtures, never real completion or health admission.

Actual CPU226 original bytes are read. Future method/job/COMPLETED snapshots
below are fabricated unit fixtures only; no candidate/intent/service is created.
"""
import copy
import importlib.util
import inspect
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as S
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('confirmation_v4',HERE/'submit_policy_confirmation_v4_native.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
import submit_policy_discovery_v4_native as d


def fixture():
    prior=m.strict_json((HERE/'policy-calibration-v3-selection-method.json').read_bytes())
    cpu=m.strict_json((HERE/m.CPU_NAME).read_bytes())
    base,parent='/mnt/data/shaoshitong/g1d-cpuv4abc','/mnt/data/shaoshitong/g1p-cpuv4abc'
    child_id,parent_id='1'*32,'2'*32;job='98765'
    output=base+'/logs/native-discovery-v4-%j.log'
    launch=dict(base=base,parent_base=parent,job_id=job,source_commit=m.FROZEN_COMMIT,
        source_sha256=cpu['source_sha256'],target_cpu_proof_sha256=m.CPU_PROOF_SHA256,
        returncode=0,stdout=job+'\n',stderr='',submitted_unix=1000.,actual_production_updates=0,
        argv=['sbatch','--chdir='+base+'/repo','--output='+output,base+'/repo/reproduction/common8/launch_calibration_native.sh'])
    launch_raw=m.canonical(launch)
    fields=dict(JobId=job,JobState='RUNNING',ExitCode='0:0',SubmitTime='2026-10-06T00:00:00',
        StartTime='2026-10-06T00:00:01',EndTime='Unknown',NodeList='h100-8',BatchHost='h100-8',
        Command=launch['argv'][-1],WorkDir=base+'/repo',StdOut=output.replace('%j',job),
        StdErr=output.replace('%j',job),Restarts='0',UserId='cpu-fixture(0)',NumNodes='1',
        MinMemoryNode='512G',TresPerNode='gres:gpu:h100:8')
    stdout=lambda f:' '.join(key+'='+value for key,value in f.items())+'\n'
    empty=dict(training_claim_exists=False,optimizer_evidence_paths=[],train_log_exists=False,
        checkpoint_entries=[],actual_updates_exists=False,segment_exists=False,segment_complete_exists=False)
    observed=dict(base=base,parent_base=parent,job_id=job,job_returncode=0,job_stderr='',job_stdout=stdout(fields),
        actual_production_updates=0,actual_updates=[],source_hashes_match=True,checked_unix=1100.,observation_finished_unix=1101.,
        child_manifest_slurm_job_id=job,child_manifest_sha256='a'*64,parent_actual_unstarted_verified=True,
        child_actual_unstarted_verified=True,parent_actual_unstarted=empty,child_actual_unstarted=empty,
        parent_run_manifest=dict(run_id=parent_id,source_commit=m.FROZEN_COMMIT,completed_updates=0,manifest_sha256='b'*64),
        **{'run-manifest.json':dict(run_id=child_id,parent_production_run_id=parent_id,source_commit=m.FROZEN_COMMIT,
             completed_updates=0,policy_calibration_role=m.DISCOVERY_ROLE)})
    observed_raw=m.canonical(observed)
    terminal=copy.deepcopy(observed);terminal.update(checked_unix=1300.,observation_finished_unix=1301.)
    final=dict(fields,JobState='COMPLETED',ExitCode='0:0',EndTime='2026-10-06T00:10:00')
    terminal['job_stdout']=stdout(final);terminal_raw=m.canonical(terminal)
    prereg=dict(role='root_policy_calibration_method_preregistration_before_native_submission',source_commit=m.FROZEN_COMMIT,
        created_unix=900.,actual_discovery_observations_for_this_source=0,production_admission=False,
        healthy_checkpoint_accepted=False,target_cpu_proof_sha256=m.CPU_PROOF_SHA256,
        **{key:prior[key] for key in ('health_rule_document','negative_tail_selection','fixed_candidates','negative_tail_limit','sampling_distribution')})
    prereg_raw=m.canonical(prereg)
    method=copy.deepcopy(prior);method.update(source_commit=m.FROZEN_COMMIT,parent_production_run_id=parent_id,
        discovery_run_id=child_id,frozen_unix=1200.,prospective_binding=dict(native_job_id=job,discovery_base=base,parent_base=parent,
        child_manifest_sha256='a'*64,parent_manifest_sha256='b'*64,zero_update_observation_sha256=m.sha(observed_raw),
        launch_sha256=m.sha(launch_raw),preregistration_sha256=m.sha(prereg_raw),preregistered_unix=900.,
        parent_actual_unstarted_verified=True,child_actual_unstarted_verified=True))
    method_raw=m.canonical(method)
    freeze=lambda doc:dict(document=doc,sha256=m.sha(m.canonical(doc)))
    selection=dict(schema_version=1,role='policy_safety_confirmation_orchestration_selection',selected_by='root_explicit_selection',
        selection_provenance={'scope':'fabricated CPU fixture only'},parent_base=parent,discovery_base=base,
        parent_production_run_id=parent_id,discovery_run_id=child_id,discovery_slurm_job_id=job,source_commit=m.FROZEN_COMMIT,
        source_map_sha256=m.SOURCE_MAP_SHA256,scientific_identity_sha256='c'*64,discovery_manifest_sha256='d'*64,
        discovery_plan_file_sha256='e'*64,discovery_plan_sha256='f'*64,discovery_evidence_sha256='a'*64,
        discovery_terminal_observation_sha256=m.sha(terminal_raw),selection_method_sha256=m.sha(method_raw),
        negative_tail_limit=.25,health_rule=freeze(method['health_rule_document']))
    selection['threshold_selection']=freeze(dict(role='policy_safety_threshold_selection',provenance={'scope':'CPU fixture only'},
        criteria=dict(discovery_evidence_sha256=selection['discovery_evidence_sha256'],health_rule_sha256=selection['health_rule']['sha256'],
        candidate_thresholds=dict(method['fixed_candidates'],negative_tail_limit=.25))))
    return S(selection=selection,launch=launch,launch_raw=launch_raw,observed_raw=observed_raw,terminal_raw=terminal_raw,
        method=method,method_raw=method_raw,prereg_raw=prereg_raw,cpu=cpu)


class ConfirmationV4Tests(unittest.TestCase):
    def setUp(self):
        self.f=fixture()
        self.sha_patch=patch.object(m,'FROZEN_METHOD_SHA256',m.sha(self.f.method_raw));self.sha_patch.start()
        self.addCleanup(self.sha_patch.stop)

    def terminal(self):
        f=self.f
        return m.completed_terminal_binding(f.selection,f.launch_raw,f.observed_raw,f.terminal_raw)

    def test_exact_frozen_terminal_helper_source_reused(self):
        self.assertEqual(m.sha(Path(d.__file__).read_bytes()),m.TERMINAL_READER_SOURCE_SHA256)
        for name in ('_digest','_number','_job_fields','_scheduler_time','_snapshot_binding','read_terminal_observation','verify_terminal_current'):
            self.assertEqual(inspect.getsource(getattr(m,name)),inspect.getsource(getattr(d,name)))

    def test_future_fixture_prospective_math_and_prereg_bytes(self):
        f=self.f;m.selection_schema(f.selection,f.launch)
        self.assertEqual(m.method_binding(f.method_raw,f.selection),f.method)
        self.assertEqual(m.prospective_binding(f.method,f.launch_raw,f.observed_raw),f.launch)
        m.preregistration_binding(f.prereg_raw,f.method,f.launch)

    def test_unknown_method_sha_has_no_default(self):
        with self.assertRaises(ValueError):m.configure_contract(m.FROZEN_COMMIT,'not-created','not-created','not-created')
        with patch.object(m,'FROZEN_METHOD_SHA256',None):
            with self.assertRaises(ValueError):m.selection_schema(self.f.selection,self.f.launch)

    def test_explicit_future_method_sha_config_and_actual_cpu_bytes(self):
        f=self.f
        with tempfile.TemporaryDirectory() as directory:
            method=Path(directory)/'CPU-only-method.json';method.write_bytes(f.method_raw)
            with patch.object(m,'METHOD_PATH',None),patch.object(m,'CPU_PATH',None),patch.object(m,'SOURCE_CLEANUP_PATH',None):
                m.configure_contract(m.FROZEN_COMMIT,method,HERE/m.CPU_NAME,Path(directory)/'none',method_sha256=m.sha(f.method_raw))
                with self.assertRaises(ValueError):m.configure_contract(m.FROZEN_COMMIT,method,HERE/m.CPU_NAME,'none',method_sha256='0'*64)
                with self.assertRaises(ValueError):m.configure_contract('a09a5530fd16fa0c04c0aa157ee56462aab825f4',method,HERE/m.CPU_NAME,'none',method_sha256=m.sha(f.method_raw))

    def test_missing_selection_and_existing_intent_no_remote_or_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            prep=Path(directory)
            with patch.object(m,'METHOD_PATH',HERE/m.METHOD_NAME),patch.object(m,'CPU_PATH',HERE/m.CPU_NAME),patch.object(m,'SOURCE_CLEANUP_PATH',prep/'none'),patch.object(m,'remote') as remote:
                with self.assertRaises(FileNotFoundError):m.submit(prep)
                remote.assert_not_called();self.assertEqual(list(prep.iterdir()),[])
                (prep/'policy-confirmation-v4-native-submit-intent.json').write_text('{}')
                with self.assertRaises(ValueError):m.submit(prep)
                remote.assert_not_called()

    def test_actual226_16_353_map_and_original_raw_hash(self):
        self.assertEqual(m.sha((HERE/m.CPU_NAME).read_bytes()),m.CPU_PROOF_SHA256)
        m.cpu_contract(self.f.cpu,self.f.launch)
        self.assertEqual(m.sha(m.canonical(self.f.cpu['source_sha256'])),m.SOURCE_MAP_SHA256)

    def test_actual_committed353_blob_hashes_are_readonly_from_explicit_new_source(self):
        files=m.frozen_sources(self.f.launch)
        self.assertEqual(len(files),353)
        self.assertEqual({name:m.sha(content) for name,content in files.items()},self.f.cpu['source_sha256'])
        self.assertIn('verl/version/version',files)
        self.assertIn('verl/single_controller/version/version',files)

    def test_missing_future_observation_terminal_or_prereg_never_creates_attempt(self):
        f=self.f
        with tempfile.TemporaryDirectory() as directory:
            prep=Path(directory)
            method=prep/m.METHOD_NAME;method.write_bytes(f.method_raw)
            (prep/m.SELECTION_NAME).write_bytes(m.canonical(f.selection))
            (prep/m.DISCOVERY_LAUNCH_NAME).write_bytes(f.launch_raw)
            with patch.object(m,'METHOD_PATH',method),patch.object(m,'CPU_PATH',HERE/m.CPU_NAME),patch.object(m,'SOURCE_CLEANUP_PATH',prep/'none'),patch.object(m,'remote') as remote:
                with self.assertRaises(FileNotFoundError):m.submit(prep)
                (prep/m.ZERO_OBSERVATION_NAME).write_bytes(f.observed_raw)
                with self.assertRaises(FileNotFoundError):m.submit(prep)
                (prep/m.PREREGISTRATION_NAME).write_bytes(f.prereg_raw)
                with self.assertRaises(FileNotFoundError):m.submit(prep)
                remote.assert_not_called()
                self.assertFalse((prep/'policy-confirmation-v4-native-submit-intent.json').exists())

    def test_cpu_old_count_source_or_skip_rejected(self):
        for change in (lambda p:p.update(stdout=json.dumps(dict(m.strict_json(p['stdout']),tests_run=222))),
                       lambda p:p.update(stdout=json.dumps(dict(m.strict_json(p['stdout']),skipped=['CPU fixture']))),
                       lambda p:p['source_sha256'].pop(next(iter(p['source_sha256']))),lambda p:p['modules'].pop()):
            bad=copy.deepcopy(self.f.cpu);change(bad)
            with self.assertRaises(ValueError):m.cpu_contract(bad,self.f.launch)

    def test_terminal_completed_fixture_is_only_terminal_not_health(self):
        result=self.terminal()
        self.assertEqual((result['slurm_terminal_state'],result['exit_code']),('COMPLETED','0:0'))
        self.assertIs(result['production_admission'],False);self.assertIs(result['healthy_checkpoint_accepted'],False)

    def test_real_old_3283_failure_never_admits_confirmation(self):
        with self.assertRaises(ValueError):
            m.completed_terminal_binding(self.f.selection,(HERE/d.OLD_LAUNCH_NAME).read_bytes(),
                (HERE/d.BIRTH_NAME).read_bytes(),(HERE/d.TERMINAL_NAME).read_bytes())
        args=dict(expected_sha256=d.TERMINAL_SHA256,launch_bytes=(HERE/d.OLD_LAUNCH_NAME).read_bytes(),
            expected_launch_sha256=d.OLD_LAUNCH_SHA256,expected_run_id=d.OLD_RUN,expected_parent_run_id=d.OLD_PARENT_RUN,
            expected_state='COMPLETED',expected_exit_code='0:0',birth_observation_bytes=(HERE/d.BIRTH_NAME).read_bytes(),expected_birth_sha256=d.BIRTH_SHA256)
        with self.assertRaises(ValueError):m.read_terminal_observation((HERE/d.TERMINAL_NAME).read_bytes(),**args)

    def test_terminal_raw_sha_running_exitcode_jobbirth_drift_rejects(self):
        f=self.f
        with self.assertRaises(ValueError):m.completed_terminal_binding(f.selection,f.launch_raw,f.observed_raw,f.terminal_raw+b' ')
        for change in (lambda t:t.update(job_stdout=t['job_stdout'].replace('JobState=COMPLETED','JobState=RUNNING')),
                       lambda t:t.update(job_stdout=t['job_stdout'].replace('ExitCode=0:0','ExitCode=1:0')),
                       lambda t:t.update(job_stdout=t['job_stdout'].replace('StartTime=2026-10-06T00:00:01','StartTime=2026-10-06T00:00:02')),
                       lambda t:t['run-manifest.json'].update(run_id='3'*32),
                       lambda t:t.update(job_stdout=t['job_stdout'].replace('MinMemoryNode=512G','MinMemoryNode=64G'))):
            terminal=m.strict_json(f.terminal_raw);change(terminal);raw=m.canonical(terminal)
            selection=dict(f.selection,discovery_terminal_observation_sha256=m.sha(raw))
            with self.assertRaises(ValueError):m.completed_terminal_binding(selection,f.launch_raw,f.observed_raw,raw)

    def test_current_matching_terminal_and_exact_purge_preserve_original_fact(self):
        result=self.terminal();raw=m.strict_json(self.f.terminal_raw)['job_stdout']
        self.assertEqual(m.verify_terminal_current(result,S(returncode=0,stdout=raw,stderr=''))['exit_code'],'0:0')
        purged=S(returncode=1,stdout='',stderr=m.INVALID_JOB+'\n')
        route=m.verify_terminal_current(result,purged,S(returncode=0,stdout='',stderr=''))
        self.assertEqual(route['slurm_terminal_state'],'COMPLETED')
        self.assertEqual(route['original_terminal_sha256'],self.f.selection['discovery_terminal_observation_sha256'])
        with self.assertRaises(ValueError):m.verify_terminal_current(result,purged,S(returncode=0,stdout='98765|RUNNING',stderr=''))
        with self.assertRaises(ValueError):m.verify_terminal_current(result,S(returncode=1,stdout='',stderr='Permission denied'),S(returncode=0,stdout='',stderr=''))

    def test_current_runner_uses_only_saved_job_and_scoped_queue(self):
        result=self.terminal()
        with patch.object(m.subprocess,'run',side_effect=[S(returncode=1,stdout='',stderr=m.INVALID_JOB),S(returncode=0,stdout='',stderr='')]) as run:
            m.current_completed_terminal(result)
            self.assertEqual(run.call_args_list[0].args[0],['scontrol','show','job','-o','98765'])
            self.assertEqual(run.call_args_list[1].args[0][3],'98765')

    def test_both_paths_terminal_and_behavior_link_checks_native_before_heavy(self):
        for fn in (m.source_precheck,m.target_evidence):
            source=inspect.getsource(fn)
            self.assertIn('completed_terminal_binding(selection, launch_bytes, observation_bytes, terminal_bytes)',source)
            self.assertIn('current_completed_terminal(terminal)',source)
            self.assertIn('behavior_links(discovery, run, segment, updates)',source)
        light=inspect.getsource(m.source_precheck)
        self.assertNotIn('verify_child(',light);self.assertNotIn('_checkpoint(',light)
        self.assertIn('full_checkpoint_verification_pending=True',light)
        native=inspect.getsource(m.target_evidence)
        self.assertLess(native.index('native_preparation()'),native.index('verify_child(discovery'))
        bind=inspect.getsource(m.native_bind)
        self.assertLess(bind.index('target_evidence('),bind.index("'preparation/confirmation-bind-claim.json'"))

    def test_embedded_codes_compile_new_paths_no_mac_imports_or_old_ids(self):
        compile(m.verification_code(),'<light>','exec')
        for name in ('PREPARE_CODE','SUBMIT_CODE','START_SOURCE_CODE','OBSERVE_CODE','CLEANUP_SOURCE_CODE'):
            compile(getattr(m,name),'<'+name+'>','exec')
        source=Path(m.__file__).read_text()
        self.assertNotIn('policy-confirmation-v3-',source);self.assertNotIn('3283',source)
        self.assertNotIn('d08276',source);self.assertNotIn('1791245435798697',source)
        self.assertNotIn('import submit_policy_discovery',m.verification_code())
        self.assertEqual(m.ray_socket_budget('/mnt/data/shaoshitong/g1c-12345678')['worst_plasma_socket_bytes'],105)

    def test_complete40_gate_context_max_tail_no_floor(self):
        f=self.f;observations=[]
        for ordinal in range(40):
            mini=ordinal//2
            observations.append(dict(context=dict(logical_update=1,minibatch_index=mini,prior_optimizer_writes=mini,
                behavior_probability_gate_sha256='a'*64),stage=('before_minibatch','after_minibatch')[ordinal%2],
                run_id=f.selection['discovery_run_id'],calibration_sha256=f.selection['discovery_plan_sha256'],
                optimizer_writes=mini+ordinal%2,execution_role=m.DISCOVERY_ROLE,enforcement={'negative_tail':'observe_only'},
                allowed=True,reasons=[],negative_tail_weight_fraction=.25 if ordinal==39 else .1))
        segment=dict(behavior_probability_gate={'proof_sha256':'a'*64},structural=dict(execution_role=m.DISCOVERY_ROLE,
            run_id=f.selection['discovery_run_id'],calibration_sha256=f.selection['discovery_plan_sha256'],outer_count=1,
            optimizer_writes_per_rank=20,enforcement={'negative_tail':'observe_only'},global_monitors=observations))
        self.assertEqual(m.measured_selection(f.selection,segment),.25)
        bad=copy.deepcopy(segment);bad['structural']['global_monitors'][-1]['context']['behavior_probability_gate_sha256']='b'*64
        with self.assertRaises(ValueError):m.measured_selection(f.selection,bad)
        with self.assertRaises(ValueError):m.measured_selection(dict(f.selection,negative_tail_limit=.26),segment)


if __name__=='__main__':unittest.main()
