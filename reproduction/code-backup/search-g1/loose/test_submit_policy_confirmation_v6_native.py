"""Stdlib protocol tests only, never actual completion/GPU/health evidence.

The approved264 CPU proof and pre-submit preregistration are actual original
bytes. All future jobs/methods/COMPLETED/selection/Hydra descriptors below are
explicit synthetic fixtures; no real candidate, intent, service or job is made.
"""
import ast
import copy
import importlib.util
import inspect
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace as S
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module
m = load('confirmation_v6_test', 'submit_policy_confirmation_v6_native.py')
d = load('discovery_v6_reader_test', 'submit_policy_discovery_v6_native.py')
old = load('old_confirmation_v4_hydra_readonly', 'submit_policy_confirmation_v4_hydra_native.py')


def fixture():
    prereg_raw = (HERE / m.PREREGISTRATION_NAME).read_bytes()
    prior = m.strict_json(prereg_raw)
    cpu_raw = (HERE / m.CPU_NAME).read_bytes(); cpu = m.strict_json(cpu_raw)
    stamp = prior['created_unix']
    base, parent = '/mnt/data/shaoshitong/g1d-cpuv6abc', '/mnt/data/shaoshitong/g1p-cpuv6abc'
    child_id, parent_id, job = '1' * 32, '2' * 32, '98765'
    output = base + '/logs/native-discovery-v6-%j.log'
    launch = dict(base=base, parent_base=parent, job_id=job, source_commit=m.FROZEN_COMMIT,
        source_sha256=cpu['source_sha256'], target_cpu_proof_sha256=m.CPU_PROOF_SHA256,
        actual_target_cpu_tests=264, returncode=0, stdout=job+'\n', stderr='', submitted_unix=stamp+100,
        actual_production_updates=0, argv=['sbatch', '--chdir='+base+'/repo', '--output='+output,
                                        base+'/repo/reproduction/common8/launch_calibration_native.sh'])
    launch_raw = m.canonical(launch)
    fields = dict(JobId=job, JobState='RUNNING', ExitCode='0:0', SubmitTime='2026-10-06T10:00:00',
        StartTime='2026-10-06T10:00:01', EndTime='Unknown', NodeList='h100-8', BatchHost='h100-8',
        Command=launch['argv'][-1], WorkDir=base+'/repo', StdOut=output.replace('%j',job),
        StdErr=output.replace('%j',job), Restarts='0', UserId='cpu-fixture(0)', NumNodes='1',
        MinMemoryNode='512G', TresPerNode='gres:gpu:h100:8')
    line = lambda row: ' '.join(key+'='+value for key,value in row.items())+'\n'
    empty = dict(training_claim_exists=False, optimizer_evidence_paths=[], train_log_exists=False,
        checkpoint_entries=[], actual_updates_exists=False, segment_exists=False, segment_complete_exists=False)
    observed = dict(base=base, parent_base=parent, job_id=job, job_returncode=0, job_stderr='',
        job_stdout=line(fields), actual_production_updates=0, actual_updates=[], source_hashes_match=True,
        checked_unix=stamp+200, observation_finished_unix=stamp+201,
        child_manifest_slurm_job_id=job, child_manifest_sha256='a'*64,
        parent_actual_unstarted_verified=True, child_actual_unstarted_verified=True,
        parent_actual_unstarted=empty, child_actual_unstarted=empty,
        parent_run_manifest=dict(run_id=parent_id, source_commit=m.FROZEN_COMMIT, completed_updates=0,
                                 manifest_sha256='b'*64),
        **{'run-manifest.json': dict(run_id=child_id, parent_production_run_id=parent_id,
            source_commit=m.FROZEN_COMMIT, completed_updates=0, policy_calibration_role=m.DISCOVERY_ROLE)})
    observed_raw = m.canonical(observed)
    terminal = copy.deepcopy(observed); terminal.update(checked_unix=stamp+400, observation_finished_unix=stamp+401)
    terminal['job_stdout'] = line(dict(fields, JobState='COMPLETED', ExitCode='0:0', EndTime='2026-10-06T10:10:00'))
    terminal_raw = m.canonical(terminal)
    method = copy.deepcopy(prior)
    method.update(role='root_explicit_policy_calibration_selection_method', source_commit=m.FROZEN_COMMIT,
        parent_production_run_id=parent_id, discovery_run_id=child_id, frozen_unix=stamp+300,
        before_completed_discovery_update=True, before_any_optimizer_write=True,
        prospective_binding=dict(native_job_id=job, discovery_base=base, parent_base=parent,
            child_manifest_sha256='a'*64, parent_manifest_sha256='b'*64,
            zero_update_observation_sha256=m.sha(observed_raw), launch_sha256=m.sha(launch_raw),
            preregistration_sha256=m.sha(prereg_raw), preregistered_unix=stamp,
            target_cpu_proof_sha256=m.CPU_PROOF_SHA256,actual_target_cpu_tests=264,
            parent_actual_unstarted_verified=True, child_actual_unstarted_verified=True))
    method_raw = m.canonical(method)
    freeze = lambda doc: dict(document=doc, sha256=m.sha(m.canonical(doc)))
    selection = dict(schema_version=1, role='policy_safety_confirmation_orchestration_selection',
        selected_by='root_explicit_selection', selection_provenance={'scope':'synthetic CPU protocol only'},
        parent_base=parent, discovery_base=base, parent_production_run_id=parent_id, discovery_run_id=child_id,
        discovery_slurm_job_id=job, source_commit=m.FROZEN_COMMIT, source_map_sha256=m.SOURCE_MAP_SHA256,
        scientific_identity_sha256='c'*64, discovery_manifest_sha256='d'*64, discovery_plan_file_sha256='e'*64,
        discovery_plan_sha256='f'*64, discovery_evidence_sha256='a'*64,
        discovery_terminal_observation_sha256=m.sha(terminal_raw), selection_method_sha256=m.sha(method_raw),
        negative_tail_limit=.25, health_rule=freeze(method['health_rule_document']))
    selection['threshold_selection'] = freeze(dict(role='policy_safety_threshold_selection',
        provenance={'scope':'synthetic CPU protocol only'}, criteria=dict(
            discovery_evidence_sha256=selection['discovery_evidence_sha256'],
            health_rule_sha256=selection['health_rule']['sha256'],
            candidate_thresholds=dict(method['fixed_candidates'], negative_tail_limit=.25))))
    # Synthetic future metadata descriptor, never a generated native path claim.
    selection['discovery_generated_hydra_snapshots'] = dict(schema_version=1, run_id=child_id, slurm_job_id=job,
        source_map_sha256=m.SOURCE_MAP_SHA256, training_launch_claim_sha256='b'*64,
        resolved_training_config_file_sha256='c'*64, resolved_config_sha256='d'*64,
        files={f'outputs/2026-10-06/10-10-10/.hydra/{name}.yaml': dict(size_bytes=42, sha256='e'*64)
               for name in ('config','hydra','overrides')})
    handle = dict(export='/dataset/shaoshitong/g1-original-export-discovery-v6-cpufix00',
                  source_sha256='f'*64, scope='Synthetic source-handle fixture only')
    cleanup = dict(cleaned=True, handle=handle, scope='Synthetic cleanup fixture only')
    blobs = dict(method=method_raw, cpu=cpu_raw, source_cleanup=m.canonical(cleanup),
        source_handle=m.canonical(handle), selection=m.canonical(selection), launch=launch_raw,
        observation=observed_raw, terminal=terminal_raw, preregistration=prereg_raw)
    return S(selection=selection, launch=launch, method=method, method_raw=method_raw, launch_raw=launch_raw,
        observed_raw=observed_raw, terminal_raw=terminal_raw, prereg_raw=prereg_raw, cpu=cpu, blobs=blobs)


def materialize(base, f):
    artifacts = {}
    for role, content in f.blobs.items():
        path = base / (role+'.json'); path.write_bytes(content)
        artifacts[role] = dict(path=str(path), sha256=m.sha(content))
    return artifacts


class ConfirmationV6Tests(unittest.TestCase):
    def setUp(self):
        self.f = fixture()
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.artifacts = materialize(self.base, self.f)
        for key in ('INPUT_PATHS', 'INPUT_SHA256', 'METHOD_PATH', 'CPU_PATH', 'SOURCE_CLEANUP_PATH', 'FROZEN_METHOD_SHA256'):
            context = patch.object(m, key, None); context.start(); self.addCleanup(context.stop)

    def configure(self): return m.configure_contract(m.FROZEN_COMMIT, self.artifacts)
    def terminal(self):
        f = self.f
        return m.completed_terminal_binding(f.selection, f.launch_raw, f.observed_raw, f.terminal_raw)

    def test_actual_new_cpu264_prereg_and_source354_originals(self):
        f = self.f
        self.assertEqual(m.sha(f.blobs['cpu']), m.CPU_PROOF_SHA256)
        self.assertEqual(m.sha(f.prereg_raw), m.PREREGISTRATION_SHA256)
        m.cpu_contract(f.cpu, f.launch)
        self.assertEqual(len(f.cpu['source_sha256']),354)
        self.assertEqual(m.sha(m.canonical(f.cpu['source_sha256'])),m.SOURCE_MAP_SHA256)
        self.assertEqual(m.FROZEN_COMMIT,'f0445f4edf0dbea916ff63bedafe356d6d9ebcdb')

    def test_all_explicit_original_inputs_read_and_bound(self):
        self.configure()
        content, selection, launch = m.load_inputs()
        self.assertEqual(content,self.f.blobs['selection']); self.assertEqual(selection,self.f.selection)
        self.assertEqual(launch,self.f.launch); self.assertEqual(m.INPUT_SHA256,{k:v['sha256'] for k,v in self.artifacts.items()})

    def test_missing_unknown_or_uncreated_inputs_fail_before_remote_or_intent(self):
        with patch.object(m,'remote') as remote:
            with self.assertRaises(ValueError): m.submit(prep=self.base)
            remote.assert_not_called()
        for role in m.INPUT_ROLES:
            missing=copy.deepcopy(self.artifacts); del missing[role]
            with self.subTest(role=role),self.assertRaises(ValueError): m.configure_contract(m.FROZEN_COMMIT,missing)
        unknown=copy.deepcopy(self.artifacts); unknown['future_unknown']=unknown['method']
        with self.assertRaises(ValueError):m.configure_contract(m.FROZEN_COMMIT,unknown)
        self.artifacts['method']['path']=str(self.base/'not-created-future-method.json')
        with self.assertRaises(FileNotFoundError):self.configure()
        self.assertFalse((self.base/'policy-confirmation-v6-hydra-native-submit-intent.json').exists())

    def test_every_original_byte_hash_and_symlink_is_checked(self):
        for role in m.INPUT_ROLES:
            original=self.artifacts[role]['sha256']; self.artifacts[role]['sha256']='0'*64
            with self.subTest(role=role),self.assertRaises(ValueError):self.configure()
            self.artifacts[role]['sha256']=original
        path=Path(self.artifacts['method']['path']); real=self.base/'real-method.json'; path.rename(real); path.symlink_to(real)
        with self.assertRaises(ValueError):self.configure()

    def test_old_commit_cpu_proof_and_prereg_bytes_cannot_be_relabelled(self):
        for commit in ('3fdf92f71cf624f78611cf5ab5f03dba64ce4fec','a09a5530fd16fa0c04c0aa157ee56462aab825f4'):
            with self.subTest(commit=commit),self.assertRaises(ValueError):m.configure_contract(commit,self.artifacts)
        for role,name in (('cpu','behavior-consistency-target-cpu-v2-result.json'),
                          ('preregistration','policy-calibration-v4-method-preregistration.json')):
            original=copy.deepcopy(self.artifacts[role]); raw=(HERE/name).read_bytes()
            path=self.base/(role+'-old.json');path.write_bytes(raw);self.artifacts[role]=dict(path=str(path),sha256=m.sha(raw))
            with self.subTest(role=role),self.assertRaises(ValueError):self.configure()
            self.artifacts[role]=original

    def test_configured_inputs_are_reread_with_original_sha(self):
        self.configure(); path=Path(self.artifacts['terminal']['path']); path.write_bytes(self.f.terminal_raw+b'\n')
        with self.assertRaises(ValueError):m.load_inputs()

    def test_original_pre_submit_math_and_future_prospective_identity(self):
        self.configure();f=self.f
        m.selection_schema(f.selection,f.launch);m.method_binding(f.method_raw,f.selection)
        self.assertEqual(m.prospective_binding(f.method,f.launch_raw,f.observed_raw),f.launch)
        self.assertEqual(m.preregistration_binding(f.prereg_raw,f.method,f.launch)['actual_target_cpu_tests'],264)
        wrong=copy.deepcopy(f.method);wrong['prospective_binding']['preregistration_sha256']='0'*64
        with self.assertRaises(ValueError):m.preregistration_binding(f.prereg_raw,wrong,f.launch)
        with self.assertRaises(ValueError):m.preregistration_binding(m.canonical(m.strict_json(f.prereg_raw)),f.method,f.launch)

    def test_prospective_observation_drift_and_prior_claim_reject(self):
        self.configure();f=self.f
        for name,value in (('source_hashes_match',False),('child_actual_unstarted_verified',False),
                           ('child_manifest_slurm_job_id','1'),('actual_updates',[{'fixture':True}])):
            observed=m.strict_json(f.observed_raw);observed[name]=value;raw=m.canonical(observed)
            method=copy.deepcopy(f.method);method['prospective_binding']['zero_update_observation_sha256']=m.sha(raw)
            with self.subTest(name=name),self.assertRaises(ValueError):m.prospective_binding(method,f.launch_raw,raw)

    def test_frozen_v6_terminal_reader_is_exact_source(self):
        self.assertEqual(m.sha(Path(d.__file__).read_bytes()),m.TERMINAL_READER_SOURCE_SHA256)
        for name in ('_digest','_number','_job_fields','_scheduler_time','_snapshot_binding','read_terminal_observation','verify_terminal_current'):
            self.assertEqual(inspect.getsource(getattr(m,name)),inspect.getsource(getattr(d,name)))

    def test_synthetic_completed0_terminal_current_and_purged_routes(self):
        terminal=self.terminal()
        self.assertIsNot(terminal.get('physical_execution_authenticated'), True)
        self.assertFalse(terminal['production_admission'])
        self.assertFalse(terminal['healthy_checkpoint_accepted'])
        current=S(returncode=0,stdout=m.strict_json(self.f.terminal_raw)['job_stdout'],stderr='')
        self.assertIn('matches',m.verify_terminal_current(terminal,current)['route'])
        purged=S(returncode=1,stdout='',stderr=m.INVALID_JOB+'\n')
        queue=S(returncode=0,stdout='',stderr='')
        self.assertIn('purged',m.verify_terminal_current(terminal,purged,queue)['route'])
        for command,scoped in ((S(returncode=1,stdout='',stderr='other'),queue),
                               (purged,S(returncode=0,stdout='98765|RUNNING',stderr=''))):
            with self.assertRaises(ValueError):m.verify_terminal_current(terminal,command,scoped)

    def test_running_failed_wrong_birth_and_changed_terminal_cannot_pass(self):
        f=self.f
        with self.assertRaises(ValueError):m.completed_terminal_binding(f.selection,f.launch_raw,f.observed_raw,f.observed_raw)
        for old,new in (('JobState=COMPLETED','JobState=FAILED'),('ExitCode=0:0','ExitCode=1:0'),
                         ('Restarts=0','Restarts=1'),('WorkDir='+f.launch['base']+'/repo','WorkDir=/outside')):
            raw=f.terminal_raw.replace(old.encode(),new.encode());selection=copy.deepcopy(f.selection)
            selection['discovery_terminal_observation_sha256']=m.sha(raw)
            with self.subTest(new=new),self.assertRaises(ValueError):m.completed_terminal_binding(selection,f.launch_raw,f.observed_raw,raw)

    def test_no_future_method_default_and_closed_selection_descriptor(self):
        with self.assertRaises(ValueError):m.selection_schema(self.f.selection,self.f.launch)
        self.configure();m.selection_schema(self.f.selection,self.f.launch)
        for field in ('selection_method_sha256','discovery_generated_hydra_snapshots'):
            selection=copy.deepcopy(self.f.selection);del selection[field]
            with self.subTest(field=field),self.assertRaises(Exception):m.selection_schema(selection,self.f.launch)
        descriptor=copy.deepcopy(self.f.selection['discovery_generated_hydra_snapshots']);descriptor['unknown']=True
        with self.assertRaises(ValueError):m.hydra_descriptor_schema(descriptor,self.f.selection,self.f.launch)
        for value in (True,0,131073):
            descriptor=copy.deepcopy(self.f.selection['discovery_generated_hydra_snapshots'])
            next(iter(descriptor['files'].values()))['size_bytes']=value
            with self.subTest(value=value),self.assertRaises(ValueError):m.hydra_descriptor_schema(descriptor,self.f.selection,self.f.launch)

    def test_all_original_archived_metadata_rechecks_every_hash_before_heavy(self):
        self.configure(); prep=self.base/'archive/preparation';prep.mkdir(parents=True)
        names=dict(selection=m.SELECTION_NAME,method=m.METHOD_NAME,cpu=m.CPU_NAME,launch=m.DISCOVERY_LAUNCH_NAME,
            observation=m.ZERO_OBSERVATION_NAME,terminal=m.TERMINAL_NAME,preregistration=m.PREREGISTRATION_NAME,
            source_cleanup=m.SOURCE_CLEANUP_NAME,source_handle=m.SOURCE_HANDLE_NAME)
        fields=dict(selection='selection_sha256',method='selection_method_sha256',cpu='target_cpu_proof_sha256',
            launch='prospective_launch_sha256',observation='prospective_observation_sha256',terminal='terminal_observation_sha256',
            preregistration='preregistration_sha256',source_cleanup='discovery_source_cleanup_sha256',source_handle='discovery_source_handle_sha256')
        package=dict(original_input_sha256=copy.deepcopy(m.INPUT_SHA256))
        for role,name in names.items():
            (prep/name).write_bytes(self.f.blobs[role]);package[fields[role]]=m.INPUT_SHA256[role]
        self.assertEqual(m.archived_inputs(prep.parent,package),m.INPUT_SHA256)
        for role,name in names.items():
            path=prep/name;path.write_bytes(self.f.blobs[role]+b'\n')
            with self.subTest(role=role),self.assertRaises(ValueError):m.archived_inputs(prep.parent,package)
            path.write_bytes(self.f.blobs[role])

    def test_measured40_exact_empirical_tail_context4_and_writes_order(self):
        f=self.f;gate='e'*64
        structural=dict(execution_role=m.DISCOVERY_ROLE,run_id=f.selection['discovery_run_id'],
            calibration_sha256=f.selection['discovery_plan_sha256'],outer_count=1,optimizer_writes_per_rank=20,
            enforcement={'negative_tail':'observe_only'},global_monitors=[])
        for ordinal in range(40):
            mini=ordinal//2
            structural['global_monitors'].append(dict(context=dict(logical_update=1,minibatch_index=mini,
                prior_optimizer_writes=mini,behavior_probability_gate_sha256=gate),
                stage=('before_minibatch','after_minibatch')[ordinal%2],run_id=f.selection['discovery_run_id'],
                calibration_sha256=f.selection['discovery_plan_sha256'],optimizer_writes=mini+ordinal%2,
                execution_role=m.DISCOVERY_ROLE,enforcement={'negative_tail':'observe_only'},allowed=True,reasons=[],
                negative_tail_weight_fraction=.25 if ordinal==39 else 0.))
        segment=dict(structural=structural,behavior_probability_gate={'proof_sha256':gate})
        self.assertEqual(m.measured_selection(f.selection,segment),.25)
        for kind in ('missing','gate','order','margin','nan','writes'):
            selection=copy.deepcopy(f.selection);changed=copy.deepcopy(segment)
            if kind=='missing':changed['structural']['global_monitors'].pop()
            elif kind=='gate':changed['structural']['global_monitors'][1]['context']['behavior_probability_gate_sha256']='f'*64
            elif kind=='order':changed['structural']['global_monitors'].reverse()
            elif kind=='margin':selection['negative_tail_limit']=.25001
            elif kind=='nan':changed['structural']['global_monitors'][39]['negative_tail_weight_fraction']=float('nan')
            else:changed['structural']['optimizer_writes_per_rank']=19
            with self.subTest(kind=kind),self.assertRaises(ValueError):m.measured_selection(selection,changed)

    def test_light_heavy_and_hydra_behavior_verifiers_remain_exact_prior_code(self):
        for name in ('source_precheck','target_evidence','measured_selection','behavior_links',
                     'hydra_descriptor_schema','verify_source_snapshot_sets','owned_small_bytes'):
            self.assertEqual(inspect.getsource(getattr(m,name)),inspect.getsource(getattr(old,name)).replace('353','354'))
        light=inspect.getsource(m.source_precheck);heavy=inspect.getsource(m.target_evidence)
        self.assertNotIn('verify_child(',light);self.assertNotIn('torch.load',light)
        self.assertLess(heavy.index('native_preparation()'),heavy.index('verify_child('))
        self.assertIn('behavior_links(discovery, run, segment, updates)',light)
        self.assertIn('behavior_links(discovery, run, segment, updates)',heavy)
        self.assertIn('physical_execution_authenticated=False',light)
        self.assertIn('production_admission=False',heavy)
        self.assertIn('healthy_checkpoint_accepted=False',heavy)

    def test_local_main_missing_inputs_never_calls_remote_or_creates_intent(self):
        with patch.object(sys,'argv',[str(HERE/'submit_policy_confirmation_v6_native.py')]),patch.object(m,'remote') as remote:
            with self.assertRaises(ValueError):m.main()
            remote.assert_not_called()
        self.assertFalse((self.base/'policy-confirmation-v6-hydra-native-submit-intent.json').exists())

    def test_all_embedded_code_compiles_and_native_wrapper_binds_before_pipeline(self):
        compile(m.verification_code(),'<light-reader>','exec')
        for name in ('PREPARE_CODE','START_SOURCE_CODE','SUBMIT_CODE','OBSERVE_CODE','SOURCE_HANDLE_CODE','CLEANUP_SOURCE_CODE'):
            compile(getattr(m,name),name,'exec')
        self.assertIn(m.SOURCE_CLEANUP_NAME,m.PREPARE_CODE);self.assertIn(m.SOURCE_HANDLE_NAME,m.SUBMIT_CODE)
        self.assertIn("source_cleanup_bytes",m.PREPARE_CODE);self.assertIn("source_handle_bytes",m.PREPARE_CODE)
        files={'reproduction/common8/launch_calibration_native.sh':b'#!/bin/sh\nexec python -m reproduction.common8.run_pipeline\n'}
        wrapper=m.native_wrapper(files).decode()
        self.assertLess(wrapper.index('--native-bind'),wrapper.index('exec python -m reproduction.common8.run_pipeline'))
        bind=inspect.getsource(m.native_bind)
        self.assertLess(bind.index('archived_inputs('),bind.index('target_evidence('))
        self.assertLess(bind.index('native_preparation()'),bind.index('target_evidence('))

    def test_actual_frozen_git354_map_matches_approved_cpu_without_models(self):
        files=m.frozen_sources(self.f.launch)
        self.assertEqual(len(files),354)
        self.assertEqual({name:m.sha(raw) for name,raw in files.items()},self.f.cpu['source_sha256'])

    def test_previous_failed3294_originals_cannot_be_newsource_confirmation(self):
        method_raw=(HERE/'policy-calibration-v5-selection-method.json').read_bytes()
        launch_raw=(HERE/'policy-discovery-v5-native-launch.json').read_bytes()
        birth_raw=(HERE/'policy-discovery-v5-observation-1791254632444188.json').read_bytes()
        with self.assertRaises(ValueError):m.prospective_binding(m.strict_json(method_raw),launch_raw,birth_raw)
        old_cpu=m.strict_json((HERE/'behavior-consistency-target-cpu-v4-result.json').read_bytes())
        with self.assertRaises(ValueError):m.cpu_contract(old_cpu,m.strict_json(launch_raw))
        self.assertFalse(self.f.selection.get('production_admission',False))


if __name__=='__main__':unittest.main()
