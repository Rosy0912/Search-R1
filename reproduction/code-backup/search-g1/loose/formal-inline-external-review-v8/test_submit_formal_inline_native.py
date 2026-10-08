"""Constructed stdlib fixtures only: no SSH, jobs, source execution or models.

These tests must be reviewed before execution. Mocked CPU/native fixtures are
not measured physical evidence and cannot authorize an actual submission.
"""
import ast
import copy
import contextlib
import importlib.util
import io
import json
import os
import subprocess
import tempfile
import types
import unittest
from pathlib import Path, PosixPath
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('formal_inline_submitter',
                                            HERE / 'submit_formal_inline_native.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)
COMMIT = 'b' * 40
BASE = '/mnt/data/shaoshitong/g1f-a1234567'
MODULES = list(m.CPU_MODULES)
CPU_PRODUCER_SHA = m.sha((HERE / m.CPU_PRODUCER_FILE).read_bytes())
PROTOCOL = m.load_cpu_protocol(CPU_PRODUCER_SHA)
# Original raw RUNNING record from the sealed saved h100-8 observation
# policy-discovery-v6-observation-1791275757425127.json. Only fixture job
# identity/timestamps are changed below; scheduler field spelling is inherited.
SAVED_NATIVE_RUNNING = r'''JobId=3298 JobName=search-g1-policy-discovery-v6 UserId=root(0) GroupId=root(0) MCS_label=N/A Priority=4294898768 Nice=0 Account=(null) QOS=(null) JobState=RUNNING Reason=None Dependency=(null) Requeue=1 Restarts=0 BatchFlag=1 Reboot=0 ExitCode=0:0 RunTime=00:25:35 TimeLimit=08:00:00 TimeMin=N/A SubmitTime=2026-10-06T16:10:22 EligibleTime=2026-10-06T16:10:22 AccrueTime=2026-10-06T16:10:22 StartTime=2026-10-06T16:10:22 EndTime=2026-10-07T00:10:22 Deadline=N/A SuspendTime=None SecsPreSuspend=0 LastSchedEval=2026-10-06T16:10:22 Scheduler=Main Partition=gpu AllocNode:Sid=h100-8:2470314 ReqNodeList=h100-8 ExcNodeList=(null) NodeList=h100-8 BatchHost=h100-8 NumNodes=1 NumCPUs=192 NumTasks=1 CPUs/Task=64 ReqB:S:C:T=0:0:*:* TRES=cpu=192,mem=512G,node=1,billing=192 Socks/Node=* NtasksPerN:B:S:C=0:0:*:* CoreSpec=* MinCPUsNode=64 MinMemoryNode=512G MinTmpDiskNode=0 Features=(null) DelayBoot=00:00:00 OverSubscribe=NO Contiguous=0 Licenses=(null) Network=(null) Command=/mnt/data/shaoshitong/g1d-302569b1/repo/reproduction/common8/launch_calibration_native.sh WorkDir=/mnt/data/shaoshitong/g1d-302569b1/repo StdErr=/mnt/data/shaoshitong/g1d-302569b1/logs/native-discovery-v6-3298.log StdIn=/dev/null StdOut=/mnt/data/shaoshitong/g1d-302569b1/logs/native-discovery-v6-3298.log Power= TresPerNode=gres:gpu:h100:8 \n'''


def source_fixture():
    names = set(m.VERSION_FILES) | {
        'reproduction/common8/launch_native.sh',
        'reproduction/common8/formal_inline_safety.py',
        'reproduction/common8/train.sh', 'reproduction/common8/launch_calibration_native.sh'}
    names.update(module.replace('.', '/') + '.py' for module in MODULES)
    files = {name: ('# constructed fixture: ' + name + '\n').encode() for name in sorted(names)}
    files['reproduction/common8/budget.json'] = m.canonical(dict(m.MATHEMATICS,
        contract_kind='fresh200_warmup57', policy_safety_mode=m.SAFETY_MODE))
    return files, {name: m.sha(value) for name, value in files.items()}


def cpu_fixture(mapping):
    modules = list(MODULES)
    summary = dict(successful=True, cuda_available=False, cuda_initialized_before=False,
        cuda_initialized_after=False, cuda_visible_devices='', system='Linux', skipped=[],
        tests_run=len(modules), failures=0, errors=0, expected_failures=0, unexpected_successes=0,
        runner_error=None, expected_distribution_versions_matched=True, modules=modules,
        module_test_counts={name: 1 for name in modules}, expected_failure_details=[],
        unexpected_success_details=[], failure_details=[], error_details=[],
        distribution_versions=dict(torch='2.4.0+cu121', transformers='4.46.3', vllm='0.6.3', tensordict='0.5.0'),
        torch_version='2.4.0+cu121', transformers_module_version='4.46.3', vllm_module_version='0.6.3.dev-fixture',
        python_version='3.10.12', python_executable=PROTOCOL.TARGET_PYTHON,
        duration_seconds=.25, output='constructed CPU fixture; no actual target execution')
    base='/dataset/shaoshitong/g1-formal-inline-cpu-' + 'a' * 32
    runner_sha=m.sha(PROTOCOL.CPU_RUNNER.encode())
    return dict(schema_version=2, role='formal_inline_target_cpu_regression',
        base=base, repo_path=base+'/repo', runner_path=base+'/runner/cpu_runner.py', source_commit=COMMIT,
        source_sha256=mapping, source_map_sha256=m.sha(m.canonical(mapping)),
        source_initial_sha256=mapping, source_before_tests_sha256=mapping,
        source_after_tests_sha256=mapping, source_unchanged=True, source_inventory_error=None,
        expected_source_count=len(mapping), actual_source_count=len(mapping),
        cpu_runner_sha256=runner_sha, cpu_runner_before_sha256=runner_sha,
        cpu_runner_after_sha256=runner_sha, cpu_runner_unchanged=True, runner_read_error=None,
        target_python=PROTOCOL.TARGET_PYTHON, cuda_visible_devices='', actual_production_updates=0,
        production_admission=False, healthy_checkpoint_accepted=False, scope=PROTOCOL.SCOPE,
        started_unix=2., checked_unix=3., duration_seconds=1., runner_timeout_seconds=PROTOCOL.CPU_TIMEOUT_SECONDS,
        modules=modules, module_count=len(modules), returncode=0,
        observation_timed_out=False, owned_process_group_cleaned=True, owned_process_group_remaining=False,
        shell_syntax_checks=[dict(path=name, returncode=0, stdout='', stderr='', observation_timed_out=False,
            owned_process_group_cleaned=True, owned_process_group_remaining=False) for name in PROTOCOL.SHELL_FILES],
        stdout=m.CPU_SUMMARY_PREFIX + json.dumps(summary) + '\n', stderr='')


def cpu_claim_fixture(proof, mapping, files=None):
    if files is None:
        files, unused = source_fixture()
    payload = m.cpu_payload(PROTOCOL, proof, COMMIT, mapping, files, MODULES)
    claim = {key: value for key, value in payload.items() if key not in ('files', 'cpu_runner')}
    claim.update(role='formal_inline_target_cpu_execution_claim', claimed_unix=1.,
        launcher_sha256=CPU_PRODUCER_SHA, remote_code_sha256=m.sha(PROTOCOL.REMOTE_CODE.encode()),
        complete_payload_sha256=m.sha(PROTOCOL.canonical(payload)), target_host='h100-8')
    return claim


def verify_cpu_fixture(proof, mapping, execution_claim=None):
    files, unused = source_fixture()
    if execution_claim is None:
        execution_claim = cpu_claim_fixture(cpu_fixture(mapping), mapping, files)
    return m.verify_cpu(proof, COMMIT, mapping, MODULES, files, execution_claim, PROTOCOL, CPU_PRODUCER_SHA)


def method_fixture():
    _, mapping = source_fixture()
    return dict(schema_version=1, role='formal_inline_training_method', created_unix=1.,
        source_commit=COMMIT, source_map_sha256=m.sha(m.canonical(mapping)), target_updates=200,
        warmup_steps=57, safeguards=copy.deepcopy(m.THRESHOLDS), policy_safety_mode=m.SAFETY_MODE,
        same_process_after_five=True, fresh_states=True, optimizer_counting=m.FORMAL_COUNTING,
        negative_tail_semantics='prospective operational candidate; not calibrated health')


def old_fixture():
    launch = dict(base=m.OLD_BASE, parent_base=m.OLD_PARENT, job_id=m.OLD_JOB)
    observed = dict(base=m.OLD_BASE, job_id=m.OLD_JOB,
        **{'run-manifest.json': dict(run_id=m.OLD_RUN),
           'pipeline-status.json': dict(run_id=m.OLD_RUN, stage='FAILED'),
           'cleanup-status.json': dict(run_id=m.OLD_RUN, cleaned=True),
           'process-ledger.json': dict(run_id=m.OLD_RUN, jobs={'trainer': dict(state='STOPPED')})},
        actual_active_namespace_processes=[], job_returncode=1, job_stdout='',
        job_stderr='slurm_load_jobs error: Invalid job id specified\n')
    handle = dict(export='/dataset/shaoshitong/g1-original-export-discovery-v6-302569b1',
                  source_sha256='c' * 64, pid=7, start_ticks=8, sid=7, uid=0)
    return launch, observed, dict(cleaned=True, handle=handle)


def native_fixture():
    _, mapping = source_fixture()
    intent = dict(base=BASE, source_commit=COMMIT, source_sha256=mapping,
        source_map_sha256=m.sha(m.canonical(mapping)), source_count=len(mapping),
        method_sha256='e' * 64, cpu_producer_sha256=CPU_PRODUCER_SHA, export_port=19878, time_limit='2-00:00:00',
        job_name='search-g1-formal-inline-a1234567')
    common = dict(source_commit=COMMIT, source_sha256=mapping, target_updates=200,
        contract_kind='fresh200_warmup57', completed_updates=0,
        profile='common8', acceptance_update=5, policy_safety_mode=m.SAFETY_MODE,
        budget=dict(copy.deepcopy(m.MATHEMATICS), policy_safety_mode=m.SAFETY_MODE),
        initial_model=m.ASSET_ROOT + '/bootstrap-assets/instruct300/assets/qwen2.5-3b-instruct-aa8e725',
        reference_model=m.ASSET_ROOT + '/bootstrap-assets/instruct300/assets/qwen2.5-3b-instruct-aa8e725',
        initial_model_sha256={'config.json': '1' * 64},
        input_sha256={'manifest.json': '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'},
        calibration_question_ids=['fixture-id'],
        model_revision='aa8e72537993ba99e69dfaafa59ed015b17504d1')
    run = dict(common, run_id='c' * 32, policy_calibration_role='production')
    receipt = dict(intent, execution_role=m.ROLE, returncode=0, job_id='4321', submitted_unix=5.)
    observed = dict(base=BASE, job_id='4321', submission_outcome_known=True,
        submission=receipt, source_hashes_match=True, checked_unix=10., observation_finished_unix=11.,
        method_sha256=intent['method_sha256'], **{'run-manifest.json': run})
    return intent, observed


def scheduler_line(state):
    # The legacy archived wrapper encoded a trailing separator literally. It
    # is normalized only here, leaving new raw subprocess parsing unchanged.
    fields = m.job_fields(SAVED_NATIVE_RUNNING.removesuffix(r'\n').rstrip())
    fields.update(JobId='4321', JobName='search-g1-formal-inline-a1234567', JobState=state, TimeLimit='2-00:00:00',
        SubmitTime='2026-10-06T01:00:00', StartTime='2026-10-06T01:01:00',
        EndTime='2026-10-06T02:00:00' if state == 'COMPLETED' else 'Unknown',
        Command=BASE + '/repo/reproduction/common8/launch_native.sh', WorkDir=BASE + '/repo',
        StdOut=BASE + '/logs/native-formal-inline-4321.log',
        StdErr=BASE + '/logs/native-formal-inline-4321.log')
    return ' '.join(name + '=' + value for name, value in fields.items())


def terminal_fixture():
    intent, observed = native_fixture()
    launch = m.actual_launch(intent, observed)
    birth = dict(observed, job_returncode=0, job_stdout=scheduler_line('RUNNING'))
    terminal = dict(observed, checked_unix=20., observation_finished_unix=21.,
        job_returncode=0, job_stdout=scheduler_line('COMPLETED'), queue_returncode=0, queue_stdout='',
        actual_active_namespace_processes=[],
        **{'pipeline-status.json': dict(stage='FINISHED_FULL200_FINAL51713_AND_OWNED_CLEANUP'),
           'cleanup-status.json': dict(cleaned=True, run_id=launch['run_id']),
           'process-ledger.json': dict(jobs={'trainer': dict(state='STOPPED')})})
    return birth, terminal, launch


class LocalInputTests(unittest.TestCase):
    def test_unique_finite_json_and_prospective_method(self):
        for value in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                m.strict_json(value)
        _, mapping = source_fixture()
        m.verify_method(method_fixture(), COMMIT, mapping)
        for key, value in (('schema_version', True), ('role', 'old_constant300'),
                           ('same_process_after_five', False), ('source_map_sha256', '0' * 64)):
            method = method_fixture()
            method[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                m.verify_method(method, COMMIT, mapping)

    def test_frozen_native_walltime_is_exactly_fortyeight_hours(self):
        m.verify_walltime('2-00:00:00')
        for value in ('00:01:00','48:00:00','3-00:00:00',None,48):
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'exactly48hours'):
                m.verify_walltime(value)

    def test_formal_thresholds_and_same_process_horizon_cannot_drift(self):
        _, mapping = source_fixture()
        for part, key, value in (('safeguards', 'target_old_policy_kl', .01),
                ('safeguards', 'target_old_policy_kl', .02),
                ('safeguards', 'negative_tail_limit', None), ('safeguards', 'negative_tail_limit', .1)):
            method = method_fixture()
            method[part][key] = value
            with self.subTest(part=part, key=key), self.assertRaises(ValueError):
                m.verify_method(method, COMMIT, mapping)
        for key, value in (('target_updates', 5), ('warmup_steps', 0), ('fresh_states', False),
                           ('negative_tail_semantics', 'calibrated from failed3298'),
                           ('optimizer_counting', 'exactly100_actual_writes_before_acceptance')):
            method = method_fixture()
            method[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                m.verify_method(method, COMMIT, mapping)

    def _shared_committed(self, files, *, additions=None, limits=None, declared=None):
        """Real sealed producer loader; only Git bytes are constructed fixtures."""
        with tempfile.TemporaryDirectory() as folder:
            repo=Path(folder)/'repo';repo.mkdir()
            committed=dict(files,**(additions or {}))
            for name,data in committed.items():
                path=repo/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
            def git(actual_repo,*args,stdout_consumer=None):
                self.assertEqual(Path(actual_repo),repo)
                if args==('rev-parse','HEAD'):raw=(COMMIT+'\n').encode()
                elif args==('ls-tree','-r','-z',COMMIT):
                    raw=b''.join(('100644 blob '+'a'*40+'\t'+name).encode()+b'\0'
                                 for name in sorted(committed))
                elif args[:2]==('cat-file','-s'):raw=str(len(committed[args[2][41:]])).encode()
                elif args[0]=='show':raw=committed[args[1][41:]]
                else:raise AssertionError('Unexpected shared bounded Git call')
                if stdout_consumer is not None:
                    # Split within Unicode and tab-containing NUL records.
                    for offset in range(0,len(raw),7):stdout_consumer(raw[offset:offset+7])
                    return b''
                return raw
            mapping=declared if declared is not None else {name:m.sha(data) for name,data in files.items()}
            cpu_base='/dataset/shaoshitong/g1-formal-inline-cpu-'+'a'*32
            with patch.object(PROTOCOL,'git',side_effect=git),contextlib.ExitStack() as stack:
                for key,value in (limits or {}).items():stack.enter_context(patch.object(PROTOCOL,key,value))
                return m.committed_sources(COMMIT,mapping,len(mapping),repo,PROTOCOL,cpu_base)

    def test_dynamic_exact_committed_source_set_and_no_fixed_count(self):
        files,mapping=source_fixture()
        self.assertEqual(self._shared_committed(files),files)
        files['new_runtime.py']=b''
        self.assertEqual(len(self._shared_committed(files)),len(files))
        with self.assertRaisesRegex(ValueError,'exact complete'):
            self._shared_committed(files,declared=mapping)
        self.assertNotIn('EXPECTED_FILES',vars(m))

    def test_actual_shared_nul_loader_preserves_unicode_tabs_and_exclusions(self):
        files,_=source_fixture();files['分析\t模块.py']=b'# unicode raw path\n'
        actual=self._shared_committed(files,additions={'cache/ignored.py':b'',
            '.pytest_cache/ignored.json':b'{}','evidence/private.json':b'{}'})
        self.assertEqual(actual,files)

    def test_shared_loader_rejects_graph_and_bytes_before_blob_read(self):
        files,_=source_fixture()
        with self.assertRaisesRegex(ValueError,'graph budget exceeded'):
            self._shared_committed(files,limits={'MAX_RUNTIME_FILES':2})
        files['oversized.py']=b'x'*(PROTOCOL.MAX_FILE_BYTES+1)
        with self.assertRaisesRegex(ValueError,'size differs from bound'):
            self._shared_committed(files)

    def test_source_map_traversal_versions_size_and_wrong_bytes_rejected(self):
        files,mapping=source_fixture()
        for name in ('../foreign.py','/absolute.py','evidence/raw.json','cache/ignored.py'):
            bad=dict(mapping);bad[name]='1'*64
            with self.subTest(name=name),self.assertRaises(ValueError):m.verify_mapping(bad,len(bad))
        bad=dict(mapping);bad.pop(m.VERSION_FILES[0])
        with self.assertRaisesRegex(ValueError,'version'):m.verify_mapping(bad,len(bad))
        bad=dict(mapping);bad['reproduction/common8/train.sh']='1'*64
        with self.assertRaisesRegex(ValueError,'exact complete'):
            self._shared_committed(files,declared=bad)

    def test_actual_dynamic_suite_union_cpu_zero_skip_source_binding(self):
        _, mapping = source_fixture()
        proof = cpu_fixture(mapping)
        summary=verify_cpu_fixture(proof, mapping)
        self.assertEqual(summary['tests_run'], len(MODULES))
        for key, value in (('source_commit', 'f' * 40), ('source_unchanged', False),
                ('actual_source_count', len(mapping) + 1), ('cpu_runner_after_sha256', '0' * 64),
                ('observation_timed_out', True), ('returncode', True), ('source_initial_sha256', {})):
            bad = copy.deepcopy(proof)
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                verify_cpu_fixture(bad, mapping)

    def test_explicit_dynamic_cpu_union_rejects_missing_old_or_new_scientific_suite(self):
        _, mapping = source_fixture()
        m.verify_cpu_modules(MODULES, mapping)
        for missing in (m.ORIGINAL_CPU_MODULES[0], *m.INLINE_CPU_MODULES):
            with self.subTest(missing=missing), self.assertRaisesRegex(ValueError, 'union'):
                m.verify_cpu_modules([name for name in MODULES if name != missing], mapping)
        with self.assertRaisesRegex(ValueError, 'union'):
            m.verify_cpu_modules(MODULES + ['reproduction.common8.test_unknown_uncommitted'], mapping)

    def test_cpu_adverse_outcomes_missing_suite_or_summary_rejected(self):
        _, mapping = source_fixture()
        proof = cpu_fixture(mapping)
        summary = m.strict_json(proof['stdout'][len(m.CPU_SUMMARY_PREFIX):])
        for key, value in (('skipped', ['actual skipped test']), ('expected_failures', 1),
                ('unexpected_successes', 1), ('cuda_available', True), ('runner_error', 'failure')):
            bad = copy.deepcopy(proof)
            changed = dict(summary, **{key: value})
            bad['stdout'] = m.CPU_SUMMARY_PREFIX + json.dumps(changed)
            with self.subTest(key=key), self.assertRaises(ValueError):
                verify_cpu_fixture(bad, mapping)
        for stdout in ('{}', proof['stdout'] * 2):
            bad = dict(proof, stdout=stdout)
            with self.subTest(stdout=stdout[:20]), self.assertRaisesRegex(ValueError, 'Exactly one'):
                verify_cpu_fixture(bad, mapping)
        bad = copy.deepcopy(proof)
        bad['modules'].pop()
        with self.assertRaises(ValueError):
            verify_cpu_fixture(bad, mapping)

    def test_each_new_suite_and_each_shell_is_required_by_typed_cpu_protocol(self):
        unused, mapping=source_fixture()
        proof=cpu_fixture(mapping)
        for missing in m.INLINE_CPU_MODULES:
            bad=copy.deepcopy(proof)
            bad['modules'].remove(missing)
            bad['module_count']-=1
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                verify_cpu_fixture(bad,mapping)
        for missing in PROTOCOL.SHELL_FILES:
            bad=copy.deepcopy(proof)
            bad['shell_syntax_checks']=[row for row in bad['shell_syntax_checks'] if row['path']!=missing]
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                verify_cpu_fixture(bad,mapping)

    def test_old_cpu_role_prefix_namespace_and_claim_cannot_authorize_new_source(self):
        unused,mapping=source_fixture()
        proof=cpu_fixture(mapping)
        for key,value in (('role','fresh200_target_cpu_regression'),
                          ('base','/dataset/shaoshitong/g1-fresh200-cpu-'+'a'*32),
                          ('base','/mnt/data/shaoshitong/g1-formal-inline-cpu-'+'a'*32),
                          ('stdout',proof['stdout'].replace(m.CPU_SUMMARY_PREFIX,'FRESH200_CPU_SUMMARY='))):
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                verify_cpu_fixture(dict(proof,**{key:value}),mapping)
        claim=cpu_claim_fixture(proof,mapping)
        for key,value in (('role','fresh200_target_cpu_execution_claim'),('base','/dataset/shaoshitong/old'),
                          ('source_commit','0'*40),('complete_payload_sha256','0'*64),
                          ('launcher_sha256','0'*64),('remote_code_sha256','0'*64),
                          ('target_host','h100-9'),('claimed_unix',proof['started_unix']+1)):
            with self.subTest(claim=key),self.assertRaises(ValueError):
                verify_cpu_fixture(proof,mapping,dict(claim,**{key:value}))

    def test_typed_cpu_evidence_binds_interpreter_scope_environment_runner_and_timeline(self):
        unused,mapping=source_fixture()
        proof=cpu_fixture(mapping)
        for key,value in (('target_python','/usr/bin/python3'),('repo_path',proof['base']+'/foreign'),
                          ('runner_path',proof['base']+'/runner/other.py'),('scope','GPU admission'),
                          ('actual_production_updates',1),('production_admission',True),
                          ('healthy_checkpoint_accepted',True),('runner_timeout_seconds',901),
                          ('duration_seconds',2.),('started_unix',4.),('owned_process_group_remaining',True)):
            with self.subTest(key=key),self.assertRaises(ValueError):
                verify_cpu_fixture(dict(proof,**{key:value}),mapping)
        summary=m.strict_json(proof['stdout'][len(m.CPU_SUMMARY_PREFIX):])
        for key,value in (('python_executable','/usr/bin/python3'),('cuda_initialized_after',True),
                          ('system','Darwin'),('expected_distribution_versions_matched',False),
                          ('distribution_versions',dict(summary['distribution_versions'],vllm='0.6.2'))):
            bad=dict(proof,stdout=m.CPU_SUMMARY_PREFIX+json.dumps(dict(summary,**{key:value})))
            with self.subTest(summary=key),self.assertRaises(ValueError):
                verify_cpu_fixture(bad,mapping)

    def test_actual_test_count_is_derived_from_complete_summary_not_a_future_pin(self):
        unused,mapping=source_fixture()
        proof=cpu_fixture(mapping)
        summary=m.strict_json(proof['stdout'][len(m.CPU_SUMMARY_PREFIX):])
        summary['module_test_counts'][MODULES[0]]+=7
        summary['tests_run']+=7
        proof['stdout']=m.CPU_SUMMARY_PREFIX+json.dumps(summary)
        self.assertEqual(verify_cpu_fixture(proof,mapping)['tests_run'],len(MODULES)+7)

    def test_cpu_producer_load_rejects_unsealed_bytes(self):
        with self.assertRaisesRegex(ValueError,'producer bytes changed'):
            m.load_cpu_protocol('0'*64)

    def test_legacy_failed_cleaned_scope_is_not_required_to_be_successful(self):
        launch, observed, cleanup = old_fixture()
        m.verify_old_failure(launch, observed, cleanup)
        observed.pop('actual_active_namespace_processes')
        m.verify_old_failure(launch, observed, cleanup)  # Execute still obtains actual census.
        for action in ('running', 'uncleaned', 'active', 'foreign'):
            launch, observed, cleanup = old_fixture()
            if action == 'running':
                observed['pipeline-status.json']['stage'] = 'RUNNING'
            elif action == 'uncleaned':
                observed['cleanup-status.json']['cleaned'] = False
            elif action == 'active':
                observed['actual_active_namespace_processes'] = [dict(pid=1)]
            else:
                observed['run-manifest.json']['run_id'] = 'f' * 32
            with self.subTest(action=action), self.assertRaises(ValueError):
                m.verify_old_failure(launch, observed, cleanup)

    def test_legacy_literal_separator_is_read_only_and_new_native_parser_remains_strict(self):
        launch,observed,cleanup=old_fixture()
        raw=SAVED_NATIVE_RUNNING.replace('JobState=RUNNING','JobState=FAILED')
        observed.update(job_returncode=0,job_stdout=raw)
        before=copy.deepcopy(observed)
        m.verify_old_failure(launch,observed,cleanup)
        self.assertEqual(observed,before)
        with self.assertRaises(ValueError):m.job_fields(raw)
        with self.assertRaises(ValueError):m.historical_job_fields(raw+r'\n')

    def test_review_binds_actual_report_full_map_and_both_external_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            report = root / 'review.md'
            report.write_text('Constructed static review fixture; no actual review or execution admission.\n')
            names = ('submit_formal_inline_native.py', 'test_submit_formal_inline_native.py', m.CPU_PRODUCER_FILE, m.CPU_PRODUCER_TEST_FILE)
            for name in names:
                (root / name).write_text('# fixture\n')
            _, mapping = source_fixture()
            review = dict(role='formal_inline_preexecution_review', approved=True,
                source_commit=COMMIT, source_sha256=mapping, target_cpu_proof_sha256='a' * 64,
                cpu_producer_sha256=m.sha((root / m.CPU_PRODUCER_FILE).read_bytes()), cpu_execution_claim_sha256='c' * 64,
                method_sha256='b' * 64, blocking_findings=[], scope='staticonly',
                physical_execution_authenticated=False, production_admission=False,
                healthy_checkpoint_accepted=False, actualreview_report_path=str(report),
                actualreview_report_sha256=m.sha(report.read_bytes()),
                external_file_sha256={name: m.sha((root / name).read_bytes()) for name in names})
            m.verify_review(review, COMMIT, mapping, 'a' * 64, 'b' * 64, root, review['cpu_producer_sha256'], 'c' * 64)
            for name in names:
                target = root / name
                original = target.read_bytes()
                target.write_bytes(b'changed after review')
                with self.assertRaisesRegex(ValueError, 'bytes changed'):
                    m.verify_review(review, COMMIT, mapping, 'a' * 64, 'b' * 64, root, review['cpu_producer_sha256'], 'c' * 64)
                target.write_bytes(original)

    def test_sealed_ordinary_proof_and_exclusive_intent(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / 'intent.json'
            m.save(path, dict(role='constructed fixture'))
            content = path.read_bytes()
            self.assertEqual(m.sealed(path, m.sha(content))[1]['role'], 'constructed fixture')
            with self.assertRaises(FileExistsError):
                m.save(path, dict(role='overwrite forbidden'))
            link = root / 'link.json'
            link.symlink_to(path)
            with self.assertRaises(ValueError):
                m.sealed(link, m.sha(content))
            self.assertEqual(path.read_bytes(), content)

    def test_load_inputs_is_local_readonly_without_controller_write_or_rpc(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            files, mapping = source_fixture()
            cpu = cpu_fixture(mapping)
            method = method_fixture()
            previous_launch, previous_observation, previous_cleanup = old_fixture()
            report = root / 'review.md'
            report.write_text('Constructed static review bytes; not an actual review or admission.\n')
            def put(name, document):
                path = root / (name + '.json')
                content = m.canonical(document)
                path.write_bytes(content)
                return str(path), m.sha(content)
            paths = {name: put(name, document) for name, document in (
                ('source_map', mapping), ('cpu_modules', MODULES), ('target_cpu_proof', cpu),
                ('cpu_claim', cpu_claim_fixture(cpu,mapping,files)),
                ('method', method), ('previous_launch', previous_launch),
                ('previous_observation', previous_observation), ('previous_source_cleanup', previous_cleanup))}
            review = dict(role='formal_inline_preexecution_review', approved=True,
                source_commit=COMMIT, source_sha256=mapping,
                target_cpu_proof_sha256=paths['target_cpu_proof'][1], method_sha256=paths['method'][1],
                cpu_producer_sha256=CPU_PRODUCER_SHA,cpu_execution_claim_sha256=paths['cpu_claim'][1],
                blocking_findings=[], scope='staticonly', physical_execution_authenticated=False,
                production_admission=False, healthy_checkpoint_accepted=False,
                actualreview_report_path=str(report), actualreview_report_sha256=m.sha(report.read_bytes()),
                external_file_sha256={name: m.sha((HERE / name).read_bytes()) for name in (
                    'submit_formal_inline_native.py', 'test_submit_formal_inline_native.py', m.CPU_PRODUCER_FILE, m.CPU_PRODUCER_TEST_FILE)})
            paths['review_proof'] = put('review_proof', review)
            fields = dict(source_commit=COMMIT, source_count=len(mapping), cpu_producer_sha256=CPU_PRODUCER_SHA,
                          export_port=19878, time_limit='2-00:00:00')
            for name, (path, digest) in paths.items():
                fields[name] = path
                fields[name + '_sha256'] = digest
            before = {path.name: path.read_bytes() for path in root.iterdir()}
            with patch.object(m, 'committed_sources', return_value=files), \
                    patch.object(m, 'remote', side_effect=AssertionError('pure check cannot call RPC')), \
                    patch.object(m, 'save', side_effect=AssertionError('pure check cannot write')), \
                    patch.object(m.subprocess, 'run', side_effect=AssertionError('no process in pure fixture')):
                inputs, actual_files, method_bytes = m.load_inputs(types.SimpleNamespace(**fields))
            self.assertEqual(actual_files, files)
            self.assertEqual(inputs['actual_cpu_modules'], MODULES)
            self.assertEqual(inputs['actual_target_cpu_tests'], len(MODULES))
            self.assertEqual(m.sha(method_bytes), fields['method_sha256'])
            self.assertEqual(before, {path.name: path.read_bytes() for path in root.iterdir()})
            self.assertEqual(inputs['execution_role'], m.ROLE)
            self.assertIs(inputs['production_admission'], False)


class BoundedRpcTests(unittest.TestCase):
    def test_actual_remote_routes_once_through_sha_sealed_bounded_capture(self):
        controller=types.SimpleNamespace(route=lambda host,command:['MOCKED_ROUTE',host,command])
        result=subprocess.CompletedProcess([],0,stdout='{"actual":"constructed"}',stderr='')
        with patch.dict('sys.modules',control=controller),patch('sys.path',list(__import__('sys').path)), \
             patch.object(PROTOCOL,'bounded_capture',return_value=result) as capture, \
             patch.object(m,'load_cpu_protocol',return_value=PROTOCOL) as load:
            self.assertEqual(m.remote('h100-8','print(1)',{'x':1},producer_sha256=CPU_PRODUCER_SHA),
                             {'actual':'constructed'})
        load.assert_called_once_with(CPU_PRODUCER_SHA)
        self.assertEqual(capture.call_count,1)
        self.assertEqual(capture.call_args.args[1:],(90,2*m.MAX_FILE_BYTES,64*1024))
        self.assertEqual(capture.call_args.kwargs,{'input':m.canonical({'x':1})})

    def test_rpc_observation_errors_preserve_cause_and_never_repeat(self):
        controller=types.SimpleNamespace(route=lambda host,command:['MOCKED_ROUTE',host,command])
        for error in (subprocess.TimeoutExpired(['mock'],1),ValueError('bounded output overflow'),
            UnicodeDecodeError('utf-8',b'\xff',0,1,'invalid'),RuntimeError('orphan group')):
            with self.subTest(error=type(error).__name__),patch.dict('sys.modules',control=controller), \
                 patch('sys.path',list(__import__('sys').path)),patch.object(m,'load_cpu_protocol',return_value=PROTOCOL), \
                 patch.object(PROTOCOL,'bounded_capture',side_effect=error) as capture:
                with self.assertRaisesRegex(RuntimeError,'unknown; retain') as raised:
                    m.remote('h100-8','print(1)',{},producer_sha256=CPU_PRODUCER_SHA)
                self.assertIs(raised.exception.__cause__,error)
                self.assertEqual(capture.call_count,1)

    def test_oversized_rpc_input_rejects_before_route_or_process(self):
        with patch.object(m,'load_cpu_protocol',return_value=PROTOCOL),patch.object(m,'MAX_SOURCE_BYTES',2), \
             patch.object(PROTOCOL,'bounded_capture') as capture:
            with self.assertRaisesRegex(ValueError,'input byte budget'):
                m.remote('h100-8','print(1)',{'too_large':'bounded fixture'},producer_sha256=CPU_PRODUCER_SHA)
        capture.assert_not_called()


class NativeBindingTests(unittest.TestCase):
    def test_actual_single_formal_uuid_only_from_prepared_manifest(self):
        intent, observed = native_fixture()
        launch = m.actual_launch(intent, observed)
        self.assertEqual(launch['run_id'], observed['run-manifest.json']['run_id'])
        self.assertNotIn('parent_run_id', launch)
        self.assertNotIn('parent_base', launch)
        self.assertEqual(launch['execution_role'], m.ROLE)
        self.assertNotIn('plan_sha256', launch)
        for key in ('run-manifest.json',):
            bad = copy.deepcopy(observed)
            bad.pop(key)
            with self.subTest(key=key), self.assertRaises(KeyError):
                m.actual_launch(intent, bad)
        bad = copy.deepcopy(observed)
        bad['run-manifest.json']['diagnostic_proof'] = '/foreign/failed-diagnostic.json'
        with self.assertRaises(ValueError):
            m.actual_launch(intent, bad)

    def test_new_science_and_initial_source_identity_cannot_relabel_old(self):
        intent, observed = native_fixture()
        for document, key, value in (
                ('run-manifest.json', 'policy_calibration_role', 'policy_safety_calibration_discovery'),
                ('run-manifest.json', 'source_commit', 'f' * 40),
                ('run-manifest.json', 'policy_safety_mode', 'calibrated_constant300')):
            bad = copy.deepcopy(observed)
            bad[document][key] = value
            with self.subTest(document=document, key=key), self.assertRaises(ValueError):
                m.actual_launch(intent, bad)

    def test_actual_terminal_birth_restart_and_owned_cleanup_required(self):
        birth, terminal, launch = terminal_fixture()
        proof = m.terminal_proof(birth, terminal, launch)
        self.assertEqual(proof['execution_role'], m.ROLE)
        self.assertIn('captured_birth', proof)
        for old, new in (('JobState=COMPLETED', 'JobState=FAILED'), ('ExitCode=0:0', 'ExitCode=1:0'),
                ('Restarts=0', 'Restarts=1'), ('NodeList=h100-8', 'NodeList=h100-9'),
                ('StartTime=2026-10-06T01:01:00', 'StartTime=2026-10-06T01:02:00')):
            bad = dict(terminal, job_stdout=terminal['job_stdout'].replace(old, new))
            with self.subTest(old=old), self.assertRaises(ValueError):
                m.terminal_proof(birth, bad, launch)
        bad = dict(terminal, actual_active_namespace_processes=[dict(pid=9)])
        with self.assertRaises(ValueError):
            m.terminal_proof(birth, bad, launch)

    def test_actual_birth_and_terminal_require_exact_fortyeight_hour_walltime(self):
        for value in ('2-00:00:00','48:00:00'):
            birth,terminal,launch=terminal_fixture()
            for observation in (birth,terminal):
                observation['job_stdout']=observation['job_stdout'].replace('TimeLimit=2-00:00:00','TimeLimit='+value)
            self.assertEqual(m.native_walltime_seconds(value),172800)
            m.terminal_proof(birth,terminal,launch)
        for value in ('08:00:00','1-23:59:59','2-00:00:01','UNLIMITED','2-00:60:00'):
            for changed in ('birth','terminal'):
                birth,terminal,launch=terminal_fixture();target=birth if changed=='birth' else terminal
                target['job_stdout']=target['job_stdout'].replace('TimeLimit=2-00:00:00','TimeLimit='+value)
                with self.subTest(value=value,changed=changed),self.assertRaises(ValueError):m.terminal_proof(birth,terminal,launch)

    def test_purged_scheduler_or_stage_only_cannot_infer_success(self):
        birth, terminal, launch = terminal_fixture()
        for bad in (dict(terminal, job_returncode=1, job_stdout='', job_stderr='Invalid job id specified'),
                    dict(terminal, job_stdout=terminal['job_stdout'] + ' JobId=4321')):
            with self.assertRaises(ValueError):
                m.terminal_proof(birth, bad, launch)

    def test_ray_budget_and_only_three_export_transport_literals_change(self):
        self.assertLessEqual(m.ray_budget(BASE)['worst_plasma_socket_bytes'], 107)
        with self.assertRaises(ValueError):
            m.ray_budget(BASE + '/too-long-child-directory')
        original = ("ROOT=Path('" + m.EXPORT_ROOT + "')\n"
            "BASE=Path('/dataset/shaoshitong/g1-common8-original-retrieval-export')\n"
            "with Server(('10.33.0.13',19874),Handler):\n handle=dict(port=19874)\n").encode()
        source = m.exporter_bytes({'reproduction/common8/assets_export.py': original},
                                  '/dataset/shaoshitong/g1-formal-inline-export-a1234567', 19878)
        self.assertIn(b"ROOT=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')", source)
        self.assertEqual(source.count(b'19878'), 2)
        with self.assertRaises(ValueError):
            m.exporter_bytes({'reproduction/common8/assets_export.py': original + b'port=19874'}, '/tmp/x', 19878)


class ExporterStartupFixture:
    """Memory-only startup surface: never opens a file, process or socket.

    Executes only the helper definitions, after the required fresh review.
    All OS/process/time/path effects below are constructed stdlib doubles.
    """
    def __init__(self):
        self.events = []
        self.failures = {}
        self.clock = 0.
        self.identity = dict(pid=321, start_ticks=777, sid=321, uid=41)
        self.handle = dict(self.identity, port=19878, files={'original': 7})
        row = ['S'] + ['0'] * 19
        row[3], row[19] = '321', '777'
        self.data = {'/proc/321/stat': ('321 (fixture) ' + ' '.join(row)).encode(),
                     '/proc/321/cmdline': b'python3\0/export/serve.py\0',
                     '/export/serve.py': b'original source',
                     '/export/handle.json': json.dumps(self.handle).encode()}
        fixture = self

        class MemoryFile:
            def __init__(self, name):
                self.name = name
            def __enter__(self):
                return self
            def __exit__(self, *args):
                self.close()
                return False
            def close(self):
                fixture.event('close:' + self.name)
            def write(self, value):
                fixture.event('write:' + self.name)
                fixture.data[self.name] = value
                return len(value)
            def flush(self):
                fixture.event('flush:' + self.name)
            def fileno(self):
                return self.name

        class MemoryPath:
            def __init__(self, name):
                self.name = str(name)
            def __str__(self):
                return self.name
            def __truediv__(self, name):
                return MemoryPath(self.name + '/' + str(name))
            @property
            def parent(self):
                return MemoryPath(self.name.rsplit('/', 1)[0])
            def read_bytes(self):
                fixture.event('read:' + self.name)
                if self.name not in fixture.data:
                    raise FileNotFoundError(self.name)
                return fixture.data[self.name]
            def read_text(self):
                return self.read_bytes().decode()
            def stat(self):
                fixture.event('stat:' + self.name)
                return types.SimpleNamespace(st_uid=41)
            def is_file(self):
                return self.name in fixture.data
            def is_symlink(self):
                return False
            def open(self, mode):
                fixture.event('open:' + self.name + ':' + mode)
                if mode == 'xb':
                    if self.name in fixture.data:
                        raise FileExistsError(self.name)
                    fixture.data[self.name] = b''
                return MemoryFile(self.name)

        class Process:
            pid = 321
            def poll(self):
                raise AssertionError('Leader must never be polled/reaped before signals')
            def wait(self, timeout):
                fixture.event('wait:' + str(timeout))
                return -9

        self.process = Process()
        self.log = types.SimpleNamespace(close=lambda: self.event('close:log'))
        self.path = MemoryPath
        self.ns = {}
        exec(m.SOURCE_STARTUP_HELPERS_CODE, self.ns)
        self.ns.update(Path=MemoryPath,
            os=types.SimpleNamespace(O_RDONLY=0, P_PID=1, WEXITED=4, WNOHANG=1, WNOWAIT=8,
                getuid=lambda: 41,
                open=lambda name, flags: self.directory_open(name),
                close=lambda fd: self.event('fdclose:' + fd),
                fsync=lambda fd: self.event('fsync:' + fd),
                waitid=lambda kind, pid, flags: self.observe(kind, pid, flags),
                killpg=lambda pid, sig: self.event('signal:' + str(pid) + ':' + str(sig))),
            signal=types.SimpleNamespace(SIGTERM=15, SIGKILL=9),
            time=types.SimpleNamespace(monotonic=self.now, sleep=self.sleep))

    def event(self, name):
        self.events.append(name)
        failure = self.failures.get(name)
        if failure is not None:
            raise failure

    def directory_open(self, name):
        self.event('diropen:' + name)
        return name

    def observe(self, kind, pid, flags):
        assert (kind, pid, flags) == (1, 321, 13)
        self.event('observe')
        return None

    def now(self):
        self.event('clock')
        return self.clock

    def sleep(self, seconds):
        self.event('sleep')
        self.clock += seconds

    def spawn(self):
        self.event('spawn')
        return self.process

    def emit(self, handle):
        self.event('emit')
        self.result = handle

    def run(self):
        self.ns['exporter_start'](self.spawn, self.log, self.path('/export'),
            self.path('/export/serve.py'), b'original source',
            dict(export_port=19878, export_source_sha256=m.sha(b'original source')), self.emit)


class ExporterStartupTests(unittest.TestCase):
    def assert_cleaned(self, fixture):
        events = fixture.events
        signals = [event for event in events if event.startswith('signal:')]
        self.assertEqual(signals, ['signal:321:15', 'signal:321:9'])
        self.assertLess(events.index('signal:321:9'), events.index('wait:5'))
        self.assertIn('close:log', events)
        self.assertEqual(events.count('spawn'), 1)
        self.assertEqual(fixture.data['/export/serve.py'], b'original source')

    def test_proc_clock_write_flush_and_file_directory_fsync_failures(self):
        phases = ('clock', 'read:/proc/321/stat', 'stat:/proc/321',
            'open:/export/launcher-process.json:xb', 'write:/export/launcher-process.json',
            'flush:/export/launcher-process.json', 'fsync:/export/launcher-process.json',
            'diropen:/export', 'fsync:/export', 'fdclose:/export', 'observe',
            'read:/proc/321/cmdline', 'read:/export/handle.json',
            'read:/export/launcher-process.json', 'fsync:/export/handle.json',
            'close:log', 'emit')
        for phase in phases:
            with self.subTest(phase=phase):
                fixture = ExporterStartupFixture()
                original = OSError('constructed ' + phase)
                fixture.failures[phase] = original
                with self.assertRaises(OSError) as caught:
                    fixture.run()
                self.assertIs(caught.exception, original)
                self.assert_cleaned(fixture)

    def test_timeout_missing_or_incomplete_handle_is_bounded(self):
        for raw in (None, b'{'):
            with self.subTest(raw=raw):
                fixture = ExporterStartupFixture()
                if raw is None:
                    del fixture.data['/export/handle.json']
                else:
                    fixture.data['/export/handle.json'] = raw
                with self.assertRaises(TimeoutError):
                    fixture.run()
                self.assertLessEqual(fixture.events.count('sleep'), 100)
                self.assert_cleaned(fixture)

    def test_handoff_deadline_and_final_durable_bytes_are_rechecked(self):
        fixture = ExporterStartupFixture()
        def slow_fsync(fd):
            fixture.event('fsync:' + fd)
            if fd == '/export/handle.json':
                fixture.clock = 11.
        fixture.ns['os'].fsync = slow_fsync
        with self.assertRaises(TimeoutError):
            fixture.run()
        self.assertNotIn('emit', fixture.events)
        self.assert_cleaned(fixture)
        fixture = ExporterStartupFixture()
        read = fixture.path.read_bytes
        reads = []
        def changed_after_sync(path):
            if str(path) == '/export/launcher-process.json':
                reads.append(str(path))
                if len(reads) == 2:
                    return b'changed durable launcher'
            return read(path)
        fixture.path.read_bytes = changed_after_sync
        with self.assertRaises(AssertionError):
            fixture.run()
        self.assertNotIn('emit', fixture.events)
        self.assert_cleaned(fixture)

    def test_actual_handle_identity_and_port_mismatches(self):
        for key in ('pid', 'start_ticks', 'sid', 'uid', 'port'):
            with self.subTest(key=key):
                fixture = ExporterStartupFixture()
                fixture.handle[key] += 1
                fixture.data['/export/handle.json'] = json.dumps(fixture.handle).encode()
                with self.assertRaises(AssertionError):
                    fixture.run()
                self.assert_cleaned(fixture)

    def test_exit_is_observed_without_reaping_and_live_birth_is_rechecked(self):
        fixture = ExporterStartupFixture()
        fixture.ns['os'].waitid = lambda *args: types.SimpleNamespace(si_pid=321)
        with self.assertRaises(AssertionError):
            fixture.run()
        self.assert_cleaned(fixture)
        fixture = ExporterStartupFixture()
        identity = fixture.ns['exporter_identity']
        observations = []
        def changed_birth(pid):
            result = identity(pid)
            observations.append(result)
            if len(observations) == 2:
                result['start_ticks'] += 1
            return result
        fixture.ns['exporter_identity'] = changed_birth
        with self.assertRaises(AssertionError):
            fixture.run()
        self.assert_cleaned(fixture)

    def test_failed_spawn_has_no_group_authority(self):
        fixture = ExporterStartupFixture()
        original = OSError('Popen did not return an owned child')
        fixture.failures['spawn'] = original
        with self.assertRaises(OSError) as caught:
            fixture.run()
        self.assertIs(caught.exception, original)
        self.assertEqual(fixture.events, ['spawn', 'close:log'])

    def test_fsync_error_survives_file_and_directory_close_failures(self):
        for name, close in (('/export/launcher-process.json', 'close:/export/launcher-process.json'),
                            ('/export', 'fdclose:/export'),
                            ('/export/handle.json', 'close:/export/handle.json')):
            with self.subTest(name=name):
                fixture = ExporterStartupFixture()
                original = OSError('constructed fsync failure')
                fixture.failures['fsync:' + name] = original
                fixture.failures[close] = OSError('secondary close failure')
                with self.assertRaises(OSError) as caught:
                    fixture.run()
                self.assertIs(caught.exception, original)
                self.assert_cleaned(fixture)

    def test_source_and_durable_launcher_bytes_must_match(self):
        for name in ('/export/serve.py', '/export/launcher-process.json'):
            with self.subTest(name=name):
                fixture = ExporterStartupFixture()
                read = fixture.path.read_bytes
                def corrupt(path):
                    if str(path) == name:
                        fixture.event('read:' + name)
                        return b'changed bytes'
                    return read(path)
                fixture.path.read_bytes = corrupt
                with self.assertRaises(AssertionError):
                    fixture.run()
                self.assert_cleaned(fixture)

    def test_cleanup_observation_signal_wait_close_failures_preserve_original(self):
        fixture = ExporterStartupFixture()
        original = KeyboardInterrupt('constructed initial proc observation failure')
        fixture.failures.update({'read:/proc/321/stat': original,
            'clock': None, 'signal:321:15': OSError('TERM failed'),
            'signal:321:9': OSError('KILL failed'), 'wait:5': TimeoutError('reap timeout'),
            'close:log': OSError('close failed')})
        with self.assertRaises(KeyboardInterrupt) as caught:
            fixture.run()
        self.assertIs(caught.exception, original)
        self.assertEqual(fixture.events.count('read:/proc/321/stat'), 1)
        self.assertEqual(fixture.events.count('clock'), 1)
        self.assert_cleaned(fixture)

    def test_success_verifies_and_fsyncs_before_handoff_without_reap(self):
        fixture = ExporterStartupFixture()
        fixture.run()
        self.assertEqual(fixture.result, dict(fixture.handle, export='/export',
            source_sha256=m.sha(b'original source')))
        launcher = json.loads(fixture.data['/export/launcher-process.json'])
        self.assertEqual(launcher, dict(fixture.identity, source_sha256=m.sha(b'original source')))
        self.assertLess(fixture.events.index('fsync:/export/launcher-process.json'),
                        fixture.events.index('read:/export/launcher-process.json'))
        self.assertLess(fixture.events.index('fsync:/export/handle.json'), fixture.events.index('emit'))
        self.assertEqual(fixture.events.count('read:/export/launcher-process.json'), 2)
        self.assertFalse(any(event.startswith(('signal:', 'wait:')) for event in fixture.events))
        self.assertEqual(fixture.events.count('spawn'), 1)


class NoImplicitExecutionTests(unittest.TestCase):
    def test_remote_fragments_are_syntax_only_and_scoped(self):
        for name in ('OLD_READONLY_CODE', 'PREPARE_CODE', 'START_SOURCE_CODE', 'SUBMIT_CODE',
                     'OBSERVE_CODE', 'CLEANUP_SOURCE_CODE', 'SOURCE_HANDLE_CODE',
                     'RESERVE_NEVER_SUBMITTED_CLEANUP_CODE'):
            tree = ast.parse(getattr(m, name))
            self.assertTrue(tree.body, name)
        self.assertNotIn('ray stop', m.PREPARE_CODE + m.SUBMIT_CODE + m.CLEANUP_SOURCE_CODE)
        self.assertNotIn('scancel', m.PREPARE_CODE + m.SUBMIT_CODE + m.CLEANUP_SOURCE_CODE)
        self.assertIn("native-submit-claim.json", m.SUBMIT_CODE)
        self.assertLess(m.SUBMIT_CODE.index("write(prep/'native-submit-claim.json'"),
                        m.SUBMIT_CODE.index('subprocess.run(args'))
        self.assertLess(m.SUBMIT_CODE.index("write(prep/'native-submit-disposition.json',entered)"),
                        m.SUBMIT_CODE.index('subprocess.run(args'))
        self.assertNotIn('calibration-300', m.PREPARE_CODE)
        self.assertNotIn('SEARCH_G1_PRODUCTION_BASE', m.SUBMIT_CODE)
        self.assertNotIn('MEASUREMENT', m.SUBMIT_CODE)
        self.assertIn('SEARCH_G1_EXPORT_PORT', m.SUBMIT_CODE)
        self.assertIn("'scontrol','show','partition'", m.SUBMIT_CODE)
        self.assertNotIn("'parent_base'", m.PREPARE_CODE)

    def test_existing_unknown_submission_intent_is_never_retried(self):
        with tempfile.TemporaryDirectory() as folder:
            prep = Path(folder)
            m.save(prep / m.NAMES['intent'], dict(unknown_submission_outcome=True))
            with patch.object(m, 'load_inputs', side_effect=AssertionError('must reject before inputs')), \
                    patch.object(m, 'remote', side_effect=AssertionError('must not contact remote')), \
                    self.assertRaisesRegex(ValueError, 'Existing attempt'):
                m.submit(types.SimpleNamespace(), prep=prep)

    def test_unknown_mutation_outcome_keeps_intent_and_never_resubmits(self):
        inputs = dict(source_commit=COMMIT, source_sha256={}, source_count=0,
                      export_port=19878, cpu_producer_sha256=CPU_PRODUCER_SHA, execution_role=m.ROLE)
        with tempfile.TemporaryDirectory() as folder:
            prep = Path(folder)
            calls = []
            def remote(host, code, payload, **kwargs):
                calls.append((host, code))
                if code == m.OLD_READONLY_CODE:
                    return dict(actual_active_namespace_processes=[], stage='FAILED')
                raise TimeoutError('constructed unknown target copy result')
            with patch.object(m, 'load_inputs', return_value=(inputs, {}, b'{}')), \
                    patch.object(m, 'exporter_bytes', return_value=b'# fixture exporter'), \
                    patch.object(m, 'remote', side_effect=remote), \
                    self.assertRaises(TimeoutError):
                m.submit(types.SimpleNamespace(), prep=prep)
            intent = (prep / m.NAMES['intent']).read_bytes()
            with patch.object(m, 'remote', side_effect=AssertionError('no retry')), \
                    self.assertRaisesRegex(ValueError, 'Existing attempt'):
                m.submit(types.SimpleNamespace(), prep=prep)
            self.assertEqual((prep / m.NAMES['intent']).read_bytes(), intent)
            self.assertEqual(len(calls), 2)

    def test_birth_is_preserved_before_manifest_without_guessing_run_uuid(self):
        intent, observed = native_fixture()
        observed.pop('run-manifest.json')
        observed.update(job_returncode=0, job_stdout=scheduler_line('RUNNING'))
        with tempfile.TemporaryDirectory() as folder:
            prep = Path(folder)
            m.save(prep / m.NAMES['intent'], intent)
            with patch.object(m, 'remote', return_value=observed) as remote:
                result = m.observe(prep)
            remote.assert_called_once_with('h100-8', m.OBSERVE_CODE, intent,producer_sha256=CPU_PRODUCER_SHA)
            self.assertEqual(result, observed)
            self.assertTrue((prep / m.NAMES['birth']).exists())
            self.assertFalse((prep / m.NAMES['launch']).exists())
            self.assertNotIn('run_id', m.strict_json((prep / m.NAMES['birth']).read_bytes()))

    def test_recoverable_brake_observation_never_exits_or_relaunches_native_job(self):
        intent, observed = native_fixture()
        observed.update(job_returncode=0, job_stdout=scheduler_line('RUNNING'),
            **{'pipeline-status.json': dict(stage='FORMAL_TRAINING_SAME_PROCESS',
                completed_outer_updates=2, actual_optimizer_writes_per_rank=3,
                abandoned_attempt_partial_writes_per_rank=1, guard_recovery='fresh_rollout_same_job')})
        with tempfile.TemporaryDirectory() as folder:
            prep = Path(folder)
            m.save(prep / m.NAMES['intent'], intent)
            with patch.object(m, 'remote', return_value=observed) as remote:
                result = m.observe(prep)
            remote.assert_called_once_with('h100-8', m.OBSERVE_CODE, intent,producer_sha256=CPU_PRODUCER_SHA)
            self.assertEqual(result['pipeline-status.json']['actual_optimizer_writes_per_rank'], 3)
            self.assertTrue((prep / m.NAMES['launch']).exists())
            self.assertFalse((prep / m.NAMES['terminal']).exists())

    def test_check_inputs_cli_and_import_do_not_lazy_load_controller(self):
        tree = ast.parse(Path(m.__file__).read_text())
        global_imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        self.assertFalse(any(isinstance(node, ast.Import) and any(alias.name == 'control' for alias in node.names)
                             for node in global_imports))
        self.assertFalse(any(isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('reproduction')
                             for node in global_imports))
        args = m.parser().parse_args(['--check-inputs'])
        self.assertTrue(args.check_inputs)
        self.assertFalse(args.execute)


class EmbeddedNativeSurfaceTests(unittest.TestCase):
    """Run the real embedded fragments against private files and mocked CLI.

    These tests are authored for review first; no fragment has been executed
    while implementing this draft. They never call a controller or real Slurm.
    """
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.base = self.root / 'g1f-a1234567'
        self.base.mkdir()
        (self.base / 'preparation').mkdir()
        (self.base / 'logs').mkdir()
        self.proc = self.root / 'proc'
        self.proc.mkdir()
        intent, unused = native_fixture()
        files, mapping = source_fixture()
        self.payload = dict(intent, base=str(self.base), execution_role=m.ROLE,
            target_cpu_proof_sha256='f' * 64, export_base='/dataset/shaoshitong/g1-formal-inline-export-a1234567',
            export_source_sha256='d' * 64, source_count=len(mapping), source_sha256=mapping,
            handle=dict(files={'assets-root/assets/wiki/e5_Flat.index': 7}))
        for name, content in files.items():
            path = self.base / 'repo' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        method = b'{"constructed_prospective_fixture":true}\n'
        self.payload['method_sha256'] = m.sha(method)
        (self.base / 'preparation/formal-inline-method.json').write_bytes(method)
        self.record('namespace-owner.json', dict(base=str(self.base), uid=os.getuid(),
            execution_role=m.ROLE, source_commit=COMMIT))
        self.record('source-package-intent.json', self.payload)
        self.stamp = m.datetime.datetime.fromisoformat('2026-10-06T01:00:00').timestamp()
        self.calls = []

    def tearDown(self):
        self.folder.cleanup()

    def record(self, name, value):
        (self.base / 'preparation' / name).write_text(json.dumps(value))

    def path(self, value):
        if str(value) == '/mnt/data/shaoshitong':
            return self.root
        if str(value) == '/proc':
            return self.proc
        return PosixPath(value)

    def scheduler(self, job='4321', **changes):
        fields = m.job_fields(scheduler_line('RUNNING').replace(BASE, str(self.base)))
        fields.update(JobId=job, UserId='fixture(' + str(os.getuid()) + ')',
            StdOut=str(self.base / 'logs') + '/native-formal-inline-' + job + '.log',
            StdErr=str(self.base / 'logs') + '/native-formal-inline-' + job + '.log')
        fields.update(changes)
        return ' '.join(key + '=' + value for key, value in fields.items())

    def cli(self, args, **kwargs):
        self.calls.append(list(args))
        if args[0] == 'sbatch':
            raise subprocess.TimeoutExpired(args, 60)
        if args[:4] == ['scontrol', 'show', 'partition', 'gpu']:
            return types.SimpleNamespace(returncode=0, stdout='PartitionName=gpu State=UP MaxTime=UNLIMITED\n', stderr='')
        if args == ['scontrol', 'show', 'job', '-o']:
            return types.SimpleNamespace(returncode=0, stdout=self.scheduler_records, stderr='')
        if args[:4] == ['scontrol', 'show', 'job', '-o']:
            return types.SimpleNamespace(returncode=0, stdout=self.scheduler(args[4]), stderr='')
        if args[0] == 'squeue':
            return types.SimpleNamespace(returncode=0, stdout=self.queue_records, stderr='')
        raise AssertionError('Unexpected mocked CLI: ' + repr(args))

    def fragment(self, code, cli=None, opener=None):
        response = types.SimpleNamespace(status=200, headers={'Content-Length': '7'})
        class Response:
            def __enter__(inner):
                return response
            def __exit__(inner, *args):
                return False
        fake_opener = opener or types.SimpleNamespace(open=lambda *args, **kwargs: Response())
        stdout = io.StringIO()
        with patch('pathlib.Path', side_effect=self.path), \
                patch('sys.stdin', io.StringIO(json.dumps(self.payload))), \
                patch('sys.stdout', stdout), patch('time.time', return_value=self.stamp), \
                patch('subprocess.run', side_effect=cli or self.cli), \
                patch('urllib.request.build_opener', return_value=fake_opener):
            exec(code, {})
        return m.strict_json(stdout.getvalue())

    def enter_timeout(self):
        self.scheduler_records = self.queue_records = ''
        with self.assertRaises(subprocess.TimeoutExpired):
            self.fragment(m.SUBMIT_CODE)
        self.assertFalse((self.base / 'preparation/native-submit-receipt.json').exists())
        self.assertEqual(json.loads((self.base / 'preparation/native-submit-disposition.json').read_text())['state'],
                         'sbatch_entered')

    def test_saved_actual_scheduler_spelling_is_retained(self):
        fields = m.job_fields(SAVED_NATIVE_RUNNING.removesuffix(r'\n').rstrip())
        self.assertEqual(fields['TresPerNode'], 'gres:gpu:h100:8')
        birth, unused, launch = terminal_fixture()
        self.assertEqual(m.native_record(birth, launch, 'RUNNING', '0:0')['TresPerNode'], fields['TresPerNode'])

    def test_actual_submit_timeout_recovers_unique_bound_job_without_resubmission(self):
        self.enter_timeout()
        self.scheduler_records = self.scheduler() + '\n'
        self.queue_records = '4321|' + self.payload['job_name'] + '\n'
        observed = self.fragment(m.OBSERVE_CODE)
        self.assertEqual(observed['job_id'], '4321')
        self.assertTrue(observed['submission_outcome_known'])
        self.assertIsNone(observed['submission']['returncode'])
        self.assertEqual(observed['submission']['submission_authentication'], 'actual_scheduler_claim_recovery')
        self.assertTrue(m.submission_authenticated(observed['submission']))
        self.assertTrue((self.base / 'preparation/native-submit-recovered-receipt.json').exists())
        self.fragment(m.OBSERVE_CODE)
        self.assertEqual(sum(args[0] == 'sbatch' for args in self.calls), 1)

    def test_ambiguous_or_wrong_identity_is_not_recovered(self):
        self.enter_timeout()
        self.queue_records = ''
        for records in (self.scheduler() + '\n' + self.scheduler('4322') + '\n',
                        self.scheduler(Command='/some/foreign/job.sh') + '\n',
                        self.scheduler(WorkDir='/some/foreign') + '\n',
                        self.scheduler(UserId='foreign(' + str(os.getuid() + 1) + ')') + '\n',
                        self.scheduler(TimeLimit='08:00:00') + '\n',
                        self.scheduler(SubmitTime='2026-10-05T01:00:00') + '\n'):
            self.scheduler_records = records
            with self.subTest(records=records), self.assertRaises(AssertionError):
                self.fragment(m.OBSERVE_CODE)
            self.assertFalse((self.base / 'preparation/native-submit-recovered-receipt.json').exists())
        self.assertEqual(sum(args[0] == 'sbatch' for args in self.calls), 1)

    def test_recovery_rejects_changed_claim_uid_source_or_exact_sbatch_argv(self):
        self.enter_timeout()
        path = self.base / 'preparation/native-submit-claim.json'
        original = json.loads(path.read_text())
        self.scheduler_records = self.scheduler() + '\n'
        self.queue_records = ''
        for key, value in (('uid', os.getuid() + 1), ('source_commit', '0' * 40),
                           ('argv', original['argv'] + ['--extra-foreign-option'])):
            altered = dict(original, **{key: value})
            path.write_text(json.dumps(altered))
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.fragment(m.OBSERVE_CODE)
            self.assertFalse((self.base / 'preparation/native-submit-recovered-receipt.json').exists())
        path.write_text(json.dumps(original))
        self.assertEqual(sum(args[0] == 'sbatch' for args in self.calls), 1)

    def test_entered_but_absent_or_purged_job_remains_unknown_and_not_cleanable(self):
        self.enter_timeout()
        self.scheduler_records = self.queue_records = ''
        observed = self.fragment(m.OBSERVE_CODE)
        self.assertFalse(observed['submission_outcome_known'])
        self.assertEqual(observed['submission_state'], 'UNKNOWN_OR_PURGED_AFTER_SBATCH_ENTERED')
        with self.assertRaises(AssertionError):
            self.fragment(m.RESERVE_NEVER_SUBMITTED_CLEANUP_CODE)
        self.assertEqual(sum(args[0] == 'sbatch' for args in self.calls), 1)

    def test_precheck_failure_is_never_submitted_and_cleanup_reservation_blocks_later_sbatch(self):
        self.scheduler_records = self.queue_records = ''
        bad = types.SimpleNamespace(open=lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError('HEAD failed')))
        with self.assertRaises(TimeoutError):
            self.fragment(m.SUBMIT_CODE, opener=bad)
        self.assertTrue((self.base / 'preparation/native-precheck-failed.json').exists())
        self.assertFalse((self.base / 'preparation/native-submit-disposition.json').exists())
        result = self.fragment(m.RESERVE_NEVER_SUBMITTED_CLEANUP_CODE)
        self.assertTrue(result['cleanup_eligible'])
        self.assertEqual(result['disposition']['state'], 'never_submitted_cleanup')
        self.assertEqual(sum(args[0] == 'sbatch' for args in self.calls), 0)
        with self.assertRaises(FileExistsError):
            self.fragment(m.SUBMIT_CODE)
        self.assertEqual(sum(args[0] == 'sbatch' for args in self.calls), 0)

    def test_old_scope_recheck_failure_before_submit_can_reserve_exact_never_entered_namespace(self):
        self.scheduler_records = self.queue_records = ''
        self.assertFalse((self.base / 'preparation/native-submit-claim.json').exists())
        result = self.fragment(m.RESERVE_NEVER_SUBMITTED_CLEANUP_CODE)
        self.assertTrue(result['cleanup_eligible'])
        observed = self.fragment(m.OBSERVE_CODE)
        self.assertEqual(observed['submission_state'], 'VERIFIED_NEVER_SUBMITTED_CLEANUP_RESERVED')
        self.assertEqual(sum(args[0] == 'sbatch' for args in self.calls), 0)
        # This starts the first submit after cleanup won, so the new submit
        # claim is published and the later disposition collision is exercised.
        with self.assertRaises(FileExistsError):
            self.fragment(m.SUBMIT_CODE)
        self.assertTrue((self.base / 'preparation/native-submit-claim.json').exists())
        self.assertEqual(json.loads((self.base / 'preparation/native-submit-disposition.json').read_text())['state'],
                         'never_submitted_cleanup')
        self.assertEqual(sum(args[0] == 'sbatch' for args in self.calls), 0)

    def test_never_submitted_cleanup_rejects_scheduler_candidate_or_changed_source(self):
        self.scheduler_records = self.scheduler() + '\n'
        self.queue_records = ''
        with self.assertRaises(AssertionError):
            self.fragment(m.RESERVE_NEVER_SUBMITTED_CLEANUP_CODE)
        self.assertFalse((self.base / 'preparation/native-submit-disposition.json').exists())
        self.scheduler_records = ''
        changed = self.base / 'repo' / next(iter(self.payload['source_sha256']))
        changed.write_bytes(b'changed source')
        with self.assertRaises(AssertionError):
            self.fragment(m.RESERVE_NEVER_SUBMITTED_CLEANUP_CODE)
        self.assertFalse((self.base / 'preparation/native-submit-disposition.json').exists())


if __name__ == '__main__':
    unittest.main()
