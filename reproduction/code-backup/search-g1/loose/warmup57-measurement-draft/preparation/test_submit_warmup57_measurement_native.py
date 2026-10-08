"""Constructed stdlib fixtures only: no SSH, jobs, source execution or models.

These tests must be reviewed before execution. Mocked CPU/native fixtures are
not measured physical evidence and cannot authorize an actual submission.
"""
import ast
import copy
import importlib.util
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('warmup57_measurement_submitter',
                                            HERE / 'submit_warmup57_measurement_native.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)
COMMIT = 'b' * 40
BASE = '/mnt/data/shaoshitong/g1m-a1234567'
PARENT = '/mnt/data/shaoshitong/g1q-a1234567'


def source_fixture():
    names = set(m.VERSION_FILES) | {
        'reproduction/common8/launch_measurement_native.sh',
        'reproduction/common8/warmup_measurement_plan.py',
        'reproduction/common8/warmup_measurement_runtime.py',
        'reproduction/common8/train.sh', 'reproduction/common8/launch_calibration_native.sh'}
    names.update(module.replace('.', '/') + '.py' for module in m.CPU_MODULES)
    files = {name: ('# constructed fixture: ' + name + '\n').encode() for name in sorted(names)}
    files['reproduction/common8/budget.json'] = m.canonical(dict(m.MATHEMATICS, contract_kind='fresh200_warmup57'))
    return files, {name: m.sha(value) for name, value in files.items()}


def cpu_fixture(mapping):
    modules = list(m.CPU_MODULES)
    summary = dict(successful=True, cuda_available=False, cuda_initialized_before=False,
        cuda_initialized_after=False, cuda_visible_devices='', system='Linux', skipped=[],
        tests_run=len(modules), failures=0, errors=0, expected_failures=0, unexpected_successes=0,
        runner_error=None, expected_distribution_versions_matched=True, modules=modules,
        module_test_counts={name: 1 for name in modules})
    return dict(schema_version=2, role='fresh200_target_cpu_regression',
        base='/dataset/shaoshitong/g1-fresh200-cpu-' + 'a' * 32, source_commit=COMMIT,
        source_sha256=mapping, source_map_sha256=m.sha(m.canonical(mapping)),
        source_initial_sha256=mapping, source_before_tests_sha256=mapping,
        source_after_tests_sha256=mapping, source_unchanged=True, source_inventory_error=None,
        expected_source_count=len(mapping), actual_source_count=len(mapping),
        cpu_runner_sha256='d' * 64, cpu_runner_before_sha256='d' * 64,
        cpu_runner_after_sha256='d' * 64, cpu_runner_unchanged=True, runner_read_error=None,
        modules=modules, module_count=len(modules), returncode=0,
        observation_timed_out=False, owned_process_group_remaining=False,
        shell_syntax_checks=[dict(path=name, returncode=0, observation_timed_out=False,
            owned_process_group_remaining=False) for name in (
                'reproduction/common8/train.sh', 'reproduction/common8/launch_calibration_native.sh',
                'reproduction/common8/launch_measurement_native.sh')],
        stdout=m.CPU_SUMMARY_PREFIX + json.dumps(summary) + '\n')


def method_fixture():
    return dict(schema_version=2, role='warmup57_measurement_method', created_unix=1.,
        phase_budget=copy.deepcopy(m.PHASE_BUDGET), health_rule=copy.deepcopy(m.HEALTH),
        negative_tail_method=m.TAIL_METHOD, candidate_thresholds=copy.deepcopy(m.THRESHOLDS),
        mathematics_sha256=m.sha(m.canonical(m.MATHEMATICS)))


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
    intent = dict(base=BASE, parent_base=PARENT, source_commit=COMMIT, source_sha256=mapping,
        source_map_sha256=m.sha(m.canonical(mapping)), source_count=len(mapping),
        method_sha256='e' * 64, export_port=19876)
    common = dict(source_commit=COMMIT, source_sha256=mapping, target_updates=200,
        contract_kind='fresh200_warmup57', completed_updates=0,
        budget=copy.deepcopy(m.MATHEMATICS), initial_model_sha256={'config.json': '1' * 64},
        input_sha256={'manifest.json': '2' * 64}, calibration_question_ids=['fixture-id'],
        model_revision='aa8e72537993ba99e69dfaafa59ed015b17504d1')
    parent = dict(common, run_id='a' * 32, policy_calibration_role='production')
    run = dict(common, run_id='c' * 32, policy_calibration_role=m.ROLE,
        measurement_parent_run_id=parent['run_id'], measurement_method_sha256=intent['method_sha256'])
    receipt = dict(intent, execution_role=m.ROLE, returncode=0, job_id='4321', submitted_unix=5.)
    observed = dict(base=BASE, parent_base=PARENT, job_id='4321', submission_outcome_known=True,
        submission=receipt, source_hashes_match=True, checked_unix=10., observation_finished_unix=11.,
        parent_manifest_sha256='f' * 64,
        **{'run-manifest.json': run, 'parent-run-manifest.json': parent})
    return intent, observed


def scheduler_line(state):
    return ' '.join([
        'JobId=4321', 'JobName=search-g1-warmup57-a1234567', 'UserId=root(0)',
        'JobState=' + state, 'ExitCode=0:0', 'Restarts=0',
        'SubmitTime=2026-10-06T01:00:00', 'StartTime=2026-10-06T01:01:00',
        'EndTime=' + ('2026-10-06T02:00:00' if state == 'COMPLETED' else 'Unknown'),
        'NodeList=h100-8', 'BatchHost=h100-8', 'NumNodes=1', 'MinMemoryNode=512G',
        'TresPerNode=gres/gpu:h100:8',
        'Command=' + BASE + '/repo/reproduction/common8/launch_measurement_native.sh',
        'WorkDir=' + BASE + '/repo', 'StdOut=' + BASE + '/logs/native-warmup57-4321.log',
        'StdErr=' + BASE + '/logs/native-warmup57-4321.log'])


def terminal_fixture():
    intent, observed = native_fixture()
    launch = m.actual_launch(intent, observed)
    birth = dict(observed, job_returncode=0, job_stdout=scheduler_line('RUNNING'))
    terminal = dict(observed, checked_unix=20., observation_finished_unix=21.,
        job_returncode=0, job_stdout=scheduler_line('COMPLETED'), queue_returncode=0, queue_stdout='',
        actual_active_namespace_processes=[],
        **{'pipeline-status.json': dict(stage='FINISHED_WARMUP57_MEASUREMENTS_AND_OWNED_CLEANUP'),
           'cleanup-status.json': dict(cleaned=True, run_id=launch['run_id']),
           'process-ledger.json': dict(jobs={'trainer': dict(state='STOPPED')})})
    return birth, terminal, launch


class LocalInputTests(unittest.TestCase):
    def test_unique_finite_json_and_prospective_method(self):
        for value in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                m.strict_json(value)
        m.verify_method(method_fixture())
        for key, value in (('schema_version', True), ('role', 'old_constant300'),
                           ('negative_tail_method', 'rounded_floor'), ('mathematics_sha256', '0' * 64)):
            method = method_fixture()
            method[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                m.verify_method(method)

    def test_health_threshold_and_two_outer_budget_cannot_drift(self):
        for part, key, value in (('health_rule', 'method', 'allow_drop'),
                ('candidate_thresholds', 'target_old_policy_kl', .02),
                ('candidate_thresholds', 'negative_tail_limit', .1),
                ('phase_budget', 'outer_count', 1), ('phase_budget', 'optimizer_writes_per_rank', 20)):
            method = method_fixture()
            method[part][key] = value
            with self.subTest(part=part, key=key), self.assertRaises(ValueError):
                m.verify_method(method)

    def test_dynamic_exact_committed_source_set_and_no_fixed_count(self):
        files, mapping = source_fixture()
        def git(argv, **kwargs):
            if argv[:3] == ['git', 'ls-tree', '-r']:
                return '\n'.join('100644 blob ' + 'a' * 40 + '\t' + name for name in sorted(files))
            name = argv[-1][41:]
            return str(len(files[name])).encode() if argv[2] == '-s' else files[name]
        with patch.object(m.subprocess, 'check_output', side_effect=git):
            self.assertEqual(m.committed_sources(COMMIT, mapping, len(mapping)), files)
            files['new_runtime.py'] = b''
            mapping['new_runtime.py'] = m.sha(b'')
            self.assertEqual(len(m.committed_sources(COMMIT, mapping, len(mapping))), len(files))
            bad = dict(mapping)
            bad.pop('new_runtime.py')
            with self.assertRaisesRegex(ValueError, 'exact complete'):
                m.committed_sources(COMMIT, bad, len(bad))
        self.assertNotIn('EXPECTED_FILES', vars(m))

    def test_source_map_traversal_versions_size_and_wrong_bytes_rejected(self):
        files, mapping = source_fixture()
        for name in ('../foreign.py', '/absolute.py', 'evidence/raw.json'):
            bad = dict(mapping)
            bad[name] = '1' * 64
            with self.subTest(name=name), self.assertRaises(ValueError):
                m.verify_mapping(bad, len(bad))
        bad = dict(mapping)
        bad.pop(m.VERSION_FILES[0])
        with self.assertRaisesRegex(ValueError, 'version'):
            m.verify_mapping(bad, len(bad))
        with patch.object(m.subprocess, 'check_output', return_value='120000 blob ' + 'a' * 40 + '\t' + m.VERSION_FILES[0]), \
                self.assertRaisesRegex(ValueError, 'exact complete'):
            m.committed_sources(COMMIT, mapping, len(mapping))

    def test_actual_new_thirty_suite_cpu_zero_skip_source_binding(self):
        _, mapping = source_fixture()
        proof = cpu_fixture(mapping)
        m.verify_cpu(proof, COMMIT, mapping, len(m.CPU_MODULES))
        for key, value in (('source_commit', 'f' * 40), ('source_unchanged', False),
                ('actual_source_count', len(mapping) + 1), ('cpu_runner_after_sha256', '0' * 64),
                ('observation_timed_out', True), ('returncode', True), ('source_initial_sha256', {})):
            bad = copy.deepcopy(proof)
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                m.verify_cpu(bad, COMMIT, mapping, len(m.CPU_MODULES))

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
                m.verify_cpu(bad, COMMIT, mapping, len(m.CPU_MODULES))
        for stdout in ('{}', proof['stdout'] * 2):
            bad = dict(proof, stdout=stdout)
            with self.subTest(stdout=stdout[:20]), self.assertRaisesRegex(ValueError, 'Exactly one'):
                m.verify_cpu(bad, COMMIT, mapping, len(m.CPU_MODULES))
        bad = copy.deepcopy(proof)
        bad['modules'].pop()
        with self.assertRaises(ValueError):
            m.verify_cpu(bad, COMMIT, mapping, len(m.CPU_MODULES))

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

    def test_review_binds_actual_report_full_map_and_both_external_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            report = root / 'review.md'
            report.write_text('Constructed static review fixture; no actual review or execution admission.\n')
            names = ('submit_warmup57_measurement_native.py', 'test_submit_warmup57_measurement_native.py')
            for name in names:
                (root / name).write_text('# fixture\n')
            _, mapping = source_fixture()
            review = dict(role='warmup57_measurement_preexecution_review', approved=True,
                source_commit=COMMIT, source_sha256=mapping, target_cpu_proof_sha256='a' * 64,
                method_sha256='b' * 64, blocking_findings=[], scope='staticonly',
                physical_execution_authenticated=False, production_admission=False,
                healthy_checkpoint_accepted=False, actualreview_report_path=str(report),
                actualreview_report_sha256=m.sha(report.read_bytes()),
                external_file_sha256={name: m.sha((root / name).read_bytes()) for name in names})
            m.verify_review(review, COMMIT, mapping, 'a' * 64, 'b' * 64, root)
            for name in names:
                target = root / name
                original = target.read_bytes()
                target.write_bytes(b'changed after review')
                with self.assertRaisesRegex(ValueError, 'bytes changed'):
                    m.verify_review(review, COMMIT, mapping, 'a' * 64, 'b' * 64, root)
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


class NativeBindingTests(unittest.TestCase):
    def test_actual_parent_uuid_only_from_prepared_manifest(self):
        intent, observed = native_fixture()
        launch = m.actual_launch(intent, observed)
        self.assertEqual(launch['parent_run_id'], observed['parent-run-manifest.json']['run_id'])
        self.assertEqual(launch['execution_role'], m.ROLE)
        self.assertNotIn('plan_sha256', launch)
        for key in ('run-manifest.json', 'parent-run-manifest.json'):
            bad = copy.deepcopy(observed)
            bad.pop(key)
            with self.subTest(key=key), self.assertRaises(KeyError):
                m.actual_launch(intent, bad)
        bad = copy.deepcopy(observed)
        bad['run-manifest.json']['measurement_parent_run_id'] = 'f' * 32
        with self.assertRaises(ValueError):
            m.actual_launch(intent, bad)

    def test_new_science_and_initial_source_identity_cannot_relabel_old(self):
        intent, observed = native_fixture()
        for document, key, value in (
                ('run-manifest.json', 'policy_calibration_role', 'policy_safety_calibration_discovery'),
                ('run-manifest.json', 'source_commit', 'f' * 40),
                ('parent-run-manifest.json', 'completed_updates', 1)):
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
                                  '/dataset/shaoshitong/g1-warmup57-export-a1234567', 19876)
        self.assertIn(b"ROOT=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')", source)
        self.assertEqual(source.count(b'19876'), 2)
        with self.assertRaises(ValueError):
            m.exporter_bytes({'reproduction/common8/assets_export.py': original + b'port=19874'}, '/tmp/x', 19876)


class NoImplicitExecutionTests(unittest.TestCase):
    def test_remote_fragments_are_syntax_only_and_scoped(self):
        for name in ('OLD_READONLY_CODE', 'PREPARE_CODE', 'START_SOURCE_CODE', 'SUBMIT_CODE',
                     'OBSERVE_CODE', 'CLEANUP_SOURCE_CODE'):
            tree = ast.parse(getattr(m, name))
            self.assertTrue(tree.body, name)
        self.assertNotIn('ray stop', m.PREPARE_CODE + m.SUBMIT_CODE + m.CLEANUP_SOURCE_CODE)
        self.assertNotIn('scancel', m.PREPARE_CODE + m.SUBMIT_CODE + m.CLEANUP_SOURCE_CODE)
        self.assertIn("native-submit-claim.json", m.SUBMIT_CODE)
        self.assertLess(m.SUBMIT_CODE.index("write(b/'preparation/native-submit-claim.json'"),
                        m.SUBMIT_CODE.index('subprocess.run(args'))
        self.assertNotIn('calibration-300', m.PREPARE_CODE)

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
                      export_port=19876, execution_role=m.ROLE)
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


if __name__ == '__main__':
    unittest.main()
