"""Local CPU fixture tests of the repair launcher; no Slurm/GPU/native state proof.

All replacement allocation, ownership and native reports are explicit fixtures.
Only the wait tests spawn real local Python child processes. Production source
is compiled unchanged and no remote paths, training or signed records are touched.
"""
import copy
import builtins
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import checkpoint_repair_cpu as repair

SOURCE_SHA = 'a740a5106b7b8c338da294d14cd6437a711ac79e74b50b28770ed9814b5a60e2'
HERE = Path(__file__).resolve().parent
CASE_EVENTS = []


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode() + b'\n'


def unique(raw):
    def pairs(values):
        out = {}
        for key, value in values:
            if key in out:
                raise ValueError('Duplicate fixture JSON key')
            out[key] = value
        return out
    return json.loads(raw, object_pairs_hook=pairs)


def need(ok, message):
    if not ok:
        raise ValueError(message)


class FixtureLauncher:
    """Fixture physical ownership/allocation only; byte pins and writes stay real."""
    canonical = staticmethod(canonical)
    unique = staticmethod(unique)
    need = staticmethod(need)
    now = staticmethod(lambda: 'SYNTHETIC_LOCAL_CPU_FIXTURE_TIME')
    exit_status = staticmethod(lambda code: code if code >= 0 else 128 - code)

    def __init__(self):
        self.fail_process_write = False
        self.last_pid = None

    def physical(self, path, directory=False):
        p = Path(path)
        need(p.is_absolute() and p.resolve(strict=True) == p, 'Fixture physical path required')
        need(p.is_dir() if directory else p.is_file(), 'Fixture path kind differs')
        return p

    def pinned_bytes(self, path, expected):
        raw = self.physical(path).read_bytes()
        need(hashlib.sha256(raw).hexdigest() == expected, 'Fixture immutable pin differs')
        return raw

    def sha(self, path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def write(self, path, value):
        if self.fail_process_write and Path(path).name.endswith('-process-original.json'):
            raise ValueError('Intentional fixture process metadata failure')
        with Path(path).open('xb') as handle:
            handle.write(canonical(value))

    def process(self, pid, require_live=False):
        self.last_pid = pid
        return {'pid': pid, 'scope': 'SYNTHETIC_NODE_UID_SLURM_CONTEXT_NOT_REMOTE_OBSERVATION'}

    def allocation(self):
        return {'scope': 'SYNTHETIC_ALLOCATION_NOT_ACTUAL_JOB_INSPECTION'}


class FixtureCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.old_output = self.root / 'old-output'
        self.old_output.mkdir()
        self.old_marker = self.root / 'old-marker.json'
        self.old_marker.write_bytes(b'IMMUTABLE_ORIGINAL_FIXTURE_MARKER\n')
        (self.old_output / 'sentinel').write_bytes(b'IMMUTABLE_OLD_FIXTURE_FAILURE\n')
        self.output = self.root / 'new-output'
        self.marker = self.root / 'new-marker.json'
        self.launcher = FixtureLauncher()
        self.manifest_sha = 'f' * 64
        self.patches = [patch.object(repair, 'OUTPUT', self.output),
                        patch.object(repair, 'MARKER', self.marker),
                        patch.object(repair, 'ORIGINAL_OUTPUT', self.old_output),
                        patch.object(repair, 'ORIGINAL_MARKER', self.old_marker)]
        for item in self.patches:
            item.start()
        self.original_bytes = {self.old_marker: self.old_marker.read_bytes(),
                               self.old_output / 'sentinel': (self.old_output / 'sentinel').read_bytes()}

    def tearDown(self):
        for path, raw in self.original_bytes.items():
            self.assertEqual(path.read_bytes(), raw, 'An old fixture marker/output changed')
        for item in reversed(self.patches):
            item.stop()
        self.tmp.cleanup()

    def make_request(self):
        self.output.mkdir()
        self.launcher.write(self.marker, repair.request_value(self.manifest_sha))
        return self.launcher.sha(self.marker)


class RequestAndStartTests(FixtureCase):
    def test_source_and_exact_failure_lineage_constants(self):
        self.assertEqual(hashlib.sha256((HERE / 'checkpoint_repair_cpu.py').read_bytes()).hexdigest(), SOURCE_SHA)
        value = repair.request_value(self.manifest_sha)
        self.assertEqual(value['checkpoint_step'], 25)
        self.assertEqual(value['training_job_id'], '3261')
        self.assertEqual(value['original_request_sha256'], 'feac6e0c56a35e7725307b0d65c6f0437a0ae18bba13bea07e774721b1a2e88f')
        self.assertEqual(value['inventory_sha256'], '32f2c38c121b0c59ab26036c358b4fd749218f8e3980d6700f54c78fabbe6c4e')
        self.assertEqual(value['original_failed_exit_sha256'], repair.FAILURES)
        self.assertEqual(value['output_directory'], str(self.output))
        self.assertEqual(value['scope'], 'CPU25_FAILED_CHECKER_REPAIR_STORED_CONTENT_ONLY')

    def test_exact_request_read_and_bad_pin_rejected(self):
        pin = self.make_request()
        self.assertEqual(repair.read_request(self.launcher, self.manifest_sha, pin),
                         repair.request_value(self.manifest_sha))
        with self.assertRaisesRegex(ValueError, 'pin differs'):
            repair.read_request(self.launcher, self.manifest_sha, '0' * 64)

    def test_unknown_step_foreign_job_inventory_or_extra_request_rejected(self):
        self.output.mkdir()
        original = repair.request_value(self.manifest_sha)
        for key, bad in [('checkpoint_step', 50), ('training_job_id', '3210'),
                         ('inventory_sha256', '0' * 64), ('unexpected', 1)]:
            changed = copy.deepcopy(original)
            changed[key] = bad
            self.marker.write_bytes(canonical(changed))
            with self.assertRaisesRegex(ValueError, 'failure lineage'):
                repair.read_request(self.launcher, self.manifest_sha, self.launcher.sha(self.marker))

    def test_duplicate_json_request_rejected_even_hash_matches(self):
        self.output.mkdir()
        self.marker.write_bytes(b'{"checkpoint_step":25,"checkpoint_step":50}\n')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            repair.read_request(self.launcher, self.manifest_sha, self.launcher.sha(self.marker))

    def test_existing_marker_output_or_dangling_link_never_retried(self):
        self.marker.write_bytes(b'PREEXISTING_REPAIR_MARKER\n')
        with self.assertRaisesRegex(ValueError, 'Existing repair'):
            repair.start(self.launcher, self.manifest_sha)
        self.assertEqual(self.marker.read_bytes(), b'PREEXISTING_REPAIR_MARKER\n')
        self.marker.unlink()
        self.output.mkdir()
        with self.assertRaises(ValueError):
            repair.start(self.launcher, self.manifest_sha)
        self.output.rmdir()
        self.marker.symlink_to(self.root / 'missing-target')
        with self.assertRaises(ValueError):
            repair.start(self.launcher, self.manifest_sha)

    def test_start_fresh_local_supervisor_real_pid_and_duplicate_refusal(self):
        children = []
        real_popen = subprocess.Popen
        def spawn(*args, **kwargs):
            child = real_popen(*args, **kwargs)
            children.append(child)
            return child
        command = [sys.executable, '-c', 'import time; print("LOCAL_SUPERVISOR_FIXTURE",flush=True); time.sleep(.05)']
        with patch.object(repair, 'self_command', return_value=command), \
                patch.object(repair.subprocess, 'Popen', spawn):
            repair.start(self.launcher, self.manifest_sha)
        child = children[0]
        code = child.wait(timeout=5)
        record = unique((self.output / 'durable-launch-original.json').read_bytes())
        self.assertEqual(record['pid'], child.pid)
        self.assertGreater(record['pid'], 0)
        self.assertEqual(record['command'], command)
        self.assertEqual(code, 0)
        CASE_EVENTS.append(dict(case=self.id(), actual_local_pid=child.pid, actual_local_exit=code,
                                stdout=(self.output / 'supervisor-stdout-original.txt').read_text(),
                                scope='REAL_LOCAL_CHILD_WITH_SYNTHETIC_ALLOCATION_ONLY'))
        with self.assertRaisesRegex(ValueError, 'Existing repair'):
            repair.start(self.launcher, self.manifest_sha)

    def test_controller_cpu_srun_argv_and_actual_exit_forwarding(self):
        pin = self.make_request()
        commands = []
        def waiting(launcher, role, command):
            commands.append((role, command))
            return -15
        with patch.object(repair, 'bootstrap', return_value=(self.launcher, {})), \
                patch.object(repair, 'wait_child', waiting), \
                patch.object(sys, 'argv', ['repair', 'controller', '--tools-manifest-sha256',
                                          self.manifest_sha, '--request-sha256', pin]), \
                self.assertRaises(SystemExit) as outcome:
            repair.main()
        self.assertEqual(outcome.exception.code, 143)
        role, command = commands[0]
        self.assertEqual(role, 'srun')
        self.assertEqual(command[:8], ['srun', '--jobid=3261', '--overlap', '--nodes=1',
                                      '--ntasks=1', '--cpus-per-task=1', '--gres=none', '--nodelist=h100-10'])
        self.assertIn('CUDA_VISIBLE_DEVICES=', command)
        self.assertIn('OMP_NUM_THREADS=1', command)
        self.assertIn('PYTHONDONTWRITEBYTECODE=1', command)
        self.assertNotIn('sbatch', command)
        self.assertEqual(command[-1], pin)

    def test_unknown_cli_mode_rejected_before_bootstrap(self):
        with patch.object(repair, 'bootstrap') as bootstrap, \
                patch.object(sys, 'argv', ['repair', 'restart-training', '--tools-manifest-sha256', self.manifest_sha]), \
                self.assertRaises(SystemExit) as result:
            repair.main()
        self.assertEqual(result.exception.code, 2)
        bootstrap.assert_not_called()


class RealChildWaitTests(FixtureCase):
    def test_real_local_exit0_nonzero_and_sigterm_are_saved(self):
        self.output.mkdir()
        cases = [('zero', 'print("FIXTURE_EXIT_ZERO",flush=True)', 0),
                 ('nonzero', 'import sys; print("FIXTURE_EXIT_SEVEN",flush=True); sys.exit(7)', 7),
                 ('term', 'import os,signal; print("FIXTURE_SIGTERM",flush=True); os.kill(os.getpid(),signal.SIGTERM)', -signal.SIGTERM)]
        for role, code, expected in cases:
            returned = repair.wait_child(self.launcher, role, [sys.executable, '-c', code])
            record = unique((self.output / (role + '-exit-original.json')).read_bytes())
            self.assertEqual(returned, expected)
            self.assertEqual(record['returncode'], expected)
            self.assertEqual(record['pid'], self.launcher.last_pid)
            self.assertEqual(record['waited_by_pid'], os.getpid())
            with self.assertRaises(ChildProcessError):
                os.waitpid(record['pid'], os.WNOHANG)
            CASE_EVENTS.append(dict(case=self.id(), role=role, exit_record=record,
                                    stdout=(self.output / (role + '-stdout-original.txt')).read_text(),
                                    stderr=(self.output / (role + '-stderr-original.txt')).read_text(),
                                    scope='REAL_LOCAL_CHILD_WAIT_NOT_NATIVE_INSPECTION'))

    def test_metadata_failure_still_waits_and_never_completes(self):
        self.output.mkdir()
        self.launcher.fail_process_write = True
        with self.assertRaisesRegex(ValueError, 'metadata failure'):
            repair.wait_child(self.launcher, 'failedmeta', [sys.executable, '-c', 'print("WAITED_METADATA_FAILURE",flush=True)'])
        pid = self.launcher.last_pid
        self.assertGreater(pid, 0)
        with self.assertRaises(ChildProcessError):
            os.waitpid(pid, os.WNOHANG)
        self.assertFalse((self.output / 'inspection-completed-review-required.json').exists())
        # The root fix keeps actual exit evidence in finally, while preserving
        # the original metadata error and refusing a completion marker.
        exit_record = unique((self.output / 'failedmeta-exit-original.json').read_bytes())
        self.assertEqual(exit_record['returncode'], 0)
        self.assertEqual(exit_record['pid'], pid)
        CASE_EVENTS.append(dict(case=self.id(), actual_local_pid=pid, reaped=True,
                                stdout=(self.output / 'failedmeta-stdout-original.txt').read_text(),
                                exit_record=exit_record,
                                scope='REAL_LOCAL_CHILD_METADATA_FAILURE_REAPED_AND_EXIT_RECORDED'))


class InspectorBoundaryTests(FixtureCase):
    def setUp(self):
        super().setUp()
        self.request_sha = self.make_request()
        self.cgroup = '0::/SYNTHETIC/job_3261/step_fixture\n'
        self.context = {'job_id': '3261', 'step_id': 'SYNTHETIC_CPU_STEP',
                        'hostname': 'h100-10', 'process': {'cgroup_original': self.cgroup}}
        self.pipeline = types.SimpleNamespace(NATIVE_HELPER=self.root / 'fixture-helper',
            BASE=self.root / 'fixture-base', MODEL_SHA='a' * 64,
            execution_context=lambda _: dict(self.context))
        self.tools = {'source_commit': '188f4e5e7b63cf479290d0888a4a075499886750',
                      'tools': {'inspect_author200_checkpoint_cpu.py': 'b' * 64}}

    def synthetic_report(self):
        return {'scope': 'NATIVE_CPU_STORED_CHECKPOINT_CONTENT_INSPECTION', 'passed': True,
                'source_commit': self.tools['source_commit'], 'job_id': '3261',
                'checkpoint_step': 25, 'cuda_is_initialized': False,
                'inspector_source_sha256': 'b' * 64,
                'execution_identity': {'execution_job_id': '3261',
                    'execution_step_id': 'SYNTHETIC_CPU_STEP', 'pid': 123456,
                    'node': 'h100-10', 'uid': 0, 'uids': [0] * 4,
                    'start_ticks': 123, 'cgroup': self.cgroup}}

    def inspector_call(self, report=None, child_exit=0):
        def waiting(_, role, command):
            self.assertEqual(role, 'native')
            self.assertIn('--step', command)
            self.assertEqual(command[command.index('--step') + 1], '25')
            self.assertEqual(command[command.index('--inventory-sha256') + 1], repair.INVENTORY_SHA)
            if report is not None:
                self.launcher.write(self.output / 'native.json', report)
                self.launcher.write(self.output / 'native-process-original.json', {
                    'pid': 123456, 'process': {'present': True, 'start_ticks': '123',
                        'uids': [0] * 4, 'cgroup_original': self.cgroup}})
            return child_exit
        with patch.object(repair, 'module', return_value=self.pipeline), \
                patch.object(repair, 'wait_child', waiting), \
                patch.object(repair, 'bootstrap', return_value=(self.launcher, self.tools)):
            repair.inspect(self.launcher, self.tools, self.manifest_sha, self.request_sha)

    def test_pid_step_cuda_or_report_source_error_has_no_completion(self):
        fields = [('pid', 987654), ('execution_step_id', 'FOREIGN_CPU_STEP'),
                  ('execution_job_id', '3210'), ('uid', 501), ('uid', False),
                  ('start_ticks', 124), ('start_ticks', True),
                  ('uids', [False] * 4), ('node', 'h100-11'),
                  ('cgroup', '0::/FOREIGN/job_3210/step_1\n')]
        for key, bad in fields:
            with self.subTest(key=key):
                report = self.synthetic_report()
                report['execution_identity'][key] = bad
                with self.assertRaisesRegex(ValueError, 'execution identity'):
                    self.inspector_call(report)
                self.assertFalse((self.output / 'inspection-completed-review-required.json').exists())
                for path in self.output.iterdir():
                    path.unlink()
        report = self.synthetic_report()
        report['cuda_is_initialized'] = True
        with self.assertRaisesRegex(ValueError, 'identity/scope'):
            self.inspector_call(report)
        self.assertFalse((self.output / 'inspection-completed-review-required.json').exists())

    def test_nonzero_native_child_never_creates_completion(self):
        with self.assertRaisesRegex(ValueError, 'Actual repaired inspection failed'):
            self.inspector_call(child_exit=1)
        self.assertFalse((self.output / 'inspection-completed-review-required.json').exists())

    def test_zero_exit_without_report_is_not_completion(self):
        with self.assertRaises(FileNotFoundError):
            self.inspector_call(child_exit=0)
        self.assertFalse((self.output / 'inspection-completed-review-required.json').exists())

    def test_synthetic_complete_marker_still_requires_independent_review(self):
        self.inspector_call(self.synthetic_report())
        value = unique((self.output / 'inspection-completed-review-required.json').read_bytes())
        self.assertTrue(value['independent_acceptance_pending'])
        self.assertNotIn('passed', value)
        self.assertEqual(value['scope'], repair.SCOPE)
        self.assertEqual(value['original_failure_exits_preserved'], repair.FAILURES)


class BootstrapTests(unittest.TestCase):
    """Verified unchanged code is executed; runtime host/file owner are fixtures."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        (self.root / 'checkpoint_overlap.py').write_bytes((HERE / 'checkpoint_overlap.py').read_bytes())
        (self.root / 'checkpoint_repair_cpu.py').write_bytes((HERE / 'checkpoint_repair_cpu.py').read_bytes())

    def tearDown(self):
        self.tmp.cleanup()

    def test_bootstrap_wrong_owner_or_changed_launcher_never_executes(self):
        with patch.object(repair, 'HERE', self.root), patch.object(repair.os, 'getuid', return_value=501), \
                self.assertRaisesRegex(ValueError, 'physical owner'):
            repair.bootstrap('a' * 64)
        (self.root / 'checkpoint_overlap.py').write_bytes(b'raise AssertionError("MUST_NOT_EXECUTE")\n')
        with patch.object(repair, 'HERE', self.root), patch.object(repair.os, 'getuid', return_value=0), \
                self.assertRaisesRegex(ValueError, 'launcher bytes'):
            repair.bootstrap('a' * 64)

    def test_module_reader_sha_validation_precedes_execution(self):
        path = self.root / 'fixture-module.py'
        path.write_bytes(b'raise AssertionError("MUST_NOT_EXECUTE")\n')
        launcher = FixtureLauncher()
        with self.assertRaisesRegex(ValueError, 'pin differs'):
            repair.module(path, '0' * 64, launcher.pinned_bytes)

    def bootstrap_fixture(self, mutate=None):
        """Replay actual failed JSON/stream pins with explicit local owner fixtures.

        The old helper's already-reviewed request validation is a fixture
        authority; its module bytes are still hashed and compiled unchanged.
        The current bootstrap and current tools validator execute normally.
        """
        work = HERE.parent
        snap = work / 'actual-snapshots/3261-20261005T221857135211Z/original-files'
        old_output = self.root / 'old-original-output'
        old_output.mkdir()
        old_marker = self.root / 'old-marker.json'
        old_marker.write_bytes((snap / 'evidence/checkpoint-cpu-25-launch.json').read_bytes())
        inventory = old_output / 'inventory.json'
        inventory.write_bytes((snap / 'evidence/checkpoint-25-cpu-inspection-once-v1/observation/observed-checkpoint-inventory.json').read_bytes())
        failures = unique((work / 'checkpoint25-native-status-20261005T222408Z-original.json').read_bytes())['files']
        for name, pin in repair.FAILURES.items():
            raw = canonical(failures[name]['content'])
            self.assertEqual(hashlib.sha256(raw).hexdigest(), pin)
            (old_output / name).write_bytes(raw)
        stderr = unique((work / 'checkpoint25-cpu-failure-streams-original.json').read_bytes())['files']['native-stderr-original.txt']['content'].encode()
        self.assertEqual(hashlib.sha256(stderr).hexdigest(), repair.FAILURE_STDERR)
        (old_output / 'native-stderr-original.txt').write_bytes(stderr)
        old_tools = self.root / 'old-reviewed-tools'
        old_tools.mkdir()
        (old_tools / 'checkpoint_overlap.py').write_bytes((HERE / 'checkpoint_overlap.py').read_bytes())
        required = {'checkpoint_overlap.py', 'checkpoint_cpu_pipeline.py',
                    'observe_author200_checkpoint.py', 'inspect_author200_checkpoint_cpu.py',
                    'hf_author200_cpu.py', 'checkpoint_repair_cpu.py'}
        for name in required - {'checkpoint_overlap.py', 'checkpoint_repair_cpu.py'}:
            (self.root / name).write_bytes(b'EXPLICIT_LOCAL_SOFTWARE_FIXTURE_NOT_NATIVE_CONTENT\n')
        original = unique(old_marker.read_bytes())
        value = {'version': 1, 'source_commit': original['source_commit'],
                 'kind': 'AUTHOR200_EXTERNAL_CHECKPOINT_TOOLS',
                 'scope': 'AUTHOR200_CPU_STORED_CONTENT_INSPECTION_NOT_TRAINING_CONTROL',
                 'tools': {name: hashlib.sha256((self.root / name).read_bytes()).hexdigest() for name in required},
                 'dependencies': {}, 'contract': {'training_job_id': '3261', 'horizon': 200,
                    'checkpoint_steps': [25,50,75,100,125,150,175,200], 'global_mini_batch': 32,
                    'adam_steps_per_rank_per_outer': 16, 'warmup_steps': 0,
                    'learning_rate': 5e-6, 'cuda_work': False}}
        mp = self.root / 'manifest.json'
        mp.write_bytes(canonical(value))
        manifest_sha = hashlib.sha256(mp.read_bytes()).hexdigest()
        old_bytes = {p: p.read_bytes() for p in old_output.iterdir()}
        real_stat, real_fstat = Path.stat, os.fstat
        class FixtureRootOwner:
            def __init__(self, original): self.original = original
            def __getattr__(self, name): return 0 if name == 'st_uid' else getattr(self.original, name)
        def execute(code, namespace):
            builtins.exec(code, namespace)
            namespace['host'] = lambda: None
            namespace['physical'] = FixtureLauncher().physical
        def old_module(path, expected, reader):
            self.assertEqual(path, old_tools / 'checkpoint_overlap.py')
            self.assertEqual(expected, repair.LAUNCHER)
            raw = reader(path, expected)
            module = types.ModuleType('actual_old_helper_fixture_authority')
            module.__file__ = str(path)
            builtins.exec(compile(raw, str(path), 'exec'), module.__dict__)
            def old_request(path, expected):
                self.assertEqual(path, old_marker)
                self.assertEqual(expected, repair.ORIGINAL_REQUEST)
                parsed = unique(reader(path, expected))
                if mutate == 'wrong_old_step': parsed['checkpoint_step'] = 50
                if mutate == 'wrong_old_manifest': parsed['tools_manifest_sha256'] = '0' * 64
                return parsed, old_output
            module.load_request = old_request
            return module
        if mutate == 'inventory': inventory.write_bytes(b'MUTATED_LOCAL_INVENTORY\n')
        if mutate == 'previous_completed': (old_output / 'native.json').write_bytes(b'{}\n')
        if mutate == 'stderr': (old_output / 'native-stderr-original.txt').write_bytes(b'OTHER_FAILURE\n')
        if mutate == 'exit': (old_output / 'native-exit-original.json').write_bytes(b'{"returncode":0}\n')
        with patch.object(repair, 'HERE', self.root), patch.object(repair, 'OLD_TOOLS', old_tools), \
                patch.object(repair, 'ORIGINAL_MARKER', old_marker), \
                patch.object(repair, 'ORIGINAL_OUTPUT', old_output), patch.object(repair, 'INVENTORY', inventory), \
                patch.object(repair, 'exec', execute, create=True), patch.object(repair, 'module', old_module), \
                patch.object(repair.os, 'getuid', return_value=0), \
                patch.object(Path, 'stat', lambda path,*a,**k: FixtureRootOwner(real_stat(path,*a,**k))), \
                patch.object(repair.os, 'fstat', lambda fd: FixtureRootOwner(real_fstat(fd))):
            result = repair.bootstrap(manifest_sha)
        for p, raw in old_bytes.items():
            if mutate not in ('inventory', 'stderr', 'exit'):
                self.assertEqual(p.read_bytes(), raw)
        return result

    def test_current_bootstrap_accepts_actual_failure_pins_under_declared_local_fixtures(self):
        launcher, tools = self.bootstrap_fixture()
        self.assertEqual(tools['tools']['checkpoint_repair_cpu.py'], SOURCE_SHA)
        self.assertEqual(tools['contract']['cuda_work'], False)

    def test_current_bootstrap_rejects_failure_inventory_exit_stderr_and_prior_completion(self):
        for mutation in ('inventory', 'exit', 'stderr', 'previous_completed', 'wrong_old_step', 'wrong_old_manifest'):
            with self.subTest(mutation=mutation):
                with tempfile.TemporaryDirectory() as td:
                    original = self.root
                    self.root = Path(td).resolve()
                    (self.root / 'checkpoint_overlap.py').write_bytes((HERE / 'checkpoint_overlap.py').read_bytes())
                    (self.root / 'checkpoint_repair_cpu.py').write_bytes((HERE / 'checkpoint_repair_cpu.py').read_bytes())
                    try:
                        with self.assertRaises(ValueError):
                            self.bootstrap_fixture(mutation)
                    finally:
                        self.root = original

    def test_real_launcher_tools_manifest_rejects_unsafe_name_and_contract(self):
        path = self.root / 'checkpoint_overlap.py'
        module = repair.module(path, repair.LAUNCHER, FixtureLauncher().pinned_bytes)
        with patch.object(module, 'host'), patch.object(module, 'physical', FixtureLauncher().physical):
            required = {'checkpoint_overlap.py', 'checkpoint_cpu_pipeline.py',
                        'observe_author200_checkpoint.py', 'inspect_author200_checkpoint_cpu.py', 'hf_author200_cpu.py'}
            for name in required - {'checkpoint_overlap.py'}:
                (self.root / name).write_bytes(b'EXPLICIT_LOCAL_SOFTWARE_FIXTURE\n')
            value = {'version': 1, 'source_commit': module.SOURCE,
                     'kind': 'AUTHOR200_EXTERNAL_CHECKPOINT_TOOLS', 'scope': module.SCOPE,
                     'tools': {name: module.sha(self.root / name) for name in required},
                     'dependencies': {}, 'contract': {'training_job_id': '3261', 'horizon': 200,
                        'checkpoint_steps': list(module.STEPS), 'global_mini_batch': 32,
                        'adam_steps_per_rank_per_outer': 16, 'warmup_steps': 0,
                        'learning_rate': 5e-6, 'cuda_work': False}}
            mp = self.root / 'manifest.json'
            mp.write_bytes(canonical(value))
            self.assertEqual(module.tools(module.sha(mp)), value)
            with self.assertRaisesRegex(ValueError, 'SHA differs'):
                module.tools('0' * 64)
            for mutation in ('unsafe_name', 'horizon', 'cuda'):
                changed = copy.deepcopy(value)
                if mutation == 'unsafe_name': changed['tools']['../outside.py'] = 'c' * 64
                elif mutation == 'horizon': changed['contract']['horizon'] = 300
                else: changed['contract']['cuda_work'] = True
                mp.write_bytes(canonical(changed))
                with self.assertRaisesRegex(ValueError, 'Unsafe tool|contract differs'):
                    module.tools(module.sha(mp))


if __name__ == '__main__':
    unittest.main()
