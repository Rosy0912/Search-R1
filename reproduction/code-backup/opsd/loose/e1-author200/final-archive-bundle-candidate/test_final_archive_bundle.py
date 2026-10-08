"""Local CPU-only constructor/publication-packet tests; no production entry."""
import base64
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import types
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent


def load(name):
    path = HERE / (name + '.py')
    value = types.ModuleType(name)
    value.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), 'exec'), value.__dict__)
    return value


f = load('freeze_final_archive_bundle')
p = load('publish_final_archive_bundle')
io, runtime_raw = f.bootstrap_runtime()
remote = types.ModuleType('portable_publisher_no_execution')
exec(compile(p.CODE, '<portable-publisher-fixture>', 'exec'), remote.__dict__)


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Actual fixed originals assembled locally, never frozen or deployed.
        cls.manifest, cls.members = f.construct(io, runtime_raw, HERE / 'FINAL_ARCHIVE_BUNDLE_PREPARATION_DESIGN.md')

    def packet(self):
        raw = f.canonical(self.manifest)
        return {'manifest_sha256': f.digest(raw), 'manifest_b64': base64.b64encode(raw).decode(),
                'members': {n: base64.b64encode(v).decode() for n, v in self.members.items()}}

    def test_real_source_and_collector_closure_count(self):
        self.assertEqual(self.manifest['tools'], {f.RUNTIME_NAME: f.RUNTIME})
        self.assertEqual(len(self.manifest['dependencies']), 31)
        self.assertEqual(len(self.manifest['review_records']), 21)
        self.assertEqual(self.manifest['contract']['original_driver_pid'], 1533312)
        self.assertEqual(self.manifest['contract']['original_driver_start_ticks'], '1829085316')
        self.assertFalse(self.manifest['contract']['requires_live_training_driver'])
        self.assertTrue(self.manifest['contract']['deployment_requires_original_driver_alive'])
        self.assertFalse(self.manifest['contract']['runtime_execution_authorized_by_freeze'])

    def test_real_original20_inodes_and11_prefixes(self):
        history = self.manifest['historical_file_bindings']
        self.assertEqual(len(history), 20)
        self.assertEqual(sum(v['historical_prefix_sha256'] is not None for v in history.values()), 11)
        self.assertEqual(history['ray-pid-1533312-stdout']['inode'], 7356335)
        self.assertEqual(history['ray-pid-1533312-stderr']['inode'], 7356336)
        self.assertEqual(history['slurm-formal-3261.err']['minimum_final_bytes'], 0)
        self.assertIsNone(history['ray-pid-1533312-stdout']['historical_prefix_sha256'])

    def test_history_wrong_original_start_rejected(self):
        raw = self.members['ACTUAL_FD_BINDINGS_SUMMARY_ORIGINAL.json']
        current = io.unique_json(raw)
        current['processes']['1533312']['start_ticks'] = 'new process'
        with self.assertRaises(ValueError):
            f.history_contract(io, f.canonical(current))

    def test_history_bool_uid_or_inode_rejected(self):
        raw = self.members['ACTUAL_FD_BINDINGS_SUMMARY_ORIGINAL.json']
        for mutate in (
            lambda v: v['processes']['1533312'].update(uids=[False, 0, 0, 0]),
            lambda v: v['processes']['1533312']['stdout']['before'].update(inode=True),
            lambda v: v['files']['slurm-formal-3261.err'].update(physical_path='/tmp/guessed.err'),
        ):
            changed = io.unique_json(raw)
            mutate(changed)
            with self.assertRaises(ValueError):
                f.history_contract(io, f.canonical(changed))

    def test_exact_packet_validates_without_publication(self):
        manifest, raw, members = remote.validate_packet(self.packet())
        self.assertEqual(manifest, self.manifest)
        self.assertEqual(members, self.members)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), self.packet()['manifest_sha256'])

    def test_packet_mutated_runtime_rejected(self):
        packet = self.packet()
        packet['members'][f.RUNTIME_NAME] = base64.b64encode(b'print("new runtime")').decode()
        with self.assertRaises(ValueError):
            remote.validate_packet(packet)

    def test_packet_missing_or_extra_member_rejected(self):
        packet = self.packet()
        packet['members'].pop(f.RUNTIME_NAME)
        with self.assertRaises(ValueError):
            remote.validate_packet(packet)
        packet = self.packet()
        packet['members']['../wrong.py'] = base64.b64encode(b'bad').decode()
        with self.assertRaises(ValueError):
            remote.validate_packet(packet)

    def test_manifest_wrong_job_driver_and_scope_rejected(self):
        for mutate in (
            lambda v: v['contract'].update(training_job_id='3210'),
            lambda v: v['contract'].update(original_driver_pid=1533312.0),
            lambda v: v['contract'].update(original_driver_start_ticks='other'),
            lambda v: v['contract'].update(requires_live_training_driver=True),
            lambda v: v['contract'].update(runtime_execution_authorized_by_freeze=True),
            lambda v: v.update(version=True),
        ):
            changed = copy.deepcopy(self.manifest)
            mutate(changed)
            raw = f.canonical(changed)
            packet = self.packet()
            packet.update(manifest_sha256=f.digest(raw), manifest_b64=base64.b64encode(raw).decode())
            with self.assertRaises(ValueError):
                remote.validate_packet(packet)

    def test_packet_duplicate_or_nonfinite_manifest_rejected(self):
        for raw in (b'{"version":1,"version":2}', b'{"value":1e999}'):
            packet = self.packet()
            packet.update(manifest_sha256=f.digest(raw), manifest_b64=base64.b64encode(raw).decode())
            with self.assertRaises(ValueError):
                remote.validate_packet(packet)

    def test_remote_program_contains_no_collector_or_job_entry_call(self):
        self.assertNotIn('c.main(', p.CODE)
        self.assertNotIn('scancel', p.CODE)
        self.assertNotIn('sbatch', p.CODE)
        self.assertNotIn('torch', p.CODE)
        self.assertNotIn('flock(', p.CODE)
        self.assertNotIn('os.kill(', p.CODE)
        self.assertNotIn('nvidia-smi', p.CODE)

    def test_portable_dependency_allowlist_equals_actual_constructed31(self):
        basepath = f.ROOT / 'tools/final-acceptance-90c2695a7f46/manifest.json'
        sourcepath = f.RUN / 'bindings/source.json'
        actual = {str(basepath): (f.W / 'final-acceptance-90c2695a7f46/manifest.json').read_bytes(),
                  str(sourcepath): (f.SNAP / 'bindings/source.json').read_bytes()}
        def reader(_, path, pin):
            raw = actual[str(path)]
            self.assertEqual(f.digest(raw), pin)
            return raw
        with mock.patch.object(remote, 'read', side_effect=reader):
            expected = remote.expected_dependencies(io)
        self.assertEqual(expected, self.manifest['dependencies'])
        self.assertNotIn('/root/.ssh/id_rsa', expected)

    def test_portable_deployment_guard_accepts_only_original_live_typed_allocation(self):
        original = (f.SNAP / 'bindings/allocation.json').read_bytes()
        allocation = io.unique_json(original)['scontrol'].encode()
        verified = {str(f.RUN / 'bindings/allocation.json'): original}
        identity = {'status': 'ORIGINAL_OR_CURRENT_PROCESS_ALIVE', 'uids': [0, 0, 0, 0],
                    'cgroup': io.CGROUP}
        def command(argv):
            if argv[-2:] == ['rev-parse', 'HEAD']:
                return (f.SOURCE + '\n').encode()
            if argv[-2:] == ['status', '--porcelain']:
                return b''
            self.assertEqual(argv, ['scontrol', 'show', 'job', '3261', '-o'])
            return allocation
        with mock.patch.object(remote.os, 'getuid', return_value=0), \
             mock.patch.object(remote.socket, 'gethostname', return_value='h100-10'), \
             mock.patch.object(remote.subprocess, 'check_output', side_effect=command), \
             mock.patch.object(io, 'process_observation', return_value=identity):
            result = remote.source_and_allocation(io, verified)
        self.assertEqual(result['scontrol_original'], allocation.decode())
        self.assertTrue(result['source_clean'])

    def test_portable_deployment_rejects_restart_terminal_or_reused_driver(self):
        original = (f.SNAP / 'bindings/allocation.json').read_bytes()
        initial = io.unique_json(original)['scontrol'].encode()
        verified = {str(f.RUN / 'bindings/allocation.json'): original}
        good = {'status': 'ORIGINAL_OR_CURRENT_PROCESS_ALIVE', 'uids': [0, 0, 0, 0], 'cgroup': io.CGROUP}
        for allocation, identity in (
            (initial.replace(b'Restarts=0', b'Restarts=1'), good),
            (initial.replace(b'JobState=RUNNING', b'JobState=COMPLETED'), good),
            (initial, dict(good, status='PID_REUSED_ORIGINAL_GONE')),
            (initial, dict(good, cgroup=io.CGROUP.replace('step_0', 'step_6'))),
            (initial + b' JobState=RUNNING', good),
            (initial, dict(good, uids=[False, 0, 0, 0])),
        ):
            def command(argv):
                if argv[-2:] == ['rev-parse', 'HEAD']:return (f.SOURCE + '\n').encode()
                if argv[-2:] == ['status', '--porcelain']:return b''
                return allocation
            with mock.patch.object(remote.os, 'getuid', return_value=0), \
                 mock.patch.object(remote.socket, 'gethostname', return_value='h100-10'), \
                 mock.patch.object(remote.subprocess, 'check_output', side_effect=command), \
                 mock.patch.object(io, 'process_observation', return_value=identity):
                with self.assertRaises(ValueError):remote.source_and_allocation(io, verified)

    def test_exact_transport_source_loads_without_network_or_pyc(self):
        with mock.patch('subprocess.run', side_effect=AssertionError('Network must not execute')):
            route = p.approved_transport(f, io)
        self.assertTrue(callable(route))
        self.assertEqual(f.digest((f.W / 'transport.py').read_bytes()), p.TRANSPORT)

    def test_modified_transport_rejected_before_code_execution(self):
        with mock.patch.object(f, 'read', side_effect=ValueError('Source pin differs')):
            with self.assertRaises(ValueError):p.approved_transport(f, io)

    def test_real_modified_freezer_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve()
            marker = directory / 'must-not-execute'
            (directory / 'freeze_final_archive_bundle.py').write_text('from pathlib import Path\nPath(%r).write_text("executed")\n' % str(marker))
            with mock.patch.object(p, '__file__', str(directory / 'publisher.py')):
                with self.assertRaises(ValueError):p.helpers()
            self.assertFalse(marker.exists())

    def test_missing_or_symlink_freezer_never_executes(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve()
            with mock.patch.object(p, '__file__', str(directory / 'publisher.py')):
                with self.assertRaises(FileNotFoundError):p.helpers()
                (directory / 'freeze_final_archive_bundle.py').symlink_to(HERE / 'freeze_final_archive_bundle.py')
                with self.assertRaises(OSError):p.helpers()

    def test_exact_freezer_same_fd_bytes_verified_before_exec(self):
        with mock.patch('subprocess.run', side_effect=AssertionError('No process may execute')):
            exact, collector = p.helpers()
        self.assertEqual(exact.RUNTIME, f.RUNTIME)
        self.assertEqual(collector.SOURCE, f.SOURCE)
        self.assertEqual(f.digest((HERE / 'freeze_final_archive_bundle.py').read_bytes()), p.FREEZER)


class FreezeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.parent = Path(self.temp.name).resolve()
        self.members = {'collect.py': b'original source', 'review.json': b'original review'}
        self.manifest = {'tools': {'collect.py': f.digest(self.members['collect.py'])},
                         'review_records': {'review.json': f.digest(self.members['review.json'])},
                         'dependencies': {}}

    def tearDown(self):
        for path in self.parent.glob('*'):
            if path.is_dir():
                path.chmod(0o700)
        self.temp.cleanup()

    def test_exclusive_manifest_last_and_exact_modes(self):
        result = f.freeze(io, self.parent, self.manifest, self.members)
        directory = Path(result['directory'])
        self.assertEqual(result['published_members'][-1], 'manifest.json')
        self.assertEqual(directory.stat().st_mode & 0o777, 0o555)
        self.assertTrue(all(p.stat().st_mode & 0o777 == 0o444 for p in directory.iterdir()))
        self.assertEqual((directory / 'collect.py').read_bytes(), self.members['collect.py'])
        self.assertEqual(f.digest((directory / 'manifest.json').read_bytes()), result['manifest_sha256'])

    def test_existing_bundle_never_overwritten(self):
        result = f.freeze(io, self.parent, self.manifest, self.members)
        before = {p.name: p.read_bytes() for p in Path(result['directory']).iterdir()}
        with self.assertRaises(FileExistsError):
            f.freeze(io, self.parent, self.manifest, self.members)
        self.assertEqual(before, {p.name: p.read_bytes() for p in Path(result['directory']).iterdir()})

    def test_member_hash_alias_or_unsafe_name_rejected_before_write(self):
        bad = dict(self.members, **{'../bad': b'escape'})
        badmanifest = copy.deepcopy(self.manifest)
        badmanifest['review_records']['../bad'] = f.digest(b'escape')
        with self.assertRaises(ValueError):
            f.freeze(io, self.parent, badmanifest, bad)
        changed = dict(self.members, **{'collect.py': b'modified source'})
        with self.assertRaises(ValueError):
            f.freeze(io, self.parent, self.manifest, changed)
        self.assertEqual(list(self.parent.iterdir()), [])

    def test_manifest_is_not_written_after_member_failure(self):
        real = io.write_exclusive
        def write(path, raw):
            if Path(path).name == 'review.json':
                raise OSError('simulated member failure')
            return real(path, raw)
        with mock.patch.object(io, 'write_exclusive', side_effect=write):
            with self.assertRaises(OSError):
                f.freeze(io, self.parent, self.manifest, self.members)
        directory = next(self.parent.iterdir())
        self.assertFalse((directory / 'manifest.json').exists())
        self.assertEqual((directory / 'collect.py').read_bytes(), self.members['collect.py'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
