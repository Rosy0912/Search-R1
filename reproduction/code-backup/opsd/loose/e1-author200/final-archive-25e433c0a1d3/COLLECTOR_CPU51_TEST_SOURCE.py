"""CPU-only fixtures. Never invoke production main, Slurm, GPU or network."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('final_collector_candidate',
        HERE / 'collect_actual3261_final_originals.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
ACTUAL_REVIEW = HERE.parent / 'ACTUAL3261_INFLIGHT_STDERR_SLURM_PREFIX_INDEPENDENT_REVIEW.json'


def scheduler(state='FAILED', exit_code='1:0'):
    return f'JobId=3261 JobState={state} NodeList=h100-10 UserId=root(0) ExitCode={exit_code}\n'.encode()


def proc_fixture(root, pid=123, start='123456', state='S', job='3261'):
    base = root / str(pid)
    base.mkdir()
    (base / 'fd').mkdir()
    (base / 'fdinfo').mkdir()
    fields = [state] + ['0'] * 18 + [start] + ['0'] * 4
    (base / 'stat').write_text(f'{pid} (name with ) parentheses) ' + ' '.join(fields) + '\n')
    (base / 'status').write_text('Name:\tfixture\nUid:\t0\t0\t0\t0\n')
    (base / 'cgroup').write_text(f'1:freezer:/slurm_h100-10/uid_0/job_{job}/step_0\n')
    return base


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.source = self.root / 'source.log'
        self.output = self.root / 'copy.original'
        self.source.write_bytes(b'first\x00raw\xffline\nsecond\n')

    def tearDown(self):
        self.temporary.cleanup()

    def capture(self, chunk=3):
        return c.copy_original(self.source, self.output, uid=os.getuid(), chunk_size=chunk)

    def mutate_on_source_read(self, operation):
        actual_read, source_stat = os.read, self.source.stat()
        called = [False]
        def reader(fd, count):
            current = os.fstat(fd)
            if (current.st_dev, current.st_ino) == (source_stat.st_dev, source_stat.st_ino) and not called[0]:
                result = actual_read(fd, count)
                called[0] = True
                operation()
                return result
            return actual_read(fd, count)
        return mock.patch.object(c.os, 'read', side_effect=reader)

    def test_full_binary_bytes_and_opened_fd_stable(self):
        record = self.capture()
        self.assertTrue(record['stable_full_bytes'])
        self.assertEqual(self.output.read_bytes(), self.source.read_bytes())
        self.assertEqual(record['sha256'], hashlib.sha256(self.source.read_bytes()).hexdigest())
        self.assertEqual(record['opened_fd_second_sha256'], record['sha256'])
        self.assertEqual(record['before'], record['opened_fd'])
        self.assertEqual(record['before'], record['after'])
        self.assertEqual(record['before'], record['path_after'])
        self.assertEqual(record['phase'], 'INFLIGHT_PREFIX')
        self.assertFalse(record['closed_writer_lifetime'])

    def test_zero_byte_stderr_is_valid_original(self):
        self.source.write_bytes(b'')
        record = self.capture()
        self.assertTrue(record['stable_full_bytes'])
        self.assertEqual(record['bytes'], 0)
        self.assertEqual(record['sha256'], hashlib.sha256(b'').hexdigest())

    def test_no_legacy_prefix_limit_large_stream(self):
        # A whole capture must cross both old 32MiB and 256MiB prefix limits.
        payload = bytes(range(256)) * 4096
        with self.source.open('wb') as handle:
            for _ in range(259):
                handle.write(payload)
        record = self.capture(chunk=1024 * 1024)
        self.assertTrue(record['stable_full_bytes'])
        self.assertEqual(record['bytes'], 259 * len(payload))
        self.assertEqual(record['sha256'], hashlib.sha256(self.output.read_bytes()).hexdigest())

    def test_missing_is_retained_and_no_fabricated_copy(self):
        self.source.unlink()
        record = self.capture()
        self.assertEqual(record['status'], 'MISSING_OR_DISAPPEARED')
        self.assertFalse(record['stable_full_bytes'])
        self.assertFalse(self.output.exists())

    def test_final_symlink_rejected(self):
        other = self.root / 'other'
        self.source.rename(other)
        self.source.symlink_to(other)
        self.assertFalse(self.capture()['stable_full_bytes'])
        self.assertFalse(self.output.exists())

    def test_parent_symlink_rejected(self):
        real = self.root / 'real'
        real.mkdir()
        (real / 'f').write_bytes(b'data')
        link = self.root / 'link'
        link.symlink_to(real, target_is_directory=True)
        record = c.copy_original(link / 'f', self.output, uid=os.getuid())
        self.assertFalse(record['stable_full_bytes'])
        self.assertFalse(self.output.exists())

    def test_source_uid_mismatch_is_rejected(self):
        record = c.copy_original(self.source, self.output, uid=os.getuid() + 1)
        self.assertFalse(record['stable_full_bytes'])

    def test_directory_and_parent_traversal_rejected(self):
        self.assertFalse(c.copy_original(self.root, self.output)['stable_full_bytes'])
        with self.assertRaises(ValueError):
            c.open_physical(self.root / '..' / self.root.name / 'source.log')

    def test_exclusive_destination_is_never_replaced(self):
        self.output.write_bytes(b'prior attempt must survive')
        record = self.capture()
        self.assertFalse(record['stable_full_bytes'])
        self.assertNotIn('attempted_archive_original', record)
        self.assertEqual(self.output.read_bytes(), b'prior attempt must survive')

    def test_source_growth_is_not_closed_or_full_final(self):
        with self.mutate_on_source_read(lambda: self.source.open('ab').write(b'new writer bytes')):
            record = self.capture()
        self.assertFalse(record['stable_full_bytes'])
        self.assertEqual(record['phase'], 'INFLIGHT_PREFIX')
        self.assertTrue(self.output.exists())
        self.assertEqual(record['attempted_archive_original']['bytes'], record['opened_fd']['bytes'])

    def test_same_length_overwrite_during_copy_rejected(self):
        with self.mutate_on_source_read(lambda: self.source.write_bytes(b'X' * self.source.stat().st_size)):
            record = self.capture()
        self.assertFalse(record['stable_full_bytes'])
        self.assertTrue(self.output.exists())

    def test_path_replacement_during_read_rejected(self):
        def replace():
            self.source.rename(self.root / 'old-inode')
            self.source.write_bytes(b'replacement same original path')
        with self.mutate_on_source_read(replace):
            record = self.capture()
        self.assertFalse(record['stable_full_bytes'])
        self.assertIn('attempted_archive_original', record)

    def test_replacement_between_precheck_and_open_rejected(self):
        actual_open = c.open_physical
        count = [0]
        def opener(path, *args, **kwargs):
            if Path(path) == self.source:
                count[0] += 1
                if count[0] == 2:
                    self.source.rename(self.root / 'old-before-open')
                    self.source.write_bytes(b'new before open')
            return actual_open(path, *args, **kwargs)
        with mock.patch.object(c, 'open_physical', side_effect=opener):
            record = self.capture()
        self.assertFalse(record['stable_full_bytes'])
        self.assertFalse(self.output.exists())

    def test_second_same_fd_hash_checks_more_than_file_metadata(self):
        actual_seek, actual_meta = os.lseek, c.metadata
        info = self.source.stat()
        old = actual_meta(info)
        def seek(fd, offset, how):
            result = actual_seek(fd, offset, how)
            if os.fstat(fd).st_ino == info.st_ino:
                self.source.write_bytes(b'X' * info.st_size)
            return result
        def same_metadata(value):
            # Deliberately model unchanged/replayed metadata: bytes must still differ.
            return old if value.st_ino == info.st_ino else actual_meta(value)
        with mock.patch.object(c.os, 'lseek', side_effect=seek), mock.patch.object(c, 'metadata', side_effect=same_metadata):
            record = self.capture()
        self.assertFalse(record['stable_full_bytes'])
        self.assertNotEqual(record['opened_fd_second_sha256'], record['sha256'])
        self.assertIn('Same FD full bytes changed', record['error'])

    def test_truncation_preserves_short_attempt(self):
        with self.mutate_on_source_read(lambda: self.source.write_bytes(b'')):
            record = self.capture()
        self.assertFalse(record['stable_full_bytes'])
        self.assertLess(record['attempted_archive_original']['bytes'], record['opened_fd']['bytes'])

    def test_output_is_private(self):
        self.capture()
        self.assertEqual(stat.S_IMODE(self.output.stat().st_mode), 0o600)

    def test_write_exclusive_and_fsync_not_overwrite(self):
        c.write_exclusive(self.output, b'first')
        with self.assertRaises(FileExistsError):
            c.write_exclusive(self.output, b'second')
        self.assertEqual(self.output.read_bytes(), b'first')

    def test_output_parent_symlink_rejected(self):
        real = self.root / 'real'
        real.mkdir()
        link = self.root / 'link'
        link.symlink_to(real, target_is_directory=True)
        with self.assertRaises(OSError):
            c.write_exclusive(link / 'new', b'bytes')
        self.assertFalse((real / 'new').exists())


class ContractsTests(unittest.TestCase):
    def test_actual_pinned_review_and_31_literal_paths(self):
        value, raw = c.pinned_json(ACTUAL_REVIEW, c.BINDINGS_SHA)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), c.BINDINGS_SHA)
        bindings = c.validate_bindings(value)
        specs = c.source_specs(bindings)
        self.assertEqual(len(specs), 31)
        self.assertEqual(specs['resources.ndjson'], c.RUN / 'resources.ndjson')
        self.assertEqual(sum(k.startswith('ray-pid-') for k in specs), 18)
        self.assertEqual(specs['slurm-formal-3261.err'], c.ROOT / 'logs/formal-3261.err')

    def test_binding_mutations_rejected(self):
        value = json.loads(ACTUAL_REVIEW.read_bytes())
        for mutate in (
            lambda v: v.update(version=True),
            lambda v: v.update(training_job_id='3210'),
            lambda v: v['final_exit_read_path_bindings'][0].update(pid=1533312.0),
            lambda v: v['final_exit_read_path_bindings'][0].update(start_ticks=1829085316),
            lambda v: v['final_exit_read_path_bindings'][0].update(stderr_fd_target='/tmp/guessed.err'),
            lambda v: v['final_exit_read_path_bindings'][0].update(cgroup=c.CGROUP.replace('3261', '3007')),
            lambda v: v['final_exit_read_path_bindings'].reverse(),
        ):
            changed = copy.deepcopy(value)
            mutate(changed)
            with self.assertRaises(ValueError):
                c.validate_bindings(changed)

    def test_duplicate_nonfinite_and_overflow_json_rejected(self):
        for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":1e999}'):
            with self.assertRaises(ValueError):
                c.unique_json(raw)

    def test_scheduler_failed_remains_failed(self):
        value = c.scheduler_fields(scheduler())
        self.assertTrue(value['terminal'])
        self.assertEqual(value['state'], 'FAILED')
        self.assertEqual(value['exit_code'], '1:0')

    def test_running_cannot_use_old_terminal_watcher(self):
        value = c.scheduler_observation({'status': 'COMMAND_RETURNED', 'returncode': 0},
                                        scheduler('RUNNING', '0:0'), scheduler())
        self.assertFalse(value['terminal'])
        self.assertEqual(value['state'], 'RUNNING')

    def test_successful_wrong_job_does_not_fallback(self):
        value = c.scheduler_observation({'status': 'COMMAND_RETURNED', 'returncode': 0},
                                        scheduler().replace(b'3261', b'9999'), scheduler())
        self.assertFalse(value['terminal'])

    def test_unavailable_scontrol_may_use_pinned_terminal_original(self):
        value = c.scheduler_observation({'status': 'COMMAND_RETURNED', 'returncode': 1}, b'', scheduler())
        self.assertTrue(value['terminal'])
        self.assertEqual(value['state'], 'FAILED')
        self.assertIn('WATCHER', value['origin'])

    def test_bool_and_float_command_returncodes_cannot_authorize_terminal_or_empty_query(self):
        for code in (False, True, 0.0, 1.0, '0', None):
            command = {'status': 'COMMAND_RETURNED', 'returncode': code}
            value = c.scheduler_observation(command, scheduler(), scheduler())
            self.assertFalse(value['terminal'])
            owners = c.command_owner_identities(b'', 'nvml', command=command)
            self.assertEqual(owners['status'], 'UNAVAILABLE_OR_UNPARSED')

    def test_scheduler_identity_and_exit_types_rejected(self):
        for raw in (scheduler().replace(b'h100-10', b'h100-8'),
                    scheduler().replace(b'root(0)', b'other(1000)'),
                    scheduler() + b'JobState=COMPLETED\n',
                    scheduler().replace(b'ExitCode=1:0', b'ExitCode=unknown')):
            with self.assertRaises(ValueError):
                c.scheduler_fields(raw)

    def test_only_fixed_read_only_commands_are_executable(self):
        self.assertEqual(set(c.COMMANDS), {'scontrol', 'sacct', 'nvml-inventory',
                         'nvml-applications-unfiltered', 'listeners-unfiltered'})
        for argv in c.COMMANDS.values():
            self.assertNotIn('ssh', argv)
            self.assertNotIn('scancel', argv)
            self.assertNotIn('flock', argv)
            self.assertNotIn('pkill', argv)
        self.assertNotIn('torch', c.__dict__)


class WriterLifetimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.proc = self.root / 'proc'
        self.proc.mkdir()
        self.target = self.root / 'log'
        self.target.write_bytes(b'log')
        self.info = c.physical_metadata(self.target)

    def tearDown(self):
        self.temporary.cleanup()

    def scan(self):
        return c.scan_processes_and_writers({'log': self.info}, [(123, '123456')], self.proc)

    def test_original_uid_and_start_are_observed(self):
        proc_fixture(self.proc)
        value = c.process_observation(123, '123456', self.proc)
        self.assertEqual(value['uids'], [0, 0, 0, 0])
        self.assertTrue(value['in_training_job'])
        self.assertEqual(value['status'], 'ORIGINAL_OR_CURRENT_PROCESS_ALIVE')

    def test_pid_reuse_is_original_gone_not_kill_target(self):
        proc_fixture(self.proc, start='654321', job='9999')
        value = self.scan()
        self.assertEqual(value['original_processes'][0]['status'], 'PID_REUSED_ORIGINAL_GONE')
        self.assertTrue(c.all_originals_gone(value))
        self.assertEqual(value['training_job_processes'], [])

    def test_zombie_means_exited_and_not_running(self):
        proc_fixture(self.proc, state='Z')
        self.assertTrue(c.all_originals_gone(self.scan()))

    def test_matching_start_with_foreign_uid_or_cgroup_does_not_prove_exit(self):
        base = proc_fixture(self.proc, state='Z', job='9999')
        self.assertFalse(c.all_originals_gone(self.scan()))
        (base / 'cgroup').write_text('1:freezer:/slurm_h100-10/uid_0/job_3261/step_0\n')
        (base / 'status').write_text('Uid:\t1000\t1000\t1000\t1000\n')
        self.assertFalse(c.all_originals_gone(self.scan()))

    def test_nvml_and_listener_owners_have_current_start_uid_cgroup(self):
        proc_fixture(self.proc, pid=999, start='999000', job='9999')
        nvml = c.command_owner_identities(b'GPU-foreign, 999, 123\n', 'nvml', self.proc)
        self.assertEqual(nvml['owners'][0]['start_ticks'], '999000')
        self.assertFalse(nvml['owners'][0]['in_training_job'])
        listener = c.command_owner_identities(b'LISTEN 127.0.0.1:18084 users:(("python",pid=999,fd=4))\n',
                                             'listeners', self.proc)
        self.assertEqual(listener['owners'][0]['start_ticks'], '999000')

    def test_invalid_nvml_query_not_empty_release_evidence(self):
        value = c.command_owner_identities(b'query failed', 'nvml', self.proc)
        self.assertEqual(value['status'], 'UNAVAILABLE_OR_UNPARSED')
        self.assertNotIn('released', value)
        failed = c.command_owner_identities(b'', 'nvml', self.proc,
                                           command={'status': 'COMMAND_RETURNED', 'returncode': 1})
        self.assertEqual(failed['status'], 'UNAVAILABLE_OR_UNPARSED')

    def test_missing_four_uid_does_not_prove_exit(self):
        base = proc_fixture(self.proc)
        (base / 'status').write_text('Uid: 0 0\n')
        value = self.scan()
        self.assertFalse(value['visibility_complete'])
        self.assertFalse(c.all_originals_gone(value))

    def test_foreign_process_write_fd_still_prevents_closure(self):
        base = proc_fixture(self.proc, pid=999, start='999000', job='9999')
        (base / 'fd/4').symlink_to(self.target)
        (base / 'fdinfo/4').write_text('pos:\t0\nflags:\t0100001\n')
        value = self.scan()
        self.assertTrue(c.all_originals_gone(value))
        self.assertEqual(value['writers']['log'][0]['pid'], 999)
        self.assertEqual(value['training_job_processes'], [])

    def test_readonly_fd_is_not_a_writer(self):
        base = proc_fixture(self.proc, pid=999, start='999000', job='9999')
        (base / 'fd/4').symlink_to(self.target)
        (base / 'fdinfo/4').write_text('flags:\t0100000\n')
        self.assertEqual(self.scan()['writers']['log'], [])

    def test_matching_fd_without_flags_is_unproven(self):
        base = proc_fixture(self.proc, pid=999, start='999000', job='9999')
        (base / 'fd/4').symlink_to(self.target)
        (base / 'fdinfo/4').write_text('pos:\t0\n')
        self.assertFalse(self.scan()['visibility_complete'])

    def test_proc_permission_error_cannot_prove_writer_absence(self):
        base = proc_fixture(self.proc, pid=999, start='999000', job='9999')
        actual_iterdir = Path.iterdir
        def iterate(path):
            if path == base / 'fd':
                raise PermissionError(13, 'permission denied', str(path))
            return actual_iterdir(path)
        with mock.patch.object(c.Path, 'iterdir', iterate):
            value = self.scan()
        self.assertFalse(value['visibility_complete'])
        self.assertTrue(value['errors'])

    def test_closure_requires_terminal_gone_inode_and_two_fd_scans(self):
        scan = self.scan()
        record = {'stable_full_bytes': True, 'initial_inode_matches_opened': True}
        terminal = c.scheduler_fields(scheduler())
        value = c.closure_for(record, 'log', scan, scan, terminal, terminal)
        self.assertEqual(value['phase'], 'CLOSED_WRITER_LIFETIME')
        for field in ('stable_full_bytes', 'initial_inode_matches_opened'):
            mutated = dict(record, **{field: False})
            self.assertFalse(c.closure_for(mutated, 'log', scan, scan, terminal, terminal)['closed_writer_lifetime'])
        for mutate in (
            lambda v: v.update(visibility_complete=False),
            lambda v: v['writers']['log'].append({'pid': 999}),
            lambda v: v['original_processes'][0].update(status='UNAVAILABLE_IDENTITY_NOT_EXIT_PROOF'),
            lambda v: v['training_job_processes'].append({'state': 'S'}),
        ):
            changed = copy.deepcopy(scan)
            mutate(changed)
            self.assertFalse(c.closure_for(record, 'log', scan, changed, terminal, terminal)['closed_writer_lifetime'])
        self.assertFalse(c.closure_for(record, 'log', scan, scan, terminal,
                                      c.scheduler_fields(scheduler('RUNNING', '0:0')))['closed_writer_lifetime'])

    def test_failed_training_can_still_have_closed_writer_bytes(self):
        scan = self.scan()
        value = c.closure_for({'stable_full_bytes': True, 'initial_inode_matches_opened': True},
                             'log', scan, scan, c.scheduler_fields(scheduler()), c.scheduler_fields(scheduler()))
        self.assertTrue(value['closed_writer_lifetime'])
        # Closure is byte lifetime only, never an experiment-success status.
        self.assertNotIn('passed', value)


class SeparateEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()

    def tearDown(self):
        self.temporary.cleanup()

    def record(self, name, raw):
        source = self.root / (name + '.source')
        output = self.root / (name + '.original')
        source.write_bytes(raw)
        return c.copy_original(source, output)

    def test_child_zero_wrapper_one_is_preserved_not_combined_success(self):
        files = {'evidence--train-process-exit.json': self.record('child', b'{"returncode":0}\n'),
                 'job-exit.txt': self.record('wrapper', b'1\n')}
        result = c.exit_layers(files)
        self.assertEqual(result['training_entry_process']['returncode'], 0)
        self.assertEqual(result['formal_wrapper_shell']['returncode'], 1)
        self.assertNotIn('passed', result)

    def test_bool_child_code_and_missing_finish_not_fabricated(self):
        files = {'evidence--train-process-exit.json': self.record('child', b'{"returncode":false}\n')}
        result = c.exit_layers(files)
        self.assertEqual(result['training_entry_process']['status'], 'PENDING_OR_UNAVAILABLE')
        self.assertNotIn('process-finished-review-required.json', files)

    def test_full_resource_sha_join(self):
        resource = self.record('resources', b'{"raw":1}\n{"raw":2}\n')
        files = {'resources.ndjson': resource,
                 'evidence--resource-data-audit.json': self.record('audit', json.dumps({'sha256': resource['sha256']}).encode())}
        self.assertEqual(c.exit_layers(files)['resource_audit_join']['status'], 'OBSERVED_FULL_BYTE_SHA_MATCH')

    def test_resource_prefix_sha_cannot_join_full_audit(self):
        files = {'resources.ndjson': self.record('resources', b'full final resources\n'),
                 'evidence--resource-data-audit.json': self.record('audit', b'{"sha256":"prefix"}')}
        self.assertEqual(c.exit_layers(files)['resource_audit_join']['status'], 'PENDING_OR_UNAVAILABLE')

    def test_all_lease_uuid_paths_derive_from_real_typed_original(self):
        leases = [{'fd': i + 3, 'uuid': u} for i, u in enumerate(sorted(c.UUIDS))]
        files = {'evidence--leases.json': self.record('leases', json.dumps(leases).encode())}
        paths = c.lease_paths(files)
        self.assertEqual(len(paths), 9)
        self.assertEqual(str(paths[-1]), '/mnt/data/shaoshitong/search-r1-rl/opsd/gpu-job.lock')

    def test_bad_lease_uuid_and_boolean_fd_reject(self):
        for fd, uuid in ((True, sorted(c.UUIDS)[0]), (3, 'GPU-foreign')):
            leases = [{'fd': i + 3, 'uuid': u} for i, u in enumerate(sorted(c.UUIDS))]
            leases[0] = {'fd': fd, 'uuid': uuid}
            files = {'evidence--leases.json': self.record('lease-' + str(fd) + uuid, json.dumps(leases).encode())}
            self.assertEqual(c.lease_paths(files), [])

    def test_lock_reads_do_not_acquire_any_lock(self):
        proc = self.root / 'proc'
        proc.mkdir()
        lock = self.root / 'lease.lock'
        lock.write_bytes(b'')
        info = lock.stat()
        proc_fixture(proc, start='123456')
        actual = f'1: FLOCK ADVISORY WRITE 123 {os.major(info.st_dev):02x}:{os.minor(info.st_dev):02x}:{info.st_ino} 0 EOF\n'
        (proc / 'locks').write_text(actual)
        result = c.lock_observation([lock], self.root, 'before', proc)
        self.assertEqual(result['paths'][0]['matching_original_lock_lines'], [actual.strip()])
        self.assertEqual((self.root / 'before-proc-locks.original').read_bytes(), actual.encode())
        self.assertEqual(result['current_lock_owner_identities'][0]['start_ticks'], '123456')
        self.assertNotIn('released', result)

    def test_read_only_command_nonzero_keeps_both_raw_streams(self):
        completed = subprocess.CompletedProcess(list(c.COMMANDS['scontrol']), 1,
                                               stdout=b'actual stdout\x00', stderr=b'accounting unavailable\xff')
        with mock.patch.object(c.subprocess, 'run', return_value=completed):
            result, raw = c.command_original('scontrol', self.root, 'before')
        self.assertEqual(result['returncode'], 1)
        self.assertEqual(raw, completed.stdout)
        self.assertEqual(Path(result['stderr']['archive_path']).read_bytes(), completed.stderr)

    def test_command_timeout_preserves_partial_raw(self):
        with mock.patch.object(c.subprocess, 'run', side_effect=subprocess.TimeoutExpired(
                list(c.COMMANDS['scontrol']), 30, output=b'partial', stderr=b'partial error')):
            result, raw = c.command_original('scontrol', self.root, 'before')
        self.assertEqual(result['status'], 'COMMAND_TIMED_OUT')
        self.assertIsNone(result['returncode'])
        self.assertEqual(raw, b'partial')

    def test_unlisted_command_cannot_execute(self):
        with mock.patch.object(c.subprocess, 'run') as run:
            with self.assertRaises(ValueError):
                c.command_original('ssh', self.root, 'before')
            run.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
