"""Local synthetic CPU fixtures only. No SSH, Slurm, GPU or terminal claims."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import tracemalloc
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('stream_candidate', HERE / 'stream_actual3261_archive.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
transport, collector = m.load_helpers()


def fixture_manifest():
    archive = m.RUN / 'evidence/final-archive-local-synthetic-fixture'
    def saved(name, raw=b'fixture'):
        return dict(archive_path=str(archive / name), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    value = {key: {} for key in m.TOP_KEYS}
    value.update(version=1, kind='ACTUAL3261_ORIGINAL_FINAL_ARCHIVE_OBSERVATION', scope=m.SCOPE,
        source_commit=m.SOURCE, config_sha256=m.CONFIG, training_job_id='3261',
        source_binding_scope='EXPECTED_IDENTITY_FROM_REVIEWED_FD_BINDINGS_NOT_NEW_GIT_CLEAN_OR_CONFIG_VERIFICATION',
        run_root=str(m.RUN), hostname='h100-10', reader_uid=0, reader_pid=123,
        reviewed_fd_binding_sha256=m.BINDINGS_SHA, original_process_bindings=collector.expected_bindings(),
        release_accepted=False, training_or_effect_accepted=False, closed_writer_members=[],
        missing_or_unstable_members=[], archive_directory_identity=dict(device=1, inode=2, uid=0, mode=0o40700))
    value['files'] = {}
    for role, origin in collector.source_specs(collector.expected_bindings()).items():
        item = saved(role + '.original')
        value['files'][role] = dict(physical_path=str(origin), status='STABLE_FULL_BYTES_AT_CAPTURE',
            stable_full_bytes=True, phase='INFLIGHT_PREFIX', closed_writer_lifetime=False,
            attempted_archive_original=copy.deepcopy(item), **item)
    binding = saved('bindings-review.original.json')
    binding['sha256'] = m.BINDINGS_SHA
    value['bindings_review_original'] = binding
    value['watcher_terminal_original'] = dict(status='NOT_SUPPLIED_PENDING_OR_ABSENT', terminal_claim=False)
    value['process_scan_originals'] = {stage: saved(f'proc-{stage}.original.json') for stage in ('before', 'after')}
    for stage in ('before', 'after'):
        value['commands_' + stage] = {name: dict(argv=list(argv),
            stdout=saved(f'{stage}-{name}.stdout.original'), stderr=saved(f'{stage}-{name}.stderr.original'))
            for name, argv in collector.COMMANDS.items()}
    value['lease_lock_observations'] = [dict(status='RAW_CAPTURED', original=saved(stage + '-proc-locks.original'))
                                        for stage in ('before', 'after')]
    return value, archive / 'manifest.json'


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.value, self.path = fixture_manifest()

    def reject(self, mutate):
        value = copy.deepcopy(self.value)
        mutate(value)
        with self.assertRaises((ValueError, KeyError, TypeError)):
            m.validate_manifest(value, self.path, collector)

    def test_exact31_and_auxiliary_closure(self):
        copies, absent, identity = m.validate_manifest(self.value, self.path, collector)
        self.assertEqual(len(copies), 56)
        self.assertEqual(absent, [])
        self.assertEqual(identity['inode'], 2)

    def test_missing_does_not_create_original(self):
        role = 'process-finished-review-required.json'
        origin = self.value['files'][role]['physical_path']
        self.value['files'][role] = dict(physical_path=origin, status='MISSING_OR_DISAPPEARED',
            stable_full_bytes=False, phase='INFLIGHT_PREFIX', closed_writer_lifetime=False)
        self.value['missing_or_unstable_members'] = [role]
        copies, absent, _ = m.validate_manifest(self.value, self.path, collector)
        self.assertNotIn(role, copies)
        self.assertEqual(absent, [role])

    def test_unstable_attempt_copied_without_upgrade(self):
        role = 'resources.ndjson'
        item = self.value['files'][role]
        for key in ('archive_path', 'bytes', 'sha256'):
            del item[key]
        item.update(status='CAPTURE_FAILED_OR_UNSTABLE', stable_full_bytes=False)
        self.value['missing_or_unstable_members'] = [role]
        copies, absent, _ = m.validate_manifest(self.value, self.path, collector)
        self.assertIn(role, copies)
        self.assertEqual(absent, [])
        self.assertFalse(item['stable_full_bytes'])

    def test_optional_actual_watcher(self):
        saved = dict(archive_path=str(self.path.parent / 'watcher-terminal.original'), bytes=0,
                     sha256=hashlib.sha256(b'').hexdigest())
        self.value['watcher_terminal_original'] = dict(status='ACTUAL_PINNED_TERMINAL_ORIGINAL',
            physical_path=str(m.ROOT / 'evidence/terminal-inputs/job3261-slurm-final-fixture.original'),
            sha256=saved['sha256'], archived_original=saved)
        self.assertIn('watcher-terminal', m.validate_manifest(self.value, self.path, collector)[0])

    def test_wrong_source(self):
        self.reject(lambda v: v.update(source_commit='wrong'))
    def test_wrong_job(self):
        self.reject(lambda v: v.update(training_job_id='32610'))
    def test_wrong_config(self):
        self.reject(lambda v: v.update(config_sha256='0' * 64))
    def test_bool_version(self):
        self.reject(lambda v: v.update(version=True))
    def test_bool_uid(self):
        self.reject(lambda v: v.update(reader_uid=False))
    def test_wrong_host(self):
        self.reject(lambda v: v.update(hostname='h100-100'))
    def test_false_acceptance_typed(self):
        self.reject(lambda v: v.update(release_accepted=0))
    def test_unknown_top_key(self):
        self.reject(lambda v: v.update(fake_nonce='future'))
    def test_unknown_role(self):
        self.reject(lambda v: v['files'].update(extra=v['files']['train.log']))
    def test_missing_role(self):
        self.reject(lambda v: v['files'].pop('train.log'))
    def test_foreign_origin(self):
        self.reject(lambda v: v['files']['train.log'].update(physical_path='/tmp/wrong'))
    def test_foreign_copy(self):
        self.reject(lambda v: v['files']['train.log'].update(archive_path='/tmp/wrong'))
    def test_attempt_alias(self):
        self.reject(lambda v: v['files']['train.log']['attempted_archive_original'].update(
            archive_path=v['files']['resources.ndjson']['archive_path']))
    def test_bool_bytes(self):
        self.reject(lambda v: v['files']['train.log'].update(bytes=True))
    def test_negative_bytes(self):
        self.reject(lambda v: v['files']['train.log'].update(bytes=-1))
    def test_bad_sha(self):
        self.reject(lambda v: v['files']['train.log'].update(sha256='future'))
    def test_unknown_status(self):
        self.reject(lambda v: v['files']['train.log'].update(status='PASS'))
    def test_false_closure(self):
        self.reject(lambda v: v['files']['train.log'].update(closed_writer_lifetime=True))
    def test_mismatched_closed_list(self):
        self.reject(lambda v: v.update(closed_writer_members=['train.log']))
    def test_binding_pid(self):
        self.reject(lambda v: v['original_process_bindings'][0].update(pid=1))
    def test_binding_clock(self):
        self.reject(lambda v: v['original_process_bindings'][0].update(start_ticks='fake'))
    def test_command_origin(self):
        self.reject(lambda v: v['commands_before']['scontrol'].update(argv=['kill', '1']))
    def test_command_copy_path(self):
        self.reject(lambda v: v['commands_after']['sacct']['stderr'].update(archive_path='/tmp/fake'))
    def test_duplicate_json_key(self):
        with self.assertRaises(ValueError):
            m.unique_json(b'{"a":1,"a":2}')
    def test_nonfinite_json(self):
        for raw in (b'{"x":NaN}', b'{"x":1e1000}'):
            with self.assertRaises(ValueError):
                m.unique_json(raw)
    def test_manifest_foreign_and_traversal_paths(self):
        for path in ('/tmp/manifest.json', str(m.RUN / 'evidence/final-archive-../manifest.json'),
                     str(m.RUN / 'evidence/final-archive-fixture/../manifest.json')):
            with self.assertRaises(ValueError):
                m.validate_manifest_path(path)


class ActualLocalStreamTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name).resolve()
        self.source = self.base / 'archive.raw'
        self.output = self.base / 'copy.original'
        self.source.write_bytes(b'actual local synthetic binary\x00\xff\n' * 1000)

    def tearDown(self):
        self.temp.cleanup()

    def request(self):
        return dict(path=str(self.source), bytes=self.source.stat().st_size,
            sha256=hashlib.sha256(self.source.read_bytes()).hexdigest(), uid=os.getuid(),
            hostname=socket.gethostname().split('.')[0], directory_identity={
                k: m.metadata(self.base.stat())[k] for k in ('device', 'inode', 'uid', 'mode')})

    def run_reader(self, request=None, prefix='', suffix='', timeout=3):
        request = request or self.request()
        program = prefix + m.reader_program(request) + suffix
        return m.transfer_one([sys.executable, '-c', program], request, self.output, timeout)

    def assert_failed(self, result, rc=None):
        self.assertEqual(result['status'], 'FAILED_TRANSFER_ORIGINAL_BYTES_PRESERVED')
        if rc is not None:
            self.assertEqual(result['actual_returncode'], rc)
        self.assertTrue(Path(str(self.output) + '.transfer.json').is_file())
        self.assertTrue(Path(str(self.output) + '.transport.stderr.original').is_file())
        self.assertEqual(result['local_bytes'], self.output.stat().st_size)
        self.assertEqual(result['local_sha256'], hashlib.sha256(self.output.read_bytes()).hexdigest())

    def mutation_prefix(self, mutation):
        return f'''import os
_original_write=os.write
_did_mutation=False
def _fixture_write(fd,data):
 global _did_mutation
 count=_original_write(fd,data)
 if fd==1 and not _did_mutation:
  _did_mutation=True
  {mutation}
 return count
os.write=_fixture_write
'''

    def test_real_binary_success(self):
        result = self.run_reader()
        self.assertEqual(result['status'], 'TRANSFERRED_EXACT_ARCHIVED_BYTES_NOT_ACCEPTANCE')
        self.assertEqual(result['actual_returncode'], 0)
        self.assertEqual(self.output.read_bytes(), self.source.read_bytes())
        self.assertIn(m.CERT.encode(), Path(result['stderr_original']).read_bytes())

    def test_empty_success(self):
        self.source.write_bytes(b'')
        self.assertEqual(self.run_reader()['actual_returncode'], 0)

    def test_32mib_constant_memory_stream(self):
        with self.source.open('wb') as handle:
            for _ in range(32):
                handle.write(b'x' * 1024 * 1024)
        request = self.request()
        tracemalloc.start()
        try:
            result = self.run_reader(request, timeout=15)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertEqual(result['status'], 'TRANSFERRED_EXACT_ARCHIVED_BYTES_NOT_ACCEPTANCE')
        self.assertLess(peak, 8 * 1024 * 1024)

    def test_full_matching_bytes_nonzero_transport_still_fails(self):
        result = self.run_reader(suffix='\nraise SystemExit(23)\n')
        self.assert_failed(result, 23)
        self.assertEqual(result['local_sha256'], self.request()['sha256'])

    def test_real_sigterm_preserved(self):
        result = self.run_reader(suffix='\nimport signal\nos.kill(os.getpid(),signal.SIGTERM)\n')
        self.assert_failed(result, -15)

    def test_growth_matching_fullbody_then_metadata_failure(self):
        request = self.request()
        prefix = self.mutation_prefix(f"open({str(self.source)!r},'ab').write(b'growth')")
        result = self.run_reader(request, prefix=prefix)
        self.assert_failed(result, 1)
        self.assertEqual(result['local_sha256'], request['sha256'])

    def test_replace_matching_fullbody_then_metadata_failure(self):
        request = self.request()
        prefix = self.mutation_prefix(f"os.rename({str(self.source)!r},{str(self.source)+'.old'!r});open({str(self.source)!r},'wb').write(b'replaced')")
        result = self.run_reader(request, prefix=prefix)
        self.assert_failed(result, 1)
        self.assertEqual(result['local_sha256'], request['sha256'])

    def test_truncate_actual_partial_preserved(self):
        self.source.write_bytes(b'x' * (3 * 1024 * 1024))
        prefix = self.mutation_prefix(f"os.truncate({str(self.source)!r},0)")
        result = self.run_reader(prefix=prefix)
        self.assert_failed(result, 1)
        self.assertLess(result['local_bytes'], 3 * 1024 * 1024)

    def test_same_size_overwrite_failure(self):
        prefix = self.mutation_prefix(f"open({str(self.source)!r},'r+b').write(b'CHANGED')")
        self.assert_failed(self.run_reader(prefix=prefix), 1)

    def test_after_path_metadata_read_failure_even_fullsha_match(self):
        request = self.request()
        prefix = '''import os
_stat_original=os.stat
_calls=0
def _fixture_stat(*a,**k):
 global _calls
 _calls+=1
 if _calls==2:raise OSError('synthetic post metadata failure')
 return _stat_original(*a,**k)
os.stat=_fixture_stat
'''
        result = self.run_reader(request, prefix=prefix)
        self.assert_failed(result, 1)
        self.assertEqual(result['local_sha256'], request['sha256'])

    def test_wrong_pin_failure(self):
        request = self.request()
        request['sha256'] = '0' * 64
        self.assert_failed(self.run_reader(request), 1)

    def test_wrong_size_failure(self):
        request = self.request()
        request['bytes'] += 1
        self.assert_failed(self.run_reader(request), 1)

    def test_wrong_uid_failure(self):
        request = self.request()
        request['uid'] += 1
        self.assert_failed(self.run_reader(request), 1)

    def test_wrong_host_failure(self):
        request = self.request()
        request['hostname'] = 'foreign-host'
        self.assert_failed(self.run_reader(request), 1)

    def test_wrong_directory_identity(self):
        request = self.request()
        request['directory_identity']['inode'] += 1
        self.assert_failed(self.run_reader(request), 1)

    def test_symlink_source_rejected(self):
        request = self.request()
        self.source.rename(self.base / 'real')
        self.source.symlink_to(self.base / 'real')
        self.assert_failed(self.run_reader(request), 1)

    def test_hardlink_source_rejected(self):
        os.link(self.source, self.base / 'alias')
        self.assert_failed(self.run_reader(), 1)

    def test_parent_symlink_rejected(self):
        other = self.base / 'other'
        other.mkdir()
        self.source.rename(other / 'archive.raw')
        link = self.base / 'link'
        link.symlink_to(other, target_is_directory=True)
        self.source = link / 'archive.raw'
        self.assert_failed(self.run_reader(), 1)

    def test_short_local_transport_zero_exit_fails(self):
        request = self.request()
        result = m.transfer_one([sys.executable, '-c', "import os;os.write(1,b'short')"], request, self.output, 3)
        self.assert_failed(result, 0)

    def test_missing_certificate_zero_exit_fails(self):
        request = self.request()
        code = f"import os;os.write(1,open({str(self.source)!r},'rb').read())"
        result = m.transfer_one([sys.executable, '-c', code], request, self.output, 3)
        self.assert_failed(result, 0)

    def test_overlong_stream_rejected_and_actual_bytes_preserved(self):
        request = self.request()
        result = self.run_reader(request, suffix="\nos.write(1,b'extra')\n")
        self.assert_failed(result)
        self.assertGreater(result['local_bytes'], request['bytes'])

    def test_deadline_sigterm_owned_child_real_exit_preserved(self):
        request = self.request()
        code = "import os,time;os.write(1,b'partial');time.sleep(30)"
        result = m.transfer_one([sys.executable, '-c', code], request, self.output, 0.15)
        self.assert_failed(result, -15)
        self.assertEqual(self.output.read_bytes(), b'partial')

    def test_nooverwrite_existing_original(self):
        self.output.write_bytes(b'existing')
        result = self.run_reader()
        self.assertEqual(result['status'], 'FAILED_TRANSFER_ORIGINAL_BYTES_PRESERVED')
        self.assertIsNone(result['actual_returncode'])
        self.assertEqual(self.output.read_bytes(), b'existing')

    def test_nooverwrite_existing_stderr(self):
        stderr = Path(str(self.output) + '.transport.stderr.original')
        stderr.write_bytes(b'existing stderr')
        result = self.run_reader()
        self.assertIsNone(result['actual_returncode'])
        self.assertEqual(stderr.read_bytes(), b'existing stderr')

    def test_duplicate_report_not_replaced(self):
        report = Path(str(self.output) + '.transfer.json')
        report.write_bytes(b'original report')
        with self.assertRaises(FileExistsError):
            self.run_reader()
        self.assertEqual(report.read_bytes(), b'original report')

    def test_created_destination_replacement_detected(self):
        real_open = m.open_file
        swapped = False
        def intercept(path, uid, flags=os.O_RDONLY):
            nonlocal swapped
            if str(path) == str(self.output) and flags == os.O_RDONLY and not swapped:
                swapped = True
                self.output.rename(self.base / 'owned-created-original')
                self.output.write_bytes(b'foreign replacement')
            return real_open(path, uid, flags)
        with mock.patch.object(m, 'open_file', side_effect=intercept):
            result = self.run_reader()
        self.assertEqual(result['status'], 'FAILED_TRANSFER_ORIGINAL_BYTES_PRESERVED')
        self.assertEqual(result['actual_returncode'], 0)
        self.assertEqual((self.base / 'owned-created-original').read_bytes(), self.source.read_bytes())

    def test_fsync_failure_still_waits_and_records_real_exit(self):
        actual_fsync = os.fsync
        did_fail = False
        def once(fd):
            nonlocal did_fail
            if not did_fail:
                did_fail = True
                raise OSError('synthetic fsync failure')
            return actual_fsync(fd)
        with mock.patch.object(m.os, 'fsync', side_effect=once):
            result = self.run_reader()
        self.assert_failed(result, 0)

    def test_fresh_directory_no_reuse(self):
        path = self.base / 'fresh'
        m.fresh_directory(path)
        with self.assertRaises(FileExistsError):
            m.fresh_directory(path)

    def test_negative_or_bool_length_before_child(self):
        for length in (-1, True, False):
            request = self.request()
            request['bytes'] = length
            with self.assertRaises(ValueError):
                self.run_reader(request)
        self.assertFalse(self.output.exists())

    def test_zero_floating_or_bool_timeout_rejected_before_child(self):
        for timeout in (0, -1, True, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                self.run_reader(timeout=timeout)
        self.assertFalse(self.output.exists())

    def test_local_output_parent_replace_is_not_success(self):
        nested = self.base / 'owned-directory'
        nested.mkdir()
        self.output = nested / 'copy.original'
        request = self.request()
        # The fixture child replaces only its own synthetic local output parent
        # after producing the first raw block. No remote or training object exists.
        prefix = self.mutation_prefix(f"os.rename({str(nested)!r},{str(nested)+'.old'!r});os.mkdir({str(nested)!r})")
        result = self.run_reader(request, prefix=prefix)
        self.assertEqual(result['status'], 'FAILED_TRANSFER_ORIGINAL_BYTES_PRESERVED')
        self.assertEqual(result['actual_returncode'], 0)
        self.assertIn('directory_fsync_error', result)
        self.assertEqual((Path(str(nested) + '.old') / 'copy.original').read_bytes(), self.source.read_bytes())


if __name__ == '__main__':
    unittest.main(verbosity=2)
