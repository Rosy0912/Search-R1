"""Targeted local fixtures only. No collector main, Slurm, NVML or GPU call."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('collector3304_delta', HERE / 'collect_actual3304_final_originals.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
ACTUAL_MAP = HERE.parent / 'job3304-live-stdio18-binding-originals-v1/stdout.original'


def fake_proc(root, pid, start, state='S', cgroup=c.CGROUP):
    base = root / str(pid)
    base.mkdir()
    fields = [state] + ['0'] * 18 + [str(start)] + ['0'] * 4
    (base / 'stat').write_text(f'{pid} (fixture) ' + ' '.join(fields) + '\n')
    (base / 'status').write_text(f'Pid:\t{pid}\nUid:\t0\t0\t0\t0\n')
    (base / 'cgroup').write_text(cgroup)
    (base / 'fd').mkdir()
    (base / 'fdinfo').mkdir()
    return base


def closed_fixture(role='log'):
    scan = {'visibility_complete': True, 'original_processes': [{'status': 'ABSENT_AT_OBSERVATION'}],
            'training_job_processes': [], 'writers': {role: []}}
    scheduler = {'terminal': True, 'state': 'FAILED', 'exit_code': '1:0'}
    record = {'stable_full_bytes': True, 'initial_inode_matches_opened': True}
    return record, scan, scheduler


class Terminal3304Delta(unittest.TestCase):
    def test_actual18_origin_map_and_replacement_cannot_close(self):
        raw = ACTUAL_MAP.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), c.BINDINGS_SHA)
        value = c.unique_json(raw)
        bindings = c.validate_bindings(value)
        self.assertEqual(len(bindings), 9)
        self.assertEqual(len(c.source_specs(bindings)), 31)
        self.assertEqual(bindings[5]['pid'], 2549231)  # rank4 is not sorted PID order
        changed = copy.deepcopy(value)
        changed['rows']['2548319']['pid'] = True
        with self.assertRaises(ValueError):
            c.validate_bindings(changed)
        changed = copy.deepcopy(value)
        changed['rows']['2548319']['stdio']['1']['fd_stat']['inode'] += 1
        with self.assertRaises(ValueError):
            c.validate_bindings(changed)
        # Live append may change size/time; historical inode provenance remains fixed.
        changed = copy.deepcopy(value)
        changed['rows']['2548319']['stdio']['1']['after']['bytes'] += 1
        c.validate_bindings(changed)
        role = 'ray-pid-2548319-stdout'
        record, scan, scheduler = closed_fixture(role)
        expected = dict(bindings[0]['historical_fd_identity']['stdout_fd_target'])
        expected['nlink'] = expected.pop('link_count')
        record['opened_fd'] = expected
        c.bind_historical_origin(record, role, bindings)
        self.assertTrue(c.closure_for(record, role, scan, scan, scheduler, scheduler)['closed_writer_lifetime'])
        record['opened_fd'] = {**expected, 'inode': expected['inode'] + 1}
        c.bind_historical_origin(record, role, bindings)
        result = c.closure_for(record, role, scan, scan, scheduler, scheduler)
        self.assertFalse(result['closed_writer_lifetime'])
        self.assertIn('OPENED_RAY_INODE_DIFFERS_FROM_ACTUAL_HISTORICAL_FD_ORIGIN', result['closure_unproven_reasons'])

    def test_whole_resources_bad_row_and_unclosed_tail_are_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            source = root / 'resources.ndjson'
            raw = b'{"row":65}\n{"row":66,"in_slurm_job":false}\n{"row":67}\n{"partial":'
            source.write_bytes(raw)
            out = root / 'all.original'
            result = c.copy_original(source, out, os.getuid(), chunk_size=7)
            self.assertTrue(result['stable_full_bytes'])
            self.assertEqual(out.read_bytes(), raw)
            self.assertEqual(result['sha256'], hashlib.sha256(raw).hexdigest())
            self.assertFalse(result['closed_writer_lifetime'])
            hardlink = root / 'alias'
            os.link(source, hardlink)
            blocked = c.copy_original(source, root / 'blocked.original', os.getuid())
            self.assertFalse(blocked['stable_full_bytes'])
            self.assertIn('one physical link', blocked['error'])

    def test_absent_pids_do_not_replace_terminal_or_global_writer_proof(self):
        record, scan, terminal = closed_fixture()
        missing = {'terminal': False, 'state': None, 'exit_code': None}
        self.assertFalse(c.closure_for(record, 'log', scan, scan, missing, missing)['closed_writer_lifetime'])
        unknown = copy.deepcopy(scan)
        unknown['original_processes'][0]['status'] = 'UNAVAILABLE_IDENTITY_NOT_EXIT_PROOF'
        self.assertFalse(c.closure_for(record, 'log', unknown, scan, terminal, terminal)['closed_writer_lifetime'])
        writer = copy.deepcopy(scan)
        writer['writers']['log'] = [{'pid': 999999, 'fd': '8', 'identity': {'in_training_job': False}}]
        self.assertFalse(c.closure_for(record, 'log', writer, scan, terminal, terminal)['closed_writer_lifetime'])
        with tempfile.TemporaryDirectory() as temp:
            proc = Path(temp).resolve()
            fake_proc(proc, 7, 101)
            self.assertEqual(c.process_observation(7, '100', proc)['status'], 'PID_REUSED_ORIGINAL_GONE')
            fake_proc(proc, 8, 102, cgroup='1:freezer:/slurm_h100-10/uid_0/job_33040/step_0\n')
            self.assertEqual(c.process_observation(8, '102', proc)['status'], 'UNAVAILABLE_IDENTITY_NOT_EXIT_PROOF')

    def test_real_query_wait_and_fresh_running_not_overridden(self):
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp).resolve()
            argv = (sys.executable, '-c', 'import sys;sys.stdout.write("fixture-only");sys.exit(7)')
            with mock.patch.dict(c.COMMANDS, {'scontrol': argv}):
                command, raw = c.command_original('scontrol', archive, 'before')
            self.assertIs(command['waited'], True)
            self.assertEqual(command['returncode'], 7)
            self.assertEqual(raw, b'fixture-only')
            self.assertEqual(Path(command['stdout']['archive_path']).read_bytes(), raw)
        running = b'JobId=3304 NodeList=h100-10 UserId=root(0) JobState=RUNNING ExitCode=0:0 Restarts=0\n'
        watcher = running.replace(b'RUNNING', b'FAILED').replace(b'ExitCode=0:0', b'ExitCode=1:0')
        command = {'status': 'COMMAND_RETURNED', 'waited': True, 'returncode': 0}
        result = c.scheduler_observation(command, running, watcher)
        self.assertFalse(result['terminal'])
        self.assertEqual(result['origin'], 'FRESH_SCONTROL_ORIGINAL')
        with self.assertRaises(ValueError):
            c.scheduler_fields(watcher.replace(b'Restarts=0', b'Restarts=1'))

    def test_cgroup_raw_double_read_and_missing_not_terminal(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            group, archive = root / 'job', root / 'archive'
            group.mkdir()
            archive.mkdir()
            (group / 'step_0').mkdir()
            for directory in (group, group / 'step_0'):
                (directory / 'cgroup.procs').write_bytes(b'123\n456\n')
                (directory / 'tasks').write_bytes(b'123\n456\n789\n')
            result = c.cgroup_observation(archive, 'before', group)
            self.assertEqual(result['status'], 'FIXED_JOB_AND_IMMEDIATE_STEPS_OBSERVED')
            self.assertEqual(len(result['virtual_files']), 4)
            self.assertTrue(all(v['same_fd_two_reads_equal'] for v in result['virtual_files']))
            self.assertFalse(result['release_accepted'])
            absent = c.cgroup_observation(archive, 'after', root / 'absent')
            self.assertEqual(absent['status'], 'ABSENT_AT_OBSERVATION_NOT_SCHEDULER_TERMINAL_PROOF')

    def test_read_only_lock_and_literal_port_records_do_not_release(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            lock, proc, archive = root / 'lease.lock', root / 'proc', root / 'archive'
            lock.write_bytes(b'untouched')
            proc.mkdir()
            archive.mkdir()
            fake_proc(proc, 9, 100)
            info = lock.stat()
            raw = f'1: FLOCK ADVISORY WRITE 9 {os.major(info.st_dev):02x}:{os.minor(info.st_dev):02x}:{info.st_ino} 0 EOF\n'.encode()
            (proc / 'locks').write_bytes(raw)
            result = c.lock_observation([lock], archive, 'before', proc)
            self.assertEqual(result['paths'][0]['matching_original_lock_lines'], [raw.decode().strip()])
            self.assertEqual(lock.read_bytes(), b'untouched')
            self.assertEqual((proc / 'locks').read_bytes(), raw)
        command = {'status': 'COMMAND_RETURNED', 'waited': True, 'returncode': 0}
        raw = b'State Recv-Q Send-Q Local Peer Process\nLISTEN 0 128 127.0.0.1:18084 0.0.0.0:* users:(("fixture",pid=9,fd=3))\nLISTEN 0 128 *:16384 *:*\n'
        result = c.port_observation(command, raw)
        self.assertEqual(len(result['matching_original_lines']), 2)
        self.assertFalse(result['release_accepted'])

    def test_physical8_record_detects_rewrite_and_literal_child0_wrapper1(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            child, wrapper = root / 'child.json', root / 'wrapper.txt'
            child.write_bytes(b'{"returncode":0}')  # Ordinary JSON does not require LF.
            wrapper.write_bytes(b'1\n')
            files = {}
            for role, path in [('evidence--train-process-exit.json', child), ('job-exit.txt', wrapper)]:
                files[role] = c.copy_original(path, root / (path.name + '.original'), os.getuid())
            layers = c.exit_layers(files)
            self.assertEqual(layers['training_entry_process']['returncode'], 0)
            self.assertEqual(layers['formal_wrapper_shell']['returncode'], 1)
            record = c.source_physical_record(files['evidence--train-process-exit.json'])
            self.assertEqual(set(record), {'physical_path', 'sha256', 'bytes', 'device', 'inode', 'uid', 'mtime_ns', 'ctime_ns'})
            saved = c.write_exclusive(root / 'scheduler.original', b'literal')
            c.archive_physical_record(saved)
            Path(saved['archive_path']).write_bytes(b'changed')
            with self.assertRaises(ValueError):
                c.archive_physical_record(saved)

    def test_exclusive_archive_is_synced_before_output_and_never_retried(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            evidence = root / 'evidence'
            evidence.mkdir()
            path = evidence / 'final-archive-fixture'
            original_stat = Path.stat
            # Local test only: declare the production UID0 evidence-parent guard.
            def parent_uid_fixture(p, *args, **kwargs):
                value = original_stat(p, *args, **kwargs)
                if p == evidence:
                    fields = list(value)
                    fields[4] = 0
                    return os.stat_result(fields)
                return value
            synced = []
            actual_sync = c.fsync_directory
            def track(p):
                actual_sync(p)
                synced.append(p)
            with mock.patch.object(c, 'RUN', root), mock.patch.object(Path, 'stat', parent_uid_fixture), mock.patch.object(c, 'fsync_directory', track):
                c.create_archive(path)
                self.assertEqual(synced, [evidence, path])
                with self.assertRaises(FileExistsError):
                    c.create_archive(path)


if __name__ == '__main__':
    unittest.main()
