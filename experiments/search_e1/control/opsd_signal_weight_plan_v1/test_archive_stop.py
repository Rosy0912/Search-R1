"""CPU-only fixtures. No /proc scan, SSH, nvidia-smi or real signals."""
import contextlib
import copy
import datetime as dt
import importlib.util
import io
import json
from pathlib import Path
import signal
import struct
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('archive_stop', Path(__file__).with_name('archive_stop.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture():
    result = {}
    for pid, tick in {**m.SUPERVISORS, **m.ANCHORS, **m.PROTECTED}.items():
        env = {'E1_BASE': str(m.BASE), 'CUDA_VISIBLE_DEVICES': '4,5,6,7'}
        if pid in m.ANCHORS:
            env['E1_RUN_ID'] = m.RUN
        result[pid] = {'pid': pid, 'uid': m.UID, 'starttick': tick, 'state': 'S',
                       'ppid': 1, 'argv': str(m.BASE) + '/test-' + str(pid) + ' ',
                       'cwd': str(m.BASE) + '/runtime/archive', 'run_env': env}
    result[1432632]['ppid'] = 1405392
    result[1434631]['ppid'] = 1432632
    result[1434645]['ppid'] = 1434631
    result[1434651]['ppid'] = 1434645
    return result


def gpus_released():
    result = {}
    for i, uuid in m.GPU_UUIDS.items():
        used = 738 + 18 if i == 7 else 18
        result[i] = {'index': i, 'uuid': uuid, 'total_mib': 49140,
                     'used_mib': used, 'free_mib': 49140 - 379 - used}
    return result


def apps_released():
    return [{'uuid': m.GPU_UUIDS[7], 'pid': 1749337, 'name': 'oracle', 'mib': 738}]


class Selection(unittest.TestCase):
    def setUp(self):
        self.records = fixture()

    def test_exact_original_scope_and_protected_exclusion(self):
        found = m.select_owned(self.records, self.records)
        self.assertEqual(set(found), set(m.SUPERVISORS) | set(m.ANCHORS))
        self.assertFalse(set(found) & set(m.PROTECTED))

    def test_foreign_uid_and_other_run_not_descendant_ignored(self):
        table = copy.deepcopy(self.records)
        table[999] = dict(table[1434651], pid=999, uid=1020)
        table[1000] = dict(table[1434651], pid=1000, run_env={'E1_RUN_ID': 'other'}, ppid=1)
        self.assertNotIn(999, m.select_owned(table, self.records))
        self.assertNotIn(1000, m.select_owned(table, self.records))

    def test_known_pid_reuse_rejected(self):
        table = copy.deepcopy(self.records)
        table[1434651]['starttick'] += 1
        with self.assertRaisesRegex(m.Hold, 'identity changed'):
            m.select_owned(table, self.records)

    def test_exact_argv_changed_rejected(self):
        table = copy.deepcopy(self.records)
        table[1432632]['argv'] += '--other'
        with self.assertRaises(m.Hold):
            m.select_owned(table, self.records)

    def test_true_owned_infrastructure_descendant(self):
        table = copy.deepcopy(self.records)
        table[2000] = dict(table[1434651], pid=2000, ppid=1434651, starttick=999,
                           run_env={}, argv=str(m.BASE) + '/env/bin/python rayhelper')
        self.assertIn(2000, m.select_owned(table, self.records))

    def test_descendant_other_run_holds_without_signal(self):
        table = copy.deepcopy(self.records)
        table[2000] = dict(table[1434651], pid=2000, ppid=1434651, starttick=999,
                           run_env={'E1_RUN_ID': 'a-new-opsd-run'})
        with self.assertRaisesRegex(m.Hold, 'another E1 run'):
            m.select_owned(table, self.records)

    def test_unattributed_descendant_holds(self):
        table = copy.deepcopy(self.records)
        table[2000] = dict(table[1434651], pid=2000, ppid=1434651, starttick=999,
                           run_env={}, cwd='/tmp/foreign', argv='python unrelated')
        with self.assertRaisesRegex(m.Hold, 'Unattributed'):
            m.select_owned(table, self.records)

    def test_orphan_same_run_still_selected(self):
        table = copy.deepcopy(self.records)
        table[2000] = dict(table[1434651], pid=2000, ppid=1, starttick=999)
        self.assertIn(2000, m.select_owned(table, self.records))

    def test_marker_wrong_gpu_holds(self):
        table = copy.deepcopy(self.records)
        table[1434651]['run_env']['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
        with self.assertRaisesRegex(m.Hold, 'GPU scope'):
            m.select_owned(table, self.records)

    def test_protected_replacement_holds(self):
        table = copy.deepcopy(self.records)
        table[1749337]['starttick'] += 1
        with self.assertRaisesRegex(m.Hold, 'Protected'):
            m.check_protected(table, self.records)

    def test_ray_actor_method_title_change_is_not_pid_reuse(self):
        expected = copy.deepcopy(self.records)
        expected[1437900]['argv'] = 'ray::WorkerDict.actor_rollout_generate_sequences  '
        current = copy.deepcopy(expected)
        current[1437900]['argv'] = 'ray::WorkerDict.actor_rollout_update_actor  '
        self.assertIn(1437900, m.select_owned(current, expected))
        current[1437900]['run_env']['E1_RUN_ID'] = 'other'
        with self.assertRaises(m.Hold):
            m.select_owned(current, expected)

    def test_unrelated_opaque_own_environment_is_ignored(self):
        table = copy.deepcopy(self.records)
        table[2001] = dict(table[1434651], pid=2001, ppid=1, starttick=999,
                           env_readable=False, run_env={})
        self.assertNotIn(2001, m.select_owned(table, self.records))
        table[2001]['ppid'] = 1434651
        with self.assertRaisesRegex(m.Hold, 'Opaque environment'):
            m.select_owned(table, self.records)

    def test_reparented_captured_helper_remains_owned(self):
        table = copy.deepcopy(self.records)
        old = dict(table[1434651], pid=2000, ppid=1434651, starttick=999, run_env={})
        captured = {2000: old}
        table[2000] = dict(old, ppid=1)
        self.assertNotIn(2000, m.select_owned(table, self.records))
        remaining, ended = m.remaining_owned(table, self.records, captured)
        self.assertIn(2000, remaining)
        self.assertFalse(ended)

    def test_reused_captured_helper_not_signaled_as_original(self):
        table = copy.deepcopy(self.records)
        old = dict(table[1434651], pid=2000, ppid=1434651, starttick=999, run_env={})
        table[2000] = dict(old, ppid=1, starttick=1000)
        remaining, ended = m.remaining_owned(table, self.records, {2000: old})
        self.assertNotIn(2000, remaining)
        self.assertEqual(ended[0]['old_identity'], 'PID_REUSED_NOT_SIGNALED')

    def test_opaque_or_missing_tick_historical_target_holds(self):
        old = dict(self.records[1434651], pid=2000, ppid=1434651, starttick=999, run_env={})
        for actual in ({'pid': 2000, 'opaque_identity': True},
                       {'pid': 2000, 'uid': m.UID, 'state': 'S'},
                       {'pid': 2000, 'uid': m.UID, 'state': 'Z'},
                       {'pid': 2000, 'uid': m.UID, 'starttick': None},
                       {'pid': 2000, 'uid': m.UID, 'starttick': '1000'},
                       {'pid': 2000, 'uid': m.UID, 'starttick': True}):
            with self.subTest(actual=actual):
                table = copy.deepcopy(self.records)
                table[2000] = actual
                with self.assertRaisesRegex(m.Hold, 'Opaque/missing starttick'):
                    m.remaining_owned(table, self.records, {2000: old})

    def test_nonstrict_known_missing_tick_is_not_pid_reuse(self):
        table = copy.deepcopy(self.records)
        table[1434651] = {'pid': 1434651, 'opaque_identity': True}
        with self.assertRaisesRegex(m.Hold, 'Opaque/missing starttick'):
            m.select_owned(table, self.records, strict_known=False)


class Pidfd(unittest.TestCase):
    def setUp(self):
        self.r = fixture()[1434651]
        self.events = []
        self.closed = []

    def make(self, actual):
        return m.ExactSignals(reader=lambda pid: actual,
                              opener=lambda pid, flags: 100,
                              sender=lambda fd, sig, info, flags: self.events.append((fd, sig)),
                              closer=self.closed.append)

    def test_send_uses_only_verified_pidfd(self):
        h = self.make(self.r)
        h.send(self.r, signal.SIGTERM)
        self.assertEqual(self.events, [(100, signal.SIGTERM)])
        h.close()
        self.assertEqual(self.closed, [100])

    def test_pid_reuse_between_open_and_check_is_not_signaled(self):
        changed = dict(self.r, starttick=self.r['starttick'] + 1)
        with self.assertRaisesRegex(m.Hold, 'opening pidfd'):
            self.make(changed).send(self.r, signal.SIGTERM)
        self.assertFalse(self.events)
        self.assertEqual(self.closed, [100])

    def test_protected_pid_denied_before_open(self):
        with self.assertRaisesRegex(m.Hold, 'Denied'):
            self.make(fixture()[1749337]).send(fixture()[1749337], signal.SIGTERM)
        self.assertFalse(self.events)
        self.assertFalse(self.closed)

    def test_second_signal_detects_reuse(self):
        actual = copy.deepcopy(self.r)
        h = self.make(actual)
        h.send(self.r, signal.SIGSTOP)
        actual['starttick'] += 1
        with self.assertRaisesRegex(m.Hold, 'reused'):
            h.send(self.r, signal.SIGCONT)
        self.assertEqual(len(self.events), 1)


class ProcIdentity(unittest.TestCase):
    def test_effective_uid_is_used_and_foreign_environment_never_read(self):
        with tempfile.TemporaryDirectory() as td:
            proc = Path(td)
            p = proc / '7'
            p.mkdir()
            fields = ['S', '1', '7', '7'] + ['0'] * 15 + ['12345']
            (p / 'stat').write_text('7 (worker) ' + ' '.join(fields))
            (p / 'status').write_text('Name:\tworker\nUid:\t1001\t1020\t1020\t1020\n')
            # Deliberately no environ/cmdline files: foreign branch must not read them.
            with patch.object(m, 'Path', side_effect=lambda value: proc if value == '/proc' else Path(value)):
                r = m.read_process(7)
            self.assertEqual(r['uid'], 1020)
            self.assertEqual(r['real_uid'], 1001)
            self.assertTrue(r['opaque_foreign'])

    def test_effective_owner_can_differ_from_real_uid(self):
        with tempfile.TemporaryDirectory() as td:
            proc = Path(td)
            p = proc / '7'
            p.mkdir()
            fields = ['S', '1', '7', '7'] + ['0'] * 15 + ['12345']
            (p / 'stat').write_text('7 (worker) ' + ' '.join(fields))
            (p / 'status').write_text('Name:\tworker\nUid:\t1020\t1001\t1001\t1001\n')
            (p / 'cmdline').write_bytes(b'python\0owned\0')
            (p / 'environ').write_bytes(b'E1_RUN_ID=' + m.RUN.encode() + b'\0')
            (p / 'cwd').symlink_to(m.BASE / 'runtime/archive')
            with patch.object(m, 'Path', side_effect=lambda value: proc if value == '/proc' else Path(value)):
                r = m.read_process(7)
            self.assertEqual(r['uid'], 1001)
            self.assertEqual(r['real_uid'], 1020)
            self.assertTrue(r['env_readable'])


class GPU(unittest.TestCase):
    def test_direct_free_reserved_accounting(self):
        rows = m.check_gpu_ownership(gpus_released(), apps_released(), {}, released=True)
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]['reserved_mib'], 379)
        self.assertGreater(rows[0]['minimum_direct_free_mib'], 48000)

    def test_old_context_blocks_release(self):
        apps = apps_released() + [{'uuid': m.GPU_UUIDS[4], 'pid': 1437900,
                                   'name': 'oldworker', 'mib': 100}]
        with self.assertRaisesRegex(m.Hold, 'context remains'):
            m.check_gpu_ownership(gpus_released(), apps, {1437900: {}}, released=True)

    def test_foreign_selected_gpu_blocks(self):
        apps = apps_released() + [{'uuid': m.GPU_UUIDS[4], 'pid': 102020, 'mib': 100}]
        with self.assertRaisesRegex(m.Hold, 'Foreign compute'):
            m.check_gpu_ownership(gpus_released(), apps, {})

    def test_foreign_gpu_zero_untouched(self):
        apps = apps_released() + [{'uuid': 'foreign-GPU0', 'pid': 102020, 'mib': 33000}]
        self.assertEqual(len(m.check_gpu_ownership(gpus_released(), apps, {}, released=True)), 4)

    def test_hidden_memory_blocks(self):
        gs = gpus_released()
        gs[4]['used_mib'] += 500
        gs[4]['free_mib'] -= 500
        with self.assertRaisesRegex(m.Hold, 'not fully released'):
            m.check_gpu_ownership(gs, apps_released(), {}, released=True)

    def test_oracle_cannot_be_lost_or_excessive(self):
        with self.assertRaisesRegex(m.Hold, 'oracle'):
            m.check_gpu_ownership(gpus_released(), [], {}, released=True)
        apps = apps_released()
        apps[0]['mib'] = 2048
        with self.assertRaises(m.Hold):
            m.check_gpu_ownership(gpus_released(), apps, {}, released=True)


class Evidence(unittest.TestCase):
    def checkpoint(self, base, step, complete=True):
        p = base / 'runs/archive/round-1-grpo/actor' / ('global_step_' + str(step))
        p.mkdir(parents=True)
        names = ('added_tokens.json config.json generation_config.json merges.txt '
                 'model-00001-of-00003.safetensors model-00002-of-00003.safetensors '
                 'model-00003-of-00003.safetensors model.safetensors.index.json '
                 'special_tokens_map.json tokenizer.json tokenizer_config.json vocab.json').split()
        for name in names if complete else names[:5]:
            (p / name).write_bytes(b'CPUfixture')
        if complete:
            (p / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {
                'x': names[4], 'y': names[5], 'z': names[6]}}))
            for key, name in zip(('x', 'y', 'z'), names[4:7]):
                header = json.dumps({key: {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}}).encode()
                (p / name).write_bytes(struct.pack('<Q', len(header)) + header + b'\0' * 4)
        return p

    def test_preserve150_and_partial200_without_resume_claim(self):
        with tempfile.TemporaryDirectory() as td:
            b = Path(td) / 'base'
            cp = self.checkpoint(b, 150)
            partial = self.checkpoint(b, 200, complete=False)
            (b / 'receipts').mkdir()
            (b / 'receipts' / 'observer.json').write_text('{}')
            d = b / 'preserved' / 'fixture'
            d.mkdir(parents=True)
            result = m.preserve(b, d)
            self.assertEqual(result['latest_layout_valid_checkpoint_step'], 150)
            self.assertFalse(result['complete_Adam_resume_state'])
            self.assertFalse(result['checkpoints'][-1]['structurally_complete_12_file_checkpoint'])
            self.assertTrue(cp.exists() and partial.exists())
            self.assertTrue((d / 'copies/receipts/observer.json').exists())

    def test_complete200_is_latest_actual_directory(self):
        with tempfile.TemporaryDirectory() as td:
            b = Path(td) / 'base'
            self.checkpoint(b, 150)
            self.checkpoint(b, 200)
            d = Path(td) / 'out'
            d.mkdir()
            self.assertEqual(m.preserve(b, d)['latest_layout_valid_checkpoint_step'], 200)

    def test_nonempty_truncated200_is_not_complete(self):
        with tempfile.TemporaryDirectory() as td:
            b = Path(td) / 'base'
            self.checkpoint(b, 150)
            p = self.checkpoint(b, 200)
            q = p / 'model-00003-of-00003.safetensors'
            q.write_bytes(q.read_bytes()[:-1])
            d = Path(td) / 'out'
            d.mkdir()
            result = m.preserve(b, d)
            self.assertEqual(result['latest_layout_valid_checkpoint_step'], 150)
            self.assertIn('EOF', result['checkpoints'][-1]['layout_error'])
            self.assertTrue(q.exists())

    def test_safetensors_shape_bytes_and_index_keys_are_checked(self):
        with tempfile.TemporaryDirectory() as td:
            b = Path(td) / 'base'
            self.checkpoint(b, 150)
            p = self.checkpoint(b, 200)
            idx = p / 'model.safetensors.index.json'
            d = json.loads(idx.read_text())
            d['weight_map']['bad-key'] = d['weight_map'].pop('x')
            idx.write_text(json.dumps(d))
            out = Path(td) / 'out'
            out.mkdir()
            result = m.preserve(b, out)
            self.assertEqual(result['latest_layout_valid_checkpoint_step'], 150)
            self.assertIn('key mismatch', result['checkpoints'][-1]['layout_error'])

    def test_copy_symlink_does_not_follow_external_asset(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / 'source'
            source.mkdir()
            (root / 'outside').write_bytes(b'private-model-not-to-copy')
            (source / 'model-link').symlink_to(root / 'outside')
            rows = m.copy_evidence_snapshot(source, root / 'out')
            self.assertTrue((root / 'out/model-link').is_symlink())
            self.assertEqual(rows, [{'relative_path': 'model-link', 'symlink': str(root / 'outside')}])

    def test_bound_input_sha_rejects_changed_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'x'
            p.write_bytes(b'old')
            with self.assertRaisesRegex(m.Hold, 'SHA mismatch'):
                m.bound_bytes(p, m.sha(b'new'))

    def test_expired_authorization_holds(self):
        now = dt.datetime(2026, 10, 9, 9, tzinfo=dt.timezone.utc)
        x = {'schema': 'K4-human-resource-readonly-derived-v1', 'base': str(m.BASE),
             'hostname': m.HOST, 'uid': m.UID, 'actual_observed_utc': now.isoformat(),
             'owned_processes': list(fixture().values())}
        with self.assertRaisesRegex(m.Hold, 'Authorization'):
            m.load_request(json.dumps(x).encode(), b'human stop old archive',
                           '2026-10-07T00:00:00+00:00', now)


class Lifecycle(unittest.TestCase):
    def run_main(self, execute, preservation_fails=False, protected_disappears=False):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        b = Path(td.name) / 'base'
        b.mkdir()
        calls = []
        live = fixture()
        now = dt.datetime.now(dt.timezone.utc)
        with patch.object(m, 'BASE', b):
            for r in live.values():
                r['run_env']['E1_BASE'] = str(b)
                r['argv'] = str(b) + '/fixture-' + str(r['pid'])
                r['cwd'] = str(b) + '/runtime/archive'
            x = {'schema': 'K4-human-resource-readonly-derived-v1', 'base': str(b),
                 'hostname': m.HOST, 'uid': m.UID, 'actual_observed_utc': now.isoformat(),
                 'owned_processes': list(copy.deepcopy(live).values())}
            s = Path(td.name) / 'snapshot.json'
            s.write_text(json.dumps(x))
            auth = Path(td.name) / 'authorization.txt'
            auth.write_text('Latest human explicitly allows stopping old K4 archive.')
            args = ['--snapshot', str(s), '--snapshot-sha256', m.file_sha(s),
                    '--authorization-file', str(auth), '--authorization-sha256', m.file_sha(auth),
                    '--authorization-observed-utc', now.isoformat()]
            if execute:
                args.append('--execute')

            def fake_send(fd, sig, info, flags):
                calls.append((fd, sig))
                if sig == signal.SIGSTOP:
                    live[fd]['state'] = 'T'
                if sig in (signal.SIGTERM, signal.SIGKILL):
                    live.pop(fd, None)

            def fake_preserve(base, dest):
                self.assertTrue(all(r['state'] == 'T' for pid, r in live.items() if pid not in m.PROTECTED))
                if preservation_fails:
                    raise m.Hold('fixture disk preservation failure')
                return {'latest_layout_valid_checkpoint_step': 150, 'complete_Adam_resume_state': False}

            gpu_reads = [0]
            def fake_gpu():
                gpu_reads[0] += 1
                if protected_disappears and gpu_reads[0] > 1:
                    live.pop(2062794, None)
                return gpus_released(), apps_released(), {}

            uname = type('Uname', (), {'nodename': m.HOST})()
            output = io.StringIO()
            with patch.object(m.os, 'getuid', return_value=m.UID), \
                 patch.object(m.os, 'uname', return_value=uname), \
                 patch.object(m.os, 'pidfd_open', side_effect=lambda pid, flags: pid, create=True), \
                 patch.object(m.signal, 'pidfd_send_signal', side_effect=fake_send, create=True), \
                 patch.object(m.os, 'close'), \
                 patch.object(m, 'read_process', side_effect=lambda pid: copy.deepcopy(live.get(pid))), \
                 patch.object(m, 'process_table', side_effect=lambda: copy.deepcopy(live)), \
                 patch.object(m, 'checked_gpu', side_effect=fake_gpu), \
                 patch.object(m, 'preserve', side_effect=fake_preserve), \
                 contextlib.redirect_stdout(output):
                rc = m.main(args)
            return rc, json.loads(output.getvalue()), calls, live, list(b.glob('preserved/*'))

    def test_validation_only_has_no_signals_or_node_writes(self):
        rc, result, calls, _, directories = self.run_main(False)
        self.assertEqual(rc, 0)
        self.assertEqual(result['status'], 'VALIDATED_ONLY')
        self.assertFalse(calls)
        self.assertFalse(directories)

    def test_stop_ancestors_first_preserve_then_term_only_own(self):
        rc, result, calls, live, directories = self.run_main(True)
        self.assertEqual(rc, 0)
        self.assertTrue(result['accepted'])
        self.assertEqual([pid for pid, sig in calls[:3]], list(m.SUPERVISORS))
        self.assertTrue(all(sig == signal.SIGSTOP for _, sig in calls[:3]))
        self.assertFalse(set(pid for pid, _ in calls) & set(m.PROTECTED))
        self.assertEqual(set(live), set(m.PROTECTED))
        self.assertEqual(len(directories), 1)

    def test_failed_preservation_reports_hold_and_never_resumes_supervisor(self):
        rc, result, calls, _, _ = self.run_main(True, preservation_fails=True)
        self.assertEqual(rc, 2)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['status'], 'HOLD_STOP_INCOMPLETE')
        self.assertTrue(calls)
        self.assertTrue(all(sig == signal.SIGSTOP for _, sig in calls))

    def test_protected_exit_during_gpu_release_prevents_acceptance(self):
        rc, result, calls, _, _ = self.run_main(True, protected_disappears=True)
        self.assertEqual(rc, 2)
        self.assertFalse(result['accepted'])
        self.assertIn('Protected', result['error'])
        self.assertFalse(set(pid for pid, _ in calls) & set(m.PROTECTED))


if __name__ == '__main__':
    unittest.main()
