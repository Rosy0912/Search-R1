"""Small read-only terminal-audit fixtures; no SSH, signals, GPU, or Node writes."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location('terminal_verify_under_test', HERE / 'terminal_verify.py')
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)
h = v.load_helper(HERE / 'archive_stop.py')


def captured(pid, tick):
    return {'pid': pid, 'uid': h.UID, 'starttick': tick, 'argv': 'ordinary helper ',
            'cwd': str(h.BASE / 'runtime/archive'), 'ppid': 1, 'state': 'S', 'run_env': {}}


def receipts():
    owners = [captured(900000 + i, 100000 + i) for i in range(188)]
    events = []
    for name in ('SIGSTOP', 'SIGTERM', 'SIGCONT'):
        for r in owners:
            events.append({k: r[k] for k in ('pid', 'uid', 'starttick')})
            events[-1].update(signal=name, argv_sha256='a' * 64,
                              pidfd_bound_actual_argv_sha256='b' * 64,
                              signal_observed_actual_argv_sha256='c' * 64)
    preservation = {'checkpoint_weights_only': True, 'complete_Adam_resume_state': False}
    result = {'schema': 'k4-archive-replacement-stop-v1', 'status': 'HOLD_STOP_INCOMPLETE',
              'accepted': False, 'signals_executed': True, 'error_type': 'Hold',
              'error': 'Known PID identity changed: 1434651', 'base': str(h.BASE),
              'run_id': h.RUN, 'uid': h.UID, 'hostname': h.HOST,
              'helper_source_sha256': v.HELPER_SHA, 'authorization_sha256': v.AUTH_SHA,
              'preserved_directory': str(h.BASE / 'preserved' / v.DESTINATION),
              'authorization_raw_utf8': 'fixture-auth', 'snapshot_sha256': 'd' * 64,
              'authorization_observed_utc': '2026-10-09T08:30:00Z',
              'initial_owned': owners, 'stopped_by_this_helper': list(reversed(owners)),
              'protected': [], 'signal_events': events, 'preservation': preservation}
    q = copy.deepcopy(result)
    q.update(status='QUIESCED_PRESERVING', signal_events=events[:188])
    return result, preservation, q


class TerminalAuditTests(unittest.TestCase):
    def valid_receipts(self, result, preservation, q):
        # The fixture exercises structure with its own explicit authorization SHA;
        # production main still requires the fixed published real authorization.
        with patch.object(v, 'AUTH_SHA', v.sha(b'fixture-auth')):
            result['authorization_sha256'] = q['authorization_sha256'] = v.AUTH_SHA
            return v.validate_receipts(result, preservation, q, h)

    def test_exact_receipts_accept_without_changing_old_hold(self):
        r, p, q = receipts()
        self.assertEqual(len(self.valid_receipts(r, p, q)), 188)
        self.assertFalse(r['accepted'])
        self.assertEqual(r['status'], 'HOLD_STOP_INCOMPLETE')

    def test_wrong_error_or_missing_signal_rejected(self):
        for change in ('error', 'signal'):
            with self.subTest(change=change):
                r, p, q = receipts()
                if change == 'error':
                    r['error'] = 'Some other failure'
                else:
                    r['signal_events'].pop()
                with self.assertRaises(v.Hold):
                    self.valid_receipts(r, p, q)

    def test_protected_signal_and_preservation_mismatch_rejected(self):
        r, p, q = receipts()
        r['initial_owned'][0]['pid'] = next(iter(h.PROTECTED))
        with self.assertRaises(v.Hold):
            self.valid_receipts(r, p, q)
        r, p, q = receipts()
        with self.assertRaises(v.Hold):
            self.valid_receipts(r, dict(p, extra='not-in-bound-result'), q)

    def make_checkpoints(self, base):
        checkpoints = []
        for step in (50, 100, 150):
            root = base / 'runs/archive/round-1-grpo/actor' / ('global_step_' + str(step))
            root.mkdir(parents=True)
            files = []
            for index in range(12):
                f = root / ('model-' + str(index) + '.safetensors')
                f.write_bytes(b'fixture-not-actual-model')
                files.append({'relative_path': f.name, 'bytes': f.stat().st_size,
                              'mtime_ns': f.stat().st_mtime_ns, 'sha256': 'a' * 64})
            checkpoints.append({'path': str(root), 'step': step, 'files': files})
        return checkpoints

    def test_checkpoint_continuity_never_opens_payload(self):
        with tempfile.TemporaryDirectory() as t:
            base = Path(t)
            cps = self.make_checkpoints(base)
            with patch.object(v, 'file_sha', side_effect=AssertionError('must not hash model')):
                result = v.checkpoint_metadata(cps, base)
            self.assertEqual(result['members'], 36)
            self.assertFalse(result['fresh_payload_sha'])
            self.assertFalse(result['Adam_or_RNG_captured'])

    def test_checkpoint_mtime_or_added_member_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            base = Path(t)
            cps = self.make_checkpoints(base)
            p = Path(cps[0]['path']) / cps[0]['files'][0]['relative_path']
            st = p.stat()
            os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1000000))
            with self.assertRaises(v.Hold):
                v.checkpoint_metadata(cps, base)
            cps[0]['files'][0]['mtime_ns'] = p.stat().st_mtime_ns
            (Path(cps[0]['path']) / 'extra').write_text('added')
            with self.assertRaises(v.Hold):
                v.checkpoint_metadata(cps, base)

    def test_copy_fresh_sha_and_corruption_detection(self):
        with tempfile.TemporaryDirectory() as t:
            base = Path(t) / 'base'
            dest = base / 'preserved' / v.DESTINATION
            entries = []
            for rel in v.COPY_SCOPES:
                root = dest / 'copies' / rel
                single = rel == v.COPY_SCOPES[-1]
                p = root if single else root / 'file.py'
                p.parent.mkdir(parents=True)
                p.write_bytes(b'read-only copy bytes')
                records = [{'relative_path': p.name, 'bytes': p.stat().st_size,
                            'sha256': v.file_sha(p)}]
                entries.append({'path': str(base / rel), 'copy': str(root), 'files': records})
            result = v.copied_evidence(entries, base, dest)
            self.assertEqual(result['regular_files_fresh_sha'], 6)
            p.write_bytes(b'changed copy bytes!!')
            with self.assertRaises(v.Hold):
                v.copied_evidence(entries, base, dest)

    def protected_table(self):
        return {pid: captured(pid, tick) for pid, tick in h.PROTECTED.items()}

    def test_historical_reparent_and_opaque_identity_hold(self):
        for actual in (captured(990000, 12345), {'pid': 990000, 'opaque_identity': True}):
            with self.subTest(actual=actual):
                table = self.protected_table()
                table[990000] = actual
                old = {990000: captured(990000, 12345)}
                with patch.object(h, 'process_table', return_value=table):
                    with self.assertRaises((v.Hold, h.Hold)):
                        v.terminal_sample(h, self.protected_table(), old)

    def test_fresh_release_only_oracle_and_protected_required(self):
        table = self.protected_table()
        gpus = {i: {'index': i, 'uuid': u, 'total_mib': 49152,
                    'used_mib': 750 if i == 7 else 12,
                    'free_mib': 48000 if i == 7 else 48738}
                for i, u in h.GPU_UUIDS.items()}
        oracle = {'pid': 1749337, 'uuid': h.GPU_UUIDS[7], 'mib': 738}
        with patch.object(h, 'process_table', return_value=table), \
                patch.object(h, 'checked_gpu', return_value=(gpus, [oracle], {})):
            proof = v.terminal_sample(h, table, {990000: captured(990000, 12345)})
            self.assertEqual(len(proof['released_gpu']), 4)
            self.assertEqual(len(proof['historical_identity_terminal_checks']), 1)
        for apps, current in (([oracle, {'pid': 4444, 'uuid': h.GPU_UUIDS[4], 'mib': 1}], table),
                              ([oracle], {})):
            with patch.object(h, 'process_table', return_value=current), \
                    patch.object(h, 'checked_gpu', return_value=(gpus, apps, {})):
                with self.assertRaises((v.Hold, h.Hold)):
                    v.terminal_sample(h, table, {})


if __name__ == '__main__':
    unittest.main()
