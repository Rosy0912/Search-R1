"""Pure CPU fixtures: mocked pidfds/signals only, never real process signaling."""
import copy
import json
from pathlib import Path
import signal
import tempfile
import types
import unittest
import stop


def row(pid=stop.OWNER['pid'], uid=1005, tick=stop.OWNER['starttick'], run=stop.RUN, ppid=1, state='S'):
    return dict(pid=pid, uid=uid, starttick=tick, ppid=ppid, pgid=pid, state=state,
                run_id=run, cuda_visible_devices='', environment_readable=True,
                argv_tokens=[stop.PYTHON, '-B', stop.ENTRY, 'parent', '--config',
                    str(stop.BASE / 'receipts/config.json'), '--config-sha256', stop.CONFIG_SHA,
                    '--run-id', stop.RUN, '--resume'])


class StopTests(unittest.TestCase):
    def test_exact_owner_and_negative_identity(self):
        valid = row(); stop.match_owner(valid)
        for field, value in [('pid', 660173), ('uid', 0), ('starttick', valid['starttick'] + 1), ('run_id', 'foreign'), ('cuda_visible_devices', '0')]:
            wrong = copy.deepcopy(valid); wrong[field] = value
            with self.assertRaises(RuntimeError): stop.match_owner(wrong)

    def test_exact_options_not_substrings(self):
        for index, value in [(2, stop.ENTRY + '.foreign'), (5, str(stop.BASE / 'receipts/config.json') + '.foreign'), (7, stop.CONFIG_SHA + 'x')]:
            wrong = row(); wrong['argv_tokens'][index] = value
            with self.assertRaises(RuntimeError): stop.match_owner(wrong)
        wrong = row(); wrong['argv_tokens'] += ['--run-id', stop.RUN]
        with self.assertRaises(RuntimeError): stop.match_owner(wrong)
        wrong = row(); wrong['argv_tokens'] += ['unexpected']
        with self.assertRaises(RuntimeError): stop.match_owner(wrong, row()['argv_tokens'])

    def test_missing_pidfd_capability_sends_nothing(self):
        with self.assertRaises(RuntimeError): stop.Signaler(opener=False, sender=lambda *a: None)

    def test_pidfd_reused_pid_reject_before_signal(self):
        sent = []; closed = []; calls = [row(), row(tick=1)]
        runner = stop.Signaler(reader=lambda pid: calls.pop(0), opener=lambda pid: 88,
            sender=lambda *args: sent.append(args), closer=lambda fd: closed.append(fd))
        with self.assertRaises(RuntimeError): runner.send(stop.OWNER, signal.SIGTERM, lambda *a, **k: None, owner=True)
        self.assertEqual(sent, []); self.assertEqual(closed, [88])

    def test_pidfd_target_uid_change_reject(self):
        sent = []; calls = [row(), row(uid=0)]
        runner = stop.Signaler(reader=lambda pid: calls.pop(0), opener=lambda pid: 88,
            sender=lambda *args: sent.append(args), closer=lambda fd: None)
        with self.assertRaises(RuntimeError): runner.send(stop.OWNER, signal.SIGTERM, lambda *a, **k: None)
        self.assertEqual(sent, [])

    def test_explicit_pidfd_signal_no_group_or_false_exit0(self):
        sent = []; events = []
        runner = stop.Signaler(reader=lambda pid: row(), opener=lambda pid: 88,
            sender=lambda *args: sent.append(args), closer=lambda fd: None)
        runner.send(stop.OWNER, signal.SIGSTOP, lambda event, **k: events.append((event, k)), owner=True)
        self.assertEqual(sent, [(88, signal.SIGSTOP)])
        self.assertEqual([e[0] for e in events], ['BEFORE_SIGNAL', 'SIGNAL_SENT'])
        self.assertNotIn('actual_exit_code', events[-1][1])

    def test_cohort_positive_ancestry_or_run_only(self):
        parent = row(); child = row(10, tick=10, run=None, ppid=parent['pid'])
        grandchild = row(11, tick=11, run=None, ppid=10)
        reparented = row(12, tick=12, run=stop.RUN)
        reader = row(13, tick=13, run='cpu-reader')
        reader['argv_tokens'] = ['python', 'reader.py', '--input', str(stop.BASE / 'runs')]
        oracle = row(stop.ORACLE['pid'], tick=stop.ORACLE['starttick'], run='retriever')
        observer = row(15, tick=15, run='online-memory-' + stop.RUN)
        backup = row(57199, uid=501, tick=16, run='backup')
        table = {r['pid']: r for r in (parent, child, grandchild, reparented, reader, oracle, observer, backup)}
        self.assertEqual({r['pid'] for r in stop.owned_children(table)}, {10, 11, 12})

    def test_foreign_uid_and_protected_identity_block(self):
        child = row(10, uid=0, tick=10, ppid=stop.OWNER['pid'])
        with self.assertRaises(RuntimeError): stop.owned_children({10: child})
        oracle = row(stop.ORACLE['pid'], tick=stop.ORACLE['starttick'])
        with self.assertRaises(RuntimeError): stop.owned_children({oracle['pid']: oracle})
        observer = row(10, tick=10)
        with self.assertRaises(RuntimeError): stop.owned_children({10: observer}, protected=[observer])

    def test_known_reparented_retained_but_reused_not_targeted(self):
        prior = row(10, tick=10, run=None); now = copy.deepcopy(prior)
        self.assertEqual([r['pid'] for r in stop.owned_children({10: now}, [prior])], [10])
        now['starttick'] += 1
        self.assertEqual(stop.owned_children({10: now}, [prior]), [])

    def test_zero_children_no_child_signal(self):
        signaler = types.SimpleNamespace(send=lambda *a, **k: self.fail('No child signals expected'))
        self.assertEqual(stop.drain_children(signaler, lambda: {stop.OWNER['pid']: row()}, lambda *a, **k: None, []), [])

    def test_drain_term_then_exact_kill_after_finite_wait(self):
        child = row(10, tick=10); table = {10: child}; sends = []; clock = [0]
        def send(bound, number, journal):
            sends.append((bound['pid'], number))
            if number == signal.SIGKILL: table.clear()
        def sleep(seconds): clock[0] += seconds
        known = stop.drain_children(types.SimpleNamespace(send=send), lambda: dict(table), lambda *a, **k: None,
            [], timeout=.5, clock=lambda: clock[0], sleep=sleep)
        self.assertEqual(sends, [(10, signal.SIGTERM), (10, signal.SIGKILL)])
        self.assertEqual(known[0]['starttick'], 10)

    def test_canonical_latest_excludes_pending_checks_actual_native(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, update in [('update-00000364-outer-00036', 364), ('update-00000365-outer-00037.pending-9', 365)]:
                path = root / name; path.mkdir()
                (path / 'manifest.json').write_text(json.dumps(dict(accepted=True, run_id=stop.RUN,
                    config_sha256=stop.CONFIG_SHA, optimizer_updates=update, cursor=dict(outer_index=36))))
            calls = []
            def check(ref, config, run, update, outer):
                calls.append((ref, run, update, outer)); return {'optimizer_updates': update}, {'actual_files_hashed': True}
            native = types.SimpleNamespace(fileproof=stop.proof, check_checkpoint=check)
            cp, evidence = stop.latest_checkpoint(root, {}, native)
            self.assertEqual(cp['optimizer_updates'], 364); self.assertTrue(evidence['actual_files_hashed'])
            self.assertEqual(calls[0][2:], (364, 36))


if __name__ == '__main__': unittest.main()
