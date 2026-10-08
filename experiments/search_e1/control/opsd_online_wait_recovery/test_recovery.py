"""Pure CPU tests of external recovery; no CUDA, HTTP, node or signals."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('wait_recovery_cpu_fixture', HERE / 'recovery.py')
r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
sys.path.insert(0, str(HERE.parent / 'opsd_online'))
import controller as original


class StopHold(BaseException):
    pass


class Process:
    def __init__(self, rc, polls=None):
        self.rc = rc; self.polls = list(polls or [rc]); self.wait_calls = 0
    def poll(self):
        return self.polls.pop(0) if len(self.polls) > 1 else self.polls[0]
    def wait(self):
        self.wait_calls += 1; return self.rc


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name); self.stages = []; self.gate_calls = []
        self.gate_errors = []; self.audit_errors = []; self.binding_reads = 0
        self.frozen = SimpleNamespace(Controller=original.Controller, require_gpu_release=lambda c, i: dict(actual_free=True))
        self.cls = r.recovery_controller(self.frozen, {'source': 'fixture'}, reverify=self.reverify, sleeper=lambda _: None)
        self.c = self.cls.__new__(self.cls)
        self.c.config = dict(paths=dict(base=str(self.base), helper_root=str(HERE.parent / 'opsd_online'),
            python=sys.executable, runtime=str(self.base)), owner=dict(physical_gpus=list(range(8))))
        self.c.config_ref = dict(path=str(self.base / 'config.json'), sha256=r.CONFIG_SHA)
        self.c.run_id = r.RUN_ID; self.c.owner = dict(pid=100, uid=1005, starttick=1)
        self.c.known = []; self.c.active_jobs = []; self.c.lock = SimpleNamespace(closed=False)
        self.c.shared_locks = [SimpleNamespace(closed=False) for _ in range(4)]
        self.c.gate = self.gate; self.c.stage = lambda phase, **fields: self.stages.append((phase, fields))
        def admission(*args, **kwargs): return dict(accepted=True)
        def audit(*args, **kwargs):
            if self.audit_errors:
                error = self.audit_errors.pop(0)
                if error: raise error
            return dict(accepted=True, actual_full_owned_release=True)
        admission.require_owned_release = audit; self.c.admit = admission

    def reverify(self):
        self.binding_reads += 1

    def gate(self, phase, children=None):
        self.gate_calls.append(phase)
        if self.gate_errors:
            error = self.gate_errors.pop(0)
            if error: raise error
        return dict(accepted=True, phase=phase)

    def jobs(self, codes):
        result = []
        for rank, rc in enumerate(codes):
            handle = (self.base / ('worker-%d.log' % rank)).open('x'); handle.write('CPU fixture log\n'); handle.flush()
            identity = dict(pid=200 + rank, uid=1005, starttick=10 + rank, run_id=r.RUN_ID)
            result.append((Process(rc), handle, identity)); self.c.known.append(identity)
            self.addCleanup(lambda h=handle: None if h.closed else h.close())
        return result

    def records(self):
        path = next((self.base / 'receipts' / 'wait-recovery').glob('*/events.jsonl'))
        return [json.loads(line) for line in path.read_text().splitlines()]

    def test_transient_gate_error_then_actual_zero_and_release_succeeds(self):
        self.gate_errors = [ValueError('temporary ownership read unavailable')]
        jobs = self.jobs([0] * 8)
        proof = self.c.wait(jobs, 'sampling-running')
        self.assertTrue(proof['accepted']); self.assertEqual([j[0].wait_calls for j in jobs], [1] * 8)
        self.assertEqual([e['actual_wait_returncode'] for e in proof['child_exit_evidence']], [0] * 8)
        events = [e['event'] for e in self.records()]
        self.assertIn('ADMISSION_UNCERTAIN', events); self.assertIn('WAIT_ACCEPTED', events)
        self.assertLess(events.index('ACTUAL_WAIT_RETURNED'), events.index('TERMINAL_RELEASE_ACCEPTED'))
        self.assertFalse(self.c.lock.closed); self.assertTrue(all(not l.closed for l in self.c.shared_locks))

    def test_nonzero_is_durable_before_hard_rejection(self):
        with self.assertRaisesRegex(ValueError, 'Actual child nonzero'):
            self.c.wait(self.jobs([0, 7]), 'sampling-running')
        terminal_path = next((self.base / 'receipts' / 'wait-recovery').glob('*/actual-terminal.json'))
        terminal = json.loads(terminal_path.read_text())
        self.assertEqual([e['actual_wait_returncode'] for e in terminal['child_exit_evidence']], [0, 7])
        summary = json.loads(next(terminal_path.parent.glob('phase-complete.json')).read_text())
        self.assertFalse(summary['accepted']); self.assertTrue(summary['gpu_release']['all_owned_release']['accepted'])
        self.assertEqual(self.records()[-1]['event'], 'HARD_CHILD_FAILURE')

    def test_persistent_admission_holds_leases_after_durable_actual_rc(self):
        self.c.gate = lambda *a, **k: (_ for _ in ()).throw(ValueError('persistent foreign/unknown fact'))
        cls = r.recovery_controller(self.frozen, {}, sleeper=lambda _: (_ for _ in ()).throw(StopHold()))
        self.c.__class__ = cls
        with self.assertRaises(StopHold): self.c.wait(self.jobs([0]), 'training-running')
        self.assertFalse(self.c.lock.closed); self.assertTrue(all(not l.closed for l in self.c.shared_locks))
        self.assertEqual(self.records()[-1]['event'], 'TERMINAL_RELEASE_UNCERTAIN_HOLD')
        self.assertTrue(list((self.base / 'receipts' / 'wait-recovery').glob('*/actual-terminal.json')))
        self.assertFalse(list((self.base / 'receipts' / 'wait-recovery').glob('*/phase-complete.json')))

    def test_owned_release_unknown_retries_without_next_phase(self):
        self.audit_errors = [ValueError('opaque descendant'), None]
        result = self.c.wait(self.jobs([0]), 'training-running')
        self.assertTrue(result['accepted']); self.assertEqual(self.binding_reads, 2)
        self.assertIn('TERMINAL_RELEASE_UNCERTAIN_HOLD', [e['event'] for e in self.records()])

    def test_partial_journal_append_error_retries_one_complete_event(self):
        actual = r.Journal.record; injected = []
        def faulty(journal, event, **fields):
            if event == 'ACTUAL_WAIT_RETURNED' and not injected:
                injected.append(True)
                with journal.path.open('ab') as handle:
                    handle.write(b'{"partial":'); handle.flush()
                raise OSError('injected partial append / full disk')
            return actual(journal, event, **fields)
        with patch.object(r.Journal, 'record', faulty):
            value = self.c.wait(self.jobs([0]), 'sampling-running')
        self.assertTrue(value['accepted'])
        rows = self.records()  # Every line must still parse after recovery.
        self.assertEqual(sum(e['event'] == 'ACTUAL_WAIT_RETURNED' for e in rows), 1)
        self.assertFalse(self.c.lock.closed)

    def test_persistent_evidence_io_holds_with_live_child_no_unwind(self):
        cls = r.recovery_controller(self.frozen, {}, sleeper=lambda _: (_ for _ in ()).throw(StopHold()))
        self.c.__class__ = cls; jobs = self.jobs([0]); jobs[0][0].polls = [None]
        with patch.object(r.Journal, 'record', side_effect=OSError('persistent disk failure')):
            with self.assertRaises(StopHold): self.c.wait(jobs, 'sampling-running')
        self.assertEqual(jobs[0][0].wait_calls, 0)
        self.assertFalse(self.c.lock.closed); self.assertTrue(all(not l.closed for l in self.c.shared_locks))
        self.assertFalse(list((self.base / 'receipts' / 'wait-recovery').glob('*/phase-complete.json')))

    def test_stage_and_terminal_evidence_io_recover_before_acceptance(self):
        self.gate_errors = [ValueError('uncertain admission')]
        actual_stage = self.c.stage; stages = []
        def stage(phase, **fields):
            if not stages:
                stages.append('injected'); raise OSError('status fsync failed')
            return actual_stage(phase, **fields)
        self.c.stage = stage
        actual_json = r.durable_json; json_errors = []
        def durable(path, value):
            if Path(path).name == 'actual-terminal.json' and not json_errors:
                json_errors.append(True); raise OSError('terminal fsync failed')
            return actual_json(path, value)
        with patch.object(r, 'durable_json', durable):
            value = self.c.wait(self.jobs([0]), 'training-running')
        self.assertTrue(value['accepted']); self.assertEqual(len(json_errors), 1)
        self.assertFalse(self.c.lock.closed)

    def test_log_hash_io_retries_without_reaping_or_signalling_again(self):
        actual = r.fileproof; errors = []
        def hashing(path):
            if Path(path).name == 'worker-0.log' and not errors:
                errors.append(True); raise OSError('log read temporarily unavailable')
            return actual(path)
        jobs = self.jobs([0])
        with patch.object(r, 'fileproof', hashing): value = self.c.wait(jobs, 'sampling-running')
        self.assertTrue(value['accepted']); self.assertEqual(jobs[0][0].wait_calls, 1)
        self.assertFalse(self.c.lock.closed)

    def test_subclass_only_overrides_wait(self):
        methods = {name for name, value in self.cls.__dict__.items() if callable(value)}
        self.assertEqual(methods, {'wait'})
        self.assertIs(self.cls.run, original.Controller.run)
        self.assertIs(self.cls.sample, original.Controller.sample)
        self.assertIs(self.cls.spawn, original.Controller.spawn)

    def test_actual_inherited_sample_creates_new_full_attempt_not_old_worker_evidence(self):
        outer = self.base / 'outer'; outer.mkdir()
        old = outer / 'sampling-attempt-00000'; old.mkdir()
        poison = old / 'rollouts-0.jsonl.worker-complete.json'; poison.write_text('{"accepted":true,"rows":320}')
        old_before = r.fileproof(poison)
        questions = [dict(id='q%d' % i) for i in range(512)]
        identity = dict(sha256='model-sha')
        calls = []
        def spawn(command, log, phase, env):
            def arg(key): return command[command.index(key) + 1]
            rank = int(arg('--rank')); output = Path(arg('--output')); trace = Path(arg('--trace')); http = Path(arg('--http-trace'))
            selected = original.read_rows(arg('--questions')); self.assertEqual(len(selected), 64)
            original.write_rows(output, [dict(id=q['id'], sibling=s) for q in selected for s in range(5)])
            trace.write_text(''); http.write_text('')
            original.exclusive_json(str(output) + '.worker-complete.json', dict(accepted=True, rank=rank, rows=320,
                run_id=r.RUN_ID, sampling_model_identity=identity['sha256'], rollouts=original.fileproof(output),
                generation_trace=original.fileproof(trace), http_trace=original.fileproof(http), http_requests=0))
            handle = Path(log).open('x'); handle.write('new actual worker attempt\n'); handle.flush()
            child = dict(pid=400 + rank, uid=1005, starttick=100 + rank, run_id=r.RUN_ID)
            self.c.known.append(child); calls.append(output); return Process(0), handle, child
        self.c.spawn = spawn
        module = ModuleType('search_e1'); module.core = SimpleNamespace()
        with patch.object(original, 'bind_model_files', return_value=identity), \
             patch.object(original, 'verify_scoring', return_value=dict(accepted=True)), \
             patch.object(original, 'verify_rollouts', side_effect=lambda rows, qs, core: ([], dict(P=0, rows=len(rows)))), \
             patch.object(original, 'verify_http_trace', side_effect=lambda ref, *args: original.checked_file(ref)), \
             patch.dict(sys.modules, {'search_e1': module}):
            value, proof = self.c.sample(outer, questions, identity)
        self.assertEqual(len(calls), 8); self.assertTrue(all(p.parent.name == 'sampling-attempt-00001' for p in calls))
        self.assertEqual(value['stats']['rows'], 2560); self.assertTrue(value['exit_proof']['accepted'])
        self.assertEqual([e['actual_wait_returncode'] for e in value['exit_proof']['child_exit_evidence']], [0] * 8)
        self.assertEqual(r.fileproof(poison), old_before)
        self.assertTrue(Path(proof['path']).exists()); self.assertFalse((old / 'actual-terminal.json').exists())


class BindingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup); self.root = Path(self.tmp.name)

    def terminal(self):
        rows = [dict(pid=612077, uid=1005, starttick=924207665)] + [dict(pid=700 + i, uid=1005, starttick=20 + i) for i in range(8)]
        value = dict(schema='opsd-online-failed-original-owner-terminal-v1', accepted=True, run_id=r.RUN_ID,
            config_sha256=r.CONFIG_SHA, old_wait_returncodes='NOT_PERSISTED_UNKNOWN', old_controller_tmux_exit_status=1,
            old_owned_processes=rows)
        return value

    def test_old_terminal_requires_fresh_absence_and_never_rc_zero_inference(self):
        proc = self.root / 'proc'; proc.mkdir(); value = self.terminal()
        ref = r.exclusive_json(self.root / 'terminal.json', value)
        self.assertTrue(r.validate_old_terminal(ref, proc)['fresh_all_old_pids_absent'])
        (proc / '612077').mkdir()
        with self.assertRaisesRegex(ValueError, 'Old PID is present'): r.validate_old_terminal(ref, proc)

    def test_old_unknown_wait_codes_cannot_be_replaced_with_inferred_zero(self):
        value = self.terminal(); value['old_wait_returncodes'] = [0] * 8
        with self.assertRaisesRegex(ValueError, 'Do not infer old child rc'):
            r.validate_old_terminal(r.exclusive_json(self.root / 'terminal.json', value), self.root / 'proc')

    def publication(self):
        (self.root / 'recovery.py').write_text('fixture source\n')
        item = r.fileproof(self.root / 'recovery.py')
        manifest = r.exclusive_json(self.root / 'source-manifest.json', dict(files={'recovery.py': dict(sha256=item['sha256'], bytes=item['bytes'])}))
        pub = dict(accepted=True, branch='search-e1', source_commit='a' * 40, remote_readback_sha='a' * 40,
            source_manifest_sha256=manifest['sha256'], run_id=r.RUN_ID, config_sha256=r.CONFIG_SHA,
            files=[dict(path=r.REPO_PREFIX + 'recovery.py', sha256=item['sha256'], bytes=item['bytes'], equal=True),
                   dict(path=r.REPO_PREFIX + 'source-manifest.json', sha256=manifest['sha256'], bytes=manifest['bytes'], equal=True)])
        return manifest, pub

    def test_external_publication_requires_exact_actual_source_and_remote_readback(self):
        manifest, pub = self.publication(); ref = r.exclusive_json(self.root / 'publication.json', pub)
        # Publication receipt lives outside the source directory in production.
        source = self.root / 'source'; source.mkdir()
        for name in ('recovery.py', 'source-manifest.json'): (self.root / name).rename(source / name)
        manifest['path'] = str(source / 'source-manifest.json')
        self.assertEqual(r.verify_external_binding(manifest, ref, source)['source_commit'], 'a' * 40)
        (source / 'recovery.py').write_text('changed source\n')
        with self.assertRaisesRegex(ValueError, 'Frozen source SHA/bytes differ'): r.verify_external_binding(manifest, ref, source)

    def test_unpublished_readback_never_passes(self):
        manifest, pub = self.publication(); pub['remote_readback_sha'] = 'b' * 40
        source = self.root / 'source'; source.mkdir()
        for name in ('recovery.py', 'source-manifest.json'): (self.root / name).rename(source / name)
        manifest['path'] = str(source / 'source-manifest.json')
        ref = r.exclusive_json(self.root / 'publication.json', pub)
        with self.assertRaisesRegex(ValueError, 'remote commit readback'): r.verify_external_binding(manifest, ref, source)


if __name__ == '__main__': unittest.main()
