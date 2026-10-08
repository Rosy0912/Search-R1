"""Exact helper-function metadata fixtures, no main/route/subprocess execution."""
import ast
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import time
from types import SimpleNamespace
import unittest

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
HELPER = ROOT / 'work/tree-fresh200-dispatch-v12'
CAPTURE = ROOT / 'work/tree-fresh200-freeze-02b1-v11/actual-zero-dispatch'


def exact_functions(path, names, extra=None):
    parsed = ast.parse(path.read_text())
    nodes = [node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    ns = dict(ast=ast, hashlib=hashlib, json=json, os=os, Path=Path, re=re, stat=stat,
              __file__=str(path), LOCAL_CLUSTER='wuhan-h100')
    if extra:
        ns.update(extra)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
    return SimpleNamespace(**{name: ns[name] for name in names})


def literal(path, name):
    tree = ast.parse(path.read_text())
    return ast.literal_eval(next(n.value for n in tree.body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)))


submit = exact_functions(HELPER / 'submit_tree_fresh200_once.py', [
    'accounting_disabled', 'read_failed_zero_run', 'live_failed_zero_terminal',
    'previous_journals_clear', 'validate_dispatch'])


class IndependentTerminalCPU(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='metadata-fixture-', dir=OUT)
        self.addCleanup(self.temp.cleanup)
        cpath = HELPER / 'fresh200_dispatch_common.py'
        self.c = exact_functions(cpath, ['require', 'digest', 'valid_sha', 'local_path',
            'read_bound', 'json_bound', 'queue_clear'], extra={
                'ROOT': Path(self.temp.name), 'OWN_PREFIXES': literal(cpath, 'OWN_PREFIXES')})
        self.c.ROOT = Path(self.temp.name)
        self.c.STAGE = 'fresh_mu_calibration'
        self.c.COMMIT = '53082e1667a6ab131a036ca2ef66e004989ed1a9'
        self.c.WRAPPER_HASHES = literal(cpath, 'WRAPPER_HASHES')
        (self.c.ROOT / 'instruct300').mkdir()
        self.prior = json.loads((CAPTURE / 'NATIVE_JOURNAL3314.json').read_bytes())
        # Only path namespace translated into the owned fixture; actual captured
        # journal/proof hashes are checked separately against unmodified bytes.
        self.prior['reviewed_launch_path_provenance_only'] = str(
            self.c.ROOT / 'instruct300/candidates/fresh200-mu-02b1-v11/launch.sbatch')
        self.journal = self.c.ROOT / 'instruct300/SUBMISSION_ONCE-fresh200-mu-zero-old.json'
        self.current = self.c.ROOT / 'instruct300/SUBMISSION_ONCE-fresh200-mu-zero-new.json'
        self.run_path = self.c.ROOT / 'runs/tree-fresh200-3314/run-result.json'
        self.run_path.parent.mkdir(parents=True)
        self.run_raw = (CAPTURE / 'terminal-native/run-result.json').read_bytes()
        self.run_path.write_bytes(self.run_raw)
        audit = json.loads((CAPTURE / 'TERMINAL3314_AUDIT.json').read_bytes())
        self.control = copy.deepcopy(audit['scontrol'])
        self.old_queue = copy.deepcopy(audit['squeue'])
        self.sacct = SimpleNamespace(returncode=1, stdout='', stderr='Slurm accounting storage is disabled\n')
        self.queue = SimpleNamespace(returncode=0, stdout='', stderr='')
        self.previous = {}
        self.calls = []
        self.save_journal()

    def save_journal(self):
        self.journal.write_text(json.dumps(self.prior, sort_keys=True))

    def retained(self):
        self.save_journal()
        self.proof = json.loads((CAPTURE / 'RETAINED_TERMINAL3314_PROOF_V2.json').read_bytes())
        self.proof['journal_sha256'] = hashlib.sha256(self.journal.read_bytes()).hexdigest()
        self.proof_path = self.c.ROOT / 'instruct300/positive-terminal.json'
        self.save_proof()
        self.control = dict(returncode=1, stdout='', stderr='slurm_load_jobs error: Invalid job id specified\n')

    def save_proof(self):
        self.proof_path.write_text(json.dumps(self.proof, sort_keys=True))
        self.previous = {'3314': dict(path=str(self.proof_path),
            sha256=hashlib.sha256(self.proof_path.read_bytes()).hexdigest())}

    def runner(self, argv, **kwargs):
        self.assertEqual(kwargs, dict(env={}, text=True, capture_output=True, timeout=30))
        self.calls.append(argv)
        if argv == ['sacct', '--local', '-X', '-n', '-P', '-j', '3314', '-o',
                    'JobIDRaw,Cluster,State,ExitCode']:
            return self.sacct
        if argv == ['scontrol', '--local', 'show', 'job', '3314']:
            return SimpleNamespace(**self.control)
        if argv == ['squeue', '--local', '-h', '-w', 'h100-9', '-o', '%i|%j|%T|%N']:
            return self.queue
        raise AssertionError('Unexpected inquiry; no real tool runner exists: ' + repr(argv))

    def check(self):
        return submit.previous_journals_clear(self.c, self.current, self.runner, {}, self.previous)

    def refused(self):
        with self.assertRaises((ValueError, FileNotFoundError, OSError)):
            self.check()

    def test_captured_positive_failed3314_and_exact_native_run_bytes(self):
        result = self.check()[0]
        self.assertEqual(result['scheduler_state'], 'FAILED')
        self.assertEqual(result['exit_code'], '1:0')
        self.assertEqual(result['failed_run']['sha256'], hashlib.sha256(self.run_raw).hexdigest())
        self.assertFalse(result['actual_zero_or_training_accepted'])

    def test_bound_expired_positive_plus_new_queue_and_native_bytes(self):
        self.retained()
        result = self.check()[0]
        self.assertEqual(result['evidence_source'],
            'sha_bound_retained_positive_terminal_live_expired_and_empty_queue')
        self.assertEqual([x[0] for x in self.calls], ['sacct', 'scontrol', 'squeue'])
        self.assertFalse(result['actual_zero_or_training_accepted'])

    def test_normal_sacct_path_does_not_consume_fallback(self):
        self.sacct = SimpleNamespace(returncode=0, stdout='3314|wuhan-h100|FAILED|1:0\n', stderr='')
        self.assertEqual(self.check()[0]['scheduler_state'], 'FAILED')
        self.assertEqual([x[0] for x in self.calls], ['sacct'])

    def test_exact_disabled_only_no_mixed_diagnostics_or_success_error(self):
        for value in [SimpleNamespace(returncode=0, stdout='', stderr='Slurm accounting storage is disabled'),
                      SimpleNamespace(returncode=1, stdout='', stderr='Slurm accounting storage is disabled\ncontroller failed'),
                      SimpleNamespace(returncode=1, stdout='', stderr='controller unavailable'),
                      SimpleNamespace(returncode=1, stdout=b'', stderr='Slurm accounting storage is disabled')]:
            with self.subTest(value=value):
                self.assertFalse(submit.accounting_disabled(value))
        for text in ['Slurm accounting storage is disabled', 'sacct: error: Slurm accounting storage is disabled.']:
            self.assertTrue(submit.accounting_disabled(SimpleNamespace(returncode=1, stdout='', stderr=text)))

    def test_invalid_id_alone_and_other_controller_errors_refuse(self):
        self.control = dict(returncode=1, stdout='', stderr='slurm_load_jobs error: Invalid job id specified')
        self.refused()
        self.retained()
        for error in ['Invalid job id specified', 'controller unavailable',
                      'slurm_load_jobs error: Invalid job id specified\nUNKNOWN']:
            with self.subTest(error=error):
                self.control['stderr'] = error
                self.refused()

    def test_retained_positive_control_running_missing_duplicate_requeue_refuse(self):
        self.retained()
        original = copy.deepcopy(self.proof)
        control = original['scontrol']['stdout']
        for old, new in [('JobState=FAILED', 'JobState=RUNNING'), ('Requeue=0', 'Requeue=1'),
                         ('Restarts=0', 'Restarts=1'), ('NumCPUs=192', 'NumCPUs=True'),
                         ('ExitCode=1:0', 'ExitCode=0:0'), ('TresPerNode=gres:gpu:8', '')]:
            with self.subTest(new=new):
                self.proof = copy.deepcopy(original)
                self.proof['scontrol']['stdout'] = control.replace(old, new)
                self.save_proof(); self.refused()
        self.proof = copy.deepcopy(original)
        self.proof['scontrol']['stdout'] += '\nJobId=3314\n'
        self.save_proof(); self.refused()

    def test_retained_old_queue_missing_or_failed_refuses(self):
        self.retained()
        for value in [{}, dict(returncode=1, stdout='', stderr=''),
                      dict(returncode=0, stdout='3314|tree-fresh200-mu-zero|RUNNING|h100-9\n', stderr='')]:
            with self.subTest(value=value):
                self.proof['squeue'] = value; self.save_proof(); self.refused()

    def test_new_queue_oldid_foreign_name_ownprefix_unknown_and_errors_refuse(self):
        self.retained()
        for value in [SimpleNamespace(returncode=0, stdout='3314|foreign|RUNNING|h100-9\n', stderr=''),
                      SimpleNamespace(returncode=0, stdout='3315|tree-fresh200-mu-zero|RUNNING|h100-9\n', stderr=''),
                      SimpleNamespace(returncode=0, stdout='UNKNOWN', stderr=''),
                      SimpleNamespace(returncode=1, stdout='', stderr=''),
                      SimpleNamespace(returncode=0, stdout='', stderr='controller error')]:
            with self.subTest(value=value):
                self.queue = value; self.refused()
        self.queue = SimpleNamespace(returncode=0, stdout='5000|unrelated|RUNNING|h100-9\n', stderr='')
        self.assertEqual(self.check()[0]['scheduler_state'], 'FAILED')

    def test_current_native_bytes_changed_missing_or_symlink_refuse(self):
        self.retained()
        changed = json.loads(self.run_raw); changed['error'] = 'Different bytes'
        self.run_path.write_text(json.dumps(changed)); self.refused()
        self.run_path.unlink(); self.refused()
        alternate = self.run_path.parent / 'alternate.json'; alternate.write_bytes(self.run_raw)
        self.run_path.symlink_to(alternate); self.refused()

    def test_cleanup_or_protected_eight_missing_refuse(self):
        original = json.loads(self.run_raw)
        for key, value in [('cleanup_residual', None), ('cleanup_residual', {'pid': 1}),
                           ('original_eight_pid_start_times_preserved', False), ('allocated_gpus', True)]:
            with self.subTest(key=key, value=value):
                changed = copy.deepcopy(original); changed[key] = value
                self.run_path.write_text(json.dumps(changed)); self.refused()

    def test_unknown_prior_and_wrong_bound_hashes_still_refuse(self):
        self.retained()
        original = copy.deepcopy(self.prior)
        for key, value in [('status', 'SUBMITTING_DO_NOT_RETRY_ON_UNKNOWN'), ('cluster_name', 'foreign')]:
            with self.subTest(key=key):
                self.prior = copy.deepcopy(original); self.prior[key] = value
                self.save_journal(); self.refused()
        self.prior = original; self.save_journal()
        self.previous['3314']['sha256'] = '0' * 64; self.refused()

    def test_final_release_must_bind_same_previous_terminal_map(self):
        target = self.c.ROOT / 'instruct300/candidates/new'
        refs = dict(manifest='a' * 64)
        review = dict(status='READY_FOR_FRESH200_BEHAVIOR_MU_ZERO_WRITE_CALIBRATION',
            manifest_sha256=refs['manifest'], code_commit=self.c.COMMIT,
            submit_helper_sha256=hashlib.sha256((HELPER / 'submit_tree_fresh200_once.py').read_bytes()).hexdigest(),
            actual_3b_training_accepted=False, long_training_ready=False,
            original_sampler_actor_gate_accepted=False, production_thresholds_accepted=False,
            requires_zero_optimizer_writes=True, requires_actual_behavior_mu_binding=True,
            previous_terminal_evidence={})
        inputs = dict(manifest_sha256=refs['manifest'], code_commit=self.c.COMMIT,
            stage=self.c.STAGE, target=str(target),
            previous_terminal_evidence={'3314': dict(path='/changed', sha256='a' * 64)})
        with self.assertRaisesRegex(ValueError, 'not bound by independent release'):
            submit.validate_dispatch(self.c, target, {}, review, {}, inputs, {}, refs)


if __name__ == '__main__':
    started = time.perf_counter()
    log = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(IndependentTerminalCPU)
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    (OUT / 'TINY_TERMINAL_CPU.stdout').write_text(log.getvalue())
    record = dict(status='INDEPENDENT_TARGETED_HELPER_FUNCTION_CPU_PASSED' if result.wasSuccessful() else 'FAILED',
        actual_test_count=result.testsRun, errors=len(result.errors), failures=len(result.failures),
        wall_seconds=time.perf_counter()-started,
        scope='Exact AST-extracted functions. Captured actual3314 control/native run metadata and V2 proof in owned translated-path fixtures. Injected pure runner responses only; no main/route/subprocess/tool inquiries.',
        source_helper_files={name:hashlib.sha256((HELPER/name).read_bytes()).hexdigest() for name in
            ['prepare_tree_fresh200_package.py','submit_tree_fresh200_once.py','fresh200_dispatch_common.py']},
        historical_owner20_rerun=False, actual_remote_executed=False, actual_subprocess_tools_executed=False,
        actual_gpu_executed=False, actual_slurm_executed=False, actual_prepare_executed=False,
        actual_zero_accepted=False, actual_training_accepted=False)
    (OUT / 'TINY_TERMINAL_CPU.json').write_text(json.dumps(record, indent=2, sort_keys=True)+'\n')
    print(log.getvalue(), end='')
    print(json.dumps(record, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
