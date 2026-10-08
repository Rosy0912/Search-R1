"""Narrow CPU counterexamples; actual preflight checker and own CPU children only."""
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
RUNTIME = Path(__file__).resolve().parent.parent / 'safety-calibration-v6'
sys.path.insert(0, str(RUNTIME))
from calibration_run_v1 import wait_retriever_gpu_coexistence, query_gpu_coexistence_snapshot

SOURCE = ROOT / 'work/gigpo-e1-quality-resume/search_r1/baselines/preflight.py'
sys.path.insert(0, str(SOURCE.parents[2]))
spec = importlib.util.spec_from_file_location('source_preflight_readonly_v6_test', SOURCE)
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)


class Clock:
    now = 0.
    def monotonic(self): return self.now
    def sleep(self, seconds): self.now += seconds


class Child:
    def __init__(self, pid): self.pid, self.exit = pid, None
    def poll(self): return self.exit


def inventory(util=0., used=10000., total=100000.):
    return {str(i): dict(uuid=f'GPU-test-{i}', memory_used=used,
                        memory_total=total, utilization=util) for i in range(8)}


class CooldownTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.run = Path(self.temp.name)
        self.clock = Clock()
        self.guard, self.retriever = Child(101), Child(102)
        self.snapshot_calls = 0
        self.checker_calls = 0
        self.samples = [inventory()]
        self.eligible_hook = None
        self.snapshot_hook = None

    def tearDown(self): self.temp.cleanup()

    def snapshot(self, remaining):
        self.snapshot_calls += 1
        self.assertGreater(remaining, 0.)
        self.assertLessEqual(remaining, 60.)
        if self.snapshot_hook: self.snapshot_hook()
        sample = self.samples[min(self.snapshot_calls - 1, len(self.samples) - 1)]
        return sample, {'GPU-test-0'}, dict(gpu_inventory_csv='RAW ORIGINAL CPU FIXTURE',
                                           compute_apps_csv='GPU-test-0,101\n')

    def checker(self, selected, records, processes):
        self.checker_calls += 1
        result = preflight.check_selected_gpus(selected, records, processes)
        if self.eligible_hook: self.eligible_hook()
        return result

    def run_gate(self, **kwargs):
        options = dict(snapshot=self.snapshot, checker=self.checker,
                       monotonic=self.clock.monotonic, sleep=self.clock.sleep)
        options.update(kwargs)
        return wait_retriever_gpu_coexistence(self.run, [str(i) for i in range(8)],
                                              self.guard, self.retriever, **options)

    def rows(self):
        return [json.loads(v) for v in (self.run / 'gpu-retriever-coexistence-before-ray.jsonl').read_text().splitlines()]

    def receipt(self):
        return json.loads((self.run / 'gpu-retriever-coexistence-before-ray.json').read_text())

    def test_exact15_sm_and_memory_still_pass(self):
        self.samples = [inventory(util=15., used=15000.)]
        row = self.run_gate()
        self.assertEqual(row['decision'], 'PASS_ORIGINAL_TOTAL15')
        self.assertEqual(self.clock.now, 0.)

    def test_own_probe16_cools_to15_with_original_snapshot(self):
        self.samples = [inventory(util=16.), inventory(util=15.)]
        row = self.run_gate()
        self.assertEqual(self.clock.now, 3.)
        self.assertEqual(row['attempt'], 2)
        rows = self.rows()
        self.assertEqual(rows[0]['gate_snapshot']['inventory']['0']['utilization'], 16.)
        self.assertEqual(rows[0]['decision'], 'WAIT_RETRY')
        self.assertEqual(rows[1]['decision'], 'PASS_ORIGINAL_TOTAL15')
        self.assertEqual(rows[0]['raw_gpu_sample']['gpu_inventory_csv'], 'RAW ORIGINAL CPU FIXTURE')

    def test_memory15_point01_cools_but_is_not_admitted_early(self):
        self.samples = [inventory(used=15001.), inventory(used=15000.)]
        self.run_gate()
        self.assertEqual([r['decision'] for r in self.rows()], ['WAIT_RETRY', 'PASS_ORIGINAL_TOTAL15'])

    def test_both_thresholds_must_pass_on_all8gpus(self):
        first = inventory(); first['7']['utilization'] = 16.
        second = inventory(); second['3']['memory_used'] = 15001.
        self.samples = [first, second, inventory()]
        self.run_gate()
        self.assertEqual(self.snapshot_calls, 3)
        self.assertEqual(self.clock.now, 6.)

    def test_persistent_sm_ends_at60_no_admission(self):
        self.samples = [inventory(util=16.)]
        with self.assertRaisesRegex(RuntimeError, 'after60s'): self.run_gate()
        self.assertEqual(self.clock.now, 60.)
        self.assertEqual(self.snapshot_calls, 20)
        self.assertEqual(self.receipt()['decision'], 'FAILED_CLOSED')
        self.assertFalse(any(r['decision'] == 'PASS_ORIGINAL_TOTAL15' for r in self.rows()))

    def test_persistent_memory_ends_at60_no_admission(self):
        self.samples = [inventory(used=15001.)]
        with self.assertRaisesRegex(RuntimeError, 'after60s'): self.run_gate()
        self.assertEqual(self.clock.now, 60.)

    def test_nonfinite_unknown_reading_never_passes_and_is_retained(self):
        self.samples = [inventory(util=float('nan')), inventory()]
        with self.assertRaisesRegex(RuntimeError, 'unknown/invalid occupancy'): self.run_gate()
        self.assertEqual(self.snapshot_calls, 1)
        self.assertEqual(self.clock.now, 0.)
        self.assertEqual(self.rows()[0]['gate_snapshot']['inventory']['0']['utilization'], {'nonfinite_float_repr':'nan'})
        self.assertNotIn('PASS_ORIGINAL_TOTAL15', [r['decision'] for r in self.rows()])

    def test_nonpositive_total_memory_never_passes(self):
        self.samples = [inventory(total=0.), inventory()]
        with self.assertRaisesRegex(RuntimeError, 'unknown/invalid occupancy'): self.run_gate()
        self.assertEqual(self.snapshot_calls, 1)
        self.assertEqual(self.clock.now, 0.)

    def test_invalid_sm_above100_then_normal_is_refused_on_first_sample(self):
        self.samples = [inventory(util=101.), inventory()]
        with self.assertRaisesRegex(RuntimeError, 'unknown/invalid occupancy'): self.run_gate()
        self.assertEqual(self.snapshot_calls, 1)
        self.assertEqual(self.clock.now, 0.)

    def test_invalid_memory_nan_then_normal_is_refused_on_first_sample(self):
        self.samples = [inventory(used=float('nan')), inventory()]
        with self.assertRaisesRegex(RuntimeError, 'unknown/invalid occupancy'): self.run_gate()
        self.assertEqual(self.snapshot_calls, 1)
        self.assertEqual(self.clock.now, 0.)

    def test_unknown_checker_runtimeerror_does_not_retry(self):
        def unknown_checker(*args): raise RuntimeError('Unknown resource state')
        with self.assertRaisesRegex(RuntimeError, 'Unknown resource state'):
            self.run_gate(checker=unknown_checker)
        self.assertEqual(self.snapshot_calls, 1)
        self.assertEqual(self.clock.now, 0.)

    def test_fake_threshold_message_other_device_is_not_retried(self):
        def unknown_checker(*args): raise RuntimeError('Allocated GPU unbound exceeds the 15% idle thresholds')
        with self.assertRaisesRegex(RuntimeError, 'unbound'): self.run_gate(checker=unknown_checker)
        self.assertEqual(self.snapshot_calls, 1)

    def test_mixed_first_busy_later_nan_cannot_wait_for_valid_next_sample(self):
        sample = inventory(); sample['0']['utilization'] = 16.; sample['7']['utilization'] = float('nan')
        self.samples = [sample, inventory()]
        with self.assertRaisesRegex(RuntimeError, 'unknown/invalid occupancy'): self.run_gate()
        self.assertEqual(self.snapshot_calls, 1)
        self.assertEqual(self.clock.now, 0.)
        self.assertEqual(self.receipt()['decision'], 'FAILED_CLOSED')

    def test_mixed_first_busy_later_zero_total_cannot_wait(self):
        sample = inventory(); sample['0']['utilization'] = 16.; sample['7']['memory_total'] = 0.
        self.samples = [sample, inventory()]
        with self.assertRaisesRegex(RuntimeError, 'unknown/invalid occupancy'): self.run_gate()
        self.assertEqual(self.snapshot_calls, 1)

    def test_mixed_first_busy_later_sm101_cannot_wait(self):
        sample = inventory(); sample['0']['utilization'] = 16.; sample['7']['utilization'] = 101.
        self.samples = [sample, inventory()]
        with self.assertRaisesRegex(RuntimeError, 'unknown/invalid occupancy'): self.run_gate()
        self.assertEqual(self.snapshot_calls, 1)

    def test_mixed_first_busy_later_missing_gpu_cannot_wait(self):
        sample = inventory(); sample['0']['utilization'] = 16.; del sample['7']
        self.samples = [sample, inventory()]
        with self.assertRaisesRegex(RuntimeError, 'Cannot resolve every allocated GPU'): self.run_gate()
        self.assertEqual(self.snapshot_calls, 1)

    def test_mixed_first_busy_later_duplicate_uuid_cannot_wait(self):
        sample = inventory(); sample['0']['utilization'] = 16.; sample['7']['uuid'] = sample['6']['uuid']
        self.samples = [sample, inventory()]
        with self.assertRaisesRegex(RuntimeError, 'Duplicate allocated GPU UUIDs'): self.run_gate()
        self.assertEqual(self.snapshot_calls, 1)

    def test_unknown_gpu_identity_fails_immediately(self):
        sample = inventory(); del sample['7']; self.samples = [sample]
        with self.assertRaisesRegex(ValueError, 'Cannot resolve'): self.run_gate()
        self.assertEqual(self.clock.now, 0.)
        self.assertEqual(self.receipt()['decision'], 'FAILED_CLOSED')

    def test_duplicate_uuid_fails_immediately(self):
        sample = inventory(); sample['7']['uuid'] = sample['6']['uuid']; self.samples = [sample]
        with self.assertRaisesRegex(ValueError, 'duplicate devices'): self.run_gate()
        self.assertEqual(self.clock.now, 0.)

    def test_snapshot_exception_fails_closed_before_checker(self):
        def broken(remaining): raise OSError('nvidia-smi unavailable')
        with self.assertRaisesRegex(OSError, 'unavailable'): self.run_gate(snapshot=broken)
        self.assertEqual(self.checker_calls, 0)
        self.assertEqual(self.receipt()['error_type'], 'OSError')

    def test_snapshot_timeout_is_terminal_not_retried(self):
        def broken(remaining): raise subprocess.TimeoutExpired('nvidia-smi', remaining)
        with self.assertRaises(subprocess.TimeoutExpired): self.run_gate(snapshot=broken)
        self.assertEqual(self.snapshot_calls, 0)
        self.assertEqual(self.checker_calls, 0)
        self.assertEqual(self.receipt()['decision'], 'FAILED_CLOSED')

    def test_guard_dead_before_sample_refused(self):
        self.guard.exit = 1
        with self.assertRaisesRegex(RuntimeError, 'exited during'): self.run_gate()
        self.assertEqual(self.snapshot_calls, 0)
        self.assertEqual(self.receipt()['dependency_exit_codes']['guard'], 1)

    def test_retriever_exit0_is_still_refused(self):
        self.retriever.exit = 0
        with self.assertRaisesRegex(RuntimeError, 'exited during'): self.run_gate()
        self.assertEqual(self.snapshot_calls, 0)

    def test_guard_dies_during_snapshot_cannot_be_admitted(self):
        self.snapshot_hook = lambda: setattr(self.guard, 'exit', 1)
        with self.assertRaisesRegex(RuntimeError, 'while reading'): self.run_gate()
        self.assertEqual(self.checker_calls, 0)
        self.assertIsNotNone(self.receipt()['gate_snapshot'])

    def test_retriever_dies_during_checker_cannot_be_admitted(self):
        self.eligible_hook = lambda: setattr(self.retriever, 'exit', 1)
        with self.assertRaisesRegex(RuntimeError, 'before GPU coexistence'): self.run_gate()
        self.assertNotEqual(self.receipt()['decision'], 'PASS_ORIGINAL_TOTAL15')

    def test_snapshot_running_beyond_deadline_cannot_admit(self):
        self.snapshot_hook = lambda: setattr(self.clock, 'now', 60.)
        with self.assertRaisesRegex(RuntimeError, 'snapshot exceeded60s'): self.run_gate()
        self.assertEqual(self.checker_calls, 0)

    def test_checker_running_beyond_deadline_cannot_admit(self):
        self.eligible_hook = lambda: setattr(self.clock, 'now', 60.)
        with self.assertRaisesRegex(RuntimeError, 'admission exceeded60s'): self.run_gate()

    def test_runtime_cooldown_precedes_ray_and_driver_start(self):
        source = (RUNTIME / 'calibration_run_v1.py').read_text()
        begin = source.index("result['retriever_coexistence_admission'] =")
        self.assertLess(begin, source.index("head = owned.spawn"))
        self.assertLess(begin, source.index("driver = owned.spawn"))
        self.assertEqual(source[begin:].count('check_selected_gpus('), 0)

    def test_bounded_query_raw_and_parsed_use_same_original_read(self):
        raw = ''.join(f'{i},GPU-test-{i},10000,100000,16\n' for i in range(8))
        calls = []
        def runner(command, **kwargs):
            calls.append((command, kwargs))
            if len(calls) == 1:
                self.clock.now += 7.
                return raw
            return 'GPU-test-0,101\n'
        parsed, processes, original = query_gpu_coexistence_snapshot(
            60., command_runner=runner, monotonic=self.clock.monotonic)
        self.assertEqual(original['gpu_inventory_csv'], raw)
        self.assertEqual(parsed['0']['utilization'], 16.)
        self.assertEqual(processes, {'GPU-test-0'})
        self.assertEqual(calls[0][1]['timeout'], 60.)
        self.assertEqual(calls[1][1]['timeout'], 53.)

    def test_first_query_timeout_no_second_command(self):
        calls = []
        def runner(command, **kwargs):
            calls.append(command)
            raise subprocess.TimeoutExpired(command, kwargs['timeout'])
        with self.assertRaises(subprocess.TimeoutExpired):
            query_gpu_coexistence_snapshot(60., command_runner=runner, monotonic=self.clock.monotonic)
        self.assertEqual(len(calls), 1)

    def test_first_query_at_deadline_no_second_command(self):
        calls = []
        def runner(command, **kwargs):
            calls.append(command); self.clock.now = 60.
            return '0,GPU-test-0,10000,100000,0\n'
        with self.assertRaisesRegex(RuntimeError, 'inventory query exceeded'):
            query_gpu_coexistence_snapshot(60., command_runner=runner, monotonic=self.clock.monotonic)
        self.assertEqual(len(calls), 1)

    def test_second_query_timeout_uses_only_remaining_budget(self):
        calls = []
        def runner(command, **kwargs):
            calls.append(kwargs['timeout'])
            if len(calls) == 1:
                self.clock.now = 17.
                return '0,GPU-test-0,10000,100000,0\n'
            raise subprocess.TimeoutExpired(command, kwargs['timeout'])
        with self.assertRaises(subprocess.TimeoutExpired):
            query_gpu_coexistence_snapshot(60., command_runner=runner, monotonic=self.clock.monotonic)
        self.assertEqual(calls, [60., 43.])

    def test_malformed_original_inventory_not_fabricated(self):
        def runner(command, **kwargs): return 'malformed-gpu-csv\n'
        with self.assertRaises(ValueError) as caught:
            query_gpu_coexistence_snapshot(60., command_runner=runner, monotonic=self.clock.monotonic)
        self.assertEqual(caught.exception.gpu_coexistence_raw_snapshot['gpu_inventory_csv'], 'malformed-gpu-csv\n')

    def test_failed_query_originals_reach_failed_gate_receipt(self):
        def runner(command, **kwargs): return 'malformed-gpu-csv\n'
        def snapshot(remaining):
            return query_gpu_coexistence_snapshot(remaining, command_runner=runner, monotonic=self.clock.monotonic)
        with self.assertRaises(ValueError): self.run_gate(snapshot=snapshot)
        self.assertEqual(self.receipt()['raw_gpu_sample']['gpu_inventory_csv'], 'malformed-gpu-csv\n')
        self.assertIsNone(self.receipt()['gate_snapshot'])

    def test_partial_timeout_stdout_retained_without_gate_substitution(self):
        def runner(command, **kwargs):
            raise subprocess.TimeoutExpired(command, kwargs['timeout'], output=b'PARTIAL GPU RAW')
        def snapshot(remaining):
            return query_gpu_coexistence_snapshot(remaining, command_runner=runner, monotonic=self.clock.monotonic)
        with self.assertRaises(subprocess.TimeoutExpired): self.run_gate(snapshot=snapshot)
        self.assertEqual(self.receipt()['raw_gpu_sample']['failed_command_partial_output'], 'PARTIAL GPU RAW')
        self.assertIsNone(self.receipt()['gate_snapshot'])

    def test_command_timeout_budget_cannot_exceed60(self):
        with self.assertRaisesRegex(RuntimeError, 'Invalid GPU cooldown command budget'):
            query_gpu_coexistence_snapshot(61., command_runner=lambda *a, **k: '')

    def test_actual_cpu_command_timeout_is_terminated_by_runner(self):
        # No nvidia-smi invocation: only our own CPU sleep is launched, timed out
        # and waited by subprocess.check_output's context manager.
        def runner(command, **kwargs):
            return subprocess.check_output([sys.executable, '-c', 'import time; time.sleep(10)'], **kwargs)
        with self.assertRaises(subprocess.TimeoutExpired):
            query_gpu_coexistence_snapshot(.05, command_runner=runner)

    def own_cpu_pair(self):
        self.guard = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])
        self.retriever = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])
        return self.guard, self.retriever

    @staticmethod
    def cleanup_own(children):
        for child in children:
            if child.poll() is None: child.terminate()
        for child in children: child.wait(timeout=10)

    def test_actual_cpu_dependency_processes_live_at_admission(self):
        children = self.own_cpu_pair()
        try:
            self.samples = [inventory(util=16.), inventory()]
            row = self.run_gate()
            self.assertEqual(row['guard_pid'], children[0].pid)
            self.assertEqual(row['retriever_pid'], children[1].pid)
            self.assertTrue(all(c.poll() is None for c in children))
        finally: self.cleanup_own(children)

    def test_actual_cpu_guard_terminates_during_cooldown_zero_admission(self):
        children = self.own_cpu_pair()
        try:
            self.samples = [inventory(util=16.)]
            def terminate_guard(seconds):
                children[0].terminate(); children[0].wait(timeout=10)
                self.clock.sleep(seconds)
            with self.assertRaisesRegex(RuntimeError, 'exited during'):
                self.run_gate(sleep=terminate_guard)
            self.assertEqual(self.snapshot_calls, 1)
            self.assertEqual(self.receipt()['decision'], 'FAILED_CLOSED')
        finally: self.cleanup_own(children)


if __name__ == '__main__': unittest.main(verbosity=2)
