"""Synthetic receipt/sample and process-boundary tests, not real GPU admission."""
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from reproduction.common8 import gpu_training_admission as gpu
from reproduction.common8.policy_calibration_plan import canonical_bytes


class ActualGPUCoverageTests(unittest.TestCase):
    def fixture(self):
        run = dict(profile='common8',training_world_size=8,run_id='fresh200', started_unix=1., source_sha256={'runtime.py': 'a' * 64})
        claim = dict(run_id=run['run_id'], slurm_job_id='123', claimed_unix=10., shell_pid=41, shell_start_ticks=71)
        producer = dict(pid=40, start_ticks=70, sid=40, uid=os.getuid())
        allocation = dict(accepted=True, run_id=run['run_id'], slurm_job_id='123',
            actual_allocated_indices=list(range(8)),
            physical_gpu_inventory=[[str(i), 'GPU-' + str(i), '1000'] for i in range(8)],
            actual_cuda_runtime_uuids=['GPU-' + str(i) for i in range(8)])
        source = hashlib.sha256(canonical_bytes(run['source_sha256'])).hexdigest()
        rows = [dict(schema_version=1, role='actual_common8_gpu_occupancy', run_id=run['run_id'],
            slurm_job_id='123', source_map_sha256=source, sampler_identity=producer,
            sampling_interval_seconds=2, query_started_unix=end - .25, unix=float(end),
            gpus=[dict(index=i, uuid='GPU-' + str(i), used_mib=500 + i, total_mib=1000, util_percent=100)
                  for i in range(8)]) for end in range(8, 123, 2)]
        scheduler = {}
        for rank in range(8):
            scheduler[str(rank)] = dict(selected_rows=[dict(logical_update=step, kind=kind,
                observed_unix=step * 20 - (5 if kind == 'BEGIN' else -5))
                for step in range(1, 6) for kind in ('BEGIN', 'COMPLETE')])
        return rows, run, claim, allocation, scheduler

    def verify(self, item=None):
        rows, run, claim, allocation, scheduler = self.fixture() if item is None else item
        return gpu._coverage(rows, run, claim, allocation, scheduler, entered=120.)

    def test_first_five_window_reports_device_peaks_and_monitor_only_policy(self):
        item = self.fixture(); item[0][15]['gpus'][7]['used_mib'] = 980
        proof = self.verify(item)
        self.assertTrue(proof['accepted'])
        self.assertEqual(proof['sampled_peak_fraction']['7'], .98)
        self.assertEqual(proof['occupancy_policy'], 'monitor_only')
        self.assertIsNone(proof['sampled_peak_fraction_stop_limit'])
        self.assertEqual(proof['historical_sampled_peak_fraction_limit'], .98)
        self.assertFalse(proof['historical_limit_exceeded'])
        self.assertEqual(proof['maximum_sampling_gap_seconds'], 2.)
        self.assertEqual(proof['maximum_query_duration_seconds'], .25)
        self.assertEqual(len(proof['actual_device_inventory']), 8)
        self.assertTrue(all(window['sample_count_inside_optimizer_window'] > 0
                            for window in proof['actual_optimizer_windows']))
        self.assertGreater(proof['last_sample_unix'], proof['acceptance_entered_unix'])

    def test_foreign_run_job_source_sampler_uuid_total_or_missing_device_rejected(self):
        mutations = (
            lambda r: r.update(run_id='diagnostic'),
            lambda r: r.update(slurm_job_id='old'),
            lambda r: r.update(source_map_sha256='b' * 64),
            lambda r: r.update(sampler_identity=dict(r['sampler_identity'], start_ticks=99)),
            lambda r: r['gpus'][0].update(uuid='GPU-foreign'),
            lambda r: r['gpus'][0].update(total_mib=2000),
            lambda r: r['gpus'].pop(),
            lambda r: r['gpus'][0].update(index=1),
            lambda r: r['gpus'][0].update(used_mib=1001),
            lambda r: r['gpus'][0].update(used_mib=True),
            lambda r: r.update(query_started_unix=float('nan')),
        )
        for index, mutation in enumerate(mutations):
            item = self.fixture(); mutation(item[0][10])
            with self.subTest(index=index), self.assertRaises(AssertionError): self.verify(item)

    def test_preflight_only_stale_samples_no_training_bracket_or_missing_actual_outer_coverage_rejected(self):
        for mode in ('preflight', 'historical', 'no_before_claim', 'missing_outer'):
            rows, *rest = self.fixture()
            if mode == 'preflight': rows = rows[:2]
            if mode == 'historical': rows = [row for row in rows if row['query_started_unix'] < 120.]
            if mode == 'no_before_claim': rows = [row for row in rows if row['unix'] > 10.]
            if mode == 'missing_outer': rows = [row for row in rows if not (55 <= row['query_started_unix'] and row['unix'] <= 65)]
            with self.subTest(mode=mode), self.assertRaises(AssertionError): self.verify((rows, *rest))

    def test_reordered_sampling_or_scheduler_window_rejected(self):
        item = self.fixture(); item[0][10], item[0][11] = item[0][11], item[0][10]
        with self.assertRaises(AssertionError): self.verify(item)
        item = self.fixture(); item[4]['2']['selected_rows'][3]['observed_unix'] = 4.
        with self.assertRaises(AssertionError): self.verify(item)

    def test_gap_is_reported_without_silent_new_gap_threshold(self):
        rows, *rest = self.fixture()
        # Retain samples inside all five true optimizer windows plus brackets.
        rows = [row for row in rows if row['unix'] in (8, 10, 20, 40, 60, 80, 100, 122)]
        proof = self.verify((rows, *rest))
        self.assertEqual(proof['maximum_sampling_gap_seconds'], 22.)
        self.assertIn('instantaneous peaks between samples are unmeasured', proof['semantics'])

    def test_completed_attempt_window_excludes_retry_begin_but_peak_coverage_keeps_retry(self):
        item=self.fixture()
        for receipt in item[4].values():
            for row in receipt['selected_rows']:row['attempt_id']=f"{row['logical_update']:032x}"
            receipt['selected_rows'].insert(0,dict(logical_update=1,kind='BEGIN',
                attempt_id='a'*32,observed_unix=11.))
            receipt['selected_rows'].insert(1,dict(logical_update=1,kind='ABORTED_ATTEMPT',
                attempt_id='a'*32,observed_unix=14.))
        item[0][2]['gpus'][7]['used_mib']=980
        proof=self.verify(item)
        self.assertEqual(proof['actual_optimizer_windows'][0]['begin_unix'],15.)
        self.assertEqual(proof['sampled_peak_fraction']['7'],.98)
        item[0][2]['gpus'][7]['used_mib']=981
        proof=self.verify(item)
        self.assertTrue(proof['accepted'])
        self.assertEqual(proof['sampled_peak_fraction']['7'],.981)
        self.assertTrue(proof['historical_limit_exceeded'])

    def test_reader_waits_for_new_sample_after_gate_entry_and_checks_live_producer(self):
        rows, run, claim, *_ = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve(); (base / 'logs').mkdir()
            path = base / 'logs' / ('gpu-samples-' + run['run_id'] + '.jsonl')
            def write(selected):
                path.write_bytes(b'\n'.join(canonical_bytes(row) for row in selected) + b'\n')
            write(rows[:-1])
            def advance(_):
                with path.open('ab') as stream: stream.write(canonical_bytes(rows[-1]) + b'\n')
            with mock.patch.object(gpu, '_live_producer') as live, mock.patch.object(gpu.time, 'sleep', side_effect=advance):
                _, raw, selected, producer = gpu._sample_after_entry(base, run, claim, 120.)
            self.assertEqual(len(selected), len(rows))
            self.assertEqual(producer, rows[-1]['sampler_identity'])
            self.assertGreaterEqual(live.call_count, 2)
            self.assertTrue(raw.endswith(b'\n'))
            with mock.patch.object(gpu, '_live_producer', side_effect=AssertionError('dead sampler')):
                with self.assertRaises(AssertionError): gpu._sample_after_entry(base, run, claim, 120.)

    def test_unfinished_append_is_ignored_but_completed_malformed_row_rejected(self):
        rows, run, *_ = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / 'samples.jsonl'
            raw = canonical_bytes(rows[0]) + b'\n'
            path.write_bytes(raw + b'{"unfinished":')
            complete, selected = gpu._complete_rows(path)
            self.assertEqual(complete, raw); self.assertEqual(selected, rows[:1])
            path.write_bytes(raw + b'{"unfinished":\n')
            with self.assertRaises((ValueError, json.JSONDecodeError)): gpu._complete_rows(path)

    def test_live_but_stalled_sampler_is_rejected_with_bounded_wait(self):
        rows,run,claim,*_=self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory).resolve();(base/'logs').mkdir()
            path=base/'logs'/('gpu-samples-'+run['run_id']+'.jsonl')
            historical=[r for r in rows if r['query_started_unix']<120.]
            path.write_bytes(b'\n'.join(canonical_bytes(r) for r in historical)+b'\n')
            with mock.patch.object(gpu,'_live_producer'),mock.patch.object(gpu.time,'sleep'), \
                 mock.patch.object(gpu.time,'monotonic',side_effect=[0.,1.,61.]):
                with self.assertRaisesRegex(AssertionError,'fresh query progress'):
                    gpu._sample_after_entry(base,run,claim,120.)

    def test_sampler_operational_query_timeout_is_explicit_and_not_a_peak_threshold(self):
        from reproduction.common8.run_pipeline import gpu_rows
        with mock.patch('subprocess.check_output',return_value='0, GPU-fixture, 1, 1000, 0\n') as query:
            self.assertEqual(len(gpu_rows('index,uuid,memory.used,memory.total,utilization.gpu')),1)
        self.assertEqual(query.call_args.kwargs['timeout'],30)

    def test_live_sampler_training_shell_birth_and_owned_ledger_required(self):
        rows, _, claim, *_ = self.fixture(); producer = rows[0]['sampler_identity']
        leader = dict(pid=41, start_ticks=71, sid=41, uid=os.getuid())
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve(); (base / 'logs').mkdir()
            ledger = dict(run_id=claim['run_id'], jobs={'trainer': dict(state='STARTED', leader=leader)})
            path = base / 'logs/process-ledger.json'; path.write_bytes(canonical_bytes(ledger))
            def actual(pid): return producer if pid == 40 else leader
            with mock.patch.object(gpu, '_identity', side_effect=actual), \
                 mock.patch.object(Path, 'read_text', return_value='41 (bash) S 40'):
                gpu._live_producer(base, claim, producer)
            with mock.patch.object(gpu, '_identity', return_value=dict(producer, start_ticks=99)):
                with self.assertRaises(AssertionError): gpu._live_producer(base, claim, producer)
            with mock.patch.object(gpu, '_identity', side_effect=actual), \
                 mock.patch.object(Path, 'read_text', return_value='41 (bash) S 999'):
                with self.assertRaises(AssertionError): gpu._live_producer(base, claim, producer)
            ledger['jobs']['trainer']['state'] = 'STOPPED'; path.write_bytes(canonical_bytes(ledger))
            with mock.patch.object(gpu, '_identity', side_effect=actual):
                with self.assertRaises(AssertionError): gpu._live_producer(base, claim, producer)


if __name__ == '__main__': unittest.main()
