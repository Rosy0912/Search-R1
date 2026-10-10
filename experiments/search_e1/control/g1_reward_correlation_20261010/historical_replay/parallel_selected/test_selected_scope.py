import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE=Path(__file__).parent

def load(name):
    spec=importlib.util.spec_from_file_location(name,HERE/(name+'.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
core=load('replay_core');runner=load('replay_gpu')

class SelectedScope(unittest.TestCase):
    def test_exact_seven_new_points(self):
        self.assertEqual(runner.validate_selected_steps([20,40,60,80,100,120,140],core),[20,40,60,80,100,120,140])
        self.assertEqual(runner.FINAL_SELECTED_STEPS,[1,20,40,60,80,100,120,140])
    def test_worker_subset_allowed(self):
        self.assertEqual(runner.validate_selected_steps([60],core),[60])
        self.assertEqual(runner.validate_selected_steps([20,40],core),[20,40])
    def test_extra_unselected_point_rejected(self):
        with self.assertRaises(core.Hold):runner.validate_selected_steps([20,40,60,80,100,120,140,146],core)
    def test_duplicates_rejected(self):
        with self.assertRaises(core.Hold):runner.validate_selected_steps([20,40,60,80,100,120,120],core)
    def test_order_rejected(self):
        with self.assertRaises(core.Hold):runner.validate_selected_steps([40,20,60,80,100,120,140],core)
    def test_bool_not_integer_point(self):
        with self.assertRaises(core.Hold):runner.validate_selected_steps([True,40,60,80,100,120,140],core)
    def test_dense_range_not_selected(self):
        with self.assertRaises(core.Hold):runner.validate_selected_steps(list(range(20,141)),core)
    def test_step1_not_replayed(self):
        with self.assertRaises(core.Hold):runner.validate_selected_steps([1,20,40,60,80,100,120,140],core)
    def files(self,p):
        p=p.resolve()
        rows=p/'rows.jsonl'
        records=[dict(run_id='fixture-run',step=1,attempt_id='a'*32,probe_version='fixture:0:head',
            original_online_R_available=False,balanced_request_index=i,original_sample_index=2559-i,
            trajectory_id='fixture-run-1-attempt-'+('a'*32)+'-sample-'+str(i)) for i in range(2560)]
        rows.write_bytes(b''.join(core.canonical(r)+b'\n' for r in records))
        doc=dict(status='ACCEPTED_RECONSTRUCTED_R_STEP',step=1,N=2560,trajectory_count=2560,
            probe_version='fixture:0:head',original_trace_sha256='b'*64,
            native_scalar_comparison={'passed':True},original_online_row_rewards_available=False,
            output_path=str(rows),output_sha256=hashlib.sha256(rows.read_bytes()).hexdigest())
        summary=p/'summary.json';summary.write_bytes(core.canonical(doc))
        cfg=dict(expected_run_id='fixture-run',reused_step1_summary_path=str(summary),reused_step1_summary_sha256=hashlib.sha256(summary.read_bytes()).hexdigest())
        return cfg,rows,summary,doc
    def test_accepted_reused_point_no_rescore(self):
        with tempfile.TemporaryDirectory() as d:
            cfg,rows,summary,doc=self.files(Path(d))
            v=runner.accepted_reused_step1(cfg,core)
            self.assertNotIn('attempt_id',doc)  # Exact preserved dense-summary schema.
            self.assertEqual(v['reused_native_attempt_id'],'a'*32)
            self.assertEqual(v['reused_verified_row_count'],2560)
            runner.bind_reused_step1_to_original_ledger(v,[dict(step=1,attempt_id='a'*32,
                behavior_trace_sha256='b'*64,probe_version='fixture:0:head')],core)
            self.assertEqual(json.loads(summary.read_bytes()),doc)  # Original summary remains unmodified.
    def test_reused_rows_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            cfg,rows,summary,doc=self.files(Path(d));rows.write_bytes(b'changed')
            with self.assertRaises(core.Hold):runner.accepted_reused_step1(cfg,core)
    def test_reused_incomplete_count_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            cfg,rows,summary,doc=self.files(Path(d));doc['N']=2559;summary.write_bytes(core.canonical(doc))
            cfg['reused_step1_summary_sha256']=hashlib.sha256(summary.read_bytes()).hexdigest()
            with self.assertRaises(core.Hold):runner.accepted_reused_step1(cfg,core)
    def test_reused_first_ledger_attempt_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            cfg,rows,summary,doc=self.files(Path(d));v=runner.accepted_reused_step1(cfg,core)
            with self.assertRaises(core.Hold):runner.bind_reused_step1_to_original_ledger(v,[dict(step=1,attempt_id='c'*32,
                behavior_trace_sha256='b'*64,probe_version='fixture:0:head')],core)
    def test_reused_multiple_attempts_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            cfg,rows,summary,doc=self.files(Path(d));records=[json.loads(x) for x in rows.read_bytes().splitlines()]
            records[-1]['attempt_id']='c'*32;records[-1]['trajectory_id']='fixture-run-1-attempt-'+('c'*32)+'-sample-2559'
            rows.write_bytes(b''.join(core.canonical(r)+b'\n' for r in records));doc['output_sha256']=hashlib.sha256(rows.read_bytes()).hexdigest()
            summary.write_bytes(core.canonical(doc));cfg['reused_step1_summary_sha256']=hashlib.sha256(summary.read_bytes()).hexdigest()
            with self.assertRaises(core.Hold):runner.accepted_reused_step1(cfg,core)
    def test_summary_explicit_selected_not_contiguous(self):
        source=(HERE/'replay_gpu.py').read_text()
        self.assertIn('ACCEPTED_COMPLETE_PARALLEL_WORKER_SELECTED_RECONSTRUCTED_R_SCOPE',source)
        self.assertIn('contiguous_TRAIN_scope_complete=False',source)
        self.assertNotIn("list(range(config['start_step'], config['end_step'] + 1))",source)

if __name__=='__main__':unittest.main()
