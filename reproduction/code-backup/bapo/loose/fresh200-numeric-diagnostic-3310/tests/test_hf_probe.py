import importlib.util
from pathlib import Path
import unittest
import torch
import tempfile
import json
from types import SimpleNamespace
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('probe', Path(__file__).parents[1] / 'hf-probe.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class ProbeTests(unittest.TestCase):
    def test_selected_mask_ties_and_double_delta(self):
        tensors = dict(old_log_probs=torch.tensor([[-1., -2.], [-3., -4.]]),
            sampling_log_probs=torch.tensor([[-2., -1.], [-100., -4.]]),
            sampling_mask=torch.tensor([[1, 1], [0, 1]]), responses=torch.tensor([[2, 3], [4, 5]]))
        points = probe.selected_points(tensors, 2)
        self.assertEqual([(p['row'], p['position']) for p in points], [(0, 0), (0, 1)])
        self.assertEqual([p['token'] for p in points], [2, 3])

    def test_selected_nonfinite_fail_closed(self):
        tensors = dict(old_log_probs=torch.tensor([[float('nan')]]), sampling_log_probs=torch.tensor([[-1.]]),
                       sampling_mask=torch.tensor([[1]]), responses=torch.tensor([[2]]))
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            probe.selected_points(tensors)

    def test_turn_mapping_environment_and_whole_boundary_token(self):
        turns = [dict(actual_request_token_ids=[10], raw_generated_token_ids=[2, 3, 99],
                      kept_token_count=2, environment_token_ids=[8, 9]),
                 dict(actual_request_token_ids=[10, 2, 3, 8, 9], raw_generated_token_ids=[4, 5],
                      kept_token_count=2, environment_token_ids=[])]
        ordinal, offset, turn = probe.locate_turn(turns, 5)
        self.assertEqual((ordinal, offset), (1, 1))
        batch = probe.turn_batch(turn)
        self.assertEqual(batch['input_ids'].tolist(), [[10, 2, 3, 8, 9, 4, 5]])
        self.assertEqual(batch['responses'].tolist(), [[4, 5]])
        self.assertEqual(batch['position_ids'].tolist(), [[0, 1, 2, 3, 4, 5, 6]])
        truncated = probe.turn_batch(turn, 0)
        self.assertEqual(truncated['input_ids'].tolist(), [[10, 2, 3, 8, 9, 4]])
        self.assertEqual(truncated['responses'].tolist(), [[4]])
        with self.assertRaises(ValueError):
            probe.locate_turn(turns, 3)

    def test_low_error_controls_distinct_turns(self):
        tensors = dict(old_log_probs=torch.tensor([[-1., -2., 0., -3., -4.]]),
            sampling_log_probs=torch.tensor([[-1.2, -2., 0., -3.1, -4.3]]), responses=torch.tensor([[2, 3, 8, 4, 5]]))
        journals = [[dict(kept_token_count=2, environment_token_ids=[8]),
                     dict(kept_token_count=2, environment_token_ids=[])]]
        points = probe.low_error_controls(tensors, journals)
        self.assertEqual([p['position'] for p in points], [1, 3])
        self.assertEqual([p['point_role'] for p in points], ['LOW_ERROR_TURN_0', 'LOW_ERROR_TURN_1'])

    def test_new_neighbors_label_and_target_preserved(self):
        cases = probe.groups([dict(row=5), dict(row=7)], 16)
        self.assertIn(('NEW_PACKED_PROBE_MICRO4', [4, 5, 6, 7]), cases)
        changed = [rows for mode, rows in cases if mode == 'NEW_PACKED_PROBE_CHANGED_NEIGHBORS']
        self.assertEqual(changed[0][1], 5)
        self.assertEqual(changed[1][3], 7)
        self.assertEqual(len(set(changed[0])), 4)
        self.assertTrue(all(0 <= row < 16 for rows in changed for row in rows))

    def test_future_token_control_preserves_shape_target_and_padding(self):
        batch = dict(input_ids=torch.tensor([[10, 11, 2, 3, 4, 0]]), responses=torch.tensor([[2, 3, 4, 0]]),
            attention_mask=torch.tensor([[1, 1, 1, 1, 1, 0]]), position_ids=torch.tensor([[0, 1, 2, 3, 4, 0]]))
        changed, indices = probe.future_token_control(batch, 0, 1, 100)
        self.assertEqual(indices, [2])
        self.assertEqual(changed['input_ids'].tolist(), [[10, 11, 2, 3, 5, 0]])
        self.assertEqual(changed['responses'].tolist(), [[2, 3, 5, 0]])
        self.assertTrue(torch.equal(changed['attention_mask'], batch['attention_mask']))
        self.assertTrue(torch.equal(changed['position_ids'], batch['position_ids']))
        self.assertEqual(batch['input_ids'].tolist(), [[10, 11, 2, 3, 4, 0]])

    def test_parameter_hash_detects_actual_mutation_no_optimizer(self):
        model = torch.nn.Linear(3, 2, dtype=torch.bfloat16)
        before = probe.parameter_digest(model)
        self.assertEqual(before, probe.parameter_digest(model))
        with torch.no_grad():
            model.weight[0, 0] += 1
        self.assertNotEqual(before, probe.parameter_digest(model))

    def test_bound_diagnostic_receipt_and_wrong_job_source_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            nonce = 'a' * 32
            run = root / 'runs' / ('hf-zero-adam-replay-' + nonce + '-123')
            receipt_path = root / 'receipt.json'
            receipt = dict(phase='hf-zero-adam-replay', source=probe.SOURCE, nonce=nonce, job_id=123,
                comment='BAPO_ZERO_ADAM:hf-zero-adam-replay:' + probe.SOURCE + ':' + nonce)
            receipt_path.write_text(json.dumps(receipt))
            approval = dict(protocol='BAPO_ZERO_ADAM_DIAGNOSTIC_V1', status='ALLOW_ZERO_ADAM_PROBABILITY_REPLAY',
                source_commit=probe.SOURCE, raw_sha256=probe.RAW_SHA, hf_probe_sha256=probe.digest(probe.__file__),
                native_acceptance=False, production_release=False, optimizer_writes=0, formal_outer_updates=0,
                scheduler_updates=0, allocation_gpu_count=8, probe_gpu_count=1, root=str(root), nonce=nonce,
                job_receipt_path=str(receipt_path), repo=str(root/'repo'), raw_batch=str(root/'raw'), model=str(root/'model'))
            approval_path = root / 'approval.json'
            approval_path.write_text(json.dumps(approval))
            args = SimpleNamespace(repo=approval['repo'], raw_batch=approval['raw_batch'], model=approval['model'],
                                   output=str(run/'result.json'))
            env = dict(FRESH200_ZERO_ADAM_GUARD_READY='1', BAPO_DIAGNOSTIC_RECEIPT=str(receipt_path),
                       SLURM_JOB_ID='123', BAPO_RUN_ID=str(run))
            with patch.dict(probe.os.environ, env, clear=True):
                self.assertEqual(probe.check_approval(approval_path, args)['job_id'], 123)
                with patch.dict(probe.os.environ, {'SLURM_JOB_ID': '124'}):
                    with self.assertRaisesRegex(ValueError, 'Slurm'):
                        probe.check_approval(approval_path, args)
                args.output = str(root/'foreign.json')
                with self.assertRaisesRegex(ValueError, 'outside'):
                    probe.check_approval(approval_path, args)
                args.output = str(run/'result.json')
                approval['source_commit'] = 'old-source'
                approval_path.write_text(json.dumps(approval))
                with self.assertRaisesRegex(ValueError, 'source/raw'):
                    probe.check_approval(approval_path, args)


if __name__ == '__main__':
    unittest.main()
