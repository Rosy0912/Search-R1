"""Constructed owned scalar logs; no native GPU or production admission claim."""
import copy
import json
import tempfile
import unittest

from reproduction.common8.actor_probe_rank_tail import read_unreturned_tail
from reproduction.common8.policy_calibration_plan import document_sha256
from reproduction.common8.formal_contract import consumed_lr
from reproduction.common8.scheduler import expected_lr
from tests.reproduction.test_actor_probe_namespace_audit import fixture, rows


def open_tail(root, worker_lengths=(10, 8, 8, 0), scheduler_lengths=(1, 1, 1, 1)):
    base, run, _, updates = fixture(root, start=50)
    last = updates[-1]; cursor = last['attempt_index']; step = last['step'] + 1
    attempt = f'{cursor + 10001:032x}'; gate = document_sha256(cursor + 1)
    def advance(value):
        value = copy.deepcopy(value)
        if 'context' in value:
            value['context'] = advance(value['context'])
        for key in ('logical_update', 'completed_outer_before', 'completed_outer_after',
                    'scheduler_last_epoch_before', 'scheduler_last_epoch_after',
                    'scheduler_step_count_before', 'scheduler_step_count_after'):
            if key in value: value[key] += 1
        if 'attempt_index' in value:
            value.update(attempt_index=cursor + 1, attempt_id=attempt)
        for key in ('prior_optimizer_writes', 'optimizer_writes_before', 'optimizer_writes_after',
                    'write_count', 'writes_before', 'writes_after', 'cumulative_writes'):
            if key in value: value[key] += 20
        if value.get('kind') == 'POLICY_MONITOR': value['optimizer_writes'] += 20
        if 'observed_unix' in value: value['observed_unix'] += 100
        for key in ('behavior_probability_gate_sha256', 'batch_gate_sha256'):
            if key in value: value[key] = gate
        if 'learning_rates' in value: value['learning_rates'] = [consumed_lr(step)]
        if 'consumed_learning_rates' in value: value['consumed_learning_rates'] = [consumed_lr(step)]
        if 'next_learning_rates' in value:
            value['next_learning_rates'] = [expected_lr(step if value['kind'] == 'COMPLETE' else step - 1)]
        return value
    for rank in range(4):
        for kind, length in (('optimizer', worker_lengths[rank]), ('scheduler', scheduler_lengths[rank])):
            path = base/'logs'/f'{kind}-rank-{rank}.jsonl'
            values = [json.loads(line) for line in path.read_text().splitlines()]
            selected = [row for row in values if (row.get('context') or row).get('attempt_index') == cursor]
            rows(path, values + [advance(row) for row in selected[:length]])
    return base, run, last


class RankTailTests(unittest.TestCase):
    def test_partial_rank_writes_are_lower_bound_evidence_and_not_a_complete_outer(self):
        with tempfile.TemporaryDirectory() as root:
            base, run, last = open_tail(root)
            path = base/'logs/optimizer-rank-2.jsonl'
            path.write_bytes(path.read_bytes() + b'{"incomplete')
            proof = read_unreturned_tail(base, run, retained_through=last['step'])
            self.assertEqual([proof['per_rank'][str(rank)]['recorded_adam_writes'] for rank in range(4)], [3, 2, 2, 0])
            self.assertEqual(proof['all_rank_recorded_adam_writes'], 0)
            self.assertEqual(proof['counted_new_complete_outers'], 0)
            fragment = next(row for row in proof['files'] if row['path'] == str(path))
            self.assertEqual(fragment['trailing_fragment_bytes'], 12)
            self.assertFalse(proof['production_admission'])
            self.assertFalse(proof['current_process_exit_proven'])
            self.assertFalse(proof['full_training_or_final_acceptance'])

    def test_all_rank_terminal_and_scheduler_complete_without_driver_return_still_count_zero(self):
        with tempfile.TemporaryDirectory() as root:
            base, run, last = open_tail(root, (81, 81, 81, 81), (3, 3, 3, 3))
            proof = read_unreturned_tail(base, run, retained_through=last['step'])
            self.assertEqual(proof['all_rank_recorded_adam_writes'], 20)
            self.assertEqual(proof['counted_new_complete_outers'], 0)
            self.assertEqual(proof['segment']['observed_completed_through'], last['step'])
            self.assertTrue(all(row['terminal_kind'] == 'OUTER_BUDGET' for row in proof['per_rank'].values()))
            self.assertTrue(all(row['scheduler_tail_kinds'][-1] == 'COMPLETE' for row in proof['per_rank'].values()))

    def test_unscored_return_and_separate_unreturned_rank_tail_are_not_combined(self):
        with tempfile.TemporaryDirectory() as root:
            base, run, last = open_tail(root, (10, 10, 10, 10))
            path = base/'logs/actual-updates.jsonl'
            values = [json.loads(line) for line in path.read_text().splitlines()]
            rows(path, values[:-1])
            proof = read_unreturned_tail(base, run, retained_through=last['step'] - 1)
            self.assertEqual(proof['segment']['discarded_successful_adam'], 20)
            self.assertEqual(proof['segment']['discarded_unscored_attempts'][0]['logical_update'], last['step'])
            self.assertEqual(proof['all_rank_recorded_adam_writes'], 3)
            self.assertEqual(proof['after_driver_writes'], last['optimizer_writes_after'])
            self.assertEqual(proof['counted_new_complete_outers'], 0)

    def test_unbound_actual_fatal_snapshot_is_preserved_and_requires_fault_review(self):
        with tempfile.TemporaryDirectory() as root:
            base, run, last = open_tail(root)
            path = base/'logs/scheduler-rank-2.jsonl'
            values = [json.loads(line) for line in path.read_text().splitlines()]
            failure = dict(values[-1], kind='POST_COMMIT_FAILURE', reason='actual receipt write failed',
                retained_pending=None, retained_prepared=None, fatal_no_automatic_resume=True)
            failure.pop('attempt_index'); failure.pop('attempt_id')
            values.append(failure); rows(path, values)
            proof = read_unreturned_tail(base, run, retained_through=last['step'])
            self.assertTrue(proof['requires_fault_review'])
            self.assertEqual(proof['per_rank']['2']['scheduler_unbound_failure_count'], 1)
            self.assertEqual(proof['counted_new_complete_outers'], 0)

    def test_bound_fatal_snapshots_and_rejection_always_require_fault_review(self):
        for kind in ('POST_COMMIT_FAILURE', 'PRE_COMMIT_EVIDENCE_FAILURE', 'REJECTED'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as root:
                base, run, last = open_tail(root)
                path = base/'logs/scheduler-rank-2.jsonl'
                values = [json.loads(line) for line in path.read_text().splitlines()]
                failure = dict(values[-1], kind=kind, reason='actual receipt fault')
                if kind != 'REJECTED':
                    failure.update(retained_pending=copy.deepcopy(values[-1]), retained_prepared=None,
                        fatal_no_automatic_resume=True)
                    failure.pop('attempt_index'); failure.pop('attempt_id')
                values.append(failure); rows(path, values)
                proof = read_unreturned_tail(base, run, retained_through=last['step'])
                self.assertTrue(proof['requires_fault_review'])
                self.assertEqual(proof['per_rank']['2']['scheduler_fatal_failure_count'], 1)
                self.assertEqual(proof['per_rank']['2']['scheduler_unbound_failure_count'], 0)
                self.assertEqual(proof['counted_new_complete_outers'], 0)

    def test_wrong_attempt_gate_write_position_rank_birth_and_interior_fragment_rejected(self):
        for kind in ('cursor', 'gate', 'write', 'birth', 'interior_fragment'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as root:
                base, run, last = open_tail(root)
                path = base/'logs'/('scheduler-rank-2.jsonl' if kind == 'birth' else 'optimizer-rank-0.jsonl')
                values = [json.loads(line) for line in path.read_text().splitlines()]
                if kind == 'cursor': values[-1]['context']['attempt_index'] += 1
                elif kind == 'gate': values[-1]['context']['behavior_probability_gate_sha256'] = 'e'*64
                elif kind == 'write': values[-1]['write_count'] += 1
                elif kind == 'birth': values[-1]['process_start_ticks'] += 1
                else:
                    path.write_bytes(path.read_bytes() + b'{"incomplete\n')
                if kind != 'interior_fragment': rows(path, values)
                with self.assertRaises((AssertionError, ValueError)):
                    read_unreturned_tail(base, run, retained_through=last['step'])


if __name__ == '__main__': unittest.main()
