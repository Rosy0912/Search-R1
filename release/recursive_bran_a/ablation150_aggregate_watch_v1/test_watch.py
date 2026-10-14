"""CPU fixtures for closed evidence, reuse, stale waits, and terminal acceptance."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from release.recursive_bran_a.ablation150_aggregate_watch_v1 import watch
from release.recursive_bran_a.ablation150_v1 import PROTOCOL
from release.recursive_bran_a.ablation150_v1.contracts import planned_conditions, sha


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False)+'\n')


def dev(hits=92):
    return {'val/qa_count/all': 512., 'val/qa_count/nq': 256., 'val/qa_count/hotpotqa': 256.,
            'val/em/all': hits/512, 'val/em/nq': 52/256, 'val/em/hotpotqa': (hits-52)/256}


class Fixture:
    def __init__(self, base, count=5, row=None):
        self.base = Path(base)
        self.plan = planned_conditions()
        self.row = row or self.plan[0]
        self.group = self.base/self.row['group']
        self.output = self.group/'output/attempt-current'
        self.config = self.group/'receipts/resolved.yaml'
        self.config.parent.mkdir(parents=True)
        self.config.write_text('frozen: fixture\n')
        self.receipt = dict(protocol=PROTOCOL, group=self.row['group'], run_id='run-fixture',
            config=str(self.config), config_sha256=sha(self.config), source_commit='s'*40,
            data_identity='fixture-data', condition=self.row['options'], actual_update_limit=150,
            evaluation_steps=list(watch.MILESTONES))
        write(self.group/'receipts/configuration.json', self.receipt)
        self.entry = dict(time=100., pid=50, uid=0, tick=1000, output=str(self.output),
            **{key:self.receipt[key] for key in ('run_id','config_sha256','source_commit','group')},
            zero_write_manifest='preserved-zero-write-state')
        write(self.group/'receipts/current-entry.json', self.entry)
        self.study = dict(status='ACTIVE', source_commit=self.receipt['source_commit'],
            conditions=self.plan, requested_labels=11, complete={}, current=self.row['group'],
            actual_parent_wait_rc=1)
        write(self.base/'study-progress.json', self.study)
        self.closed, self.log_lines = [], []
        for outer in range(1, count+1):
            metric = {'timing_s/step':10., 'training/actual_adam_calls_per_rank':float(outer*20),
                'actor/scheduler_step_before_outer':float(outer-1), 'actor/scheduler_step_after_outer':float(outer),
                'actor/lr':1e-6}
            log_metric = dict(metric, **{'rollout/raw_emitted_tokens':100.})
            if outer in watch.MILESTONES:
                metric.update(dev())
                log_metric.update(dev())
            self.log_lines.append('step:'+str(outer)+' - '+' - '.join(f'{key}:{value}' for key,value in log_metric.items())+'\n')
            checkpoint = self.group/'checkpoints/training_state'/f'global_step_{outer}'/'post-evaluation/manifest.json'
            native = dict(completed_step=outer, config_sha256=self.receipt['config_sha256'],
                code_commit=self.receipt['source_commit'], data_identity=self.receipt['data_identity'],
                actual_update_limit=150)
            workers = [dict(identity=dict(native, rank=rank, world_size=8, cuda_visible_devices='0,1,2,3,4,5,6,7'),
                full_worker_state_finite=True, scheduler_last_epoch=outer, actual_total_optimizer_calls=outer*20)
                for rank in range(8)]
            write(checkpoint, dict(status='COMPLETE_SAME_WORLD_STATE_SAVED', boundary_phase='post-evaluation',
                identity=native, driver=dict(next_step=outer+1), worker_states=workers))
            value = dict(time=100.+outer, **{key:self.receipt[key] for key in ('run_id','group','config_sha256','source_commit')},
                completed_outer=outer, next_outer=outer+1, metrics=metric,
                optimizer_calls_per_rank={str(rank):outer*20 for rank in range(8)},
                checkpoint=dict(path=str(checkpoint), sha256=sha(checkpoint), native_only=False))
            self.closed.append(value)
            ranks = [dict(rank=rank, optimizer_calls=20, opportunities_skipped=0,
                opportunity_ledger=[dict(index=index, actual_writes=1) for index in range(1,21)]) for rank in range(8)]
            write(self.output/'attempt-results'/f'attempt-{outer:06d}.json', dict(accepted_outer=outer,
                actual_completed_optimizer_calls_per_rank={str(rank):outer*20 for rank in range(8)},
                all_rank_result=dict(identity={key:self.receipt[key] for key in ('run_id','config_sha256')},
                    decision='COMPLETED_OUTER', ranks=ranks)))
        self.write_closed()
        self.log = self.group/'logs/training.log'
        self.log.parent.mkdir(parents=True)
        self.log.write_text(''.join(self.log_lines))

    def write_closed(self):
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output/'completed-outers.jsonl').write_text(''.join(json.dumps(value)+'\n' for value in self.closed))

    def accepted150(self):
        last = self.closed[-1]
        complete = dict(status=watch.COMPLETE_TRAINING, time=300., completed_outer=150,
            evaluation_steps=list(watch.MILESTONES), checkpoint=last['checkpoint'],
            final_dev_metrics=dev(), actual_optimizer_calls_per_rank=last['optimizer_calls_per_rank'],
            **{key:self.receipt[key] for key in ('run_id','group','config_sha256','source_commit')})
        write(self.output/'training-complete.json', complete)
        write(self.output/'entry-exit.json', dict(time=301., rc=0,
            **{key:self.entry[key] for key in ('pid','uid','tick','run_id','group')}))
        write(self.group/'receipts/actual-parent-wait.json', dict(time=302., rc=0,
            pid=self.entry['pid'], group=self.entry['group']))
        self.study['complete'][self.row['group']] = dict(training_complete=str(self.output/'training-complete.json'),
            complete_sha256=sha(self.output/'training-complete.json'),
            actual_parent_wait=str(self.group/'receipts/actual-parent-wait.json'), actual_parent_wait_rc=0,
            owned_gpu_release=dict(time=303., complete=True, applications=[], query_rc=0))


class WatchTest(unittest.TestCase):
    def test_closed_five_stale_wait_and_reuse_are_not_goal_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory)/'study')
            # The old return is retained even while a different exact entry is active.
            write(fixture.group/'receipts/actual-parent-wait.json', dict(time=99., rc=1, pid=49,
                group=fixture.entry['group']))
            output = Path(directory)/'reports'
            active = lambda owner: owner
            result = watch.poll(fixture.base, output, read_process=active)
            self.assertEqual(result['status'], 'ACTIVE')
            self.assertFalse(result['goal_complete'])
            self.assertEqual(result['required_report_count'], 44)
            self.assertEqual(result['collected_report_count'], 14)
            state = result['groups']['C1-root-only']['terminal']
            self.assertTrue(state['ignored_previous_parent_wait'])
            self.assertIsNone(state['actual_parent_wait_rc'])
            report_path = Path(result['reports']['generation-1']['5']['directory'])/'aggregate.json'
            report = json.loads(report_path.read_text())
            self.assertEqual(report['reuse_of'], 'C1-root-only')
            self.assertFalse(report['generated_new_training'])
            self.assertEqual(report['conditions']['generation-1']['milestones']['5']['dev512']['all']['hits'], 92)
            self.assertTrue(report['conditions']['generation-1']['milestones']['5']['cost_coverage']['original_sampling_cost_unknown'])
            self.assertNotIn('analysis_complete', report)
            again = watch.poll(fixture.base, output, read_process=active)
            self.assertEqual(again['new_reports'], [])
            self.assertEqual(again['reports'], result['reports'])
            fixture.log.write_text(''.join(fixture.log_lines)+'active outer6 is still generating\n')
            with patch.object(watch, 'native_lines', side_effect=AssertionError('Idle evidence was rescanned')):
                again = watch.poll(fixture.base, output, read_process=active)
            self.assertEqual(again['status'], 'ACTIVE')

    def test_unclosed_dev_and_unterminated_log_cannot_be_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory)/'study')
            fixture.closed.pop()
            fixture.write_closed()
            # Even an emitted full native dev line and all Adam receipts cannot close outer5.
            result = watch.poll(fixture.base, Path(directory)/'reports')
            self.assertEqual(result['reports']['C1-root-only'], {})
            fixture.closed.append(dict(time=105., completed_outer=5, next_outer=6,
                **{key:fixture.receipt[key] for key in ('run_id','group','config_sha256','source_commit')},
                metrics=dev(), optimizer_calls_per_rank={str(rank):100 for rank in range(8)},
                checkpoint=dict(path=str(fixture.group/'checkpoints/training_state/global_step_5/post-evaluation/manifest.json'),
                    sha256=sha(fixture.group/'checkpoints/training_state/global_step_5/post-evaluation/manifest.json'))))
            fixture.write_closed()
            fixture.log.write_text(''.join(fixture.log_lines).rstrip('\n'))
            result = watch.poll(fixture.base, Path(directory)/'reports')
            self.assertEqual(result['status'], 'ATTENTION_REQUIRED')
            self.assertEqual(result['reports']['C1-root-only'], {})
            # An actual live owner can still be between closure flush and native scalar flush.
            result = watch.poll(fixture.base, Path(directory)/'reports', read_process=lambda owner:owner)
            self.assertEqual(result['status'], 'ACTIVE')
            self.assertEqual(result['groups']['C1-root-only']['pending_milestones'], [5])
            fixture.log.write_text(''.join(fixture.log_lines))
            result = watch.poll(fixture.base, Path(directory)/'reports', read_process=lambda owner:owner)
            self.assertIn('5', result['reports']['C1-root-only'])

    def test_native_manifest_hash_and_finite_evidence_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory)/'study')
            path = Path(fixture.closed[-1]['checkpoint']['path'])
            value = json.loads(path.read_text())
            value['worker_states'][0]['scheduler_last_epoch'] = 4
            write(path, value)
            fixture.closed[-1]['checkpoint']['sha256'] = sha(path)
            fixture.write_closed()
            result = watch.poll(fixture.base, Path(directory)/'reports')
            self.assertEqual(result['status'], 'ATTENTION_REQUIRED')
            self.assertTrue(any('scheduler/Adam' in row['error'] for row in result['errors']))
            with self.assertRaises(ValueError):
                watch.finite({'timing_s/step': float('nan')})

    def test_complete150_requires_current_wait_exit_release_and_full_study(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory)/'study', count=150)
            fixture.accepted150()
            closed = {value['completed_outer']:value for value in fixture.closed}
            result = watch.group_terminal(fixture.group, fixture.receipt, fixture.entry, fixture.study,
                closed, read_process=lambda owner:None)
            self.assertTrue(result['accepted'])
            wait = fixture.group/'receipts/actual-parent-wait.json'
            write(wait, dict(time=302., rc=0, pid=49, group=fixture.entry['group']))
            with self.assertRaisesRegex(ValueError, 'current parent wait'):
                watch.group_terminal(fixture.group, fixture.receipt, fixture.entry, fixture.study, closed)
            write(wait, dict(time=302., rc=0, pid=50, group=fixture.entry['group']))
            fixture.study['complete']['C1-root-only']['owned_gpu_release']['complete'] = False
            with self.assertRaisesRegex(ValueError, 'release receipt'):
                watch.group_terminal(fixture.group, fixture.receipt, fixture.entry, fixture.study, closed)
            # A terminal-looking study cannot complete a subset publication.
            fixture.study['status'] = watch.TERMINAL_STUDY
            fixture.study['current'] = None
            fixture.study['complete'] = {}
            write(fixture.base/'study-progress.json', fixture.study)
            write(fixture.base/'study-actual-exit.json', dict(time=400., rc=0))
            result = watch.poll(fixture.base, Path(directory)/'reports')
            self.assertEqual(result['collected_report_count'], 20)
            self.assertFalse(result['goal_complete'])
            self.assertFalse(result['study_terminal_accepted'])

    def test_current_wait_failure_is_attention_even_when_study_says_active(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory)/'study')
            write(fixture.group/'receipts/actual-parent-wait.json', dict(time=110., rc=1,
                pid=fixture.entry['pid'], group=fixture.entry['group']))
            result = watch.poll(fixture.base, Path(directory)/'reports')
            self.assertEqual(result['status'], 'ATTENTION_REQUIRED')
            self.assertFalse(result['goal_complete'])
            self.assertTrue(any('actual parent wait 1' in row['error'] for row in result['errors']))

    def test_full_eleven_labels_and_actual_study_exit_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)/'study'
            accepted = {}
            for row in planned_conditions():
                if row.get('reuse'):
                    continue
                fixture = Fixture(base, count=150, row=row)
                fixture.accepted150()
                accepted.update(fixture.study['complete'])
            original_path = base/'retained-A-post-evaluation-manifest.json'
            write(original_path, dict(status='COMPLETE_SAME_WORLD_STATE_SAVED', boundary_phase='post-evaluation',
                identity=dict(completed_step=150, code_commit=watch.A_COMMIT, config_sha256=watch.A_CONFIG)))
            original = dict(path=str(original_path), sha256=sha(original_path), completed_outer=150)
            for row in planned_conditions():
                if row.get('reuse'):
                    accepted[row['group']] = dict(reuse_of=row['reuse'], condition=row['options'],
                        generated_new_training=False,
                        accepted=accepted['C1-root-only'] if row['reuse'] == 'C1-root-only' else original)
            fixture.study.update(status=watch.TERMINAL_STUDY, current=None, complete=accepted)
            write(base/'study-progress.json', fixture.study)
            output = Path(directory)/'reports'
            result = watch.poll(base, output)
            self.assertEqual(result['collected_report_count'], 44)
            self.assertFalse(result['goal_complete'])
            write(base/'study-actual-exit.json', dict(time=400., rc=0))
            result = watch.poll(base, output)
            self.assertEqual(result['status'], 'GOAL_COMPLETE')
            self.assertTrue(result['goal_complete'])
            self.assertTrue(result['study_terminal_accepted'])


if __name__ == '__main__':
    unittest.main()
