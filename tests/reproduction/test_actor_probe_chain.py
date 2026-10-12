"""Synthetic scalar ledgers only; never a production completion claim."""
import copy
import unittest

from reproduction.common8.actor_probe_chain import account_segment, assemble_counter_chain


def segment(run, start, observed, retained, writes=0, attempt_index=0, zero_guard_at=None):
    updates = []; attempts = []
    for step in range(start+1, observed+1):
        counts = ([0] if step == zero_guard_at else []) + [13 if step == 103 else 20]
        for count in counts:
            attempt_index += 1; before = writes; writes += count
            row = dict(run_id=run, attempt_id=f'{run}-{attempt_index}', attempt_index=attempt_index,
                logical_update=step, optimizer_writes_before=before, optimizer_writes_after=writes,
                optimizer_writes_this_attempt=count, opportunities_processed=20,
                opportunity_statuses=['written']*count+['skipped_guard']*(20-count),
                all_ranks_agreed=True, real_rank_count=4, complete_outer=count>0,
                scheduler_advances=int(count>0), completed_outer=step if count else step-1,
                status='complete' if count else 'recoverable_guard')
            attempts.append(row)
            if count:
                updates.append(dict(run_id=run,step=step,attempt_id=row['attempt_id'],
                    attempt_index=attempt_index,optimizer_writes_before=before,optimizer_writes_after=writes,
                    opportunities_processed=20,actual_optimizer_writes=count,
                    reward_schedule='g1_through100_then_binary',reward_logical_update=step,
                    reward_mode='search_g1' if step<=100 else 'binary_em',
                    metrics={'critic/rewards/mean': .3 if step<=100 else .5}))
    return account_segment(run, updates, attempts, retained_through=retained,
        start_outer=start, start_writes=updates[0]['optimizer_writes_before'],
        start_attempt_index=attempts[0]['attempt_index']-1)


class ActorProbeChainTests(unittest.TestCase):
    def chain(self):
        root = segment('root',0,53,50,zero_guard_at=3)
        middle = segment('middle',50,102,100,root['checkpoint_writes'],root['checkpoint_attempt_index'])
        final = segment('last',100,200,200,middle['checkpoint_writes'],middle['checkpoint_attempt_index'])
        return [root,middle,final]

    def test_retained200_excludes_discarded_updates_and_keeps_separate_reward_phases(self):
        segments = self.chain(); before = copy.deepcopy(segments)
        proof = assemble_counter_chain(segments, final_outer=200)
        self.assertEqual(segments,before)
        self.assertEqual(proof['complete_outers'],200)
        self.assertEqual(proof['successful_adam'],3993)  # Seven legitimate skips at103, replicas once.
        self.assertEqual(proof['discarded_by_namespace'][0]['complete_outers'],[51,52,53])
        self.assertEqual(proof['discarded_by_namespace'][0]['successful_adam'],60)
        self.assertEqual(proof['discarded_by_namespace'][1]['complete_outers'],[101,102])
        self.assertEqual(proof['logical_owners'][50],'root')
        self.assertEqual(proof['logical_owners'][51],'middle')
        self.assertEqual(proof['logical_owners'][101],'last')
        self.assertEqual(proof['reward_phases']['g1']['count'],100)
        self.assertEqual(proof['reward_phases']['binary_em']['count'],100)
        self.assertAlmostEqual(proof['reward_phases']['g1']['mean'],.3)
        self.assertEqual(proof['reward_phases']['binary_em']['mean'],.5)
        self.assertFalse(proof['full_training_or_final_acceptance'])

    def test_recovery_cannot_inherit_updates_after_the_saved_checkpoint(self):
        segments = self.chain()
        wrong = copy.deepcopy(segments); wrong[1]['start_writes']=wrong[0]['observed_writes']
        with self.assertRaises(AssertionError):
            assemble_counter_chain(wrong,final_outer=200)

    def test_holes_duplicate_namespace_and_wrong_attempt_cursor_are_rejected(self):
        for kind in ('hole','duplicate_namespace','attempt_cursor'):
            wrong=self.chain()
            if kind=='hole': wrong[1]['retained_updates'].pop(2)
            if kind=='duplicate_namespace': wrong[1]['run_id']=wrong[0]['run_id']
            if kind=='attempt_cursor': wrong[1]['start_attempt_index']+=1
            with self.assertRaises(AssertionError): assemble_counter_chain(wrong,final_outer=200)

    def test_skip_only_attempt_does_not_advance_an_outer(self):
        proof=segment('unit',0,5,5,zero_guard_at=3)
        self.assertEqual(len(proof['retained_updates']),5)
        self.assertEqual(len(proof['retained_attempts']),6)
        self.assertEqual(proof['checkpoint_writes'],100)
        self.assertEqual(proof['checkpoint_attempt_index'],6)

    def test_mutated_summary_cannot_inflate_saved_writes(self):
        wrong=self.chain();wrong[0]['checkpoint_writes']+=40
        with self.assertRaisesRegex(AssertionError,'summary differs'):
            assemble_counter_chain(wrong,final_outer=200)

    def test_wrong_reward_phase_and_attempt201_are_rejected(self):
        normal=self.chain()[-1]
        updates=copy.deepcopy(normal['retained_updates']); attempts=copy.deepcopy(normal['retained_attempts'])
        updates[0]['reward_mode']='search_g1'
        with self.assertRaises(AssertionError):
            account_segment('last',updates,attempts,retained_through=200,start_outer=100,
                start_writes=normal['start_writes'],start_attempt_index=normal['start_attempt_index'])
        updates=copy.deepcopy(normal['retained_updates'])
        extra=dict(attempts[-1],logical_update=201,attempt_id='last-extra',
            attempt_index=attempts[-1]['attempt_index']+1,optimizer_writes_before=normal['checkpoint_writes'],
            optimizer_writes_after=normal['checkpoint_writes'],optimizer_writes_this_attempt=0,
            complete_outer=False,scheduler_advances=0,status='recoverable_guard',
            opportunity_statuses=['skipped_guard']*20)
        with self.assertRaises(AssertionError):
            account_segment('last',updates,attempts+[extra],retained_through=200,start_outer=100,
                start_writes=normal['start_writes'],start_attempt_index=normal['start_attempt_index'])

    def test_writer_metric_whitelist_shape_is_accepted(self):
        from reproduction.common8.update_evidence import METRICS
        normal=segment('unit',0,5,5)
        self.assertNotIn('train/reward_binary_phase', METRICS)
        self.assertNotIn('train/reward_logical_update', METRICS)
        for row in normal['retained_updates']:
            self.assertTrue(set(row['metrics']).issubset(METRICS))
            self.assertEqual(row['reward_logical_update'], row['step'])
        self.assertEqual(assemble_counter_chain([normal], final_outer=5)['complete_outers'],5)


if __name__=='__main__': unittest.main()
