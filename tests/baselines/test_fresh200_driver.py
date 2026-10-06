import copy
import unittest
from search_r1.baselines.fresh200_driver import Fresh200Progress
from search_r1.baselines.fresh200_gate import validate_first_five_attempts


def events(state, outer, status='COMPLETE', writes=None, processed=None, postabort=False):
    expected=20 if outer<50 else 10
    if writes is None:writes=expected
    if processed is None:processed=expected
    return [dict(rank=r,actual_outer=outer,attempt_id=state.attempts,status=status,
                 optimizer_steps=writes,expected_optimizer_steps=expected,optimizer_opportunities_total=expected,
                 processed_opportunities=processed,skipped_opportunities=processed-writes,
                 unprocessed_opportunities=expected-processed,postabort=postabort,
                 attempt_complete=processed==expected and not postabort,
                 update_complete=status in ('COMPLETE','COMPLETE_WITH_SKIPS')) for r in range(8)]


class DriverAccounting(unittest.TestCase):
    def test_whole_skip_and_postabort_do_not_advance_complete_outer(self):
        state=Fresh200Progress()
        state.begin_attempt(1)
        self.assertFalse(state.record(events(state,1,'SKIPPED',0,20),1))
        state.begin_attempt(1)
        self.assertFalse(state.record(events(state,1,'ABORTED',3,3,True),1))
        self.assertEqual(state.completed_outers,0)
        self.assertEqual(state.actual_adam_writes_per_rank,[3]*8)
        state.begin_attempt(1)
        self.assertTrue(state.record(events(state,1,'COMPLETE_WITH_SKIPS',19),1))
        self.assertEqual((state.completed_outers,state.attempts),(1,3))
        self.assertEqual(state.actual_adam_writes_per_rank,[22]*8)

    def test_boundaries_and_actual200_are_not200_attempts(self):
        state=Fresh200Progress()
        for outer in range(1,201):
            if outer in (49,50,200):
                state.begin_attempt(outer)
                self.assertFalse(state.record(events(state,outer,'ABORTED',1,1,True),outer))
            state.begin_attempt(outer)
            self.assertTrue(state.record(events(state,outer),outer))
        self.assertEqual(state.completed_outers,200)
        self.assertEqual(state.attempts,203)
        self.assertEqual(state.actual_adam_writes_per_rank,[2493]*8)
        self.assertEqual(state.state_dict()['no_skip_expected_writes_per_rank'],2490)
        with self.assertRaises(ValueError):state.begin_attempt(201)

    def test_underfilled_complete_zero_writes_duplicate_and_rank_drift_reject(self):
        state=Fresh200Progress();state.begin_attempt(1)
        for bad in (events(state,1,'COMPLETE_WITH_SKIPS',0),events(state,1,'COMPLETE',1,1)):
            with self.assertRaises(ValueError):state.record(bad,1)
        bad=events(state,1);bad[-1]['optimizer_steps']=19;bad[-1]['skipped_opportunities']=1
        with self.assertRaises(ValueError):state.record(bad,1)
        self.assertEqual(state.actual_adam_writes_per_rank,[0]*8)
        state.record(events(state,1),1)
        with self.assertRaises(ValueError):state.record(events(state,1),1)

    def test_firstfive_replay_allows_skips_and_retained_abort_writes(self):
        state=Fresh200Progress();attempts=[]
        layouts=[(1,'ABORTED',3,3,True),(1,'COMPLETE_WITH_SKIPS',19,20,False),
                 (2,'SKIPPED',0,20,False),(2,'COMPLETE',20,20,False),
                 (3,'COMPLETE',20,20,False),(4,'COMPLETE',20,20,False),(5,'COMPLETE_WITH_SKIPS',18,20,False)]
        for outer,status,writes,processed,abort in layouts:
            attempt=state.begin_attempt(outer);ranks=events(state,outer,status,writes,processed,abort)
            complete=state.record(ranks,outer)
            attempts.append(dict(attempt_id=attempt,candidate_outer=outer,driver_pid=123,
                status=status,accepted_complete_outer=complete,rank_write_ledger=ranks,progress=state.state_dict()))
        self.assertEqual(len(validate_first_five_attempts(attempts,state.state_dict(),123)),5)
        self.assertEqual(state.actual_adam_writes_per_rank,[100]*8)
        for bad in (attempts[:-1],attempts[1:]):
            with self.assertRaises(ValueError):validate_first_five_attempts(bad,state.state_dict(),123)
        bad=copy.deepcopy(attempts);bad[3]['driver_pid']=456
        with self.assertRaises(ValueError):validate_first_five_attempts(bad,state.state_dict(),123)
        bad=copy.deepcopy(attempts);bad[0]['progress']['actual_adam_writes_per_rank'][0]=0
        with self.assertRaises(ValueError):validate_first_five_attempts(bad,state.state_dict(),123)
        for key,value in [('actual_outer',True),('attempt_id',True),('expected_optimizer_steps',20.),('optimizer_opportunities_total',20.)]:
            bad=copy.deepcopy(attempts)
            for event in bad[0]['rank_write_ledger']:event[key]=value
            with self.assertRaises(ValueError):validate_first_five_attempts(bad,state.state_dict(),123)
        bad=copy.deepcopy(attempts);bad[0]['candidate_outer']=True
        with self.assertRaises(ValueError):validate_first_five_attempts(bad,state.state_dict(),123)
        bad=copy.deepcopy(attempts);bad[0]['progress']['attempts']=1.
        with self.assertRaises(ValueError):validate_first_five_attempts(bad,state.state_dict(),123)
        with self.assertRaises(ValueError):Fresh200Progress().begin_attempt(True)


if __name__=='__main__':unittest.main()
