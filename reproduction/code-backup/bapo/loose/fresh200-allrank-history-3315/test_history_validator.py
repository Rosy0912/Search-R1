import copy
import json
from pathlib import Path
import unittest
import history_validator as v
from search_r1.baselines.fresh200_driver import Fresh200Progress

BASE = Path(__file__).resolve().parents[1]
ACTUAL = BASE / 'evidence/fresh200-controller-initfix-formal3315-metrics4-rank.actual.json'


def fixture():
    # Real observed numerical vectors, paired with clearly synthetic ledger for parser counterexamples only.
    actual = json.loads(ACTUAL.read_text())
    ranks = copy.deepcopy(actual['latest']['ACTOR_RANK_UPDATE_JSON='])
    driver = copy.deepcopy(actual['latest']['BAPO_TRAIN_METRICS_JSON='][0])
    events = [dict(rank=r, actual_outer=1, attempt_id=1, status='COMPLETE', optimizer_steps=20,
        expected_optimizer_steps=20, optimizer_opportunities_total=20, processed_opportunities=20,
        skipped_opportunities=0, unprocessed_opportunities=0, postabort=False,
        update_complete=True, attempt_complete=True) for r in range(8)]
    progress=Fresh200Progress();progress.begin_attempt(1);progress.record(events,1)
    path=v.RUN+'/diagnostics/attempt_1_candidate_outer_1/all-rank-write-ledger.json'
    outcome=dict(source_commit=v.SOURCE,driver_pid=4321,attempt_id=1,candidate_outer=1,accepted_complete_outer=True,
                 ledger_path=path,rank_write_ledger=events,progress=progress.state_dict())
    records=[(v.MARKERS[0],r) for r in ranks]+[(v.MARKERS[1],outcome),(v.MARKERS[2],driver)]
    return records,{path:outcome}


def lines(records): return '\n'.join(m+json.dumps(r) for m,r in records)


class ReplayTests(unittest.TestCase):
    def test_observed_numeric_vectors_synthetic_ledger(self):
        records,ledgers=fixture(); out=v.replay(lines(records),ledgers)
        self.assertEqual(out['progress']['actual_adam_writes_per_rank'],[20]*8)
        self.assertFalse(out['native_acceptance'])
    def test_missing_rank(self):
        r,l=fixture();del r[2]
        with self.assertRaises(ValueError):v.replay(lines(r),l)
    def test_duplicate_rank(self):
        r,l=fixture();r[1]=copy.deepcopy(r[0])
        with self.assertRaises(ValueError):v.replay(lines(r),l)
    def test_reordered_across_attempt_boundary(self):
        r,l=fixture();r[7],r[8]=r[8],r[7]
        with self.assertRaises(ValueError):v.replay(lines(r),l)
    def test_driver_mean_mismatch(self):
        r,l=fixture();r[-1][1]['metrics']['actor/pg_loss']+=.01
        with self.assertRaises(ValueError):v.replay(lines(r),l)
    def test_nonfinite_rank(self):
        r,l=fixture();r[0][1]['metrics']['actor/grad_norm'][0]=float('nan')
        with self.assertRaises(ValueError):v.replay(lines(r),l)
    def test_wrong_scheduler(self):
        r,l=fixture();r[0][1]['scheduler_last_epoch']=2
        with self.assertRaises(ValueError):v.replay(lines(r),l)
    def test_durable_ledger_missing(self):
        r,l=fixture()
        with self.assertRaises(ValueError):v.replay(lines(r),{})
    def test_truncated_write_vector(self):
        r,l=fixture();r[0][1]['metrics']['actor/pg_loss'].pop()
        with self.assertRaises(ValueError):v.replay(lines(r),l)
    def test_bool_identity(self):
        r,l=fixture();r[-1][1]['attempt_id']=True
        with self.assertRaises(ValueError):v.replay(lines(r),l)
    def test_duplicate_json_keys(self):
        with self.assertRaises(ValueError):v.strict_json('{"rank":0,"rank":1}')
    def test_abort_retained_then_accept_with_skips(self):
        records, unused = fixture()
        templates=[x[1] for x in records[:8]]
        progress=Fresh200Progress(); all_records=[]; ledgers={}
        for attempt, status, writes, processed, skipped in [(1,'ABORTED',3,3,0),(2,'COMPLETE_WITH_SKIPS',19,20,1)]:
            progress.begin_attempt(1)
            accepted=status!='ABORTED'
            events=[dict(rank=r,actual_outer=1,attempt_id=attempt,status=status,optimizer_steps=writes,
                expected_optimizer_steps=20,optimizer_opportunities_total=20,processed_opportunities=processed,
                skipped_opportunities=skipped,unprocessed_opportunities=20-processed,postabort=not accepted,
                update_complete=accepted,attempt_complete=accepted) for r in range(8)]
            progress.record(events,1)
            raw=[]
            for template in templates:
                x=copy.deepcopy(template);x['optimizer_steps']=writes
                x['lr_after']=v.lr(int(accepted));x['scheduler_last_epoch']=int(accepted)
                if not accepted:
                    x['metrics']={key:x['metrics'][key] for key in ('actor/optimizer_steps','mfu/actor','actor/lr','actor/lr_before_outer','actor/scheduler_outer_steps')}
                else:
                    # Loss can include the opportunity rejected after backward; grad vector cannot.
                    x['metrics']['actor/grad_norm']=x['metrics']['actor/grad_norm'][:writes]
                    x['metrics']['actor/skipped_empty_minibatches']=1
                x['metrics'].update({'actor/optimizer_steps':writes,'actor/lr':x['lr_after'],'actor/scheduler_outer_steps':int(accepted)})
                raw.append(x)
            from search_r1.baselines.actor_metrics import merge_actor_metric_records
            merged=merge_actor_metric_records([x['metrics']for x in raw])
            metric={k:sum(val)/len(val) if isinstance(val,list) else val for k,val in merged.items()}
            path=v.RUN+'/diagnostics/attempt_%d_candidate_outer_1/all-rank-write-ledger.json'%attempt
            outcome=dict(source_commit=v.SOURCE,driver_pid=4321,attempt_id=attempt,candidate_outer=1,accepted_complete_outer=accepted,
                         ledger_path=path,rank_write_ledger=events,progress=progress.state_dict())
            ledgers[path]=outcome
            all_records.extend([(v.MARKERS[0],x)for x in raw]+[(v.MARKERS[1],outcome),
                (v.MARKERS[2],dict(attempt_id=attempt,step=1,accepted_complete_outer=accepted,metrics=metric))])
        result=v.replay(lines(all_records),ledgers)
        self.assertEqual(result['progress']['actual_adam_writes_per_rank'],[22]*8)
        self.assertEqual(result['progress']['completed_outers'],1)
        self.assertTrue(result['attempts'][0]['numerical_coverage_gaps'])
        self.assertIn('GAPS',result['status'])

    def test_actual_snapshot_is_not_full_history(self):
        a=json.loads(ACTUAL.read_text())
        self.assertEqual(a['source'],v.SOURCE);self.assertEqual(a['job_id'],3315)
        self.assertEqual(a['counts'][v.MARKERS[0]],8)
        with self.assertRaises(ValueError):v.replay(lines([(v.MARKERS[0],r)for r in a['latest'][v.MARKERS[0]]]),{})

if __name__=='__main__':unittest.main()
