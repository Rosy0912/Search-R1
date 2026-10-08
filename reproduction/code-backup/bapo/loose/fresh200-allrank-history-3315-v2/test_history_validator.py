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
    path=v.RUN+'/checkpoints/diagnostics/attempt_1_candidate_outer_1/all-rank-write-ledger.json'
    outcome=dict(source_commit=v.SOURCE,driver_pid=4321,attempt_id=1,candidate_outer=1,accepted_complete_outer=True,
                 ledger_path=path,rank_write_ledger=events,progress=progress.state_dict())
    records=[(v.MARKERS[0],r) for r in ranks]+[(v.MARKERS[1],outcome),(v.MARKERS[2],driver)]
    return records,{path:outcome}


def boundary_trajectory():
    original,_=fixture();templates=[x for _,x in original[:8]]
    progress=Fresh200Progress();records=[];ledgers={}
    for outer in range(1,59):
        sequence=[False,True] if outer==50 else [True]
        for accepted in sequence:
            attempt=progress.begin_attempt(outer);expected=20 if outer<50 else 10;writes=expected if accepted else 3
            events=[dict(rank=r,actual_outer=outer,attempt_id=attempt,status='COMPLETE' if accepted else 'ABORTED',optimizer_steps=writes,
                expected_optimizer_steps=expected,optimizer_opportunities_total=expected,processed_opportunities=writes,
                skipped_opportunities=0,unprocessed_opportunities=expected-writes,postabort=not accepted,
                update_complete=accepted,attempt_complete=accepted)for r in range(8)]
            before=progress.completed_outers;progress.record(events,outer);raw=[]
            for template in templates:
                x=copy.deepcopy(template);x.update(optimizer_steps=writes,lr_before=v.lr(before),lr_after=v.lr(progress.completed_outers),scheduler_last_epoch=progress.completed_outers)
                if not accepted:
                    x['metrics']={k:x['metrics'][k]for k in ('actor/optimizer_steps','mfu/actor','actor/lr','actor/lr_before_outer','actor/scheduler_outer_steps')}
                else:
                    for key,val in x['metrics'].items():
                        if isinstance(val,list):x['metrics'][key]=val[:expected]
                x['metrics'].update({'actor/optimizer_steps':writes,'actor/lr_before_outer':x['lr_before'],'actor/lr':x['lr_after'],'actor/scheduler_outer_steps':progress.completed_outers})
                raw.append(x)
            from search_r1.baselines.actor_metrics import merge_actor_metric_records
            merged=merge_actor_metric_records([x['metrics']for x in raw])
            metrics={k:sum(val)/len(val) if isinstance(val,list) else val for k,val in merged.items()}
            path=v.RUN+'/checkpoints/diagnostics/attempt_%d_candidate_outer_%d/all-rank-write-ledger.json'%(attempt,outer)
            out=dict(source_commit=v.SOURCE,driver_pid=4321,attempt_id=attempt,candidate_outer=outer,accepted_complete_outer=accepted,
                     ledger_path=path,rank_write_ledger=events,progress=progress.state_dict())
            ledgers[path]=out
            records.extend([(v.MARKERS[0],x)for x in raw]+[(v.MARKERS[1],out),
                (v.MARKERS[2],dict(step=outer,attempt_id=attempt,accepted_complete_outer=accepted,metrics=metrics))])
    return records,ledgers


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
            path=v.RUN+'/checkpoints/diagnostics/attempt_%d_candidate_outer_1/all-rank-write-ledger.json'%attempt
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

    def test_original_short_path_rejected(self):
        r,l=fixture();out=r[8][1];old=out['ledger_path'];out['ledger_path']=old.replace('/checkpoints/diagnostics/','/diagnostics/')
        with self.assertRaises(ValueError):v.replay(lines(r),{out['ledger_path']:out})
    def test_canonical_wrong_attempt_directory_rejected(self):
        r,l=fixture();out=r[8][1];out['ledger_path']=out['ledger_path'].replace('attempt_1_','attempt_2_')
        with self.assertRaises(ValueError):v.replay(lines(r),{out['ledger_path']:out})
    def test_accepted_short_loss_mapping_gap(self):
        r,l=fixture();out=r[8][1]
        for e in out['rank_write_ledger']:
            e.update(status='COMPLETE_WITH_SKIPS',optimizer_steps=19,skipped_opportunities=1)
        p=Fresh200Progress();p.begin_attempt(1);p.record(out['rank_write_ledger'],1);out['progress']=p.state_dict()
        for _,x in r[:8]:
            x['optimizer_steps']=19;x['metrics']['actor/optimizer_steps']=19;x['metrics']['actor/skipped_empty_minibatches']=1
            for key in ('actor/pg_loss','actor/entropy_loss','actor/pg_clipfrac','actor/ppo_kl','actor/grad_norm'):
                x['metrics'][key]=x['metrics'][key][:19]
        from search_r1.baselines.actor_metrics import merge_actor_metric_records
        merged=merge_actor_metric_records([x['metrics']for _,x in r[:8]])
        r[-1][1]['metrics'].update({k:sum(val)/len(val) if isinstance(val,list) else val for k,val in merged.items()})
        result=v.replay(lines(r),l)
        gaps=result['attempts'][0]['numerical_coverage_gaps']
        self.assertIn('GAPS',result['status'])
        self.assertTrue(any(g['reason']=='fewer_observed_values_than_processed_opportunities' and g['observed_count']==19 for g in gaps))
        self.assertTrue(any(g['reason']=='runtime_vector_has_no_explicit_opportunity_index_mapping' for g in gaps))
    def test_lr_and_geometry_boundaries_real_shared_contract(self):
        from search_r1.baselines.fresh200_contract import outer_contract
        self.assertEqual(outer_contract(49)['expected_optimizer_steps'],20)
        self.assertEqual(outer_contract(50)['expected_optimizer_steps'],10)
        self.assertAlmostEqual(v.lr(48),49e-6/57)
        self.assertAlmostEqual(v.lr(49),50e-6/57)
        self.assertAlmostEqual(v.lr(55),56e-6/57)
        self.assertAlmostEqual(v.lr(56),1e-6)
        self.assertEqual(v.lr(57),v.lr(200))

    def test_full_49_50_57_58_replay_with_retained_abort(self):
        r,l=boundary_trajectory();out=v.replay(lines(r),l)
        self.assertEqual(out['progress']['completed_outers'],58)
        self.assertEqual(out['progress']['attempts'],59)
        self.assertEqual(out['progress']['actual_adam_writes_per_rank'],[49*20+9*10+3]*8)
        aborted=out['attempts'][49];self.assertFalse(aborted['accepted']);self.assertEqual(aborted['candidate_outer'],50)
        self.assertAlmostEqual(out['attempts'][50]['rank_metrics']['actor/lr_before_outer'],50e-6/57)
        self.assertEqual(out['attempts'][57]['rank_metrics']['actor/lr_before_outer'],1e-6)
    def test_outer50_mislabelled_geometry_fails_full_replay(self):
        r,l=boundary_trajectory()
        target=next(x for m,x in r if m==v.MARKERS[1] and x['candidate_outer']==50)
        for e in target['rank_write_ledger']:e['expected_optimizer_steps']=20
        with self.assertRaises(ValueError):v.replay(lines(r),l)
    def test_outer57_incorrect_warmup_fails_full_replay(self):
        r,l=boundary_trajectory()
        attempt=next(x['attempt_id']for m,x in r if m==v.MARKERS[1] and x['candidate_outer']==57)
        r[(attempt-1)*10][1]['lr_before']=56e-6/57
        with self.assertRaises(ValueError):v.replay(lines(r),l)

    def test_actual_snapshot_is_not_full_history(self):
        a=json.loads(ACTUAL.read_text())
        self.assertEqual(a['source'],v.SOURCE);self.assertEqual(a['job_id'],3315)
        self.assertEqual(a['counts'][v.MARKERS[0]],8)
        with self.assertRaises(ValueError):v.replay(lines([(v.MARKERS[0],r)for r in a['latest'][v.MARKERS[0]]]),{})

if __name__=='__main__':unittest.main()
