"""Owned scalar prefix/fault files only; gate receipts are synthetic, no GPU."""
import copy
import json
import tempfile
import unittest
from unittest.mock import patch

from reproduction.common8.actor_probe_retained_prefix import load_prefix, rank_prefix, console_evidence
from reproduction.common8.actor_probe_evidence import verify_actor_readouts
from reproduction.common8.behavior_runtime import verify_write_receipt_links
from reproduction.common8.formal_attempt_audit import verify_attempt_history
from reproduction.common8.policy_calibration_plan import document_sha256
from reproduction.common8.runtime import budget_rows, verify_scheduler_receipts
from reproduction.common8.worker_evidence import read_console_form
from tests.reproduction.test_actor_probe_namespace_audit import fixture, rows
from tests.reproduction.test_actor_probe_rank_tail import open_tail


def fault_tail(base, run, rank=2):
    path=base/'logs'/f'scheduler-rank-{rank}.jsonl'
    values=[json.loads(line) for line in path.read_text().splitlines()]
    pending=copy.deepcopy(values[-1])
    failure=dict(pending,kind='POST_COMMIT_FAILURE',reason='synthetic durable receipt failure',
        retained_pending=pending,retained_prepared=None,fatal_no_automatic_resume=True)
    failure.pop('attempt_id');failure.pop('attempt_index')
    rows(path,values+[failure])
    with open(run['train_log'],'a') as stream:
        stream.write('SEARCH_G1_FATAL_SCHEDULER_EVIDENCE='+json.dumps(failure)+'\n')
    return failure


def actual_fixture_plan(base):
    return next(json.loads(line)['calibration_sha256']
        for line in (base/'logs/optimizer-rank-0.jsonl').read_text().splitlines()
        if json.loads(line)['kind']=='POLICY_MONITOR')


class RetainedPrefixTests(unittest.TestCase):
    def audit(self,base,run,last,plan,prefix):
        complete=prefix['selected_updates']
        gates={str(row['step']):{'proof_sha256':document_sha256(row['attempt_index'])} for row in complete}
        def gate(_base,_run,step,**kwargs):
            return dict(proof_sha256=document_sha256(kwargs['update_record']['attempt_index']
                if 'update_record' in kwargs else kwargs['attempt_index']))
        with patch('reproduction.common8.behavior_runtime.read_gate_receipt',side_effect=gate):
            schedulers=verify_scheduler_receipts(base,run,complete[0]['step'],last,gates,retained_prefix=prefix)
            audit=verify_attempt_history(base,run,complete,schedulers,last=last,plan_sha256=plan,retained_prefix=prefix)
            verify_write_receipt_links(base,gates,complete[0]['step'],last,retained_prefix=prefix)
        budget=budget_rows(read_console_form(base,run['run_id'],retained_prefix=prefix),
            complete[0]['step'],last,complete_records=complete,run=run)
        return schedulers,audit,budget

    def test_restored_prefix_survives_later_bound_fault_and_fragments_without_counting_tail(self):
        with tempfile.TemporaryDirectory() as root:
            base,run,last=open_tail(root);failure=fault_tail(base,run)
            # All fragments are after a genuinely fully returned/scored prefix.
            for name in ('actual-updates','actual-attempts','scheduler-rank-2','optimizer-rank-2'):
                path=base/'logs'/(name+'.jsonl');path.write_bytes(path.read_bytes()+b'{"partial')
            before={path:path.read_bytes() for path in (base/'logs').glob('*.jsonl')}
            prefix=load_prefix(base,run,last['step'])
            schedulers,audit,budget=self.audit(base,run,last['step'],actual_fixture_plan(base),prefix)
            self.assertEqual(audit['namespace_new_complete_outers'],2)
            self.assertEqual(audit['actual_optimizer_writes_per_rank'],last['optimizer_writes_after'])
            self.assertEqual(len(budget),8)
            self.assertEqual(schedulers['2']['retained_prefix_files']['scheduler']['trailing_fragment_bytes'],9)
            self.assertTrue(schedulers['2']['retained_prefix_files']['scheduler']['requires_fault_review'])
            self.assertEqual(schedulers['2']['fatal_console_check']['discarded_fatal_records'],[failure])
            self.assertTrue(schedulers['2']['fatal_console_check']['requires_fault_review'])
            self.assertEqual(before,{path:path.read_bytes() for path in before})
            # The original full-file reader still refuses a fatal completion.
            with self.assertRaises(AssertionError):
                verify_scheduler_receipts(base,run,51,52,{})

    def test_fresh_prefix_keeps_zero_origin_and_returned_lost_updates_explicit(self):
        with tempfile.TemporaryDirectory() as root:
            base,run,plan,updates=fixture(root,start=0,recovery=False)
            prefix=load_prefix(base,run,1)
            schedulers,audit,budget=self.audit(base,run,1,plan,prefix)
            self.assertEqual(audit['actual_optimizer_writes_per_rank'],20)
            self.assertEqual(audit['namespace_new_complete_outers'],1)
            self.assertEqual(len(budget),4)
            self.assertEqual(prefix['segment']['discarded_successful_adam'],20)
            self.assertIsNone(schedulers['0']['restore_anchor'])
            self.assertFalse(prefix['full_training_or_final_acceptance'])

    def test_fatal_inside_retained_prefix_or_unbound_fault_is_rejected(self):
        for mode in ('retained','unbound'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as root:
                base,run,last=open_tail(root);fault_tail(base,run)
                path=base/'logs/scheduler-rank-2.jsonl'
                values=[json.loads(line) for line in path.read_text().splitlines()]
                if mode=='retained':
                    values[-1]['retained_pending']['attempt_index']=last['attempt_index']
                    values[-1]['retained_pending']['attempt_id']=last['attempt_id']
                    values[-1]['retained_pending']['logical_update']=last['step']
                else:values[-1]['retained_pending']=None
                rows(path,values)
                prefix=load_prefix(base,run,last['step'])
                with self.assertRaises(AssertionError):rank_prefix(base,run,prefix,2,'scheduler')

    def test_fatal_console_cannot_be_rebound_to_another_rank_birth_or_retained_attempt(self):
        for mode in ('birth','retained'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as root:
                base,run,last=open_tail(root);failure=fault_tail(base,run)
                if mode=='birth':failure['process_start_ticks']+=1
                else:failure['retained_pending']['attempt_index']=last['attempt_index']
                from pathlib import Path
                Path(run['train_log']).write_text('SEARCH_G1_FATAL_SCHEDULER_EVIDENCE='+json.dumps(failure)+'\n')
                prefix=load_prefix(base,run,last['step'])
                with self.assertRaises(AssertionError):
                    self.audit(base,run,last['step'],actual_fixture_plan(base),prefix)

    def test_forged_prefix_interior_json_damage_and_extra_optimizer_restore_reject(self):
        with tempfile.TemporaryDirectory() as root:
            base,run,last=open_tail(root);prefix=load_prefix(base,run,last['step'])
            changed=copy.deepcopy(prefix);changed['cutoff_attempt_index']+=1
            with self.assertRaises(AssertionError):rank_prefix(base,run,changed,0,'optimizer')
            path=base/'logs/optimizer-rank-0.jsonl';original=path.read_bytes()
            path.write_bytes(original+b'{"partial\n')
            with self.assertRaises((AssertionError,ValueError)):rank_prefix(base,run,prefix,0,'optimizer')
            path.write_bytes(original)
            values=[json.loads(line) for line in original.splitlines()]
            rows(path,[dict(run_id=run['run_id'],rank=0,kind='RESTORE',last_attempt_index=prefix['origin']['attempt_index'])]+values)
            with self.assertRaises(AssertionError):rank_prefix(base,run,prefix,0,'optimizer')

    def test_fatal_identity_sources_cannot_mask_retained_failure_as_later_tail(self):
        for mode in ('context','prepared','top_level','float_top','float_prepared'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as root:
                base,run,last=open_tail(root);failure=fault_tail(base,run)
                if mode=='context':
                    failure['context']=dict(attempt_index=last['attempt_index']+1,
                        attempt_id='b'*32,logical_update=last['step']+1)
                elif mode=='prepared':
                    failure['retained_prepared']=dict(failure['retained_pending'],attempt_index=last['attempt_index'])
                elif mode=='top_level':failure['logical_update']=last['step']
                elif mode=='float_top':failure['logical_update']=float(failure['logical_update'])
                else:failure['retained_prepared']=dict(failure['retained_pending'],attempt_index=float(failure['retained_pending']['attempt_index']))
                path=base/'logs/scheduler-rank-2.jsonl'
                values=[json.loads(line) for line in path.read_text().splitlines()];values[-1]=failure;rows(path,values)
                from pathlib import Path
                Path(run['train_log']).write_text('SEARCH_G1_FATAL_SCHEDULER_EVIDENCE='+json.dumps(failure)+'\n')
                prefix=load_prefix(base,run,last['step'])
                with self.assertRaises(AssertionError):rank_prefix(base,run,prefix,2,'scheduler')
                with self.assertRaises(AssertionError):console_evidence(base,run,prefix)

    def test_unscored_return_is_preserved_as_discarded_without_fabricated_reward(self):
        with tempfile.TemporaryDirectory() as root:
            base,run,last=open_tail(root)
            path=base/'logs/actual-updates.jsonl';updates=[json.loads(line) for line in path.read_text().splitlines()]
            rows(path,updates[:-1]);prefix=load_prefix(base,run,last['step']-1)
            self.assertEqual(prefix['segment']['discarded_unscored_attempts'][0]['logical_update'],last['step'])
            self.assertEqual(prefix['segment']['discarded_successful_adam'],20)
            self.assertFalse(prefix['production_admission'])

    def test_tail_index_logical_uuid_and_console_only_fault_are_bound(self):
        for mode in ('retained_logical','retained_uuid','cursor'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as root:
                base,run,last=open_tail(root)
                path=base/'logs/scheduler-rank-2.jsonl';values=[json.loads(line) for line in path.read_text().splitlines()]
                if mode=='retained_logical':values[-1]['logical_update']=last['step']
                elif mode=='retained_uuid':values[-1]['attempt_id']=last['attempt_id']
                else:values[-1]['attempt_index']+=1
                rows(path,values);prefix=load_prefix(base,run,last['step'])
                with self.assertRaises(AssertionError):rank_prefix(base,run,prefix,2,'scheduler')
        with tempfile.TemporaryDirectory() as root:
            base,run,last=open_tail(root);failure=fault_tail(base,run)
            path=base/'logs/scheduler-rank-2.jsonl';values=[json.loads(line) for line in path.read_text().splitlines()]
            rows(path,values[:-1]);prefix=load_prefix(base,run,last['step'])
            self.assertFalse(rank_prefix(base,run,prefix,2,'scheduler')[1]['requires_fault_review'])
            self.assertTrue(console_evidence(base,run,prefix)['requires_fault_review'])

    def test_actor_readout_prefix_keeps_rank_order_and_rejects_duplicate_or_changed_binding(self):
        with tempfile.TemporaryDirectory() as root:
            base,run,plan,updates=fixture(root,start=50)
            for record in updates:
                record['probe_version']='fixture-version-'+str(record['step'])
                record['probe_actor_binding']=dict(row_order_verified=True,feature_rows=5120)
            rows(base/'logs/actual-updates.jsonl',updates)
            prefix=load_prefix(base,run,updates[-1]['step'])
            for rank in range(4):
                receipts=[]
                for record in updates:
                    binding=dict(run_id=run['run_id'],execution_id=run['execution_id'],
                        attempt_id=record['attempt_id'],attempt_index=record['attempt_index'],
                        logical_update=record['step'],completed_outer=record['step']-1,
                        optimizer_writes=record['optimizer_writes_before'],head_version=record['probe_version'])
                    receipts.append(dict(rank=rank,detached=True,binding=binding,feature_rows=1280,
                        row_ids=list(range(rank*1280,(rank+1)*1280)),feature_sha256='a'*64))
                path=base/'logs'/f'actor-probe-features-rank-{rank}.jsonl'
                rows(path,receipts);path.write_bytes(path.read_bytes()+b'{"partial')
            proof=verify_actor_readouts(base,run,updates,retained_prefix=prefix)
            self.assertEqual(proof['actual_completed_outers'],2)
            self.assertTrue(all(file['trailing_fragment_bytes']==9 for file in proof['retained_prefix_files']))
            path=base/'logs/actor-probe-features-rank-3.jsonl';original=path.read_bytes()
            values=[json.loads(line) for line in original[:original.rfind(b'\n')+1].splitlines()]
            rows(path,values+[copy.deepcopy(values[-1])])
            with self.assertRaises(AssertionError):verify_actor_readouts(base,run,updates,retained_prefix=prefix)
            values[-1]['binding']['optimizer_writes']+=1;rows(path,values)
            with self.assertRaises(AssertionError):verify_actor_readouts(base,run,updates,retained_prefix=prefix)

    def test_console_open_identity_matches_its_rank_and_cannot_diverge_between_fallbacks(self):
        for mode in ('rank_mismatch','duplicate_mismatch'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as root:
                base,run,last=open_tail(root);failure=fault_tail(base,run)
                changed=copy.deepcopy(failure);changed['retained_pending']['attempt_id']='d'*32
                from pathlib import Path
                text='SEARCH_G1_FATAL_SCHEDULER_EVIDENCE='+json.dumps(changed)+'\n'
                if mode=='duplicate_mismatch':text='SEARCH_G1_FATAL_SCHEDULER_EVIDENCE='+json.dumps(failure)+'\n'+text
                Path(run['train_log']).write_text(text)
                prefix=load_prefix(base,run,last['step'])
                with self.assertRaises(AssertionError):console_evidence(base,run,prefix)


if __name__=='__main__':unittest.main()
