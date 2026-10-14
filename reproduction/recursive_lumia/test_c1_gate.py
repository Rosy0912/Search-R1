"""Completion must follow real mechanism evidence and actual update accounting."""
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest

from search_r1.baselines.recursive_ablation import Planner, Options
from search_r1.baselines.recursive_branch import Record
from .c1_evidence import validate_question, validate_rounds
from .gate_c1_complete import validate_exit, verify_final_native
from .inspect_run import sha
from .prepare_ablation import condition_for


class CompletionGate(unittest.TestCase):
    def events(self, reward):
        events=[]
        def emit(event, **fields):
            events.append(dict(event=event,event_seq=len(events)+1,**fields))
        root=Record('root',None,'tree',0,0,None,[0,1,2],['a','b','c'],'initial')
        planner=Planner(root,random.Random(1),emit,options=Options(variant='root_only'))
        while True:
            request=planner.next_request()
            if request is None:break
            parent,index,attempt=request
            steps=parent.steps[:index]+['new-'+str(attempt)]
            planner.accept_result(Record(str(attempt),parent.trajectory_id,'tree',1,
                reward(attempt),None,[None]*len(steps),steps,'pending'))
        planner.finish()
        return events

    def test_accepts_actual_planner_contrast_and_budget_fill_paths(self):
        contrast=validate_question(self.events(lambda n:1))
        self.assertEqual((contrast['branches'],contrast['contrast']),(4,4))
        filled=validate_question(self.events(lambda n:0))
        self.assertEqual((filled['branches'],filled['filled']),(10,4))
        mixed=validate_question(self.events(lambda n:n%3==0))
        self.assertEqual(mixed['contrast'],3)

    def test_rejects_nonroot_parent_or_changed_acceptance_or_missing_end(self):
        for event,key,value in [('search_start','parent_id','child'),
                                ('branch_end','generation',2),
                                ('branch_end','source','same_reward')]:
            rows=self.events(lambda n:1)
            next(row for row in rows if row['event']==event)[key]=value
            with self.assertRaises(ValueError):validate_question(rows)
        with self.assertRaises(ValueError):validate_question(self.events(lambda n:1)[:-1])

    def rounds(self):
        attempts=[];total=0
        for outer in range(1,151):
            writes=0 if outer==2 else 13 if outer==4 else 20
            guard=(dict(status='NORMAL_SAFETY_GUARD_ROUND_CONSUMED',optimizer_writes=writes,
                opportunities_processed=20,opportunities_skipped=20-writes) if writes<20 else None)
            total+=writes
            attempts.append(dict(outer=outer,actual_adam_writes_per_rank=writes,normal_guard=guard,
                reported_pg_loss_finite=True,ranks=[dict(all_observed_loss_finite=True)]*8,
                cumulative_adam_writes_per_rank={str(i):total for i in range(8)}))
        return dict(completed_rounds=150,attempts=attempts,sixth_real_generation_observed=True,
            native5=dict(pre_evaluation_saved=True,post_evaluation_saved=True,
                same_world_reload_verified=True,all_native_files_hash_verified=True)),total

    def test_normal_guard_zero_is_a_consumed_round_not_twenty_updates(self):
        audit,total=self.rounds()
        result=validate_rounds(audit,{str(i):total for i in range(8)})
        self.assertEqual(result['actual_adam_per_rank'],2973)
        self.assertEqual(result['zero_write_rounds'],1)
        with self.assertRaises(ValueError):validate_rounds(audit,{str(i):3000 for i in range(8)})

    def test_incomplete_nonfinite_or_unexplained_skip_never_releases_next_run(self):
        for mutation in ('incomplete','loss','guard','sixth','checkpoint'):
            audit,total=self.rounds()
            if mutation=='incomplete':audit['completed_rounds']=149
            if mutation=='loss':audit['attempts'][4]['reported_pg_loss_finite']=False
            if mutation=='guard':audit['attempts'][1]['normal_guard']=None
            if mutation=='sixth':audit['sixth_real_generation_observed']=False
            if mutation=='checkpoint':audit['native5']['all_native_files_hash_verified']=False
            with self.assertRaises(ValueError):validate_rounds(audit,{str(i):total for i in range(8)})

    def completion(self):
        record=dict(group='C1-root-only',condition=condition_for('C1-root-only'),
            run_id='run',source_commit='source',config_sha256='config',data_identity='data')
        entry=dict(record,allocation='123',pid=44,tick=55,uid=1066)
        complete=dict(record,status='COMPLETE_NATIVE_150_TRAINING_AND_FOUR_SCHEDULED_DEV',
            completed_outer=150,evaluation_steps=[5,50,100,150])
        terminal=dict(record,rc=0,error=None,pid=44,tick=55,uid=1066)
        wait=dict(rc=0,pid=44,allocation='123',group=record['group'],source='source')
        return record,entry,complete,terminal,wait

    def test_scheduler_success_alone_does_not_establish_completion(self):
        args=self.completion();scheduler='123|jmzhang|COMPLETED|0:0|\n'
        validate_exit(*args,'123',scheduler)
        for index,key,value in [(2,'completed_outer',149),(3,'rc',1),(4,'pid',45),
                                (4,'source','other'),(2,'config_sha256','other')]:
            changed=copy.deepcopy(args);changed[index][key]=value
            with self.assertRaises(ValueError):validate_exit(*changed,'123',scheduler)
        for state in ('RUNNING','FAILED','CANCELLED'):
            with self.assertRaises(ValueError):validate_exit(*args,'123',scheduler.replace('COMPLETED',state))

    def test_c2_requires_exact_condition_and_cannot_pass_the_c1_gate(self):
        record,entry,complete,terminal,wait=self.completion()
        for value in (record,entry,complete,terminal,wait):
            value['group']='C2-random-final'
        record['condition']=condition_for('C2-random-final')
        args=(record,entry,complete,terminal,wait,'123','123|jmzhang|COMPLETED|0:0|\n')
        validate_exit(*args,expected_group='C2-random-final')
        with self.assertRaises(ValueError):validate_exit(*args)
        record['condition']=condition_for('C1-root-only')
        with self.assertRaises(ValueError):validate_exit(*args,expected_group='C2-random-final')

    def test_final_native_checks_actual_bytes_not_only_manifest_claim(self):
        with tempfile.TemporaryDirectory() as temporary:
            run=Path(temporary).resolve();record,entry,complete,_,_=self.completion()
            directory=run/'checkpoints/training_state/global_step_150/post-evaluation'
            directory.mkdir(parents=True)
            identity=dict(job_id='123',code_commit='source',config_sha256='config',
                data_identity='data',completed_step=150,actual_update_limit=150)
            members=[]
            for i in range(9):
                file=directory/str(i);file.write_bytes(str(i).encode())
                members.append(dict(path=str(file),sha256=sha(file),bytes=1))
            workers=[dict(members[i],identity=dict(identity,rank=i,world_size=8),
                full_worker_state_finite=True,scheduler_last_epoch=150,
                actual_total_optimizer_calls=2973,observed_adam_steps=[2973]) for i in range(8)]
            manifest=directory/'manifest.json'
            manifest.write_text(json.dumps(dict(status='COMPLETE_SAME_WORLD_STATE_SAVED',
                boundary_phase='post-evaluation',identity=identity,worker_states=workers,
                driver=dict(members[8],next_step=151))))
            complete['checkpoint']=dict(path=str(manifest),sha256=sha(manifest))
            self.assertTrue(verify_final_native(run,record,entry,complete,2973)['all_nine_native_files_hash_verified'])
            (directory/'0').write_bytes(b'x')
            with self.assertRaises(ValueError):verify_final_native(run,record,entry,complete,2973)


if __name__=='__main__':unittest.main()
