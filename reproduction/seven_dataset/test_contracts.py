import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from . import contracts as c


class Contracts(unittest.TestCase):
    def test_queued_replacement_requires_never_started_cancelled_allocation(self):
        c.validate_replacement_state('CANCELLED by 1000',0,'Unknown','UNSTARTED_DEPENDENCY_BLOCKED')
        c.validate_replacement_state('CANCELLED by 1000',0,'None','UNSTARTED_DEPENDENCY_BLOCKED')
        c.validate_replacement_state('FAILED',30,'2026-10-16T00:00:00')
        for state,elapsed,start,reason in [
            ('RUNNING',0,'Unknown','UNSTARTED_DEPENDENCY_BLOCKED'),
            ('CANCELLED',1,'Unknown','UNSTARTED_DEPENDENCY_BLOCKED'),
            ('CANCELLED',0,'2026-10-16T00:00:00','UNSTARTED_DEPENDENCY_BLOCKED'),
            ('PENDING',0,'Unknown','UNSTARTED_DEPENDENCY_BLOCKED'),
            ('CANCELLED',0,'Unknown',None),('FAILED',0,'Unknown','other')]:
            with self.assertRaises(ValueError):c.validate_replacement_state(state,elapsed,start,reason)

    def test_jsonl_preserves_embedded_unicode_line_separators(self):
        rows=[{'prediction':'a'+char+'b'} for char in ('\u0085','\u2028','\u2029','\n','\r','\v','\f')]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'predictions.jsonl'
            payload=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)
            path.write_bytes(payload.encode('utf-8'))
            self.assertEqual(c.read_prediction_rows(path),rows)
            path.write_bytes(payload.encode('utf-8')[:-1])
            with self.assertRaises(ValueError):c.read_prediction_rows(path)
            path.write_bytes(b'{"prediction":"truncated\n')
            with self.assertRaises(ValueError):c.read_prediction_rows(path)
            path.write_bytes(b'\n')
            with self.assertRaises(ValueError):c.read_prediction_rows(path)

    def test_scope_exactly_twenty_one(self):
        accepted=[]
        for g in c.GROUPS:
            for step in (5,50,100,150,200):
                try:c.validate_milestone(g,step)
                except ValueError:continue
                accepted.append((g,step))
        self.assertEqual(len(accepted),21)
        self.assertEqual([s for g,s in accepted if g=='position-early'],[50])
        self.assertNotIn(('budget-5',200),accepted)
        for g,s in [('curriculum',50),('original',True),('original',5)]:
            with self.assertRaises(ValueError):c.validate_milestone(g,s)

    def rows(self):
        expected=[dict(question_id='q'+str(i),source=s,gold_answers=['yes'])
                  for i,s in enumerate(['nq','nq','bamboogle'])]
        rows=[dict(r,final_index=i,prediction='yes' if i==0 else 'no',em=int(i==0),checkpoint_identity='model')
              for i,r in enumerate(expected)]
        return expected,rows

    def accept(self,rows,expected):
        with patch.multiple(c,TOTAL=3,COUNTS={'nq':2,'bamboogle':1},IDS_SHA=c.ids_hash(['q0','q1','q2'])):
            return c.accept_rows(rows,expected,'model',lambda p,g:p in g)

    def test_missing_repeated_reordered_ids_cannot_be_complete(self):
        expected,rows=self.rows()
        self.assertEqual(self.accept(rows,expected)['correct'],1)
        for bad in [rows[:-1],rows+[rows[0]], [rows[1],rows[0],rows[2]], [rows[0],rows[0],rows[2]]]:
            with self.assertRaises(ValueError):self.accept(bad,expected)

    def test_gold_source_score_model_cannot_be_substituted(self):
        expected,rows=self.rows()
        for key,value in [('gold_answers',['no']),('source','popqa'),('em',0),('em',float('nan')),
                          ('checkpoint_identity','other'),('prediction','no')]:
            bad=copy.deepcopy(rows);bad[0][key]=value
            with self.assertRaises(ValueError):self.accept(bad,expected)

    def test_incomplete_source_identity_fails(self):
        expected,rows=self.rows();expected[0]['source']='popqa'
        with self.assertRaises(ValueError):self.accept(rows,expected)

    def test_paired_counts_question_disagreements(self):
        expected,rows=self.rows();other=copy.deepcopy(rows);other[0]['em']=0;other[1]['em']=1
        with patch.object(c,'TOTAL',3):
            self.assertEqual(c.compare_paired(rows,other)['candidate_only_correct'],1)
            self.assertEqual(c.compare_paired(rows,other)['baseline_only_correct'],1)
            other[0]['question_id']='changed'
            with self.assertRaises(ValueError):c.compare_paired(rows,other)

    def test_actual_file_hash_and_index_are_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'model.safetensors').write_bytes(b'fixture')
            (p/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'weight':'model.safetensors'}}))
            plan={'model':tmp,'model_files':{f.name:c.sha(f) for f in p.iterdir()}}
            c.validate_model_files(plan)
            (p/'model.safetensors').write_bytes(b'changed')
            with self.assertRaises(ValueError):c.validate_model_files(plan)
            plan['model_files']['../escape']='bad'
            with self.assertRaises(ValueError):c.validate_model_files(plan)

    def test_fixed_seven_dataset_count(self):
        self.assertEqual(sum(c.COUNTS.values()),51713)
        self.assertEqual(len(c.COUNTS),7)

    def test_slurm_allocation_requires_exact_submission_binding(self):
        p=dict(job='seven-model',backend='slurm_lumia',evaluation_code_commit='code')
        intent=dict(job=p['job'],plan_sha256='sha');submission=dict(intent,job_id='123')
        env=dict(SEARCH_R1_EXECUTION_BACKEND='slurm_lumia',RECURSIVE_SOURCE_COMMIT='code',
                 SLURM_JOB_NAME='seven-model',SLURM_JOB_ID='123')
        self.assertEqual(c.validate_scheduler_binding(p,intent,submission,env,'sha'),'123')
        with self.assertRaises(ValueError):c.validate_scheduler_binding(p,intent,None,env,'sha')
        for key,value in [('SLURM_JOB_ID','124'),('SLURM_JOB_NAME','other'),('RECURSIVE_SOURCE_COMMIT','other')]:
            bad=dict(env);bad[key]=value
            with self.assertRaises(ValueError):c.validate_scheduler_binding(p,intent,submission,bad,'sha')

    def test_pt_binding_rejects_fabricated_slurm_environment(self):
        p=dict(job='pt-seven',backend='pj_pt',evaluation_code_commit='code')
        intent=dict(job=p['job'],plan_sha256='sha')
        env=dict(SEARCH_R1_EXECUTION_BACKEND='pj_pt',RECURSIVE_SOURCE_COMMIT='code',SEARCH_R1_JOB_ID='pt-seven')
        self.assertEqual(c.validate_scheduler_binding(p,intent,None,env,'sha'),'pt-seven')
        env['SLURM_JOB_ID']='123'
        with self.assertRaises(ValueError):c.validate_scheduler_binding(p,intent,None,env,'sha')

    def test_macro_and_micro_are_distinct(self):
        expected,rows=self.rows();result=self.accept(rows,expected)
        self.assertEqual(result['equal_dataset_em'],0.25)
        self.assertEqual(result['all_question_em'],1/3)

    def test_manifest_requires_completed_finite_eight_rank_identity(self):
        files={n:'sha' for n in ['config.json','generation_config.json','tokenizer.json',
            'tokenizer_config.json','model.safetensors.index.json','model.safetensors']}
        plan=dict(group='generation-2',outer=50,training_config_sha256='cfg',
            training_job='job',training_code_commit='code',model_files=files)
        identity=dict(completed_step=50,config_sha256='cfg',data_identity=c.DATA_SHA,
            job_id='job',code_commit='code',guard_continue_accounting={'completed_outer_updates':50})
        manifest=dict(status='COMPLETE_SAME_WORLD_STATE_SAVED',identity=identity,driver={'next_step':51},
            hf_identity={'files':files},worker_states=[dict(identity=dict(identity,rank=i,world_size=8),
            full_worker_state_finite=True,scheduler_last_epoch=50) for i in range(8)])
        c.validate_manifest(plan,manifest)
        bad=copy.deepcopy(manifest);bad['worker_states'][0]['identity']['rank']=1
        with self.assertRaises(ValueError):c.validate_manifest(plan,bad)
        for field,value in [('full_worker_state_finite',False),('scheduler_last_epoch',49)]:
            bad=copy.deepcopy(manifest);bad['worker_states'][0][field]=value
            with self.assertRaises(ValueError):c.validate_manifest(plan,bad)
        bad=copy.deepcopy(manifest);bad['identity']['guard_continue_accounting']['completed_outer_updates']=49
        with self.assertRaises(ValueError):c.validate_manifest(plan,bad)

if __name__=='__main__':unittest.main()
