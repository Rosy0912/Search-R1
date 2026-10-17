import copy,unittest
from unittest.mock import patch
from pathlib import Path
from . import test_base_contracts as base_tests
from . import base_pj as m
from .contracts import digest

class Migration(unittest.TestCase):
    def setUp(self):
        fixture=base_tests.BaseContracts();fixture.setUp();self.addCleanup(fixture.doCleanups)
        self.parent=copy.deepcopy(fixture.plan);self.export=dict(status='EXACT_BASE_PROTOCOL_INPUTS_VERIFIED',questions=51713,
            template=fixture.template,template_sha256=self.parent['template_plan_sha256'],plan=self.parent,
            config={'immutable_science':4},config_digest=digest({'immutable_science':4}))
        self.addCleanup(patch.stopall);patch.object(m,'EXPORT_CONFIG_DIGEST',self.export['config_digest']).start()
        self.sub=dict(job_id='138989',job=self.parent['job'],plan_sha256='a'*64,checkpoint_identity=self.parent['checkpoint_identity'],output=self.parent['output'],repeat_index=1)
        self.plan=copy.deepcopy(self.parent);self.plan.update(protocol='base_seven_pj_migration_v1',backend='pj_pt',base=m.BASE,
            model=m.MODEL,data_root=m.BASE+'/common-data',retriever_receipt=m.BASE+'/receipts/pj-retriever-ready.json',
            evaluation_code_commit='b'*40,evaluation_run_id='c'*32,job='zhangjunming-ttt-base-r1-pj',
            output=str(Path(m.BASE)/'baseline-seven-evaluations'/self.parent['checkpoint_identity']/self.parent['protocol_identity']/'r1/pj-v1/output'),
            migration={k:'a'*64 if k.endswith('sha256') else '/bound/'+k for k in m.LINKS})
        self.retired=dict(job_id='138989',plan_sha256='a'*64,prediction_bytes=0,progress_exists=False,
            status='CANCELLED_NEVER_STARTED_ZERO_PREDICTIONS',observations=[dict(job_id='138989',state='CANCELLED',elapsed=0,start='Unknown',active_queue=[]) for _ in range(2)])
    def validate(self,plan=None,retired=None):
        return m.validate_migration(plan or self.plan,self.parent,self.sub,self.export,self.retired if retired is None else retired)
    def test_valid_retired_origin(self):self.assertEqual(str(self.validate())+'/output',self.plan['output'])
    def test_rejects_started_running_or_partial_origin(self):
        for key,value in [('state','RUNNING'),('elapsed',1),('start','2026-10-17T00:00:00'),('active_queue',['138989']),('job_id','138992')]:
            r=copy.deepcopy(self.retired);r['observations'][0][key]=value
            with self.assertRaises(ValueError):self.validate(retired=r)
        for key,value in [('prediction_bytes',1),('progress_exists',True),('job_id','wrong'),('observations',[])]:
            r=copy.deepcopy(self.retired);r[key]=value
            with self.assertRaises(ValueError):self.validate(retired=r)
    def test_rejects_scientific_or_repeat_change(self):
        for k,v in [('model_files',{}),('repeat_index',2),('repeat_count',5),('ordered_ids_sha256','0'*64),('protocol_identity','0'*64)]:
            p=copy.deepcopy(self.plan);p[k]=v
            with self.assertRaises(ValueError):self.validate(plan=p)
    def test_rejects_unbound_export_configuration(self):
        self.export['config']['immutable_science']=5
        with self.assertRaises(ValueError):self.validate()
        self.export['config_digest']=digest(self.export['config'])
        with self.assertRaises(ValueError):self.validate()
    def test_rejects_wrong_backend_paths_or_job(self):
        for k,v in [('backend','slurm_lumia'),('model','/other'),('data_root','/other'),('output',self.parent['output']),('job','other'),('evaluation_run_id',self.parent['evaluation_run_id'])]:
            p=copy.deepcopy(self.plan);p[k]=v
            with self.assertRaises(ValueError):self.validate(plan=p)
    def test_rejects_missing_or_extra_evidence(self):
        for key in ('origin_plan_sha256','retirement_sha256'):
            p=copy.deepcopy(self.plan);del p['migration'][key]
            with self.assertRaises(ValueError):self.validate(plan=p)
        p=copy.deepcopy(self.plan);p['skip_validation']=True
        with self.assertRaises(ValueError):self.validate(plan=p)


class AutomaticPlacement(unittest.TestCase):
    def setUp(self):
        f=Migration();f.setUp();self.addCleanup(f.doCleanups)
        self.parent,self.sub,self.export,self.retired=f.parent,f.sub,f.export,f.retired
        self.previous=copy.deepcopy(f.plan);self.auto=copy.deepcopy(f.plan)
        self.auto.update(job='zhangjunming-ttt-base-r1-auto',output=f.plan['output'].replace('/pj-v1/','/pj-auto-v2/'),evaluation_run_id='d'*32)
        self.auto['migration'].update({k:'e'*64 if k.endswith('sha256') else '/bound/'+k for k in m.PJ_LINKS})
        self.pj_retired=dict(job=self.previous['job'],uid='uid1',plan_sha256='e'*64,status='TERMINATED_NEVER_STARTED_ZERO_PREDICTIONS',observations=[dict(job=self.previous['job'],uid='uid1',state='TERMINATED',ever_running=False,start_exists=False,log_exists=False,output_exists=False,prediction_bytes=0) for _ in range(2)])
    def validate(self,p=None,r=None,previous=True):
        return m.validate_migration(p or self.auto,self.parent,self.sub,self.export,self.retired,self.previous if previous else None,r or self.pj_retired)
    def test_automatic_valid(self):self.assertEqual(str(self.validate())+'/output',self.auto['output'])
    def test_automatic_requires_zero_run_and_output(self):
        for key,value in [('state','PENDING'),('ever_running',True),('start_exists',True),('log_exists',True),('output_exists',True),('prediction_bytes',1),('uid','other')]:
            r=copy.deepcopy(self.pj_retired);r['observations'][0][key]=value
            with self.assertRaises(ValueError):self.validate(r=r)
    def test_automatic_rejects_wrong_lineage(self):
        for key,value in [('job','other'),('plan_sha256','0'*64),('observations',[])]:
            r=copy.deepcopy(self.pj_retired);r[key]=value
            with self.assertRaises(ValueError):self.validate(r=r)
        with self.assertRaises(ValueError):self.validate(previous=False)
    def test_automatic_rejects_changed_repeat_or_same_execution(self):
        for key,value in [('repeat_index',2),('model','/other'),('evaluation_run_id',self.previous['evaluation_run_id'])]:
            p=copy.deepcopy(self.auto);p[key]=value
            with self.assertRaises(ValueError):self.validate(p=p)

if __name__=='__main__':unittest.main()
