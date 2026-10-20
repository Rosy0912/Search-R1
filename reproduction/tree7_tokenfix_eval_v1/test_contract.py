"""CPU fault injection for the new lineage and allocation adapter, no GPU call."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from reproduction.official_eval_v2 import driver, port_recovery
from . import contract as c
from .prepare import build_plan
from .scheduler import exact, zero, actual
from .submit import command, params


def fixture():
    # This synthetic proof exercises validation only and is never a production receipt.
    native=dict(uid=c.UID,source_manifest_sha256=c.TRAIN_SOURCE,condition="common_initialization_tree7_sampled_token_prefix_repair_v1",not_old_tree179_model=True,logical_parameters=339,complete_iterations_history=179,plan_sha256=c.TRAIN_PLAN,
                status='CPU_ACTOR_NATIVE_FILE_TENSOR_ADAM_RNG_SCHEDULER_AUDIT_PASSED',
                actual_iterations=179,directory_label=180,actual_adam_per_rank=1432,world_size=8,
                local_small_backup_verified=True,files={str(i):{} for i in range(24)},
                workers=[dict(rank=i,adam_min=1432,adam_max=1432,scheduler=dict(last_epoch=179)) for i in range(8)])
    files={n:'a'*64 for n in ['config.json','generation_config.json','tokenizer.json','tokenizer_config.json','model.safetensors']}
    hf=dict(uid=c.UID,plan_sha256=c.TRAIN_PLAN,actual_training_iterations=179,final_directory_label=180,
            status='FINAL_HF_OFFICIAL_CPU_MERGE_ALL_TENSORS_AND_METADATA_ACCEPTED',
            local_small_backup_verified=True,route_sha256='15ce94973d0b3cdc8dcba2c4d3c879422e0449ba7a9f3fb350c748cf5e946470',native_route_sha256='ae83cf3655613242deabd47a36613f4e5acdbc8a3b1bc9df5e44e5a743b61ad0',full_hf_cpu_reload=dict(logical_parameters=339,elements=7615616512,all_logical_parameters_bit_exact_to_exported_files=True,tied_storage=False),native_acceptance_sha256='b'*64,target=str(c.MODEL),cuda_initialized=False,
            parameter_comparison=dict(exact_native_bf16_projection_all_parameters=True,logical_parameters=339,
                                      rank_parameter_comparisons=2712,parameters_different_from_initialization=['fixture.weight']),
            files={n:dict(sha256=h) for n,h in files.items()},
            hf_file_identity_sha256=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest())
    reload=dict(status='ACTUAL_HF_CPU_RELOAD_ALL339_PARAMETERS_AND_COMMONV2_51713_PROMPTS_VERIFIED',uid=c.UID,
                hf_file_identity_sha256=hf['hf_file_identity_sha256'],hf_acceptance_sha256='c'*64,
                all_loaded_parameters_exact=True,logical_parameters_compared=339,parameter_elements=7615616512,
                cuda_initialized=False,all51713_prompt_tokenization_verified=True,max_prompt_tokens=400,
                data_sha256=c.FINAL_JSONL,id_gold_identity=c.ID_GOLD,profile=c.PROFILE,science=copy.deepcopy(c.SCIENCE))
    evidence={n:dict(path=str(c.ROOT/(n+'-acceptance.json')),sha256=h*64) for n,h in [('native','b'),('hf','c'),('reload','d')]}
    with patch('reproduction.tree7_tokenfix_eval_v1.prepare.sha',return_value='f'*64):
        plan=build_plan(1,'e'*40,copy.deepcopy(c.FROZEN),native,hf,reload,evidence,c.FINAL_JSONL,c.ID_GOLD,
                        'f'*64,c.COMMON/'receipts/pj-retriever-ready.json','http://10.120.1.198:29147/retrieve')
    return plan,native,hf,reload


class Contract(unittest.TestCase):
    def setUp(self):self.p,self.n,self.h,self.r=fixture()
    def valid(self):return c.validate_plan(self.p,self.n,self.h,self.r)
    def test_valid_three_independent_slots(self):
        roots=set()
        for i in (1,2,3):
            self.p.update(repeat=i,job=c.job_name(i),root=str(c.ROOT/('r'+str(i))))
            roots.add(self.valid())
        self.assertEqual(len(roots),3)
        for value in (0,4,True,'1'):
            with self.assertRaises(AssertionError):c.job_name(value)
    def test_reject_base_search_or_old3b(self):
        for k,v in [('kind','base'),('kind','searchr1'),('training_uid','227708e6-4b7b-4262-9407-7aa8c2644731'),('model',str(c.COMMON/'official-treegrpo/old'))]:
            p=copy.deepcopy(self.p);p[k]=v
            with self.assertRaises(AssertionError):c.validate_plan(p,self.n,self.h,self.r)
    def test_reject_wrong_native_iteration_and_adam(self):
        for k,v in [('actual_iterations',180),('directory_label',179),('world_size',16),('actual_adam_per_rank',0),('local_small_backup_verified',False)]:
            n=copy.deepcopy(self.n);n[k]=v
            with self.assertRaises(AssertionError):c.validate_plan(self.p,n,self.h,self.r)
    def test_reject_missing_duplicate_or_wrong_rank(self):
        for key,value in [('workers',self.n['workers'][:-1]),('workers',[self.n['workers'][0]]*8),('files',{})]:
            n=copy.deepcopy(self.n);n[key]=value
            with self.assertRaises(AssertionError):c.validate_plan(self.p,n,self.h,self.r)
        n=copy.deepcopy(self.n);n['workers'][7]['adam_min']=1431
        with self.assertRaises(AssertionError):c.validate_plan(self.p,n,self.h,self.r)
        n=copy.deepcopy(self.n);n['workers'][7]['scheduler']['last_epoch']=178
        with self.assertRaises(AssertionError):c.validate_plan(self.p,n,self.h,self.r)
    def test_reject_wrong_export_projection_or_sha(self):
        for key,value in [('logical_parameters',435),('rank_parameter_comparisons',2711),('exact_native_bf16_projection_all_parameters',False),('parameters_different_from_initialization',[])]:
            h=copy.deepcopy(self.h);h['parameter_comparison'][key]=value
            with self.assertRaises(AssertionError):c.validate_plan(self.p,self.n,h,self.r)
        h=copy.deepcopy(self.h);h['files']['model.safetensors']['sha256']='0'*64
        with self.assertRaises(AssertionError):c.validate_plan(self.p,self.n,h,self.r)
    def test_reject_missing_reload_or_prompt_validation(self):
        for k,v in [('logical_parameters_compared',338),('parameter_elements',1),('cuda_initialized',True),('all51713_prompt_tokenization_verified',False),('max_prompt_tokens',2049),('hf_acceptance_sha256','0'*64)]:
            r=copy.deepcopy(self.r);r[k]=v
            with self.assertRaises(AssertionError):c.validate_plan(self.p,self.n,self.h,r)
    def test_reject_scientific_data_or_worker_change(self):
        for k,v in [('data_sha256','0'*64),('id_gold_identity','0'*64),('model_revision','a09a35458c702b33eeacc393d103063234e8bc28'),('repeat_count',5)]:
            p=copy.deepcopy(self.p);p[k]=v
            with self.assertRaises(AssertionError):c.validate_plan(p,self.n,self.h,self.r)
        for key in c.SCIENCE:
            p=copy.deepcopy(self.p);p['science'][key]=None
            with self.assertRaises(AssertionError):c.validate_plan(p,self.n,self.h,self.r)
        p=copy.deepcopy(self.p);p['source_files'][next(iter(c.FROZEN))]='0'*64
        with self.assertRaises(AssertionError):c.validate_plan(p,self.n,self.h,self.r)
    def test_gate_exact_uid_source_and_resources(self):
        sub=dict(job=self.p['job'],uid='fixture-uid',plan_sha256='0'*64)
        gate=dict(plan_sha256='0'*64,source=self.p['source_commit'],actual=dict(name=self.p['job'],uid='fixture-uid',state='RUNNING',resources=c.ALLOCATION))
        c.validate_gate(self.p,sub,gate)
        for key,value in [('uid','other'),('state','PENDING'),('resources',dict(c.ALLOCATION,gpus=16))]:
            g=copy.deepcopy(gate);g['actual'][key]=value
            with self.assertRaises(AssertionError):c.validate_gate(self.p,sub,g)
    def test_same_collection_ast_and_original_worker(self):
        source=Path(driver.__file__).read_text();entry=Path(__file__).with_name('entry.py').read_text()
        a=source[source.index(" ctx=mp.get_context('spawn');processes=[]"):source.index("\nif __name__=='__main__':")]
        b=entry[entry.index(" ctx=mp.get_context('spawn');processes=[]"):entry.index("\n\nif __name__=='__main__':")]
        a=a.replace('target=worker,args=','target=startup_worker,args=')
        self.assertEqual(ast.dump(ast.parse('def f():\n'+a)),ast.dump(ast.parse('def f():\n'+b)))
        self.assertIs(port_recovery.ORIGINAL_WORKER,driver.worker)
        for n,h in c.FROZEN.items():self.assertEqual(c.sha(Path(__file__).resolve().parents[2]/n),h)
    def test_command_only_new_entry_and_fixed_resources(self):
        p=params(self.p,c.ROOT/'plans/r1.json')
        self.assertEqual((p['num_nodes'],p['gpus_per_task'],p['cpus_per_task'],p['memory_per_task'],p['retry']),(1,8,64,600,0))
        self.assertFalse(p.get('include'))
        self.assertIn('reproduction.tree7_tokenfix_eval_v1.launch',p['cmd'])
        self.assertIn('unset HTTP_PROXY',p['cmd'])
        self.assertNotIn('official_treegrpo_final_launch',p['cmd'])


class RepairLineage(unittest.TestCase):
    def test_old_parent_model_rejected(self):
        p,n,h,r=fixture();n['uid']='432a4519-d9fd-494e-819d-b5220632230c'
        with self.assertRaises(AssertionError):c.validate_plan(p,n,h,r)
    def test_missing_real_full_hf_reload_rejected(self):
        p,n,h,r=fixture();h['full_hf_cpu_reload']['all_logical_parameters_bit_exact_to_exported_files']=False
        with self.assertRaises(AssertionError):c.validate_plan(p,n,h,r)
    def test_wrong_frozen_route_rejected(self):
        p,n,h,r=fixture();h['route_sha256']='0'*64
        with self.assertRaises(AssertionError):c.validate_plan(p,n,h,r)
    def test_training_source_manifest_rejected(self):
        p,n,h,r=fixture();n['source_manifest_sha256']='0'*64
        with self.assertRaises(AssertionError):c.validate_plan(p,n,h,r)

class Scheduler(unittest.TestCase):
    def test_exact_zero_and_positive_control(self):
        class Fake:
            def list_training_jobs(self,**kw):return {}
            def get_training_job(self,name):
                e=RuntimeError('notfound');e.response=type('Response',(),dict(status_code=404))();raise e
        self.assertEqual(exact(Fake(),'new-slot'),[]);zero(Fake(),'new-slot')
    def test_reject_pagination_or_nonexact_or_400(self):
        for response in [dict(next_page_token='more'),dict(total_size=1),dict(training_jobs=[dict(name='other')],total_size=1),dict(error='denied')]:
            fake=type('Fake',(),dict(list_training_jobs=lambda self,**kw:response))()
            with self.assertRaises(AssertionError):exact(fake,'slot')
        class Fake:
            def list_training_jobs(self,**kw):return {}
            def get_training_job(self,name):
                e=RuntimeError('bad request');e.response=type('Response',(),dict(status_code=400))();raise e
        with self.assertRaises(AssertionError):zero(Fake(),'slot')
    def test_actual_resource_projection(self):
        r=dict(accelerate_device_count=8,cpu_count=64,memory_gib=600,machine_types=['n2ls.ie.i60b'],accelerate_device_model='A100',accelerate_device_memory_gib=80)
        job=dict(name='slot',uid='uid',status=dict(state='RUNNING'),spec=dict(vc_job=dict(tasks=[dict(replicas=1,resource_spec=r)])))
        self.assertEqual(actual(job)['resources'],c.ALLOCATION)
        job['spec']['vc_job']['tasks'][0]['replicas']=2
        with self.assertRaises(AssertionError):actual(job)


if __name__=='__main__':unittest.main()
