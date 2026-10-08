"""Small CPU fixtures test software contracts, never actual training acceptance."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES']=''
import torch

HERE=Path(__file__).resolve().parent
WORK=HERE.parents[1]
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);out=importlib.util.module_from_spec(spec);spec.loader.exec_module(out);return out
n=module('candidate_native',HERE/'inspect_author200_checkpoint_cpu.py')
h=module('candidate_hf',HERE/'hf_author200_cpu.py')
mathhelper=n.exact_module(HERE.parent/'first5-native-1f209cab7e02/inspect_payloads.py',n.HELPER,'unchanged_storage_math',{})
state=n.exact_module(WORK/'opsd-e1-300/search_r1/baselines/full_state.py',n.FULL_STATE,'unchanged_full_state',{})
oldhf=n.exact_module(HERE.parent/'first5-12424d01025e/hf_cpu_reload.py',h.HELPER,'unchanged_full_hf_loader',{})


class Contracts(unittest.TestCase):
    def inventory(self,step=25):
        workers=[dict(rank=i,pid=100+i,object_ids=dict(model=1,optimizer=2,scheduler=3,teacher=4)) for i in range(8)]
        return dict(version=1,kind=n.KIND,source_commit=n.SOURCE,job_id='3261',run_root=n.RUN_ROOT,checkpoint_step=step,
            config_sha256=n.CONFIG,profile='author200',horizon=200,world_size=8,native_import_closure_sha256=n.CLOSURE,
            formal_manifest_sha256=n.FORMAL,accepted5=dict(manifest_sha256=n.ACCEPTED_MANIFEST,signed_receipt_sha256=n.ACCEPTED_RECEIPT,
            public_key_sha256=n.PUBLIC,driver_pid=n.DRIVER_PID,driver_start_ticks=n.DRIVER_START,workers=workers),
            rank_reports=[dict(rank=i) for i in range(8)],driver_report=dict(status='DRIVER_STATE_SERIALIZED_OK'),
            raw_statuses=dict(ranks=['FULL_STATE_SERIALIZED_OK']*8,driver='DRIVER_STATE_SERIALIZED_OK'))
    def scheduler(self,step=25):
        return dict(last_epoch=step,_step_count=step+1,base_lrs=[5e-6],_last_lr=[5e-6],lr_lambdas=[None])
    def sampler(self,step):
        return dict(size=169028,shuffle=True,seed=None,epoch=0,cursor=512*step,consumed_cursor=512*step,
                    order=list(range(169028)),generator=torch.Generator().manual_seed(42).get_state())
    def check_sampler(self,current,old,step):
        return n.validate_sampler(current,old,step,torch,n.sampler_settings(state,{'data':{'shuffle_train_dataloader':True},'baselines':{'seed':42}}))
    def test_all_saved_step_clocks(self):
        for step in n.STEPS:
            c=n.clock(step);self.assertEqual(c['adam_steps'],16*step);self.assertEqual(c['scheduler_step_count'],step+1)
            self.assertEqual(c['cursor'],512*step);self.assertEqual(n.scheduler(self.scheduler(step),step)['last_epoch'],step)
    def test_200_saved_next201_valid_without_execution_claim(self):
        c=n.validate_inventory(self.inventory(200),200);self.assertEqual(c['next_global_step'],201);self.assertEqual(c['adam_steps'],3200)
    def test_old5_and_future201_not_saved_current_scope(self):
        for step in (0,5,24,201,300,True,25.0):
            with self.assertRaises(ValueError):n.clock(step)
    def test_scheduler_wrong_outer_vs_adam_clock(self):
        for key,value in [('last_epoch',400),('_step_count',400),('last_epoch',True),('_step_count',26.0)]:
            s=self.scheduler();s[key]=value
            with self.assertRaises(ValueError):n.scheduler(s,25)
    def test_scheduler_no_old_warmup_or_zero(self):
        for lr in (0.0,1e-6,5e-6*25/57,float('nan')):
            s=self.scheduler();s['_last_lr']=[lr]
            with self.assertRaises(ValueError):n.scheduler(s,25)
    def test_scheduler_not_serialized_callable_lambda(self):
        s=self.scheduler();s['lr_lambdas']=[{}]
        with self.assertRaises(ValueError):n.scheduler(s,25)
    def test_inventory_exact_training_and_accepted_ids(self):
        n.validate_inventory(self.inventory(),25)
        for key,bad in [('source_commit','0'*40),('job_id','3210'),('profile','e1_300'),('horizon',300),('world_size',8.0),('checkpoint_step',25.0)]:
            v=self.inventory();v[key]=bad
            with self.assertRaises(ValueError):n.validate_inventory(v,25)
    def test_inventory_refuses_upgrade_serialized_to_reload(self):
        v=self.inventory();v['raw_statuses']['ranks'][4]='EXACT_FULL_STATE_RELOAD_OK'
        with self.assertRaises(ValueError):n.validate_inventory(v,25)
    def test_inventory_boolean_numeric_aliases_rejected(self):
        for mutate in (lambda v:v.update(version=True),lambda v:v['rank_reports'][0].update(rank=False)):
            v=self.inventory();mutate(v)
            with self.assertRaises(ValueError):n.validate_inventory(v,25)
    def test_rank_report_all_numeric_headers_are_exact_integers(self):
        report=dict(version=3,rank=0,world_size=8,step=25,pid=100,scheduler_last_epoch=25,optimizer_state_entries=37,model_dtensor_entries=435,
                    status='FULL_STATE_SERIALIZED_OK',algorithm='pure_opsd',model_state_dict_type='SHARDED_STATE_DICT',object_ids=dict(model=1,optimizer=2,scheduler=3,teacher=4))
        n.validate_rank_report(report,0,25,100)
        for key in ('version','rank','world_size','step','pid','scheduler_last_epoch','optimizer_state_entries','model_dtensor_entries'):
            bad=copy.deepcopy(report);bad[key]=float(bad[key])
            with self.assertRaises(ValueError):n.validate_rank_report(bad,0,25,100)
        bad=copy.deepcopy(report);bad['rank']=False
        with self.assertRaises(ValueError):n.validate_rank_report(bad,0,25,100)
    def test_driver_report_numeric_headers_and_stored_job_exact_types(self):
        report=dict(status='DRIVER_STATE_SERIALIZED_OK',pid=n.DRIVER_PID,step=25,uid=os.getuid(),job_id='3261')
        n.validate_driver_report(report,25)
        for key in ('pid','step','uid'):
            bad=copy.deepcopy(report);bad[key]=float(bad[key])
            with self.assertRaises(ValueError):n.validate_driver_report(bad,25)
        bad=copy.deepcopy(report);bad['job_id']=3261
        with self.assertRaises(ValueError):n.validate_driver_report(bad,25)
    def test_inventory_rejects_missing_alias_rank_and_boolean_object(self):
        for mutate in (lambda v:v['accepted5']['workers'].pop(),
                       lambda v:v['accepted5']['workers'][7].update(pid=100),
                       lambda v:v['accepted5']['workers'][7]['object_ids'].update(model=True),
                       lambda v:v['rank_reports'].reverse()):
            v=self.inventory();mutate(v)
            with self.assertRaises(ValueError):n.validate_inventory(v,25)
    def test_sampler_exact_continuous_200_and_permutation(self):
        out=self.check_sampler(self.sampler(200),self.sampler(5),200)
        self.assertEqual(out['sampler_cursor'],102400);self.assertTrue(out['unchanged_epoch0_generator'])
    def test_sampler_cursor_seed_shuffle_and_boolean_type(self):
        for key,bad in [('cursor',12799),('consumed_cursor',12801),('seed',0),('shuffle',False),('epoch',1),('epoch',False),('cursor',12800.0)]:
            v=self.sampler(25);v[key]=bad
            with self.assertRaises(ValueError):self.check_sampler(v,self.sampler(5),25)
    def test_sampler_same_permutation_set_changed_order_rejected(self):
        v=self.sampler(25);v['order'][0],v['order'][1]=v['order'][1],v['order'][0]
        with self.assertRaises(ValueError):self.check_sampler(v,self.sampler(5),25)
    def test_sampler_duplicate_order_and_bool_element_rejected(self):
        for bad in (0,True):
            v=self.sampler(25);v['order'][1]=bad
            with self.assertRaises(ValueError):self.check_sampler(v,self.sampler(5),25)
    def test_sampler_generator_changed_even_when_valid(self):
        v=self.sampler(25);v['generator']=torch.Generator().manual_seed(99).get_state()
        with self.assertRaises(ValueError):self.check_sampler(v,self.sampler(5),25)
    def test_sampler_generator_truncated(self):
        v=self.sampler(25);v['generator']=torch.zeros(16,dtype=torch.uint8)
        with self.assertRaises(RuntimeError):self.check_sampler(v,self.sampler(5),25)
    def test_frozen_default_sampler_actual5_to25_schema(self):
        config_path=WORK/'e1-author200/actual-snapshots/3261-20261005T212707464146Z/original-files/full_state/global_step_5/resolved-config.json'
        raw=config_path.read_bytes();self.assertEqual(hashlib.sha256(raw).hexdigest(),n.CONFIG);config=n.strict_json(raw)
        settings=n.sampler_settings(state,config);self.assertEqual(config['baselines']['seed'],42);self.assertIsNone(settings['seed'])
        sampler=state.StatefulSampler(169028,shuffle=config['data']['shuffle_train_dataloader']);iterator=iter(sampler)
        for _ in range(2560):next(iterator)
        sampler.commit_consumed(2560);old=sampler.state_dict()
        for _ in range(10240):next(iterator)
        sampler.commit_consumed(10240);current=sampler.state_dict()
        result=n.validate_sampler(current,old,25,torch,settings)
        self.assertIsNone(result['seed']);self.assertEqual(result['sampler_cursor'],12800)
        self.assertTrue(torch.equal(old['generator'],current['generator']));self.assertEqual(old['order'],current['order'])
    def test_none_seed_contract_not_arbitrarily_relaxed(self):
        for bad_seed in (0,42,True,42.0,'42'):
            current=self.sampler(25);current['seed']=bad_seed
            with self.assertRaises(ValueError):self.check_sampler(current,self.sampler(5),25)
            old=self.sampler(5);old['seed']=bad_seed
            with self.assertRaises(ValueError):self.check_sampler(self.sampler(25),old,25)
        with self.assertRaises(ValueError):n.validate_sampler(self.sampler(25),self.sampler(5),25,torch,{'seed':42,'shuffle':True})
    def test_shuffle_from_config_and_frozen_default_signature(self):
        settings=n.sampler_settings(state,{'data':{'shuffle_train_dataloader':False},'baselines':{'seed':42}})
        self.assertEqual(settings,{'seed':None,'shuffle':False})
        for bad in (0,1,None):
            with self.assertRaises(ValueError):n.sampler_settings(state,{'data':{'shuffle_train_dataloader':bad}})
        class WrongSampler:
            def __init__(self,data_source,seed=42,shuffle=True):pass
        with self.assertRaises(ValueError):n.sampler_settings(types.SimpleNamespace(StatefulSampler=WrongSampler),{'data':{'shuffle_train_dataloader':True}})
    def test_actual_signed5_and_observed25_small_driver_pt_sampler_join(self):
        originals=WORK/'e1-author200/actual3261-sampler-driver-small-originals'
        old_path=originals/'accepted5-driver.pt';current_path=originals/'current25-driver.pt'
        self.assertEqual(n.stable_sha(old_path),'5a8c2f6141a09cceee068e08ac8e8eed3033cca58fc05abf3139beaf487782c1')
        self.assertEqual(n.stable_sha(current_path),'dfbf01906077afbc6249ef2d4d57ecf665dcfa3f4f63c675d4d11a96608f9e22')
        inventory_path=WORK/'e1-author200/actual-snapshots/3261-20261005T221857135211Z/original-files/evidence/checkpoint-25-cpu-inspection-once-v1/observation/observed-checkpoint-inventory.json'
        inventory=n.strict_json(inventory_path.read_bytes());self.assertEqual(inventory['files']['driver/driver.pt']['sha256'],n.stable_sha(current_path))
        receipt_path=WORK/'e1-author200/actual3261-first5-signed-publication-once-v1/signed-envelope.json'
        self.assertEqual(n.stable_sha(receipt_path),n.ACCEPTED_RECEIPT);receipt=n.strict_json(receipt_path.read_bytes())
        self.assertEqual(receipt['payload']['checkpoint_files']['driver/driver.pt']['sha256'],n.stable_sha(old_path))
        old=torch.load(old_path,map_location='cpu',weights_only=False);current=torch.load(current_path,map_location='cpu',weights_only=False)
        self.assertEqual(state.state_digest(current),inventory['driver_report']['digest'])
        prior_path=WORK/'e1-author200/actual-snapshots/3261-20261005T210038278787Z/original-files/evidence/first5-cpu-inspection-once-v1/native.json'
        self.assertEqual(n.stable_sha(prior_path),receipt['payload']['evidence']['rank_audit']['sha256']);prior=n.strict_json(prior_path.read_bytes())
        self.assertEqual(state.state_digest(old),prior['driver']['state_digest'])
        config_path=WORK/'e1-author200/actual-snapshots/3261-20261005T212707464146Z/original-files/full_state/global_step_5/resolved-config.json'
        self.assertEqual(n.stable_sha(config_path),n.CONFIG);config=n.strict_json(config_path.read_bytes())
        self.assertIsNone(old['sampler']['seed']);self.assertIsNone(current['sampler']['seed'])
        self.assertEqual(current['sampler']['order'],old['sampler']['order']);self.assertTrue(torch.equal(current['sampler']['generator'],old['sampler']['generator']))
        old_validator=n.exact_module(WORK/'e1-author200/final-acceptance-8441e10fa456/inspect_author200_checkpoint_cpu.py','ca09cbca5084b1c1c8a1bf57906aca61e9805dd1176d6cfc954da5fb9981b471','old_rejected_actual_schema',{})
        with self.assertRaisesRegex(ValueError,'Sampler position/config differs: seed'):old_validator.validate_sampler(current['sampler'],old['sampler'],25,torch)
        result=n.validate_sampler(current['sampler'],old['sampler'],25,torch,n.sampler_settings(state,config))
        self.assertEqual(result['sampler_cursor'],12800);self.assertIsNone(result['seed']);self.assertTrue(result['unchanged_epoch0_generator'])
        self.assertFalse(torch.cuda.is_initialized())
    def test_algorithm_exact_pure_and_component_clip(self):
        config=dict(baselines=dict(distillation=dict(component_clip=.05,temperature=1,beta=0)))
        value=dict(version=1,method='pure_opsd',global_step=25,distillation=config['baselines']['distillation'],resume_scope='same_live_objects_same_topology',outcome_rewards_train_policy=False)
        n.validate_algorithm(value,config,25)
        for mutate in (lambda v:v.update(outcome_rewards_train_policy=True),lambda v:v['distillation'].update(component_clip=None),lambda v:v.update(global_step=25.0)):
            bad=copy.deepcopy(value);mutate(bad)
            with self.assertRaises(ValueError):n.validate_algorithm(bad,config,25)
    def test_strict_json_duplicate_constant_and_overflow(self):
        for bad in ('{"a":1,"a":2}','{"a":NaN}','{"a":Infinity}','{"a":1e999}'):
            with self.assertRaises(ValueError):n.strict_json(bad)
    def test_exact_source_no_pyc_and_wrong_pin(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory).resolve()/'helper.py';p.write_text('value=3\n');sha=n.stable_sha(p)
            self.assertEqual(n.exact_module(p,sha,'tiny_exact',{}).value,3)
            p.write_text('value=4\n')
            with self.assertRaises(ValueError):n.exact_module(p,sha,'tiny_exact',{})
    def test_source_symlink_cannot_be_helper(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory).resolve()/'helper.py';p.write_text('value=3\n');alias=p.parent/'alias.py';alias.symlink_to(p)
            with self.assertRaises(ValueError):n.exact_module(alias,n.stable_sha(p),'alias_exact',{})
    def test_json_parsed_bytes_not_second_unverified_path_read(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory).resolve()/'evidence.json';p.write_text('{"actual":true}')
            pins={};n.pinned(p,n.stable_sha(p),pins)
            self.assertEqual(n.read_pinned_json(p,pins),{'actual':True})
            p.write_text('{"actual":false}')
            with self.assertRaises(ValueError):n.read_pinned_json(p,pins)
    def test_exclusive_output_preserves_prior_original(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory).resolve()/'result.json';sha=n.write_exclusive(p,dict(fixture_only=True))
            self.assertEqual(sha,n.stable_sha(p))
            with self.assertRaises(FileExistsError):n.write_exclusive(p,dict(passed=True))
            self.assertEqual(n.strict_json(p.read_bytes()),dict(fixture_only=True))
    def test_postterminal_execution_job_separate_from_stored_training(self):
        procstat='100 (CPU inspect name) '+' '.join(['R']+['0']*18+['12345']+['0']*8)
        original=Path.read_text
        def content(p,*args,**kwargs):
            return {'/proc/self/stat':procstat,'/proc/self/status':f'Uid:\t{os.getuid()}\t{os.getuid()}\t{os.getuid()}\t{os.getuid()}\n','/proc/self/cgroup':'0::/slurm/job_9999/step_2\n'}.get(str(p),original(p,*args,**kwargs)) if not str(p).startswith('/proc/') else {'/proc/self/stat':procstat,'/proc/self/status':f'Uid:\t{os.getuid()}\t{os.getuid()}\t{os.getuid()}\t{os.getuid()}\n','/proc/self/cgroup':'0::/slurm/job_9999/step_2\n'}[str(p)]
        with patch.object(n.sys,'platform','linux'),patch.object(n.socket,'gethostname',return_value='h100-10'),patch.dict(os.environ,SLURM_JOB_ID='9999',SLURM_STEP_ID='2'),patch.object(Path,'read_text',content):
            identity=n.execution_identity('9999','2');self.assertEqual(identity['stored_training_job_id'],'3261');self.assertEqual(identity['execution_job_id'],'9999');self.assertEqual(identity['start_ticks'],12345)
    def test_execution_wrong_current_job_rejected_before_read(self):
        with patch.object(n.sys,'platform','linux'),patch.object(n.socket,'gethostname',return_value='h100-10'),patch.dict(os.environ,SLURM_JOB_ID='8888',SLURM_STEP_ID='2'):
            with self.assertRaises(ValueError):n.execution_identity('9999','2')
    def test_execution_forged_env_step_vs_actual_cgroup_rejected(self):
        procstat='100 (CPU inspect name) '+' '.join(['R']+['0']*18+['12345']+['0']*8)
        records={'/proc/self/stat':procstat,'/proc/self/status':f'Uid:\t{os.getuid()}\t{os.getuid()}\t{os.getuid()}\t{os.getuid()}\n','/proc/self/cgroup':'0::/slurm/job_9999/step_4\n'}
        with patch.object(n.sys,'platform','linux'),patch.object(n.socket,'gethostname',return_value='h100-10'),patch.dict(os.environ,SLURM_JOB_ID='9999',SLURM_STEP_ID='3'),patch.object(Path,'read_text',lambda p:records[str(p)]):
            with self.assertRaisesRegex(ValueError,'step cgroup'):n.execution_identity('9999','3')
        with patch.object(n.sys,'platform','linux'),patch.object(n.socket,'gethostname',return_value='h100-10'),patch.dict(os.environ,SLURM_JOB_ID='9999',SLURM_STEP_ID='../4'):
            with self.assertRaises(ValueError):n.execution_identity('9999','../4')


class StoredMath(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture=WORK/'e1-300/job3190-actual-evidence/fixture/baseline-fp32/roundtrip-step5/rank_0.pt'
        cls.raw=torch.load(fixture,map_location='cpu',weights_only=False)
    def optimizer(self,step):
        value=copy.deepcopy(self.raw['optimizer'])
        for row in value['state'].values():row['step'].fill_(16*step)
        for group in value['param_groups']:group['lr']=5e-6
        return value
    def test_real_stored_cuda_mesh_local_cpu_no_cuda(self):
        value=mathhelper.layout(self.raw['model']['0.weight'],0)
        self.assertEqual(value['mesh_ranks'],list(range(8)));self.assertEqual(value['local_shape'],[1,8]);self.assertFalse(torch.cuda.is_initialized())
    def test_layout_wrong_rank_and_flat_tensor(self):
        for tensor,rank in [(self.raw['model']['0.weight'],1),(self.raw['model']['0.weight'].to_local(),0)]:
            with self.assertRaises(ValueError):mathhelper.layout(tensor,rank)
    def test_all_saved_adam_clocks_with_full_moment_hashes(self):
        for step in n.STEPS:
            value=mathhelper.adam(self.optimizer(step),12,expected_steps=16*step);self.assertEqual(value['steps'],[16*step]);self.assertTrue(value['entries'][0]['exp_avg_sha256'])
    def test_old80_clock_cannot_pass_saved25(self):
        with self.assertRaises(ValueError):mathhelper.adam(self.optimizer(5),12,expected_steps=400)
    def test_adam_nonfinite_missing_and_wrong_local_geometry(self):
        for mutate in (lambda v:next(iter(v['state'].values()))['exp_avg'].__setitem__(0,float('nan')),
                       lambda v:next(iter(v['state'].values())).pop('exp_avg_sq'),
                       lambda v:v['param_groups'][0].update(params=[])):
            value=self.optimizer(25);mutate(value)
            with self.assertRaises(ValueError):mathhelper.adam(value,12,expected_steps=400)
        with self.assertRaises(ValueError):mathhelper.adam(self.optimizer(25),13,expected_steps=400)
    def teacher(self,step=25):
        config=dict(model_type='qwen2',hidden_size=8,intermediate_size=16,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,vocab_size=10,tie_word_embeddings=True)
        teacher=dict(version=1,parameter_count=n.NUMEL,storage_device='cpu',execution_device='cuda:0',model_content_digest='b'*64,
            asset_directory='/fixture-only/official',asset_files={'fixture':'a'*64},counters={k:1.0 for k in mathhelper.COUNTERS},model_config=config,
            caches={'rotary_emb':{'max_seq_len_cached':4096,'original_max_seq_len':32768}},buffers={'rotary_emb.inv_freq':torch.ones(2)},model_bytes=n.NUMEL*4+8)
        teacher['counters'].update(cuda_update_entries=step,scored_tokens=128);teacher['buffer_content_digest']=state.state_digest(teacher['buffers'])
        return teacher,types.SimpleNamespace(directory=Path('/fixture-only/official'),config=config)
    def test_teacher_whole_raw_projection_and_history_cache(self):
        teacher,base=self.teacher();report=mathhelper.teacher_report_projection({k:v for k,v in teacher.items() if k!='buffers'})
        value=n.teacher_check(teacher,report,base,teacher['asset_files'],'b'*64,25,mathhelper,state.state_digest);self.assertEqual(value['cuda_update_entries'],25)
        self.assertFalse(value['direct_saved_teacher_parameter_arrays'])
    def test_teacher_counter_clock_cache_bool_missing_and_unknown(self):
        for mutate in (lambda t:t['counters'].update(cuda_update_entries=5),lambda t:t['counters'].update(cuda_update_entries=25.0),
                       lambda t:t['caches']['rotary_emb'].update(max_seq_len_cached=True),lambda t:t['caches']['rotary_emb'].update(unknown=1),
                       lambda t:t.update(caches={'rotary_emb':{}})):
            teacher,base=self.teacher();mutate(teacher);report=mathhelper.teacher_report_projection({k:v for k,v in teacher.items() if k!='buffers'})
            with self.assertRaises(ValueError):n.teacher_check(teacher,report,base,teacher['asset_files'],'b'*64,25,mathhelper,state.state_digest)
    def test_teacher_buffer_bytes_and_nonfinite_fail(self):
        for bad in (2.0,float('inf')):
            teacher,base=self.teacher();teacher['buffers']['rotary_emb.inv_freq'][0]=bad;report=mathhelper.teacher_report_projection({k:v for k,v in teacher.items() if k!='buffers'})
            with self.assertRaises(ValueError):n.teacher_check(teacher,report,base,teacher['asset_files'],'b'*64,25,mathhelper,state.state_digest)
    def test_teacher_nested_report_type_not_python_alias(self):
        teacher,base=self.teacher();report=mathhelper.teacher_report_projection({k:v for k,v in teacher.items() if k!='buffers'});report['counters']['cuda_update_entries']=25.0
        with self.assertRaises(ValueError):n.teacher_check(teacher,report,base,teacher['asset_files'],'b'*64,25,mathhelper,state.state_digest)
    def test_rng_original_cpu_philox_no_device_restore(self):
        out=mathhelper.rng(self.raw['rng'],True);self.assertEqual(out['cuda'][0]['bytes'],16);self.assertFalse(torch.cuda.is_initialized())
    def test_rng_truncated_philox_and_nonfinite_numpy(self):
        for mutate in (lambda v:v['cuda'][0].update(state=torch.zeros(15,dtype=torch.uint8)),lambda v:v.update(numpy=(*v['numpy'][:4],float('inf')))):
            value=copy.deepcopy(self.raw['rng']);mutate(value)
            with self.assertRaises(ValueError):mathhelper.rng(value,True)


class WholeHF(unittest.TestCase):
    def native(self,inventory):
        layouts={str(i):dict(mesh_coordinate=[r],local_dtype='torch.float32',global_dtype='torch.float32') for r in () for i in ()}
        ranks=[]
        for r in range(8):
            ranks.append(dict(rank=r,model_dtensor_entries=435,model_hf_slice_exact=True,raw_saved_status='FULL_STATE_SERIALIZED_OK',
                pid=inventory['accepted5']['workers'][r]['pid'],object_ids=inventory['accepted5']['workers'][r]['object_ids'],optimizer_step_values=[3200],
                model_layouts={str(i):dict(mesh_coordinate=[r],local_dtype='torch.float32',global_dtype='torch.float32') for i in range(435)},model_hf_slice_keys=[str(i) for i in range(435)]))
        return dict(version=1,passed=True,scope=n.SCOPE,source_commit=n.SOURCE,config_sha256=n.CONFIG,job_id='3261',checkpoint_step=200,
            inventory_sha256='a'*64,run_root=n.RUN_ROOT,cuda_is_initialized=False,hf_all_rank_slices_exact=True,original_raw_statuses=inventory['raw_statuses'],ranks=ranks)
    def test_native_evidence_scope_and_actual200_binding(self):
        inv=Contracts().inventory(200);h.validate_native(self.native(inv),inv,'a'*64,n.require)
        for key,bad in [('checkpoint_step',25),('passed',1),('scope','EXACT_FULL_STATE_RELOAD_OK'),('inventory_sha256','b'*64),('cuda_is_initialized',True)]:
            value=self.native(inv);value[key]=bad
            with self.assertRaises(ValueError):h.validate_native(value,inv,'a'*64,n.require)
    def test_native_missing_rank435_or_saved_layout(self):
        inv=Contracts().inventory(200)
        for mutate in (lambda v:v['ranks'].pop(),lambda v:v['ranks'][5]['model_layouts'].pop('4'),lambda v:v['ranks'][4].update(optimizer_step_values=[80]),lambda v:v['ranks'][0]['model_layouts']['0'].update(mesh_coordinate=[1])):
            value=self.native(inv);mutate(value)
            with self.assertRaises(ValueError):h.validate_native(value,inv,'a'*64,n.require)
    def test_native_boolean_rank_float_adam_and_coordinate_aliases(self):
        inv=Contracts().inventory(200)
        for mutate in (lambda v:v['ranks'][0].update(rank=False),lambda v:v['ranks'][1].update(optimizer_step_values=[3200.0]),lambda v:v['ranks'][0]['model_layouts']['0'].update(mesh_coordinate=[False])):
            value=self.native(inv);mutate(value)
            with self.assertRaises(ValueError):h.validate_native(value,inv,'a'*64,n.require)
    def test_hf_exact_inspector_source_pin_and_symlink(self):
        sha=n.stable_sha(HERE/'inspect_author200_checkpoint_cpu.py')
        self.assertEqual(h.load_inspector(HERE/'inspect_author200_checkpoint_cpu.py',sha).SOURCE,n.SOURCE)
        with self.assertRaises(ValueError):h.load_inspector(HERE/'inspect_author200_checkpoint_cpu.py','0'*64)
    def test_loader_missing_loading_keys_and_tokenizer_semantics(self):
        oldhf.empty_loading(dict(missing_keys=[],unexpected_keys=[],mismatched_keys=[],error_msgs=[]))
        with self.assertRaises(ValueError):oldhf.empty_loading(dict(missing_keys=['one'],unexpected_keys=[],mismatched_keys=[],error_msgs=[]))
    def tiny_model_pair(self,directory):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import Qwen2Config,Qwen2ForCausalLM,PreTrainedTokenizerFast
        from safetensors import safe_open
        base=Path(directory).resolve()/'base';hf=base.parent/'hf'
        config=Qwen2Config(vocab_size=16,hidden_size=8,intermediate_size=16,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,
                          max_position_embeddings=128,tie_word_embeddings=True,bos_token_id=1,eos_token_id=2,pad_token_id=0)
        tok=Tokenizer(WordLevel({'<pad>':0,'<s>':1,'</s>':2,'<unk>':3,'Return':4,'the':5,'word':6,'test':7,'.':8},unk_token='<unk>'))
        tok.pre_tokenizer=Whitespace()
        tokenizer=PreTrainedTokenizerFast(tokenizer_object=tok,bos_token='<s>',eos_token='</s>',pad_token='<pad>',unk_token='<unk>',chat_template="{{ messages[0]['content'] }}")
        model=Qwen2ForCausalLM(config);count=sum(v.numel() for v in model.parameters())
        model.save_pretrained(base,safe_serialization=True);tokenizer.save_pretrained(base)
        with torch.no_grad():model.model.embed_tokens.weight[4,0]+=0.125
        model.save_pretrained(hf,safe_serialization=True);tokenizer.save_pretrained(hf)
        for path in (base,hf):
            with safe_open(str(path/'model.safetensors'),framework='pt',device='cpu') as stream:keys=list(stream.keys())
            (path/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{key:'model.safetensors' for key in keys}}))
        return base,hf,count
    def tiny_load(self,base,hf,count):
        sys.path.insert(0,str(WORK/'opsd-e1-300'))
        try:
            with patch.dict(sys.modules,{'search_r1.baselines.full_state':state}):return oldhf.load_compare(base,hf,expected_count=count)
        finally:sys.path.pop(0)
    def test_actual_small_qwen_whole_cpu_load_values_tie_and_forward(self):
        with tempfile.TemporaryDirectory() as directory:
            base,hf,count=self.tiny_model_pair(directory);result=self.tiny_load(base,hf,count)
            self.assertEqual(result['actual_unique_parameters'],count);self.assertTrue(result['tied_embeddings'])
            self.assertTrue(result['all_loaded_parameters_finite']);self.assertTrue(result['actual_cpu_forward_finite']);self.assertGreater(result['total_changed_elements'],0)
            self.assertFalse(torch.cuda.is_initialized())
    def test_actual_small_qwen_whole_loader_nonfinite_and_unchanged_rejected(self):
        from safetensors.torch import load_file,save_file
        with tempfile.TemporaryDirectory() as directory:
            base,hf,count=self.tiny_model_pair(directory)
            values=load_file(str(hf/'model.safetensors'));next(iter(values.values())).reshape(-1)[0]=float('nan');save_file(values,str(hf/'model.safetensors'),metadata={'format':'pt'})
            with self.assertRaises(ValueError):self.tiny_load(base,hf,count)
        with tempfile.TemporaryDirectory() as directory:
            base,hf,count=self.tiny_model_pair(directory);(hf/'model.safetensors').write_bytes((base/'model.safetensors').read_bytes())
            with self.assertRaises(ValueError):self.tiny_load(base,hf,count)


if __name__=='__main__':unittest.main(verbosity=2)
