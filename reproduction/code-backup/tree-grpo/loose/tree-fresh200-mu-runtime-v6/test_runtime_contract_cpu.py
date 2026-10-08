"""Tiny synthetic wrapper contract tests; no assets/GPU/full DataProto proof."""
import copy
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import calibration_contract as contract
from render_dispatch import render
from tokenizer_raw_preflight import inspect_row

HERE=Path(__file__).resolve().parent


def launch_heredoc():
    text=(HERE/'launch.sbatch').read_text()
    return text.split("<<'PY'\n",1)[1].split('\nPY\n',1)[0]


def candidate_owned_class():
    tree=ast.parse((HERE/'supervisor.py').read_bytes())
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CandidateOwnedProcesses')
    class ToyOwned:
        def __init__(self):self.calls=[]
        def spawn(self,command,log,env=None,pass_fds=()):
            result=dict(command=command,log=log,environment=env,pass_fds=pass_fds)
            self.calls.append(result);return result
    def require(value,message):
        if not value:raise RuntimeError(message)
    scope=dict(OwnedProcesses=ToyOwned,sys=sys,os=os,require=require)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'actual-v6-CandidateOwnedProcesses','exec'),scope)
    return scope['CandidateOwnedProcesses'],scope


# Exact fixed external guard source executes below, with every NVML/proc/cancel
# dependency replaced by pure CPU stubs. This never imports a hardware backend.
GUARD_STUB_HARNESS=r'''
import hashlib,json,os,pathlib,subprocess,sys,time,types
source_path=pathlib.Path(sys.argv[1]);output=pathlib.Path(sys.argv[2]);raw=source_path.read_bytes()
if hashlib.sha256(raw).hexdigest()!='284c014cd6e273816c112ea8a7b515f3afa4e55bef5c74d23f319f93f146ff88':
 raise ValueError('External guard source changed')
uuids=['GPU-CPU-STUB-'+str(i) for i in range(8)]
read_text=pathlib.Path.read_text
def synthetic_read(path,*args,**kwargs):
 text=str(path)
 if text=='/proc/self/cgroup':return '0::/slurm/job_9001/step_batch\n'
 if text.endswith('/tools/expected-gpu-uuids.json'):return json.dumps(uuids)
 if text=='/proc/999999/stat':return '999999 (foreign cpu stub) '+' '.join(['S']+['0']*18+['1'])
 if text=='/proc/999999/cgroup':return '0::/other/job_4000\n'
 if text.startswith('/proc/'):raise RuntimeError('Actual proc access prohibited: '+text)
 return read_text(path,*args,**kwargs)
pathlib.Path.read_text=synthetic_read
nv=types.ModuleType('pynvml');nv.NVMLError_NotFound=type('NVMLError_NotFound',(Exception,),{})
nv.nvmlInit=lambda:None;nv.nvmlDeviceGetCount=lambda:8;nv.nvmlDeviceGetHandleByIndex=lambda i:i
nv.nvmlDeviceGetUUID=lambda i:uuids[i]
nv.nvmlDeviceGetMemoryInfo=lambda h:types.SimpleNamespace(total=100,used=80)
nv.nvmlDeviceGetUtilizationRates=lambda h:types.SimpleNamespace(gpu=90)
nv.nvmlDeviceGetComputeRunningProcesses=lambda h:[types.SimpleNamespace(pid=999999,usedGpuMemory=50)]
nv.nvmlDeviceGetGraphicsRunningProcesses=lambda h:[]
nv.nvmlDeviceGetProcessUtilization=lambda h,t:[types.SimpleNamespace(pid=999999,smUtil=90,timeStamp=int(time.time()*1e6)+1)]
sys.modules['pynvml']=nv
calls=[]
def mock_cancel(command,**kwargs):calls.append(command);return types.SimpleNamespace(returncode=0)
subprocess.run=mock_cancel
def no_sleep(seconds):raise RuntimeError('CPU_PROBE_STOP_AFTER_COMPLETE_SAMPLE')
time.sleep=no_sleep
sys.argv=[str(source_path),'--output',str(output)]
terminal=None
try:exec(compile(raw,str(source_path),'exec'),{'__name__':'__main__','__file__':str(source_path)})
except BaseException as error:terminal=repr(error)
records=[json.loads(line) for line in output.read_text().splitlines()] if output.exists() else []
print(json.dumps(dict(optimization=sys.flags.optimize,pythonoptimize=os.environ.get('PYTHONOPTIMIZE'),
 source_sha256=hashlib.sha256(raw).hexdigest(),terminal=terminal,records=records,
 healthy_samples=sum('gpus' in r and not r.get('event') for r in records),
 scancel_calls_stubbed=calls,actual_scancel_calls=0,actual_nvml_calls=0,actual_proc_calls=0),allow_nan=False))
'''


class FreshMetadataContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.run=Path(self.tmp.name)/'tree-fresh200-9001';(self.run/'output').mkdir(parents=True)
        self.package=dict(code_commit='a'*40,common_data_identity='b'*64,
            official_model_path='/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/assets/qwen2.5-3b-instruct-aa8e725',
            model_identity={'sha256':'c'*64})
        self.expected=contract.expected_runtime(self.run,self.package,'d'*64)
        workers=[dict(self.expected,rank=rank,world_size=8,
            training_protocol=contract.PROFILE,calibration_profile='fresh_base_zero_v1',
            optimizer_state_entries=0,optimizer_step_values=[],scheduler_step=0,
            optimizer_committed=0,scheduler_advanced=False,checkpoint_recovery_environment_absent=True,
            configured_model_path=self.package['official_model_path'],expected_model_path=self.package['official_model_path'],
            complete_model_parameter_bytes_verified=False) for rank in range(8)]
        self.record=dict(self.expected,status='FRESH_OFFICIAL_INITIAL_STATE_OBSERVED',training_protocol=contract.PROFILE,
            calibration_profile='fresh_base_zero_v1',initialized_global_steps=0,parent_completed_step=0,parent_cursor=0,
            driver_completed_outer=0,checkpoint_loaded=False,workers=workers)

    def validate(self, record=None):
        path=self.run/'output/fresh-initialization.json'
        path.write_text(json.dumps(self.record if record is None else record))
        return contract.validate_fresh_start(self.run,self.package,'d'*64)

    def test_all_eight_direct_fresh_metadata(self):
        self.assertEqual(self.validate()['workers'],8)

    def test_old_model_path_is_not_fresh(self):
        record=copy.deepcopy(self.record);record['workers'][3]['configured_model_path']='/old/HF110'
        with self.assertRaisesRegex(ValueError,'official'):self.validate(record)

    def test_nonempty_adam_refuses_even_with_empty_step_claim(self):
        record=copy.deepcopy(self.record);record['workers'][5]['optimizer_state_entries']=1
        with self.assertRaisesRegex(ValueError,'Adam'):self.validate(record)

    def test_scheduler_and_rank_coverage_refuse(self):
        record=copy.deepcopy(self.record);record['workers'][7]['scheduler_step']=110
        with self.assertRaisesRegex(ValueError,'Adam'):self.validate(record)
        record=copy.deepcopy(self.record);record['workers'].pop()
        with self.assertRaisesRegex(ValueError,'eight'):self.validate(record)

    def test_real_receipt_and_collector_env_names_do_not_alias(self):
        mapped=contract.expected_mu_provenance(self.expected)
        self.assertEqual(set(mapped),{'SEARCH_R1_BASELINE_COMMIT','SEARCH_R1_BASELINE_CONFIG_SHA256',
            'SEARCH_R1_BASELINE_DATA_IDENTITY','SEARCH_R1_MODEL_IDENTITY','SEARCH_R1_BASELINE_RUN_ID','SLURM_JOB_ID'})
        self.assertEqual(mapped['SLURM_JOB_ID'],'9001');self.assertNotIn('job_id',mapped)

    def test_missing_checkpoint_evidence_is_not_default_false(self):
        record=copy.deepcopy(self.record);del record['checkpoint_loaded']
        with self.assertRaises(KeyError):self.validate(record)

    def test_recovery_env_refuses(self):
        contract.forbid_recovery_environment({'TREE_FRESH_INITIALIZATION_ONLY':'1'})
        with self.assertRaises(ValueError):contract.forbid_recovery_environment({'TREE_RECOVERY_MANIFEST':'/old'})

    def test_renderer_has_no_submit_and_rejects_non_task_path(self):
        text=render('/dataset/shaoshitong/search-r1-rl/tree-grpo/candidates/fresh-test','a'*64,'b'*64)
        self.assertIn('#SBATCH --gres=gpu:8',text);self.assertIn('h100-9',text)
        self.assertNotIn('sbatch ',text);self.assertNotIn('TREE_RECOVERY_',text)
        with self.assertRaises(ValueError):render('/other-cluster','a'*64,'b'*64)


class ActualReadyBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.expected=dict(code_commit='a'*40,config_sha256='b'*64,data_identity='c'*64,
                           model_identity='d'*64,job_id='9001',run_id='tree-fresh200-9001')
        # Exact source producer keys: no invented training_protocol/fresh flag,
        # top-level scheduler, run_id or actual_completed_outer.
        self.ready=dict({k:v for k,v in self.expected.items() if k!='run_id'},
            status='FRESH_BEHAVIOR_MU_CALIBRATION_OBSERVED_ZERO_WRITES',calibration_profile='fresh_base_zero_v1',
            initialized_global_steps=0,behavior_mu_corrected_profile=True,
            original_sampler_actor_probability_gate_accepted=False,parent_completed_outer=0,
            observed_outer=1,new_completed_outer=0,optimizer_committed=0,scheduler_advanced=False,
            sampler_cursor_before_outer=0,sampler_cursor_after_observation=512)
        self.worker=dict(self.expected,rank=2,world_size=8,calibration_profile='fresh_base_zero_v1',
            parent_completed_outer=0,observed_outer=1,optimizer_state_entries=0,restored_optimizer_steps=[],
            optimizer_committed=0,scheduler_step=0,scheduler_advanced=False,
            fresh_post_monitor_state=dict(calibration_profile='fresh_base_zero_v1',restored_optimizer_steps=[],
                                         optimizer_state_entries=0,scheduler_step=0))

    def test_actual_ready_fields_accept_without_fabricated_fields(self):
        contract.validate_ready_boundary(self.ready,self.expected)
        contract.validate_calibration_rank_state(self.worker,2,self.expected)

    def test_old_or_wrong_profile_ready_does_not_alias_fresh(self):
        ready=copy.deepcopy(self.ready);ready['status']='BEHAVIOR_MU_CALIBRATION_OBSERVED_ZERO_WRITES'
        with self.assertRaisesRegex(ValueError,'status'):contract.validate_ready_boundary(ready,self.expected)
        ready=copy.deepcopy(self.ready);ready['sampler_cursor_after_observation']=640
        with self.assertRaisesRegex(ValueError,'cursor'):contract.validate_ready_boundary(ready,self.expected)

    def test_post_monitor_state_writes_or_wrong_source_refuse(self):
        worker=copy.deepcopy(self.worker);worker['fresh_post_monitor_state']['optimizer_state_entries']=1
        with self.assertRaisesRegex(ValueError,'changed actual'):contract.validate_calibration_rank_state(worker,2,self.expected)
        worker=copy.deepcopy(self.worker);worker['run_id']='old-run'
        with self.assertRaisesRegex(ValueError,'identity'):contract.validate_calibration_rank_state(worker,2,self.expected)


class HistoricalBoundaryOnlyTests(unittest.TestCase):
    def setUp(self):
        source=Path(__file__).resolve().parent.parent/'tree-behavior-mu-ppo/search_r1/baselines/tree_raw_tokens.py'
        spec=importlib.util.spec_from_file_location('original_raw_helper_fixture',source)
        self.helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.helper)
        self.record=dict(token_ids=[1,2],prompt_token_ids=[9],input_row=0,request_id='original-request',
            true_response_length=2,logprobs=[{'1':{'logprob':-.5}},{'2':{'logprob':-.25}}],
            finished=True,completion_index=0,finish_reason='length',stop_reason=None)
        self.params=dict(n=1,temperature=1.0,top_p=1.0,top_k=-1,min_p=0.0,presence_penalty=0.0,
            frequency_penalty=0.0,repetition_penalty=1.0,min_tokens=0,max_tokens=500,ignore_eos=False,
            stop=[],stop_token_ids=[],include_stop_str_in_output=False,logprobs=1,prompt_logprobs=None,seed=None)
        self.identity={'pad_token_id':151643,'eos_token_id':151645}

    def test_missing_historical_best_of_and_processors_stay_unknown_without_default(self):
        class Tokenizer:
            pad_token_id=151643;eos_token_id=151645
            def decode(self,ids,**kwargs):return {(): '',(1,):'a',(1,2):'ab'}[tuple(ids)]
            def encode(self,text,**kwargs):return [1,2]
        result=inspect_row(self.helper,Tokenizer(),self.identity,{'sha256':'a'*64},self.record,self.params)
        self.assertFalse(result['actual_native24_complete'])
        self.assertEqual(result['missing_actual_sampling_fields'],['best_of','_real_n','logits_processors',
            'truncate_prompt_tokens','guided_decoding','logit_bias','allowed_token_ids'])
        self.assertEqual(result['actual_best_of_status'],'UNKNOWN_NOT_RECORDED')
        self.assertTrue(all(value=='UNKNOWN_NOT_RECORDED' for value in result['actual_control_fields_status'].values()))
        self.assertFalse(result['actual_sampling_support_validated'])
        self.assertEqual(result['retained_tokens'],2)

    def test_recorded_historical_best_of_is_observed_without_repairing_distribution(self):
        class Tokenizer:
            pad_token_id=151643;eos_token_id=151645
            def decode(self,ids,**kwargs):return {(): '',(1,):'a',(1,2):'ab'}[tuple(ids)]
            def encode(self,text,**kwargs):return [1,2]
        for value, supported in ((1,True),(None,False),(2,False),(1.,False),(True,False)):
            actual=dict(self.params,best_of=value,_real_n=1,logits_processors=None,
                truncate_prompt_tokens=None,guided_decoding=None,logit_bias=None,allowed_token_ids=None)
            result=inspect_row(self.helper,Tokenizer(),self.identity,{'sha256':'a'*64},self.record,actual)
            self.assertTrue(result['actual_native24_complete'])
            self.assertEqual(result['actual_best_of_status'],'RECORDED')
            self.assertEqual(result['actual_sampling_support_validated'],supported)
            self.assertEqual(result['retained_tokens'],2)

    def test_historical_new_controls_missing_or_nontrivial_stay_observations(self):
        class Tokenizer:
            pad_token_id=151643;eos_token_id=151645
            def decode(self,ids,**kwargs):return {(): '',(1,):'a',(1,2):'ab'}[tuple(ids)]
            def encode(self,text,**kwargs):return [1,2]
        base=dict(self.params,best_of=1,_real_n=1,logits_processors=None,
            truncate_prompt_tokens=None,guided_decoding=None,logit_bias=None,allowed_token_ids=None)
        for key,value in (('_real_n',2),('truncate_prompt_tokens',1),('guided_decoding',{'json':{}}),
                          ('logit_bias',{'151645':1.}),('allowed_token_ids',[151645])):
            for missing in (False,True):
                actual=dict(base)
                if missing:del actual[key]
                else:actual[key]=value
                result=inspect_row(self.helper,Tokenizer(),self.identity,{'sha256':'a'*64},self.record,actual)
                self.assertFalse(result['actual_sampling_support_validated'])
                self.assertEqual(result['actual_control_fields_status'][key],
                                 'UNKNOWN_NOT_RECORDED' if missing else 'RECORDED')
                self.assertEqual(result['retained_tokens'],2)

    def test_unicode_intermediate_refusal_survives_metadata_gap(self):
        class Tokenizer:
            pad_token_id=151643;eos_token_id=151645
            def decode(self,ids,**kwargs):return '\ufffd' if len(ids)==1 else '中'
            def encode(self,text,**kwargs):return [1,2]
        with self.assertRaisesRegex(ValueError,'UTF8'):
            inspect_row(self.helper,Tokenizer(),self.identity,{'sha256':'a'*64},self.record,self.params)


class OptimizationBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)

    def launch_fixture(self,change_manifest=None,change_receipt=None,wrong_digest=None):
        payload=b'# synthetic wrapper bytes, no execution\n';(self.root/'payload.py').write_bytes(payload)
        manifest=dict(stage='fresh_mu_calibration',training_protocol='tree_fresh200_five_leaf_v1',code_commit='a'*40,
            scope='FROZEN_PENDING_INDEPENDENT_FRESH200_MU_PRE_DISPATCH_REVIEW',production_thresholds_accepted=False,
            probability_calibration=dict(environment={'TREE_FRESH_INITIALIZATION_ONLY':'1','TREE_PPO_CALIBRATION_ONLY':'1'}),
            files={'payload.py':hashlib.sha256(payload).hexdigest()})
        if change_manifest:change_manifest(manifest)
        raw=json.dumps(manifest).encode();(self.root/'manifest.json').write_bytes(raw);msha=hashlib.sha256(raw).hexdigest()
        receipt=dict(status='READY_FOR_FRESH200_BEHAVIOR_MU_ZERO_WRITE_CALIBRATION',manifest_sha256=msha,code_commit='a'*40,
            actual_3b_training_accepted=False,long_training_ready=False,requires_zero_optimizer_writes=True,
            requires_actual_behavior_mu_binding=True,original_sampler_actor_gate_accepted=False)
        if change_receipt:change_receipt(receipt)
        rr=json.dumps(receipt).encode();(self.root/'PRE_DISPATCH_RECEIPT.json').write_bytes(rr)
        env=dict(os.environ,TREE_INSTRUCT_TOOLS=str(self.root),TREE_INSTRUCT_PACKAGE_SHA256=msha,
            TREE_INSTRUCT_REVIEW_SHA256=hashlib.sha256(rr).hexdigest(),CUDA_VISIBLE_DEVICES='')
        if wrong_digest:env[wrong_digest]='0'*64
        return env

    def run_heredoc(self,env,opt):
        return subprocess.run([sys.executable,'-c',launch_heredoc()],env=dict(env,PYTHONOPTIMIZE=str(opt)),
            capture_output=True,text=True,timeout=10)

    def test_actual_launch_heredoc_valid_only_unoptimized(self):
        env=self.launch_fixture()
        self.assertEqual(self.run_heredoc(env,0).returncode,0)
        for opt in (1,2):
            result=self.run_heredoc(env,opt)
            self.assertNotEqual(result.returncode,0);self.assertIn('unoptimized',result.stderr)

    def test_actual_launch_bad_release_status_stage_hash_scope_always_refuse(self):
        cases=(('receipt_status',None,lambda r:r.update(status='UNREVIEWED'),None),
            ('stage',lambda m:m.update(stage='fresh_mu_train200'),None,None),
            ('manifest_sha',None,None,'TREE_INSTRUCT_PACKAGE_SHA256'),
            ('receipt_sha',None,None,'TREE_INSTRUCT_REVIEW_SHA256'),
            ('partial_scope',None,lambda r:r.update(requires_zero_optimizer_writes=False),None),
            ('training_claim',None,lambda r:r.update(actual_3b_training_accepted=True),None),
            ('unknown_mu',None,lambda r:r.pop('requires_actual_behavior_mu_binding'),None),
            ('wrong_protocol',lambda m:m.update(training_protocol='old'),None,None),
            ('scope',lambda m:m.update(scope='LOCAL_DRAFT_NOT_RELEASED'),None,None),
            ('changed_package',lambda m:m['files'].update({'payload.py':'0'*64}),None,None))
        for name,m,r,digest in cases:
            env=self.launch_fixture(m,r,digest)
            for opt in (0,1):
                with self.subTest(case=name,opt=opt):self.assertNotEqual(self.run_heredoc(env,opt).returncode,0)

    def test_launch_static_explicit_gate_and_export_zero_before_interpreter(self):
        text=(HERE/'launch.sbatch').read_text();tree=ast.parse(launch_heredoc())
        self.assertFalse(any(isinstance(n,ast.Assert) for n in ast.walk(tree)))
        self.assertLess(text.index('export PYTHONOPTIMIZE=0'),text.index("/env/bin/python - <<'PY'"))
        self.assertIn("require(sys.flags.optimize==0",launch_heredoc())

    def test_actual_supervisor_prefix_rejects_opt1_before_external_import_and_sets_env0(self):
        tree=ast.parse((HERE/'supervisor.py').read_bytes())
        external=next(i for i,node in enumerate(tree.body) if isinstance(node,ast.ImportFrom)
                      and node.module in ('calibration_contract','tree_shared_owned_runtime_fixed'))
        prefix=ast.unparse(ast.Module(body=tree.body[:external],type_ignores=[]))
        code="import os;os.environ['PYTHONOPTIMIZE']='1'\n"+prefix+"\nprint(json.dumps({'optimization':sys.flags.optimize,'environment':os.environ['PYTHONOPTIMIZE']}))"
        for opt in (0,1):
            result=subprocess.run([sys.executable,'-c',code],env=dict(os.environ,PYTHONOPTIMIZE=str(opt)),
                text=True,capture_output=True,timeout=10)
            if opt:
                self.assertNotEqual(result.returncode,0);self.assertIn('before any runtime import',result.stderr)
            else:
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertEqual(json.loads(result.stdout),dict(optimization=0,environment='0'))

    def test_actual_owned_class_forces_both_child_envs_without_mutating_caller(self):
        klass,scope=candidate_owned_class();instance=klass()
        with patch.dict(os.environ,{'PYTHONOPTIMIZE':'1','SEARCH_R1_REPO':'/synthetic/source'}):
            inherited=instance.spawn(['python','synthetic-only'],'log',None,pass_fds=(77,))
        explicit={'PYTHONOPTIMIZE':'1','SEARCH_R1_REPO':'/synthetic/explicit','EXPLICIT_TEST':'preserved'}
        original=dict(explicit);result=instance.spawn(['python','synthetic-only'],'log',explicit,pass_fds=(78,))
        self.assertEqual(explicit,original);self.assertIsNot(result['environment'],explicit)
        self.assertEqual(result['environment']['EXPLICIT_TEST'],'preserved')
        for record in (inherited,result):
            self.assertEqual(record['environment']['PYTHONOPTIMIZE'],'0')
            self.assertEqual(record['command'][:5],['bash','-c','cd "$SEARCH_R1_REPO" && exec "$@"','candidate-cwd','python'])
        self.assertEqual(inherited['pass_fds'],(77,));self.assertEqual(result['pass_fds'],(78,))
        scope['sys']=SimpleNamespace(flags=SimpleNamespace(optimize=1))
        with self.assertRaisesRegex(RuntimeError,'unoptimized parent'):instance.spawn(['python'],'log')
        self.assertEqual(len(instance.calls),2)

    def test_actual_external_guard_with_captured_child_env_still_refuses_dual15_stub(self):
        klass,_=candidate_owned_class();owned=klass()
        explicit={'PYTHONOPTIMIZE':'1','SEARCH_R1_REPO':'/synthetic/explicit'}
        with patch.dict(os.environ,{'PYTHONOPTIMIZE':'1','SEARCH_R1_REPO':'/synthetic/inherited'}):
            environments=[owned.spawn(['python'],'log',None)['environment'],owned.spawn(['python'],'log',explicit)['environment']]
        results=[]
        for i,child_env in enumerate(environments):
            actual_env=dict(child_env,SLURM_JOB_ID='9001',SLURM_JOB_NODELIST='h100-9',CUDA_VISIBLE_DEVICES='')
            guard=HERE.parent/'resource_guard_identity.py'
            completed=subprocess.run([sys.executable,'-c',GUARD_STUB_HARNESS,str(guard),str(self.root/f'guard-{i}.jsonl')],
                env=actual_env,capture_output=True,text=True,timeout=10)
            self.assertEqual(completed.returncode,0,completed.stderr);result=json.loads(completed.stdout)
            self.assertEqual(result['optimization'],0);self.assertEqual(result['pythonoptimize'],'0')
            self.assertEqual(result['healthy_samples'],0);self.assertIn('dual15%',result['terminal'])
            self.assertIn('0.5',result['terminal']);self.assertIn('90',result['terminal'])
            self.assertEqual(result['actual_scancel_calls'],0);self.assertEqual(result['actual_nvml_calls'],0)
            self.assertEqual(result['actual_proc_calls'],0);self.assertEqual(result['scancel_calls_stubbed'],[['scancel','9001']])
            results.append(dict(env_kind='inherited' if i==0 else 'explicit',**result))
        proof=dict(scope='Actual fixed guard bytes compiled in actual spawned CPU subprocess with captured V6 child env; all hardware/proc/cancel dependencies stubbed; not full launch or hardware acceptance',
            guard_sha256='284c014cd6e273816c112ea8a7b515f3afa4e55bef5c74d23f319f93f146ff88',results=results,
            external_guard_changed=False,external_guard_other_entrypoints_repaired=False)
        path=HERE/'V6_RESOURCE_CHILD_ENV_CPU_PROOF.json'
        if not path.exists():path.write_text(json.dumps(proof,indent=2,sort_keys=True,allow_nan=False)+'\n')


if __name__=='__main__': unittest.main(verbosity=2)
