"""CPU-only evidence-schema contracts; these fixtures are not actual GPU proof."""
import ast,copy,pathlib
import pytest
from source_binding import SOURCE_SHA,WORKER_SHA,check_device_correctness,validate_device_results
HERE=pathlib.Path(__file__).resolve().parent

def records():
 parent={'job':'99999','status':'DEVICE_PARENT_AND_ORIGINAL_MATH_PASS_NOT_CAPACITY_NOT_TRAINING','source_manifest_sha256':SOURCE_SHA,'worker_source_sha256':WORKER_SHA,'leased_uuids':['GPU-FIXTURE-'+str(i) for i in range(8)],'guardian_errors':[],'cleanup':{'actual_cgroup_empty':True,'remaining':[]},'parent_cuda_initialized':False,'baseline_preserved':True,'worker_identity':{'pid':999}}
 worker={'job':'99999','pid':999,'status':'DEVICE_AND_ORIGINAL_MATH_PASS_NOT_CAPACITY_NOT_TRAINING','source_manifest_sha256':SOURCE_SHA,'forced_wrapper':False,'steps':3,'all_parameter_and_state_equal':True,'allocator_after_release':{'allocated':0,'reserved':0},'all_parameters_and_states_cpu':True,'namespace_restored':True,'hooks':['pre','post']*3,'initializer_metadata_after_compile_context':{'wrapped':True,'dynamo_disabled':True,'orig_callable_is_body':True},'production_entropy_compile_context':{'dynamic':True,'invoked':False},'initializer_events':[{'input_device':'cuda:0','output_device':'cpu','dtype':'torch.float32'} for _ in range(4)]}
 terminal={'job_id':99999,'returncode':0,'stdout':'JobId=99999 JobState=COMPLETED ExitCode=0:0'}
 return parent,worker,terminal

def test_pending_device_proof_and_actual_runtime_callers():
 for c in ({},{'source_device_correctness':None}):
  with pytest.raises(ValueError,match='pending'):check_device_correctness(c)
 for name in ('supervise.py','train_entry.py'):
  tree=ast.parse((HERE/name).read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
  assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='check_device_correctness' for n in ast.walk(main))

def test_schema_rejects_wrong_source_forced_wrapper_release_or_terminal():
 p,w,t=records();assert validate_device_results(p,w,t)==99999
 for mutate in ('source','forced','release','cpu_events','terminal','guardian','lease'):
  parent,worker,terminal=records()
  if mutate=='source':worker['source_manifest_sha256']='old-source'
  if mutate=='forced':worker['forced_wrapper']=True
  if mutate=='release':worker['allocator_after_release']['allocated']=32*1024*1024
  if mutate=='cpu_events':worker['initializer_events'][0]['output_device']='cuda:0'
  if mutate=='terminal':terminal['stdout']='JobId=99998 JobState=COMPLETED ExitCode=0:0'
  if mutate=='guardian':parent['guardian_errors']=['failed fullfresh']
  if mutate=='lease':parent['leased_uuids']=parent['leased_uuids'][:7]
  with pytest.raises(ValueError):validate_device_results(parent,worker,terminal)
