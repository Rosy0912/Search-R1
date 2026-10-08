"""CPU schema/caller boundaries; no fixture is an actual CUDA approval."""
import ast
from pathlib import Path
import pytest
from source_binding import check_entropy_correctness,check_entropy_native,validate_native_result,INITIALIZER_SOURCE_SHA,SOURCE_SHA
HERE=Path(__file__).resolve().parent

def test_new_entropy_none_rejected_before_actual_launch():
 for c in ({},{'source_entropy_native_cpu_evidence':None,'source_entropy_correctness':None}):
  with pytest.raises(ValueError,match='pending'):check_entropy_correctness(c)
 for name in ('supervise.py','train_entry.py'):
  tree=ast.parse((HERE/name).read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
  assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='check_entropy_correctness' for n in ast.walk(main))

def test_inherited_optimizer_proof_cannot_be_relabelled_as_new_source():
 base={'job':'3200','tests':26,'skips':0,'failures':0,'exit_code':0,'cuda_initialized':False,'source_manifest_sha256':INITIALIZER_SOURCE_SHA}
 assert validate_native_result(base)==base
 assert SOURCE_SHA!=INITIALIZER_SOURCE_SHA
 with pytest.raises(ValueError):validate_native_result(dict(base,source_manifest_sha256=SOURCE_SHA))

def schema_records():
 from source_binding import ENTROPY_OPTIONS,ENTROPY_FILE_SHA,ENTROPY_ACTOR_SHA
 p={'job':'99999','status':'ENTROPY_PARENT_AND_ORIGINAL_MATH_PASS_NOT_CAPACITY_NOT_TRAINING','source_manifest_sha256':SOURCE_SHA,'leased_uuids':['CPU-SCHEMA-ONLY-'+str(i) for i in range(8)],'guardian_errors':[],'cleanup':{'actual_cgroup_empty':True,'remaining':[]},'parent_cuda_initialized':False,'baseline_preserved':True,'worker_identity':{'pid':999}}
 specs=[(4,17,'float32'),(5,17,'bfloat16'),(2,152064,'float32'),(3,152064,'bfloat16'),(6144,152064,'float32'),(6143,152064,'float32'),(6144,152064,'float32'),(6144,152064,'bfloat16'),(6143,152064,'bfloat16'),(6144,152064,'bfloat16')];cases=[]
 for length,vocab,dtype in specs:
  c={'length':length,'vocab':vocab,'input_dtype':dtype}
  for name in ('candidate_vs_original_compiled_value','candidate_vs_original_compiled_gradient','candidate_vs_eager_value','candidate_vs_eager_gradient'):
   compiled='original_compiled' in name;grad=name.endswith('gradient')
   atol,rtol=((1e-7,1e-4) if grad else (1e-4,1e-5)) if compiled or dtype=='float32' else ((.0001,.032) if grad else (.125,.016))
   c[name]={'all_elements_passed':True,'elements':length*vocab if grad else length,'max_absolute_delta':0.,'max_relative_delta':0.,'atol':atol,'rtol':rtol}
  cases.append(c)
 w={'job':'99999','pid':999,'status':'ENTROPY_DEVICE_CORRECTNESS_AND_BENCH_SUPPRESSION_PASS_NOT_CAPACITY_NOT_TRAINING','source_manifest_sha256':SOURCE_SHA,'options':dict(ENTROPY_OPTIONS),'actual_device':{'uuid':p['leased_uuids'][0]},'allocator_after_release':{'allocated':0,'reserved':0},'source_identity':{'entropy_file_sha256':ENTROPY_FILE_SHA,'actor_source_sha256':ENTROPY_ACTOR_SHA,'codegen_source_sha256':'2f0105a59f6e341c2e8a695e02e192e2603a6ef31e6e914c8b6dd33b9a46235a'},'cases':cases,'events':[{'kind':'ACTUAL_PRECOMPILE','scope':'candidate','launchers_after':1,'inductor_meta':{k.replace('triton.',''):v for k,v in ENTROPY_OPTIONS.items() if k!='coordinate_descent_tuning'}}],'generated_files':[{'path':'CPU-FIXTURE-NOT-ACTUAL-KERNEL'}]}
 t={'job_id':99999,'returncode':0,'stdout':'JobId=99999 JobState=COMPLETED ExitCode=0:0'}
 return p,w,t

def test_complete_cpu_schema_contract_not_gpu_evidence():
 from source_binding import validate_entropy_results
 assert validate_entropy_results(*schema_records())==99999

@pytest.mark.parametrize('bad',['bench','length','tolerance','metadata','coordinate','terminal'])
def test_incomplete_or_loosened_entropy_schema_rejected(bad):
 from source_binding import validate_entropy_results
 p,w,t=schema_records()
 if bad=='bench':w['events'].append({'kind':'DO_BENCH_CALL','scope':'candidate'})
 if bad=='length':w['cases'].pop()
 if bad=='tolerance':w['cases'][0]['candidate_vs_original_compiled_value']['atol']=.1
 if bad=='metadata':w['events'][0]['inductor_meta'].pop('dynamic_scale_rblock')
 if bad=='coordinate':w['events'][0]['inductor_meta']['coordinate_descent_tuning']=True
 if bad=='terminal':t['stdout']='JobId=99998 JobState=COMPLETED ExitCode=0:0'
 with pytest.raises(ValueError):validate_entropy_results(p,w,t)
