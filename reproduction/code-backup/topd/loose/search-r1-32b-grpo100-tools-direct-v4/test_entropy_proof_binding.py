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
 from source_binding import ENTROPY_OPTIONS,ENTROPY_FILE_SHA,ENTROPY_ACTOR_SHA,ENTROPY_COMPILED_POLICY
 p={'job':'99999','status':'ENTROPY_PARENT_AND_ORIGINAL_MATH_PASS_NOT_CAPACITY_NOT_TRAINING','source_manifest_sha256':SOURCE_SHA,'leased_uuids':['CPU-SCHEMA-ONLY-'+str(i) for i in range(8)],'guardian_errors':[],'cleanup':{'actual_cgroup_empty':True,'remaining':[]},'parent_cuda_initialized':False,'baseline_preserved':True,'worker_identity':{'pid':999}}
 specs=[(4,17,'float32'),(5,17,'bfloat16'),(2,152064,'float32'),(3,152064,'bfloat16'),(6144,152064,'float32'),(6143,152064,'float32'),(6144,152064,'float32'),(6144,152064,'bfloat16'),(6143,152064,'bfloat16'),(6144,152064,'bfloat16')];cases=[]
 for length,vocab,dtype in specs:
  c={'length':length,'vocab':vocab,'input_dtype':dtype,'candidate':{'output_dtype':'torch.'+dtype,'gradient_dtype':'torch.'+dtype},'original_compiled':{'output_dtype':'torch.'+dtype,'gradient_dtype':'torch.'+dtype}}
  for name in ('candidate_vs_original_compiled_value','candidate_vs_original_compiled_gradient','candidate_vs_eager_value','candidate_vs_eager_gradient'):
   compiled='original_compiled' in name;grad=name.endswith('gradient')
   atol,rtol=((1e-7,1e-4) if grad else (1e-4,1e-5)) if compiled or dtype=='float32' else ((.0001,.032) if grad else (.125,.016))
   c[name]={'all_elements_passed':True,'elements':length*vocab if grad else length,'max_absolute_delta':0.,'max_relative_delta':0.,'atol':atol,'rtol':rtol}
   if compiled and dtype=='bfloat16':
    c[name].pop('atol');c[name].pop('rtol');c[name].update(criterion='ACTUAL_REFERENCE_BF16_NEXTAFTER_CLOSED_NEIGHBOR_INTERVAL_NO_EXTRA_ABSOLUTE_FLOOR',nonadjacent_elements=0,allowed_reference_adjacent_steps=1,max_reference_adjacent_steps=0.)
  cases.append(c)
 w={'job':'99999','pid':999,'status':'ENTROPY_DEVICE_CORRECTNESS_AND_BENCH_SUPPRESSION_PASS_NOT_CAPACITY_NOT_TRAINING','source_manifest_sha256':SOURCE_SHA,'options':dict(ENTROPY_OPTIONS),'compiled_comparison_policy':dict(ENTROPY_COMPILED_POLICY),'actual_device':{'uuid':p['leased_uuids'][0]},'allocator_after_release':{'allocated':0,'reserved':0},'source_identity':{'entropy_file_sha256':ENTROPY_FILE_SHA,'actor_source_sha256':ENTROPY_ACTOR_SHA,'codegen_source_sha256':'2f0105a59f6e341c2e8a695e02e192e2603a6ef31e6e914c8b6dd33b9a46235a'},'cases':cases,'events':[{'kind':'ACTUAL_PRECOMPILE','scope':'candidate','launchers_after':1,'inductor_meta':{k.replace('triton.',''):v for k,v in ENTROPY_OPTIONS.items() if k!='coordinate_descent_tuning'}}],'generated_files':[{'path':'CPU-FIXTURE-NOT-ACTUAL-KERNEL'}]}
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


@pytest.mark.parametrize('dtype_name',['float32','bfloat16'])
def test_actual_final_worker_comparison_dicts_pass_and_real_violation_rejects(dtype_name):
 # Execute original final producer comparison functions on real CPU tensors;
 # this checks schema fidelity, not entropy/CUDA numerical correctness.
 import hashlib,torch
 from source_binding import validate_entropy_case
 p=HERE/'entropy_reference_worker.py'
 assert hashlib.sha256(p.read_bytes()).hexdigest()=='cae701443d3d0cdad0b52c03aea17c5d95967af9b94bcb2916cdbf92234a7e03'
 tree=ast.parse(p.read_text());names={'need','difference','adjacent_difference','compiled_difference'}
 ns={'torch':torch};exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),str(p),'exec'),ns)
 dtype=getattr(torch,dtype_name);reference=torch.tensor([1.,2.],dtype=dtype);gradient=torch.ones((2,2),dtype=dtype)
 candidate=torch.nextafter(reference,torch.full_like(reference,float('inf'))) if dtype_name=='bfloat16' else reference.clone()
 case={'length':2,'vocab':2,'input_dtype':dtype_name,'candidate':{'output_dtype':str(dtype),'gradient_dtype':str(dtype)},'original_compiled':{'output_dtype':str(dtype),'gradient_dtype':str(dtype)},'candidate_vs_original_compiled_value':ns['compiled_difference'](candidate,reference,False),'candidate_vs_original_compiled_gradient':ns['compiled_difference'](gradient,gradient,True),'candidate_vs_eager_value':ns['difference'](candidate,reference+1,0.,0.),'candidate_vs_eager_gradient':ns['difference'](gradient,gradient+1,0.,0.)}
 assert case['candidate_vs_eager_value']['all_elements_passed'] is False and validate_entropy_case(case)
 bad=torch.nextafter(candidate,torch.full_like(candidate,float('inf'))) if dtype_name=='bfloat16' else candidate+1
 case['candidate_vs_original_compiled_value']=ns['compiled_difference'](bad,reference,False)
 with pytest.raises(ValueError):validate_entropy_case(case)
