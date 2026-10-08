import pathlib,json,hashlib,copy
import pytest
from standard_r1_formal_profile_admission import job_id,terminal_ok,profile_admission,SCOPE
ROOT=pathlib.Path(__file__).resolve().parent
@pytest.mark.parametrize('v',[True,'03163','3163x',0,-1])
def test_bad_job_identity(v):
 with pytest.raises(ValueError):job_id(v)
@pytest.mark.parametrize('raw',['JobId=3162 JobState=COMPLETED ExitCode=0:0','JobId=3163 JobState=FAILED ExitCode=1:0','JobId=3163 JobState=COMPLETED ExitCode=0:2',''])
def test_actual_terminal_not_queue_absence(raw):
 with pytest.raises(ValueError):terminal_ok({'job_id':3163,'returncode':0,'stdout':raw},3163)
def fixture(tmp):
 def put(name,obj):
  p=tmp/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj));return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
 evidence=tmp/'evidence';model=tmp/'model';tools=tmp/'tools';model.mkdir();tools.mkdir();weight=model/'model.safetensors';weight.write_bytes(b'CPU_CONTRACT_ONLY_NOT_A_REAL_32B_CHECKPOINT')
 inv={'directory':str(model),'files':{weight.name:{'size':weight.stat().st_size,'sha256':hashlib.sha256(weight.read_bytes()).hexdigest()}},'shards':[weight.name],'tensor_count':771}
 module=(ROOT/'search-r1-32b-grpo100-tools-v4/resource_audit.py').read_bytes();(tools/'resource_audit.py').write_bytes(module)
 manifest=put('tools/files.json',{'files':{'resource_audit.py':{'size':len(module),'sha256':hashlib.sha256(module).hexdigest()}}})
 records=[]
 for rank in range(8):
  u={'local_rows':320,'input_finite':dict.fromkeys(('old_log_probs','advantages','loss_mask','ref_log_prob'),True),'micro_forward_records':[{'finite':True}]*40,'optimizer_step_records':[{'finite':True,'grad_norm':0.}]*10,'parameter_sample_before':'a','parameter_sample_after':'b'}
  records.append(put('evidence/actor-resource-audit/actor-updates-rank%d-outer2.json'%rank,{'rank':rank,'updates':[dict(copy.deepcopy(u),actual_outer_update_ordinal=n) for n in (1,2)]}))
 import importlib.util
 spec=importlib.util.spec_from_file_location('real_resource_audit',tools/'resource_audit.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);counts=m.verify_updates([json.loads(pathlib.Path(r['path']).read_text()) for r in records],2)
 cleanup=dict.fromkeys(('own_processes_remaining','baseline_identity_missing','original_gpu_baseline_missing','own_gpu_records_remaining','parent_gpu_records','guardian_errors','sampler_errors','failure_history'),[]);cleanup.update(cleanup={'actual_cgroup_empty':True},error=None,primary_error=None,parent_cuda_initialized=False,terminal_monitors_drained_before_authorized_exit=True)
 cl=put('evidence/owned-cleanup.json',cleanup);outer=put('outer.json',{'actual_completed_outer_steps':[1,2],'export_step':2,'checkpoint_directory':str(model)})
 cfg=put('evidence/effective-runtime-config.json',{'job':'3163','mode':'profile2','expected_outer_steps':2,'final_checkpoint':str(model),'evidence_root':str(evidence),'tools_manifest':manifest,'source_manifest':manifest,'tools':str(tools)})
 result=put('evidence/train-result.json',{'job':'3163','status':'COMPLETE_UNREVIEWED','cleanup_passed':True,'error':None,'scope':'ACTUAL_SEARCH_R1_PROFILE2_UNREVIEWED_NOT_100_NOT_TOPD','evidence_root':str(evidence),'effective_config_sha256':cfg['sha256'],'cleanup':cl,'actual_outer_evidence':outer,'export':inv,'actual_update_evidence':counts})
 verifier=put('verifier.json',{'job_id':3163,'rank_records':records,'actual_update_evidence':counts})
 semantic=put('semantic.json',{'status':'FULL_HF_FINITE_SEMANTIC_CHANGED_UNREVIEWED','scope':'CPU_WEIGHT_SCAN_ONLY_NOT_TEACHER_QUALITY_NOT_TOPD_ACCEPTANCE','job_id':3163,'outer_steps':2,'train_result':result,'effective_config':cfg,'cleanup':cl,'outer_evidence':outer,'actual_hf_inventory':inv,'source_manifest':manifest,'tools_manifest':manifest,'semantic':{'all_finite':True,'change_beyond_base_to_teacher_dtype_cast':True,'changed_tensor_names':['CPU_CONTRACT_ONLY'],'compared_elements_including_tied_aliases':32763876352,'output_vocab_size':152064}})
 terminal=put('terminal.json',{'job_id':3163,'returncode':0,'stdout':'JobId=3163 JobState=COMPLETED ExitCode=0:0'});md=tmp/'review.md';md.write_text('CPU CONTRACT FIXTURE ONLY')
 review=put('review.json',{'status':'PASS','scope':SCOPE,'profile_job_id':3163,'reviewed_profile_result_sha256':result['sha256'],'prior_profile_terminal':terminal,'profile_effective_config':cfg,'profile_resource_verifier':verifier,'semantic_report':semantic,'review_md':{'path':str(md),'sha256':hashlib.sha256(md.read_bytes()).hexdigest()}})
 return {'profile_resource_proof':result,'source_manifest':manifest},{'prior_profile_job_id':3163,'independent_profile_review':review,'prior_profile_terminal':terminal,'prior_run_cleanup':cl},put

def test_full_same_run_contract_with_actual_counter_function(tmp_path):
 c,g,_=fixture(tmp_path);assert profile_admission(c,g,tmp_path)['counts']['optimizer_steps_per_rank']==20
@pytest.mark.parametrize('field,value',[('prior_profile_job_id',3159),('prior_run_cleanup',{'path':'/elsewhere','sha256':'0'*64})])
def test_same_run_reference_negative(tmp_path,field,value):
 c,g,_=fixture(tmp_path);g[field]=value
 with pytest.raises(ValueError):profile_admission(c,g,tmp_path)
def test_actual_rank_9_plus_11_cannot_compensate(tmp_path):
 c,g,put=fixture(tmp_path);review=json.loads(pathlib.Path(g['independent_profile_review']['path']).read_text());vf=json.loads(pathlib.Path(review['profile_resource_verifier']['path']).read_text());r=vf['rank_records'][0];raw=json.loads(pathlib.Path(r['path']).read_text());raw['updates'][0]['optimizer_step_records'].pop();raw['updates'][1]['optimizer_step_records'].append({'finite':True,'grad_norm':0.});vf['rank_records'][0]=put('evidence/actor-resource-audit/actor-updates-rank0-outer2.json',raw);review['profile_resource_verifier']=put('verifier.json',vf);g['independent_profile_review']=put('review.json',review)
 with pytest.raises(ValueError):profile_admission(c,g,tmp_path)
def test_full_checkpoint_bytes_drift(tmp_path):
 c,g,_=fixture(tmp_path);(tmp_path/'model/model.safetensors').write_bytes(b'x')
 with pytest.raises(ValueError):profile_admission(c,g,tmp_path)
