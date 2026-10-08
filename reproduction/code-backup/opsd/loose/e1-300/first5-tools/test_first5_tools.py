"""CPU negative tests only. No actual job/source acceptance or receipt signing."""
import copy,hashlib,importlib.util,json,os,sys
from pathlib import Path
import pytest
DIR=Path(__file__).resolve().parent;sys.path.insert(0,str(DIR));sys.path.insert(0,str(DIR.parents[1]/'opsd-e1-300'))
import receipt_payload as r
import hf_cpu_reload as hf

def manifest(root):return {'version':1,'method':'pure_opsd','run_root':str(root),'source_commit':r.SOURCE,'config_sha256':'c'*64,'global_step':5,'actual_outer_updates':5,'next_global_step':6,'horizon':300,'warmup_steps':57,'world_size':8,'driver_pid':100,'uid':os.getuid(),'job_id':'123','pause_nonce':'n'*64}

def actual_metrics():
 d={'actor/optimizer_steps':20,'actor/skipped_empty_minibatches':0,'actor/samples':320,'actor/loss_mask_valid':1,'actor/opsd_loss_finite':1,'actor/grad_norm_finite':1,'teacher/trainable_parameters_per_rank':0,'teacher/gradient_tensors_per_rank':0,'teacher/student_optimizer_membership_per_rank':0,'teacher/storage_is_cpu_per_rank':1,'teacher/all_parameter_dtype_is_fp32_per_rank':1,'teacher/attention_is_eager_per_rank':1,'actor/allocator_is_native':1,'actor/allocator_configuration_is_default':1,'actor/old_log_probs_present':1,'actor/old_log_probs_finite':1,'actor/supervised_tokens':10}
 return {k+'/rank_'+str(rank):v for rank in range(8) for k,v in d.items()}

def frozen_gate():
 path=DIR.parents[1]/'opsd-e1-300/search_r1/baselines/checkpoint_gate.py';return r.frozen_gate(path.parents[2],r.sha(path))

def log_fixture(tmp_path):
 m=manifest(tmp_path);m['pause_nonce']='a'*64;metrics=actual_metrics();rows=[]
 for step in range(1,6):
  d=dict(metrics);d.update({'teacher/cuda_update_entries_cumulative_per_rank/rank_'+str(rank):step for rank in range(8)})
  if step==5:d.update({'opsd_stage/val/qa_count/all':64,'opsd_stage/val/evaluated_count/fixture':64})
  rows.append({'global_step':step,'actual_outer_updates':step,'driver_pid':100,'job_id':'123','source_commit':r.SOURCE,'config_sha256':'c'*64,'metrics':d})
 m['outer_metrics']={k:v for k,v in rows[-1]['metrics'].items() if 'cuda' in k};pause={'manifest_sha256':'b'*64,'driver_pid':100,'job_id':'123','pause_nonce':'a'*64}
 path=tmp_path/'prefix'
 def write(rows_=rows,pause_=pause,extra=''):path.write_text('\n'.join('OPSD_E1_OUTER_JSON='+json.dumps(x) for x in rows_)+'\nBASELINE_EVALUATION_METRICS_JSON='+json.dumps({'val/qa_count/all':512,'val/evaluated_count/fixture':512})+'\nBASELINE_EVALUATION_METRICS_JSON='+json.dumps({'val/qa_count/all':64,'val/evaluated_count/fixture':64})+'\nOPSD_E1_STEP5_PAUSED_JSON='+json.dumps(pause_)+extra)
 write();return m,rows,pause,path,write

def test_actual_five_plus_lp_cudaentry_stage_and_pause(tmp_path):
 m,rows,pause,path,write=log_fixture(tmp_path);gate=frozen_gate();assert len(r.check_log(path,m,'b'*64,gate)[0])==5
 for count in (4,6):
  altered=rows[:count] if count==4 else rows+[rows[-1]];write(altered)
  with pytest.raises(ValueError):r.check_log(path,m,'b'*64,gate)
 for key,value in [('actor/optimizer_steps/rank_7',19),('actor/old_log_probs_finite/rank_3',0),('teacher/cuda_update_entries_cumulative_per_rank/rank_0',4),('opsd_stage/val/qa_count/all',63)]:
  altered=copy.deepcopy(rows);altered[-1]['metrics'][key]=value;write(altered)
  with pytest.raises((ValueError,RuntimeError)):r.check_log(path,m,'b'*64,gate)
 write(rows,dict(pause,driver_pid=101))
 with pytest.raises(ValueError):r.check_log(path,m,'b'*64,gate)
 write(extra='\nOPSD_E1_STEP5_RELEASED_JSON={}');
 with pytest.raises(ValueError):r.check_log(path,m,'b'*64,gate)


def test_exact_file_sha_escape_duplicate_and_freshprefix(tmp_path):
 live=tmp_path/'train.log';live.write_bytes(b'original first5 fixture\n');snap=tmp_path/'snapshot';p=r.log_prefix(live,snap,tmp_path);assert p['sha256']==r.sha(live) and snap.read_bytes()==live.read_bytes()
 with pytest.raises(FileExistsError):r.log_prefix(live,snap,tmp_path)
 with pytest.raises(ValueError):r.file({'path':str(live),'sha256':'0'*64},tmp_path)
 p=tmp_path/'duplicate.json';p.write_text('{"x":1,"x":2}')
 with pytest.raises(ValueError):r.read(p)
 link=tmp_path/'link';link.symlink_to(live)
 with pytest.raises(ValueError):r.file({'path':str(link),'sha256':r.sha(live)},tmp_path)


def test_unsigned_payload_review_requires_actual_identity(tmp_path):
 m=manifest(tmp_path);evidence={'raw':{'path':'fixture','sha256':'f'*64}};p=tmp_path/'review.json'
 d={'version':1,'role':'INDEPENDENT_REVIEWER','status':'PASS_ACTUAL_FIRST5_EVIDENCE','source_commit':r.SOURCE,'job_id':'123','driver_pid':100,'pause_nonce':m['pause_nonce'],'checkpoint_manifest_sha256':'b'*64,'evidence':evidence}
 p.write_bytes(r.canonical(d));assert r.review_record(p,r.sha(p),tmp_path,m,'b'*64,'INDEPENDENT_REVIEWER',evidence)['sha256']==r.sha(p)
 for key,value in [('job_id','old'),('driver_pid',101),('status','PASS_STATIC_ONLY'),('role','ROOT_AUDITOR'),('checkpoint_manifest_sha256','0'*64)]:
  p.write_bytes(r.canonical(dict(d,**{key:value})))
  with pytest.raises(ValueError):r.review_record(p,r.sha(p),tmp_path,m,'b'*64,'INDEPENDENT_REVIEWER',evidence)


def test_native_cpu_report_and_teacher_value_digest_cross_binding(tmp_path):
 m=manifest(tmp_path);reports=[{'pid':200+i,'object_ids':{'model':i},'state_sha256':'a'*64,'state_digest':'b'*64} for i in range(8)]
 native={'scope':'NATIVE_CPU_FIRST5_STORED_STATE_INSPECTION_NOT_LIVE_RESTORE','cpu_runtime':{'torch_version':'2.4.0'},'passed':True,'source_commit':r.SOURCE,'job_id':'123','run_root':str(tmp_path),'checkpoint_manifest_sha256':'f'*64,'cuda_is_initialized':False,'driver':{'pid':100,'sampler_cursor':2560,'consumed_cursor':2560,'size':169028,'epoch':0},'ranks':[{'rank':i,**x,'optimizer_step_values':[100],'scheduler_last_epoch':5,'scheduler_last_lr':[1e-6*5/57],'rng_verified':True,'teacher_content_digest':'c'*64,'teacher_cuda_update_entries':5} for i,x in enumerate(reports)]}
 h={'passed':True,'source_commit':r.SOURCE,'job_id':'123','run_root':str(tmp_path),'checkpoint_manifest_sha256':'f'*64,'cuda_is_initialized':False,'scope':'ACTUAL_FULL_CPU_HF5_RELOAD_NO_CUDA','actual_unique_parameters':r.COUNT,'all_loaded_parameters_finite':True,'actual_cpu_forward_finite':True,'total_changed_elements':1,'config_computation_equal':True,'base_named_parameter_digest':'c'*64}
 keys=[f'param_{i:03d}' for i in range(435)];rec={'path':str(tmp_path/'model-1.safetensors'),'sha256':'e'*64};idx={'path':str(tmp_path/'model.safetensors.index.json'),'sha256':'d'*64}
 m['files']={'actor/model-1.safetensors':rec,'actor/model.safetensors.index.json':idx};h['files']=copy.deepcopy(m['files']);h['hf_map']={key:'model-1.safetensors' for key in keys};native.update(hf_all_rank_slices_exact=True,hf_files={rec['path']:rec['sha256'],idx['path']:idx['sha256']})
 for rank in native['ranks']:rank.update(model_hf_slice_exact=True,model_hf_slice_keys=keys,model_dtensor_entries=435,model_layouts={key:{} for key in keys})
 r.check_cpu_evidence(native,h,m,'f'*64,reports)
 for key,value in [('optimizer_step_values',[99]),('rng_verified',False),('teacher_content_digest','d'*64),('pid',999),('model_hf_slice_exact',False),('model_hf_slice_keys',keys[:-1])]:
  bad=copy.deepcopy(native);bad['ranks'][7][key]=value
  with pytest.raises(ValueError):r.check_cpu_evidence(bad,h,m,'f'*64,reports)
 bad=copy.deepcopy(native);bad['hf_files'][rec['path']]='f'*64
 with pytest.raises(ValueError):r.check_cpu_evidence(bad,h,m,'f'*64,reports)
 bad=copy.deepcopy(native);bad['hf_all_rank_slices_exact']=False
 with pytest.raises(ValueError):r.check_cpu_evidence(bad,h,m,'f'*64,reports)
 bad=dict(h,total_changed_elements=0)
 with pytest.raises(ValueError):r.check_cpu_evidence(native,bad,m,'f'*64,reports)
 bad=copy.deepcopy(native);bad['driver']['consumed_cursor']=3072
 with pytest.raises(ValueError):r.check_cpu_evidence(bad,h,m,'f'*64,reports)


def test_real_rank_pid_start_uuid_resource_join(tmp_path):
 m=manifest(tmp_path);reports=[{'pid':200+i,'rank':i} for i in range(8)];service={'pid':1000,'start':'1'};uuids=[f'GPU-{i}' for i in range(8)]
 apps=[{'pid':200+i,'uuid':u,'uid':str(os.getuid()),'in_slurm_job':True,'process_start_ticks':str(200+i)} for i,u in enumerate(uuids)]+[{'pid':1000,'uuid':u,'uid':str(os.getuid()),'in_slurm_job':True,'process_start_ticks':'1'} for u in uuids]
 a={'time':'2026-10-04T00:00:00+00:00','job_id':'123','gpus':[{'uuid':u,'util':0.,'used_mib':1000.} for u in uuids],'gpu_processes':apps};b=copy.deepcopy(a);b['time']='2026-10-04T00:00:02+00:00';p=tmp_path/'resources-prefix.ndjson'
 def write(x,y):p.write_text('\n'.join(json.dumps(v) for v in (x,y)))
 write(a,b);assert len(r.resource_join(p,m,reports,service)['samples'])==2
 bad=copy.deepcopy(b);bad['gpu_processes'][7]['process_start_ticks']='recycled';write(a,bad)
 with pytest.raises(ValueError):r.resource_join(p,m,reports,service)
 bad=copy.deepcopy(b);bad['gpu_processes'][7]['pid']=777;write(a,bad)
 with pytest.raises(ValueError):r.resource_join(p,m,reports,service)
 write(a,a)
 with pytest.raises(ValueError):r.resource_join(p,m,reports,service)
 bad=copy.deepcopy(b);bad['query_error']='TimeoutExpired';write(a,bad)
 with pytest.raises(ValueError):r.resource_join(p,m,reports,service)
 for value in (float('nan'),101.):
  bad=copy.deepcopy(b);bad['gpus'][0]['util']=value;write(a,bad)
  with pytest.raises(ValueError):r.resource_join(p,m,reports,service)


def tiny_hf(tmp_path):
 import torch
 from transformers import Qwen2Config,Qwen2ForCausalLM,PreTrainedTokenizerFast,AutoModelForCausalLM
 from tokenizers import Tokenizer
 from tokenizers.models import WordLevel
 from tokenizers.pre_tokenizers import Whitespace
 from safetensors import safe_open
 base=tmp_path/'base';trained=tmp_path/'hf';base.mkdir();trained.mkdir()
 config=Qwen2Config(vocab_size=8,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=64,tie_word_embeddings=True,bos_token_id=None,eos_token_id=1,pad_token_id=0)
 model=Qwen2ForCausalLM(config).to(torch.bfloat16);count=sum(p.numel() for p in model.parameters());model.save_pretrained(base)
 token=Tokenizer(WordLevel({'<pad>':0,'<end>':1,'<unk>':2,'a':3,'Return':4,'the':5,'word':6,'test':7},unk_token='<unk>'));token.pre_tokenizer=Whitespace();tok=PreTrainedTokenizerFast(tokenizer_object=token,unk_token='<unk>',pad_token='<pad>',eos_token='<end>',chat_template="{{ 'a' }}");tok.save_pretrained(base)
 student=AutoModelForCausalLM.from_pretrained(base,torch_dtype=torch.float32);student.save_pretrained(trained);tok.save_pretrained(trained)
 for directory in (base,trained):
  with safe_open(str(directory/'model.safetensors'),framework='pt',device='cpu') as handle:m={k:'model.safetensors' for k in handle.keys()}
  (directory/'model.safetensors.index.json').write_text(json.dumps({'weight_map':m}))
 return base,trained,student,count


def test_real_tiny_full_cpu_hf_precision_change_is_not_training(tmp_path):
 base,trained,student,count=tiny_hf(tmp_path)
 with pytest.raises(ValueError,match='No real same-FP32'):hf.load_compare(base,trained,count)


def test_real_tiny_full_cpu_hf_change_and_missing_tensor_reject(tmp_path):
 import torch
 base,trained,student,count=tiny_hf(tmp_path)
 with torch.no_grad():student.get_input_embeddings().weight.add_(.001)
 student.save_pretrained(trained)
 result=hf.load_compare(base,trained,count);assert result['total_changed_elements']>0 and result['actual_cpu_forward_finite'] and not result['cuda_is_initialized'] and result['actual_unique_parameters']==count
 index=json.loads((trained/'model.safetensors.index.json').read_text());del index['weight_map']['model.layers.0.self_attn.q_proj.weight'];(trained/'model.safetensors.index.json').write_text(json.dumps(index))
 with pytest.raises(ValueError,match='Actual tensors'):hf.load_compare(base,trained,count)


def test_loading_schema_no_random_params():
 good={'missing_keys':[],'unexpected_keys':[],'mismatched_keys':[],'error_msgs':[]};hf.empty_loading(good)
 for bad in [{k:v for k,v in good.items() if k!='missing_keys'},dict(good,missing_keys=['head']),dict(good,extra=[])]:
  with pytest.raises(ValueError):hf.empty_loading(bad)


def test_live_process_all_four_uids_job_start_and_nonterminated(tmp_path):
 m=manifest(tmp_path);p=tmp_path/'proc'/'200';p.mkdir(parents=True)
 uid=os.getuid();valid=f'Uid:\t{uid} {uid} {uid} {uid}\nState:\tS (sleeping)\n'
 (p/'status').write_text(valid);(p/'cgroup').write_text('0::/slurm/uid_0/job_123/step_batch\n')
 (p/'stat').write_text('200 (worker with space) '+' '.join(['S']+['0']*18+['4321']+['0']*3))
 assert r.actual_live_process(200,m,tmp_path/'proc')['start']=='4321'
 for status,cgroup in [(valid.replace(f'{uid} {uid} {uid} {uid}',f'{uid} {uid} {uid+1} {uid}'),'0::/slurm/job_123/step_batch'),(valid,'0::/slurm/job_1234/step_batch'),(valid.replace('S (sleeping)','Z (zombie)'),'0::/slurm/job_123/step_batch')]:
  (p/'status').write_text(status);(p/'cgroup').write_text(cgroup)
  with pytest.raises(ValueError):r.actual_live_process(200,m,tmp_path/'proc')


def test_full_cpu_reload_config_and_allfiles_pin_drift_rejected(tmp_path):
 p=tmp_path/'resolved.json';p.write_bytes(r.canonical({'model':'original'}));q=tmp_path/'state.pt';q.write_bytes(b'actual bytes')
 m={'files':{'resolved_config':{'path':str(p),'sha256':r.sha(p)},'native':{'path':str(q),'sha256':r.sha(q)}},'config_sha256':hashlib.sha256(r.canonical({'model':'original'})).hexdigest()}
 assert hf.verify_manifest_inputs(m,tmp_path)=={'model':'original'}
 q.write_bytes(b'mutated bytes')
 with pytest.raises(ValueError):hf.verify_manifest_inputs(m,tmp_path)
 q.write_bytes(b'actual bytes');p.write_bytes(r.canonical({'model':'other'}));m['files']['resolved_config']['sha256']=r.sha(p)
 with pytest.raises(ValueError):hf.verify_manifest_inputs(m,tmp_path)
