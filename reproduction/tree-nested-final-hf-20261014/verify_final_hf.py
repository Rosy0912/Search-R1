"""Future asset-local CPU cold HF/tokenizer/native200 verification for actual778.

Does not stop, start, restore or modify training; refuses before real200 exists.
"""
import argparse,gc,hashlib,json,math,os,resource,socket,stat,subprocess,sys,time
from pathlib import Path
import chain_contract as chain
ROOT=chain.ROOT;SOURCE=chain.TRAIN_SOURCE;CONFIG=chain.CONFIG_SHA
PACKAGE_SHA=chain.PACKAGE_SHA;RUN=chain.RUN;PACKAGE=chain.PACKAGE
BASE=ROOT/'instruct300/assets/qwen2.5-3b-instruct-aa8e725'
CHECKPOINT=chain.HF;NATIVE=chain.NATIVE
VERIFIER_REL='source/repo/reproduction/baselines/tree-validation-20261002/evidence/runtime/check_real_checkpoint.py'
VERIFIER_SHA='b3075d7b9346bb7682e0ea05962de0624c117b354590b728ed8a18b7e8ba3c97'
OUT=RUN/'output/final-hf-native5-acceptance'

def require(condition,message):
 if not condition:raise ValueError(message)
def file_sha(path):
 require(stat.S_ISREG(path.lstat().st_mode),'Evidence member must be ordinary file: '+str(path))
 digest=hashlib.sha256()
 with path.open('rb') as stream:
  before=os.fstat(stream.fileno())
  for data in iter(lambda:stream.read(8*1024*1024),b''):digest.update(data)
  after=os.fstat(stream.fileno())
 require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),'Evidence changed while hashing')
 return digest.hexdigest()
def read_json(path,expected=None):
 digest=file_sha(path);require(expected is None or expected==digest,'JSON SHA changed: '+str(path))
 value=json.loads(path.read_text());require(file_sha(path)==digest,'JSON changed while reading');return value,digest
def identity(path):
 files={str(p.relative_to(path)):file_sha(p) for p in sorted(path.rglob('*')) if p.is_file()}
 return dict(files=files,sha256=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest())
def write_once(path,value):
 with path.open('x') as out:json.dump(value,out,indent=2,allow_nan=False);out.flush();os.fsync(out.fileno())
def normalize_tokenizer_value(value):
 # AddedToken behavior must be retained, not reduced to its string content.
 fields=('content','single_word','lstrip','rstrip','normalized','special')
 if all(hasattr(value,key) for key in fields):return {key:getattr(value,key) for key in fields}
 if isinstance(value,dict):return {str(key):normalize_tokenizer_value(item) for key,item in value.items()}
 if isinstance(value,(list,tuple)):return [normalize_tokenizer_value(item) for item in value]
 require(value is None or isinstance(value,(str,int,float,bool)),'Unexpected tokenizer property type')
 return value

def normalized_tokenizer_config(path):
 value,_=read_json(path/'tokenizer_config.json')
 # These fields locate serialized files/revisions; they do not alter tokenization.
 ignored=('name_or_path','_name_or_path','tokenizer_file','_commit_hash')
 return normalize_tokenizer_value({key:item for key,item in value.items() if key not in ignored})

def tokenizer_semantics(tokenizer):
 require(hasattr(tokenizer,'backend_tokenizer'),'Fast tokenizer backend unavailable')
 options=('padding_side','truncation_side','model_max_length','clean_up_tokenization_spaces',
          'split_special_tokens','add_bos_token','add_prefix_space','errors')
 return dict(backend=json.loads(tokenizer.backend_tokenizer.to_str()),
     full_special_roles=normalize_tokenizer_value(tokenizer.special_tokens_map_extended),
     added_tokens=normalize_tokenizer_value(tokenizer.added_tokens_decoder),
     runtime_options={key:normalize_tokenizer_value(getattr(tokenizer,key,None)) for key in options})

def verify_native_cpu(native,expected_provenance):
 """Sequentially deserialize all eight saved CPU states, never construct an actor."""
 from search_r1.baselines.training_state import verified_state_file,digest,finite
 import torch
 torch.set_num_threads(8)
 accounting=native['identity']['guard_continue_accounting'];records=[]
 for record in sorted(native['worker_states'],key=lambda x:x['identity']['rank']):
  rank=record['identity']['rank'];path=Path(record['path'])
  require(path==NATIVE.parent/f'rank-{rank:02d}.pt','Native rank path must stay at actual200')
  require(stat.S_ISREG(path.lstat().st_mode),'Ordinary saved native rank file required')
  saved=verified_state_file(record);finite(saved)
  require(saved['version']==2 and saved['identity']==record['identity']
          and digest(saved)==record['logical_state_sha256']
          and saved['optimizer']['state'] and saved['scheduler']['last_epoch']==200,
          'Actual native CPU saved identity/logical/optimizer/scheduler differs')
  calls=accounting['total_optimizer_calls_per_rank'][str(rank)]
  for state in saved['optimizer']['state'].values():
   step=state['step'];require(isinstance(step,torch.Tensor) and step.numel()==1
       and step.item()==int(step.item()) and step.item()>=0
       and all(isinstance(state.get(key),torch.Tensor) for key in ('exp_avg','exp_avg_sq')),
       'Complete Adam moments/integer step required')
  group_keys=[key for group in saved['optimizer']['param_groups'] for key in group['params']]
  require(len(group_keys)==len(set(group_keys)) and set(group_keys)==set(saved['optimizer']['state'])
          and len(saved['optimizer']['state'])==record['optimizer_state_entries'],
          'Complete saved optimizer group/state coverage differs')
  steps=sorted({int(state['step']) for state in saved['optimizer']['state'].values()})
  require(steps==[calls] and record['actual_total_optimizer_calls']==calls
          and record['observed_adam_steps']==[calls]
          and all(isinstance(value,torch.Tensor) and value.device.type=='cpu' for value in saved['parameters'].values())
          and saved['parameters'] and saved['local_shards'],
          'Actual complete native CPU parameters/Adam coverage differs')
  records.append(dict(rank=rank,world_size=8,state_path=str(path),state_file_sha256=record['sha256'],
      logical_state_sha256=record['logical_state_sha256'],actual_pt_deserialization=True,
      all_saved_state_finite=True,actual_adam_steps=steps,scheduler_step=200,
      actual_total_optimizer_calls=calls,optimizer_state_entries=len(saved['optimizer']['state'])))
  del saved;gc.collect()
 require(len(records)==8 and [x['rank'] for x in records]==list(range(8)),'Actual eight native rank states required')
 record=native['driver'];require(Path(record['path'])==NATIVE.parent/'driver.pt'
         and stat.S_ISREG(Path(record['path']).lstat().st_mode),'Ordinary native driver path differs')
 saved=verified_state_file(record);finite(saved);attempts=accounting['training_attempts']
 require(saved['version']==1 and saved['identity']==native['identity']
         and digest(saved)==record['logical_state_sha256']
         and saved['completed_step']==saved['actual_outer_updates']==200 and saved['next_step']==201
         and saved['guard_continue_accounting']==accounting
         and saved['recovery_provenance']==expected_provenance
         and saved['algorithm']['method']=='tree_grpo' and saved['algorithm']['tree_rng'] is not None,
         'Actual native200 driver/attempt/ancestry/Tree state differs')
 sampler=saved['sampler'];limit=(169028//512)*512;consumed=2560+(attempts-5)*512
 require(sampler['rows']==169028 and sampler['batch_size']==512
         and sampler['identity']==native['identity']['data_identity']
         and sampler['epoch']==(consumed-1)//limit and sampler['cursor']==(consumed-1)%limit+1
         and len(sampler['order'])==169028 and set(sampler['order'])==set(range(169028)),
         'Actual native200 consumed sampler/permutation differs')
 driver=dict(state_path=record['path'],state_file_sha256=record['sha256'],
     logical_state_sha256=record['logical_state_sha256'],actual_pt_deserialization=True,
     all_saved_state_finite=True,completed_outer=200,next_outer=201,
     training_attempts=attempts,sampler_cursor=sampler['cursor'],sampler_epoch=sampler['epoch'],
     rng_and_tree_rng_serialized_and_logical_hash_verified=True)
 del saved;gc.collect()
 return dict(status='ACTUAL_NATIVE200_EIGHT_RANK_CPU_STATE_VERIFIED',workers=records,driver=driver,
     actual_pt_deserialization=True,all_saved_state_finite=True,
     optimizer_or_driver_restored=False,gpu_cold_world_restore_performed=False)


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--code-manifest',required=True,type=Path);p.add_argument('--code-manifest-sha',required=True)
 a=p.parse_args();started=time.time();cpu_started=time.process_time()
 require(socket.gethostname().split('.')[0]=='h100-10','Only retained18 same-node asset-local verification')
 require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU verification must explicitly hide all GPUs')
 require(sys.flags.optimize==0,'Unoptimized Python required for frozen checker assertions')
 available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
 require(available>=64*1024**3,'At least64GiB actual free CPU memory required; no GPU fallback')
 tools=Path(__file__).resolve().parent;code,code_sha=read_json(a.code_manifest,a.code_manifest_sha)
 require(a.code_manifest.resolve()==tools/'code-manifest.json'
         and code['schema']=='nested_final_hf_code_v1' and code['training_source_commit']==SOURCE
         and code['readiness']=='PENDING_REAL_TRAIN200_AND_FINAL_NATIVE_HF_COLD_RELOAD'
         and code['operational_changes_independently_reviewed'] is False
         and code['training_chain']==dict(package_sha256=PACKAGE_SHA,config_sha256=CONFIG,
             native5_loaded_receipt_sha256=chain.RESTORE_RECEIPT_SHA,effective_training_segments=chain.effective_segments())
         and set(code['files'])=={'verify_final_hf.py','chain_contract.py','build_code_package.py','README.md'},
         'Published independent future verifier source inventory differs')
 for name,digest in code['files'].items():require(file_sha(tools/name)==digest,'Verifier wrapper code changed: '+name)
 package,package_sha=read_json(PACKAGE/'manifest.json',PACKAGE_SHA)
 require(package['code_commit']==SOURCE and package['files'][VERIFIER_REL]==VERIFIER_SHA
         and file_sha(PACKAGE/VERIFIER_REL)==VERIFIER_SHA,'Unchanged successful full-HF checker must remain pinned')
 # Freeze actual current code before JSON accounting imports; no old experiment scan.
 for name,digest in package['files'].items():require(file_sha(PACKAGE/name)==digest,'Actual training package file changed: '+name)
 package,native,native_sha,ranks=chain.training_metadata()
 load=chain.read_json(chain.RESTORE_RECEIPT,chain.RESTORE_RECEIPT_SHA)
 chain_evidence=chain.restore_and_fault_evidence()
 binding_paths=dict(wrapper=Path(__file__).resolve(),chain_contract=tools/'chain_contract.py',code_manifest=a.code_manifest,
     resolved_config=chain.CONFIG,package_manifest=PACKAGE/'manifest.json',frozen_verifier=PACKAGE/VERIFIER_REL,
     native_manifest=NATIVE,native5_loaded=chain.RESTORE_RECEIPT)
 bindings_before={name:file_sha(path) for name,path in binding_paths.items()}
 require(bindings_before['resolved_config']==CONFIG and bindings_before['package_manifest']==PACKAGE_SHA
         and bindings_before['frozen_verifier']==VERIFIER_SHA and bindings_before['native_manifest']==native_sha,
         'Real200 package/config/native/loaded5 identity changed')
 require(Path(native['hf_checkpoint'])==CHECKPOINT and CHECKPOINT.resolve().is_relative_to(RUN.resolve()),
         'Native/HF200 must stay in current own remote run')
 base_before=identity(BASE);require(base_before==package['model_identity'],'Official immutable base files differ')
 before=identity(CHECKPOINT);require(before==native['hf_identity'],'HF200 complete inventory differs from saved native200')
 out=OUT;require(not out.exists(),'Existing namespace: preserve prior evidence; no overwrite/reload retry');out.mkdir()
 native_report=verify_native_cpu(native,load['provenance']);write_once(out/'native200-cpu-state.json',native_report)
 native_files={r['path']:r['sha256'] for r in native['worker_states']+[native['driver']]}
 env={k:v for k,v in os.environ.items() if not k.startswith('TREE_')}
 env.update(CUDA_VISIBLE_DEVICES='',PYTHONPATH='',PYTHONNOUSERSITE='1',PYTHONOPTIMIZE='0',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HOME=str(out/'hf-cache'),OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8')
 with (out/'verifier.log').open('x') as log:
  result=subprocess.run([sys.executable,str(PACKAGE/VERIFIER_REL),'--base',str(BASE),'--checkpoint',str(CHECKPOINT),'--output',str(out/'full-HF-cold-reload.json')],env=env,stdout=log,stderr=subprocess.STDOUT)
 require(result.returncode==0,'Actual full CPU cold reload failed; preserve log/evidence, no automatic retry')
 full,full_sha=read_json(out/'full-HF-cold-reload.json')
 require(full['status']=='HF_RELOAD_PARAMETER_DELTA_ACCEPTED' and full['all_parameters_finite'] is True and full['cpu_forward_logits_finite'] is True and full['changed_parameters']>0 and full['max_abs_parameter_delta']>0 and full['total_parameters']>1_000_000_000,'Complete finite/nonzero cold model proof missing')
 require(Path(full['checkpoint']).resolve()==CHECKPOINT.resolve() and Path(full['base']).resolve()==BASE.resolve(),'Cold loader used different weights')
 require(full['checkpoint_file_sha256']==before['files'],'Cold reload observed a different HF inventory')
 # Tokenizer-only extra checks: no second model load or generation batch.
 os.environ.update({k:v for k,v in env.items() if k in ('CUDA_VISIBLE_DEVICES','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_HOME')})
 from transformers import AutoTokenizer
 base_tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
 tok=AutoTokenizer.from_pretrained(CHECKPOINT,local_files_only=True)
 require(normalized_tokenizer_config(BASE)==normalized_tokenizer_config(CHECKPOINT),'Tokenizer semantic config differs (only serialization paths ignored)')
 require(tokenizer_semantics(base_tok)==tokenizer_semantics(tok),'Full tokenizer backend/AddedToken/special-role/runtime behavior differs')
 special=lambda t:dict(bos=t.bos_token_id,eos=t.eos_token_id,pad=t.pad_token_id,unk=t.unk_token_id,all_special_tokens=t.all_special_tokens,all_special_ids=t.all_special_ids,chat_template=t.chat_template)
 require(base_tok.get_vocab()==tok.get_vocab() and special(base_tok)==special(tok),'Full tokenizer/special/PAD/chat identity differs')
 probes=['The capital of France is','你好，世界。','<|im_start|>user\nFind an answer.<|im_end|>']
 for text in probes:
  b=base_tok.encode(text,add_special_tokens=True);c=tok.encode(text,add_special_tokens=True)
  require(b==c and base_tok.decode(b)==tok.decode(c),'Actual cold tokenizer encode/decode differs')
 messages=[dict(role='system',content='Answer clearly.'),dict(role='user',content='你好，what is 2+2?')]
 for add_generation_prompt in (False,True):
  for tokenize in (False,True):
   require(base_tok.apply_chat_template(messages,tokenize=tokenize,add_generation_prompt=add_generation_prompt)==tok.apply_chat_template(messages,tokenize=tokenize,add_generation_prompt=add_generation_prompt),'Actual chat rendering/token IDs differ')
 tokenizer=dict(status='ACTUAL_COLD_TOKENIZER_IDENTITY_VERIFIED',vocab_equal=True,special_pad_bos_eos_unk_equal=True,all_special_tokens_ids_equal=True,chat_template_equal=True,actual_encode_decode_probes=len(probes),canonical_backend_equal=True,full_added_token_attributes_equal=True,full_special_roles_equal=True,normalized_semantic_config_equal=True,actual_chat_template_comparisons=4)
 write_once(out/'tokenizer-cold-reload.json',tokenizer)
 require(identity(BASE)==base_before and identity(CHECKPOINT)==before,'Base/checkpoint changed during cold verification')
 bindings_after={name:file_sha(path) for name,path in binding_paths.items()}
 require(bindings_after==bindings_before,'Wrapper/config/package/verifier/native/terminal evidence changed during cold verification')
 require(bindings_after['resolved_config']==CONFIG and bindings_after['package_manifest']==PACKAGE_SHA and bindings_after['frozen_verifier']==VERIFIER_SHA,'Pinned config/package/verifier changed')
 require(read_json(out/'full-HF-cold-reload.json')[1]==full_sha,'Full cold report changed during final input checks')

 require(all(file_sha(Path(path))==digest for path,digest in native_files.items()),'Saved native file changed during verification')
 require(chain.restore_and_fault_evidence()==chain_evidence,'Source ancestry/fault evidence changed during verification')
 receipt=dict(status='NESTED_TRAIN200_HF_TOKENIZER_NATIVE_COLD_READ_IDENTITY_BOUND',job_id=chain.TRAIN_JOB,
     run_id=RUN.name,source_commit=SOURCE,config_sha256=CONFIG,package_manifest_sha256=package_sha,
     verifier_wrapper_source_commit=code['verifier_wrapper_source_commit'],code_manifest_sha256=code_sha,
     native_manifest_sha256=native_sha,native5_loaded_receipt_sha256=chain.RESTORE_RECEIPT_SHA,
     effective_training_segments=chain.effective_segments(),restore_and_fault_evidence=chain_evidence,
     input_hashes_before=bindings_before,input_hashes_after=bindings_after,native_file_sha256=native_files,
     native_cpu_report_sha256=file_sha(out/'native200-cpu-state.json'),hf_identity_sha256=before['sha256'],
     full_hf_report_sha256=full_sha,tokenizer_report_sha256=file_sha(out/'tokenizer-cold-reload.json'),
     wrapper_sha256=bindings_before['wrapper'],frozen_full_verifier_sha256=VERIFIER_SHA,
     actual_completed_training_outer=200,actual_native_cpu_deserialization=True,all_native_saved_state_finite=True,
     all_parameters_finite=True,shape_keys_full_checked=True,actual_cpu_model_cold_reload=True,
     actual_tokenizer_cold_reload=True,total_parameters=full['total_parameters'],changed_parameters=full['changed_parameters'],
     max_abs_parameter_delta=full['max_abs_parameter_delta'],cpu_forward_logits_finite=True,
     wall_seconds=time.time()-started,wrapper_process_cpu_seconds=time.process_time()-cpu_started,
     wrapper_peak_self_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
     full_hf_checker_process_cpu_seconds=full['process_cpu_seconds'],
     full_hf_checker_wall_seconds=full['wall_seconds'],
     full_hf_checker_peak_self_rss_bytes=full['peak_self_rss_bytes'],
     weights_left_on_h100_10=True,training_or_optimizer_restored=False,training_signals_sent=False,
     native_gpu_cold_world_restore_performed=False,native_hf_parameter_equivalence_checked=False,
     final51713_scores_accepted=False,independent_final_acceptance=False)
 write_once(out/'identity-bound-final-hf.json',receipt);print(json.dumps(receipt,allow_nan=False))

if __name__=='__main__':main()
