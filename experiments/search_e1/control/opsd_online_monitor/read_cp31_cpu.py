"""Independent read-only CPU audit of permanent online CP31; no model or CUDA entry."""
from pathlib import Path
import os, sys, json, hashlib, math, time
B=Path('/data/shaoshitong/search-e1-opsd-online-archive100-8gpu-20261008')
HELPER=B/'source/revision-7cf2853/opsd_online'
RUN='1d1d67b60f0e46a2afc0536a36ab69b1'
CONFIG_SHA='d0c30df9b035e73d87bb5402bbcd32c16b2314a4180c451a543522ca5b321b17'
CP31_SHA='078c569e2cd5ff009f901a3185e7f6e6e949937d30b52e03ab7707dca395829d'
CP5_ADAPTER_DIGEST='af9fc6fa6a5fbc2f0568604bce313351218621f9ab9284b0f864c1ce139939c9'
SOURCE_MANIFEST_SHA='66cb43dd0830b9a7776bd644a0ef5bbf3d1661a3db3b159cc9e781db582c5233'
started=time.time()
def require(ok,message):
 if not ok: raise ValueError(message)
def stat_from(s):
 return dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
def stable_bytes(path, expected_sha=None):
 path=Path(path)
 with path.open('rb') as handle:
  before=stat_from(os.fstat(handle.fileno()));raw=handle.read();after=stat_from(os.fstat(handle.fileno()))
 require(before==after==stat_from(path.stat()),'Artifact mutated/replaced while reading '+str(path))
 digest=hashlib.sha256(raw).hexdigest()
 if expected_sha:require(digest==expected_sha,'SHA mismatch '+str(path))
 return raw,dict(path=str(path),sha256=digest,**before)
require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU audit requires explicitly empty CUDA mask')
require(os.geteuid()==1005,'Expected HKUST1 effective UID1005')
raw, config_proof=stable_bytes(B/'receipts/config.json',CONFIG_SHA);config=json.loads(raw)
config['_config_sha256']=config_proof['sha256'] # In-memory binding only; never write config.
raw, helper_proof=stable_bytes(HELPER/'source-manifest.json',SOURCE_MANIFEST_SHA);helper_manifest=json.loads(raw)
source_proofs={}
for name in ('checkpoint.py','trainer.py','objective.py'):
 _,proof=stable_bytes(HELPER/name,helper_manifest['files'][name]['sha256']);source_proofs[name]=proof
sys.path.insert(0,str(HELPER))
from checkpoint import check_checkpoint, checked_json, checked_file, fileproof
from trainer import verify_rank_state, tensor_digest, optimizer_tensor_digest, actual_adam_state
import torch
from safetensors.torch import load as load_safetensors
cuda_before=torch.cuda.is_initialized();require(not cuda_before,'CPU reader unexpectedly initialized CUDA')
torch.set_num_threads(1)
cpref=dict(path=str(B/'runs/checkpoints/update-00000031-outer-00002/manifest.json'),sha256=CP31_SHA)
cp,checkpoint_fileproofs=check_checkpoint(cpref,config,RUN,31,2)
require('outer_end' in cp['protection_reason'],'CP31 not protected as completed outer end')
require(cp['cursor']['next_pair_cursor']==cp['cursor']['usable_pair_count']==292,'CP31 cursor not completed292')
require(cp['cumulative_pair_exposure']==925,'Wrong cumulative exposure')

def recursive_equal(a,b,path='root'):
 if isinstance(a,torch.Tensor):
  require(isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b),'Cross-rank tensor mismatch '+path)
 elif isinstance(a,dict):
  require(isinstance(b,dict) and set(a)==set(b),'Cross-rank dict inventory mismatch '+path)
  for k in a:recursive_equal(a[k],b[k],path+'/'+str(k))
 elif isinstance(a,(list,tuple)):
  require(type(a)==type(b) and len(a)==len(b),'Cross-rank sequence mismatch '+path)
  for i,(av,bv) in enumerate(zip(a,b)):recursive_equal(av,bv,path+'/'+str(i))
 else:require(type(a)==type(b) and a==b,'Cross-rank scalar/group mismatch '+path)

def fixed_handle_state(ref):
 p=Path(ref['path'])
 with p.open('rb') as handle:
  before=stat_from(os.fstat(handle.fileno()));h=hashlib.sha256()
  for block in iter(lambda:handle.read(4<<20),b''):h.update(block)
  require(h.hexdigest()==ref['sha256'],'State content SHA mismatch '+str(p))
  for k,v in before.items():require(ref[k]==v,'State fixed inode/stat mismatch '+str(p)+' '+k)
  handle.seek(0);state=torch.load(handle,map_location='cpu',weights_only=False)
  after=stat_from(os.fstat(handle.fileno()))
  require(before==after==stat_from(p.stat()),'State changed while CPU loading '+str(p))
 return state,dict(path=str(p),sha256=h.hexdigest(),**before)
rank_results=[];baseline=None
for row in sorted(checkpoint_fileproofs['ranks'],key=lambda q:q['rank']):
 rank=row['rank'];state,proof=fixed_handle_state(row['file'])
 verify_rank_state(state,cp,rank,config)
 require(len(state['optimizer']['state'])==504,'Expected504 actual Adam parameters')
 actual_adam_state(state['optimizer'],31,504)
 finite=0
 for entry in state['optimizer']['state'].values():
  for name,t in entry.items():
   require(t.device.type=='cpu' and torch.isfinite(t).all().item(),'Nonfinite/nonCPU actual saved Adam '+name)
   finite+=1
 if baseline is None:baseline=state['optimizer']
 else:recursive_equal(baseline,state['optimizer'])
 rank_results.append(dict(rank=rank,file=proof,actual_state_parameters=504,actual_adam_steps=sorted(set(float(q['step']) for q in state['optimizer']['state'].values())),finite_optimizer_tensors=finite,full_step_moments_and_param_groups_equal_rank0=True,optimizer_digest=optimizer_tensor_digest(state['optimizer']),adapter_digest=state['adapter_tensor_sha256'],rng_cuda_state_count=len(state['rng']['cuda_all']),rng_restored_or_sampled=False))
 del state

def adapter_from_manifest(manifest):
 ref=next(q for q in manifest['adapter']['files'] if q['path'].endswith('/adapter_model.safetensors'))
 raw,proof=stable_bytes(ref['path'],ref['sha256']);tensors=load_safetensors(raw)
 require(len(tensors)==504 and all(t.device.type=='cpu' and torch.isfinite(t).all().item() for t in tensors.values()),'Adapter tensor inventory/finite check failed')
 digest=tensor_digest(tensors)
 require(digest==manifest['all_rank_adapter_sha256'][0],'Actual exported adapter differs from full-state manifest')
 return tensors,proof,digest
current,current_proof,current_digest=adapter_from_manifest(cp)
raw,cp5proof=stable_bytes(B/'runs/checkpoints/update-00000005-outer-00000/manifest.json');cp5=json.loads(raw)
require(cp5['run_id']==RUN and cp5['config_sha256']==CONFIG_SHA and cp5['optimizer_updates']==5 and cp5['source_model_identity']==cp['source_model_identity'],'Wrong preservedCP5 identity')
require(cp5['all_rank_adapter_sha256'][0]==CP5_ADAPTER_DIGEST,'CP5 no longer matches previous independent CPU accepted adapter')
prior,prior_proof,prior_digest=adapter_from_manifest(cp5)
require(set(current)==set(prior),'Adapter tensor names changed')
changed=0;maximum=0.;groups={}
for name,t in current.items():
 p=prior[name];require(t.dtype==p.dtype and t.shape==p.shape,'Adapter shape/dtype changed '+name)
 delta=(t-p).abs().max().item();maximum=max(maximum,delta)
 changed+=int(not torch.equal(t,p));role='LoRA_A' if 'lora_A' in name else 'LoRA_B' if 'lora_B' in name else 'other'
 group=groups.setdefault(role,dict(tensors=0,changed=0,max_absolute_delta=0.))
 group['tensors']+=1;group['changed']+=int(not torch.equal(t,p));group['max_absolute_delta']=max(group['max_absolute_delta'],delta)
require(changed>0 and maximum>0,'No actual adapter change afterCP5')

# Check all native summary receipts for this completed outer. Only the four
# latest tail token files are read; prior raw token files are counted by path.
updates=[];all_token_refs=[];all_pairs=[];last=None
for ref in cp['outer_update_receipts']:
 value,proof=checked_json(ref);require(value['accepted'] is True and value['run_id']==RUN and value['outer_index']==2 and value['loss_reduction']=='trajectory_mean','Wrong native update summary')
 pairs=[pair for rank in value['all_rank_actual_token_metrics'] for pair in rank['pairs']]
 require(len(value['all_rank_actual_token_metrics'])==8 and {q['rank'] for q in value['all_rank_actual_token_metrics']}==set(range(8)),'Missing rank summary')
 require(len(pairs)==value['global_actual_pairs'] and sum(q['valid_tokens'] for q in pairs)==value['global_valid_tokens'],'Native pair/token count mismatch')
 require(all(math.isfinite(float(rank['grad_norm'])) for rank in value['all_rank_actual_token_metrics']),'Nonfinite actual native gradnorm')
 require(all(math.isfinite(float(pair[k])) for pair in pairs for k in ('raw_loss','post_loss')),'Nonfinite native token loss summary')
 for pair in pairs:
  p=Path(pair['token_log']['path']);require(p.is_file() and p.stat().st_size==pair['token_log']['bytes'],'Missing native token log');all_token_refs.append(pair['token_log']);all_pairs.append(pair['pair_index'])
 loss_pre=sum(q['raw_loss'] for q in pairs)/len(pairs);loss_post=sum(q['post_loss'] for q in pairs)/len(pairs)
 updates.append(dict(optimizer_updates=value['optimizer_updates'],receipt=proof,global_actual_pairs=value['global_actual_pairs'],global_valid_tokens=value['global_valid_tokens'],trajectory_mean_pre_clip_kl=loss_pre,trajectory_mean_post_clip_loss=loss_post,clipped_vocabulary_count=sum(q['clipped_vocabulary_count'] for q in pairs),learning_rate=value['learning_rate'],grad_norm_unique=sorted(set(q['grad_norm'] for q in value['all_rank_actual_token_metrics'])),all_native_pair_loss_metrics_finite=True))
 last=value
require([v['optimizer_updates'] for v in updates]==list(range(22,32)),'Committed native updates not continuous22..31')
require(sum(v['global_actual_pairs'] for v in updates)==292 and len(all_token_refs)==len({r['path'] for r in all_token_refs})==292 and len(set(all_pairs))==292,'292 pair traces/order exposure inventory differs')
sampled=[]
for rank in last['all_rank_actual_token_metrics']:
 for pair in rank['pairs']:
  raw,proof=stable_bytes(pair['token_log']['path'],pair['token_log']['sha256']);lines=raw.splitlines();header=json.loads(lines[0]);records=[json.loads(line) for line in lines[1:]]
  require(header['run_id']==RUN and header['outer_index']==2 and header['optimizer_update']==31 and header['component_clip']==10 and header['loss_scope']=='actual_training_forward_before_optimizer','Wrong latest actual token header')
  require(len(records)==pair['valid_tokens']==sum(header['policy_mask']),'Tail trajectory masked token count mismatch')
  require({r['token_index'] for r in records}=={i for i,v in enumerate(header['policy_mask']) if v},'Tail token positions duplicate/missing')
  for r in records:
   require(r['policy_mask'] and r['token_id']==header['token_ids'][r['token_index']],'Tail token/mask provenance differs')
   for k in ('pre_clip_full_vocabulary_kl','post_clip_full_vocabulary_loss','clipped_teacher_probability_mass','trajectory_contribution','global_batch_contribution'):
    require(math.isfinite(r[k]),'Nonfinite actual token metric')
   require(r['clipped_vocabulary_count']>=0,'Invalid clipped contribution count')
   require(math.isclose(r['trajectory_contribution'],r['post_clip_full_vocabulary_loss']/len(records),rel_tol=1e-12,abs_tol=1e-12),'Trajectory mean normalization mismatch')
   require(math.isclose(r['global_batch_contribution'],r['trajectory_contribution']/last['global_actual_pairs'],rel_tol=1e-12,abs_tol=1e-12),'Actual global tail denominator mismatch')
  require(math.isclose(sum(r['pre_clip_full_vocabulary_kl'] for r in records)/len(records),pair['raw_loss'],rel_tol=1e-9,abs_tol=1e-9),'Raw token sum differs from native summary')
  require(math.isclose(sum(r['post_clip_full_vocabulary_loss'] for r in records)/len(records),pair['post_loss'],rel_tol=1e-9,abs_tol=1e-9),'Postclip token sum differs from native summary')
  require(sum(r['clipped_vocabulary_count'] for r in records)==pair['clipped_vocabulary_count'],'Tail clipped contribution count mismatch')
  sampled.append(dict(rank=rank['rank'],pair_index=pair['pair_index'],file=proof,valid_tokens=len(records),all_actual_record_metrics_finite=True,token_ids_and_policy_masks_exact=True,trajectory_and_global_tail_denominators_exact=True,post_loss=pair['post_loss'],raw_loss=pair['raw_loss'],clipped_vocabulary_count=pair['clipped_vocabulary_count']))
require(len(sampled)==4 and last['global_actual_pairs']==4,'Expected four real tail trajectories at update31')
raw,statusproof=stable_bytes(B/'runs/status.json');status=json.loads(raw)
cuda_after=torch.cuda.is_initialized();require(not cuda_after,'Audit initialized CUDA')
output=dict(schema='opsd-online-protected-cp31-independent-CPU-audit-v1',accepted=True,started_unix=started,ended_unix=time.time(),run_id=RUN,config=config_proof,config_binding_scope='Actual config rawSHA inserted only as _config_sha256 in in-memory dict; disk config unchanged.',source_helper_manifest=helper_proof,source_helper_modules=source_proofs,checkpoint=checkpoint_fileproofs['manifest'],optimizer_updates=31,cursor=cp['cursor'],cumulative_pair_exposure=925,protection_reason=cp['protection_reason'],ranks=rank_results,adapter=dict(current_file=current_proof,baseline5_file=prior_proof,baseline5_manifest=cp5proof,actual_tensor_count=504,all_finite=True,current_tensor_digest=current_digest,baseline5_tensor_digest=prior_digest,changed_vs_checkpoint5=changed,max_absolute_delta_vs_checkpoint5=maximum,per_role=groups),outer2_native_update_summaries=updates,actual_native_training_token_trace_count=292,actual_trace_inventory_scope='All292 committed token file paths exist with expected size and unique path; no fullSHA/all-record reread except the four update31 tail trajectories below.',tail_update31_actual_token_readbacks=sampled,token_readback_scope='All records of the four latest tail trajectory files only; matches native headers/summary/masks and per-trajectory then actual4 global mean. No model/logit reevaluation, no entire earlier trace validation or inference claimed.',current_status=dict(proof=statusproof,run_id=status.get('run_id'),phase=status.get('phase'),outer_index=status.get('outer_index'),optimizer_updates=status.get('optimizer_updates'),updated_at_unix=status.get('updated_at_unix')),cuda_initialized_before=cuda_before,cuda_initialized_after=cuda_after,no_model_loaded=True,no_optimizer_created_or_mutated=True,no_global_or_production_rng_set_or_restored=True,rng_validation_scope='Saved RNG containers checked with private CPU generator instances; no production or global Torch CPU/CUDA RNG restored.',no_node_output_written=True,no_http_gpu_signals_git=True,scope='Fresh CPU bytes/inodes/hash and actual8rank31 Adam full tensors/groups/readback; finite changed actual504 adapter vs acceptedCP5. Does not claim GRPOAdam restoration, new native512 evaluation, effectiveness or complete200outer run.')
print(json.dumps(output,sort_keys=True,indent=2))
