import os,json,gzip,hashlib,time,resource,math,collections,gc
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from pathlib import Path
torch.set_num_threads(4);assert not torch.cuda.is_initialized()
root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-safety300-3268/output');start=time.perf_counter();cpu=time.process_time();proof=[];ranks={};results=[]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(8*1024**2),b''):h.update(c)
 return h.hexdigest()
def read(p):
 raw=p.read_bytes();proof.append(dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw)));return json.loads(raw)
def finite(x):
 if isinstance(x,float):assert math.isfinite(x)
 elif isinstance(x,dict):
  for y in x.values():finite(y)
 elif isinstance(x,list):
  for y in x:finite(y)
for step in [111]:
 norms=[];ratio=[]
 assert not list((root/'tree-rank-audit').glob('step-*.json')) and not list((root/'tree-rank-audit/ratio-tail').glob('*'))
 rolloutpath=next(p for p in (root/'tree-audit').glob(f'step_{step}_rollout_*.json') if 'policy_batch' not in p.name);roll=read(rolloutpath);pol=read(rolloutpath.with_name(rolloutpath.stem+'_policy_batch.json'))
 assert pol['rollout_file_sha256']==sha(rolloutpath) and roll['leaf_count']==pol['leaf_count']==3840
 assert len(roll['question_groups'])==640 and all(g['roots']==2 and g['leaves']==6 for g in roll['question_groups'])
 assert all(b['inherited'] is True and b['parent_fields']==b['child_fields'] and len(b['parent_fields'])==16 for b in roll['branches'])
 assert all(n==2 for n in collections.Counter(b['source'] for b in roll['branches']).values())
 assert all(l['context_matches'] and l['generation_prefix_hash']==l['scoring_prefix_hash'] for l in roll['leaves'])
 assert all(abs(row['policy_weight_sum']-(0 if row['empty_leaf'] else 1))<1e-6 for row in pol['rows'])
 directory=root/f'trajectories/outer-{step:06d}';m=read(directory/'manifest.json');assert m['rows']==3840 and m['identity']==dict(outer_step=step,job_id='3268',source='a632ba91840fb116aacfc0e770681a860647b295',config='e25214a90cf6ac063174787b7aa2c607ef5c942218fd31c2af0cfdf2c206b71b',data='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24')
 for name,expected in m['payloads'].items():
  p=directory/name;actual=sha(p);assert actual==expected;proof.append(dict(path=str(p),sha256=actual,bytes=p.stat().st_size))
 payload=torch.load(directory/'actor-inputs.pt',map_location='cpu',weights_only=False);assert payload['identity']==m['identity'] and payload['rows']==3840
 t=payload['tensors'];meta=payload['non_tensor_batch'];width=t['responses'].shape[1];effective=t['attention_mask'][:,-width:]*t['loss_mask'];info=t['info_mask'][:,-width:]
 assert torch.equal(t['loss_mask'],info)
 count=effective.sum(-1);expected=effective/count.clamp(min=1)[:,None];assert torch.allclose(t['policy_weights'],expected,atol=1e-8,rtol=1e-6)
 for key in ['old_log_probs','ref_log_prob','advantages']:
  assert torch.isfinite(t[key][effective.bool()]).all()
 assert torch.isfinite(t['token_level_rewards']).all()
 score=t['token_level_rewards'].sum(-1);adv=torch.zeros_like(score)
 for keys in [meta['uid'],list(zip(meta['uid'],meta['tree_uid']))]:
  groups=collections.defaultdict(list)
  for i,key in enumerate(keys):groups[key].append(i)
  for indices in groups.values():
   v=score[indices];adv[indices]+=(v-v.mean())/(v.std(correction=1)+1e-6)
 assert torch.allclose(t['advantages'],adv[:,None]*effective,atol=1e-6,rtol=1e-6)
 uidcounts=collections.Counter(meta['uid']);assert len(uidcounts)==640 and set(uidcounts.values())=={6}
 for uid in uidcounts:
  assert sorted(collections.Counter(meta['tree_uid'][i] for i,u in enumerate(meta['uid']) if u==uid).values())==[3,3]
 assert len(set(meta['trajectory_uid']))==3840
 reward=0;actions=0;searches=0
 with gzip.open(directory/'trajectories.jsonl.gz','rt') as stream:
  for i,line in enumerate(stream):
   x=json.loads(line);assert x['row']==i and x['identity']==m['identity']
   assert x['metadata']=={k:v[i] for k,v in meta.items()}
   assert x['response_token_ids']==t['responses'][i].tolist()
   assert x['token_ids']==t['input_ids'][i][t['attention_mask'][i].bool()].tolist()
   assert x['reward']==float(t['token_level_rewards'][i].sum());reward+=x['reward']
   records=x['metadata']['action_records'];actionids=[];obsids=[];timeline=[];masktimeline=[]
   for a in records:
    assert a['scoring_context_matches_generation'] is True and 'raw_generation_token_ids' in a and 'observation_text' in a
    assert a['action_tokens']==len(a['action_token_ids']) and a['observation_tokens']==len(a['observation_token_ids'])
    actionids+=a['action_token_ids'];obsids+=a['observation_token_ids'];timeline+=a['action_token_ids']+a['observation_token_ids'];masktimeline += [1]*a['action_tokens']+[0]*a['observation_tokens'];searches+=int(a['valid_search']);actions+=1
   retained=t['responses'][i][t['attention_mask'][i,-width:].bool()].tolist()
   assert timeline==retained
   assert effective[i][t['attention_mask'][i,-width:].bool()].tolist()==masktimeline
   assert int(effective[i].sum())==len(actionids)
  assert i+1==3840
 results.append(dict(observed_step=step,actual_new_completed_outers=0,actual_optimizer_writes=0,trajectory_rows=3840,mean_reward=reward/3840,records=actions,retained_leaf_search_records=searches,all_actor_effective_inputs_finite=True,branch_fields=16,branches=len(roll['branches'])))
 del payload,t,meta,roll,pol,effective,expected;gc.collect()
print(json.dumps(dict(status='ACTUAL_CALIBRATION_3840_TREE_TRAJECTORY_INPUTS_VERIFIED_ZERO_WRITE_ONLY',records=results,evidence=proof,wall_seconds=time.perf_counter()-start,process_cpu_seconds=time.process_time()-cpu,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,torch_threads=4,cuda_initialized=torch.cuda.is_initialized())),flush=True)
