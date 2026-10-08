"""Original native concat and actual submitted manager; known optional field only."""
import copy,hashlib,json,random,types
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from verl import DataProto
from verl.experimental.agent_loop.agent_loop import CoSearchAgentLoopManager
from verl.trainer.ppo.search_r1_reranker_ray_trainer import CoSearchRayTrainer
import verl.experimental.agent_loop.agent_loop as am
import observe_worker,empty_ranker,optional_metadata

records=[];raw=[]
record=lambda k,v:records.append((k,v))
preserve=lambda k,v:raw.append((k,v))
def batch(n=512,field='missing',offset=0):
 non={'uid':np.array([str(offset+i) for i in range(n)],dtype=object),'__num_turns__':np.ones(n,dtype=np.int32)}
 if field!='missing':
  a=np.empty(n,dtype=object)
  for i in range(n):a[i]=None if field=='none' or i%2==0 else [{'query':str(offset+i)}]
  for key in optional_metadata.KEYS:non[key]=copy.deepcopy(a)
 return DataProto.from_dict(tensors={'prompts':torch.ones(n,2,dtype=torch.long),'responses':torch.ones(n,2,dtype=torch.long),'response_mask':torch.ones(n,2,dtype=torch.long),'input_ids':torch.ones(n,4,dtype=torch.long),'attention_mask':torch.ones(n,4,dtype=torch.long),'position_ids':torch.arange(4).repeat(n,1),'rm_scores':torch.zeros(n,2)},non_tensors=non,meta_info={'metrics':[{'generate_sequences':1.,'tool_calls':.5} for _ in range(n)],'reward_extra_keys':[]})
def equal(a,b):
 assert set(a.batch.keys())==set(b.batch.keys()) and set(a.non_tensor_batch)==set(b.non_tensor_batch)
 for k in a.batch.keys():assert torch.equal(a.batch[k],b.batch[k])
 for k in a.non_tensor_batch:assert a.non_tensor_batch[k].tolist()==b.non_tensor_batch[k].tolist()
inputs=[batch(field='details' if i<6 else 'missing',offset=i*512) for i in range(8)]
try:DataProto.concat(inputs)
except AssertionError as e:assert 'tool_call_details length 3072' in str(e) and '4096' in str(e)
else:raise AssertionError('original shortage not reproduced')
state=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone());before=copy.deepcopy(inputs)
out=optional_metadata.concat_main(inputs,record,preserve);assert len(out)==4096
assert random.getstate()==state[0] and np.array_equal(np.random.get_state()[1],state[1][1]) and torch.equal(torch.get_rng_state(),state[2])
for a,b in zip(inputs,before):equal(a,b)
expected=copy.deepcopy(before)
for x in expected:
 for key in optional_metadata.KEYS:
  if key not in x.non_tensor_batch:x.non_tensor_batch[key]=np.full(len(x),None,dtype=object)
equal(out,DataProto.concat(expected));assert all(v['missing_rows']==1024 and v['present_explicit_none_rows']==1536 for v in records[-1][1]['fields'].values())
assert [x['fields']['tool_call_details']['field_present'] for x in raw[-1][1]]==[True]*6+[False]*2
for field in ['missing','none','details']:
 a=[batch(4,field),batch(4,field,4)];p0=len(raw);r0=len(records)
 equal(optional_metadata.concat_main(a,record,preserve),DataProto.concat(a));assert len(raw)==p0 and len(records)==r0
# Existing malformed value is never treated as a missing shard.
for key in optional_metadata.KEYS:
 for malformed in [None,np.ones(4,dtype=np.int64),np.empty((4,1),dtype=object),np.empty(3,dtype=object)]:
  bad=batch(4,'details');bad.non_tensor_batch[key]=malformed
  try:optional_metadata.concat_main([bad,batch(4)],record,preserve)
  except AssertionError:pass
  else:raise AssertionError('malformed present accepted')
# Required training metadata/tensors remain subject to original concat errors.
for kind in ['uid','rm_scores']:
 bad=batch(4,'details')
 if kind=='uid':del bad.non_tensor_batch['uid']
 else:del bad.batch['rm_scores']
 try:optional_metadata.concat_main([bad,batch(4)],record,preserve)
 except (AssertionError,KeyError,RuntimeError):pass
 else:raise AssertionError('required field silently repaired')
for stage in ['preserve','record']:
 def fail(*a):raise OSError(stage)
 try:optional_metadata.concat_main([batch(4,'none'),batch(4)],fail if stage=='record' else record,fail if stage=='preserve' else preserve)
 except OSError as e:assert str(e)==stage
 else:raise AssertionError('evidence failure concealed')

# Execute actual source-pinned manager installation (both training and validation).
manager=CoSearchAgentLoopManager.__new__(CoSearchAgentLoopManager)
manager.config=OmegaConf.create({'actor_rollout_ref':{'rollout':{'free_cache_engine':True}},'trainer':{'reranker_sampling_val_start_step':10000}})
manager.reward_model_manager=None;wake=[];manager.wake_up=lambda:wake.append(1);manager.sleep=lambda:wake.append(0)
manager._aggregate_reranker_metrics=lambda ms:{'post_filter/num_groups':sum(m.get('post_filter/num_groups',0) for m in ms)}
empty_ranker.install_controller(manager,CoSearchRayTrainer,record)
class Remote:
 def __init__(self,v):self.v=v
 def remote(self,*a):return self.v
oldget=am.ray.get;oldpreserve=observe_worker.preserve_raw_payload
am.ray.get=lambda v:v;observe_worker.preserve_raw_payload=preserve
try:
 for validation in [False,True]:
  main=copy.deepcopy(before)
  if validation:results=main
  else:
   results=[]
   for i,m in enumerate(main):
    r=batch(4,offset=i*4);r.meta_info['worker_reranker_metrics']={'post_filter/num_groups':1};results.append((m,r))
  manager.agent_loop_workers=[types.SimpleNamespace(generate_sequences=Remote(x)) for x in results]
  prompt=DataProto.from_dict(tensors={'input_ids':torch.ones(4096,2,dtype=torch.long)},meta_info={'validate':validation,'global_steps':13})
  result=manager.generate_sequences(prompt);m=result if validation else result[0]
  equal(m,out)
  assert len(m)==4096
  if not validation:assert len(result[1])==32 and result[1].meta_info['reranker_rollout_metrics']['post_filter/num_groups']==8
 assert wake==[1,0,1,0]
finally:am.ray.get=oldget;observe_worker.preserve_raw_payload=oldpreserve
assert not torch.cuda.is_initialized()
output={'passed':True,'actual_original_concat_3072_vs4096_reproduced':True,'known_optional_fields':list(optional_metadata.KEYS),'actual_manager_train_and_validation':True,'main_training_tensors_rewards_UID_unchanged':True,'missing_vs_explicit_None_evidence':True,'malformed_present_rejections':16,'required_field_rejections':2,'evidence_errors_propagated':2,'allpresent_allabsent_parity':True,'RNG_unchanged':True,'CUDA_initialized':False,'PPO_updates':0,'optional_sha256':hashlib.sha256(Path(optional_metadata.__file__).read_bytes()).hexdigest(),'empty_sha256':hashlib.sha256(Path(empty_ranker.__file__).read_bytes()).hexdigest()}
Path(__file__).with_name('cpu-optional-metadata-gate.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output),flush=True)
