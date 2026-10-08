import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['TREE_FIXED_ACTOR_PROFILE']='0'
import torch,pathlib,json,hashlib,importlib.util,time
assert not torch.cuda.is_initialized();torch.set_num_threads(4)
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-recovery300-3234/native-fullstate')
source=pathlib.Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/candidates/recovery-502e9fa-f91c4f17/source/repo/search_r1/baselines/training_state.py')
spec=importlib.util.spec_from_file_location('review_state',source);st=importlib.util.module_from_spec(spec);spec.loader.exec_module(st)
start=time.monotonic();rows=[]
for rank in range(8):
 rec=json.loads((root/f'rank-{rank:02d}.json').read_bytes());diag=json.loads((root/f'rank-{rank:02d}-diagnostic.json').read_bytes())
 states={};shas={}
 for label,name in [('saved5',f'rank-{rank:02d}.pt'),('uninterrupted',f'rank-{rank:02d}-uninterrupted6.pt'),('replayed',f'rank-{rank:02d}-replayed6.pt')]:
  path=root/name;shas[label]=st.file_sha(path);v=torch.load(path,map_location='cpu',weights_only=False);st.finite(v)
  assert v['identity']==rec['identity'];assert v['version']==2
  steps=sorted(set(int(s['step']) for s in v['optimizer']['state'].values()));assert steps==[120 if label=='saved5' else 130]
  assert v['scheduler']['last_epoch']==(5 if label=='saved5' else 6)
  states[label]=st.digest(v)
  del v
 assert shas['saved5']==rec['sha256'] and states['saved5']==rec['logical_state_sha256']
 assert states['uninterrupted']==states['replayed']==diag['uninterrupted_digest']==diag['replayed_digest'] and diag['differences']==[]
 rows.append(dict(rank=rank,actual_tensor_file_sha256=shas,actual_logical_digest=states,all_serialized_finite=True,adam_steps=[120,130],scheduler_steps=[5,6],complete_replay_equal=True))
print(json.dumps(dict(status='INDEPENDENT_ACTUAL8_NATIVE_TINY_SERIALIZED_STATES_CPU_VERIFIED',job_id='3234',rows=rows,wall_seconds=time.monotonic()-start,cuda_initialized=torch.cuda.is_initialized(),limits='Reads only already produced tiny native probe tensor snapshots; not actual3B checkpoint100 or105 acceptance')),flush=True)
