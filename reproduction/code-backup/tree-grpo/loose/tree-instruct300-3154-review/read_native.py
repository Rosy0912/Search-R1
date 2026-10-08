from pathlib import Path
import json,hashlib,time,socket
import torch
from search_r1.baselines.training_state import digest,finite
root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-instruct300-3154/native-fullstate');rows=[]
for rank in range(8):
 states={};files=[]
 for suffix in ['','-uninterrupted6','-replayed6']:
  p=root/('rank-%02d'%rank+suffix+'.pt');b=p.read_bytes();v=torch.load(p,map_location='cpu',weights_only=False);finite(v)
  assert v['version']==2 and v['identity']==dict(rank=rank,world_size=8,job='3154',code='f994af33c6c93b6b9844863be912cab2a4020dc1')
  assert v['local_shards'].keys()==v['parameters'].keys()
  assert all(torch.equal(v['local_shards'][n],t) for n,t in v['parameters'].items())
  step=5 if suffix=='' else 6;assert v['scheduler']['last_epoch']==step
  assert all(t['step'].item()==step*24 for t in v['optimizer']['state'].values())
  states[suffix]=digest(v);files.append(dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b),logical_sha256=states[suffix]))
 rec=json.loads((root/('rank-%02d.json'%rank)).read_bytes());d=json.loads((root/('rank-%02d-diagnostic.json'%rank)).read_bytes())
 assert states['']==rec['logical_state_sha256'] and files[0]['sha256']==rec['sha256']
 assert states['-uninterrupted6']==states['-replayed6']==d['uninterrupted_digest']==d['replayed_digest'] and d['differences']==[]
 rows.append(dict(rank=rank,actual_serialized_states_finite=True,step5_adam=120,step6_adam=144,exact_replay=True,files=files))
print(json.dumps(dict(status='ACTUAL_NATIVE8_SERIALIZED_FULLSTATE_EXACT_REPLAY_PASS',hostname=socket.gethostname(),epoch=time.time(),torch=torch.__version__,state_module=__import__('search_r1.baselines.training_state',fromlist=['']).__file__,ranks=rows)))
