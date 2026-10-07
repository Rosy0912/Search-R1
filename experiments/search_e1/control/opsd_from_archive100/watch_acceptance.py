from pathlib import Path
import json,time,torch,hashlib
from safetensors import safe_open
b=Path('/data/shaoshitong/search-e1-opsd-from-archive100-8gpu-20261007');out=b/'runs/opsd'
while not (out/'five-update-acceptance.json').exists():time.sleep(10)
a=json.loads((out/'five-update-acceptance.json').read_text());assert a['world_size']==8 and a['opsd_updates']==5 and a['source_grpo_step']==100
assert sorted(r['rank'] for r in a['ranks'])==list(range(8))
assert all(r['finite'] and r['changed_from_initial'] and r['adam_steps']==[5.] for r in a['ranks'])
files=[out/f'adapter_update_{s}/adapter_model.safetensors' for s in [1,5]]
with safe_open(str(files[0]),framework='pt',device='cpu') as one,safe_open(str(files[1]),framework='pt',device='cpu') as five:
 assert set(one.keys())==set(five.keys());changed=[]
 for k in five.keys():
  v=five.get_tensor(k);assert torch.isfinite(v).all(),k
  u=one.get_tensor(k)
  if not torch.equal(u,v):changed.append(dict(key=k,max_abs_delta=float((v.float()-u.float()).abs().max())))
 assert changed,'No adapter weights changed between actual updates1 and5'
optimizer=[]
for rank in range(8):
 p=out/f'update_5_rank_{rank}_state.pt';r=torch.load(p,map_location='cpu',weights_only=False);assert r['rank']==rank and r['updates']==5
 for state in r['optimizer']['state'].values():
  for key,v in state.items():
   if torch.is_tensor(v):assert torch.isfinite(v).all(),(rank,key)
 steps=sorted(set(float(v['step']) for v in r['optimizer']['state'].values() if 'step' in v));assert steps==[5.]
 optimizer.append(dict(rank=rank,steps=steps,finite=True,path=str(p),bytes=p.stat().st_size));del r
m=json.loads((b/'receipts/memory-continuous.json').read_text());assert m['minimum_reserve_mib']>=1024,'Less than1GiB sampled total-card reserve; do not certify GPU headroom'
report=dict(status='ACCEPTED',time=time.time(),source_grpo_step=100,actual_opsd_updates=5,display_step=105,continuous_run=True,source_model=a['source_model'],global_trajectory_batch=32,world=8,adapter_tensors_all_finite=True,changed_adapter_tensors_1_to_5=changed,actual_saved_optimizer_ranks=optimizer,memory=m,live_update_evidence=a,scope='Five actual OPSD updates and numerical/memory acceptance, not completed effect reproduction')
(b/'receipts/five-real-opsd-updates-accepted.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
