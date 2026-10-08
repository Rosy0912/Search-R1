from pathlib import Path
import json,hashlib,base64,subprocess,os,gc
import torch
from search_r1.baselines.recovery import file_sha256,tensor_digest
root=Path('/mnt/data/shaoshitong/search-r1-rl/gigpo');run=root/'runs/gigpo-resume200to300-job3130-1791086878335123421';cp=run/'checkpoints/actor/global_step_205'
ready=json.loads((run/'resume205-ready.json').read_text());manifest=json.loads(Path(ready['path']).read_text())
assert file_sha256(ready['path'])==ready['sha256'];assert manifest==ready['identity']
assert manifest['job_id']=='3130' and manifest['logical_step']==205 and manifest['actual_new_updates']==5
files=[]
for receipt in manifest['files']:
 p=Path(receipt['path']);assert p.stat().st_size==receipt['bytes'] and file_sha256(p)==receipt['sha256'];files.append(dict(receipt))
states=[]
for rank in range(8):
 p=cp/'training_state'/f'rank_{rank}.pt';s=torch.load(p,map_location='cpu',weights_only=False)
 assert s['logical_step']==205 and s['layout']['rank']==rank and s['layout']['world_size']==8
 for t in s['model_shards'].values():assert torch.isfinite(t).all()
 adam_steps=[]
 for moment in s['optimizer']['state'].values():
  for key,t in moment.items():
   if isinstance(t,torch.Tensor):assert torch.isfinite(t).all()
  adam_steps.append(float(moment['step']))
 assert set(adam_steps)=={50.0},set(adam_steps)
 ident=s['pause_identity'];assert ident['scheduler_step']==205 and s['scheduler']['last_epoch']==205
 assert tensor_digest(s['model_shards'].items())==ident['model_shards_sha256']
 names=[p['name'] for p in s['layout']['parameters']];mapping={}
 for group,lay in zip(s['optimizer']['param_groups'],s['layout']['optimizer_groups']):mapping.update(zip(lay['names'],group['params']))
 moments=[(str(index)+'/'+key,value) for index,name in enumerate(names) for key,value in sorted(s['optimizer']['state'].get(mapping[name],{}).items()) if isinstance(value,torch.Tensor)]
 assert tensor_digest(moments)==ident['optimizer_tensors_sha256']
 generators=[('torch_cpu',s['rng']['torch']),('torch_cuda',s['rng']['cuda'])]+[(k,v) for k,v in s['sharding_rng'].items() if v is not None]
 assert tensor_digest(generators)==ident['rng_tensors_sha256']
 pid=ident['pid'];proc=Path('/proc')/str(pid);tick=(proc/'stat').read_text().rsplit(')',1)[1].split()[19];assert tick==ident['start_ticks']
 assert '/job_3130/' in (proc/'cgroup').read_text()
 states.append(dict(rank=rank,logical_step=205,pause_identity=ident,live_pid_start_matches=True,model_sha_verified=True,optimizer_sha_verified=True,rng_sha_verified=True,adam_step_values=sorted(set(adam_steps)),scheduler=s['scheduler'],shard_parameters=len(s['model_shards']),optimizer_parameters=len(s['optimizer']['state']),rng_present={k:v is not None for k,v in s['rng'].items()},layout=s['layout']))
 del s;gc.collect()
d=torch.load(cp/'training_state/driver.pt',map_location='cpu',weights_only=False);assert d['logical_step']==205 and d['actual_new_updates']==5
sampler=d['sampler'];assert sampler['cursor']==2560 and len(sampler['order'])==166222 and sorted(sampler['order'])==list(range(166222))
driver=dict(logical_step=d['logical_step'],actual_new_updates=d['actual_new_updates'],source=d['source'],sampler={k:v for k,v in sampler.items() if k not in ('order','generator')},order_sha256=hashlib.sha256(json.dumps(sampler['order']).encode()).hexdigest(),rng_present={k:v is not None for k,v in d['rng'].items()},kl_controller=d['kl_controller'],config=d['config'])
raw={}
paths=[run/'resume205-ready.json',cp/'training_state/manifest.json',run/'checkpoint205-cpu-verification.json',run/'checkpoint205-cpu-verification.log',run/'launch-manifest.json',run/'input-config.yaml',run/'resolved-config.yaml',run/'audit/entry-config.yaml',run/'training.log',run/'guard.log',run/'ray-audit-environment-verification.json',run/'checkpoints/algorithm/global_step_205/state.json',*sorted((run/'audit').rglob('*.jsonl'))]
for p in paths:
 b=p.read_bytes();raw[str(p)]=dict(sha256=hashlib.sha256(b).hexdigest(),base64=base64.b64encode(b).decode())
obj=dict(status='COLLECTED_NOT_INDEPENDENT_ACCEPTANCE',run=str(run),ready=ready,files_verified=files,worker_states=states,driver_state=driver,raw_files=raw,queue=subprocess.check_output(['squeue','-j','3130','-o','%.18i %.9T %.10M %.20R'],text=True),gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True))
p=root/'logs/resume3130-205-evidence-v1.json';p.write_text(json.dumps(obj,indent=2));print(json.dumps(obj))
