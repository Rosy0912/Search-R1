"""Read-only CPU verification of physical saved continuation state; never mutates trainer."""
from pathlib import Path
import json,hashlib,os,gc,sys
import torch
from search_r1.baselines.recovery import file_sha256,tensor_digest
ROOT=Path('/mnt/data/shaoshitong/search-r1-rl/gigpo')
RUN=ROOT/'runs/gigpo-resume200to300-job3130-1791086878335123421'
SOURCE=str(ROOT/'runs/gigpo-formal500-job2992-1791016340597739432/checkpoints/actor/global_step_200')
step=int(sys.argv[1]);assert step in (205,225,250,275,300);actual=step-200
cp=RUN/'checkpoints/actor'/f'global_step_{step}';manifest_path=cp/'training_state/manifest.json';manifest=json.loads(manifest_path.read_text())
assert manifest['job_id']=='3130' and manifest['source']==SOURCE and manifest['logical_step']==step and manifest['actual_new_updates']==actual
assert manifest['code_commit']=='60a472b156812c707f9d4ec48d57512773016333'
assert manifest['config_sha256']=='45b4c55462069aa37ddcd01403eb799681f1fe69d8564dda34724d14593ba15f'
assert manifest['data_identity']=='fac19c74fe28df2016bdff12f74f035fe272bed0d65d37e9740071e674195524'
hf_names=('added_tokens.json','config.json','generation_config.json','merges.txt','model-00001-of-00003.safetensors','model-00002-of-00003.safetensors','model-00003-of-00003.safetensors','model.safetensors.index.json','special_tokens_map.json','tokenizer.json','tokenizer_config.json','vocab.json')
expected={str(cp/name) for name in hf_names}|{str(cp/'training_state'/f'rank_{rank}.pt') for rank in range(8)}|{str(cp/'training_state/driver.pt')}
assert len(manifest['files'])==21 and {x['path'] for x in manifest['files']}==expected
receipts={x['path']:x for x in manifest['files']}
for row in manifest['files']:
 p=Path(row['path']);assert p.is_file() and p.stat().st_size==row['bytes'];assert file_sha256(p)==row['sha256'],p
initial_payload_path=ROOT/'logs/resume3130-205-evidence-v1.json'
assert file_sha256(initial_payload_path)=='8d6d656832c265e4f403250643e2f5d0bac5f5d906660cf043b79e16e2a778b6'
initial_payload=json.loads(initial_payload_path.read_text())
initial_layouts={x['rank']:hashlib.sha256(json.dumps(x['layout'],sort_keys=True).encode()).hexdigest() for x in initial_payload['worker_states']}
states=[];initial_workers=json.loads((RUN/'resume205-ready.json').read_text())['identity']['files'];initial_ids={x['pause_identity']['rank']:x['pause_identity'] for x in initial_workers if 'pause_identity' in x}
for rank in range(8):
 p=cp/'training_state'/f'rank_{rank}.pt';s=torch.load(p,map_location='cpu',weights_only=False)
 assert s['schema']==1 and s['logical_step']==step and s['layout']['rank']==rank and s['layout']['world_size']==8
 for t in s['model_shards'].values():assert torch.isfinite(t).all()
 opt_steps=[]
 for state in s['optimizer']['state'].values():
  for t in state.values():
   if isinstance(t,torch.Tensor):assert torch.isfinite(t).all()
  opt_steps.append(float(state['step']))
 assert set(opt_steps)=={float(actual*10)}
 ident=s['pause_identity'];assert ident==receipts[str(p)]['pause_identity'] and ident['rank']==rank
 assert hashlib.sha256(json.dumps(s['layout'],sort_keys=True).encode()).hexdigest()==initial_layouts[rank]
 assert ident['pid']==initial_ids[rank]['pid'] and ident['start_ticks']==initial_ids[rank]['start_ticks']
 assert ident['scheduler_step']==step and s['scheduler']['last_epoch']==step and s['scheduler']['_step_count']==step+1
 assert s['scheduler']['_last_lr']==ident['learning_rates']==[g['lr'] for g in s['optimizer']['param_groups']]
 assert tensor_digest(s['model_shards'].items())==ident['model_shards_sha256']
 names=[x['name'] for x in s['layout']['parameters']];mapping={}
 for group,layout in zip(s['optimizer']['param_groups'],s['layout']['optimizer_groups']):mapping.update(zip(layout['names'],group['params']))
 moments=[(str(index)+'/'+key,value) for index,name in enumerate(names) for key,value in sorted(s['optimizer']['state'].get(mapping[name],{}).items()) if isinstance(value,torch.Tensor)]
 assert tensor_digest(moments)==ident['optimizer_tensors_sha256']
 generators=[('torch_cpu',s['rng']['torch']),('torch_cuda',s['rng']['cuda'])]+[(k,v) for k,v in s['sharding_rng'].items() if v is not None]
 assert tensor_digest(generators)==ident['rng_tensors_sha256']
 states.append(dict(rank=rank,logical_step=step,actual_new_updates=actual,pause_identity=ident,finite=True,optimizer_step_values=sorted(set(opt_steps)),scheduler=s['scheduler'],model_optimizer_rng_digests_verified=True,same_worker_pid_start_as205=True,layout_sha256=hashlib.sha256(json.dumps(s['layout'],sort_keys=True).encode()).hexdigest()))
 del s;gc.collect()
d=torch.load(cp/'training_state/driver.pt',map_location='cpu',weights_only=False);assert d['logical_step']==step and d['actual_new_updates']==actual and d['source']==SOURCE
sampler=d['sampler'];assert sampler['cursor']==actual*512 and len(sampler['order'])==166222 and sorted(sampler['order'])==list(range(166222))
assert d['config']['trainer']['resume']['stop_step']==300
summary=dict(status='PASS_SAVED_FILES_CPU_INSPECTION_ONLY',job_id='3130',source=SOURCE,logical_step=step,actual_new_updates=actual,manifest_sha256=file_sha256(manifest_path),manifest=manifest,workers=states,driver=dict(logical_step=step,actual_new_updates=actual,source=SOURCE,sampler={k:v for k,v in sampler.items() if k not in ('order','generator')},permutation_sha256=hashlib.sha256(json.dumps(sampler['order']).encode()).hexdigest(),kl_controller=d['kl_controller'],rng_present={k:v is not None for k,v in d['rng'].items()}),scope='Actual complete files/SHA/finite model and Adam/layout/CPU digests/scheduler/RNG/permutation; not a live 3B restart or final acceptance')
p=ROOT/'logs'/f'resume3130-checkpoint{step}-cpu-state-verification-v1.json';assert not p.exists(),'Already inspected: read existing receipt, do not repeat';p.write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
