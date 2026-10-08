"""Independent read-only native CPU state oracle audit; no distributed init/CUDA use."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
REPO=Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004/repo-04f8cc5a777f')
ROOT=Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004/runs/state-hardware-3190/fixture')
sys.path.insert(0,str(REPO))
from search_r1.baselines.full_state import state_digest
assert not torch.cuda.is_initialized()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def layout(x):
 return (tuple(x.shape),tuple(x.stride()),str(x.dtype),tuple(map(str,x.placements)),x.device_mesh.device_type,tuple(x.device_mesh.mesh.flatten().tolist()),tuple(x.device_mesh.mesh_dim_names),tuple(x.device_mesh.get_coordinate()))
def equal(a,b):
 assert type(a)==type(b),(type(a),type(b))
 if hasattr(a,'to_local') and hasattr(a,'placements'):
  assert layout(a)==layout(b);return equal(a.to_local(),b.to_local())
 if torch.is_tensor(a):
  assert a.device.type==b.device.type=='cpu'
  assert a.shape==b.shape and a.dtype==b.dtype and a.stride()==b.stride() and torch.equal(a,b)
 elif isinstance(a,np.ndarray):assert a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b)
 elif isinstance(a,dict):
  assert a.keys()==b.keys()
  for k in a:equal(a[k],b[k])
 elif isinstance(a,(list,tuple)):
  assert len(a)==len(b)
  for x,y in zip(a,b):equal(x,y)
 else:assert a==b,(a,b)
results=[];files={};pids=set()
for rank in range(8):
 record=json.loads((ROOT/f'rank-{rank}.json').read_bytes())
 assert record['rank']==record['device']==rank and record['world_size']==8
 assert record['source']=='04f8cc5a777f8dfafbb9e48f877673c2c5751f33'
 scalar=record['independent_scalar_warmup_oracle']
 assert scalar['optimizer_calls']==scalar['scheduler_calls']==57
 assert scalar['first_lr']==0 and scalar['after57_lr']==1e-6
 assert abs(scalar['last_before_lr']-1e-6*56/57)<1e-18
 assert len(record['records'])==2
 for r in record['records']:
  name=r['profile'];loads={};reports={}
  for label,folder in [('initial','initial-step5'),('roundtrip','roundtrip-step5'),('oracle','no-roundtrip-step6'),('reloaded','reloaded-step6')]:
   p=ROOT/name/folder/f'rank_{rank}.pt';j=p.with_suffix('.json');report=json.loads(j.read_bytes())
   assert report['rank']==rank and report['world_size']==8 and report['version']==3
   assert report['step']==(5 if label in ('initial','roundtrip') else 6)
   assert report['path']==str(p) and report['sha256']==report['state_sha256']==sha(p)
   assert report['model_state_dict_type']=='SHARDED_STATE_DICT' and report['model_dtensor_entries']==(4 if name=='baseline-fp32' else 12)
   x=torch.load(p,map_location='cpu',weights_only=False)
   assert state_digest(x)==report['state_digest'];loads[label]=x;reports[label]=report
   files[str(p)]=sha(p);files[str(j)]=sha(j)
  equal(loads['initial'],loads['roundtrip']);equal(loads['oracle'],loads['reloaded'])
  assert reports['roundtrip']['status']=='EXACT_FULL_STATE_RELOAD_OK'
  assert len({v['pid'] for v in reports.values()})==1 and all(v['object_ids']==reports['initial']['object_ids'] for v in reports.values())
  pids.add(reports['initial']['pid'])
  components={}
  for key in ('model','optimizer','scheduler','rng','teacher'):
   equal(loads['oracle'][key],loads['reloaded'][key]);components[key]=state_digest(loads['oracle'][key])
  assert loads['initial']['scheduler']['last_epoch']==5 and loads['oracle']['scheduler']['last_epoch']==6
  assert r['initial_optimizer_calls']==20 and r['total_actual_optimizer_calls']==28
  assert len(r['gradient_norms'])==28 and r['gradient_norms'][20:24]==r['gradient_norms'][24:28]
  for n,lr in enumerate(r['outer_lrs'],1):
   assert lr['outer']==n and abs(lr['before']-1e-6*(n-1)/57)<1e-18 and abs(lr['after']-1e-6*n/57)<1e-18
  assert r['oracle_sixth_lr']==r['reloaded_sixth_lr'] and abs(r['oracle_sixth_lr']['after']-1e-6*6/57)<1e-18
  assert loads['initial']['teacher']['counters']['cuda_update_entries']==5 and loads['oracle']['teacher']['counters']['cuda_update_entries']==6
  assert loads['initial']['teacher']['counters']['scored_tokens']==80 and loads['oracle']['teacher']['counters']['scored_tokens']==96
  asset_dir=Path(reports['initial']['teacher']['asset_directory'])
  for file,expected in reports['initial']['teacher']['asset_files'].items():
   p=asset_dir/file;assert sha(p)==expected;files[str(p)]=sha(p)
  results.append({'rank':rank,'profile':name,'initial_vs_roundtrip_raw_exact':True,'oracle_vs_reloaded_raw_exact':True,'all_payload_sha_and_report_digest_verified':True,'pid':reports['initial']['pid'],'object_ids':reports['initial']['object_ids'],'component_digests':components,'dtensor_entries':reports['initial']['model_dtensor_entries']})
assert len(pids)==8 and len(results)==16 and not torch.cuda.is_initialized()
print(json.dumps({'scope':'INDEPENDENT_NATIVE_CPU_RAW_CHECKPOINT_COMPARISON_OF_ACTUAL_JOB3190_NOT_NEW_GPU','torch_version':torch.__version__,'cuda_initialized':False,'pairs':results,'all_checkpoint_and_report_files_sha256':files,'distinct_worker_pids':sorted(pids),'passed':True},sort_keys=True))
