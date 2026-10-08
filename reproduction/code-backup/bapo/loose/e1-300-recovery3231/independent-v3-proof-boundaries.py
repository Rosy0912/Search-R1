import sys,json,tempfile
from pathlib import Path
from pytest import MonkeyPatch
sys.path.insert(0,str(Path('work/bapo-e1-300-recovery').resolve()));sys.path.insert(0,str(Path('work/bapo-e1-300-recovery/tests/recovery').resolve()))
from test_gpu_proof import setup_proof
from search_r1.baselines import recovery as r
cases=('missing_v3','different_gen_after','different_state_after','wrong_pid','wrong_dtype','bool_mesh_shape','bool_mesh_rank','roundtrip_false','native_version_bool')
results=[]
for case in cases:
 with tempfile.TemporaryDirectory()as t,MonkeyPatch.context()as mp:
  m,p,c,root,run,ranks,d,bind=setup_proof(Path(t).resolve(),mp);v3=p['gpu_validation']['native_v3_roundtrip'];vr=r.read(v3[0]['report']['path'])
  if case=='missing_v3':p['gpu_validation'].pop('native_v3_roundtrip')
  if case=='different_gen_after':vr['generation_rng_after_digest']='a'*64
  if case=='different_state_after':vr['reloaded_state_digest']='a'*64
  if case=='wrong_pid':vr['pid']=999
  if case=='wrong_dtype':vr['generation_rng_topology']['gen_random_states_dtype']='torch.float32'
  if case=='bool_mesh_shape':vr['generation_rng_topology']['device_mesh']['shape']=[8,True]
  if case=='bool_mesh_rank':vr['generation_rng_topology']['device_mesh']['ranks'][0]=[False]
  if case=='roundtrip_false':vr['roundtrip']=False
  if case=='native_version_bool':vr['native_state_version']=True
  v3[0]['report']=bind(Path(v3[0]['report']['path']),vr)
  records=[r.read(x['report']['path'])for x in v3]
  raw=p['gpu_validation']['raw_artifacts'];text=Path(raw['train_log']['path']).read_text();text='\n'.join(x for x in text.splitlines()if 'BAPO_RECOVERY_V3_RANK_JSON='not in x)+'\n'+'\n'.join('BAPO_RECOVERY_V3_RANK_JSON='+json.dumps(x)for x in records)+'\n';raw['train_log']=bind(run/'train.log',text,False)
  try:r.validate_gpu_proof(p,m,c,root)
  except (ValueError,KeyError,TypeError)as e:results.append(dict(case=case,rejected=True,error=str(e)));continue
  results.append(dict(case=case,rejected=False))
o=dict(scope='INDEPENDENT_LOCAL_SCHEMA_ONLY_NOT_ACTUAL_GPU_V3_ROUNDTRIP',results=results,all_rejected=all(x['rejected']for x in results));Path('work/evidence/recovery3231-independent-v3-proof-boundaries.json').write_text(json.dumps(o,indent=2)+'\n');print(json.dumps(o,indent=2));assert o['all_rejected']
