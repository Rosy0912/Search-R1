"""Read-only checkpoint25 metadata and realCPU driver check; not full HF acceptance."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='8';os.environ['MKL_NUM_THREADS']='8';os.environ['PYTHONDONTWRITEBYTECODE']='1'
import pathlib,json,hashlib,sys,time
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-stabilityfix-r2-20261005');RUN=ROOT/'runs/bapo-instruct300-formal300-3231';D=RUN/'checkpoints/full_state/global_step_25'
sys.path.insert(0,str(ROOT/'repo'))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((D/'manifest.json').read_text());assert m['global_step']==m['actual_outer_updates']==25 and m['horizon']==300 and m['warmup_steps']==114 and m['source_commit']=='39cdcdd1404213b897bb0de99cc52d72707682f0'
assert len(m['rank_roundtrips'])==8 and sorted(x['rank']for x in m['rank_roundtrips'])==list(range(8))
for x in m['rank_roundtrips']:assert x['status']=='EXACT_FULL_STATE_RELOAD_OK'and x['step']==x['scheduler_last_epoch']==25 and x['world_size']==8
for x in m['files']:
 p=pathlib.Path(x['path']);assert RUN in p.resolve().parents and p.is_file()and p.stat().st_size==x['bytes']
import torch
from search_r1.baselines.full_state import state_digest
torch.set_num_threads(8);d=torch.load(D/'driver.pt',map_location='cpu',weights_only=False);r=json.loads((D/'driver-reload.json').read_text());a=json.loads((RUN/'checkpoints/algorithm/global_step_25/state.json').read_text())
assert d['global_step']==d['actual_outer_updates']==25 and d['next_global_step']==26 and d['driver_pid']==40564 and d['sampler']['cursor']==12800 and d['sampler']['size']==169028 and d['rng']['cuda']==[]
assert r['status']=='EXACT_DRIVER_RELOAD_OK'and r['step']==25 and r['pid']==40564 and r['sha256']==digest(D/'driver.pt')and r['state_digest']==state_digest(d)
assert a['global_step']==a['actual_outer_updates']==25 and a['algorithm_source_commit']==m['source_commit']and a['bapo']==d['bapo']
assert not torch.cuda.is_initialized()
print(json.dumps({'scope':'CHECKPOINT25_METADATA_FILE_SIZES_AND_REAL_CPU_DRIVER_ONLY_NOT_FULL_NATIVE_HASH_OR_HF_ACCEPTANCE','status':'PASS_METADATA_AND_CPU_DRIVER25','time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'job_id':3231,'manifest_sha256':digest(D/'manifest.json'),'manifest_files':len(m['files']),'manifest_total_bytes':sum(x['bytes']for x in m['files']),'native_producer_reload_reports':m['rank_roundtrips'],'driver_sha256':digest(D/'driver.pt'),'driver_digest':state_digest(d),'sampler_cursor':d['sampler']['cursor'],'bapo_state':d['bapo'],'CUDA_initialized':False,'running_source_changed':False},indent=2))
