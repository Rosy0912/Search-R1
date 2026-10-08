"""Periodic25 serialized metadata/realCPU driver; no full HF or model reload acceptance."""
import os,sys
sys.dont_write_bytecode=True
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='8';os.environ['MKL_NUM_THREADS']='8'
import pathlib,json,hashlib,time
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-stabilityfix-r2-20261005');RUN=ROOT/'runs/bapo-instruct300-formal300-3231';D=RUN/'checkpoints/full_state/global_step_25';A=RUN/'checkpoints/actor/global_step_25'
sys.path.insert(0,str(ROOT/'repo'))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
reports=[]
for rank in range(8):
 report=A/'training_state'/('rank_%d.json'%rank);x=json.loads(report.read_text());p=pathlib.Path(x['path'])
 assert x['status']=='FULL_STATE_SERIALIZED_OK'and x['rank']==rank and x['step']==x['scheduler_last_epoch']==25 and x['world_size']==8 and x['model_dtensor_entries']==435 and x['optimizer_state_entries']>0
 assert p==A/'training_state'/('rank_%d.pt'%rank) and p.is_file()and not p.is_symlink()and p.stat().st_size>0
 reports.append({'report':str(report),'report_sha256':digest(report),'serialized_report':x,'native_bytes':p.stat().st_size})
assert len([p for p in A.iterdir()if p.is_file()])==12
import torch
from search_r1.baselines.full_state import state_digest
torch.set_num_threads(8);d=torch.load(D/'driver.pt',map_location='cpu',weights_only=False);r=json.loads((D/'driver-reload.json').read_text());a=json.loads((RUN/'checkpoints/algorithm/global_step_25/state.json').read_text())
assert d['global_step']==d['actual_outer_updates']==25 and d['next_global_step']==26 and d['driver_pid']==40564 and d['sampler']['cursor']==12800 and d['sampler']['size']==169028 and d['rng']['cuda']==[]
assert r['status']=='DRIVER_SERIALIZED_OK'and r['step']==25 and r['pid']==40564 and r['sha256']==digest(D/'driver.pt')and r['state_digest']==state_digest(d)
assert a['global_step']==a['actual_outer_updates']==25 and a['algorithm_source_commit']=='39cdcdd1404213b897bb0de99cc52d72707682f0'and a['bapo']==d['bapo']
assert not torch.cuda.is_initialized()
print(json.dumps({'scope':'PERIODIC25_NATIVE_SERIALIZATION_REPORTS_SIZES_AND_REAL_CPU_DRIVER_NOT_FULL_NATIVE_HASH_OR_HF_ACCEPTANCE','status':'PASS_PERIODIC25_METADATA_AND_CPU_DRIVER','time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'job_id':3231,'first5_gate_manifest_only':True,'native_roundtrip_restore_performed_at25':False,'native_serialization_reports':reports,'hf_file_inventory':[{'name':p.name,'bytes':p.stat().st_size}for p in sorted(A.iterdir())if p.is_file()],'driver_sha256':digest(D/'driver.pt'),'driver_digest':state_digest(d),'sampler_cursor':d['sampler']['cursor'],'bapo_state':d['bapo'],'CUDA_initialized':False,'running_source_changed':False},indent=2))
