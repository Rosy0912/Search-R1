"""Read-only full immutable25 candidate inventory/hash and realCPU driver; not resume authorization."""
import os,sys
sys.dont_write_bytecode=True
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='8';os.environ['MKL_NUM_THREADS']='8'
import pathlib,json,hashlib,time,resource,subprocess
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-stabilityfix-r2-20261005');RUN=ROOT/'runs/bapo-instruct300-formal300-3231';SOURCE='39cdcdd1404213b897bb0de99cc52d72707682f0';step=25
sys.path.insert(0,str(ROOT/'repo'));start=time.monotonic();cpu=time.process_time()
def digest(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
assert (RUN/'source-commit.txt').read_text().strip()==SOURCE
paths=[]
for kind in ['actor','algorithm','full_state']:
 d=RUN/'checkpoints'/kind/'global_step_25';assert d.is_dir()and not d.is_symlink()
 for p in sorted(d.rglob('*')):
  assert not p.is_symlink()
  if p.is_file():
   before=p.stat();sha=digest(p);after=p.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
   paths.append(dict(path=str(p),bytes=before.st_size,sha256=sha))
A=RUN/'checkpoints/actor/global_step_25';D=RUN/'checkpoints/full_state/global_step_25';reports=[]
for rank in range(8):
 r=json.loads((A/'training_state'/('rank_%d.json'%rank)).read_text());p=A/'training_state'/('rank_%d.pt'%rank)
 assert r['rank']==rank and r['step']==r['scheduler_last_epoch']==step and r['world_size']==8 and r['status']=='FULL_STATE_SERIALIZED_OK'
 assert r['path']==str(p)and r['sha256']==next(x['sha256']for x in paths if x['path']==str(p))
 reports.append(r)
import torch
from search_r1.baselines.full_state import state_digest
from omegaconf import OmegaConf
torch.set_num_threads(8);d=torch.load(D/'driver.pt',map_location='cpu',weights_only=False);r=json.loads((D/'driver-reload.json').read_text());a=json.loads((RUN/'checkpoints/algorithm/global_step_25/state.json').read_text());c=OmegaConf.load(RUN/'resolved-config.yaml')
assert d['global_step']==d['actual_outer_updates']==step and d['next_global_step']==26 and d['sampler']['cursor']==12800 and d['sampler']['size']==169028 and not d['rng']['cuda']
assert r['status']=='DRIVER_SERIALIZED_OK'and r['sha256']==digest(D/'driver.pt')and r['state_digest']==state_digest(d)
assert d['bapo']==a['bapo']and a['algorithm_source_commit']==SOURCE
assert c.trainer.total_training_steps==301 and c.actor_rollout_ref.actor.optim.num_warmup_steps==114
assert not torch.cuda.is_initialized()
print(json.dumps(dict(status='FULL25_HASH_AND_CPU_DRIVER_CANDIDATE_NOT_NEW_PROCESS_RELOAD',source=SOURCE,run=str(RUN),step=step,next_step=26,files=paths,native_reports=reports,driver_digest=state_digest(d),sampler_cursor=d['sampler']['cursor'],sampler_epoch=d['sampler']['epoch'],bapo=d['bapo'],config_sha256=digest(RUN/'resolved-config.yaml'),generation_rng_in_old_checkpoint=False,wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),indent=2))
