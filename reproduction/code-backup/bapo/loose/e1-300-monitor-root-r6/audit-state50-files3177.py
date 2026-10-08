import pathlib,json,hashlib,time
r=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-monitor-r6-20261004/runs/bapo-instruct300-formal300-3177'); c=r/'checkpoints'; start=time.monotonic(); results=[]
def digest(p):
 assert p.is_file() and not p.is_symlink() and r in p.resolve().parents
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
for rank in range(8):
 p=c/'actor/global_step_50/training_state'/f'rank_{rank}.json'; z=json.loads(p.read_text()); assert z['rank']==rank and z['world_size']==8 and z['step']==50 and z['scheduler_last_epoch']==50 and z['optimizer_state_entries']>0
 f=pathlib.Path(z['path']); h=digest(f); assert h==z['sha256']; results.append(dict(rank=rank,report=z,bytes=f.stat().st_size,actual_sha256=h))
p=c/'full_state/global_step_50/driver-reload.json'; z=json.loads(p.read_text()); assert z['step']==50 and z['status']=='DRIVER_SERIALIZED_OK'; f=pathlib.Path(z['path']); h=digest(f); assert h==z['sha256']
a=json.loads((c/'algorithm/global_step_50/state.json').read_text()); assert a['global_step']==50 and a['actual_outer_updates']==50 and a['algorithm_source_commit']=='733097feeae371b4efd8fd48940c46ff36da839b'
hf=[]
for p in sorted((c/'actor/global_step_50').iterdir()):
 if p.is_file():hf.append(dict(name=p.name,bytes=p.stat().st_size,sha256=digest(p)))
assert len(hf)==12
print(json.dumps(dict(status='STATE50_FILES_HASH_CLOCK_PASS',ranks=results,driver=z,driver_actual_sha256=h,algorithm=a,hf=hf,seconds=time.monotonic()-start,limitations='No state50 CPU load or resume performed; first5 exact reload separately accepted. No files modified.'),sort_keys=True))
