import pathlib,json,hashlib,tarfile,time
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-recovery-20261005');r=root/'runs/bapo-instruct300-formal300-3246';dest=root/'logs/formal3246-fault-raw.tar.gz';record=dest.with_suffix('.record.json');assert not dest.exists()and not record.exists();files=[p for p in r.rglob('*')if p.is_file()and not p.is_symlink()];files+=[root/'logs/slurm-formal300-3246.out',root/'logs/slurm-formal300-3246.err'];inv=[]
for p in files:
 b=p.read_bytes();inv.append({'path':str(p),'relative':str(p.relative_to(root)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
assert sum(x['bytes']for x in inv)<200_000_000
with tarfile.open(dest,'x:gz')as t:
 for p in files:t.add(p,arcname=str(p.relative_to(root)),recursive=False)
b=dest.read_bytes();out={'job_id':3246,'time_unix':time.time(),'path':str(dest),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'inventory':inv,'slurm_live_handle_retired':True,'terminal_basis':'Own resource-watch recovery_failed, phase RESOURCE_PAUSE, srun actual exit42, cleanup remaining[]; final scontrol unavailable, no invented Slurm fields'};record.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
