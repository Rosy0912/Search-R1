import pathlib,json,hashlib,os,time
r=pathlib.Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-unclipped-multitraj200-20261006/runs/formal-3304');o={'epoch':time.time(),'records':[],'df':os.statvfs('/mnt/data').f_bavail*os.statvfs('/mnt/data').f_frsize}
for f in list((r/'full_state/global_step_200').rglob('*'))+list((r/'checkpoints/algorithm/global_step_200').rglob('*')):
 if not f.is_file():continue
 if f.stat().st_size>16*1024**2:continue
 h=hashlib.sha256(f.read_bytes()).hexdigest();o['records'].append({'group':'opsd-driver-step200','archive':str(f),'archive_bytes':f.stat().st_size,'archive_sha256':h,'repo_path':'h100-cleanup-20261012/opsd/opsd-best-step200/full_state/'+str(f.relative_to(r))})
print(json.dumps(o))
