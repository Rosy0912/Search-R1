import pathlib,os,json,subprocess,time,shutil,concurrent.futures,hashlib
root=pathlib.Path('/mnt/data/shaoshitong/search-r1-rl/tree-grpo-retained18-20261008');audit=pathlib.Path('/mnt/data/shaoshitong/h100-cleanup-20261012-audit');audit.mkdir(exist_ok=True)
candidates=[root/'runs'/run/'checkpoints/training_state' for run in ['tree-current200-continue-3331','tree-current200-continue-3339']]
out={'epoch':time.time(),'before_df':subprocess.check_output(['df','-B1','/mnt/data'],text=True),'candidates':[]}
queue=subprocess.check_output(['squeue','-h','-o','%i|%T'],text=True);assert not any(line.split('|')[0] in ['3331','3339'] for line in queue.splitlines())
for c in candidates:
 assert c.is_dir() and not c.is_symlink() and str(c.resolve()).startswith(str(root.resolve())+'/runs/tree-current200-continue-')
 refs=[]
 for proc in pathlib.Path('/proc').glob('[0-9]*'):
  if int(proc.name)==os.getpid():continue
  try:
   cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
   if str(c) in cmd or str(c.parent.parent) in cmd:refs.append({'pid':int(proc.name),'kind':'cmd'})
   for p in [proc/'cwd']+list((proc/'fd').iterdir()):
    try:
     real=str(p.resolve())
     if real==str(c.resolve()) or real.startswith(str(c.resolve())+'/'):refs.append({'pid':int(proc.name),'kind':'fd/cwd'})
    except:pass
   maps=(proc/'maps').read_text(errors='replace')
   if str(c) in maps:refs.append({'pid':int(proc.name),'kind':'mmap'})
  except:pass
 assert not refs,refs
 inventory=[]
 for f in c.rglob('*'):
  if f.is_file() and f.suffix=='.json' and f.stat().st_size<5*1024**2:
   dest=audit/c.parent.parent.name/'training_state'/f.relative_to(c);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest);inventory.append({'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
 bytes_=int(subprocess.check_output(['du','-sb',str(c)],text=True).split()[0]);out['candidates'].append({'path':str(c),'bytes':bytes_,'refs':refs,'json_preserved':inventory})
(audit/'failed-delete-preflight.json').write_text(json.dumps(out,indent=2))
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(shutil.rmtree,candidates))
for c in candidates:assert not c.exists()
out['after_df']=subprocess.check_output(['df','-B1','/mnt/data'],text=True);out['completed_epoch']=time.time();out['status']='FAILED_CHECKPOINTS_DELETED_WITH_NO_LIVE_REFERENCES';(audit/'failed-delete-complete.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
