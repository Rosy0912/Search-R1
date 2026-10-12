import pathlib,os,json,subprocess,time,shutil,hashlib
scope=SCOPE
try:os.nice(10)
except:pass
gate=GATE
for rec in gate['source_records']:
 p=pathlib.Path(rec['archive']);assert p.stat().st_size==rec['archive_bytes'];h=hashlib.sha256()
 with p.open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
 assert h.hexdigest()==rec['archive_sha256']

root=pathlib.Path('/mnt/data/shaoshitong/search-r1-rl');audit=pathlib.Path('/mnt/data/shaoshitong/h100-cleanup-20261012-audit')/scope;audit.mkdir(parents=True,exist_ok=True)
candidates=[]
if scope=='tree-grpo':
 tree=root/'tree-grpo-retained18-20261008'
 for runname in ['tree-current200-continue-3331','tree-current200-continue-3335','tree-current200-continue-3339','tree-current200-continue-3341']:
  run=tree/'runs'/runname
  if runname.endswith('3335'):
   candidates += [run/'checkpoints/actor',run/'checkpoints/training_state']
  if runname.endswith('3341'):
   for cat in ['actor','training_state']:
    candidates += [run/'checkpoints'/cat/('global_step_'+str(n)) for n in [50,100,150]]
  candidates += [run/'output/trajectories',run/'output/tree-audit']
 protected=[tree/'runs/tree-branpo-fresh200-3360',tree/'runs/tree-current200-continue-3341/checkpoints/actor/global_step_200',tree/'runs/tree-current200-continue-3341/checkpoints/training_state/global_step_200',tree/'runs/tree-current200-continue-3341/checkpoints/training_state/rolling-completed']
else:
 run=root/'opsd-e1-unclipped-multitraj200-20261006/runs/formal-3304'
 candidates=[run/'checkpoints/actor/global_step_50',run/'checkpoints/actor/global_step_100']
 protected=[run/'checkpoints/actor/global_step_5',run/'checkpoints/actor/global_step_150',run/'checkpoints/actor/global_step_200',run/'full_state',run/'bindings',run/'evidence']
candidates=[p for p in candidates if p.exists()];out={'epoch':time.time(),'scope':scope,'before_df':subprocess.check_output(['df','-B1','/mnt/data'],text=True),'hf_gate':GATE,'protected':[str(p) for p in protected],'candidates':[]}
for c in candidates:
 assert c.is_dir() and not c.is_symlink() and str(c.resolve()).startswith(str(root.resolve())+'/')
 assert not any(c.resolve()==p.resolve() or c.resolve().is_relative_to(p.resolve()) or p.resolve().is_relative_to(c.resolve()) for p in protected)
refs=[]
for proc in pathlib.Path('/proc').glob('[0-9]*'):
 if int(proc.name)==os.getpid():continue
 try:
  cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  for c in candidates:
   if str(c) in cmd:refs.append({'pid':int(proc.name),'path':str(c),'kind':'cmd'})
  for p in [proc/'cwd']+list((proc/'fd').iterdir()):
   try:
    real=str(p.resolve())
    for c in candidates:
     if real==str(c.resolve()) or real.startswith(str(c.resolve())+'/'):refs.append({'pid':int(proc.name),'path':str(c),'kind':'fd/cwd'})
   except:pass
  maps=(proc/'maps').read_text(errors='replace')
  for c in candidates:
   if str(c) in maps:refs.append({'pid':int(proc.name),'path':str(c),'kind':'mmap'})
 except:pass
assert not refs,refs
for c in candidates:
 manifest=[]
 for f in c.rglob('*.json'):
  if f.is_file() and not f.is_symlink() and f.stat().st_size<1024**2:
   dest=audit/'preserved-metadata'/f.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest);manifest.append({'source':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'backup':str(dest)})
 out['candidates'].append({'path':str(c),'bytes':int(subprocess.check_output(['du','-sb',str(c)],text=True).split()[0]),'metadata_preserved':manifest})
(audit/'delete-preflight.json').write_text(json.dumps(out,indent=2))
for c in candidates:shutil.rmtree(c);assert not c.exists()
out['after_df']=subprocess.check_output(['df','-B1','/mnt/data'],text=True);out['completed_epoch']=time.time();out['status']='HF_FULL_SHA_VERIFIED_STALE_CHECKPOINTS_DELETED';(audit/'delete-complete.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
