import pathlib,os,json,subprocess,datetime,hashlib,base64,stat
R=pathlib.Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-unclipped-multitraj200-20261006');RUN=R/'runs/formal-3304'
def c(a):
 p=subprocess.run(a,capture_output=True,text=True,timeout=15);return {'argv':a,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
files={}
for rel in ['train.log','formal-wrapper.log','job-exit.txt','evidence/train-process-exit.json']:
 p=RUN/rel;s=p.lstat();assert stat.S_ISREG(s.st_mode) and s.st_uid==0 and s.st_nlink==1 and s.st_size<16*1024*1024
 b=p.read_bytes();t=p.lstat();assert (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)==(t.st_dev,t.st_ino,t.st_size,t.st_mtime_ns,t.st_ctime_ns)
 files[rel]={'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'dev':s.st_dev,'inode':s.st_ino,'mtime_ns':s.st_mtime_ns,'text':b.decode('utf-8',errors='strict')}
layouts={}
for rel in ['checkpoints','full_state','trajectories','evidence']:
 p=RUN/rel;layouts[rel]=[{'name':x.name,'dir':x.is_dir(),'bytes':x.lstat().st_size} for x in p.iterdir()]
procs=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  cg=(p/'cgroup').read_text()
  if '/job_3304/' not in cg:continue
  q={'pid':int(p.name),'cgroup':cg,'stat':(p/'stat').read_text(),'cmdline':(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode(errors='replace'),'cwd':os.readlink(p/'cwd')}
  q['stdio']={n:os.readlink(p/'fd'/n) for n in ['0','1','2'] if (p/'fd'/n).exists()};procs.append(q)
 except OSError:continue
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':c(['hostname']),'slurm_job':c(['scontrol','show','job','-o','3304']),'slurm_steps':c(['scontrol','show','step','3304']),'slurm_local_pids':c(['scontrol','listpids','3304']),'files':files,'layouts':layouts,'actual_job3304_procs':procs},sort_keys=True))
