python3 - <<'PY'
import os,pathlib,json,time,subprocess,hashlib
job=3144;r=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd/logs/g1-initial3102-r4r10-allocation-evidence');rows=[];errors=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  cg=(p/'cgroup').read_text().splitlines()
  if not any('/job_3144/' in s or s.endswith('/job_3144') for s in cg):continue
  stat=(p/'stat').read_text();sf=stat.rsplit(')',1)[1].split();cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace');env={}
  for item in (p/'environ').read_bytes().split(b'\0'):
   k,sep,v=item.partition(b'=')
   if k.decode(errors='replace') in ('SEARCH_R1_GPU_LEASES','CUDA_VISIBLE_DEVICES','SLURM_JOB_ID'):env[k.decode()]=v.decode(errors='replace')
  fd=[]
  for f in (p/'fd').iterdir():
   try:
    target=os.readlink(f)
    if 'gpu' in target.lower() and ('lease' in target.lower() or 'lock' in target.lower()):fd.append({'fd':int(f.name),'target':target,'fdinfo':(p/'fdinfo'/f.name).read_text()})
   except FileNotFoundError:pass
  rows.append({'pid':int(p.name),'uid':p.stat().st_uid,'state':sf[0],'ppid':int(sf[1]),'starttime_ticks':int(sf[19]),'cgroup':cg,'command':cmd,'safe_identity_env':env,'gpu_lease_fds':fd})
 except FileNotFoundError:pass
 except Exception as e:errors.append({'pid':p.name,'error':repr(e)})
cmd=['scontrol','show','job',str(job),'-o'];q=subprocess.run(cmd,capture_output=True,text=True)
print(json.dumps({'scope':'READ_ONLY_3144_BEFORE_DIRECTION_CHANGE_NO_SIGNALS','epoch':time.time(),'job_id':job,'hostname':os.uname().nodename,'scontrol':{'returncode':q.returncode,'stdout':q.stdout,'stderr':q.stderr},'own_job_processes':rows,'errors':errors}))
PY
