import os,time,json,subprocess,fcntl
base='/data/user/jxu957/experiments/cosearch-continue300-x203-20261008'
lock=open(base+'/cleanup-other-data.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
state=base+'/cleanup-other-data.json'
paths=['/data/user/jxu957/independent_project','/data/user/jxu957/envs1']
attempt=0
while True:
 remaining=[];errors=[]
 for p in paths:
  try:os.lstat(p);remaining.append(p)
  except FileNotFoundError:pass
  except OSError as e:remaining.append(p);errors.append(str(e))
 status={'pid':os.getpid(),'epoch':time.time(),'remaining':remaining,'attempt':attempt,'errors':errors,'status':'cleaning' if remaining else 'complete'}
 with open(state+'.tmp','w') as f:json.dump(status,f)
 os.replace(state+'.tmp',state)
 print(json.dumps(status),flush=True)
 if not remaining:break
 attempt+=1
 for p in remaining:
  try:
   r=subprocess.run(['/usr/bin/rm','-rf','--one-file-system','--',p],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=600)
   print(json.dumps({'path':p,'exit':r.returncode,'errors':r.stderr[-3000:]}),flush=True)
  except subprocess.TimeoutExpired:print(json.dumps({'path':p,'timeout':True}),flush=True)
 time.sleep(30)
