import subprocess,sys,os,json,pathlib,hashlib,base64
KEYS=('BAPO_RUN_ID','SLURM_JOB_ID','SEARCH_R1_GPU_LEASES')
code='''import os,json,sys,ray,setproctitle
keys=('BAPO_RUN_ID','SLURM_JOB_ID','SEARCH_R1_GPU_LEASES')
print(json.dumps({'ready':True,'pid':os.getpid(),'extension':setproctitle.__file__,'env':{k:os.environ.get(k)for k in keys}}),flush=True)
sys.stdin.readline();setproctitle.setproctitle('ray::IDLE')
print(json.dumps({'after':True,'env':{k:os.environ.get(k)for k in keys}}),flush=True)
sys.stdin.readline()
'''
def snap(pid):
 p=pathlib.Path('/proc')/str(pid);raw=(p/'environ').read_bytes().split(b'\0'); vals={k:[x.split(b'=',1)[1].decode(errors='replace')for x in raw if x.startswith(k.encode()+b'=')]for k in KEYS}
 return {'markers':vals,'cmdline_base64':base64.b64encode((p/'cmdline').read_bytes()).decode(),'exe':str((p/'exe').resolve()),'environment_bytes':sum(len(x)+1 for x in raw)}
results=[]
for noenv in (False,True):
 env=dict(os.environ,BAPO_RUN_ID='BAPO_CPU_PROBE_TITLE_ENV_ONLY',SLURM_JOB_ID='CPU_PROBE_NOT_A_SLURM_JOB',SEARCH_R1_GPU_LEASES='[]',CUDA_VISIBLE_DEVICES='')
 env.pop('SPT_NOENV',None)
 if noenv:env['SPT_NOENV']='1'
 proc=subprocess.Popen([sys.executable,'-B','-c',code],env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
 ready=json.loads(proc.stdout.readline());before=snap(proc.pid);proc.stdin.write('\n');proc.stdin.flush();aftermsg=json.loads(proc.stdout.readline());after=snap(proc.pid);proc.stdin.write('\n');proc.stdin.flush();out,err=proc.communicate(timeout=15)
 results.append({'SPT_NOENV':noenv,'ready':ready,'before':before,'after_message':aftermsg,'after':after,'natural_exit_code':proc.returncode,'stderr':err,'signals':0})
print(json.dumps({'scope':'ACTUAL_TARGET_CPU_TITLE_PROC_ENV_DIAGNOSTIC_ONLY_NOT_JOB_OR_GPU','results':results},indent=2))
