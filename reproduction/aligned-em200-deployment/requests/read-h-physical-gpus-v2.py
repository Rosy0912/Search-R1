"""User-requested physical H GPU inventory via existing strict SSH; no mutations."""
import subprocess,json,datetime,os,re,shlex
if os.uname().nodename!='h100-5':raise RuntimeError('wrong query origin')
raw=subprocess.check_output(['scontrol','show','nodes','-o'],text=True,timeout=15)
rows=[]
for line in raw.splitlines():
 fields=dict(re.findall(r'(\w+)=([^\s]*)',line));name=fields.get('NodeName');addr=fields.get('NodeAddr')
 if name not in ('h100-5','h100-8','h100-10','h100-11'):continue
 expected={'h100-5':'10.33.0.5','h100-8':'10.33.0.8','h100-10':'10.33.0.18','h100-11':'10.32.0.11'}
 if addr!=expected[name]:raise RuntimeError('actual node address differs from observed inventory')
 queries={}
 for key,args in [('gpus',['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total','--format=csv,noheader']),('gpu_processes',['nvidia-smi','--query-compute-apps=pid,process_name,gpu_uuid,used_memory','--format=csv,noheader'])]:
  if name=='h100-5':cmd=args
  else:cmd=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=8','root@'+addr,'env -u LD_LIBRARY_PATH '+shlex.join(args)]
  try:
   p=subprocess.run(cmd,capture_output=True,text=True,timeout=18);queries[key]={'exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
  except subprocess.TimeoutExpired:queries[key]={'timeout':True}
 rows.append({'node':name,'address':addr,'slurm_state':fields.get('State'),'queries':queries})
print(json.dumps({'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'READONLY_KNOWN_FOUR_NODES_PHYSICAL_GPU_NO_MUTATION','nodes':rows}))
