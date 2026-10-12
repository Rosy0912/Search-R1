"""Small CPU-only sampled memory summary; never signal or allocate a GPU."""
import argparse,json,os,pathlib,subprocess,time
p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--job',required=True);a=p.parse_args()
b=pathlib.Path(a.base);path=b/'control/memory-live.json';samples=0;peak={};minimum={}
while True:
 try:
  rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,memory.total','--format=csv,noheader,nounits'],text=True,timeout=8)
  gpu=[dict(index=int(i),uuid=u.strip(),free_mib=int(f),total_mib=int(t)) for i,u,f,t in [line.split(',') for line in rows.splitlines()]]
  apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=8)
  own={};foreign=[]
  for line in apps.splitlines():
   u,pid,mem=[x.strip() for x in line.split(',')];proc=pathlib.Path('/proc',pid)
   try:
    env=(proc/'environ').read_bytes().split(b'\0')
    owned=(('SLURM_JOB_ID='+a.job).encode() in env or ('RBA_H100_ALLOCATION_ID='+a.job).encode() in env)
   except OSError:owned=False
   if owned:own[u]=own.get(u,0)+int(mem)
   else:foreign.append(dict(uuid=u,pid=int(pid),used_mib=int(mem)))
  for g in gpu:
   key=str(g['index']);peak[key]=max(peak.get(key,0),own.get(g['uuid'],0));minimum[key]=min(minimum.get(key,g['free_mib']),g['free_mib'])
  samples+=1;result=dict(time=time.time(),observer_pid=os.getpid(),job=a.job,samples=samples,current=gpu,own_peak_sampled_mib=peak,directfree_min_sampled_mib=minimum,foreign=foreign,scope='2s sampled peaks, not continuous hardware maximum')
  tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(result));tmp.replace(path)
 except Exception as error:
  (b/'control/memory-observer-errors.log').open('a').write(str(time.time())+' '+repr(error)+'\n')
 time.sleep(2)
