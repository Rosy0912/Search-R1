"""Small read-only CPU observer: direct free memory and sampled GPU peak footprint."""
import argparse,json,os,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);assert os.getuid()==1005
peak=[0]*8;minimum=[None]*8;samples=accepted=0
while True:
 try:
  rows=[x.split(',') for x in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,memory.used','--format=csv,noheader,nounits'],text=True).splitlines()]
  assert len(rows)==8
  pids=set(subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).split())
  foreign=[]
  for pid in pids:
   try:
    if Path('/proc',pid).stat().st_uid!=1005:foreign.append(int(pid))
   except FileNotFoundError:pass
  samples+=1
  if not foreign:
   accepted+=1
   for row in rows:
    i,free,used=map(int,row);peak[i]=max(peak[i],used);minimum[i]=free if minimum[i] is None else min(minimum[i],free)
  value=dict(time=time.time(),pid=os.getpid(),samples=samples,accepted_no_foreign_samples=accepted,foreign_pids=foreign,
    current_directfree_mib=[int(x[1]) for x in rows],observed_peak_used_mib=peak,observed_min_directfree_mib=minimum,
    scope='Sampled at 0.5s; whole GPU footprint includes own oracle/context; excludes any sample with foreign UID process; not a mathematical bound between samples')
 except Exception as e:value=dict(time=time.time(),pid=os.getpid(),error=repr(e),samples=samples)
 tmp=out.with_suffix('.pending');tmp.write_text(json.dumps(value));os.replace(tmp,out);time.sleep(.5)
