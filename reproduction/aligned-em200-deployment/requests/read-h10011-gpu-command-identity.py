"""User-requested GPU experiment command identity; no environ/source read or signals."""
import os,subprocess,json,datetime
from pathlib import Path
if os.uname().nodename!='h100-11':raise RuntimeError('wrong node')
p=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
if p.returncode:raise RuntimeError('GPU inventory failed')
pids=sorted({int(v.strip()) for v in p.stdout.splitlines() if v.strip().isdigit()});rows=[]
def record(pid):
 root=Path('/proc')/str(pid)
 fields=(root/'stat').read_text().rsplit(')',1)[1].split();args=[v.decode(errors='replace') for v in (root/'cmdline').read_bytes().split(b'\0') if v]
 # Do not print environment or long arguments beyond command identification.
 return {'pid':pid,'ppid':int(fields[1]),'start_ticks':int(fields[19]),'cwd':str((root/'cwd').resolve()),'command':[v[:1600] for v in args[:6]]}
for pid in pids:
 try:
  r=record(pid)
  try:r['parent']=record(r['ppid'])
  except (OSError,ValueError):r['parent_unavailable']=True
  rows.append(r)
 except (OSError,ValueError):rows.append({'pid':pid,'disappeared':True})
print(json.dumps({'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':'h100-11','scope':'READONLY_GPU_PROCESS_COMMAND_NOT_EXPERIMENT_RESULT','rows':rows}))
