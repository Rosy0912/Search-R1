import os,json,datetime
from pathlib import Path
r=Path('/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v10-20261009');d=r/'evidence/first5-cpu3358-20261009-1806'
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':[{'name':p.name,'bytes':p.stat().st_size}for p in d.iterdir()if p.is_file()],'cpu_evidence_complete':(d/'cpu-evidence.json').exists(),'processes':[]}
for p in Path('/proc').iterdir():
 if p.name.isdigit():
  try:
   c=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
   if 'aligned_first5_cpu_evidence.py' in c and str(r) in c and '-c import'not in c:out['processes'].append({'pid':int(p.name),'command':c[:600],'io':(p/'io').read_text(),'status':(p/'status').read_text()[:700]})
  except (OSError,ValueError):pass
print(json.dumps(out))
