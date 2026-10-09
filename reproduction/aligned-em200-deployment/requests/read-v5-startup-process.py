import os,json,subprocess
from pathlib import Path
r=Path('/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v5-20261009');b=json.loads((r/'scheduler-head/formal-submit-receipt.json').read_text());run=b['run'];out=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  env=(p/'environ').read_bytes().split(b'\0')
  if ('BAPO_RUN_ID='+run).encode()not in env:continue
  out.append({'pid':p.name,'cmd':(p/'cmdline').read_bytes().decode(errors='replace').replace('\0',' ')[:400],'stat':(p/'stat').read_text(),'wchan':(p/'wchan').read_text(),'io':(p/'io').read_text()})
 except OSError:pass
print(json.dumps(out))
