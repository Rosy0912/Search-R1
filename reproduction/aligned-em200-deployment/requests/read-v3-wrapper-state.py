import os,json
from pathlib import Path
r=Path('/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v3-20261009');out={'wrappers':[]}
for p in Path('/proc').iterdir():
 if p.name.isdigit() and int(p.name)!=os.getpid():
  try:
   c=(p/'cmdline').read_bytes();args=c.split(b'\0')
   if len(args)>=4 and args[0].endswith(b'python3') and args[1:3]==[b'-B',b'-c'] and b'formal3352-failure-full.tar' in args[3] and b'archive_sha256' in args[3]:out['wrappers'].append({'pid':int(p.name),'stat':(p/'stat').read_text(),'io':(p/'io').read_text()})
  except OSError:pass
for n in ('formal-submit-intent.json','formal-submit-receipt.json'):
 p=r/'scheduler-head'/n;out[n]=p.read_text()if p.exists()else None
print(json.dumps(out))
