"""Read-only exact CPU3277 scheduler and selection metadata; no tensor reads."""
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import time

assert socket.gethostname().split('.')[0]=='h100-9'
root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300')
tools=root/'probe-selection-v2-source';result=root/'probe-selection-v2-result'
journal=json.loads((root/'PROBE_SELECTION_ONCE-3268-v2.json').read_bytes())
assert journal['status']=='SUBMITTED' and journal['job_id']==3277
reports={}
for name in ['selection-3277.json','selection-3277.err','target-cpu-tests-3277.txt']:
    path=tools/name
    if path.exists():
        data=path.read_bytes();reports[name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),text=data.decode())
manifest=result/'selection.json'
if manifest.exists():
    data=manifest.read_bytes();reports['selection.json']=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),text=data.decode())
commands={}
for args in [['scontrol','show','job','3277'],['squeue','-h','-j','3277','-o','%i %j %T %M %N']]:
    p=subprocess.run(args,capture_output=True,text=True,timeout=15)
    commands[' '.join(args)]=dict(rc=p.returncode,stdout=p.stdout,stderr=p.stderr)
print(json.dumps(dict(job_id=3277,captured_epoch=time.time(),journal=journal,commands=commands,reports=reports),sort_keys=True,allow_nan=False))
