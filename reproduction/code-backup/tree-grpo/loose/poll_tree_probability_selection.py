"""Read-only exact CPU3275 scheduler and selection metadata; no tensor reads."""
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import time

assert socket.gethostname().split('.')[0]=='h100-9'
root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300')
tools=root/'probe-selection-32593d9-source';result=root/'probe-selection-32593d9-result'
journal=json.loads((root/'PROBE_SELECTION_ONCE-3268-32593d9.json').read_bytes())
assert journal['status']=='SUBMITTED' and journal['job_id']==3275
reports={}
for name in ['selection-3275.json','selection-3275.err','target-cpu-tests-3275.txt']:
    path=tools/name
    if path.exists():
        data=path.read_bytes();reports[name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),text=data.decode())
manifest=result/'selection.json'
if manifest.exists():
    data=manifest.read_bytes();reports['selection.json']=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),text=data.decode())
commands={}
for args in [['scontrol','show','job','3275'],['squeue','-h','-j','3275','-o','%i %j %T %M %N']]:
    p=subprocess.run(args,capture_output=True,text=True,timeout=15)
    commands[' '.join(args)]=dict(rc=p.returncode,stdout=p.stdout,stderr=p.stderr)
print(json.dumps(dict(job_id=3275,captured_epoch=time.time(),journal=journal,commands=commands,reports=reports),sort_keys=True,allow_nan=False))
