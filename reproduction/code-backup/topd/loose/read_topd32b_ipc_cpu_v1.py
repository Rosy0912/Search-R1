"""Read-only3031 raw status/log capture; no scheduling or recovery actions."""
import subprocess
from pathlib import Path
remote=r'''python3 - <<'INNER'
import base64,hashlib,json,pathlib,subprocess,time
r=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd');j=3031
status=subprocess.run(['scontrol','show','job','-dd',str(j)],capture_output=True,text=True,timeout=20)
p=r/'logs'/('topd32b-ipc-native-cpu-'+str(j)+'.log');raw=p.read_bytes() if p.exists() else b''
row=dict(epoch=time.time(),job_id=j,scontrol_stdout=status.stdout,scontrol_stderr=status.stderr,scontrol_returncode=status.returncode,raw_log_sha256=hashlib.sha256(raw).hexdigest(),raw_log_base64=base64.b64encode(raw).decode())
with (r/'logs/topd32b-ipc-native-cpu-3031-readonly.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush()
print(json.dumps(row))
INNER'''
result=subprocess.run(['python3','work/remote_topd.py'],input=remote,text=True,capture_output=True,timeout=90)
Path('outputs/topd32b-ipc-native-cpu-3031-observations.jsonl').open('a').write(result.stdout)
if result.returncode:print(result.stderr);raise SystemExit(result.returncode)
import json,base64
row=json.loads(result.stdout.strip().splitlines()[-1])
Path('outputs/topd32b-ipc-native-cpu-3031-raw.log').write_bytes(base64.b64decode(row['raw_log_base64']))
Path('outputs/topd32b-ipc-native-cpu-3031-slurm.log').write_text(row['scontrol_stdout']+row['scontrol_stderr'])
print(row['scontrol_stdout']);print(base64.b64decode(row['raw_log_base64']).decode())
