"""Resolve current GPU process ancestry without signals or unfiltered environment."""
import json
from pathlib import Path
import subprocess
import time

raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory',
    '--format=csv,noheader,nounits'],text=True)
records=[]
for line in raw.strip().splitlines():
    pid,gpu,memory=[value.strip() for value in line.split(',')]
    chain=[];current=int(pid)
    for _ in range(6):
        if current<2:break
        root=Path('/proc')/str(current)
        try:
            fields=(root/'stat').read_text().rsplit(')',1)[1].split()
            allowed={'SLURM_JOB_ID','SLURM_STEP_ID','SLURM_STEPID','E1_RUN_ID','RBA_H100_ALLOCATION_ID'}
            env={}
            for item in (root/'environ').read_bytes().split(b'\0'):
                key,separator,value=item.partition(b'=')
                if separator and key.decode(errors='replace') in allowed:
                    env[key.decode()]=value.decode(errors='replace')
            chain.append(dict(pid=current,uid=root.stat().st_uid,starttick=int(fields[19]),
                ppid=int(fields[1]),comm=(root/'comm').read_text().strip(),job_environment=env))
            current=int(fields[1])
        except (FileNotFoundError,PermissionError):break
    records.append(dict(pid=int(pid),gpu_uuid=gpu,used_mib=memory,ancestry=chain))
print(json.dumps(dict(time=time.time(),records=records,scope='Read-only GPU process ancestry; no credentials or signals')))
