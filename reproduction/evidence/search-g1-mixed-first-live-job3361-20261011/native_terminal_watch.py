"""Read-only native terminal receipt capture; never submits or signals a job."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


def atomic(path,value):
    temp=path.with_suffix('.tmp')
    with temp.open('w') as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def run(base,job):
    assert job.isdecimal();base=Path(base).resolve()
    target=base/'logs/native-terminal-receipt.json'
    assert not target.exists()
    source=Path(__file__).read_bytes()
    stat=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
    birth=dict(pid=os.getpid(),start_ticks=int(stat[19]),sid=os.getsid(0),job=job,
        base=str(base),source_sha256=hashlib.sha256(source).hexdigest(),scope='read-only Slurm observation; no submit/cancel/signal')
    atomic(base/'logs/native-terminal-watcher.json',birth)
    last=None;misses=0
    while True:
        observed=time.time()
        r=subprocess.run(['scontrol','show','job',job,'-o'],capture_output=True,text=True,timeout=20)
        fields=dict(re.findall(r'(\w+)=(\S+)',r.stdout))
        state=fields.get('JobState')
        value=dict(observed_unix=observed,job=job,returncode=r.returncode,
                   raw_stdout=r.stdout,raw_stderr=r.stderr,fields=fields,watcher=birth)
        if r.returncode==0 and fields.get('JobId')==job:
            last=value;misses=0
            atomic(base/'logs/native-current-receipt.json',value)
            if state not in ('PENDING','RUNNING','CONFIGURING','COMPLETING','SUSPENDED','REQUEUED','RESIZING'):
                assert 'ExitCode' in fields
                value['terminal_proved']=True
                atomic(target,value)
                print(json.dumps({'terminal':state,'ExitCode':fields['ExitCode']}),flush=True)
                return
        else:
            misses+=1
            # Never infer an exit code from disappeared or temporarily unreachable Slurm state.
            atomic(base/'logs/native-observation-error.json',dict(value,last_successful=last))
            if misses>=40:
                atomic(base/'logs/native-terminal-unproved.json',dict(value,last_successful=last))
                return
        time.sleep(15)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--job',required=True)
    a=p.parse_args();run(a.base,a.job)
