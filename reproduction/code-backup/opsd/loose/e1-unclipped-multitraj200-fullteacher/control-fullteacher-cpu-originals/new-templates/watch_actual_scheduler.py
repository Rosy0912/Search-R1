"""Preserve actual own scheduler terminal before retention expires; read-only."""
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import time

HERE=Path(__file__).resolve().parent
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL',
          'OUT_OF_MEMORY','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def main():
    p=argparse.ArgumentParser();p.add_argument('--job-id',required=True)
    p.add_argument('--source',required=True);p.add_argument('--hours',type=float,default=192)
    a=p.parse_args()
    if re.fullmatch(r'\d+',a.job_id) is None or int(a.job_id)<=0 or re.fullmatch('[0-9a-f]{40}',a.source) is None:
        raise ValueError('Actual positive job ID and real source required')
    actual=(HERE/'actual-job-id.txt').read_text().strip()
    submitted=json.loads((HERE/'submit-intent.json').read_bytes())
    if actual!=a.job_id or submitted['source_commit']!=a.source:
        raise ValueError('Actual submitted job/source differ')
    prefix=HERE/('job'+a.job_id+'-scheduler')
    ready=dict(utc=now(),watcher_pid=os.getpid(),job_id=a.job_id,source_commit=a.source,
               expected_scheduler_uid=0,hours=a.hours,poll_seconds=15,scope='Read-only scheduler observation')
    with Path(str(prefix)+'-watcher-ready.json').open('x') as f:json.dump(ready,f,sort_keys=True);f.write('\n')
    deadline=time.monotonic()+a.hours*3600
    with Path(str(prefix)+'-originals.ndjson').open('x',buffering=1) as handle:
        while time.monotonic()<deadline:
            observation=dict(utc=now(),job_id=a.job_id)
            try:
                result=subprocess.run(['python3',str(HERE.parent/'head.py'),
                    'scontrol show job -o '+a.job_id],capture_output=True,text=True,timeout=65)
                observation.update(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
                if result.returncode==0:
                    if not re.search(r'\bJobId='+re.escape(a.job_id)+r'\b',result.stdout):
                        observation['validation_error']='Unexpected scheduler job identity'
                    elif not re.search(r'\bUserId=[^\s(]+\(0\)(?:\s|$)',result.stdout):
                        observation['validation_error']='Scheduler job is not owned by actual root UID0'
                    else:
                        match=re.search(r'\bJobState=([^\s]+)',result.stdout)
                        if match is None:observation['validation_error']='Actual scheduler state absent'
                        else:observation['actual_state']=match.group(1)
            except subprocess.TimeoutExpired as exc:
                observation.update(timeout=True,stdout=(exc.stdout or b'').decode(),stderr=(exc.stderr or b'').decode())
            except OSError as exc:
                observation['read_error']=dict(type=type(exc).__name__,errno=exc.errno,message=str(exc))
            handle.write(json.dumps(observation,sort_keys=True)+'\n');handle.flush();os.fsync(handle.fileno())
            if observation.get('actual_state') in TERMINAL:
                # Preserve the scheduler's literal original, never infer it
                # from training exit0, NVML absence, or disappearance of a job.
                with Path(str(prefix)+'-terminal-original.txt').open('x') as f:
                    f.write(observation['stdout']);f.flush();os.fsync(f.fileno())
                print('Observed actual scheduler terminal '+observation['actual_state'],flush=True)
                return
            time.sleep(15)
    raise SystemExit('No actual scheduler terminal observed before watcher timeout')

if __name__=='__main__':main()
