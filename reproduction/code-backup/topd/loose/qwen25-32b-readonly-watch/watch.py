"""Local 30-second observation loop, remote records survive scontrol TTL."""
import argparse
import base64
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--job',type=int,required=True);p.add_argument('--hours',type=float,default=4);args=p.parse_args()
    if args.job<=0 or not 0<args.hours<=12:raise ValueError('Actual positive job ID and bounded hours required')
    own=Path(__file__).resolve().parent
    remote=own.parent/'remote_topd.py'
    code=(own/'snapshot.py').read_bytes()
    command='python3 -c '+shlex.quote("import base64,sys;sys.argv=['snapshot.py','--job',"+repr(str(args.job))+"];exec(compile(base64.b64decode("+repr(base64.b64encode(code).decode())+"),'readonly_watch_snapshot','exec'))")
    log=own/('job-'+str(args.job)+'.jsonl');deadline=time.monotonic()+args.hours*3600
    while time.monotonic()<deadline:
        start=time.monotonic()
        try:
            result=subprocess.run([sys.executable,str(remote)],input=command,text=True,capture_output=True,timeout=90)
            row=dict(epoch=time.time(),returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
            with log.open('a') as f:f.write(json.dumps(row)+'\n')
            if result.returncode==0:
                actual=json.loads(result.stdout.strip().splitlines()[-1])
                if actual['final_state_observed']:return
        except (subprocess.TimeoutExpired,json.JSONDecodeError,KeyError) as exc:
            with log.open('a') as f:f.write(json.dumps(dict(epoch=time.time(),observation_error=str(exc)))+'\n')
        time.sleep(max(0,30-(time.monotonic()-start)))


if __name__=='__main__':main()
