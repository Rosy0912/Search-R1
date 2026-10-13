"""Reuse exact dual15% foreign guard; supervisor handles own cgroup cleanup."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
source=HERE/'resource_guard_retained18.py'


def main():
    # Source SHA is checked in the release before this child is spawned.
    import json
    release=json.loads(Path(os.environ['TREE_EVAL_RELEASE']).read_text())
    if hashlib.sha256(source.read_bytes()).hexdigest()!=release['files'][source.name]:
        raise RuntimeError('Frozen foreign-resource guard differs')
    original=subprocess.run
    def run(command,*args,**kwargs):
        if Path(str(command[0])).name=='scancel':
            raise RuntimeError('Bare cancellation refused; evaluation supervisor cleans only own cgroup')
        return original(command,*args,**kwargs)
    subprocess.run=run
    try:
        scope=Path('/proc/self/cgroup').read_text()
        job=os.environ['SLURM_JOB_ID']
        if not any(('job_'+job) in l.split(':',2)[-1].split('/') for l in scope.splitlines()):
            raise RuntimeError('Guard not inside own allocation')
        exec(compile(source.read_bytes(),str(source),'exec'),dict(__name__='__main__',__file__=str(source)))
    finally:
        subprocess.run=original


if __name__=='__main__':
    main()
