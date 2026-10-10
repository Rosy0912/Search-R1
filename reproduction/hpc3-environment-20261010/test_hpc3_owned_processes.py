"""CPU integration check on the actual Slurm cgroup, including protected baseline."""
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time
from hpc3_owned_processes import OwnedProcesses, members

row = next(x.split(':', 2) for x in pathlib.Path('/proc/self/cgroup').read_text().splitlines()
           if 'freezer' in x.split(':', 2)[1].split(','))
scope = pathlib.Path('/sys/fs/cgroup/freezer') / row[2].lstrip('/')
baseline = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
try:
    owner = OwnedProcesses(scope, row[2])
    log = pathlib.Path(tempfile.mkdtemp(dir='/data/user/jxu957/experiments/topd-20261008/outputs')) / 'child.log'
    # The root exits; its live grandchild must be adopted and cleaned up.
    owner.spawn([sys.executable, '-c',
                 'import subprocess,sys;subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"])'], log)
    # A second root ignores TERM, exercising escalation while its PID is held.
    resistant = owner.spawn([sys.executable, '-c',
                             'import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)'], log.with_name('resistant.log'))
    time.sleep(1)
    owner.cleanup()
    assert baseline.poll() is None, 'Protected baseline was signalled'
    assert resistant.returncode is not None
    assert not (members(scope) - owner.baseline)
    print(json.dumps({'status': 'OWN_CHILD_TREE_CLEANUP_PASS_BASELINE_PROTECTED',
                      'job': os.environ['SLURM_JOB_ID'], 'cleanup': owner.cleanup_record}))
finally:
    baseline.terminate()
    baseline.wait(timeout=10)
