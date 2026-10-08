"""Native own-Slurm cleanup probe only; never submits or loads GPU/model assets."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

root = Path(os.environ['AGENT_ROOT'])
spec = importlib.util.spec_from_file_location('topd_native_launcher', root/'tools/topd-short-run.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)
job = os.environ['SLURM_JOB_ID']
scope, relative = launcher.cgroup_scope(job)
# A deliberately retained process born before the OwnedProcesses baseline.
guardian = subprocess.Popen(['/bin/sleep','300'])
guardian_fd = os.pidfd_open(guardian.pid)
owned = launcher.OwnedProcesses(scope, relative)
base = root/'logs'/('cleanup-native-'+job)
base.mkdir()
result = dict(scope='Native own-step cleanup infrastructure only; no GPU/model/training',
              job_id=job, cgroup=str(scope), launcher_sha256=launcher.sha(root/'tools/topd-short-run.py'),
              guardian_pid=guardian.pid, baseline_pids=sorted(owned.baseline))
try:
    direct=owned.spawn(['/bin/sleep','300'], base/'direct.log')
    # Controlled own child requires a real3s TERM grace period; old2s TERM budget killed it.
    slow_file=base/'slow-ready.pid'
    slow_code='''import os,signal,time
from pathlib import Path
def stop(*_):
 time.sleep(3)
 raise SystemExit(0)
signal.signal(signal.SIGTERM,stop)
Path(os.environ['SLOW_READY_FILE']).write_text(str(os.getpid()))
while True:time.sleep(1)
'''
    slow=owned.spawn([sys.executable,'-c',slow_code],base/'slow-direct.log',dict(os.environ,SLOW_READY_FILE=str(slow_file)))
    deadline=time.monotonic()+15
    while not slow_file.exists():
        assert slow.poll() is None and time.monotonic()<deadline
        time.sleep(.1)
    code = '''import os,signal,time
from pathlib import Path
pid=os.fork()
if pid:
 raise SystemExit(0)
os.setsid()
signal.signal(signal.SIGTERM,signal.SIG_IGN)
Path(os.environ['DAEMON_PID_FILE']).write_text(str(os.getpid()))
while True:time.sleep(1)
'''
    daemon_file = base/'daemon.pid'
    parent = owned.spawn([sys.executable,'-c',code],base/'daemon.log',dict(os.environ,DAEMON_PID_FILE=str(daemon_file)))
    parent.wait(timeout=30)
    deadline=time.monotonic()+15
    while not daemon_file.exists():
        assert time.monotonic()<deadline
        time.sleep(.1)
    daemon=int(daemon_file.read_text())
    assert daemon in launcher.cgroup_pids(scope)-owned.baseline
    result.update(daemon_pid=daemon,slow_direct_pid=slow.pid,slow_direct_identity=launcher.process_identity(slow.pid), new_owned_pids_before=sorted(launcher.cgroup_pids(scope)-owned.baseline))
    owned.cleanup()
    result.update(owned_pids_after=sorted(launcher.cgroup_pids(scope)-owned.baseline), guardian_preserved=guardian.poll() is None)
    result.update(cleanup_record=owned.cleanup_record,direct_returncode=direct.returncode,slow_direct_returncode=slow.returncode)
    assert not result['owned_pids_after'] and result['guardian_preserved'] and slow.returncode==0
    result['status']='PASS'
    print('NATIVE_OWN_CGROUP_PIDFD_DIRECT_DAEMON_KILL_GUARDIAN_PASS_NO_TRAINING',flush=True)
except BaseException as exc:
    result.update(status='FAIL',error_type=type(exc).__name__,error=str(exc),cleanup_record=getattr(owned,'cleanup_record',None),remaining_identity=[launcher.process_identity(pid) for pid in sorted(launcher.cgroup_pids(scope)-owned.baseline)])
    raise
finally:
    if guardian.poll() is None:signal.pidfd_send_signal(guardian_fd,signal.SIGTERM)
    guardian.wait(timeout=10)
    os.close(guardian_fd)
    (root/'logs'/('launcher-cleanup-native-'+job+'.json')).write_text(json.dumps(result,indent=2))
