import json,os,sys,time,subprocess
from pathlib import Path
sys.path.insert(0,'/mnt/data/shaoshitong/search-r1-rl/gigpo/tools')
from owned_runtime import OwnedProcesses,cgroup_scope,cgroup_pids
scope,relative=cgroup_scope(os.environ['SLURM_JOB_ID']);owned=OwnedProcesses(scope,relative)
r=Path('/mnt/data/shaoshitong/search-r1-rl/gigpo/logs')
child=owned.spawn([sys.executable,'-c','import subprocess,sys,time; subprocess.Popen([sys.executable,"-c","import time;time.sleep(300)"]);time.sleep(300)'],r/('cleanup-child-'+os.environ['SLURM_JOB_ID']+'.log'))
time.sleep(2);start=time.monotonic();owned.cleanup();pending=cgroup_pids(scope)-owned.baseline
print(json.dumps(dict(status='PASS',seconds=time.monotonic()-start,cleanup_residual=owned.cleanup_residual,pending=sorted(pending),own_scope=relative)),flush=True)
