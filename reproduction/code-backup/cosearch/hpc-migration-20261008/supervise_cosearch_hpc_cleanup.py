"""Continue explicitly authorized deletion of two unrelated HPC environment dirs."""
import fcntl, importlib.util, json, os, shlex, subprocess, time
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
E=ROOT/'outputs/evidence'
lock=(E/'native-x-hpc-cleanup-supervisor.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
s=importlib.util.spec_from_file_location('ctl','/Users/shaoshitong/Documents/remote_codex_control/control.py')
c=importlib.util.module_from_spec(s);s.loader.exec_module(c)
code=(E/'native-x-hpc-cleanup-remaining-worker-20261008.py').read_text()
attempt=0
while True:
 state={'pid':os.getpid(),'attempt':attempt,'epoch':time.time(),'status':'cleaning','paths':['/data/user/jxu957/independent_project','/data/user/jxu957/envs1']}
 (E/'native-x-hpc-cleanup-supervisor.json').write_text(json.dumps(state,indent=2))
 with (E/'native-x-hpc-cleanup-supervisor.remote.log').open('a') as log:
  try:r=subprocess.run(c.route('hpc3','/usr/bin/python3 -u -c '+shlex.quote(code)),stdout=log,stderr=log,timeout=1800);exit_code=r.returncode
  except subprocess.TimeoutExpired:exit_code=124
 if exit_code==0:
  state.update(status='complete_all_other_data_deleted',epoch=time.time())
  (E/'native-x-hpc-cleanup-supervisor.json').write_text(json.dumps(state,indent=2));break
 attempt+=1
 state.update(status='retry_backoff',exit=exit_code,epoch=time.time())
 (E/'native-x-hpc-cleanup-supervisor.json').write_text(json.dumps(state,indent=2))
 time.sleep(min(120,15*attempt))
