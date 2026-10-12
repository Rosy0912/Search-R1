"""HK3 single controller waits for transfer then runs CPU SHA/submission gate on HPC."""
import fcntl,json,os,shlex,subprocess,sys,time
from pathlib import Path
from hkust3_archive_transport import SSH
BASE=Path('/backup/shaoshitong/cosearch-instruct200-common-em-20261011')
ROOT='/data/user/jxu957/experiments/cosearch-continue300-x203-20261008/runtime/cosearch-instruct200-common-em-20261011'
lock=(BASE/'recovery100-chain.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
env=dict(os.environ,SSHPASS=sys.stdin.readline().rstrip('\n'))
def state(status,**kw):(BASE/'recovery100-chain-state.json').write_text(json.dumps(dict(status=status,pid=os.getpid(),epoch=time.time(),**kw),indent=2)+'\n')
try:
 while True:
  d=json.loads((BASE/'repair100-stage.json').read_text()) if (BASE/'repair100-stage.json').exists() else {}
  if d.get('status')=='FAILED':raise RuntimeError('Recovery100 staging failed: '+d.get('error',''))
  if d.get('status')=='TRANSFER_COMPLETE_PENDING_COMPUTE_SHA_AND_BOUNDARY_EVALUATION':break
  state('WAITING_FOR_SOURCE_SHA_AND_TRANSFER',stage=d);time.sleep(15)
 state('RUNNING_HPC_CPU_SHA_AND_SINGLE_SUBMISSION_GATE')
 command='COSEARCH_RECOVERY_PUBLICATION='+shlex.quote(os.environ['COSEARCH_RECOVERY_PUBLICATION'])+' PYTHONPATH='+shlex.quote(ROOT+'/tools')+' python3 '+shlex.quote(ROOT+'/tools/finalize_recovery100.py')
 subprocess.run(SSH+[command],env=env,check=True)
 state('HPC_GATE_FINISHED_CHECK_RUNTIME_NOT_JUST_SUBMISSION')
except BaseException as e:state('FAILED_REQUIRES_REVIEW',error=str(e));raise
