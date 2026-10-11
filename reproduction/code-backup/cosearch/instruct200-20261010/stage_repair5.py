"""HK3 to HPC historical checkpoint5 staging; password stdin/RAM, durable receipts."""
import json,os,shlex,subprocess,sys,time
from pathlib import Path
from hkust3_archive_transport import SSH
SOURCE=Path('/backup/shaoshitong/cosearch-instruct200-common-em-20261011/checkpoints/global_step_5')
ROOT='/data/user/jxu957/experiments/cosearch-continue300-x203-20261008/runtime/cosearch-instruct200-common-em-20261011'
STATUS=Path('/backup/shaoshitong/cosearch-instruct200-common-em-20261011/repair5-stage.json')
env=dict(os.environ,SSHPASS=sys.stdin.readline().rstrip('\n'))
def remote(code):return subprocess.run(SSH+['python3 -c '+shlex.quote(code)],env=env,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
def state(status,**kw):STATUS.write_text(json.dumps(dict(status=status,pid=os.getpid(),epoch=time.time(),**kw),indent=2)+'\n')
try:
    assert (SOURCE/'hkust3-archive-receipt.json').exists()
    remote('from pathlib import Path;Path('+repr(ROOT+'/checkpoints/global_step_5')+').mkdir(parents=True,exist_ok=True)')
    state('TRANSFERRING')
    subprocess.run(['rsync','-a','--partial','--append-verify','-e',' '.join(shlex.quote(x) for x in SSH[:-1]),str(SOURCE)+'/', 'mou176@10.120.81.30:'+ROOT+'/checkpoints/global_step_5/'],env=env,check=True)
    state('TRANSFER_COMPLETE_PENDING_COMPUTE_SHA_AND_BOUNDARY_EVALUATION')
except BaseException as e:state('FAILED',error=str(e));raise
