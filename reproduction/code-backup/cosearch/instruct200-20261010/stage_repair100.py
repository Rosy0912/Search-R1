"""HK3 to HPC historical checkpoint100 staging; password stdin/RAM, durable receipts."""
import fcntl,hashlib,json,os,shlex,subprocess,sys,time
from pathlib import Path
from hkust3_archive_transport import SSH
SOURCE=Path('/backup/shaoshitong/cosearch-instruct200-common-em-20261011/checkpoints/global_step_100')
ROOT='/data/user/jxu957/experiments/cosearch-continue300-x203-20261008/runtime/cosearch-instruct200-common-em-20261011'
STATUS=Path('/backup/shaoshitong/cosearch-instruct200-common-em-20261011/repair100-stage.json')
lock=STATUS.with_suffix('.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
env=dict(os.environ,SSHPASS=sys.stdin.readline().rstrip('\n'))
def remote(code):return subprocess.run(SSH+['python3 -c '+shlex.quote(code)],env=env,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
def state(status,**kw):STATUS.write_text(json.dumps(dict(status=status,pid=os.getpid(),epoch=time.time(),**kw),indent=2)+'\n')
try:
    assert (SOURCE/'hkust3-archive-receipt.json').exists()
    receipt=json.loads((SOURCE/'hkust3-archive-receipt.json').read_text())
    assert receipt['step']==100 and len(receipt['files'])==33
    for item in receipt['files']:
        p=SOURCE/item['path'].split('/global_step_100/')[1]
        h=hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
        assert p.stat().st_size==item['bytes'] and h.hexdigest()==item['sha256'],str(p)
        state('VERIFYING_SOURCE',verified_file=str(p))
    remote('from pathlib import Path;Path('+repr(ROOT+'/checkpoints/global_step_100')+').mkdir(parents=True,exist_ok=True)')
    state('TRANSFERRING')
    subprocess.run(['rsync','-a','--partial','--append-verify','-e',' '.join(shlex.quote(x) for x in SSH[:-1]),str(SOURCE)+'/', 'mou176@10.120.81.30:'+ROOT+'/checkpoints/global_step_100/'],env=env,check=True)
    state('TRANSFER_COMPLETE_PENDING_COMPUTE_SHA_AND_BOUNDARY_EVALUATION')
except BaseException as e:state('FAILED',error=str(e));raise
