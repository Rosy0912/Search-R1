"""Single reviewed recovery submission; unknown outcomes require inspection."""
import hashlib,json,os,socket,subprocess,time
from pathlib import Path

ROOT=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
TOOLS=ROOT/'instruct300/candidates/recovery-502e9fa-f91c4f17'
PACKAGE='f91c4f17a3a3454bb16a81e9ca400f4db07a56a0a27cf6cfac2c1814459c3087'
REVIEW=os.environ['TREE_RECOVERY_DISPATCH_REVIEW_SHA256']
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

assert socket.gethostname().split('.')[0]=='h100-9'
assert sha(TOOLS/'manifest.json')==PACKAGE
assert sha(TOOLS/'PRE_DISPATCH_RECEIPT.json')==REVIEW
review=json.loads((TOOLS/'PRE_DISPATCH_RECEIPT.json').read_bytes())
assert review['status']=='READY_FOR_NATIVE_COLD_GATE_THEN_RECOVERY300'
assert review['manifest_sha256']==PACKAGE
assert review['submit_helper_sha256']==sha(__file__),'Unreviewed submission helper'
assert review['requires_native8_fullstate_before3b'] is True
assert review['requires_independent_recovery105_before106'] is True
prior=json.loads((ROOT/'runs/tree-instruct300-3154/run-result.json').read_bytes())
assert prior['job_id']=='3154' and prior['cleanup_residual']=={}
assert prior['status']=='FAILED' and prior['code_commit']=='f994af33c6c93b6b9844863be912cab2a4020dc1'
assert prior['error']=='Own Instruct300 supervisor interrupted 15'
for pid in (1680782,1680783,1680784,1680785,1696443,1696444,1696445,1696446):
    assert Path(f'/proc/{pid}').exists(),'Protected background PID missing'
active=subprocess.check_output(['squeue','-h','-w','h100-9','-o','%i %j %T'],text=True)
assert 'tree-instruct300' not in active and 'tree-recovery300' not in active,'Existing own training, no duplicate'
m=json.loads((TOOLS/'manifest.json').read_bytes())
for name,expected in m['files'].items():assert sha(TOOLS/name)==expected,name
for name,expected in m['external_files'].items():assert sha(name)==expected,name
assert sha(m['parent_manifest']['path'])==m['parent_manifest']['sha256']
journal=ROOT/'instruct300/SUBMISSION_ONCE-recovery-f91c4f17.json'
intent=dict(status='SUBMITTING_DO_NOT_RETRY_ON_UNKNOWN',epoch=time.time(),parent_job=3154,
    parent_completed=100,parent_failed_history=262,target=300,package_sha256=PACKAGE,review_sha256=REVIEW)
with journal.open('x') as f:json.dump(intent,f);f.flush();os.fsync(f.fileno())
env=os.environ.copy();env.update(TREE_INSTRUCT_TOOLS=str(TOOLS),TREE_INSTRUCT_PACKAGE_SHA256=PACKAGE,
                                TREE_INSTRUCT_REVIEW_SHA256=REVIEW)
try:
    p=subprocess.run(['sbatch','--parsable','--output='+str(ROOT/'instruct300/recovery-slurm-%j.out'),
        '--error='+str(ROOT/'instruct300/recovery-slurm-%j.err'),str(TOOLS/'launch.sbatch')],
        env=env,text=True,capture_output=True,timeout=60)
    intent.update(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
    if p.returncode==0:
        job=p.stdout.strip().split(';')[0];assert job.isdigit(),p.stdout
        intent.update(status='SUBMITTED',job_id=int(job))
    else:intent['status']='SUBMISSION_FAILED_NO_AUTOMATIC_RETRY'
except BaseException as e:
    intent.update(status='SUBMISSION_UNKNOWN_DO_NOT_RETRY',exception=repr(e));raise
finally:
    temp=journal.with_suffix('.final.tmp')
    with temp.open('x') as f:json.dump(intent,f,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(temp,journal);print(json.dumps(intent),flush=True)
