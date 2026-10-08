"""One independently reviewed same-node zero-write calibration submission."""
import hashlib, json, os, socket, subprocess, time
from pathlib import Path

ROOT = Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
TOOLS = Path(os.environ['TREE_SAFETY110_PACKAGE_PATH'])
PACKAGE = os.environ['TREE_SAFETY110_PACKAGE_SHA256']
REVIEW = os.environ['TREE_SAFETY110_REVIEW_SHA256']

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''): h.update(block)
    return h.hexdigest()

assert socket.gethostname().split('.')[0] == 'h100-9'
assert TOOLS.resolve().is_relative_to(ROOT/'instruct300/candidates')
assert sha(TOOLS/'manifest.json') == PACKAGE
assert sha(TOOLS/'PRE_DISPATCH_RECEIPT.json') == REVIEW
review = json.loads((TOOLS/'PRE_DISPATCH_RECEIPT.json').read_bytes())
manifest = json.loads((TOOLS/'manifest.json').read_bytes())
assert review['status'] == 'READY_FOR_HEALTHY110_ZERO_WRITE_CALIBRATION'
assert review['manifest_sha256'] == PACKAGE and review['code_commit'] == manifest['code_commit']
assert review['submit_helper_sha256'] == sha(__file__)
assert review['requires_zero_optimizer_writes'] is True and review['long_training_ready'] is False
assert manifest['stage'] == 'calibration110' and manifest['parent_manifest']['identity']['completed_step'] == 110
assert manifest['probability_calibration']['environment'] == {'TREE_PPO_CALIBRATION_ONLY': '1'}
prior = json.loads((ROOT/'runs/tree-recovery300-3234/run-result.json').read_bytes())
assert prior['status'] == 'FAILED' and prior['cleanup_residual'] == {} and prior['job_id'] == '3234'
protected = {**{pid:'1502352181' for pid in (1680782,1680783,1680784,1680785)},
             **{pid:'1502398762' for pid in (1696443,1696444,1696445,1696446)}}
for pid, start in protected.items():
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()
    assert fields[19] == start and fields[0] not in ('Z','X'), 'Protected process changed'
active = subprocess.check_output(['squeue','-h','-w','h100-9','-o','%i %j %T'],text=True)
for own_name in ('tree-instruct300','tree-recovery300','tree-safety300','tree-safety110-calibration','tree-policy-safety-native'):
    assert own_name not in active, 'Existing own GPU job; do not duplicate: '+own_name
for name, expected in manifest['files'].items(): assert sha(TOOLS/name) == expected, name
for name, expected in manifest['external_files'].items(): assert sha(name) == expected, name
assert sha(manifest['parent_manifest']['path']) == manifest['parent_manifest']['sha256']
journal = ROOT/'instruct300'/('SUBMISSION_ONCE-safety110-'+PACKAGE[:12]+'.json')
intent = dict(status='SUBMITTING_DO_NOT_RETRY_ON_UNKNOWN',epoch=time.time(),stage='calibration110',
              parent_completed=110,expected_new_optimizer_writes=0,historical_failed_completed=128,
              package_sha256=PACKAGE,review_sha256=REVIEW,source=manifest['code_commit'])
with journal.open('x') as stream:
    json.dump(intent,stream);stream.flush();os.fsync(stream.fileno())
env = os.environ.copy()
env.update(TREE_INSTRUCT_TOOLS=str(TOOLS),TREE_INSTRUCT_PACKAGE_SHA256=PACKAGE,TREE_INSTRUCT_REVIEW_SHA256=REVIEW)
try:
    proc = subprocess.run(['sbatch','--parsable','--output='+str(ROOT/'logs/tree-safety110-calibration-%j.out'),
                           '--error='+str(ROOT/'logs/tree-safety110-calibration-%j.err'),str(TOOLS/'launch.sbatch')],
                          env=env,text=True,capture_output=True,timeout=60)
    intent.update(returncode=proc.returncode,stdout=proc.stdout,stderr=proc.stderr)
    if proc.returncode == 0:
        job = proc.stdout.strip().split(';')[0]
        assert job.isdigit(), proc.stdout
        intent.update(status='SUBMITTED',job_id=int(job))
    else: intent['status'] = 'SUBMISSION_FAILED_NO_AUTOMATIC_RETRY'
except BaseException as exc:
    intent.update(status='SUBMISSION_UNKNOWN_DO_NOT_RETRY',exception=repr(exc));raise
finally:
    temporary = journal.with_suffix('.final.tmp')
    with temporary.open('x') as stream:
        json.dump(intent,stream,indent=2);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,journal)
    print(json.dumps(intent),flush=True)
