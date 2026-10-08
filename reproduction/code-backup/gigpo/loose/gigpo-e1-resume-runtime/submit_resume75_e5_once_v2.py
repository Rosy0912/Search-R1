"""Claim an immutable intent once; a lost SSH response never authorizes a second submit."""
import hashlib,json,os,subprocess
from pathlib import Path
ROOT=Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004')
CANDIDATE=ROOT/'logs/deployment-candidate-e5b30d7-formal-v2.json'
REVIEW=ROOT/'logs/RESUME75-E5B30D7-DEPLOYMENT-INDEPENDENT-REVIEW-v2.json'
BINDING=ROOT/'logs/deployment-binding-resume75-e5b30d7-v2.json'
INTENT=ROOT/'logs/resume75-e5b30d7-submit-intent-v2.json'
RECEIPT=ROOT/'logs/resume75-e5b30d7-submit-receipt-v2.json'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,value):
    with path.open('x') as stream:
        stream.write(json.dumps(value,indent=2)+'\n');stream.flush();os.fsync(stream.fileno())
    directory_fd=os.open(str(path.parent),os.O_RDONLY)
    try:os.fsync(directory_fd)
    finally:os.close(directory_fd)


def main():
    assert not BINDING.exists() and not INTENT.exists() and not RECEIPT.exists(), 'Existing intent/receipt: inspect scheduler and original receipt; never resubmit'
    queued=subprocess.check_output(['squeue','-h','-w','h100-8','-n','gigpo-e1-resume75','-o','%i %T'],text=True)
    assert not queued.strip(), 'Existing live continuation: inspect rather than submit'
    candidate=json.loads(CANDIDATE.read_text());review=json.loads(REVIEW.read_text())
    assert sha(CANDIDATE)=='c43df5f9df2eb6f05d53401c5de95a592a78bb8f2ba3bcd113a72d389890df39'
    assert review['status']=='PASS' and review['candidate_sha256']==sha(CANDIDATE)
    assert candidate['status']=='CANDIDATE_NOT_READY'
    candidate.update(status='REVIEWED',review_path=str(REVIEW),review_sha256=sha(REVIEW))
    candidate['evidence'].append(dict(path=str(REVIEW),sha256=sha(REVIEW)))
    # Leave the claim even on any exception. Explicitly preserve sbatch stderr and
    # its return code before interpretation, so an uncertain transport is auditable.
    write(INTENT,dict(status='CLAIMED_NO_RESUBMISSION',candidate_sha256=sha(CANDIDATE),binding=str(BINDING),restore_checkpoint=candidate['restore_checkpoint']))
    write(BINDING,candidate)
    completed=subprocess.run(['sbatch','--parsable',str(Path(candidate['helpers'])/'formal_resume_v1.sbatch'),str(BINDING),sha(BINDING),candidate['repo'],candidate['helpers']],text=True,capture_output=True)
    raw=dict(stdout=completed.stdout,stderr=completed.stderr,returncode=completed.returncode,binding=str(BINDING),binding_sha256=sha(BINDING))
    job=completed.stdout.strip().split(';')[0]
    if completed.returncode==0 and job.isdigit():raw['job']=job
    write(RECEIPT,raw)
    print(json.dumps(raw),flush=True)
    assert completed.returncode==0 and job.isdigit(), 'Submission did not yield a definite job: preserve receipt and inspect; do not repeat'


if __name__=='__main__':main()
