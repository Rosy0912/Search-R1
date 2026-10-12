"""Claim an immutable intent once; a lost SSH response never authorizes a second submit."""
import hashlib,json,os,subprocess
from pathlib import Path
ROOT=Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004')
CANDIDATE=ROOT/'logs/deployment-candidate-28a5017-v1.json'
REVIEW=ROOT/'logs/RESUME75-28A5017-DEPLOYMENT-INDEPENDENT-REVIEW.json'
BINDING=ROOT/'logs/deployment-binding-resume75-28a5017-v1.json'
INTENT=ROOT/'logs/resume75-28a5017-submit-intent-v1.json'
RECEIPT=ROOT/'logs/resume75-28a5017-submit-receipt-v1.json'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,value):
    with path.open('x') as stream:
        stream.write(json.dumps(value,indent=2)+'\n');stream.flush();os.fsync(stream.fileno())


def main():
    assert not BINDING.exists() and not INTENT.exists() and not RECEIPT.exists(), 'Existing intent/receipt: inspect scheduler and original receipt; never resubmit'
    candidate=json.loads(CANDIDATE.read_text());review=json.loads(REVIEW.read_text())
    assert sha(CANDIDATE)=='2dd3a65825177d18641fc56fc3581afa8bfd9701fbb9b64cd81b61da115ced60'
    assert review['status']=='PASS' and review['candidate_sha256']==sha(CANDIDATE)
    assert candidate['status']=='PENDING_INDEPENDENT_DEPLOYMENT_REVIEW'
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
