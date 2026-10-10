"""Submit once only after authorized storage and a verified deployed source manifest."""
import fcntl,hashlib,json,subprocess,time
from pathlib import Path
ROOT=Path('/data/user/jxu957/experiments/cosearch-continue300-x203-20261008/runtime/cosearch-instruct200-20261010')

def main():
    lock=(ROOT/'submit.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    approval=json.loads((ROOT/'storage-authorization.json').read_text())
    if approval.get('human_authorized') is not True or approval.get('budget_bytes',0)<1000000000000 or approval.get('archive_strategy')!='hkust3_verified_then_delete':raise RuntimeError('Storage authorization pending')
    heartbeat=json.loads((ROOT/'archive-worker-heartbeat.json').read_text())
    if time.time()-heartbeat['epoch']>120:raise RuntimeError('HKUST3 archive worker not freshly reachable')
    manifest=json.loads((ROOT/'deployment-manifest.json').read_text())
    if not manifest.get('published_remote_sha_verified'):raise RuntimeError('Publication not verified')
    for name,expected in manifest['sha256'].items():
        actual=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        if actual!=expected:raise RuntimeError('Deployed source changed: '+name)
    claim=ROOT/'submit-claim.json'
    with claim.open('x') as f:json.dump({'epoch':time.time(),'state':'SUBMISSION_ATTEMPT_CLAIMED','publication':manifest['publication_commit']},f)
    # A connection failure after this point must inspect claim/queue, never submit again blindly.
    r=subprocess.run(['sbatch','--parsable','--output='+str(ROOT/'slurm-%j.log'),str(ROOT/'tools/job.sbatch')],capture_output=True,text=True)
    (ROOT/'submission-result.json').write_text(json.dumps({'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'epoch':time.time()},indent=2)+'\n')
    if r.returncode:raise RuntimeError('Slurm rejected submission; inspect recorded result')
    job=r.stdout.strip().split(';')[0]
    if not job.isdigit():raise RuntimeError('Unrecognized sbatch response; inspect queue before any action')
    (ROOT/'job.id').write_text(job+'\n');print(json.dumps({'status':'SUBMITTED_NOT_RUNTIME_ACCEPTED','job':job}))
if __name__=='__main__':main()
