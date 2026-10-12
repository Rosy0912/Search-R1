"""CPU-only full SHA gate before a single authorized recovery submission and audit."""
import fcntl,hashlib,json,os,shlex,subprocess,time
from pathlib import Path
ROOT=Path('/data/user/jxu957/experiments/cosearch-continue300-x203-20261008/runtime/cosearch-instruct200-common-em-20261011')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(8*1024*1024),b''):h.update(c)
 return h.hexdigest()
def main():
 lock=(ROOT/'recovery100-finalize.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 statepath=ROOT/'recovery100-finalize-state.json'
 def state(status,**kw):statepath.write_text(json.dumps(dict(status=status,epoch=time.time(),pid=os.getpid(),**kw),indent=2)+'\n')
 try:
  checkpoint=ROOT/'checkpoints/global_step_100'
  expected=json.loads((ROOT/'checkpoint-receipt-100.json').read_text())
  archived=json.loads((checkpoint/'hkust3-archive-receipt.json').read_text())
  def entries(d):return {x['path'].split('/global_step_100/')[1]:(x['bytes'],x['sha256']) for x in d['files']}
  wanted=entries(expected)
  if expected['step']!=100 or archived['step']!=100 or archived['status']!='HKUST3_ALL_CHECKPOINT_SHA_PASS' or len(wanted)!=33 or wanted!=entries(archived):raise RuntimeError('Source100/archive receipt mismatch')
  for i,(name,(size,digest)) in enumerate(wanted.items(),1):
   p=checkpoint/name
   if p.stat().st_size!=size or sha(p)!=digest:raise RuntimeError('Target100 SHA differs: '+name)
   state('VERIFYING_TARGET_SHA',files_pass=i,total=33,path=name)
  saved=checkpoint/'full-sha-receipt.json'
  if entries(json.loads(saved.read_text()))!=wanted:raise RuntimeError('Full recovery receipt differs')
  plan=json.loads((ROOT/'recovery100-plan-pending-stage.json').read_text());plan['receipt_sha256']=sha(saved)
  (ROOT/'recovery-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
  (ROOT/'repair100-full-sha-verified.json').write_text(json.dumps(dict(status='ALL33_SOURCE_ARCHIVE_TARGET_SHA_PASS',bytes=sum(v[0] for v in wanted.values()),epoch=time.time(),receipt_sha256=plan['receipt_sha256']),indent=2)+'\n')
  from verify_updates import verify
  verify(ROOT/'evidence',100,'709923',require_joint=False,**plan['prior_history'])
  from submit import main as submit
  state('TARGET_SHA_PASS_SUBMITTING_ONCE')
  submit()
  job=(ROOT/'job.id').read_text().strip()
  out=ROOT/'dependent-final-audit'
  # Original CPU710171 outcome is preserved in failed-job-709923; bind one new audit to this job.
  claim=out/('submission-claim-'+job+'.json')
  with claim.open('x') as f:json.dump(dict(dependency=int(job),epoch=time.time(),status='SUBMISSION_CLAIMED'),f)
  python='/data/user/jxu957/experiments/cosearch-continue300-x203-20261008/source/author-native/python/cpython-3.11.17-linux-x86_64-gnu/bin/python3.11'
  wrap=' '.join(shlex.quote(x) for x in [python,str(out/'audit.py'),str(ROOT),job])
  result=subprocess.run(['sbatch','--parsable','--job-name=cosearch-final-cpu-audit','--partition=debug','--cpus-per-task=2','--mem=16G','--time=00:25:00','--dependency=afterany:'+job,'--output='+str(out/'slurm-%j.log'),'--wrap='+wrap],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
  (out/('submission-result-'+job+'.json')).write_text(json.dumps(dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr),indent=2)+'\n')
  if result.returncode:raise RuntimeError('Dependent CPU audit submission rejected: '+result.stderr)
  audit=result.stdout.strip().split(';')[0]
  if not audit.isdigit():raise RuntimeError('Unknown CPU audit submission outcome')
  (out/'job.id').write_text(audit+'\n');(out/'submission-claim.json').write_text(json.dumps(dict(dependency=int(job),audit_job=int(audit),publication=os.environ['COSEARCH_RECOVERY_PUBLICATION'],epoch=time.time()),indent=2)+'\n')
  state('RECOVERY_AND_CPU_DEPENDENCY_SUBMITTED_NOT_RUNTIME_ACCEPTED',job=job,audit_job=audit,restored_step=100,recompute=[101,140])
 except BaseException as e:state('FAILED_REQUIRES_REVIEW',error_type=type(e).__name__,error=str(e));raise
if __name__=='__main__':main()
