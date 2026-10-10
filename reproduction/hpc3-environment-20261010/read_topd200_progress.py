"""Read-only progress audit; scheduler state is not proof of real training."""
import json,pathlib,sys
BASE=pathlib.Path('/data/user/jxu957/experiments/topd-20261008')
def main(job):
 if not job.isdigit():raise ValueError('Numeric Slurm identity required')
 root=BASE/'runtime/logs'/('topd-hpc3-'+job)/'evidence'
 outer=root/'actual-outer-updates.json'
 report={'job':job,'outer_receipt_present':outer.is_file(),'verified_full_outer_iterations':0,'rank_audit_pending':False,'errors':[]}
 if outer.is_file():
  raw=json.loads(outer.read_text());steps=raw['actual_completed_outer_steps']
  if steps!=list(range(1,len(steps)+1)) or len(steps)>200:raise ValueError('Invalid completed outer sequence')
  sys.path.insert(0,str(BASE/'runtime/tools/topd-instruct300-launch-v6/runtime'))
  from resource_audit import verify_updates
  n=len(steps);paths=[root/'actor-resource-audit'/('actor-updates-rank%d-outer%d.json'%(rank,n)) for rank in range(4)]
  if n and all(p.is_file() for p in paths):
   audit=verify_updates([json.loads(p.read_text()) for p in paths],n)
   if len(paths)!=4 or audit['optimizer_steps_per_rank']!=20*n:raise ValueError('Actual rank update geometry differs')
   audit['observed_rank_receipts']=len(paths)
   report.update(verified_full_outer_iterations=n,rank_update_audit=audit)
  elif n:report['rank_audit_pending']=True
 for name in ['train-result.json','trainer-first-error.json']:
  p=root/name
  if p.exists():
   r=json.loads(p.read_text());report[name]={k:r.get(k) for k in ['status','error','message','error_type','cleanup_passed'] if k in r}
 report['goal_complete']=False # Full final evaluation and checkpoint audit require separate acceptance.
 print(json.dumps(report,indent=2))
if __name__=='__main__':main(sys.argv[1])
