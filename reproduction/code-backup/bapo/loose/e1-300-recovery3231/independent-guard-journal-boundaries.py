import sys,json,copy
from pathlib import Path
sys.path.insert(0,str(Path('work/e1-300-recovery3231/tools').resolve()))
import test_guard_journal as f
results=[]
t=f.Tests()
for case in ('cpu_bad_parent','cpu_parent_cycle','cpu_missing_certificate','certificate_foreign_run','certificate_wrong_job','certificate_claims_missing_leases','prelaunch_cont','late_safe'):
 x=f.recovery()
 if case.startswith('cpu_'):
  # CPU child not present in raw GPU identity inventory; only complete certified ancestry authorizes freeze.
  c=f.certificate(103,'phase_descendant',lease=False) if hasattr(f,'certificate')else None
  if c is None: raise RuntimeError('inspect fixture function name')
  c['ppid']=101
  e=f.event('owned_freeze_signal',pid=103,start_ticks=113,uid=0,role='phase_descendant',lease_uuids=f.UUIDS,certificate=c)
  if case=='cpu_bad_parent':c['ppid']=999
  if case=='cpu_parent_cycle':c['ppid']=103
  if case=='cpu_missing_certificate':e.pop('certificate')
  x.insert(4,e)
 elif case=='certificate_foreign_run':x[2]['certificate']['run']='/foreign'
 elif case=='certificate_wrong_job':x[2]['certificate']['slurm_job_id']='901'
 elif case=='certificate_claims_missing_leases':x[2]['certificate']['gpu_leases_inherited']=False;x[2]['certificate']['inherited_gpu_lease_uuids']=[]
 elif case=='prelaunch_cont':x=t.prelaunch();x.insert(2,f.event('owned_resume_signal',pid=101,start_ticks=111,signal=18));x[2]['phase_pid']=None
 elif case=='late_safe':
  for z in x:
   if z.get('event')=='recovery_sample'and z.get('pause')is False:z['elapsed_seconds']=60;break
 try:t.run_rows(x)
 except (ValueError,TypeError,KeyError)as e:results.append(dict(case=case,rejected=True,error=str(e)));continue
 results.append(dict(case=case,rejected=False))
o=dict(scope='LOCAL_RAW_ASSESSOR_AND_GRAMMAR_ONLY_NOT_GPU_ACCEPTANCE',results=results,all_rejected=all(x['rejected']for x in results))
Path('work/evidence/recovery3231-independent-guard-journal-boundaries.json').write_text(json.dumps(o,indent=2)+'\n');print(json.dumps(o,indent=2));assert o['all_rejected']
