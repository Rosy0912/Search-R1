"""Synthetic provenance checks; these do not prove GPU recovery."""
import json,tempfile
from pathlib import Path
from verify_updates import verify
with tempfile.TemporaryDirectory() as directory:
 root=Path(directory)
 for role in ('main','reranker'):
  for rank in range(4):
   rows=[]
   for job in ('old','new'):
    rows.append(dict(kind='initialized',job=job,role=role,rank=rank,uuid='GPU-'+str(rank),cgroup='job_'+job))
   steps=range(1,7) if role=='main' else (6,)
   for step in steps:
    rows.append(dict(kind='update',job='old' if step<=5 else 'new',role=role,rank=rank,outer=step,optimizer_calls=20 if role=='main' else 1,local_rows=640,world_size=4,metrics={'loss':1.},scheduler_before=step-1,scheduler_after=step))
   (root/(role+'-'+str(rank)+'.jsonl')).write_text(''.join(json.dumps(x)+'\n' for x in rows))
 result=verify(root,6,'new',prior_job='old',prior_through=5)
 assert result['joint_iterations']==[6] and result['main_updates_per_rank']==120
 p=root/'main-0.jsonl';rows=[json.loads(l) for l in p.read_text().splitlines()];repeat=dict(rows[-1],outer=5);rows.append(repeat);p.write_text(''.join(json.dumps(x)+'\n' for x in rows))
 try:verify(root,6,'new',prior_job='old',prior_through=5)
 except RuntimeError as e:assert 'repeated' in str(e)
 else:raise AssertionError('Repeated completed step accepted')
print('SYNTHETIC_RECOVERY_PROVENANCE_AND_NO_REPEAT_PASS')
