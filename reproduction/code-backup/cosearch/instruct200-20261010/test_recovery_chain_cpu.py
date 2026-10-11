"""Synthetic multi-job evidence; no GPU recovery claim."""
import json,tempfile
from pathlib import Path
from verify_updates import verify
with tempfile.TemporaryDirectory() as d:
 root=Path(d)
 for role in ('main','reranker'):
  for rank in range(4):
   rows=[dict(kind='initialized',job=j,role=role,rank=rank,uuid='GPU-'+str(rank),cgroup='job_'+j) for j in ('a','b','c')]
   for step in range(1,12):
    rows.append(dict(kind='update',job='a' if step<=5 else 'b' if step<=10 else 'c',role=role,rank=rank,outer=step,optimizer_calls=20 if role=='main' else 1,local_rows=640,world_size=4,metrics={'loss':1.},scheduler_before=step-1,scheduler_after=step))
   (root/(role+'-'+str(rank)+'.jsonl')).write_text(''.join(json.dumps(x)+'\n' for x in rows))
 args=dict(prior_job='b',prior_through=10,prior_history=dict(prior_job='a',prior_through=5))
 assert verify(root,11,'c',**args)['joint_iterations']==list(range(1,12))
 p=root/'main-0.jsonl';xs=[json.loads(x) for x in p.read_text().splitlines()];xs.append(dict(xs[-1],outer=10));p.write_text(''.join(json.dumps(x)+'\n' for x in xs))
 try:verify(root,11,'c',**args)
 except RuntimeError as e:assert 'repeated' in str(e)
 else:raise AssertionError('Repeated outer accepted')
print('MULTI_JOB_EXACT_INTERVAL_AND_NO_REPEAT_PASS')
