"""Synthetic evidence only: validate rejection of wrong updates or GPU sharing."""
import json,tempfile
from pathlib import Path
from verify_updates import verify
with tempfile.TemporaryDirectory() as directory:
    root=Path(directory)
    for role in ('main','reranker'):
        for rank in range(4):
            identity=dict(kind='initialized',job='SYNTHETIC',rank=rank,role=role,uuid='GPU-SYNTHETIC-'+str(rank),cgroup='job_SYNTHETIC')
            update=dict(kind='update',job='SYNTHETIC',outer=1,rank=rank,role=role,world_size=4,metrics={'grad_norm':[1.]},scheduler_before=0,scheduler_after=1,optimizer_calls=20 if role=='main' else 1,local_rows=640 if role=='main' else 32)
            (root/(role+'-'+str(rank)+'.jsonl')).write_text(json.dumps(identity)+'\n'+json.dumps(update)+'\n')
    assert verify(root,1,'SYNTHETIC')['joint_iterations']==[1]
    path=root/'main-0.jsonl';identity,update=map(json.loads,path.read_text().splitlines())
    for field,value in [('optimizer_calls',19),('metrics',{'loss':float('nan')})]:
        bad=dict(update);bad[field]=value
        path.write_text(json.dumps(identity)+'\n'+json.dumps(bad)+'\n')
        try:verify(root,1,'SYNTHETIC')
        except RuntimeError:pass
        else:raise AssertionError('Invalid '+field+' accepted')
    path.write_text(json.dumps(identity)+'\n'+json.dumps(update)+'\n')
    path=root/'reranker-0.jsonl';identity,update=map(json.loads,path.read_text().splitlines());identity['uuid']='GPU-FOREIGN'
    path.write_text(json.dumps(identity)+'\n'+json.dumps(update)+'\n')
    try:verify(root,1,'SYNTHETIC')
    except RuntimeError:pass
    else:raise AssertionError('Foreign reranker GPU accepted')
print('PASS: complete synthetic shared-four evidence; wrong update count, NaN and foreign GPU rejected')
