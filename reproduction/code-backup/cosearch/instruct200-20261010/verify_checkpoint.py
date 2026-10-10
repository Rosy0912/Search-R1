"""Verify full both-role four-rank checkpoint files and publish an atomic SHA receipt."""
import hashlib,json,os,time
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def verify(root,step):
    root=Path(root);checkpoint=root/'checkpoints'/('global_step_'+str(step))
    expected=[]
    for role in ('actor','reranker_actor_rollout'):
        for rank in range(4):
            for component in ('model','optim','extra_state'):
                expected.append(checkpoint/role/(component+'_world_size_4_rank_'+str(rank)+'.pt'))
            expected.append(checkpoint/role/('recovery-state-rank-'+str(rank)+'.json'))
    expected.append(checkpoint/'data.pt')
    for p in expected:
        if not p.is_file() or not p.stat().st_size:raise RuntimeError('Incomplete checkpoint: '+str(p))
    receipt={'status':'BOTH_ROLE_FULL_CHECKPOINT_FILES_SHA_PASS','step':step,'epoch':time.time(),'files':[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in expected],'actual_reload_pending':True}
    destination=checkpoint/'full-sha-receipt.json';temporary=destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(receipt,indent=2)+'\n');os.replace(temporary,destination)
    return receipt
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('step',type=int);a=p.parse_args()
    print(json.dumps(verify(a.root,a.step)))
