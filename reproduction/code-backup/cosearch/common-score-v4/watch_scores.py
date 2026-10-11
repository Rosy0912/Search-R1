"""Lightweight stateless scoring beside training; immutable producer files and no GPUs."""
import argparse,fcntl,json,os,time
from pathlib import Path
from score_saved_boundaries import evaluation,training
from common_em import VERSION
p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();root=Path(a.root)
folder=root/'common-scoring-v4';folder.mkdir(exist_ok=True)
lock=(folder/'scorer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def state(**kw):
    out=folder/'worker-state.json';tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(dict(pid=os.getpid(),epoch=time.time(),scorer_version=VERSION,**kw),indent=2)+'\n');os.replace(tmp,out)
while True:
    try:
        for source in (root/'common-training').glob('*/completed-outer-*.json'):
            job=source.parent.name;step=int(source.stem.rsplit('-',1)[-1]);dest=folder/'common-training'/job/source.name
            if not dest.exists():
                state(status='SCORING_COMPLETE_OUTER',job=job,step=step);receipt=training(root,job,step)
                state(status='COMPLETE_OUTER_SCORED',job=job,step=step,correct=receipt['selected_update_trajectories']['correct'],denominator=2560)
        for name in ['common-historical-validation','common-validation']:
            for receipt in (root/name).glob('*.receipt.json'):
                step=int(receipt.name.split('.')[0]);dest=folder/name/receipt.name
                if not dest.exists():
                    state(status='SCORING_COMPLETE_VALIDATION',source=name,step=step);evaluation(root,name,step,512)
        state(status='WAITING_FOR_COMPLETE_PRODUCER_RECEIPTS')
    except Exception as e:state(status='SCORING_ERROR_REQUIRES_REVIEW',error=type(e).__name__+': '+str(e))
    time.sleep(20)
