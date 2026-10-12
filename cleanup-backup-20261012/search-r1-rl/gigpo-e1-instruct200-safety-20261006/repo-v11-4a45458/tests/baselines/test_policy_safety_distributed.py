"""Real two-rank Gloo reductions; GPU/FSDP acceptance is a separate requirement."""
import json
from pathlib import Path
import tempfile
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from search_r1.baselines.policy_safety import local_statistics, global_statistics, rejection_reason


def worker(rank, rendezvous, output):
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2)
    reports=[]
    for case in ('empty_rank','invalid_rank'):
        delta=torch.tensor([[.1,2.]],dtype=torch.float64)
        mask=torch.ones_like(delta)
        if rank==0:
            if case=='empty_rank':mask.zero_()
            else:delta[0,0]=float('nan')
        part=local_statistics(torch.zeros_like(delta),delta,-torch.ones_like(delta),mask,mask/2,
                              torch.tensor([[0,1]]),3.,30.)
        stats=global_statistics([part])
        reports.append(dict(case=case,statistics=stats,reason=rejection_reason(stats,
                             dict(target_old_policy_kl=100.,negative_tail_limit=.75))))
    (Path(output)/f'rank{rank}.json').write_text(json.dumps(reports,allow_nan=False))
    dist.destroy_process_group()


def test_empty_and_invalid_rank_global_decisions_and_exact_quantiles():
    with tempfile.TemporaryDirectory() as root:
        mp.spawn(worker,args=(str(Path(root)/'store'),root),nprocs=2,join=True)
        a=json.loads((Path(root)/'rank0.json').read_text())
        b=json.loads((Path(root)/'rank1.json').read_text())
        assert a==b
        assert a[0]['reason'] is None
        stats=a[0]['statistics']
        assert stats['negative_action_count']==2 and stats['negative_tail_fraction_count']==.5
        assert abs(stats['negative_action_logratio_p95']-1.905)<1e-12
        assert abs(stats['negative_action_logratio_p99']-1.981)<1e-12
        assert a[1]['reason']=='NONFINITE_OR_EXP_RANGE_OR_EMPTY_EFFECTIVE_MONITOR'
        assert a[1]['statistics']['sampled_old_policy_kl'] is None
