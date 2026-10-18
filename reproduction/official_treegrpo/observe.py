"""Read-only scientific observation, called after an upstream optimizer/iteration event."""
import json
import math
import os
import time
from pathlib import Path

def write(name, record):
    root=Path(os.environ['TREEGRPO_AUDIT_DIR']);root.mkdir(parents=True,exist_ok=True)
    p=root/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(record,allow_nan=False)+'\n');t.replace(p)

def observe_adam(optimizer, grad_norm):
    import torch
    steps=[int(s['step'].item()) for s in optimizer.state.values() if 'step' in s]
    record=dict(at=time.time(),rank=torch.distributed.get_rank(),world_size=torch.distributed.get_world_size(),
        actual_adam_min=min(steps) if steps else None,actual_adam_max=max(steps) if steps else None,
        optimizer_state_count=len(steps),grad_norm=float(grad_norm.item()),
        learning_rates=[g['lr'] for g in optimizer.param_groups],job=os.environ['SLURM_JOB_ID'])
    write('adam-rank-%02d.json'%record['rank'],record)

def observe_iteration(step, metrics):
    clean={k:float(v) for k,v in metrics.items() if isinstance(v,(int,float)) and math.isfinite(v)}
    record=dict(at=time.time(),completed_upstream_iteration=int(step),metrics=clean,job=os.environ['SLURM_JOB_ID'])
    write('progress.json',record)
    with (Path(os.environ['TREEGRPO_AUDIT_DIR'])/'iterations.jsonl').open('a') as f:
        f.write(json.dumps(record,allow_nan=False)+'\n')
