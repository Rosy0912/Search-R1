"""Post-event observation; no optimizer, advantage, or rollout changes."""
import json
import math
import os
from pathlib import Path
import time


def write(name, record):
    root = Path(os.environ['BRANPO_AUDIT_DIR'])
    root.mkdir(parents=True, exist_ok=True)
    target = root / name
    temp = target.with_suffix('.tmp')
    temp.write_text(json.dumps(record, allow_nan=False) + '\n')
    temp.replace(target)


def observe_adam(optimizer, grad_norm):
    import torch
    distributed = torch.distributed.is_initialized()
    steps = [int(x['step'].item()) for x in optimizer.state.values() if 'step' in x]
    record = dict(at=time.time(), rank=torch.distributed.get_rank() if distributed else 0,
                  world_size=torch.distributed.get_world_size() if distributed else 1,
                  actual_adam_min=min(steps) if steps else None,
                  actual_adam_max=max(steps) if steps else None,
                  optimizer_state_count=len(steps), grad_norm=float(grad_norm),
                  learning_rates=[g['lr'] for g in optimizer.param_groups],
                  job=os.environ.get('SLURM_JOB_ID', 'CPU_EQUIVALENCE_TEST'))
    write('adam-rank-%02d.json' % record['rank'], record)


def observe_iteration(step, metrics):
    clean = {key: float(value) for key, value in metrics.items()
             if isinstance(value, (int, float)) and math.isfinite(value)}
    if not any('pg_loss' in key for key in clean):
        raise ValueError('An iteration observation requires completed policy-update metrics')
    label = int(step)
    assert 101 <= label <= 300, 'Unexpected official start/total-step semantics'
    record = dict(at=time.time(), completed_upstream_label=label,
                  completed_actual_iterations=label - 100, metrics=clean,
                  job=os.environ['SLURM_JOB_ID'])
    write('progress.json', record)
    with (Path(os.environ['BRANPO_AUDIT_DIR']) / 'iterations.jsonl').open('a') as f:
        f.write(json.dumps(record, allow_nan=False) + '\n')
