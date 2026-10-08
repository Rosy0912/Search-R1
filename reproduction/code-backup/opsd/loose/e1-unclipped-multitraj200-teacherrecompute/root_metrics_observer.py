"""Pinned root observations only; never grant first5 acceptance or alter a run."""
import argparse
import csv
import hashlib
import json
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = 'fcecbea5d939e41dcfb3b118a26ef049a9bda3d6'
GATE = HERE.parent / 'opsd-unclipped-multitraj200-teacherrecompute/search_r1/baselines/checkpoint_gate.py'
GATE_SHA = '09561719673a7c273ad39b3c9363b1a250eab0a94c20c80f40dd1dfc05589e16'


def observe(snapshot):
    code = GATE.read_bytes()
    assert hashlib.sha256(code).hexdigest() == GATE_SHA
    gate = types.ModuleType('root_pinned_3304_observer')
    gate.__file__ = str(GATE)
    exec(compile(code, str(GATE), 'exec'), gate.__dict__)
    raw = (snapshot / 'train.log').read_bytes()
    transport = json.loads((snapshot / 'transport.stdout.original').read_bytes())
    assert transport['job_id'] == '3304'
    app_rows = list(csv.reader(transport['live']['gpu_apps']['stdout'].splitlines()))
    workers = {int(v[1]) for v in app_rows} - {2543106}
    assert len(workers) == 8
    rows = []
    partial = []
    for line in raw.splitlines(keepends=True):
        if b'OPSD_E1_OUTER_JSON=' not in line:
            continue
        try:
            value = gate.parse_log_marker(line.decode(), 'OPSD_E1_OUTER_JSON=',
                driver_pid=2548319, job_id='3304', worker_pids=workers,
                checkpoint_warning_path='/mnt/data/shaoshitong/search-r1-rl/opsd/env/lib/python3.10/site-packages/torch/utils/checkpoint.py')
        except json.JSONDecodeError:
            assert not line.endswith(b'\n') and raw.endswith(line)
            partial.append({'bytes': len(line), 'sha256': hashlib.sha256(line).hexdigest()})
            continue
        assert value['source_commit'] == SOURCE and value['job_id'] == '3304'
        assert value['driver_pid'] == 2548319
        gate.validate_outer(value['metrics'], profile='multitraj200')
        m = value['metrics']
        rows.append(dict(outer=value['actual_outer_updates'],
            rank0_loss=m['actor/opsd_loss'],
            rank_mean_loss=sum(m[f'actor/opsd_loss_mean/rank_{r}'] for r in range(8))/8,
            reward=m['critic/score/mean'], finish_ratio=m['env/finish_ratio'],
            optimizer_per_rank=[m[f'actor/optimizer_steps/rank_{r}'] for r in range(8)],
            outer_seconds=m['timing_s/step'], generation_seconds=m['timing_s/gen'],
            update_seconds=m['timing_s/update_actor'],
            used_learning_rate=m['actor/optimizer_lr_used'], post_outer_lr=m['actor/lr'],
            has_LF=line.endswith(b'\n')))
    assert [v['outer'] for v in rows] == list(range(1, len(rows)+1))
    return dict(source_commit=SOURCE, job_id=3304, original_log=str(snapshot/'train.log'),
        sha256=hashlib.sha256(raw).hexdigest(), current_pinned_gate_validate_outer='PASS',
        actual_complete_JSON_outer_observations=len(rows), rows=rows, incomplete_EOF=partial,
        scope='Root actual metrics only; independent first5 full state/HF/archive acceptance separate')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--snapshot', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    assert args.snapshot.is_absolute() and args.snapshot.resolve(strict=True) == args.snapshot
    result = observe(args.snapshot)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(result, allow_nan=False))
