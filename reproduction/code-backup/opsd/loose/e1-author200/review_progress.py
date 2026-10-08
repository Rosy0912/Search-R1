"""Local read-only join of actual captured author200 progress; no remote control."""
import argparse
import hashlib
import json
from pathlib import Path
import types

HERE = Path(__file__).resolve().parent
SOURCE = '188f4e5e7b63cf479290d0888a4a075499886750'
CONFIG = '73f6401e818dc93a11c705da2357d0f3d91b2a51609a51e78a599fb7cf416c60'
WORKERS = [1533910, 1534229, 1534230, 1534231, 1534232, 1534233, 1534234, 1534235]
RECEIPT = '812f0baf1d70ee87e847865f62ff03151e1de0c4a4c0eab2e9ae72e0dbb3793b'
GATE = HERE.parent / 'opsd-e1-300/search_r1/baselines/checkpoint_gate.py'
GATE_SHA = 'c738911559a56780555de219306075b810074a7b0287fc4b271edece6ae73cbb'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def review(snapshot):
    raw_gate = GATE.read_bytes()
    need(hashlib.sha256(raw_gate).hexdigest() == GATE_SHA, 'Fixed source gate changed')
    gate = types.ModuleType('actual_progress_source_gate')
    gate.__file__ = str(GATE)
    exec(compile(raw_gate, str(GATE), 'exec'), gate.__dict__)
    summary = gate.load_json(snapshot / 'observation-summary.json')
    commands = summary['commands']
    need(summary['job_id'] == '3261' and summary['node'] == 'h100-10', 'Wrong snapshot')
    need(commands['source_head']['returncode'] == 0
        and commands['source_head']['stdout'].strip() == SOURCE
        and commands['source_porcelain']['returncode'] == 0
        and not commands['source_porcelain']['stdout'].strip(), 'Source drift')
    fields = dict(x.split('=', 1) for x in commands['scontrol']['stdout'].split() if '=' in x)
    need(tuple(fields.get(k) for k in ['JobId', 'JobState', 'Restarts', 'NodeList', 'TresPerNode'])
        == ('3261', 'RUNNING', '0', 'h100-10', 'gres:gpu:h100:8'), 'Not the same running job')
    for pid in WORKERS + [1533312, 1531763]:
        p = summary['processes'][str(pid)]
        need(p['in_observed_job'] and p['uids'] == [0] * 4
            and p['state'] in ('R', 'S', 'D', 'I'), 'Process identity changed')
    need(summary['processes']['1533312']['start_ticks'] == '1829085316'
        and summary['processes']['1531763']['start_ticks'] == '1829074494', 'Process restart')
    apps = summary['unfiltered_application_csv_rows']
    need(len(apps) == 16 and set(int(a[1]) for a in apps) == set(WORKERS + [1531763]),
        'Unmatched GPU application')
    def parse(line, marker):
        return gate.parse_log_marker(line, marker, driver_pid=1533312, job_id='3261',
            worker_pids=WORKERS, checkpoint_warning_path=
            '/mnt/data/shaoshitong/search-r1-rl/opsd/env/lib/python3.10/site-packages/torch/utils/checkpoint.py')
    logs = {}
    for label, name in [('driver', 'ray-stdout-pid-1533312.out'), ('aggregate', 'train.log')]:
        raw = (snapshot / 'original-files' / name).read_bytes()
        rows, releases = [], []
        for line_bytes in raw.splitlines(keepends=True):
            if not line_bytes.endswith(b'\n'):
                continue
            line = line_bytes.decode()
            if 'OPSD_E1_OUTER_JSON=' in line:
                row = parse(line, 'OPSD_E1_OUTER_JSON=')
                step = row['global_step']
                need((row['actual_outer_updates'], row['source_commit'], row['config_sha256'])
                    == (step, SOURCE, CONFIG), 'Outer identity/config drift')
                gate.validate_outer(row['metrics'], profile='author200')
                for rank in range(8):
                    need(row['metrics'][f'teacher/cuda_update_entries_cumulative_per_rank/rank_{rank}']
                        == step, 'Teacher update clock differs')
                rows.append(row)
            if 'OPSD_E1_STEP5_RELEASED_JSON=' in line:
                releases.append(parse(line, 'OPSD_E1_STEP5_RELEASED_JSON='))
        need([r['global_step'] for r in rows] == list(range(1, len(rows) + 1))
            and 5 <= len(rows) <= 200, 'Missing/duplicate/excess outer')
        need(releases == [{'actual_outer_updates': 5, 'driver_pid': 1533312,
            'next_global_step': 6, 'receipt_sha256': RECEIPT}], 'Actual release differs')
        logs[label] = (rows, hashlib.sha256(raw).hexdigest())
    rows, driver_sha = logs['driver']
    aggregate, aggregate_sha = logs['aggregate']
    need(len(aggregate) <= len(rows) and aggregate == rows[:len(aggregate)], 'Aggregate is not driver prefix')
    keys = ['critic/rewards/mean', 'actor/opsd_loss', 'actor/lr', 'actor/grad_norm',
        'env/finish_ratio', 'env/ratio_of_valid_action', 'timing_s/step']
    return {'status': 'PASS_CURRENT_LOGGED_OUTERS_NOT_FINAL200', 'job_id': '3261',
        'source_commit': SOURCE, 'config_sha256': CONFIG, 'driver_pid': 1533312,
        'driver_start_ticks': '1829085316', 'snapshot': str(snapshot),
        'snapshot_finished_utc': summary['finished_utc'], 'actual_complete_outer': len(rows),
        'aggregate_complete_outer': len(aggregate), 'same_process_adam_calls_per_rank': 16 * len(rows),
        'latest_metrics': {k: rows[-1]['metrics'][k] for k in keys},
        'reward_history': [r['metrics']['critic/rewards/mean'] for r in rows],
        'observed_gpu_used_gib': [float(a[3]) / 1024 for a in summary['gpu_inventory_csv_rows']],
        'slurm_fields': {k: fields[k] for k in ['JobState', 'Restarts', 'RunTime', 'NodeList', 'TresPerNode']},
        'driver_original_sha256': driver_sha, 'aggregate_original_sha256': aggregate_sha,
        'scope': 'Captured complete original log rows and observed process/resource identity only; '
            'not later checkpoint, final evaluation, whole terminal or effect acceptance'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = review(args.snapshot.resolve(strict=True))
    with args.output.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({k: result[k] for k in ['actual_complete_outer', 'aggregate_complete_outer',
        'latest_metrics', 'observed_gpu_used_gib']}, ensure_ascii=False))
