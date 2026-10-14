"""Read-only final evidence audit; no imports of training/model libraries or writes."""
import hashlib
import json
import math
import re
from pathlib import Path

BASE = Path('/dataset/shaoshitong/recursive-bran-a-h100-20261012/recovery-from100-20261013')
KEYS = ('f1', 'em', 'answer_format', 'format_rate', 'generated_tokens', 'search_calls')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def collect_rows(path, count):
    totals = {}
    indices = set()
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            index = row['final_index']
            assert type(index) is int and 0 <= index < count and index not in indices
            indices.add(index)
            assert row['source'] != 'all'
            for key in KEYS:
                value = row[key]
                assert isinstance(value, (int, float)) and math.isfinite(value)
                assert value >= 0
                if key in ('em', 'answer_format', 'format_rate'):
                    assert value in (0, 1)
                if key == 'f1':
                    assert value <= 1
            for source in (row['source'], 'all'):
                group = totals.setdefault(source, dict(count=0, **{key: 0. for key in KEYS}))
                group['count'] += 1
                for key in KEYS:
                    group[key] += row[key]
    assert indices == set(range(count))
    return totals


def collect_gpu_records(paths):
    records = {}
    for path in paths:
        with Path(path).open(errors='replace') as stream:
            for line in stream:
                if 'FINAL51713_GPU=' not in line:
                    continue
                row = json.JSONDecoder().raw_decode(line.split('FINAL51713_GPU=', 1)[1])[0]
                key = (row['event'], row['rank'], row['pid'], row['call'])
                assert key not in records or records[key] == row, 'Conflicting worker evidence'
                records[key] = row
    return records


def audit(base=BASE):
    folder = base / 'final51713-supervisor-recovery-v1'
    recovery = base / 'final51713-shortpath-recovery-v1'
    if (recovery/'evaluation-launch.json').exists():
        folder = recovery
    tokenizer_recovery = base/'final51713-tokenizer-recovery-v1'
    if (tokenizer_recovery/'evaluation-launch.json').exists():
        folder = tokenizer_recovery
    worker_recovery = base/'final51713-worker-recovery-v1'
    if (worker_recovery/'evaluation-launch.json').exists():
        folder = worker_recovery
    plan_path = folder / 'execution-plan.json'
    exit_path = folder / 'evaluation-actual-exit.json'
    if not plan_path.exists() or not exit_path.exists():
        return dict(status='WAIT_FINAL_EXIT', accepted=False)
    plan, terminal = load(plan_path), load(exit_path)
    launch = load(folder / 'evaluation-launch.json')
    assert terminal['rc'] == 0 and terminal['pid'] == launch['pid']
    assert terminal['run_id'] == launch['run_id'] == plan['run_id']
    assert plan['training_complete_outer'] == 200 and plan['owned_training_released']
    assert sha(plan['config']) == plan['config_sha256']
    assert sha(plan['training_start']) == plan['training_start_sha256']
    assert load(plan['training_start'])['run_id'] == plan['training_run']
    for receipt in plan['admission_receipts']:
        assert sha(receipt['path']) == receipt['sha256']
    assert sha(plan['checkpoint_manifest']) == plan['checkpoint_manifest_sha256']
    manifest = load(plan['checkpoint_manifest'])
    assert manifest['status'] == 'COMPLETE_SAME_WORLD_STATE_SAVED'
    identity = manifest['identity']
    assert identity['completed_step'] == identity['guard_continue_accounting']['completed_outer_updates'] == 200
    assert identity['config_sha256'] == plan['config_sha256']
    assert manifest['driver']['next_step'] == 201
    assert sorted(w['identity']['rank'] for w in manifest['worker_states']) == list(range(8))
    output = Path(plan['output'])
    summary = load(output / 'summary.json')
    assert summary['status'] == 'COMPLETE' and summary['count'] == 51713
    assert summary['workers'] == 8 and summary['batch_size'] == 512
    assert summary['checkpoint_sha'] == plan['checkpoint_manifest_sha256']
    rows_path = output / 'question-metrics.jsonl'
    assert sha(rows_path) == summary['question_metrics_sha']
    totals = collect_rows(rows_path, 51713)
    for source, group in totals.items():
        assert summary['metrics']['val/qa_count/' + source] == group['count']
        for key in KEYS:
            assert math.isclose(summary['metrics']['val/' + key + '/' + source],
                                group[key] / group['count'], rel_tol=1e-10, abs_tol=1e-10)
    runtime_path = folder/'runtime-recovery.json'
    ray_root = Path(load(runtime_path)['ray_tmpdir']) if runtime_path.exists() else Path('/tmp/rba-final-' + plan['run_id'])
    logs = list(ray_root.glob('**/logs/worker-*.out'))
    records = collect_gpu_records(logs + [folder / 'evaluation.log'])
    workers = {}
    for key, row in records.items():
        if row['event'] != 'complete':
            continue
        start = records[('start', row['rank'], row['pid'], row['call'])]
        assert start['checkpoint_sha'] == plan['checkpoint_manifest_sha256']
        assert start['gpu_uuid'] == row['gpu_uuid'] and start['rows'] == row['rows'] > 0
        assert row['generated_tokens'] > 0 and math.isfinite(row['seconds']) and row['seconds'] > 0
        worker = workers.setdefault(row['rank'], dict(pid=row['pid'], gpu_uuid=row['gpu_uuid'], calls=0, rows=0, tokens=0))
        assert worker['pid'] == row['pid'] and worker['gpu_uuid'] == row['gpu_uuid']
        worker['calls'] += 1
        worker['rows'] += row['rows']
        worker['tokens'] += row['generated_tokens']
    if set(workers) != set(range(8)):
        return dict(status='HOLD_GPU_EVIDENCE', accepted=False, ranks=sorted(workers))
    assert len({w['gpu_uuid'] for w in workers.values()}) == 8
    resource = load(folder / 'resource-admission.json')
    assert {w['gpu_uuid'] for w in workers.values()} == {card[1] for card in resource['cards']}
    # The worker hook measures each action RPC, including further search turns
    # and shrinking active tails. There are at least 102 initial batch RPCs;
    # exactly 102 would incorrectly reject ordinary multi-turn evaluation.
    assert all(w['calls'] >= 102 for w in workers.values()), 'All eight workers must cover at least 102 batch starts'
    utilization = {w['gpu_uuid']: [] for w in workers.values()}
    with (folder / 'gpu-device-observer.jsonl').open() as stream:
        for line in stream:
            sample = json.loads(line)
            if sample['rc'] or not launch['time'] <= sample['time'] <= terminal['time']:
                continue
            cards = [row.split(', ') for row in sample['device_sample'].strip().splitlines()]
            assert len(cards) == 8 and {row[1] for row in cards} == set(utilization)
            for row in cards:
                value = float(row[2])
                assert math.isfinite(value) and 0 <= value <= 100
                utilization[row[1]].append(value)
    assert all(utilization.values()), 'Missing actual eight-device utilization samples'
    device_stats = {uuid: dict(samples=len(values), mean=sum(values)/len(values),
                              peak=max(values), nonzero_fraction=sum(v > 0 for v in values)/len(values))
                    for uuid, values in utilization.items()}
    return dict(status='ACCEPTED_ROWS_AND_EIGHT_GPU', accepted=True, count=51713,
                training_run=plan['training_run'], final_run=plan['run_id'],
                checkpoint_sha=plan['checkpoint_manifest_sha256'],
                rows_sha=summary['question_metrics_sha'], totals=totals, workers=workers,
                whole_device_utilization=device_stats,
                elapsed_seconds=summary['elapsed_seconds'],
                scope='Numeric rows, native aggregate, actual wait, checkpoint manifest, eight worker generation records and whole-device utilization including possible cohost tasks; not fresh tensor hashes or utilization attribution')


if __name__ == '__main__':
    print(json.dumps(audit(), sort_keys=True))
