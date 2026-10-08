import json
from pathlib import Path

run = Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-monitor-r6-20261004/runs/bapo-instruct300-formal300-3177')
out = {'job_id': 3177, 'run': str(run), 'actual_completed_outer': 0}
for name in ('source-commit.txt', 'resolved-config.yaml', 'retriever-health.json', 'train.log', 'phase.log'):
    p = run / name
    out[name] = {'exists': p.is_file(), 'bytes': p.stat().st_size if p.is_file() else None}
train = run / 'train.log'
if train.is_file():
    lines = train.read_text(errors='replace').splitlines()
    events = [json.loads(x.split('BAPO_TRAIN_METRICS_JSON=', 1)[1]) for x in lines if 'BAPO_TRAIN_METRICS_JSON=' in x]
    out['actual_completed_outer'] = max((e['step'] for e in events), default=0)
    out['latest_outer_event'] = events[-1] if events else None
    out['step5_paused_marker'] = any('E1_STEP5_PAUSED_JSON=' in x for x in lines)
    out['train_tail'] = [x[:1800] for x in lines[-6:]]
watch = run / 'resource-watch.jsonl'
if watch.is_file():
    rows = [json.loads(x) for x in watch.read_text().splitlines() if x]
    out['resource_samples'] = len(rows)
    out['resource_pauses'] = sum(bool(x.get('pause')) for x in rows)
    out['latest_resource_time'] = rows[-1].get('time') if rows else None
    if rows and rows[-1].get('pause'):
        out['last_resource_failure'] = {k: rows[-1].get(k) for k in ('reason', 'error')}
for name in ('retriever.log', 'phase.log'):
    p = run / name
    if p.is_file():
        out[name + '_tail'] = [x[:1800] for x in p.read_text(errors='replace').splitlines()[-4:]]
print(json.dumps(out, ensure_ascii=False, indent=2))
