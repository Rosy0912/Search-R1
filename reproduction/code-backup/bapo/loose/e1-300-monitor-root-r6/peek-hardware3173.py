import json
from pathlib import Path

run = Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-monitor-r6-20261004/runs/bapo-instruct300-hardware-3173')
out = {'job_id': 3173, 'run': str(run)}
for name in ('model-gpu.json', 'retrieval-gpu.json', 'retrieval-cpu.json', 'retrieval-comparison.json', 'HARDWARE_MEASUREMENTS_RETURNED_SUCCESSFULLY', 'own-process-cleanup.json'):
    p = run / name
    out[name] = {'exists': p.is_file(), 'bytes': p.stat().st_size if p.is_file() else None}
watch = run / 'resource-watch.jsonl'
if watch.is_file():
    rows = [json.loads(x) for x in watch.read_text().splitlines() if x]
    out['resource_samples'] = len(rows)
    out['resource_pauses'] = sum(bool(x.get('pause')) for x in rows)
    out['last_resource_time'] = rows[-1].get('time') if rows else None
    out['discarded_lifecycle_sample_count'] = sum(len(x.get('discarded_lifecycle_samples', [])) for x in rows)
    if rows and rows[-1].get('pause'):
        out['last_resource_failure'] = {k: rows[-1].get(k) for k in ('reason', 'error')}
for name in ('parity-gpu.log', 'parity-gpu-time.log', 'parity-cpu.log', 'parity-cpu-time.log'):
    p = run / name
    if p.is_file():
        out[name] = p.read_text(errors='replace').splitlines()[-6:]
print(json.dumps(out, ensure_ascii=False, indent=2))
