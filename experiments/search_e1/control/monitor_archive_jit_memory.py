"""Continuous read-only post-JIT sampling; preserve pre-JIT peak evidence."""
import json
import subprocess
import time
from pathlib import Path

base = Path('/weka_3/shaoshitong/search-e1-config-study-20261004')
proof = json.loads((base / 'receipts/archive-live-jit-optimizer.json').read_text())
assert len(proof['ranks']) == 4 and all(r['jit_enabled'] for r in proof['ranks'])
initial = json.loads((base / 'receipts/archive-post-jit-memory.json').read_text())
target = base / 'receipts/archive-jit-memory-continuous.json'
assert not target.exists(), 'Preserve existing observer output'
record = dict(run_id=proof['run_id'], started=initial['started'],
              continuous_observer_started=time.time(), initial_source='archive-post-jit-memory.json',
              initial_source_samples=initial['samples'], peaks_mib=dict(initial['peaks_mib']), samples=0)
last = 0
while True:
    raw = subprocess.check_output(['nvidia-smi', '-i', '4,5,6,7',
                                   '--query-gpu=index,memory.used,memory.total',
                                   '--format=csv,noheader,nounits'], text=True, timeout=15)
    totals = {}
    for line in raw.splitlines():
        idx, used, total = [v.strip() for v in line.split(',')]
        record['peaks_mib'][idx] = max(record['peaks_mib'].get(idx, 0), int(used))
        totals[idx] = int(total)
    record['minimum_reserve_mib'] = {i: totals[i] - v for i, v in record['peaks_mib'].items()}
    record['shared_gpu_margin_met'] = record['minimum_reserve_mib']['7'] >= 6144
    record['samples'] += 1
    if time.time() - last >= 30:
        record['updated'] = time.time()
        record['pipeline_phase'] = (base / 'receipts/pipeline-stage.txt').read_text()
        source = json.loads((base / 'receipts/archive-round-1-grpo.json').read_text())
        record['source_run_status'] = source['status']
        record['source_run_matches'] = source['run_id'] == proof['run_id']
        temp = target.with_suffix('.tmp'); temp.write_text(json.dumps(record, indent=2)); temp.replace(target)
        last = time.time()
    time.sleep(2)
