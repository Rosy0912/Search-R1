"""Read-only post-adjustment GPU sampling; retain original historical peaks."""
import json
import re
import subprocess
import time
from pathlib import Path

base = Path('/weka_3/shaoshitong/search-e1-config-study-20261004')
run = 'f04fe593830c4831918ede872812c016'
proof = json.loads((base / 'receipts/archive-live-physical-score1.json').read_text())
assert proof['run_id'] == run and all(r['physical_score'] == 1 for r in proof['ranks'])
record = dict(run_id=run, started=time.time(), physical_score=1, peaks_mib={},
              samples=0, full_iterations_to_observe=[39, 40], status='OBSERVING')
target = base / 'receipts/archive-post-score1-memory.json'
last = 0
while True:
    receipt = json.loads((base / 'receipts/archive-round-1-grpo.json').read_text())
    assert receipt['run_id'] == run, 'Run identity changed'
    if receipt['status'].startswith('FAILED'):
        record['status'] = 'TRAINING_FAILED'; break
    raw = subprocess.check_output(['nvidia-smi', '-i', '4,5,6,7',
                                   '--query-gpu=index,memory.used,memory.total',
                                   '--format=csv,noheader,nounits'], text=True, timeout=15)
    totals = {}
    for line in raw.splitlines():
        idx, used, total = [x.strip() for x in line.split(',')]
        record['peaks_mib'][idx] = max(record['peaks_mib'].get(idx, 0), int(used))
        totals[idx] = int(total)
    record['minimum_reserve_mib'] = {i: totals[i] - v for i, v in record['peaks_mib'].items()}
    record['samples'] += 1
    if time.time() - last >= 30:
        text = (base / 'runs/archive/round-1-grpo/train.log').read_text(errors='replace')
        steps = {int(s) for s in re.findall(r'\bstep:(\d+) -', text)}
        record['last_completed_step'] = max(steps, default=0)
        record['updated'] = time.time()
        if {39, 40} <= steps:
            record['status'] = 'OBSERVATION_COMPLETE'
        temp = target.with_suffix('.tmp')
        temp.write_text(json.dumps(record, indent=2)); temp.replace(target)
        last = time.time()
        if record['status'] == 'OBSERVATION_COMPLETE': break
    time.sleep(2)
record['finished'] = time.time()
target.write_text(json.dumps(record, indent=2))
print(json.dumps(record), flush=True)
