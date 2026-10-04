"""Read-only post-adjustment GPU sampling; retain original historical peaks."""
import argparse
import json
import re
import subprocess
import time
from pathlib import Path

base = Path('/weka_3/shaoshitong/search-e1-config-study-20261004')
run = 'f04fe593830c4831918ede872812c016'
parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['score1', 'jit'], default='score1')
parser.add_argument('--steps', type=int, nargs=2, default=[39, 40])
args = parser.parse_args()
if args.mode == 'jit':
    jit_path = base / 'receipts/archive-live-jit-optimizer.json'
    while not jit_path.exists():
        receipt = json.loads((base / 'receipts/archive-round-1-grpo.json').read_text())
        assert receipt['run_id'] == run and not receipt['status'].startswith('FAILED')
        time.sleep(5)
    jit = json.loads(jit_path.read_text())
    assert jit['run_id'] == run and all(r['jit_enabled'] for r in jit['ranks'])
    assert max(r['scheduler']['last_epoch'] for r in jit['ranks']) < min(args.steps)
proof = json.loads((base / 'receipts/archive-live-physical-score1.json').read_text())
assert proof['run_id'] == run and all(r['physical_score'] == 1 for r in proof['ranks'])
record = dict(run_id=run, started=time.time(), physical_score=1, jit=args.mode == 'jit', peaks_mib={},
              samples=0, full_iterations_to_observe=args.steps, status='OBSERVING')
target = base / ('receipts/archive-post-' + args.mode + '-memory.json')
assert not target.exists(), 'Preserve existing observation; do not overwrite it'
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
        if set(args.steps) <= steps:
            record['status'] = 'OBSERVATION_COMPLETE'
        temp = target.with_suffix('.tmp')
        temp.write_text(json.dumps(record, indent=2)); temp.replace(target)
        last = time.time()
        if record['status'] == 'OBSERVATION_COMPLETE': break
    time.sleep(2)
record['finished'] = time.time()
target.write_text(json.dumps(record, indent=2))
print(json.dumps(record), flush=True)
