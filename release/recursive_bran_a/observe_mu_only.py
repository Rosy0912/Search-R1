"""Read-only compact progress; no weights, model imports, GPU writes or signals."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument('--base', required=True)
p.add_argument('--owner', required=True, type=int)
p.add_argument('--tick', required=True, type=int)
p.add_argument('--run', required=True)
p.add_argument('--source', required=True)
p.add_argument('--output', default='output-sparse-mu-only-resume4')
p.add_argument('--log', default='logs/mu-only-resume4-20261012.log')
p.add_argument('--seconds', type=int, default=3600)
a = p.parse_args()
b = Path(a.base)
out = b/a.output
deadline = time.monotonic() + a.seconds
previous = None
while time.monotonic() < deadline:
    row = dict(time=time.time(), owner=a.owner, source=a.source, run=a.run)
    try:
        proc = Path('/proc', str(a.owner))
        stat = (proc/'stat').read_text().rsplit(') ', 1)[1].split()
        assert proc.stat().st_uid == 1005 and int(stat[19]) == a.tick
        row['owner_state'] = stat[0]
    except FileNotFoundError:
        row['owner_state'] = 'ABSENT'
    migrated = out/'sparse-schedule-migrated.json'
    if migrated.exists():
        m = json.loads(migrated.read_text())
        row['migration'] = m['status']
        row['restored_calls'] = [r['actual_total_optimizer_calls'] for r in m['workers']]
        row['next_outer'] = m['next_outer']
    events = list((out/'recursive-events').glob('*/*/events.jsonl'))
    row['recorded_questions'] = len(events)
    row['events_bytes'] = sum(f.stat().st_size for f in events)
    if events:
        row['latest_event_mtime'] = max(f.stat().st_mtime for f in events)
    completed = []
    for f in sorted((b/'checkpoints/training_state/attempt-results').glob('attempt-*.json')):
        value = json.loads(f.read_text())
        result = value.get('all_rank_result', {})
        identity = result.get('identity', {})
        if identity.get('run_id') == a.run and identity.get('code_commit') == a.source:
            completed.append(dict(outer=value['accepted_outer'], status=value['status'],
                calls=value['actual_completed_optimizer_calls_per_rank']))
    row['new_completed'] = completed
    safety_files = list((out/'rank-audit/policy-safety').glob('*.json'))
    row['safety_event_count'] = len(safety_files)
    if safety_files:
        newest = max(safety_files, key=lambda f: f.stat().st_mtime)
        safety = json.loads(newest.read_text())
        row['safety'] = {k: safety[k] for k in ('outer_step', 'rank', 'phase', 'committed_optimizer_calls')}
        row['safety']['decision'] = {k: safety['decision'].get(k) for k in
            ('allowed', 'reasons', 'probability_denominator', 'full_actor_beta_available')}
    log = (b/a.log).read_text(errors='replace')
    lines = log.splitlines()
    row['errors'] = [s[-500:] for s in lines if any(k in s for k in ('Traceback', 'ValueError:', 'RuntimeError:'))][-3:]
    row['cache_release_lines'] = [s[s.index('RECURSIVE_WEIGHT_CACHE_PHASE='):] for s in lines if 'RECURSIVE_WEIGHT_CACHE_PHASE=' in s][-8:]
    step_lines = [s for s in lines if re.search(r'\bstep:\d+\b', s)]
    if step_lines:
        row['last_step_metrics'] = step_lines[-1][-2600:]
    row['compute'] = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_gpu_memory',
        '--format=csv,noheader,nounits'], text=True).splitlines()
    row['direct_free'] = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
        '--format=csv,noheader,nounits'], text=True).splitlines()
    signature = json.dumps({k:v for k,v in row.items() if k != 'time'}, sort_keys=True)
    if signature != previous:
        print(json.dumps(row), flush=True)
        previous = signature
    if row['owner_state'] in ('ABSENT', 'Z') or row['errors']:
        break
    time.sleep(30)
