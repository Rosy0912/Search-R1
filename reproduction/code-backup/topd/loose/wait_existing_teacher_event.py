"""Wait on the existing local read-only observer; never submit or restart training."""
import argparse
import json
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--known-sha', required=True)
a = p.parse_args()
log = Path('outputs/remote-evidence/r1-standard32b-direct100-3218-readonly.jsonl')
last = None
for _ in range(180):
    with log.open('rb') as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 2_000_000))
        lines = f.read().splitlines()
    try:
        s = json.loads(lines[-1])
    except (json.JSONDecodeError, IndexError):
        time.sleep(30)
        continue
    queue = s.get('queue', {}).get('stdout', '')
    event = None
    if time.time() - s['epoch'] > 180:
        event = 'STALE_OBSERVATION_NOT_JOB_FAILURE'
    elif s.get('errors') or any(
        'first-fault' in x.get('relative_path', '') or
        'train-result' in x.get('relative_path', '') for x in s['files']
    ):
        event = 'CHECK_ACTUAL_FAULT_OR_TERMINAL'
    elif '|RUNNING|' not in queue:
        event = 'CONTROLLER_CHANGED_REPOLL_SAME_JOB'
    outer = next((x for x in s['files'] if x.get('relative_path') == '../actual-outer.json'), {})
    if not event and outer.get('sha256') and outer['sha256'] != a.known_sha:
        event = 'ACTUAL_OUTER_CHANGED'
    last = {'epoch': s['epoch'], 'queue': queue, 'errors': s['errors'],
            'sha256': outer.get('sha256'),
            'completed': outer.get('data', {}).get('actual_completed_outer_steps')}
    if event:
        print(json.dumps({'event': event, **last}), flush=True)
        break
    time.sleep(30)
else:
    print(json.dumps({'event': 'OBSERVATION_WINDOW_EXPIRED_NOT_JOB_FAILURE', 'last': last}), flush=True)
