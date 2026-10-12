"""One-shot CPU observation of an explicit B recovery, without actor callbacks."""
import argparse
from collections import Counter
import csv
import datetime
import io
import json
import os
from pathlib import Path
import re
import socket
import subprocess

def observe(run, source, log, pane):
    assert os.getuid() == 1001 and socket.gethostname().split('.')[0] == 'xie-lab-4'
    assert re.fullmatch('[0-9a-f]{32}', run) and re.fullmatch('[0-9a-f]{40}', source)
    root = Path('/home/shaoshitong/search-e1-migration-20261011/weka_3/shaoshitong/recursive-bran-b-4gpu-hkust5-20261011')
    log = Path(log)
    log.resolve().relative_to(root / 'logs')
    output = root / ('output-speed-resume3-' + run)
    result = dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(), run=run,
                  source=source, log_bytes=log.stat().st_size, events={}, latest={}, metrics={}, errors=[])
    if (output / 'start.json').exists():
        start = json.loads((output / 'start.json').read_text())
        assert start['run_id'] == run and start['source_commit'] == source
        proc = Path('/proc') / str(start['pid'])
        result['controller_pid'] = start['pid']
        result['controller_present'] = proc.exists()
        if proc.exists():
            stat = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
            result['controller_state'], result['controller_starttick'] = stat[0], int(stat[19])
    result['pane'] = subprocess.check_output(['tmux', 'display-message', '-p', '-t', pane,
                          '#{pane_id}|#{pane_pid}|#{pane_dead}|#{pane_dead_status}'], text=True).strip()
    events = Counter()
    with log.open(errors='replace') as stream:
        for line in stream:
            for tag in ('TREE_TIMING', 'TREE_ACTION_TIMING', 'TREE_DRIVER_STAGE', 'RECURSIVE_WEIGHT_CACHE_PHASE', 'RETRIEVER_PHASE'):
                marker = tag + '='
                if marker not in line:
                    continue
                try:
                    row = json.JSONDecoder().raw_decode(line.split(marker, 1)[1])[0]
                except (ValueError, TypeError):
                    continue  # An in-flight log write is not a completed event.
                key = tag + ':' + str(row.get('event', row.get('stage', 'phase')))
                events[key] += 1
                result['latest'][tag] = {k: v for k, v in row.items() if k in
                    ('event', 'stage', 'outer', 'unix_time', 'wall_seconds', 'cpu_seconds', 'successful',
                     'active_requests', 'rows', 'policy_version', 'generation_calls', 'weight_sync_calls',
                     'sync_wall_seconds', 'kv_init_calls', 'kv_init_wall_seconds', 'released_before_hf_or_policy_change', 'cost_delta')}
            match = re.search(r'step:(\d+) -', line)
            if match:
                result['metrics'][match.group(1)] = {k: float(v) for k, v in re.findall(
                    r'([a-zA-Z0-9_/-]+):(-?[0-9.eE+]+)', line) if
                    k.startswith(('timing_s/', 'actor/', 'tree/', 'memory/'))}
            if any(s in line for s in ('Traceback (most recent call last)', 'NameError:', 'RayTaskError(', 'CUDA out of memory', 'RuntimeError:', 'PolicySafetyRefusal:')):
                result['errors'].append(line.strip()[:600])
    result['events'] = dict(events)
    result['errors'] = result['errors'][-12:]
    restore = output / 'sparse-schedule-migrated.json'
    if restore.exists():
        p = json.loads(restore.read_text())
        result['restore'] = dict(driver_equal=p['driver_actual_state_equal'], next_outer=p['next_outer'],
            workers=[{k: w[k] for k in ('rank', 'actual_cold_native_state_equal', 'actual_total_optimizer_calls', 'scheduler_step', 'adam_steps')} for w in p['workers']])
    try:
        p = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
                           text=True, capture_output=True, timeout=20)
        result['gpu_samples'] = [row for row in csv.reader(io.StringIO(p.stdout)) if row and row[0].strip() in ('0', '1', '2', '3')]
        result['gpu_sample_rc'] = p.returncode
    except subprocess.TimeoutExpired:
        result['gpu_sample_timeout'] = True
    return result

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True); p.add_argument('--source', required=True)
    p.add_argument('--log', required=True); p.add_argument('--pane', required=True)
    a = p.parse_args()
    print(json.dumps(observe(a.run, a.source, a.log, a.pane), sort_keys=True))
