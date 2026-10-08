import collections
import datetime
import json
import pathlib
import subprocess
import sys

base = pathlib.Path(__file__).resolve().parent
number = int(sys.argv[1])
previous_path = base / ('runtime-watch-3322-observation-%02d.json' % (number - 1))
prior_observation = json.loads(previous_path.read_text())
snapshot_path = base / 'runtime-watch-3322-current-process-metadata.json'
if snapshot_path.exists():
    previous = json.loads(snapshot_path.read_text())
else:
    previous = {'observed_at': prior_observation['observed_at'], **prior_observation.get('own_job_processes', {})}
response = subprocess.run([sys.executable, 'work/remote.py', '--stdin', str(base / 'runtime-watch-3322-safe-meta.py'), 'python3 -'], text=True, capture_output=True, timeout=50)
if response.returncode != 0:
    print(json.dumps({'remote_exit_code': response.returncode, 'remote_stderr': response.stderr[:200]}))
    sys.exit(response.returncode)
observation = json.loads(response.stdout)
current = observation.pop('own_job_processes')
snapshot_path.write_text(json.dumps({'observed_at': observation['observed_at'], **current}, sort_keys=True))
old = {(x['pid'], x['starttime_jiffies']): x for x in previous.get('processes', [])}
by_comm = collections.defaultdict(lambda: {'count': 0, 'utime_delta_jiffies': 0, 'stime_delta_jiffies': 0})
states = collections.Counter()
for x in current['processes']:
    states[x['state']] += 1
    y = old.get((x['pid'], x['starttime_jiffies']))
    if y is not None:
        group = by_comm[x['comm']]
        group['count'] += 1
        group['utime_delta_jiffies'] += x['utime_jiffies'] - y['utime_jiffies']
        group['stime_delta_jiffies'] += x['stime_jiffies'] - y['stime_jiffies']
observation['own_cpu_delta'] = {
    'prior_observed_at': previous.get('observed_at'),
    'own_count': current['count'],
    'same_pid_starttime_count': sum(g['count'] for g in by_comm.values()),
    'clock_ticks_per_second': current['clock_ticks_per_second'],
    'utime_delta_jiffies': sum(g['utime_delta_jiffies'] for g in by_comm.values()),
    'stime_delta_jiffies': sum(g['stime_delta_jiffies'] for g in by_comm.values()),
    'states': dict(states),
    'by_comm': dict(by_comm),
}
destination = base / ('runtime-watch-3322-observation-%02d.json' % number)
destination.write_text(json.dumps(observation, sort_keys=True, indent=2))
print(json.dumps({
    'observed_at': observation['observed_at'],
    'slurm': {k: v for k, v in observation['slurm_identity']['fields'].items() if k in {'JobState', 'ExitCode', 'RunTime', 'Reason'}},
    'published_counts': {k: v.get('count') for k, v in observation['files'].items() if k not in {'output/fresh-initialization.json', 'resource-samples.jsonl', 'output/resource-samples.jsonl'}},
    'new_publications': {k: v for k, v in observation['files'].items() if v.get('count', 0) > 0 and k != 'output/fresh-initialization.json' and v != prior_observation['files'].get(k)},
    'resource_latest': observation['files'].get('resource-samples.jsonl', {}).get('recent', [])[-1:],
    'own_cpu_delta': {k: v for k, v in observation['own_cpu_delta'].items() if k != 'by_comm'},
    'changed_anchored_logs': {k: v for k, v in observation.get('anchored_log_metadata', {}).items() if v != prior_observation.get('anchored_log_metadata', {}).get(k)},
    'py_spy_installed_path': observation.get('py_spy_installed_path'),
    'own_measured_task_wait': observation.get('own_measured_task_wait'),
}, sort_keys=True))
