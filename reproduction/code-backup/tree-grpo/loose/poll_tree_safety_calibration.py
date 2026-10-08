"""Read-only, bounded metadata view of one own h100-9 calibration run."""
import json
import subprocess
import sys
import time
from pathlib import Path

job = sys.argv[1]
assert job.isdigit()
root = Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
run = root / 'runs' / ('tree-safety300-' + job)

def tail(path, limit=3000):
    if not path.exists():
        return None
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - limit))
        return stream.read().decode('utf-8', errors='replace')

def read_json(path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_bytes())
    except Exception as error:
        return {'unreadable': type(error).__name__, 'path': str(path)}

commands = {}
for args in (['scontrol', 'show', 'job', job],
             ['sacct', '-j', job, '--format=JobID,State,ExitCode,Elapsed,AllocCPUS,AllocTRES', '-P']):
    result = subprocess.run(args, capture_output=True, text=True, timeout=15)
    commands[' '.join(args)] = {'rc': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}

files = [str(path.relative_to(run)) for path in run.iterdir() if path.is_file()]
files += [str(path.relative_to(run)) for name in ('output', 'gpu-kl-proof', 'native-fullstate')
          for path in (run / name).rglob('*')
          if path.is_file() and path.suffix in ('.json', '.jsonl', '.log', '.ready')]
logs = {relative: tail(run / relative) for relative in ('stage.log', 'native-fullstate.log',
        'gpu-kl.log', 'retriever.log', 'guard.log')
        if (run / relative).exists()}
for name in ('tree-safety110-calibration-' + job + '.out',
             'tree-safety110-calibration-' + job + '.err'):
    logs[name] = tail(root / 'logs' / name)
guard_paths = [run / 'resource-samples.jsonl'] if (run / 'resource-samples.jsonl').exists() else []
guards = [{'path': str(path), 'tail': tail(path, 5000),
           'age_seconds': time.time() - path.stat().st_mtime} for path in guard_paths]
ready = read_json(run / 'output/probability-calibration/ready.json')
raw = sorted((run / 'output/sampling-consistency').glob('rank-*-call-*.json'))
calls = {}
for path in raw:
    record = read_json(path)
    if record is not None and 'generation_call' in record:
        call = str(record['generation_call'])
        calls.setdefault(call, {'ranks': [], 'saved_rows': 0, 'decision': record['decision']})
        calls[call]['ranks'].append(record['rank'])
        calls[call]['saved_rows'] += record['saved_rows']
reports = [read_json(path) for path in sorted((run / 'output/validation-cost').glob('*.json'))]
print(json.dumps({'job_id': int(job), 'epoch': time.time(), 'run': str(run),
      'commands': commands, 'metadata_files': files, 'logs': logs, 'guards': guards,
      'run_result': read_json(run / 'run-result.json'), 'calibration_ready': ready,
      'raw_sampling_calls': calls, 'validation_reports': reports,
      'scope': 'Read-only metadata; no optimizer, GPU work, asset transfers or other jobs'}, allow_nan=False))
