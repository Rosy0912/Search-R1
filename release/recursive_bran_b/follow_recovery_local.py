"""Bounded read-only followup; small summaries only, never restart training."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
p.add_argument('--observer', required=True)
p.add_argument('--run', required=True); p.add_argument('--source', required=True)
p.add_argument('--log', required=True); p.add_argument('--pane', required=True)
p.add_argument('--until-outer', type=int, default=10)
p.add_argument('--hours', type=float, default=12)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
command = shlex.join(['python3', a.observer, '--run', a.run, '--source', a.source,
                     '--log', a.log, '--pane', a.pane])
deadline = time.monotonic() + a.hours * 3600
failures = 0
while time.monotonic() < deadline:
    try:
        process = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'hkust5', command],
                                 capture_output=True, text=True, timeout=55)
        if process.returncode:
            raise RuntimeError(f'observer rc={process.returncode}: {process.stderr[-600:]}')
        row = json.loads(process.stdout)
        failures = 0
    except (subprocess.TimeoutExpired, RuntimeError, ValueError) as error:
        failures += 1
        print(json.dumps(dict(observer_error=str(error), training_terminal_unknown=True,
                              consecutive_observation_failures=failures)), flush=True)
        if failures >= 3:
            break  # Observation failure is never authorization to restart.
        time.sleep(45)
        continue
    with (a.output / 'progress.jsonl').open('a') as stream:
        stream.write(json.dumps(row, sort_keys=True) + '\n')
    latest = row['latest']
    summary = {k: row[k] for k in ('time', 'log_bytes', 'pane', 'errors')}
    summary.update(completed_outer=list(row['metrics']), events=row['events'], latest=latest,
                   restore=row.get('restore'), gpu_samples=row.get('gpu_samples'))
    print(json.dumps(summary, sort_keys=True), flush=True)
    pane = row['pane'].split('|')
    if row['errors'] or pane[2] == '1':
        (a.output / 'attention.json').write_text(json.dumps(row, indent=2))
        break
    needed = set(map(str, range(4, a.until_outer + 1)))
    sampling = latest.get('TREE_ACTION_TIMING', {})
    if needed.issubset(row['metrics']) and sampling.get('outer', 0) > a.until_outer:
        (a.output / 'initial-outer-followup-passed.json').write_text(json.dumps(row, indent=2))
        print(json.dumps(dict(initial_followup_complete=True, training_budget=200,
                              training_not_stopped=True)), flush=True)
        break
    time.sleep(45)
