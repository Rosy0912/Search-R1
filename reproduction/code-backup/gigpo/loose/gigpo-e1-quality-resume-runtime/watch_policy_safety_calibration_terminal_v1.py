"""Read-only scheduler observer for one calibration job; preserves actual terminal bytes."""
import datetime
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004')
job = sys.argv[1]
assert job.isdigit()
output = ROOT / f'logs/policy-safety-calibration-terminal-{job}-raw-v1.json'
log = ROOT / f'logs/policy-safety-calibration-scheduler-watch-{job}-raw-v1.jsonl'
assert not output.exists() and not log.exists(), 'Existing observer evidence: inspect, do not overwrite'
terminal = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL',
            'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}
with log.open('x') as stream:
    for _ in range(5000):
        command = ['scontrol', 'show', 'job', job, '-o']
        observed = subprocess.run(command, capture_output=True, text=True, timeout=15)
        record = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), args=command,
                      returncode=observed.returncode, stdout=observed.stdout, stderr=observed.stderr)
        stream.write(json.dumps(record) + '\n')
        stream.flush()
        fields = dict(item.split('=', 1) for item in observed.stdout.split() if '=' in item)
        if observed.returncode == 0 and fields.get('JobState') in terminal:
            assert fields.get('JobId') == job and fields.get('NodeList') == 'h100-8'
            record.update(fields=fields, scope='Actual raw terminal observation only; not independent acceptance')
            with output.open('x') as result:
                result.write(json.dumps(record, indent=2) + '\n')
            print(json.dumps(dict(path=str(output), sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                                  state=fields['JobState'], exit_code=fields.get('ExitCode'))), flush=True)
            break
        time.sleep(20)
    else:
        raise TimeoutError('Observer exhausted: inspect original job; never infer success or resubmit')
