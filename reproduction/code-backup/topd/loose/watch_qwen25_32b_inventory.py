"""Wait for official download; submit one CPU-only inventory check, never GPU."""
import fcntl
import json
from pathlib import Path
import subprocess
import time

BASE = Path(__file__).resolve().parent.parent
# Current authorized v6 downloader owns this state; never dispatch from legacy state.
STATE = BASE / 'outputs/qwen25-32b-download-v6-state.json'
RECEIPT = BASE / 'outputs/qwen25-32b-inventory-dispatch.json'
lock = (BASE / 'outputs/qwen25-32b-inventory-watch.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
if RECEIPT.exists():
    raise SystemExit('Dispatch record exists; inspect it, never resubmit automatically')
deadline = time.monotonic() + 24 * 3600
while time.monotonic() < deadline:
    if not STATE.is_file():
        time.sleep(30)
        continue  # Missing v6 remains pending; no fallback to stale legacy completion.
    data = json.loads(STATE.read_text())
    shards = {n: row for n, row in data['files'].items() if n.endswith('.safetensors')}
    if data.get('complete') and not data.get('last_error'):
        assert data['revision'] == '1818d35814b8319459f4bd55ed1ac8709630f003'
        assert len(shards) == 17 and all(row['remote_sha_verified'] and row['sha256'] == row['official_lfs_sha256'] for row in shards.values())
        record = dict(status='SUBMISSION_UNKNOWN_CHECK_REMOTE_BEFORE_RETRY', scope='CPU_ONLY_OFFICIAL_INVENTORY_NO_GPU_NO_TRAINING', started_epoch=time.time(), download_state_path=str(STATE), download_revision=data['revision'])
        with RECEIPT.open('x') as stream:
            json.dump(record, stream, indent=2)
        command = '''set -e
ROOT=/dataset/shaoshitong/search-r1-rl/topd
printf '%s\n' '6d92fd84e7c0230e654433ac431ddfb4926c8cce8976b4525cd30e45e123c12a  /dataset/shaoshitong/search-r1-rl/topd/tools/qwen25-32b-inventory.sbatch' | sha256sum -c - >&2
if test -f "$ROOT/logs/qwen25-32b-eval64-binding.json"; then
  echo EXISTING_BINDING_REVIEW_REQUIRED
else
  active=$(squeue -h -w h100-11 -n topd32b-official-inventory -o '%i')
  if test -n "$active"; then
    echo EXISTING_INVENTORY_JOB "$active"
  else
    sbatch --parsable "$ROOT/tools/qwen25-32b-inventory.sbatch"
  fi
fi
'''
        try:
            result = subprocess.run(['python3', str(BASE / 'work/remote_topd.py')], input=command,
                                    text=True, capture_output=True, timeout=180, cwd=BASE)
            record.update(returncode=result.returncode, stdout=result.stdout.strip(), stderr=result.stderr[-2000:], completed_epoch=time.time())
            if result.returncode == 0:
                record['status'] = 'CPU_INVENTORY_DISPATCH_RESPONSE_RECEIVED'
        except subprocess.TimeoutExpired:
            record.update(error='SSH response timeout; query records/jobs before any retry', completed_epoch=time.time())
        RECEIPT.write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(record), flush=True)
        break
    time.sleep(30)
else:
    print('WATCH_TIMEOUT_NO_SUBMISSION_NO_GPU', flush=True)
