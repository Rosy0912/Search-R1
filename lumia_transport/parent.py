"""Keep the native parent-wait format for a transported entry subprocess."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid


def main():
    binding = json.loads(Path(sys.argv[1]).read_text())
    command = [sys.executable, '-m', 'search_r1.baselines.preflight', '--',
        sys.executable, '-m', 'lumia_transport.entry', '--binding', sys.argv[1], '--sha256', sys.argv[2]]
    child = subprocess.Popen(command)
    rc = child.wait()
    record = dict(at=time.time(), pid=child.pid, rc=rc, group=binding['group'],
        allocation=os.environ['SLURM_JOB_ID'], source=binding['frozen_source_commit'],
        command=command, scope='Actual Popen.wait return for the bound transport entry')
    base = Path(binding['configuration']).parent.parent
    with (base / 'receipts' / ('parent-wait-' + uuid.uuid4().hex + '.json')).open('x') as stream:
        json.dump(record, stream, indent=2)
    current = base / 'receipts/current-entry.json'
    if current.exists():
        entry = json.loads(current.read_text())
        if entry['pid'] == child.pid and entry['allocation'] == record['allocation']:
            from release.recursive_bran_a.ablation150_v1.progress import replace_json
            replace_json(base / 'receipts/actual-parent-wait.json', record)
    print('LUMIA_ABLATION_PARENT_WAIT=' + json.dumps(record), flush=True)
    raise SystemExit(rc if rc >= 0 else 128-rc)


if __name__ == '__main__':
    main()
