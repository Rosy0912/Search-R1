"""Record an actual child wait, including failures before native initialization."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid


def main():
    receipt = Path(sys.argv[1]).resolve()
    record = json.loads(receipt.read_text())
    command = [sys.executable,'-m','search_r1.baselines.preflight','--',
        sys.executable,'-m','reproduction.recursive_lumia.ablation_entry',
        '--receipt',str(receipt),*sys.argv[2:]]
    child = subprocess.Popen(command)
    rc = child.wait()
    value = dict(at=time.time(),pid=child.pid,rc=rc,group=record['group'],
        allocation=os.environ['SLURM_JOB_ID'],source=os.environ['RECURSIVE_SOURCE_COMMIT'],
        command=command,scope='Actual Popen.wait return, not inferred from timestamps')
    base = Path(record['base'])
    path = base/'receipts'/('parent-wait-'+uuid.uuid4().hex+'.json')
    with path.open('x') as stream:json.dump(value,stream,indent=2)
    current = base/'receipts/current-entry.json'
    if current.exists():
        entry = json.loads(current.read_text())
        if entry['pid'] == child.pid and entry['allocation'] == value['allocation']:
            from release.recursive_bran_a.ablation150_v1.progress import replace_json
            replace_json(base/'receipts/actual-parent-wait.json',value)
    print('LUMIA_ABLATION_PARENT_WAIT='+json.dumps(value),flush=True)
    raise SystemExit(rc if rc >= 0 else 128-rc)


if __name__ == '__main__':
    main()
