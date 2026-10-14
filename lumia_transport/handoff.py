"""Bounded readiness gate: no model is loaded before explicit parent handoff."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .runtime import PREFIX, load, worker_setup


def main():
    os.environ[PREFIX + 'BINDING'] = str(Path(sys.argv[1]).resolve())
    os.environ[PREFIX + 'BINDING_SHA256'] = sys.argv[2]
    binding, record = load()
    worker_setup()
    from reproduction.recursive_lumia.ablation_entry import validate_allocation
    validate_allocation(dict(record, host=binding['target_host']), os.environ,
                        os.getuid(), binding['target_host'])
    job = os.environ['SLURM_JOB_ID']
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,uuid',
        '--format=csv,noheader'], text=True).splitlines()
    if len(inventory) != 8 or any('L40S' not in row for row in inventory):
        raise ValueError('This handoff requires the actual eight-L40S allocation')
    directory = Path(record['base']) / 'receipts'
    ready = dict(at=time.time(), allocation=job, binding_sha256=sys.argv[2],
        node=os.environ['SLURM_JOB_NODELIST'], host=binding['target_host'], inventory=inventory,
        scope='Actual host/source/import admission only; no model initialization or training yet.')
    with (directory / ('transport-ready-' + job + '.json')).open('x') as stream:
        json.dump(ready, stream, indent=2)
    print('LUMIA_TRANSPORT_READY=' + json.dumps(ready), flush=True)
    release = directory / ('transport-release-' + job + '.json')
    for _ in range(120):
        if release.exists():
            value = json.loads(release.read_text())
            if (value['allocation'] != job or value['binding_sha256'] != sys.argv[2]
                    or str(value['parent_job']) != str(binding['parent_job'])):
                raise ValueError('Explicit handoff release does not match this exact allocation')
            return  # Native entry independently requires terminal parent and exact state.
        time.sleep(5)
    raise TimeoutError('Explicit handoff not released within ten minutes; no training started')


if __name__ == '__main__':
    main()
