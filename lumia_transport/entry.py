"""Continue one frozen native run after its parent allocation is terminal."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from .runtime import PREFIX, install, load, sha, worker_setup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binding', required=True)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    os.environ[PREFIX + 'BINDING'] = str(Path(args.binding).resolve())
    os.environ[PREFIX + 'BINDING_SHA256'] = args.sha256
    binding, record = load()
    from reproduction.recursive_lumia import ablation_entry as native
    from release.recursive_bran_a.ablation150_v1 import bootstrap
    frozen = Path(binding['frozen_source']).resolve()
    for module in (native, bootstrap):
        if not Path(module.__file__).resolve().is_relative_to(frozen):
            raise ValueError('Entry/bootstrap must import from the original frozen checkout')
    previous = json.loads((Path(record['base']) / 'receipts/current-entry.json').read_text())
    if (str(previous['allocation']) != str(binding['parent_job'])
            or previous['host'] != binding['parent_host']
            or sha(Path(record['base']) / 'receipts/current-entry.json') != binding['parent_entry_sha256']):
        raise ValueError('The bound parent is no longer the exact current attempt')
    native.check_previous_scheduler(previous, os.environ['SLURM_JOB_ID'],
        lambda job: subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', job,
            '--format=JobIDRaw,State'], text=True))
    manifest = Path(binding['manifest'])
    if sha(manifest) != binding['manifest_sha256']:
        raise ValueError('Bound native boundary changed')
    value = json.loads(manifest.read_text())
    if (str(value['identity']['job_id']) != str(binding['parent_job'])
            or value['identity']['code_commit'] != record['source_commit']
            or value['identity']['config_sha256'] != record['config_sha256']):
        raise ValueError('Native boundary is not from the exact frozen parent')
    original = native.validate_allocation

    def admission(actual, env, uid, host):
        load()
        # All original device/source/baseline/condition checks remain in force.
        projected = dict(actual, host=binding['target_host'])
        return original(projected, env, uid, host)

    native.validate_allocation = admission
    install()
    bootstrap.worker_setup = worker_setup
    receipt = dict(version=1, at=time.time(), allocation=os.environ['SLURM_JOB_ID'],
        host=binding['target_host'], node=os.environ['SLURM_JOB_NODELIST'], uid=os.getuid(),
        binding_path=str(Path(args.binding).resolve()), binding_sha256=args.sha256,
        binding_original_text=Path(args.binding).read_text(), binding=binding,
        scope='Explicit physical-host transport; actual eight-rank native restoration is separately required.')
    target = Path(record['base']) / 'receipts' / ('transport-' + os.environ['SLURM_JOB_ID'] + '.json')
    with target.open('x') as stream:
        json.dump(receipt, stream, indent=2)
    print('LUMIA_NATIVE_TRANSPORT=' + json.dumps(receipt), flush=True)
    native.execute(argparse.Namespace(receipt=binding['configuration'], retry_unstarted=False,
        resume_manifest=binding['manifest'], resume_sha256=binding['manifest_sha256']))


if __name__ == '__main__':
    main()
