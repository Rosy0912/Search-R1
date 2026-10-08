"""Read-only HKUST-to-Mac checkpoint transfer with complete member SHA readback.

Never deletes or changes node outputs. Protected CP0/1/5 and outer0-end CP11
are immutable retained artifacts of the one fixed run. No weights enter Git.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile
import time

BASE = '/data/shaoshitong/search-e1-opsd-online-archive100-8gpu-20261008'
RUN = '1d1d67b60f0e46a2afc0536a36ab69b1'
CONFIG_SHA = 'd0c30df9b035e73d87bb5402bbcd32c16b2314a4180c451a543522ca5b321b17'
CHECKPOINTS = ['runs/checkpoints/update-%08d-outer-00000' % step for step in (0, 1, 5, 11)]
COMPACT = ['receipts/five-real-updates-accepted.json', 'receipts/five-real-updates-accepted.json.direct-memory.json',
           'receipts/config.json', 'receipts/memory-binding.json', 'runs/run-identity.json',
           'runs/outers/outer-00000/outer-manifest.json', 'runs/outers/outer-00000/training/outer-complete.json']


def sha_stream(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(4 << 20), b''):
        h.update(block)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())


def run(destination):
    destination = Path(destination).resolve(); destination.mkdir(parents=True, exist_ok=False)
    assert shutil.disk_usage(destination).free >= 20 << 30, 'Mac space below 20GiB; node originals retained'
    remote = '''
import json,sys,hashlib
from pathlib import Path
b=Path(%r);sys.path.insert(0,str(b/'source/revision-7cf2853/opsd_online'))
from checkpoint import fileproof,checked_json,check_checkpoint
config,cref=checked_json(dict(path=str(b/'receipts/config.json'),sha256=%r));config['_config_sha256']=%r
proofs=[]
for step,name in zip((0,1,5,11),%r):
    cp,evidence=check_checkpoint(fileproof(b/name/'manifest.json'),config,%r,step,0)
    declared=[evidence['manifest']]+[r['file'] for r in evidence['ranks']]+evidence['adapter_files']
    assert {str(p.resolve()) for p in (b/name).rglob('*') if p.is_file()}=={p['path'] for p in declared}
    assert not any(p.is_symlink() for p in (b/name).rglob('*'))
    declared.extend(fileproof(cp['training_input'][n]['path']) for n in ('outer_manifest','pairs','order','encoding'))
    proofs.extend(dict(relative=str(Path(p['path']).relative_to(b)),**p) for p in declared)
for name in %r:
    p=fileproof(b/name);proofs.append(dict(relative=name,**p))
unique={p['relative']:p for p in proofs}
assert all(not (b/name).is_symlink() for name in unique)
print(json.dumps(dict(accepted=True,run_id=%r,config=cref,files=list(unique.values()),source_files_unchanged=True)))
''' % (BASE, CONFIG_SHA, CONFIG_SHA, CHECKPOINTS, RUN, COMPACT, RUN)
    inventory = subprocess.run(['ssh', '-oConnectTimeout=15', 'hkust1', "CUDA_VISIBLE_DEVICES='' python3 -"],
        input=remote, text=True, capture_output=True, check=True)
    value = json.loads(inventory.stdout); write(destination / 'node-member-inventory.json', value)
    total = sum(f['bytes'] for f in value['files'])
    assert shutil.disk_usage(destination).free >= 2 * total + (5 << 30), 'Insufficient space for archive plus verified copy; originals retained'
    command = shlex.join(['tar', '-cf', '-', '-C', BASE] + sorted(f['relative'] for f in value['files']))
    archive = destination / 'protected-checkpoints.tar.partial'
    with archive.open('xb') as output, (destination / 'transfer.stderr').open('xb') as error:
        process = subprocess.Popen(['ssh', '-oConnectTimeout=15', 'hkust1', command], stdout=output, stderr=error)
        write(destination / 'transfer-start.json', dict(run_id=RUN, pid=process.pid, expected_bytes=total,
            node_base=BASE, destination=str(destination), started_unix=time.time(), accepted=False,
            no_node_release=True, no_node_files_changed=True))
        rc = process.wait(); output.flush(); os.fsync(output.fileno())
    assert rc == 0, 'SSH transfer failed; partial and node originals retained'
    expected = {f['relative']: f for f in value['files']}; observed = {}; extraction = destination / 'verified-files'
    extraction.mkdir()
    with tarfile.open(archive, 'r:') as tar:
        for member in tar:
            name = member.name.removeprefix('./')
            relative = Path(name)
            assert not relative.is_absolute() and '..' not in relative.parts
            if member.isdir():
                continue
            assert member.isfile() and name in expected and name not in observed, 'Unexpected/aliased/duplicate member'
            proof = expected[name]; assert member.size == proof['bytes']
            target = extraction / name; target.parent.mkdir(parents=True, exist_ok=True)
            h = hashlib.sha256()
            with tar.extractfile(member) as source, target.open('xb') as output:
                for block in iter(lambda: source.read(4 << 20), b''):
                    h.update(block); output.write(block)
                output.flush(); os.fsync(output.fileno())
            assert h.hexdigest() == proof['sha256'], 'Transferred bytes differ: ' + name
            observed[name] = dict(bytes=member.size, sha256=h.hexdigest(), local_path=str(target))
    assert set(observed) == set(expected), 'Incomplete checkpoint backup'
    final = archive.with_suffix(''); os.rename(archive, final)
    with final.open('rb') as stream:
        archive_sha = sha_stream(stream)
    write(destination / 'complete-sha-verified.json', dict(accepted=True, run_id=RUN, files=observed,
        archive=str(final), archive_sha256=archive_sha, archive_bytes=final.stat().st_size,
        completed_unix=time.time(), checkpoints=[0,1,5,11], node_outputs_retained=True))
    print(json.dumps(dict(accepted=True, files=len(observed), destination=str(destination))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--destination', required=True)
    run(parser.parse_args().destination)
