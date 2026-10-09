"""Run frozen recorded-log readers against independent HKUST snapshots."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE = Path('/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008')
GOAL = BASE / 'receipts/outer-backup-v2-554c52c-20261009/goal.json'
GOAL_SHA = 'c65bea46dbb4ef6451ed56b2c4e26adafde5fd247e6b67fb5bea593b8689c94f'
SOURCES = dict(collector=('collector.py', '50fc6bf8609ef93f4ca68907b9f688252010d950e59188efc467552b5c2c220c'),
    clipmass=('aggregate_clipmass.py', 'b7a264119fd18b9ac744f452c86e8deb6db550c318050178e94adb38a1f39c8a'),
    coverage=('debug_coverage.py', '99e97c08e8eb1378f5b90b2437a826715d6568abb7d82c954a3923c5bc778fb9'))
ORDER = ('C0', 'C1', 'M', 'R', 'W', 'P', 'J')


def checked(path, expected=None):
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Explicit canonical nonsymlink file required')
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError('Recorded input changed while reading')
    sha = hashlib.sha256(raw).hexdigest()
    if expected is not None and sha != expected:
        raise ValueError('Pinned source/input SHA differs')
    return dict(path=str(path), sha256=sha, bytes=len(raw)), raw


def value(ref):
    proof, raw = checked(ref['path'], ref['sha256'])
    if proof['bytes'] != ref['bytes']:
        raise ValueError('Declared byte count differs')
    return json.loads(raw)


def save(path, item):
    with path.open('x') as handle:
        handle.write(json.dumps(item, sort_keys=True, ensure_ascii=False, indent=2) + '\n')
    return checked(path)[0]


def main():
    parser = argparse.ArgumentParser()
    for name in ('copies-complete', 'copies-complete-sha256', 'copies-wait', 'copies-wait-sha256', 'source-root', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    if os.getuid() != 1005 or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or 'torch' in sys.modules:
        raise ValueError('Original HKUST UID CPU-only process required')
    source_root, output = Path(args.source_root), Path(args.output)
    if not output.is_absolute() or not output.resolve().is_relative_to(BASE / 'analysis') or output.exists():
        raise ValueError('New independent remote audit output required')
    source_refs = {}
    for role, (name, sha) in SOURCES.items():
        source_refs[role] = checked(source_root / name, sha)[0]
    copies_ref, raw = checked(args.copies_complete, args.copies_complete_sha256)
    copies = json.loads(raw)
    if not copies.get('accepted') or copies['schema'] != 'opsd-ablation-remote-raw-copy-v1':
        raise ValueError('Original actually completed remote copies required')
    wait_ref, raw = checked(args.copies_wait, args.copies_wait_sha256)
    waited = json.loads(raw)
    if waited.get('actual_ssh_returncode') != 0 or waited.get('output') != str(Path(args.copies_complete).parent):
        raise ValueError('Captured actual complete-copy wait0 required; absence is not success')
    groups = []
    for variant in ORDER:
        if variant not in copies['groups']:
            continue
        closed = value(copies['groups'][variant])
        if not closed.get('accepted') or closed['variant'] != variant or not closed['independent_inode_copy']:
            raise ValueError('Original accepted independent copy identity differs')
        groups.append(dict(variant=variant, config=closed['config'], backups=closed['backups']))
    if not groups:
        raise ValueError('No closed copied group')
    checked(GOAL, GOAL_SHA)
    output.mkdir(parents=True, exist_ok=False)
    executions = {}

    def run(name, command, stdout_path=None):
        path = stdout_path or output / (name + '.stdout')
        started = time.time()
        print(json.dumps(dict(event='recorded_reader_started', role=name, processing_host='HKUST1')), flush=True)
        with path.open('xb') as stdout, (output / (name + '.stderr')).open('xb') as stderr:
            child = subprocess.Popen(command, stdout=stdout, stderr=stderr,
                                     env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1'))
            rc = child.wait()
        execution = dict(actual_child_pid=child.pid, actual_wait_returncode=rc, argv=command,
            started_unix=started, ended_unix=time.time(), stdout=checked(path)[0],
            processing_host='HKUST1', Mac_raw_payloads=False, no_training_locks=True,
            unchanged_scientific_code=True)
        executions[name] = save(output / (name + '-execution.json'), execution)
        print(json.dumps(dict(event='recorded_reader_terminal', role=name, actual_wait_returncode=rc,
                              execution=executions[name])), flush=True)
        if rc:
            raise ValueError('Recorded reader failed; preserve stderr and partial output: ' + name)
        return execution

    facts = output / 'facts.private.json'
    run('facts', [sys.executable, '-B', source_refs['collector']['path'], '--goal', str(GOAL), '--goal-sha256', GOAL_SHA], facts)
    facts_ref, raw = checked(facts)
    if not json.loads(raw).get('accepted'):
        raise ValueError('Unaccepted actual native facts')
    inputs = save(output / 'clipmass-inputs.private.json', dict(schema='seven-group-closed-clipmass-inputs-v1', facts=facts_ref, groups=groups))
    clipmass = output / 'clipmass'
    run('clipmass', [sys.executable, '-B', source_refs['clipmass']['path'], '--manifest', inputs['path'],
        '--manifest-sha256', inputs['sha256'], '--output', str(clipmass)])
    coverage_inputs = save(output / 'coverage-inputs.private.json', dict(schema='opsd-ablation-recorded-debug-coverage-inputs-v1',
        aggregate_source=source_refs['clipmass'], clipmass_inputs=inputs,
        committed_proof=checked(clipmass / 'referenced-files-proof.private.json')[0],
        clipmass_summary=checked(clipmass / 'clipmass-publication.json')[0],
        clipmass_derived_manifest=checked(clipmass / 'derived-manifest.json')[0]))
    coverage = output / 'coverage'
    run('coverage', [sys.executable, '-B', source_refs['coverage']['path'], '--manifest', coverage_inputs['path'],
        '--manifest-sha256', coverage_inputs['sha256'], '--output', str(coverage)])
    complete = save(output / 'complete.json', dict(schema='opsd-ablation-remote-recorded-readers-v1', accepted=True,
        groups=[g['variant'] for g in groups], copies=copies_ref, copies_actual_wait=wait_ref, executions=executions, source_refs=source_refs,
        processing_host='HKUST1', Mac_raw_payloads=False,
        clipmass_summary=checked(clipmass / 'clipmass-publication.json')[0],
        debug_summary=checked(coverage / 'debug-coverage-publication.json')[0],
        ended_unix=time.time(), scope='Recorded committed scalar/top32/NPY coverage; no logits, gradient, inference or causal reconstruction'))
    print(json.dumps(dict(accepted=True, complete=complete, groups=[g['variant'] for g in groups])), flush=True)


if __name__ == '__main__':
    main()
