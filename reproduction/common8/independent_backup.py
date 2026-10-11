"""Read-only source copy for this run onto retained h100-10 local ext4.

No model loading or GPU access. A checkpoint copy is complete only after every
source file has been read in full and every destination file independently read.
The launcher owns the exclusive namespace and creates INTENT before spawning.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time


RUN_ID = '15bb15365346401da1d919e9cde8f813'
SOURCE_COMMIT = '2de313e3e455e005e0d7db83104942bad8bd977a'
ORIGINAL_BASE = Path('/mnt/data/shaoshitong/search-g1-mixed-2de313e-20261011')
SOURCE_BASE = Path('/mnt/shared/workspace/training_data_node_8') / ORIGINAL_BASE.relative_to('/mnt/data')
BACKUP_BASE = Path('/mnt/data/shaoshitong/search-g1-mixed-first-independent-backup-20261011')
RATE = 20 * 1024**2
RESERVE = 128 * 1024**3


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def record(root, name, value):
    tmp = root / (name + '.tmp')
    with tmp.open('w') as output:
        json.dump(value, output, indent=2)
        output.write('\n')
        output.flush()
        os.fsync(output.fileno())
    os.replace(tmp, root / name)
    fd = os.open(root, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def identity(path):
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def paced_read(path, target=None, progress=None, rate=RATE):
    before = identity(path)
    total = 0
    digest = hashlib.sha256()
    started = time.monotonic()
    reported = started
    with path.open('rb') as source:
        while True:
            block = source.read(1024**2)
            if not block:
                break
            if target is not None:
                target.write(block)
            digest.update(block)
            total += len(block)
            time.sleep(max(0, total / rate - (time.monotonic() - started)))
            if progress and time.monotonic() - reported >= 10:
                progress(total)
                reported = time.monotonic()
    require(before == identity(path), 'File changed during full read: ' + str(path))
    require(total == before[2], 'Short full read: ' + str(path))
    if progress:
        progress(total)
    return total, digest.hexdigest()


def admit(step, root):
    require(step in (5, 50, 100, 150, 200), 'Unrequested save point')
    require(socket.gethostname() == 'h100-10', 'Backup must run on h100-10')
    expected = BACKUP_BASE / f'job3364-{RUN_ID[:12]}-step{step}-once-v1'
    require(root == expected and root.resolve() == expected, 'Unexpected or symlinked destination')
    destination_fs = subprocess.check_output(['findmnt', '-n', '-T', str(root), '-o', 'FSTYPE'], text=True).strip()
    source_mount = subprocess.check_output(['findmnt', '-n', '-T', str(SOURCE_BASE), '-o', 'SOURCE'], text=True)
    require(destination_fs == 'ext4' and '10.32.0.8:/mnt/data' in source_mount, 'Independent mount identity differs')
    require(root.stat().st_dev != SOURCE_BASE.stat().st_dev, 'Destination is the source filesystem')
    run = json.loads((SOURCE_BASE / 'logs/run-manifest.json').read_text())
    require(run['run_id'] == RUN_ID and run['source_commit'] == SOURCE_COMMIT, 'Actual source run differs')
    require(run['model_revision'] == 'aa8e72537993ba99e69dfaafa59ed015b17504d1', 'Official model revision differs')
    require(run['budget']['common_manifest_sha256'] == '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24', 'Canonical data identity differs')
    original = ORIGINAL_BASE / 'checkpoints/search-g1-common8/actor' / f'global_step_{step}'
    source = SOURCE_BASE / original.relative_to(ORIGINAL_BASE)
    require(run['checkpoint_dir'] == str(original.parent.parent), 'Checkpoint path differs from actual run')
    raw = (source / 'complete.json').read_bytes()
    complete = json.loads(raw)
    require(complete['run_id'] == RUN_ID and complete['step'] == step, 'Complete receipt identity differs')
    require(complete['finite_all_saved_tensors'] is True and complete.get('training_state_readback'), 'Full state save is not complete')
    manifest = json.loads((source / 'training_state/manifest.json').read_text())
    require(manifest['completed_outer'] == step, 'Training state step differs')
    proofs = manifest['worker_files']
    require(sorted(p['rank'] for p in proofs) == list(range(8)), 'Missing actual rank state')
    writes = complete['actual_optimizer_writes_per_rank']
    require(writes > 0 and all(p['actual_optimizer_writes'] == writes and p['run_id'] == RUN_ID for p in proofs), 'Adam receipt differs')
    expected_hashes = dict(complete['files_sha256'])
    expected_hashes['complete.json'] = hashlib.sha256(raw).hexdigest()
    for proof in proofs + [manifest['driver_file']]:
        relative = str(Path(proof['path']).relative_to(original))
        require(expected_hashes.get(relative) == proof['sha256'], 'State manifest and complete receipt differ')
    files = []
    for path in sorted(source.rglob('*')):
        require(not path.is_symlink(), 'Checkpoint has a symlink')
        if path.is_dir():
            continue
        require(path.is_file() and not path.name.endswith(('.partial', '.tmp')), 'Checkpoint has unfinished or special files')
        relative = str(path.relative_to(source))
        files.append(dict(relative_path=relative, bytes=path.stat().st_size,
                          expected_sha256=expected_hashes.get(relative)))
    require(set(expected_hashes) <= {p['relative_path'] for p in files}, 'Receipt references absent files')
    need = sum(p['bytes'] for p in files)
    fs = os.statvfs(root)
    free = fs.f_bavail * fs.f_frsize
    require(free >= need + RESERVE, 'Insufficient ordinary destination space; preserve source')
    intent = json.loads((root / 'INTENT.json').read_text())
    require(intent['run_id'] == RUN_ID and intent['source_commit'] == SOURCE_COMMIT and intent['step'] == step, 'Exclusive intent identity differs')
    require(hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == intent['code_sha256'], 'Backup executable differs from exclusive intent')
    return source, files, need, free, hashlib.sha256(raw).hexdigest()


def backup(step, root):
    os.nice(19)
    result = dict(utc_start=utc(), pid=os.getpid(), hostname=socket.gethostname(),
                  run_id=RUN_ID, source_commit=SOURCE_COMMIT, step=step,
                  bytes_per_second_limit=RATE, semantic_reload_performed=False,
                  all_experiment_backup_complete=False)
    try:
        source, files, need, free, receipt_sha = admit(step, root)
        staging = root / 'checkpoint.partial'
        staging.mkdir()
        copied = []
        result.update(source=str(source), source_complete_sha256=receipt_sha,
                      expected_bytes=need, expected_files=len(files), admitted_free_bytes=free)
        for row in files:
            src = source / row['relative_path']
            dst = staging / row['relative_path']
            fs = os.statvfs(root)
            require(fs.f_bavail * fs.f_frsize >= row['bytes'] + RESERVE, 'Destination ordinary space reserve exhausted')
            dst.parent.mkdir(parents=True, exist_ok=True)
            def progress(count, phase='copy'):
                record(root, 'PROGRESS.json', dict(utc=utc(), pid=os.getpid(),
                    completed_files=len(copied), completed_bytes=sum(p['bytes'] for p in copied),
                    expected_bytes=need, expected_files=len(files), current_file=row['relative_path'],
                    current_phase=phase, current_file_bytes=count))
            with dst.open('xb') as output:
                count, sha = paced_read(src, output, progress)
                output.flush()
                os.fsync(output.fileno())
            require(count == row['bytes'], 'Source file size differs')
            if row['expected_sha256']:
                require(sha == row['expected_sha256'], 'Source full read differs from saved receipt')
            read_count, read_sha = paced_read(dst, progress=lambda count: progress(count, 'independent_target_readback'))
            require((read_count, read_sha) == (count, sha), 'Independent target full readback differs')
            copied.append(dict(row, source_full_read_sha256=sha, destination_full_read_sha256=read_sha))
            progress(count, 'verified')
        require(hashlib.sha256((source / 'complete.json').read_bytes()).hexdigest() == receipt_sha, 'Source completion receipt changed')
        directories = [p for p in staging.rglob('*') if p.is_dir()] + [staging]
        for directory in sorted(directories, key=lambda p: len(p.parts), reverse=True):
            fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        final = root / 'checkpoint'
        require(not final.exists(), 'Final checkpoint already exists')
        os.rename(staging, final)
        result.update(status='INDEPENDENT_CHECKPOINT_FULL_SHA_PASS', utc_end=utc(),
                      destination=str(final), files=copied, bytes=need,
                      independent_destination_mount='h100-10 local ext4')
        record(root, 'RESULT.json', result)
    except BaseException as error:
        result.update(status='FAILED_PRESERVE_SOURCE_AND_PARTIAL', utc_end=utc(), error=repr(error))
        record(root, 'RESULT.json', result)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, required=True)
    args = parser.parse_args()
    namespace = BACKUP_BASE / f'job3364-{RUN_ID[:12]}-step{args.step}-once-v1'
    backup(args.step, namespace)
