"""Copy a completed B checkpoint to an independent HKUST4 archive; never delete.

Run on HKUST5 through an existing authenticated/agent-forwarded SSH session.
The destination is an ext4 data disk accessed through SSH, not a shared mount.
Recovery first copies the verified bundle back to its recorded checkpoint paths.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import time

SOURCE_BASE = Path('/home/shaoshitong/search-e1-migration-20261011/weka_3/shaoshitong/recursive-bran-b-4gpu-hkust5-20261011')
DEST_BASE = Path('/weka_3/shaoshitong/recursive-bran-b-hkust5-archive-20261012')
DEST_HOST = 'shaoshitong@10.123.4.11'
SSH = ['ssh', '-oBatchMode=yes', '-oConnectTimeout=8', '-oStrictHostKeyChecking=yes', DEST_HOST]


def emit(**data):
    print(json.dumps(dict(at=time.time(), **data)), flush=True)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def regular(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_uid != os.getuid():
        raise RuntimeError(f'Not an owned regular file: {path}')


def within(path, base):
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def safe_dirs(path, base):
    if not within(path, base):
        raise RuntimeError('Path outside archive boundary')
    for parent in [base, *reversed(path.relative_to(base).parents), path.relative_to(base)]:
        p = parent if parent == base else base / parent
        if p.exists() and (p.is_symlink() or not p.is_dir() or p.stat().st_uid != os.getuid()):
            raise RuntimeError(f'Archive directory owner/type changed: {p}')


def pin_root(run, step):
    return SOURCE_BASE / 'checkpoint-archive-pins' / run / f'global_step_{step}'


def payload(root, run, step, source, physical_root=None):
    if len(run) != 32 or any(c not in '0123456789abcdef' for c in run):
        raise RuntimeError('Invalid run identity')
    expected_root = SOURCE_BASE / f'output-speed-resume3-{run}' / 'checkpoints'
    if root != expected_root or root.resolve() != root:
        raise RuntimeError('Unexpected checkpoint root')
    logical_root = root
    if physical_root is not None:
        if physical_root != pin_root(run, step) or physical_root.resolve() != physical_root:
            raise RuntimeError('Unexpected archive pin root')
        root = physical_root
    native = root / 'training_state' / f'global_step_{step}'
    safe_dirs(native, SOURCE_BASE)
    regular(native / 'manifest.json')
    raw = (native / 'manifest.json').read_bytes()
    m = json.loads(raw)
    identity = m['identity']
    if (m['status'] != 'COMPLETE_SAME_WORLD_STATE_SAVED'
            or identity['completed_step'] != step
            or identity['job_id'] != f'hkust-direct-{run}'
            or identity['code_commit'] != source
            or m['driver']['next_step'] != step + 1
            or {w['identity']['rank'] for w in m['worker_states']} != set(range(4))):
        raise RuntimeError('Checkpoint is not this run complete boundary')
    files = {}

    def add(relative, sha):
        rel = Path(relative)
        if rel.is_absolute() or '..' in rel.parts:
            raise RuntimeError('Unsafe checkpoint member')
        p = root / rel
        safe_dirs(p.parent, SOURCE_BASE)
        regular(p)
        files[str(rel)] = dict(bytes=p.stat().st_size, sha256=sha)

    add(str((native / 'manifest.json').relative_to(root)), hashlib.sha256(raw).hexdigest())
    add(str((native / Path(m['driver']['path']).name).relative_to(root)), m['driver']['sha256'])
    for w in m['worker_states']:
        if not w['full_worker_state_finite'] or w['identity']['world_size'] != 4:
            raise RuntimeError('Incomplete worker state')
        add(str((native / Path(w['path']).name).relative_to(root)), w['sha256'])
        sampling = w['sampling_stream']
        add(str((native / Path(sampling['path']).name).relative_to(root)), sampling['file_sha256'])
    directories = [native]
    if m.get('hf_checkpoint'):
        hf = root / 'actor' / f'global_step_{step}'
        for name, sha in m['hf_identity']['files'].items():
            add(str((hf / name).relative_to(root)), sha)
        directories.append(hf)
    for directory in directories:
        actual = {str(p.relative_to(root)) for p in directory.rglob('*') if p.is_file() or p.is_symlink()}
        expected = {n for n in files if within(Path(n), directory.relative_to(root))}
        if actual != expected:
            raise RuntimeError('Unexpected checkpoint payload members')
    data = dict(run=run, step=step, source_commit=source, source_checkpoint_root=str(logical_root),
                manifest_sha256=hashlib.sha256(raw).hexdigest(), files=files,
                destination_root=str(DEST_BASE / run),
                storage_kind='independent_node_ext4_archive_over_SSH_not_shared_mount')
    return data, directories


def pin_snapshot(root, run, step, source):
    """Keep immutable file inodes alive if normal rolling pruning advances."""
    pinned = pin_root(run, step)
    ready = pinned / 'PIN_READY.json'
    safe_dirs(pinned, SOURCE_BASE)
    if ready.exists():
        regular(ready)
        data, directories = payload(root, run, step, source, physical_root=pinned)
        if json.loads(ready.read_text()) != data:
            raise RuntimeError('Existing pinned snapshot identity differs')
        return data, directories, pinned
    data, _ = payload(root, run, step, source)
    pinned.mkdir(parents=True, exist_ok=True)
    for relative in data['files']:
        src, dest = root / relative, pinned / relative
        safe_dirs(dest.parent, SOURCE_BASE)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            regular(dest)
            if not src.samefile(dest):
                raise RuntimeError('Incomplete pin belongs to another inode')
        else:
            os.link(src, dest)
    current, directories = payload(root, run, step, source, physical_root=pinned)
    if current != data:
        raise RuntimeError('Pinned snapshot differs from producer manifest')
    with ready.open('x') as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    emit(event='archive_source_pinned', step=step, bytes=sum(e['bytes'] for e in data['files'].values()),
         scope='Hardlinks only; original checkpoint paths and bytes unchanged')
    return data, directories, pinned


def receiver(data):
    assert os.getuid() == 1001 and socket.gethostname().split('.')[0] == 'xielab5'
    root = Path(data['destination_root'])
    if root != DEST_BASE / data['run']:
        raise RuntimeError('Unexpected destination root')
    staging = root / '.partial' / f"global_step_{data['step']}"
    final = root / 'checkpoint-bundles' / staging.name
    safe_dirs(staging, DEST_BASE)
    safe_dirs(final, DEST_BASE)
    selected = final if final.exists() else staging
    expected = set(data['files'])
    actual = {str(p.relative_to(selected)) for p in selected.rglob('*') if p.is_file() or p.is_symlink()}
    if actual - {'archive-complete.json'} != expected:
        raise RuntimeError('Archive member inventory differs')
    for relative, entry in data['files'].items():
        p = selected / relative
        safe_dirs(p.parent, DEST_BASE)
        regular(p)
        if p.stat().st_size != entry['bytes'] or digest(p) != entry['sha256']:
            raise RuntimeError(f'Archive payload checksum differs: {relative}')
        with p.open('rb') as f:
            os.fsync(f.fileno())
        emit(event='archive_file_verified', relative=relative, bytes=entry['bytes'])
    receipt = dict(data, status='COMPLETE_CHECKPOINT_ARCHIVE_SHA_VERIFIED',
                   verified_at=time.time(), local_source_deleted_by_archiver=False,
                   recovery='Copy bundle members back to source_checkpoint_root and verify SHA before native restore.')
    if not final.exists():
        with (staging / 'archive-complete.json').open('x') as f:
            json.dump(receipt, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        for directory in [p for p in staging.rglob('*') if p.is_dir()] + [staging]:
            fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        final.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(final)
        fd = os.open(final.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    else:
        old = json.loads((final / 'archive-complete.json').read_text())
        if old['manifest_sha256'] != data['manifest_sha256'] or old['files'] != data['files']:
            raise RuntimeError('Existing archive identity differs')
    emit(event='archive_complete', destination=str(final), receipt=receipt)


def copy(args):
    assert os.getuid() == 1001 and socket.gethostname().split('.')[0] == 'xie-lab-4'
    data, directories, pinned = pin_snapshot(Path(args.checkpoint_root), args.run, args.step, args.source_commit)
    staging = Path(data['destination_root']) / '.partial' / f'global_step_{args.step}'
    final = Path(data['destination_root']) / 'checkpoint-bundles' / staging.name
    preparation = f'''import os,json,shutil,socket
from pathlib import Path
{inspect.getsource(within)}
{inspect.getsource(safe_dirs)}
assert os.getuid()==1001 and socket.gethostname().split('.')[0]=='xielab5'
p=Path({str(staging)!r})
b=Path({str(DEST_BASE)!r})
safe_dirs(p,b)
p.mkdir(parents=True,exist_ok=True)
safe_dirs(p,b)
exists=Path({str(final)!r}).exists()
if not exists:
    assert shutil.disk_usage(p).free>{sum(e['bytes'] for e in data['files'].values())}+1073741824
print(json.dumps(dict(existing_complete=exists)))
'''
    state = json.loads(subprocess.check_output(SSH + ['python3 -c ' + shlex.quote(preparation)], text=True))
    if not state['existing_complete']:
        for directory in directories:
            relative = directory.relative_to(pinned)
            dest = staging / relative
            subprocess.run(SSH + ['mkdir -p ' + shlex.quote(str(dest))], check=True)
            command = ['rsync', '-rt', '--partial', '--append-verify', '--bwlimit=65536',
                       '-e', 'ssh -oBatchMode=yes -oConnectTimeout=8 -oStrictHostKeyChecking=yes',
                       str(directory) + '/', DEST_HOST + ':' + str(dest) + '/']
            process = subprocess.Popen(command)
            while process.poll() is None:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    program = f"from pathlib import Path;print(sum(p.stat().st_size for p in Path({str(staging)!r}).rglob('*') if p.is_file()))"
                    copied = int(subprocess.check_output(SSH + ['python3 -c ' + shlex.quote(program)], text=True))
                    emit(event='archive_copy_progress', staged_bytes=copied,
                         expected_bytes=sum(e['bytes'] for e in data['files'].values()))
            if process.returncode:
                raise RuntimeError(f'Checkpoint copy failed: {process.returncode}')
    # The producer must remain the same complete immutable manifest after copying.
    current, _ = payload(Path(args.checkpoint_root), args.run, args.step, args.source_commit, physical_root=pinned)
    if current != data:
        raise RuntimeError('Source checkpoint changed during copy')
    command = 'python3 - --receiver ' + shlex.quote(json.dumps(data))
    subprocess.run(SSH + [command], input=Path(__file__).read_text(), text=True, check=True)
    # Only this private transfer staging is reclaimed, after complete remote SHA.
    safe_dirs(pinned, SOURCE_BASE)
    for member in pinned.rglob('*'):
        if member.is_symlink():
            raise RuntimeError('Pinned staging changed before cleanup')
        if member.is_file():
            regular(member)
    shutil.rmtree(pinned)
    emit(event='archive_pin_released', step=args.step, original_checkpoint_deleted_by_archiver=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--receiver')
    p.add_argument('--run')
    p.add_argument('--step', type=int)
    p.add_argument('--checkpoint-root')
    p.add_argument('--source-commit')
    args = p.parse_args()
    if args.receiver:
        receiver(json.loads(args.receiver))
    else:
        if not all([args.run, args.step, args.checkpoint_root, args.source_commit]):
            p.error('run, step, checkpoint-root and source-commit required')
        copy(args)


if __name__ == '__main__':
    main()
