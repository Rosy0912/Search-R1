#!/usr/bin/env python3
"""One low-priority same-H100 native backup; never loads tensors or alters training.

Run only on h100-10. A request is never automatically relaunched after an unknown
result. Original training paths and existing shared assets are read-only.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import time

SOURCE = Path('/mnt/data/shaoshitong/search-r1-rl/tree-grpo-retained18-20261008')
ALIAS = Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
SHARED = Path('/mnt/shared/workspace/h100-11_from_h100-9/shaoshitong/search-r1-rl/tree-grpo')
DEST = SHARED / 'retained18-backup-20261008'
CONTROL = SOURCE / '.shared-output-backup-20261008'
CODE = 'ebef786697b78abafc2248c0c794827606c7214a'
DATA = '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
MODEL = '744d9f616373b3890cf815412560e34041e21cd4915c026967d8cc13a423aea0'
CONFIG = '72502a1e48e757540f33fd848bacf681ba2a8796fc8b448e8b18d2f9cdf4ba47'
FALLBACK_MANIFEST = '2421b7f1f8f8db0d5b73e342e9de1a9c8b61a5fe08cb6596da492a0d3a04fbc3'
CHUNK = 4 * 1024 * 1024
RATE = 128 * 1024 * 1024
GIB = 1024**3
SMALL_LIMIT = 32 * 1024 * 1024
META_LIMIT = 512 * 1024 * 1024
SUFFIXES = {'.py', '.sh', '.sbatch', '.yaml', '.yml', '.json', '.jsonl', '.md', '.txt', '.toml', '.cfg', '.ini', '.out', '.err', '.log'}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def command(args):
    return subprocess.check_output(args, text=True, timeout=30).strip()


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, value):
    tmp = path.with_name(path.name + '.writing-' + str(os.getpid()))
    with tmp.open('x') as f:
        json.dump(value, f, sort_keys=True, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fsync_dir(path.parent)


def read_small(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        st = os.fstat(fd)
        require(stat.S_ISREG(st.st_mode) and st.st_size <= SMALL_LIMIT, 'Not a bounded regular metadata file: ' + str(path))
        raw = bytearray()
        while len(raw) < st.st_size:
            block = os.read(fd, min(CHUNK, st.st_size - len(raw)))
            require(block, 'Metadata truncated: ' + str(path))
            raw.extend(block)
        final = os.fstat(fd)
        require((final.st_size, final.st_mtime_ns) == (st.st_size, st.st_mtime_ns), 'Metadata changed while reading: ' + str(path))
        return bytes(raw)
    finally:
        os.close(fd)


def write_bytes(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    require(sha(read_small(path)) == sha(raw), 'Metadata readback hash failed: ' + str(path))


def physical(path):
    path = Path(path)
    if path.is_relative_to(ALIAS):
        path = SOURCE / path.relative_to(ALIAS)
    require(path.is_relative_to(SOURCE), 'Source path escaped task namespace')
    require(path.resolve(strict=True) == path, 'Unexpected source symlink')
    return path


def identity_check(identity, job):
    expected = dict(job_id=job, run_id='tree-current200-continue-' + job,
                    code_commit=CODE if job == '3341' else '62271081ba1bb74d8fc3f4958d623758efa74fa5',
                    config_sha256=CONFIG if job == '3341' else '98a88882c09d8bbba89908b04d1f72930b4fccccc2bbcb5b6eb6886584655f3b',
                    data_identity=DATA, model_identity=MODEL, actual_update_limit=200)
    for key, value in expected.items():
        require(identity.get(key) == value, 'Identity mismatch: ' + key)
    step = identity['completed_step']
    require(isinstance(step, int) and (12 <= step <= 200 if job == '3341' else step == 6), 'Unexpected completed step')
    return step


class Race(Exception):
    pass


def pin_native(job, manifest_path=None, expected_sha=None):
    """All descriptors are acquired before any destination large file is opened."""
    fds = []
    directory_fd = None
    try:
        run = SOURCE / 'runs' / ('tree-current200-continue-' + job)
        latest_raw = None
        if manifest_path is None:
            latest_raw = read_small(run / 'checkpoints/training_state/rolling-completed/latest.json')
            latest = json.loads(latest_raw)
            require(latest['status'] == 'COMPLETE_NATIVE_ROLLING_STATE_SAVED', 'Latest is not complete')
            identity_check(latest['identity'], job)
            manifest_path, expected_sha = latest['manifest_path'], latest['manifest_sha256']
        path = physical(manifest_path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        mfd = os.open('manifest.json', os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
        fds.append(mfd)
        mst = os.fstat(mfd)
        require(stat.S_ISREG(mst.st_mode) and mst.st_size < SMALL_LIMIT, 'Bad manifest stat')
        raw = os.read(mfd, mst.st_size + 1)
        require(len(raw) == mst.st_size and sha(raw) == expected_sha, 'Original manifest SHA mismatch')
        manifest = json.loads(raw)
        require(manifest['status'] == 'COMPLETE_NATIVE_ROLLING_STATE_SAVED', 'Incomplete native')
        identity = manifest['identity']
        step = identity_check(identity, job)
        require(path.parent == run / 'checkpoints/training_state/rolling-completed' / ('global_step_' + str(step)), 'Unexpected checkpoint location')
        if latest_raw:
            require(latest['identity'] == identity, 'Latest and manifest identities differ')
        required = ['adam_optimizer', 'scheduler', 'sampler_order_cursor_generator', 'tree_rng_and_algorithm_state',
                    'worker_model_shards', 'worker_parameters', 'worker_python_numpy_torch_cuda_rng', 'driver_python_numpy_torch_rng']
        require(all(manifest['saved_components'].get(k) is True for k in required), 'Missing native components')
        workers = manifest['worker_states']
        require(len(workers) == 8 and sorted(w['identity']['rank'] for w in workers) == list(range(8)), 'Need eight distinct ranks')
        entries = []
        for w in sorted(workers, key=lambda x: x['identity']['rank']):
            rank = w['identity']['rank']
            require(identity_check(w['identity'], job) == step and w['identity'].get('world_size') == 8, 'Rank identity mismatch')
            require(w['scheduler_last_epoch'] == step, 'Scheduler state identity mismatch')
            require(w.get('full_worker_state_finite') is True, 'Rank finite-state receipt missing')
            entries.append((w, 'rank-%02d.pt' % rank))
        require(manifest['driver']['next_step'] == step + 1, 'Driver next-step mismatch')
        entries.append((manifest['driver'], 'driver.pt'))
        opened = []
        for entry, name in entries:
            require(physical(entry['path']) == path.parent / name, 'Manifest file path mismatch')
            require(re.fullmatch('[0-9a-f]{64}', entry['sha256']) is not None, 'Missing complete file SHA')
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
            fds.append(fd)
            st = os.fstat(fd)
            require(stat.S_ISREG(st.st_mode) and st.st_size > 0, 'Invalid native regular file')
            if 'bytes' in entry:
                require(st.st_size == entry['bytes'], 'Native size differs from original manifest')
            opened.append(dict(fd=fd, name=name, bytes=st.st_size, sha256=entry['sha256'],
                               inode=st.st_ino, device=st.st_dev, mtime_ns=st.st_mtime_ns))
        for item in opened:
            st = os.fstat(item['fd'])
            require((st.st_size, st.st_mtime_ns) == (item['bytes'], item['mtime_ns']), 'Native changed during descriptor acquisition')
        require(os.fstat(mfd).st_mtime_ns == mst.st_mtime_ns, 'Manifest changed during acquisition')
        return dict(job_id=job, step=step, manifest=manifest, manifest_raw=raw, manifest_sha256=expected_sha,
                    latest_raw=latest_raw, files=opened, fds=fds, directory_fd=directory_fd,
                    total_bytes=sum(i['bytes'] for i in opened) + len(raw), source_path=str(path.parent))
    except (FileNotFoundError, NotADirectoryError) as exc:
        for fd in fds:
            os.close(fd)
        if directory_fd is not None:
            os.close(directory_fd)
        raise Race(str(exc)) from exc
    except BaseException:
        for fd in fds:
            os.close(fd)
        if directory_fd is not None:
            os.close(directory_fd)
        raise


def close_pin(snapshot):
    for fd in snapshot.pop('fds', []):
        os.close(fd)
    fd = snapshot.pop('directory_fd', None)
    if fd is not None:
        os.close(fd)


def free_bytes(path):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize


def record_metadata(target, job):
    run = SOURCE / 'runs' / ('tree-current200-continue-' + job)
    package = SOURCE / 'instruct300/candidates' / ('current200-retained18-speed-v14-ebef786' if job == '3341' else 'current200-retained18-speed-v13-6227108')
    cfg = read_small(run / 'resolved.yaml')
    require(sha(cfg) == (CONFIG if job == '3341' else '98a88882c09d8bbba89908b04d1f72930b4fccccc2bbcb5b6eb6886584655f3b'), 'Resolved config SHA mismatch')
    split = read_small(SOURCE / 'instruct300/common-data/manifest.json')
    require(sha(split) == DATA, 'Split manifest SHA mismatch')
    package_manifest = json.loads(read_small(package / 'manifest.json'))
    require(package_manifest['code_commit'] == (CODE if job == '3341' else '62271081ba1bb74d8fc3f4958d623758efa74fa5'), 'Source package identity mismatch')
    allocation_raw = read_small(run / 'allocation-identity.json')
    allocation = json.loads(allocation_raw)
    require(str(allocation['job_id']) == job and allocation['node_list'] == 'h100-10' and allocation['cluster_name'] == 'wuhan-h100', 'Scheduler allocation identity mismatch')
    launch = json.loads(read_small(run / 'launch-manifest.json'))
    require(launch['allocation_identity_sha256'] == sha(allocation_raw), 'Launch allocation SHA mismatch')
    candidates = [run / n for n in ['resolved.yaml', 'input-config.yaml', 'launch-manifest.json', 'allocation-identity.json', 'allocation.txt']]
    candidates += [SOURCE / 'instruct300/common-data/manifest.json', SOURCE / 'instruct300/common-data/exclusions-audit.json']
    candidates += list((SOURCE / '.node-migration').glob('*.json'))
    for directory in [package, run / 'output']:
        for base, dirs, files in os.walk(directory, followlinks=False):
            dirs[:] = [n for n in dirs if n not in {'.git', '__pycache__', 'cache', 'tmp', 'ray'} and not (Path(base) / n).is_symlink()]
            for name in files:
                p = Path(base) / name
                if p.suffix in SUFFIXES and not p.is_symlink():
                    candidates.append(p)
    candidates += [p for p in run.iterdir() if p.is_file() and p.suffix in SUFFIXES and not p.is_symlink()]
    for p in (SOURCE / 'instruct300/node-move-logs').glob(job + '.*'):
        candidates.append(p)
    inventory, used = [], 0
    for p in sorted(set(candidates)):
        if not p.is_file() or p.is_symlink():
            continue
        if p.stat().st_size > SMALL_LIMIT:
            inventory.append(dict(source=str(p), status='SKIPPED_SIZE_LIMIT'))
            continue
        try:
            raw = read_small(p)
        except (OSError, RuntimeError) as exc:
            inventory.append(dict(source=str(p), status='LIVE_READ_UNSTABLE', error=str(exc)))
            continue
        require(used + len(raw) <= META_LIMIT, 'Bounded metadata budget exceeded')
        rel = p.relative_to(SOURCE)
        dst = target / 'metadata' / rel
        write_bytes(dst, raw)
        used += len(raw)
        inventory.append(dict(source=str(p), destination=str(dst.relative_to(target)), bytes=len(raw), sha256=sha(raw),
                              captured_at=now(), source_immutable_claim=False, byte_range=[0, len(raw)]))
    atomic_json(target / 'metadata-inventory.json', dict(captured_at=now(), files=inventory, bytes=used,
                live_files_are_labelled_byte_range_snapshots=True, source_and_split_and_config_preserved=True))
    return dict(files=len(inventory), bytes=used)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--request-id', required=True)
    args = parser.parse_args()
    require(args.execute and re.fullmatch('[a-z0-9-]{8,90}', args.request_id), 'Explicit execute and safe request id required')
    require(socket.gethostname() == 'h100-10', 'Wrong host')
    require(ALIAS.resolve() == SOURCE and SOURCE.resolve() == SOURCE, 'Source root alias changed')
    require(SHARED.resolve() == SHARED and SHARED.is_dir(), 'Shared own namespace changed')
    mount = command(['findmnt', '-n', '-T', str(SHARED), '-o', 'SOURCE,FSTYPE,TARGET'])
    require(mount.split()[:2] == ['10.32.0.11:/dataset/from_h100-9', 'nfs4'], 'NFS11 mount source/type changed')
    os.nice(max(0, 19 - os.getpriority(os.PRIO_PROCESS, 0)))
    subprocess.check_call(['ionice', '-c', '3', '-p', str(os.getpid())], timeout=10)
    # One ordinary CPU thread; no torch, GPU, executor, scheduler mutation or training CPU affinity changes.
    CONTROL.mkdir(exist_ok=True)
    lock = (CONTROL / 'native-once.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    request = CONTROL / (args.request_id + '.request.json')
    require(not request.exists(), 'Request already journalled: inspect existing PID/status/receipt; do not duplicate')
    state = dict(request_id=args.request_id, pid=os.getpid(), host=socket.gethostname(), started_at=now(),
                 boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                 proc_start_ticks=Path('/proc/self/stat').read_text().split()[21], entry_sha256=sha(Path(__file__).read_bytes()),
                 source_root=str(SOURCE), destination_root=str(DEST), mount=mount, shared_provider_release_date='2026-10-21',
                 source_preserved=True, training_unchanged=True, status='STARTING', bytes_copied=0, bytes_readback=0)
    with request.open('x') as f:
        json.dump(state, f, indent=2); f.flush(); os.fsync(f.fileno())
    status_path = CONTROL / (args.request_id + '.status.json')
    def update(**values):
        state.update(values, observed_at=now())
        atomic_json(status_path, state)
        print(json.dumps(state, sort_keys=True), flush=True)
    snap = None
    try:
        DEST.mkdir(exist_ok=True)
        require(free_bytes(SOURCE) > 120 * GIB and free_bytes(DEST) > 80 * GIB, 'Insufficient initial storage reserve')
        probe = DEST / ('.write-probe-' + args.request_id)
        write_bytes(probe, (args.request_id + '\n').encode())
        probe.unlink(); fsync_dir(DEST)
        for attempt in range(3):
            try:
                snap = pin_native('3341')
                break
            except Race as exc:
                update(status='LATEST_OPEN_RACE', attempt=attempt+1, error=str(exc))
        if snap is None:
            snap = pin_native('3339', SOURCE / 'runs/tree-current200-continue-3339/checkpoints/training_state/rolling-completed/global_step_6/manifest.json', FALLBACK_MANIFEST)
        total = snap['total_bytes']
        require(free_bytes(SOURCE) > total * 2 + 16 * GIB and free_bytes(DEST) > total + 16 * GIB, 'Insufficient space for one pinned checkpoint and destination')
        final = DEST / ('native-' + snap['job_id'] + '-step-' + str(snap['step']) + '-' + snap['manifest_sha256'][:16])
        require(not final.exists(), 'Destination already exists: inspect complete receipt, do not replace')
        partial = DEST / ('.partial-' + args.request_id)
        partial.mkdir()
        update(status='PINNED_ALL_NATIVE_FDS', native_job_id=snap['job_id'], native_step=snap['step'], total_bytes=total,
               manifest_sha256=snap['manifest_sha256'], pinned_bytes=total, partial=str(partial), final=str(final),
               native_identity=snap['manifest']['identity'], automatic_resume_supported=snap['manifest'].get('automatic_resume_supported'),
               exact_executor_resume_verified=snap['manifest'].get('exact_executor_resume_verified'))
        write_bytes(partial / 'manifest.json', snap['manifest_raw'])
        if snap['latest_raw']:
            write_bytes(partial / 'source-latest-at-pin.json', snap['latest_raw'])
        meta = record_metadata(partial, snap['job_id'])
        update(status='COPYING_NATIVE', metadata=meta)
        inventory = []
        copied = 0
        last = time.monotonic()
        for item in snap['files']:
            h = hashlib.sha256()
            written = 0
            start = time.monotonic()
            dst = partial / item['name']
            with dst.open('xb') as f:
                while written < item['bytes']:
                    block = os.read(item['fd'], min(CHUNK, item['bytes']-written))
                    require(block, 'Native source truncated: ' + item['name'])
                    f.write(block); h.update(block)
                    written += len(block); copied += len(block)
                    delay = written / RATE - (time.monotonic()-start)
                    if delay > 0:
                        time.sleep(min(delay, 0.1))
                    if time.monotonic()-last >= 20:
                        require(free_bytes(SOURCE) > 64 * GIB and free_bytes(DEST) > (total-copied) + 16 * GIB, 'Storage reserve dropped; abort own copy')
                        update(status='COPYING_NATIVE', current_file=item['name'], bytes_copied=copied, file_bytes_copied=written)
                        last = time.monotonic()
                f.flush(); os.fsync(f.fileno())
            after = os.fstat(item['fd'])
            require((after.st_size, after.st_mtime_ns) == (item['bytes'], item['mtime_ns']), 'Native source changed during copy')
            require(h.hexdigest() == item['sha256'], 'Source stream SHA mismatch: ' + item['name'])
            inventory.append({k:v for k,v in item.items() if k != 'fd'})
            update(status='COPYING_NATIVE', current_file=item['name'], bytes_copied=copied, completed_copy_files=len(inventory))
        close_pin(snap)
        update(status='READBACK_VERIFY', pinned_bytes=0, bytes_copied=copied)
        readback = 0
        for item in inventory:
            h = hashlib.sha256()
            size = 0
            start = time.monotonic()
            with (partial / item['name']).open('rb') as f:
                while True:
                    block = f.read(CHUNK)
                    if not block:
                        break
                    h.update(block); size += len(block); readback += len(block)
                    delay = size / RATE - (time.monotonic()-start)
                    if delay > 0:
                        time.sleep(min(delay, 0.1))
                    if time.monotonic()-last >= 20:
                        update(status='READBACK_VERIFY', current_file=item['name'], bytes_readback=readback)
                        last = time.monotonic()
            require(size == item['bytes'] and h.hexdigest() == item['sha256'], 'Destination full SHA mismatch: ' + item['name'])
            item['full_destination_readback_verified'] = True
            update(status='READBACK_VERIFY', current_file=item['name'], bytes_readback=readback)
        require(sha(read_small(partial / 'manifest.json')) == snap['manifest_sha256'], 'Original manifest was changed')
        atomic_json(partial / 'native-file-inventory.json', dict(files=inventory, manifest_sha256=snap['manifest_sha256'], original_manifest_preserved=True))
        receipt = dict(state, status='COMPLETE_SHARED_NATIVE_BACKUP', completed_at=now(),
                       all_8_ranks_and_driver_stream_and_readback_sha256_verified=True,
                       manifest_sha256=snap['manifest_sha256'], metadata=meta, bytes_copied=copied, bytes_readback=readback,
                       source_checkpoint=snap['source_path'], destination=str(final), pinned_bytes=0)
        atomic_json(partial / '.complete', receipt)
        fsync_dir(partial)
        os.rename(partial, final)
        fsync_dir(DEST)
        require(json.loads(read_small(final / '.complete'))['manifest_sha256'] == snap['manifest_sha256'], 'Published receipt readback failed')
        atomic_json(DEST / (args.request_id + '.complete.json'), receipt)
        update(status='COMPLETE_SHARED_NATIVE_BACKUP', completed_at=receipt['completed_at'], bytes_readback=readback, receipt=str(final / '.complete'))
    except BaseException as exc:
        if snap:
            close_pin(snap)
        update(status='FAILED_OWN_BACKUP_ONLY', pinned_bytes=0, error=repr(exc))
        raise


if __name__ == '__main__':
    main()
