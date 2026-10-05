"""Journal the private corpus Arrow cache separately from original asset staging."""
import json
import os
import time
import shutil
from pathlib import Path
from reproduction.a800.owned_processes import identity
from reproduction.common8.stage_retrieval import atomic, assert_quiescent, ASSETS


def capacity_floor():
    # Allow simultaneous temporary and final Arrow shards plus metadata margin.
    # This is a launch allowance, not evidence of the eventual measured peak.
    return 2 * ASSETS['wiki-18.jsonl']['size'] + 2**30


def admit_capacity(base, root):
    path = base / 'logs/corpus-cache-claim.json'
    claim = json.loads(path.read_text())
    assert Path(claim['root']) == root and claim['phase'] == 'CREATED'
    assert not root.is_symlink() and root.stat().st_uid == claim['uid'] == os.getuid()
    assert (root.stat().st_dev, root.stat().st_ino) == (claim['root_device'], claim['root_inode'])
    creator = identity(os.getpid())
    assert all(creator[k] == claim['creator'][k] for k in ('pid', 'start_ticks', 'sid', 'uid'))
    available = shutil.disk_usage(root).free
    memory = {line.split(':')[0]: int(line.split()[1])*1024
              for line in Path('/proc/meminfo').read_text().splitlines() if ':' in line}
    proof = dict(required_capacity_bytes=capacity_floor(), actual_tmpfs_free_bytes=available,
                 actual_host_memory_available_bytes=memory['MemAvailable'], checked_unix=time.time(),
                 scope='Arrow temporary+final launch allowance; actual cache peak remains sampled')
    claim['capacity_admission'] = proof; atomic(path, claim)
    assert available >= capacity_floor(), 'Insufficient actual tmpfs space for corpus Arrow construction'
    assert memory['MemAvailable'] >= capacity_floor(), 'Insufficient available host RAM for corpus cache'
    return proof


def create(base, run, job):
    stage = json.loads((base / 'logs/retrieval-staging-claim.json').read_text())
    assert stage['run_id'] == run['run_id'] and stage['slurm_job_id'] == job['JobId']
    assert stage['phase'] == 'STAGED' and stage['uid'] == os.getuid()
    parent = Path(stage['root'])
    expected = Path('/dev/shm') / ('search-g1-common8-' + run['run_id'] + '-' + job['JobId'])
    assert parent == expected and not parent.is_symlink()
    assert (parent.stat().st_dev, parent.stat().st_ino) == (stage['root_device'], stage['root_inode'])
    assert parent.stat().st_uid == stage['uid']
    root = parent / 'datasets-cache'
    assert not root.exists() and not root.is_symlink()
    claim = dict(run_id=run['run_id'], slurm_job_id=job['JobId'], uid=os.getuid(), root=str(root),
                 parent_device=parent.stat().st_dev, parent_inode=parent.stat().st_ino,
                 creator=identity(os.getpid()), phase='RESERVED', created_unix=time.time())
    path = base / 'logs/corpus-cache-claim.json'
    # Reserve before mkdir; only this new empty namespace may later be removed.
    with path.open('x') as f:
        json.dump(claim, f, indent=2); f.flush(); os.fsync(f.fileno())
    fd = os.open(str(path.parent), os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)
    root.mkdir(mode=0o700)
    claim.update(phase='CREATED', root_device=root.stat().st_dev, root_inode=root.stat().st_ino)
    atomic(path, claim)
    return root


def cleanup(base):
    run = json.loads((base / 'logs/run-manifest.json').read_text())
    stage = json.loads((base / 'logs/retrieval-staging-claim.json').read_text())
    path = base / 'logs/corpus-cache-claim.json'
    claim = json.loads(path.read_text())
    assert run['profile'] == 'common8'
    assert claim['run_id'] == stage['run_id'] == run['run_id']
    assert claim['slurm_job_id'] == stage['slurm_job_id'] and claim['uid'] == os.getuid()
    parent = Path('/dev/shm') / ('search-g1-common8-' + run['run_id'] + '-' + claim['slurm_job_id'])
    root = Path(claim['root']); assert root == parent / 'datasets-cache'
    assert not parent.is_symlink() and not root.is_symlink()
    assert_quiescent(base, stage)
    try: actual = identity(claim['creator']['pid'])
    except (FileNotFoundError, ProcessLookupError): actual = None
    assert actual is None or actual['state'] == 'Z' or actual['start_ticks'] != claim['creator']['start_ticks'], 'Cache creator remains live'
    sizes = claim.get('removed_file_sizes', {})
    if root.exists():
        assert parent.stat().st_uid == root.stat().st_uid == claim['uid']
        assert (parent.stat().st_dev, parent.stat().st_ino) == (claim['parent_device'], claim['parent_inode'])
        if 'root_inode' in claim:
            assert (root.stat().st_dev, root.stat().st_ino) == (claim['root_device'], claim['root_inode'])
        entries = list(root.rglob('*'))
        assert all(not p.is_symlink() and p.stat().st_uid == claim['uid']
                   and (p.is_file() or p.is_dir()) for p in entries), 'Unowned/symlink/special cache entry'
        sizes.update({str(p.relative_to(root)): p.stat().st_size for p in entries if p.is_file()})
        claim.update(phase='CLEANING', removed_file_sizes=sizes); atomic(path, claim)
        for p in sorted(entries, key=lambda p: len(p.parts), reverse=True):
            if p.is_dir(): p.rmdir()
            else: p.unlink()
        root.rmdir()
    claim.update(phase='CLEANED', removed_file_sizes=sizes); atomic(path, claim)
    proof = dict(run_id=run['run_id'], slurm_job_id=claim['slurm_job_id'], cleaned=True,
                 root=str(root), removed_file_sizes=sizes, checked_unix=time.time())
    atomic(base / 'logs/corpus-cache-cleanup.json', proof)
    return proof
