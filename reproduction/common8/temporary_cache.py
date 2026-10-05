"""Journal the private corpus Arrow cache separately from original asset staging."""
import json
import os
import time
import shutil
import hashlib
import stat
import tarfile
from pathlib import Path
from reproduction.a800.owned_processes import identity
from reproduction.common8.stage_retrieval import atomic, assert_quiescent, ASSETS


# The immutable wiki-18.jsonl asset is a TAR, despite its filename. Never restore
# its archived uid/gid into the private cache: read only its one regular member.
CORPUS_MEMBER = dict(name='data00/jiajie_jin/flashrag_indexes/wiki_dpr_100w/wiki_dump.jsonl',
                     size=14393573105, uid=1012, gid=1012, mode=0o664, type='0', linkname='')


def capacity_floor():
    # Retain the original member plus simultaneous temporary/final Arrow shards
    # and metadata margin. TAR size conservatively bounds the member's size.
    # This is a launch allowance, not evidence of the eventual measured peak.
    return 3 * ASSETS['wiki-18.jsonl']['size'] + 2**30


def _cache_parent(run_id, job_id):
    return Path('/dev/shm') / ('search-g1-common8-' + run_id + '-' + job_id)


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
                 scope='Retained JSONL+Arrow temporary+final launch allowance; actual cache peak remains sampled')
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


def extract_corpus(base, root, archive):
    """Stream the verified original member into an exclusively created own file.

    This changes filesystem ownership only by creating a new file as the caller;
    it does not change corpus bytes/order or grant any model/training admission.
    Failed partial files remain in the durably claimed cache for strict cleanup.
    """
    base, root, archive = Path(base), Path(root), Path(archive)
    path = base / 'logs/corpus-cache-claim.json'
    claim = json.loads(path.read_text())
    stage = json.loads((base / 'logs/retrieval-staging-claim.json').read_text())
    run = json.loads((base / 'logs/run-manifest.json').read_text())
    assert run['profile'] == 'common8'
    assert claim['run_id'] == stage['run_id'] == run['run_id']
    assert claim['slurm_job_id'] == stage['slurm_job_id']
    parent = Path(stage['root'])
    assert parent == _cache_parent(run['run_id'], claim['slurm_job_id'])
    assert root == Path(claim['root']) == parent / 'datasets-cache'
    assert archive == parent / 'wiki-18.jsonl' and stage['phase'] == 'STAGED'
    assert claim['phase'] == 'CREATED' and 'corpus_extraction' not in claim
    assert not parent.is_symlink() and not root.is_symlink() and not archive.is_symlink()
    assert parent.stat().st_uid == root.stat().st_uid == stage['uid'] == claim['uid'] == os.getuid()
    assert (parent.stat().st_dev, parent.stat().st_ino) == (claim['parent_device'], claim['parent_inode'])
    assert (parent.stat().st_dev, parent.stat().st_ino) == (stage['root_device'], stage['root_inode'])
    assert (root.stat().st_dev, root.stat().st_ino) == (claim['root_device'], claim['root_inode'])
    actual = identity(os.getpid())
    assert all(actual[k] == claim['creator'][k] for k in ('pid', 'start_ticks', 'sid', 'uid'))
    # Capacity must be admitted before opening an output or extracting anything.
    claim['capacity_admission'] = admit_capacity(base, root)
    partial, target = root / 'wiki_dump.jsonl.partial', root / 'wiki_dump.jsonl'
    assert not partial.exists() and not partial.is_symlink() and not target.exists() and not target.is_symlink()
    source_fd = os.open(archive, os.O_RDONLY | os.O_NOFOLLOW)
    record = None
    try:
        with os.fdopen(source_fd, 'rb') as source:
            source_stat = os.fstat(source.fileno())
            assert stat.S_ISREG(source_stat.st_mode) and source_stat.st_uid == claim['uid']
            assert source_stat.st_size == ASSETS['wiki-18.jsonl']['size']
            assert source_stat.st_nlink == 1
            archive_hash = hashlib.sha256()
            while chunk := source.read(8 * 1024 * 1024): archive_hash.update(chunk)
            assert archive_hash.hexdigest() == ASSETS['wiki-18.jsonl']['sha256'], 'Original corpus TAR differs'
            source.seek(0)
            with tarfile.open(fileobj=source, mode='r:') as tar:
                members = tar.getmembers()
                assert len(members) == 1 and members[0].isfile(), 'Corpus TAR must have exactly one regular member'
                member = members[0]
                header = dict(name=member.name, size=member.size, uid=member.uid, gid=member.gid,
                              mode=member.mode, type=member.type.decode('ascii'), linkname=member.linkname)
                assert header == CORPUS_MEMBER, 'Original corpus TAR member header differs'
                record = dict(run_id=run['run_id'], slurm_job_id=claim['slurm_job_id'], uid=claim['uid'],
                              archive=str(archive), archive_sha256=archive_hash.hexdigest(),
                              archive_size=source_stat.st_size, archive_device=source_stat.st_dev,
                              archive_inode=source_stat.st_ino, member=header, partial=str(partial),
                              target=str(target), phase='RESERVED', created_unix=time.time())
                claim['corpus_extraction'] = record; atomic(path, claim)
                output_fd = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                with os.fdopen(output_fd, 'wb') as output:
                    output_stat = os.fstat(output.fileno())
                    assert stat.S_ISREG(output_stat.st_mode) and output_stat.st_uid == claim['uid']
                    assert output_stat.st_dev == claim['root_device'] and output_stat.st_nlink == 1
                    record.update(phase='WRITING', output_device=output_stat.st_dev, output_inode=output_stat.st_ino)
                    atomic(path, claim)
                    content_hash = hashlib.sha256(); count = 0
                    with tar.extractfile(member) as content:
                        while chunk := content.read(8 * 1024 * 1024):
                            output.write(chunk); content_hash.update(chunk); count += len(chunk)
                            assert count <= member.size, 'Corpus member exceeds its original size'
                    assert count == member.size, 'Incomplete original corpus member'
                    output.flush(); os.fsync(output.fileno())
                    output_stat = os.fstat(output.fileno())
                    assert output_stat.st_size == count and output_stat.st_uid == claim['uid']
                current_source = os.fstat(source.fileno())
                assert (current_source.st_size, current_source.st_mtime_ns, current_source.st_ctime_ns) == (
                    source_stat.st_size, source_stat.st_mtime_ns, source_stat.st_ctime_ns), 'Corpus TAR changed while reading'
                assert (archive.lstat().st_dev, archive.lstat().st_ino) == (source_stat.st_dev, source_stat.st_ino)
                assert (partial.lstat().st_dev, partial.lstat().st_ino) == (record['output_device'], record['output_inode'])
                # Exclusive publication cannot replace an existing caller's file.
                os.link(partial, target, follow_symlinks=False); partial.unlink()
                final_stat = target.lstat()
                assert stat.S_ISREG(final_stat.st_mode) and final_stat.st_uid == claim['uid'] and final_stat.st_nlink == 1
                assert (final_stat.st_dev, final_stat.st_ino, final_stat.st_size) == (
                    record['output_device'], record['output_inode'], count)
                fd = os.open(root, os.O_RDONLY)
                try: os.fsync(fd)
                finally: os.close(fd)
                record.update(phase='READY', content_sha256=content_hash.hexdigest(), content_size=count,
                              completed_unix=time.time(), scope='original corpus bytes; private CPU filesystem extraction only')
                atomic(path, claim)
                atomic(base / 'logs/corpus-extraction.json', record)
                return target
    except BaseException as error:
        if record is not None:
            record.update(phase='FAILED', error=repr(error), failed_unix=time.time())
            try: atomic(path, claim)
            except OSError: pass  # The durable reservation still owns the partial.
        raise


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
