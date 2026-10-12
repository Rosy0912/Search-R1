"""Streaming byte seals for fixed evidence after the owned trainer exits.

The sampler, process inventory, cleanup and pipeline status remain writable
until the pipeline finishes, so they have separate actual post-exit audits.
"""
import hashlib
import os
from pathlib import Path

DYNAMIC = {'pipeline-status.json','process-ledger.json','corpus-cache-claim.json',
           'corpus-cache-cleanup.json','cleanup-status.json','full-run-acceptance.json',
           'actor-probe-final-cleanup-acceptance.json'}


def sealed_digest(base,path,*,prefix_bytes=None):
    base=Path(base);path=Path(path)
    assert path.is_relative_to(base) and path == path.resolve() and path.is_file()
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        before=os.fstat(fd);assert before.st_uid == os.getuid()
        assert prefix_bytes is None or type(prefix_bytes) is int and 0<prefix_bytes<=before.st_size
        h=hashlib.sha256();size=0
        with os.fdopen(fd,'rb',closefd=False) as stream:
            target=before.st_size if prefix_bytes is None else prefix_bytes
            while size<target:
                block=stream.read(min(1024**2,target-size));assert block, 'Sealed evidence was truncated'
                h.update(block);size+=len(block)
        after=os.fstat(fd)
        if prefix_bytes is None:
            assert (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(
                after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
        else:assert after.st_size>=prefix_bytes
        current=path.stat();assert (current.st_dev,current.st_ino)==(before.st_dev,before.st_ino)
        assert size==target
    finally:os.close(fd)
    return dict(bytes=size,sha256=h.hexdigest())


def fixed_logs(base,run):
    base=Path(base)
    paths={path for path in (base/'logs').rglob('*') if path.is_file()
        and path.suffix in ('.json','.jsonl') and path.name not in DYNAMIC
        and not path.name.startswith('gpu-samples-')}
    paths.add(Path(run['train_log']))
    return {path.relative_to(base).as_posix():sealed_digest(base,path) for path in sorted(paths)}


def verify_fixed_logs(base,run,saved):
    assert saved and fixed_logs(base,run)==saved, 'Fixed training/scalar evidence bytes changed after acceptance'
    return len(saved)
