"""Same-descriptor byte and stat binding for already-frozen local probe files."""
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path

import torch

from .probability_probe_capture import require

STAT_KEYS=('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')


def descriptor_sha(stream):
    stream.seek(0);digest=hashlib.sha256()
    for chunk in iter(lambda:stream.read(8*1024**2),b''):digest.update(chunk)
    return digest.hexdigest()


def stat_identity(value):return tuple(getattr(value,key) for key in STAT_KEYS)


@contextmanager
def bound_descriptor(path,expected_sha):
    """The consumer reads the verified inode, never a separate pathname open.

    The final path must still identify the same unchanged file. Nanosecond ctime
    catches replacing and restoring its name or in-place mutation/restore. Full
    same-FD SHA is rechecked after consumption. This is not a general privileged
    malicious-kernel/undetectable-storage-tampering claim.
    """
    path=Path(path)
    require(not path.is_symlink(),'Frozen probe file must not be a symlink')
    fd=os.open(str(path),os.O_RDONLY|getattr(os,'O_NOFOLLOW',0))
    with os.fdopen(fd,'rb') as stream:
        before=stat_identity(os.fstat(stream.fileno()))
        require(descriptor_sha(stream)==expected_sha,'Frozen probe descriptor SHA changed: '+str(path))
        require(stat_identity(os.fstat(stream.fileno()))==before==stat_identity(path.stat()),
            'Frozen probe file changed during initial descriptor hashing: '+str(path))
        stream.seek(0)
        yield stream
        require(stat_identity(os.fstat(stream.fileno()))==before==stat_identity(path.stat()),
            'Frozen probe file changed during descriptor consumption: '+str(path))
        require(descriptor_sha(stream)==expected_sha,'Frozen probe descriptor bytes changed during consumption: '+str(path))
        require(stat_identity(os.fstat(stream.fileno()))==before==stat_identity(path.stat()),
            'Frozen probe file changed during final descriptor hashing: '+str(path))


def load_tensor_verified(path,expected_sha):
    with bound_descriptor(path,expected_sha) as stream:
        value=torch.load(stream,map_location='cpu',weights_only=False)
    return value


@contextmanager
def safetensor_verified(path,expected_sha,safe_open):
    """Linux same-inode safetensors memory mapping via held /proc/self/fd/N.

    This does not copy a HF shard. The actual runner is h100-9 Linux-only; CPU
    tests use an explicit synthetic reader of that descriptor path.
    """
    with bound_descriptor(path,expected_sha) as stream:
        descriptor_path='/proc/self/fd/'+str(stream.fileno())
        with safe_open(descriptor_path,framework='pt',device='cpu') as shard:yield shard
