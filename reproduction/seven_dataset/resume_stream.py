"""Verified append-only transport; partial files survive every failure."""
import fcntl
import hashlib
import os
import stat

CHUNK = 1024 * 1024


def inspect_prefix(path, size):
    with open(path, 'rb') as f:
        before = os.fstat(f.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 <= size <= before.st_size:
            raise ValueError('Invalid regular-file prefix')
        h = hashlib.sha256()
        remaining = size
        while remaining:
            data = f.read(min(CHUNK, remaining))
            if not data:
                raise ValueError('Truncated prefix')
            h.update(data)
            remaining -= len(data)
        after = os.fstat(f.fileno())
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('File changed during prefix verification')
        return dict(inode=before.st_ino, device=before.st_dev, size=before.st_size,
                    mtime_ns=before.st_mtime_ns, prefix_bytes=size, prefix_sha256=h.hexdigest())


def send_suffix(path, expected, output):
    actual = inspect_prefix(path, expected['prefix_bytes'])
    if actual != expected:
        raise ValueError('Source identity or prefix changed')
    with open(path, 'rb') as f:
        s = os.fstat(f.fileno())
        if (s.st_ino, s.st_dev, s.st_size, s.st_mtime_ns) != tuple(expected[k] for k in ('inode', 'device', 'size', 'mtime_ns')):
            raise ValueError('Source changed before send')
        f.seek(expected['prefix_bytes'])
        while True:
            data = f.read(CHUNK)
            if not data:
                break
            output.write(data)
        output.flush()


def receive_suffix(path, expected, total, full_sha256, incoming):
    # Never create, replace, truncate, or remove the target. Lock through verification.
    fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
    with os.fdopen(fd, 'r+b', buffering=0) as f:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        s = os.fstat(fd)
        if not stat.S_ISREG(s.st_mode) or (s.st_ino, s.st_dev, s.st_size, s.st_mtime_ns) != tuple(expected[k] for k in ('inode', 'device', 'size', 'mtime_ns')):
            raise ValueError('Target identity changed')
        if expected['prefix_bytes'] != s.st_size or not s.st_size < total:
            raise ValueError('Invalid resume offset')
        h = hashlib.sha256()
        while True:
            data = f.read(CHUNK)
            if not data:
                break
            h.update(data)
        if h.hexdigest() != expected['prefix_sha256']:
            raise ValueError('Target prefix mismatch')
        remaining = total - s.st_size
        try:
            while remaining:
                data = incoming.read(min(CHUNK, remaining))
                if not data:
                    raise ValueError('Incomplete suffix; preserve partial')
                view = memoryview(data)
                while view:
                    written = f.write(view)
                    if not written:
                        raise OSError('No write progress')
                    view = view[written:]
                remaining -= len(data)
            if incoming.read(1):
                raise ValueError('Excess suffix; preserve target for inspection')
        finally:
            os.fsync(fd)
        f.seek(0)
        h = hashlib.sha256()
        while True:
            data = f.read(CHUNK)
            if not data:
                break
            h.update(data)
        if h.hexdigest() != full_sha256:
            raise ValueError('Full target SHA mismatch; preserve evidence')
        return dict(bytes=os.fstat(fd).st_size, sha256=h.hexdigest(), status='ACTUAL_FULL_SHA_VERIFIED')
