"""Reuse a digest within one verifier process only while file identity is unchanged."""
import hashlib
import os
import stat
from pathlib import Path

class VerifiedDigestCache:
    def __init__(self):
        self.records = {}
        self.bytes_read = 0
        self.cache_hits = 0

    @staticmethod
    def identity(p):
        s = p.lstat()
        if not stat.S_ISREG(s.st_mode) or p.resolve(strict=True) != p:
            raise ValueError('Canonical regular file required')
        return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns

    def digest(self, path):
        p = Path(path).absolute()
        before = self.identity(p)
        key = str(p)
        if key in self.records:
            original, value = self.records[key]
            if before != original:
                raise ValueError('Verified input changed since hashing: ' + key)
            self.cache_hits += 1
            return value
        h = hashlib.sha256()
        with p.open('rb') as f:
            s = os.fstat(f.fileno())
            opened = (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
            if opened != before:
                raise ValueError('Input changed while opening: ' + key)
            for b in iter(lambda: f.read(1048576), b''):
                h.update(b)
                self.bytes_read += len(b)
        if self.identity(p) != before:
            raise ValueError('Input changed while hashing: ' + key)
        value = h.hexdigest()
        self.records[key] = (before, value)
        return value

    def verify_unchanged(self):
        for name, (identity, _) in self.records.items():
            if self.identity(Path(name)) != identity:
                raise ValueError('Verified input changed before publication: ' + name)

    def exact_file(self, entry, root):
        p = Path(entry['path'])
        if root not in p.parents or type(entry.get('bytes')) is not int:
            raise ValueError('Original member scope/size invalid')
        if self.identity(p)[2] != entry['bytes'] or self.digest(p) != entry['sha256']:
            raise ValueError('Original member bytes/SHA differ')
        return p
