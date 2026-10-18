"""Four verified immutable ranges; append to an inspected, preserved prefix."""
import hashlib
import json
import os
from pathlib import Path
from .resume_stream import inspect_prefix, receive_suffix

BLOCK = 1024 * 1024


def fingerprint(path):
    s = os.stat(path)
    return [s.st_ino, s.st_dev, s.st_size, s.st_mtime_ns]


def manifest(path, prefix, total, expected_sha):
    before = fingerprint(path)
    assert before[2] == total and 0 <= prefix < total
    boundaries = [prefix + (total-prefix)*i//4 for i in range(5)]
    full, pre = hashlib.sha256(), hashlib.sha256()
    ranges = []
    with open(path, 'rb') as f:
        remain = prefix
        while remain:
            b = f.read(min(BLOCK, remain)); assert b
            full.update(b); pre.update(b); remain -= len(b)
        for i in range(4):
            length = boundaries[i+1]-boundaries[i]
            h = hashlib.sha256(); remain = length
            while remain:
                b = f.read(min(BLOCK, remain)); assert b
                full.update(b); h.update(b); remain -= len(b)
            ranges.append(dict(index=i, offset=boundaries[i], length=length,
                               sha256=h.hexdigest()))
        assert not f.read(1)
    assert fingerprint(path) == before and full.hexdigest() == expected_sha
    return dict(source_fingerprint=before, total=total, sha256=expected_sha,
                prefix_bytes=prefix, prefix_sha256=pre.hexdigest(), ranges=ranges)


def send(path, expected, item, output):
    assert fingerprint(path) == expected
    with open(path, 'rb') as f:
        f.seek(item['offset']); remaining = item['length']; h = hashlib.sha256()
        while remaining:
            b = f.read(min(BLOCK, remaining)); assert b
            h.update(b); output.write(b); remaining -= len(b)
        output.flush()
    assert fingerprint(path) == expected and h.hexdigest() == item['sha256']


def receive(path, item, incoming):
    # New range files are exclusive. Any failed partial remains for inspection.
    h = hashlib.sha256(); remaining = item['length']
    with open(path, 'xb', buffering=0) as f:
        try:
            while remaining:
                b = incoming.read(min(BLOCK, remaining))
                if not b: raise ValueError('Incomplete range; preserve partial')
                assert len(b) <= remaining
                h.update(b); view = memoryview(b)
                while view:
                    n = f.write(view); assert n; view = view[n:]
                remaining -= len(b)
            assert not incoming.read(1), 'Excess range bytes'
        finally:
            os.fsync(f.fileno())
    assert h.hexdigest() == item['sha256'], 'Range SHA mismatch; preserve file'
    result = dict(item, status='ACTUAL_RANGE_SHA_VERIFIED', path=str(path))
    with open(str(path)+'.verified.json', 'x') as f: json.dump(result, f)
    return result


class PartsReader:
    def __init__(self, paths):
        self.paths = iter(paths); self.current = None

    def read(self, n):
        while True:
            if self.current is None:
                try: self.current = open(next(self.paths), 'rb')
                except StopIteration: return b''
            b = self.current.read(n)
            if b: return b
            self.current.close(); self.current = None

    def close(self):
        if self.current is not None: self.current.close()


def combine(path, prefix, proof, paths):
    assert len(paths) == len(proof['ranges']) == 4
    assert prefix['prefix_bytes'] == proof['prefix_bytes']
    assert prefix['prefix_sha256'] == proof['prefix_sha256']
    offset = prefix['prefix_bytes']
    for item, part in zip(proof['ranges'], paths):
        assert item['offset'] == offset
        record = json.loads(Path(str(part)+'.verified.json').read_text())
        assert all(record[k] == item[k] for k in item)
        actual = inspect_prefix(part, item['length'])
        assert actual['size'] == item['length']
        assert actual['prefix_sha256'] == item['sha256']
        offset += item['length']
    assert offset == proof['total']
    incoming = PartsReader(paths)
    try:
        return receive_suffix(path, prefix, proof['total'], proof['sha256'], incoming)
    finally:
        incoming.close()
