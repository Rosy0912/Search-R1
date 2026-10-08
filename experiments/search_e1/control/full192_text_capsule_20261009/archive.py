#!/usr/bin/env python3
"""Lossless Mac-only packaging of the already accepted closed debug text files.

This operational entry never contacts a node, imports model code, or deletes files.
Original NPY files, original node assets, and accepted inventory remain separate.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tarfile
import time


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def record(path, value):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=True, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def checked_file(root, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or not relative.parts or any(x in ('', '.', '..') for x in relative.parts):
        raise ValueError('unsafe member name')
    path = root
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError('symlink in source path')
    mode = path.stat().st_mode
    if not stat.S_ISREG(mode):
        raise ValueError('source is not a regular file')
    return path


def identity(path):
    s = path.stat()
    return dict(device=s.st_dev, inode=s.st_ino, size=s.st_size, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


class HashedReader:
    def __init__(self, stream):
        self.stream = stream
        self.hash = hashlib.sha256()
        self.count = 0

    def read(self, size=-1):
        data = self.stream.read(size)
        self.hash.update(data)
        self.count += len(data)
        return data


def verify(archive, rows):
    """Independently read every decompressed member without a second disk copy."""
    seen = set()
    with archive.open('rb') as compressed:
        with gzip.GzipFile(fileobj=compressed, mode='rb') as stream:
            with tarfile.open(fileobj=stream, mode='r|') as tar:
                for member in tar:
                    if not member.isreg() or member.name not in rows or member.name in seen:
                        raise ValueError('unexpected, duplicate, or non-file archive member')
                    expected = rows[member.name]
                    if member.size != expected['bytes']:
                        raise ValueError('archive member size mismatch')
                    source = tar.extractfile(member)
                    h = hashlib.sha256()
                    count = 0
                    for data in iter(lambda: source.read(1024 * 1024), b''):
                        h.update(data)
                        count += len(data)
                    if count != expected['bytes'] or h.hexdigest() != expected['sha256']:
                        raise ValueError('decompressed member SHA mismatch')
                    seen.add(member.name)
                # Drain the SAME tar wrapper: it owns any already-prefetched bytes.
                # Reading only the underlying gzip object would discard that buffer.
                while True:
                    trailing = tar.fileobj.read(1024 * 1024)
                    if not trailing:
                        break
                    if any(trailing):
                        raise ValueError('nonzero trailing archive data')
            # Consume through the gzip footer (CRC/truncation), checking every block.
            while True:
                trailing = stream.read(1024 * 1024)
                if not trailing:
                    break
                if any(trailing):
                    raise ValueError('nonzero data after tar wrapper')
    if seen != set(rows):
        raise ValueError('missing archive members')
    return len(seen)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--entry', required=True)
    parser.add_argument('--entry-sha256', required=True)
    args = parser.parse_args()
    entry_path = Path(args.entry)
    if digest(entry_path) != args.entry_sha256:
        raise ValueError('entry SHA mismatch')
    entry = json.loads(entry_path.read_text())
    proof_path = Path(entry['accepted_inventory']['path'])
    if digest(proof_path) != entry['accepted_inventory']['sha256']:
        raise ValueError('accepted inventory SHA mismatch')
    proof = json.loads(proof_path.read_text())
    if not proof.get('accepted') or proof['file_count'] != 6603 or proof['bytes'] != 39215210380:
        raise ValueError('incorrect accepted full192 inventory')
    rows = {x['relative_path']: x for x in proof['files'] if PurePosixPath(x['relative_path']).suffix in ('.json', '.jsonl')}
    if len(rows) != 6547 or sum(x['bytes'] for x in rows.values()) != 22419590028:
        raise ValueError('incorrect text subset')
    root = Path(entry['source_root']).resolve(strict=True)
    destination = Path(entry['destination']).resolve()
    destination.mkdir(exist_ok=False)
    # Worst-case gzip can be slightly larger; leave room for the independent CP queue.
    if shutil.disk_usage(destination).free < 22419590028 + 64 * 1024 * 1024 + 5 * 1024**3:
        raise ValueError('insufficient local margin; no source changed')
    record(destination / 'start.json', {'pid': os.getpid(), 'uid': os.getuid(), 'started_unix': time.time(), 'entry_sha256': args.entry_sha256, 'source_count': len(rows), 'source_bytes': 22419590028, 'no_node_io': True, 'source_deletion': False})
    partial = destination / 'full192-text.tar.gz.partial'
    before = {}
    with partial.open('xb') as output:
        with gzip.GzipFile(filename='', fileobj=output, mode='wb', compresslevel=1, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode='w|', format=tarfile.PAX_FORMAT) as tar:
                for index, (name, expected) in enumerate(sorted(rows.items())):
                    path = checked_file(root, name)
                    before[name] = identity(path)
                    if before[name]['size'] != expected['bytes']:
                        raise ValueError('source size changed')
                    info = tarfile.TarInfo(name)
                    info.size = expected['bytes']
                    info.mode = 0o600
                    info.mtime = 0
                    with path.open('rb') as source:
                        reader = HashedReader(source)
                        tar.addfile(info, reader)
                        if reader.count != info.size or reader.hash.hexdigest() != expected['sha256']:
                            raise ValueError('fresh source SHA mismatch')
                        if identity(path) != before[name]:
                            raise ValueError('source metadata changed during read')
                    if index % 100 == 0:
                        record(destination / 'progress.json', {'pid': os.getpid(), 'phase': 'COMPRESSING', 'files': index + 1, 'total': len(rows), 'observed_unix': time.time()})
        output.flush()
        os.fsync(output.fileno())
    record(destination / 'progress.json', {'pid': os.getpid(), 'phase': 'INDEPENDENT_DECOMPRESSED_MEMBER_SHA', 'files': len(rows), 'observed_unix': time.time()})
    member_count = verify(partial, rows)
    for name in rows:
        if identity(checked_file(root, name)) != before[name]:
            raise ValueError('source changed after packaging')
    compressed_sha = digest(partial)
    final = destination / 'full192-text.tar.gz'
    partial.replace(final)
    record(destination / 'source-metadata.json', before)
    record(destination / 'complete-sha-verified.json', {'schema': 'full192-text-lossless-capsule-v1', 'accepted': True, 'ended_unix': time.time(), 'entry_sha256': args.entry_sha256, 'accepted_inventory': entry['accepted_inventory'], 'archive': {'path': str(final), 'bytes': final.stat().st_size, 'sha256': compressed_sha}, 'text_members': member_count, 'text_bytes': 22419590028, 'npy_untouched_count': 56, 'npy_untouched_bytes': 16795620352, 'source_metadata_sha256': digest(destination / 'source-metadata.json'), 'fresh_source_SHA_and_independent_member_SHA': True, 'all_original_text_files_retained': True, 'Node_originals_retained': True, 'no_node_io': True, 'no_model_GPU_or_HTTP': True})
    print(json.dumps({'accepted': True, 'members': member_count, 'compressed_bytes': final.stat().st_size}), flush=True)


if __name__ == '__main__':
    main()
