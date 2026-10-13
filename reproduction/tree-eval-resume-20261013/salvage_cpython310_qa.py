#!/usr/bin/env python3
"""Read append-only _validate qa_rows externally. No ptrace, signals or injection.

Supports stock little-endian 64-bit CPython 3.10 only. Run on the target node;
outputs must remain there. Runtime/frame offsets match Ubuntu Python 3.10
headers. Fail closed on unsupported object layouts or inconsistent snapshots.
"""
import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

KEYS = {'source', 'f1', 'em', 'answer_format', 'format_rate', 'generated_tokens', 'search_calls'}


class Reader:
    def __init__(self, pid):
        self.pid = pid
        self.starttime = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
        exe = os.readlink(f'/proc/{pid}/exe')
        if 'python3.10' not in exe or sys.byteorder != 'little':
            raise RuntimeError('Only 64-bit little-endian stock CPython 3.10 supported')
        maps = Path(f'/proc/{pid}/maps').read_text().splitlines()
        bases = [int(x.split()[0].split('-')[0], 16) for x in maps
                 if x.split()[2] == '00000000' and x.endswith(exe)]
        if len(bases) != 1:
            raise RuntimeError('Ambiguous executable relocation')
        symbols = subprocess.check_output(['nm', '-D', f'/proc/{pid}/exe'], text=True)
        self.types = {}
        for line in symbols.splitlines():
            parts = line.split()
            if len(parts) == 3 and parts[2] in ('_PyRuntime', 'PyList_Type', 'PyTuple_Type', 'PyDict_Type', 'PyUnicode_Type', 'PyFloat_Type', 'PyLong_Type', 'PyCell_Type'):
                self.types[parts[2]] = bases[0] + int(parts[0], 16)
        self.fd = os.open(f'/proc/{pid}/mem', os.O_RDONLY)
        self.string_cache = {}

    def read(self, address, size):
        if not address or size < 0 or size > 32 * 1024 * 1024:
            raise ValueError('Invalid memory request')
        value = os.pread(self.fd, size, address)
        if len(value) != size:
            raise RuntimeError('Short process memory read')
        return value

    def ptr(self, address):
        return struct.unpack('<Q', self.read(address, 8))[0]

    def typ(self, address, name):
        if self.ptr(address + 8) != self.types[name]:
            raise ValueError(f'Expected {name} at {address:x}')

    def string(self, address):
        if address in self.string_cache:
            return self.string_cache[address]
        self.typ(address, 'PyUnicode_Type')
        size = struct.unpack('<q', self.read(address + 16, 8))[0]
        state = struct.unpack('<I', self.read(address + 32, 4))[0]
        kind, compact, ascii_flag = (state >> 2) & 7, (state >> 5) & 1, (state >> 6) & 1
        if not compact or kind not in (1, 2, 4) or not 0 <= size <= 20000:
            raise ValueError('Unsupported unicode')
        raw = self.read(address + (48 if ascii_flag else 72), size * kind) if size else b''
        value = raw.decode({1: 'latin1', 2: 'utf-16-le', 4: 'utf-32-le'}[kind])
        self.string_cache[address] = value
        return value

    def primitive(self, address):
        typ = self.ptr(address + 8)
        if typ == self.types['PyUnicode_Type']:
            return self.string(address)
        if typ == self.types['PyFloat_Type']:
            value = struct.unpack('<d', self.read(address + 16, 8))[0]
            if not math.isfinite(value):
                raise ValueError('Nonfinite float')
            return value
        if typ == self.types['PyLong_Type']:
            size = struct.unpack('<q', self.read(address + 16, 8))[0]
            if abs(size) > 4:
                raise ValueError('Large unexpected integer')
            digits = struct.unpack('<' + 'I' * abs(size), self.read(address + 24, abs(size) * 4)) if size else []
            return (-1 if size < 0 else 1) * sum(x << (30 * i) for i, x in enumerate(digits))
        raise ValueError(f'Unexpected primitive type {typ:x}')

    def tuple_ptrs(self, address):
        self.typ(address, 'PyTuple_Type')
        count = struct.unpack('<q', self.read(address + 16, 8))[0]
        if not 0 <= count <= 10000:
            raise ValueError('Tuple too large')
        return list(struct.unpack('<' + 'Q' * count, self.read(address + 24, count * 8))) if count else []

    def find_qa(self):
        interp = self.ptr(self.types['_PyRuntime'] + 32)
        found, seen_threads = [], set()
        while interp:
            thread = self.ptr(interp + 8)
            while thread and thread not in seen_threads:
                seen_threads.add(thread)
                frame = self.ptr(thread + 24)
                seen_frames = set()
                while frame and frame not in seen_frames:
                    seen_frames.add(frame)
                    code = self.ptr(frame + 32)
                    name = self.string(self.ptr(code + 112))
                    if name == '_validate':
                        filename = self.string(self.ptr(code + 104))
                        nlocals = struct.unpack('<i', self.read(code + 28, 4))[0]
                        varnames = [self.string(p) for p in self.tuple_ptrs(self.ptr(code + 72))]
                        cellvars = [self.string(p) for p in self.tuple_ptrs(self.ptr(code + 88))]
                        if 'qa_rows' in varnames:
                            qa = self.ptr(frame + 352 + 8 * varnames.index('qa_rows'))
                        elif 'qa_rows' in cellvars:
                            cell = self.ptr(frame + 352 + 8 * (nlocals + cellvars.index('qa_rows')))
                            self.typ(cell, 'PyCell_Type')
                            qa = self.ptr(cell + 16)
                        else:
                            frame = self.ptr(frame + 24)
                            continue
                        self.typ(qa, 'PyList_Type')
                        found.append(dict(frame=frame, qa=qa, filename=filename))
                    frame = self.ptr(frame + 24)
                thread = self.ptr(thread + 8)
            interp = self.ptr(interp)
        unique = {x['qa']: x for x in found}
        if len(unique) != 1:
            raise RuntimeError(f'Expected one _validate qa_rows, found {len(unique)}')
        return next(iter(unique.values()))

    def row(self, address):
        self.typ(address, 'PyDict_Type')
        before = self.read(address, 48)
        used, keys, values = struct.unpack_from('<q', before, 16)[0], struct.unpack_from('<Q', before, 32)[0], struct.unpack_from('<Q', before, 40)[0]
        if used != 7 or values:
            raise ValueError('Expected combined seven-key metric dict')
        header = self.read(keys, 40)
        size, entries = struct.unpack_from('<q', header, 8)[0], struct.unpack_from('<q', header, 32)[0]
        if size <= 0 or size & (size - 1) or size > 1024 or entries != 7:
            raise ValueError('Unexpected dict keys table')
        ixwidth = 1 if size <= 128 else 2 if size <= 32768 else 4 if size <= 2**31 else 8
        data = self.read(keys + 40 + size * ixwidth, entries * 24)
        result = {}
        for i in range(entries):
            key, value = struct.unpack_from('<QQ', data, i * 24 + 8)
            if key and value:
                result[self.string(key)] = self.primitive(value)
        if self.read(address, 48) != before or set(result) != KEYS:
            raise ValueError('Metric dict changed or wrong schema')
        if not isinstance(result['source'], str) or not result['source']:
            raise ValueError('Invalid source')
        for key in ('f1', 'em', 'answer_format', 'format_rate'):
            if not isinstance(result[key], (int, float)) or not 0 <= result[key] <= 1:
                raise ValueError(f'Invalid {key}')
        for key in ('generated_tokens', 'search_calls'):
            if type(result[key]) is not int or result[key] < 0:
                raise ValueError(f'Invalid {key}')
        return result

    def snapshot(self, qa):
        header = self.read(qa, 40)
        count, items, allocated = struct.unpack_from('<qQq', header, 16)
        if not 0 <= count <= allocated <= 1000000:
            raise ValueError('Unexpected list size')
        pointers_raw = self.read(items, count * 8) if count else b''
        pointers = struct.unpack('<' + 'Q' * count, pointers_raw)
        rows = [self.row(p) for p in pointers]
        after = self.read(qa, 40)
        after_count, after_items = struct.unpack_from('<qQ', after, 16)
        if after_count < count or self.read(after_items, count * 8) != pointers_raw:
            raise RuntimeError('List prefix changed during snapshot')
        return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pid', type=int, required=True)
    ap.add_argument('--out-dir', type=Path, default=Path('/run/output/eval-resume'))
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--minimum-rows', type=int, default=30000)
    ap.add_argument('--expected-count', type=int)
    ap.add_argument('--observer-ledger', type=Path)
    ap.add_argument('--interval', type=float, default=2)
    args = ap.parse_args()
    def observer():
        if args.observer_ledger is None:
            return None
        # Ignore an incomplete final line while the writer is active.
        raw = args.observer_ledger.read_bytes()
        records = [json.loads(line) for line in raw.split(b'\n')[:-1] if line]
        batches = [r for r in records if r.get('kind') == 'real_sample_batch'
                   and r.get('pid') == args.pid and r.get('phase') == 'final']
        if not batches or any(r['rows'] != args.batch_size for r in batches):
            raise RuntimeError('Missing or unexpectedly sized observer batches')
        return dict(path=str(args.observer_ledger), started_rows=sum(r['rows'] for r in batches),
                    latest_started_batch=batches[-1], ledger_bytes=len(raw))
    reader = Reader(args.pid)
    origin = reader.find_qa()
    observer_first = observer()
    first = reader.snapshot(origin['qa'])
    print(json.dumps({'stage': 'first_snapshot', 'rows': len(first), 'frame': hex(origin['frame']), 'qa_rows': hex(origin['qa'])}), flush=True)
    time.sleep(args.interval)
    second_origin = reader.find_qa()
    if second_origin != origin:
        raise RuntimeError('Validation frame changed between snapshots')
    second = reader.snapshot(origin['qa'])
    observer_second = observer()
    if len(second) < len(first) or second[:len(first)] != first:
        raise RuntimeError('Two snapshots do not have an identical immutable prefix')
    stable_count = len(first) // args.batch_size * args.batch_size
    if observer_first is not None:
        # Ledger logs batch start, whereas qa_rows is appended after decoding
        # and scoring. Its only possible live difference is one active batch.
        for rows_count, obs in ((len(first), observer_first), (len(second), observer_second)):
            if abs(obs['started_rows'] - rows_count) > args.batch_size:
                raise RuntimeError('Observer and completed metric count differ by more than one batch')
    if stable_count < args.minimum_rows:
        raise RuntimeError(f'Only {stable_count} batch-aligned stable rows')
    if args.expected_count is not None and stable_count != args.expected_count:
        raise RuntimeError(f'Expected observer count {args.expected_count}, stable rows={stable_count}')
    if Path(f'/proc/{args.pid}/stat').read_text().rsplit(')', 1)[1].split()[19] != reader.starttime:
        raise RuntimeError('PID identity changed')
    rows = first[:stable_count]
    raw = (json.dumps(rows, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    path = args.out_dir / f'preserved-prefix-pid{args.pid}-{stamp}.json'
    with path.open('xb') as fp:
        fp.write(raw)
        fp.flush()
        os.fsync(fp.fileno())
    if hashlib.sha256(path.read_bytes()).hexdigest() != hashlib.sha256(raw).hexdigest():
        raise RuntimeError('Persisted snapshot checksum mismatch')
    receipt = dict(pid=args.pid, process_starttime=reader.starttime, python_version='3.10',
        method='external /proc/PID/mem pread; no ptrace/signals/injection',
        frame=hex(origin['frame']), qa_rows=hex(origin['qa']), source_file=origin['filename'],
        first_snapshot_count=len(first), second_snapshot_count=len(second), stable_batch_aligned_count=stable_count,
        batch_size=args.batch_size, source_counts=dict(collections.Counter(r['source'] for r in rows)),
        observer_first=observer_first, observer_second=observer_second,
        file=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
        warning='Metrics have no question IDs: independently bind this prefix to the exact effective ordered validation dataset before resume.')
    receipt_path = path.with_suffix('.receipt.json')
    with receipt_path.open('x') as fp:
        fp.write(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
        fp.flush()
        os.fsync(fp.fileno())
    directory_fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
