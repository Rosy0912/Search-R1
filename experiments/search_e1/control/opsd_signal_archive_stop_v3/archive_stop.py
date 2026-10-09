"""Stop only the explicitly replaced HKUST4 archive; never execute on import.

Linux only. Root publishes/reviews this separate helper before invoking --execute.
The required private snapshot and human authorization bytes are SHA-bound inputs.
No killpg/pkill, no CUDA/Torch/Ray import, and no model/optimizer mutation.
An incomplete preservation leaves the identified processes quiesced and reports
HOLD; it never resumes a supervisor which could automatically launch OPSD.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import datetime as dt
import errno
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import signal
import struct
import subprocess
import sys
import time
import uuid

BASE = Path('/weka_3/shaoshitong/search-e1-config-study-20261004')
RUN = 'f04fe593830c4831918ede872812c016'
UID = 1001
HOST = 'xielab5'
SUPERVISORS = {1405392: 637554706, 1432632: 637837817, 1434631: 637850112}
ANCHORS = {1434645: 637850668, 1434648: 637850669, 1434651: 637850670,
           1435488: 637851418, 1437474: 637851512, 1437900: 637852269,
           1438381: 637852857, 1438382: 637852858, 1438383: 637852859}
PROTECTED = {1749337: 641133475, 2062794: 645271546, 1751713: 641148734}
GPU_UUIDS = {
    4: 'GPU-ed24344f-9cd8-6d29-e8b9-d7582a8cfd65',
    5: 'GPU-79de0d51-c2a2-897e-f76a-ec84f03b38e2',
    6: 'GPU-3a89b647-73fb-27bc-c8a8-e9f663b96d5f',
    7: 'GPU-21352a4a-5b55-8adf-8e82-c70b7a94ecf3',
}


class Hold(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise Hold(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def argv_sha(record):
    # Matches the actual read-only snapshot's cmdline normalization, including
    # trailing spaces. No shell parsing or whitespace normalization is applied.
    return sha(record['argv'].encode('utf-8'))


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def bound_bytes(path, expected):
    require(re.fullmatch(r'[0-9a-f]{64}', expected) is not None, 'Invalid input SHA')
    raw = Path(path).read_bytes()
    require(sha(raw) == expected, 'Input SHA mismatch: ' + str(path))
    return raw


def utc(value):
    result = dt.datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(result.tzinfo is not None, 'Observation must include timezone')
    return result.astimezone(dt.timezone.utc)


def load_request(snapshot_raw, authorization_raw, authorization_observed, now):
    x = json.loads(snapshot_raw)
    require(x.get('schema') == 'K4-human-resource-readonly-derived-v1', 'Wrong snapshot schema')
    require(x.get('base') == str(BASE) and x.get('hostname') == HOST and x.get('uid') == UID,
            'Wrong snapshot base/host/UID')
    observed = utc(x['actual_observed_utc'])
    auth_time = utc(authorization_observed)
    require(observed <= now and (now - observed).total_seconds() <= 12 * 3600,
            'Snapshot expired or from future')
    require(auth_time <= now and (now - auth_time).total_seconds() <= 24 * 3600,
            'Authorization observation expired or from future')
    require(authorization_raw.strip(), 'Empty human authorization')
    # Exact human text is evidence, not parsed instructions or executable code.
    records = {r['pid']: r for r in x['owned_processes']}
    expected = dict(SUPERVISORS)
    expected.update(ANCHORS)
    expected.update(PROTECTED)
    for pid, tick in expected.items():
        r = records.get(pid)
        require(r and r.get('uid') == UID and r.get('starttick') == tick,
                'Snapshot missing exact approved process: ' + str(pid))
        require(isinstance(r.get('argv'), str) and r['argv'], 'Missing snapshot argv')
        if pid in ANCHORS:
            require(r.get('run_env', {}).get('E1_RUN_ID') == RUN,
                    'Snapshot anchor belongs to another run')
        if pid in SUPERVISORS:
            require(r.get('run_env', {}).get('E1_BASE') == str(BASE), 'Supervisor base mismatch')
    return x, records


def read_process(pid):
    p = Path('/proc') / str(pid)
    try:
        st = p.joinpath('stat').read_text()
        tail = st[st.rfind(')') + 2:].split()
        status = p.joinpath('status').read_text()
        uid_match = re.search(r'^Uid:\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)', status, re.M)
        require(uid_match is not None, 'Unreadable effective UID')
        real_uid, effective_uid = int(uid_match[1]), int(uid_match[2])
        if effective_uid != UID:
            return {'pid': pid, 'uid': effective_uid, 'real_uid': real_uid,
                    'starttick': int(tail[19]), 'ppid': int(tail[1]), 'opaque_foreign': True}
        record = {'pid': pid, 'uid': effective_uid, 'real_uid': real_uid, 'state': tail[0],
                'ppid': int(tail[1]), 'pgid': int(tail[2]), 'sid': int(tail[3]),
                'starttick': int(tail[19]),
                'argv': p.joinpath('cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace'),
                'cwd': str(p.joinpath('cwd').resolve()),
                'run_env': {}, 'env_readable': False}
        try:
            env = p.joinpath('environ').read_bytes().split(b'\0')
            env = {v.split(b'=', 1)[0].decode(errors='replace'):
                   v.split(b'=', 1)[1].decode(errors='replace') for v in env if b'=' in v}
            record['run_env'] = {k: env[k] for k in ('E1_BASE', 'E1_RUN_ID', 'E1_PROFILE',
                                                    'E1_STAGE', 'CUDA_VISIBLE_DEVICES') if k in env}
            record['env_readable'] = True
        except PermissionError:
            record['env_error'] = 'PermissionError; opaque unrelated processes are not targets'
        final = p.joinpath('stat').read_text()
        final_tail = final[final.rfind(')') + 2:].split()
        if int(final_tail[19]) != record['starttick']:
            return None  # Inconsistent PID sample: never a signalable identity.
        return record
    except (FileNotFoundError, ProcessLookupError):
        return None
    except PermissionError:
        return {'pid': pid, 'opaque_identity': True, 'env_readable': False}


def process_table():
    result = {}
    for p in Path('/proc').iterdir():
        if p.name.isdigit():
            r = read_process(int(p.name))
            if r:
                result[r['pid']] = r
    return result


def same_identity(expected, actual):
    if not (actual and expected['pid'] == actual.get('pid') and
            expected['uid'] == actual.get('uid') and
            expected['starttick'] == actual.get('starttick')):
        return False
    def dynamic_actor(r):
        return (r.get('argv', '').startswith('ray::') and
                r.get('run_env', {}).get('E1_RUN_ID') == RUN and
                r.get('run_env', {}).get('E1_BASE') == str(BASE) and
                r.get('cwd') == str(BASE / 'runtime/archive'))
    # Ray's title changes with the actor method. Its actual title is recorded,
    # not normalized into a fabricated match. UID/tick/run/cwd stay mandatory.
    if dynamic_actor(expected) or dynamic_actor(actual):
        return dynamic_actor(expected) and dynamic_actor(actual)
    return argv_sha(expected) == argv_sha(actual)


def in_tree(pid, roots, table):
    seen = set()
    while pid in table and pid not in seen:
        if pid in roots:
            return True
        seen.add(pid)
        pid = table[pid].get('ppid')
    return False


def select_owned(table, bound_records, strict_known=True, historical=None):
    targets = {}
    known = set(SUPERVISORS) | set(ANCHORS)
    roots = set()
    reused = set()
    for pid in known:
        actual = table.get(pid)
        if actual and actual.get('state') != 'Z':
            if not same_identity(bound_records[pid], actual):
                if not strict_known:
                    require(type(actual.get('starttick')) is int and actual['starttick'] > 0,
                            'Opaque/missing starttick for known historical PID: ' + str(pid))
                    if actual['starttick'] != bound_records[pid]['starttick']:
                        reused.add(pid)
                        continue
                raise Hold('Known PID identity changed: ' + str(pid))
            require(actual.get('env_readable', True), 'Opaque environment for known target')
            targets[pid] = actual
            roots.add(pid)
    for pid, previous in (historical or {}).items():
        actual = table.get(pid)
        if actual:
            require(type(actual.get('starttick')) is int and actual['starttick'] > 0,
                    'Opaque/missing starttick for captured historical PID: ' + str(pid))
            if actual['starttick'] != previous['starttick']:
                reused.add(pid)
    for pid, r in table.items():
        if pid in PROTECTED or pid in reused or r.get('uid') != UID or r.get('state') == 'Z':
            continue
        env = r.get('run_env', {})
        marker = env.get('E1_RUN_ID')
        descendant = in_tree(pid, roots, table)
        if descendant:
            require(r.get('env_readable', True), 'Opaque environment for attributed descendant')
        if marker == RUN:
            require(env.get('E1_BASE') == str(BASE), 'Run marker base mismatch')
            cvd = env.get('CUDA_VISIBLE_DEVICES')
            devices = cvd.split(',') if isinstance(cvd, str) else []
            require(bool(devices) and all(d in ('4', '5', '6', '7') for d in devices) and
                    len(devices) == len(set(devices)), 'Run marker GPU scope mismatch')
            targets[pid] = r
        elif descendant and pid not in SUPERVISORS:
            require(marker in (None, ''), 'Descendant is another E1 run: ' + str(pid))
            require(str(BASE) in r.get('argv', '') or
                    r.get('cwd', '').startswith(str(BASE) + '/'),
                    'Unattributed descendant: ' + str(pid))
            targets[pid] = r
    require(not (set(targets) & set(PROTECTED)), 'Protected process selected')
    return targets


def remaining_owned(table, bound_records, historical):
    """Current scope UNION every captured exact identity, including reparented helpers."""
    remaining = select_owned(table, bound_records, strict_known=False, historical=historical)
    ended = []
    for pid, old in historical.items():
        actual = table.get(pid)
        if actual is None:
            ended.append({'pid': pid, 'starttick': old['starttick'], 'old_identity': 'ABSENT_OR_ZOMBIE'})
        else:
            require(type(actual.get('starttick')) is int and actual['starttick'] > 0,
                    'Opaque/missing starttick for captured historical PID: ' + str(pid))
            if actual['starttick'] != old['starttick']:
                ended.append({'pid': pid, 'starttick': old['starttick'], 'old_identity': 'PID_REUSED_NOT_SIGNALED'})
            elif actual.get('state') == 'Z':
                require(actual.get('uid') == old['uid'], 'Captured zombie UID changed')
                ended.append({'pid': pid, 'starttick': old['starttick'], 'old_identity': 'ABSENT_OR_ZOMBIE'})
            else:
                require(same_identity(old, actual), 'Captured identity changed: ' + str(pid))
                remaining[pid] = actual
    return remaining, ended


def check_protected(table, bound_records):
    for pid in PROTECTED:
        require(same_identity(bound_records[pid], table.get(pid)),
                'Protected retriever/observer changed or absent: ' + str(pid))


class NativePidfd:
    """Linux x86_64 ABI only; no PID-number or process-group signal fallback."""
    def __init__(self, libc=None, system=None, machine=None):
        system = system if system is not None else sys.platform
        machine = machine if machine is not None else os.uname().machine
        require(system == 'linux' and machine == 'x86_64' and ctypes.sizeof(ctypes.c_void_p) == 8,
                'Native pidfd adapter requires Linux x86_64')
        self.libc = libc if libc is not None else ctypes.CDLL(None, use_errno=True)
        self.libc.syscall.restype = ctypes.c_long

    def call(self, number, *args):
        ctypes.set_errno(0)
        value = self.libc.syscall(ctypes.c_long(number), *args)
        if value < 0:
            error = ctypes.get_errno() or errno.EIO
            raise OSError(error, os.strerror(error))
        return int(value)

    def open(self, pid, flags=0):
        require(type(pid) is int and pid > 0 and type(flags) is int and flags == 0,
                'Native pidfd_open requires positive PID and flags0')
        return self.call(434, ctypes.c_int(pid), ctypes.c_uint(flags))

    def send(self, pidfd, sig, siginfo=None, flags=0):
        require(type(pidfd) is int and pidfd >= 0 and siginfo is None and
                type(flags) is int and flags == 0,
                'Native pidfd_send_signal requires valid fd, null info and flags0')
        require(sig in (signal.SIGSTOP, signal.SIGTERM, signal.SIGCONT, signal.SIGKILL),
                'Native pidfd signal outside reviewed stop scope')
        return self.call(424, ctypes.c_int(pidfd), ctypes.c_int(int(sig)),
                         ctypes.c_void_p(None), ctypes.c_uint(flags))


def pidfd_backend():
    """Use built-ins when present, otherwise exact native syscalls; probe self fd only."""
    opener = getattr(os, 'pidfd_open', None)
    sender = getattr(signal, 'pidfd_send_signal', None)
    native = None
    if not callable(opener) or not callable(sender):
        native = NativePidfd()
    open_function = opener if callable(opener) else native.open
    send_function = sender if callable(sender) else native.send
    proof = {'open_backend': 'python_builtin' if callable(opener) else 'linux_x86_64_syscall434',
             'send_backend': 'python_builtin' if callable(sender) else 'linux_x86_64_syscall424',
             'probe_pid': os.getpid(), 'probe_flags': 0, 'target_signal_sent_by_probe': False}
    try:
        fd = open_function(proof['probe_pid'], 0)
        require(type(fd) is int and fd >= 0, 'Invalid pidfd self-probe descriptor')
        os.close(fd)
    except OSError as exc:
        raise Hold('Exact pidfd self-open/close probe failed: errno=' +
                   str(exc.errno) + ' ' + str(exc)) from exc
    proof['self_open_close_accepted'] = True
    return open_function, send_function, proof


class ExactSignals:
    def __init__(self, reader=None, opener=None, sender=None, closer=None):
        self.reader = reader or read_process
        self.opener = opener or os.pidfd_open
        self.sender = sender or signal.pidfd_send_signal
        self.closer = closer or os.close
        self.handles = {}
        self.events = []

    def bind(self, record):
        pid = record['pid']
        require(pid not in PROTECTED and record['uid'] == UID, 'Denied signal target')
        if pid in self.handles:
            require(same_identity(record, self.handles[pid][1]), 'Handle identity mismatch')
            return
        fd = self.opener(pid, 0)
        actual = self.reader(pid)
        if not same_identity(record, actual):
            self.closer(fd)
            raise Hold('PID changed while opening pidfd: ' + str(pid))
        self.handles[pid] = (fd, dict(actual, run_env=dict(actual.get('run_env', {}))))

    def send(self, record, sig):
        self.bind(record)
        actual = self.reader(record['pid'])
        if actual is None or actual.get('state') == 'Z':
            return False
        require(same_identity(record, actual), 'PID reused before signal')
        try:
            self.sender(self.handles[record['pid']][0], sig, None, 0)
        except ProcessLookupError:
            return False
        self.events.append({'pid': record['pid'], 'uid': record['uid'],
                            'starttick': record['starttick'], 'argv_sha256': argv_sha(record),
                            'pidfd_bound_actual_argv_sha256': argv_sha(self.handles[record['pid']][1]),
                            'signal_observed_actual_argv_sha256': argv_sha(actual),
                            'signal': signal.Signals(sig).name, 'unix': time.time()})
        return True

    def close(self):
        for fd, _ in self.handles.values():
            self.closer(fd)


def checked_gpu():
    def rows(fields, kind):
        a = ['nvidia-smi', '--query-' + kind + '=' + fields, '--format=csv,noheader,nounits']
        r = subprocess.run(a, capture_output=True, text=True, timeout=15)
        require(r.returncode == 0, 'nvidia-smi failed: ' + r.stderr)
        return list(csv.reader(io.StringIO(r.stdout), skipinitialspace=True)), {'argv': a, 'stdout': r.stdout}
    gs, gp = rows('index,uuid,memory.total,memory.used,memory.free', 'gpu')
    apps, ap = rows('gpu_uuid,pid,process_name,used_gpu_memory', 'compute-apps')
    gpus = {int(r[0]): {'index': int(r[0]), 'uuid': r[1], 'total_mib': int(r[2]),
                        'used_mib': int(r[3]), 'free_mib': int(r[4])} for r in gs}
    for index, expected in GPU_UUIDS.items():
        require(gpus.get(index, {}).get('uuid') == expected, 'Physical GPU UUID mismatch')
    return gpus, [{'uuid': r[0], 'pid': int(r[1]), 'name': r[2], 'mib': int(r[3])}
                  for r in apps if r], {'gpu': gp, 'apps': ap}


def check_gpu_ownership(gpus, apps, targets, released=False):
    results = []
    uuids = set(GPU_UUIDS.values())
    for a in apps:
        if a['pid'] in targets:
            require(a['uuid'] in uuids, 'Own target has context outside GPU4-7')
        if a['uuid'] in uuids:
            require(a['pid'] in targets or (a['pid'] == 1749337 and a['uuid'] == GPU_UUIDS[7]),
                    'Foreign compute on selected GPU: ' + str(a))
            if released:
                require(a['pid'] == 1749337, 'Old archive context remains')
    if released:
        oracle = [a for a in apps if a['pid'] == 1749337]
        require(len(oracle) == 1 and oracle[0]['uuid'] == GPU_UUIDS[7] and oracle[0]['mib'] <= 1024,
                'Protected oracle GPU identity/usage failed')
        for index in GPU_UUIDS:
            g = gpus[index]
            allowance = oracle[0]['mib'] if index == 7 else 0
            reserved = g['total_mib'] - g['used_mib'] - g['free_mib']
            require(0 <= reserved <= 1024, 'Unexpected GPU driver reserved memory')
            # Direct memory.free is tested; total-used is not substituted for it.
            minimum = g['total_mib'] - reserved - allowance - 64
            require(g['used_mib'] <= allowance + 64 and g['free_mib'] >= minimum,
                    'Selected GPU not fully released: ' + str(g))
            results.append(dict(g, oracle_allowance_mib=allowance, reserved_mib=reserved,
                                minimum_direct_free_mib=minimum, guard_mib=64))
    return results


def stable_inventory(root):
    records = []
    for p in sorted(root.rglob('*')):
        if p.is_symlink():
            records.append({'relative_path': str(p.relative_to(root)), 'symlink': os.readlink(p)})
        elif p.is_file():
            before = p.stat()
            digest = file_sha(p)
            after = p.stat()
            require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
                    (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
                    'File changed during preservation: ' + str(p))
            records.append({'relative_path': str(p.relative_to(root)), 'bytes': after.st_size,
                            'sha256': digest, 'mtime_ns': after.st_mtime_ns})
    return records


DTYPE_BYTES = {'BOOL': 1, 'U8': 1, 'I8': 1, 'U16': 2, 'I16': 2,
               'U32': 4, 'I32': 4, 'U64': 8, 'I64': 8,
               'F16': 2, 'BF16': 2, 'F32': 4, 'F64': 8,
               'F8_E4M3': 1, 'F8_E4M3FN': 1, 'F8_E5M2': 1}


def safetensors_layout(path):
    """CPU header/shape/offset/EOF validation only, never tensor arithmetic."""
    before = path.stat()
    with path.open('rb') as stream:
        prefix = stream.read(8)
        require(len(prefix) == 8, 'Truncated safetensors length')
        header_bytes = struct.unpack('<Q', prefix)[0]
        require(2 <= header_bytes <= 64 * 1024 * 1024 and 8 + header_bytes <= before.st_size,
                'Truncated/invalid safetensors header length')
        raw = stream.read(header_bytes)
    try:
        header = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise Hold('Invalid safetensors JSON header') from exc
    require(isinstance(header, dict), 'Safetensors header is not object')
    intervals = []
    names = []
    for name, entry in header.items():
        if name == '__metadata__':
            continue
        require(isinstance(entry, dict) and entry.get('dtype') in DTYPE_BYTES,
                'Unknown/invalid safetensors dtype')
        shape, offsets = entry.get('shape'), entry.get('data_offsets')
        require(isinstance(shape, list) and all(type(n) is int and n >= 0 for n in shape),
                'Invalid safetensors shape')
        require(isinstance(offsets, list) and len(offsets) == 2 and
                all(type(n) is int and n >= 0 for n in offsets) and offsets[0] <= offsets[1],
                'Invalid safetensors offsets')
        elements = 1
        for n in shape:
            elements *= n
        require(offsets[1] - offsets[0] == elements * DTYPE_BYTES[entry['dtype']],
                'Safetensors dtype/shape byte length mismatch')
        intervals.append((offsets[0], offsets[1]))
        names.append(name)
    require(names, 'No safetensors entries')
    cursor = 0
    for start, end in sorted(intervals):
        require(start == cursor, 'Safetensors data hole/overlap')
        cursor = end
    require(8 + header_bytes + cursor == before.st_size, 'Safetensors data EOF mismatch/truncation')
    after = path.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
            'Safetensors changed during layout check')
    return {'tensor_keys': sorted(names), 'data_bytes': cursor,
            'header_sha256': sha(raw), 'header_bytes': header_bytes,
            'scope': 'layout only; numerical finite/optimizer/RNG not checked'}


def preserve(base, destination):
    actor = base / 'runs/archive/round-1-grpo/actor'
    checkpoints = []
    for p in sorted(actor.glob('global_step_*'), key=lambda x: int(x.name.rsplit('_', 1)[1])):
        require(p.is_dir() and not p.is_symlink(), 'Checkpoint path must be real directory')
        inventory = stable_inventory(p)
        names = {r['relative_path'] for r in inventory}
        required = {'added_tokens.json', 'config.json', 'generation_config.json', 'merges.txt',
                    'model-00001-of-00003.safetensors', 'model-00002-of-00003.safetensors',
                    'model-00003-of-00003.safetensors', 'model.safetensors.index.json',
                    'special_tokens_map.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json'}
        complete = required <= names and all(r.get('bytes', 0) > 0 for r in inventory
                                            if r['relative_path'] in required)
        layouts = {}
        layout_error = None
        if complete:
            try:
                index = json.loads((p / 'model.safetensors.index.json').read_text())
                weight_map = index.get('weight_map', {})
                shards = {'model-00001-of-00003.safetensors', 'model-00002-of-00003.safetensors',
                          'model-00003-of-00003.safetensors'}
                require(isinstance(weight_map, dict) and set(weight_map.values()) == shards,
                        'Index does not map all three shards')
                layouts = {s: safetensors_layout(p / s) for s in shards}
                for s in shards:
                    require(set(layouts[s]['tensor_keys']) == {k for k, v in weight_map.items() if v == s},
                            'Index/tensor header key mismatch')
                if 'total_size' in index.get('metadata', {}):
                    require(index['metadata']['total_size'] == sum(z['data_bytes'] for z in layouts.values()),
                            'Index total_size mismatch')
                require(inventory == stable_inventory(p), 'Checkpoint changed during layout/hash verification')
            except (Hold, ValueError, UnicodeError) as exc:
                complete = False
                layout_error = str(exc)
        checkpoints.append({'path': str(p), 'step': int(p.name.rsplit('_', 1)[1]), 'files': inventory,
                            'structurally_complete_12_file_checkpoint': complete,
                            'safetensors_layout': layouts, 'layout_error': layout_error,
                            'numerically_accepted': False,
                            'note': 'All original files retained; layout validity plus SHA is not full tensor/Adam resume acceptance.'})
    structural = [c for c in checkpoints if c['structurally_complete_12_file_checkpoint']]
    require(structural, 'No layout-valid checkpoint to preserve')
    copied = []
    for relative in ('source', 'runtime', 'opsd-runtime', 'logs', 'receipts',
                     'runs/archive/round-1-grpo/train.log'):
        source = base / relative
        if not source.exists():
            copied.append({'path': str(source), 'absent': True})
            continue
        target = destination / 'copies' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        copied_inventory = copy_evidence_snapshot(source, target)
        copied.append({'path': str(source), 'copy': str(target), 'files': copied_inventory})
    return {'checkpoints': checkpoints, 'latest_layout_valid_checkpoint_step': max(p['step'] for p in structural),
            'checkpoint_weights_only': True, 'complete_Adam_resume_state': False,
            'warning': 'No live Adam/scheduler/RNG capture. Completed metric updates beyond durable checkpoint are not recoverable state.',
            'copied_evidence': copied}


def copy_evidence_snapshot(source, target):
    """Per-file byte snapshots; protected observer receipts may keep updating.

    A replaced observer receipt is explicitly recorded, not required to equal a
    later live receipt. Checkpoint stability uses the separate stricter inventory.
    No external symlink is followed or copied into a private evidence directory.
    """
    records = []
    paths = sorted(source.rglob('*')) if source.is_dir() else [source]
    if source.is_dir():
        target.mkdir(exist_ok=False)
    for p in paths:
        relative = p.relative_to(source) if source.is_dir() else Path(p.name)
        q = target / relative if source.is_dir() else target
        if p.is_symlink():
            q.parent.mkdir(parents=True, exist_ok=True)
            q.symlink_to(os.readlink(p))
            records.append({'relative_path': str(relative), 'symlink': os.readlink(p)})
        elif p.is_dir():
            q.mkdir(exist_ok=True)
        elif p.is_file():
            q.parent.mkdir(parents=True, exist_ok=True)
            for attempt in range(4):
                h = hashlib.sha256()
                with p.open('rb') as src, q.open('wb') as dst:
                    before = os.fstat(src.fileno())
                    for block in iter(lambda: src.read(8 * 1024 * 1024), b''):
                        h.update(block)
                        dst.write(block)
                    dst.flush()
                    os.fsync(dst.fileno())
                    after = os.fstat(src.fileno())
                if (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns):
                    break
            else:
                raise Hold('Continuously changing evidence file: ' + str(p))
            require(file_sha(q) == h.hexdigest() and q.stat().st_size == before.st_size,
                    'Preserved evidence byte mismatch')
            live = p.stat()
            records.append({'relative_path': str(relative), 'bytes': before.st_size,
                            'sha256': h.hexdigest(), 'source_mtime_ns': before.st_mtime_ns,
                            'source_inode_at_copy': before.st_ino,
                            'source_replaced_or_updated_after_copy':
                            (before.st_ino, before.st_size, before.st_mtime_ns) !=
                            (live.st_ino, live.st_size, live.st_mtime_ns)})
    return records


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def wait_stopped(records, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        pending = []
        for r in records.values():
            actual = read_process(r['pid'])
            if actual and actual.get('state') != 'Z':
                require(same_identity(r, actual), 'Identity changed while quiescing')
                if actual['state'] not in ('T', 't'):
                    pending.append(r['pid'])
        if not pending:
            return
        time.sleep(.1)
    raise Hold('Quiesce timeout: ' + str(pending))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot', required=True)
    p.add_argument('--snapshot-sha256', required=True)
    p.add_argument('--authorization-file', required=True)
    p.add_argument('--authorization-sha256', required=True)
    p.add_argument('--authorization-observed-utc', required=True)
    p.add_argument('--execute', action='store_true', help='Without this flag: validation-only, no signals or Node writes')
    a = p.parse_args(argv)
    snapshot_raw = bound_bytes(a.snapshot, a.snapshot_sha256)
    auth_raw = bound_bytes(a.authorization_file, a.authorization_sha256)
    snapshot, records = load_request(snapshot_raw, auth_raw, a.authorization_observed_utc,
                                     dt.datetime.now(dt.timezone.utc))
    require(os.getuid() == UID and os.uname().nodename == HOST, 'Wrong execution host/UID')
    pidfd_open_function, pidfd_send_function, pidfd_probe = pidfd_backend()
    table = process_table()
    check_protected(table, records)
    owned = select_owned(table, records)
    require(owned, 'No live approved process found; use separately reviewed terminal verifier')
    gpus, apps, raw_gpu = checked_gpu()
    check_gpu_ownership(gpus, apps, owned)
    result = {'schema': 'k4-archive-replacement-stop-v1', 'base': str(BASE), 'run_id': RUN,
              'helper_source_sha256': file_sha(__file__), 'uid': UID, 'hostname': HOST,
              'started_unix': time.time(), 'snapshot_sha256': a.snapshot_sha256,
              'authorization_sha256': a.authorization_sha256,
              'authorization_observed_utc': a.authorization_observed_utc,
              'authorization_raw_utf8': auth_raw.decode('utf-8'), 'initial_owned': list(owned.values()),
              'protected': [records[pid] for pid in PROTECTED], 'initial_gpu': raw_gpu,
              'signals_executed': False, 'status': 'VALIDATED_ONLY', 'accepted': False,
              'pidfd_backend_probe': pidfd_probe}
    if not a.execute:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    destination = BASE / 'preserved' / ('user-stop-archive-20261009-' + uuid.uuid4().hex)
    destination.mkdir(parents=True, exist_ok=False)
    result['preserved_directory'] = str(destination)
    (destination / 'input-snapshot.private.json').write_bytes(snapshot_raw)
    (destination / 'authorization.private.txt').write_bytes(auth_raw)
    signals = ExactSignals(opener=pidfd_open_function, sender=pidfd_send_function)
    stopped = {}
    try:
        # Stop ancestors first so child exit cannot launch a new pipeline stage.
        for pid in SUPERVISORS:
            if pid in owned and signals.send(owned[pid], signal.SIGSTOP):
                stopped[pid] = owned[pid]
        wait_stopped(stopped)
        # Freeze the bounded run too: preservation hashes/copies must be stable.
        for _ in range(3):
            table = process_table()
            check_protected(table, records)
            refreshed = select_owned(table, records)
            for pid, r in refreshed.items():
                if pid not in stopped and signals.send(r, signal.SIGSTOP):
                    stopped[pid] = r
            owned.update(refreshed)
            wait_stopped(stopped)
        require(set(select_owned(process_table(), records)) <= set(stopped), 'New unquiesced descendant')
        result['signals_executed'] = True
        result['status'] = 'QUIESCED_PRESERVING'
        result['signal_events'] = signals.events
        write_json(destination / 'quiesced.private.json', result)
        result['preservation'] = preserve(BASE, destination)
        write_json(destination / 'preservation.private.json', result['preservation'])
        # TERM is queued while stopped. CONT is sent only to exact processes
        # stopped by this helper, solely to deliver their pending TERM.
        for r in owned.values():
            signals.send(r, signal.SIGTERM)
        for r in stopped.values():
            signals.send(r, signal.SIGCONT)
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            remaining, ended = remaining_owned(process_table(), records, owned)
            if not remaining:
                break
            for pid, r in remaining.items():
                if pid not in owned:
                    # Exact prior ancestry/run checks above still apply. Never
                    # allow a late descendant to escape as a reparented orphan.
                    owned[pid] = r
                    signals.send(r, signal.SIGTERM)
            time.sleep(.2)
        remaining, ended = remaining_owned(process_table(), records, owned)
        for r in remaining.values():
            signals.send(r, signal.SIGKILL)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            remaining, ended = remaining_owned(process_table(), records, owned)
            if not remaining:
                break
            time.sleep(.2)
        remaining, ended = remaining_owned(process_table(), records, owned)
        require(not remaining, 'Owned processes remain after exact signals')
        result['historical_identity_terminal_checks'] = ended
        check_protected(process_table(), records)
        # GPU release can lag process termination, but no threshold is relaxed.
        deadline = time.monotonic() + 15
        while True:
            final_gpus, final_apps, final_raw = checked_gpu()
            try:
                result['released_gpu'] = check_gpu_ownership(final_gpus, final_apps, owned, released=True)
                break
            except Hold:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(.5)
        # Refresh after the release polling interval, not just before it.
        check_protected(process_table(), records)
        result.update(status='STOPPED_AND_PRESERVED', accepted=True,
                      final_gpu=final_raw, final_protected=[read_process(pid) for pid in PROTECTED],
                      stopped_all_original_archive=True, automatic_followon_prevented=True)
    except BaseException as exc:
        result.update(status='HOLD_STOP_INCOMPLETE', accepted=False,
                      error_type=type(exc).__name__, error=str(exc),
                      note='Never infer success. Any remaining quiesced owner must be handled using this exact receipt; no automatic supervisor resume.',
                      stopped_by_this_helper=list(stopped.values()))
    finally:
        result['signal_events'] = signals.events
        result['signals_executed'] = bool(signals.events)
        result['finished_unix'] = time.time()
        signals.close()
        try:
            write_json(destination / 'result.private.json', result)
        except BaseException as exc:
            result.update(accepted=False, status='HOLD_RESULT_DURABILITY',
                          result_receipt_write_error=str(exc))
        print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if result['accepted'] else 2


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Hold as exc:
        print(json.dumps({'accepted': False, 'status': 'HOLD_PREFLIGHT', 'error': str(exc)}), flush=True)
        raise SystemExit(2)
