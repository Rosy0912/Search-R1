"""Isolated, read-only source collector. Writing is limited to a new evidence archive.

This does not evaluate checkpoints, approve training, release resources, change
training source, signal processes, acquire locks, or repair the frozen supervisor.
The production entry has exactly one existing training identity and literal paths.
"""
import argparse
import csv
import datetime
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys

ROOT = Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct200-author-20261006')
RUN = ROOT / 'runs/formal-3261'
SOURCE = '188f4e5e7b63cf479290d0888a4a075499886750'
CONFIG = '73f6401e818dc93a11c705da2357d0f3d91b2a51609a51e78a599fb7cf416c60'
BINDINGS_SHA = 'e47ebd3b3a9e73619b5bdd0ec700760ed6aa0ea045266854baff76cab73b83c6'
SESSION = Path('/tmp/o4e-3261/ray/session_2026-10-06_03-56-08_188781_1526172/logs')
CGROUP = '1:freezer:/slurm_h100-10/uid_0/job_3261/step_0\n0::/system.slice/slurmd.service\n'
RAY = [
    (1533312, '1829085316', '8350026ae4bc8387c3d3b6880792734dd21e46ad5a9c09e08669a24e'),
    (1533910, '1829086072', '29c1f19308dadf572ed2b91a1792df098eeda628db2a448cfbe0b11c'),
    (1534229, '1829086582', 'cdf5291397084014536684595ad8aa13d96a9ab29489f317c40f23ad'),
    (1534230, '1829086583', 'bd449d4f08caea91153a584d1f1f8f60835126cb5cad09e189f67a33'),
    (1534231, '1829086584', '61fd9986380ffc812d9a409257287b1cfb6ccc121e1097ef1426b6f3'),
    (1534232, '1829086585', 'd6373931557bb2845c7093fcca34f0291f79a73413b2da38e5864c35'),
    (1534233, '1829086586', '306ae17389e9387b1ed0f5b2ebae458e3a8c43eb70fce98e6bd22ffc'),
    (1534234, '1829086587', '4806f8a6b39290499d3d307f830b04abb0fd6d71d2a30979c0783aa9'),
    (1534235, '1829086588', '8e647b685e950f2af7dda89994d1694503aa674b628aeb1c3df842c4'),
]
ORIGINALS = [(pid, start) for pid, start, _ in RAY] + [
    (1518778, '1828996485'), (1531763, '1829074494')]
UUIDS = {
    'GPU-2c3bdc08-b976-0fb5-79bd-04def458046b',
    'GPU-308b13c1-44ea-47b7-7b33-01a928099618',
    'GPU-525a9efc-446e-333d-11ff-e1f2213d15c8',
    'GPU-6e00764e-3916-65f2-436e-568aa0c70fc5',
    'GPU-d61d94cd-3d24-fa07-8761-33961a14d995',
    'GPU-db94211b-42d5-88d0-f626-60e777af3be6',
    'GPU-f3a967b0-0b4a-4d13-0917-0a4f70aa740b',
    'GPU-fca5fcba-4036-0e79-4cb6-ceb9ed121ad5',
}
RUN_MEMBERS = (
    'train.log', 'formal-wrapper.log', 'retriever.log', 'resources.ndjson',
    'job-exit.txt', 'evidence/train-process-exit.json', 'evidence/sampler-stop.json',
    'evidence/resource-data-audit.json', 'evidence/owned-child-cleanup.json',
    'process-finished-review-required.json', 'evidence/leases.json',
)
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'NODE_FAIL',
            'OUT_OF_MEMORY', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}
COMMANDS = {
    'scontrol': ('scontrol', 'show', 'job', '3261', '-o'),
    'sacct': ('sacct', '-j', '3261', '--noheader', '--parsable2',
              '--format=JobIDRaw,State,ExitCode,NodeList,Start,End'),
    'nvml-inventory': ('nvidia-smi',
                       '--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu',
                       '--format=csv,noheader,nounits'),
    'nvml-applications-unfiltered': ('nvidia-smi',
                                    '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                                    '--format=csv,noheader,nounits'),
    'listeners-unfiltered': ('ss', '-lntp'),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def unique_json(raw):
    def pairs(values):
        result = {}
        for key, value in values:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    def bad(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    result = json.loads(raw, object_pairs_hook=pairs, parse_constant=bad)
    # JSON exponent overflow produces inf without parse_constant being invoked.
    json.dumps(result, allow_nan=False)
    return result


def metadata(value):
    return {'device': value.st_dev, 'inode': value.st_ino, 'uid': value.st_uid,
            'mode': value.st_mode, 'bytes': value.st_size,
            'mtime_ns': value.st_mtime_ns, 'ctime_ns': value.st_ctime_ns,
            'nlink': value.st_nlink}


def open_physical(path, flags=os.O_RDONLY, uid=None, mode=0o600):
    """Open each parent and the final component without following any symlink."""
    path = Path(path)
    require(path.is_absolute() and '..' not in path.parts and path.name,
            'Absolute physical path required')
    current = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:-1]:
            nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                          dir_fd=current)
            os.close(current)
            current = nxt
        fd = os.open(path.name, flags | os.O_NOFOLLOW, mode, dir_fd=current)
        try:
            info = os.fstat(fd)
            require(stat.S_ISREG(info.st_mode), 'Source is not a regular file')
            if uid is not None:
                require(info.st_uid == uid, 'Source UID differs')
            return fd
        except BaseException:
            os.close(fd)
            raise
    finally:
        os.close(current)


def physical_metadata(path, uid=None):
    fd = open_physical(path, uid=uid)
    try:
        return metadata(os.fstat(fd))
    finally:
        os.close(fd)


def write_exclusive(path, raw):
    """No replacement, no retry, fsync each original and its directory."""
    path = Path(path)
    fd = open_physical(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, uid=os.getuid())
    try:
        with os.fdopen(fd, 'wb', closefd=False) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(fd)
    finally:
        os.close(fd)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return {'archive_path': str(path), 'bytes': len(raw),
            'sha256': hashlib.sha256(raw).hexdigest()}


def write_json(path, value):
    raw = (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                      allow_nan=False) + '\n').encode()
    return write_exclusive(path, raw)


def copy_original(source, destination, uid=None, chunk_size=4 * 1024 * 1024):
    """Copy the complete size at open, then hash that same FD again.

    Growth/replacement/overwrite is retained as an unstable original attempt;
    even a stable copy does not by itself prove that a writer has exited.
    """
    source, destination = Path(source), Path(destination)
    require(type(chunk_size) is int and chunk_size > 0, 'Positive chunk size required')
    record = {'physical_path': str(source), 'capture_started_utc': utc(),
              'status': 'UNREAD', 'stable_full_bytes': False,
              'phase': 'INFLIGHT_PREFIX', 'closed_writer_lifetime': False}
    fd = out = None
    created = False
    try:
        record['before'] = physical_metadata(source, uid)
        fd = open_physical(source, uid=uid)
        record['opened_fd'] = metadata(os.fstat(fd))
        require(record['before'] == record['opened_fd'], 'Source replaced before open')
        out = open_physical(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, uid=os.getuid())
        created = True
        remaining, captured, digest = record['opened_fd']['bytes'], 0, hashlib.sha256()
        while remaining:
            block = os.read(fd, min(chunk_size, remaining))
            require(bool(block), 'Source shortened during copy')
            view = memoryview(block)
            while view:
                count = os.write(out, view)
                require(count > 0, 'Archive write made no progress')
                view = view[count:]
            digest.update(block)
            captured += len(block)
            remaining -= len(block)
        os.fsync(out)
        os.close(out)
        out = None
        record.update(archive_path=str(destination), bytes=captured,
                      sha256=digest.hexdigest(), complete_prefix_at_open=True)
        os.lseek(fd, 0, os.SEEK_SET)
        second, remaining = hashlib.sha256(), record['opened_fd']['bytes']
        while remaining:
            block = os.read(fd, min(chunk_size, remaining))
            require(bool(block), 'Source shortened during second full hash')
            second.update(block)
            remaining -= len(block)
        record['opened_fd_second_sha256'] = second.hexdigest()
        record['after'] = metadata(os.fstat(fd))
        record['path_after'] = physical_metadata(source, uid)
        require(record['before'] == record['opened_fd'] == record['after'] == record['path_after'],
                'Source metadata changed during capture')
        require(second.hexdigest() == record['sha256'], 'Same FD full bytes changed during capture')
        record.update(status='STABLE_FULL_BYTES_AT_CAPTURE', stable_full_bytes=True)
    except FileNotFoundError as error:
        record.update(status='MISSING_OR_DISAPPEARED', error=str(error))
    except (OSError, ValueError) as error:
        record.update(status='CAPTURE_FAILED_OR_UNSTABLE', error=type(error).__name__ + ': ' + str(error))
    finally:
        if out is not None:
            os.fsync(out)
            os.close(out)
        if fd is not None:
            os.close(fd)
        if created:
            # Preserve the actual attempted copy even on a shortened source/error.
            read_fd = open_physical(destination, uid=os.getuid())
            try:
                digest, size = hashlib.sha256(), 0
                while True:
                    block = os.read(read_fd, chunk_size)
                    if not block:
                        break
                    digest.update(block)
                    size += len(block)
                record['attempted_archive_original'] = {
                    'archive_path': str(destination), 'bytes': size, 'sha256': digest.hexdigest()}
                require(metadata(os.fstat(read_fd)) == physical_metadata(destination, os.getuid()),
                        'Attempted archive changed')
            finally:
                os.close(read_fd)
            directory = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        record['capture_finished_utc'] = utc()
    return record


def pinned_json(path, pin):
    require(re.fullmatch('[0-9a-f]{64}', pin) is not None, 'Invalid full SHA pin')
    fd = open_physical(path, uid=os.getuid())
    try:
        before = metadata(os.fstat(fd))
        require(before['bytes'] <= 8 * 1024 * 1024, 'JSON input too large')
        chunks = []
        while True:
            value = os.read(fd, 1024 * 1024)
            if not value:
                break
            chunks.append(value)
        raw = b''.join(chunks)
        require(len(raw) == before['bytes'] and metadata(os.fstat(fd)) == before
                and physical_metadata(path, os.getuid()) == before, 'Pinned JSON changed')
        require(hashlib.sha256(raw).hexdigest() == pin, 'Pinned JSON bytes differ')
        return unique_json(raw), raw
    finally:
        os.close(fd)


def expected_bindings():
    return [{'pid': pid, 'start_ticks': start, 'cgroup': CGROUP,
             'stdout_fd_target': str(SESSION / f'worker-{worker}-02000000-{pid}.out'),
             'stderr_fd_target': str(SESSION / f'worker-{worker}-02000000-{pid}.err')}
            for pid, start, worker in RAY]


def validate_bindings(value):
    require(type(value.get('version')) is int and value['version'] == 1,
            'Typed reviewed bindings version required')
    require(value.get('training_job_id') == '3261' and value.get('source_commit') == SOURCE,
            'Foreign reviewed binding identity')
    require(value.get('status') == 'PASS_ACTUAL_INFLIGHT_STDERR_AND_SLURM_PREFIX_BYTES_AND_RECORDED_IDENTITY_ONLY',
            'Wrong reviewed binding scope')
    actual = value.get('final_exit_read_path_bindings')
    require(type(actual) is list and len(actual) == 9, 'Exactly nine actual bindings required')
    for current, expected in zip(actual, expected_bindings()):
        require(type(current) is dict and set(current) == set(expected), 'Wrong binding schema')
        require(type(current['pid']) is int and type(current['start_ticks']) is str,
                'Typed PID/start binding required')
        require(current == expected, 'Reviewed FD identity/path differs')
    return actual


def parse_stat(raw, pid):
    text = raw.decode()
    require(text.startswith(str(pid) + ' ('), 'Proc stat PID differs')
    fields = text.rsplit(')', 1)[1].split()
    require(len(fields) > 19 and fields[19].isdigit(), 'Malformed proc start clock')
    return {'state': fields[0], 'start_ticks': fields[19]}


def process_observation(pid, expected_start=None, proc=Path('/proc')):
    result = {'pid': pid, 'expected_start_ticks': expected_start, 'observed_utc': utc()}
    base = proc / str(pid)
    try:
        first_raw = (base / 'stat').read_bytes()
        first = parse_stat(first_raw, pid)
        status_raw = (base / 'status').read_bytes()
        cgroup_raw = (base / 'cgroup').read_bytes()
        second_raw = (base / 'stat').read_bytes()
        second = parse_stat(second_raw, pid)
        require(first['start_ticks'] == second['start_ticks'], 'PID reused during observation')
        match = re.search(rb'^Uid:\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$', status_raw, re.M)
        require(match is not None, 'Missing typed four-UID process identity')
        uids = [int(v) for v in match.groups()]
        cgroup = cgroup_raw.decode()
        result.update(start_ticks=second['start_ticks'], state=second['state'],
                      uids=uids, cgroup=cgroup, in_training_job=bool(
                          re.search(r'(?:^|/)job_3261(?:/|$)', cgroup, re.M)),
                      raw={'stat_before': first_raw.decode(), 'stat_after': second_raw.decode(),
                           'status': status_raw.decode(), 'cgroup': cgroup})
        if expected_start is not None and second['start_ticks'] == expected_start:
            require(uids == [0, 0, 0, 0] and result['in_training_job'],
                    'Original start matched but UID/cgroup identity differs')
        if expected_start is not None and second['start_ticks'] != expected_start:
            result['status'] = 'PID_REUSED_ORIGINAL_GONE'
        elif second['state'] in ('Z', 'X'):
            result['status'] = 'ORIGINAL_EXITED_ZOMBIE'
        else:
            result['status'] = 'ORIGINAL_OR_CURRENT_PROCESS_ALIVE'
    except FileNotFoundError:
        result['status'] = 'ABSENT_AT_OBSERVATION'
    except (OSError, ValueError, UnicodeError) as error:
        result.update(status='UNAVAILABLE_IDENTITY_NOT_EXIT_PROOF', error=str(error))
    return result


def scan_processes_and_writers(target_metadata, originals=ORIGINALS, proc=Path('/proc')):
    """Read-only /proc inventory. Permission/inconsistent errors prevent closure.

    Every process's open FD is compared to explicit target device/inode. A matching
    writable FD is evidence of a live writer, including a foreign descendant.
    No resource release conclusion is based only on known PID absence.
    """
    result = {'started_utc': utc(), 'visibility_complete': True, 'errors': [],
              'original_processes': [process_observation(pid, start, proc) for pid, start in originals],
              'training_job_processes': [], 'writers': {key: [] for key in target_metadata}}
    targets = {}
    for role, info in target_metadata.items():
        targets.setdefault((info['device'], info['inode']), []).append(role)
    try:
        children = sorted((p for p in proc.iterdir() if p.name.isdigit()), key=lambda p: int(p.name))
    except OSError as error:
        result['visibility_complete'] = False
        result['errors'].append(str(error))
        children = []
    for base in children:
        pid = int(base.name)
        identity = process_observation(pid, proc=proc)
        if identity['status'] == 'UNAVAILABLE_IDENTITY_NOT_EXIT_PROOF':
            result['visibility_complete'] = False
            result['errors'].append({'pid': pid, 'identity': identity})
        if identity.get('in_training_job'):
            result['training_job_processes'].append(identity)
        try:
            for fd_path in (base / 'fd').iterdir():
                try:
                    first = os.stat(fd_path)
                    roles = targets.get((first.st_dev, first.st_ino))
                    if not roles:
                        continue
                    info = (base / 'fdinfo' / fd_path.name).read_text()
                    match = re.search(r'^flags:\s+([0-7]+)\s*$', info, re.M)
                    require(match is not None, 'Matching FD flags unavailable')
                    second = os.stat(fd_path)
                    require((first.st_dev, first.st_ino) == (second.st_dev, second.st_ino),
                            'Matching FD changed during observation')
                    access = int(match.group(1), 8) & os.O_ACCMODE
                    if access in (os.O_WRONLY, os.O_RDWR):
                        for role in roles:
                            result['writers'][role].append({'pid': pid, 'fd': fd_path.name,
                                'flags_octal': match.group(1), 'identity': identity})
                except OSError as error:
                    if error.errno not in (errno.ENOENT, errno.ESRCH):
                        raise
        except (OSError, ValueError) as error:
            if not isinstance(error, OSError) or error.errno not in (errno.ENOENT, errno.ESRCH):
                result['visibility_complete'] = False
                result['errors'].append({'pid': pid, 'fd_visibility_error': str(error)})
    result['finished_utc'] = utc()
    return result


def all_originals_gone(scan):
    return bool(scan.get('original_processes')) and all(
        v['status'] in ('ABSENT_AT_OBSERVATION', 'PID_REUSED_ORIGINAL_GONE', 'ORIGINAL_EXITED_ZOMBIE')
        for v in scan['original_processes'])


def scheduler_fields(raw):
    text = raw.decode()
    fields = {}
    for key, value in re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S*)', text):
        require(key not in fields, 'Duplicate scheduler field')
        fields[key] = value
    require(fields.get('JobId') == '3261', 'Scheduler job differs')
    require(fields.get('NodeList') == 'h100-10', 'Scheduler node differs')
    require(re.fullmatch(r'[^\s]+\(0\)', fields.get('UserId', '')) is not None,
            'Scheduler UID differs')
    require(re.fullmatch(r'[0-9]+:[0-9]+', fields.get('ExitCode', '')) is not None,
            'Actual scheduler exit code required')
    return {'state': fields.get('JobState'), 'terminal': fields.get('JobState') in TERMINAL,
            'exit_code': fields['ExitCode'], 'fields': fields}


def closure_for(record, role, before, after, scheduler_before, scheduler_after):
    reasons = []
    if not record.get('stable_full_bytes'):
        reasons.append('FILE_COPY_MISSING_FAILED_OR_UNSTABLE')
    if record.get('initial_inode_matches_opened') is not True:
        reasons.append('PRESCAN_TARGET_INODE_DIFFERS_FROM_OPENED_SOURCE')
    if not (scheduler_before.get('terminal') is True and scheduler_after.get('terminal') is True
            and scheduler_before.get('state') == scheduler_after.get('state')
            and scheduler_before.get('exit_code') == scheduler_after.get('exit_code')):
        reasons.append('DOUBLE_ACTUAL_TERMINAL_NOT_PROVEN')
    for label, scan in (('before', before), ('after', after)):
        if not scan.get('visibility_complete'):
            reasons.append(label.upper() + '_PROC_VISIBILITY_INCOMPLETE')
        if not all_originals_gone(scan):
            reasons.append(label.upper() + '_ORIGINAL_PROCESS_EXIT_NOT_PROVEN')
        if any(v.get('state') not in ('Z', 'X') for v in scan.get('training_job_processes', [])):
            reasons.append(label.upper() + '_TRAINING_CGROUP_PROCESS_STILL_ALIVE')
        if role not in scan.get('writers', {}) or scan['writers'][role]:
            reasons.append(label.upper() + '_WRITE_FD_ABSENCE_NOT_PROVEN')
    return {'phase': 'INFLIGHT_PREFIX' if reasons else 'CLOSED_WRITER_LIFETIME',
            'closed_writer_lifetime': not reasons, 'closure_unproven_reasons': reasons}


def command_original(name, archive, stage, timeout=30):
    require(name in COMMANDS and stage in ('before', 'after'), 'Command not allowed')
    argv = list(COMMANDS[name])
    result = {'argv': argv, 'started_utc': utc(), 'status': 'NOT_EXECUTED'}
    stdout = stderr = b''
    try:
        completed = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   timeout=timeout, check=False)
        stdout, stderr = completed.stdout, completed.stderr
        result.update(status='COMMAND_RETURNED', returncode=completed.returncode)
    except subprocess.TimeoutExpired as error:
        stdout, stderr = error.stdout or b'', error.stderr or b''
        result.update(status='COMMAND_TIMED_OUT', returncode=None)
    except OSError as error:
        result.update(status='COMMAND_UNAVAILABLE', returncode=None, error=str(error))
    result['stdout'] = write_exclusive(archive / f'{stage}-{name}.stdout.original', stdout)
    result['stderr'] = write_exclusive(archive / f'{stage}-{name}.stderr.original', stderr)
    result['finished_utc'] = utc()
    return result, stdout


def command_owner_identities(raw, kind, proc=Path('/proc'), command=None):
    """Join current positive PIDs without attributing a reused PID to old training.

    This subsequent proc read is not atomic with the command's original snapshot.
    Missing/malformed query data is never treated as a clear-resource proof.
    """
    result = {'scope': 'CURRENT_PID_START_UID_CGROUP_AFTER_RAW_QUERY_NOT_ATOMIC_RESOURCE_RELEASE',
              'status': 'UNAVAILABLE_OR_UNPARSED', 'owners': []}
    try:
        require(kind in ('nvml', 'listeners'), 'Unlisted owner query kind')
        if command is not None:
            require(command.get('status') == 'COMMAND_RETURNED' and command.get('returncode') == 0,
                    'Original owner query did not return zero; empty output is not release evidence')
        text = raw.decode()
        pids = set()
        if kind == 'nvml':
            for row in csv.reader(text.splitlines()):
                require(len(row) == 3 and re.fullmatch(r'[1-9][0-9]*', row[1].strip()) is not None,
                        'Malformed original unfiltered NVML application PID')
                pids.add(int(row[1]))
        else:
            pids.update(int(v) for v in re.findall(r'\bpid=([1-9][0-9]*)\b', text))
        known = dict(ORIGINALS)
        result['owners'] = [process_observation(pid, known.get(pid), proc) for pid in sorted(pids)]
        result['status'] = 'CURRENT_IDENTITIES_OBSERVED'
    except (ValueError, UnicodeError) as error:
        result['error'] = str(error)
    return result


def scheduler_observation(command, raw, watcher=None):
    try:
        require(command.get('status') == 'COMMAND_RETURNED' and command.get('returncode') == 0,
                'Fresh scontrol command did not return zero')
        result = scheduler_fields(raw)
        result['origin'] = 'FRESH_SCONTROL_ORIGINAL'
        return result
    except (ValueError, UnicodeError) as error:
        result = {'terminal': False, 'state': None, 'exit_code': None, 'error': str(error)}
        # A contradictory successful scheduler reply cannot be replaced with an
        # older watcher reply. The watcher is a fallback for unavailable scontrol.
        if watcher is not None and not (command.get('status') == 'COMMAND_RETURNED'
                                        and command.get('returncode') == 0):
            try:
                fields = scheduler_fields(watcher)
                require(fields['terminal'], 'Watcher original is not terminal')
                fields['origin'] = 'PINNED_ACTUAL_WATCHER_TERMINAL_ORIGINAL_FRESH_SCONTROL_UNAVAILABLE'
                fields['fresh_scontrol_error'] = str(error)
                return fields
            except (ValueError, UnicodeError) as watcher_error:
                result['watcher_error'] = str(watcher_error)
        return result


def source_specs(bindings):
    specs = {name.replace('/', '--'): RUN / name for name in RUN_MEMBERS}
    specs.update({'slurm-formal-3261.out': ROOT / 'logs/formal-3261.out',
                  'slurm-formal-3261.err': ROOT / 'logs/formal-3261.err'})
    for item in bindings:
        for descriptor, suffix in (('stdout_fd_target', 'stdout'), ('stderr_fd_target', 'stderr')):
            specs[f'ray-pid-{item["pid"]}-{suffix}'] = Path(item[descriptor])
    require(len(specs) == 31 and len(set(specs.values())) == 31, 'Exact 31 distinct original paths required')
    return specs


def copied_json(record):
    require(record.get('stable_full_bytes') is True, 'JSON original unavailable/unstable')
    value, _ = pinned_json(record['archive_path'], record['sha256'])
    return value


def exit_layers(files):
    result = {'scope': 'RAW_SEPARATE_EXIT_LAYERS_NO_SUCCESS_OR_RESTORE_RECEIPT',
              'training_entry_process': {'status': 'PENDING_OR_UNAVAILABLE'},
              'formal_wrapper_shell': {'status': 'PENDING_OR_UNAVAILABLE'},
              'resource_audit_join': {'status': 'PENDING_OR_UNAVAILABLE'}}
    try:
        original = copied_json(files['evidence--train-process-exit.json'])
        require(type(original) is dict and set(original) == {'returncode'}
                and type(original['returncode']) is int, 'Typed actual child returncode required')
        result['training_entry_process'] = {'status': 'OBSERVED', 'returncode': original['returncode'],
            'original_sha256': files['evidence--train-process-exit.json']['sha256']}
    except (OSError, ValueError, KeyError) as error:
        result['training_entry_process']['error'] = str(error)
    try:
        record = files['job-exit.txt']
        require(record.get('stable_full_bytes'), 'Actual wrapper exit original missing/unstable')
        fd = open_physical(record['archive_path'], uid=os.getuid())
        try:
            raw = os.read(fd, 256)
        finally:
            os.close(fd)
        require(hashlib.sha256(raw).hexdigest() == record['sha256']
                and re.fullmatch(rb'[0-9]+\n', raw) is not None, 'Malformed wrapper exit original')
        result['formal_wrapper_shell'] = {'status': 'OBSERVED', 'returncode': int(raw.strip()),
                                         'original_sha256': record['sha256']}
    except (OSError, ValueError, KeyError) as error:
        result['formal_wrapper_shell']['error'] = str(error)
    try:
        audit = copied_json(files['evidence--resource-data-audit.json'])
        actual = files['resources.ndjson']
        require(actual.get('stable_full_bytes') is True and audit.get('sha256') == actual['sha256'],
                'Full original resources SHA does not match frozen resource audit')
        result['resource_audit_join'] = {'status': 'OBSERVED_FULL_BYTE_SHA_MATCH',
            'resources_sha256': actual['sha256'], 'resources_bytes': actual['bytes'],
            'audit_original_sha256': files['evidence--resource-data-audit.json']['sha256']}
    except (OSError, ValueError, KeyError) as error:
        result['resource_audit_join']['error'] = str(error)
    return result


def lease_paths(files):
    try:
        leases = copied_json(files['evidence--leases.json'])
        require(type(leases) is list and len(leases) == 8, 'Exactly eight actual lease records required')
        for row in leases:
            require(type(row) is dict and set(row) == {'fd', 'uuid'} and type(row['fd']) is int
                    and row['fd'] >= 0 and type(row['uuid']) is str, 'Typed recorded lease fields required')
        require({v['uuid'] for v in leases} == UUIDS and len({v['fd'] for v in leases}) == 8,
                'Actual original eight lease UUIDs/FDs differ')
        return [Path('/tmp/search-r1-gpu-leases-0') / (uuid + '.lock') for uuid in sorted(UUIDS)] + [
            Path('/mnt/data/shaoshitong/search-r1-rl/opsd/gpu-job.lock')]
    except (OSError, ValueError, KeyError):
        return []


def lock_observation(paths, archive, stage, proc=Path('/proc')):
    result = {'scope': 'READ_ONLY_PROC_LOCKS_AND_LITERAL_LEASE_INODES_NO_FLOCK_PROBE',
              'paths': [], 'status': 'UNAVAILABLE', 'started_utc': utc(),
              'timing_scope': 'Independent current lock samples after log captures; stage is sample label'}
    try:
        raw = (proc / 'locks').read_bytes()
        result['original'] = write_exclusive(archive / f'{stage}-proc-locks.original', raw)
        result['status'] = 'RAW_CAPTURED'
        lock_pids = {int(v) for v in re.findall(
            r'\b(?:FLOCK|POSIX|OFDLCK)\s+\S+\s+\S+\s+([1-9][0-9]*)\s+[0-9a-fA-F]+:[0-9a-fA-F]+:[0-9]+', raw.decode())}
        known = dict(ORIGINALS)
        result['current_lock_owner_identities'] = [
            process_observation(pid, known.get(pid), proc) for pid in sorted(lock_pids)]
        result['owner_identity_scope'] = 'Subsequent proc read; OFD negative-PID/unknown owners retained in raw text'
        for path in paths:
            row = {'physical_path': str(path), 'status': 'UNAVAILABLE'}
            try:
                before = physical_metadata(path, uid=os.getuid())
                after = physical_metadata(path, uid=os.getuid())
                require(before == after, 'Lock file changed during metadata observation')
                row.update(status='OBSERVED_STABLE_INODE', before=before, after=after)
                wanted = (os.major(before['device']), os.minor(before['device']), before['inode'])
                matches = []
                for line in raw.decode().splitlines():
                    for token in line.split():
                        match = re.fullmatch(r'([0-9a-fA-F]+):([0-9a-fA-F]+):([0-9]+)', token)
                        if match and (int(match[1], 16), int(match[2], 16), int(match[3])) == wanted:
                            matches.append(line)
                row['matching_original_lock_lines'] = matches
            except (OSError, ValueError, UnicodeError) as error:
                row['error'] = str(error)
            result['paths'].append(row)
    except OSError as error:
        result['error'] = str(error)
    result['finished_utc'] = utc()
    return result


def create_archive(path):
    path = Path(path)
    require(path.parent == RUN / 'evidence' and
            re.fullmatch(r'final-archive-[A-Za-z0-9][A-Za-z0-9_.-]{0,100}', path.name) is not None,
            'A new literal-run evidence archive is required')
    require(path.parent.resolve(strict=True) == path.parent and path.parent.stat().st_uid == 0,
            'Physical own evidence parent required')
    os.mkdir(path, 0o700)
    require(path.resolve(strict=True) == path and path.stat().st_uid == os.getuid(),
            'Archive directory identity differs')
    return path


def main():
    require(sys.flags.optimize == 0, 'Optimized Python is not accepted')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bindings-review', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--watcher-terminal-original')
    parser.add_argument('--watcher-terminal-sha256')
    args = parser.parse_args()
    require(os.getuid() == 0 and socket.gethostname().split('.')[0] == 'h100-10',
            'Only the current owned h100-10 UID0 collection route is accepted')
    review, review_raw = pinned_json(args.bindings_review, BINDINGS_SHA)
    bindings = validate_bindings(review)
    require(bool(args.watcher_terminal_original) == bool(args.watcher_terminal_sha256),
            'Watcher original and its actual SHA must be supplied together')
    watcher = None
    watcher_record = {'status': 'NOT_SUPPLIED_PENDING_OR_ABSENT', 'terminal_claim': False}
    if args.watcher_terminal_original:
        watcher_path = Path(args.watcher_terminal_original)
        require(watcher_path.parent == ROOT / 'evidence/terminal-inputs' and
                re.fullmatch(r'job3261-slurm-final-[A-Za-z0-9_.-]+\.original', watcher_path.name) is not None,
                'Only a real uploaded task watcher original path is accepted')
        fd = open_physical(watcher_path, uid=0)
        try:
            info = metadata(os.fstat(fd))
            require(info['bytes'] <= 1024 * 1024, 'Watcher original unexpectedly large')
            watcher = os.read(fd, 1024 * 1024 + 1)
            require(len(watcher) == info['bytes'] and metadata(os.fstat(fd)) == info
                    and physical_metadata(watcher_path, 0) == info, 'Watcher original changed')
            require(re.fullmatch('[0-9a-f]{64}', args.watcher_terminal_sha256) is not None
                    and hashlib.sha256(watcher).hexdigest() == args.watcher_terminal_sha256,
                    'Watcher actual original SHA differs')
            fields = scheduler_fields(watcher)
            require(fields['terminal'], 'Uploaded watcher original is not terminal')
            watcher_record = {'status': 'ACTUAL_PINNED_TERMINAL_ORIGINAL', 'physical_path': str(watcher_path),
                              'sha256': args.watcher_terminal_sha256, 'scheduler': fields}
        finally:
            os.close(fd)
    archive = create_archive(args.output)
    archive_identity = metadata(archive.stat())
    started = utc()
    bindings_original = write_exclusive(archive / 'bindings-review.original.json', review_raw)
    if watcher is not None:
        watcher_record['archived_original'] = write_exclusive(archive / 'watcher-terminal.original', watcher)
    specs = source_specs(bindings)
    targets, stat_errors = {}, {}
    for role, path in specs.items():
        try:
            targets[role] = physical_metadata(path, uid=0)
        except (OSError, ValueError) as error:
            stat_errors[role] = str(error)
    before_commands, before_raw = {}, {}
    for name in COMMANDS:
        before_commands[name], before_raw[name] = command_original(name, archive, 'before')
    before_owners = {
        'nvml': command_owner_identities(before_raw['nvml-applications-unfiltered'], 'nvml',
                                        command=before_commands['nvml-applications-unfiltered']),
        'listeners': command_owner_identities(before_raw['listeners-unfiltered'], 'listeners',
                                             command=before_commands['listeners-unfiltered'])}
    scheduler_before = scheduler_observation(before_commands['scontrol'], before_raw['scontrol'], watcher)
    before = scan_processes_and_writers(targets)
    before_original = write_json(archive / 'proc-before.original.json', before)
    files = {role: copy_original(path, archive / (role + '.original'), uid=0)
             for role, path in specs.items()}
    for role, record in files.items():
        initial, opened_info = targets.get(role), record.get('opened_fd')
        record['initial_inode_matches_opened'] = bool(initial is not None and opened_info is not None
            and all(initial[key] == opened_info[key] for key in ('device', 'inode', 'uid')))
    # Rebind postscan targets to the opened original inode, not a possibly replaced pathname.
    opened = {role: value['opened_fd'] for role, value in files.items() if 'opened_fd' in value}
    after = scan_processes_and_writers(opened)
    after_original = write_json(archive / 'proc-after.original.json', after)
    after_commands, after_raw = {}, {}
    for name in COMMANDS:
        after_commands[name], after_raw[name] = command_original(name, archive, 'after')
    after_owners = {
        'nvml': command_owner_identities(after_raw['nvml-applications-unfiltered'], 'nvml',
                                        command=after_commands['nvml-applications-unfiltered']),
        'listeners': command_owner_identities(after_raw['listeners-unfiltered'], 'listeners',
                                             command=after_commands['listeners-unfiltered'])}
    scheduler_after = scheduler_observation(after_commands['scontrol'], after_raw['scontrol'], watcher)
    for role, record in files.items():
        record.update(closure_for(record, role, before, after, scheduler_before, scheduler_after))
    paths = lease_paths(files)
    # Two current, unfiltered raw lock reads. No lock is acquired or released.
    lock_before = lock_observation(paths, archive, 'before')
    lock_after = lock_observation(paths, archive, 'after')
    manifest = {'version': 1, 'kind': 'ACTUAL3261_ORIGINAL_FINAL_ARCHIVE_OBSERVATION',
        'scope': 'RAW_BYTES_AND_EXIT_RESOURCE_OBSERVATIONS_NOT_TRAINING_EFFECT_OR_RESTORE_ACCEPTANCE',
        'source_commit': SOURCE, 'config_sha256': CONFIG, 'training_job_id': '3261',
        'source_binding_scope': 'EXPECTED_IDENTITY_FROM_REVIEWED_FD_BINDINGS_NOT_NEW_GIT_CLEAN_OR_CONFIG_VERIFICATION',
        'run_root': str(RUN), 'hostname': socket.gethostname(), 'reader_uid': os.getuid(),
        'reader_pid': os.getpid(), 'reader_identity': process_observation(os.getpid()),
        'capture_started_utc': started, 'capture_finished_utc': utc(),
        'reviewed_fd_binding_sha256': BINDINGS_SHA, 'original_process_bindings': bindings,
        'bindings_review_original': bindings_original,
        'watcher_terminal_original': watcher_record, 'initial_stat_errors': stat_errors,
        'files': files, 'commands_before': before_commands, 'commands_after': after_commands,
        'current_command_owner_identities': {'before': before_owners, 'after': after_owners},
        'scheduler_before': scheduler_before, 'scheduler_after': scheduler_after,
        'process_scan_before': before, 'process_scan_after': after,
        'process_scan_originals': {'before': before_original, 'after': after_original},
        'lease_lock_observations': [lock_before, lock_after], 'exit_layers': exit_layers(files),
        'closed_writer_members': [role for role, value in files.items() if value['closed_writer_lifetime']],
        'missing_or_unstable_members': [role for role, value in files.items() if not value['stable_full_bytes']],
        'release_accepted': False, 'training_or_effect_accepted': False,
        'limitations': [
            'Only exact original paths are copied; no reconstruction or normalization of training markers.',
            'Stable full bytes at capture alone are only an in-flight prefix until terminal and writer checks close.',
            'A source file changed later cannot be excluded forever by a finite read-only observation.',
            'Raw NVML, listeners, locks, process identity and Slurm observations require separate independent review.',
            'Training child exit and supervisor/Slurm exit remain separate, including child zero with wrapper FAILED.',
            'No checkpoint content/HF load/Adam recovery/scientific effect acceptance is performed here.',
        ]}
    current_archive = metadata(archive.stat())
    require(all(current_archive[k] == archive_identity[k] for k in ('device', 'inode', 'uid', 'mode'))
            and archive.resolve(strict=True) == archive, 'Exclusive archive directory replaced')
    manifest['archive_directory_identity'] = {
        k: current_archive[k] for k in ('device', 'inode', 'uid', 'mode')}
    saved = write_json(archive / 'manifest.json', manifest)
    print(json.dumps({'status': 'ARCHIVE_OBSERVATION_WRITTEN_NOT_ACCEPTANCE',
                      'manifest': saved, 'missing_or_unstable': manifest['missing_or_unstable_members'],
                      'closed_writer_member_count': len(manifest['closed_writer_members'])}, sort_keys=True))


if __name__ == '__main__':
    main()
