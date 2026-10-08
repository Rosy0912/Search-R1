"""Stream reviewed collector originals to a fresh local directory.

This is a byte-transfer client, not a terminal, release, checkpoint, restore or
effect validator. Production only uses the existing read-only transport route.
Partial bytes, actual stderr and actual transport exit codes are preserved.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import shlex
import socket
import stat
import subprocess
import sys
import time
import types

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
TRANSPORT_SHA = 'a64bf52a97cbef6e3b0cc7948e095546687c8f1690094469c4937180f9449a85'
COLLECTOR_SHA = 'd83ef9dcbc950978a96b280789574917b0c8a81148c52d73ea1109a22f828406'
REVIEW_SHA = '62826381f10160bc6452fe8b26a9338b586f0f3a9142b43b4767a05a58b5aed4'
BINDINGS_SHA = 'e47ebd3b3a9e73619b5bdd0ec700760ed6aa0ea045266854baff76cab73b83c6'
ROOT = Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct200-author-20261006')
RUN = ROOT / 'runs/formal-3261'
SOURCE = '188f4e5e7b63cf479290d0888a4a075499886750'
CONFIG = '73f6401e818dc93a11c705da2357d0f3d91b2a51609a51e78a599fb7cf416c60'
SCOPE = 'RAW_BYTES_AND_EXIT_RESOURCE_OBSERVATIONS_NOT_TRAINING_EFFECT_OR_RESTORE_ACCEPTANCE'
CERT = 'AUTHOR200_STREAM_READ_CERTIFICATE '
CHUNK = 1024 * 1024
TOP_KEYS = set(('version kind scope source_commit config_sha256 training_job_id '
 'source_binding_scope run_root hostname reader_uid reader_pid reader_identity '
 'capture_started_utc capture_finished_utc reviewed_fd_binding_sha256 original_process_bindings '
 'bindings_review_original watcher_terminal_original initial_stat_errors files commands_before '
 'commands_after current_command_owner_identities scheduler_before scheduler_after process_scan_before '
 'process_scan_after process_scan_originals lease_lock_observations exit_layers closed_writer_members '
 'missing_or_unstable_members release_accepted training_or_effect_accepted limitations '
 'archive_directory_identity').split())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def integer(value, minimum=0):
    require(type(value) is int and value >= minimum, 'Typed nonnegative integer required')
    return value


def digest(value):
    require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value), 'Full SHA256 required')
    return value


def unique_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    def bad(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=bad)
    json.dumps(value, allow_nan=False)
    return value


def metadata(info):
    return dict(device=info.st_dev, inode=info.st_ino, uid=info.st_uid,
                mode=info.st_mode, bytes=info.st_size, mtime_ns=info.st_mtime_ns,
                ctime_ns=info.st_ctime_ns, nlink=info.st_nlink)


def open_directory(path, uid):
    path = Path(path)
    require(path.is_absolute() and '..' not in path.parts, 'Absolute physical path required')
    fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:]:
            nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = nxt
        require(os.fstat(fd).st_uid == uid, 'Directory owner differs')
        return fd
    except BaseException:
        os.close(fd)
        raise


def open_file(path, uid, flags=os.O_RDONLY):
    path = Path(path)
    parent = open_directory(path.parent, uid)
    try:
        fd = os.open(path.name, flags | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600, dir_fd=parent)
        info = os.fstat(fd)
        require(stat.S_ISREG(info.st_mode) and info.st_uid == uid and info.st_nlink == 1,
                'Single-link owned regular file required')
        return fd
    except BaseException:
        if 'fd' in locals():
            os.close(fd)
        raise
    finally:
        os.close(parent)


def stable_raw(path, sha, limit=8 * 1024 * 1024):
    digest(sha)
    fd = open_file(path, os.getuid())
    try:
        before = metadata(os.fstat(fd))
        require(before['bytes'] <= limit, 'Pinned local input too large')
        chunks = []
        while True:
            block = os.read(fd, CHUNK)
            if not block:
                break
            chunks.append(block)
        raw = b''.join(chunks)
        check = open_file(path, os.getuid())
        try:
            require(metadata(os.fstat(fd)) == before == metadata(os.fstat(check)),
                    'Pinned local file changed')
        finally:
            os.close(check)
        require(len(raw) == before['bytes'] and hashlib.sha256(raw).hexdigest() == sha,
                'Pinned local bytes differ')
        return raw
    finally:
        os.close(fd)


def load_helpers():
    definitions = [('transport', WORK / 'transport.py', TRANSPORT_SHA),
        ('collector', WORK / 'final-archive-v2-candidate/collect_actual3261_final_originals.py', COLLECTOR_SHA)]
    modules = []
    for name, path, pin in definitions:
        raw = stable_raw(path, pin)
        module = types.ModuleType('stream_verified_' + name)
        module.__file__ = str(path)
        exec(compile(raw, str(path), 'exec'), module.__dict__)
        modules.append(module)
    peer = unique_json(stable_raw(WORK / 'FINAL_ARCHIVE_COLLECTOR_V2_INDEPENDENT_SOFTWARE_REVIEW.json', REVIEW_SHA))
    require(peer['status'] == 'PASS_INDEPENDENT_LOCAL_FINAL_ORIGINAL_ARCHIVE_COLLECTOR_SOFTWARE_ONLY'
            and peer['runtime']['sha256'] == COLLECTOR_SHA, 'Collector peer scope/pin differs')
    binding = unique_json(stable_raw(WORK / 'ACTUAL3261_INFLIGHT_STDERR_SLURM_PREFIX_INDEPENDENT_REVIEW.json', BINDINGS_SHA))
    modules[1].validate_bindings(binding)
    return modules[0], modules[1]


def validate_manifest_path(path):
    path = Path(path)
    archive = path.parent
    require(path.is_absolute() and '..' not in path.parts and path.name == 'manifest.json'
        and archive.parent == RUN / 'evidence'
        and re.fullmatch(r'final-archive-[A-Za-z0-9][A-Za-z0-9_.-]{0,100}', archive.name),
        'Only an actual literal3261 collector manifest is accepted')
    return archive


def file_record(value, archive, filename):
    require(type(value) is dict and set(value) == {'archive_path', 'bytes', 'sha256'},
            'Exact archive byte record required')
    require(value['archive_path'] == str(archive / filename), 'Archived copy path differs')
    integer(value['bytes'])
    digest(value['sha256'])
    return dict(value)


def validate_manifest(value, path, collector):
    archive = validate_manifest_path(path)
    require(type(value) is dict and set(value) == TOP_KEYS, 'Collector manifest schema differs')
    require(type(value['version']) is int and value['version'] == 1
        and value['kind'] == 'ACTUAL3261_ORIGINAL_FINAL_ARCHIVE_OBSERVATION'
        and value['scope'] == SCOPE and value['source_commit'] == SOURCE
        and value['config_sha256'] == CONFIG and value['training_job_id'] == '3261'
        and value['run_root'] == str(RUN) and value['reviewed_fd_binding_sha256'] == BINDINGS_SHA
        and value['source_binding_scope'] == 'EXPECTED_IDENTITY_FROM_REVIEWED_FD_BINDINGS_NOT_NEW_GIT_CLEAN_OR_CONFIG_VERIFICATION',
        'Foreign observation identity or scope')
    require(value['release_accepted'] is False and value['training_or_effect_accepted'] is False,
            'Observation cannot assert acceptance')
    require(type(value['reader_uid']) is int and value['reader_uid'] == 0
        and type(value['reader_pid']) is int and value['reader_pid'] > 0
        and type(value['hostname']) is str and value['hostname'].split('.')[0] == 'h100-10',
        'Collector physical reader identity differs')
    bindings = collector.validate_bindings(dict(version=1, training_job_id='3261', source_commit=SOURCE,
        status='PASS_ACTUAL_INFLIGHT_STDERR_AND_SLURM_PREFIX_BYTES_AND_RECORDED_IDENTITY_ONLY',
        final_exit_read_path_bindings=value['original_process_bindings']))
    identity = value['archive_directory_identity']
    require(type(identity) is dict and set(identity) == {'device', 'inode', 'uid', 'mode'},
            'Archive directory identity schema differs')
    for key in identity:
        integer(identity[key])
    require(identity['uid'] == 0 and stat.S_ISDIR(identity['mode']), 'Archive directory owner/type differs')
    specs = collector.source_specs(bindings)
    require(type(value['files']) is dict and set(value['files']) == set(specs),
            'Exactly31 literal source-role records required')
    copies, unavailable = {}, []
    for role, origin in specs.items():
        record = value['files'][role]
        require(type(record) is dict and record.get('physical_path') == str(origin), 'Source origin differs')
        require(type(record.get('stable_full_bytes')) is bool and type(record.get('closed_writer_lifetime')) is bool,
                'Typed source observation booleans required')
        require(record.get('status') in {'STABLE_FULL_BYTES_AT_CAPTURE', 'MISSING_OR_DISAPPEARED', 'CAPTURE_FAILED_OR_UNSTABLE'},
                'Unknown source capture status')
        require(record.get('phase') in {'INFLIGHT_PREFIX', 'CLOSED_WRITER_LIFETIME'}
            and (record['phase'] == 'CLOSED_WRITER_LIFETIME') == record['closed_writer_lifetime'],
            'Observation closure phase inconsistent')
        require((record['status'] == 'STABLE_FULL_BYTES_AT_CAPTURE') == record['stable_full_bytes']
            and (not record['closed_writer_lifetime'] or record['stable_full_bytes']), 'Source status inconsistent')
        attempt = record.get('attempted_archive_original')
        if attempt is not None:
            attempt = file_record(attempt, archive, role + '.original')
        if record['stable_full_bytes']:
            saved = file_record({k: record[k] for k in ('archive_path', 'bytes', 'sha256')}, archive, role + '.original')
            require(attempt == saved, 'Stable copy and attempted copy differ')
            copies[role] = saved
        elif attempt is not None:
            copies[role] = attempt
        else:
            require(not any(k in record for k in ('archive_path', 'bytes', 'sha256')),
                    'Failed copy has unbound body fields')
            unavailable.append(role)
    closed = [role for role, record in value['files'].items() if record['closed_writer_lifetime']]
    unstable = [role for role, record in value['files'].items() if not record['stable_full_bytes']]
    require(type(value['closed_writer_members']) is list and set(value['closed_writer_members']) == set(closed)
        and len(value['closed_writer_members']) == len(closed), 'Closed member declaration differs')
    require(type(value['missing_or_unstable_members']) is list and set(value['missing_or_unstable_members']) == set(unstable)
        and len(value['missing_or_unstable_members']) == len(unstable), 'Unstable member declaration differs')
    copies['bindings-review'] = file_record(value['bindings_review_original'], archive, 'bindings-review.original.json')
    require(copies['bindings-review']['sha256'] == BINDINGS_SHA, 'Binding original SHA differs')
    watcher = value['watcher_terminal_original']
    require(type(watcher) is dict, 'Watcher observation required')
    if watcher.get('status') == 'ACTUAL_PINNED_TERMINAL_ORIGINAL':
        copies['watcher-terminal'] = file_record(watcher['archived_original'], archive, 'watcher-terminal.original')
        require(copies['watcher-terminal']['sha256'] == digest(watcher['sha256']), 'Watcher body pin differs')
        origin = Path(watcher['physical_path'])
        require(origin.parent == ROOT / 'evidence/terminal-inputs'
            and re.fullmatch(r'job3261-slurm-final-[A-Za-z0-9_.-]+\.original', origin.name), 'Watcher origin differs')
    else:
        require(watcher == {'status': 'NOT_SUPPLIED_PENDING_OR_ABSENT', 'terminal_claim': False}, 'Unknown watcher state')
    require(type(value['process_scan_originals']) is dict and set(value['process_scan_originals']) == {'before', 'after'},
            'Process scan original pair required')
    for stage in ('before', 'after'):
        copies['proc-' + stage] = file_record(value['process_scan_originals'][stage], archive, f'proc-{stage}.original.json')
        commands = value['commands_' + stage]
        require(type(commands) is dict and set(commands) == set(collector.COMMANDS), 'Exact original command roles required')
        for name, command in commands.items():
            require(command['argv'] == list(collector.COMMANDS[name]), 'Command origin differs')
            for stream in ('stdout', 'stderr'):
                role = f'{stage}-{name}-{stream}'
                copies[role] = file_record(command[stream], archive, f'{stage}-{name}.{stream}.original')
    require(type(value['lease_lock_observations']) is list and len(value['lease_lock_observations']) == 2,
            'Exact two lock observations required')
    for stage, observation in zip(('before', 'after'), value['lease_lock_observations']):
        if 'original' in observation:
            copies[stage + '-proc-locks'] = file_record(observation['original'], archive, stage + '-proc-locks.original')
        else:
            require(observation.get('status') == 'UNAVAILABLE', 'Missing lock body has unsupported status')
    require(len({v['archive_path'] for v in copies.values()}) == len(copies), 'Archived bodies alias')
    return copies, unavailable, identity


# This complete program is sent as one quoted Python argument. It never writes
# remote files or signals any remote process. Its stdout is only original bytes.
READER_SOURCE = r'''
import hashlib,json,os,socket,stat,sys
def require(ok,message):
 if not ok: raise ValueError(message)
def meta(s):
 return dict(device=s.st_dev,inode=s.st_ino,uid=s.st_uid,mode=s.st_mode,bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,nlink=s.st_nlink)
def directory(path,uid):
 require(path.startswith('/') and '..' not in path.split('/'),'Physical absolute path required')
 fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY)
 try:
  for part in path.split('/')[1:]:
   nxt=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=nxt
  require(os.fstat(fd).st_uid==uid,'Directory UID differs')
  return fd
 except BaseException:
  os.close(fd);raise
def read_original(request):
 require(type(request['bytes']) is int and request['bytes']>=0,'Typed nonnegative bytes required')
 require(type(request['uid']) is int and os.getuid()==request['uid'],'Reader UID differs')
 require(socket.gethostname().split('.')[0]==request['hostname'],'Reader host differs')
 path=request['path'];parent,name=os.path.split(path)
 require(name and name not in ('.','..'),'Literal file required')
 d=directory(parent,request['uid']);fd=None
 try:
  initial_dir=meta(os.fstat(d))
  expected_dir=request['directory_identity']
  if expected_dir is not None:
   require(all(initial_dir[k]==expected_dir[k] for k in ('device','inode','uid','mode')),'Archive directory replaced')
  before=meta(os.stat(name,dir_fd=d,follow_symlinks=False))
  require(stat.S_ISREG(before['mode']) and before['uid']==request['uid'] and before['nlink']==1,'Owned single-link regular original required')
  fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=d)
  opened=meta(os.fstat(fd));require(before==opened and opened['bytes']==request['bytes'],'Original size/identity differs')
  remaining=request['bytes'];digest=hashlib.sha256();copied=0
  while remaining:
   block=os.read(fd,min(1048576,remaining));require(bool(block),'Original shortened')
   digest.update(block);copied+=len(block);remaining-=len(block)
   view=memoryview(block)
   while view:
    count=os.write(1,view);require(count>0,'Stream write made no progress');view=view[count:]
  require(digest.hexdigest()==request['sha256'],'Original full SHA differs')
  os.lseek(fd,0,os.SEEK_SET);second=hashlib.sha256();remaining=request['bytes']
  while remaining:
   block=os.read(fd,min(1048576,remaining));require(bool(block),'Original shortened in rehash')
   second.update(block);remaining-=len(block)
  after=meta(os.fstat(fd));path_after=meta(os.stat(name,dir_fd=d,follow_symlinks=False))
  require(before==opened==after==path_after,'Original metadata changed')
  require(second.hexdigest()==digest.hexdigest(),'Same opened original changed')
  fresh=directory(parent,request['uid'])
  try:
   final_dir=meta(os.fstat(fresh));require(all(initial_dir[k]==final_dir[k] for k in ('device','inode','uid','mode')),'Parent directory replaced')
  finally:os.close(fresh)
  cert=dict(version=1,status='STREAMED_RAW_BYTES_DOUBLE_HASH_STABLE_NOT_ACCEPTANCE',path=path,bytes=copied,sha256=digest.hexdigest(),second_sha256=second.hexdigest(),before=before,opened_fd=opened,after=after,path_after=path_after,directory_identity={k:initial_dir[k] for k in ('device','inode','uid','mode')},reader_uid=os.getuid(),reader_hostname=socket.gethostname(),reader_pid=os.getpid())
  os.write(2,('AUTHOR200_STREAM_READ_CERTIFICATE '+json.dumps(cert,sort_keys=True,allow_nan=False)+'\n').encode())
 finally:
  if fd is not None:os.close(fd)
  os.close(d)
'''


def reader_program(request):
    return READER_SOURCE + '\nread_original(' + repr(request) + ')\n'


def write_exclusive(path, raw):
    fd = open_file(path, os.getuid(), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        view = memoryview(raw)
        while view:
            count = os.write(fd, view)
            require(count > 0, 'Local write made no progress')
            view = view[count:]
        os.fsync(fd)
    finally:
        os.close(fd)
    parent = open_directory(Path(path).parent, os.getuid())
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def transfer_one(argv, request, output, timeout):
    """Actual Popen stream/wait; caller supplies route argv or local fixture argv."""
    require(type(timeout) in (int, float) and 0 < timeout <= 7200, 'Bounded transfer timeout required')
    integer(request['bytes'])
    digest(request['sha256'])
    output = Path(output)
    out = err = child = None
    result = dict(version=1, status='FAILED_TRANSFER_ORIGINAL_BYTES_PRESERVED', request=request,
                  argv=argv, local_original=str(output), stderr_original=str(output) + '.transport.stderr.original',
                  actual_returncode=None, local_bytes=0, local_sha256=None, started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    current_digest = hashlib.sha256()
    parent_identity = None
    try:
        parent_fd = open_directory(output.parent, os.getuid())
        try:
            parent_identity = {k: metadata(os.fstat(parent_fd))[k] for k in ('device', 'inode', 'uid', 'mode')}
            result['local_directory_identity'] = parent_identity
        finally:
            os.close(parent_fd)
        out = open_file(output, os.getuid(), os.O_RDWR | os.O_CREAT | os.O_EXCL)
        err = open_file(result['stderr_original'], os.getuid(), os.O_RDWR | os.O_CREAT | os.O_EXCL)
        child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=err)
        result['local_transport_pid'] = child.pid
        deadline = time.monotonic() + timeout
        selector = selectors.DefaultSelector()
        selector.register(child.stdout, selectors.EVENT_READ)
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('Owned transfer deadline reached')
                if not selector.select(min(remaining, 0.2)):
                    continue
                block = os.read(child.stdout.fileno(), CHUNK)
                if not block:
                    break
                current_digest.update(block)
                result['local_bytes'] += len(block)
                view = memoryview(block)
                while view:
                    count = os.write(out, view)
                    require(count > 0, 'Local streamed write made no progress')
                    view = view[count:]
                require(result['local_bytes'] <= request['bytes'], 'Transport emitted more than pinned size')
        finally:
            selector.close()
        result['actual_returncode'] = child.wait(timeout=max(0.001, deadline - time.monotonic()))
        os.fsync(out)
        os.fsync(err)
        result['local_sha256'] = current_digest.hexdigest()
        require(result['actual_returncode'] == 0, 'Actual transport returned nonzero')
        require(result['local_bytes'] == request['bytes'] and result['local_sha256'] == request['sha256'],
                'Transferred full bytes/SHA differ')
        # Stderr stays on disk; only its final bounded certificate window is read.
        stderr_fd = open_file(result['stderr_original'], os.getuid())
        try:
            os.lseek(stderr_fd, max(0, os.fstat(stderr_fd).st_size - 65536), os.SEEK_SET)
            tail = os.read(stderr_fd, 65536)
        finally:
            os.close(stderr_fd)
        lines = tail.splitlines()
        cert_lines = [line for line in lines if line.startswith(CERT.encode())]
        require(len(cert_lines) == 1 and lines[-1] == cert_lines[0], 'One actual final reader certificate required')
        cert = unique_json(cert_lines[0][len(CERT):])
        require(type(cert['version']) is int and cert['version'] == 1
            and cert['status'] == 'STREAMED_RAW_BYTES_DOUBLE_HASH_STABLE_NOT_ACCEPTANCE'
            and cert['path'] == request['path'] and type(cert['bytes']) is int and cert['bytes'] == request['bytes']
            and cert['sha256'] == cert['second_sha256'] == request['sha256']
            and type(cert['reader_uid']) is int and cert['reader_uid'] == request['uid']
            and cert['reader_hostname'].split('.')[0] == request['hostname']
            and type(cert['reader_pid']) is int and cert['reader_pid'] > 0,
            'Final remote read certificate identity differs')
        require(cert['before'] == cert['opened_fd'] == cert['after'] == cert['path_after'], 'Remote stability evidence differs')
        if request['directory_identity'] is not None:
            require(cert['directory_identity'] == request['directory_identity'], 'Remote archive directory differs')
        local = metadata(os.fstat(out))
        check = open_file(output, os.getuid())
        try:
            require(local == metadata(os.fstat(check)) and local['bytes'] == result['local_bytes'],
                    'Local original replaced')
        finally:
            os.close(check)
        result.update(status='TRANSFERRED_EXACT_ARCHIVED_BYTES_NOT_ACCEPTANCE', reader_certificate=cert)
    except BaseException as error:
        result['error'] = type(error).__name__ + ': ' + str(error)
    finally:
        if child is not None:
            if child.poll() is None:
                child.terminate()  # Only this client's owned local transport child.
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            result['actual_returncode'] = child.wait()
            if child.stdout is not None:
                child.stdout.close()
        # Record the actual created local bytes even after a write/fsync error.
        # This same-FD reread cannot mistake a replaced pathname for our copy.
        for label, fd in (('local', out), ('stderr', err)):
            if fd is None:
                continue
            try:
                os.fsync(fd)
                info = metadata(os.fstat(fd))
                os.lseek(fd, 0, os.SEEK_SET)
                actual_digest, actual_bytes = hashlib.sha256(), 0
                while True:
                    block = os.read(fd, CHUNK)
                    if not block:
                        break
                    actual_digest.update(block)
                    actual_bytes += len(block)
                result[label + '_bytes'] = actual_bytes
                result[label + '_sha256'] = actual_digest.hexdigest()
                require(info == metadata(os.fstat(fd)), 'Local created bytes changed in rehash')
                local_path = output if label == 'local' else result['stderr_original']
                fresh = open_file(local_path, os.getuid())
                try:
                    require(info == metadata(os.fstat(fresh)), 'Created local pathname was replaced')
                finally:
                    os.close(fresh)
                if label == 'local':
                    if result['status'] == 'TRANSFERRED_EXACT_ARCHIVED_BYTES_NOT_ACCEPTANCE':
                        require(actual_bytes == request['bytes'] and actual_digest.hexdigest() == request['sha256'],
                                'Final local opened bytes differ')
                result[label + '_metadata'] = info
            except BaseException as error:
                result['status'] = 'FAILED_TRANSFER_ORIGINAL_BYTES_PRESERVED'
                result[label + '_finalization_error'] = type(error).__name__ + ': ' + str(error)
            finally:
                os.close(fd)
        parent = open_directory(output.parent, os.getuid())
        try:
            if parent_identity is not None:
                require({k: metadata(os.fstat(parent))[k] for k in parent_identity} == parent_identity,
                        'Local output directory was replaced')
            os.fsync(parent)
        except (OSError, ValueError) as error:
            result['status'] = 'FAILED_TRANSFER_ORIGINAL_BYTES_PRESERVED'
            result['directory_fsync_error'] = str(error)
        finally:
            os.close(parent)
        result['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write_exclusive(str(output) + '.transfer.json', (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n').encode())
    return result


def fresh_directory(path):
    path = Path(path)
    parent = open_directory(path.parent, os.getuid())
    try:
        os.mkdir(path.name, 0o700, dir_fd=parent)
        os.fsync(parent)
    finally:
        os.close(parent)
    fd = open_directory(path, os.getuid())
    os.close(fd)
    return path


def main():
    require(sys.flags.optimize == 0, 'Optimized Python not accepted')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest-path', required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--manifest-bytes', required=True, type=int)
    parser.add_argument('--output', required=True)
    parser.add_argument('--timeout', type=int, default=1800)
    args = parser.parse_args()
    archive = validate_manifest_path(args.manifest_path)
    digest(args.manifest_sha256)
    require(0 < integer(args.manifest_bytes, 1) <= 8 * 1024 * 1024, 'Manifest size bound differs')
    transport, collector = load_helpers()
    output = fresh_directory(args.output)
    directory_fd = open_directory(output, os.getuid())
    try:
        local_directory_identity = {k: metadata(os.fstat(directory_fd))[k] for k in ('device', 'inode', 'uid', 'mode')}
    finally:
        os.close(directory_fd)
    report = dict(version=1, kind='LOCAL_STREAMED_ACTUAL3261_ARCHIVE_BYTE_TRANSFER_NOT_ACCEPTANCE',
        source_commit=SOURCE, config_sha256=CONFIG, training_job_id='3261', manifest_path=args.manifest_path,
        manifest_sha256=args.manifest_sha256, manifest_bytes=args.manifest_bytes,
        release_accepted=False, training_or_effect_accepted=False, transfers={}, unavailable_roles=[],
        status='FAILED_TRANSFER_PRESERVING_ACTUAL_ORIGINALS',
        helper_pins=dict(transport=TRANSPORT_SHA, collector=COLLECTOR_SHA, collector_review=REVIEW_SHA))
    report['local_directory_identity'] = local_directory_identity
    def recheck_output():
        fd = open_directory(output, os.getuid())
        try:
            require({k: metadata(os.fstat(fd))[k] for k in local_directory_identity} == local_directory_identity,
                    'Exclusive local evidence directory was replaced')
        finally:
            os.close(fd)
    def transfer(role, request, filename):
        recheck_output()
        command = 'python3 -c ' + shlex.quote(reader_program(request))
        argv = transport.route(command)
        result = transfer_one(argv, request, output / filename, args.timeout)
        report['transfers'][role] = result
        require(result['status'] == 'TRANSFERRED_EXACT_ARCHIVED_BYTES_NOT_ACCEPTANCE', 'Original transfer failed: ' + role)
        recheck_output()
    try:
        request = dict(path=args.manifest_path, bytes=args.manifest_bytes, sha256=args.manifest_sha256,
                       uid=0, hostname='h100-10', directory_identity=None)
        transfer('observation-manifest', request, 'manifest.original.json')
        raw = stable_raw(output / 'manifest.original.json', args.manifest_sha256)
        value = unique_json(raw)
        copies, unavailable, identity = validate_manifest(value, args.manifest_path, collector)
        require(report['transfers']['observation-manifest']['reader_certificate']['directory_identity'] == identity,
                'Manifest archive directory identity differs from actual reader')
        report['unavailable_roles'] = unavailable
        report['observation_missing_or_unstable_members'] = value['missing_or_unstable_members']
        report['observation_closed_writer_members'] = value['closed_writer_members']
        for role, record in copies.items():
            request = dict(path=record['archive_path'], bytes=record['bytes'], sha256=record['sha256'],
                           uid=0, hostname='h100-10', directory_identity=identity)
            transfer(role, request, Path(record['archive_path']).name)
        report['status'] = 'TRANSFERRED_AVAILABLE_EXACT_ARCHIVE_ORIGINALS_NOT_ACCEPTANCE'
    except BaseException as error:
        report['error'] = type(error).__name__ + ': ' + str(error)
    finally:
        # If the directory was replaced, do not write a misleading final report
        # into the replacement. Per-file original records remain on the owned FD.
        recheck_output()
        write_exclusive(output / 'transfer-report.json', (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n').encode())
    # No transport stdout or original body is printed to the conversation.
    return 0 if report['status'] == 'TRANSFERRED_AVAILABLE_EXACT_ARCHIVE_ORIGINALS_NOT_ACCEPTANCE' else 1


if __name__ == '__main__':
    sys.exit(main())
