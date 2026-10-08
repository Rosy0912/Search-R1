"""Scoped resolver compatibility so original cancelled stepd can report completion.

No Slurm state fabrication, service restart, new submission, or training mutation.
The hosts edit is append-only; cleanup comments our own line without rewriting
the file. A detached watchdog is acknowledged before the append is permitted.
"""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import stat
import subprocess
import sys
import time

ROOT = Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct200-safety-20261006')
NS = ROOT / 'logs/human-preempt-searchg1-3329-for-gigpo3332-20261008-v1'
HOSTS = Path('/etc/hosts')
PREFIX = 'LEGACY-CONTROLLER-V2-'
# Ownership marker is an earlier hostname in the SAME alias line. Any short
# write that reaches h100-16 must already contain this complete owned prefix.
OWN_PREFIX = b'10.33.0.8 gigpo-close-3329-once-20261008-v2 '
LINE = OWN_PREFIX + b'h100-16\n'
BINDING = ROOT / 'logs/fresh200-evidence-repair3325-once-v1/BINDING.json'
BINDING_SHA = '2c161a96fd4f1313e1c5de935a98319bf85b9247b4c7632c73afac282ed500ec'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def run(args):
    p = subprocess.run(args, capture_output=True, text=True, timeout=5)
    return dict(returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)


def field(text, key):
    m = re.search(r'(?:^|\s)' + re.escape(key) + r'=([^\s]+)', text)
    return m.group(1) if m else None


def save(name, value):
    final = NS / (PREFIX + name + '.json')
    temporary = NS / (PREFIX + name + '.tmp-' + str(os.getpid()))
    with temporary.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.link(temporary, final)  # Publish complete bytes atomically, never replace.
    temporary.unlink()


def locked_hosts():
    fd = os.open(HOSTS, os.O_RDWR | os.O_APPEND | os.O_NOFOLLOW)
    fcntl.flock(fd, fcntl.LOCK_EX)
    st = os.fstat(fd)
    if not stat.S_ISREG(st.st_mode) or st.st_uid != 0:
        os.close(fd)
        raise RuntimeError('Unexpected hosts type/owner')
    return fd, st


def deactivate_once(plan):
    """Touch only our full unique prefix's first byte; never truncate/replace."""
    fd, st = locked_hosts()
    try:
        data = os.pread(fd, st.st_size, 0)
        if data.count(b'#' + OWN_PREFIX[1:]) == 1 and OWN_PREFIX not in data:
            return dict(status='ALREADY_DEACTIVATED', utc=now())
        if data.count(OWN_PREFIX) == 1:
            offset = data.index(OWN_PREFIX)
            segment = OWN_PREFIX
            # This whole IP + unique hostname prefix came from our append. It
            # cannot be the first byte of somebody's line after a marker-only
            # short write. Commenting our first byte also covers a short tail.
        elif not re.search(rb'(?m)^[^#\n]*\bh100-16\b', data):
            return dict(status='EXTERNAL_REPLACEMENT_HAS_NO_ACTIVE_ALIAS', utc=now())
        else:
            raise RuntimeError('Own line absent/duplicated with active alias; refusing external edits')
        # O_APPEND affects pwrite on Linux: use a second descriptor without it.
        clean_fd = os.open(HOSTS, os.O_RDWR | os.O_NOFOLLOW)
        try:
            other = os.fstat(clean_fd)
            if (other.st_dev, other.st_ino) != (st.st_dev, st.st_ino):
                raise RuntimeError('hosts identity changed before own-byte cleanup')
            if os.pread(clean_fd, len(segment), offset) != segment:
                raise RuntimeError('Own range differs before cleanup')
            if os.pwrite(clean_fd, b'#', offset) != 1:
                raise RuntimeError('Incomplete own-byte cleanup')
            os.fsync(clean_fd)
            if os.pread(clean_fd, len(segment), offset) != b'#' + segment[1:]:
                raise RuntimeError('Own-byte cleanup verification differs')
        finally:
            os.close(clean_fd)
        return dict(status='OWN_LINE_COMMENTED_NO_FILE_REWRITE', utc=now(),
            semantic_restore_only=True, original_bytes_restored=False)
    finally:
        os.close(fd)


def deactivate(plan):
    for attempt in range(5):
        result = deactivate_once(plan)
        fd, st = locked_hosts()
        try:
            data = os.pread(fd, st.st_size, 0)
            if not re.search(rb'(?m)^[^#\n]*\bh100-16\b', data):
                current = HOSTS.stat()
                if (current.st_dev, current.st_ino) == (st.st_dev, st.st_ino):
                    result['current_path_has_no_active_legacy_alias'] = True
                    result['semantic_restore_only'] = True
                    result['original_bytes_restored'] = False
                    return result
        finally:
            os.close(fd)
    raise RuntimeError('Current hosts path still has active alias after bounded cleanup')


def watchdog():
    plan = json.loads((NS / (PREFIX + 'INTENT.json')).read_text())
    save('WATCHDOG-READY', dict(pid=os.getpid(), utc=now(), deadline_epoch=plan['deadline_epoch'],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    while time.time() < plan['deadline_epoch']:
        if (NS / (PREFIX + 'RECEIPT.json')).exists():
            break
        time.sleep(1)
    result = dict(status='RESTORE_FAILED')
    try:
        result = deactivate(plan)
    except BaseException as error:
        result['error'] = repr(error)
    finally:
        save('WATCHDOG-RESTORE', result)
    return 0 if result['status'] != 'RESTORE_FAILED' else 1


def main():
    if socket.gethostname() != 'h100-8' or os.geteuid() != 0:
        raise RuntimeError('Wrong host or uid')
    receipt = json.loads((NS / 'SCANCEL-RECEIPT.json').read_text())
    auth_raw = (NS / 'AUTHORIZATION.json').read_bytes()
    auth = json.loads(auth_raw)
    intent = json.loads((NS / 'SCANCEL-INTENT.json').read_text())
    if receipt['job_id'] != '3329' or receipt['command_result']['returncode'] != 0:
        raise RuntimeError('Missing actual cancellation')
    if intent.get('authorization_sha256') != hashlib.sha256(auth_raw).hexdigest() or intent.get('argv') != ['scancel', '3329']:
        raise RuntimeError('Cancellation authorization/intent binding differs')
    if auth['human_instruction_original'] != '\u4f60\u53ef\u4ee5kill\u6389\u4ed6\u8dd1\u7684\uff0c\u7136\u540e\u8dd1\u4f60\u7684':
        raise RuntimeError('Wrong current human authorization')
    if hashlib.sha256(BINDING.read_bytes()).hexdigest() != BINDING_SHA:
        raise RuntimeError('Existing GiGPO binding differs')
    cancelled = run(['scontrol', 'show', 'job', '3329', '-o'])
    queued = run(['scontrol', 'show', 'job', '3332', '-o'])
    expected_foreign = dict(JobId='3329', JobName='search-g1-formal-inline-ef498ac3',
        NodeList='h100-8', JobState='COMPLETING', UserId='root(0)',
        StartTime='2026-10-07T19:10:26',
        Command='/mnt/data/shaoshitong/g1f-ef498ac3/repo/reproduction/common8/launch_native.sh')
    if cancelled['returncode'] or any(field(cancelled['stdout'], k) != v for k, v in expected_foreign.items()):
        raise RuntimeError('Exact original3329 identity/state differs')
    if intent.get('foreign_job_identity') != dict(id='3329', name=expected_foreign['JobName'],
            node='h100-8', user='root(0)', start_time=expected_foreign['StartTime']):
        raise RuntimeError('Original cancellation intent refers to another job')
    expected_ours = dict(JobId='3332', JobName='gigpo-e1-fresh200-live',
        ReqNodeList='h100-8', JobState='PENDING', SubmitTime='2026-10-07T19:46:26',
        Command=str(ROOT / 'tools/fresh200-continuation-v14-evidenceclone1/calibration_v1.sbatch'))
    if queued['returncode'] or any(field(queued['stdout'], k) != v for k, v in expected_ours.items()):
        raise RuntimeError('Exact original3332 identity/state differs')
    if str(BINDING) + ' ' + BINDING_SHA not in queued['stdout']:
        raise RuntimeError('Queued command does not use frozen binding')
    if auth['actual_before']['foreign']['stdout'].split(' StartTime=')[1].split()[0] != expected_foreign['StartTime']:
        raise RuntimeError('Authorization refers to another job start')
    gpu = run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'])
    pids = run(['scontrol', 'listpids', '3329'])
    if gpu['returncode'] or gpu['stdout'].strip() or pids['returncode'] or len(pids['stdout'].strip().splitlines()) != 1:
        raise RuntimeError('Live compute/tasks remain')
    if Path('/sys/fs/cgroup/freezer/slurm_h100-8/uid_0/job_3329').exists():
        raise RuntimeError('Cancelled task cgroup remains')
    active = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            command = (proc / 'cmdline').read_bytes().replace(b'\x00', b' ')
            if command.startswith(b'slurmstepd:') or command.startswith(b'/usr/sbin/slurmstepd'):
                active.append(dict(pid=int(proc.name), command=command.decode(), stat=(proc / 'stat').read_text()))
        except OSError:
            pass
    if len(active) != 1 or active[0]['command'].strip() != 'slurmstepd: [3329.batch]':
        raise RuntimeError('Other stepd exists; legacy alias would affect it')
    config = run(['scontrol', 'show', 'config'])
    if config['returncode'] or not re.search(r'^SlurmctldHost\[0\]\s*=\s*h100-8\s*$', config['stdout'], re.M):
        raise RuntimeError('Current controller differs')
    local_config = Path('/etc/slurm/slurm.conf').read_text()
    if not re.search(r'^SlurmctldHost\s*=\s*h100-8(?:\s|$)', local_config, re.M):
        raise RuntimeError('Local controller differs')
    listeners = run(['ss', '-ltnp', 'sport = :6817'])
    if listeners['returncode'] or 'slurmctld' not in listeners['stdout']:
        raise RuntimeError('No actual local controller listener')
    log_path = Path('/var/log/slurm/slurmd.log')
    with log_path.open('rb') as f:
        f.seek(max(0, log_path.stat().st_size - 24000))
        lines = f.read().decode(errors='replace').splitlines()
    fresh = []
    for line in lines:
        if '[3329.batch]' in line and 'Unable to resolve "h100-16"' in line:
            stamp = datetime.datetime.fromisoformat(line.split(']')[0][1:]).timestamp()
            if 0 <= time.time() - stamp <= 60:
                fresh.append(line)
    if not fresh:
        raise RuntimeError('No recent target3329 legacy DNS failure')
    if not any(a[4][0] == '10.33.0.8' for a in socket.getaddrinfo('h100-8', 6817, socket.AF_INET)):
        raise RuntimeError('Current h1008 resolves elsewhere')
    fd, st = locked_hosts()
    try:
        original = os.pread(fd, st.st_size, 0)
        if not original.endswith(b'\n') or re.search(rb'(?m)^[^#\n]*\bh100-16\b', original):
            raise RuntimeError('Unexpected hosts layout or existing legacy alias')
        plan = dict(utc=now(), offset=st.st_size, dev=st.st_dev, ino=st.st_ino,
            deadline_epoch=time.time() + 65, alias_line=LINE.decode(), controller='h100-8/10.33.0.8',
            authorization_sha256=hashlib.sha256(auth_raw).hexdigest(), scancel_intent=intent,
            original_sha256=hashlib.sha256(original).hexdigest(), target_stepd=active,
            cancelled=cancelled, queued=queued, binding_sha256=BINDING_SHA, fresh_error=fresh[-1])
        save('INTENT', plan)
        with (NS / (PREFIX + 'original-hosts')).open('xb') as f:
            f.write(original); f.flush(); os.fsync(f.fileno())
        log_fd = os.open(NS / (PREFIX + 'watchdog.log'), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            watcher = subprocess.Popen([sys.executable, '-E', '-B', '-S', str(Path(__file__).resolve()), '--restore-watchdog'],
                start_new_session=True, stdin=subprocess.DEVNULL, stdout=log_fd, stderr=log_fd, close_fds=True)
        finally:
            os.close(log_fd)
        deadline = time.monotonic() + 5
        while True:
            ready_path = NS / (PREFIX + 'WATCHDOG-READY.json')
            if ready_path.exists():
                ready = json.loads(ready_path.read_text())
                if ready['pid'] != watcher.pid or ready['deadline_epoch'] != plan['deadline_epoch'] or ready['source_sha256'] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
                    raise RuntimeError('Watchdog handshake differs; hosts untouched')
                if watcher.poll() is not None:
                    raise RuntimeError('Watchdog exited before mutation; hosts untouched')
                break
            if watcher.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError('Watchdog not acknowledged; hosts untouched')
            time.sleep(.1)
        if os.fstat(fd).st_size != st.st_size or os.pread(fd, st.st_size, 0) != original:
            raise RuntimeError('hosts changed before append; untouched')
        # O_APPEND never discards simultaneous appenders' data. Cleanup does not
        # replace the file, so external appended lines/atomic replacements survive.
        if os.write(fd, LINE) != len(LINE):
            raise RuntimeError('Partial append; watchdog owns prefix cleanup')
        os.fsync(fd)
        if os.pread(fd, len(LINE), st.st_size) != LINE:
            raise RuntimeError('Concurrent host append changed owned offset; no continuation')
    finally:
        os.close(fd)
    outcome = dict(status='UNPROVEN', observations=[])
    try:
        deadline = time.monotonic() + 40
        while time.monotonic() < deadline:
            row = dict(utc=now(), foreign=run(['scontrol', 'show', 'job', '3329', '-o']),
                ours=run(['scontrol', 'show', 'job', '3332', '-o']))
            outcome['observations'].append(row)
            if row['foreign']['returncode'] == 0 and field(row['foreign']['stdout'], 'JobState') == 'CANCELLED':
                outcome['status'] = 'REAL_CANCELLED_ACKNOWLEDGED'
                break
            time.sleep(2)
    except BaseException as error:
        outcome['error'] = repr(error)
    finally:
        try:
            outcome['cleanup'] = deactivate(plan)
        except BaseException as error:
            outcome['cleanup_error'] = repr(error)
        save('RECEIPT', outcome)
        print(json.dumps(outcome), flush=True)
    return 0 if outcome['status'] == 'REAL_CANCELLED_ACKNOWLEDGED' and 'cleanup_error' not in outcome else 1


if __name__ == '__main__':
    sys.exit(watchdog() if sys.argv[1:] == ['--restore-watchdog'] else main())
