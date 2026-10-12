"""One read-only diagnosis of stopped3315; never signal or alter scheduler state."""
import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess

JOB = '3315'
OLD = Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-controller-initfix-20261006')
COMMENT = 'BAPO_FRESH200:formal200:60f566b3ddabc88ce9de56ead5ad30a3d6795dd2:bab8981d9fd14afc98923eb4ebb89e33'
TAG_MAN = 'https://github.com/SchedMD/slurm/blob/slurm-21-08-5-1/doc/man/man8/slurmstepd.8'
LIMIT = 2 * 1024 * 1024


def command(args, timeout=15):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return dict(command=args, exit=p.returncode, stdout=p.stdout, stderr=p.stderr)
    except subprocess.TimeoutExpired as e:
        text = lambda value: value.decode(errors='replace') if isinstance(value, bytes) else (value or '')
        return dict(command=args, exit=None, timeout=True, stdout=text(e.stdout), stderr=text(e.stderr))
    except OSError as e:
        return dict(command=args, exit=None, error=str(e))


def matches_own(argv):
    return re.fullmatch(r'slurmstepd: \[3315\.(?:[0-9]+|batch|extern)\]', argv.strip()) is not None


def stat_identity(proc):
    fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=int(proc.name), state=fields[0], ppid=int(fields[1]), start_ticks=int(fields[19]), uid=proc.stat().st_uid)


def process(proc):
    before = stat_identity(proc)
    argv = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace').strip()
    if not matches_own(argv):
        return None
    result = dict(before, argv=argv, exe_link=os.readlink(proc / 'exe'),
                  status=(proc / 'status').read_text(), cgroup=(proc / 'cgroup').read_text(),
                  wchan=(proc / 'wchan').read_text(), threads=[])
    try:
        st = (proc / 'exe').stat()
        result['exe_identity'] = dict(device=st.st_dev, inode=st.st_ino, bytes=st.st_size)
    except OSError as e:
        result['exe_identity_error'] = str(e)
    try:
        # Read existing process executable, including a deleted mapped binary.
        with (proc / 'exe').open('rb') as handle:
            raw = handle.read(64 * 1024 * 1024 + 1)
        result['exe_bytes'] = len(raw)
        result['exe_sha256'] = hashlib.sha256(raw).hexdigest() if len(raw) <= 64 * 1024 * 1024 else None
    except OSError as e:
        result['exe_hash_error'] = str(e)
    for thread in sorted((proc / 'task').iterdir(), key=lambda p: int(p.name))[:64]:
        row = dict(tid=int(thread.name))
        for name in ('stat', 'wchan', 'stack', 'children'):
            try: row[name] = (thread / name).read_text()[:16384]
            except OSError as e: row[name + '_error'] = str(e)
        result['threads'].append(row)
    sockets = []
    for fd in (proc / 'fd').iterdir():
        try:
            link = os.readlink(fd)
            if '3315' in link: sockets.append(dict(fd=fd.name, link=link))
        except OSError: pass
    result['job_named_fds'] = sockets
    result['after'] = stat_identity(proc)
    result['same_birth'] = all(before[k] == result['after'][k] for k in ('pid', 'start_ticks', 'uid', 'ppid'))
    result['job_specific_cgroup'] = re.search(r'(?:job[_-]?3315|/3315)(?:/|$)', result['cgroup']) is not None
    return result


def own_lines(raw):
    return [line for line in raw.splitlines() if re.search(r'(?<![0-9])3315(?![0-9])', line)][-200:]


def log_tail(path, own_file=False):
    p = Path(path)
    try:
        before = p.stat()
        with p.open('rb') as handle:
            offset = max(0, before.st_size - LIMIT); handle.seek(offset); raw = handle.read(LIMIT)
        lines = raw.decode(errors='replace').splitlines()
        selected = lines[-120:] if own_file else own_lines(raw.decode(errors='replace'))
        after = p.stat()
        return dict(path=str(p), bytes=before.st_size, offset=offset, tail_bytes=len(raw),
                    tail_sha256=hashlib.sha256(raw).hexdigest(), selected_lines=selected,
                    selection='OWN3315_FILE_LAST120' if own_file else 'EXACT_JOB3315_ROWS_ONLY',
                    stable=before.st_ino == after.st_ino and before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns)
    except OSError as e:
        return dict(path=str(p), unavailable=str(e))


def main():
    v = dict(scope='READONLY_STOPPED3315_DAEMON_DIAGNOSIS_NO_SIGNAL_NO_NODE_OR_SERVICE_MUTATION',
             utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), host=socket.gethostname(),
             stop_authorization_utc='2026-10-08T18:07:39.855509Z', target_job=JOB, never_resume3315=True,
             official_matching_version_manual=TAG_MAN, official21_08_expected_TERM='IGNORED_NOT_A_GRACEFUL_SHUTDOWN_REMEDY')
    if v['host'] != 'h100-5':
        raise RuntimeError('read-only probe restricted to h100-5')
    v['boot_id'] = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    # Root already collected20:52 scheduler/listpids/steps/ps; do not poll them again.
    v['baseline_observation'] = dict(file='aligned-em200-old3315-completion-readonly-observation2.actual.json',
        sha256='e128229263c19537cc4722284be31d9aaef7a30e167937508fa010ebaaf5d9c2', utc='2026-10-08T20:52:41.687698+00:00')
    v['version'] = command(['scontrol', '--version'])
    v['expected_pid_from_fresh_baseline'] = 755855
    v['daemons'] = []
    proc = Path('/proc/755855')
    try:
        value = process(proc)
        if value is not None:
            v['daemons'].append(value)
    except OSError as e:
        v['daemons'].append(dict(pid=755855, observation_error=str(e)))
    v['matched_original_step_daemon'] = bool(v['daemons'] and v['daemons'][0].get('argv') == 'slurmstepd: [3315.0]'
        and v['daemons'][0].get('uid') == 0 and v['daemons'][0].get('same_birth') is True
        and Path(v['daemons'][0].get('exe_link', '').removesuffix(' (deleted)')).name == 'slurmstepd')
    # Filter config IN MEMORY: never print raw config or credentials.
    config = command(['scontrol', 'show', 'config'])
    allowed = {'SlurmdLogFile', 'SlurmctldLogFile', 'SlurmdSpoolDir'}
    paths = {}
    for line in config.get('stdout', '').splitlines():
        if '=' in line:
            key, value = [part.strip() for part in line.split('=', 1)]
            if key in allowed: paths[key] = value
    v['configuration_log_paths_only'] = dict(exit=config.get('exit'), paths=paths)
    v['logs'] = []
    if v['matched_original_step_daemon']:
        for name in ('slurm-formal200-3315.out', 'slurm-formal200-3315.err'):
            v['logs'].append(log_tail(OLD / 'logs' / name, own_file=True))
        for key in ('SlurmdLogFile', 'SlurmctldLogFile'):
            if paths.get(key, '').startswith('/'): v['logs'].append(log_tail(paths[key]))
        v['journal_own3315'] = []
        for args in (['journalctl', '-u', 'slurmd', '--no-pager', '-n', '500', '-o', 'short-iso'],
                     ['journalctl', '-u', 'slurmd', '--no-pager', '-n', '500', '-o', 'short-iso', '--since', '2026-10-08 18:00:00 UTC', '--until', '2026-10-08 18:20:00 UTC']):
            row = command(args); row['stdout'] = '\n'.join(own_lines(row.get('stdout', ''))); v['journal_own3315'].append(row)
    # Existing installed manual only; no daemon executable invocation or man cache.
    located = command(['man', '-w', 'slurmstepd'])
    candidates = [Path(line) for line in located.get('stdout', '').splitlines() if line.startswith('/')]
    candidates += [Path('/usr/share/man/man8/slurmstepd.8.gz'), Path('/usr/local/share/man/man8/slurmstepd.8.gz')]
    v['installed_manual'] = []
    for path in dict.fromkeys(candidates):
        if not path.is_file(): continue
        raw = path.read_bytes()
        text = gzip.decompress(raw).decode(errors='replace') if path.suffix == '.gz' else raw.decode(errors='replace')
        lines = text.splitlines(); indices = [i for i, line in enumerate(lines) if 'SIGTERM' in line or 'SIGNALS' in line]
        excerpts = [lines[max(0, i-2): min(len(lines), i+7)] for i in indices]
        v['installed_manual'].append(dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw), signal_excerpts=excerpts))
    v['manual_locator'] = located
    v['job_cgroup_records'] = []
    if v['matched_original_step_daemon']:
        root = Path('/sys/fs/cgroup')
        for controller in [root] + [p for p in root.iterdir() if p.is_dir()]:
            for candidate in (controller / 'slurm/uid_0/job_3315', controller / 'slurm/job_3315'):
                if not candidate.is_dir(): continue
                for directory in [candidate] + [p for p in candidate.iterdir() if p.is_dir()]:
                    row = dict(path=str(directory))
                    for name in ('tasks', 'cgroup.procs', 'cgroup.events'):
                        try: row[name] = (directory / name).read_text()[:65536]
                        except OSError as e: row[name + '_unavailable'] = str(e)
                    v['job_cgroup_records'].append(row)
    v['diagnosis_only'] = True
    v['action_taken'] = 'NONE'
    print(json.dumps(v, sort_keys=True))


if __name__ == '__main__':
    main()
