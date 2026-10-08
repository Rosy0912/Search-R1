"""Follow-up missing3315 logs/cgroups only; exact observed birth/exe; no signals."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess

LIMIT = 2 * 1024 * 1024
OLD = Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-controller-initfix-20261006')
BASELINE = {'file': 'aligned-em200-old3315-stepdaemon-diagnosis.actual.json', 'sha256': '5961713de438e50f17abea7118e843b34d41ac36d80ecbc585777d79bd8f3979', 'utc': '2026-10-08T20:57:16.747241+00:00', 'boot_id': '9471cbc1-9ae1-4be5-8edb-1e84e850e2cc'}
EXPECTED = {'pid': 755855, 'start_ticks': 1433986891, 'ppid': 1, 'uid': 0, 'argv': 'slurmstepd: [3315.0]', 'exe_link': '/usr/sbin/slurmstepd-wlm', 'exe_sha256': 'be7770948fcfd24b074745d8add09005f77c1ae0ddb380b86abceb71d2bfdd6a', 'exe_identity': {'bytes': 266032, 'device': 64769, 'inode': 288378}, 'cgroup': '1:freezer:/\n0::/system.slice/slurmd.service\n'}


def command(args, timeout=15):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return dict(command=args, exit=p.returncode, stdout=p.stdout, stderr=p.stderr)
    except subprocess.TimeoutExpired as e:
        text = lambda value: value.decode(errors='replace') if isinstance(value, bytes) else (value or '')
        return dict(command=args, exit=None, timeout=True, stdout=text(e.stdout), stderr=text(e.stderr))
    except OSError as e:
        return dict(command=args, exit=None, error=str(e))

def stat_identity(proc):
    fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=int(proc.name), state=fields[0], ppid=int(fields[1]), start_ticks=int(fields[19]), uid=proc.stat().st_uid)

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

def current_birth():
    proc = Path('/proc') / str(EXPECTED['pid'])
    before = stat_identity(proc)
    argv = (proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace').strip()
    exe = os.readlink(proc/'exe')
    st = (proc/'exe').stat()
    executable = dict(device=st.st_dev,inode=st.st_ino,bytes=st.st_size)
    with (proc/'exe').open('rb') as handle: raw=handle.read(1024*1024)
    sha = hashlib.sha256(raw).hexdigest()
    after = stat_identity(proc)
    expected_birth = {key:EXPECTED[key] for key in ('pid','start_ticks','ppid','uid')}
    if (any(before[key]!=value or after[key]!=value for key,value in expected_birth.items())
        or before['state'] in ('Z','X') or after['state'] in ('Z','X')
        or argv!=EXPECTED['argv'] or exe!=EXPECTED['exe_link']
        or Path(exe).name not in ('slurmstepd','slurmstepd-wlm')
        or executable!=EXPECTED['exe_identity'] or sha!=EXPECTED['exe_sha256']):
        raise RuntimeError('exact original3315 daemon birth/argv/executable mismatch; refuse further collection')
    cgroup=(proc/'cgroup').read_text()
    return dict(**after,argv=argv,exe_link=exe,exe_identity=executable,exe_sha256=sha,cgroup=cgroup)


def main():
    if socket.gethostname()!='h100-5':raise RuntimeError('read-only ownnode only')
    if Path('/proc/sys/kernel/random/boot_id').read_text().strip()!=BASELINE['boot_id']:
        raise RuntimeError('node boot changed; old PID/startticks binding invalid')
    v=dict(scope='READONLY_STOPPED3315_MISSING_LOGS_CGROUPS_ONLY_NO_SIGNAL_NO_NODE_OR_SERVICE_MUTATION',
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),baseline_observation=BASELINE,
        stop_authorization_utc='2026-10-08T18:07:39.855509Z',never_resume3315=True,action_taken='NONE')
    v['before']=current_birth()
    v['installed_manual_in_original_probe']='actual21.08 installed manual confirms SIGTERM ignored; not re-polled'
    v['logs']=[log_tail(OLD/'logs'/name,own_file=True)for name in ('slurm-formal200-3315.out','slurm-formal200-3315.err')]
    # Paths already obtained from the hash-bound original probe; no raw config.
    v['logs'] += [log_tail(path)for path in ('/var/log/slurm/slurmd.log','/var/log/slurm/slurmctld.log')]
    v['journal_own3315']=[]
    for args in (['journalctl','-u','slurmd','--no-pager','-n','500','-o','short-iso'],
        ['journalctl','-u','slurmd','--no-pager','-n','500','-o','short-iso','--since','2026-10-08 18:00:00 UTC','--until','2026-10-08 18:20:00 UTC']):
        row=command(args);row['stdout']='\n'.join(own_lines(row.get('stdout','')));v['journal_own3315'].append(row)
    v['job_cgroup_records']=[]
    for controller in ('','freezer','cpuset','memory','devices','pids','unified'):
        for relative in ('slurm/uid_0/job_3315','slurm/job_3315'):
            candidate=Path('/sys/fs/cgroup')/controller/relative
            if not candidate.is_dir():
                v['job_cgroup_records'].append(dict(path=str(candidate),exists=False));continue
            for directory in [candidate]+[p for p in candidate.iterdir()if p.is_dir()]:
                row=dict(path=str(directory),exists=True)
                for name in ('tasks','cgroup.procs','cgroup.events'):
                    try:row[name]=(directory/name).read_text()[:65536]
                    except OSError as e:row[name+'_unavailable']=str(e)
                v['job_cgroup_records'].append(row)
    # Do not inspect/control all tasks in the shared slurmd service cgroup.
    v['service_cgroup_is_not_3315_job_identity']=True
    spool=Path('/var/spool/slurmd');v['own_spool_entries']=[]
    try:
        for path in spool.iterdir():
            if re.search(r'(?<![0-9])3315(?![0-9])',path.name):
                st=path.lstat();v['own_spool_entries'].append(dict(path=str(path),mode=st.st_mode,inode=st.st_ino,bytes=st.st_size))
    except OSError as e:v['spool_unavailable']=str(e)
    try:v['own_unix_socket_rows']=own_lines(Path('/proc/net/unix').read_text())
    except OSError as e:v['unix_socket_rows_unavailable']=str(e)
    v['after']=current_birth()
    v['same_original_birth_before_after']=True
    print(json.dumps(v,sort_keys=True))


if __name__=='__main__':main()
