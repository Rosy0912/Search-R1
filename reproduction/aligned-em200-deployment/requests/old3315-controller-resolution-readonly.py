"""Read-only current controller/config/resolution diagnosis for stopped3315."""
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
BASELINE = {'file': 'aligned-em200-old3315-stepdaemon-missing-logs.actual.json', 'sha256': 'f6acf880d6269b79e3504cca3bd9cf5ee59d9944cfa51138c5c6e3021d870397', 'utc': '2026-10-08T21:01:48.702916+00:00', 'boot_id': '9471cbc1-9ae1-4be5-8edb-1e84e850e2cc'}
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

ALLOWED={'ClusterName','SlurmctldHost','SlurmctldAddr','SlurmctldPort','ControlMachine','ControlAddr','BackupController','BackupAddr','SlurmdPort','SLURM_CONF'}


def selected_config(text):
    rows=[]
    for line in text.splitlines():
        line=line.split('#',1)[0]
        for match in re.finditer(r'(?:^|\s)([A-Za-z_]+)(\[[0-9]+\])?\s*=\s*(\S+)',line):
            key,index,value=match.groups()
            if key in ALLOWED:rows.append(dict(key=key+(index or ''),value=value))
    return rows


def selected_env(pid):
    # Inspect only one selected key; never emit full environment/token strings.
    value=None
    for part in (Path('/proc')/str(pid)/'environ').read_bytes().split(b'\0'):
        if part.startswith(b'SLURM_CONF='):value=part.split(b'=',1)[1].decode(errors='replace')
    return value


def main():
    if socket.gethostname()!='h100-5':raise RuntimeError('ownnode only')
    if Path('/proc/sys/kernel/random/boot_id').read_text().strip()!=BASELINE['boot_id']:raise RuntimeError('boot changed')
    v=dict(scope='READONLY_OWN3315_CONTROLLER_CONFIGURATION_AND_RESOLUTION_NO_ADMIN_MUTATION',
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),baseline_observation=BASELINE,
        action_taken='NONE',never_resume3315=True)
    v['old_daemon_before']=current_birth()
    v['old_daemon_selected_SLURM_CONF']=selected_env(EXPECTED['pid'])
    v['cli_selected_SLURM_CONF']=os.environ.get('SLURM_CONF')
    config=command(['scontrol','show','config'])
    v['current_cli_selected_config']=dict(exit=config.get('exit'),timeout=config.get('timeout',False),rows=selected_config(config.get('stdout','')))
    # Raw config/stdout never leaves memory. A ping only reports controller status.
    v['current_cli_controller_ping']=command(['scontrol','ping'])
    v['slurmd_processes']=[]
    paths=[v['old_daemon_selected_SLURM_CONF'],v['cli_selected_SLURM_CONF'],'/etc/slurm/slurm.conf','/etc/slurm-llnl/slurm.conf']
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            if (proc/'comm').read_text().strip() not in ('slurmd','slurmd-wlm'):continue
            before=stat_identity(proc);exe=os.readlink(proc/'exe')
            if Path(exe).name not in ('slurmd','slurmd-wlm'):continue
            argv=[p.decode(errors='replace')for p in (proc/'cmdline').read_bytes().split(b'\0')if p]
            explicit=argv[argv.index('-f')+1]if '-f'in argv and argv.index('-f')+1<len(argv)else None
            environment=selected_env(int(proc.name));after=stat_identity(proc)
            v['slurmd_processes'].append(dict(before=before,after=after,exe_link=exe,selected_config_arg=explicit,selected_SLURM_CONF=environment,
                same_birth=before['start_ticks']==after['start_ticks']))
            paths += [explicit,environment]
        except OSError as e:v['slurmd_processes'].append(dict(pid=int(proc.name),unavailable=str(e)))
    v['selected_config_files']=[]
    for path in dict.fromkeys(p for p in paths if p and p.startswith('/')):
        p=Path(path)
        try:
            before=p.stat()
            if before.st_size>4*1024*1024:raise ValueError('unexpected large config; not read')
            raw=p.read_bytes();after=p.stat()
            v['selected_config_files'].append(dict(path=str(p),resolved_path=str(p.resolve()),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),
                mtime_ns=after.st_mtime_ns,stable=before.st_ino==after.st_ino and before.st_mtime_ns==after.st_mtime_ns,
                selected_rows=selected_config(raw.decode(errors='replace'))))
        except (OSError,ValueError)as e:v['selected_config_files'].append(dict(path=path,unavailable=str(e)))
    names={'h100-16'}
    rows=v['current_cli_selected_config']['rows']+[r for file in v['selected_config_files']for r in file.get('selected_rows',[])]
    for row in rows:
        if any(row['key'].startswith(key)for key in ('SlurmctldHost','SlurmctldAddr','ControlMachine','ControlAddr','BackupController','BackupAddr')):
            for value in re.split(r'[(),]',row['value']):
                if re.fullmatch(r'[A-Za-z0-9_.:-]+',value):names.add(value)
    v['resolution']=[command(['getent','ahosts',name],timeout=8)for name in sorted(names)[:8]]
    try:
        selected=[]
        for line in Path('/etc/hosts').read_text().splitlines():
            parts=line.split('#',1)[0].split()
            if len(parts)>1 and any(value in names for value in parts[1:]):selected.append(' '.join(parts))
        v['selected_etc_hosts_rows']=selected
    except OSError as e:v['hosts_unavailable']=str(e)
    v['old_daemon_after']=current_birth()
    v['in_memory_old_controller_config_not_read']=True
    v['resolution_alias_is_not_authorized']=True
    print(json.dumps(v,sort_keys=True))


if __name__=='__main__':main()
