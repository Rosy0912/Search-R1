import csv, datetime, errno, json, os, re, subprocess, sys, time
from pathlib import Path
job = os.environ.get('SLURM_JOB_ID', '')
# Prior readable ownership is retained verbatim; it is not current ownership.
prior_owned = {}

def resolve_exit_race(app, seen_time, confirmed_apps, proc_exists):
    if not app.get('unavailable'):
        if app.get('in_slurm_job') is True and str(app.get('uid')) == str(os.getuid()):
            prior_owned[(app['uuid'], app['pid'])] = {'time': seen_time, 'record': dict(app)}
        return app
    prior = prior_owned.get((app['uuid'], app['pid']))
    absent = (app['uuid'], app['pid']) not in confirmed_apps
    error = app.get('read_error', {})
    if prior is not None and absent and proc_exists is False and error.get('errno') == errno.ENOENT:
        app['exit_race_resolution'] = {'status': 'EXITED_AFTER_NVML_PRIOR_OWNED',
            'prior': prior, 'proc_exists_after': False, 'absent_from_confirmed_nvml': True}
    return app
def owned_cgroup(text):
    return bool(job and re.search(r'(?:^|/)job_' + re.escape(job) + r'(?:/|$)', text, re.M))


def usage(pid):
    try:
        text = Path(f'/proc/{pid}/status').read_text()
        fields = dict(line.split(':',1) for line in text.splitlines() if ':' in line)
        cgroup = Path(f'/proc/{pid}/cgroup').read_text()
        stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return dict(cpu_user_ticks=int(stat[11]), cpu_system_ticks=int(stat[12]),
            clock_ticks_per_second=os.sysconf('SC_CLK_TCK'), pid=pid, uid=fields['Uid'].split()[0],
            rss_kib=int(fields.get('VmRSS','0 kB').split()[0]),
            peak_rss_kib=int(fields.get('VmHWM','0 kB').split()[0]),
            name=fields.get('Name','').strip(),
            process_start_ticks=stat[19],
            in_slurm_job=owned_cgroup(cgroup))
    except (OSError, KeyError, ValueError, IndexError) as exc:
        return dict(pid=pid, unavailable=True, read_error=dict(
            type=type(exc).__name__, errno=getattr(exc, 'errno', None), message=str(exc)))

def proc_exists(pid):
    # Only an explicit ENOENT proves absence. Permission and parse failures
    # remain unknown and cannot be converted into an ordinary exit race.
    try:
        Path(f'/proc/{pid}').stat()
        return True
    except FileNotFoundError as exc:
        if exc.errno == errno.ENOENT:
            return False
        raise

def nvml_app_rows(raw):
    rows = list(csv.reader(raw.splitlines()))
    if any(len(row) != 3 for row in rows):
        raise ValueError('Malformed NVML application record')
    return rows

def safe_cgroup(path):
    try: return (path/'cgroup').read_text()
    except OSError: return ''

def main():
    out, parent = Path(sys.argv[1]), int(sys.argv[2])
    parent_start = Path(f'/proc/{parent}/stat').read_text().rsplit(')',1)[1].split()[19]
    cgroup_paths = []
    for line in Path(f'/proc/{parent}/cgroup').read_text().splitlines():
        _, controllers, path = line.split(':', 2)
        if not controllers:
            cgroup_paths.append(Path('/sys/fs/cgroup')/path.lstrip('/'))
        if 'memory' in controllers.split(','):
            cgroup_paths.append(Path('/sys/fs/cgroup/memory')/path.lstrip('/'))
    with out.open('a', buffering=1) as handle:
        while True:
            try:
                os.kill(parent,0)
                if Path(f'/proc/{parent}/stat').read_text().rsplit(')',1)[1].split()[19] != parent_start:
                    break
            except (ProcessLookupError, FileNotFoundError):
                break
            row=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(), job_id=job, nominal_sleep_seconds=2)
            try:
                gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],text=True,timeout=8)
                apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=8)
                row['gpus']=[dict(index=r[0].strip(),uuid=r[1].strip(),util=float(r[2]),used_mib=float(r[3]),total_mib=float(r[4])) for r in csv.reader(gpu.splitlines())]
                row['nvml_gpu_original'] = gpu
                row['nvml_apps_original'] = apps
                records = [dict(uuid=r[0].strip(),gpu_used_mib=r[2].strip(),**usage(int(r[1]))) for r in nvml_app_rows(apps)]
                # Retain the initial per-PID reads even if the confirmation
                # query fails; an unresolved application must remain visible.
                row['gpu_processes'] = records
                confirmed = set()
                if any(v.get('unavailable') for v in records):
                    raw = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=8)
                    row['nvml_apps_recheck_original'] = raw
                    confirmed = {(r[0].strip(),int(r[1])) for r in nvml_app_rows(raw)}
                row['gpu_processes'] = [resolve_exit_race(v,row['time'],confirmed,proc_exists(v['pid'])) for v in records]
            except (subprocess.SubprocessError,ValueError,OSError) as e:
                row['query_error']=type(e).__name__
                row['query_error_detail'] = dict(errno=getattr(e,'errno',None), message=str(e))
            row['job_processes'] = [usage(int(p.name)) for p in Path('/proc').iterdir()
                if p.name.isdigit() and owned_cgroup(safe_cgroup(p))]
            row['cgroup']={}
            for path in cgroup_paths:
                for name in ('memory.current','memory.peak','memory.events','memory.max','memory.usage_in_bytes','memory.max_usage_in_bytes','memory.limit_in_bytes','memory.failcnt'):
                    try: row['cgroup'][str(path/name)]=(path/name).read_text().strip()
                    except OSError: pass
            handle.write(json.dumps(row,sort_keys=True)+'\n')
            time.sleep(2)

if __name__ == '__main__':
    main()
