import csv, datetime, json, os, re, subprocess, sys, time
from pathlib import Path
out, parent = Path(sys.argv[1]), int(sys.argv[2])
job = os.environ.get('SLURM_JOB_ID', '')
parent_start = Path(f'/proc/{parent}/stat').read_text().rsplit(')',1)[1].split()[19]
def owned_cgroup(text):
    return bool(job and re.search(r'(?:^|/)job_' + re.escape(job) + r'(?:/|$)', text, re.M))

cgroup_paths = []
for line in Path(f'/proc/{parent}/cgroup').read_text().splitlines():
    _, controllers, path = line.split(':', 2)
    if not controllers:
        cgroup_paths.append(Path('/sys/fs/cgroup')/path.lstrip('/'))
    if 'memory' in controllers.split(','):
        cgroup_paths.append(Path('/sys/fs/cgroup/memory')/path.lstrip('/'))

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
    except (OSError, KeyError, ValueError, IndexError):
        return dict(pid=pid, unavailable=True)

def safe_cgroup(path):
    try: return (path/'cgroup').read_text()
    except OSError: return ''

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
            row['gpu_processes']=[dict(uuid=r[0].strip(),gpu_used_mib=r[2].strip(),**usage(int(r[1]))) for r in csv.reader(apps.splitlines()) if len(r)==3]
        except (subprocess.SubprocessError,ValueError,OSError) as e:
            row['query_error']=type(e).__name__
        row['job_processes'] = [usage(int(p.name)) for p in Path('/proc').iterdir()
            if p.name.isdigit() and owned_cgroup(safe_cgroup(p))]
        row['cgroup']={}
        for path in cgroup_paths:
            for name in ('memory.current','memory.peak','memory.events','memory.max','memory.usage_in_bytes','memory.max_usage_in_bytes','memory.limit_in_bytes','memory.failcnt'):
                try: row['cgroup'][str(path/name)]=(path/name).read_text().strip()
                except OSError: pass
        handle.write(json.dumps(row,sort_keys=True)+'\n')
        time.sleep(2)
