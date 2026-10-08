from pathlib import Path
import json,time
pid=3067832
raw=Path(f'/proc/{pid}/cgroup').read_text()
records=[]
for line in raw.splitlines():
 hierarchy,controllers,relative=line.split(':',2)
 if hierarchy=='0' and not controllers:
  path=Path('/sys/fs/cgroup')/relative.lstrip('/')
  fields={}
  for name in ['memory.current','memory.peak','memory.max','memory.events','cpu.max','cpu.stat','cpuset.cpus.effective']:
   p=path/name
   if p.exists():fields[name]=p.read_text().strip()
  records.append(dict(path=str(path),dedicated_job_scope=('slurm' in relative and '2872' in relative),fields=fields))
print(json.dumps(dict(time=time.time(),pid=pid,raw_cgroup=raw,records=records,note='Do not attribute a shared/root cgroup peak to BAPO'),indent=2))
