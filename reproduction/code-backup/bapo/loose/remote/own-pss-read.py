from pathlib import Path
import json,time
run='/dataset/shaoshitong/search-r1-rl/bapo/runs/bapo-smoke2-2872'
marker=('BAPO_RUN_ID='+run).encode();out=[];skipped=[];start=time.time()
for d in Path('/proc').iterdir():
 if not d.name.isdigit():continue
 try:
  before=(d/'stat').read_text().rsplit(')',1)[1].split()[19]
  if marker not in (d/'environ').read_bytes().split(b'\0'):continue
  fields={}
  for line in (d/'smaps_rollup').read_text().splitlines():
   parts=line.split()
   if parts[0] in ('Pss:','Rss:','Swap:'):fields[parts[0][:-1]+'_bytes']=int(parts[1])*1024
  after=(d/'stat').read_text().rsplit(')',1)[1].split()[19]
  if after!=before:raise RuntimeError('PID lifecycle changed')
  if not {'Pss_bytes','Rss_bytes'}.issubset(fields):raise RuntimeError('Incomplete memory fields')
  out.append(dict(pid=int(d.name),start_ticks=before,**fields))
 except (OSError,RuntimeError,IndexError) as e:skipped.append(dict(pid=int(d.name),error=type(e).__name__))
print(json.dumps(dict(run=run,start_time=start,end_time=time.time(),own_processes=len(out),sum_pss_bytes=sum(x['Pss_bytes'] for x in out),sum_rss_bytes=sum(x['Rss_bytes'] for x in out),skipped=skipped,processes=out,note='Own exact inherited marker, stable PID ticks. PSS is sampled proportional mapped memory; excludes unmapped cache/slurm overhead; not a cgroup peak.'),indent=2))
