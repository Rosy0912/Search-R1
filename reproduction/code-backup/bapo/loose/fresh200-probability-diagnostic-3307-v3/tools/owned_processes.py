"""Observe or stop only processes carrying this run's exact inherited marker."""
import argparse,json,os,signal,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('action',choices=['monitor','stop']);p.add_argument('--run',required=True);p.add_argument('--retriever',type=int);p.add_argument('--ray',type=int);p.add_argument('--output');args=p.parse_args()
job=os.environ.get('SLURM_JOB_ID')
if not job or not job.isdigit(): raise SystemExit('Scoped ownership requires numeric SLURM_JOB_ID')
marker=('BAPO_RUN_ID='+args.run).encode();job_marker=('SLURM_JOB_ID='+job).encode();exclude={os.getpid(),os.getppid()}
def snapshot():
 found={}
 for d in Path('/proc').iterdir():
  if not d.name.isdigit() or int(d.name) in exclude: continue
  try:
   environment=(d/'environ').read_bytes().split(b'\0')
   if marker not in environment or job_marker not in environment or d.stat().st_uid!=os.getuid():continue
   s=(d/'stat').read_text().rsplit(')',1)[1].split()
   status=(d/'status').read_text().splitlines()
   found[int(d.name)]={'pid':int(d.name),'ppid':int(s[1]),'start_ticks':int(s[19]),'cpu_ticks':int(s[11])+int(s[12]),'rss_bytes':int(s[21])*os.sysconf('SC_PAGE_SIZE'),'threads':int(s[17])}
  except (OSError,ValueError,IndexError):pass
 return found
if args.action=='stop':
 records=snapshot();fds={}
 for pid,rec in records.items():
  try:
   fd=os.pidfd_open(pid)
   current=snapshot().get(pid)
   if current is None or current['start_ticks']!=rec['start_ticks']:os.close(fd);continue
   fds[pid]=fd;signal.pidfd_send_signal(fd,signal.SIGTERM)
  except ProcessLookupError:pass
 time.sleep(5)
 for pid,fd in fds.items():
  try:signal.pidfd_send_signal(fd,signal.SIGKILL)
  except ProcessLookupError:pass
  os.close(fd)
 time.sleep(1)
 remaining=snapshot()
 print(json.dumps({'own_marker':args.run,'stopped_pids':sorted(fds),'remaining':list(remaining.values())}))
 if remaining:raise SystemExit('Own marked processes remain after scoped cleanup')
else:
 while True:
  records=snapshot()
  for rec in records.values():
   seen=set();ancestor=rec['pid'];role='training_or_ray'
   while ancestor in records and ancestor not in seen:
    if ancestor==args.retriever:role='retriever';break
    seen.add(ancestor);ancestor=records[ancestor]['ppid']
   rec['role']=role
  with open(args.output,'a') as f:f.write(json.dumps({'time':time.time(),'clock_ticks_per_second':os.sysconf('SC_CLK_TCK'),'processes':list(records.values())})+'\n')
  time.sleep(10)
