import argparse,json,os,pathlib,re,time
p=argparse.ArgumentParser();p.add_argument('--run',type=pathlib.Path,required=True);p.add_argument('--job',required=True);a=p.parse_args()
log=(a.run/'retriever.log').read_text();pid=int(re.search(r'Started server process \[(\d+)\]',log).group(1));proc=pathlib.Path('/proc')/str(pid)
assert re.search(r'(?:^|/)job_'+re.escape(a.job)+r'(?:/|$)',(proc/'cgroup').read_text()),'PID outside own Slurm job'
initial=(proc/'stat').read_text().rsplit(')',1)[1].split()[19]
with (a.run/'retriever-cpu-samples.jsonl').open('x') as out:
 while True:
  try:
   fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
   if fields[19]!=initial:break
   status=(proc/'status').read_text();hwm=re.search(r'^VmHWM:\s+(\d+) kB',status,re.M)
   r={'epoch':time.time(),'pid':pid,'job_id':a.job,'process_cpu_seconds':(int(fields[11])+int(fields[12]))/os.sysconf('SC_CLK_TCK'),'rss_bytes':int(fields[21])*os.sysconf('SC_PAGE_SIZE'),'peak_rss_bytes':int(hwm.group(1))*1024 if hwm else None,'scope':'own retriever process; sampling started after initial loading'}
   out.write(json.dumps(r)+'\n');out.flush()
  except FileNotFoundError:break
  time.sleep(5)
