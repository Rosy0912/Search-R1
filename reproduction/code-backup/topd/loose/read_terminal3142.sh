python3 - <<'PY'
import subprocess,json,time,pathlib,hashlib
commands=[['hostname'],['date','--iso-8601=seconds'],['sacct','-j','3142','--format=JobID,JobName,State,ExitCode,Start,End,NodeList','--parsable2','--noheader'],['systemctl','show','slurmd','-p','FragmentPath','-p','ExecStart'],['journalctl','-u','slurmd','--since','2026-10-04 13:20:00','--until','2026-10-04 13:50:00','--no-pager','-o','short-iso'],['scontrol','show','config']]
rows=[]
for cmd in commands:
 try:
  q=subprocess.run(cmd,capture_output=True,text=True,timeout=30);out=q.stdout
  if cmd[0]=='journalctl':out='\n'.join(l for l in out.splitlines() if '3142' in l or 'job' in l.lower() and ('complete' in l.lower() or 'exit' in l.lower()))
  if cmd[:3]==['scontrol','show','config']:out='\n'.join(l for l in out.splitlines() if any(k in l for k in ('SlurmdLogFile','SlurmctldLogFile','AccountingStorageType','JobComp','SlurmdSpoolDir','MinJobAge')))
  rows.append({'cmd':cmd,'returncode':q.returncode,'stdout':out,'stderr':q.stderr})
 except Exception as e:rows.append({'cmd':cmd,'error':repr(e)})
print(json.dumps({'epoch':time.time(),'scope':'READ_ONLY_TERMINAL_SEARCH_3142','commands':rows}))
PY
