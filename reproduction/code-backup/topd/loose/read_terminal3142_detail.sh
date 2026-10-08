python3 - <<'PY'
import pathlib,subprocess,json,time,hashlib
rows=[]
for cmd in [['journalctl','--since','2026-10-04 13:43:35','--until','2026-10-04 13:44:05','--no-pager','-o','short-iso'],['grep','-n','3142','/var/log/slurm/slurmd.log','/var/log/slurm/slurmctld.log','/var/log/slurm_jobcomp.log']]:
 q=subprocess.run(cmd,capture_output=True,text=True,timeout=30);rows.append({'cmd':cmd,'returncode':q.returncode,'stdout':q.stdout,'stderr':q.stderr})
print(json.dumps({'epoch':time.time(),'scope':'READ_ONLY_TERMINAL_DETAIL_3142','commands':rows}))
PY
