/dataset/shaoshitong/search-r1-rl/cosearch/env/bin/python - <<'PYCODE'
import subprocess,json
from pathlib import Path
rows=subprocess.check_output(['ps','-eo','pid,ppid,etimes,time,pcpu,rss,args'],text=True).splitlines()
for row in rows:
 if 'ray::main_task' in row or 'five_iteration_job_d.py' in row: print(row)
r=Path('/dataset/shaoshitong/search-r1-rl/cosearch')
for p in (r/'session_latest/logs').glob('worker-*.out'):
 if '134' in p.name:
  lines=p.read_text(errors='replace').splitlines()
  if any('COSEARCH_RANK_PROGRESS' in x for x in lines):print(p,lines[-4:])

PYCODE
