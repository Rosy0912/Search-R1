set -euo pipefail
python3 - <<'CHECK'
import json,hashlib,re,time
from pathlib import Path
root=Path('/dataset/shaoshitong/search-r1-rl/cosearch');run=root/'runs/cosearch-local-gpu-five-20261003-n'
audit=json.loads((run/'actor-gpu-ownership-first.json').read_text());provenance=json.loads((run/'worker-log-provenance.json').read_text());session=Path(provenance['session']);out=[]
assert audit['job']==3002 and audit['actor_distinct_eight_physical_gpus']
actors={pid:v for pid,v in audit['records'].items() if 'ray::WorkerDict' in v['title']};assert len(actors)==8
for pid,live in actors.items():
 assert '/job_3002/' in live['cgroup'] and live['start_ticks']>0
 files=list((session/'logs').glob('worker-*-'+pid+'.out'));assert len(files)==1,(pid,files)
 path=files[0];data=path.read_bytes();stat=path.stat();rows=[]
 for line in data.decode(errors='replace').splitlines():
  if 'COSEARCH_ACTOR_INITIALIZATION_JSON=' in line:rows.append(json.JSONDecoder().raw_decode(line.split('COSEARCH_ACTOR_INITIALIZATION_JSON=',1)[1])[0])
 assert len(rows)==1,(pid,rows)
 assert rows[0]['pid']==int(pid) and rows[0]['job']=='3002'
 out.append(dict(source=str(path),source_inode=stat.st_ino,source_device=stat.st_dev,source_sha256=hashlib.sha256(data).hexdigest(),source_bytes=len(data),historical_live_identity=live,initialization=rows[0]))
value=dict(job=3002,session=str(session),main_provenance=provenance,captured_at=time.time(),live_actor_audit_sha256=hashlib.sha256((run/'actor-gpu-ownership-first.json').read_bytes()).hexdigest(),actors=out)
(run/'actor-initialization-evidence.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
CHECK
