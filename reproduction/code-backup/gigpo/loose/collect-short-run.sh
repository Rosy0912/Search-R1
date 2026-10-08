python3 - "$1" "$2" <<'PY'
from pathlib import Path
import json,hashlib,sys
root=Path(sys.argv[1]);job=sys.argv[2];runs=list(root.glob('runs/*-job'+job+'-*'));assert len(runs)==1,runs;r=runs[0]
paths=[p for p in r.rglob('*') if p.is_file() and not any(x in p.relative_to(r).parts for x in ['ray','tmp','cache','checkpoints'])]
files={};hashes=[]
for p in paths:
 data=p.read_bytes();files[str(p.relative_to(r))]=data.decode(errors='replace');hashes.append(dict(path=str(p),sha256=hashlib.sha256(data).hexdigest()))
logs=list(root.glob('logs/*-short-'+job+'.log'));assert len(logs)==1,logs;log=logs[0];data=log.read_bytes();hashes.append(dict(path=str(log),sha256=hashlib.sha256(data).hexdigest()))
check={str(p.relative_to(r)):p.stat().st_size for p in (r/'checkpoints').rglob('*') if p.is_file()}
print(json.dumps(dict(run=str(r),files=files,raw_hashes=hashes,slurm_log=data.decode(errors='replace'),checkpoint_files=check)))
PY
