set -euo pipefail
R=/dataset/shaoshitong/search-r1-rl/cosearch
"$R/env/bin/python" - <<'CHECK'
import pathlib,json,time,math,hashlib
r=pathlib.Path('/dataset/shaoshitong/search-r1-rl/cosearch/runs/cosearch-local-gpu-five-20261003-n')
b=json.loads((r/'worker-log-provenance.json').read_text());p=pathlib.Path(b['source']);st=p.stat()
assert (st.st_ino,st.st_dev)==(b['source_inode'],b['source_device'])
raw=p.read_bytes();records=[]
for l in raw.decode(errors='replace').splitlines():
 if 'COSEARCH_OUTER_METRICS_JSON=' in l:
  x=json.JSONDecoder().raw_decode(l.split('COSEARCH_OUTER_METRICS_JSON=',1)[1])[0]
  assert all(math.isfinite(float(v)) for v in x['metrics'].values());records.append(x)
print(json.dumps(dict(job=3002,worker_pid=b['worker_pid'],process_start_ticks=b['process_start_ticks'],source=str(p),source_inode=st.st_ino,source_device=st.st_dev,source_sha256=hashlib.sha256(raw).hexdigest(),source_bytes=len(raw),checked_at=time.time(),outer=records,finite=True)))
CHECK
