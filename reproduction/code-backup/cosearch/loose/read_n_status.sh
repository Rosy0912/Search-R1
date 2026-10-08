set -euo pipefail
R=/dataset/shaoshitong/search-r1-rl/cosearch
squeue -h -j 3002 -o '%i %T %N'
"$R/env/bin/python" - <<'CHECK'
import pathlib,json,time
r=pathlib.Path('/dataset/shaoshitong/search-r1-rl/cosearch/runs/cosearch-local-gpu-five-20261003-n')
b=json.loads((r/'worker-log-provenance.json').read_text());p=pathlib.Path(b['source']);st=p.stat()
assert (st.st_ino,st.st_dev)==(b['source_inode'],b['source_device'])
lines=p.read_text(errors='replace').splitlines()
tags=('COSEARCH_BATCH_PROGRESS_JSON=','COSEARCH_RANK_PROGRESS_JSON=','COSEARCH_FIRST_BATCH_JSON ','COSEARCH_OUTER_METRICS_JSON=','Traceback','RuntimeError','OutOfMemoryError')
print(json.dumps(dict(source_bytes=st.st_size,mtime_age=time.time()-st.st_mtime,driver_exit=json.loads((r/'driver-exit.json').read_text()) if (r/'driver-exit.json').exists() else None,last_events=[l for l in lines if any(t in l for t in tags)][-5:])))
CHECK
