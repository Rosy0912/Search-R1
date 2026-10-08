/dataset/shaoshitong/search-r1-rl/cosearch/env/bin/python - <<'PYCODE'
import json,time,hashlib,os
from pathlib import Path
r=Path('/dataset/shaoshitong/search-r1-rl/cosearch/runs/cosearch-local-gpu-five-20261003-d')
session=Path('/dataset/shaoshitong/search-r1-rl/cosearch/session_2026-10-03_11-36-08_043143_1345803')
p=session/'logs/worker-dbc997fb9294c224f0fb983b51e8d4a0ebb6607de3f77123c00d1478-01000000-1348811.out'
pid=1348811;proc=Path('/proc')/str(pid)
source_stat=p.stat();verified_live=proc.is_dir()
if verified_live:
 cgroup=(proc/'cgroup').read_text();assert '/job_2901/' in cgroup
 stat=(proc/'stat').read_text().rsplit(')',1)[1].split();start_ticks=int(stat[19]);assert start_ticks==277280496
 live_binding_at=time.time()
else:
 prior=json.loads((r/'worker-log-provenance.json').read_text())
 assert prior['process_cgroup_verified'] and prior['process_start_ticks']==277280496
 assert prior['source']==str(p) and prior['source_inode']==source_stat.st_ino and prior['source_device']==source_stat.st_dev
 assert (r/'driver-exit.json').is_file()
 cgroup=prior['cgroup'];start_ticks=prior['process_start_ticks'];live_binding_at=prior['live_binding_at']
payload=p.read_bytes();dest=r/'worker-training.log';temp=dest.with_suffix('.pending');temp.write_bytes(payload);os.replace(temp,dest)
record=dict(captured_at=time.time(),job=2901,session=str(session),source=str(p),worker_pid=pid,destination=str(dest),process_start_ticks=start_ticks,cgroup=cgroup,process_cgroup_verified=True,currently_live=verified_live,live_binding_at=live_binding_at,source_inode=source_stat.st_ino,source_device=source_stat.st_dev,sha256=hashlib.sha256(payload).hexdigest(),bytes=len(payload))
prov=r/'worker-log-provenance.json';temp=prov.with_suffix('.pending');temp.write_text(json.dumps(record,indent=2)+'\n');os.replace(temp,prov)
print(json.dumps([line for line in payload.decode(errors='replace').splitlines() if 'COSEARCH_' in line or 'epoch ' in line][-6:]))

PYCODE
