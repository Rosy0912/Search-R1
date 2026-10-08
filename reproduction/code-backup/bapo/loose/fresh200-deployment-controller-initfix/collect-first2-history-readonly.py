from pathlib import Path
import sys,json,hashlib,base64,gzip,datetime
R=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-controller-initfix-20261006');S='60f566b3ddabc88ce9de56ead5ad30a3d6795dd2';RUN=R/'runs/formal200-bab8981d9fd14afc98923eb4ebb89e33-3315'
B=R/'first5-evidence-tools-3315-v3';assert hashlib.sha256((B/'first5_helpers.py').read_bytes()).hexdigest()=='716f9410465c2f642b5cf9af7cbf6d7f503ff43c58aeb4e195f2c82deae525df';sys.path.insert(0,str(B))
from first5_helpers import bind_source,immutable_prefix,verify_snapshot_prefix
bind_source();out=RUN/'logs/history-first2-readonly-20261007';out.mkdir();snapshot=out/'train-prefix.log';binding=immutable_prefix(RUN/'train.log',snapshot);verify_snapshot_prefix(binding);raw=snapshot.read_bytes()
records=[]
for line in raw.decode().splitlines():
 marker='FRESH200_ATTEMPT_JSON='
 if marker in line:records.append(json.loads(line.split(marker,1)[1]))
assert len(records)==2 and [a['candidate_outer']for a in records]==[1,2] and all(a['accepted_complete_outer']for a in records)
ledgers=[]
for a in records:
 p=Path(a['ledger_path']);assert p==RUN/'checkpoints/diagnostics'/('attempt_'+str(a['attempt_id'])+'_candidate_outer_'+str(a['candidate_outer']))/'all-rank-write-ledger.json';data=p.read_bytes();assert json.loads(data)==a and a['source_commit']==S
 ledgers.append(dict(original_path=str(p),bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),text=data.decode()))
verify_snapshot_prefix(binding)
value=dict(status='ACTUAL_FIRST2_HISTORY_PREFIX_ONLY_NOT_FINAL',observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source=S,job_id=3315,run_root=str(RUN),train_prefix_binding=binding,train_prefix_bytes=len(raw),train_prefix_sha256=hashlib.sha256(raw).hexdigest(),train_prefix_gzip_base64=base64.b64encode(gzip.compress(raw,mtime=0)).decode(),ledgers=ledgers)
with (out/'prefix-binding.json').open('x')as f:json.dump(binding,f)
print(json.dumps(value,allow_nan=False))
