"""Small full-index HTTP equality gate. No training predictions or pressure load."""
import argparse,hashlib,json,os,time
from pathlib import Path
import requests
p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--oracle',default='http://10.120.17.101:18004');p.add_argument('--candidate',default='http://10.120.17.101:18006');a=p.parse_args()
b=Path(a.base);identity=json.loads((b/'receipts/source-identity.json').read_text());pid=identity['pid']
proc=Path('/proc',str(pid));assert proc.stat().st_uid==1005 and os.getuid()==1005
starttick=int((proc/'stat').read_text().rsplit(') ',1)[1].split()[19])
token=(b/'receipts/private-probe-token').read_text().strip()
old=Path('/proc/2239529');assert old.stat().st_uid==1005 and int((old/'stat').read_text().rsplit(') ',1)[1].split()[19])==914664099
health=requests.get(a.candidate+'/health',timeout=5).json();assert health['gpu_ready'] and health['gpu_shards']==8 and health['pid']==pid and health['source_identity_sha256']==identity['source_identity_sha256']
proof=dict(scope='eight_gpu_full_scan_and_http_oracle_equality',candidate_pid=pid,candidate_starttick=starttick,
 source_identity_sha256=identity['source_identity_sha256'],started_at=time.time(),accepted=False,
 ntotal=health['ntotal'],d=health['d'],tf32=health['tf32'],tests=[])
assert (proof['ntotal'],proof['d'],proof['tf32'])==(21015324,768,False)
queries=['What is the capital of France?']+['History of '+x for x in ('London','Paris','Tokyo','Beijing','Rome','Athens','Berlin','Cairo')]*8
for selected in (queries[:1],queries):
 payload=dict(queries=selected,topk=3,return_scores=True)
 t=time.perf_counter();oracle=requests.post(a.oracle+'/retrieve',json=payload,timeout=120);oracle.raise_for_status();ot=time.perf_counter()-t
 t=time.perf_counter();candidate=requests.post(a.candidate+'/probe/retrieve',json=payload,headers={'x-e1-probe-token':token},timeout=120);candidate.raise_for_status();ct=time.perf_counter()-t
 oj=oracle.json();cj=candidate.json();assert oj==cj,'Full-index document/score/order HTTP mismatch'
 assert len(cj['result'])==len(selected) and all(len(row)==3 for row in cj['result'])
 proof['tests'].append(dict(queries=len(selected),topk=3,full_scan=True,complete_json_equal=True,
  oracle_seconds=ot,candidate_seconds=ct,json_sha256=hashlib.sha256(json.dumps(cj,sort_keys=True).encode()).hexdigest()))
 print(json.dumps(proof['tests'][-1]),flush=True)
proof.update(accepted=True,finished_at=time.time(),limitations='66 queries across two requests; not exhaustive queries or training effect proof')
gate=b/'receipts/full-scan-http-gate.json';assert not gate.exists();gate.write_text(json.dumps(proof,indent=2))
admission=dict(pid=pid,source_identity_sha256=identity['source_identity_sha256'],accepted=True,gate_path=str(gate),gate_sha256=hashlib.sha256(gate.read_bytes()).hexdigest())
(b/'receipts/admit-requests.json').write_text(json.dumps(admission,indent=2));print(json.dumps(dict(accepted=True,gate=str(gate))))
