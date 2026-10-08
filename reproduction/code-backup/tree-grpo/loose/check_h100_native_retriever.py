"""Bind own immutable acceptance to a live same-cluster Slurm service instance."""
import argparse, hashlib, json, pathlib, time, urllib.request
DIRECT=urllib.request.build_opener(urllib.request.ProxyHandler({}))
p=argparse.ArgumentParser(); p.add_argument('--receipt',type=pathlib.Path,required=True); p.add_argument('--sha256',required=True); p.add_argument('--stage',choices=['two','five','formal'],required=True); p.add_argument('--required-seconds',type=int,required=True); a=p.parse_args()
assert a.required_seconds>=0
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
assert a.receipt.resolve().parent==(root/'logs').resolve()
raw=a.receipt.read_bytes(); assert hashlib.sha256(raw).hexdigest()==a.sha256
r=json.loads(raw); assert r['status']=='ACCEPTED_H100_GPU_RETRIEVAL_AFTER_INDEPENDENT_REVIEW' and a.stage in r['permitted_stages']
required={'baseline','gate','independent_review','service_source','guard_source','service_launcher_source','checker_source','training_launcher_source','probe_source'}
assert set(r['evidence_roles'])==required and r['evidence_file_sha256']
for role,f in r['evidence_roles'].items(): assert f in r['evidence_file_sha256']
for f,h in r['evidence_file_sha256'].items(): assert hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest()==h
with DIRECT.open('http://10.32.0.11:18082/provenance',timeout=10) as f: live=json.load(f)
assert live==r['instance']
for key,role in [('source_sha256','service_source'),('guard_source_sha256','guard_source'),('launcher_source_sha256','service_launcher_source')]:assert live[key]==r['evidence_file_sha256'][r['evidence_roles'][role]]
assert live['cluster']=='wuhan-h100' and live['node']=='h100-11' and live['training_node']=='h100-9'
assert live['deadline_epoch']-time.time()>=a.required_seconds
print(json.dumps({'status':'BOUND_LIVE_SAME_CLUSTER_RETRIEVER','instance':live,'receipt_sha256':a.sha256}),flush=True)
