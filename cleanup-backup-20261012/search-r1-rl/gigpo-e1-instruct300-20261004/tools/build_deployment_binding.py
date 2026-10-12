import json,hashlib,subprocess,shutil
from pathlib import Path
root=Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
 return h.hexdigest()
evidence={}
def add(p,s=None):
 p=str(p);evidence[p]=dict(path=p,sha256=s or sha(p))
for x in json.loads((root/'logs/native-binding-v5.json').read_text())['files']:add(x['path'],x['sha256'])
for n in ['actual-environment-full-sha-v1.json','actual-config-data-tokenizer-validation.json','input-copy-and-full-hashes-v4.json','retrieval-assets-full-sha-common-identity.json','FORMAL-V3-STATIC-INDEPENDENT-REVIEW.json','NATIVE3184-INDEPENDENT-REVIEW.json']:
 p=root/'logs'/n;v=json.loads(p.read_text());add(p)
 if n=='actual-environment-full-sha-v1.json':
  for x in v['files']:add(x['path'],x['sha256'])
 if n=='input-copy-and-full-hashes-v4.json':
  for x in v['all_model_actual_files']+v['verified_split_files']:add(x['path'],x['sha256'])
 if n=='retrieval-assets-full-sha-common-identity.json':
  assert v['status']=='PASS'
  for x in v['records']:add(x['path'],x['sha256'])
 if n=='NATIVE3184-INDEPENDENT-REVIEW.json':assert v['status']=='PASS'
for p in (root/'common-data').rglob('*'):
 if p.is_file():add(p)
for n in ['formal_run_v3.py','formal_v3.sbatch','ray-audit-environment-probe-v3.py','check_http_retrieval.py','verify-gigpo-checkpoint.py','checkpoint_semantics.py','common-retrieval-identity.json','verify_retrieval_assets.py','collect_environment.py']:add(root/'tools'/n)
add(root/'logs/cpu-gpu-retrieval-comparison.json')
r=root/'runs/gigpo-e1-native-job3184';add(r/'result.json')
v=dict(status='PENDING_INDEPENDENT_DEPLOYMENT_REVIEW',code_commit=subprocess.check_output(['git','-C',str(root/'repo'),'rev-parse','HEAD'],text=True).strip(),fresh_model_revision='aa8e72537993ba99e69dfaafa59ed015b17504d1',common_manifest_sha256='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24',native_status='PASS',native_job='3184',evidence=list(evidence.values()),disk=shutil.disk_usage(root)._asdict(),target_actual_outer=300,first5_same_process=True,checkpoints=[5]+list(range(25,301,25)),node='h100-8')
p=root/'logs/deployment-binding-bb1adde-candidate-v1.json';assert not p.exists();p.write_text(json.dumps(v,indent=2));print(json.dumps(dict(path=str(p),sha256=sha(p),evidence_count=len(evidence),disk=v['disk'])))
