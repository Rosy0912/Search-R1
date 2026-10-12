import json,hashlib,subprocess,os
from pathlib import Path
root=Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
candidate=root/'logs/deployment-binding-bb1adde-candidate-v1.json';review=root/'logs/DEPLOYMENT-INDEPENDENT-REVIEW.json'
r=json.loads(review.read_text());assert r['status']=='PASS';assert r['candidate_sha256']==sha(candidate)
v=json.loads(candidate.read_text());assert v['status']=='PENDING_INDEPENDENT_DEPLOYMENT_REVIEW';v['status']='REVIEWED';v['review_path']=str(review);v['review_sha256']=sha(review);v['evidence'].append(dict(path=str(review),sha256=sha(review)))
binding=root/'logs/deployment-binding-bb1adde-v1.json';intent=root/'logs/formal-submit-intent-v1.json';receipt=root/'logs/formal-submission-v1.json'
assert not binding.exists() and not intent.exists() and not receipt.exists(),'Existing intent: never submit twice'
def exclusive_write(path,value):
 with path.open('x') as f:
  f.write(json.dumps(value,indent=2));f.flush();os.fsync(f.fileno())
exclusive_write(intent,dict(binding=str(binding),candidate_sha256=sha(candidate),status='CLAIMED_NO_RESUBMISSION'))
exclusive_write(binding,v);h=sha(binding)
p=subprocess.run(['sbatch','--parsable',str(root/'tools/formal_v3.sbatch'),str(binding),h],capture_output=True,text=True,check=True)
job=p.stdout.strip().split(';')[0];assert job.isdigit();exclusive_write(receipt,dict(job=job,binding=str(binding),binding_sha256=h,stdout=p.stdout));print(receipt.read_text())
