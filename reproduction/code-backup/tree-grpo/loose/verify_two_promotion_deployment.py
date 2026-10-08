import hashlib,json,tarfile
from pathlib import Path
root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo');out=root/'logs/shared-two-2934-review';out.mkdir(exist_ok=True)
with tarfile.open(root/'logs/two-promotion-bundle.tar.gz') as tar:
 for member in tar.getmembers():
  assert member.isfile() and Path(member.name).name==member.name
  data=tar.extractfile(member).read();target=out/member.name
  if target.exists():assert target.read_bytes()==data
  else:
   with target.open('xb') as stream:stream.write(data)
p=out/'shared-two-2934-combined-acceptance.json';assert hashlib.sha256(p.read_bytes()).hexdigest()=='b34557770089ba1e5c1959d8ceb26602b2a84df672b56b85379f616c56669d14'
v=json.loads(p.read_text());assert v['original_job_failed'] and v['scientific_training_accepted'] and v['operational_repair_accepted'] and v['permit_five_only']
for x in v['evidence']:
 h=hashlib.sha256()
 with Path(x['path']).open('rb') as stream:
  for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b)
 assert h.hexdigest()==x['sha256'],x['path']
print('ALL_COMBINED_PROMOTION_EVIDENCE_SHA_VERIFIED',len(v['evidence']),flush=True)
