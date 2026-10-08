import hashlib,json,tarfile
from pathlib import Path
root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo');out=root/'logs/shared-five-2956-review';out.mkdir(exist_ok=True)
with tarfile.open(root/'logs/five-promotion-bundle.tar.gz') as tar:
 for member in tar.getmembers():
  assert member.isfile() and Path(member.name).name==member.name
  data=tar.extractfile(member).read();target=out/member.name
  if target.exists():assert target.read_bytes()==data
  else:
   with target.open('xb') as stream:stream.write(data)
p=out/'shared-five-2956-acceptance.json';assert hashlib.sha256(p.read_bytes()).hexdigest()=='46174c92932f35f30c836a2fd7b28043b706569bc1ef7d54d7a7b0489fcbe598'
v=json.loads(p.read_text());assert v['status']=='PASS' and v['stage']=='five' and v['code_commit']=='4d133efeb1f675139d2558640f1e21d50780f133' and v['actual_outer_updates']==5
for x in v['evidence']:
 h=hashlib.sha256()
 with Path(x['path']).open('rb') as stream:
  for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b)
 assert h.hexdigest()==x['sha256'],x['path']
print('ALL_COMBINED_PROMOTION_EVIDENCE_SHA_VERIFIED',len(v['evidence']),flush=True)
