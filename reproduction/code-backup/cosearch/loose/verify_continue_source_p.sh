python3 - <<'PY'
from pathlib import Path
import json,hashlib,time
r=Path('/dataset/shaoshitong/search-r1-rl/cosearch/author-native');cp=r/'runs/author-native-checkpoint-fixed-five-p/checkpoints/global_step_5';out=[]
for role in ['actor','reranker_actor_rollout']:
 for rank in range(4):
  p=cp/role/f'native-integrity-rank-{rank}.json';x=json.loads(p.read_text())
  assert str(x['source_job'])=='3125'
  for n,v in x['files'].items():
   f=p.parent/n;h=hashlib.sha256()
   with f.open('rb') as s:
    for b in iter(lambda:s.read(8388608),b''):h.update(b)
   assert h.hexdigest()==v['sha256'] and f.stat().st_size==v['bytes'],str(f)
  out.append(dict(role=role,rank=rank,source_job=x['source_job'],integrity_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),files=x['files']))
assert len(out)==8
x=dict(checked_at=time.time(),source=cp.parent.parent.name,ranks=8,actual_full_file_shas_reverified=True,files=out,data_sha256=hashlib.sha256((cp/'data.pt').read_bytes()).hexdigest())
assert x['data_sha256']=='ba7a5fb645fd0b3ab552329fa73bdbc57a6b0fb94cc3eacd4e01e17afae9b14a'
(r/'continue-300-tools-q/source-checkpoint-verification.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x),flush=True)

PY
