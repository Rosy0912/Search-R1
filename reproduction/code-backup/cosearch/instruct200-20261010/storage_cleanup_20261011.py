from pathlib import Path
import hashlib,json,time,shutil,os
b=Path('/data/user/jxu957/experiments');co=b/'cosearch-continue300-x203-20261008';audit=co/'runtime/storage-cleanup-20261011';audit.mkdir(exist_ok=True)
def record(s,**kw):
 (audit/'status.json').write_text(json.dumps(dict(epoch=time.time(),status=s,**kw),indent=2))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
try:
 runs=co/'source/author-native/runs'
 record('deleting_authorized_cosearch_old_states')
 for name in ['author-native-grammar-warm-start-v2','author-native-continue-300-x','author-native-grammar-actor-continue-v4']:
  p=runs/name
  if p.exists():shutil.rmtree(str(p))
 record('cosearch_old_states_deleted_verifying_retrieval')
 src=co/'source/assets-root/assets';target=b/'topd-20261008/bundle/retrieval';receipts=[]
 for p in sorted(target.rglob('*')):
  if not p.is_file():continue
  rel=p.relative_to(target);q=src/rel
  record('verifying_retrieval',current=str(rel),verified=receipts)
  h=sha(p);hq=sha(q)
  assert h==hq,(str(rel),h,hq)
  receipts.append(dict(path=str(rel),sha256=h,bytes=p.stat().st_size))
 (audit/'retrieval-equality.json').write_text(json.dumps(receipts,indent=2))
 for name in ['wiki','e5-base-v2']:
  p=src/name;old=src/(name+'.verified-duplicate-removal')
  p.rename(old);p.symlink_to(target/name,target_is_directory=True)
  assert p.resolve()==(target/name).resolve()
  shutil.rmtree(str(old))
 record('COMPLETE',deleted_old_states=True,retrieval_shared_with_topd=True,verified=receipts,topd_mutated=False)
except Exception as e:
 record('FAILED',error=repr(e));raise
