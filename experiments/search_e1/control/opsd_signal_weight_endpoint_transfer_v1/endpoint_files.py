"""Pinned GP/GN raw15/25 evaluation assets only; CPU and remote storage only."""
import argparse,hashlib,json,os,sys
from pathlib import Path
S=Path('/weka_3/shaoshitong/search-e1-opsd-signal-weight-4gpu-20261010')
T=Path('/home/shaoshitong/search-e1-migration-20261011')
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def safe(path):
 p=Path(path);p.relative_to(S)
 if '..' in p.parts:raise ValueError('Traversal')
 return p
def inventory():
 refs={};ends=[]
 def add(ref):
  p=safe(ref['path'])
  if not p.is_file() or p.is_symlink():raise ValueError('Not regular source file')
  n=p.stat().st_size;sha=digest(p)
  if sha!=ref['sha256'] or n!=ref['bytes']:raise ValueError('Source reference differs: '+str(p))
  r=dict(path=str(p),bytes=n,sha256=sha)
  if str(p) in refs and refs[str(p)]!=r:raise ValueError('Conflicting refs')
  refs[str(p)]=r
 def newref(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p))
 for group in ['G_POS','G_NEG']:
  for outer in [14,24]:
   done=S/('group-'+group)/'runs/outers'/('outer-%05d'%outer)/'training/outer-complete.json'
   value=json.loads(done.read_text())
   if not value['accepted'] or value['outer_index']!=outer:raise ValueError('Wrong endpoint')
   add(newref(done));add(value['checkpoint']);cp=json.loads(Path(value['checkpoint']['path']).read_text())
   if not cp['accepted'] or cp['cursor']['outer_index']!=outer:raise ValueError('Wrong checkpoint')
   add(cp['outer_old_adapter']);add(cp['shared_optimizer'])
   for rank in cp['ranks']:add(rank['file'])
   for ref in cp['adapter']['files']:add(ref)
   for name in ['outer_manifest','pairs','order','encoding']:add(cp['training_input'][name])
   ends.append(dict(group=group,completed_outer=outer+1,checkpoint=value['checkpoint'],optimizer_updates=cp['optimizer_updates']))
 return dict(schema='signal-endpoint-transfer-v1',endpoints=ends,files=sorted(refs.values(),key=lambda r:r['path']),member_bytes=sum(r['bytes'] for r in refs.values()),source='HKUST4',target='HKUST5',no_gpu=True)
def install(value,stage):
 if value['schema']!='signal-endpoint-transfer-v1' or len(value['endpoints'])!=4:raise ValueError('Wrong inventory')
 checked=[]
 # Verify the entire staging set before installing any final file.
 for r in value['files']:
  p=safe(r['path']);relative=str(p).lstrip('/');src=stage/relative;dst=T/relative
  if src.is_symlink() or src.stat().st_size!=r['bytes'] or digest(src)!=r['sha256']:raise ValueError('Stage SHA mismatch')
  if dst.exists() and (dst.is_symlink() or dst.stat().st_size!=r['bytes'] or digest(dst)!=r['sha256']):raise ValueError('Existing destination differs; never overwrite')
  checked.append((r,src,dst))
 for r,src,dst in checked:
  dst.parent.mkdir(parents=True,exist_ok=True)
  if not dst.exists():os.link(src,dst)
  if digest(dst)!=r['sha256']:raise ValueError('Final SHA mismatch')
 return dict(accepted=True,files=len(checked),member_bytes=value['member_bytes'],endpoints=value['endpoints'],all_fresh_sha=True,original_sources_preserved=True,large_local_files=0)
def main():
 a=argparse.ArgumentParser();a.add_argument('mode',choices=['inventory','install']);a.add_argument('--inventory');a.add_argument('--inventory-sha');a.add_argument('--stage');v=a.parse_args()
 if v.mode=='inventory':out=inventory()
 else:
  p=Path(v.inventory)
  if digest(p)!=v.inventory_sha:raise ValueError('Inventory changed')
  out=install(json.loads(p.read_text()),Path(v.stage))
 print(json.dumps(out,sort_keys=True))
if __name__=='__main__':main()
