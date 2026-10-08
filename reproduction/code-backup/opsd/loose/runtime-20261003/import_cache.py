import hashlib,json,os,shutil,sys,time
from pathlib import Path
root=Path('/mnt/data/shaoshitong/search-r1-rl/opsd')
name=sys.argv[1]
assert name in ('part_aa','part_ab')
source=Path('/mnt/shared/workspace/training_data_node_5/shaoshitong/search-r1-rl/bapo/assets-root/assets/wiki')/name
dest=root/'assets-root/assets/wiki'/name
assert not dest.exists()
manifest=json.loads((root/'assets-root/assets.json').read_text())
a=next(a for a in manifest if a['path']=='wiki/'+name)
def identity(p):
 s=p.stat();return dict(device=s.st_dev,inode=s.st_ino,size=s.st_size,mtime_ns=s.st_mtime_ns)
before=identity(source);assert before['size']==a['size']
partial=dest.with_name(name+'.cache-import.partial')
assert not partial.exists(), 'Inspect existing cache candidate before retry'
start=time.monotonic();print('COPY_START',name,before,flush=True)
shutil.copyfile(source,partial)
h=hashlib.sha256()
with partial.open('rb') as f:
 for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
after=identity(source)
assert before==after, 'Source changed while copying'
assert partial.stat().st_size==a['size'] and h.hexdigest()==a['sha256']
partial.replace(dest)
record=dict(source=str(source),target=str(dest),source_before=before,source_after=after,size=a['size'],sha256=h.hexdigest(),seconds=time.monotonic()-start)
(root/'logs'/('cache-import-'+name+'-20261003.json')).write_text(json.dumps(record,indent=2))
print('CACHE_IMPORT_FULL_SHA_VERIFIED',json.dumps(record),flush=True)
