import json,pathlib,subprocess,time,hashlib,os
root=pathlib.Path('/data/shaoshitong/search-r1-rl/tree-grpo-retrieval')
root.mkdir(parents=True,exist_ok=True)
(root/'logs').mkdir(exist_ok=True)
marker=root/'logs/preparation-started.json'
with marker.open('x') as f:json.dump({'epoch':time.time(),'scope':'own auxiliary retrieval environment; peer assets/environment read only','source_env':'/data/search-r1/env'},f)
subprocess.run(['cp','-a','/data/search-r1/env',str(root/'env')],check=True)
subprocess.run(['/usr/bin/python3.10','-m','venv','--without-pip',str(root/'env')],check=True)
subprocess.run(['cp','-a','/data/search-r1/retriever-gpu-lib',str(root/'retriever-gpu-lib')],check=True)
# Immutable inputs independently read and hash; do not trust source ready flags.
records={}
for name,expected in [('e5_Flat.index','69c98463fdb41fc08737d88513c597725f311c44f7ba6dca4b05d8c7c658d166'),('wiki-18.jsonl','85a787a692e73fdd657e86411bfd1ac810ed193b1eea4a478de449d4871c06b9')]:
 p=pathlib.Path('/data/search-r1/assets/wiki')/name;before=p.stat();h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 after=p.stat();assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
 assert h.hexdigest()==expected,(name,h.hexdigest())
 records[name]={'path':str(p),'size':after.st_size,'sha256':h.hexdigest(),'mtime_ns':after.st_mtime_ns,'inode':after.st_ino}
print(json.dumps(records),flush=True)
(root/'logs/assets-verified.json').write_text(json.dumps(records,indent=2)+'\n')
subprocess.run([str(root/'env/bin/python'),'-m','pip','freeze'],stdout=(root/'logs/environment-freeze.txt').open('w'),check=True)
(root/'logs/preparation-completed.json').write_text(json.dumps({'epoch':time.time(),'scope':'own env copy, independent immutable SHA verification; no GPU operation'})+'\n')
