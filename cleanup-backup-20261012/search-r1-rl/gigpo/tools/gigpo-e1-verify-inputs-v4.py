from pathlib import Path
import hashlib,json,datetime,socket
ROOT=Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004')
assert socket.gethostname()=='h100-8'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
 return h.hexdigest()
common=ROOT/'common-data';manifest=common/'manifest.json'
assert sha(manifest)=='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
m=json.loads(manifest.read_text());verified=[]
for name,n in [('train',169028),('validation',512),('stage',64),('final',51713)]:
 d=m['splits'][name];p=common/d['path'];assert d['rows']==n and sha(p)==d['sha256'];verified.append(dict(split=name,path=str(p),rows=n,sha256=d['sha256']))
model=ROOT/'assets/qwen2.5-3b-instruct-aa8e725';mp=model/'VERIFIED-MODEL-MANIFEST.json';data=json.loads(mp.read_text())
assert data['repo']=='Qwen/Qwen2.5-3B-Instruct' and data['revision']=='aa8e72537993ba99e69dfaafa59ed015b17504d1'
for d in data['files']:
 p=model/d['file'];assert p.stat().st_size==d['bytes'] and sha(p)==d['sha256']
 if d['expected_lfs_sha256'] is not None:assert d['expected_lfs_sha256']==d['sha256']
 else:assert hashlib.sha1(b'blob '+str(p.stat().st_size).encode()+b'\0'+p.read_bytes()).hexdigest()==d['git_blob_sha1']
for d in m['preparation']['raw_sources']:assert sha(common/d['path'])==d['sha256']
receipt=dict(time=datetime.datetime.now().isoformat(),host=socket.gethostname(),source_node='h100-9',destination_root=str(ROOT),common_manifest_sha256=sha(manifest),verified_split_files=verified,model_manifest_sha256=sha(mp),model_manifest=data,all_model_actual_files=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(model.iterdir()) if p.is_file()],scope='Full original prebuilt manifest/splits/raw source hashes and official revision LFS+git identities after same-H100 offline relay; no training')
out=ROOT/'logs/input-copy-and-full-hashes-v4.json';assert not out.exists();out.write_text(json.dumps(receipt,indent=2));print(json.dumps(dict(receipt=str(out),sha256=sha(out))),flush=True)
