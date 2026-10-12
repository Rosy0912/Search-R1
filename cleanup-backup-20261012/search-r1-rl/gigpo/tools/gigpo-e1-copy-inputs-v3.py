from pathlib import Path
import subprocess,hashlib,json,datetime,socket,os
ROOT=Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004')
assert socket.gethostname()=='h100-8'
ROOT.mkdir(exist_ok=True)
fs=os.statvfs(ROOT);assert fs.f_bavail*fs.f_frsize>10*1024**3,'Insufficient free space before copy'
(ROOT/'assets').mkdir(exist_ok=True);(ROOT/'logs').mkdir(exist_ok=True)
for source,parent,name in [('/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/common-data',ROOT,'common-data'),('/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/assets/qwen2.5-3b-instruct-aa8e725',ROOT/'assets','qwen2.5-3b-instruct-aa8e725')]:
 target=parent/name
 assert not target.exists(),f'Reuse or inspect existing target, never overwrite: {target}'
 source_path=Path(source)
 cmd=['env','-u','LD_LIBRARY_PATH','ssh','-o','ConnectTimeout=15','-o','HostKeyAlias=gigpo-e1-common-source-h1009','-o','UserKnownHostsFile=/mnt/data/shaoshitong/search-r1-rl/gigpo/tools/gigpo-e1-h1009-known-hosts','-o','StrictHostKeyChecking=yes','-o','BatchMode=yes','h100-9','tar','-C',str(source_path.parent),'-cf','-',name]
 print('COPY_START',source,str(target),flush=True)
 producer=subprocess.Popen(cmd,stdout=subprocess.PIPE)
 consumer=subprocess.run(['tar','-C',str(parent),'-xf','-'],stdin=producer.stdout)
 producer.stdout.close();rc=producer.wait();assert rc==0 and consumer.returncode==0
 print('COPY_DONE',str(target),flush=True)
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
receipt=dict(time=datetime.datetime.now().isoformat(),host=socket.gethostname(),source_node='h100-9',destination_root=str(ROOT),common_manifest_sha256=sha(manifest),verified_split_files=verified,model_manifest_sha256=sha(mp),model_manifest=data,all_model_actual_files=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(model.iterdir()) if p.is_file()],scope='Offline same-H100-cluster asset copy only; no training or algorithm change')
out=ROOT/'logs/input-copy-and-full-hashes-v1.json';assert not out.exists();out.write_text(json.dumps(receipt,indent=2));print(json.dumps(dict(receipt=str(out),sha256=sha(out))),flush=True)
