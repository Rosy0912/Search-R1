"""Remove only verified duplicate bootstrap artifacts after retained copies pass."""
import sys,json,subprocess,shlex,hashlib,time
from pathlib import Path
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control');import control
root=Path('/Users/shaoshitong/Documents/Search-G1/preparation')
manifest=Path('/Users/shaoshitong/Documents/remote_codex_control/.transfer-staging/g1-common8-h100-assets/source-hashes.json')
verified=json.loads((root/'assets-target-verification.json').read_text())
r=subprocess.run(control.route('h100-8','cat /dataset/shaoshitong/Search-G1-common8/preparation/source-hashes.json'),capture_output=True,timeout=55);assert r.returncode==0
assert verified['accepted'] and hashlib.sha256(r.stdout).hexdigest()==verified['source_hash_manifest_sha256']
expected=json.loads(r.stdout)
(root/'bootstrap-verified-source-manifest.json').write_bytes(r.stdout)
selected={n:h for n,h in expected.items() if n.startswith('wheelhouse/') or n.startswith('env/')}
source_code="""import json,hashlib,os
from pathlib import Path
base=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
expected=json.loads(EXPECTED)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
proof={}
for n,h in expected.items():
 p=base/n;assert p.is_file() and not p.is_symlink() and sha(p)==h
 proof[n]={'sha256':h,'size':p.stat().st_size}
print(json.dumps(proof))
""".replace('EXPECTED',repr(json.dumps(selected)))
r=subprocess.run(control.route('h100-9','nice -n 19 python3 -c '+shlex.quote(source_code)),capture_output=True,text=True,timeout=55);assert r.returncode==0,r.stderr
source=json.loads(r.stdout)
(root/'bootstrap-retained-source-proof.json').write_text(json.dumps(dict(source_node='h100-9',files=source,checked_unix=time.time()),indent=2)+'\n')
code="""import json,hashlib,os,time,shutil
from pathlib import Path
base=Path('/dataset/shaoshitong/Search-G1-common8');artifact=base/'bootstrap-assets';expected=json.loads(EXPECTED);source=json.loads(SOURCE)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
assert base.resolve()==base and artifact.resolve()==artifact
proof={'phase':'VERIFIED_BEFORE_CLEANUP','checked_unix':time.time(),'files':{},'scope':'only own verified duplicate wheelhouse and staged FlashAttention; installed env/model/data retained','free_before':shutil.disk_usage(base).free}
actual={str(p.relative_to(artifact)) for prefix in ('wheelhouse','env') for p in (artifact/prefix).rglob('*') if p.is_file()}
assert actual==set(expected), 'Unknown bootstrap file; refuse broad deletion'
for n,h in expected.items():
 p=artifact/n;stat=p.stat();assert not p.is_symlink() and stat.st_uid==os.getuid()
 assert sha(p)==h and stat.st_size==source[n]['size'] and source[n]['sha256']==h
 if n.startswith('env/') and p.suffix!='.pyc':
  retained=base/n;assert retained.is_file() and not retained.is_symlink() and retained.resolve()!=p.resolve()
  assert sha(retained)==h, 'Installed runtime does not retain original bytes'
 proof['files'][n]={'sha256':h,'bytes':stat.st_size,'device':stat.st_dev,'inode':stat.st_ino}
claim=base/'logs/bootstrap-duplicate-cleanup.json';assert not claim.exists()
with claim.open('x') as f:json.dump(proof,f,indent=2);f.flush();os.fsync(f.fileno())
for n,item in proof['files'].items():
 p=artifact/n;stat=p.stat();assert not p.is_symlink() and (stat.st_dev,stat.st_ino,stat.st_uid)==(item['device'],item['inode'],os.getuid());p.unlink()
for prefix in ('wheelhouse','env'):
 parent=artifact/prefix
 for p in sorted(parent.rglob('*'),key=lambda p:len(p.parts),reverse=True):
  assert p.is_dir() and not p.is_symlink();p.rmdir()
 parent.rmdir()
proof.update(phase='CLEANED',free_after=shutil.disk_usage(base).free,reclaimed_payload_bytes=sum(i['bytes'] for i in proof['files'].values()),finished_unix=time.time())
tmp=claim.with_suffix('.tmp');tmp.write_text(json.dumps(proof,indent=2));os.replace(tmp,claim)
print(json.dumps(proof))
""".replace('EXPECTED',repr(json.dumps(selected))).replace('SOURCE',repr(json.dumps(source)))
r=subprocess.run(control.route('h100-8','nice -n 19 python3 -c '+shlex.quote(code)),capture_output=True,text=True,timeout=55)
assert r.returncode==0,r.stderr
proof=json.loads(r.stdout);(root/'bootstrap-duplicate-cleanup.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps({k:v for k,v in proof.items() if k!='files'},indent=2))
