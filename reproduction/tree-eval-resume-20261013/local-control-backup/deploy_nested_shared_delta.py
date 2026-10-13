"""Reuse immutable remote code bytes, stream only changed compressed code."""
import hashlib,importlib.util,json,shlex,subprocess,tarfile,tempfile
from pathlib import Path
HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location('ssh',HERE/'ssh_retained18.py')
ssh=importlib.util.module_from_spec(spec);spec.loader.exec_module(ssh)
local=Path('work/tree-nested-shared-package-0befbaf')
root='/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/candidates'
target=root+'/nested-branch-fresh200-0befbaf99fc44942105831b3fbec94e4bf1b8ccb'
base=root+'/nested-branch-fresh200-0a123e4a0265134e5c9871e7d7a1251f2425cd4f'
stage=target+'.delta-stage'
manifest_sha=hashlib.sha256((local/'manifest.json').read_bytes()).hexdigest()
release_sha=hashlib.sha256((local/'release.json').read_bytes()).hexdigest()
def call(code,timeout=120):
 r=ssh.run('python3 -c '+shlex.quote(code),timeout=timeout)
 if r.returncode:raise RuntimeError(r.stdout+r.stderr)
 return json.loads(r.stdout)
prepare='''import hashlib,json,os,shutil,stat,time
from pathlib import Path
target=Path(TARGET);base=Path(BASE);stage=Path(STAGE)
assert not Path('/proc/300323').exists(),'Prior owned transfer still exists; reconcile, never race'
assert target.is_dir() and base.is_dir() and not stage.exists()
assert hashlib.sha256((target/'manifest.json').read_bytes()).hexdigest()==MANIFEST_SHA
new=json.loads((target/'manifest.json').read_text());old=json.loads((base/'manifest.json').read_text())
assert old['code_commit']=='0a123e4a0265134e5c9871e7d7a1251f2425cd4f'
assert new['code_commit']=='0befbaf99fc44942105831b3fbec94e4bf1b8ccb'
partial=target.with_name(target.name+'.partial-transfer-'+str(int(time.time())));target.rename(partial)
stage.mkdir();shutil.copyfile(partial/'manifest.json',stage/'manifest.json');needed=[];reused=0
for name,expected in new['files'].items():
 rel=Path(name);assert not rel.is_absolute() and '..' not in rel.parts
 src=base/rel;dst=stage/rel;dst.parent.mkdir(parents=True,exist_ok=True)
 if old['files'].get(name)==expected and src.is_file() and stat.S_ISREG(src.lstat().st_mode):
  assert hashlib.sha256(src.read_bytes()).hexdigest()==expected,name
  shutil.copyfile(src,dst);reused+=1
 else:needed.append(name)
needed+=['package-build.json','release.json']
print(json.dumps(dict(stage=str(stage),partial_preserved=str(partial),needed=needed,reused_files=reused)))
'''
for key,val in [('TARGET',target),('BASE',base),('STAGE',stage),('MANIFEST_SHA',manifest_sha)]:prepare=prepare.replace(key,repr(val))
plan=call(prepare)
Path('outputs/tree-nested-branch-handoff-20261013/DELTA_DEPLOYMENT_PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
with tempfile.TemporaryFile() as archive:
 with tarfile.open(fileobj=archive,mode='w:gz') as tar:
  for name in plan['needed']:
   rel=Path(name);assert not rel.is_absolute() and '..' not in rel.parts
   tar.add(local/rel,arcname=name,recursive=False)
 archive.seek(0,2);compressed_bytes=archive.tell();archive.seek(0)
 subprocess.run(ssh.route('tar -xzf - -C '+shlex.quote(stage)),stdin=archive,check=True)
finalize='''import hashlib,json,time
from pathlib import Path
stage=Path(STAGE);target=Path(TARGET)
assert not target.exists()
assert hashlib.sha256((stage/'manifest.json').read_bytes()).hexdigest()==MANIFEST_SHA
m=json.loads((stage/'manifest.json').read_text())
assert all(hashlib.sha256((stage/name).read_bytes()).hexdigest()==expected for name,expected in m['files'].items())
assert hashlib.sha256((stage/'release.json').read_bytes()).hexdigest()==RELEASE_SHA
stage.rename(target)
print(json.dumps(dict(status='REMOTE_CODE_DELTA_ALL_BYTES_VERIFIED',observed_epoch=time.time(),package=str(target),manifest_sha256=MANIFEST_SHA,release_sha256=RELEASE_SHA,source_commit=m['code_commit'],verified_files=len(m['files']),training_submitted=False)))
'''
for key,val in [('TARGET',target),('STAGE',stage),('MANIFEST_SHA',manifest_sha),('RELEASE_SHA',release_sha)]:finalize=finalize.replace(key,repr(val))
receipt=call(finalize);receipt.update(compressed_delta_bytes=compressed_bytes,reused_files=plan['reused_files'],transferred_files=len(plan['needed']))
Path('outputs/tree-nested-branch-handoff-20261013/DELTA_DEPLOYMENT.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
