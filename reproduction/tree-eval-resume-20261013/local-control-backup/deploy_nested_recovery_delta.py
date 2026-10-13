"""Publish immutable same-node code delta only; never submit training."""
import argparse,hashlib,json,shlex,subprocess,tarfile,tempfile
from pathlib import Path
import ssh_retained18 as ssh

parser=argparse.ArgumentParser()
parser.add_argument('--package',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
local=args.package.resolve()
manifest=json.loads((local/'manifest.json').read_text())
source=manifest['code_commit']
root='/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/candidates'
base=root+'/nested-branch-fresh200-0befbaf99fc44942105831b3fbec94e4bf1b8ccb'
target=root+'/nested-branch-fresh200-'+source
stage=target+'.delta-stage'
old_local=Path('work/tree-nested-shared-package-0befbaf/manifest.json')
old=json.loads(old_local.read_text())
old_sha=hashlib.sha256(old_local.read_bytes()).hexdigest()
manifest_sha=hashlib.sha256((local/'manifest.json').read_bytes()).hexdigest()
release_sha=hashlib.sha256((local/'release.json').read_bytes()).hexdigest()
needed=[name for name,h in manifest['files'].items() if old['files'].get(name)!=h]
needed+=['manifest.json','package-build.json','release.json']

def call(code):
    result=ssh.run('env CUDA_VISIBLE_DEVICES= python3 -c '+shlex.quote(code),timeout=120)
    if result.returncode:raise RuntimeError(result.stdout+result.stderr)
    return json.loads(result.stdout)

prepare='''import json,hashlib
from pathlib import Path
base=Path(BASE);target=Path(TARGET);stage=Path(STAGE)
assert base.is_dir() and not target.exists() and not stage.exists(),'Reconcile existing deployment before retry'
assert hashlib.sha256((base/'manifest.json').read_bytes()).hexdigest()==OLD_SHA
stage.mkdir()
print(json.dumps(dict(stage=str(stage),training_submitted=False)))
'''
for key,value in [('BASE',base),('TARGET',target),('STAGE',stage),('OLD_SHA',old_sha)]:prepare=prepare.replace(key,repr(value))
call(prepare)
with tempfile.TemporaryFile() as stream:
    with tarfile.open(fileobj=stream,mode='w:gz') as tar:
        for name in needed:
            rel=Path(name)
            if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe code path')
            tar.add(local/rel,arcname=name,recursive=False)
    stream.seek(0,2);compressed_bytes=stream.tell();stream.seek(0)
    subprocess.run(ssh.route('tar -xzf - -C '+shlex.quote(stage)),stdin=stream,check=True)
finish='''import json,hashlib,stat,shutil,time
from pathlib import Path
base=Path(BASE);stage=Path(STAGE);target=Path(TARGET)
assert not target.exists()
assert hashlib.sha256((stage/'manifest.json').read_bytes()).hexdigest()==MANIFEST_SHA
old=json.loads((base/'manifest.json').read_text());m=json.loads((stage/'manifest.json').read_text());reused=0
for name,expected in m['files'].items():
 rel=Path(name);assert not rel.is_absolute() and '..' not in rel.parts
 dst=stage/rel
 if not dst.exists():
  src=base/rel;assert old['files'].get(name)==expected and stat.S_ISREG(src.lstat().st_mode)
  dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst);reused+=1
 assert stat.S_ISREG(dst.lstat().st_mode) and hashlib.sha256(dst.read_bytes()).hexdigest()==expected,name
assert hashlib.sha256((stage/'release.json').read_bytes()).hexdigest()==RELEASE_SHA
assert target.name=='nested-branch-fresh200-'+m['code_commit']
stage.rename(target)
print(json.dumps(dict(status='REMOTE_RECOVERY_CODE_DELTA_BYTES_VERIFIED',epoch=time.time(),source_commit=m['code_commit'],package=str(target),manifest_sha256=MANIFEST_SHA,release_sha256=RELEASE_SHA,verified_files=len(m['files']),reused_files=reused,training_submitted=False)))
'''
for key,value in [('BASE',base),('STAGE',stage),('TARGET',target),('MANIFEST_SHA',manifest_sha),('RELEASE_SHA',release_sha)]:finish=finish.replace(key,repr(value))
receipt=call(finish);receipt.update(compressed_delta_bytes=compressed_bytes,transferred_files=len(needed))
args.output.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
