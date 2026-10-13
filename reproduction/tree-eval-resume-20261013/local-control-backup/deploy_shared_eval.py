import importlib.util,json,base64,hashlib,shlex
from pathlib import Path
spec=importlib.util.spec_from_file_location('ssh',Path(__file__).with_name('ssh_retained18.py'));ssh=importlib.util.module_from_spec(spec);spec.loader.exec_module(ssh)
commit='1c15c492cf5619da9225dd60acdd3688653df614'
root='/dataset/shaoshitong/search-r1-rl/tree-grpo';dest=root+'/instruct300/eval-shared-'+commit
src=Path(__file__).parent/'branpo-source/reproduction/tree-eval-resume-20261013'
old=json.loads(Path('outputs/tree-eval-resume-20261013/DEPLOYMENT.txt').read_text().split('\n\nexit')[0])['release']
review=json.loads((src/'SHARED_CODE_REVIEW.json').read_text())
release=dict(old,files=review['files'],source_commit=commit,resource_protocol='memory_headroom_shared_v1',human_authorization='Latest explicit human instruction permits cohosting on four same-cluster H100 nodes with available memory; no requirement for empty/exclusive GPUs; keep foreign processes unchanged')
files={n:(src/n).read_bytes() for n in [*review['files'],'CODE_REVIEW.json','SHARED_CODE_REVIEW.json','README.md']}
files['release.json']=(json.dumps(release,indent=2)+'\n').encode()
def call(code):
 r=ssh.run('python3 -c '+shlex.quote(code),timeout=120)
 if r.returncode:raise RuntimeError(r.stderr+r.stdout)
 return r.stdout
print(call('from pathlib import Path\np=Path('+repr(dest)+')\np.mkdir()'))
groups=[];group={};size=0
for n,data in files.items():
 if n in review['files']:assert hashlib.sha256(data).hexdigest()==review['files'][n]
 if size+len(data)>20000 and group:groups.append(group);group={};size=0
 group[n]=base64.b64encode(data).decode();size+=len(data)
if group:groups.append(group)
for group in groups:
 code='import base64,os,hashlib\nfrom pathlib import Path\nfiles='+repr(group)+'\nfor n,b64 in files.items():\n p=Path('+repr(dest)+')/n;data=base64.b64decode(b64)\n with p.open("xb") as f:f.write(data);f.flush();os.fsync(f.fileno())\n assert p.read_bytes()==data'
 call(code)
receipt=dict(tools=dest,release_sha256=hashlib.sha256(files['release.json']).hexdigest(),release=release,bytes=sum(map(len,files.values())))
Path('outputs/tree-eval-resume-20261013/SHARED_DEPLOYMENT.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
