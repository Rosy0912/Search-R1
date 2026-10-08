"""Copy only pinned H100-cluster input bytes into the new OPSD root; no compute."""
import importlib.util, json, pathlib, shlex, subprocess, sys, fcntl
HERE=pathlib.Path(__file__).resolve().parent
ROOT='/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004'
SOURCE='/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300'
spec=importlib.util.spec_from_file_location('tree_management_route','/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/work/remote.py')
route=importlib.util.module_from_spec(spec);spec.loader.exec_module(route)
opts=['-T','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2','-o','ControlMaster=no','-o','ControlPath=none']
def destination(command):
 worker='env -u LD_LIBRARY_PATH ssh '+shlex.join(opts+['h100-10',command])
 head='env -u LD_LIBRARY_PATH ssh '+shlex.join(opts+['-o','HostName=10.33.0.16','-o','HostKeyAlias=100.108.173.4','-o','StrictHostKeyChecking=yes','h100-16',worker])
 return ['ssh']+opts+['-o','ProxyCommand=/usr/bin/nc -X 5 -x 127.0.0.1:7892 %h %p','mgmt-host',head]
model=json.loads((HERE/'official-model-manifest-original.json').read_text());data=json.loads((HERE/'common-manifest-original.json').read_text())
files=['common-data/manifest.json','common-data/exclusions-audit.json']+['common-data/'+v['path'] for v in data['splits'].values()]
modelprefix='assets/qwen2.5-3b-instruct-aa8e725/'
files += [modelprefix+'VERIFIED-MODEL-MANIFEST.json']+[modelprefix+v['file'] for v in model['files']]
assert all(pathlib.PurePosixPath(p).is_relative_to('common-data') or pathlib.PurePosixPath(p).is_relative_to('assets/qwen2.5-3b-instruct-aa8e725') for p in files)
lock=(HERE/'asset-copy.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
prep=f'''import pathlib,json,socket
r=pathlib.Path({ROOT!r})
assert socket.gethostname()=='h100-10'
r.mkdir(exist_ok=True)
assert r.is_dir() and not r.is_symlink()
p=r/'asset-transfer-intent.json'
with p.open('x') as f: json.dump(dict(source_host='h100-9',target_host='h100-10',files={files!r}),f)
print('OWNED_NEW_ASSET_ROOT_READY',flush=True)
'''
subprocess.run(destination('python3 -c '+shlex.quote(prep)),check=True)
extract=f'''import pathlib,subprocess
r=pathlib.Path({ROOT!r})
assert (r/'asset-transfer-intent.json').is_file()
subprocess.run(['tar','xf','-','-C',str(r)],check=True)
'''
source=subprocess.Popen(route.route(shlex.join(['tar','cf','-','-C',SOURCE]+files)),stdout=subprocess.PIPE)
target=subprocess.Popen(destination('python3 -c '+shlex.quote(extract)),stdin=source.stdout)
source.stdout.close()
t=target.wait();s=source.wait()
print(json.dumps(dict(source_exit=s,target_exit=t,root=ROOT,files=len(files))),flush=True)
if s or t:sys.exit(1)
verify=f'''import pathlib,json,hashlib,socket
r=pathlib.Path({ROOT!r})
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''): h.update(b)
 return h.hexdigest()
m=r/'assets/qwen2.5-3b-instruct-aa8e725/VERIFIED-MODEL-MANIFEST.json';d=r/'common-data/manifest.json'
assert sha(m)=='7f53ce6c010ac076b1ec482031bea7c9e6b8d0a096a61b880fe8ad4971bc9e44'
assert sha(d)=='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
model=json.loads(m.read_text()); data=json.loads(d.read_text());out=[]
for row in model['files']:
 p=m.parent/row['file'];assert p.is_file() and not p.is_symlink();assert p.stat().st_size==row['bytes'];actual=sha(p);assert actual==row['sha256'];out.append(dict(path=str(p),bytes=p.stat().st_size,sha256=actual))
for row in list(data['splits'].values())+[data['exclusion_audit']]:
 p=d.parent/row['path'];assert p.is_file() and not p.is_symlink();actual=sha(p);assert actual==row['sha256'];out.append(dict(path=str(p),bytes=p.stat().st_size,sha256=actual))
result=dict(state='ACTUAL_ALL_INPUT_BYTES_MATCH',node=socket.gethostname(),root=str(r),model_revision=model['revision'],model_manifest_sha256=sha(m),data_manifest_sha256=sha(d),files=out)
with (r/'asset-copy-verified.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result),flush=True)
'''
subprocess.run(destination('python3 -c '+shlex.quote(verify)),check=True)
