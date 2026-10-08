"""Transfer immutable H100 inputs entirely inside the cluster via management SSH."""
import json,pathlib,shlex,subprocess,importlib.util
HERE=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('prior',HERE/'copy-shared-assets.py')
# Reuse only the route's code, not import the prior executable transfer script.
ROOT='/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004'
SOURCE='/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300'
data=json.loads((HERE/'common-manifest-original.json').read_text());model=json.loads((HERE/'official-model-manifest-original.json').read_text())
files=['common-data/manifest.json','common-data/exclusions-audit.json']+['common-data/'+v['path'] for v in data['splits'].values()]+['common-data/'+v['path'] for v in data['preparation']['raw_sources']]
files+=['assets/qwen2.5-3b-instruct-aa8e725/VERIFIED-MODEL-MANIFEST.json']+['assets/qwen2.5-3b-instruct-aa8e725/'+v['file'] for v in model['files']]
peer='10.33.0.13'
ssh=['env','-u','LD_LIBRARY_PATH','ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2','-o','ControlMaster=no','-o','ControlPath=none']
sourceargs=ssh+['-o','HostName='+peer,'-o','HostKeyAlias=h100-9','-o','StrictHostKeyChecking=yes','root@h100-9',shlex.join(['tar','cf','-','-C',SOURCE]+files)]
extract=f'''import pathlib,subprocess,json,os
r=pathlib.Path({ROOT!r}); stage=r/'transfer-direct-lan-v2';stage.mkdir()
with (r/'direct-lan-v2-intent.json').open('x') as f:json.dump(dict(source_peer={peer!r},target='h100-10',files={files!r}),f)
subprocess.run(['tar','xf','-','-C',str(stage)],check=True)
print('ALL_V2_TAR_BYTES_RECEIVED',flush=True)
'''
worker=shlex.join(ssh+['h100-10','python3 -c '+shlex.quote(extract)])
targetargs=ssh+['-o','HostName=10.33.0.16','-o','HostKeyAlias=100.108.173.4','-o','StrictHostKeyChecking=yes','h100-16',worker]

native=f'''import pathlib,subprocess,json,socket
r=pathlib.Path({ROOT!r});stage=r/'transfer-native-lan-v3';stage.mkdir()
assert socket.gethostname()=='h100-10'
with (r/'native-lan-v3-intent.json').open('x') as f:json.dump(dict(source_peer={peer!r},target='h100-10',files={files!r},route='native LAN direct; ephemeral forwarded existing identity'),f)
args={ssh!r}+['-o','UserKnownHostsFile='+str(r/'peer-known-hosts'),'-o','StrictHostKeyChecking=yes','root@'+{peer!r},{shlex.join(['tar','cf','-','-C',SOURCE]+files)!r}]
source=subprocess.Popen(args,stdout=subprocess.PIPE)
target=subprocess.Popen(['tar','xf','-','-C',str(stage)],stdin=source.stdout);source.stdout.close()
t=target.wait();s=source.wait();print(json.dumps(dict(source_exit=s,target_exit=t,files={len(files)})),flush=True)
raise SystemExit(0 if s==t==0 else 1)
'''
opts=ssh+['-A']
worker=shlex.join(opts+['h100-10','python3 -c '+shlex.quote(native)])
route=opts+['-o','HostName=10.33.0.16','-o','HostKeyAlias=100.108.173.4','-o','StrictHostKeyChecking=yes','h100-16',worker]
program='import subprocess; subprocess.run('+repr(['ssh-add','/home/lichen/.ssh/id_ed25519'])+',check=True); subprocess.run('+repr(route)+',check=True)'
command='ssh-agent -a /home/lichen/.ssh/opsd-e1-copy-v3-20261004.sock python3 -c '+shlex.quote(program)
subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2','-o','ProxyCommand=/usr/bin/nc -X 5 -x 127.0.0.1:7892 %h %p','mgmt-host',command],check=True)
verify=f'''import pathlib,json,hashlib,os,socket
root=pathlib.Path({ROOT!r});r=root/'transfer-native-lan-v3'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
m=r/'assets/qwen2.5-3b-instruct-aa8e725/VERIFIED-MODEL-MANIFEST.json';d=r/'common-data/manifest.json'
assert sha(m)=='7f53ce6c010ac076b1ec482031bea7c9e6b8d0a096a61b880fe8ad4971bc9e44'
assert sha(d)=='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
model=json.loads(m.read_text());data=json.loads(d.read_text());out=[]
for row in model['files']:
 p=m.parent/row['file'];assert p.is_file() and not p.is_symlink() and p.stat().st_size==row['bytes'];v=sha(p);assert v==row['sha256'];out.append(dict(path=p.relative_to(r).as_posix(),bytes=p.stat().st_size,sha256=v))
for row in list(data['splits'].values())+[data['exclusion_audit']]+data['preparation']['raw_sources']:
 p=d.parent/row['path'];assert p.is_file() and not p.is_symlink();v=sha(p);assert v==row['sha256'];out.append(dict(path=p.relative_to(r).as_posix(),bytes=p.stat().st_size,sha256=v))
assert socket.gethostname()=='h100-10'
failed=root/'preserved-slow-transfer-v1-v3';failed.mkdir()
for name in ('assets','common-data'):
 if (root/name).exists():os.rename(root/name,failed/name)
 os.rename(r/name,root/name)
result=dict(state='ACTUAL_ALL_INPUT_BYTES_MATCH',node=socket.gethostname(),root=str(root),official_file_count=len(model['files']),raw_source_file_count=len(data['preparation']['raw_sources']),model_revision=model['revision'],model_manifest_sha256=sha(root/'assets/qwen2.5-3b-instruct-aa8e725/VERIFIED-MODEL-MANIFEST.json'),data_manifest_sha256=sha(root/'common-data/manifest.json'),files=out)
with (root/'asset-copy-verified.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result),flush=True)
'''
subprocess.run(['python3',str(HERE.parent/'remote.py'),'python3 -c '+shlex.quote(verify)],check=True)
