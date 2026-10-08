import hashlib,json,pathlib,shlex,subprocess,time
from remote import route
base=pathlib.Path('/Users/shaoshitong/Documents/remote_codex_control/.transfer-staging/tree-grpo-deps-20261002')
manifest=json.loads((base/'manifest.json').read_text())
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
selected=[a for a in manifest if (base/a['filename']).is_file() and digest(base/a['filename'])==a['sha256']]
assert selected
root='/dataset/shaoshitong/search-r1-rl/tree-grpo'
incoming=root+'/wheelhouse-incoming-'+str(time.time_ns())
verify='''import hashlib,json,pathlib
root=pathlib.Path(%r); selected=json.loads(%r)
for a in selected:
 p=root/a['filename'];h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 assert p.stat().st_size==a['size'] and h.hexdigest()==a['sha256'],a['filename']
 p.replace(pathlib.Path(%r)/a['filename'])
 print('WHEEL_SHA_VERIFIED',a['filename'],flush=True)
(root/'transfer.json').write_text(json.dumps(selected,indent=2))
print('WHEELHOUSE_BATCH_SHA_OK',len(selected))
'''%(incoming,json.dumps(selected),root+'/wheelhouse')
command='set -e; mkdir -p '+shlex.quote(incoming)+' '+shlex.quote(root+'/wheelhouse')+'; tar -xf - -C '+shlex.quote(incoming)+'; /usr/bin/python3.10 -c '+shlex.quote(verify)
print('TRANSFER_START',len(selected),sum(a['size'] for a in selected),flush=True)
a=subprocess.Popen(['tar','-cf','-','-C',str(base)]+[a['filename'] for a in selected],stdout=subprocess.PIPE)
b=subprocess.Popen(route(command),stdin=a.stdout);a.stdout.close();status=b.wait();tarstatus=a.wait();assert status==tarstatus==0,(status,tarstatus)
(base/'transfer-batch-'+str(time.time_ns())+'.json').write_text(json.dumps(selected,indent=2))
