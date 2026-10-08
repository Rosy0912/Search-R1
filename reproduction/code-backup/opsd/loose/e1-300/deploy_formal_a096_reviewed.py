import io,json,hashlib,pathlib,subprocess,shlex,tarfile,datetime
H=pathlib.Path(__file__).resolve().parent
R='/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004'
D=R+'/tools/formal-7dfaeb6ef51c'
M='7dfaeb6ef51cca4d928b028c52bedd0d64bde91dfc4a9b86c360183c13b07601'
opts=['-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=3','-o','ControlMaster=no','-o','ControlPath=none']
def route(c):
 w='env -u LD_LIBRARY_PATH ssh '+shlex.join(opts+['h100-10',c]);h='env -u LD_LIBRARY_PATH ssh '+shlex.join(opts+['-o','HostName=10.33.0.16','-o','HostKeyAlias=100.108.173.4','-o','StrictHostKeyChecking=yes','h100-16',w])
 return ['ssh']+opts+['-o','ProxyCommand=/usr/bin/nc -X 5 -x 127.0.0.1:7892 %h %p','mgmt-host',h]
def remote(c,data=None):return subprocess.run(route(c),input=data,capture_output=True,check=True,timeout=60)
m=json.loads((H/'formal-tools-a0968c2-strict-marker-candidate/manifest.json').read_bytes());assert hashlib.sha256((H/'formal-tools-a0968c2-strict-marker-candidate/manifest.json').read_bytes()).hexdigest()==M
for n,s in m['tools'].items():assert hashlib.sha256((H/'formal-tools-a0968c2-strict-marker-candidate'/n).read_bytes()).hexdigest()==s
key=H/'review-signing/public.json';assert hashlib.sha256(key.read_bytes()).hexdigest()=='ff53cfde6dd4bf96142b23364ae72da1d9610e1a834663ba70e750f604cee41c'
r=remote('mkdir '+shlex.quote(D)+' && mkdir -p '+shlex.quote(R+'/review-signing'))
b=io.BytesIO()
with tarfile.open(fileobj=b,mode='w') as t:
 for n in m['tools']:t.add(H/'formal-tools-a0968c2-strict-marker-candidate'/n,arcname=n)
r=remote('tar --no-same-owner -xf - -C '+shlex.quote(D),b.getvalue())
# Existing reviewed public key remains unchanged; native postcheck pins exact bytes.
r=remote('test ! -e '+shlex.quote(D+'/manifest.json')+' && cat > '+shlex.quote(D+'/manifest.json'),(H/'formal-tools-a0968c2-strict-marker-candidate/manifest.json').read_bytes())
code=f'''import pathlib,hashlib,json,subprocess,os,socket
r=pathlib.Path({R!r});d=pathlib.Path({D!r});m=d/'manifest.json'
assert hashlib.sha256(m.read_bytes()).hexdigest()=={M!r}
v=json.loads(m.read_bytes());repo=r/'repo-a0968c2d24be'
assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==v['source_commit']
assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()
for n,s in v['tools'].items():
 p=d/n;assert p.stat().st_uid==os.getuid()==0 and hashlib.sha256(p.read_bytes()).hexdigest()==s
for n,s in v['source_files'].items():assert hashlib.sha256((repo/n).read_bytes()).hexdigest()==s
k=r/'review-signing/public.json';assert k.stat().st_uid==0 and hashlib.sha256(k.read_bytes()).hexdigest()=='ff53cfde6dd4bf96142b23364ae72da1d9610e1a834663ba70e750f604cee41c'
print(json.dumps({{'node':socket.gethostname(),'uid':os.getuid(),'source':v['source_commit'],'tools':str(d),'manifest_sha256':{M!r},'public_key_only':True,'runtime_tools':len(v['tools']),'source_files':len(v['source_files'])}},sort_keys=True))
'''
r=remote('python3 -c '+shlex.quote(code));(H/'formal-a096-deployment-actual.json').write_bytes(r.stdout);print(r.stdout.decode())
