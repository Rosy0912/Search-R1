"""Task-authorized CPU-only native inventory transfer/execution; never GPU init."""
import hashlib,io,json,pathlib,shlex,subprocess,tarfile
HERE=pathlib.Path(__file__).resolve().parent
ROOT='/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004'
COMMIT='04f8cc5a777f8dfafbb9e48f877673c2c5751f33'
REPO=ROOT+'/repo-'+COMMIT[:12]
CHECKER='04304d563a71b3123e0da6c5350398bf0b8510e0b477ce47742798214bd5b3f0'
PRODUCER=HERE/'native_import_closure_producer.py';PIN=hashlib.sha256(PRODUCER.read_bytes()).hexdigest()
DEST=ROOT+'/tools/native-closure-'+PIN[:12]
OUTPUT=ROOT+'/native-source-inventory-'+COMMIT[:12]+'.json'
opts=['-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=3','-o','ControlMaster=no','-o','ControlPath=none']
def route(command):
 worker='env -u LD_LIBRARY_PATH ssh '+shlex.join(opts+['h100-10',command])
 head='env -u LD_LIBRARY_PATH ssh '+shlex.join(opts+['-o','HostName=10.33.0.16','-o','HostKeyAlias=100.108.173.4','-o','StrictHostKeyChecking=yes','h100-16',worker])
 return ['ssh']+opts+['-o','ProxyCommand=/usr/bin/nc -X 5 -x 127.0.0.1:7892 %h %p','mgmt-host',head]
def remote(code,input=None):
 return subprocess.run(route('python3 -c '+shlex.quote(code)),input=input,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
pre=f'''import pathlib,subprocess,socket,os,hashlib,json
repo=pathlib.Path({REPO!r});checker=repo/'reproduction/baselines/e1_import_bootstrap.py'
assert socket.gethostname()=='h100-10' and os.getuid()==0
assert repo.resolve()==repo
assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()=={COMMIT!r}
assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()
assert hashlib.sha256(checker.read_bytes()).hexdigest()=={CHECKER!r}
p=pathlib.Path({DEST!r});p.mkdir(exist_ok=False)
assert p.stat().st_uid==os.getuid()
print(json.dumps(dict(host=socket.gethostname(),uid=os.getuid(),source={COMMIT!r},checker_sha256={CHECKER!r},producer_sha256={PIN!r},tool_dir=str(p))))
'''
r=remote(pre);(HERE/'native-closure-native-precheck.json').write_bytes(r.stdout)
buffer=io.BytesIO()
with tarfile.open(fileobj=buffer,mode='w') as tf:tf.add(PRODUCER,arcname=PRODUCER.name)
cmd='tar --no-same-owner -xf - -C '+shlex.quote(DEST)
r=subprocess.run(route(cmd),input=buffer.getvalue(),capture_output=True,check=True)
run=f'''import pathlib,subprocess,os,json,hashlib,socket
repo=pathlib.Path({REPO!r});p=pathlib.Path({DEST!r})/'native_import_closure_producer.py'
assert p.stat().st_uid==os.getuid()==0 and hashlib.sha256(p.read_bytes()).hexdigest()=={PIN!r}
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',PYTHONPATH=str(repo),HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1')
result=subprocess.run(['/mnt/data/shaoshitong/search-r1-rl/opsd/env/bin/python',str(p),'--repo',str(repo),'--checker-sha256',{CHECKER!r},'--output',{OUTPUT!r}],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
print(result.stdout.decode(),end='')
print(result.stderr.decode(),file=__import__('sys').stderr,end='')
assert result.returncode==0
assert hashlib.sha256(p.read_bytes()).hexdigest()=={PIN!r}
assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()=={COMMIT!r}
assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()
assert hashlib.sha256((repo/'reproduction/baselines/e1_import_bootstrap.py').read_bytes()).hexdigest()=={CHECKER!r}
raw=pathlib.Path({OUTPUT!r}).read_bytes();value=json.loads(raw);assert len(value['modules'])==210
assert all(pathlib.Path(v['path']).is_file() and hashlib.sha256(pathlib.Path(v['path']).read_bytes()).hexdigest()==v['sha256'] for v in value['modules'].values())
print(json.dumps(dict(scope='actual h100-10 CPU inventory only; no GPU acceptance',host=socket.gethostname(),uid=os.getuid(),source={COMMIT!r},producer_sha256={PIN!r},checker_sha256={CHECKER!r},output={OUTPUT!r},output_sha256=hashlib.sha256(raw).hexdigest(),records=len(value['modules']),libraries={{n:value['modules'][n] for n in ['hydra.main','torch.utils._contextlib','contextlib','dataclasses']}})))
'''
r=remote(run);(HERE/'native-closure-native-producer-stdout.txt').write_bytes(r.stdout);(HERE/'native-closure-native-producer-stderr.txt').write_bytes(r.stderr)
r=remote(f'import pathlib,sys;sys.stdout.buffer.write(pathlib.Path({OUTPUT!r}).read_bytes())');(HERE/'native-source-inventory-04f8cc5a777f-ACTUAL.json').write_bytes(r.stdout)
assert hashlib.sha256(PRODUCER.read_bytes()).hexdigest()==PIN
print((HERE/'native-closure-native-producer-stdout.txt').read_text())
