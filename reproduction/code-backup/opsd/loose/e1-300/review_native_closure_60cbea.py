import hashlib,json,pathlib,shlex,subprocess
B=pathlib.Path(__file__).resolve().parent
new=B/'native-source-inventory-60cbea97d61f-ACTUAL.json';old=B/'native-source-inventory-04f8cc5a777f-ACTUAL.json'
assert hashlib.sha256(new.read_bytes()).hexdigest()=='020369593d82cbf77997787366daf4cb55cf1e4f694ac449021454103d8288c4'
assert hashlib.sha256(old.read_bytes()).hexdigest()=='e863d128385442af9a11fa63e7f4b1d9ea558ad0ae7fd358c7cf78badd387d59'
a=json.loads(old.read_text())['modules'];z=json.loads(new.read_text())['modules'];assert set(a)==set(z) and len(z)==210
libraries=('hydra.main','torch.utils._contextlib','contextlib','dataclasses')
changed=[]
for n,v in z.items():
 if n in libraries: assert v==a[n],n
 else:
  assert v['path']==a[n]['path'].replace('/repo-04f8cc5a777f/','/repo-60cbea97d61f/'),n
  if v['sha256']!=a[n]['sha256']:changed.append(n)
assert changed==['reproduction.baselines.e1_import_bootstrap'],changed
assert z[changed[0]]['sha256']=='82433ba203fa8b8ed02cc3c4d0be4d0d0ceb7de096ebe657561f1fd726194164'
print('Independent local actual-inventory comparison PASS: 210 keys, exact path migration, checker-only hash change; four native libraries identical')
code=r'''import sys,os,pathlib,hashlib,json,subprocess,socket
sys.dont_write_bytecode=True
repo=pathlib.Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004/repo-60cbea97d61f')
assert socket.gethostname().split('.')[0]=='h100-10' and os.environ['CUDA_VISIBLE_DEVICES']==''
assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()=='60cbea97d61ff95a8f7dfa58de7b79962e79bd10'
assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()
p=repo/'reproduction/baselines/e1_import_bootstrap.py'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='82433ba203fa8b8ed02cc3c4d0be4d0d0ceb7de096ebe657561f1fd726194164'
raw=pathlib.Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004/native-source-inventory-60cbea97d61f.json').read_bytes()
assert hashlib.sha256(raw).hexdigest()=='020369593d82cbf77997787366daf4cb55cf1e4f694ac449021454103d8288c4'
sys.path.insert(0,str(repo))
from reproduction.baselines import e1_import_bootstrap as checker
assert pathlib.Path(checker.__file__).resolve()==p
import torch
assert not torch.cuda.is_initialized()
actual,count=checker.verify_imports(repo,checker.unique_json(raw))
assert not torch.cuda.is_initialized()
assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()
print(json.dumps(dict(status='INDEPENDENT_NATIVE_CPU_IMPORT_CHECK_PASS',hostname=socket.gethostname(),uid=os.getuid(),pid=os.getpid(),cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],cuda_initialized=False,source_commit='60cbea97d61ff95a8f7dfa58de7b79962e79bd10',repo=str(repo),inventory_sha256=hashlib.sha256(raw).hexdigest(),inventory_records=210,actual_modules=len(actual['modules']),actual_functions=count,actual=actual),sort_keys=True))
'''
command='env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct300-20261004/repo-60cbea97d61f /mnt/data/shaoshitong/search-r1-rl/opsd/env/bin/python -c '+shlex.quote(code)
r=subprocess.run(['python3',str(B.parent/'remote.py'),command],capture_output=True,text=True)
(B/'native-closure-60cbea-independent-native-cpu-original.txt').write_text(r.stdout+r.stderr+'\nactual_exit='+str(r.returncode)+'\n');print(r.stdout,r.stderr)
assert r.returncode==0
obj=json.loads(r.stdout.splitlines()[-1]);(B/'native-closure-60cbea-independent-native-cpu.json').write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
