import base64,json
from pathlib import Path
r=Path('work/numericfix/evidence-tools')
names=[p for p in r.iterdir() if p.is_file() and p.suffix in ['.py','.json','.md']]
records={p.name:base64.b64encode(p.read_bytes()).decode() for p in names}
s='''import pathlib,base64,subprocess,os,json,hashlib
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-rewardfix-numeric');dest=root/'tools/rewardfix-evidence';dest.mkdir(exist_ok=True)
assert not dest.is_symlink()
for name,data in RECORDS.items():
 p=dest/name;b=base64.b64decode(data)
 if p.exists():assert p.read_bytes()==b
 else:p.write_bytes(b)
record=json.loads((dest/'tool-hashes.json').read_text())
print('ACTUAL_EVIDENCE_TOOLS_DEPLOYED_FIXED')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
with (root/'logs/rewardfix-evidence-tools-native-tests.log').open('w') as f:
 subprocess.run([str(root/'env/bin/python'),'-m','unittest','discover','-s',str(dest),'-p','test*.py','-v'],check=True,env=env,stdout=f,stderr=subprocess.STDOUT)
print((root/'logs/rewardfix-evidence-tools-native-tests.log').read_text()[-250:])
'''.replace('RECORDS',repr(records))
Path('work/numericfix/deploy-evidence-tools.py').write_text(s)
