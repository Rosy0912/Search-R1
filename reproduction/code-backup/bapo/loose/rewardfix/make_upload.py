import json,hashlib,base64
from pathlib import Path
r=Path('work/rewardfix/remote')
files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in r.iterdir() if p.is_file() and p.name!='hardware-launch-hashes-b09736d.json'}
(r/'hardware-launch-hashes-b09736d.json').write_text(json.dumps(files,indent=2,sort_keys=True)+'\n')
records={p.name:base64.b64encode(p.read_bytes()).decode() for p in r.iterdir() if p.is_file()}
s='''import pathlib,base64,json,subprocess,os
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-rewardfix-b097')
records=RECORDS
for name,data in records.items():
 p=root/'tools'/name
 assert not p.is_symlink()
 p.write_bytes(base64.b64decode(data))
subprocess.run([str(root/'env/bin/python'),str(root/'tools/verify_shared_inputs.py'),'--record'],check=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
print('NEW_OWNED_TOOLS_AND_SHARED_RECORD_INSTALLED')
'''.replace('RECORDS',repr(records))
Path('work/rewardfix/upload-tools.py').write_text(s)
