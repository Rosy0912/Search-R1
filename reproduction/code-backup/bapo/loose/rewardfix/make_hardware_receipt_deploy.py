from pathlib import Path
import base64
names={'bapo-rewardfix-hardware3064-independent-review.md':'outputs/bapo-rewardfix-hardware3064-independent-review.md','hardware-acceptance.json':'outputs/bapo-rewardfix-hardware3064-acceptance.json','bapo-rewardfix-hardware3064-acceptance.json':'outputs/bapo-rewardfix-hardware3064-acceptance.json'}
records={n:base64.b64encode(Path(p).read_bytes()).decode() for n,p in names.items()}
s='''import pathlib,base64,subprocess,os
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-rewardfix-b097')
for name,data in RECORDS.items():
 p=root/'logs'/name;b=base64.b64decode(data)
 assert not p.is_symlink()
 if p.exists():assert p.read_bytes()==b,'Receipt overwrite forbidden'
 else:p.write_bytes(b)
subprocess.run([str(root/'env/bin/python'),str(root/'tools/verify_hardware_receipt.py')],check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',PYTHONDONTWRITEBYTECODE='1'))
'''.replace('RECORDS',repr(records))
Path('work/rewardfix/deploy-hardware-receipt.py').write_text(s)
