from pathlib import Path
import json,os
assert os.uname().nodename=='h100-11'
r={};b=Path('/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-baseline-aligned-em200-20261009/env')
for rel in ['pyvenv.cfg','lib','lib/python3.10','lib/python3.10/site-packages','bin/python']:
 p=b/rel;r[rel]={'link':os.readlink(p) if p.is_symlink() else None,'exists':p.exists(),'resolved':str(p.resolve())}
 if p.is_file() and rel=='pyvenv.cfg':r[rel]['text']=p.read_text()
 if p.is_dir():r[rel]['children']=[x.name for x in p.iterdir()][:30]
print(json.dumps(r))
