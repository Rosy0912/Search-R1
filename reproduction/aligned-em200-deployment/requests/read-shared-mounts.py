from pathlib import Path
import json,os,subprocess
v={'host':os.uname().nodename}
for s in ['/mnt/shared','/mnt/shared/workspace','/mnt/shared/workspace/shaoshitong','/mnt/shared/workspace/shaoshitong/search-r1-rl','/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-baseline-aligned-em200-20261009']:
 p=Path(s);r=subprocess.run(['findmnt','-T',s,'-J'],capture_output=True,text=True,timeout=10)
 v[s]={'exists':p.exists(),'is_symlink':p.is_symlink(),'realpath':str(p.resolve()),'mount':r.stdout,'mount_exit':r.returncode}
print(json.dumps(v))
