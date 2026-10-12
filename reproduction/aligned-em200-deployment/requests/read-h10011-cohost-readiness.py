"""Read local mounts, BAPO asset visibility and available memory; no mutation."""
import datetime,json,os,subprocess
from pathlib import Path
assert os.uname().nodename=='h100-11'
root=Path('/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-baseline-aligned-em200-20261009')
r={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':os.uname().nodename,'root':str(root),'paths':{}}
for name in ['repo','env/bin/python','assets-root','assets-root/assets/wiki/e5_Flat.index','assets-root/assets/wiki/wiki-18.jsonl','assets-root/assets/e5-base-v2']:
 p=root/name
 r['paths'][name]={'exists':p.exists(),'symlink':p.is_symlink(),'resolved':str(p.resolve())}
 if p.exists():r['paths'][name]['bytes']=p.stat().st_size
for k,args in [('mounts',['findmnt','-rn','-o','TARGET,SOURCE,FSTYPE']),('disks',['df','-h','/dataset','/mnt/shared','/dev/shm']),('memory',['free','-m']),('ports',['ss','-ltn']),('local_assets',['ls','-ld','/dataset/shaoshitong','/dataset/shaoshitong/search-r1-rl'])]:
 p=subprocess.run(args,capture_output=True,text=True,timeout=15);r[k]={'exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
print(json.dumps(r))
