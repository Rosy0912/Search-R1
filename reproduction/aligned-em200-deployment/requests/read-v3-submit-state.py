import json,os,subprocess
from pathlib import Path
r=Path('/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v3-20261009');old=Path(str(r).replace('v3','v2'));run=next((old/'runs').glob('*3352'));out={}
for n in ('formal-submit-intent.json','formal-submit-receipt.json'):
 p=r/'scheduler-head'/n;out[n]=p.read_text()if p.exists()else None
out['old_run_files']=[(str(p.relative_to(run)),p.stat().st_size)for p in run.rglob('*')if p.is_file()];a=old/'evidence/formal3352-failure-full.tar';out['archive_size']=a.stat().st_size if a.exists()else None
p=subprocess.run(['squeue','-h','-u','root','-o','%i|%T|%N|%k'],capture_output=True,text=True);out['queue']=p.stdout;print(json.dumps(out))
