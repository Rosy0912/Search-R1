"""Reserve a separate allocation after the old parent completed; no other job mutation."""
import json,subprocess,time
from pathlib import Path
b=Path('/dataset/shaoshitong/recursive-bran-a-h100-20261012')
r=b/'recovery-from100-20261013';r.mkdir(exist_ok=True);(r/'logs').mkdir(exist_ok=True)
receipt=r/'allocation-submitted.json'
if receipt.exists():
 print(receipt.read_text());raise SystemExit(0)
old=subprocess.run(['scontrol','show','job','3389'],capture_output=True,text=True)
assert (old.returncode==0 and 'JobState=COMPLETED' in old.stdout) or (old.returncode!=0 and 'Invalid job id specified' in old.stderr)
assert all(not Path('/proc/'+str(pid)).exists() for pid in [3714909,3721269,3715225,3713677,3713706])
cp=b/'checkpoints/training_state/global_step_100/manifest.json'
m=json.loads(cp.read_text());assert m['identity']['completed_step']==100
assert all(Path(w['path']).is_file() and Path(w['path']).stat().st_size==w['bytes'] for w in m['worker_states'])
script='#!/bin/bash\nset -euo pipefail\nr=/dataset/shaoshitong/recursive-bran-a-h100-20261012/recovery-from100-20261013\nprintf \'%s\\n\' "$SLURM_JOB_ID" > "$r/allocation-job.txt"\nwhile [[ ! -f "$r/launch-ready.sh" ]]; do sleep 3; done\nmv "$r/launch-ready.sh" "$r/launch-executing-$SLURM_JOB_ID.sh"\nbash "$r/launch-executing-$SLURM_JOB_ID.sh"\n'
p=subprocess.run(['ssh','-oBatchMode=yes','-oConnectTimeout=8','h100-5','sbatch --parsable --job-name=rba-resume100-200 --nodelist=h100-11 --gres=gpu:h100:8 --exclusive --cpus-per-task=96 --mem=768G --time=7-00:00:00 --output='+str(r/'logs/slurm-%j.log')],input=script,text=True,capture_output=True,timeout=30)
d=dict(time=time.time(),returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,parent_completed=100,reason='old shared allocation ended; no saved118 state',old_parent_job=3389)
if p.returncode==0:receipt.write_text(json.dumps(d,indent=2))
print(json.dumps(d),flush=True);raise SystemExit(p.returncode)
