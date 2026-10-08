import subprocess,shlex,json
from pathlib import Path
code="""import json,hashlib
from pathlib import Path
r=Path('/mnt/data/shaoshitong/search-r1-rl/opsd');p=r/'runs/opsd-new-acceptance-2994';h=r/'runs/opsd-new-hf-reload-2996'
paths={'pilot_raw_log':p/'training.log','pilot_commit':p/'evidence/commit.txt','pilot_slurm':p/'evidence/controller-normal-exit-proof.json','pilot_exit':p/'exit.txt','pilot_finished':p/'training-finished.txt','reload_result':h/'checkpoint-reload.json','reload_commit':h/'commit.txt','reload_slurm':h/'slurm-final.txt','reload_exit':h/'exit.txt','reload_cleanup':h/'previous-job-cleanup.json'}
summary=json.loads((p/'evidence/micro-capacity.json').read_text());c=[x for x in summary['candidates'] if x['local_micro']==2][0]
aux={'capacity_summary':p/'evidence/micro-capacity.json','capacity_candidate':Path(c['aggregate_json']),'pilot_config':p/'evidence/reviewed-entry-config.yaml'}
def item(f):return {'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
print(json.dumps({'roles':{k:item(v) for k,v in paths.items()},'aux':{k:item(v) for k,v in aux.items()}}))
"""
r=subprocess.run(['python3','work/remote.py','python3 -c '+shlex.quote(code)],capture_output=True,text=True)
if r.returncode:print(r.stderr);raise SystemExit(r.returncode)
Path('work/job2994-2996-ten-roles-and-aux-actual.json').write_text(r.stdout);print(r.stdout)
