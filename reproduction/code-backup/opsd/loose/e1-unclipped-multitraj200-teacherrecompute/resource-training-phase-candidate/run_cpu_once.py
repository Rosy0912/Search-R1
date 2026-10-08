import datetime,hashlib,json,os,subprocess
from pathlib import Path
h=Path(__file__).resolve().parent;o=h/'cpu-originals';o.mkdir(exist_ok=True);stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
deps=[h/'snapshot_training_resources_once.py',h/'test_snapshot_training_resources_once.py',h.parent/'first5-1b16c2227caf/receipt_payload.py',h.parent/'formal-71d2501d4512/formal_supervisor.py',h.parent.parent/'opsd-unclipped-multitraj200-teacherrecompute/search_r1/baselines/checkpoint_gate.py']
def pins():return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in deps}
before=pins();cmd=['/Users/shaoshitong/Documents/Search-R1-RL-Baselines/.venv-cpu/bin/python','-B','-m','pytest','-q',str(h/'test_snapshot_training_resources_once.py')]
with (o/(stamp+'-stdout.original')).open('xb') as out,(o/(stamp+'-stderr.original')).open('xb') as err:
 r=subprocess.run(cmd,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
after=pins();status=dict(scope='LOCAL_CPU_SYNTHETIC_BOUNDARIES_AND_EXPLICIT_ACTUAL_RAW_READONLY_JOINS_NOT_PRODUCTION_SNAPSHOT',argv=cmd,exit_code=r.returncode,before=before,after=after,stable=before==after,stdout=str(o/(stamp+'-stdout.original')),stderr=str(o/(stamp+'-stderr.original')))
p=o/(stamp+'-status.json');p.write_text(json.dumps(status,sort_keys=True,indent=2)+'\n');print(json.dumps(dict(status=str(p),exit_code=r.returncode,stable=before==after)));print((o/(stamp+'-stdout.original')).read_text()[-6000:])
