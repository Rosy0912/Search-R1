import json, subprocess, time
from pathlib import Path
root=Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-envfix-20261006')
record=root/'scheduler-head/formal3292-superseded-by-fresh200-cancel-record.json'
assert not record.exists(), 'Do not repeat a previously recorded cancellation'
before=subprocess.check_output(['scontrol','show','job','3292'],text=True)
for expected in ('JobId=3292','JobName=bapo-instruct300-formal300','NodeList=h100-5',str(root/'scheduler-head/formal300.sbatch')):
 assert expected in before, expected
assert 'JobState=RUNNING' in before, 'Only cancel the verified live superseded job'
result=subprocess.run(['scancel','3292'],capture_output=True,text=True)
with record.open('x') as f:
 json.dump(dict(job_id=3292,basis='Latest direct human instruction replaces old300 with independently initialized fresh200; preserve old outputs; no inherited state',observed_unix=time.time(),before=before,cancel_returncode=result.returncode,stdout=result.stdout,stderr=result.stderr),f,indent=2)
assert result.returncode==0, result.stderr
print(record.read_text())
