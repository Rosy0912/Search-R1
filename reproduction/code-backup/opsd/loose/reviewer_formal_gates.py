import subprocess,shlex,json,sys,importlib.util
from pathlib import Path
code="""from pathlib import Path
import json
p=Path('/mnt/data/shaoshitong/search-r1-rl/opsd/runs/opsd-formal-2999');e=p/'evidence';out={}
for n in ['gpu-teacher-semantic-diagnostic.json','semantic-device-gate.json','semantic-device-diagnostic.log','gpu-loss-checkpoint-equivalence.json','micro-capacity.json','micro-2.json','reviewed-entry-config.yaml','expected-final-counts.json']:
 out[n]=(e/n).read_text()
out['training.log']=(p/'training.log').read_text()
print(json.dumps(out))
"""
r=subprocess.run(['python3','work/remote.py','python3 -c '+shlex.quote(code)],capture_output=True,text=True,check=True);p=Path('work/opsd/reproduction/baselines/opsd-validation-20261003/evidence');(p/'job2999_numeric_capacity_yaml_reviewer_raw.json').write_text(r.stdout);x=json.loads(r.stdout)
sys.path[:0]=['work/formal-preparation/formal-2994-2996-no-purged-dependency-bound-preview'];from micro_evidence import validate_capacity_record
c=json.loads(x['micro-capacity.json']);m=json.loads(x['micro-2.json']);validate_capacity_record(c,m,'2999','8e60976d364ab7d7ae51714415813c1359dafe9a',2)
print('CURRENT_CAPACITY_STRICT_ALL8_PASS')
for n in ['semantic-device-gate.json','gpu-loss-checkpoint-equivalence.json']:
 y=json.loads(x[n]);print(n,list(y),str(y)[:1300])
print('yaml desired keys', '\n'.join(l for l in x['reviewed-entry-config.yaml'].splitlines() if any(k in l for k in ['total_epochs:','total_training_steps:','val_batch_size:','test_freq:','save_freq:','smoke_only:'])))
print('log relevant', '\n'.join(l for l in x['training.log'].splitlines() if any(k in l for k in ['filtered training dataset size','Size of train dataloader','Total training steps:','BASELINE_EVALUATION_METRICS_JSON'])))
