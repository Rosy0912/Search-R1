set -euo pipefail
python3 - <<'PY'
from pathlib import Path
import json,hashlib,subprocess,shutil,time
r=Path('/dataset/shaoshitong/search-r1-rl/cosearch/author-native');s=r/'continue-300-tools-q';src=r/'runs/author-native-checkpoint-fixed-five-p/tools';run='author-native-continue-300-q'
assert not subprocess.check_output(['squeue','-h','-w','h100-16','-o','%i'],text=True).strip()
gate=json.loads((s/'cpu-continue-gate.json').read_text());assert gate['passed'] and gate['single_original_load'] and gate['data_state_equal'] and not gate['cuda_initialized']
ready=json.loads((src/'preparation-ready.json').read_text())
changed={'launch_observed.py','native_job.py'}
for n,h in ready['tools_sha256'].items():
 if n not in changed:assert hashlib.sha256((s/n).read_bytes()).hexdigest()==h,n
for n in changed|{'continuation.py','cpu_continue_gate.py','cpu-continue-gate.json','start_continue.py','continuation-authorization.json'}:ready['tools_sha256'][n]=hashlib.sha256((s/n).read_bytes()).hexdigest()
ready.update(run=run,created_at=time.time(),initialization='Resume verified P global5 model+RLAdam+scheduler+data; total300; natural324; val64/save50; retention2; explicit human long-training override')
(s/'preparation-ready.json').write_text(json.dumps(ready,indent=2)+'\n')
print(json.dumps(dict(run=run,ready_sha256=hashlib.sha256((s/'preparation-ready.json').read_bytes()).hexdigest(),cpu_gate=gate,tools_sha256=ready['tools_sha256'],disk_free=shutil.disk_usage(r).free)),flush=True)
PY
