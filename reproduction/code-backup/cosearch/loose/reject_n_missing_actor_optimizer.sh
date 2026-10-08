set -euo pipefail
python3 - <<'REJECT'
import json,hashlib,subprocess
from pathlib import Path
root=Path('/dataset/shaoshitong/search-r1-rl/cosearch');run=root/'runs/cosearch-local-gpu-five-20261003-n'
assert not subprocess.check_output(['git','-C',str(root/'repo'),'status','--porcelain'],text=True).strip()
source_commit=subprocess.check_output(['git','-C',str(root/'repo'),'rev-parse','HEAD'],text=True).strip();assert source_commit=='e2e401c10ea0a20131eb9b0ed39f4c0464df7bc3'
assert json.loads((run/'driver-exit.json').read_text())['returncode']==0
import sys;sys.path.insert(0,str(root/'tools'))
from metric_log_source import validated_metric_log
text,metric_source=validated_metric_log(run,root);records=[]
for line in text.splitlines():
 if 'COSEARCH_OUTER_METRICS_JSON=' in line:records.append(json.JSONDecoder().raw_decode(line.split('COSEARCH_OUTER_METRICS_JSON=',1)[1])[0])
import math
assert [x['step'] for x in records]==[1,2,3,4,5]
assert all(math.isfinite(float(v)) for x in records for v in x['metrics'].values())
assert records[-1]['metrics']['val/qa_count/all']==64
actor=run/'output/checkpoints'/run.name/'actor/global_step_5'
files={p.name:p.stat().st_size for p in actor.iterdir() if p.is_file()}
assert len([n for n in files if n.endswith('.safetensors')])==3
assert not any('optim' in n.lower() for n in files)
def file_sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
file_hashes={name:file_sha(actor/name) for name in files}
source=root/'repo/verl/workers/fsdp_workers.py';data=source.read_bytes()
value=dict(status='strict_rejected_actor_optimizer_not_saved',accepted=False,run=run.name,job=3002,source_commit=source_commit,
 exact_outer_steps=[x['step'] for x in records],all_finite=True,validation_count=64,missing_actor_optimizer=True,
 actor_checkpoint=str(actor),actor_files=files,actor_files_sha256=file_hashes,no_resume=True,actor_save_source_sha256=hashlib.sha256(data).hexdigest(),
 metric_source=metric_source,metric_provenance=json.loads((run/'worker-log-provenance.json').read_text()),
 note='Five joint updates completed but actor Adam was not saved; cannot reconstruct it after workers exited. Author-native comparison remains required.')
(run/'strict-rejection.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
REJECT
