set -euo pipefail
python3 - <<'PY'
from pathlib import Path
import json,hashlib,shutil,subprocess,time
r=Path('/dataset/shaoshitong/search-r1-rl/cosearch/author-native');s=r/'continue-300-tools-q';t=r/'tools'
assert not subprocess.check_output(['squeue','-h','-w','h100-16','-o','%i'],text=True).strip()
v=json.loads((s/'source-checkpoint-verification.json').read_text());assert v['actual_full_file_shas_reverified'] and v['ranks']==8
x=json.loads((s/'warm-initialization.json').read_text());x.update(run='author-native-continue-300-q',fresh_RL_optimizers_required=False,RL_optimizer_restored_from='author-native-checkpoint-fixed-five-p/global_step_5',sft_optimizer_not_loaded_into_RL=True,initial_warm_HF_identity_retained_for_reference=True)
(s/'warm-initialization.json').write_text(json.dumps(x,indent=2)+'\n')
a=json.loads((s/'continuation-authorization.json').read_text());a['checkpoints_to_keep']=2;(s/'continuation-authorization.json').write_text(json.dumps(a,indent=2)+'\n')
ready=json.loads((s/'preparation-ready.json').read_text())
for n in ['warm-initialization.json','continuation-authorization.json','source-checkpoint-verification.json']:ready['tools_sha256'][n]=hashlib.sha256((s/n).read_bytes()).hexdigest()
ready['source_checkpoint_full24file_sha_verified']=True
(s/'preparation-ready.json').write_text(json.dumps(ready,indent=2)+'\n')
for n,h in ready['tools_sha256'].items():assert hashlib.sha256((s/n).read_bytes()).hexdigest()==h,n
for p in s.iterdir():
 if p.is_file() and p.suffix in ['.py','.json','.sh','.yaml']:shutil.copy2(p,t/p.name)
for n,h in ready['tools_sha256'].items():assert hashlib.sha256((t/n).read_bytes()).hexdigest()==h,n
assert (s/'preparation-ready.json').read_bytes()==(t/'preparation-ready.json').read_bytes()
print(json.dumps(dict(run=ready['run'],ready_sha256=hashlib.sha256((t/'preparation-ready.json').read_bytes()).hexdigest(),tools_sha256=ready['tools_sha256'],source_verified=True,deployed_readback_equal=True,checked_at=time.time())),flush=True)
PY
