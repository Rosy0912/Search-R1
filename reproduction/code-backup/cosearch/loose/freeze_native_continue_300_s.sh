set -euo pipefail
python3 - <<'PY'
from pathlib import Path
import json,hashlib,subprocess,time,shutil
r=Path('/dataset/shaoshitong/search-r1-rl/cosearch/author-native');s=r/'continue-300-tools-s';src=r/'runs/author-native-continue-300-r/tools'
assert not subprocess.check_output(['squeue','-h','-w','h100-16','-o','%i'],text=True).strip()
assert not (s/'preparation-ready.json').exists()
ready=json.loads((src/'preparation-ready.json').read_text())
changed={'launch_observed.py','start_continue.py','warm-initialization.json','continuation-authorization.json','source-checkpoint-verification.json','cpu_continue_gate.py','cpu-continue-gate.json'}
for n,h in ready['tools_sha256'].items():
 if n not in changed:assert hashlib.sha256((s/n).read_bytes()).hexdigest()==h,n
cg=json.loads((s/'cpu-continue-gate.json').read_text());rg=json.loads((s/'cpu-bounded-gate.json').read_text());v=json.loads((s/'source-checkpoint-verification.json').read_text())
assert cg['save_freq']==1 and cg['test_freq']==50 and cg['passed'] and cg['data_state_equal'] and cg['single_original_load'] and not cg['cuda_initialized']
assert rg['passed'] and rg['actual_submitted_wrapper_executed'] and not rg['CUDA_initialized'] and rg['RL_updates']==0
assert v['actual_full_file_shas_reverified'] and v['ranks']==8 and v['checked_at']>1791106000
assert v['data_sha256']=='ba7a5fb645fd0b3ab552329fa73bdbc57a6b0fb94cc3eacd4e01e17afae9b14a'
for n in changed|{'bounded_sampling.py','cpu_bounded_gate.py','cpu-bounded-gate.json'}:ready['tools_sha256'][n]=hashlib.sha256((s/n).read_bytes()).hexdigest()
ready.update(run='author-native-continue-300-s',created_at=time.time(),initialization='P5 fullstate -> total300/natural324/val64 every50/save EACH completed outer/retention2; same two-attempt cap; all attempt costs and original checkpoint save cost disclosed',rollout_budget_changed=True,accepted_batch_selection_changed=True,zero_support_PPO_prohibited=True,old_R_failed_step11=True,checkpoint_frequency_changed=True,original_save_freq=1,original_test_freq=50)
(s/'preparation-ready.json').write_text(json.dumps(ready,indent=2)+'\n')
print(json.dumps(dict(run=ready['run'],ready_sha256=hashlib.sha256((s/'preparation-ready.json').read_bytes()).hexdigest(),ready=ready,cpu_continue=cg,cpu_bounded=rg,source_full_verification=v,disk_free=shutil.disk_usage(r).free)),flush=True)
PY
