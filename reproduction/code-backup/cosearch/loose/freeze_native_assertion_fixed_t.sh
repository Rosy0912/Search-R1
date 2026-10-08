set -euo pipefail
python3 - <<'PY'
from pathlib import Path
import json,hashlib,time,subprocess
r=Path('/dataset/shaoshitong/search-r1-rl/cosearch/author-native');s=r/'continue-300-tools-t';src=r/'runs/author-native-continue-300-s/tools'
assert not subprocess.check_output(['squeue','-h','-w','h100-16','-o','%i'],text=True).strip()
assert not (s/'preparation-ready.json').exists()
ready=json.loads((src/'preparation-ready.json').read_text())
changed={'launch_observed.py','phase_sampling.py','bounded_sampling.py','native_job.py','start_continue.py','cpu_continue_gate.py','cpu_bounded_gate.py','warm-initialization.json','continuation-authorization.json','source-checkpoint-verification.json','cpu-continue-gate.json','cpu-bounded-gate.json'}
for n,h in ready['tools_sha256'].items():
 if n not in changed:assert hashlib.sha256((s/n).read_bytes()).hexdigest()==h,n
cg=json.loads((s/'cpu-continue-gate.json').read_text());pg=json.loads((s/'cpu-original-fallback-gate.json').read_text());bg=json.loads((s/'cpu-bounded-gate.json').read_text());v=json.loads((s/'source-checkpoint-verification.json').read_text())
assert cg['passed'] and cg['save_freq']==1 and cg['test_freq']==50 and cg['source']==v['source_run'] and cg['source_step']==v['source_step'] and cg['next_step']==v['source_step']+1 and cg['data_state_equal'] and cg['single_original_load'] and cg['data_sha256']==v['data_sha256'] and not cg['cuda_initialized']
assert pg['passed'] and not pg['CUDA_initialized'] and pg['actual_native_update_policy'] and pg['current_settings_preserved'] and pg['bounded_two_attempts'] and pg['RL_updates']==0
assert bg['passed'] and not bg['CUDA_initialized']
assert v['actual_full_file_shas_reverified'] and v['source_resources_released'] and v['ranks']==8 and sum(len(x['files']) for x in v['files'])==24
for n in ['warm-initialization.json','continuation-authorization.json']:
 p=s/n;x=json.loads(p.read_text());x.update(run='author-native-continue-300-t',source_checkpoint_run=v['source_run'],source_checkpoint_step=v['source_step'],checkpoint_frequency=1,validation_frequency=50,checkpoint_retention=2,correction='Human requested only learning support assertions removed; bounded2/current PPO/math/reward/min4/settings unchanged; terminal zero-support original fallback forwarded; source is latest complete S checkpoint after controlled restart',learning_support_diagnostic_only=True,bounded_two_attempts_preserved=True,operational_acceptance_required_completed_joint_iterations=5)
 p.write_text(json.dumps(x,indent=2)+'\n')
for n in changed|{'cpu_original_fallback_gate.py','cpu-original-fallback-gate.json'}:ready['tools_sha256'][n]=hashlib.sha256((s/n).read_bytes()).hexdigest()
ready.update(run='author-native-continue-300-t',created_at=time.time(),initialization=f"S{v['source_step']} fullstate -> total300/natural324/save1/val50/retention2; only learning support aborts removed; bounded2 preserved",resume_source=v['source_run'],resume_source_step=v['source_step'],resume_source_job=v['source_job'],zero_support_PPO_prohibited=False,qualified_zero_is_diagnostic_only=True,bounded_two_attempts_preserved=True,training_math_unchanged=True,operational_acceptance_required_completed_joint_iterations=5,strict_accepted=False)
(s/'preparation-ready.json').write_text(json.dumps(ready,indent=2)+'\n')
print(json.dumps(dict(run=ready['run'],ready_sha256=hashlib.sha256((s/'preparation-ready.json').read_bytes()).hexdigest(),ready=ready,cpu_continue=cg,cpu_original_fallback=pg,cpu_bounded=bg,source_full_verification=v)),flush=True)
PY
