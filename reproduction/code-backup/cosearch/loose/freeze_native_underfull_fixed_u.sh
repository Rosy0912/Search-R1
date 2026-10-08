set -euo pipefail
python3 - <<'FREEZE_U'
from pathlib import Path
import json,hashlib,time,subprocess
r=Path('/dataset/shaoshitong/search-r1-rl/cosearch/author-native');s=r/'continue-300-tools-u';src=r/'runs/author-native-continue-300-t/tools'
assert not subprocess.check_output(['squeue','-h','-w','h100-16','-o','%i'],text=True).strip()
assert not (s/'preparation-ready.json').exists()
assert hashlib.sha256((src/'preparation-ready.json').read_bytes()).hexdigest()=='4fbf415779151ee020ab5275d156fdffa97c12e24fc776c0d0ab6d9341cbcbce'
ready=json.loads((src/'preparation-ready.json').read_text())
changed={'observe_worker.py','native_job.py','start_continue.py','cpu_continue_gate.py','warm-initialization.json','continuation-authorization.json','source-checkpoint-verification.json','cpu-continue-gate.json'}
for n,h in ready['tools_sha256'].items():
 if n not in changed:assert hashlib.sha256((s/n).read_bytes()).hexdigest()==h,n
v=json.loads((s/'source-checkpoint-verification.json').read_text());cg=json.loads((s/'cpu-continue-gate.json').read_text());fg=json.loads((s/'cpu-fallback-alignment-gate.json').read_text())
assert v['source_run']=='author-native-continue-300-t' and v['source_step']==11 and v['source_job']=='3191' and v['actual_full_file_shas_reverified'] and v['source_resources_released'] and v['ranks']==8 and sum(len(x['files']) for x in v['files'])==24
assert cg['passed'] and cg['source']==v['source_run'] and cg['source_step']==11 and cg['next_step']==12 and cg['data_state_equal'] and cg['single_original_load'] and cg['data_sha256']==v['data_sha256'] and not cg['cuda_initialized'] and cg['save_freq']==1 and cg['test_freq']==50 and cg['target_step']==300 and cg['natural_horizon']==324 and cg['argv_count']==102
assert fg['passed'] and fg['actual_ray_modified_class'] and not fg['CUDA_initialized'] and fg['PPO_updates']==0 and fg['qualification_not_fabricated'] and fg['fix_sha256']==hashlib.sha256((s/'fallback_alignment.py').read_bytes()).hexdigest()
original=r/'repo/verl/verl/experimental/agent_loop/search_r1_reranker_agent_loop_worker.py';assert fg['source_sha256']==hashlib.sha256(original.read_bytes()).hexdigest()
for n in ['warm-initialization.json','continuation-authorization.json']:
 p=s/n;x=json.loads(p.read_text());x.update(run='author-native-continue-300-u',source_checkpoint_run=v['source_run'],source_checkpoint_step=11,checkpoint_frequency=1,validation_frequency=50,checkpoint_retention=2,correction='Nonempty underfull original fallback only: reuse original author _align_to_multiple; preserve groups/rewards/UIDs and full raw before/after; added copies do not qualify learning. All T sampling/PPO/Adam/scheduler/min4/support diagnostics unchanged.',operational_acceptance_required_completed_joint_iterations=5)
 p.write_text(json.dumps(x,indent=2)+'\n')
for n in changed|{'fallback_alignment.py','cpu_fallback_alignment_gate.py','cpu-fallback-alignment-gate.json'}:ready['tools_sha256'][n]=hashlib.sha256((s/n).read_bytes()).hexdigest()
ready.update(run='author-native-continue-300-u',created_at=time.time(),initialization='T11 fullstate -> total300/natural324/save1/val50/retention2; original underfull fallback alignment fix only',resume_source=v['source_run'],resume_source_step=11,resume_source_job='3191',underfull_fallback_author_alignment=True,operational_acceptance_required_completed_joint_iterations=5,strict_accepted=False)
(s/'preparation-ready.json').write_text(json.dumps(ready,indent=2)+'\n')
print(json.dumps(dict(run=ready['run'],ready_sha256=hashlib.sha256((s/'preparation-ready.json').read_bytes()).hexdigest(),ready=ready,cpu_continue=cg,cpu_fallback_alignment=fg,source_full_verification=v)),flush=True)
FREEZE_U
