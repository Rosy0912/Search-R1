set -euo pipefail
python3 - <<'FREEZE_W'
from pathlib import Path
import json,hashlib,time,subprocess
r=Path('/dataset/shaoshitong/search-r1-rl/cosearch/author-native');s=r/'continue-300-tools-w';src=r/'runs/author-native-continue-300-v/tools'
hashfile=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not subprocess.check_output(['squeue','-h','-w','h100-16','-o','%i'],text=True).strip()
assert not (s/'preparation-ready.json').exists()
assert not (r/'runs/author-native-continue-300-w').exists()
assert not (r/'author-native-continue-300-w-launch-claim.json').exists()
assert hashfile(src/'preparation-ready.json')=='e806a4d75b03a9de615233f3d8a7d17724969188869c9d2b460f7b2451977e1c'
ready=json.loads((src/'preparation-ready.json').read_text())
changed={'empty_ranker.py','start_continue.py','warm-initialization.json','continuation-authorization.json','source-checkpoint-verification.json','cpu-continue-gate.json','cpu-empty-ranker-gate.json'}
for n,h in ready['tools_sha256'].items():
 if n not in changed:assert hashfile(s/n)==h,n
expected_core={'empty_ranker.py': '6060e73f367162a946e3bc8edf428dd1f955ff9a8b4a803dfc22d7d50aab2778', 'start_continue.py': 'af9a2fa1a65d1f45176d4a123d5f6c7fe2bea9927c4729d954916b7ae6ea8044', 'cpu_worker_serialization_gate.py': '9e17bd14b582bae78aec9a3e7ccfc4f24a0d87f731b2b8b80a87f299120b0894'}
for n,h in expected_core.items():assert hashfile(s/n)==h,('Reviewed V source changed',n)
v=json.loads((s/'source-checkpoint-verification.json').read_text());cg=json.loads((s/'cpu-continue-gate.json').read_text());eg=json.loads((s/'cpu-empty-ranker-gate.json').read_text());pg=json.loads((s/'cpu-original-fallback-gate.json').read_text());sg=json.loads((s/'cpu-worker-serialization-gate.json').read_text())
assert v['source_run']=='author-native-continue-300-u' and v['source_step']==12 and v['source_job']=='3201'
assert v['actual_full_file_shas_reverified'] and v['source_resources_released'] and v['ranks']==8
assert {(x['role'],x['rank']) for x in v['files']}=={(role,rank) for role in ['actor','reranker_actor_rollout'] for rank in range(4)}
assert len(v['files'])==8 and all(len(x['files'])==3 for x in v['files'])
assert v['data_sha256']=='e900aca11f4f1b9b36c9c82ae91b517181bca38952713caf49ea5718f19f9341'
assert str(v['source_job']) not in subprocess.check_output(['squeue','-h','-o','%i'],text=True).split()
release=json.loads((s/'previous-launch-release.json').read_text())
assert release['run']=='author-native-continue-300-v' and release['job']=='3205' and release['driver_exit']['returncode']==1
assert release['old8_PIDticks_gone'] and release['node_queue_empty']
assert '3205' not in subprocess.check_output(['squeue','-h','-o','%i'],text=True).split()
assert cg['passed'] and cg['source']==v['source_run'] and cg['source_step']==12 and cg['next_step']==13
assert cg['data_state_equal'] and cg['single_original_load'] and cg['data_sha256']==v['data_sha256'] and not cg['cuda_initialized']
assert cg['save_freq']==1 and cg['test_freq']==50 and cg['target_step']==300 and cg['natural_horizon']==324 and cg['argv_count']==102
assert eg['passed'] and eg['actual_modified_worker'] and eg['actual_manager_mixed_cases']==6
assert eg['actual_transformed_fit_main_and_ranker_RPC_cases']==4 and eg['actual_original_save_both_roles_cases']==4
assert eg['bounded_terminal_empty_two_calls'] and not eg['CUDA_initialized'] and eg['PPO_updates']==0
assert eg['fix_sha256']==hashfile(s/'empty_ranker.py')
assert sg['passed'] and sg['actual_ray_modified_class_full_hooks'] and sg['legacy_ContextVar_failure_reproduced'] and sg['actual_ray_cloudpickle_roundtrip'] and sg['fresh_interpreter_original_generate_passed']
assert not sg['CUDA_initialized'] and not sg['Ray_initialized'] and sg['PPO_updates']==0
assert sg['fix_sha256']==hashfile(s/'empty_ranker.py') and sg['observer_sha256']==hashfile(s/'observe_worker.py')
assert pg['passed'] and pg['actual_native_update_policy'] and pg['actual_submitted_wrapper'] and pg['current_settings_preserved'] and pg['bounded_two_attempts']
assert not pg['CUDA_initialized'] and pg['RL_updates']==0 and len(pg['cases'])==6
assert all(x['all_gradient_clip_parameter_adam_scheduler_rng_data_exact'] for x in pg['cases'])
ps={Path(k).name:h for k,h in pg['source_sha256'].items()}
assert ps['launch_observed.py']==hashfile(s/'launch_observed.py') and ps['cpu_original_fallback_gate.py']==hashfile(s/'cpu_original_fallback_gate.py')
assert ps['dp_actor.py']==hashfile(r/'repo/verl/verl/workers/actor/dp_actor.py')
for n in ['warm-initialization.json','continuation-authorization.json']:
 p=s/n;x=json.loads(p.read_text());x.update(run='author-native-continue-300-w',source_checkpoint_run=v['source_run'],source_checkpoint_step=12,checkpoint_frequency=1,validation_frequency=50,checkpoint_retention=2,correction='Serialization-only fix: resolve per-process ContextVar via runtime module import; V canonical empty handling and all model/sampling/PPO/save settings unchanged. Full actual actor-class cloudpickle/fresh-interpreter CPU gate required.',operational_acceptance_required_completed_joint_iterations=5,main_only_empty_ranker_outer_excluded_from_joint_acceptance=True)
 p.write_text(json.dumps(x,indent=2)+'\n')
new={'cpu_worker_serialization_gate.py','cpu-worker-serialization-gate.json','previous-launch-release.json'}
for n in changed|new:ready['tools_sha256'][n]=hashfile(s/n)
ready.update(run='author-native-continue-300-w',created_at=time.time(),initialization='U12 fullstate -> total300/natural324/save1/val50/retention2; serialization-only ContextVar runtime-import fix',resume_source=v['source_run'],resume_source_step=12,resume_source_job='3201',explicit_empty_ranker_support=True,actual_worker_cloudpickle_cpu_passed=True,global_empty_ranker_optimizer_scheduler_steps=0,main_only_empty_ranker_outer_excluded_from_joint_acceptance=True,operational_acceptance_required_completed_joint_iterations=5,strict_accepted=False)
(s/'preparation-ready.json').write_text(json.dumps(ready,indent=2)+'\n')
print(json.dumps(dict(run=ready['run'],ready_sha256=hashfile(s/'preparation-ready.json'),ready=ready,cpu_continue=cg,cpu_empty_ranker=eg,cpu_original_fallback=pg,cpu_worker_serialization=sg,previous_launch_release=release,source_full_verification=v)),flush=True)
FREEZE_W
