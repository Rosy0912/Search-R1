set -euo pipefail
python3 - <<'FREEZE_V'
from pathlib import Path
import json,hashlib,time,subprocess
r=Path('/dataset/shaoshitong/search-r1-rl/cosearch/author-native');s=r/'continue-300-tools-v';src=r/'runs/author-native-continue-300-u/tools'
hashfile=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not subprocess.check_output(['squeue','-h','-w','h100-16','-o','%i'],text=True).strip()
assert not (s/'preparation-ready.json').exists()
assert not (r/'runs/author-native-continue-300-v').exists()
assert not (r/'author-native-continue-300-v-launch-claim.json').exists()
assert hashfile(src/'preparation-ready.json')=='9d387aa2d25c890332fa17f01d18ed49ee42c83925fc23d5db107a2205a8a95d'
ready=json.loads((src/'preparation-ready.json').read_text())
changed={'cpu_continue_gate.py','observe_worker.py','launch_observed.py','bounded_sampling.py','start_continue.py','warm-initialization.json','continuation-authorization.json','source-checkpoint-verification.json','cpu-continue-gate.json','cpu-original-fallback-gate.json'}
for n,h in ready['tools_sha256'].items():
 if n not in changed:assert hashfile(s/n)==h,n
expected_core={'observe_worker.py': 'f6d7a36bbd307a28ae4555e1532d5ef4210d2010c0f423e31c02d9263e090610', 'launch_observed.py': 'e3477fe53e4e540dd3ec37e8c226ad7f16086dfc5ebf1b57dcef005695f2e1f3', 'bounded_sampling.py': '56106c0148f665755f00c3be50b6b358450d60f64ac744ca1b9da17a37a38837', 'start_continue.py': 'faa2d92fbc419e137fa08929fe97ee74ad652eac4f886b34de044715fc8c010d', 'empty_ranker.py': '5c2e5043df14d784374d49f158970b2523408a6e6285584143406c55336ebd73', 'cpu_empty_ranker_gate.py': 'f14325a2a803a66cf16fbf61ad6612390872b2446470382e18b68582c320823d'}
expected_core['cpu_continue_gate.py']='3f1891d3c889e54d86159154a180b06b36892b0475a78fc54f134c8848d3585a'
for n,h in expected_core.items():assert hashfile(s/n)==h,('Reviewed V source changed',n)
v=json.loads((s/'source-checkpoint-verification.json').read_text());cg=json.loads((s/'cpu-continue-gate.json').read_text());eg=json.loads((s/'cpu-empty-ranker-gate.json').read_text());pg=json.loads((s/'cpu-original-fallback-gate.json').read_text())
assert v['source_run']=='author-native-continue-300-u' and v['source_step']==12 and v['source_job']=='3201'
assert v['actual_full_file_shas_reverified'] and v['source_resources_released'] and v['ranks']==8
assert {(x['role'],x['rank']) for x in v['files']}=={(role,rank) for role in ['actor','reranker_actor_rollout'] for rank in range(4)}
assert len(v['files'])==8 and all(len(x['files'])==3 for x in v['files'])
assert v['data_sha256']=='e900aca11f4f1b9b36c9c82ae91b517181bca38952713caf49ea5718f19f9341'
assert str(v['source_job']) not in subprocess.check_output(['squeue','-h','-o','%i'],text=True).split()
assert cg['passed'] and cg['source']==v['source_run'] and cg['source_step']==12 and cg['next_step']==13
assert cg['data_state_equal'] and cg['single_original_load'] and cg['data_sha256']==v['data_sha256'] and not cg['cuda_initialized']
assert cg['save_freq']==1 and cg['test_freq']==50 and cg['target_step']==300 and cg['natural_horizon']==324 and cg['argv_count']==102
assert eg['passed'] and eg['actual_modified_worker'] and eg['actual_manager_mixed_cases']==6
assert eg['actual_transformed_fit_main_and_ranker_RPC_cases']==4 and eg['actual_original_save_both_roles_cases']==4
assert eg['bounded_terminal_empty_two_calls'] and not eg['CUDA_initialized'] and eg['PPO_updates']==0
assert eg['fix_sha256']==hashfile(s/'empty_ranker.py')
assert pg['passed'] and pg['actual_native_update_policy'] and pg['actual_submitted_wrapper'] and pg['current_settings_preserved'] and pg['bounded_two_attempts']
assert not pg['CUDA_initialized'] and pg['RL_updates']==0 and len(pg['cases'])==6
assert all(x['all_gradient_clip_parameter_adam_scheduler_rng_data_exact'] for x in pg['cases'])
ps={Path(k).name:h for k,h in pg['source_sha256'].items()}
assert ps['launch_observed.py']==hashfile(s/'launch_observed.py') and ps['cpu_original_fallback_gate.py']==hashfile(s/'cpu_original_fallback_gate.py')
assert ps['dp_actor.py']==hashfile(r/'repo/verl/verl/workers/actor/dp_actor.py')
for n in ['warm-initialization.json','continuation-authorization.json']:
 p=s/n;x=json.loads(p.read_text());x.update(run='author-native-continue-300-v',source_checkpoint_run=v['source_run'],source_checkpoint_step=12,checkpoint_frequency=1,validation_frequency=50,checkpoint_retention=2,correction='Canonical empty local ranker shards only; preserve all true main/nonempty ranker rows; skip zero-row ranker PPO/Adam/scheduler without fake rows/rewards/qualification. Nonempty author paths unchanged. Original bounded TWO attempts retained.',operational_acceptance_required_completed_joint_iterations=5,main_only_empty_ranker_outer_excluded_from_joint_acceptance=True)
 p.write_text(json.dumps(x,indent=2)+'\n')
new={'empty_ranker.py','cpu_empty_ranker_gate.py','cpu-empty-ranker-gate.json'}
for n in changed|new:ready['tools_sha256'][n]=hashfile(s/n)
ready.update(run='author-native-continue-300-v',created_at=time.time(),initialization='U12 fullstate -> total300/natural324/save1/val50/retention2; canonical empty-ranker correction only',resume_source=v['source_run'],resume_source_step=12,resume_source_job='3201',explicit_empty_ranker_support=True,global_empty_ranker_optimizer_scheduler_steps=0,main_only_empty_ranker_outer_excluded_from_joint_acceptance=True,operational_acceptance_required_completed_joint_iterations=5,strict_accepted=False)
(s/'preparation-ready.json').write_text(json.dumps(ready,indent=2)+'\n')
print(json.dumps(dict(run=ready['run'],ready_sha256=hashfile(s/'preparation-ready.json'),ready=ready,cpu_continue=cg,cpu_empty_ranker=eg,cpu_original_fallback=pg,source_full_verification=v)),flush=True)
FREEZE_V
