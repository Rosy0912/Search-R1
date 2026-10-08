set -euo pipefail
python3 - <<'FREEZE_X'
from pathlib import Path
import json,hashlib,time,subprocess
r=Path('/dataset/shaoshitong/search-r1-rl/cosearch/author-native');s=r/'continue-300-tools-x';src=r/'runs/author-native-continue-300-w/tools'
hashfile=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not subprocess.check_output(['squeue','-h','-w','h100-16','-o','%i'],text=True).strip()
assert not (s/'preparation-ready.json').exists()
assert not (r/'runs/author-native-continue-300-x').exists()
assert not (r/'author-native-continue-300-x-launch-claim.json').exists()
assert hashfile(src/'preparation-ready.json')=='184a390530e4fc87100d48869688ed5b4aac99f803f5594721e756d915ef9106'
ready=json.loads((src/'preparation-ready.json').read_text())
changed={'empty_ranker.py','start_continue.py','warm-initialization.json','continuation-authorization.json','source-checkpoint-verification.json','cpu-continue-gate.json','cpu-empty-ranker-gate.json','cpu-worker-serialization-gate.json','previous-launch-release.json'}
for n,h in ready['tools_sha256'].items():
 if n not in changed:assert hashfile(s/n)==h,n
expected_core={'empty_ranker.py': '2416b29c08d3ac350173a0c3866511d0576427f05324d9cbd90995de4a15d6ea', 'start_continue.py': '8d6add0e59fc25292507aa431535776776679f2ced56058f46c34e9b45f336ff', 'optional_metadata.py': 'feace77acf04058db2f59b004e1907681be2bc66612da46c5d6a59ba4e8c8b85', 'cpu_optional_metadata_gate.py': '182df76144f7e678a353febc768cc5a72a86508ba58c6e723b19d6d77504268a'}
for n,h in expected_core.items():assert hashfile(s/n)==h,('Reviewed V source changed',n)
v=json.loads((s/'source-checkpoint-verification.json').read_text());cg=json.loads((s/'cpu-continue-gate.json').read_text());eg=json.loads((s/'cpu-empty-ranker-gate.json').read_text());pg=json.loads((s/'cpu-original-fallback-gate.json').read_text());sg=json.loads((s/'cpu-worker-serialization-gate.json').read_text());mg=json.loads((s/'cpu-optional-metadata-gate.json').read_text())
assert v['source_run']=='author-native-continue-300-u' and v['source_step']==12 and v['source_job']=='3201'
assert v['actual_full_file_shas_reverified'] and v['source_resources_released'] and v['ranks']==8
assert {(x['role'],x['rank']) for x in v['files']}=={(role,rank) for role in ['actor','reranker_actor_rollout'] for rank in range(4)}
assert len(v['files'])==8 and all(len(x['files'])==3 for x in v['files'])
assert v['data_sha256']=='e900aca11f4f1b9b36c9c82ae91b517181bca38952713caf49ea5718f19f9341'
assert str(v['source_job']) not in subprocess.check_output(['squeue','-h','-o','%i'],text=True).split()
release=json.loads((s/'previous-launch-release.json').read_text())
assert release['run']=='author-native-continue-300-w' and release['job']=='3206' and release['driver_exit']['returncode']==1
assert release['old8_PIDticks_gone'] and release['node_queue_empty']
assert '3206' not in subprocess.check_output(['squeue','-h','-o','%i'],text=True).split()
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
assert mg['passed'] and mg['actual_original_concat_3072_vs4096_reproduced'] and mg['actual_manager_train_and_validation'] and mg['main_training_tensors_rewards_UID_unchanged']
assert mg['known_optional_fields']==['tool_call_details','messages','initial_query','answers'] and mg['missing_vs_explicit_None_evidence'] and mg['allpresent_allabsent_parity'] and mg['RNG_unchanged']
assert mg['malformed_present_rejections']==16 and mg['required_field_rejections']==2 and mg['evidence_errors_propagated']==2 and not mg['CUDA_initialized'] and mg['PPO_updates']==0
assert mg['optional_sha256']==hashfile(s/'optional_metadata.py') and mg['empty_sha256']==hashfile(s/'empty_ranker.py')
assert pg['passed'] and pg['actual_native_update_policy'] and pg['actual_submitted_wrapper'] and pg['current_settings_preserved'] and pg['bounded_two_attempts']
assert not pg['CUDA_initialized'] and pg['RL_updates']==0 and len(pg['cases'])==6
assert all(x['all_gradient_clip_parameter_adam_scheduler_rng_data_exact'] for x in pg['cases'])
ps={Path(k).name:h for k,h in pg['source_sha256'].items()}
assert ps['launch_observed.py']==hashfile(s/'launch_observed.py') and ps['cpu_original_fallback_gate.py']==hashfile(s/'cpu_original_fallback_gate.py')
assert ps['dp_actor.py']==hashfile(r/'repo/verl/verl/workers/actor/dp_actor.py')
for n in ['warm-initialization.json','continuation-authorization.json']:
 p=s/n;x=json.loads(p.read_text());x.update(run='author-native-continue-300-x',source_checkpoint_run=v['source_run'],source_checkpoint_step=12,checkpoint_frequency=1,validation_frequency=50,checkpoint_retention=2,correction='Across-shard author extra_fields.get(None) alignment only for four confirmed trajectory-only fields: tool_call_details/messages/initial_query/answers. Preserve true missing-vs-explicitNone raw; present malformed or required fields still fail. W serialization/empty/bounded2/model/PPO/Adam/save settings unchanged.',operational_acceptance_required_completed_joint_iterations=5,main_only_empty_ranker_outer_excluded_from_joint_acceptance=True)
 p.write_text(json.dumps(x,indent=2)+'\n')
new={'optional_metadata.py','cpu_optional_metadata_gate.py','cpu-optional-metadata-gate.json'}
for n in changed|new:ready['tools_sha256'][n]=hashfile(s/n)
ready.update(run='author-native-continue-300-x',created_at=time.time(),initialization='U12 fullstate -> total300/natural324/save1/val50/retention2; four confirmed optional trajectory fields missing-shard None alignment only',resume_source=v['source_run'],resume_source_step=12,resume_source_job='3201',explicit_empty_ranker_support=True,actual_worker_cloudpickle_cpu_passed=True,known_optional_trajectory_fields=['tool_call_details','messages','initial_query','answers'],global_empty_ranker_optimizer_scheduler_steps=0,main_only_empty_ranker_outer_excluded_from_joint_acceptance=True,operational_acceptance_required_completed_joint_iterations=5,strict_accepted=False)
(s/'preparation-ready.json').write_text(json.dumps(ready,indent=2)+'\n')
print(json.dumps(dict(run=ready['run'],ready_sha256=hashfile(s/'preparation-ready.json'),ready=ready,cpu_continue=cg,cpu_empty_ranker=eg,cpu_original_fallback=pg,cpu_worker_serialization=sg,cpu_optional_metadata=mg,previous_launch_release=release,source_full_verification=v)),flush=True)
FREEZE_X
