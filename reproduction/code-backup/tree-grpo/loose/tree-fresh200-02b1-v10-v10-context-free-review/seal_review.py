"""Seal independent SOURCE/CPU scope; never promote actual execution acceptance."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
SOURCE=ROOT/'work/tree-behavior-mu-utf8-v2';COMMIT='02b1ea2688ff3fb032ee6699003a7e09d651f166'
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(name):return json.loads((OUT/name).read_text())
def write(name,value):
    path=OUT/name;path.write_text(json.dumps(value,indent=2,sort_keys=True,ensure_ascii=False)+'\n');return digest(path)
assert subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD']).decode().strip()==COMMIT
assert not subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain=v1'])
audit=read('BYTE_AUDIT.json');rawprobe=read('INDEPENDENT_UTF8_SEMANTIC_PROBE_RESULT.json')
runs=read('ACTUAL_SELECTED_CPU_RUNS.json');submit=read('INDEPENDENT_SUBMIT_V10_PROBE_RESULT.json');extra=read('INDEPENDENT_HELPER_SAFETY_PROBE_RESULT.json')
assert len(audit['source_files'])==19 and len(audit['wrapper_files'])==15 and len(audit['helper_files'])==3
assert rawprobe['all_expected_cases_passed'] and all(run['exit_code']==0 for run in runs)
assert len(submit['cases'])==10 and len(extra['cases'])==4
for name,entry in audit['source_files'].items():assert digest(SOURCE/name)==entry['sha256']
for name,entry in audit['wrapper_files'].items():assert digest(ROOT/'work/tree-fresh200-mu-runtime-v10'/name)==entry['sha256']
for name,entry in audit['helper_files'].items():assert digest(ROOT/'work/tree-fresh200-dispatch-v10'/name)==entry['sha256']
context=dict(fork_turns='none',inherited_parent_conversation=False)
false_scope=dict(actual_gpu_accepted=False,actual_gpu_executed=False,actual_gpu_ready=False,
    actual_remote_executed=False,actual_slurm_executed=False,actual_slurm_submitted=False,
    actual_zero_ready=False,actual_ready=False,production_thresholds_accepted=False,
    original_sampler_actor_gate_accepted=False,actual_3b_training_accepted=False,
    long_training_ready=False,training_release=False,actual_assets_loaded=False,
    actual_official_tokenizer_executed=False,actual_official_model_or_dataset_assets_loaded=False,
    actual_ray_executed=False,actual_optimizer_writes=0,actual_completed_outer=0)
helper=dict(schema_version=1,status='FRESH200_PREPARE_AND_SUBMIT_HELPERS_SOURCE_CPU_REVIEWED',
    reviewer_context=context,code_commit=COMMIT,binding_new_source_commit=COMMIT,
    helper_files={name:entry['sha256'] for name,entry in audit['helper_files'].items()},
    helper_inventory_sha256=audit['helper_inventory_sha256'],wrapper_inventory_sha256=audit['wrapper_inventory_sha256'],
    helper_source_capability_accepted=True,source_only=True,all_scheduler_calls_mocked=True,
    independent_submit_case_groups=10,independent_extra_safety_case_groups=4,
    immutable_stdin_and_raw_output_verified=True,durable_file_and_directory_intent_before_mock_submit_verified=True,
    whole_task_nonblocking_lock_verified=True,unknown_no_retry_verified=True,
    local_default_cluster='wuhan-h100',federation_refused=True,protected_original_eight_pid_start_identity_verified_on_synthetic_metadata=True,
    exact_source_interface_review_accepted=True,
    source_cpu_preparation_capability_only=True,
    artifacts={name:digest(OUT/name) for name in ('independent_submit_v10_probe.py','INDEPENDENT_SUBMIT_V10_PROBE_RESULT.json',
        'independent_helper_safety_probe.py','INDEPENDENT_HELPER_SAFETY_PROBE_RESULT.json','ACTUAL_SELECTED_CPU_RUNS.json')},
    limitations=['Source/local CPU capability only. No actual package or execution argv release.',
        'Same-node CPU preparation remains contingent on separate exact local envelope and argv binding review.',
        'No real scheduler command was invoked. Actual official5120 tokenizer/preflight and fresh zero/five/production stages remain independently gated.'],
    **false_scope)
helper_sha=write('HELPER_SOURCE_REVIEW.json',helper)
owner_path=ROOT/'work/tree-fresh200-utf8-v2-owner/UTF8_V2_SOURCE_INTERFACE_CPU_RECEIPT.json'
owner=json.loads(owner_path.read_text())
assert digest(owner_path)=='5648c41dc787f95d750157fd876194182c6812665b6e4039db31a0ef86140ce2'
for artifact in ('FROZEN_02B1EA2_AFFECTED_CPU_RUN.json','FROZEN_02B1EA2_AFFECTED_CPU_ACTUAL.txt',
    'FROZEN_02B1EA2_BUDGET_CPU_RUN.json','FROZEN_02B1EA2_BUDGET_CPU_ACTUAL.txt','ACTUAL_TINY_VALIDATION_CONTEXT_COST.json'):
    record=owner['artifacts'][artifact];assert digest(Path(record['path']))==record['sha256']
evidence_names=('BYTE_AUDIT.json','HELPER_SOURCE_REVIEW.json','INDEPENDENT_UTF8_SEMANTIC_PROBE_RESULT.json',
    'independent_utf8_semantic_probe.py','INDEPENDENT_UTF8_SEMANTIC_PROBE.log','SOURCE_SELECTED_8.stderr',
    'ACTUAL_SELECTED_CPU_RUNS.json','INDEPENDENT_SUBMIT_V10_PROBE_RESULT.json','INDEPENDENT_HELPER_SAFETY_PROBE_RESULT.json',
    'FIXTURE_FAILURE_RECORD.json')
receipt=dict(schema_version=1,status='FRESH200_02B1_V10_SOURCE_CPU_ACCEPTED_ONLY',review_status='completed',
    reviewer_context=context,source=dict(directory=str(SOURCE),audited_clean_head=COMMIT,
        initial_clean=True,final_clean=True,source_interface_count=15,additional_runtime_count=4,
        source_manifest_sha256=audit['source_manifest_sha256'],source_owner_receipt_sha256=digest(owner_path),
        changed_runtime_files=['search_r1/baselines/tree_raw_tokens.py','verl/workers/fsdp_workers.py'],
        unchanged_from_af67976_runtime_count=17),
    source_interface_reviewed=True,source_interface_review_accepted=True,
    execution_scope=dict(local_source_metadata_and_tiny_cpu_review_only=True,
        same_h100_9_cpu_only_preparation_allowed=True,
        same_h100_9_cpu_preparation_actually_executed=False,
        exact_local_packet_or_execution_argv_review_completed=False,
        actual_official_node5120_compatibility_verified=False,
        actual_zero_write_release=False,actual_living_five_release=False,actual_six_to200_release=False,
        restriction='CPU preparation capability accepted; exact envelope and argv must receive separate execution-before independent review.'),
    stages=dict(preparation=dict(accepted=True,source_cpu_capability_only=True,actual_same_node_executed=False),
        fresh_mu_calibration=dict(source_review_completed=True,source_review_accepted=True,
            actual_gpu_ready=False,actual_official_tokenizer_5120_accepted=False,exact_package_dispatch_accepted=False),
        living_first_five=dict(source_cpu_capability_only=True,actual_gpu_ready=False,actual_five_accepted=False),
        formal_six_to200=dict(new_source_formal_combination_review_accepted=False,actual_gpu_ready=False,actual_training_accepted=False)),
    source_review_findings=[],
    verified=dict(resealed_single_ids_reward_and_double_change_refused_after_all_hash_recomputation=True,
        json_reserialized_projection_refused=True,mu_validator_and_actual_actor_body_pre_forward_refused=True,
        actor_attack_native_forward_calls=0,actor_attack_optimizer_state_entries=0,
        legal_multibyte_prefix_cases=17,fresh_backend_illegal_byte_scan_cases=6,
        legal_ancestor_sharing_preserved=True,policy_ids_masks_and_selected_mu_scalars_not_reencoded_or_filtered=True,
        current_live_path_file_sha_backend_semantics_and_default_flags_revalidated=True,
        normal_backend_padding_truncation_not_rejected=True,no_ledger_path_loading=True,
        independent_cpu_spawn_actual_fsdp_registration_statement_passed=True,
        driver_and_colocated_ray_actor_ref_registration_before_mu_is_source_reviewed=True,
        backend_cache_scope='One fresh context per identity per validation entrance; real serialization once per trajectory, no cross-call authority cache.',
        file_cache_scope='Actual resolved path and inode/size/mtime/ctime/mode checked; same-length synthetic file rewrite rejected.',
        wrapper15_exact_previous_accepted_bytes=True,helper3_exact_bytes=True,helper_common_only_commit_repin=True,
        scheduler_all_mocked=True,original_guard_source_hash_pin_and_owned_cleanup_unchanged=True,
        child_current_cgroup_before_query=True,requested_allocation=dict(gpus=8,exclusive=True,cpus=32,mem_gib=256,hours=3,no_requeue=True)),
    human_target_contract=dict(training_protocol='tree_fresh200_five_leaf_v1',fresh_from_official_initial_state=True,
        inherited_old_weights_adam_scheduler_cursor=False,official_revision='aa8e72537993ba99e69dfaafa59ed015b17504d1',
        data_identity='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24',
        complete_outer_budget=200,questions=512,leaf_quotas=[3,2],total_true_leaves=2560,
        mini_global=128,actor_micro_global=16,actor_micro_local=2,old_ref_micro_global=32,old_ref_micro_local=4,dp=8,
        lr=1e-6,warmup_completed_outers=57,then_constant_lr=True,dual_clip_negative_advantage_c=3.,
        sampled_policy_kl_brake=.2,finite_before_exp_bound=30.,negative_tail_gate_required=True,
        opportunities_per_rank_per_complete_outer=20,minimum_actual_successful_adam_returns_for_credit=1,
        actual_successful_writes_reported_separately=True,allskip_abandon_credit_outer_or_scheduler=False,
        retain_actual_writes_without_fake_rollback=True,guard_stops_remaining_old_batch_writes=True,
        continue_with_fresh_rollout=True,living_first_five_then_external_actual_acceptance_before6_to200=True,
        same_source_checkpoints=[5,50,100,150,200],dev_every_completed_outers=5,
        final_rows=51713,final_full_hf_cost_and_own_cleanup_required=True),
    explicit_profile_scope='Behavior-mu correction is a separately opted-in alternate denominator/protocol. Original sampler/actor gate and original tolerance are not relaxed or accepted.',
    historical_evidence_not_promoted=['Prior f7 source CPU receipt is used only for unchanged byte closure.',
        'Prior af67976 source remains rejected for SOURCE_P01.',
        'Prior143/Gloo suite was not rerun on02b1 and is not02b1 GPU evidence.',
        'Historical3295/3296 best_of/_real_n None rows do not prove fresh native24 settings.'],
    actual_cpu_evidence=dict(independent_utf8_probe_process_cpu_seconds=rawprobe['process_cpu_seconds'],
        independent_utf8_probe_wall_seconds=rawprobe['wall_seconds'],selected_source_test_methods=8,
        selected_source_unittest_seconds=.796,selected_source_children_process_cpu_seconds=runs[0]['children_process_cpu_seconds'],
        selected_source_wall_seconds=runs[0]['wall_seconds'],helper10_mock_children_process_cpu_seconds=runs[1]['children_process_cpu_seconds'],
        helper10_mock_wall_seconds=runs[1]['wall_seconds'],helper_extra4_process_cpu_seconds=extra['process_cpu_seconds'],
        independent_final_byte_audit_process_cpu_seconds=audit['byte_audit_process_cpu_seconds'],
        owner52_metadata_verified_not_reexecuted_by_this_reviewer=True,
        owner14_budget_metadata_verified_not_reexecuted_by_this_reviewer=True,
        owner_tiny_context_cost=owner['validation_context_cost'],initial_fixture_failed_run_cpu_not_measured=True,
        cost_scope='These are actual tiny synthetic local CPU measurements only; no official-tokenizer or GPU performance claim.'),
    limitations=['Actual official Qwen tokenizer5120 compatibility/default flags and same-node preparation have not been executed by this reviewer.',
        'Actual zero-write 3B calibration, living production five, 6..200, checkpoints/dev/final/HF/cleanup acceptance remain false.',
        'The fixed trial negative-tail/repeat threshold is unaccepted for production; current human target remains full200.',
        'CPU synthetic tests cannot establish sampler/actor numerical probability agreement on GPU.',
        'Actor old/reference log-prob forwards are preceded by registry registration but do not themselves perform mu revalidation; actual update_policy does so before its first native forward.'],
    artifacts={name:digest(OUT/name) for name in evidence_names},**false_scope)
receipt_sha=write('RECEIPT.json',receipt)
print(json.dumps(dict(receipt_sha256=receipt_sha,byte_audit_sha256=digest(OUT/'BYTE_AUDIT.json'),helper_review_sha256=helper_sha,
    source_cpu_accepted=True,actual_gpu_ready=False)))
