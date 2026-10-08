"""Local review-only evidence writer. Never imports the training project."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
SOURCE = Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/work/tree-behavior-mu-ppo')
OWNER = SOURCE.parent/'tree-fresh200-source-owner'
WRAPPER = SOURCE.parent/'tree-fresh200-mu-runtime-v5'
HEAD = '8a3bc24db05ee1fd67660e2f89088d6f56bc0875'
BASE = '1e06c6aa6f54eee5de8e0a29dd50efea76111c10'

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def artifact(path):
    path = Path(path)
    return dict(path=str(path), sha256=sha(path.read_bytes()), bytes=path.stat().st_size)

def git(*args):
    return subprocess.check_output(['git','-C',str(SOURCE),*args])

def write_exclusive(name, value):
    with (ROOT/name).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')

owner = json.loads((OWNER/'REPAIRED_SOURCE_INTERFACE_CPU_RECEIPT.json').read_bytes())
extras = [
    'search_r1/baselines/gradient_norm.py','search_r1/baselines/tree_rank_audit.py',
    'search_r1/baselines/policy_safety.py','search_r1/baselines/objectives.py',
    'search_r1/baselines/training_state.py','search_r1/baselines/controller.py',
    'search_r1/baselines/advantages.py','search_r1/baselines/local_runtime.py',
    'search_r1/baselines/holdout.py','search_r1/baselines/tree_audit.py',
    'verl/protocol.py','verl/single_controller/base/decorator.py',
    'verl/workers/sharding_manager/fsdp_ulysses.py','verl/workers/rollout/vllm_rollout/vllm_rollout.py',
    'verl/trainer/ppo/core_algos.py','verl/utils/torch_functional.py',
    'verl/trainer/config/ppo_tree_instruct300.yaml',
    'verl/third_party/vllm/vllm_v_0_6_3/llm_engine_sp.py',
    'test_partial_worker_state_cpu.py','test_fresh_calibration_worker_cpu.py',
    'test_tree_fresh200_cpu.py','test_tree_partial_driver_cpu.py',
    'test_tree_raw_tokens_cpu.py','test_tree_behavior_mu_data_cpu.py',
]
records = {}
for relative in sorted(set(owner['source_files']) | set(extras)):
    frozen = ROOT/'immutable-source-8a'/relative
    raw = frozen.read_bytes()
    head_raw = git('show',HEAD+':'+relative)
    current = SOURCE/relative
    records[relative] = dict(sha256=sha(raw), matches_git_head=raw==head_raw,
        bytes=len(raw), immutable_path=str(frozen), original_source_path=str(current),
        initial_clean_head_binding=True, current_worktree_sha256=sha(current.read_bytes()),
        current_worktree_matches_audited_head=current.read_bytes()==raw,
        owner_interface_expected_sha256=owner['source_files'].get(relative),
        owner_interface_hash_matches=(sha(raw)==owner['source_files'][relative] if relative in owner['source_files'] else None))

wrapper_audit_path = ROOT/'WRAPPER_V5_BYTE_AUDIT.json'
wrapper_audit = json.loads(wrapper_audit_path.read_bytes())
assert wrapper_audit['wrapper_count']==15 and len(wrapper_audit['wrapper_files'])==15
assert all(v['matches'] for v in wrapper_audit['wrapper_files'].values())
source_status = git('status','--porcelain').decode()
current_head = git('rev-parse','HEAD').decode().strip()
cpu = [
    dict(name='independent_31_initialization_five_leaf_partial_driver', count=31, exit_code=0,
        seconds=13.505, log=artifact(ROOT/'SOURCE_31_CPU_LOG.txt'), source_head=HEAD),
    dict(name='independent_9_partial_worker_including_6_gloo_scenarios', count=9, exit_code=0,
        seconds=37.788, log=artifact(ROOT/'SOURCE_PARTIAL_CPU_LOG.txt'), source_head=HEAD),
    dict(name='independent_33_raw_and_mu', count=33, exit_code=1, errors=1,
        seconds=.167, log=artifact(ROOT/'SOURCE_RAW_MU_CPU_LOG.txt'), source_head=HEAD,
        failed_test='test_actual_tree_round_inherits_sample_ids_and_equal_leaf_weights_unchanged',
        failure_kind='test fixture omits required roots; no runtime quota failure established'),
    dict(name='F04_actual_methods_tiny_adam_real_2rank_gloo_counterexample', exit_code=0,
        log=artifact(ROOT/'F04_ACTUAL_CPU_LOG.txt'), script=artifact(ROOT/'reproduce_f04_cpu.py'),
        per_rank=[artifact(ROOT/f'F04_RANK_{rank}.json') for rank in range(2)],
        scope='Unmodified frozen source methods; synthetic FSDP layout and lazy-init bypass; not FSDP/GPU training'),
]
audit = dict(schema_version=1, created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    status='FROZEN_BYTES_AUDITED_IMPLEMENTATION_REJECTED', git_head=HEAD, base_commit=BASE,
    reviewer_context=dict(fork_turns='none',inherited_parent_conversation=False),
    source_initial_clean=True, source_files=records,
    source_interface_keys=list(owner['source_files']),
    all_audited_source_files_match_git_head=all(v['matches_git_head'] for v in records.values()),
    all_owner_interface_hashes_match=all(records[k]['owner_interface_hash_matches'] for k in owner['source_files']),
    current_worktree=dict(git_head=current_head, clean=not source_status, status_porcelain=source_status,
        accepted=False, newer_or_mutated_bytes_are_not_accepted=True),
    wrapper_files=wrapper_audit['wrapper_files'], wrapper_file_count=15,
    wrapper_inventory=wrapper_audit['manifest'], wrapper_cpu_receipt=wrapper_audit['cpu_receipt'],
    wrapper_subreview_byte_audit=artifact(wrapper_audit_path),
    owner_source_interface_receipt=artifact(OWNER/'REPAIRED_SOURCE_INTERFACE_CPU_RECEIPT.json'),
    owner_source_files=artifact(OWNER/'REPAIRED_SOURCE_FILES.json'),
    independent_cpu_runs=cpu, additive_success_total=None,
    actual_goal_optimizer_writes=0,actual_goal_complete_outer_updates=0,
    actual_remote_calls=0,actual_gpu_calls=0,actual_slurm_submissions=0,
    model_or_data_assets_loaded=False,
    limits=['Byte identity is not training success','F04 reproduced on already synchronized refusal',
        'Raw-token full suite failure retained','Wrapper and helper stages are separate',
        'No real Hydra/assets/tokenizer/native effective controls/FSDP/NCCL/Ray acceptance'])
write_exclusive('BYTE_AUDIT.json',audit)
artifacts = {name:artifact(ROOT/name) for name in
    ('REVIEW.md','REQUIREMENTS_AUDIT.md','BYTE_AUDIT.json','WRAPPER_V5_REVIEW.md','WRAPPER_V5_BYTE_AUDIT.json')}
receipt = dict(schema_version=1,status='REJECTED_SOURCE_AND_WRAPPER_NO_EXECUTION_RELEASE',
    created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    fork_turns='none',inherited_parent_conversation=False,
    source=dict(audited_clean_head=HEAD,base_commit=BASE,initial_clean=True,
        source_interface_reviewed=True,source_interface_accepted=False,
        current_worktree_head=current_head,current_worktree_clean=not source_status,
        current_worktree_accepted=False),
    source_interface_reviewed=True,source_interface_accepted=False,
    stages=dict(preparation=dict(review_completed=True,accepted=False,reason='F04 and wrapper resource gate defect'),
        fresh_mu_calibration=dict(source_review_completed=True,source_review_accepted=False,
            actual_gpu_observation_completed=False,ready_for_gpu_dispatch=False),
        fresh_mu_train200=dict(source_review_completed=True,source_review_accepted=False,
            implemented=False,production_thresholds_accepted=False,actual_training_accepted=False),
        helper_dispatch_v2=dict(review_completed=False,accepted=False,status='PENDING_FROZEN_INVENTORY_SEPARATE_STAGE')),
    execution_scope=dict(same_h100_9_cpu_only_preparation_allowed=False,
        same_h100_9_zero_write_gpu_calibration_allowed=False,training_allowed=False,
        cross_cluster_allowed=False,foreign_process_kill_allowed=False,automatic_rollback_allowed=False),
    artifacts=artifacts,
    blocking_findings=['F04 synchronized nonfinite gradient refusal loses worker partial state',
        'W02 inherited optimization removes dual15% foreign utilization gate',
        'P00 fresh_mu_train200 unimplemented and production thresholds null'],
    secondary_findings=['F05 raw-token full CPU fixture omits roots',
        'W01 launcher assert-only receipt/pins gate can be deleted under optimization; downstream explicit stage/source checks still apply'],
    actual_goal_complete_outers=0,actual_goal_optimizer_writes_per_rank=0,
    actual_remote_executed=False,actual_gpu_executed=False,
    original_sampler_actor_probability_gate_accepted=False,
    sampled_mu_calibration_accepted=False,production_thresholds_accepted=False,
    actual_3b_training_accepted=False,goal_complete=False,
    automatic_partial_resume_supported=False,
    limitations=['A source-owner claim or passing CPU fixture is not actual execution',
        'New repaired source or wrapper requires a separate freeze and independent review',
        'Materialized h100-9 bytes, actual Hydra, full assets and official tokenizer preflight remain unreviewed'])
write_exclusive('RECEIPT.json',receipt)
print(json.dumps(dict(receipt_sha256=sha((ROOT/'RECEIPT.json').read_bytes()),
    byte_audit_sha256=sha((ROOT/'BYTE_AUDIT.json').read_bytes()),source_files=len(records),wrapper_files=15,
    audited_head=HEAD,current_head=current_head,current_clean=not source_status,status=receipt['status'])))
