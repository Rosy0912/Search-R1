"""Mirror compact current-chain status without double-crediting failed parent work."""
import argparse, json, shutil
from pathlib import Path

LEGACY_SOURCE = '0befbaf99fc44942105831b3fbec94e4bf1b8ccb'
NATIVE1_SOURCE = 'ba9f6627ec9a4e9749830d7b6acca53f6feec6b4'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('poll')
    parser.add_argument('--source', '--source-commit', dest='source')
    parser.add_argument('--source-checkout')
    parser.add_argument('--source-branch')
    args = parser.parse_args()
    poll_path = Path(args.poll); data = json.loads(poll_path.read_text()); progress = data['progress']
    if args.source and args.source != data['source_commit']:
        parser.error('source differs from observed poll identity; poll the requested source first')
    native_requested = data.get('native1_continuation_requested', data['source_commit'] == NATIVE1_SOURCE)
    receipt = data.get('native1_load_receipt')
    inherited = progress.get('inherited_native1')
    driver = (receipt or {}).get('driver') or {}
    workers = (receipt or {}).get('workers') or []
    loaded = None if native_requested and receipt is None else bool(
        inherited and receipt and receipt.get('status') == 'COLD_NESTED_NATIVE1_STATE_LOADED'
        and len(workers) == 8 and sorted(w.get('rank') for w in workers) == list(range(8))
        and all(w.get('status') == 'COLD_NESTED_NATIVE1_WORKER_LOADED'
                and w.get('actual_cold_native_state_equal') is True
                and w.get('actual_file_sha256_verified') is True for w in workers)
        and driver.get('actual_cold_driver_state_equal') is True
        and driver.get('actual_file_sha256_verified') is True)
    counters_known = not native_requested or loaded is True
    observed_completed = progress['actual_completed_outer_from_attempt_results']
    completed = observed_completed if counters_known else None
    inherited_completed = inherited['completed_outer'] if loaded is True else 0 if not native_requested else None
    child_completed = observed_completed - inherited_completed if counters_known else None
    totals = progress['completed_actual_optimizer_calls_per_rank'] if counters_known else None
    loaded_adam = driver.get('completed_optimizer_calls_per_rank') if loaded is True else None
    child_adam = {rank: count - loaded_adam[rank] for rank, count in totals.items()} if loaded_adam else totals
    state = dict(x.split('=', 1) for x in data['systemd'].splitlines() if '=' in x)
    result = data.get('run_result')
    failed = state.get('ActiveState') == 'failed' or bool(result and result.get('status') != 'ACTUAL_HUMAN_AUTHORIZED_DIRECT_FRESH200_TRAINING_COMPLETE') or bool(progress.get('errors'))
    stage = data['logs'].get('stage.log') or ''
    if failed:
        phase = 'REPAIR_REQUIRED'
    elif native_requested and loaded is not True:
        phase = 'NATIVE1_LOAD_UNCONFIRMED'
    elif native_requested and child_completed == 0:
        phase = 'NATIVE1_LOADED_CONTINUING'
    else:
        phase = 'TRAINING_RUNNING' if completed else 'FIRST_OUTER_UPDATING' if data.get('rollout_cost') else 'FIRST_ROLLOUT_ENTERED' if data.get('first_rollout_entered') or 'epoch 0, step 1' in stage else 'ACTOR_LOADING' if stage else 'LOCAL_RETRIEVER_LOADING' if data['logs'].get('retriever.log') is not None else 'SHARED_STARTUP_ADMISSION'
    provenance = (receipt or {}).get('provenance') or {}
    parent_failed = provenance.get('failed_parent_outer2') or (inherited or {}).get('failed_parent_outer2')
    if native_requested and parent_failed is None:
        # Known historical facts supplied by the mainline recovery handoff, not child credit.
        parent_failed = dict(parent_source_commit=LEGACY_SOURCE, actual_completed_outer_before_failure=2,
            actual_optimizer_calls_per_rank_before_failure={str(rank): 40 for rank in range(8)},
            failure_component='CPU_cgroup_OOM_during_native2_save', recovered_completed_outer=1,
            recovered_optimizer_calls_per_rank={str(rank): 20 for rank in range(8)},
            outer2_must_be_recomputed=True, failed_attempt_in_effective_history=False,
            evidence_source='mainline_recovery_handoff_20261014', load_receipt_confirmation_pending=True)
    checkout = args.source_checkout or data.get('source_checkout') or ('work/tree-speed-repair-20261008/nested-native1-recovery-source' if native_requested else 'work/tree-speed-repair-20261008/nested-branch-source')
    branch = args.source_branch or data.get('source_branch') or ('codex/tree-nested-native1-recovery-20261014' if native_requested else 'codex/tree-nested-branch-20261012')
    source_verified = data.get('source_remote_sha_verified')
    publication_evidence = None
    for publication_path in [poll_path.parent / 'NATIVE1_CODE_PUBLICATION.json', Path('outputs/tree-nested-branch-handoff-20261013/NATIVE1_CODE_PUBLICATION.json')]:
        if not publication_path.is_file():
            continue
        publication = json.loads(publication_path.read_text())
        if publication.get('source_commit') != data['source_commit']:
            continue
        files = publication.get('files') or {}
        source_verified = bool(publication.get('status') == 'COMMITTED_PUSHED_AND_REMOTE_SOURCE_READBACK_VERIFIED'
            and publication.get('remote_commit') == data['source_commit'] and files
            and all(record.get('remote_readback_equal') is True for record in files.values()))
        publication_evidence = {key: publication.get(key) for key in ('status', 'source_commit', 'backup_branch', 'remote_commit', 'verified_at')}
        publication_evidence['receipt_path'] = str(publication_path)
        break
    latest = data.get('development') or []
    initialization = ('same original fresh nested chain; cold native1 restored mother outer1/20 Adam per rank, driver/RNG/sampler and eight workers; child recomputes outer2' if loaded is True else 'same original fresh nested chain; cold native1 continuation requested from mother outer1/20 Adam per rank; actual loading unconfirmed') if native_requested else 'official Qwen/Qwen2.5-3B-Instruct aa8e725; no predecessor native/Adam/scheduler inheritance'
    status = dict(status=phase, phase=phase, method='nested_tree_grpo', source_commit=data['source_commit'], source_checkout=str(Path(checkout).resolve()), source_branch=branch,
        source_remote_sha_verified=source_verified, source_publication_evidence=publication_evidence, node='h100-10 / 10.33.0.18',
        active_unit=data['unit'], job_id=data['job_id'], active_training_job=data['job_id'], run_directory=data['run'], observed_epoch=data['epoch'], last_observed_epoch=data['epoch'],
        current_outer=completed, new_completed_outer=child_completed, effective_counters_known=counters_known, target_outer=200, training_target_outer=200,
        successful_adam_per_rank=totals, child_successful_adam_per_rank=child_adam, inherited_completed_outer=inherited_completed, inherited_adam_per_rank=loaded_adam,
        all_skipped_attempts=progress['all_skipped_attempts'], direct_rank_record_count=data['direct_rank_records'], last_rank_summaries=data['last_rank_summaries'], progress_errors=progress['errors'],
        latest_dev512=latest[-1] if latest else None, dev_score_updated=bool(latest), checkpoint=data['native_manifests'], latest_guard=data['latest_guard'], current_systemd=state,
        last_actual_poll=str(poll_path), actual_training_progress=progress, health='failed; save evidence and repair own task' if failed else 'native1 receipt pending; effective loaded counters unknown' if native_requested and loaded is not True else 'active; observe actual rollout/update progress',
        physical_gpus=8, slurm_reserved_gpus=0, resource_protocol='training_memory_headroom_shared_v1', new_training_submitted=True, initialization=initialization,
        experiment_origin='fresh', native1_continuation_requested=native_requested, native1_loaded=loaded if native_requested else False, native1_load_receipt=receipt,
        actual_cold_executor_equivalence=False if native_requested else None, exact_executor_resume=False,
        inherited_native1=inherited, historical_failed_parent_outer2=parent_failed, failed_parent_outer2_in_effective_history=False,
        requested_inherited_completed_outer=1 if native_requested else 0, requested_inherited_adam_per_rank={str(rank): 20 for rank in range(8)} if native_requested else None,
        design='A,A1,B,B1,B2 leaf quotas2,3; random A fork and true B->B1->B2 finite reward-difference backoff', global_train_questions=512, training_leaves=2560,
        mini_batch=128, actor_micro_global=16, old_ref_micro_global=32, ppo_opportunities_per_rank_per_outer=20, lr=1e-6, warmup_complete_outer=57,
        forward_dtype='bfloat16', master_adam_dtype='float32', initial_baseline_rerun=False, additional_reviews_requested=False,
        latest_human_preference='Use original18 assets; same-goal native1 continuation, retain failed outer2 costs, recompute child outer2; no redundant reviews',
        predecessor=dict(job_id=3360, completed_outer=200, final_evaluated_rows=51713, final_f1=.510726647820571, final_em=.44340881403128807, independent_final_acceptance=True, independent_acceptance_sha256='11330f3b73d34d837b2a986d8d877e09f9af65d17415260268365e72e82aad1e', status_file='outputs/tree-eval-resume-20261013/STATUS.json'), goal_complete=False)
    status.update(latest_rollout_cost=data.get('rollout_cost', []), output_metadata=data.get('output_metadata', []), first_rollout_entered=data.get('first_rollout_entered', False), recent_iteration_metrics=data.get('iteration_metrics', []))
    status['native_manifest_history'] = data.get('native_manifest_history')
    status['native_manifest_history_audit_error'] = (data.get('native_manifest_history') or {}).get('audit_error')
    report = ('Current nested Tree-GRPO experiment on original H10018.\n\n'
        + f'Phase: {phase}; effective completed outer {completed}/200; child completed outer {child_completed}; successful Adam writes per rank {totals}. Unit {data["unit"]}; source {data["source_commit"]}.\n\n'
        + initialization + '. Experiment origin remains fresh.\n\n'
        + ('The native1 receipt supplies eight worker and driver restoration evidence. Parent actual outer2/40 Adam per rank is retained as failed historical cost and excluded from effective child history; child outer2 must be recomputed. Missing receipt means actual loading and effective inherited counters remain unknown.\n\n' if native_requested else '')
        + 'Same 512x5, mini128, actor16/old-ref32, BF16 forward/FP32 master+Adam, lr1e-6/warm57, existing protections and lengths. Continue the same 200-outer goal; no initial baseline repeat or additional reviews.\n\n'
        + 'Historical failed parent cost: ' + json.dumps(parent_failed) + '\n\n'
        + 'Latest actual rollout cost: ' + json.dumps(status['latest_rollout_cost']) + '\n\n'
        + 'Predecessor3360 completed200/51713 and independent acceptance. Final F1=51.072665%, EM=44.340881%; these are predecessor results, not this experiment evaluation. Detailed old result stays outputs/tree-eval-resume-20261013/FINAL_RESULT.md.\n')
    for out in [Path('outputs'), Path('outputs/tree-nested-branch')]:
        out.mkdir(parents=True, exist_ok=True)
        (out / 'STATUS.json').write_text(json.dumps(status, indent=2) + '\n')
        (out / 'REPORT.md').write_text(report)
    for base in [Path('work/tree-grpo/reproduction/baselines/tree-validation-20261002'), Path('/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/tree-grpo')]:
        destination = base / 'tree-nested-branch'
        if destination.is_dir():
            for name in ['STATUS.json', 'REPORT.md']:
                shutil.copyfile(Path('outputs/tree-nested-branch') / name, destination / name)
    print(json.dumps(dict(phase=phase, outer=completed, child_outer=child_completed, native1_loaded=status['native1_loaded'], adam=totals)))


if __name__ == '__main__':
    main()
