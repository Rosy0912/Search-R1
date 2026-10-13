"""Mirror compact current-chain status without double-crediting failed parent work."""
import argparse, json, shutil
from pathlib import Path

LEGACY_SOURCE = '0befbaf99fc44942105831b3fbec94e4bf1b8ccb'
NATIVE1_SOURCE = 'ba9f6627ec9a4e9749830d7b6acca53f6feec6b4'
NATIVE5_SOURCE = 'eb6a69e47163b179b20b66cf4992c0d5e152ea6c'


def development_status(data, previous_status):
    same_run = (previous_status.get('source_commit') == data['source_commit']
        and previous_status.get('run_directory') == data['run'])
    development = data.get('development') or []
    observed_latest = development[-1] if development else None
    if observed_latest is not None and (observed_latest.get('status') != 'VALIDATION_COST_COMPLETE'
            or observed_latest.get('rows') != 512 or (observed_latest.get('metrics') or {}).get('val/qa_count/all') != 512):
        observed_latest = None
    previous_latest = previous_status.get('latest_dev512') if same_run else None
    latest = observed_latest if observed_latest is not None else previous_latest if previous_latest is not None else data.get('inherited_dev512')
    same_evaluation = previous_latest is not None and observed_latest is not None and all(
        observed_latest.get(key) == previous_latest.get(key) for key in ('path', 'outer', 'phase', 'status', 'rows', 'metrics'))
    if same_evaluation and observed_latest.get('ledger_sha256') and previous_latest.get('ledger_sha256'):
        same_evaluation = observed_latest['ledger_sha256'] == previous_latest['ledger_sha256']
    updated = bool(observed_latest is not None
        and (not same_run or not same_evaluation))
    return latest, updated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('poll')
    parser.add_argument('--source', '--source-commit', dest='source')
    parser.add_argument('--source-checkout')
    parser.add_argument('--source-branch')
    parser.add_argument('--source-publication')
    args = parser.parse_args()
    poll_path = Path(args.poll); data = json.loads(poll_path.read_text()); progress = data['progress']
    if args.source and args.source != data['source_commit']:
        parser.error('source differs from observed poll identity; poll the requested source first')
    native_requested = data.get('native_continuation_requested', data.get('native1_continuation_requested', data['source_commit'] in (NATIVE1_SOURCE, NATIVE5_SOURCE)))
    restore_step = data.get('native_restore_step') or (5 if data['source_commit'] == NATIVE5_SOURCE else 1 if native_requested else None)
    receipt = data.get('native_load_receipt') or data.get('native1_load_receipt')
    inherited = progress.get('inherited_native_restore') or progress.get('inherited_native1')
    driver = (receipt or {}).get('driver') or {}
    workers = (receipt or {}).get('workers') or []
    loaded = None if native_requested and receipt is None else bool(
        inherited and receipt and receipt.get('status') == f'COLD_NESTED_NATIVE{restore_step}_STATE_LOADED'
        and driver.get('completed_outer_updates') == inherited.get('completed_outer') == restore_step
        and len(workers) == 8 and sorted(w.get('rank') for w in workers) == list(range(8))
        and all(w.get('status') == f'COLD_NESTED_NATIVE{restore_step}_WORKER_LOADED'
                and w.get('world_size') == 8 and w.get('child_code_commit') == data['source_commit']
                and w.get('scheduler_step') == restore_step
                and w.get('actual_total_optimizer_calls') == (driver.get('completed_optimizer_calls_per_rank') or {}).get(str(w.get('rank')))
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
        phase = 'NATIVE1_LOAD_UNCONFIRMED' if restore_step == 1 else 'NATIVE_RESTORE_UNCONFIRMED'
    elif native_requested and child_completed == 0:
        phase = 'NATIVE1_LOADED_CONTINUING' if restore_step == 1 else 'NATIVE_RESTORE_LOADED_CONTINUING'
    else:
        phase = 'TRAINING_RUNNING' if completed else 'FIRST_OUTER_UPDATING' if data.get('rollout_cost') else 'FIRST_ROLLOUT_ENTERED' if data.get('first_rollout_entered') or 'epoch 0, step 1' in stage else 'ACTOR_LOADING' if stage else 'LOCAL_RETRIEVER_LOADING' if data['logs'].get('retriever.log') is not None else 'SHARED_STARTUP_ADMISSION'
    provenance = (receipt or {}).get('provenance') or {}
    boundary = {}
    for boundary_path in [poll_path.parent / 'FAULT6_AND_NATIVE5_RECOVERY_BOUNDARY.json', Path('outputs/tree-nested-branch-handoff-20261013/FAULT6_AND_NATIVE5_RECOVERY_BOUNDARY.json')]:
        if boundary_path.is_file():
            candidate = json.loads(boundary_path.read_text())
            if restore_step == 5 or candidate.get('failed_source_commit') == data['source_commit']:
                boundary = candidate
                break
    parent_failed = provenance.get('failed_parent_outer2') or (inherited or {}).get('failed_parent_outer2')
    parent_failed = parent_failed or boundary.get('historical_failed_parent_outer2')
    parent_failed6 = provenance.get('failed_parent_outer6') or (inherited or {}).get('failed_parent_outer6') or boundary.get('failure')
    if native_requested and restore_step == 1 and parent_failed is None:
        # Known historical facts supplied by the mainline recovery handoff, not child credit.
        parent_failed = dict(parent_source_commit=LEGACY_SOURCE, actual_completed_outer_before_failure=2,
            actual_optimizer_calls_per_rank_before_failure={str(rank): 40 for rank in range(8)},
            failure_component='CPU_cgroup_OOM_during_native2_save', recovered_completed_outer=1,
            recovered_optimizer_calls_per_rank={str(rank): 20 for rank in range(8)},
            outer2_must_be_recomputed=True, failed_attempt_in_effective_history=False,
            evidence_source='mainline_recovery_handoff_20261014', load_receipt_confirmation_pending=True)
    continuation_name = 'native1' if restore_step == 1 else 'native5' if restore_step == 5 else 'native'
    checkout = args.source_checkout or data.get('source_checkout') or ('work/tree-speed-repair-20261008/nested-'+continuation_name+'-recovery-source' if native_requested else 'work/tree-speed-repair-20261008/nested-branch-source')
    branch = args.source_branch or data.get('source_branch') or ('codex/tree-nested-'+continuation_name+'-recovery-20261014' if native_requested else 'codex/tree-nested-branch-20261012')
    source_verified = data.get('source_remote_sha_verified')
    publication_evidence = None
    publication_paths = [Path(args.source_publication)] if args.source_publication else []
    publication_paths += [base / name for base in [poll_path.parent, Path('outputs/tree-nested-branch-handoff-20261013')]
        for name in ['NATIVE5_CODE_PUBLICATION.json', 'NATIVE_CODE_PUBLICATION.json', 'NATIVE1_CODE_PUBLICATION.json']]
    for publication_path in publication_paths:
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
    previous_status_path = Path('outputs/STATUS.json')
    previous_status = json.loads(previous_status_path.read_text()) if previous_status_path.is_file() else {}
    inherited_dev = provenance.get('parent_latest_dev512') or (boundary.get('latest_completed_dev512') if restore_step == 5 else None)
    latest, dev_score_updated = development_status(dict(data, inherited_dev512=inherited_dev), previous_status)
    observed_dev = (data.get('development') or [None])[-1]
    if observed_dev is not None and latest == observed_dev:
        latest_dev_source = data['source_commit']
    elif previous_status.get('source_commit') == data['source_commit'] and previous_status.get('run_directory') == data['run'] and previous_status.get('latest_dev512') is not None:
        latest_dev_source = previous_status.get('latest_dev512_source_commit', previous_status.get('source_commit'))
    else:
        latest_dev_source = ((provenance.get('parent_identity') or {}).get('code_commit') or boundary.get('failed_source_commit')) if inherited_dev else None
    ancestor_segments = provenance.get('effective_ancestor_segments') or ((boundary.get('completed_recovery_boundary') or {}).get('effective_origin_segments') if restore_step == 5 else None)
    if ancestor_segments is None and loaded is True:
        identity = provenance.get('parent_identity') or {}
        ancestor_segments = [dict(code_commit=identity.get('code_commit'), run_id=identity.get('run_id'), config_sha256=identity.get('config_sha256'), first_outer=1, last_outer=inherited_completed)]
    effective_segments = list(ancestor_segments or [])
    if counters_known and child_completed:
        effective_segments.append(dict(source_run=data['run'], run_id=Path(data['run']).name, code_commit=data['source_commit'], first_outer=(inherited_completed or 0)+1, last_outer=completed))
    requested_calls = {str(rank): 20 if restore_step == 1 else 100 for rank in range(8)} if restore_step in (1, 5) else None
    confirmed_boundary = boundary.get('completed_recovery_boundary') or {}
    confirmed_outer = completed if counters_known else confirmed_boundary.get('completed_outer')
    confirmed_adam = totals if counters_known else confirmed_boundary.get('actual_cumulative_optimizer_calls_per_rank')
    initialization = (f'same original fresh nested chain; cold native{restore_step} restored completed outer {inherited_completed}, actual Adam calls per rank {loaded_adam}, driver/RNG/sampler and eight workers; child continues outer {inherited_completed+1}' if loaded is True else f'same original fresh nested chain; native{restore_step} continuation requested; actual loading and inherited counters unconfirmed') if native_requested else 'official Qwen/Qwen2.5-3B-Instruct aa8e725; no predecessor native/Adam/scheduler inheritance'
    status = dict(status=phase, phase=phase, method='nested_tree_grpo', source_commit=data['source_commit'], source_checkout=str(Path(checkout).resolve()), source_branch=branch,
        source_remote_sha_verified=source_verified, source_publication_evidence=publication_evidence, node='h100-10 / 10.33.0.18',
        active_unit=data['unit'], job_id=data['job_id'], active_training_job=data['job_id'], run_directory=data['run'], observed_epoch=data['epoch'], last_observed_epoch=data['epoch'],
        current_outer=confirmed_outer, new_completed_outer=child_completed, effective_counters_known=counters_known, actual_child_counters_known=counters_known, target_outer=200, training_target_outer=200,
        successful_adam_per_rank=confirmed_adam, successful_adam_per_rank_scope='actual current segment plus inherited history' if counters_known else 'last confirmed durable chain; current restore pending',
        actual_current_restore_outer=completed, actual_current_restore_adam_per_rank=totals,
        child_successful_adam_per_rank=child_adam, inherited_completed_outer=inherited_completed, inherited_adam_per_rank=loaded_adam,
        all_skipped_attempts=progress['all_skipped_attempts'], direct_rank_record_count=data['direct_rank_records'], last_rank_summaries=data['last_rank_summaries'], progress_errors=progress['errors'],
        latest_dev512=latest, latest_dev512_source_commit=latest_dev_source, dev_score_updated=dev_score_updated, checkpoint=data['native_manifests'], latest_guard=data['latest_guard'], current_systemd=state,
        last_actual_poll=str(poll_path), actual_training_progress=progress, health='failed; preserve actual durable boundary and continue same goal recovery' if failed else 'native restore receipt pending; effective loaded counters unknown' if native_requested and loaded is not True else 'active; observe actual rollout/update progress',
        physical_gpus=8, slurm_reserved_gpus=0, resource_protocol='training_memory_headroom_shared_v1', new_training_submitted=True, initialization=initialization,
        experiment_origin='fresh', native_continuation_requested=native_requested, native_restore_step=restore_step, native_loaded=loaded if native_requested else False, native_load_receipt=receipt,
        native1_continuation_requested=native_requested and restore_step == 1, native1_loaded=loaded if native_requested and restore_step == 1 else False, native1_load_receipt=receipt if restore_step == 1 else None,
        actual_cold_executor_equivalence=False if native_requested else None, exact_executor_resume=False,
        inherited_native_restore=inherited, inherited_native1=inherited if restore_step == 1 else None, effective_origin_segments=effective_segments,
        historical_failed_parent_outer2=parent_failed, failed_parent_outer2_in_effective_history=False,
        historical_failed_parent_outer6=parent_failed6, failed_parent_outer6_in_effective_history=False,
        fault6_native5_recovery_boundary=boundary or None,
        last_durable_completed_outer=(boundary.get('completed_recovery_boundary') or {}).get('completed_outer'),
        last_confirmed_chain_outer=confirmed_outer, last_confirmed_chain_adam_per_rank=confirmed_adam,
        failed_unrecorded_partial_optimizer_writes_unknown=bool((result or {}).get('failed_unrecorded_partial_optimizer_writes_unknown') or (parent_failed6 or {}).get('failed_unrecorded_partial_optimizer_writes_unknown')),
        requested_inherited_completed_outer=restore_step if native_requested else 0, requested_inherited_adam_per_rank=requested_calls,
        design='A,A1,B,B1,B2 leaf quotas2,3; random A fork and true B->B1->B2 finite reward-difference backoff', global_train_questions=512, training_leaves=2560,
        mini_batch=128, actor_micro_global=16, old_ref_micro_global=32, ppo_opportunities_per_rank_per_outer=20, lr=1e-6, warmup_complete_outer=57,
        forward_dtype='bfloat16', master_adam_dtype='float32', initial_baseline_rerun=False, additional_reviews_requested=False,
        latest_human_preference='Use original18 assets; continue same fresh-origin 200-outer goal from actual durable native boundary; preserve failed attempt costs separately',
        predecessor=dict(job_id=3360, completed_outer=200, final_evaluated_rows=51713, final_f1=.510726647820571, final_em=.44340881403128807, independent_final_acceptance=True, independent_acceptance_sha256='11330f3b73d34d837b2a986d8d877e09f9af65d17415260268365e72e82aad1e', status_file='outputs/tree-eval-resume-20261013/STATUS.json'), goal_complete=False)
    status.update(latest_rollout_cost=data.get('rollout_cost', []), output_metadata=data.get('output_metadata', []), first_rollout_entered=data.get('first_rollout_entered', False), recent_iteration_metrics=data.get('iteration_metrics', []))
    status['native_manifest_history'] = data.get('native_manifest_history')
    status['native_manifest_history_audit_error'] = (data.get('native_manifest_history') or {}).get('audit_error')
    report = ('Current nested Tree-GRPO experiment on original H10018.\n\n'
        + f'Phase: {phase}; last confirmed chain outer {confirmed_outer}/200; current restore actual outer {completed}; child completed outer {child_completed}; confirmed chain Adam per rank {confirmed_adam}. Unit {data["unit"]}; source {data["source_commit"]}.\n\n'
        + initialization + '. Experiment origin remains fresh.\n\n'
        + ('Native restore requires actual eight worker and driver restoration evidence. Failed mother2 and parent6 remain historical costs without effective credit; missing receipt leaves current inherited counters unknown. Prior accepted origin segments stay recorded separately.\n\n' if native_requested else '')
        + 'Same 512x5, mini128, actor16/old-ref32, BF16 forward/FP32 master+Adam, lr1e-6/warm57, existing protections and lengths. Continue the same 200-outer goal; no initial baseline repeat or additional reviews.\n\n'
        + 'Historical failed parent cost: ' + json.dumps(parent_failed) + '\n\n'
        + 'Historical failed parent outer6: ' + json.dumps(parent_failed6) + '\n\n'
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
    print(json.dumps(dict(phase=phase, outer=confirmed_outer, child_outer=child_completed, actual_restore_outer=completed, native_loaded=status['native_loaded'], native1_loaded=status['native1_loaded'], adam=confirmed_adam)))


if __name__ == '__main__':
    main()
