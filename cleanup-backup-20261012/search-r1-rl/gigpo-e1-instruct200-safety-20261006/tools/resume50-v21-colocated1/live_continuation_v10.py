"""Human-authorized same-process200; diagnostics do not block healthy training."""
import copy
import hashlib
import json
import os
from pathlib import Path

from calibration_contract_v1 import LIVE_SCOPE, LIVE_GOAL_UPDATED_AT, require, initial_fields, summarize_guards
from calibration_state_v1 import state_digest
from continuation_timing_v10 import observe_time, publish_timing, record_duration


def next_counted_outer(ledger, payload):
    from search_r1.baselines.fresh200 import validate_attempt_ledger
    validate_attempt_ledger(ledger)
    completed = ledger['completed_outer_updates']
    require(ledger['active_attempt'] is None and type(completed) is int and 0 <= completed <= 200,
            'Cannot fetch with an active attempt or beyond200')
    require(type(payload['config_epoch']) is int and payload['config_epoch'] in (0, 1)
            and type(payload['live_goal_updated_at']) is int and payload['live_goal_updated_at'] == LIVE_GOAL_UPDATED_AT,
            'Successor loop lost the actual live goal')
    if completed == 200:
        require(payload['human_nonblocking_diagnostics'] is True
                and payload['config_epoch'] == 0 and payload['phase'] == 'C0_CALIBRATION',
                'Final200 lost the bound human nonblocking execution scope')
        return None
    require(payload['human_nonblocking_diagnostics'] is True
            and payload['config_epoch'] == 0 and payload['phase'] == 'C0_CALIBRATION',
            'Fetch requires the unchanged bound human nonblocking safety configuration')
    return completed + 1


def validate_completed5_transition(ledger, review, payload, worker_rows):
    from search_r1.baselines.fresh200 import validate_attempt_ledger
    from search_r1.baselines.fresh200_live import validate_admission
    validate_attempt_ledger(ledger); validate_admission(payload)
    require(ledger['active_attempt'] is None and ledger['completed_outer_updates'] == 5
            and review['fresh200_attempt_ledger'] == ledger and payload['config_epoch'] == 1,
            'Only this retained, idle completed5 ledger may admit C1')
    expected = sorted(review['checkpoint_pause_identity'], key=lambda row: row['rank'])
    require(len(worker_rows) == 8 and {row['rank'] for row in worker_rows} == set(range(8))
            and sorted(worker_rows, key=lambda row: row['rank']) == expected,
            'First5 approved model/Adam/RNG/worker PID identity changed before C1')


def immutable_json(path, value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    immutable_bytes(path, raw)
    return dict(path=str(Path(path).resolve()), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def immutable_bytes(path, raw):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        with os.fdopen(fd, 'wb', closefd=False) as stream:
            stream.write(raw); stream.flush(); os.fsync(fd)
    finally:
        os.close(fd)


def quality_report(metrics, baseline_metrics):
    # The human goal requires reporting degradation while continuing safe200.
    observed, baseline = metrics['val/f1/all'], baseline_metrics['val/f1/all']
    return dict(observed_f1=observed, baseline_f1=baseline,
        below_half_baseline=observed < baseline / 2, report_only=True,
        optimizer_or_outer_rejected_by_quality=False)


def audit_attempts(trainer, run):
    from calibration_audit_v1 import validate_calibration_update, validate_attempt_numeric
    from safety_audit_v1 import validate_guard
    from search_r1.baselines.fresh200_live import read_bound_json_lines
    ledger = copy.deepcopy(trainer.fresh200_attempt_ledger)
    rank_records, checks = [], []
    for rank in range(8):
        guard_path = run / f'audit/policy-safety/rank-{rank}.jsonl'
        raw = guard_path.read_bytes()
        rows = read_bound_json_lines(dict(path=str(guard_path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
        rank_records.append(rows)
        actors = [json.loads(line) for line in (run / f'audit/{rank}.jsonl').read_text().splitlines()
                  if json.loads(line)['kind'] in ('actor_outer', 'actor_attempt_incomplete')]
        require([row['attempt_id'] for row in actors] == [row['attempt_id'] for row in ledger['history']],
                'Missing original accepted/rejected attempt audit')
        counted = [row['outer_step'] for row in actors if row['kind'] == 'actor_outer']
        require(counted == list(range(1, ledger['completed_outer_updates'] + 1)), 'Counted actor audit labels differ')
        numeric = [json.loads(line) for line in (run / f'audit/numeric/micro-{rank}.jsonl').read_text().splitlines()]
        require(len(numeric) == sum(len(row['micro_stats']) for row in actors), 'Numeric observation coverage differs')
        checked = []
        for attempt, row in zip(ledger['history'], actors):
            receipt = next(value for value in attempt['rank_receipts'] if value['rank'] == rank)
            parent = getattr(trainer, 'resume50_parent', None) if attempt['completed_before'] < 50 else None
            value = validate_calibration_update(row, parent['source_commit'] if parent else os.environ['SEARCH_R1_BASELINE_COMMIT'],
                attempt['config_sha256'], Path(parent['run']).name if parent else os.environ['SEARCH_R1_BASELINE_RUN_ID'], rank,
                attempt['completed_before'] + 1, receipt)
            value['numeric'] = validate_attempt_numeric([entry for entry in numeric
                if entry['attempt_id'] == attempt['attempt_id']], row, receipt)
            checked.append(value)
        checks.append(checked)
        attempts = {attempt['attempt_id']: attempt for attempt in ledger['history']}
        for row in rows:
            attempt = attempts[row['attempt_id']]
            require(row['config_sha256'] == attempt['config_sha256'], 'Guard retroactively relabeled with C1 config')
            if attempt['config_epoch'] == 0:
                parent = getattr(trainer, 'resume50_parent', None) if attempt['completed_before'] < 50 else None
                validate_guard(row, 'calibration', parent['parent_binding_sha256'] if parent else os.environ['GIGPO_POLICY_NATIVE_BINDING_SHA256'])
            else:
                validate_guard(row, 'production', calibration_sha=trainer.fresh200_live_admission['policy_review_record']['sha256'])
    return dict(rank_records=rank_records, checks=checks, observations=summarize_guards(rank_records, ledger))


def prepare_first5_evidence(trainer, run):
    from search_r1.baselines.fresh200_live import digest, driver_admission_state
    from calibration_driver_v1 import json_save
    require(trainer.actual_outer_updates == trainer.global_steps == 5, 'First5 evidence requires actual completed5')
    checked = audit_attempts(trainer, run)
    records = []
    for rank in range(8):
        source = run / f'audit/policy-safety/rank-{rank}.jsonl'
        raw = source.read_bytes()
        target = run / f'first5-C0-evidence/guard-rank-{rank}.jsonl'
        immutable_bytes(target, raw)
        records.append(dict(rank=rank, path=str(target), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
    state = driver_admission_state(trainer)
    evidence = dict(schema='ACTUAL_RETAINED_FIRST5_C0_TAIL_OBSERVATIONS_V1', scope=LIVE_SCOPE,
        job_id=os.environ['SLURM_JOB_ID'], run_id=os.environ['SEARCH_R1_BASELINE_RUN_ID'],
        source_commit=os.environ['SEARCH_R1_BASELINE_COMMIT'], config_sha256=state['payload']['config_record']['sha256'],
        completed_outer_updates=5, actual_attempt_ledger_digest=digest(trainer.fresh200_attempt_ledger),
        actual_attempt_ledger=copy.deepcopy(trainer.fresh200_attempt_ledger),
        role_admissions=state['role_admissions'], role_admissions_digest=digest(state['role_admissions']),
        rank_guard_records=records, observations=checked['observations'],
        tail_selected=False, production_READY=False, first5_count_toward_goal=True)
    record = immutable_json(run / 'first5-C0-evidence/actual-tail-evidence.json', evidence)
    json_save(run / 'first5-C0-audit.json', dict(scope=LIVE_SCOPE, checks=checked['checks'], evidence=record))
    return record


def driver_training_projection(state):
    value = copy.deepcopy(state)
    value.pop('fresh200_live_admission')
    value['config']['actor_rollout_ref']['actor'].pop('policy_safety')
    return value


def install_production_at5(trainer, run, evidence_record):
    from omegaconf import OmegaConf
    from calibration_driver_v1 import driver_complete_pause_state, json_save, reload_saved_complete_state
    from search_r1.baselines.fresh200_live import (digest, driver_admission_state, read_bound_json,
        existing_record, validate_config_delta)
    from search_r1.baselines.instruct300 import check_manifest, equal_state
    c0 = driver_admission_state(trainer)
    before = driver_complete_pause_state(trainer)
    manifest = copy.deepcopy(trainer.instruct_manifest); check_manifest(manifest)
    approval = read_bound_json(existing_record(trainer.config.trainer.instruct300.approval_path))
    review_record = existing_record(approval['review_path'], approval['review_sha256'])
    review = read_bound_json(review_record)
    require(review['actual_tail_evidence_record'] == evidence_record
            and review['role_admissions_digest'] == digest(c0['role_admissions']), 'Policy review lacks actual all8 actor/reference C0 state')
    realized = copy.deepcopy(c0['resolved_config'])
    policy = realized['actor_rollout_ref']['actor']['policy_safety']
    policy.update(mode='production', negative_tail_limit=review['production_policy']['negative_tail_limit'],
        calibration_evidence_path=review_record['path'], calibration_evidence_sha256=review_record['sha256'])
    validate_config_delta(c0['resolved_config'], realized)
    config_record = immutable_json(run / 'resolved-config-C1.json', realized)
    payload = copy.deepcopy(c0['payload'])
    payload.update(phase='C1_PRODUCTION', config_epoch=1, config_record=config_record,
        policy=copy.deepcopy(policy), policy_review_record=review_record,
        first5_manifest_record={key: manifest[key] for key in ('path', 'bytes', 'sha256')},
        probability_approval_record=existing_record(run / 'probability-identity-approval.json'),
        parent_admission_sha256=digest(c0['payload']))
    validate_completed5_transition(trainer.fresh200_attempt_ledger, review, payload,
                                  trainer.actor_rollout_wg.checkpoint_pause_identity())
    admission_record = immutable_json(run / 'production-admission-C1.json', payload)
    installed = {}
    for role, group in (('actor', trainer.actor_rollout_wg), ('reference', trainer.ref_policy_wg)):
        rows = group.calibration_install_live_admission(payload)
        old = {row['rank']: row for row in c0['role_admissions'][role]}
        require(len(rows) == 8 and {row['rank'] for row in rows} == set(range(8))
                and all(row['role'] == role and row['state']['payload'] == payload
                    and all(row[key] == old[row['rank']][key] for key in ('pid', 'start_ticks'))
                    and row['protected_state_before_sha256'] == row['protected_state_after_sha256'] for row in rows),
                'All8 actual pure C1 installation/readback differs')
        installed[role] = rows
    OmegaConf.update(trainer.config, 'actor_rollout_ref.actor.policy_safety', policy, merge=False)
    trainer.fresh200_live_admission = copy.deepcopy(payload)
    os.environ['SEARCH_R1_BASELINE_CONFIG_SHA256'] = config_record['sha256']
    c1 = driver_admission_state(trainer)
    after = driver_complete_pause_state(trainer)
    require(equal_state(driver_training_projection(before), driver_training_projection(after)),
            'C0 toC1 admission changed driver RNG/iterator/sampler/GiGPO/ledger')
    # C0 files remain immutable; the real logical5 basename is preserved under
    # this separate parent for the actual HF loader and allrank full-state save.
    root = run / ('checkpoints-admitted-' + review_record['sha256'][:16])
    require(not root.exists(), 'C1 checkpoint namespace already exists')
    with observe_time(trainer, 'admitted_C1_logical5_fullstate_HF_save', 'checkpoint'):
        trainer._save_checkpoint(checkpoint_root=str(root))
    require(Path(trainer.instruct_manifest['path']).parent.parent.name == 'global_step_5', 'C1 fake logical-step namespace')
    with observe_time(trainer, 'admitted_C1_logical5_fullstate_reload', 'checkpoint'):
        c1_roundtrip = reload_saved_complete_state(trainer, run, 5, 'admitted-C1-step5')
    check_manifest(manifest)  # Recheck C0 exact original files after C1 save/reload.
    receipt = dict(scope=LIVE_SCOPE, status='ACTUAL_C1_ADMITTED_SAME_PROCESS_STATE_RELOADED',
        c0_manifest=manifest, c1_manifest=copy.deepcopy(trainer.instruct_manifest),
        c0_config=c0['payload']['config_record'], c1_config=config_record,
        admission_record=admission_record, installed_roles=installed, actual_role_readback=c1['role_admissions'],
        driver_protected_before_sha256=state_digest(driver_training_projection(before)),
        driver_protected_after_sha256=state_digest(driver_training_projection(after)),
        c1_roundtrip=c1_roundtrip, retained_training_updates_contributed=5,
        no_optimizer_or_forward_performed=True, no_rng_or_data_reset=True)
    json_save(run / 'actual-C0-to-C1-admission.json', receipt)
    return receipt


def _run_counted_training_body(trainer, run, binding, initial, reference, baseline):
    from calibration_driver_v1 import one_outer, json_save, reload_saved_complete_state
    from calibration_state_v1 import state_digest
    from search_r1.baselines.fresh200 import initialize_attempt_ledger
    from search_r1.baselines.fresh200_live import driver_admission_state
    from search_r1.baselines.instruct300 import validate_first5, roundtrip_and_wait, record_evaluation
    from attempt_accounting_v5 import actual_attempt_totals
    if not hasattr(trainer, 'resume50_parent'):
        trainer._calibration_training_iterator = iter(trainer.train_dataloader)
    driver_admission_state(trainer)  # Observe both actual C0 roles before any transition.
    attempts, evaluations, admissions, checkpoints = [], [], [], []
    while True:
        ledger = initialize_attempt_ledger(trainer)
        outer = next_counted_outer(ledger, trainer.fresh200_live_admission)
        if outer is None: break
        result = one_outer(trainer, run, binding, trainer._calibration_training_iterator, outer)
        attempts.append(dict(attempt_id=result['outcome']['attempt']['attempt_id'],
            counted_complete=result['outcome']['counted_complete'], driver_receipt=result['driver_receipt']))
        json_save(run / 'live-progress.json', dict(scope=LIVE_SCOPE,
            job_id=os.environ['SLURM_JOB_ID'], source_commit=os.environ['SEARCH_R1_BASELINE_COMMIT'],
            completed_outer_updates=trainer.actual_outer_updates,
            actual_adam_total=result['outcome']['ledger']['actual_adam_total'],
            actual_attempt_ledger=result['outcome']['ledger'], last_actual_attempt=attempts[-1],
            config_epoch=trainer.fresh200_live_admission['config_epoch'],
            config_sha256=trainer.fresh200_live_admission['config_record']['sha256']))
        if not result['outcome']['counted_complete']: continue
        require(trainer.actual_outer_updates == outer, 'Real completed outer did not advance once')
        trainer.baseline_controller.stage_validation(outer)
        if outer == 5:
            trainer.actor_rollout_wg.calibration_phase('updated512')
            with observe_time(trainer, 'first5_512_evaluation', 'evaluation'):
                metrics = validate_first5(trainer)
            evaluations.append(dict(step=5, metrics=metrics, quality=quality_report(metrics, baseline['full512'])))
            with observe_time(trainer, 'first5_C0_fullstate_HF_save', 'checkpoint'):
                trainer._save_checkpoint()
            checkpoints.append(copy.deepcopy(trainer.instruct_manifest))
            with observe_time(trainer, 'first5_raw_audit_hashing', 'first5_acceptance'):
                evidence = prepare_first5_evidence(trainer, run)
            # Real same-PID complete-state reload, with CPU semantic diagnostics
            # and external review pending in the background under the new scope.
            with observe_time(trainer, 'first5_source_fullstate_reload', 'checkpoint'):
                roundtrip_and_wait(trainer, timing_observer=lambda stage, wall:
                    record_duration(trainer, stage, wall))
            json_save(run / 'first5-human-nonblocking-diagnostics.json', dict(
                status='HUMAN_NONBLOCKING_DIAGNOSTICS', independent_review_status='PENDING_BACKGROUND',
                independent_review_passed=False, production_admission_accepted=False,
                actual_probability_accepted=False, negative_tail_threshold_status='PENDING_BACKGROUND',
                evidence=evidence, manifest=trainer.instruct_manifest,
                external_diagnostic_review_blocks_training=False, retained_training_updates_contributed=5))
        elif outer % 50 == 0:
            trainer.actor_rollout_wg.calibration_phase('updated512')
            with observe_time(trainer, f'periodic{outer}_512_evaluation', 'evaluation'):
                metrics = trainer._validate()
            record_evaluation(trainer, metrics, 'periodic512', trainer.config.data.val_files)
            evaluations.append(dict(step=outer, metrics=metrics, quality=quality_report(metrics, baseline['full512'])))
            with observe_time(trainer, f'completed{outer}_fullstate_HF_save', 'checkpoint'):
                trainer._save_checkpoint()
            checkpoints.append(copy.deepcopy(trainer.instruct_manifest))
        require(reference == trainer.ref_policy_wg.calibration_reference_identity(), 'Fixed official reference changed')
        publish_timing(trainer, run, 'counted_complete_checkpoint_or_evaluation')
    ledger = copy.deepcopy(trainer.fresh200_attempt_ledger)
    totals = actual_attempt_totals(ledger, 200)
    require(trainer.global_steps == trainer.actual_outer_updates == 200 and admissions == [],
            'Live continuation endpoint is not actual retained200')
    with observe_time(trainer, 'completed200_source_fullstate_HF_reload', 'checkpoint'):
        roundtrip_and_wait(trainer, final=True, timing_observer=lambda stage, wall:
            record_duration(trainer, stage, wall))
    with observe_time(trainer, 'completed200_typed_endpoint_reload', 'checkpoint'):
        endpoint = reload_saved_complete_state(trainer, run, 200, 'completed200-human-nonblocking')
    audit = audit_attempts(trainer, run)
    json_save(run / 'actual-continuation-audit.json', dict(scope=LIVE_SCOPE, checks=audit['checks'], observations=audit['observations']))
    trainer._prepare_final_validation()
    with observe_time(trainer, 'final51713_evaluation', 'evaluation'):
        final_metrics = trainer._validate()
    record_evaluation(trainer, final_metrics, 'final51713', trainer.config.data.final_val_files)
    require(final_metrics['val/qa_count/all'] == 51713, 'Actual final evaluation is not complete51713')
    publish_timing(trainer, run, 'completed200_final51713')
    production = trainer.fresh200_live_admission
    return dict(initial_fields(), status='ACTUAL_RETAINED_FRESH200_COMPLETED', scope=LIVE_SCOPE,
        live_goal_updated_at=LIVE_GOAL_UPDATED_AT, retained_training_updates_contributed=200,
        calibration_actual_outer_updates=200, official_training_outer_target=200,
        **totals, optimizer_steps_before=0, optimizer_steps_after=ledger['actual_adam_total'], scheduler_after=200,
        data_cursor=trainer.recovery_sampler.cursor, data_epoch=trainer.recovery_sampler.epoch,
        actual_attempt_ledger=ledger, fresh200_attempt_ledger=ledger, baseline=baseline,
        attempts=attempts, evaluations=evaluations, checkpoints=checkpoints, admissions=admissions,
        manifest=trainer.instruct_manifest, full_state_reloaded=True, roundtrip=endpoint, final_metrics=final_metrics,
        unselected_probability_tolerances={key: binding[key] for key in
            ('actor_recompute_abs_tolerance', 'actor_vllm_abs_tolerance',
             'actor_recompute_joint_abs_tolerance', 'actor_vllm_joint_abs_tolerance')},
        human_old_policy_kl_selected=True, target_old_policy_kl=.2, candidate_old_policy_kl=.2,
        token_kl_threshold_source='human', token_kl_human_goal_updated_at=1791282908,
        negative_tail_limit=production['policy']['negative_tail_limit'], observations=audit['observations'],
        realized_admission=production, optimal_threshold_accepted=False,
        output_quality_accepted=False, production_admission_accepted=False,
        actual_probability_accepted=False, independent_review_status='PENDING_BACKGROUND',
        negative_tail_threshold_status='PENDING_BACKGROUND', human_nonblocking_diagnostics=True,
        no_discarded_training_updates=not hasattr(trainer, 'resume50_parent'),
        cross_job_resume_parent=getattr(trainer, 'resume50_parent', None))


def run_counted_training(trainer, run, binding, initial, reference, baseline):
    """Unknown failures keep local driver/timing only; never dispatch another RPC."""
    try:
        return _run_counted_training_body(trainer, run, binding, initial, reference, baseline)
    except BaseException as error:
        from calibration_driver_v1 import driver_complete_pause_state, json_save
        from search_r1.baselines.recovery import atomic_torch_save
        saved = dict(scope=LIVE_SCOPE, status='FAILED_NOT_RESUME_ACCEPTED', original_error=repr(error),
            completed_outer_updates=getattr(trainer, 'actual_outer_updates', None),
            actual_attempt_ledger=copy.deepcopy(getattr(trainer, 'fresh200_attempt_ledger', None)),
            realized_admission=copy.deepcopy(getattr(trainer, 'fresh200_live_admission', None)),
            timings=copy.deepcopy(getattr(trainer, 'continuation_timings', [])),
            worker_recovery_rpc_dispatched=False, automatic_resume_allowed=False, preservation_errors=[])
        try:
            saved['local_driver'] = driver_complete_pause_state(trainer, local_only=True)
        except BaseException as local_error:
            saved['preservation_errors'].append('local_driver: ' + repr(local_error))
        try:
            receipt = atomic_torch_save(saved, run / 'continuation-terminal-driver.pt')
            json_save(run / 'continuation-terminal.json', dict(status=saved['status'],
                original_error=saved['original_error'], completed_outer_updates=saved['completed_outer_updates'],
                driver=receipt, worker_recovery_rpc_dispatched=False, automatic_resume_allowed=False,
                preservation_errors=saved['preservation_errors']))
        except BaseException:
            pass  # Evidence IO cannot erase the original error or authorize continuation.
        raise
