"""Explicit fresh Instruct200 protocol; this does not restore an older run."""
from pathlib import Path
import copy
import hashlib
import json
import os

GOAL_UPDATED_AT = 1791275767
TOKEN_KL_HUMAN_GOAL_UPDATED_AT = 1791282908
COUNTING_HUMAN_GOAL_UPDATED_AT = 1791283983
REVISION = 'aa8e72537993ba99e69dfaafa59ed015b17504d1'
DATA_SHA = '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
CALIBRATION_SCOPE = 'GIGPO_POLICY_SAFETY_OFFICIAL_INSTRUCT_FRESH200_CALIBRATION_V1'
CHECKPOINT_STEPS = (5, 50, 100, 150, 200)
# Explicit human-selected sampled-old token KL brake; not a parity tolerance.
TOKEN_KL_LIMIT = 0.2


def enabled(config):
    return bool(config.trainer.get('fresh200', {}).get('enabled', False))


def stop_step(config):
    return 200 if enabled(config) else 300


def calibration_initialization():
    return dict(origin_kind='OFFICIAL_MODEL_INITIALIZATION', official_revision=REVISION,
        common_manifest_sha256=DATA_SHA, initial_outer_step=0, initial_adam_steps=0,
        initial_scheduler_step=0, initial_data_cursor=0, restore_checkpoint=None,
        restore_manifest_sha256=None, ppo_mini_batch_size=128, optimizer_opportunities_per_outer=20,
        diagnostic_total_optimizer_opportunities=40, no_skip_adam_per_outer_per_rank=20,
        no_skip_gradient_micro_per_outer_per_rank=160, counters_are_no_skip_reference=True,
        full_outer_requires_all_opportunities_processed=True,
        full_outer_requires_at_least_one_actual_adam_write=True,
        actual_write_counts_must_be_reported_separately=True,
        zero_write_or_abandoned_outer_counts=False,
        token_kl_threshold_source='human', token_kl_human_goal_updated_at=TOKEN_KL_HUMAN_GOAL_UPDATED_AT,
        counting_rules_source='human', counting_human_goal_updated_at=COUNTING_HUMAN_GOAL_UPDATED_AT,
        reference_kl_coefficient=0.0)


def validate_calibration_initialization(proof):
    for key, value in calibration_initialization().items():
        if key not in proof or proof[key] != value or isinstance(proof[key], bool) != isinstance(value, bool):
            raise ValueError('Fresh official calibration initialization differs: ' + key)


def validate_protocol(config):
    """Reject the superseded25/300/mini256 recipe before sampler or CUDA work."""
    if not enabled(config):
        return
    t, a, r = config.trainer, config.actor_rollout_ref.actor, config.actor_rollout_ref.rollout
    controls = t.fresh200
    if (controls.get('enabled') is not True or type(controls.get('goal_updated_at')) is not int
            or controls.get('goal_updated_at') != GOAL_UPDATED_AT
            or t.total_training_steps != 200 or t.acceptance_step != 5
            or t.save_freq != 50 or t.test_freq != 50 or t.val_before_train is not True
            or not t.get('instruct300', {}).get('enabled', False)
            or t.get('resume', {}).get('enabled', False)
            or t.get('resume', {}).get('source') is not None
            or t.get('quality_resume', {}).get('enabled', False)
            or t.instruct300.get('restore_checkpoint') is not None
            or t.instruct300.get('restore_manifest_sha256') is not None
            or controls.get('initial_outer_step') != 0 or controls.get('initial_adam_steps') != 0
            or controls.get('initial_scheduler_step') != 0 or controls.get('initial_data_cursor') != 0
            or list(controls.get('checkpoint_steps', [])) != list(CHECKPOINT_STEPS)
            or controls.get('preserve_partial_updates') is not True
            or t.get('trajectory_logging') is not True
            or t.n_gpus_per_node != 8 or t.nnodes != 1
            or a.ppo_mini_batch_size != 128 or a.ppo_micro_batch_size != 16 or a.ppo_epochs != 1
            or a.optim.lr != 1e-6 or a.optim.lr_warmup_steps != 57
            or a.optim.lr_warmup_steps_ratio != 0 or a.optim.total_training_steps != 200
            or a.optim.get('initial_logical_step', 0) != 0
            or a.entropy_coeff != 0 or a.use_kl_loss is not True or a.kl_loss_coef != 0
            or a.kl_loss_type != 'low_var_kl' or config.algorithm.kl_ctrl.kl_coef != 0
            or a.clip_ratio != .2 or a.grad_clip != 1
            or a.get('fresh200_protocol') is not True
            or a.get('preserve_partial_policy_safety_state') is not True
            or a.get('joint_loss_scaling') is not True
            or a.policy_safety.get('calibration_scope') != CALIBRATION_SCOPE
            or a.policy_safety.get('enabled') is not True or a.policy_safety.get('dual_clip') != 3.0
            or a.policy_safety.get('kl_log_bound') != 30.0
            or a.policy_safety.get('monitor_first_global_rows') != 256
            or a.policy_safety.get('target_old_policy_kl') != TOKEN_KL_LIMIT
            or r.n_agent != 5 or r.n != 1 or r.tensor_model_parallel_size != 1
            or r.log_prob_micro_batch_size != 32 or config.actor_rollout_ref.ref.log_prob_micro_batch_size != 32):
        raise ValueError('Fresh Instruct200 differs from the current authorized protocol')
    for key in ('initial_outer_step', 'initial_adam_steps', 'initial_scheduler_step', 'initial_data_cursor'):
        if type(controls.get(key)) is not int:
            raise ValueError('Fresh initialization counters must be real integer zero: ' + key)
    for key in ('optimizer_opportunities_per_outer', 'no_skip_adam_per_outer_per_rank',
            'no_skip_gradient_micro_per_outer_per_rank', 'counters_are_no_skip_reference',
            'full_outer_requires_all_opportunities_processed',
            'full_outer_requires_at_least_one_actual_adam_write',
            'actual_write_counts_must_be_reported_separately', 'zero_write_or_abandoned_outer_counts',
            'token_kl_threshold_source', 'token_kl_human_goal_updated_at',
            'counting_rules_source', 'counting_human_goal_updated_at'):
        expected = calibration_initialization()[key]
        if controls.get(key) != expected or type(controls.get(key)) is not type(expected):
            raise ValueError('Fresh actual-write counting authorization differs: ' + key)
    raw = [config.get('actor_token_identity'), a.get('actor_token_identity'),
        r.get('actor_token_identity'), config.actor_rollout_ref.ref.get('actor_token_identity')]
    if raw != ['sampled_action_prefix_v1'] * 4:
        raise ValueError('Fresh200 must preserve actual sampled action probability identities')


def fixed_monitor_layout(local_rows, local_mini, world, global_monitor_rows):
    """Monitor complete local trajectories independently of the PPO mini split."""
    values = (local_rows, local_mini, world, global_monitor_rows)
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError('Invalid fixed monitor/mini layout')
    if (local_rows * world != 2560 or local_rows % local_mini
            or global_monitor_rows % world or global_monitor_rows > local_rows * world):
        raise ValueError('Fixed monitor/full rollout coverage differs')
    return global_monitor_rows // world, local_rows // local_mini


def select_fixed_monitor(batch, local_mini, world, global_monitor_rows):
    rows, expected_minis = fixed_monitor_layout(len(batch), local_mini, world, global_monitor_rows)
    return batch[:rows].clone(), expected_minis


def validate_initial_workers(actor, reference):
    for role, rows in (('actor', actor), ('reference', reference)):
        if (len(rows) != 8 or {v['rank'] for v in rows} != set(range(8))
                or any(v['role'] != role or v['official_revision'] != REVISION
                    or v['origin_kind'] != 'OFFICIAL_MODEL_INITIALIZATION'
                    or not isinstance(v['pid'], int) or v['pid'] <= 0
                    or not isinstance(v['start_ticks'], int) or v['start_ticks'] <= 0 for v in rows)):
            raise ValueError('Missing actual eight-rank fresh initialization: ' + role)
    for value in actor:
        if (value['optimizer_populated_slots'] != 0 or value['scheduler_last_epoch'] != 0
                or value['scheduler_step_count'] != 1 or value['learning_rates'] != [0.0]
                or value['local_mini_batch_size'] != 16 or value['local_micro_batch_size'] != 2
                or value['reference_kl_coefficient'] != 0.0):
            raise ValueError('Fresh actor inherited optimizer/scheduler state or a wrong micro/mini budget')
    if {v['model_path'] for v in actor} != {v['model_path'] for v in reference}:
        raise ValueError('Fresh actor and fixed reference model paths differ')
    by_rank = {v['rank']: v for v in reference}
    if any(value['official_bf16_parameter_sha256'] != by_rank[value['rank']]['official_bf16_parameter_sha256']
            for value in actor):
        raise ValueError('Actual actor BF16 weights differ from the fixed official reference')


def record_initialization(trainer):
    """Observe real workers and the actual zero-cursor sampler, before validation."""
    if not enabled(trainer.config):
        return
    from .recovery import atomic_json_save
    sampler = trainer.recovery_sampler.state_dict()
    if (trainer.global_steps != 0 or getattr(trainer, 'actual_outer_updates', 0) != 0
            or sampler['cursor'] != 0 or sampler['order'] is not None or sampler['epoch'] != -1):
        raise RuntimeError('Fresh200 was initialized from a historical training boundary')
    actor = trainer.actor_rollout_wg.fresh_initialization_identity()
    reference = trainer.ref_policy_wg.fresh_initialization_identity()
    validate_initial_workers(actor, reference)
    trainer.fresh200_initial_identity = dict(status='ACTUAL_FRESH_OFFICIAL_ZERO_STATE',
        initialization=calibration_initialization(), actor_workers=actor,
        reference_workers=reference, actual_completed_outer_updates=0,
        sampler=dict(size=sampler['size'], cursor=sampler['cursor'], epoch=sampler['epoch'], order=sampler['order']))
    atomic_json_save(trainer.fresh200_initial_identity,
        Path(trainer.config.trainer.default_local_dir).parent / 'fresh200-initialization.json')


_ITERATOR_FIELDS = ('_base_seed', '_num_yielded', '_num_workers', '_dataset_kind',
                    '_auto_collation', '_drop_last', '_timeout')


def driver_flow_state(trainer):
    """Capture actual single-process iteration and validation trace progress."""
    iterator = getattr(trainer, 'fresh200_active_train_iterator', None)
    if iterator is None:
        iterator = getattr(trainer, '_calibration_training_iterator', None)
    if iterator is None or not all(hasattr(iterator, key) for key in _ITERATOR_FIELDS):
        raise ValueError('Missing actual single-process training iterator metadata')
    fields = {key: getattr(iterator, key) for key in _ITERATOR_FIELDS}
    if (fields['_num_workers'] != 0 or type(fields['_base_seed']) is not int
            or type(fields['_num_yielded']) is not int or fields['_num_yielded'] < 0
            or not all(type(value) in (bool, int, float) for value in fields.values())):
        raise ValueError('Unsupported training iterator fields or worker prefetch')
    serial = getattr(trainer, 'validation_trace_serial', 0)
    if type(serial) is not int or serial < 0:
        raise ValueError('Invalid actual validation trace progress')
    return dict(iterator_class=type(iterator).__module__ + '.' + type(iterator).__qualname__,
        iterator_fields=fields, validation_trace_serial=serial,
        data_cursor=trainer.recovery_sampler.cursor,
        data_epoch=trainer.recovery_sampler.epoch,
        attempt_ledger=copy.deepcopy(initialize_attempt_ledger(trainer)),
        sample_iterator_scope='live_single_process_iterator; saved_sampler_controls_next_data_row')


def restore_driver_flow(trainer, saved):
    """Reload saved small fields in the same actual iterator without constructing another."""
    before = driver_flow_state(trainer)
    if before != saved:
        raise ValueError('Live iterator/validation progress changed before checkpoint roundtrip')
    iterator = getattr(trainer, 'fresh200_active_train_iterator', None)
    if iterator is None:
        iterator = trainer._calibration_training_iterator
    identity = id(iterator)
    for key in _ITERATOR_FIELDS:
        setattr(iterator, key, saved['iterator_fields'][key])
    trainer.validation_trace_serial = saved['validation_trace_serial']
    restore_attempt_ledger(trainer, saved['attempt_ledger'])
    if id(iterator) != identity or driver_flow_state(trainer) != saved:
        raise ValueError('Actual iterator/trace metadata reload readback differs')
    return dict(actual_iterator_metadata_reload=True, actual_validation_trace_reload=True,
        same_live_iterator=True, iterator_process_identity=identity,
        iterator_num_yielded=saved['iterator_fields']['_num_yielded'],
        validation_trace_serial=saved['validation_trace_serial'], data_cursor=saved['data_cursor'])


ATTEMPT_SCHEMA = 'FRESH200_OPPORTUNITIES_AND_ACTUAL_WRITES_V1'


def initialize_attempt_ledger(trainer):
    """A completed outer processes twenty opportunities and writes Adam at least once."""
    if not hasattr(trainer, 'fresh200_attempt_ledger'):
        trainer.fresh200_attempt_ledger = dict(schema=ATTEMPT_SCHEMA,
            completed_outer_updates=0, attempts=0, skipped_attempts=0,
            actual_adam_total=0, successful_adam_writes_completed_outers=0,
            successful_adam_writes_abandoned_attempts=0, skipped_opportunities=0,
            gradient_micro_batches=0, consumed_training_questions=0,
            history=[], active_attempt=None, epoch_ends=[])
    validate_attempt_ledger(trainer.fresh200_attempt_ledger)
    return trainer.fresh200_attempt_ledger


def _sampler_position(sampler):
    state = sampler.state_dict()
    order = state['order']
    return dict(epoch=state['epoch'], cursor=state['cursor'], size=state['size'],
        permutation_sha256=None if order is None else hashlib.sha256(
            json.dumps(order, separators=(',', ':')).encode()).hexdigest())


def begin_attempt(trainer, batch):
    """Bind one actual fetched512 batch. Never fetch, rewind, or create a permutation."""
    ledger = initialize_attempt_ledger(trainer)
    if ledger['active_attempt'] is not None or len(batch) != 512:
        raise ValueError('A fresh attempt requires one complete newly fetched512 batch')
    if getattr(trainer, 'actual_outer_updates', 0) != ledger['completed_outer_updates']:
        raise ValueError('Driver completed count differs from its attempt ledger')
    position = _sampler_position(trainer.recovery_sampler)
    previous = ledger['history'][-1]['sampler_position'] if ledger['history'] else None
    if previous is None:
        valid = position['epoch'] == 0 and position['cursor'] == 512
    elif position['epoch'] == previous['epoch']:
        valid = (position['cursor'] == previous['cursor'] + 512
                 and position['permutation_sha256'] == previous['permutation_sha256'])
    else:
        valid = (position['epoch'] == previous['epoch'] + 1 and position['cursor'] == 512
                 and ledger['epoch_ends'] and ledger['epoch_ends'][-1]['epoch'] == previous['epoch'])
    if not valid:
        raise ValueError('Attempt is not the next real512 positions in the saved sampler')
    ledger['attempts'] += 1
    ledger['consumed_training_questions'] += 512
    attempt = dict(attempt_id=ledger['attempts'], completed_before=ledger['completed_outer_updates'],
        actual_adam_total_before=ledger['actual_adam_total'], sampler_position=position,
        driver_pid=os.getpid())
    ledger['active_attempt'] = attempt
    batch.meta_info.update(fresh200_attempt_id=attempt['attempt_id'],
        fresh200_completed_before=attempt['completed_before'],
        fresh200_adam_total_before=attempt['actual_adam_total_before'],
        baseline_outer_step=attempt['completed_before'] + 1)
    return attempt['attempt_id']


def record_epoch_end(trainer):
    """Record real dropped tail rows on iterator exhaustion before another epoch."""
    ledger = initialize_attempt_ledger(trainer)
    position = _sampler_position(trainer.recovery_sampler)
    if ledger['active_attempt'] is not None or position['cursor'] != position['size']:
        raise ValueError('New epoch requires the actual exhausted sampler')
    last = ledger['history'][-1]['sampler_position'] if ledger['history'] else dict(cursor=0)
    ledger['epoch_ends'].append(dict(position, dropped_tail_rows=position['cursor'] - last['cursor']))


def consume_attempt(trainer, actor_output):
    """Read eight actual rank receipts concatenated as non-tensor rows by DataProto."""
    ledger = initialize_attempt_ledger(trainer)
    active = ledger['active_attempt']
    rows = actor_output.non_tensor_batch.get('fresh200_attempt_receipts')
    if active is None or rows is None:
        raise ValueError('Missing active attempt or actual all-rank return receipts')
    rows = list(rows)
    if (len(rows) != 8 or any(type(row.get('rank')) is not int
            or type(row.get('pid')) is not int or row['pid'] <= 0 for row in rows)
            or {row['rank'] for row in rows} != set(range(8))
            or len({row['pid'] for row in rows}) != 8):
        raise ValueError('Incomplete actual eight-rank attempt coverage')
    common = ('attempt_id', 'completed_before', 'opportunities_processed',
        'actual_successful_adam_writes', 'skipped_opportunities', 'gradient_micro_batches',
        'abandoned', 'counted_complete', 'actual_adam_total', 'scheduler_step', 'safe_to_continue')
    first = rows[0]
    if any(any(row[key] != first[key] or type(row[key]) is not type(first[key]) for key in common) for row in rows):
        raise ValueError('Rank attempt decisions/counters differ')
    processed, writes, skips = (first[key] for key in
        ('opportunities_processed', 'actual_successful_adam_writes', 'skipped_opportunities'))
    if (any(type(first[key]) is not int for key in ('attempt_id', 'completed_before',
            'actual_adam_total', 'scheduler_step', 'opportunities_processed',
            'actual_successful_adam_writes', 'skipped_opportunities', 'gradient_micro_batches'))
            or not 0 <= processed <= 20 or not 0 <= writes <= processed
            or skips != processed - writes or not writes * 8 <= first['gradient_micro_batches'] <= processed * 8
            or first['attempt_id'] != active['attempt_id']
            or first['completed_before'] != active['completed_before']
            or first['actual_adam_total'] != active['actual_adam_total_before'] + writes
            or type(first['abandoned']) is not bool or type(first['counted_complete']) is not bool
            or first['safe_to_continue'] is not True):
        raise ValueError('Attempt receipt is not the actual opportunity/write boundary')
    countable = processed == 20 and writes > 0 and first['abandoned'] is False
    if (first['counted_complete'] is not countable
            or first['scheduler_step'] != active['completed_before'] + int(countable)):
        raise ValueError('Only a countable completed outer may advance warmup')
    record = dict(active, **{key: first[key] for key in common if key not in active},
                  rank_receipts=copy.deepcopy(rows))
    ledger['history'].append(record)
    ledger['completed_outer_updates'] += int(countable)
    ledger['skipped_attempts'] += int(not countable)
    ledger['actual_adam_total'] += writes
    key = ('successful_adam_writes_completed_outers' if countable else
           'successful_adam_writes_abandoned_attempts')
    ledger[key] += writes
    ledger['skipped_opportunities'] += skips
    ledger['gradient_micro_batches'] += first['gradient_micro_batches']
    ledger['active_attempt'] = None
    trainer.actual_outer_updates = ledger['completed_outer_updates']
    validate_attempt_ledger(ledger)
    return dict(counted_complete=countable, attempt=record, ledger=copy.deepcopy(ledger))


def validate_attempt_ledger(ledger):
    if ledger.get('schema') != ATTEMPT_SCHEMA:
        raise ValueError('Unsupported fresh opportunity ledger')
    names = ('completed_outer_updates', 'attempts', 'skipped_attempts', 'actual_adam_total',
        'successful_adam_writes_completed_outers', 'successful_adam_writes_abandoned_attempts',
        'skipped_opportunities', 'gradient_micro_batches', 'consumed_training_questions')
    if any(type(ledger.get(key)) is not int or ledger[key] < 0 for key in names):
        raise ValueError('Ledger counters must be nonnegative actual integers')
    history, active = ledger['history'], ledger['active_attempt']
    completed = total = full_writes = abandoned_writes = skips = micros = 0
    for index, row in enumerate(history, 1):
        processed, writes = row['opportunities_processed'], row['actual_successful_adam_writes']
        counted = processed == 20 and writes > 0 and row['abandoned'] is False
        if (any(type(row[key]) is not int for key in ('attempt_id', 'completed_before',
                'actual_adam_total_before', 'opportunities_processed', 'actual_successful_adam_writes',
                'skipped_opportunities', 'gradient_micro_batches', 'actual_adam_total', 'scheduler_step'))
                or type(row['abandoned']) is not bool or type(row['counted_complete']) is not bool
                or row['attempt_id'] != index or row['completed_before'] != completed
                or row['actual_adam_total_before'] != total or not 0 <= writes <= processed <= 20
                or row['skipped_opportunities'] != processed - writes
                or not writes * 8 <= row['gradient_micro_batches'] <= processed * 8
                or row['counted_complete'] is not counted or row['safe_to_continue'] is not True
                or row['scheduler_step'] != completed + int(counted)
                or row['actual_adam_total'] != total + writes):
            raise ValueError('Invalid actual attempt row or scheduler/write chain')
        completed += int(counted)
        total += writes
        full_writes += writes if counted else 0
        abandoned_writes += 0 if counted else writes
        skips += row['skipped_opportunities']
        micros += row['gradient_micro_batches']
    if (ledger['attempts'] != len(history) + int(active is not None)
            or ledger['consumed_training_questions'] != ledger['attempts'] * 512
            or [row['attempt_id'] for row in history] != list(range(1, len(history) + 1))
            or ledger['completed_outer_updates'] != sum(row['counted_complete'] for row in history)
            or ledger['skipped_attempts'] != sum(not row['counted_complete'] for row in history)
            or ledger['actual_adam_total'] != sum(row['actual_successful_adam_writes'] for row in history)
            or ledger['actual_adam_total'] != ledger['successful_adam_writes_completed_outers'] +
                ledger['successful_adam_writes_abandoned_attempts']
            or ledger['successful_adam_writes_completed_outers'] != full_writes
            or ledger['successful_adam_writes_abandoned_attempts'] != abandoned_writes
            or ledger['skipped_opportunities'] != skips or ledger['gradient_micro_batches'] != micros
            or (active is not None and (active['attempt_id'] != ledger['attempts']
                or active['completed_before'] != completed or active['actual_adam_total_before'] != total))):
        raise ValueError('Ledger totals do not describe its actual unique attempts')
    return ledger


def restore_attempt_ledger(trainer, saved):
    """Same-PID complete-state roundtrip restores actual totals, never infers20×step."""
    validate_attempt_ledger(saved)
    current = initialize_attempt_ledger(trainer)
    if current != saved:
        raise ValueError('Saved attempt ledger differs from the actual live driver')
    trainer.fresh200_attempt_ledger = copy.deepcopy(saved)
    return copy.deepcopy(trainer.fresh200_attempt_ledger)


def validate_worker_attempt_ledger(ledger):
    if ledger.get('schema') != 'FRESH200_WORKER_ACTUAL_ATTEMPTS_V1':
        raise ValueError('Unsupported actual worker attempt ledger')
    history = ledger['history']
    if any(type(ledger.get(key)) is not int or ledger[key] < 0
           for key in ('attempts', 'completed_outer_updates', 'actual_adam_total')):
        raise ValueError('Worker counters must be actual nonnegative integers')
    completed = total = 0
    for index, row in enumerate(history, 1):
        processed, writes = row['opportunities_processed'], row['actual_successful_adam_writes']
        counted = processed == 20 and writes > 0 and row['abandoned'] is False
        if (any(type(row[key]) is not int for key in ('attempt_id', 'completed_before',
                'opportunities_processed', 'actual_successful_adam_writes', 'skipped_opportunities',
                'gradient_micro_batches', 'actual_adam_total', 'scheduler_step'))
                or type(row['counted_complete']) is not bool or type(row['abandoned']) is not bool
                or row['safe_to_continue'] is not True or not 0 <= writes <= processed <= 20
                or row['attempt_id'] != index or row['completed_before'] != completed
                or row['skipped_opportunities'] != processed - writes
                or not writes * 8 <= row['gradient_micro_batches'] <= processed * 8
                or row['counted_complete'] is not counted
                or row['actual_adam_total'] != total + writes
                or row['scheduler_step'] != completed + int(counted)):
            raise ValueError('Worker actual opportunity/write/scheduler chain differs')
        completed += int(counted)
        total += writes
    if (ledger['attempts'] != len(history)
            or [row['attempt_id'] for row in history] != list(range(1, len(history) + 1))
            or ledger['completed_outer_updates'] != sum(row['counted_complete'] for row in history)
            or ledger['actual_adam_total'] != sum(row['actual_successful_adam_writes'] for row in history)
            or any(row['counted_complete'] is not (row['opportunities_processed'] == 20
                and row['actual_successful_adam_writes'] > 0 and row['abandoned'] is False) for row in history)):
        raise ValueError('Worker ledger does not describe its actual opportunities/writes')
    return ledger


def record_worker_attempt(worker, receipt):
    """The actual update_actor completion seam: one scheduler step iff countable."""
    ledger = copy.deepcopy(getattr(worker, 'fresh200_worker_ledger',
        dict(schema='FRESH200_WORKER_ACTUAL_ATTEMPTS_V1', attempts=0,
             completed_outer_updates=0, actual_adam_total=0, history=[])))
    validate_worker_attempt_ledger(ledger)
    counted = (receipt['opportunities_processed'] == 20 and
               receipt['actual_successful_adam_writes'] > 0 and receipt['abandoned'] is False)
    if (receipt['counted_complete'] is not counted
            or receipt['attempt_id'] != ledger['attempts'] + 1
            or receipt['completed_before'] != ledger['completed_outer_updates']
            or worker.actor_lr_scheduler.last_epoch != ledger['completed_outer_updates']):
        raise ValueError('Actual countable worker return does not follow its scheduler/attempt ledger')
    if counted:
        worker.actor_lr_scheduler.step()  # Unknown scheduler errors escape unchanged.
    receipt['scheduler_step'] = worker.actor_lr_scheduler.last_epoch
    ledger['attempts'] += 1
    ledger['completed_outer_updates'] += int(counted)
    ledger['actual_adam_total'] = receipt['actual_adam_total']
    ledger['history'].append(copy.deepcopy(receipt))
    validate_worker_attempt_ledger(ledger)
    worker.fresh200_worker_ledger = ledger
    return receipt
