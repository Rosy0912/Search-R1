"""Explicit fresh Instruct200 protocol; this does not restore an older run."""
from pathlib import Path

GOAL_UPDATED_AT = 1791275767
REVISION = 'aa8e72537993ba99e69dfaafa59ed015b17504d1'
DATA_SHA = '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
CALIBRATION_SCOPE = 'GIGPO_POLICY_SAFETY_OFFICIAL_INSTRUCT_FRESH200_CALIBRATION_V1'
CHECKPOINT_STEPS = (5, 50, 100, 150, 200)


def enabled(config):
    return bool(config.trainer.get('fresh200', {}).get('enabled', False))


def stop_step(config):
    return 200 if enabled(config) else 300


def calibration_initialization():
    return dict(origin_kind='OFFICIAL_MODEL_INITIALIZATION', official_revision=REVISION,
        common_manifest_sha256=DATA_SHA, initial_outer_step=0, initial_adam_steps=0,
        initial_scheduler_step=0, initial_data_cursor=0, restore_checkpoint=None,
        restore_manifest_sha256=None, ppo_mini_batch_size=128, expected_adam_per_rank=20,
        expected_gradient_micro_per_rank=160, reference_kl_coefficient=0.0)


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
            or r.n_agent != 5 or r.n != 1 or r.tensor_model_parallel_size != 1
            or r.log_prob_micro_batch_size != 32 or config.actor_rollout_ref.ref.log_prob_micro_batch_size != 32):
        raise ValueError('Fresh Instruct200 differs from the current authorized protocol')
    for key in ('initial_outer_step', 'initial_adam_steps', 'initial_scheduler_step', 'initial_data_cursor'):
        if type(controls.get(key)) is not int:
            raise ValueError('Fresh initialization counters must be real integer zero: ' + key)
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
    if id(iterator) != identity or driver_flow_state(trainer) != saved:
        raise ValueError('Actual iterator/trace metadata reload readback differs')
    return dict(actual_iterator_metadata_reload=True, actual_validation_trace_reload=True,
        same_live_iterator=True, iterator_process_identity=identity,
        iterator_num_yielded=saved['iterator_fields']['_num_yielded'],
        validation_trace_serial=saved['validation_trace_serial'], data_cursor=saved['data_cursor'])
