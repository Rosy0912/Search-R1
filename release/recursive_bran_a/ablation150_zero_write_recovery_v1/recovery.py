"""Reuse the stopped first actor batch without repeating generation or resetting Adam.

There is no saved worker checkpoint in this narrowly admitted failure. The native
fit must first observe its configured official model and genuinely empty Adam and
scheduler. We restore only the saved driver state and immutable prepared batch,
then execute the original actor/outer validation endpoints once. Worker CUDA RNG
and the former vLLM executor state are unavailable; no bitwise restart is claimed.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import time

from . import PROTOCOL

MANIFEST_ENV = 'RECURSIVE_ABLATION_ZERO_WRITE_MANIFEST'
SHA_ENV = 'RECURSIVE_ABLATION_ZERO_WRITE_SHA256'
IDENTITY_ENV = dict(job_id='SLURM_JOB_ID', run_id='SEARCH_R1_BASELINE_RUN_ID',
    code_commit='SEARCH_R1_BASELINE_COMMIT', config_sha256='SEARCH_R1_BASELINE_CONFIG_SHA256',
    data_identity='SEARCH_R1_BASELINE_DATA_IDENTITY', model_identity='SEARCH_R1_MODEL_IDENTITY')
FAILURE = 'Partial state directory must remain in this own run and match attempted outer'
FIELDS = ('prompts', 'responses', 'responses_with_info_mask', 'input_ids', 'attention_mask',
    'info_mask', 'position_ids', 'step_ids', 'response_mask', 'sampled_old_log_probs',
    'old_log_probs', 'ref_log_prob', 'token_level_scores', 'token_level_rewards',
    'advantages', 'returns', 'policy_weights', 'loss_mask')


def enabled(environ=None):
    return bool((os.environ if environ is None else environ).get(MANIFEST_ENV))


def validate_zero_write_evidence(value, receipt, environ):
    """Pure admission: no nonzero, later-round, unmatched, or ordinary safety stop."""
    identity = value.get('identity', {})
    expected = {key: environ.get(env) for key, env in IDENTITY_ENV.items()}
    if (any(not isinstance(v, str) or not v for v in expected.values())
            or any(identity.get(key) != v for key, v in expected.items())
            or expected['config_sha256'] != receipt['config_sha256']
            or expected['data_identity'] != receipt['data_identity']
            or expected['code_commit'] != receipt['source_commit']
            or expected['run_id'] != receipt['run_id']
            or receipt['group'] != 'C1-root-only'):
        raise ValueError('Zero-write recovery requires this unchanged C1 source/config/data/model/run/job')
    if (value.get('protocol') != 'tree_partial_state_v1'
            or value.get('status') != 'PARTIAL_OUTER_STATE_INCOMPLETE_CONTROLLED_STOP'
            or value.get('outer_completed') is not False
            or value.get('automatic_rollback') is not False
            or value.get('completed_outer_updates') != 0 or value.get('attempted_outer') != 1
            or identity.get('attempt_id') != 1 or identity.get('attempted_outer') != 1
            or identity.get('completed_outer') != 0
            or identity.get('training_protocol') != 'tree_fresh200_five_leaf_v1'):
        raise ValueError('Only this incomplete first attempted outer with zero completed outers is recoverable')
    envelope = value.get('worker_outcome', {})
    workers = envelope.get('worker_states', [])
    if (envelope.get('protocol') != 'tree_partial_state_v1'
            or envelope.get('status') != 'PARTIAL_WORKER_STATE_INCOMPLETE'
            or envelope.get('identity_consistent') is not False
            or envelope.get('identity') != {}
            or envelope.get('scheduler_advanced_actual') != [False] * 8
            or len(workers) != 8 or {r.get('rank') for r in workers} != set(range(8))):
        raise ValueError('Missing explicit eight-rank pre-update refusal evidence')
    for rank, record in enumerate(sorted(workers, key=lambda r: r['rank'])):
        if (record.get('rank') != rank or record.get('world_size') != 8
                or record.get('matched_all_ranks') is not True or record.get('safety_guard') is not False
                or type(record.get('committed_optimizer_calls')) is not int
                or record['committed_optimizer_calls'] != 0
                or type(record.get('scheduler_step_before_actual')) is not int
                or type(record.get('scheduler_step_after_actual')) is not int
                or record['scheduler_step_before_actual'] != 0 or record['scheduler_step_after_actual'] != 0
                or record.get('status') != 'PARTIAL_WORKER_STATE_INCOMPLETE'
                or record.get('path') is not None or record.get('sha256') is not None
                or record.get('bytes') != 0
                or FAILURE not in record.get('error', '')
                or 'Global partial worker context preparation refusal' not in record.get('refusal', '')):
            raise ValueError('A rank lacks the exact zero-write, zero-scheduler startup namespace refusal')
    if (value.get('driver', {}).get('consumed_cursor') != 512
            or value['driver'].get('consumed_epoch') != 0
            or value.get('actor_inputs', {}).get('rows') != 2560
            or tuple(value['actor_inputs'].get('fields', [])) != FIELDS):
        raise ValueError('Saved consumed questions or complete prepared actor schema differs')
    return value


def manifest(environ=None):
    from release.recursive_bran_a.ablation150_v1.contracts import read_receipt
    environ = os.environ if environ is None else environ
    receipt = read_receipt(environ['RECURSIVE_ABLATION_RECEIPT'])
    source = Path(environ[MANIFEST_ENV])
    expected_dir = Path(receipt['base']) / 'checkpoints' / 'training_state' / 'partial_attempt_000001_outer_000001'
    if source.is_symlink() or source.name != 'manifest.json' or source.resolve().parent != expected_dir.resolve():
        raise ValueError('Zero-write evidence must be the original unchanged C1 first partial manifest')
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != environ.get(SHA_ENV):
        raise ValueError('Explicit zero-write manifest SHA changed')
    value = validate_zero_write_evidence(json.loads(raw), receipt, environ)
    for name, filename in (('driver', 'driver.pt'), ('actor_inputs', 'actor-inputs.pt')):
        member = Path(value[name]['path'])
        if (member.is_symlink() or not member.is_file() or member.name != filename
                or member.resolve().parent != source.resolve().parent):
            raise ValueError('Prepared recovery member is missing or outside its frozen partial evidence')
    if source.read_bytes() != raw:
        raise ValueError('Zero-write manifest changed during admission')
    return value


def validate_entry(config=None, environ=None):
    value = manifest(environ)
    if config is not None:
        from omegaconf import OmegaConf
        from release.recursive_bran_a.ablation150_v1.contracts import validate_config_dict
        validate_config_dict(OmegaConf.to_container(config, resolve=True))
    return value


def validate_initialization(provenance, identity):
    workers = provenance.get('workers', []) if isinstance(provenance, dict) else []
    if len(workers) != 8 or {r.get('rank') for r in workers} != set(range(8)):
        raise ValueError('Eight actual official initial worker observations are required')
    for record in workers:
        if (record.get('world_size') != 8 or record.get('optimizer_state_entries') != 0
                or record.get('optimizer_step_values') != [] or record.get('scheduler_step') != 0
                or record.get('optimizer_committed') != 0 or record.get('scheduler_advanced') is not False
                or not record.get('configured_model_path')
                or record.get('configured_model_path') != record.get('expected_model_path')
                or any(record.get(key) != identity[key] for key in IDENTITY_ENV)):
            raise ValueError('Official source path, empty Adam, scheduler zero, or runtime identity differs')


def restore_partial_driver(trainer, saved, value):
    from search_r1.baselines.training_state import cpu_copy, digest, restore_rng
    identity = value['identity']
    if (saved.get('version') != 3 or saved.get('snapshot_kind') != 'tree_partial_state_v1'
            or saved.get('identity') != identity or saved.get('attempted_outer') != 1
            or saved.get('completed_outer_updates') != 0 or saved.get('outer_completed') is not False
            or digest(saved) != value['driver']['logical_state_sha256']
            or trainer.global_steps != 0 or trainer.actual_outer_updates != 0
            or trainer.checkpoint_sampler.cursor != 0 or trainer.checkpoint_sampler.epoch != -1):
        raise ValueError('Saved driver identity/hash or untouched live driver differs')
    validate_initialization(saved.get('fresh_initialization_provenance'), identity)
    validate_initialization(trainer.fresh_initialization_provenance, identity)
    sampler = saved['consumed_sampler']
    algorithm = saved['algorithm']
    history = saved.get('guard_continue_accounting', {})
    if (sampler['batch_size'] != 512 or sampler['cursor'] != 512 or sampler['epoch'] != 0
            or algorithm['method'] != trainer.baseline_controller.method
            or algorithm['method'] != 'tree_grpo' or algorithm['tree_rng'] is None
            or saved.get('recovery_provenance') is not None
            or history.get('training_attempts') != 1 or history.get('completed_outer_updates') != 0
            or history.get('refused_attempts') != 0 or history.get('guard_refused_attempts') != 0
            or history.get('partial_evidence') != []
            or any(history.get(key) != {} for key in
                ('completed_optimizer_calls_per_rank', 'partial_optimizer_calls_per_rank'))
            or history.get('total_optimizer_calls_per_rank') != {str(r): 0 for r in range(8)}):
        raise ValueError('Consumed sampler/controller/attempt history does not describe the unique first batch')
    trainer.checkpoint_sampler.load_state_dict(sampler)
    trainer.baseline_controller.stage.load_state_dict(algorithm['stage'])
    trainer.baseline_controller.tree_rng.setstate(algorithm['tree_rng'])
    trainer.global_steps = 1
    trainer.actual_outer_updates = 0
    trainer.training_attempts = 1
    trainer.refused_attempts = trainer.guard_refused_attempts = 0
    trainer.completed_optimizer_calls_per_rank = {}
    trainer.partial_optimizer_calls_per_rank = {}
    trainer.partial_attempt_evidence = []
    trainer.zero_write_sampling_rounds = trainer.normal_guard_rounds = 0
    restore_rng(saved['rng'])
    if (digest(trainer.checkpoint_sampler.state_dict()) != digest(sampler)
            or digest(trainer.baseline_controller.stage.state_dict()) != digest(algorithm['stage'])
            or trainer.baseline_controller.tree_rng.getstate() != algorithm['tree_rng']):
        raise ValueError('Live consumed sampler/controller/tree RNG did not restore exactly')
    return cpu_copy(saved['fresh_initialization_provenance'])


def prepared_batch(saved, value):
    """Wrap the loaded tensors and object arrays directly; never select/reorder/recompute."""
    from tensordict import TensorDict
    from verl import DataProto
    from search_r1.baselines.tree_behavior_mu import PROVENANCE_META
    if set(saved) != {'batch', 'non_tensor_batch', 'meta_info'} or tuple(saved['batch']) != FIELDS:
        raise ValueError('Prepared batch archive schema differs')
    rows = value['actor_inputs']['rows']
    if any(tensor.shape[0] != rows for tensor in saved['batch'].values()):
        raise ValueError('Prepared actor tensor row coverage differs')
    for key in ('uid', 'tree_uid', 'trajectory_uid', 'tree_behavior_mu_binding', 'tree_raw_semantics'):
        if key not in saved['non_tensor_batch'] or len(saved['non_tensor_batch'][key]) != rows:
            raise ValueError('Prepared actor row identities or immutable probability proof coverage missing')
    meta = saved['meta_info']
    expected = {env: value['identity'][key] for key, env in IDENTITY_ENV.items()}
    if (meta.get(PROVENANCE_META) != expected or meta.get('baseline_outer_step') != 1
            or meta.get('baseline_attempt_id') != 1 or meta.get('baseline_completed_outer') != 0
            or 'global_token_num' not in meta):
        raise ValueError('Prepared actor runtime provenance/attempt/token metadata differs')
    return DataProto(batch=TensorDict(saved['batch'], batch_size=[rows]),
                     non_tensor_batch=saved['non_tensor_batch'], meta_info=meta)


def complete_prepared_outer(trainer, batch):
    """Execute the native validation/credit path, beginning immediately at update_actor."""
    from search_r1.baselines.tree_partial_state import (partial_context, stop_on_partial_actor_result,
                                                       preserve_unmatched_actor_failure)
    from search_r1.baselines.tree_guard_continue import validate_completed_result, record_completed_result
    from search_r1.baselines.tree_fresh200 import validate_completed_outer
    from verl.trainer.ppo.ray_trainer import reduce_metrics
    if trainer.use_critic or getattr(trainer.baseline_controller, 'ranker', None):
        raise ValueError('Prepared C1 replay must not omit a critic or learned ranker update')
    batch.meta_info['tree_partial_context'] = partial_context(trainer)
    started = time.monotonic()
    try:
        output = trainer.actor_rollout_wg.update_actor(batch)
    except Exception as error:
        try:
            evidence = preserve_unmatched_actor_failure(trainer, batch, error)
            print('ABLATION_ZERO_WRITE_REPLAY_UNMATCHED_EVIDENCE=' + evidence, flush=True)
        except Exception as evidence_error:
            print('ABLATION_ZERO_WRITE_REPLAY_EVIDENCE_FAILURE=' + repr(evidence_error), flush=True)
        raise
    actor_seconds = time.monotonic() - started
    if stop_on_partial_actor_result(trainer, batch, output):
        raise RuntimeError('Prepared first actor replay did not complete its native opportunity ledger')
    validate_completed_result(trainer, output)
    metrics = reduce_metrics(output.meta_info['metrics'])
    trainer.baseline_controller.after_actor(batch, metrics)
    validate_completed_outer(metrics, trainer.global_steps)
    record_completed_result(trainer, output)
    trainer.actual_outer_updates += 1
    metrics.update({'training/actual_outer_updates': 1, 'training/attempt_id': 1,
        'training/refused_attempts': 0, 'training/consumed_sampling_rounds': 1,
        'training/zero_write_sampling_rounds': trainer.zero_write_sampling_rounds,
        'training/optimizer_write_rounds': 1-trainer.zero_write_sampling_rounds,
        'training/normal_guard_rounds': trainer.normal_guard_rounds,
        'training/actual_adam_calls_per_rank': output.meta_info['tree_fresh_actor_result']['ranks'][0]['total_optimizer_calls'],
        'training/reused_saved_prepared_actor_batch': 1,
        'training/replayed_round_timing_excludes_original_sampling': 1,
        'timing_s/update_actor': actor_seconds, 'timing_s/step': time.monotonic() - started})
    return metrics


def migrate_trainer(trainer):
    from search_r1.baselines.training_state import verified_state_file, atomic_save, digest, rng_state
    if getattr(trainer, '_ablation_zero_write_replay_started', False):
        raise ValueError('Prepared actor batch may be replayed only once in one live trainer')
    value = manifest()
    saved_driver = verified_state_file(value['driver'])
    old_initialization = restore_partial_driver(trainer, saved_driver, value)
    if digest(rng_state()) != digest(saved_driver['rng']):
        raise ValueError('Live saved driver RNG differs before prepared batch replay')
    batch = prepared_batch(verified_state_file(value['actor_inputs']), value)
    trainer._guard_round_contract_active = trainer.acceptance_released = True
    trainer._ablation_zero_write_replay_started = True
    record = dict(protocol=PROTOCOL, status='ZERO_WRITE_PREPARED_BATCH_REPLAY_STARTED', time=time.time(),
        parent_manifest=os.environ[MANIFEST_ENV], parent_manifest_sha256=os.environ[SHA_ENV],
        parent_driver_sha256=value['driver']['sha256'], parent_actor_inputs_sha256=value['actor_inputs']['sha256'],
        parent_official_initialization=old_initialization, observed_live_official_initialization=trainer.fresh_initialization_provenance,
        consumed_cursor=512, consumed_epoch=0, reused_rows=len(batch), reused_attempt_id=1,
        driver_sampler_controller_rng_restored=True, worker_state_checkpoint_restored=False,
        worker_cuda_rng_restored=False, vllm_executor_rng_restored=False, repeated_generation_calls=0,
        prior_committed_adam_calls_per_rank={str(r): 0 for r in range(8)},
        prior_scheduler_steps_per_rank={str(r): 0 for r in range(8)},
        scope='Proven pre-update refusal: configured official model reload with observed empty Adam/scheduler; saved driver and full actor batch reused; no full-worker or bitwise future-run restoration claim')
    directory = Path(os.environ['SEARCH_R1_BASELINE_OUTPUT'])
    atomic_save(record, directory / 'zero-write-replay-started.json', json_format=True)
    print('ABLATION_ZERO_WRITE_REPLAY_START=' + json.dumps(record, sort_keys=True), flush=True)
    metrics = complete_prepared_outer(trainer, batch)
    trainer.logger.log(data=metrics, step=1)  # Existing logger hook durably saves native rolling1.
    record = dict(record, status='ZERO_WRITE_PREPARED_BATCH_REPLAY_COMPLETED', time=time.time(),
        completed_outer=trainer.actual_outer_updates, next_outer=2,
        actual_adam_calls_per_rank=copy.deepcopy(trainer.completed_optimizer_calls_per_rank))
    trainer.zero_write_operational_recovery_provenance = copy.deepcopy(record)
    atomic_save(record, directory / 'zero-write-replay-completed.json', json_format=True)
    print('ABLATION_ZERO_WRITE_REPLAY_COMPLETE=' + json.dumps(record, sort_keys=True), flush=True)
    return record
