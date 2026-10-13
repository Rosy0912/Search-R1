"""Pinned native5 continuation of the existing nested Tree200 goal.

Restore actual actor/Adam/scheduler and driver/sampler/RNG/tree RNG. Preserve
origin1 + ba9 2..5, redo failed6; executor/vLLM restart is not exact resume.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import stat

PROTOCOL = 'tree_nested_fresh200_native5_ray_thread_continuation_v1'
STATE_STATUS = 'COLD_NESTED_NATIVE5_STATE_LOADED'
WORKER_STATUS = 'COLD_NESTED_NATIVE5_WORKER_LOADED'
RECEIPT_NAME = 'nested-native5-loaded.json'
ORIGIN_COMMIT = '0befbaf99fc44942105831b3fbec94e4bf1b8ccb'
ORIGIN_RUN = 'tree-nested-branch-fresh200-not-slurm;systemd:' + ORIGIN_COMMIT
ORIGIN_ROOT = Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs') / ORIGIN_RUN
ORIGIN_CONFIG_SHA256 = '2a317628c497bba932698bf6d7bf64ca3503233d32080a389a36e1f69b362243'
PARENT_COMMIT = 'ba9f6627ec9a4e9749830d7b6acca53f6feec6b4'
PARENT_JOB = 'not-slurm;systemd:' + PARENT_COMMIT
PARENT_RUN = 'tree-nested-branch-fresh200-' + PARENT_JOB
PARENT_ROOT = Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs') / PARENT_RUN
PARENT_MANIFEST = PARENT_ROOT / 'checkpoints/training_state/global_step_5/manifest.json'
PARENT_MANIFEST_SHA256 = 'b112caffda6514c34a07f2f6b6ecc62096c0717eb09ebae46aac71c6b9dfe09a'
PARENT_CONFIG = PARENT_ROOT / 'resolved.yaml'
PARENT_CONFIG_SHA256 = '22beaec8aa2b449b496754aacfd9b1e7c89009c9c27a8eff3aef92b768b28b59'
PARENT_FAILED_ROLLOUT = PARENT_ROOT / 'output/training-rollout-cost/attempt-000006.json'
PARENT_FAILED_ROLLOUT_SHA256 = '585fc84e46d3a05a75e33e7cf11d363cf276ae99e0060947abb167d1452a1c3f'
PARENT_FAILED_TERMINAL = PARENT_ROOT / 'run-result.json'
PARENT_FAILED_TERMINAL_SHA256 = '1c20f2d7eff01a56cb6f713556552184383b0855e31f427ac505258a54820a1f'
PARENT_FAULT_SNAPSHOT = PARENT_ROOT / 'output/ray-fault-20261014-outer6-terminal.json'
PARENT_FAULT_SNAPSHOT_SHA256 = '55f013d537819d4c7b84eafa0790e8de205fe7d5256e6f79945aae60610e5718'
ANCESTRY_RECEIPT = PARENT_ROOT / 'output/nested-native1-loaded.json'
ANCESTRY_RECEIPT_SHA256 = 'dc6f1f065a21d048b21a9462b55f992f9762b07f9b00633cb7028af99443f096'
PARENT_CONTINUATION = PARENT_ROOT / 'output/acceptance-step5/human-authorized-continuation.json'
PARENT_CONTINUATION_SHA256 = '7acc5fd519604c8977f2bc31afa9b223b22dfd25bfa8002f35375c35aea8d333'
PARENT_COMPLETED_STEP, PARENT_OPTIMIZER_CALLS, PARENT_CURSOR = 5, 100, 2560
ENV_PREFIX = 'TREE_NESTED_CHAIN_RESUME_'
INPUTS = [('MANIFEST', 'parent_manifest', PARENT_MANIFEST, PARENT_MANIFEST_SHA256),
          ('CONFIG', 'parent_config', PARENT_CONFIG, PARENT_CONFIG_SHA256),
          ('FAILED_ROLLOUT', 'failed_rollout', PARENT_FAILED_ROLLOUT, PARENT_FAILED_ROLLOUT_SHA256),
          ('FAILED_TERMINAL', 'failed_terminal', PARENT_FAILED_TERMINAL, PARENT_FAILED_TERMINAL_SHA256),
          ('FAULT_SNAPSHOT', 'fault_snapshot', PARENT_FAULT_SNAPSHOT, PARENT_FAULT_SNAPSHOT_SHA256),
          ('ANCESTRY_RECEIPT', 'ancestry_receipt', ANCESTRY_RECEIPT, ANCESTRY_RECEIPT_SHA256),
          ('PARENT_CONTINUATION', 'parent_continuation', PARENT_CONTINUATION, PARENT_CONTINUATION_SHA256)]
ENV_FIELDS = {'AUTHORITY'} | {name for suffix, _, _, _ in INPUTS for name in (suffix, suffix+'_SHA256')}
COUNTS = {str(rank): PARENT_OPTIMIZER_CALLS for rank in range(8)}


def ancestor_segments():
    return [dict(source_run=str(ORIGIN_ROOT), run_id=ORIGIN_RUN, code_commit=ORIGIN_COMMIT,
                 config_sha256=ORIGIN_CONFIG_SHA256, first_outer=1, last_outer=1),
            dict(source_run=str(PARENT_ROOT), run_id=PARENT_RUN, code_commit=PARENT_COMMIT,
                 config_sha256=PARENT_CONFIG_SHA256, first_outer=2, last_outer=5)]


def require(condition, message):
    if not condition:
        raise ValueError('Nested native5 continuation: ' + message)


def enabled(environ=None):
    environ = os.environ if environ is None else environ
    return any(key.startswith(ENV_PREFIX) for key in environ)


def read_pinned(path, expected):
    path = Path(path)
    require(stat.S_ISREG(path.lstat().st_mode), 'ordinary preserved input required: ' + str(path))
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as stream:
        before = os.fstat(stream.fileno()); raw = stream.read(); after = os.fstat(stream.fileno())
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
            and hashlib.sha256(raw).hexdigest() == expected, 'preserved input bytes changed: ' + str(path))
    return raw


def validate_entry(config=None, environ=None):
    environ = os.environ if environ is None else environ
    if not enabled(environ):
        return False
    require(not any(key.startswith(('TREE_RECOVERY_', 'TREE_CURRENT_CHAIN_RESUME_')) for key in environ),
            'historical/current3339 recovery cannot be mixed into the nested chain')
    require({key[len(ENV_PREFIX):] for key in environ if key.startswith(ENV_PREFIX)} == ENV_FIELDS,
            'all explicit native5 entry fields are required')
    require(environ[ENV_PREFIX+'AUTHORITY'] == PROTOCOL, 'scope differs')
    for suffix, _, path, sha in INPUTS:
        require(Path(environ[ENV_PREFIX+suffix]) == path
                and environ[ENV_PREFIX+suffix+'_SHA256'] == sha, 'only pinned ba9 native5 inputs are allowed')
    require(socket.gethostname().split('.')[0] == 'h100-10', 'only the retained H10018 h100-10 host is permitted')
    if config is not None:
        validate_semantic_config(config)
    return True


def install_package_environment(package, environ=None):
    """Runtime package binds the new identity to immutable mother-chain inputs."""
    environ = os.environ if environ is None else environ
    spec = package.get('nested_native_continuation')
    if spec is None:
        require(not enabled(environ), 'native continuation env requires its package credential')
        return False
    require(set(spec) == {'protocol'} | {key for _, key, _, _ in INPUTS}, 'credential fields differ')
    require(spec['protocol'] == PROTOCOL, 'package continuation protocol differs')
    values = {'AUTHORITY': PROTOCOL}
    for suffix, key, _, _ in INPUTS:
        descriptor = spec[key]
        require(set(descriptor) == {'path', 'sha256'}, 'actual input descriptor required')
        values[suffix] = descriptor['path']; values[suffix+'_SHA256'] = descriptor['sha256']
    expected = {ENV_PREFIX+key: value for key, value in values.items()}
    existing = {key: value for key, value in environ.items() if key.startswith(ENV_PREFIX)}
    require(not existing or existing == expected, 'inherited continuation env differs from package')
    environ.update(expected)
    validate_entry(environ=environ)
    read_pinned(PARENT_MANIFEST, PARENT_MANIFEST_SHA256)
    read_pinned(PARENT_CONFIG, PARENT_CONFIG_SHA256)
    read_pinned(PARENT_FAILED_ROLLOUT, PARENT_FAILED_ROLLOUT_SHA256)
    failed_terminal_evidence()
    return True


def validate_semantic_config(config, trainer_injected=False):
    from omegaconf import OmegaConf
    from .tree_fresh200 import validate_fresh200_config
    validate_fresh200_config(config)
    parent = OmegaConf.to_container(OmegaConf.create(read_pinned(PARENT_CONFIG, PARENT_CONFIG_SHA256).decode()), resolve=True)
    child = OmegaConf.to_container(config, resolve=True)
    # Only these two own output locations change. No algorithm/resource/sampling
    # field is removed from the comparison. Trainer injects its unused critic
    # schedule deterministically in _create_dataloader; normalize that exact value.
    if trainer_injected:
        require(parent['critic']['optim']['total_training_steps'] == -1
                and child['critic']['optim']['total_training_steps'] == 201, 'critic schedule injection differs')
        child['critic']['optim']['total_training_steps'] = -1
    changes = {}
    for section, key, suffix in [('trainer', 'default_local_dir', 'checkpoints'),
                                ('baselines.holdout', 'output_dir', 'output/common-data')]:
        parent_part, child_part = parent, child
        for part in section.split('.'):
            parent_part, child_part = parent_part[part], child_part[part]
        old, new = parent_part[key], child_part[key]
        require(Path(old) == PARENT_ROOT / suffix, 'parent own output binding differs')
        child_root = Path(os.environ['SEARCH_R1_BASELINE_OUTPUT']).parent
        require(child_root != PARENT_ROOT and Path(new) == child_root / suffix,
                'new output must bind to distinct current child run')
        changes[section+'.'+key] = dict(parent=old, child=new)
        parent_part[key] = child_part[key] = '<same-chain-own-output>'
    require(parent == child, 'resolved semantic configuration differs beyond the two own output locations')
    return dict(parent_config_sha256=PARENT_CONFIG_SHA256,
                semantic_config_sha256=hashlib.sha256(json.dumps(parent, sort_keys=True, allow_nan=False).encode()).hexdigest(),
                allowed_output_identity_changes=changes, trainer_critic_schedule_injection=trainer_injected)


def parent_manifest():
    require(validate_entry(), 'explicit entry absent')
    manifest = json.loads(read_pinned(PARENT_MANIFEST, PARENT_MANIFEST_SHA256))
    identity = manifest['identity']
    expected = dict(job_id=PARENT_JOB, code_commit=PARENT_COMMIT,
        config_sha256=PARENT_CONFIG_SHA256, completed_step=5, actual_update_limit=200)
    require(manifest['status'] == 'COMPLETE_SAME_WORLD_STATE_SAVED'
            and all(identity.get(key) == value for key, value in expected.items()), 'mother chain identity differs')
    for key, env in [('data_identity', 'SEARCH_R1_BASELINE_DATA_IDENTITY'), ('model_identity', 'SEARCH_R1_MODEL_IDENTITY')]:
        require(identity[key] == os.environ[env], 'mother ' + key + ' differs')
    expected_guard = dict(protocol='tree_guard_continue_fresh_rollout_v1', training_attempts=5,
        completed_outer_updates=5, refused_attempts=0, guard_refused_attempts=0,
        partial_optimizer_calls_per_rank={}, completed_optimizer_calls_per_rank=COUNTS,
        total_optimizer_calls_per_rank=COUNTS, partial_evidence=[])
    require(identity.get('guard_continue_accounting') == expected_guard, 'mother effective accounting differs')
    records = manifest['worker_states']
    require(len(records) == 8 and {r['identity']['rank'] for r in records} == set(range(8)), 'eight native ranks required')
    for record in records:
        rank_id = record['identity']
        require(rank_id == dict(identity, rank=rank_id['rank'], world_size=8,
                    cuda_visible_devices=rank_id.get('cuda_visible_devices'))
                and isinstance(rank_id['cuda_visible_devices'], str) and bool(rank_id['cuda_visible_devices'])
                and record['actual_total_optimizer_calls'] == 100 and record['observed_adam_steps'] == [100]
                and record['scheduler_last_epoch'] == 5 and record['full_worker_state_finite'] is True
                and record['exact_same_world_reload_verified'] is True,
                'mother rank state/count differs')
    require(manifest['driver']['next_step'] == 6 and manifest['driver']['exact_reload_verified'] is True
            and Path(manifest['hf_checkpoint']) == PARENT_ROOT/'checkpoints/actor/global_step_5',
            'mother next boundary or executor scope differs')
    return manifest


def ancestry_provenance():
    receipt = json.loads(read_pinned(ANCESTRY_RECEIPT, ANCESTRY_RECEIPT_SHA256))
    provenance = receipt['provenance']
    require(receipt['status'] == 'COLD_NESTED_NATIVE1_STATE_LOADED'
            and receipt['protocol'] == 'tree_nested_fresh200_native1_oom_continuation_v1'
            and provenance['same_fresh_nested_chain'] is True and provenance['parent_completed_step'] == 1
            and provenance['parent_identity']['code_commit'] == ORIGIN_COMMIT
            and provenance['parent_manifest_sha256'] == '90f0420ccdc30a3e15245861a73b405f3f760d333722314d08b96b95765f9a1f'
            and provenance['child_code_commit'] == PARENT_COMMIT
            and provenance['child_config_sha256'] == PARENT_CONFIG_SHA256
            and provenance['child_job_id'] == PARENT_JOB and provenance['child_run_id'] == PARENT_RUN
            and provenance['failed_parent_outer2']['failed_attempt_in_effective_history'] is False
            and receipt['driver']['completed_outer_updates'] == 1
            and receipt['driver']['completed_optimizer_calls_per_rank'] == {str(r):20 for r in range(8)}
            and len(receipt['workers']) == 8
            and all(r['actual_cold_native_state_equal'] is True for r in receipt['workers']),
            'actual original native1 migration receipt differs')
    return provenance


def parent_continuation_evidence():
    value = json.loads(read_pinned(PARENT_CONTINUATION, PARENT_CONTINUATION_SHA256))
    ready_path = PARENT_ROOT/'output/acceptance-step5/ready.json'
    ready = json.loads(read_pinned(ready_path, value['ready_sha256']))
    expected = dict(job_id=PARENT_JOB, code_commit=PARENT_COMMIT, config_sha256=PARENT_CONFIG_SHA256)
    require(value['status'] == 'HUMAN_AUTHORIZED_SAME_INSTANCE_6_TO_200'
            and value['run_id'] == PARENT_RUN and value['completed_outer_updates'] == 5
            and value['next_outer_update'] == 6 and value['actual_update_limit'] == 200
            and value['same_instance_continuation'] is True and value['independent_acceptance'] is False
            and value['manifest_sha256'] == '514c4473def43e3ce781899d4b62958eee51d752383806420a3ed13a029b0d94'
            and value['human_training_authority_sha256'] == '12a0416976418908d097567dc6a95277e644c0011e502fcca8143d0e32a29acb'
            and all(value[k] == v and ready[k] == v for k,v in expected.items())
            and ready['completed_outer_updates'] == ready['completed_step'] == 5
            and ready['next_outer_update'] == 6 and ready['actual_update_limit'] == 200
            and ready['full_state_manifest'] == str(PARENT_MANIFEST)
            and ready['full_state_manifest_sha256'] == PARENT_MANIFEST_SHA256
            and ready['full_state_reload_verified'] is True
            and ready['guard_continue_accounting'] == parent_manifest()['identity']['guard_continue_accounting']
            and ready['hf_identity'] == parent_manifest()['hf_identity'],
            'actual parent fifth-outer live human continuation differs')
    return dict(path=str(PARENT_CONTINUATION), sha256=PARENT_CONTINUATION_SHA256,
                ready_path=str(ready_path), ready_sha256=value['ready_sha256'],
                independent_acceptance=False, original_dev5_repeated=False)


def failed_terminal_evidence():
    terminal = json.loads(read_pinned(PARENT_FAILED_TERMINAL, PARENT_FAILED_TERMINAL_SHA256))
    fault = json.loads(read_pinned(PARENT_FAULT_SNAPSHOT, PARENT_FAULT_SNAPSHOT_SHA256))
    observed = dict(line.split('=', 1) for line in fault['systemd'].splitlines() if '=' in line)
    bound = {entry['path']: entry for entry in fault['descriptors']}
    require(terminal['status'] == 'FAILED' and terminal['job_id'] == PARENT_JOB
            and terminal['code_commit'] == PARENT_COMMIT and terminal['own_cgroup_live_residual_empty'] is True
            and terminal['actual_progress_evidence']['actual_completed_outer_from_attempt_results'] == 5
            and terminal['actual_progress_evidence']['completed_actual_optimizer_calls_per_rank'] == COUNTS
            and terminal['native_continuation_start']['receipt_sha256'] == ANCESTRY_RECEIPT_SHA256
            and observed.get('MainPID') == '0' and observed.get('Result') == 'exit-code'
            and observed.get('ExecMainStatus') == '1' and observed.get('ActiveState') == 'failed'
            and observed.get('MemoryMax') == str(384*1024**3) and observed.get('TasksMax') == '4096'
            and fault['actual_cgroup_exists'] is False and not fault['gpu'].strip()
            and not fault['port_listeners'].strip()
            and bound[str(PARENT_FAILED_TERMINAL)]['sha256'] == PARENT_FAILED_TERMINAL_SHA256
            and bound[str(PARENT_MANIFEST)]['sha256'] == PARENT_MANIFEST_SHA256
            and bound[str(ANCESTRY_RECEIPT)]['sha256'] == ANCESTRY_RECEIPT_SHA256,
            'mother must be genuinely terminated with complete native5 and own residuals empty')
    return dict(terminal_path=str(PARENT_FAILED_TERMINAL), terminal_sha256=PARENT_FAILED_TERMINAL_SHA256,
        fault_snapshot_path=str(PARENT_FAULT_SNAPSHOT), fault_snapshot_sha256=PARENT_FAULT_SNAPSHOT_SHA256,
        original_main_pid=0, original_systemd_result='exit-code', original_exit_status=1,
        failure_component='Ray_lazy_RPC_server_thread_creation_EAGAIN_HandleLocalGC',
        exact_pids_limit_cause_proven=False, original_cpu_memory_max_bytes=384*1024**3,
        original_tasks_max=4096, original_native6_save_complete=False)


def failed_save_evidence(manifest):
    terminal = failed_terminal_evidence()
    rollout = json.loads(read_pinned(PARENT_FAILED_ROLLOUT, PARENT_FAILED_ROLLOUT_SHA256))
    require(rollout['attempt_id'] == rollout['candidate_outer'] == 6
            and rollout['completed_outer_before'] == 5
            and rollout['code_commit'] == PARENT_COMMIT and rollout['job_id'] == PARENT_JOB,
            'actual failed sixth rollout binding differs')
    for suffix in ('global_step_6/manifest.json', 'rolling-completed/global_step_6/manifest.json',
                   'attempt-results/attempt-000006.json'):
        require(not (PARENT_ROOT/'checkpoints/training_state'/suffix).exists(),
                'complete sixth decision/state appeared; do not silently recompute')
    return dict(path=str(PARENT_FAILED_ROLLOUT), sha256=PARENT_FAILED_ROLLOUT_SHA256,
        actual_completed_outer_before_failure=5, recovered_completed_outer=5,
        recovered_optimizer_calls_per_rank=COUNTS, outer6_must_be_recomputed=True,
        failed_attempt_in_effective_history=False, unrecorded_partial_optimizer_calls=None,
        original_evidence_retained=True, failed_terminal=terminal, exact_executor_resume=False)


def state_record(record, filename):
    result = dict(record); expected = PARENT_MANIFEST.parent / filename
    require(Path(result['path']) == expected and not expected.is_symlink(), 'native state path leaves mother boundary')
    if 'bytes' in result:
        require(expected.stat().st_size == result['bytes'], 'native byte count differs')
    return result


def load_worker(worker):
    from .training_state import (digest, prepare_cold_fsdp_layout, restore_worker,
                                 verified_state_file, worker_snapshot)
    require(worker._is_actor and worker.config.actor.get('training_state_enabled') is True
            and worker.world_size == 8 and not worker.actor_optimizer.state
            and worker.actor_lr_scheduler.last_epoch == 0
            and worker.actor._fresh_total_committed_optimizer_calls == 0, 'cold native worker must be untouched')
    manifest = parent_manifest()
    record = state_record(next(r for r in manifest['worker_states'] if r['identity']['rank'] == worker.rank), f'rank-{worker.rank:02d}.pt')
    saved = verified_state_file(record); logical = digest(saved)
    require(logical == record['logical_state_sha256'] and saved['identity'] == record['identity']
            and saved['optimizer']['state'] and saved['scheduler']['last_epoch'] == 5
            and all(int(v['step']) == 100 for v in saved['optimizer']['state'].values()), 'serialized worker state differs')
    prepare_cold_fsdp_layout(worker.actor_module_fsdp, worker._is_offload_param, worker._is_offload_grad)
    restore_worker(worker.actor_module_fsdp, worker.actor_optimizer, worker.actor_lr_scheduler, saved, record['identity'], cold=True)
    del saved
    worker.actor._fresh_total_committed_optimizer_calls = 100
    live = worker_snapshot(worker.actor_module_fsdp, worker.actor_optimizer, worker.actor_lr_scheduler, record['identity'], cuda=True)
    live_logical = digest(live); del live
    require(live_logical == logical and worker.actor_lr_scheduler.last_epoch == 5
            and sorted({int(v['step']) for v in worker.actor_optimizer.state.values()}) == [100], 'actual loaded worker differs')
    return dict(status=WORKER_STATUS, protocol=PROTOCOL, rank=worker.rank, world_size=8,
        parent_identity=record['identity'], parent_manifest_sha256=PARENT_MANIFEST_SHA256,
        state_path=record['path'], state_file_sha256=record['sha256'], state_file_bytes=record['bytes'],
        actual_file_sha256_verified=True, serialized_logical_state_sha256=logical,
        actual_loaded_logical_state_sha256=live_logical, actual_cold_native_state_equal=True,
        adam_steps=[100], scheduler_step=5, actual_total_optimizer_calls=100,
        child_code_commit=os.environ['SEARCH_R1_BASELINE_COMMIT'],
        executor_equivalence=False, exact_executor_resume=False, automatic_resume_supported=False)


def migrate_trainer(trainer):
    from .training_state import (atomic_save, cpu_copy, digest, driver_snapshot,
                                 restore_driver, verified_state_file)
    require(validate_entry(), 'explicit entry absent')
    semantics = validate_semantic_config(trainer.config, trainer_injected=True)
    child_commit = os.environ['SEARCH_R1_BASELINE_COMMIT']
    require(re.fullmatch('[0-9a-f]{40}', child_commit) is not None and child_commit != PARENT_COMMIT,
            'child source identity must be explicit and distinct')
    require(len(trainer.train_dataset) == 169028 and len(trainer.val_dataset) == 512, 'dataset coverage differs')
    manifest = parent_manifest(); failure = failed_save_evidence(manifest)
    workers = trainer.actor_rollout_wg.load_nested_chain_training_state()
    require(len(workers) == 8 and {r['rank'] for r in workers} == set(range(8))
            and all(r['status'] == WORKER_STATUS
                    and r['actual_cold_native_state_equal'] is True and r['actual_file_sha256_verified'] is True
                    and r['parent_manifest_sha256'] == PARENT_MANIFEST_SHA256 and r['scheduler_step'] == 5
                    and r['adam_steps'] == [100] and r['actual_total_optimizer_calls'] == 100
                    and r['child_code_commit'] == child_commit for r in workers), 'loaded worker receipts incomplete')
    record = state_record(manifest['driver'], 'driver.pt'); saved = verified_state_file(record)
    require(digest(saved) == record['logical_state_sha256'] and saved['identity'] == manifest['identity']
            and saved['completed_step'] == saved['actual_outer_updates'] == 5 and saved['next_step'] == 6
            and saved['sampler']['batch_size'] == 512 and saved['sampler']['cursor'] == 2560
            and saved['sampler']['epoch'] == 0 and saved.get('recovery_provenance') == ancestry_provenance(),
            'actual mother driver boundary/cursor differs or has historical recovery')
    restore_driver(trainer, saved, manifest['identity'])
    require(digest(driver_snapshot(trainer, manifest['identity'])) == record['logical_state_sha256'],
            'actual loaded driver/sampler/RNG/Tree RNG differs')
    provenance = dict(protocol=PROTOCOL, same_fresh_nested_chain=True,
        parent_identity=manifest['identity'], parent_manifest_path=str(PARENT_MANIFEST),
        parent_manifest_sha256=PARENT_MANIFEST_SHA256, parent_completed_step=5,
        parent_cursor=2560, parent_epoch=0, new_batch_size=512, semantic_configuration=semantics,
        child_code_commit=child_commit, child_config_sha256=os.environ['SEARCH_R1_BASELINE_CONFIG_SHA256'],
        child_job_id=os.environ['SLURM_JOB_ID'], child_run_id=os.environ['SEARCH_R1_BASELINE_RUN_ID'],
        ancestry_provenance=ancestry_provenance(), ancestry_receipt_sha256=ANCESTRY_RECEIPT_SHA256,
        effective_ancestor_segments=ancestor_segments(),
        failed_parent_outer2=ancestry_provenance()['failed_parent_outer2'], failed_parent_outer6=failure,
        parent_fifth_continuation=parent_continuation_evidence(), automatic_resume_supported=False,
        executor_equivalence=False, exact_executor_resume=False)
    trainer.recovery_provenance = cpu_copy(provenance)
    trainer.nested_chain_resume_provenance = cpu_copy(provenance)
    # The complete parent already saved dev/HF/native5 and released its live fifth gate.
    trainer.acceptance_released = True
    receipt = dict(status=STATE_STATUS, protocol=PROTOCOL, provenance=provenance,
        workers=workers, driver=dict(state_path=record['path'], state_file_sha256=record['sha256'],
            actual_file_sha256_verified=True, actual_cold_driver_state_equal=True,
            completed_outer_updates=5, next_outer_update=6, sampler_cursor=2560, sampler_epoch=0,
            training_attempts=trainer.training_attempts, completed_optimizer_calls_per_rank=trainer.completed_optimizer_calls_per_rank),
        cold_official_initialization=trainer.fresh_initialization_provenance,
        executor_equivalence=False, exact_executor_resume=False, initial_evaluation_added=False)
    atomic_save(receipt, Path(os.environ['SEARCH_R1_BASELINE_OUTPUT'])/RECEIPT_NAME, json_format=True)
    print('TREE_COLD_NESTED_NATIVE5_STATE_LOADED=1;NEXT_OUTER=6;ADAM_EACH_RANK=100;CURSOR=2560', flush=True)
    return receipt
