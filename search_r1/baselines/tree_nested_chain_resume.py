"""Only the fresh nested 0bef chain's complete native1 boundary may continue.

The failed native2 save does not provide a resumable boundary. Its real two
completed outers / forty Adam writes remain evidence, while this child restores
one outer / twenty writes and recomputes outer2. Executor/vLLM is restarted.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import stat

PROTOCOL = 'tree_nested_fresh200_native1_oom_continuation_v1'
PARENT_COMMIT = '0befbaf99fc44942105831b3fbec94e4bf1b8ccb'
PARENT_JOB = 'not-slurm;systemd:' + PARENT_COMMIT
PARENT_RUN = 'tree-nested-branch-fresh200-' + PARENT_JOB
PARENT_ROOT = Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs') / PARENT_RUN
PARENT_MANIFEST = PARENT_ROOT / 'checkpoints/training_state/rolling-completed/global_step_1/manifest.json'
PARENT_MANIFEST_SHA256 = '90f0420ccdc30a3e15245861a73b405f3f760d333722314d08b96b95765f9a1f'
PARENT_CONFIG = PARENT_ROOT / 'resolved.yaml'
PARENT_CONFIG_SHA256 = '2a317628c497bba932698bf6d7bf64ca3503233d32080a389a36e1f69b362243'
PARENT_FAILED_ATTEMPT = PARENT_ROOT / 'checkpoints/training_state/attempt-results/attempt-000002.json'
PARENT_FAILED_ATTEMPT_SHA256 = '6913b708245d63311b0e6931b057023e610f979fd7c0bd72ca45089797a673c4'
PARENT_FAILED_TERMINAL = PARENT_ROOT / 'run-result.json'
PARENT_FAILED_TERMINAL_SHA256 = '1a9ebd2a3edebce70ed1717cced93ec30823205b4a7d7cbbcded695af05f8194'
PARENT_FAULT_SNAPSHOT = PARENT_ROOT / 'output/recovery-fault-20261014.json'
PARENT_FAULT_SNAPSHOT_SHA256 = '39594f1bdd8c9604663388eda03c310a7cb091f09774c8b04a2577dec9651ed1'
PARENT_COMPLETED_STEP, PARENT_OPTIMIZER_CALLS, PARENT_CURSOR = 1, 20, 512
ENV_PREFIX = 'TREE_NESTED_CHAIN_RESUME_'
ENV_FIELDS = {'AUTHORITY', 'MANIFEST', 'MANIFEST_SHA256', 'CONFIG', 'CONFIG_SHA256',
              'FAILED_ATTEMPT', 'FAILED_ATTEMPT_SHA256', 'FAILED_TERMINAL',
              'FAILED_TERMINAL_SHA256', 'FAULT_SNAPSHOT', 'FAULT_SNAPSHOT_SHA256'}
COUNTS = {str(rank): 20 for rank in range(8)}


def require(condition, message):
    if not condition:
        raise ValueError('Nested native1 continuation: ' + message)


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
            'all eleven explicit entry fields are required')
    require(environ[ENV_PREFIX+'AUTHORITY'] == PROTOCOL, 'scope differs')
    for suffix, path, sha in [('MANIFEST', PARENT_MANIFEST, PARENT_MANIFEST_SHA256),
                              ('CONFIG', PARENT_CONFIG, PARENT_CONFIG_SHA256),
                              ('FAILED_ATTEMPT', PARENT_FAILED_ATTEMPT, PARENT_FAILED_ATTEMPT_SHA256),
                              ('FAILED_TERMINAL', PARENT_FAILED_TERMINAL, PARENT_FAILED_TERMINAL_SHA256),
                              ('FAULT_SNAPSHOT', PARENT_FAULT_SNAPSHOT, PARENT_FAULT_SNAPSHOT_SHA256)]:
        require(Path(environ[ENV_PREFIX+suffix]) == path
                and environ[ENV_PREFIX+suffix+'_SHA256'] == sha, 'only pinned 0bef native1 inputs are allowed')
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
    require(set(spec) == {'protocol', 'parent_manifest', 'parent_config', 'failed_attempt', 'failed_terminal', 'fault_snapshot'}, 'credential fields differ')
    require(spec['protocol'] == PROTOCOL, 'package continuation protocol differs')
    values = {'AUTHORITY': PROTOCOL}
    for suffix, key in [('MANIFEST', 'parent_manifest'), ('CONFIG', 'parent_config'), ('FAILED_ATTEMPT', 'failed_attempt'),
                        ('FAILED_TERMINAL', 'failed_terminal'), ('FAULT_SNAPSHOT', 'fault_snapshot')]:
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
    read_pinned(PARENT_FAILED_ATTEMPT, PARENT_FAILED_ATTEMPT_SHA256)
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
    expected = dict(job_id=PARENT_JOB, run_id=PARENT_RUN, code_commit=PARENT_COMMIT,
        config_sha256=PARENT_CONFIG_SHA256, completed_step=1, actual_update_limit=200)
    require(manifest['status'] == 'COMPLETE_NATIVE_ROLLING_STATE_SAVED'
            and manifest['snapshot_kind'] == 'rolling_completed_outer'
            and all(identity.get(key) == value for key, value in expected.items()), 'mother chain identity differs')
    for key, env in [('data_identity', 'SEARCH_R1_BASELINE_DATA_IDENTITY'), ('model_identity', 'SEARCH_R1_MODEL_IDENTITY')]:
        require(identity[key] == os.environ[env], 'mother ' + key + ' differs')
    expected_guard = dict(protocol='tree_guard_continue_fresh_rollout_v1', training_attempts=1,
        completed_outer_updates=1, refused_attempts=0, guard_refused_attempts=0,
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
                and record['actual_total_optimizer_calls'] == 20 and record['observed_adam_steps'] == [20]
                and record['scheduler_last_epoch'] == 1 and record['full_worker_state_finite'] is True,
                'mother rank state/count differs')
    require(manifest['driver']['next_step'] == 2 and manifest['exact_executor_resume_verified'] is False,
            'mother next boundary or executor scope differs')
    return manifest


def failed_terminal_evidence():
    terminal = json.loads(read_pinned(PARENT_FAILED_TERMINAL, PARENT_FAILED_TERMINAL_SHA256))
    fault = json.loads(read_pinned(PARENT_FAULT_SNAPSHOT, PARENT_FAULT_SNAPSHOT_SHA256))
    observed = dict(line.split('=', 1) for line in fault['systemd'].splitlines() if '=' in line)
    require(terminal['status'] == 'FAILED' and terminal['job_id'] == PARENT_JOB
            and terminal['code_commit'] == PARENT_COMMIT and terminal['own_cgroup_live_residual_empty'] is True
            and terminal['actual_progress_evidence']['actual_completed_outer_from_attempt_results'] == 2
            and terminal['actual_progress_evidence']['completed_actual_optimizer_calls_per_rank'] == {str(r):40 for r in range(8)}
            and observed.get('MainPID') == '0' and observed.get('Result') == 'oom-kill'
            and observed.get('ExecMainStatus') == '1' and observed.get('ActiveState') == 'failed'
            and observed.get('MemoryMax') == str(256*1024**3)
            and fault['run-result.json']['path'] == str(PARENT_FAILED_TERMINAL)
            and fault['run-result.json']['sha256'] == PARENT_FAILED_TERMINAL_SHA256
            and fault['checkpoints/training_state/attempt-results/attempt-000002.json']['sha256'] == PARENT_FAILED_ATTEMPT_SHA256,
            'mother must be genuinely terminated by CPU cgroup OOM with actual outer2 evidence')
    return dict(terminal_path=str(PARENT_FAILED_TERMINAL), terminal_sha256=PARENT_FAILED_TERMINAL_SHA256,
        fault_snapshot_path=str(PARENT_FAULT_SNAPSHOT), fault_snapshot_sha256=PARENT_FAULT_SNAPSHOT_SHA256,
        original_main_pid=0, original_systemd_result='oom-kill', original_exit_status=1,
        original_cpu_memory_max_bytes=256*1024**3, original_native2_save_complete=False)


def failed_save_evidence(manifest):
    terminal = failed_terminal_evidence()
    sha = PARENT_FAILED_ATTEMPT_SHA256
    value = json.loads(read_pinned(PARENT_FAILED_ATTEMPT, sha))
    result = value['all_rank_result']; ranks = result['ranks']
    require(value['status'] == 'COMPLETED_OPPORTUNITIES_OUTER_CREDITED'
            and value['attempt_id'] == value['accepted_outer'] == result['accepted_outer'] == 2
            and result['completed_outer_before'] == 1
            and all(result['identity'].get(k) == manifest['identity'][k] for k in
                    ('job_id', 'run_id', 'code_commit', 'config_sha256', 'data_identity', 'model_identity'))
            and len(ranks) == 8 and {r['rank'] for r in ranks} == set(range(8))
            and all(r['optimizer_calls'] == 20 and r['total_optimizer_calls'] == 40
                    and r['scheduler_before'] == 1 and r['scheduler_after'] == 2 for r in ranks),
            'actual mother outer2/forty-write evidence differs')
    require(not (PARENT_ROOT/'checkpoints/training_state/rolling-completed/global_step_2/manifest.json').exists(),
            'a complete native2 appeared; do not silently recompute a newly saved boundary')
    return dict(path=str(PARENT_FAILED_ATTEMPT), sha256=sha, actual_completed_outer_before_failure=2,
        actual_optimizer_calls_per_rank_before_failure={str(r): 40 for r in range(8)},
        failure_component='CPU_cgroup_OOM_during_native2_save', recovered_completed_outer=1,
        recovered_optimizer_calls_per_rank=COUNTS, lost_outer2_credit_in_child=0,
        outer2_must_be_recomputed=True, failed_attempt_in_effective_history=False,
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
            and saved['optimizer']['state'] and saved['scheduler']['last_epoch'] == 1
            and all(int(v['step']) == 20 for v in saved['optimizer']['state'].values()), 'serialized worker state differs')
    prepare_cold_fsdp_layout(worker.actor_module_fsdp, worker._is_offload_param, worker._is_offload_grad)
    restore_worker(worker.actor_module_fsdp, worker.actor_optimizer, worker.actor_lr_scheduler, saved, record['identity'], cold=True)
    del saved
    worker.actor._fresh_total_committed_optimizer_calls = 20
    live = worker_snapshot(worker.actor_module_fsdp, worker.actor_optimizer, worker.actor_lr_scheduler, record['identity'], cuda=True)
    live_logical = digest(live); del live
    require(live_logical == logical and worker.actor_lr_scheduler.last_epoch == 1
            and sorted({int(v['step']) for v in worker.actor_optimizer.state.values()}) == [20], 'actual loaded worker differs')
    return dict(status='COLD_NESTED_NATIVE1_WORKER_LOADED', protocol=PROTOCOL, rank=worker.rank, world_size=8,
        parent_identity=record['identity'], parent_manifest_sha256=PARENT_MANIFEST_SHA256,
        state_path=record['path'], state_file_sha256=record['sha256'], state_file_bytes=record['bytes'],
        actual_file_sha256_verified=True, serialized_logical_state_sha256=logical,
        actual_loaded_logical_state_sha256=live_logical, actual_cold_native_state_equal=True,
        adam_steps=[20], scheduler_step=1, actual_total_optimizer_calls=20,
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
            and all(r['status'] == 'COLD_NESTED_NATIVE1_WORKER_LOADED'
                    and r['actual_cold_native_state_equal'] is True and r['actual_file_sha256_verified'] is True
                    and r['parent_manifest_sha256'] == PARENT_MANIFEST_SHA256 and r['scheduler_step'] == 1
                    and r['adam_steps'] == [20] and r['actual_total_optimizer_calls'] == 20
                    and r['child_code_commit'] == child_commit for r in workers), 'loaded worker receipts incomplete')
    record = state_record(manifest['driver'], 'driver.pt'); saved = verified_state_file(record)
    require(digest(saved) == record['logical_state_sha256'] and saved['identity'] == manifest['identity']
            and saved['completed_step'] == saved['actual_outer_updates'] == 1 and saved['next_step'] == 2
            and saved['sampler']['batch_size'] == 512 and saved['sampler']['cursor'] == 512
            and saved['sampler']['epoch'] == 0 and saved.get('recovery_provenance') is None,
            'actual mother driver boundary/cursor differs or has historical recovery')
    restore_driver(trainer, saved, manifest['identity'])
    require(digest(driver_snapshot(trainer, manifest['identity'])) == record['logical_state_sha256'],
            'actual loaded driver/sampler/RNG/Tree RNG differs')
    provenance = dict(protocol=PROTOCOL, same_fresh_nested_chain=True,
        parent_identity=manifest['identity'], parent_manifest_path=str(PARENT_MANIFEST),
        parent_manifest_sha256=PARENT_MANIFEST_SHA256, parent_completed_step=1,
        parent_cursor=512, parent_epoch=0, new_batch_size=512, semantic_configuration=semantics,
        child_code_commit=child_commit, child_config_sha256=os.environ['SEARCH_R1_BASELINE_CONFIG_SHA256'],
        child_job_id=os.environ['SLURM_JOB_ID'], child_run_id=os.environ['SEARCH_R1_BASELINE_RUN_ID'],
        failed_parent_outer2=failure, automatic_resume_supported=False,
        executor_equivalence=False, exact_executor_resume=False)
    trainer.recovery_provenance = cpu_copy(provenance)
    trainer.nested_chain_resume_provenance = cpu_copy(provenance)
    # Keep the unchanged fifth-outer human continuation gate in the live child.
    trainer.acceptance_released = False
    receipt = dict(status='COLD_NESTED_NATIVE1_STATE_LOADED', protocol=PROTOCOL, provenance=provenance,
        workers=workers, driver=dict(state_path=record['path'], state_file_sha256=record['sha256'],
            actual_file_sha256_verified=True, actual_cold_driver_state_equal=True,
            completed_outer_updates=1, next_outer_update=2, sampler_cursor=512, sampler_epoch=0,
            training_attempts=trainer.training_attempts, completed_optimizer_calls_per_rank=trainer.completed_optimizer_calls_per_rank),
        cold_official_initialization=trainer.fresh_initialization_provenance,
        executor_equivalence=False, exact_executor_resume=False, initial_evaluation_added=False)
    atomic_save(receipt, Path(os.environ['SEARCH_R1_BASELINE_OUTPUT'])/'nested-native1-loaded.json', json_format=True)
    print('TREE_COLD_NESTED_NATIVE1_STATE_LOADED=1;NEXT_OUTER=2;ADAM_EACH_RANK=20;CURSOR=512', flush=True)
    return receipt
