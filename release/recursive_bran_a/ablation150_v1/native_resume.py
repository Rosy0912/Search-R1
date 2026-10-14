"""Full same-world restoration of this condition's explicit accepted boundary."""
import json
import os
from pathlib import Path

from .contracts import read_receipt, sha


def manifest():
    receipt = read_receipt(os.environ['RECURSIVE_ABLATION_RECEIPT'])
    path = Path(os.environ['RECURSIVE_ABLATION_RESUME_MANIFEST']).resolve()
    root = Path(receipt['base']) / 'checkpoints'
    if not path.is_relative_to(root) or path.is_symlink() or sha(path) != os.environ['RECURSIVE_ABLATION_RESUME_SHA256']:
        raise ValueError('Resume boundary must be this group’s explicit unchanged native state')
    value = json.loads(path.read_text())
    identity = value['identity']
    if (value['status'] != 'COMPLETE_SAME_WORLD_STATE_SAVED'
            or identity['config_sha256'] != receipt['config_sha256']
            or identity['actual_update_limit'] != 150
            or identity['data_identity'] != receipt['data_identity']
            or not 0 <= identity['completed_step'] < 150):
        raise ValueError('Wrong, incomplete, or terminal ablation state')
    if len(value['worker_states']) != 8 or {r['identity']['rank'] for r in value['worker_states']} != set(range(8)):
        raise ValueError('Full eight-rank state is required')
    for record in value['worker_states'] + [value['driver']]:
        member = Path(record['path'])
        if member.resolve().parent != path.parent or member.is_symlink() or not member.is_file():
            raise ValueError('Native member missing or outside the frozen checkpoint')
    return value


def load_worker(worker):
    from search_r1.baselines.training_state import (verified_state_file, digest, restore_worker,
        prepare_cold_fsdp_layout, worker_snapshot)
    value = manifest()
    record = next(r for r in value['worker_states'] if r['identity']['rank'] == worker.rank)
    if (worker.world_size != 8 or worker.actor_optimizer.state
            or worker.actor_lr_scheduler.last_epoch != 0
            or worker.actor._fresh_total_committed_optimizer_calls != 0):
        raise ValueError('Resume worker must start as untouched official cold initialization')
    saved = verified_state_file(record)
    expected = record['actual_total_optimizer_calls']
    if (digest(saved) != record['logical_state_sha256'] or saved['identity'] != record['identity']
            or saved['scheduler']['last_epoch'] != value['identity']['completed_step']
            or any(int(v['step']) != expected for v in saved['optimizer']['state'].values())):
        raise ValueError('Actual native Adam/scheduler/identity differs')
    prepare_cold_fsdp_layout(worker.actor_module_fsdp, worker._is_offload_param, worker._is_offload_grad)
    restore_worker(worker.actor_module_fsdp, worker.actor_optimizer, worker.actor_lr_scheduler,
                   saved, record['identity'], cold=True)
    del saved
    worker.actor._fresh_total_committed_optimizer_calls = expected
    actual = worker_snapshot(worker.actor_module_fsdp, worker.actor_optimizer,
                            worker.actor_lr_scheduler, record['identity'], cuda=True)
    if digest(actual) != record['logical_state_sha256']:
        raise ValueError('Live restored model/Adam/scheduler/RNG differs from full saved state')
    del actual
    return dict(rank=worker.rank, actual_state_equal=True, actual_total_optimizer_calls=expected,
                scheduler_step=worker.actor_lr_scheduler.last_epoch, parent_state_sha256=record['sha256'])


def migrate_trainer(trainer):
    from search_r1.baselines.training_state import verified_state_file, digest, restore_driver, driver_snapshot, atomic_save
    value = manifest()
    workers = trainer.actor_rollout_wg.load_current_chain_training_state()
    if len(workers) != 8 or {r['rank'] for r in workers} != set(range(8)) or not all(r['actual_state_equal'] for r in workers):
        raise ValueError('Incomplete real native worker restoration')
    saved = verified_state_file(value['driver'])
    if digest(saved) != value['driver']['logical_state_sha256'] or saved['identity'] != value['identity']:
        raise ValueError('Actual driver/cursor/RNG snapshot differs')
    restore_driver(trainer, saved, value['identity'])
    if digest(driver_snapshot(trainer, value['identity'])) != value['driver']['logical_state_sha256']:
        raise ValueError('Live restored driver/tree RNG/order/cursor differs')
    trainer._guard_round_contract_active = True
    trainer.acceptance_released = True
    record = dict(status='ABLATION_CURRENT_CONDITION_NATIVE_STATE_RESTORED',
        completed_outer=trainer.actual_outer_updates, next_outer=trainer.global_steps + 1,
        parent_manifest=os.environ['RECURSIVE_ABLATION_RESUME_MANIFEST'],
        parent_manifest_sha256=os.environ['RECURSIVE_ABLATION_RESUME_SHA256'], workers=workers,
        scope='Full native same-world model/Adam/scheduler/RNG/cursor; no vLLM bitwise restart claim')
    atomic_save(record, Path(os.environ['SEARCH_R1_BASELINE_OUTPUT']) / 'native-resumed.json', json_format=True)
    print('ABLATION_NATIVE_RESUMED=' + json.dumps(record, sort_keys=True), flush=True)
    return record
