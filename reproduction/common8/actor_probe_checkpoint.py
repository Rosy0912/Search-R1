"""Bind a milestone actor checkpoint to actual worker, sampler and probe state."""

import hashlib
import json
import os
import shutil
from pathlib import Path

from search_g1.actor_probe_contract import actor_probes, head_round
from search_g1.actor_probe_state import durable_torch_save, rng_state


def driver_payload(trainer, run, attempts, epoch, batch_index, live):
    identity = {k: run[k] for k in ('run_id','execution_id','source_commit','execution_backend',
                                  'training_world_size','profile','slurm_job_id')}
    return dict(schema_version=1, role='actor_probe_full_driver_state',
        identity=identity, completed_outer=trainer.global_steps, epoch=epoch,
        consumed_batch_index=batch_index, next_logical_update=trainer.global_steps + 1,
        sampler=trainer.train_dataloader.sampler.state_dict(),
        attempt_progress=dict(world_size=attempts.world_size, completed_outer=attempts.completed_outer,
            cumulative_writes=attempts.cumulative_writes, attempt_index=attempts.attempt_index,
            aborted_attempts=attempts.aborted_attempts, pending=None),
        probe_version=live['version'], probe_actor_step=live['actor_step'],
        reward_state={k: getattr(trainer.reward_fn, k) for k in ('alpha', 'ema_em', 'ema_trust', '_init', '_calls')},
        rng=rng_state())


def save_full_state(trainer, run, attempts, epoch, batch_index):
    assert actor_probes(run)
    base = Path(os.environ['SEARCH_G1_BASE'])
    step = trainer.global_steps
    directory = Path(run['checkpoint_dir'])/'actor'/f'global_step_{step}'
    actor = json.loads((directory/'complete.json').read_text())
    assert actor['run_id'] == run['run_id'] and actor['step'] == step and actor['finite_all_saved_tensors']
    assert attempts.pending is None and attempts.completed_outer == step
    live_path = base/'artifacts/live/manifest.json'
    live = json.loads(live_path.read_text())
    next_head_round = (head_round(step + 1) if step < 200 else head_round(200))
    assert live['round'] == next_head_round and live['run_id'] == run['run_id']
    receipt_path = directory/'full-state-complete.json'
    assert not receipt_path.exists(), 'Milestone full-state receipt already exists'
    workers = trainer.actor_rollout_wg.save_probe_training_state(str(directory), step)
    assert sorted(row['rank'] for row in workers) == list(range(run['training_world_size']))
    assert all(row['completed_outer'] == step and row['optimizer_writes'] == attempts.cumulative_writes for row in workers)
    probe_dir = directory/'probe-state'; probe_dir.mkdir(exist_ok=False)
    probe_hashes = {}
    for key, name in (('commit', 'commit_probe.pkl'), ('boundary', 'boundary_probe.pkl')):
        source = Path(live[key]); target = probe_dir/name
        assert hashlib.sha256(source.read_bytes()).hexdigest() == live['head_sha256'][name]
        shutil.copyfile(source, target)
        with target.open('rb') as f: os.fsync(f.fileno())
        probe_hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()
        assert probe_hashes[name] == live['head_sha256'][name]
    bound_manifest = dict(live)
    bound_manifest['commit'] = str(probe_dir/'commit_probe.pkl')
    bound_manifest['boundary'] = str(probe_dir/'boundary_probe.pkl')
    bound_manifest_path = probe_dir/'version-manifest.json'
    with bound_manifest_path.open('x') as f:
        json.dump(bound_manifest, f, indent=2, allow_nan=False); f.flush(); os.fsync(f.fileno())
    fd = os.open(probe_dir, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try: os.fsync(fd)
    finally: os.close(fd)
    for name in actor['files_sha256']:
        with (directory/name).open('rb') as f: os.fsync(f.fileno())
    identity = {k: run[k] for k in ('run_id','execution_id','source_commit','execution_backend',
                                  'training_world_size','profile','slurm_job_id')}
    driver = durable_torch_save(driver_payload(trainer, run, attempts, epoch, batch_index, live),
        directory/'training-state-driver.pt', base)
    result = dict(schema_version=1, role='actor_probe_full_state_checkpoint', identity=identity,
        completed_outer=step, next_logical_update=step + 1, actual_optimizer_writes=attempts.cumulative_writes,
        workers=workers, driver=driver, canonical_model_receipt_sha256=hashlib.sha256((directory/'complete.json').read_bytes()).hexdigest(),
        probe_head_hashes=probe_hashes, probe_manifest_sha256=hashlib.sha256(bound_manifest_path.read_bytes()).hexdigest(),
        probe_version=live['version'], probe_actor_step=live['actor_step'],
        completed_outer_reward_head_round=head_round(step),
        state_semantics='rank-local flat optimizer shards with exact layout/version binding; driver actual sampler cursor and all RNG states saved')
    with receipt_path.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False); f.flush(); os.fsync(f.fileno())
    fd = os.open(directory, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try: os.fsync(fd)
    finally: os.close(fd)
    return result


def verify_roundtrip(trainer, run, attempts, epoch, batch_index):
    """Load the actual persisted state at five without creating a new iterator."""
    import torch
    from reproduction.common8.verify_data import digest
    from search_g1.actor_probe_state import tree_digest, restore_rng
    base = Path(os.environ['SEARCH_G1_BASE']); step = trainer.global_steps
    assert actor_probes(run) and step == 5 and attempts.pending is None
    directory = Path(run['checkpoint_dir'])/'actor'/f'global_step_{step}'
    receipt = json.loads((directory/'full-state-complete.json').read_text())
    assert receipt['completed_outer'] == step and receipt['identity']['run_id'] == run['run_id']
    live = json.loads((base/'artifacts/live/manifest.json').read_text())
    current = tree_digest(driver_payload(trainer, run, attempts, epoch, batch_index, live))
    reference_before = trainer.ref_policy_wg.probe_isolation_state()
    workers = trainer.actor_rollout_wg.verify_probe_training_state_roundtrip(str(directory), step)
    row = receipt['driver']; path = Path(row['path'])
    assert path.parent.resolve() == directory.resolve() and digest(path) == row['sha256']
    payload = torch.load(path, map_location='cpu', weights_only=False)
    assert tree_digest(payload) == current, 'Persisted driver differs from live training state'
    trainer.train_dataloader.sampler.load_state_dict(payload['sampler'])
    for key, value in payload['attempt_progress'].items(): setattr(attempts, key, value)
    for key, value in payload['reward_state'].items(): setattr(trainer.reward_fn, key, value)
    restore_rng(payload['rng'])
    after = tree_digest(driver_payload(trainer, run, attempts, epoch, batch_index, live))
    assert current == after and reference_before == trainer.ref_policy_wg.probe_isolation_state()
    target = base/'logs/actor-probe-state-roundtrip-5.json'
    from reproduction.common8.calibration_runtime import write_once
    write_once(target, dict(run_id=run['run_id'], execution_id=run['execution_id'],
        source_commit=run['source_commit'], completed_outer=step, accepted=True,
        full_state_receipt_sha256=digest(directory/'full-state-complete.json'),
        workers=workers, driver_before_sha256=current, driver_after_sha256=after,
        reference_unchanged=True, no_new_optimizer_updates=True,
        scope='same-process canonical actor model plus persisted optimizer/scheduler/accounting/sampler and training/generation RNG load; cross-namespace entry remains separate'))


def verify_roundtrip_receipt(base, run, state):
    from reproduction.common8.verify_data import digest
    directory = Path(run['checkpoint_dir'])/'actor/global_step_5'
    proof = json.loads((Path(base)/'logs/actor-probe-state-roundtrip-5.json').read_text())
    assert all(proof[key] == run[key] for key in ('run_id', 'execution_id', 'source_commit'))
    assert proof['accepted'] is True and proof['completed_outer'] == 5
    assert proof['reference_unchanged'] is True and proof['no_new_optimizer_updates'] is True
    assert proof['full_state_receipt_sha256'] == digest(directory/'full-state-complete.json')
    assert proof['driver_before_sha256'] == proof['driver_after_sha256']
    assert sorted(row['rank'] for row in proof['workers']) == list(range(run['training_world_size']))
    states = {row['rank']: row for row in state['workers']}
    for row in proof['workers']:
        assert row['before_sha256'] == row['after_sha256']
        assert row['state_file_sha256'] == states[row['rank']]['sha256']
        assert row['parameters_and_gradients_unchanged'] is True
        assert row['canonical_model_loaded'] is True
        assert row['canonical_model_receipt_sha256'] == state['canonical_model_receipt_sha256']
        assert row['optimizer_scheduler_rollout_and_rng_loaded'] is True
    return proof


def record_refresh_isolation(base, step, before, after, version):
    assert before == after, 'Probe refresh changed actor/reference parameters, gradients, modes or GRPO state'
    target = Path(base)/'logs'/f'actor-probe-refresh-isolation-{step}.json'
    assert not target.exists()
    with target.open('x') as f:
        json.dump(dict(step=step, version=version, accepted=True, before=before, after=after),
                  f, sort_keys=True, allow_nan=False); f.flush(); os.fsync(f.fileno())


def verify_full_state(base, run, step, *, load_optimizer=False, retained_prefix=None):
    import torch
    from reproduction.common8.verify_data import digest
    from reproduction.common8.scheduler import expected_lr
    from search_g1.actor_probe_contract import validate_version
    base = Path(base)
    directory = Path(run['checkpoint_dir'])/'actor'/f'global_step_{step}'
    receipt = json.loads((directory/'full-state-complete.json').read_text())
    assert receipt['identity']['run_id'] == run['run_id'] and receipt['identity']['execution_id'] == run['execution_id']
    assert receipt['completed_outer'] == step and receipt['next_logical_update'] == step + 1
    if retained_prefix is None:
        actual = [json.loads(line) for line in (base/'logs/actual-updates.jsonl').read_text().splitlines()]
    else:
        from reproduction.common8.actor_probe_retained_prefix import verify_prefix
        verify_prefix(base,run,retained_prefix,step)
        actual=retained_prefix['selected_updates']
    records = [row for row in actual if row['step'] == step]
    assert len(records) == 1 and records[0]['run_id'] == run['run_id']
    assert receipt['actual_optimizer_writes'] == records[0]['optimizer_writes_after']
    assert receipt['completed_outer_reward_head_round'] == head_round(step)
    assert digest(directory/'complete.json') == receipt['canonical_model_receipt_sha256']
    assert sorted(r['rank'] for r in receipt['workers']) == list(range(run['training_world_size']))
    for row in receipt['workers'] + [receipt['driver']]:
        path = Path(row['path'])
        assert path.parent.resolve() == directory.resolve()
        assert path.stat().st_size == row['bytes'] and digest(path) == row['sha256']
        if load_optimizer and 'rank' in row:
            state = torch.load(path, map_location='cpu', weights_only=False)
            assert state['identity'] == receipt['identity'] and state['rank'] == row['rank']
            assert state['completed_outer'] == step and state['optimizer_writes'] == receipt['actual_optimizer_writes']
            assert state['scheduler']['last_epoch'] == step
            assert state['outer_accounting']['pending'] is None and state['outer_accounting']['prepared'] is None
            assert state['actor_accounting']['completed_outer'] == step
            assert state['actor_accounting']['last_attempt_index'] == state['outer_accounting']['last_attempt_index']
            assert state['actor_accounting']['used_attempt_ids'] == state['outer_accounting']['used_attempt_ids']
            assert state['optimizer']['state'] and state['flat_parameter_layout']
            for values in state['optimizer']['state'].values():
                for value in values.values():
                    if isinstance(value, torch.Tensor): assert torch.isfinite(value).all()
                assert int(values['step']) == state['optimizer_writes']
            assert all(group['lr'] == expected_lr(step) for group in state['optimizer']['param_groups'])
            assert state['rng']['torch_cpu'].dtype == torch.uint8 and state['rng']['torch_cuda']
            assert state['rollout_rng']['generation'].dtype == torch.uint8
            assert state['rollout_rng']['training'].dtype == torch.uint8
            del state
    probe = directory/'probe-state'
    assert digest(probe/'version-manifest.json') == receipt['probe_manifest_sha256']
    manifest = json.loads((probe/'version-manifest.json').read_text())
    validate_version(run, manifest, min(step + 1, 200))
    assert manifest['version'] == receipt['probe_version'] and manifest['actor_step'] == receipt['probe_actor_step']
    for name, sha in receipt['probe_head_hashes'].items(): assert digest(probe/name) == sha
    driver = torch.load(receipt['driver']['path'], map_location='cpu', weights_only=False)
    assert driver['identity'] == receipt['identity'] and driver['completed_outer'] == step
    assert driver['probe_version'] == receipt['probe_version']
    assert driver['attempt_progress']['completed_outer'] == step and driver['attempt_progress']['pending'] is None
    assert driver['attempt_progress']['cumulative_writes'] == receipt['actual_optimizer_writes']
    assert driver['attempt_progress']['attempt_index'] == records[0]['attempt_index']
    sampler = driver['sampler']
    assert sorted(sampler['permutation']) == list(range(sampler['dataset_size']))
    assert type(sampler['cursor']) is int and 0 <= sampler['cursor'] <= sampler['dataset_size']
    assert driver['next_logical_update'] == step + 1 and driver['rng']['torch_cpu'].dtype == torch.uint8
    return receipt
