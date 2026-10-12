"""Driver-side actual restore, gated by a separate admitted recovery launcher.

The fresh launcher rejects recovery manifests. This entry does not launch any
process and does not replace chain-aware final acceptance.
"""
import json
import os
import re
from pathlib import Path


def verify_worker_restores(run, context, workers):
    from search_g1.actor_probe_recovery import binding_digest, identity
    from search_g1.actor_probe_namespace import namespace_origin
    receipt = context['receipt']; binding = context['binding']
    origin = namespace_origin(run)
    assert type(workers) is list and sorted(row['rank'] for row in workers) == list(range(4))
    assert all(type(row['rank']) is int for row in workers)
    for row in workers:
        assert row['identity'] == identity(run), 'Worker confirmation belongs to another child'
        assert type(row['last_attempt_index']) is int and row['last_attempt_index'] == origin['attempt_index']
        assert type(row['policy_rejections']) is int and row['policy_rejections'] >= 0
        assert row['completed_outer'] == receipt['completed_outer']
        assert row['optimizer_writes'] == receipt['actual_optimizer_writes']
        assert row['canonical_model_loaded'] is True
        assert row['canonical_model_receipt_sha256'] == receipt['canonical_model_receipt_sha256']
        assert row['recovery_binding_sha256'] == binding_digest(binding)
        for key in ('original_worker_state_sha256', 'rebound_worker_state_sha256',
                    'local_actor_parameters_sha256'):
            assert isinstance(row[key], str) and re.fullmatch('[0-9a-f]{64}', row[key])
    assert len({row['policy_rejections'] for row in workers}) == 1
    return workers


def restore_driver_and_actor(trainer, run, attempts):
    import torch
    from reproduction.common8.verify_data import digest
    from reproduction.common8.calibration_runtime import write_once
    from reproduction.common8.actor_probe_checkpoint import driver_payload
    from search_g1.actor_probe_namespace import namespace_origin, validate_driver_origin
    from search_g1.actor_probe_recovery import (identity, binding_digest, read_recovery_context,
        rebound_driver_payload, rebound_probe_manifest, apply_driver_state)
    from search_g1.actor_probe_sampler import RecordedRandomSampler
    from search_g1.actor_probe_state import tree_digest, validate_rng

    base = Path(os.environ['SEARCH_G1_BASE']).resolve()
    origin = namespace_origin(run); assert origin['completed_outer'] > 0
    admission_path = base/'logs/actor-probe-recovery-launch-admission.json'
    assert admission_path.is_file() and admission_path == admission_path.resolve()
    assert admission_path.stat().st_uid == os.getuid()
    admission = json.loads(admission_path.read_bytes())
    assert admission['role'] == 'actor_probe_full_recovery_launch_admission' and admission['accepted'] is True
    assert admission['identity'] == identity(run)
    assert admission['recovery_binding_sha256'] == binding_digest(run['actor_probe_recovery'])
    assert admission['origin'] == origin
    assert not (base/'logs/actor-probe-recovery-restored.json').exists(), 'Restore already recorded'
    context = read_recovery_context(run, 0)
    live_path = base/'artifacts/live/manifest.json'
    live = json.loads(live_path.read_bytes())
    head_paths = {}
    for key, name in (('commit', 'commit_probe.pkl'), ('boundary', 'boundary_probe.pkl')):
        path = Path(live[key])
        assert path == path.resolve() and path.is_relative_to(base/'artifacts')
        assert path.is_file() and path.stat().st_uid == os.getuid()
        assert digest(path) == context['receipt']['probe_head_hashes'][name]
        head_paths[key] = path
    assert live == rebound_probe_manifest(context['probe_manifest'], context['binding'], run,
        context['parent'], context['receipt'], head_paths)
    saved = torch.load(context['receipt']['driver']['path'], map_location='cpu', weights_only=False)
    payload = rebound_driver_payload(saved, context['binding'], run, context['parent'], context['receipt'])
    validate_driver_origin(run, payload)
    assert trainer.global_steps == attempts.completed_outer == attempts.cumulative_writes == 0
    assert trainer.train_dataloader.num_workers == 0 and isinstance(trainer.train_dataloader.sampler, RecordedRandomSampler)
    RecordedRandomSampler(trainer.train_dataloader.dataset).load_state_dict(payload['sampler'])
    assert not payload['rng']['torch_cuda']; validate_rng(payload['rng'])
    reference_before = trainer.ref_policy_wg.probe_isolation_state()
    workers = verify_worker_restores(run, context, trainer.actor_rollout_wg.restore_probe_training_checkpoint())
    assert reference_before == trainer.ref_policy_wg.probe_isolation_state(), 'Fixed reference changed during actor recovery'
    resume = apply_driver_state(trainer, attempts, payload, run)
    actual = driver_payload(trainer, run, attempts, payload['epoch'], payload['consumed_batch_index'], live)
    assert tree_digest(actual) == tree_digest(payload), 'Actual driver state differs after restore'
    write_once(base/'logs/actor-probe-recovery-restored.json', dict(
        role='actor_probe_full_recovery_restored', identity=identity(run), accepted=True,
        origin=origin, recovery_binding_sha256=binding_digest(context['binding']),
        launch_admission_sha256=digest(admission_path), parent_full_state_receipt_sha256=context['binding']['full_state_receipt_sha256'],
        original_driver_state_sha256=tree_digest(saved), rebound_driver_state_sha256=tree_digest(actual),
        reference_unchanged=True, no_new_optimizer_updates=True, workers=workers))
    return resume
