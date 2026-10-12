"""Explicit same-experiment state rebinding; never an implicit fresh launcher."""
import hashlib
import json
import os
import re
import socket
import subprocess
from pathlib import Path

from search_g1.actor_probe_contract import actor_probes, validate_version

IDENTITY_KEYS = ('run_id', 'execution_id', 'source_commit', 'execution_backend',
                 'training_world_size', 'profile', 'slurm_job_id')
CONTRACT_KEYS = ('input_sha256', 'initial_model_sha256', 'initial_model', 'reference_model',
    'model_revision', 'budget', 'execution_host',
    'training_world_size', 'training_physical_indices', 'auxiliary_physical_indices',
    'excluded_physical_indices', 'reward_schedule', 'reward_schedule_boundary',
    'probe_representation_source', 'probe_refresh_interval', 'probe_parameter_ownership',
    'policy_safety_mode', 'contract_kind', 'target_updates', 'predecessor_completion',
    'vllm_gpu_memory_utilization')


def identity(run):
    return {key: run[key] for key in IDENTITY_KEYS}


def validate_recovery(binding, child, parent, receipt, *, rank=None):
    """Pure admission before parameters, moments, counters or RNG are mutated."""
    assert actor_probes(child) and actor_probes(parent)
    assert binding['role'] == 'actor_probe_same_experiment_full_state_recovery'
    assert binding['state_semantics'] == 'full_model_optimizer_scheduler_rng_sampler_and_heads'
    for key in ('run_id', 'execution_id', 'source_commit'):
        assert parent[key] == binding['parent_' + key]
    for key in ('run_id', 'execution_id'):
        assert child[key] != parent[key], 'Recovery must have an independent namespace identity'
    root = parent.get('actor_probe_experiment_root_run_id', parent['run_id'])
    assert re.fullmatch('[0-9a-f]{32}', root)
    assert child['actor_probe_experiment_root_run_id'] == binding['experiment_root_run_id'] == root
    assert all(child[key] == parent[key] for key in CONTRACT_KEYS), 'Recovery recipe or data changed'
    assert child['execution_backend'] == parent['execution_backend'] == 'ssh_local'
    assert child['execution_host'] == parent['execution_host'] == socket.gethostname(), \
        'This recovery entry admits only the same actual host; local proc cannot prove remote exit'
    from reproduction.common8.verify_data import MANIFEST_SHA
    assert child['input_sha256']['manifest.json'] == MANIFEST_SHA
    assert child['model_revision'] == 'aa8e72537993ba99e69dfaafa59ed015b17504d1'
    assert child['training_world_size'] == 4 and child['target_updates'] == 200
    assert child['initial_model'] == child['reference_model']
    assert parent['initial_model'] == parent['reference_model']
    step = binding['completed_outer']
    assert type(step) is int and step in (5, 50, 100, 150)
    assert receipt['identity'] == identity(parent)
    assert receipt['completed_outer'] == step and receipt['next_logical_update'] == step + 1
    for key in ('parent_manifest_sha256', 'full_state_receipt_sha256'):
        assert re.fullmatch('[0-9a-f]{64}', binding[key])
    from search_g1.actor_probe_model_state import validate_roundtrip_receipt
    row = validate_roundtrip_receipt(receipt, parent, step, 0 if rank is None else rank)
    return row


def rebound_worker_payload(payload, binding, child, parent, receipt, rank):
    validate_recovery(binding, child, parent, receipt, rank=rank)
    assert payload['identity'] == identity(parent) and payload['rank'] == rank
    assert payload['completed_outer'] == receipt['completed_outer']
    assert payload['optimizer_writes'] == receipt['actual_optimizer_writes']
    # Only namespace identity changes. Tensors, Adam, scheduler, counters and
    # random streams retain the exact saved values; the original is preserved.
    return dict(payload, identity=identity(child))


def rebound_driver_payload(payload, binding, child, parent, receipt):
    import math
    validate_recovery(binding, child, parent, receipt)
    assert payload['role'] == 'actor_probe_full_driver_state'
    assert payload['identity'] == identity(parent)
    assert payload['completed_outer'] == receipt['completed_outer']
    assert payload['next_logical_update'] == receipt['next_logical_update']
    progress = payload['attempt_progress']
    assert progress['world_size'] == 4 and progress['pending'] is None
    assert progress['completed_outer'] == receipt['completed_outer']
    assert progress['cumulative_writes'] == receipt['actual_optimizer_writes']
    assert type(progress['attempt_index']) is int and progress['attempt_index'] >= progress['completed_outer']
    assert type(progress['aborted_attempts']) is int and progress['aborted_attempts'] >= 0
    assert payload['probe_version'] == receipt['probe_version']
    assert payload['probe_actor_step'] == receipt['probe_actor_step']
    assert type(payload['epoch']) is int and payload['epoch'] >= 0
    assert type(payload['consumed_batch_index']) is int and payload['consumed_batch_index'] >= 0
    reward = payload['reward_state']
    assert set(reward) == {'alpha', 'ema_em', 'ema_trust', '_init', '_calls'}
    assert all(type(reward[key]) in (float, int) and math.isfinite(reward[key])
               for key in ('alpha', 'ema_em', 'ema_trust'))
    assert type(reward['_init']) is bool and type(reward['_calls']) is int and reward['_calls'] >= 0
    return dict(payload, identity=identity(child))


def apply_driver_state(trainer, attempts, payload, run):
    """Restore the exact next data iterator; a production caller must gate the chain."""
    import torch
    from search_g1.actor_probe_sampler import RecordedRandomSampler, restored_iterator
    from search_g1.actor_probe_state import validate_rng
    assert payload['identity'] == identity(run)
    assert trainer.global_steps == attempts.completed_outer == attempts.cumulative_writes == 0
    assert attempts.pending is None and trainer.train_dataloader.num_workers == 0
    assert isinstance(trainer.train_dataloader.sampler, RecordedRandomSampler)
    # Reject an invalid permutation/epoch/cursor before any driver field changes.
    validator = RecordedRandomSampler(trainer.train_dataloader.dataset)
    validator.load_state_dict(payload['sampler'])
    assert payload['rng']['torch_cpu'].dtype == torch.uint8 and not payload['rng']['torch_cuda']
    validate_rng(payload['rng'])
    assert payload['attempt_progress']['world_size'] == attempts.world_size
    for key, value in payload['attempt_progress'].items(): setattr(attempts, key, value)
    for key, value in payload['reward_state'].items(): setattr(trainer.reward_fn, key, value)
    trainer.global_steps = payload['completed_outer']
    trainer.reward_fn.expected_probe_version = payload['probe_version']
    iterator = restored_iterator(trainer.train_dataloader, payload['sampler'], payload['rng'])
    return dict(iterator=iterator, epoch=payload['epoch'],
                next_batch_index=payload['consumed_batch_index'] + 1,
                next_logical_update=payload['next_logical_update'])


def rebound_probe_manifest(manifest, binding, child, parent, receipt, paths):
    validate_recovery(binding, child, parent, receipt)
    validate_version(parent, manifest, receipt['next_logical_update'])
    assert manifest['version'] == receipt['probe_version']
    assert manifest['actor_step'] == receipt['probe_actor_step']
    assert manifest['head_sha256'] == receipt['probe_head_hashes']
    result = dict(manifest, run_id=child['run_id'], commit=str(paths['commit']),
                  boundary=str(paths['boundary']),
                  restored_from=dict(run_id=parent['run_id'], execution_id=parent['execution_id'],
                    source_commit=parent['source_commit'], version=manifest['version'],
                    full_state_receipt_sha256=binding['full_state_receipt_sha256']))
    validate_version(child, result, receipt['next_logical_update'])
    return result


def _owned(path, limit=None):
    path = Path(path)
    assert path.is_absolute() and path == path.resolve() and path.stat().st_uid == os.getuid()
    assert path.is_file() and (limit is None or path.stat().st_size <= limit)
    return path


def read_recovery_context(child, rank):
    """Read the bound parent bytes and prove all its owned trainers have exited."""
    from reproduction.common8.topology import validate_base
    from reproduction.common8.verify_data import digest
    from reproduction.a800.owned_processes import identity as process_identity
    binding = child['actor_probe_recovery']
    parent_base = Path(binding['parent_base'])
    assert parent_base != Path(os.environ['SEARCH_G1_BASE']).resolve()
    manifest_path = _owned(parent_base/'logs/run-manifest.json', 32 * 1024**2)
    assert digest(manifest_path) == binding['parent_manifest_sha256']
    parent = json.loads(manifest_path.read_bytes()); validate_base(parent_base, parent)
    directory = Path(parent['checkpoint_dir'])/'actor'/f"global_step_{binding['completed_outer']}"
    assert directory.is_relative_to(parent_base) and directory == directory.resolve()
    receipt_path = _owned(directory/'full-state-complete.json', 32 * 1024**2)
    assert digest(receipt_path) == binding['full_state_receipt_sha256']
    receipt = json.loads(receipt_path.read_bytes())
    row = validate_recovery(binding, child, parent, receipt, rank=rank)
    reference = Path(child['reference_model'])
    assert reference.is_absolute() and reference == reference.resolve() and reference.is_dir()
    for name, sha in child['initial_model_sha256'].items():
        assert Path(name).name == name
        assert digest(_owned(reference/name)) == sha, 'Fixed reference differs from the bound official bytes'
    repo = parent_base/'repo'
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip() == parent['source_commit']
    for name, sha in parent['source_sha256'].items():
        source = _owned(repo/name)
        assert source.is_relative_to(repo) and digest(source) == sha
    ledger_path = _owned(parent_base/'logs/process-ledger.json', 32 * 1024**2)
    ledger = json.loads(ledger_path.read_bytes()); assert ledger['run_id'] == parent['run_id']
    trainers = [job for name, job in ledger['jobs'].items() if name == 'trainer']
    assert len(trainers) == 1, 'Missing actual predecessor trainer ledger'
    records = [record for job in trainers for record in job['ledger'].values()]
    records += [job['leader'] for job in trainers]
    sessions = {(job['leader']['sid'], job['leader']['uid']) for job in trainers}
    for record in records:
        try: current = process_identity(record['pid'])
        except (FileNotFoundError, ProcessLookupError): continue
        assert current['state'] in ('Z', 'X') or any(current[k] != record[k]
            for k in ('start_ticks', 'sid', 'uid')), 'Recovery parent trainer birth is still live'
    for path in Path('/proc').iterdir():
        if not path.name.isdigit(): continue
        try: current = process_identity(int(path.name))
        except (FileNotFoundError, ProcessLookupError): continue
        assert current['state'] in ('Z', 'X') or (current['sid'], current['uid']) not in sessions, \
            'Unrecorded parent trainer session member is still live'
    # Validate both immutable Head bytes now, not after loading actor weights.
    probe = directory/'probe-state'
    assert digest(_owned(probe/'version-manifest.json')) == receipt['probe_manifest_sha256']
    probe_manifest = json.loads((probe/'version-manifest.json').read_bytes())
    validate_version(parent, probe_manifest, receipt['next_logical_update'])
    assert probe_manifest['version'] == receipt['probe_version']
    assert probe_manifest['head_sha256'] == receipt['probe_head_hashes']
    for name, sha in receipt['probe_head_hashes'].items():
        assert Path(name).name == name and digest(_owned(probe/name)) == sha
    for saved in receipt['workers'] + [receipt['driver']]:
        path = _owned(saved['path'])
        assert path.parent == directory and path.stat().st_size == saved['bytes']
        assert digest(path) == saved['sha256']
    from reproduction.common8.actor_probe_checkpoint import verify_full_state
    assert verify_full_state(parent_base, parent, binding['completed_outer']) == receipt
    return dict(binding=binding, parent=parent, parent_base=parent_base, directory=directory,
                receipt=receipt, row=row, probe_manifest=probe_manifest)


def binding_digest(binding):
    return hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()
