"""Read the scalar origin of an actual admitted restore; not final acceptance.

Canonical tensor/state byte validation belongs to the restore admission and
complete checkpoint-chain verifier. This reader never fabricates parent rows.
"""
import json
import os
from pathlib import Path

from reproduction.common8.verify_data import digest
from search_g1.actor_probe_namespace import namespace_origin


def _metadata(path, root):
    path=Path(path)
    assert path == path.resolve() and path.is_relative_to(root)
    assert path.is_file() and path.stat().st_uid == os.getuid() and path.stat().st_size <= 32*1024**2
    return json.loads(path.read_bytes())


def read_origin(base, run):
    from search_g1.actor_probe_recovery import identity, binding_digest, validate_recovery
    from reproduction.common8.actor_probe_driver_restore import verify_worker_restores
    base=Path(base).resolve(); origin=namespace_origin(run)
    proof_path=base/'logs/actor-probe-recovery-restored.json'
    if run.get('actor_probe_recovery') is None:
        assert not proof_path.exists(), 'Fresh namespace contains a recovery declaration'
        return dict(origin=origin, policy_rejections=0, restored=None, proof_sha256=None)
    binding=run['actor_probe_recovery']; parent_base=Path(binding['parent_base'])
    assert parent_base.is_absolute() and parent_base == parent_base.resolve() and parent_base != base
    parent_path=parent_base/'logs/run-manifest.json'; parent=_metadata(parent_path,parent_base)
    assert digest(parent_path) == binding['parent_manifest_sha256']
    directory=Path(parent['checkpoint_dir'])/'actor'/f"global_step_{origin['completed_outer']}"
    receipt_path=directory/'full-state-complete.json'; receipt=_metadata(receipt_path,parent_base)
    assert digest(receipt_path) == binding['full_state_receipt_sha256']
    validate_recovery(binding,run,parent,receipt)
    proof=_metadata(proof_path,base)
    assert proof['role']=='actor_probe_full_recovery_restored' and proof['accepted'] is True
    assert proof['identity']==identity(run) and proof['origin']==origin
    assert proof['recovery_binding_sha256']==binding_digest(binding)
    assert proof['parent_full_state_receipt_sha256']==binding['full_state_receipt_sha256']
    assert proof['reference_unchanged'] is True and proof['no_new_optimizer_updates'] is True
    admission_path=base/'logs/actor-probe-recovery-launch-admission.json'
    admission=_metadata(admission_path,base)
    assert digest(admission_path)==proof['launch_admission_sha256']
    assert admission['role']=='actor_probe_full_recovery_launch_admission' and admission['accepted'] is True
    assert admission['identity']==identity(run) and admission['origin']==origin
    assert admission['recovery_binding_sha256']==binding_digest(binding)
    workers=verify_worker_restores(run,dict(binding=binding,receipt=receipt),proof['workers'])
    rejections=[row['policy_rejections'] for row in workers]
    assert all(type(value) is int and value>=0 for value in rejections) and len(set(rejections))==1
    return dict(origin=origin,policy_rejections=rejections[0],restored=proof,proof_sha256=digest(proof_path))


def check_scheduler_anchor(row, run, evidence, rank):
    """Fields unique to RESTORE; the caller also verifies source/PID headers."""
    from search_g1.actor_probe_recovery import binding_digest
    from reproduction.common8.scheduler import expected_lr
    origin=evidence['origin']; binding=run['actor_probe_recovery']
    assert row['kind']=='RESTORE' and row['rank']==rank
    for key,value in dict(completed_outer=origin['completed_outer'],
        actual_writes=origin['actual_optimizer_writes'],last_attempt_index=origin['attempt_index'],
        scheduler_last_epoch=origin['completed_outer'],scheduler_step_count=origin['completed_outer']+1,
        policy_rejections=evidence['policy_rejections'],scheduler_advances=0).items():
        assert type(row[key]) is int and row[key]==value
    assert row['learning_rates'] and all(type(rate) in (int,float) and
        rate==expected_lr(origin['completed_outer']) for rate in row['learning_rates'])
    assert row['recovery_binding_sha256']==binding_digest(binding)
    assert row['parent_run_id']==binding['parent_run_id'] and row['parent_execution_id']==binding['parent_execution_id']
    assert row['full_state_receipt_sha256']==binding['full_state_receipt_sha256']
    return row
