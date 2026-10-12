"""Bind a saved-origin trainer entry to real frozen child admissions.

This is a pre-restore admission, not proof of restoration, new updates or final
chain acceptance. It starts no process. The fresh pipeline still rejects resume;
its separate owned pipeline and complete chain verifier remain required.
"""
import argparse
import os
import socket
import time
from pathlib import Path

from reproduction.common8.actor_probe_restore_guard import ROLE as GUARD_ROLE, _time, _unclaimed
from reproduction.common8.formal_inline_safety import PLAN, owned_path, read, write_once
from reproduction.common8.policy_calibration_plan import canonical_bytes, document_sha256
from reproduction.common8.verify_data import digest
from search_g1.actor_probe_namespace import namespace_origin
from search_g1.actor_probe_recovery import binding_digest, identity

ADMISSION = 'logs/actor-probe-recovery-launch-admission.json'
ROLE = 'actor_probe_full_recovery_launch_admission'


def pipeline_birth(run):
    # The native writer preserves state/command too. Those instantaneous fields
    # are not part of the immutable four-field Linux process identity contract.
    value = {key:run['pipeline_identity'][key] for key in ('pid','start_ticks','sid','uid')}
    assert all(type(v) is int for v in value.values())
    assert all(value[k] > 0 for k in ('pid','start_ticks','sid')) and value['uid'] >= 0
    return value


def build(base, run, guard, created_unix):
    """Reconstruct the receipt; caller-supplied accepted flags cannot bypass it."""
    from reproduction.common8.formal_inline_safety import verify_plan
    base = Path(base).resolve(); origin = namespace_origin(run)
    assert origin['completed_outer'] > 0 and run['actor_probe_recovery']
    assert guard['role'] == GUARD_ROLE and guard['identity'] == identity(run)
    _, guard_sha = verify_plan(base, guard, run)
    guard_path = owned_path(base, PLAN)
    assert canonical_bytes(read(guard_path)) == canonical_bytes(guard)
    assert _time(run['started_unix']) <= _time(guard['created_unix']) <= _time(created_unix)
    return dict(schema_version=1, role=ROLE, accepted=True,
        identity=identity(run), origin=origin,
        recovery_binding_sha256=binding_digest(run['actor_probe_recovery']),
        parent_full_state_receipt_sha256=run['actor_probe_recovery']['full_state_receipt_sha256'],
        parent_manifest_sha256=run['actor_probe_recovery']['parent_manifest_sha256'],
        source_map_sha256=document_sha256(run['source_sha256']),
        guard_plan=PLAN, guard_plan_sha256=digest(guard_path),
        guard_document_sha256=guard_sha, receipts_sha256=dict(guard['receipts_sha256']),
        native_birth=guard['native_birth'], resolved_config_sha256=guard['resolved_config_sha256'],
        probe_version=guard['initial_probe_version'],
        inherited_optimizer_writes=origin['actual_optimizer_writes'],
        new_namespace_optimizer_writes=0, next_logical_update=origin['completed_outer']+1,
        full_state_restore_required_before_rollout=True,
        actual_behavior_probability_accepted=False, training_completion_accepted=False,
        full_final_chain_accepted=False, created_unix=created_unix)


def verify(base, run, admission=None):
    """Historical binding reader; this does not claim current process liveness."""
    base = Path(base).resolve()
    if admission is None: admission = read(owned_path(base, ADMISSION))
    assert type(admission['schema_version']) is int and admission['schema_version'] == 1
    assert admission['role'] == ROLE
    expected = build(base, run, read(owned_path(base, PLAN)), admission['created_unix'])
    assert canonical_bytes(expected) == canonical_bytes(admission), 'Recovery launch admission differs from actual frozen inputs'
    return admission


def verify_current(base, run, *, training_claim=None):
    """The live consumer checks actual source/inputs/owned endpoint and PID birth."""
    from reproduction.common8.actor_probe_head_admission import verify_training_gate
    from reproduction.a800.owned_processes import identity as process_identity
    base = Path(base).resolve(); admission = verify(base, run)
    assert run['execution_host'] == socket.gethostname()
    assert verify_training_gate(base) == run
    producer = admission['native_birth']['pipeline_identity']
    assert producer == pipeline_birth(run)
    actual = process_identity(producer['pid'])
    assert actual['state'] not in ('Z', 'X') and actual['uid'] == os.getuid()
    assert all(actual[key] == producer[key] for key in ('pid','start_ticks','sid','uid'))
    if training_claim is not None:
        verify_claim_binding(base, run, training_claim, admission)
        ledger = read(owned_path(base, 'logs/process-ledger.json'))
        assert ledger['run_id'] == run['run_id']
        leader = ledger['jobs']['trainer']['leader']
        assert leader['pid'] == training_claim['shell_pid'] and leader['start_ticks'] == training_claim['shell_start_ticks']
        trainer = process_identity(leader['pid'])
        assert trainer['state'] not in ('Z','X') and trainer['uid'] == os.getuid()
        assert all(trainer[k] == leader[k] for k in ('pid','start_ticks','sid','uid'))
    return admission


def verify_claim_binding(base, run, claim, admission=None):
    admission = verify(base, run, admission)
    assert claim['run_id'] == run['run_id'] and claim['execution_id'] == run['execution_id']
    assert claim['execution_backend'] == 'ssh_local' and claim['slurm_job_id'] is None
    assert claim['recovery_launch_admission_sha256'] == digest(owned_path(base, ADMISSION))
    assert claim['recovery_binding_sha256'] == admission['recovery_binding_sha256']
    assert canonical_bytes(claim['origin']) == canonical_bytes(admission['origin'])
    assert claim['state_semantics'] == 'full_state_restore_required_before_rollout'
    assert _time(claim['claimed_unix']) >= _time(admission['created_unix'])
    return claim


def prepare(base):
    """Run only in the pinned child namespace with actual admitted services."""
    from reproduction.common8.actor_probe_head_admission import verify_training_gate
    from reproduction.common8.topology import validate_base
    from search_g1.actor_probe_recovery import read_recovery_context
    base = Path(base).resolve(); run = read(owned_path(base, 'logs/run-manifest.json'))
    validate_base(base, run)
    assert Path(__file__).resolve().parents[2] == base/'repo'
    relative = Path(__file__).resolve().relative_to(base/'repo').as_posix()
    assert digest(Path(__file__)) == run['source_sha256'][relative]
    _unclaimed(base, run)
    assert verify_training_gate(base) == run
    # Reads actual saved Head/model/state bytes and absence of the parent trainer
    # birth/session. It does not restore the child or add any optimizer update.
    read_recovery_context(run, 0)
    guard = read(owned_path(base, PLAN)); admission = build(base, run, guard, time.time())
    producer = admission['native_birth']['pipeline_identity']
    from reproduction.a800.owned_processes import identity as process_identity
    actual = process_identity(producer['pid'])
    assert actual['state'] not in ('Z','X') and actual['uid'] == os.getuid()
    assert producer == pipeline_birth(run) and all(actual[k] == producer[k] for k in producer)
    _unclaimed(base, run)
    write_once(base/ADMISSION, admission)
    return verify(base, run, admission)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--base', type=Path, required=True)
    args = parser.parse_args(); prepare(args.base)


if __name__ == '__main__': main()
