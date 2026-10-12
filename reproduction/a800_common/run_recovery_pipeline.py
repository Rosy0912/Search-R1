"""Independent owned full-state continuation; never a fresh0 pipeline retry.

The exact saved Heads must be staged first. This entry preserves the original
namespace, claims one independent pipeline and reuses assigned-only service and
cleanup machinery. It refuses launch until the pinned full-chain verifier exists.
"""
import argparse
import fcntl
import os
import signal
import socket
import time
from pathlib import Path

from reproduction.a800.owned_processes import publish
from reproduction.a800_common.run_pipeline import LocalPipeline, bound_ray_tmp_dir
from reproduction.common8.actor_probe_restore_guard import _manifest_without_placement, _unclaimed
from reproduction.common8.formal_contract import assert_formal_run
from reproduction.common8.formal_inline_safety import owned_path, read, write_once
from reproduction.common8.run_pipeline import request_interrupt, supported_execution_role
from reproduction.common8.topology import validate_base
from reproduction.common8.verify_data import digest
from search_g1.actor_probe_contract import actor_probes
from search_g1.actor_probe_namespace import namespace_origin
from search_g1.actor_probe_recovery import binding_digest, identity

CHAIN_ENTRY = 'reproduction/common8/verify_actor_probe_chain.py'


def verify_saved_heads(base, run):
    """Actual staged byte pair, not calibration or a newly fitted version."""
    from reproduction.common8.actor_probe_head_service import verify_manifest, head_paths
    base = Path(base).resolve(); origin = namespace_origin(run)
    proof = read(owned_path(base, 'logs/actor-probe-recovery-heads-staged.json'))
    prep_path = owned_path(base, 'logs/actor-probe-recovery-preparation.json')
    assert proof['role'] == 'actor_probe_restored_heads_staged' and proof['staged'] is True
    assert proof['production_admission'] is False and proof['identity'] == identity(run)
    assert proof['recovery_binding_sha256'] == binding_digest(run['actor_probe_recovery'])
    assert proof['preparation_sha256'] == digest(prep_path)
    assert proof['parent_full_state_receipt_sha256'] == run['actor_probe_recovery']['full_state_receipt_sha256']
    path = owned_path(base, 'artifacts/live/manifest.json'); manifest = read(path)
    verify_manifest(run, manifest, logical_update=origin['completed_outer']+1)
    assert manifest['version'] == run['initial_probe_version'] == proof['version']
    assert manifest['actor_step'] == proof['actor_step'] and manifest['round'] == proof['round']
    assert manifest['head_sha256'] == proof['head_sha256']
    archive = Path(proof['archive_manifest'])
    assert archive.is_relative_to(base/'artifacts') and archive == archive.resolve()
    archive = owned_path(base, archive.relative_to(base).as_posix())
    assert read(archive) == manifest
    assert digest(archive) == proof['archive_manifest_sha256'] == proof['live_manifest_sha256'] == digest(path)
    for name, path in head_paths(base, manifest).items():
        assert digest(path) == proof['head_sha256'][path.name]
    return manifest


def preflight(base, run):
    """Check candidate, source and saved origin before any new GPU claim."""
    from reproduction.common8.actor_probe_head_admission import verify_inputs_and_source
    from search_g1.actor_probe_recovery import read_recovery_context
    base = Path(base).resolve(); validate_base(base, run); assert_formal_run(run)
    assert actor_probes(run) and run.get('actor_probe_recovery') is not None
    assert run['run_start_semantics'] == 'same_experiment_full_state_resume'
    assert supported_execution_role(run, base) == 'production'
    assert os.environ.get('SEARCH_G1_BASE') == str(base)
    assert os.environ.get('SEARCH_G1_TRAINING_WORLD_SIZE') == '4'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0,1,2,3'
    assert run['execution_host'] == socket.gethostname()
    assert Path(__file__).resolve().parents[2] == base/'repo'
    assert os.environ.get('RAY_TMPDIR') == bound_ray_tmp_dir(run)
    _unclaimed(base, run)
    for name in ('process-ledger.json','pipeline-launch-claim.json','native-allocation-admission.json',
                 'pipeline-status.json','cleanup-status.json'):
        path = base/'logs'/name
        assert not path.exists() and not path.is_symlink(), 'Recovery namespace already attempted; inspect the same handle'
    assert not list((base/'logs').glob('gpu-samples-*.jsonl'))
    verify_inputs_and_source(base, run)
    # Mandatory final acceptance implementation must already be in the exact
    # published source; a normal fresh verifier cannot accept the child suffix.
    entry = owned_path(base, 'repo/'+CHAIN_ENTRY)
    assert CHAIN_ENTRY in run['source_sha256'] and digest(entry) == run['source_sha256'][CHAIN_ENTRY]
    candidate_path = owned_path(base, 'logs/actor-probe-recovery-candidate.json')
    prep = read(owned_path(base, 'logs/actor-probe-recovery-preparation.json'))
    assert prep['role'] == 'actor_probe_full_recovery_preparation' and prep['prepared'] is True
    assert prep['production_admission'] is False and prep['identity'] == identity(run)
    assert prep['candidate_manifest'] == 'logs/actor-probe-recovery-candidate.json'
    assert prep['candidate_manifest_sha256'] == digest(candidate_path)
    assert _manifest_without_placement(read(candidate_path)) == _manifest_without_placement(run)
    assert prep['origin'] == namespace_origin(run)
    assert prep['recovery_binding_sha256'] == binding_digest(run['actor_probe_recovery'])
    assert prep['parent_manifest_sha256'] == run['actor_probe_recovery']['parent_manifest_sha256']
    assert prep['parent_full_state_receipt_sha256'] == run['actor_probe_recovery']['full_state_receipt_sha256']
    assert prep['parent_trainer_session_exit_verified'] is True and prep['canonical_finite_tensor_count'] == 435
    verify_saved_heads(base, run)
    read_recovery_context(run, 0)
    from reproduction.common8.actor_probe_fault_review import verify as verify_fault_review
    verify_fault_review(base,run)
    return namespace_origin(run)


class RecoveryPipeline(LocalPipeline):
    def initial_probe_calibration(self):
        # Files were admitted before the pipeline claim. No initial300 refit,
        # frozen-teacher adoption or new fitting RNG is inserted into the chain.
        manifest = verify_saved_heads(self.base, self.run)
        self.stage('EXACT_SAVED_BINARY_HEADS_REUSED', version=manifest['version'],
            origin=namespace_origin(self.run), new_calibration_generated=False)

    def train_and_verify(self):
        origin = namespace_origin(self.run)
        self.stage('GRPO_SAVED_ORIGIN_TO_200', next_logical_update=origin['completed_outer']+1,
            inherited_optimizer_writes=origin['actual_optimizer_writes'], new_namespace_optimizer_writes=0)
        log = Path(self.run['train_log']); assert not log.exists() and not log.is_symlink()
        trainer = self.start('trainer', ['bash','reproduction/a800_common/train_recovery.sh'],
            self.training_gpus, {'SEARCH_G1_TRAIN_LOG':str(log)})
        self.wait('trainer', trainer)
        self.stage('FINAL200_COMPLETE_RETAINED_CHAIN_EVALUATION_ACCEPTANCE')
        verifier = self.start('final-acceptance', [self.python,'-m',
            'reproduction.common8.verify_actor_probe_chain','--base',str(self.base)], [])
        self.wait('final-acceptance', verifier)
        return 'FINISHED_FULL200_RECOVERY_CHAIN_FINAL51713_AND_OWNED_CLEANUP'


def launch(base, run):
    from reproduction.a800_common.predecessor_admission import admit as admit_predecessor
    from reproduction.common8.native_allocation import admit
    from reproduction.common8.disk_admission import admit as admit_disk
    base = Path(base).resolve(); origin = preflight(base, run)
    lock_path = base/'gpu-job.lock'; assert not lock_path.is_symlink()
    lock = lock_path.open('a'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM, request_interrupt); signal.signal(signal.SIGINT, request_interrupt)
    predecessor = admit_predecessor(base, run)
    for port in (18010,18012):
        with socket.socket() as sock: sock.bind(('127.0.0.1',port))
    allocation = admit(base)
    write_once(base/'logs/pipeline-launch-claim.json', dict(
        role='actor_probe_full_recovery_pipeline_claim', run_id=run['run_id'],
        execution_id=run['execution_id'], execution_backend='ssh_local', slurm_job_id=None,
        source_commit=run['source_commit'], origin=origin,
        node=allocation['node'],
        training_physical_indices=allocation['training_physical_indices'],
        auxiliary_physical_indices=allocation['auxiliary_physical_indices'],
        claimed_unix=time.time(),
        semantics='exclusive saved-origin pipeline startup; full state must restore before rollout; no training completion claim',
        recovery_binding_sha256=binding_digest(run['actor_probe_recovery']),
        pipeline_identity=allocation['pipeline_identity'], allocation_sha256=digest(base/'logs/native-allocation-admission.json'),
        assigned_gpu_uuids=allocation['assigned_gpu_uuids'],
        full_state_restore_required_before_rollout=True, new_namespace_optimizer_writes=0))
    assert predecessor is not None
    write_once(base/'logs/predecessor-completion-admission.json', predecessor)
    run['pipeline_identity'] = allocation['pipeline_identity']; publish(base/'logs/run-manifest.json', run)
    admit_disk(base)
    RecoveryPipeline(base, run).execute()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--base', type=Path, required=True)
    args = parser.parse_args(); base = args.base.resolve()
    launch(base, read(owned_path(base,'logs/run-manifest.json')))


if __name__ == '__main__': main()
