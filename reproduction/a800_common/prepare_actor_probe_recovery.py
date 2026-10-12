"""Prepare one explicit full-state continuation; never launch a fresh0 pipeline.

Run on the admitted A800 host after the selected parent trainer has exited.
Source/data must already be staged in an independent namespace. Preparation
validates the real parent bytes and canonical tensors, but grants no resource,
probe endpoint, prospective guard-plan or production launch admission.
"""
import argparse
import copy
import json
import os
import re
import stat
import subprocess
import time
import uuid
from pathlib import Path

from reproduction.common8.verify_data import digest
from search_g1.actor_probe_recovery import identity, validate_recovery, rebound_driver_payload
from search_g1.actor_probe_namespace import namespace_origin


def candidate_manifest(parent, receipt, driver, *, base, source_commit, source_sha256,
                       run_id, execution_id, started_unix, parent_base,
                       parent_manifest_sha256, receipt_sha256, experiment_root_run_id):
    """Derive counters from the actual saved driver, never observed lost writes."""
    from reproduction.common8.formal_contract import assert_formal_run
    assert_formal_run(parent)
    assert re.fullmatch('[0-9a-f]{40}', source_commit) and source_sha256
    assert re.fullmatch('[0-9a-f]{32}', run_id) and re.fullmatch('[0-9a-f]{32}', execution_id)
    assert driver['identity'] == receipt['identity'] == identity(parent)
    step = receipt['completed_outer']; progress = driver['attempt_progress']
    assert driver['completed_outer'] == progress['completed_outer'] == step
    assert progress['cumulative_writes'] == receipt['actual_optimizer_writes']
    parent_origin = namespace_origin(parent)
    assert step > parent_origin['completed_outer'], 'Selected checkpoint is not a new saved milestone of this parent namespace'
    child = copy.deepcopy(parent)
    # These are runtime declarations, not inherited process/service admission.
    for key in ('pipeline_identity', 'probe_logical_ordinal', 'initial_probe_adoption_source_base'):
        child.pop(key, None)
    root = parent.get('actor_probe_experiment_root_run_id', parent['run_id'])
    if parent.get('actor_probe_recovery') is None:
        assert root == parent['run_id'], 'Fresh parent cannot declare a different experiment root'
    assert root == experiment_root_run_id, 'Selected parent belongs to another explicitly bound experiment root'
    binding = dict(role='actor_probe_same_experiment_full_state_recovery',
        state_semantics='full_model_optimizer_scheduler_rng_sampler_and_heads',
        parent_base=str(parent_base), parent_run_id=parent['run_id'],
        parent_execution_id=parent['execution_id'], parent_source_commit=parent['source_commit'],
        experiment_root_run_id=root, parent_manifest_sha256=parent_manifest_sha256,
        full_state_receipt_sha256=receipt_sha256, completed_outer=step,
        actual_optimizer_writes=progress['cumulative_writes'], attempt_index=progress['attempt_index'],
        aborted_attempts=progress['aborted_attempts'])
    child.update(run_id=run_id, execution_id=execution_id, source_commit=source_commit,
        source_sha256=copy.deepcopy(source_sha256), started_unix=started_unix,
        actor_probe_experiment_root_run_id=root, actor_probe_recovery=binding,
        initial_probe_version=receipt['probe_version'], probe_visible_device=None,
        checkpoint_dir=str(Path(base)/'checkpoints/search-g1-common8'),
        train_log=str(Path(base)/'logs/search-g1-common8-training.log'),
        ray_tmp_dir=str(Path('/tmp')/('g-'+execution_id)), ray_tmp_dir_identity=None,
        run_start_semantics='same_experiment_full_state_resume',
        optimizer_semantics='same-experiment saved model/Adam/scheduler/RNG/sampler/Heads; '
                            'namespace-local logs start at saved K+1; lost parent writes excluded',
        retrieval_placement='pending child actual allocation/retrieval/service admission')
    assert_formal_run(child)
    validate_recovery(binding, child, parent, receipt)
    rebound_driver_payload(driver, binding, child, parent, receipt)
    namespace_origin(child)
    return child


def _owned(path, limit=None):
    path = Path(path)
    assert path == path.resolve() and path.is_file() and path.stat().st_uid == os.getuid()
    assert limit is None or path.stat().st_size <= limit
    return path


def _verify_official_model(parent, official):
    from reproduction.common8.prepare_run import verify_model
    model = Path(parent['initial_model'])
    assert model.is_absolute() and model == model.resolve() and model.is_dir()
    hashes = verify_model(model, official)
    assert hashes == parent['initial_model_sha256']
    return hashes


def _publish_candidate(base, child, preparation):
    """The durable candidate manifest precedes its final completion marker."""
    from reproduction.common8.calibration_runtime import write_once
    assert preparation['prepared'] is True and preparation['production_admission'] is False
    assert preparation['identity'] == identity(child)
    manifest = Path(base)/'logs/run-manifest.json'
    completion = Path(base)/'logs/actor-probe-recovery-preparation.json'
    assert not manifest.exists() and not completion.exists()
    write_once(manifest, child)
    assert json.loads(_owned(manifest, 32*1024**2).read_bytes()) == child
    record = dict(preparation, candidate_manifest_sha256=digest(manifest))
    write_once(completion, record)
    return record


def _assert_clean_candidate(base):
    for name in ('run-manifest.json', 'process-ledger.json', 'pipeline-launch-claim.json',
                 'training-launch-claim.json', 'actor-probe-recovery-preparation.json',
                 'actual-attempts.jsonl', 'actual-updates.jsonl'):
        path = Path(base)/'logs'/name
        assert not path.exists() and not path.is_symlink(), 'Existing recovery attempt; inspect, never overwrite/reprepare'
    assert not list((Path(base)/'logs').glob('*-rank-*.jsonl'))
    checkpoints = Path(base)/'checkpoints/search-g1-common8'
    assert not checkpoints.is_symlink()
    assert not checkpoints.exists() or not any(checkpoints.iterdir())


def prepare(base, parent_base, step, source_commit, experiment_root_run_id):
    """Remote-only candidate preparation with actual finite/hash/exit evidence."""
    import torch
    from reproduction.common8.topology import validate_base
    from reproduction.common8.source_files import runtime_files
    from search_g1.actor_probe_recovery import read_recovery_context, binding_digest
    from search_g1.actor_probe_model_state import read_canonical_actor_state
    from search_g1.actor_probe_state import validate_rng
    from reproduction.common8.actor_probe_rank_tail import read_unreturned_tail
    from reproduction.common8.actor_probe_restore_evidence import read_origin
    from reproduction.common8.verify_data import verify as verify_data
    from reproduction.a800_common.deploy import OFFICIAL_SHA
    base = Path(base).resolve(); parent_base = Path(parent_base).resolve()
    parent_path = _owned(parent_base/'logs/run-manifest.json', 32*1024**2)
    parent = json.loads(parent_path.read_bytes()); validate_base(base, parent); validate_base(parent_base, parent)
    read_origin(parent_base, parent)
    assert base != parent_base and base.is_dir() and base.stat().st_uid == os.getuid()
    assert os.environ.get('SEARCH_G1_TRAINING_WORLD_SIZE') == '4'
    _assert_clean_candidate(base)
    repo = base/'repo'
    assert Path(__file__).resolve().parents[2] == repo, 'Preparation must execute the pinned child source'
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip() == source_commit
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo).strip()
    source = {str(path.relative_to(repo)):digest(_owned(path)) for path in runtime_files(repo)}
    published = json.loads(_owned(base/'logs/deployment-source-receipt.json', 32*1024**2).read_bytes())
    assert published['source_commit'] == source_commit and published['source_sha256'] == source
    assert published['source_is_exact_published_commit'] is True
    for name, sha in parent['input_sha256'].items():
        assert Path(name).name == name and digest(_owned(base/'data'/name)) == sha
    data_proof = verify_data(base/'data')
    official = _owned(base/'preparation/official-model-manifest-original.json', 32*1024**2)
    assert digest(official) == OFFICIAL_SHA
    _verify_official_model(parent, official)
    directory = Path(parent['checkpoint_dir'])/'actor'/f'global_step_{step}'
    assert directory.is_relative_to(parent_base) and directory == directory.resolve()
    receipt_path = _owned(directory/'full-state-complete.json', 32*1024**2)
    receipt = json.loads(receipt_path.read_bytes()); assert receipt['completed_outer'] == step
    driver_path = _owned(receipt['driver']['path'])
    assert driver_path.parent == directory and driver_path.stat().st_size == receipt['driver']['bytes']
    assert digest(driver_path) == receipt['driver']['sha256']
    driver = torch.load(driver_path, map_location='cpu', weights_only=False)
    child = candidate_manifest(parent, receipt, driver, base=base, source_commit=source_commit,
        source_sha256=source, run_id=uuid.uuid4().hex, execution_id=uuid.uuid4().hex,
        started_unix=time.time(), parent_base=parent_base,
        parent_manifest_sha256=digest(parent_path), receipt_sha256=digest(receipt_path),
        experiment_root_run_id=experiment_root_run_id)
    assert not driver['rng']['torch_cuda']; validate_rng(driver['rng'])
    previous_base = os.environ.get('SEARCH_G1_BASE')
    os.environ['SEARCH_G1_BASE'] = str(base)
    try:
        # Proves exact original source/reference/state/Head bytes and absence
        # of the parent's actual trainer birth and its owned session survivors.
        context = read_recovery_context(child, 0)
        state = read_canonical_actor_state(directory, parent, step,
            receipt['canonical_model_receipt_sha256'], base=parent_base)
        finite_count = len(state); del state
        tail = read_unreturned_tail(parent_base, parent, retained_through=step)
    finally:
        if previous_base is None: os.environ.pop('SEARCH_G1_BASE', None)
        else: os.environ['SEARCH_G1_BASE'] = previous_base
    assert finite_count == 435
    ray = Path(child['ray_tmp_dir']); ray.mkdir(mode=0o700, exist_ok=False)
    status = ray.lstat()
    assert ray == ray.resolve() and stat.S_ISDIR(status.st_mode)
    assert status.st_uid == os.getuid() and stat.S_IMODE(status.st_mode) == 0o700
    child['ray_tmp_dir_identity'] = dict(device=status.st_dev, inode=status.st_ino, uid=status.st_uid, mode=0o700)
    for name in ('logs', 'tmp', 'cache/triton', 'cache/inductor', 'cache/hf', 'artifacts'):
        (base/name).mkdir(parents=True, exist_ok=True)
    for name, shared in (('env', '/data/search-g1/env'), ('bootstrap-tools', '/data/search-r1/bootstrap-tools')):
        link = base/name; destination = Path(shared)
        assert destination.is_dir()
        if link.exists() or link.is_symlink():
            assert link.is_symlink() and link.resolve() == destination.resolve()
        else: link.symlink_to(destination, target_is_directory=True)
    _publish_candidate(base, child, dict(
        role='actor_probe_full_recovery_preparation', prepared=True, identity=identity(child),
        origin=namespace_origin(child), recovery_binding_sha256=binding_digest(child['actor_probe_recovery']),
        parent_full_state_receipt_sha256=digest(receipt_path), parent_manifest_sha256=digest(parent_path),
        canonical_finite_tensor_count=finite_count, parent_trainer_session_exit_verified=True,
        data_proof=data_proof, official_model_manifest_sha256=OFFICIAL_SHA,
        discarded_counter_and_rank_tail=tail, production_admission=False,
        required_next='independent deployment review; actual child resource/probe/retrieval and '
                      'restored-origin guard-plan admission; separate launcher and complete chain verifier'))
    return child


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--parent-base', type=Path, required=True)
    parser.add_argument('--step', type=int, choices=(5,50,100,150), required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--experiment-root-run-id', required=True)
    args = parser.parse_args()
    run = prepare(args.base, args.parent_base, args.step, args.source_commit, args.experiment_root_run_id)
    print(json.dumps(dict(identity=identity(run), origin=namespace_origin(run),
                         prepared=True, production_admission=False)))


if __name__ == '__main__': main()
