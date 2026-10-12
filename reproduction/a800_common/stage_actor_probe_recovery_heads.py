"""Stage both saved classifier Heads before a separate recovery service launch.

This CPU-only entry neither refits Heads nor starts a teacher, actor, service or
trainer. The independent immutable archive is complete before live publication.
Actual endpoint/resource/guard-plan and full restore admission remain required.
"""
import argparse
import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path

from reproduction.common8.verify_data import digest
from search_g1.actor_probe_recovery import identity, binding_digest, rebound_probe_manifest

HEADS = (('commit', 'commit_probe.pkl'), ('boundary', 'boundary_probe.pkl'))
LIMIT = 128*1024**2


def _source(path, expected_sha):
    path = Path(path)
    assert path.is_absolute() and path == path.resolve()
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    try:
        before = os.fstat(fd)
        assert stat.S_ISREG(before.st_mode) and before.st_uid == os.getuid()
        assert 0 < before.st_size <= LIMIT
        sha = hashlib.sha256()
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            while True:
                chunk = stream.read(1024**2)
                if not chunk: break
                sha.update(chunk)
        after = os.fstat(fd); current = path.lstat()
        fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
        assert all(getattr(before,k) == getattr(after,k) == getattr(current,k) for k in fields)
        assert sha.hexdigest() == expected_sha, 'Saved Head bytes differ from the checkpoint receipt'
        return before
    finally: os.close(fd)


def _copy_head(source, target, expected_sha):
    before = _source(source, expected_sha)
    fd = os.open(source, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    try:
        opened = os.fstat(fd)
        fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
        assert all(getattr(before,k) == getattr(opened,k) for k in fields)
        sha = hashlib.sha256(); count = 0
        with os.fdopen(fd, 'rb', closefd=False) as src, Path(target).open('xb') as dst:
            while True:
                chunk = src.read(1024**2)
                if not chunk: break
                count += len(chunk); assert count <= LIMIT
                sha.update(chunk); dst.write(chunk)
            dst.flush(); os.fsync(dst.fileno())
        after = os.fstat(fd)
        assert all(getattr(before,k) == getattr(after,k) for k in fields)
        assert count == before.st_size and sha.hexdigest() == expected_sha
        assert digest(target) == expected_sha
    finally: os.close(fd)
    _source(source, expected_sha)


def _sync_directory(path):
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try: os.fsync(fd)
    finally: os.close(fd)


def _publish_live(path, document):
    from reproduction.common8.calibration_runtime import write_once
    # A durable temporary inode is linked exclusively into the live path. This
    # publishes the whole JSON atomically without replacing an existing version.
    temp = path.with_name('restored-manifest.json.pending')
    write_once(temp, document)
    os.link(temp, path, follow_symlinks=False)
    _sync_directory(path.parent)
    temp.unlink(); _sync_directory(path.parent)


def stage_context(base, child, context, preparation_sha):
    """Materialize a previously verified context; scalar tests are not admission."""
    from reproduction.common8.calibration_runtime import write_once
    base = Path(base).resolve(); directory = context['directory']/'probe-state'
    receipt = context['receipt']; binding = context['binding']
    assert binding == child['actor_probe_recovery']
    archive = base/'artifacts'/('recovered-heads-'+child['execution_id'])
    live = base/'artifacts/live/manifest.json'
    marker = base/'logs/actor-probe-recovery-heads-staged.json'
    assert not archive.exists() and not archive.is_symlink()
    assert not live.exists() and not live.is_symlink()
    assert not marker.exists() and not marker.is_symlink()
    targets = {key:archive/name for key,name in HEADS}
    # Derive and validate the pair before writing either classifier.
    manifest = rebound_probe_manifest(context['probe_manifest'], binding, child,
        context['parent'], receipt, targets)
    assert set(receipt['probe_head_hashes']) == {name for _,name in HEADS}
    for _,name in HEADS: _source(directory/name, receipt['probe_head_hashes'][name])
    artifacts = base/'artifacts'
    assert artifacts.is_dir() and artifacts == artifacts.resolve() and artifacts.stat().st_uid == os.getuid()
    live.parent.mkdir(exist_ok=True)
    assert live.parent == live.parent.resolve() and live.parent.stat().st_uid == os.getuid()
    archive.mkdir(mode=0o700, exist_ok=False)
    for key,name in HEADS: _copy_head(directory/name, targets[key], receipt['probe_head_hashes'][name])
    saved = archive/'version-manifest.json'; write_once(saved, manifest)
    _sync_directory(archive); _sync_directory(artifacts)
    assert all(digest(targets[key]) == receipt['probe_head_hashes'][name] for key,name in HEADS)
    _publish_live(live, manifest)
    assert json.loads(live.read_bytes()) == json.loads(saved.read_bytes()) == manifest
    proof = dict(role='actor_probe_restored_heads_staged', staged=True,
        identity=identity(child), recovery_binding_sha256=binding_digest(binding),
        preparation_sha256=preparation_sha, parent_full_state_receipt_sha256=binding['full_state_receipt_sha256'],
        parent_probe_manifest_sha256=receipt['probe_manifest_sha256'],
        version=manifest['version'], actor_step=manifest['actor_step'], round=manifest['round'],
        head_sha256=manifest['head_sha256'], archive_manifest=str(saved),
        archive_manifest_sha256=digest(saved), live_manifest_sha256=digest(live),
        calibration_data_sha256=manifest['calibration_data_sha256'],
        state_semantics='exact saved independent binary Heads; namespace and file paths rebound only; no refit',
        production_admission=False,
        required_next='actual restored-Head endpoint/resource/retrieval/guard-plan admission and separate full-state launcher')
    write_once(marker, proof)
    return proof


def stage(base):
    from reproduction.common8.topology import validate_base
    from reproduction.common8.formal_contract import assert_formal_run
    from search_g1.actor_probe_namespace import namespace_origin
    from search_g1.actor_probe_recovery import read_recovery_context
    from reproduction.a800_common.prepare_actor_probe_recovery import _owned
    from reproduction.common8.source_files import runtime_files
    base = Path(base).resolve()
    assert os.environ['SEARCH_G1_BASE'] == str(base)
    path = _owned(base/'logs/run-manifest.json', 32*1024**2)
    run = json.loads(path.read_bytes()); validate_base(base, run); assert_formal_run(run)
    assert Path(__file__).resolve().parents[2] == base/'repo'
    assert os.environ.get('SEARCH_G1_TRAINING_WORLD_SIZE') == '4'
    repo = base/'repo'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip() == run['source_commit']
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo).strip()
    actual_source = {str(p.relative_to(repo)):digest(_owned(p)) for p in runtime_files(repo)}
    assert actual_source == run['source_sha256'], 'Restored Head entry must be included in the pinned child source'
    prep_path = _owned(base/'logs/actor-probe-recovery-preparation.json', 32*1024**2)
    prep = json.loads(prep_path.read_bytes())
    assert prep['role'] == 'actor_probe_full_recovery_preparation' and prep['prepared'] is True
    assert prep['production_admission'] is False and prep['identity'] == identity(run)
    assert prep['candidate_manifest_sha256'] == digest(path)
    assert prep['candidate_manifest'] == 'logs/actor-probe-recovery-candidate.json'
    candidate = _owned(base/prep['candidate_manifest'], 32*1024**2)
    assert digest(candidate) == prep['candidate_manifest_sha256'] and json.loads(candidate.read_bytes()) == run
    assert prep['origin'] == namespace_origin(run)
    assert prep['recovery_binding_sha256'] == binding_digest(run['actor_probe_recovery'])
    assert prep['parent_trainer_session_exit_verified'] is True and prep['canonical_finite_tensor_count'] == 435
    assert prep['parent_manifest_sha256'] == run['actor_probe_recovery']['parent_manifest_sha256']
    assert prep['parent_full_state_receipt_sha256'] == run['actor_probe_recovery']['full_state_receipt_sha256']
    for name in ('process-ledger.json','pipeline-launch-claim.json','training-launch-claim.json',
                 'actual-updates.jsonl','actual-attempts.jsonl'):
        p = base/'logs'/name
        assert not p.exists() and not p.is_symlink(), 'Stage Heads before this candidate is launched'
    assert not list((base/'logs').glob('*-rank-*.jsonl'))
    for name,sha in run['input_sha256'].items():
        assert Path(name).name == name and digest(_owned(base/'data'/name)) == sha
    context = read_recovery_context(run, 0)
    # Deserialize on CPU only after exact checkpoint hash/identity verification.
    import joblib
    import numpy as np
    from reproduction.a800.recovery_binding import finite_fitted_values
    from search_g1.actor_probe_contract import LAYER_CHOICES
    for key,name in HEADS:
        fitted = joblib.load(context['directory']/'probe-state'/name)
        assert list(fitted['clf'].classes_) == [0,1] and np.isfinite(fitted['auc']) and 0 <= fitted['auc'] <= 1
        assert fitted['layer'] in LAYER_CHOICES
        assert (fitted['position'],fitted['label']) == (('Xa','flip') if key == 'commit' else ('Xp','closedbook'))
        finite_fitted_values(vars(fitted['clf']))
    return stage_context(base, run, context, digest(prep_path))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--base',type=Path,required=True)
    args = parser.parse_args(); print(json.dumps(stage(args.base)))


if __name__ == '__main__': main()
