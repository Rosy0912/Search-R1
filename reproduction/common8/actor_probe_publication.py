"""Durably publish an already fitted actor-only pair, without loading a model.

The original archive is immutable evidence. Its final name must never expose
partial JSON, because final acceptance parses every candidate archive. The live
rename is the commit point: a subsequent directory-sync failure propagates and
retains evidence; it does not imply rollback or permit training to continue.
"""
import json
import os
import re
import stat
import uuid
from pathlib import Path

from reproduction.common8.verify_data import digest
from search_g1.actor_probe_contract import actor_probes


def _owned(path, *, directory=False):
    path = Path(path)
    assert path.is_absolute() and path == path.resolve(), 'Noncanonical publication dependency'
    info = path.stat()
    assert info.st_uid == os.getuid()
    assert (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode))
    return path


def _sync_file(path):
    path = _owned(path)
    with path.open('rb') as stream:
        os.fsync(stream.fileno())


def _sync_dir(path):
    path = _owned(path, directory=True)
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _sync_ancestors(path, base):
    path = Path(path); base = Path(base)
    assert path.is_relative_to(base)
    while True:
        _sync_dir(path)
        if path == base:
            # The namespace itself was mkdir'ed during preparation. Syncing
            # base persists its contents, not its entry in the owning parent.
            _sync_dir(base.parent)
            return
        path = path.parent


def _names(hashes):
    assert isinstance(hashes, dict) and hashes
    for name, sha in hashes.items():
        assert isinstance(name, str) and Path(name).name == name and name not in ('.', '..')
        assert re.fullmatch('[0-9a-f]{64}', sha)
    return hashes


def sync_actor_export(base, run, step, encoder, encoder_hashes=None):
    """Sync validated model/receipt/tokenizer dependencies, including atstep5.

    The canonical producer already checks real tensor finiteness and hashes, and
    refit rehashes its encoder after fitting. This function checks that metadata
    binding and syncs the existing files; it performs no extra tensor diagnostic.
    """
    assert actor_probes(run) and type(step) is int and 0 <= step <= 200
    base = _owned(base, directory=True); encoder = _owned(encoder, directory=True)
    if step == 0:
        assert encoder == Path(run['initial_model'])
        expected = {name: sha for name, sha in run['initial_model_sha256'].items()
                    if name not in ('vocab.json', 'merges.txt', 'generation_config.json')}
    else:
        assert encoder == Path(run['checkpoint_dir'])/'actor'/f'global_step_{step}'
        assert encoder.is_relative_to(base)
        receipt = _owned(encoder/'complete.json')
        canonical = json.loads(receipt.read_bytes())
        assert canonical['run_id'] == run['run_id'] and type(canonical['step']) is int and canonical['step'] == step
        assert canonical['finite_all_saved_tensors'] is True
        expected = dict(_names(canonical['files_sha256']))
        from reproduction.common8.verify_full import checkpoint_tokenizer_hashes
        # HF save_pretrained may normalize BPE merge serialization. Reuse the
        # existing semantic check; require the actual saved tokenizer hashes.
        expected.update(checkpoint_tokenizer_hashes(encoder, Path(run['initial_model'])))
    _names(expected)
    if encoder_hashes is not None:
        assert encoder_hashes == expected, 'Fitted export differs from canonical model/tokenizer binding'
    for name in expected:
        _sync_file(encoder/name)
    if step:
        _sync_file(receipt)
        _sync_ancestors(encoder, base)
    else:
        _sync_dir(encoder)
        _sync_dir(encoder.parent)


def _write_temp(directory, label, raw):
    path = directory/('.'+label+'-'+uuid.uuid4().hex+'.tmp')
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return path


def publish_fitted_pair(base, run, round_dir, payload):
    """Return only after artifacts, original archive, live pair and dirs sync."""
    from reproduction.common8.actor_probe_head_service import verify_manifest
    verify_manifest(run, payload)
    assert 'restored_from' not in payload and 'adoption_origin' not in payload
    base = _owned(base, directory=True); round_dir = _owned(round_dir, directory=True)
    artifacts = _owned(base/'artifacts', directory=True)
    step = payload['round']
    assert round_dir.parent == artifacts
    assert re.fullmatch('round_'+str(step)+'-attempt-[0-9a-f]{32}', round_dir.name)
    assert re.fullmatch(re.escape(run['run_id'])+':'+str(step)+':[0-9a-f]{32}', payload['version'])
    assert payload['fitting_input_semantics'] == 'immutable current-actor export; production features come from current actor RPC'
    assert payload['fitted_state_semantics'] == 'serialized sklearn classifier/scaler parameters; no GRPO optimizer ownership'
    archive = round_dir/'version-manifest.json'
    assert not archive.exists(), 'Original fitting archive is immutable'
    raw = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
    heads = _owned(round_dir/'heads', directory=True)
    dumps = _owned(round_dir/'dumps', directory=True)
    head_hashes = dict(payload['head_sha256'], **{'calibration-report.json': payload['calibration_report_sha256']})
    for directory, hashes in ((heads, head_hashes), (dumps, payload['dump_sha256'])):
        _names(hashes)
        assert {p.name for p in directory.iterdir()} == set(hashes)
        for name, sha in hashes.items():
            path = _owned(directory/name)
            assert digest(path) == sha, 'Fitted publication artifact changed'
            _sync_file(path)
        _sync_dir(directory)
    assert payload['commit'] == str(heads/'commit_probe.pkl')
    assert payload['boundary'] == str(heads/'boundary_probe.pkl')
    assert json.loads((heads/'calibration-report.json').read_bytes()) == payload['calibration_report']
    assert json.loads((dumps/'calibration-label-proof.json').read_bytes()) == payload['label_proof']
    _sync_file(round_dir/'calibration.log')
    sync_actor_export(base, run, step, Path(payload['encoder']), payload['encoder_sha256'])
    _sync_ancestors(round_dir, base)
    candidate = _write_temp(round_dir, 'version-manifest', raw)
    # Exclusive linking publishes a complete fsynced inode without overwriting
    # an existing original archive, even if another writer won the race.
    os.link(candidate, archive)
    candidate.unlink()
    _sync_dir(round_dir)
    _sync_dir(artifacts)
    live = artifacts/'live'
    live.mkdir(exist_ok=True)
    _owned(live, directory=True)
    _sync_dir(artifacts)
    candidate = _write_temp(live, 'manifest', raw)
    os.replace(candidate, live/'manifest.json')
    # Do not catch this error and roll back: selection already changed. The
    # caller must fail closed, preserve evidence and authenticate actual state.
    _sync_dir(live)
    return payload['version']
