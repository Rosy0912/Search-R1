"""Authenticate original fitted Heads across retained recovery namespaces.

Restored Head archives contain only the saved classifiers and a rebound
manifest. Their genuine calibration arrays/report remain in the original
fitting namespace; this reader follows that provenance rather than pretending
that restoring a pair performed a new fit. No training or acceptance is written.
"""
import math
import re
from pathlib import Path

from reproduction.common8.actor_probe_rank_tail import _sealed_bytes
from reproduction.common8.policy_calibration_plan import _strict_json, document_sha256
from reproduction.common8.verify_data import digest
from search_g1.actor_probe_contract import head_round, checkpoint_head_round, validate_fitted_version
from search_g1.actor_probe_recovery import identity


def _document(base, path):
    path = Path(path)
    assert path.is_absolute() and path == path.resolve() and path.is_relative_to(base)
    raw, file = _sealed_bytes(base,path.relative_to(base).as_posix(),limit=32*1024**2)
    return _strict_json(raw), file


def select_versions(entries, segments):
    """Retained readouts plus the genuinely saved post-COMPLETE200 pair."""
    from reproduction.common8.actor_probe_chain import assemble_counter_chain
    assert len(entries) == len(segments)
    chain = assemble_counter_chain([s['counter_segment'] for s in segments],final_outer=200)
    versions = {}; namespaces = {}
    for entry, segment in zip(entries,segments):
        run = entry['run']
        assert segment['identity'] == identity(run) and segment['base'] == entry['base']
        assert segment['first'] == entry['origin']['completed_outer']+1 and segment['last'] == entry['last']
        assert segment['actual_interval_accepted'] is True
        for row in segment['counter_segment']['retained_updates']:
            step = row['step']; round_id = head_round(step); version = row['probe_version']
            assert isinstance(version,str) and re.fullmatch('[0-9a-f]{32}:'+str(round_id)+':[0-9a-f]{32}',version)
            assert chain['logical_owners'][step] == run['run_id']
            assert versions.setdefault(round_id,version) == version, 'Head changed inside a ten-outer interval'
            namespaces[(run['run_id'],round_id)] = version
            binding = row['probe_actor_binding']
            assert binding['run_id'] == run['run_id'] and binding['execution_id'] == run['execution_id']
            assert binding['head_version'] == version
            assert type(binding['logical_update']) is int and binding['logical_update'] == step
            assert type(binding['completed_outer']) is int and binding['completed_outer'] == step-1
            assert type(binding['optimizer_writes']) is int and binding['optimizer_writes'] == row['optimizer_writes_before']
    assert set(versions) == set(range(0,200,10))
    # Row200 still used Head190. Its owner's final checkpoint, rather than a
    # fictitious row201, authenticates the pair fitted after COMPLETE200.
    terminal_owner = chain['logical_owners'][200]
    terminal = [s for s in segments if s['identity']['run_id'] == terminal_owner]
    assert len(terminal) == 1 and terminal[0]['last'] == 200
    state = terminal[0]['full_states']['200']
    assert state['identity'] == terminal[0]['identity'] and state['completed_outer'] == 200
    assert state['probe_actor_step'] == 200 and state['completed_outer_reward_head_round'] == 190
    version = state['probe_version']
    assert isinstance(version,str) and re.fullmatch(terminal_owner+':200:[0-9a-f]{32}',version)
    assert terminal[0]['refresh_isolation']['200']['version'] == version
    versions[200] = version; namespaces[(terminal_owner,200)] = version
    root = entries[0]['run']['run_id']
    owners = {round_id:root if round_id == 0 else chain['logical_owners'][round_id] for round_id in versions}
    assert all(version.startswith(owners[round_id]+':') for round_id,version in versions.items()), \
        'Head was not fitted by the retained actor owner at that boundary'
    return versions, owners, namespaces


def verify_original(base, run, path, round_id, version):
    from reproduction.common8.actor_probe_head_service import verify_manifest, load_heads
    from reproduction.common8 import adopt_initial_probes as artifacts
    from reproduction.common8.formal_inline_safety import owned_path
    from reproduction.common8.actor_probe_segment_acceptance import verify_canonical_files
    from reproduction.common8.verify_full import checkpoint_tokenizer_hashes
    base = Path(base); manifest, file = _document(base,path)
    assert path.parent.parent == base/'artifacts'
    assert re.fullmatch('round_'+str(round_id)+'-attempt-[0-9a-f]{32}',path.parent.name)
    assert manifest['run_id'] == run['run_id'] and manifest['version'] == version
    assert manifest['round'] == round_id and 'restored_from' not in manifest and 'adoption_origin' not in manifest
    verify_manifest(run,manifest)
    assert manifest['fitting_input_semantics'] == 'immutable current-actor export; production features come from current actor RPC'
    assert manifest['fitted_state_semantics'] == 'serialized sklearn classifier/scaler parameters; no GRPO optimizer ownership'
    assert type(manifest['published_unix']) in (int,float) and math.isfinite(manifest['published_unix'])
    assert manifest['published_unix'] >= run['started_unix']
    encoder = Path(manifest['encoder']); official = Path(run['initial_model'])
    assert encoder == encoder.resolve()
    if round_id == 0:
        assert encoder == official
        expected = {name:sha for name,sha in run['initial_model_sha256'].items()
            if name not in ('vocab.json','merges.txt','generation_config.json')}
        initial, initial_file = _document(base,base/'logs/formal-inline-initial-probe.json')
        assert initial['schema_version'] == 1 and initial['role'] == 'fresh200_actual_initial_probe'
        assert initial['run_id'] == run['run_id'] and initial['version'] == run['initial_probe_version'] == version
        assert initial['source_map_sha256'] == document_sha256(run['source_sha256'])
        assert initial['model_revision'] == run['model_revision']
        assert initial['initial_model_sha256'] == run['initial_model_sha256'] and initial['input_sha256'] == run['input_sha256']
        assert initial['archive'] == path.relative_to(base).as_posix() and initial['archive_sha256'] == file['sha256']
    else:
        assert encoder == Path(run['checkpoint_dir'])/'actor'/f'global_step_{round_id}'
        snapshot, initial_file = _document(base,encoder/'complete.json')
        assert snapshot['run_id'] == run['run_id'] and type(snapshot['step']) is int and snapshot['step'] == round_id
        assert snapshot['finite_all_saved_tensors'] is True
        verify_canonical_files(base,encoder,snapshot)
        expected = dict(snapshot['files_sha256'])
        expected.update(checkpoint_tokenizer_hashes(encoder,official))
    assert manifest['encoder_sha256'] == expected
    for name, sha in expected.items():
        assert Path(name).name == name and re.fullmatch('[0-9a-f]{64}',sha)
        actual = encoder/name if round_id == 0 else owned_path(base,(encoder/name).relative_to(base).as_posix())
        assert actual == actual.resolve() and digest(actual) == sha
    heads = path.parent/'heads'; dump = path.parent/'dumps'
    assert manifest['commit'] == str(heads/'commit_probe.pkl') and manifest['boundary'] == str(heads/'boundary_probe.pkl')
    hashes = dict(manifest['head_sha256'], **{'calibration-report.json':manifest['calibration_report_sha256']})
    assert manifest['dump_sha256'] and all(re.fullmatch('[0-9a-f]{64}',sha)
        for sha in list(hashes.values())+list(manifest['dump_sha256'].values()))
    files = {**artifacts._files(heads,hashes), **artifacts._files(dump,manifest['dump_sha256'])}
    report, _ = _document(base,heads/'calibration-report.json')
    proof, _ = _document(base,dump/'calibration-label-proof.json')
    assert report == manifest['calibration_report'] and proof == manifest['label_proof']
    artifacts._calibration_arrays(dump,report,proof)
    artifacts._fitted_heads(heads,report)
    load_heads(base,run,manifest)
    assert all(digest(p) == sha for p,sha in files.items()), 'Calibration artifacts changed during revalidation'
    assert digest(path) == file['sha256']
    return manifest, dict(version=version,round=round_id,original_identity=identity(run),
        original_archive=file,encoder_proof=initial_file,head_sha256=manifest['head_sha256'],
        calibration_report_sha256=manifest['calibration_report_sha256'],
        calibration_data_sha256=manifest['calibration_data_sha256'],
        encoder=str(encoder),encoder_files_sha256=expected,
        artifact_files_sha256={p.relative_to(base).as_posix():sha for p,sha in files.items()},
        artifact_map_sha256=document_sha256({str(p.relative_to(base)):sha for p,sha in files.items()}),
        actual_calibration_arrays_and_fitted_heads_revalidated=True)


def verify_rebound(parent_entry, child_entry, parent_segment):
    from search_g1.actor_probe_recovery import rebound_probe_manifest, binding_digest
    from reproduction.common8.actor_probe_head_service import head_paths
    parent = parent_entry['run']; child = child_entry['run']; base = Path(child_entry['base'])
    step = child_entry['origin']['completed_outer']; receipt = parent_segment['full_states'][str(step)]
    assert receipt['completed_outer'] == parent_entry['last'] == step
    old = Path(parent['checkpoint_dir'])/'actor'/f'global_step_{step}'/'probe-state/version-manifest.json'
    saved, file = _document(Path(parent_entry['base']),old)
    assert file['sha256'] == receipt['probe_manifest_sha256']
    archive = base/'artifacts'/('recovered-heads-'+child['execution_id'])/'version-manifest.json'
    actual, archive_file = _document(base,archive)
    paths = head_paths(base,actual)
    expected = rebound_probe_manifest(saved,child['actor_probe_recovery'],child,parent,receipt,paths)
    assert actual == expected, 'Restored Head changed more than namespace, paths and its authenticated parent binding'
    assert set(p.name for p in archive.parent.iterdir()) == {'version-manifest.json','commit_probe.pkl','boundary_probe.pkl'}
    staged, stage_file = _document(base,base/'logs/actor-probe-recovery-heads-staged.json')
    assert staged['role'] == 'actor_probe_restored_heads_staged' and staged['staged'] is True
    assert staged['identity'] == identity(child) and staged['production_admission'] is False
    assert staged['recovery_binding_sha256'] == binding_digest(child['actor_probe_recovery'])
    assert staged['parent_full_state_receipt_sha256'] == child['actor_probe_recovery']['full_state_receipt_sha256']
    assert staged['parent_probe_manifest_sha256'] == file['sha256']
    assert staged['archive_manifest'] == str(archive) and staged['archive_manifest_sha256'] == archive_file['sha256']
    for key in ('version','actor_step','round','head_sha256','calibration_data_sha256'):
        assert staged[key] == actual[key]
    preparation, prep_file = _document(base,base/'logs/actor-probe-recovery-preparation.json')
    assert preparation['identity'] == identity(child) and staged['preparation_sha256'] == prep_file['sha256']
    return actual, dict(parent_saved_manifest=file,child_archive=archive_file,stage_receipt=stage_file)


def verify_calibration_chain(entries, segments):
    versions, owners, namespaces = select_versions(entries,segments)
    by_run = {e['run']['run_id']:e for e in entries}; originals = {}; proofs = {}
    for round_id,version in versions.items():
        entry = by_run[owners[round_id]]; base = Path(entry['base'])
        matches = []
        for path in (base/'artifacts').glob(f'round_{round_id}-attempt-*/version-manifest.json'):
            manifest, _ = _document(base,path)
            if manifest['version'] == version: matches.append(path)
        assert len(matches) == 1, 'Adopted version has no unique original fit archive'
        originals[version], proofs[str(round_id)] = verify_original(base,entry['run'],matches[0],round_id,version)
    local = {(owners[r],r):originals[v] for r,v in versions.items()}; rebound = {}
    for index in range(1,len(entries)):
        entry = entries[index]; actual, proof = verify_rebound(entries[index-1],entry,segments[index-1])
        round_id = actual['round']; version = actual['version']
        assert versions[round_id] == version
        previous = local[(entries[index-1]['run']['run_id'],round_id)]
        assert {k:v for k,v in actual.items() if k not in ('run_id','commit','boundary','restored_from')} == \
            {k:v for k,v in previous.items() if k not in ('run_id','commit','boundary','restored_from')}
        local[(entry['run']['run_id'],round_id)] = actual; rebound[entry['run']['run_id']] = proof
    for (run_id,round_id), version in namespaces.items():
        manifest = local[(run_id,round_id)]
        validate_fitted_version(by_run[run_id]['run'],manifest)
        assert manifest['version'] == version
    for entry,segment in zip(entries,segments):
        run = entry['run']; base = Path(entry['base'])
        for step,state in segment['full_states'].items():
            step = int(step); round_id = checkpoint_head_round(step)
            manifest = local[(run['run_id'],round_id)]
            path = Path(run['checkpoint_dir'])/'actor'/f'global_step_{step}'/'probe-state/version-manifest.json'
            saved, file = _document(base,path)
            assert file['sha256'] == state['probe_manifest_sha256']
            assert state['probe_head_hashes'] == saved['head_sha256'] == manifest['head_sha256']
            assert {k:v for k,v in saved.items() if k not in ('commit','boundary')} == \
                {k:v for k,v in manifest.items() if k not in ('commit','boundary')}
            assert saved['commit'] == str(path.parent/'commit_probe.pkl') and saved['boundary'] == str(path.parent/'boundary_probe.pkl')
    return dict(role='actor_probe_original_calibration_chain',original_versions=proofs,
        restored_pairs=rebound,retained_version_intervals_verified=True,
        original_calibration_provenance_verified=True,full_training_or_final_acceptance=False)
