"""Copy authenticated discovery round0 heads, never its admission or Adam state.

This CPU-only producer runs inside the new native allocation. Source evidence
describes a completed historical child; its old PIDs are never called live. The
new version still requires this run's real HTTP/encoder/coexistence admissions.
An exclusive durable claim makes partial copies fail closed without overwrites.
"""
import argparse
import hashlib
import json
import math
import os
import time
import uuid
from collections import Counter
from pathlib import Path

from reproduction.common8.policy_calibration_plan import (
    CONFIRMATION_ROLE, DISCOVERY_ROLE, PlanError, _identity, _strict_json,
    canonical_bytes, load_plan,
)
from reproduction.common8.freeze_calibration_plan import _unstarted
from reproduction.common8.verify_data import digest


def _check(ok, message):
    if not ok:
        raise PlanError(message)


def _read(path):
    return _strict_json(Path(path).read_bytes())


def _owned_file(path, root):
    path, root = Path(path), Path(root).resolve()
    _check(path.is_absolute() and not path.is_symlink() and path.is_file(), 'Not a regular owned source file')
    resolved = path.resolve()
    _check(resolved.is_relative_to(root) and resolved.stat().st_uid == os.getuid(), 'Source file outside owned namespace')
    relative = resolved.relative_to(root)
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        _check(not cursor.is_symlink(), 'Symlink in artifact path')
    return resolved


def _files(directory, expected):
    directory = Path(directory)
    _check(not directory.is_symlink() and directory.is_dir() and
           set(p.name for p in directory.iterdir()) == set(expected), 'Artifact file set differs')
    result = {}
    for name, sha in expected.items():
        _check(Path(name).name == name and name not in ('.', '..'), 'Artifact filename escapes its directory')
        path = _owned_file(directory / name, directory)
        _check(digest(path) == sha, 'Actual artifact bytes differ: ' + name)
        result[path] = sha
    return result


def _actual_target(base, run):
    """Resource/source admission only; this cannot pass a new model GPU gate."""
    from reproduction.common8.stage_retrieval import binding, memory_bytes
    from reproduction.common8.native_allocation import cuda_uuids
    from reproduction.common8.source_files import runtime_files
    actual, job, allocation = binding(base)
    _check(actual == run and memory_bytes(job['TRES']) >= 512 * 2**30, 'Wrong actual native 512GiB allocation')
    expected = allocation['actual_cuda_runtime_uuids']
    _check(len(expected) == len(set(expected)) == 8 and cuda_uuids(8) == expected,
           'Actual allocated eight CUDA identities differ')
    actual_sources = {str(p.relative_to(base / 'repo')): digest(p) for p in runtime_files(base / 'repo')}
    _check(actual_sources == run['source_sha256'], 'Actual complete runtime source differs')
    for name, sha in run['input_sha256'].items():
        _check(digest(_owned_file(base / 'data' / name, base / 'data')) == sha, 'Actual common input differs')
    for name, sha in run['initial_model_sha256'].items():
        _check(digest(Path(run['initial_model']) / name) == sha, 'Actual official actor/reference bytes differ')
    return job['JobId']


def _segment(base, run, plan, sha):
    """Re-read complete historical execution evidence without reviving its PIDs."""
    from reproduction.common8.safety_calibration_evidence import verify_segment
    from reproduction.common8.calibration_runtime import verify_optimizer_states
    from reproduction.common8.freeze_calibration_plan import _mathematics
    from reproduction.common8.production_safety_admission import _smoke, _paired_evaluations
    path = _owned_file(base / 'logs/policy-calibration-segment.json', base)
    proof = _read(path)
    for key, value in dict(role=DISCOVERY_ROLE + '_evidence', origin='actual_native_policy_segment',
                           run_id=run['run_id'], parent_production_run_id=plan['parent_production_run_id'],
                           slurm_job_id=plan['slurm_job_id'], plan_sha256=sha, identity=plan['identity'],
                           initial_reward_fingerprint=plan['initial_reward_fingerprint'],
                           resolved_config_sha256=plan['resolved_config_sha256']).items():
        _check(canonical_bytes(proof[key]) == canonical_bytes(value), 'Discovery segment identity differs: ' + key)
    _check(type(proof['ranks']) is list and all(type(r) is int for r in proof['ranks']) and
           proof['ranks'] == list(range(8)) and type(proof['completed_outer_updates']) is int and
           proof['completed_outer_updates'] == 1 and type(proof['optimizer_writes_per_rank']) is int and
           proof['optimizer_writes_per_rank'] == 20 and proof['production_admission'] is False and
           proof['healthy_checkpoint_accepted'] is False, 'Discovery has no complete bounded segment')
    files = {path: digest(path)}
    logs, states = {}, {}
    for rank in range(8):
        log = _owned_file(base / 'logs' / f'optimizer-rank-{rank}.jsonl', base)
        state = _owned_file(base / 'logs' / f'policy-calibration-optimizer-rank-{rank}.json', base)
        _check(digest(log) == proof['worker_log_sha256'][str(rank)] and
               digest(state) == proof['optimizer_state_proof_sha256'][str(rank)], 'Historical rank evidence changed')
        logs[rank] = [_strict_json(line.encode()) for line in log.read_text().splitlines()]
        states[rank] = _read(state); files.update({log: digest(log), state: digest(state)})
    actual = verify_segment(logs, dict(schema_version=1, role=DISCOVERY_ROLE, run_id=run['run_id'],
                            calibration_sha256=sha, outer_count=1, initial_optimizer_writes=0))
    _check(canonical_bytes(actual) == canonical_bytes(proof['structural']), 'Historical structural evidence differs')
    allocation_path = _owned_file(base / 'logs/native-allocation-admission.json', base)
    allocation = _read(allocation_path); files[allocation_path] = digest(allocation_path)
    config_path = _owned_file(base / 'logs/resolved-training-config.json', base)
    config = _read(config_path)
    _check(config['run_id'] == run['run_id'] and config['slurm_job_id'] == plan['slurm_job_id'] and
           hashlib.sha256(canonical_bytes(config['resolved_config'])).hexdigest() ==
           config['resolved_config_sha256'] == plan['resolved_config_sha256'], 'Historical actual resolved recipe differs')
    _mathematics(config['resolved_config'], base, run)
    files[config_path] = digest(config_path)
    historical = {}
    # Reuse the actual writer-schema readers. These reconstruct raw eight-rank
    # .05/FP32/zero-Adam and paired512 arithmetic; never consult old live PIDs.
    _smoke(base, run, plan, allocation, historical)
    _paired_evaluations(base, run, plan, sha, proof, historical, '')
    files.update({_owned_file(base / name, base): hashlib.sha256(raw).hexdigest()
                  for name, raw in historical.items()})
    actual_states = verify_optimizer_states(states, run, plan, sha, allocation)
    _check(actual_states == proof['actual_optimizer_states'], 'Historical optimizer summaries differ')
    births = proof['actual_worker_births_at_segment_close']
    _check(len(births) == 8, 'No historical worker birth snapshots')
    for state, birth in zip(actual_states, births):
        _check(birth['pid'] == state['pid'] and birth['start_ticks'] == state['start_ticks'] and
               type(birth['sid']) is int and birth['sid'] > 0 and type(birth['uid']) is int and birth['uid'] == os.getuid() and
               isinstance(birth['state'], str) and len(birth['state']) == 1 and birth['state'] != 'Z',
               'Historical worker identity differs')
    for name, key, nested in (
            ('policy-calibration-training-inputs.json', 'actual_training_inputs', ('training_input_proof_sha256', 'parents')),
            ('policy-calibration-training-groups.json', 'actual_training_groups', ('training_input_proof_sha256', 'groups')),
            ('policy-calibration-evaluation-before.json', 'evaluation_before', ('evaluation_sha256', 'before')),
            ('policy-calibration-evaluation-after.json', 'evaluation_after', ('evaluation_sha256', 'after'))):
        item = _owned_file(base / 'logs' / name, base)
        _check(digest(item) == proof[nested[0]][nested[1]] and _read(item) == proof[key], 'Historical measurement bytes changed')
        files[item] = digest(item)
    inputs, groups = proof['actual_training_inputs'], proof['actual_training_groups']
    _check(inputs['actual_parent_count'] == 512 and len(inputs['parents']) == 512 and inputs['n_agent'] == 5 and
           inputs['actual_trajectory_count'] == groups['actual_trajectory_count'] == 2560 and
           inputs['train_sha256'] == run['input_sha256']['train.parquet'], 'Historical actual batch differs')
    repeated = [row['upstream_group_sha256'] for row in inputs['parents'] for _ in range(5)]
    _check(inputs['repeated_group_sha256'] == repeated and groups['upstream_group_multiset_unchanged'] is True and
           Counter(groups['balanced_group_sha256']) == Counter(repeated), 'Historical G5 groups differ')
    before, after = proof['evaluation_before'], proof['evaluation_after']
    _check(before['question_ids'] == after['question_ids'] and len(before['question_ids']) == 512 and
           before['data_sources'] == after['data_sources'] and Counter(before['data_sources']) == Counter(nq=256, hotpotqa=256),
           'Discovery did not measure the same fixed512')
    for step, evaluation in ((0, before), (1, after)):
        _check(type(evaluation['step']) is int and evaluation['step'] == step and
               evaluation['run_id'] == run['run_id'] and evaluation['plan_sha256'] == sha and
               evaluation['validation_sha256'] == run['input_sha256']['validation.parquet'], 'Historical evaluation identity differs')
    updates_path = _owned_file(base / 'logs/actual-updates.jsonl', base)
    updates = [_strict_json(line.encode()) for line in updates_path.read_text().splitlines()]
    _check(updates == proof['actual_updates'] and len(updates) == 1 and type(updates[0]['step']) is int and
           updates[0]['step'] == 1 and updates[0]['run_id'] == run['run_id'] and
           updates[0]['probe_version'] == run['initial_probe_version'] and
           updates[0]['metrics']['actor/optimizer_steps_this_update'] == 20, 'No actual completed discovery outer')
    _check(all(type(v) in (int, float) and math.isfinite(v) for v in updates[0]['metrics'].values()), 'Nonfinite completed metrics')
    files[updates_path] = digest(updates_path)
    checkpoint = Path(run['checkpoint_dir']) / 'actor/global_step_1'
    saved = _read(_owned_file(checkpoint / 'complete.json', base))
    _check(saved == proof['checkpoint'] and saved['run_id'] == run['run_id'] and saved['step'] == 1 and
           type(saved['validated_tensor_count']) is int and saved['validated_tensor_count'] == 435 and
           saved['checkpoint'] == str(checkpoint) and saved['finite_all_saved_tensors'] is True and
           bool(proof['changed_parameters']), 'No complete finite changed discovery checkpoint proof')
    files[checkpoint / 'complete.json'] = digest(checkpoint / 'complete.json')
    # Saved tokenizer extras are permitted: the canonical producer fingerprints
    # every serialized weight shard/index/config, rather than all directory files.
    for name, expected in saved['files_sha256'].items():
        item = _owned_file(checkpoint / name, checkpoint)
        _check(Path(name).name == name and digest(item) == expected, 'Historical saved weight bytes changed')
        files[item] = expected
    return proof, files


def _calibration_arrays(dump, report, proof):
    import numpy as np
    npzs = list(dump.glob('*.npz'))
    _check(len(npzs) == 1, 'Common8 initial calibration requires its single genuine NPZ')
    with np.load(npzs[0], allow_pickle=False) as arrays:
        for name in arrays.files:
            value = arrays[name]
            _check(value.dtype.kind != 'O', 'Pickled calibration arrays forbidden')
            if value.dtype.kind in 'fc':
                _check(np.isfinite(value).all(), 'Nonfinite genuine calibration array: ' + name)
        ids = arrays['boundary_question_id'].tolist()
        _check(ids == proof['question_ids'] and len(set(ids)) == 300, 'Actual closed-book calibration IDs differ')
        labels = arrays['boundary_q_cb']
        searched = arrays['y_flip']; em = arrays['y_em']; cb = arrays['q_cb']
        _check(labels.shape == (300,) and all(np.isin(v, (0, 1)).all() for v in (labels, searched, em, cb)), 'Nonbinary actual labels')
        count = proof['searched_valid_trajectory_count']
        _check(searched.shape == em.shape == cb.shape == (count,) and count == report['commit_trajectory_count'], 'Actual searched label count differs')
        _check(np.bincount(labels.astype(int), minlength=2).tolist() == report['boundary_class_counts'] == proof['boundary_class_counts'] and
               np.bincount(searched.astype(int), minlength=2).tolist() == report['commit_class_counts'] == proof['reliance_class_counts'], 'Actual class counts differ')
        gid, qids = arrays['gid'], arrays['question_id'].tolist()
        _check(gid.shape == (count,) and gid.dtype.kind in 'iu' and len(qids) == count and
               all(0 <= int(g) < 300 and q == ids[int(g)] for g, q in zip(gid, qids)), 'Actual trajectory question groups differ')
        _check(np.array_equal(cb, labels[gid]), 'Trajectory closed-book labels differ from actual question labels')
        records = list(dump.glob('*.jsonl'))
        _check(len(records) == 1, 'Missing paired actual trajectory records')
        rows = [_strict_json(line.encode()) for line in records[0].read_text().splitlines()]
        _check(len(rows) == count, 'Raw trajectory record count differs from actual arrays')
        for index, row in enumerate(rows):
            for key, value in (('qi', int(gid[index])), ('em', int(em[index])),
                               ('flip', int(searched[index])), ('cb', int(cb[index]))):
                _check(type(row[key]) is int and row[key] == value, 'Raw trajectory labels/groups differ: ' + key)
            _check(isinstance(row['answer'], str) and isinstance(row['flip_answer'], str) and
                   type(row['counterfactual_valid']) is bool and type(row['counterfactual_strict_valid']) is bool,
                   'Malformed original trajectory record')
        for layer in (21, 24, 27):
            xa, xp = arrays['Xa_' + str(layer)], arrays['Xp_question_' + str(layer)]
            _check(xa.ndim == xp.ndim == 2 and xa.shape[0] == count and xp.shape[0] == 300 and
                   xa.shape[1] == xp.shape[1] > 0, 'Actual feature rows differ')


def _fitted_heads(heads, report):
    import joblib
    from reproduction.a800.recovery_binding import finite_fitted_values
    for name, prefix, position, label in (('commit_probe.pkl', 'commit', 'Xa', 'flip'),
                                           ('boundary_probe.pkl', 'boundary', 'Xp', 'closedbook')):
        obj = joblib.load(heads / name)
        _check(list(obj['clf'].classes_) == [0, 1] and obj['position'] == position and obj['label'] == label and
               obj['auc'] == report[prefix + '_cv_auc'], 'Actual fitted head semantics differ')
        finite_fitted_values(vars(obj['clf']))


def _initial_artifacts(base, run, plan):
    from reproduction.a800.probe_admission import calibrated_version
    live_path = _owned_file(base / 'artifacts/live/manifest.json', base)
    accepted_path = _owned_file(base / 'logs/initial-probes-accepted.json', base)
    live, accepted = _read(live_path), _read(accepted_path)
    _check(calibrated_version(run, live) == run['initial_probe_version'] == accepted['version'] and
           accepted['accepted'] is True and accepted['run_id'] == run['run_id'] and
           accepted['live_manifest_sha256'] == digest(live_path) and
           accepted['grouped_calibration_cv'] == live['calibration_report'] and accepted['label_proof'] == live['label_proof'],
           'Historical initial probe admission differs')
    heads = _owned_file(Path(live['commit']), base / 'artifacts').parent
    _check(_owned_file(Path(live['boundary']), base / 'artifacts').parent == heads and heads.name == 'heads', 'Different head round directories')
    round_dir = heads.parent
    _check(round_dir.parent == base / 'artifacts' and round_dir.name.startswith('round_0-attempt-'), 'Not the owned original initial0 round')
    archive = _owned_file(round_dir / 'version-manifest.json', base)
    _check(_read(archive) == live, 'Archived initial version differs from admitted live')
    report, proof = live['calibration_report'], live['label_proof']
    _check(report['require_all_cv_folds'] is True and type(report['configured_cv_folds']) is int and report['configured_cv_folds'] == 2 and
           report['boundary_unique_question_weighting'] is True and proof['question_ids'] == run['calibration_question_ids'] and
           len(set(proof['question_ids'])) == 300 and proof['calibration_protocol']['rollout_max_new'] == 500 and
           proof['calibration_protocol']['correction_feedback'] is True and
           proof['evidence_deletion'] == 'typed_retrieval_ranges_only_preserve_model_and_correction', 'Not actual common8 strict grouped calibration')
    fingerprint = plan['initial_reward_fingerprint']
    _check(live['encoder_sha256'] == fingerprint['encoder_sha256'] and live['head_sha256'] == fingerprint['head_sha256'], 'Discovery frozen initial reward identity differs')
    files = {live_path: digest(live_path), accepted_path: digest(accepted_path), archive: digest(archive)}
    files.update(_files(heads, dict(live['head_sha256'], **{'calibration-report.json': live['calibration_report_sha256']})))
    dump = round_dir / 'dumps'
    _check('calibration-label-proof.json' in live['dump_sha256'] and
           all(name.endswith(('.npz', '.jsonl')) or name == 'calibration-label-proof.json' for name in live['dump_sha256']), 'Unexpected calibration dump filename')
    files.update(_files(dump, live['dump_sha256']))
    _check(_read(heads / 'calibration-report.json') == report and _read(dump / 'calibration-label-proof.json') == proof,
           'Actual calibration report/label bytes differ')
    _calibration_arrays(dump, report, proof); _fitted_heads(heads, report)
    return live, archive, heads, dump, files


def _write(path, raw):
    with Path(path).open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    fd = os.open(str(Path(path).parent), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def adopt(base, discovery_base):
    base, source = Path(base).resolve(), Path(discovery_base).resolve()
    _check(base != source, 'Cannot adopt into the original discovery run')
    run = _read(_owned_file(base / 'logs/run-manifest.json', base))
    for directory in (base, base / 'logs', base / 'artifacts', base / 'preparation'):
        _check(directory.is_dir() and not directory.is_symlink() and directory.stat().st_uid == os.getuid(),
               'Target namespace is not owned by this user')
    _check(Path(run['checkpoint_dir']).resolve().is_relative_to(base) and Path(run['train_log']).resolve().is_relative_to(base),
           'Target training paths leave its owned namespace')
    _check(run.get('policy_calibration_role', 'production') in ('production', CONFIRMATION_ROLE), 'Only confirmation/production adopts discovery heads')
    _check(run.get('policy_calibration_discovery_base') == str(source), 'Required frozen discovery path differs')
    _unstarted(base, run)
    _check(run['initial_probe_version'] is None and not (base / 'logs/initial-probes-accepted.json').exists() and
           not (base / 'artifacts/live/manifest.json').exists(), 'Target already fitted/admitted initial probes')
    source_run_path = _owned_file(source / 'logs/run-manifest.json', source)
    source_run = _read(source_run_path)
    _check(source_run.get('policy_calibration_role') == DISCOVERY_ROLE and source_run['run_id'] != run['run_id'], 'Not an independent real discovery source')
    plan, plan_sha = load_plan(source, source_run, expected_slurm_job_id=source_run['policy_calibration_slurm_job_id'])
    parent_id = run['run_id'] if run.get('policy_calibration_role', 'production') == 'production' else run['parent_production_run_id']
    _check(parent_id == plan['parent_production_run_id'] and canonical_bytes(_identity(run)) == canonical_bytes(plan['identity']) and
           run['calibration_question_ids'] == source_run['calibration_question_ids'], 'Source/model/data/parent mathematics differ')
    job_id = _actual_target(base, run)
    _check(job_id != plan['slurm_job_id'], 'Fresh target must have its independent native job')
    segment, bound_files = _segment(source, source_run, plan, plan_sha)
    live, archive, heads, dump, artifact_files = _initial_artifacts(source, source_run, plan)
    bound_files.update(artifact_files)
    for path in (source_run_path, source / 'logs/policy-safety-calibration-plan.json', source / 'preparation/parent-run-manifest.json'):
        bound_files[_owned_file(path, source)] = digest(path)
    manifest_bytes = (base / 'logs/run-manifest.json').read_bytes()
    _check(_strict_json(manifest_bytes) == run, 'Target changed during source authentication')
    _unstarted(base, run)
    token = uuid.uuid4().hex
    round_dir = base / 'artifacts' / ('round_0-adopted-' + token)
    claim = dict(schema_version=1, role='initial_probe_adoption_claim', run_id=run['run_id'], slurm_job_id=job_id,
                 discovery_run_id=source_run['run_id'], discovery_base=str(source), target_round_dir=str(round_dir), uid=os.getuid(),
                 source_plan_sha256=plan_sha, source_segment_sha256=digest(source / 'logs/policy-calibration-segment.json'))
    _write(base / 'logs/initial-probe-adoption-claim.json', canonical_bytes(claim))
    # Partial output belongs to this claim. Never delete or reset it on failure.
    round_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    for name in ('heads', 'dumps', 'provenance'):
        (round_dir / name).mkdir(mode=0o700)
    for src_dir, name in ((heads, 'heads'), (dump, 'dumps')):
        for item in sorted(src_dir.iterdir()):
            _write(round_dir / name / item.name, item.read_bytes())
            _check(digest(round_dir / name / item.name) == bound_files[item], 'Copied calibration bytes differ')
    _write(round_dir / 'provenance/discovery-version-manifest.json', archive.read_bytes())
    _check(all(digest(path) == sha for path, sha in bound_files.items()), 'Immutable discovery changed during copy; preserve partial claim')
    _check((base / 'logs/run-manifest.json').read_bytes() == manifest_bytes, 'Target manifest changed during copy')
    new = dict(live, run_id=run['run_id'], version=run['run_id'] + ':0:' + token,
               encoder=run['initial_model'], commit=str(round_dir / 'heads/commit_probe.pkl'),
               boundary=str(round_dir / 'heads/boundary_probe.pkl'), published_unix=time.time(),
               adoption_origin=dict(run_id=source_run['run_id'], slurm_job_id=plan['slurm_job_id'], version=live['version'],
                                    archive_sha256=digest(archive), accepted_manifest_sha256=digest(source / 'logs/initial-probes-accepted.json'),
                                    plan_sha256=plan_sha, segment_sha256=claim['source_segment_sha256']))
    # Prove the new publication bytes are exactly those authenticated above.
    _files(round_dir / 'heads', {p.name: bound_files[p] for p in heads.iterdir()})
    _files(round_dir / 'dumps', live['dump_sha256'])
    _write(round_dir / 'version-manifest.json', canonical_bytes(new))
    live_dir = base / 'artifacts/live'; live_dir.mkdir(exist_ok=True, mode=0o700)
    _check(not live_dir.is_symlink() and live_dir.stat().st_uid == os.getuid(), 'Live publication namespace differs')
    _unstarted(base, run)
    _check((base / 'logs/run-manifest.json').read_bytes() == manifest_bytes, 'Target changed before live publication')
    temporary = live_dir / ('manifest-adoption-' + token + '.tmp')
    _write(temporary, canonical_bytes(new))
    # Hard-link publishes complete fsynced bytes atomically without replacing
    # an independently produced live version. A crash leaves owned evidence.
    os.link(temporary, live_dir / 'manifest.json')
    temporary.unlink()
    fd = os.open(str(live_dir), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    receipt = dict(schema_version=1, role='initial_probe_adoption', run_id=run['run_id'], slurm_job_id=job_id,
                   discovery_base=str(source), parent_production_run_id=parent_id, adopted=True,
                   version=new['version'], initial_reward_fingerprint=plan['initial_reward_fingerprint'],
                   origin=new['adoption_origin'], source_file_sha256={str(p.relative_to(source)): sha for p, sha in bound_files.items()},
                   live_manifest_sha256=digest(live_dir / 'manifest.json'), checked_unix=time.time(),
                   production_admission=False, healthy_checkpoint_accepted=False, child_http_admission_required=True,
                   semantics='copied real initial0 head bytes; no inherited weights/Adam or copied admission; new native model/HTTP/coexistence gates required')
    _write(base / 'logs/initial-probe-adoption.json', canonical_bytes(receipt))
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--discovery-base', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(adopt(args.base, args.discovery_base), allow_nan=False))
