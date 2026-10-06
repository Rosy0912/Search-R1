"""Adopt real initial0 probes across the explicitly authorized goal change.

The original diagnostic is authenticated by its immutable source verifier in a
separate CPU process. Its original identity and mathematics are retained. This
producer copies only initial probe/calibration bytes; it grants no production
health, model/HTTP admission, or inherited training state.
"""
import argparse
import json
import os
import time
import uuid
from pathlib import Path

from reproduction.common8 import adopt_initial_probes as original
from reproduction.common8.freeze_calibration_plan import _unstarted
from reproduction.common8.policy_calibration_plan import (
    DISCOVERY_ROLE, MATHEMATICS, PlanError, canonical_bytes, document_sha256,
)
from reproduction.common8.verify_data import digest


# These scientific producers/readers must remain byte-identical. A changed
# scheduler or new admission implementation is not a changed probe producer.
PROBE_SOURCE_FILES = (
    'ts_probe_dump.py', 'ts_probe_train.py', 'probe_server.py',
    'search_g1/signals.py', 'search_g1/calibration_evidence.py',
    'search_g1/resource.py', 'search_r1/qa_protocol.py',
    'verl/utils/reward_score/qa_em.py', 'verl/trainer/main_ppo_eco.py',
    'reproduction/a800/probe_admission.py',
    'reproduction/a800/recovery_binding.py',
)
AUTHORIZED_BUDGET_FIELDS = frozenset((
    'target_actual_outer_updates', 'learning_rate_schedule',
    'lr_warmup_steps', 'entropy_coefficient', 'reference_kl_coefficient',
    'contract_kind',
))


def _check(condition, message):
    if not condition:
        raise PlanError(message)


def authenticate_original_diagnostic(base, source, proof, terminal):
    """Use the strict cross-contract bootstrap, never the new recipe on old data.

    The implementation owns the frozen-source subprocess/full tensor audit and
    actual native terminal/cleanup authentication. No receipt-only fallback is
    provided here. Returning JSON cannot turn a failed diagnostic into success.
    """
    from reproduction.common8.fresh200_bootstrap import authenticate_original_diagnostic as authenticate
    return authenticate(base, source, proof, terminal)


def _namespace(path):
    path = Path(path)
    _check(path.is_absolute() and path == path.resolve() and
           path.is_dir() and not path.is_symlink() and
           path.stat().st_uid == os.getuid(), 'Not a canonical owned namespace')
    return path


def _bound_source_files(source, hashes):
    _check(isinstance(hashes, dict) and bool(hashes), 'Missing authenticated original file graph')
    files = {}
    for name, sha in hashes.items():
        _check(isinstance(name, str), 'Original audit path is not a string')
        relative = Path(name)
        _check(not relative.is_absolute() and
               '..' not in relative.parts and relative.as_posix() == name and
               isinstance(sha, str) and len(sha) == 64 and
               all(c in '0123456789abcdef' for c in sha), 'Unbound original audit file')
        path = original._owned_file(source / relative, source)
        _check(digest(path) == sha, 'Authenticated original bytes changed: ' + name)
        files[path] = sha
    return files


def _compatible(base, source, run, old):
    from reproduction.common8.formal_contract import assert_formal_run
    assert_formal_run(run)
    _check(old.get('policy_calibration_role') == DISCOVERY_ROLE and
           old['run_id'] != run['run_id'] and old['profile'] == run['profile'] == 'common8',
           'Not an independent original diagnostic and new production run')
    for name in ('model_revision', 'initial_model_sha256', 'input_sha256', 'calibration_question_ids'):
        _check(canonical_bytes(old[name]) == canonical_bytes(run[name]),
               'Different initial probe scientific identity: ' + name)
    _check(old['initial_model'] == old['reference_model'] and
           run['initial_model'] == run['reference_model'], 'Original fixed reference differs')
    # Original MATHEMATICS supplies only the old, authenticated contract's
    # declared defaults. It is never replaced with FORMAL_MATHEMATICS or run's
    # values when an old receipt is checked.
    previous = dict(MATHEMATICS, **old['budget'])
    current = dict(run['budget'])
    for name in AUTHORIZED_BUDGET_FIELDS:
        previous.pop(name, None)
        current.pop(name, None)
    _check(canonical_bytes(previous) == canonical_bytes(current),
           'Unauthorized common scientific budget change')
    for name in ('actor_checkpoint', 'reference_checkpoint', 'resume_path',
                 'inherited_policy_run_id', 'inherited_optimizer_run_id',
                 'inherited_scheduler_run_id'):
        _check(run.get(name) is None, 'Fresh200 cannot inherit training state: ' + name)
    hashes = {}
    for name in PROBE_SOURCE_FILES:
        old_sha, new_sha = old['source_sha256'].get(name), run['source_sha256'].get(name)
        _check(isinstance(old_sha, str) and old_sha == new_sha and
               digest(original._owned_file(source / 'repo' / name, source)) == old_sha and
               digest(original._owned_file(base / 'repo' / name, base)) == new_sha,
               'Probe/reward implementation differs: ' + name)
        hashes[name] = new_sha
    return hashes


def adopt(base, discovery_base, diagnostic_proof, terminal_proof):
    base, source = _namespace(base), _namespace(discovery_base)
    _check(base != source, 'Cannot publish into the historical diagnostic')
    run_path = original._owned_file(base / 'logs/run-manifest.json', base)
    manifest_bytes = run_path.read_bytes()
    run = original._read(run_path)
    for name in ('logs', 'artifacts', 'preparation'):
        _namespace(base / name)
    for name in ('checkpoint_dir', 'train_log'):
        path = Path(run[name])
        _check(path.is_absolute() and path == path.resolve() and path.is_relative_to(base),
               'Target training path leaves owned namespace')
    _check(run.get('policy_calibration_discovery_base') == str(source) and
           Path(run['diagnostic_proof']) == Path(diagnostic_proof) and
           Path(run['diagnostic_terminal_proof']) == Path(terminal_proof),
           'Explicit original diagnostic inputs differ from target manifest')
    _unstarted(base, run)
    _check(run['initial_probe_version'] is None and
           not any((base / relative).exists() or (base / relative).is_symlink() for relative in (
               'logs/initial-probes-accepted.json', 'logs/initial-probe-adoption-claim.json',
               'logs/initial-probe-adoption.json', 'artifacts/live/manifest.json')),
           'Target already fitted, claimed or admitted initial probes; preserve partial output')

    authenticated = authenticate_original_diagnostic(base, source, diagnostic_proof, terminal_proof)
    _check(isinstance(authenticated, dict) and {
        'run', 'plan', 'segment', 'source_file_sha256',
        'diagnostic_proof_sha256', 'terminal_proof_sha256',
    } <= set(authenticated), 'Original diagnostic authenticator returned no typed result')
    old, plan, segment = (authenticated[name] for name in ('run', 'plan', 'segment'))
    _check(old == original._read(original._owned_file(source / 'logs/run-manifest.json', source)) and
           plan == original._read(original._owned_file(source / 'logs/policy-safety-calibration-plan.json', source)) and
           segment == original._read(original._owned_file(source / 'logs/policy-calibration-segment.json', source)),
           'Original authenticated objects differ from original bytes')
    for name, path in (('diagnostic_proof_sha256', diagnostic_proof),
                       ('terminal_proof_sha256', terminal_proof)):
        _check(authenticated[name] == digest(Path(path)), 'Original authentication receipt hash differs: ' + name)
    bound_files = _bound_source_files(source, authenticated['source_file_sha256'])
    for name in ('logs/run-manifest.json', 'logs/policy-safety-calibration-plan.json',
                 'logs/policy-calibration-segment.json', 'preparation/parent-run-manifest.json'):
        _check(source / name in bound_files, 'Original identity audit bytes are not bound: ' + name)
    probe_hashes = _compatible(base, source, run, old)
    job_id = original._actual_target(base, run)
    _check(job_id != plan['slurm_job_id'], 'Fresh target must have a separate actual native job')
    live, archive, heads, dump, artifact_files = original._initial_artifacts(source, old, plan)
    _check(type(live['round']) is int and live['round'] == 0 and
           live['version'].startswith(old['run_id'] + ':0:'), 'Only the real initial0 probes may be adopted')
    _check(all(bound_files.get(path) == sha for path, sha in artifact_files.items()),
           'Actual initial0 artifacts are absent from original authentication graph')
    _check(run_path.read_bytes() == manifest_bytes, 'Target changed during original authentication')
    _unstarted(base, run)

    token = uuid.uuid4().hex
    round_dir = base / 'artifacts' / ('round_0-adopted-' + token)
    origin = dict(run_id=old['run_id'], parent_production_run_id=plan['parent_production_run_id'],
                  slurm_job_id=plan['slurm_job_id'], version=live['version'],
                  source_commit=old['source_commit'], source_sha256=old['source_sha256'],
                  original_identity=plan['identity'], original_mathematics=plan['mathematics'],
                  diagnostic_proof_sha256=authenticated['diagnostic_proof_sha256'],
                  terminal_proof_sha256=authenticated['terminal_proof_sha256'],
                  plan_sha256=digest(source / 'logs/policy-safety-calibration-plan.json'),
                  segment_sha256=digest(source / 'logs/policy-calibration-segment.json'),
                  archive_sha256=digest(archive))
    claim = dict(schema_version=3, role='fresh200_initial_probe_adoption_claim', run_id=run['run_id'],
                 slurm_job_id=job_id, discovery_base=str(source), target_round_dir=str(round_dir),
                 uid=os.getuid(), origin=origin)
    original._write(base / 'logs/initial-probe-adoption-claim.json', canonical_bytes(claim))
    # An unsuccessful copy remains exclusively claimed. Never remove/retry it
    # or overwrite a live version under the same training identity.
    round_dir.mkdir(exist_ok=False, mode=0o700)
    for name in ('heads', 'dumps', 'provenance'):
        (round_dir / name).mkdir(mode=0o700)
    for directory, name in ((heads, 'heads'), (dump, 'dumps')):
        for item in sorted(directory.iterdir()):
            destination = round_dir / name / item.name
            original._write(destination, item.read_bytes())
            _check(digest(destination) == artifact_files[item], 'Copied actual initial0 bytes differ')
    for name, path in (('original-version-manifest.json', archive),
                       ('original-diagnostic-proof.json', Path(diagnostic_proof)),
                       ('original-native-terminal-proof.json', Path(terminal_proof))):
        original._write(round_dir / 'provenance' / name, path.read_bytes())
    _check(all(digest(path) == sha for path, sha in bound_files.items()),
           'Original evidence changed during copy; preserve partial claim')
    _check(digest(diagnostic_proof) == origin['diagnostic_proof_sha256'] and
           digest(terminal_proof) == origin['terminal_proof_sha256'], 'Original proof changed during copy')
    _compatible(base, source, run, old)
    _check(run_path.read_bytes() == manifest_bytes, 'Target manifest changed during copy')
    new = dict(live, run_id=run['run_id'], version=run['run_id'] + ':0:' + token,
               encoder=run['initial_model'], commit=str(round_dir / 'heads/commit_probe.pkl'),
               boundary=str(round_dir / 'heads/boundary_probe.pkl'), published_unix=time.time(),
               adoption_origin=origin)
    original._files(round_dir / 'heads', {path.name: sha for path, sha in artifact_files.items() if path.parent == heads})
    original._files(round_dir / 'dumps', live['dump_sha256'])
    original._write(round_dir / 'version-manifest.json', canonical_bytes(new))
    live_dir = base / 'artifacts/live'
    live_dir.mkdir(exist_ok=True, mode=0o700)
    _namespace(live_dir)
    _unstarted(base, run)
    _check(run_path.read_bytes() == manifest_bytes, 'Target changed before publication')
    temporary = live_dir / ('manifest-adoption-' + token + '.tmp')
    original._write(temporary, canonical_bytes(new))
    # Publish complete durable bytes atomically, without replacement.
    os.link(temporary, live_dir / 'manifest.json')
    temporary.unlink()
    fd = os.open(str(live_dir), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    receipt = dict(schema_version=3, role='fresh200_initial_probe_adoption', run_id=run['run_id'],
                   contract_kind=run['contract_kind'], slurm_job_id=job_id, discovery_base=str(source),
                   adopted=True, version=new['version'], origin=origin,
                   initial_reward_fingerprint=plan['initial_reward_fingerprint'],
                   probe_source_sha256=probe_hashes,
                   source_file_sha256={str(path.relative_to(source)): sha for path, sha in bound_files.items()},
                   target_source_sha256=run['source_sha256'], target_budget_sha256=document_sha256(run['budget']),
                   initial_archive=str((round_dir / 'version-manifest.json').relative_to(base)),
                   initial_archive_sha256=digest(round_dir / 'version-manifest.json'),
                   live_manifest_sha256=digest(live_dir / 'manifest.json'), checked_unix=time.time(),
                   production_admission=False, healthy_checkpoint_accepted=False,
                   child_http_admission_required=True, new_native_model_admission_required=True,
                   new_all2560_behavior_admission_required=True,
                   semantics='Only actual initial0 head/dump bytes adopted; original diagnostic identity retained; '
                             'fresh production must run its own native/model/HTTP/encoder/coexistence and sampling gates')
    original._write(base / 'logs/initial-probe-adoption.json', canonical_bytes(receipt))
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--discovery-base', type=Path, required=True)
    parser.add_argument('--diagnostic-proof', type=Path, required=True)
    parser.add_argument('--terminal-proof', type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(adopt(arguments.base, arguments.discovery_base,
                           arguments.diagnostic_proof, arguments.terminal_proof), allow_nan=False))
