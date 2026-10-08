"""Guarded fresh200 short training, with explicitly different diagnostic math.

The original diagnostic is authenticated by its immutable original verifier.
It supplies candidate brake measurements, never same-recipe health admission.
The new production must pass its own all-token gate and same-process five gate.
"""
import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path

from reproduction.common8.policy_calibration_plan import canonical_bytes, document_sha256, _strict_json
from reproduction.common8.verify_data import digest

ROLE = 'guarded_fresh200_short_training_bootstrap'
ORIGINAL_DIAGNOSTIC_ROLE = 'original_f044_diagnostic'
WARMUP_MEASUREMENT_ROLE = 'warmup57_measurement'
WARMUP_AUTHENTICATION_ROLE = 'warmup57_measurement_authentication'
WARMUP_FINISHED_STAGE = 'FINISHED_WARMUP57_MEASUREMENTS_AND_OWNED_CLEANUP'
ORIGINAL_COMMIT = 'f0445f4edf0dbea916ff63bedafe356d6d9ebcdb'
ORIGINAL_LAUNCH_SHA = '4d28b864d5fc9eeafdff3a3a1cf14b55d8add6be4f9c5a657d6fa51bc7c2ae49'
ORIGINAL_SOURCE_MAP_SHA = '0caacf25e7eae533760b52139ee86b009a11b0c3ce1fa191834f362b94c9da3e'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    path = Path(path)
    need(path.is_file() and not path.is_symlink(), 'Not an original regular evidence file')
    return _strict_json(path.read_bytes())


def write_once(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open('xb') as stream:
        os.chmod(path, 0o600)
        stream.write(canonical_bytes(obj)); stream.flush(); os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _bound_path(base, value):
    path = Path(value)
    need(path.is_absolute() and not path.is_symlink() and path.resolve().is_relative_to(base),
         'Evidence must be bound to the new owned namespace')
    return path


def authenticate_original_diagnostic(base, source, proof, terminal):
    """Full old-contract CPU audit at every adoption/bootstrap producer.

    A saved result is an immutable archive, never a substitute for the audit.
    Per-outer verification later checks its frozen digest, without model reads.
    """
    base, source = Path(base).resolve(), Path(source).resolve()
    run = read(base / 'logs/run-manifest.json')
    from reproduction.common8.formal_contract import assert_formal_run
    assert_formal_run(run)
    proof, terminal = _bound_path(base, proof), _bound_path(base, terminal)
    need(str(proof) == run['diagnostic_proof'] and str(terminal) == run['diagnostic_terminal_proof'],
         'Original diagnostic descriptor differs')
    need(digest(proof) == run['diagnostic_proof_sha256'] == ORIGINAL_LAUNCH_SHA and
         digest(terminal) == run['diagnostic_terminal_proof_sha256'], 'Original captured receipt bytes differ')
    launch, observed = read(proof), read(terminal)
    need(source != base and str(source) == run['policy_calibration_discovery_base'] == launch['base'],
         'Wrong original diagnostic namespace')
    need(launch['source_commit'] == ORIGINAL_COMMIT and document_sha256(launch['source_sha256']) == ORIGINAL_SOURCE_MAP_SHA,
         'Original diagnostic verifier code is not the independently reviewed source')
    need(launch['job_id'] == observed['job_id'] == '3298' and observed['base'] == str(source) and
         observed['job_returncode'] == 0 and observed['source_hashes_match'] is True,
         'Wrong original native observation')
    fields = dict(re.findall(r'(?:^|\s)(\w+)=(\S+)', observed['job_stdout']))
    need(fields.get('JobId') == '3298' and fields.get('JobState') == 'COMPLETED' and
         fields.get('ExitCode') == '0:0' and fields.get('Restarts') == '0' and
         fields.get('NodeList') == 'h100-8', 'A real captured completed native terminal is required')
    original = read(source / 'logs/run-manifest.json')
    need(original['source_sha256'] == launch['source_sha256'] and original['source_commit'] == ORIGINAL_COMMIT and
         original['run_id'] == observed['run-manifest.json']['run_id'] != run['run_id'],
         'Original source/run identity differs')
    for name, sha in original['source_sha256'].items():
        need(digest(source / 'repo' / name) == sha, 'Original diagnostic source changed: ' + name)
    cache = base / 'logs/diagnostic-authentication.json'
    cached = None
    if cache.exists():
        cached = read(cache)
        need(cached['role'] == 'original_f044_diagnostic_authentication' and cached['new_run_id'] == run['run_id'] and
             cached['diagnostic_proof_sha256'] == digest(proof) and cached['terminal_proof_sha256'] == digest(terminal),
             'Cached original authentication identity differs')
        for name, sha in cached['source_file_sha256'].items():
            need(digest(source / name) == sha, 'Original cached diagnostic receipt changed')
    # Imports resolve only to the old source. No new200 MATHEMATICS is supplied.
    script = '''import json,sys,hashlib
from pathlib import Path
from reproduction.common8.production_safety_admission import verify_child
from reproduction.common8.policy_calibration_plan import DISCOVERY_ROLE
from reproduction.common8.adopt_initial_probes import _initial_artifacts
from reproduction.common8.verify_data import digest
base=Path(sys.argv[1]); parent=json.loads((base/'preparation/parent-run-manifest.json').read_text())
receipts={};run,plan,segment=verify_child(base,parent,DISCOVERY_ROLE,receipts,'')
live,archive,heads,dump,files=_initial_artifacts(base,run,plan)
hashes={}
for name,raw in receipts.items():
 if name=='checkpoint-complete.json':
  name=str((Path(run['checkpoint_dir'])/'actor/global_step_1/complete.json').relative_to(base))
 hashes[name]=hashlib.sha256(raw).hexdigest()
hashes.update({str(path.relative_to(base)):sha for path,sha in files.items()})
for name in ('logs/run-manifest.json','logs/policy-safety-calibration-plan.json','logs/policy-calibration-segment.json'):
 hashes[name]=digest(base/name)
print(json.dumps(dict(run=run,plan=plan,segment=segment,source_file_sha256=hashes),allow_nan=False))
'''
    env = dict(os.environ, SEARCH_G1_BASE=str(source), SEARCH_G1_PROFILE='common8',
               PYTHONPATH=str(source / 'repo'), CUDA_VISIBLE_DEVICES='', PYTHONNOUSERSITE='1')
    env.pop('PYTHONHOME', None)
    process = subprocess.run([str(source / 'env/bin/python'), '-c', script, str(source)],
                             cwd=source / 'repo', env=env, capture_output=True, text=True, check=True)
    result = _strict_json(process.stdout.strip().splitlines()[-1].encode())
    need(result['run'] == original and result['segment']['completed_outer_updates'] == 1 and
         result['segment']['production_admission'] is False, 'Original full execution audit differs')
    result.update(role='original_f044_diagnostic_authentication', new_run_id=run['run_id'],
                  diagnostic_proof_sha256=digest(proof), terminal_proof_sha256=digest(terminal),
                  checked_unix=time.time(), production_admission=False, same_recipe_health_accepted=False)
    for name, sha in result['source_file_sha256'].items():
        need(digest(source / name) == sha, 'Original bytes changed during authentication')
    need(bool(result['source_file_sha256']) and
         {'logs/run-manifest.json','logs/policy-safety-calibration-plan.json',
          'logs/policy-calibration-segment.json'}<=set(result['source_file_sha256']),
         'Full original verifier did not return its required receipt graph')
    if cached is not None:
        # Every scientific value and the complete required graph must equal
        # the fresh original-source reconstruction, including actual monitors,
        # paired evaluations, tensor audit and per-rank writes. Never overwrite
        # a differing cache or use a supplied summary to choose thresholds.
        for key in ('run','plan','segment','source_file_sha256','production_admission','same_recipe_health_accepted'):
            need(cached[key]==result[key], 'Cached authentication differs from complete original audit: '+key)
        return cached
    write_once(cache, result)
    return result


def _owned_regular(root, path):
    """Historical files remain owned/canonical even after all workers exit."""
    root, path = Path(root).resolve(), Path(path)
    need(path.is_absolute() and path.resolve() == path and path.is_relative_to(root) and
         path.is_file() and not path.is_symlink() and path.stat().st_uid == os.getuid(),
         'Not a canonical owned historical receipt')
    cursor = root
    for part in path.relative_to(root).parts:
        cursor /= part
        need(not cursor.is_symlink(), 'Historical receipt contains a symlink')
    return path


def _warmup_cleanup(source, run, plan):
    """Reconstruct completed owned cleanup; never require an old PID live."""
    from reproduction.a800.owned_processes import identity, same
    status, cleanup, ledger = (read(_owned_regular(source, source / 'logs' / name))
                              for name in ('pipeline-status.json', 'cleanup-status.json', 'process-ledger.json'))
    need(status['run_id'] == cleanup['run_id'] == ledger['run_id'] == run['run_id'] and
         status['stage'] == WARMUP_FINISHED_STAGE and cleanup['cleaned'] is True,
         'Warmup measurement did not finish its own cleanup')
    need(isinstance(ledger['jobs'], dict) and {'trainer', 'retriever', 'probe-server'} <= set(ledger['jobs']),
         'Missing original owned component ledger')
    for job in ledger['jobs'].values():
        need(job['state'] == 'STOPPED' and isinstance(job['ledger'], dict), 'Owned measurement component not stopped')
        for process in job['ledger'].values():
            need(process['uid'] == os.getuid() and type(process['pid']) is int and process['pid'] > 0 and
                 type(process['start_ticks']) is int and process['start_ticks'] > 0,
                 'Original component birth/owner is not bound')
            # Absence/recycled PIDs are normal for historical evidence. A
            # matching, still-live owned process is an actual cleanup failure.
            need(not same(process) or identity(process['pid'])['state'] == 'Z',
                 'Owned historical measurement process remains live')
    for name in ('retrieval-staging-claim.json', 'corpus-cache-claim.json'):
        claim = read(_owned_regular(source, source / 'logs' / name))
        root = Path('/dev/shm') / ('search-g1-common8-' + run['run_id'] + '-' + plan['slurm_job_id'])
        if name == 'corpus-cache-claim.json':
            root /= 'datasets-cache'
        need(claim['run_id'] == run['run_id'] and claim['slurm_job_id'] == plan['slurm_job_id'] and
             claim['phase'] == 'CLEANED' and claim['uid'] == os.getuid() and Path(claim['root']) == root,
             'Historical measurement temporary-resource cleanup differs')
        need(not root.exists() and not root.is_symlink(), 'Owned measurement temporary root remains')


def authenticate_warmup57_measurement(base, source, proof, terminal):
    """Authenticate a successful new typed measurement in its frozen source.

    This has no fallback to the failed constant-LR3298 attempt. The source's
    full reader reopens actual tensors, Adam/rank/scheduler, all2560 gates for
    both outers, paired512 and fitted probe arrays in a separate CPU process.
    """
    from reproduction.common8.formal_contract import assert_formal_run, assert_scientific_run
    from reproduction.common8.source_files import runtime_files
    base, source = Path(base).resolve(), Path(source).resolve()
    need(base != source and not base.is_relative_to(source) and not source.is_relative_to(base),
         'Measurement and production need independent namespaces')
    run = read(_owned_regular(base, base / 'logs/run-manifest.json'))
    assert_formal_run(run)
    need(run.get('diagnostic_execution_role') == WARMUP_MEASUREMENT_ROLE,
         'New measurement authentication needs its explicit typed descriptor')
    proof, terminal = _owned_regular(base, proof), _owned_regular(base, terminal)
    need(str(proof) == run['diagnostic_proof'] and str(terminal) == run['diagnostic_terminal_proof'] and
         digest(proof) == run['diagnostic_proof_sha256'] and
         digest(terminal) == run['diagnostic_terminal_proof_sha256'], 'Captured new measurement descriptors differ')
    launch, observed = read(proof), read(terminal)
    need(launch['execution_role'] == WARMUP_MEASUREMENT_ROLE and
         str(source) == run['policy_calibration_discovery_base'] == launch['base'] == observed['base'],
         'Not the independent typed warmup57 measurement source')
    job = launch['job_id']
    need(isinstance(job, str) and re.fullmatch('[1-9][0-9]*', job) is not None and
         observed['job_id'] == job and observed['job_returncode'] == 0 and
         observed['source_hashes_match'] is True, 'Original new native observation differs')
    fields = dict(re.findall(r'(?:^|\s)(\w+)=(\S+)', observed['job_stdout']))
    need(fields.get('JobId') == job and fields.get('JobState') == 'COMPLETED' and
         fields.get('ExitCode') == '0:0' and fields.get('Restarts') == '0' and fields.get('NodeList') == 'h100-8',
         'Actual captured completed native measurement terminal is required')
    original = read(_owned_regular(source, source / 'logs/run-manifest.json'))
    assert_scientific_run(original, expected_role=WARMUP_MEASUREMENT_ROLE)
    need(original['run_id'] != run['run_id'] and original['source_commit'] == launch['source_commit'] and
         original['source_sha256'] == launch['source_sha256'] and
         original == observed['run-manifest.json'], 'Measurement source/run was relabeled or changed')
    for name in ('budget', 'initial_model_sha256', 'input_sha256', 'calibration_question_ids', 'model_revision'):
        need(canonical_bytes(original[name]) == canonical_bytes(run[name]), 'Different new measurement science: ' + name)
    actual_source = {str(path.relative_to(source / 'repo')): digest(_owned_regular(source, path))
                     for path in runtime_files(source / 'repo')}
    need(actual_source == original['source_sha256'], 'Complete frozen measurement runtime source differs')
    # The original typed source is the only imported verifier. No current
    # production configuration or worker liveness is substituted into it.
    script = '''import json,sys,hashlib,os
from pathlib import Path
from reproduction.common8.warmup_measurement_plan import load_plan,ROLE
from reproduction.common8.warmup_measurement_runtime import verify_measurement
from reproduction.common8.adopt_initial_probes import _initial_artifacts,_owned_file
from reproduction.common8.verify_data import digest
base=Path(sys.argv[1]);run=json.loads((base/'logs/run-manifest.json').read_text())
plan,sha=load_plan(base,run,expected_slurm_job_id=sys.argv[2])
segment=verify_measurement(base,run=run,candidate=None,require_live_workers=False)
live,archive,heads,dump,artifacts=_initial_artifacts(base,run,plan)
hashes={}
for folder in ('logs','preparation'):
 for path in sorted((base/folder).rglob('*')):
  if path.is_file() and path.suffix in ('.json','.jsonl'):
   path=_owned_file(path,base);hashes[str(path.relative_to(base))]=digest(path)
checkpoint=Path(run['checkpoint_dir'])/'actor/global_step_2'
complete=json.loads((checkpoint/'complete.json').read_text())
for name in ('complete.json',*complete['files_sha256']):
 path=_owned_file(checkpoint/name,base);hashes[str(path.relative_to(base))]=digest(path)
hashes.update({str(path.relative_to(base)):sha for path,sha in artifacts.items()})
print(json.dumps(dict(run=run,plan=plan,segment=segment,source_file_sha256=hashes),allow_nan=False))
'''
    env = dict(os.environ, SEARCH_G1_BASE=str(source), SEARCH_G1_PROFILE='common8',
               PYTHONPATH=str(source / 'repo'), CUDA_VISIBLE_DEVICES='', PYTHONNOUSERSITE='1')
    env.pop('PYTHONHOME', None)
    process = subprocess.run([str(source / 'env/bin/python'), '-c', script, str(source), job],
                             cwd=source / 'repo', env=env, capture_output=True, text=True, check=True)
    result = _strict_json(process.stdout.strip().splitlines()[-1].encode())
    need(result['run'] == original and result['plan']['role'] == WARMUP_MEASUREMENT_ROLE and
         result['plan']['slurm_job_id'] == job and result['segment']['run_id'] == original['run_id'] and
         type(result['segment']['completed_outer_updates']) is int and result['segment']['completed_outer_updates'] == 2 and
         type(result['segment']['optimizer_writes_per_rank']) is int and result['segment']['optimizer_writes_per_rank'] == 40 and
         result['segment']['production_admission'] is False and
         result['segment']['healthy_checkpoint_accepted'] is False, 'Incomplete actual typed measurement audit')
    _warmup_cleanup(source, original, result['plan'])
    need(isinstance(result['source_file_sha256'], dict) and {
        'logs/run-manifest.json', 'logs/warmup57-measurement-plan.json',
        'logs/warmup57-measurement-segment.json', 'preparation/measurement-parent-run-manifest.json',
        'preparation/warmup57-measurement-method.json', 'logs/cleanup-status.json',
        'logs/pipeline-status.json', 'logs/process-ledger.json',
    } <= set(result['source_file_sha256']), 'Missing complete new measurement receipt graph')
    for name, sha in result['source_file_sha256'].items():
        relative = Path(name)
        need(isinstance(name, str) and not relative.is_absolute() and '..' not in relative.parts and
             relative.as_posix() == name and digest(_owned_regular(source, source / relative)) == sha,
             'New measurement bytes changed during original-source audit')
    result.update(role=WARMUP_AUTHENTICATION_ROLE, new_run_id=run['run_id'],
                  diagnostic_proof_sha256=digest(proof), terminal_proof_sha256=digest(terminal),
                  checked_unix=time.time(), production_admission=False, same_recipe_health_accepted=False,
                  measurement_recipe_health_accepted=True, measurements_counted_as_production=0)
    # Selection is frozen max80/zero-floor provenance, never an adjustment
    # made to accommodate a failed KL candidate or an unhealthy paired score.
    candidate_thresholds(result)
    cache = base / 'logs/diagnostic-authentication.json'
    if cache.exists():
        cached = read(_owned_regular(base, cache))
        for key in ('role', 'new_run_id', 'run', 'plan', 'segment', 'source_file_sha256',
                    'diagnostic_proof_sha256', 'terminal_proof_sha256', 'production_admission',
                    'same_recipe_health_accepted', 'measurement_recipe_health_accepted',
                    'measurements_counted_as_production'):
            need(cached[key] == result[key], 'Cached measurement differs from complete original audit: ' + key)
        return cached
    write_once(cache, result)
    return result


def authenticate_diagnostic(base, source, proof, terminal):
    """Explicit identity dispatch. Failed/missing evidence never selects another role."""
    run = read(Path(base) / 'logs/run-manifest.json')
    role = run.get('diagnostic_execution_role')
    if role == WARMUP_MEASUREMENT_ROLE:
        return authenticate_warmup57_measurement(base, source, proof, terminal)
    if role == ORIGINAL_DIAGNOSTIC_ROLE:
        return authenticate_original_diagnostic(base, source, proof, terminal)
    raise ValueError('An explicit recognized diagnostic_execution_role is required')


def candidate_thresholds(diagnostic):
    """Use the prospectively scoped actual maximum; never widen a failed brake."""
    segment, plan = diagnostic['segment'], diagnostic['plan']
    rows = segment['structural']['global_monitors']
    role = diagnostic.get('role')
    if role == WARMUP_AUTHENTICATION_ROLE:
        from reproduction.common8.warmup_measurement_plan import verify_method, TAIL_METHOD, PHASE_BUDGET, HEALTH
        need(plan['role'] == WARMUP_MEASUREMENT_ROLE and plan['phase_budget'] == PHASE_BUDGET and
             segment['completed_outer_updates'] == 2 and segment['optimizer_writes_per_rank'] == 40,
             'Not the complete two-outer typed measurement')
        method = verify_method(plan['method'], diagnostic['run'])
        need(method['negative_tail_method'] == TAIL_METHOD and len(rows) == 80,
             'Incomplete or prospectively different max80 monitor selection')
        health_rule = method['health_rule']
        need(canonical_bytes(health_rule) == canonical_bytes(HEALTH),
             'New measurement prospective health rule differs')
    else:
        need(role == 'original_f044_diagnostic_authentication', 'Unknown diagnostic measurement role')
        need(len(rows) == 40, 'Incomplete old global monitor sequence')
        health_rule = dict(method='paired_fixed512_absolute_allowed_drop',
                           allowed_drop=dict(weighted_em=0., weighted_f1=0., valid_action_rate=0.))
    candidate = plan['candidate_thresholds']
    need(candidate['dual_clip'] == 3 and candidate['epsilon'] == .2 and
         candidate['target_old_policy_kl'] == .01 and candidate['kl_log_bound'] == 30,
         'Original frozen candidate differs')
    tails = []
    for row in rows:
        need(row['allowed'] is True and row['kl_estimate_complete'] is True and
             type(row['sampled_old_policy_kl']) in (float, int) and
             math.isfinite(row['sampled_old_policy_kl']) and 0 <= row['sampled_old_policy_kl'] <= .01,
             'Original diagnostic violated the KL candidate')
        tail = row['negative_tail_weight_fraction']
        need(type(tail) in (int, float) and math.isfinite(tail) and 0 <= tail <= 1, 'Invalid actual tail')
        tails.append(tail)
    # The new branch reads its own preregistered method; the historical old
    # branch retains the original explicit zero-drop rule. Neither is widened.
    before, after = segment['evaluation_before'], segment['evaluation_after']
    need(before['question_ids'] == after['question_ids'] and before['question_count'] == after['question_count'] == 512,
         'Not actual paired same512')
    for key, limit in health_rule['allowed_drop'].items():
        need(type(before[key]) in (int, float) and type(after[key]) in (int, float) and
             math.isfinite(before[key]) and math.isfinite(after[key]) and before[key] - after[key] <= limit,
             'Original prospective zero-drop criterion failed: ' + key)
    return dict(dual_clip=3., target_old_policy_kl=.01, kl_log_bound=30., negative_tail_limit=max(tails))


def initial_probe_files(base, run, diagnostic):
    """Authenticate immutable adopted initial0 bytes, not the mutable live pointer."""
    base = Path(base).resolve()
    adoption_path = base / 'logs/initial-probe-adoption.json'
    adoption = read(adoption_path)
    need(adoption['schema_version'] == 3 and adoption['role'] == 'fresh200_initial_probe_adoption' and
         adoption['run_id'] == run['run_id'] and adoption['adopted'] is True and
         adoption['version'] == run['initial_probe_version'] and
         adoption['target_source_sha256'] == run['source_sha256'] and
         adoption['target_budget_sha256'] == document_sha256(run['budget']) and
         adoption['origin']['run_id'] == diagnostic['run']['run_id'],
         'Actual new initial-probe adoption lineage differs')
    relative = Path(adoption['initial_archive'])
    need(not relative.is_absolute() and '..' not in relative.parts and
         relative.as_posix() == adoption['initial_archive'], 'Unbound initial archive')
    archive_path = _bound_path(base, base / relative)
    round_dir = archive_path.parent
    need(round_dir.parent == base / 'artifacts' and round_dir.name.startswith('round_0-adopted-') and
         archive_path.name == 'version-manifest.json' and
         digest(archive_path) == adoption['initial_archive_sha256'] == adoption['live_manifest_sha256'],
         'Immutable adopted initial0 manifest changed')
    initial = read(archive_path)
    fingerprint = diagnostic['plan']['initial_reward_fingerprint']
    need(initial['run_id'] == run['run_id'] and type(initial['round']) is int and initial['round'] == 0 and
         initial['version'] == run['initial_probe_version'] and
         initial['version'].startswith(run['run_id'] + ':0:') and
         initial['encoder'] == run['initial_model'] and
         initial['encoder_sha256'] == fingerprint['encoder_sha256'] and
         initial['head_sha256'] == fingerprint['head_sha256'] and
         adoption['initial_reward_fingerprint'] == fingerprint and
         initial['adoption_origin'] == adoption['origin'], 'Archived initial0 scientific identity differs')
    need(initial['commit'] == str(round_dir / 'heads/commit_probe.pkl') and
         initial['boundary'] == str(round_dir / 'heads/boundary_probe.pkl') and
         set(initial['head_sha256']) == {'commit_probe.pkl', 'boundary_probe.pkl'},
         'Archived initial0 heads leave their adopted round')
    files = {adoption_path: digest(adoption_path), archive_path: digest(archive_path)}
    for directory, hashes in (
            ('heads', dict(initial['head_sha256'], **{'calibration-report.json': initial['calibration_report_sha256']})),
            ('dumps', initial['dump_sha256'])):
        need(bool(hashes), 'Missing actual initial0 artifact bytes')
        for name, sha in hashes.items():
            need(Path(name).name == name and name not in ('.', '..'), 'Unbound initial0 artifact filename')
            path = _bound_path(base, round_dir / directory / name)
            need(digest(path) == sha, 'Copied initial0 artifact changed')
            files[path] = sha
    for name, sha in (
            ('original-version-manifest.json', adoption['origin']['archive_sha256']),
            ('original-diagnostic-proof.json', adoption['origin']['diagnostic_proof_sha256']),
            ('original-native-terminal-proof.json', adoption['origin']['terminal_proof_sha256'])):
        path = _bound_path(base, round_dir / 'provenance' / name)
        need(digest(path) == sha, 'Original initial0 adoption provenance changed')
        files[path] = sha
    return {path.relative_to(base).as_posix(): sha for path, sha in files.items()}


def verify_bootstrap(base, artifact, run):
    """Read-only on every rank/outer; no new phase gets historical health flags."""
    from reproduction.common8.formal_contract import assert_formal_run
    from reproduction.common8.production_safety_admission import _verify_parent_probability
    assert_formal_run(run)
    need(type(artifact['schema_version']) is int and artifact['schema_version'] == 3 and
         artifact['role'] == ROLE and artifact['accepted'] is True and artifact['run_id'] == run['run_id'] and
         artifact['healthy_checkpoint_accepted'] is False and artifact['same_recipe_health_accepted'] is False and
         artifact['actual_production_short_train_accepted'] is False and artifact['old_updates_counted_as_production'] == 0,
         'Not explicit guarded fresh200 short-training bootstrap')
    for key in ('source_sha256', 'initial_model_sha256', 'input_sha256', 'budget'):
        need(artifact[key] == run[key], 'Fresh200 scientific identity differs: ' + key)
    receipts = artifact['receipts_sha256']
    need(bool(receipts), 'No bound fresh native evidence')
    for name, sha in receipts.items():
        path = _bound_path(Path(base).resolve(), Path(base) / name)
        need(digest(path) == sha, 'Frozen bootstrap receipt changed')
    diagnostic = read(Path(base) / artifact['diagnostic_authentication'])
    need(diagnostic['new_run_id'] == run['run_id'] and diagnostic['run']['run_id'] == artifact['diagnostic_run_id'],
         'Original diagnostic was relabeled')
    need(artifact['diagnostic_execution_role'] == run['diagnostic_execution_role'] and
         diagnostic['role'] == (WARMUP_AUTHENTICATION_ROLE if run['diagnostic_execution_role'] == WARMUP_MEASUREMENT_ROLE
                                else 'original_f044_diagnostic_authentication') and
         run['diagnostic_execution_role'] in (WARMUP_MEASUREMENT_ROLE, ORIGINAL_DIAGNOSTIC_ROLE),
         'Frozen bootstrap diagnostic execution role differs')
    thresholds = candidate_thresholds(diagnostic)
    need(all(artifact[key] == value for key, value in thresholds.items()), 'Brake candidate provenance differs')
    probability = artifact['probability']
    need(probability['run_id'] == run['run_id'] and probability['actual_optimizer_writes'] == 0 and
         probability['source_sha256'] == run['source_sha256'], 'Not new production native fresh0 probability')
    _verify_parent_probability(Path(base), run, probability, receipts)
    initial_files = initial_probe_files(base, run, diagnostic)
    need(all(receipts.get(name) == sha for name, sha in initial_files.items()) and
         'artifacts/live/manifest.json' not in receipts and
         artifact['initial_probe_version'] == run['initial_probe_version'],
         'Frozen actual initial0 lineage is incomplete or bound to mutable live probes')
    return thresholds['negative_tail_limit'], document_sha256(artifact)


def produce(base, source, proof, terminal):
    from reproduction.common8.freeze_calibration_plan import _unstarted
    from reproduction.common8.configuration import resolved
    from reproduction.common8.formal_contract import verify_formal_config
    base = Path(base).resolve()
    run, config, _ = resolved(base)
    from omegaconf import OmegaConf
    document = OmegaConf.to_container(config, resolve=True)
    verify_formal_config(document, run, base)
    _unstarted(base, run)
    diagnostic = authenticate_diagnostic(base, source, proof, terminal)
    thresholds = candidate_thresholds(diagnostic)
    root = base / 'logs/policy-safety-admission-receipts/parent'
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    receipts = {}
    def archive(source_path, target):
        raw = source_path.read_bytes(); path = root / target
        with path.open('xb') as stream:
            os.chmod(path, 0o600); stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        relative = path.relative_to(base).as_posix(); receipts[relative] = digest(path)
        return relative, receipts[relative]
    phases = {}
    for phase in ('initial', 'coexistence'):
        phases[phase] = archive(base / ('logs/model-smoke-' + phase + '/complete.json'),
                                'model-smoke-' + phase + '-complete.json')
        for rank in range(8):
            archive(base / ('logs/model-smoke-' + phase + f'/rank-{rank}.json'),
                    'model-smoke-' + phase + f'-rank-{rank}.json')
    receipts['logs/diagnostic-authentication.json'] = digest(base / 'logs/diagnostic-authentication.json')
    receipts.update(initial_probe_files(base, run, diagnostic))
    initial = read(base / 'logs/model-smoke-initial/complete.json')
    allocation = read(base / 'logs/native-allocation-admission.json')
    need(allocation['accepted'] is True and allocation['run_id'] == run['run_id'] and
         allocation['slurm_job_id'] == os.environ['SLURM_JOB_ID'], 'New actual native allocation differs')
    probability = dict(run_id=run['run_id'], slurm_job_id=os.environ['SLURM_JOB_ID'],
        actual_optimizer_writes=0, source_sha256=run['source_sha256'],
        resolved_config_sha256=document_sha256(document),
        question_ids=[qid for row in initial['ranks'] for qid in row['question_ids']],
        actual_cuda_runtime_uuids=allocation['actual_cuda_runtime_uuids'],
        initial_proof=phases['initial'][0], initial_proof_sha256=phases['initial'][1],
        coexistence_proof=phases['coexistence'][0], coexistence_proof_sha256=phases['coexistence'][1])
    artifact = dict(schema_version=3, role=ROLE, accepted=True, run_id=run['run_id'],
        source_sha256=run['source_sha256'], initial_model_sha256=run['initial_model_sha256'],
        input_sha256=run['input_sha256'], budget=run['budget'], probability=probability,
        diagnostic_authentication='logs/diagnostic-authentication.json', diagnostic_run_id=diagnostic['run']['run_id'],
        diagnostic_execution_role=run['diagnostic_execution_role'],
        old_updates_counted_as_production=0, same_recipe_health_accepted=False,
        healthy_checkpoint_accepted=False, actual_production_short_train_accepted=False,
        initial_probe_version=run['initial_probe_version'], receipts_sha256=receipts, **thresholds,
        semantics='Guarded fresh production1..5 is this run; independent diagnostic is candidate provenance only. '
                  'Same-process five gate precedes6; no old weights, Adam or scheduler are inherited.')
    verify_bootstrap(base, artifact, run)
    _unstarted(base, run)
    write_once(base / 'logs/policy-safety-calibration.json', artifact)
    return artifact


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('base', 'discovery-base', 'diagnostic-proof', 'terminal-proof'):
        parser.add_argument('--' + key, required=True, type=Path)
    args = parser.parse_args()
    result = produce(args.base, args.discovery_base, args.diagnostic_proof, args.terminal_proof)
    print(json.dumps(dict(run_id=result['run_id'], bootstrap=True, healthy_checkpoint_accepted=False,
                          proof_sha256=document_sha256(result))))
