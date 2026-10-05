"""Freeze a child calibration plan after real admissions, before launch claim.

This writes identity and requirements, never a healthy artifact. Production is
untouched. A crash after the exclusive plan write may be completed only by an
identical, still-unclaimed child; any claim/write/checkpoint evidence rejects
re-entry. Actual actor/empty-Adam admission remains an all-rank runtime gate.
"""
import fcntl
import hashlib
import json
import os
import uuid
from pathlib import Path

from reproduction.common8.policy_calibration_plan import (
    CONFIRMATION_ROLE, DISCOVERY_ROLE, PlanError, _identity, _strict_json,
    canonical_bytes, create_confirmation_plan, create_discovery_plan,
    document_sha256, load_plan,
)


def _check(condition, message):
    if not condition:
        raise PlanError(message)


def _read(path):
    return _strict_json(Path(path).read_bytes())


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _unstarted(base, run):
    logs = base / 'logs'
    _check(type(run['completed_updates']) is int and run['completed_updates'] == 0,
           'Calibration plan requires a fresh child counter')
    forbidden = ('training-launch-claim.json', 'actual-updates.jsonl',
                 'policy-calibration-segment.json', 'policy-calibration-segment-complete.json')
    _check(not any((logs / name).exists() for name in forbidden), 'Child launch/update was already attempted')
    _check(not list(logs.glob('optimizer-rank-*.jsonl')) and
           not list(logs.glob('policy-calibration-optimizer-rank-*.json')),
           'Existing actual or partial optimizer evidence; never reset its identity')
    checkpoint = Path(run['checkpoint_dir'])
    _check(not checkpoint.exists() or not any(checkpoint.iterdir()), 'Existing child training checkpoint')
    _check(not Path(run['train_log']).exists(), 'Existing canonical trainer log belongs to an earlier attempt')


def _resolved_document(base, run):
    from omegaconf import OmegaConf
    from reproduction.common8.configuration import resolved
    actual, config, _ = resolved(base)
    _check(actual == run, 'Manifest changed while resolving child configuration')
    return OmegaConf.to_container(config, resolve=True)


def _mathematics(document, base, run):
    """Check actual resolved fields; a fixed contract hash alone is insufficient."""
    actor = document['actor_rollout_ref']['actor']
    reference = document['actor_rollout_ref']['ref']
    rollout = document['actor_rollout_ref']['rollout']
    expected = dict(ppo_mini_batch_size=128, ppo_micro_batch_size=16, ppo_epochs=1,
                    grad_clip=1.0, clip_ratio=0.2, state_masking=True, use_kl_loss=True,
                    kl_loss_coef=0.001, kl_loss_type='low_var_kl', use_dynamic_bsz=False,
                    ulysses_sequence_parallel_size=1, compute_dtype='float32', allow_tf32=False)
    for name, value in expected.items():
        _check(canonical_bytes(actor[name]) == canonical_bytes(value), 'Actual actor mathematics differs: ' + name)
    for name, value in dict(lr=1e-6, lr_warmup_steps_ratio=0.0, warmup_style='constant').items():
        _check(canonical_bytes(actor['optim'][name]) == canonical_bytes(value), 'Actual optimizer differs: ' + name)
    for part in (actor, reference):
        _check(part['compute_dtype'] == 'float32' and part['allow_tf32'] is False and
               part['fsdp_config']['model_dtype'] == 'fp32', 'Actual reference/actor precision differs')
    _check(all(actor['fsdp_config']['mixed_precision'][name] == 'fp32'
               for name in ('param_dtype', 'reduce_dtype', 'buffer_dtype')), 'Actual FSDP precision differs')
    for name, value in dict(n=1, n_agent=5, temperature=1.0, top_p=0.95,
                            log_prob_micro_batch_size=32, dtype='float32').items():
        _check(type(rollout[name]) is not bool and rollout[name] == value, 'Actual rollout differs: ' + name)
    _check(type(reference['log_prob_micro_batch_size']) is int and reference['log_prob_micro_batch_size'] == 32,
           'Actual reference log-prob micro batch differs')
    _check(document['algorithm']['adv_estimator'] == 'grpo' and document['algorithm']['no_think_rl'] is False,
           'Actual advantages/reasoning differ')
    for name, value in dict(train_batch_size=512, val_batch_size=64, max_prompt_length=4096,
                            max_response_length=500, max_start_length=2048, max_obs_length=500).items():
        _check(type(document['data'][name]) is int and document['data'][name] == value, 'Actual data budget differs: ' + name)
    _check(document['data']['train_files'] == str(base / 'data/train.parquet') and
           document['data']['val_files'] == str(base / 'data/stage.parquet') and
           document['actor_rollout_ref']['model']['path'] == run['initial_model'],
           'Actual original model/data paths differ')
    trainer = document['trainer']
    for name, value in dict(n_gpus_per_node=8, nnodes=1, total_training_steps=301,
                            total_epochs=15, save_freq=50, test_freq=50, acceptance_step=5).items():
        _check(type(trainer[name]) is int and trainer[name] == value, 'Actual training horizon/budget differs: ' + name)
    _check(trainer['default_local_dir'] == run['checkpoint_dir'] and trainer['val_only'] is False and
           trainer['val_before_train'] is True, 'Actual initial evaluation/checkpoint behavior differs')
    policy = actor['policy_safety']
    _check(policy['enabled'] is True and policy['calibration_manifest'] == str(base / 'logs/policy-safety-calibration.json'),
           'Actual required safety path differs')
    for name, value in dict(dual_clip=3.0, target_old_policy_kl=0.01, kl_log_bound=30.0).items():
        _check(type(policy[name]) in (int, float) and policy[name] == value, 'Actual safeguard candidate differs: ' + name)
    _check(policy['negative_tail_limit'] is None, 'Resolved tail must remain unset; confirmation binds its explicit plan')


def _actual_prerequisites(base, run, document, config_sha, job_id):
    """Use existing actual native/HTTP/probe/model gates; no mockable env bypass."""
    from reproduction.common8.stage_retrieval import binding, memory_bytes
    from reproduction.common8.probe_admission import verify_training_gate
    from reproduction.common8.model_smoke import aggregate, coexistence
    from reproduction.common8.validation_identity import expected
    actual_run, job, allocation = binding(base)
    _check(actual_run == run and job['JobId'] == job_id and memory_bytes(job['TRES']) >= 512 * 2**30,
           'Actual native allocation/run differs')
    _check(verify_training_gate(base) == run, 'Actual fitted-probe gate changed manifest')
    _, _, ids = expected(base, 'stage')
    for phase in ('initial', 'coexistence'):
        root = base / 'logs' / ('model-smoke-' + phase)
        proof = _read(root / 'complete.json')
        _check(proof['accepted'] is True and proof['run_id'] == run['run_id'] and
               proof['slurm_job_id'] == job_id and proof['phase'] == phase and
               type(proof['actual_grpo_updates']) is int and proof['actual_grpo_updates'] == 0,
               'Actual initial/coexistence model gate differs')
        ranks = []
        for rank in range(8):
            path = root / f'rank-{rank}.json'
            _check(_sha(path) == proof['rank_file_sha256'][str(rank)], 'Actual model rank bytes changed')
            row = _read(path)
            _check(type(row['rank']) is int and row['rank'] == rank and
                   type(row['actual_optimizer_writes']) is int and row['actual_optimizer_writes'] == 0 and
                   row['initial_parameters_finite'] is True, 'Actual initial rank did not prove finite/fresh0')
            ranks.append(row)
        _check(ranks == proof['ranks'] and aggregate(ranks, run['run_id'], job_id, phase, ids,
               allocation['actual_cuda_runtime_uuids']) == config_sha, 'Actual eight-rank model/config gate differs')
    coexistence(base, run, job)  # Owned PID births and live endpoint identity.


def _reward_fingerprint(base, run, environ):
    from search_g1.policy_safety import reward_math_fingerprint
    live = _read(base / 'artifacts/live/manifest.json')
    accepted = _read(base / 'logs/initial-probes-accepted.json')
    _check(type(live['round']) is int and live['round'] == 0 and live['run_id'] == run['run_id'] and
           live['version'] == run['initial_probe_version'] == accepted['version'] and
           accepted['accepted'] is True and accepted['run_id'] == run['run_id'] and
           accepted['live_manifest_sha256'] == _sha(base / 'artifacts/live/manifest.json'),
           'Actual initial probe admission/version differs')
    required = set(run['initial_model_sha256']) - {'vocab.json', 'merges.txt', 'generation_config.json'}
    _check(set(live['encoder_sha256']) == required and live['encoder'] == run['initial_model'] and
           all(live['encoder_sha256'][name] == run['initial_model_sha256'][name] for name in required),
           'Actual initial encoder fingerprint differs')
    _check(set(live['head_sha256']) == {'commit_probe.pkl', 'boundary_probe.pkl'}, 'Both actual initial heads required')
    for name, key in (('commit_probe.pkl', 'commit'), ('boundary_probe.pkl', 'boundary')):
        _check(_sha(live[key]) == live['head_sha256'][name], 'Actual initial head bytes changed')
    return dict(probe_round=0, encoder_sha256=live['encoder_sha256'], head_sha256=live['head_sha256'],
                reward_math_sha256=reward_math_fingerprint(run, environ))


def _confirmation(base, parent, run, job_id, gate_sha, config_sha, reward):
    discovery = _read(base / 'preparation/policy-safety-discovery-plan.json')
    evidence = (base / 'preparation/policy-safety-discovery-evidence.json').read_bytes()
    candidate = _read(base / 'preparation/policy-safety-confirmation-candidates.json')
    _check(set(candidate) == {'schema_version', 'role', 'parent_production_run_id', 'confirmation_run_id',
                             'discovery_run_id', 'discovery_evidence_sha256', 'negative_tail_limit',
                             'health_rule', 'threshold_selection'} and
           type(candidate['schema_version']) is int and candidate['schema_version'] == 1 and
           candidate['role'] == 'policy_safety_confirmation_candidates' and
           candidate['parent_production_run_id'] == parent['run_id'] and
           candidate['confirmation_run_id'] == run['run_id'] and candidate['discovery_run_id'] == discovery['run_id'] and
           candidate['discovery_evidence_sha256'] == hashlib.sha256(evidence).hexdigest(),
           'Explicit real discovery/confirmation candidate identity differs')
    plan = create_confirmation_plan(discovery, parent, run['run_id'], slurm_job_id=job_id,
                                    probability_proof_sha256=gate_sha, resolved_config_sha256=config_sha,
                                    discovery_evidence_bytes=evidence, negative_tail_limit=candidate['negative_tail_limit'],
                                    threshold_selection=candidate['threshold_selection'], health_rule=candidate['health_rule'])
    _check(plan['initial_reward_fingerprint'] == reward, 'Confirmation refitted/replaced initial reward identity')
    return plan


def _fsync_directory(path):
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _replace_manifest(path, old_bytes, document):
    """Durably replace only the same manifest read under the exclusive lock."""
    temporary = path.with_name(path.name + '.freeze-' + uuid.uuid4().hex + '.tmp')
    with temporary.open('xb') as stream:
        stream.write(canonical_bytes(document)); stream.flush(); os.fsync(stream.fileno())
    _check(path.read_bytes() == old_bytes, 'Child manifest changed before atomic plan binding; preserve temporary evidence')
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def freeze(base, environ=None):
    """Freeze/recover an identical unclaimed child plan; production is a no-op."""
    environ = os.environ if environ is None else environ
    base = Path(base).resolve()
    manifest_path = base / 'logs/run-manifest.json'
    run = _read(manifest_path)
    role = run.get('policy_calibration_role', 'production')
    if role == 'production':
        return dict(role='production', plan_created=False, production_admission=False,
                    scope='production uses its existing strict calibrated-admission binder')
    _check(role in (DISCOVERY_ROLE, CONFIRMATION_ROLE), 'Unknown calibration role')
    job_id = environ.get('SLURM_JOB_ID')
    _check(isinstance(job_id, str) and job_id.isdecimal() and int(job_id) > 0,
           'Calibration freeze requires actual native Slurm identity')
    lock_path = base / 'logs/policy-safety-plan-freeze.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        original = manifest_path.read_bytes(); run = _strict_json(original)
        _check(run.get('policy_calibration_role') == role, 'Calibration role changed before lock')
        _unstarted(base, run)
        parent_bytes = (base / 'preparation/parent-run-manifest.json').read_bytes()
        _check(hashlib.sha256(parent_bytes).hexdigest() == run['policy_calibration_parent_manifest_sha256'],
               'Original parent manifest bytes changed')
        parent = _strict_json(parent_bytes)
        _check(parent.get('policy_calibration_role', 'production') == 'production' and
               run['parent_production_run_id'] == parent['run_id'] and run['run_id'] != parent['run_id'] and
               canonical_bytes(_identity(run)) == canonical_bytes(_identity(parent)),
               'Child source/model/data/parent identity differs')
        document = _resolved_document(base, run)
        _mathematics(document, base, run)
        config_sha = document_sha256(document)
        _actual_prerequisites(base, run, document, config_sha, job_id)
        gate_sha = _sha(base / 'logs/model-smoke-initial/complete.json')
        reward = _reward_fingerprint(base, run, environ)
        if role == DISCOVERY_ROLE:
            plan = create_discovery_plan(parent, run['run_id'], slurm_job_id=job_id,
                                         probability_proof_sha256=gate_sha, resolved_config_sha256=config_sha,
                                         initial_reward_fingerprint=reward)
        else:
            plan = _confirmation(base, parent, run, job_id, gate_sha, config_sha, reward)
        plan_sha = document_sha256(plan)
        bound = dict(run, policy_calibration_plan_sha256=plan_sha, policy_calibration_slurm_job_id=job_id)
        for name in ('policy_calibration_plan_sha256', 'policy_calibration_slurm_job_id'):
            _check(name not in run or run[name] == bound[name], 'Existing frozen child binding differs: ' + name)
        target = base / 'logs/policy-safety-calibration-plan.json'
        existing = target.exists()
        if existing:
            _check(canonical_bytes(_read(target)) == canonical_bytes(plan), 'Existing immutable child plan differs')
        else:
            with target.open('xb') as stream:
                stream.write(canonical_bytes(plan)); stream.flush(); os.fsync(stream.fileno())
            _fsync_directory(target.parent)
        # If interrupted here, identical unclaimed replay verifies the original
        # plan, rechecks all real gates, and completes this binding; never reset.
        _unstarted(base, run)
        if canonical_bytes(bound) != canonical_bytes(run):
            _replace_manifest(manifest_path, original, bound)
        loaded, digest = load_plan(base, bound, expected_slurm_job_id=job_id)
        _check(loaded == plan and digest == plan_sha, 'Final durable plan binding differs')
        return dict(schema_version=1, role='policy_safety_calibration_plan_freeze', execution_role=role,
                    run_id=run['run_id'], parent_production_run_id=parent['run_id'], slurm_job_id=job_id,
                    plan_sha256=plan_sha, plan_created=not existing, plan_frozen=True,
                    production_admission=False, healthy_checkpoint_accepted=False,
                    scope='actual admission hashes frozen before claim; no short-segment/health proof')


if __name__ == '__main__':
    print(json.dumps(freeze(os.environ['SEARCH_G1_BASE']), sort_keys=True))
