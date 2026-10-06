"""Exact live fresh200 authority and policy/config state; no training arithmetic."""
import copy
import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path

LIVE_SCOPE = 'GIGPO_POLICY_SAFETY_OFFICIAL_INSTRUCT_FRESH200_LIVE_V1'
POLICY_SCOPE = 'FRESH200_SAME_JOB_FIRST5_STATE_AND_PRODUCTION_POLICY_V1'
LIVE_GOAL_UPDATED_AT = 1791295951
LIVE_AUTHORIZATION_GOAL_UPDATED_AT = 1791293201
ADMISSION_SCHEMA = 'FRESH200_REALIZED_POLICY_CONFIG_V1'
AUDIT_ENV_KEYS = ('SEARCH_R1_BASELINE_CONFIG_SHA256', 'GIGPO_POLICY_NATIVE_BINDING',
                  'GIGPO_POLICY_NATIVE_BINDING_SHA256')
POLICY_DELTA_KEYS = {'mode', 'negative_tail_limit', 'calibration_evidence_path',
                     'calibration_evidence_sha256'}


def require(value, reason):
    if not value:
        raise ValueError(reason)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def core(record):
    return {key: record[key] for key in ('path', 'bytes', 'sha256')}


def read_bound_json(record):
    """Hash and parse one canonical no-follow FD; never reopen after hashing."""
    path = Path(record['path'])
    require(path.is_absolute() and path.resolve(strict=True) == path
            and type(record['bytes']) is int and record['bytes'] >= 0
            and isinstance(record['sha256'], str) and re.fullmatch(r'[0-9a-f]{64}', record['sha256']),
            'Noncanonical live evidence record')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and before.st_size == record['bytes'], 'Live evidence type/size differs')
        blocks = []
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            blocks.append(block)
            require(sum(map(len, blocks)) <= record['bytes'], 'Live evidence grew')
        raw = b''.join(blocks)
        after, named = os.fstat(fd), os.lstat(path)
        identity = lambda info: (info.st_dev, info.st_ino, info.st_mode, info.st_size,
                                 info.st_mtime_ns, info.st_ctime_ns)
        require(identity(before) == identity(after) == identity(named)
                and hashlib.sha256(raw).hexdigest() == record['sha256'], 'Live evidence identity/SHA changed')
        def pairs(items):
            result = {}
            for key, value in items:
                require(key not in result, 'Duplicate live evidence JSON key')
                result[key] = value
            return result
        def invalid(value):
            raise ValueError('Nonfinite live evidence JSON: ' + value)
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
    finally:
        os.close(fd)


def existing_record(path, expected_sha=None):
    path = Path(path)
    raw = path.read_bytes()
    value = dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    require(expected_sha is None or value['sha256'] == expected_sha, 'Actual live input SHA differs')
    read_bound_json(value)
    return value


def live_initialization():
    from .fresh200 import calibration_initialization
    value = calibration_initialization()
    value['diagnostic_total_optimizer_opportunities'] = 100
    return value


def validate_live_initialization(value):
    for key, expected in live_initialization().items():
        require(key in value and type(value[key]) is type(expected) and value[key] == expected,
                'Live official-zero protocol differs: ' + key)


def validate_live_binding(proof, actual_source):
    """An old isolated-two READY can never admit this five-to200 session."""
    from .fresh200 import CALIBRATION_SCOPE, TOKEN_KL_LIMIT
    expected = dict(status='REVIEWED_FOR_FRESH200_LIVE_TRAINING_ONLY',
        validation_scope=LIVE_SCOPE, calibration_scope=CALIBRATION_SCOPE,
        live_goal_updated_at=LIVE_GOAL_UPDATED_AT, live_authorization_goal_updated_at=LIVE_AUTHORIZATION_GOAL_UPDATED_AT,
        official_training_outer_target=200,
        initial_calibration_counted_outer_limit=5, first_full_acceptance_step=5,
        first5_count_toward_goal=True, restart_after_calibration=False,
        first5_known_token_kl_enforced=True, first5_tail_observation_only=True,
        production_tail_required_before_outer6=True, retained_training_updates_contributed=0,
        target_old_policy_kl=TOKEN_KL_LIMIT, candidate_old_policy_kl=TOKEN_KL_LIMIT,
        token_kl_threshold_source='human', token_kl_human_goal_updated_at=1791282908,
        outer_counting_authorization_source='human', outer_counting_human_goal_updated_at=1791283983)
    for key, value in expected.items():
        require(key in proof and type(proof[key]) is type(value) and proof[key] == value,
                'Different live human goal binding: ' + key)
    require(proof.get('allowed_safety_modes') == ['calibration', 'production']
            and proof.get('code_commit') == proof.get('source_commit') == actual_source
            and type(actual_source) is str and re.fullmatch(r'[0-9a-f]{40}', actual_source)
            and proof.get('source_native_acceptance') is None
            and proof.get('negative_tail_limit') is None
            and proof.get('actual_gpu_acceptance') is None
            and proof.get('actual_probability_acceptance') is None,
            'Old proof, unbound source, or invented initial live acceptance')
    validate_live_initialization(proof)


def _initial_payload():
    from .fresh200 import TOKEN_KL_LIMIT
    binding = existing_record(os.environ['GIGPO_POLICY_NATIVE_BINDING'],
                              os.environ['GIGPO_POLICY_NATIVE_BINDING_SHA256'])
    proof = read_bound_json(binding)
    validate_live_binding(proof, os.environ['SEARCH_R1_BASELINE_COMMIT'])
    config = existing_record(Path(os.environ['SEARCH_R1_BASELINE_OUTPUT']) / 'resolved-config-C0.json',
                             os.environ['SEARCH_R1_BASELINE_CONFIG_SHA256'])
    resolved = read_bound_json(config)
    policy = resolved['actor_rollout_ref']['actor']['policy_safety']
    require(policy['mode'] == 'calibration' and policy['negative_tail_limit'] is None
            and policy['first5_known_token_kl_enforced'] is True and policy['first5_tail_observation_only'] is True
            and policy['candidate_old_policy_kl'] == policy['target_old_policy_kl'] == TOKEN_KL_LIMIT,
            'Initial realized calibration policy differs')
    return dict(schema=ADMISSION_SCHEMA, phase='C0_CALIBRATION', config_epoch=0,
        job_id=os.environ['SLURM_JOB_ID'], run_id=os.environ['SEARCH_R1_BASELINE_RUN_ID'],
        source_commit=os.environ['SEARCH_R1_BASELINE_COMMIT'], live_goal_updated_at=LIVE_GOAL_UPDATED_AT,
        live_authorization_goal_updated_at=LIVE_AUTHORIZATION_GOAL_UPDATED_AT,
        initial_binding_record=binding, c0_config_record=config, config_record=config,
        policy=copy.deepcopy(policy), policy_review_record=None, first5_manifest_record=None,
        probability_approval_record=None, parent_admission_sha256=None)


def validate_admission(payload):
    require(payload.get('schema') == ADMISSION_SCHEMA and type(payload.get('config_epoch')) is int
            and payload['config_epoch'] in (0, 1) and payload['live_goal_updated_at'] == LIVE_GOAL_UPDATED_AT
            and payload['live_authorization_goal_updated_at'] == LIVE_AUTHORIZATION_GOAL_UPDATED_AT,
            'Unsupported live realized admission')
    for key, name in (('job_id', 'SLURM_JOB_ID'), ('run_id', 'SEARCH_R1_BASELINE_RUN_ID'),
                      ('source_commit', 'SEARCH_R1_BASELINE_COMMIT')):
        require(payload[key] == os.environ[name], 'Different realized live identity: ' + key)
    root = Path(os.environ['SEARCH_R1_BASELINE_OUTPUT'])
    require(payload['initial_binding_record']['path'] == os.environ['GIGPO_POLICY_NATIVE_BINDING']
            and payload['initial_binding_record']['sha256'] == os.environ['GIGPO_POLICY_NATIVE_BINDING_SHA256']
            and payload['config_record']['path'] == str(root / f"resolved-config-C{payload['config_epoch']}.json")
            and payload['c0_config_record']['path'] == str(root / 'resolved-config-C0.json'),
            'Actual launcher/config byte identities differ')
    initial = read_bound_json(payload['initial_binding_record'])
    validate_live_binding(initial, payload['source_commit'])
    resolved = read_bound_json(payload['config_record'])
    require(resolved['actor_rollout_ref']['actor']['policy_safety'] == payload['policy'],
            'Realized policy does not match actual full config bytes')
    require(resolved['actor_rollout_ref']['actor']['fresh200_live_protocol'] is True,
            'Live policy installed in a legacy actor')
    require(payload['policy']['live_training_protocol'] is True
            and payload['policy']['first5_known_token_kl_enforced'] is True
            and payload['policy']['first5_tail_observation_only'] is True,
            'Retained live first5 lost the known human KL brake provenance')
    if payload['config_epoch'] == 0:
        require(payload['phase'] == 'C0_CALIBRATION' and payload['policy']['mode'] == 'calibration'
                and payload['policy']['negative_tail_limit'] is None
                and payload['config_record'] == payload['c0_config_record']
                and all(payload[key] is None for key in ('policy_review_record', 'first5_manifest_record',
                    'probability_approval_record', 'parent_admission_sha256')), 'Invented initial live admission')
    else:
        c0 = read_bound_json(payload['c0_config_record'])
        validate_config_delta(c0, resolved)
        parent = copy.deepcopy(payload)
        parent.update(phase='C0_CALIBRATION', config_epoch=0, config_record=payload['c0_config_record'],
            policy=c0['actor_rollout_ref']['actor']['policy_safety'], policy_review_record=None,
            first5_manifest_record=None, probability_approval_record=None, parent_admission_sha256=None)
        require(payload['parent_admission_sha256'] == digest(parent), 'C1 admission is not the actual C0 byte chain')
        review = read_bound_json(payload['policy_review_record'])
        validate_production_review(review, payload)
        require(payload['phase'] == 'C1_PRODUCTION', 'Wrong production admission phase')
    return resolved


def validate_config_delta(c0, c1):
    left, right = copy.deepcopy(c0), copy.deepcopy(c1)
    old = left['actor_rollout_ref']['actor'].pop('policy_safety')
    new = right['actor_rollout_ref']['actor'].pop('policy_safety')
    require(left == right and set(old) == set(new)
            and all(old[key] == new[key] for key in old if key not in POLICY_DELTA_KEYS),
            'Production approval changed mathematical/configuration inputs')
    require(old['mode'] == 'calibration' and old['negative_tail_limit'] is None
            and new['mode'] == 'production' and type(new['negative_tail_limit']) in (int, float)
            and math.isfinite(new['negative_tail_limit']) and 0 <= new['negative_tail_limit'] <= 1,
            'Missing actual approved production tail')


def validate_production_review(review, payload):
    from .fresh200 import validate_attempt_ledger
    expected = dict(status='PASS', scope=POLICY_SCOPE, independent_review=True,
        implementation_authorship=False, root_conversation_inherited=False,
        job_id=payload['job_id'], run_id=payload['run_id'], source_commit=payload['source_commit'],
        live_goal_updated_at=LIVE_GOAL_UPDATED_AT,
        live_authorization_goal_updated_at=LIVE_AUTHORIZATION_GOAL_UPDATED_AT, first5_count_toward_goal=True,
        logical_step=5, actual_new_updates=5, manifest_sha256=payload['first5_manifest_record']['sha256'],
        config_sha256=payload['c0_config_record']['sha256'],
        initial_binding_sha256=payload['initial_binding_record']['sha256'],
        probability_approval_sha256=payload['probability_approval_record']['sha256'])
    for key, value in expected.items():
        require(key in review and type(review[key]) is type(value) and review[key] == value,
                'Different independent first5 production review: ' + key)
    ledger = review['fresh200_attempt_ledger']; validate_attempt_ledger(ledger)
    require(ledger['completed_outer_updates'] == 5 and ledger['active_attempt'] is None
            and review['actual_adam_total'] == ledger['actual_adam_total']
            and review['scheduler_step'] == 5 and review['rank_count'] == 8,
            'First5 policy evidence did not retain actual complete/partial writes')
    evidence = read_bound_json(review['actual_tail_evidence_record'])
    require(evidence['completed_outer_updates'] == 5 and evidence['actual_attempt_ledger_digest'] == digest(ledger)
            and evidence['job_id'] == payload['job_id'] and evidence['run_id'] == payload['run_id']
            and evidence['source_commit'] == payload['source_commit']
            and evidence['config_sha256'] == payload['c0_config_record']['sha256']
            and len(evidence['rank_guard_records']) == 8,
            'Tail was not selected from this actual first5 evidence')
    require(review['role_admissions_digest'] == evidence['role_admissions_digest'] == digest(evidence['role_admissions'])
            and set(evidence['role_admissions']) == {'actor', 'reference'},
            'First5 production review lost actual actor/reference config identity')
    for role, rows in evidence['role_admissions'].items():
        require(len(rows) == 8 and {row['rank'] for row in rows} == set(range(8))
                and len({row['pid'] for row in rows}) == 8
                and all(type(row['rank']) is int and type(row['pid']) is int and row['pid'] > 0
                    and type(row['start_ticks']) is str and row['start_ticks'].isdecimal()
                    and row['role'] == role and row['state']['payload']['config_epoch'] == 0
                    and row['state']['payload']['config_record'] == payload['c0_config_record'] for row in rows),
                'First5 actual role/config all8 coverage differs')
    require({row['rank'] for row in evidence['rank_guard_records']} == set(range(8))
            and len({row['path'] for row in evidence['rank_guard_records']}) == 8,
            'Missing/duplicate actual first5 rank guard evidence')
    for record in evidence['rank_guard_records']:
        rows = read_bound_json_lines(record)
        require(rows and all(row['rank'] == record['rank'] and row['execution_mode'] == 'calibration'
                and row['commit'] == payload['source_commit'] and row['run_id'] == payload['run_id']
                and row['config_sha256'] == payload['c0_config_record']['sha256']
                and row['native_binding_sha256'] == payload['initial_binding_record']['sha256']
                and row['attempt_id'] in {attempt['attempt_id'] for attempt in ledger['history']}
                for row in rows), 'Tail log is not actual original C0 allrank evidence')
    approved = review['production_policy']
    policy = payload['policy']
    require(approved == {key: policy[key] for key in
        ('target_old_policy_kl', 'negative_tail_limit', 'dual_clip', 'kl_log_bound')}
        and approved['target_old_policy_kl'] == .2 and approved['dual_clip'] == 3.0
        and approved['kl_log_bound'] == 30.0 and type(approved['negative_tail_limit']) in (int, float)
        and math.isfinite(approved['negative_tail_limit']) and 0 <= approved['negative_tail_limit'] <= 1
        and policy['calibration_evidence_path'] == payload['policy_review_record']['path']
        and policy['calibration_evidence_sha256'] == payload['policy_review_record']['sha256'],
        'Realized policy differs from independently approved production policy')
    manifest = read_bound_json(payload['first5_manifest_record'])
    require(manifest['job_id'] == payload['job_id'] and manifest['code_commit'] == payload['source_commit']
            and manifest['logical_step'] == manifest['actual_new_updates'] == 5
            and manifest['config_sha256'] == payload['c0_config_record']['sha256'],
            'Production policy belongs to another saved first5 state')
    ready = read_bound_json(review['first5_state_ready_record'])
    require(ready['manifest']['sha256'] == payload['first5_manifest_record']['sha256']
            and ready['manifest']['identity'] == manifest
            and ready['checkpoint_pause_identity'] == review['checkpoint_pause_identity']
            and len(ready['actual_state_roundtrip']) == 8
            and ready['actual_driver_state_roundtrip']['actual_live_admission_reload'] is True
            and ready['actual_worker_runtime_roundtrip']['complete_live_before_after_equal'] is True,
            'First5 production review lacks actual same-PID fullstate/admission readback')
    pauses = ready['checkpoint_pause_identity']
    require(len(pauses) == 8 and {row['rank'] for row in pauses} == set(range(8))
            and len({row['pid'] for row in pauses}) == 8
            and all(row['scheduler_step'] == 5 for row in pauses), 'First5 saved worker identity coverage differs')
    probability = read_bound_json(payload['probability_approval_record'])
    require(probability['status'] == 'PASS' and probability['actual_adam_writes'] == 0
            and probability['scope'] == 'GIGPO_OFFICIAL_INSTRUCT_FRESH200_RAW_ACTION_PROBABILITY_IDENTITY_V1'
            and probability['actual_hf_role_micro_batch_sizes'] == {'old': 4, 'current': 2, 'reference': 4}
            and probability['actual_dropout_neutral_train_eval_verified'] is True
            and probability['job_id'] == payload['job_id'] and probability['run_id'] == payload['run_id']
            and probability['code_commit'] == payload['source_commit'], 'Production lost actual before-Adam parity approval')


def read_bound_json_lines(record):
    """JSONL is read as one stable bound byte buffer, using the JSON reader's FD rule."""
    # The wrapper is data, not executable source. Bound the actual raw log then
    # parse each original line without manufacturing a combined PASS artifact.
    path = Path(record['path'])
    require(path.is_absolute() and path.resolve(strict=True) == path
            and type(record['bytes']) is int and record['bytes'] >= 0,
            'Noncanonical actual rank guard log')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        before = os.fstat(fd); raw = b''
        require(stat.S_ISREG(before.st_mode) and before.st_size == record['bytes'], 'Guard log size/type differs')
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block: break
            raw += block
            require(len(raw) <= record['bytes'], 'Guard log grew')
        after, named = os.fstat(fd), os.lstat(path)
        identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
        require(identity(before) == identity(after) == identity(named)
                and hashlib.sha256(raw).hexdigest() == record['sha256'], 'Actual guard log identity/SHA changed')
        def pairs(items):
            value = {}
            for key, item in items:
                require(key not in value, 'Duplicate actual rank guard JSON key')
                value[key] = item
            return value
        def invalid(value):
            raise ValueError('Nonfinite actual rank guard JSON: ' + value)
        return [json.loads(line, object_pairs_hook=pairs, parse_constant=invalid) for line in raw.splitlines()]
    finally:
        os.close(fd)


def driver_admission_state(trainer):
    if not trainer.config.actor_rollout_ref.actor.get('fresh200_live_protocol', False): return None
    if not hasattr(trainer, 'fresh200_live_admission'): trainer.fresh200_live_admission = _initial_payload()
    payload = copy.deepcopy(trainer.fresh200_live_admission)
    resolved = validate_admission(payload)
    from omegaconf import OmegaConf
    require(resolved == OmegaConf.to_container(trainer.config, resolve=True), 'Driver actual resolved config is not admitted bytes')
    require(os.environ['SEARCH_R1_BASELINE_CONFIG_SHA256'] == payload['config_record']['sha256'],
            'Driver audit producers use another config SHA')
    roles = {}
    for role, group in (('actor', trainer.actor_rollout_wg), ('reference', trainer.ref_policy_wg)):
        rows = group.calibration_live_admission_identity()
        require(len(rows) == 8 and {row['rank'] for row in rows} == set(range(8))
                and len({row['pid'] for row in rows}) == 8
                and all(type(row['rank']) is int and type(row['pid']) is int and row['pid'] > 0
                    and type(row['start_ticks']) is str and row['start_ticks'].isdecimal()
                    and row['role'] == role and row['state']['payload'] == payload for row in rows),
                'Driver lacks actual coherent all8 actor/reference admissions')
        roles[role] = sorted(rows, key=lambda row: row['rank'])
    trainer.fresh200_live_role_admissions = copy.deepcopy(roles)
    return dict(payload=payload, resolved_config=resolved,
                role_admissions=roles, audit_environment={key: os.environ[key] for key in AUDIT_ENV_KEYS})


def worker_admission_state(worker):
    if not worker.config.actor.get('fresh200_live_protocol', False): return None
    if not hasattr(worker, 'fresh200_live_admission'): worker.fresh200_live_admission = _initial_payload()
    payload = copy.deepcopy(worker.fresh200_live_admission)
    validate_admission(payload)
    from omegaconf import OmegaConf
    policy = OmegaConf.to_container(worker.config.actor.policy_safety, resolve=True)
    require(policy == payload['policy'] and os.environ['SEARCH_R1_BASELINE_CONFIG_SHA256'] == payload['config_record']['sha256'],
            'Worker actor/reference policy or audit env is not admitted config')
    if worker._is_actor:
        require(OmegaConf.to_container(worker.actor.config.policy_safety, resolve=True) == policy,
                'Actual actor uses a different safety policy object')
    return dict(payload=payload, worker_config=OmegaConf.to_container(worker.config, resolve=True),
                audit_environment={key: os.environ[key] for key in AUDIT_ENV_KEYS})


def install_worker_admission(worker, payload):
    """Called only at a known idle completed5 boundary; pure policy/env metadata."""
    from omegaconf import OmegaConf
    validate_admission(payload)
    require(hasattr(worker, 'fresh200_live_admission'), 'Worker C0 admission was never actually observed')
    old = worker.fresh200_live_admission
    validate_admission(old)
    require(old['config_epoch'] == 0 and payload['config_epoch'] == 1
            and payload['parent_admission_sha256'] == digest(old)
            and os.environ['SEARCH_R1_BASELINE_CONFIG_SHA256'] in
                (old['config_record']['sha256'], payload['config_record']['sha256']),
            'Admission is not the actual C0 to C1 chain')
    before = OmegaConf.to_container(worker.config, resolve=True)
    require(before['actor']['policy_safety'] == old['policy'], 'Worker actual C0 policy differs')
    OmegaConf.update(worker.config, 'actor.policy_safety', copy.deepcopy(payload['policy']), merge=False)
    worker.fresh200_live_admission = copy.deepcopy(payload)
    os.environ['SEARCH_R1_BASELINE_CONFIG_SHA256'] = payload['config_record']['sha256']
    after = OmegaConf.to_container(worker.config, resolve=True)
    left, right = copy.deepcopy(before), copy.deepcopy(after)
    left['actor'].pop('policy_safety'); right['actor'].pop('policy_safety')
    require(left == right, 'Admission mutated normalized worker mathematical config')
    return worker_admission_state(worker)


def restore_worker_admission_state(worker, state):
    from omegaconf import OmegaConf
    payload = state['payload']; validate_admission(payload)
    live = OmegaConf.to_container(worker.config, resolve=True)
    saved = copy.deepcopy(state['worker_config'])
    left, right = copy.deepcopy(live), copy.deepcopy(saved)
    left['actor'].pop('policy_safety'); right['actor'].pop('policy_safety')
    require(left == right and saved['actor']['policy_safety'] == payload['policy'], 'Saved worker mathematical config differs')
    require(set(state['audit_environment']) == set(AUDIT_ENV_KEYS)
            and state['audit_environment']['GIGPO_POLICY_NATIVE_BINDING'] == payload['initial_binding_record']['path']
            and state['audit_environment']['SEARCH_R1_BASELINE_CONFIG_SHA256'] == payload['config_record']['sha256']
            and state['audit_environment']['GIGPO_POLICY_NATIVE_BINDING_SHA256'] == payload['initial_binding_record']['sha256'],
            'Saved worker audit/admission identity differs')
    OmegaConf.update(worker.config, 'actor.policy_safety', copy.deepcopy(payload['policy']), merge=False)
    worker.fresh200_live_admission = copy.deepcopy(payload)
    os.environ.update(state['audit_environment'])
    require(worker_admission_state(worker) == state, 'Actual restored worker admission/config/audit readback differs')


def restore_driver_admission_state(trainer, state):
    from omegaconf import OmegaConf
    resolved = validate_admission(state['payload'])
    require(state['resolved_config'] == resolved == OmegaConf.to_container(trainer.config, resolve=True),
            'Saved driver admission differs from actual live config')
    require(set(state['audit_environment']) == set(AUDIT_ENV_KEYS)
            and state['audit_environment']['SEARCH_R1_BASELINE_CONFIG_SHA256'] == state['payload']['config_record']['sha256']
            and state['audit_environment']['GIGPO_POLICY_NATIVE_BINDING'] == state['payload']['initial_binding_record']['path']
            and state['audit_environment']['GIGPO_POLICY_NATIVE_BINDING_SHA256'] == state['payload']['initial_binding_record']['sha256']
            and set(state['role_admissions']) == {'actor', 'reference'}, 'Saved driver audit/role identity differs')
    trainer.fresh200_live_admission = copy.deepcopy(state['payload'])
    os.environ.update(state['audit_environment'])
    for role, group in (('actor', trainer.actor_rollout_wg), ('reference', trainer.ref_policy_wg)):
        restored = group.calibration_restore_live_admission(state['role_admissions'][role])
        require(sorted(restored, key=lambda row: row['rank']) == state['role_admissions'][role],
                'Actual saved actor/reference admission restore differs')
    require(driver_admission_state(trainer) == state, 'Actual restored driver admission/audit readback differs')


def validate_live_safety_config(config):
    require(config.get('live_training_protocol') is True, 'Live source safety opt-in absent')
    if config.get('mode') == 'calibration':
        payload = _initial_payload()
    else:
        require(config.get('mode') == 'production', 'Unknown live safety phase')
        payload = read_bound_json(existing_record(Path(os.environ['SEARCH_R1_BASELINE_OUTPUT']) /
                                                 'production-admission-C1.json'))
    validate_admission(payload)
    require(dict(config) == payload['policy']
            and payload['config_record']['sha256'] == os.environ['SEARCH_R1_BASELINE_CONFIG_SHA256'],
            'Actual live safety policy/audit SHA is not approved')


def validate_live_update_boundary(worker, data):
    """Local checks before the known-idle runtime admission discriminator."""
    state = worker_admission_state(worker)
    require(state is not None, 'Live update has no actual worker admission')
    payload = state['payload']
    before = data.meta_info['fresh200_completed_before']
    ledger = getattr(worker, 'fresh200_worker_ledger', {'completed_outer_updates': 0})
    require(type(before) is int and before == ledger['completed_outer_updates']
            and 0 <= before < 200
            and type(data.meta_info['fresh200_config_epoch']) is int
            and data.meta_info['fresh200_config_epoch'] == payload['config_epoch']
            and data.meta_info['fresh200_config_sha256'] == payload['config_record']['sha256'],
            'Actual update boundary/config epoch differs')
    require((payload['config_epoch'] == 0 and before < 5)
            or (payload['config_epoch'] == 1 and before >= 5),
            'Sixth update requires actual same-job first5 production admission')
