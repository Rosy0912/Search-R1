"""Pure, frozen identities for separate discovery and confirmation child runs.

This module is a plan reader, not an admission producer. It does not run workers,
authenticate JSON as physical execution, choose health thresholds, or mutate a
production run. Collective validation/decisions belong to the calling launcher.
Even a valid plan or discovery declaration never grants production admission.
"""
import copy
import hashlib
import json
import math
import os
import re
from collections.abc import Mapping
from pathlib import Path


SCHEMA_VERSION = 1
DISCOVERY_ROLE = 'policy_safety_calibration_discovery'
CONFIRMATION_ROLE = 'policy_safety_calibration_confirmation'
MODEL_ID = 'Qwen/Qwen2.5-3B-Instruct'
MODEL_REVISION = 'aa8e72537993ba99e69dfaafa59ed015b17504d1'
COMMON_MANIFEST_SHA256 = '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
_SHA = re.compile(r'[0-9a-f]{64}\Z')
_RUN = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z')
_COMMIT = re.compile(r'[0-9a-f]{40}\Z')
_JOB = re.compile(r'[1-9][0-9]*\Z')

# A frozen mathematical contract, not measured runtime or healthy-run evidence.
MATHEMATICS = dict(
    world_size=8, train_batch_size=512, n_agent=5, rollout_n=1,
    sequence_parallel_size=1, trajectory_count=2560, ppo_mini_batch_size=128,
    ppo_epochs=1, optimizer_steps_per_outer=20, ppo_micro_batch_size=16,
    per_rank_actor_micro_batch=2, log_prob_micro_batch_size=32,
    per_rank_log_prob_micro_batch=4, learning_rate=1e-6,
    lr_warmup_steps_ratio=0.0, learning_rate_schedule='constant',
    gradient_clip_global_l2=1.0, target_actual_outer_updates=300,
    training_start_checkpoint=None, fresh_actor=True, fixed_original_reference=True,
    split_counts=dict(train=169028, validation=512, stage=64, final=51713),
    advantage_estimator='grpo', original_state_masking=True,
    objective_reduction='original two-row micro masked_mean',
    reference_kl_coefficient=0.001, reference_kl_type='low_var_kl',
    sampling_temperature=1.0, sampling_top_p=1.0, epsilon=0.2,
    compute_dtype='float32', rollout_dtype='float32', allow_tf32=False,
)
PHASE_BUDGET = dict(
    outer_count=1, initial_logical_update=0, initial_optimizer_writes=0,
    optimizer_writes_per_rank=20, monitors_per_rank=40,
    fixed_validation_split='validation', fixed_validation_count=512,
    fixed_validation_source_counts=dict(nq=256, hotpotqa=256),
    fixed_validation_before_and_after=True,
    required_health_metrics=['EM', 'F1', 'effective_executed_action_rate'],
    final_evaluation_in_this_phase=False,
)
INITIALIZATION = dict(
    actor='fresh_official_revision', reference='fixed_original_official_revision',
    logical_update=0, optimizer_writes=0, adam_state='empty',
    checkpoint=None, inherited_policy_run_id=None, inherited_optimizer_run_id=None,
    rng_and_sampler='fresh',
)


class PlanError(ValueError):
    """The plan or its frozen external binding is not the required contract."""


def _require(condition, message):
    if not condition:
        raise PlanError(message)


def canonical_bytes(document):
    """Stable JSON bytes; reject NaN/Infinity instead of hashing them."""
    try:
        return json.dumps(document, sort_keys=True, separators=(',', ':'),
                          ensure_ascii=False, allow_nan=False).encode('utf-8')
    except (ValueError, TypeError) as failure:
        raise PlanError('Not finite JSON: ' + str(failure)) from failure


def document_sha256(document):
    return hashlib.sha256(canonical_bytes(document)).hexdigest()


def _hash(value, name):
    _require(isinstance(value, str) and _SHA.fullmatch(value) is not None,
             'Invalid SHA256: ' + name)


def _run(value, name):
    _require(isinstance(value, str) and _RUN.fullmatch(value) is not None,
             'Invalid run identity: ' + name)


def _hash_map(value, name):
    _require(isinstance(value, Mapping) and bool(value), 'Empty hash map: ' + name)
    for path, digest in value.items():
        _require(isinstance(path, str) and path and not path.startswith('/') and
                 '..' not in path.split('/'), 'Unbound path in ' + name)
        _hash(digest, name + '/' + path)


def _identity(parent):
    _require(isinstance(parent, Mapping) and parent['profile'] == 'common8', 'Wrong parent profile')
    _run(parent['run_id'], 'parent production run')
    _require(type(parent['completed_updates']) is int and parent['completed_updates'] == 0,
             'Parent production run must be fresh0')
    _require(parent['model_revision'] == MODEL_REVISION, 'Wrong official model revision')
    _require(parent['initial_model'] == parent['reference_model'], 'Original reference differs')
    _require(isinstance(parent['source_commit'], str) and
             _COMMIT.fullmatch(parent['source_commit']) is not None, 'Invalid source commit')
    for name in ('source_sha256', 'initial_model_sha256', 'input_sha256'):
        _hash_map(parent[name], name)
    _require(parent['input_sha256']['manifest.json'] == COMMON_MANIFEST_SHA256,
             'Not the exact common data manifest')
    for name in ('train.parquet', 'validation.parquet', 'stage.parquet', 'final.parquet'):
        _hash(parent['input_sha256'][name], name)
    budget = parent['budget']
    _require(budget['profile'] == 'common8' and budget['model_id'] == MODEL_ID and
             budget['model_revision'] == MODEL_REVISION and
             budget['common_manifest_sha256'] == COMMON_MANIFEST_SHA256,
             'Wrong parent model/data budget')
    for name in MATHEMATICS:
        if name in budget:
            _require(canonical_bytes(budget[name]) == canonical_bytes(MATHEMATICS[name]),
                     'Parent mathematical budget differs: ' + name)
    # All approved batch/precision-independent budget fields are mandatory.
    for name in ('world_size', 'train_batch_size', 'n_agent', 'rollout_n',
                 'sequence_parallel_size', 'trajectory_count', 'ppo_mini_batch_size',
                 'ppo_epochs', 'optimizer_steps_per_outer', 'ppo_micro_batch_size',
                 'per_rank_actor_micro_batch', 'log_prob_micro_batch_size',
                 'per_rank_log_prob_micro_batch', 'learning_rate', 'lr_warmup_steps_ratio',
                 'learning_rate_schedule', 'gradient_clip_global_l2',
                 'target_actual_outer_updates', 'training_start_checkpoint',
                 'fresh_actor', 'fixed_original_reference', 'split_counts'):
        _require(name in budget, 'Missing parent mathematical budget: ' + name)
    return dict(model_id=MODEL_ID, model_revision=MODEL_REVISION,
                initial_model_sha256=copy.deepcopy(parent['initial_model_sha256']),
                source_commit=parent['source_commit'],
                source_sha256=copy.deepcopy(parent['source_sha256']),
                input_sha256=copy.deepcopy(parent['input_sha256']),
                common_manifest_sha256=COMMON_MANIFEST_SHA256,
                mathematical_config_sha256=document_sha256(MATHEMATICS))


def freeze_rule(document):
    """Freeze a caller-defined rule or provenance without choosing its values."""
    _require(isinstance(document, Mapping) and bool(document), 'Empty frozen document')
    return dict(document=copy.deepcopy(document), sha256=document_sha256(document))


def _frozen(value, name, role):
    _require(isinstance(value, Mapping) and set(value) == {'document', 'sha256'},
             'Wrong frozen document schema: ' + name)
    _hash(value['sha256'], name)
    _require(document_sha256(value['document']) == value['sha256'],
             'Frozen document hash differs: ' + name)
    document = value['document']
    _require(isinstance(document, Mapping) and document['role'] == role,
             'Wrong frozen document role: ' + name)
    _require(isinstance(document['provenance'], Mapping) and bool(document['provenance']),
             'Missing rule provenance: ' + name)
    _require(isinstance(document['criteria'], Mapping) and bool(document['criteria']),
             'Missing explicit rule criteria: ' + name)


def _strict_json(content):
    _require(isinstance(content, bytes), 'Frozen evidence must be original bytes')
    def pairs(items):
        value = {}
        for key, item in items:
            _require(key not in value, 'Duplicate JSON key: ' + key)
            value[key] = item
        return value
    try:
        return json.loads(content, object_pairs_hook=pairs,
                          parse_constant=lambda value: (_ for _ in ()).throw(
                              PlanError('Nonfinite JSON constant: ' + value)))
    except (UnicodeDecodeError, json.JSONDecodeError) as failure:
        raise PlanError('Invalid JSON: ' + str(failure)) from failure


def create_discovery_plan(parent, run_id, *, slurm_job_id, probability_proof_sha256,
                          resolved_config_sha256, initial_reward_fingerprint):
    """Freeze real artifact hashes after smoke/probes, before any Adam write.

    Actual gate, resolved recipe and probe verification remain the caller's
    responsibility. This reader binds their supplied hashes, not their health.
    """
    plan = dict(schema_version=SCHEMA_VERSION, role=DISCOVERY_ROLE, run_id=run_id,
                parent_production_run_id=parent['run_id'], slurm_job_id=slurm_job_id, identity=_identity(parent),
                resolved_config_sha256=resolved_config_sha256,
                initial_reward_fingerprint=copy.deepcopy(initial_reward_fingerprint),
                initialization=copy.deepcopy(INITIALIZATION),
                mathematics=copy.deepcopy(MATHEMATICS), phase_budget=copy.deepcopy(PHASE_BUDGET),
                initial_probability_gate=dict(tolerance=0.05, proof_relative_path='logs/model-smoke-initial/complete.json',
                                              proof_sha256=probability_proof_sha256,
                                              required_actual_optimizer_writes=0),
                candidate_thresholds=dict(dual_clip=3.0, epsilon=0.2,
                                          target_old_policy_kl=0.01, kl_log_bound=30.0,
                                          negative_tail_limit=None),
                threshold_policy=dict(negative_tail='observe_only', old_policy_kl='candidate_brake',
                                      numerical_finite='mandatory_reject', thresholds_are_candidates=True),
                health_rule=None, discovery=None)
    verify_plan(plan, parent, expected_plan_sha256=document_sha256(plan))
    return plan


def create_confirmation_plan(discovery_plan, parent, run_id, *, slurm_job_id,
                             probability_proof_sha256, resolved_config_sha256,
                             discovery_evidence_bytes, negative_tail_limit,
                             threshold_selection, health_rule):
    """Freeze the separate fresh confirmation branch and discovery provenance."""
    discovery_hash = document_sha256(discovery_plan)
    verify_plan(discovery_plan, parent, expected_plan_sha256=discovery_hash)
    _require(discovery_plan['role'] == DISCOVERY_ROLE, 'Not a discovery plan')
    plan = copy.deepcopy(discovery_plan)
    plan.update(role=CONFIRMATION_ROLE, run_id=run_id, slurm_job_id=slurm_job_id,
                resolved_config_sha256=resolved_config_sha256, health_rule=copy.deepcopy(health_rule))
    plan['initial_probability_gate']['proof_sha256'] = probability_proof_sha256
    plan['candidate_thresholds']['negative_tail_limit'] = negative_tail_limit
    plan['threshold_policy']['negative_tail'] = 'candidate_brake'
    plan['discovery'] = dict(run_id=discovery_plan['run_id'], plan_sha256=discovery_hash,
                             evidence_sha256=hashlib.sha256(discovery_evidence_bytes).hexdigest(),
                             threshold_selection=copy.deepcopy(threshold_selection))
    verify_plan(plan, parent, expected_plan_sha256=document_sha256(plan),
                discovery_plan=discovery_plan, discovery_evidence_bytes=discovery_evidence_bytes)
    return plan


def verify_plan(plan, parent, *, expected_plan_sha256, discovery_plan=None,
                discovery_evidence_bytes=None):
    """Strictly verify frozen roles/identities, never physical or healthy admission.

    ``expected_plan_sha256`` must be kept by the caller outside the plan file.
    A later actor binder must collectively agree on this digest and separately
    verify the actual probability/worker/finite/512-evaluation proofs. A valid
    declaration in JSON does not establish that a native job ran or was healthy.
    """
    try:
        return _verify_plan(plan, parent, expected_plan_sha256, discovery_plan,
                            discovery_evidence_bytes)
    except PlanError:
        raise
    except (KeyError, TypeError, OverflowError) as failure:
        raise PlanError('Malformed calibration plan: ' + str(failure)) from failure


def _verify_plan(plan, parent, expected_hash, discovery_plan, evidence_bytes):
    _hash(expected_hash, 'externally frozen plan')
    _require(isinstance(plan, Mapping) and document_sha256(plan) == expected_hash,
             'Externally frozen plan SHA differs')
    _require(set(plan) == {'schema_version', 'role', 'run_id', 'parent_production_run_id',
                          'slurm_job_id', 'resolved_config_sha256', 'initial_reward_fingerprint',
                          'identity', 'initialization', 'mathematics', 'phase_budget',
                          'initial_probability_gate', 'candidate_thresholds',
                          'threshold_policy', 'health_rule', 'discovery'}, 'Unexpected plan schema')
    _require(type(plan['schema_version']) is int and plan['schema_version'] == SCHEMA_VERSION,
             'Unsupported plan schema')
    _require(plan['role'] in (DISCOVERY_ROLE, CONFIRMATION_ROLE), 'Not an actual-policy calibration role')
    _run(plan['run_id'], 'child run')
    _require(isinstance(plan['slurm_job_id'], str) and _JOB.fullmatch(plan['slurm_job_id']) is not None,
             'Not a native Slurm job identity')
    _require(plan['parent_production_run_id'] == parent['run_id'] and
             plan['run_id'] != parent['run_id'], 'Child and production identities conflict')
    _require(canonical_bytes(plan['identity']) == canonical_bytes(_identity(parent)),
             'Cross-run source/model/data/mathematics identity differs')
    _hash(plan['resolved_config_sha256'], 'actual resolved child recipe artifact')
    reward = plan['initial_reward_fingerprint']
    _require(isinstance(reward, Mapping) and set(reward) == {'probe_round', 'encoder_sha256',
                 'head_sha256', 'reward_math_sha256'}, 'Unexpected initial reward fingerprint schema')
    _require(type(reward['probe_round']) is int and reward['probe_round'] == 0, 'Not initial probe round0')
    official = plan['identity']['initial_model_sha256']
    # This is the exact calibrated_version runtime encoder fingerprint. The
    # three excluded tokenizer/generation files remain bound by the full
    # official model identity above; do not invent extra live encoder entries.
    encoder_files = set(official) - {'vocab.json', 'merges.txt', 'generation_config.json'}
    _require(isinstance(reward['encoder_sha256'], Mapping) and
             set(reward['encoder_sha256']) == encoder_files and
             all(reward['encoder_sha256'][name] == official[name] for name in encoder_files),
             'Initial reward encoder differs from actual official encoder fingerprint')
    _hash_map(reward['head_sha256'], 'actual initial probe heads')
    _require(set(reward['head_sha256']) == {'commit_probe.pkl', 'boundary_probe.pkl'},
             'Both actual probe heads are required')
    _hash(reward['reward_math_sha256'], 'actual initial reward mathematics')
    for name, expected in (('initialization', INITIALIZATION), ('mathematics', MATHEMATICS),
                           ('phase_budget', PHASE_BUDGET)):
        _require(canonical_bytes(plan[name]) == canonical_bytes(expected), 'Changed contract: ' + name)
    gate = plan['initial_probability_gate']
    _require(set(gate) == {'tolerance', 'proof_relative_path', 'proof_sha256', 'required_actual_optimizer_writes'} and
             canonical_bytes(gate['tolerance']) == canonical_bytes(0.05) and
             type(gate['required_actual_optimizer_writes']) is int and
             gate['required_actual_optimizer_writes'] == 0, 'Original probability gate changed')
    _require(gate['proof_relative_path'] == 'logs/model-smoke-initial/complete.json',
             'Initial probability proof must belong to the fresh child namespace')
    _hash(gate['proof_sha256'], 'actual initial probability artifact')
    thresholds = plan['candidate_thresholds']
    _require(set(thresholds) == {'dual_clip', 'epsilon', 'target_old_policy_kl',
                                'kl_log_bound', 'negative_tail_limit'}, 'Wrong candidate threshold schema')
    for name, expected in dict(dual_clip=3.0, epsilon=0.2, target_old_policy_kl=0.01,
                               kl_log_bound=30.0).items():
        _require(type(thresholds[name]) in (int, float) and math.isfinite(thresholds[name]) and
                 thresholds[name] == expected, 'Candidate changed: ' + name)
    expected_policy = dict(negative_tail='observe_only' if plan['role'] == DISCOVERY_ROLE else 'candidate_brake',
                           old_policy_kl='candidate_brake', numerical_finite='mandatory_reject',
                           thresholds_are_candidates=True)
    _require(canonical_bytes(plan['threshold_policy']) == canonical_bytes(expected_policy),
             'Threshold decisions or candidate status changed')
    if plan['role'] == DISCOVERY_ROLE:
        _require(thresholds['negative_tail_limit'] is None and plan['discovery'] is None and
                 plan['health_rule'] is None,
                 'Discovery must observe tail without a fake threshold')
        _require(discovery_plan is None and evidence_bytes is None, 'Discovery inherits evidence unexpectedly')
    else:
        tail = thresholds['negative_tail_limit']
        _require(type(tail) in (int, float) and math.isfinite(tail) and 0 <= tail <= 1,
                 'Confirmation needs an explicit finite candidate tail limit')
        previous = plan['discovery']
        _require(set(previous) == {'run_id', 'plan_sha256', 'evidence_sha256', 'threshold_selection'},
                 'Wrong discovery provenance schema')
        _run(previous['run_id'], 'discovery run')
        _require(previous['run_id'] not in (parent['run_id'], plan['run_id']), 'Discovery is not a separate child')
        _hash(previous['plan_sha256'], 'discovery plan')
        _hash(previous['evidence_sha256'], 'discovery evidence')
        _require(discovery_plan is not None and evidence_bytes is not None, 'Missing frozen actual discovery artifacts')
        verify_plan(discovery_plan, parent, expected_plan_sha256=previous['plan_sha256'])
        _require(discovery_plan['role'] == DISCOVERY_ROLE and
                 discovery_plan['run_id'] == previous['run_id'], 'Wrong discovery role/run')
        _frozen(plan['health_rule'], 'health rule', 'policy_safety_health_rule')
        _require(plan['initial_reward_fingerprint'] == discovery_plan['initial_reward_fingerprint'],
                 'Actual initial encoder/probe/reward mathematics changed after discovery')
        _require(hashlib.sha256(evidence_bytes).hexdigest() == previous['evidence_sha256'],
                 'Actual discovery bytes changed')
        evidence = _strict_json(evidence_bytes)
        _require(evidence['role'] == 'policy_safety_calibration_discovery_evidence' and
                 evidence['origin'] == 'actual_native_policy_segment' and
                 evidence['run_id'] == previous['run_id'] and
                 evidence['parent_production_run_id'] == parent['run_id'] and
                 evidence['slurm_job_id'] == discovery_plan['slurm_job_id'] and
                 evidence['plan_sha256'] == previous['plan_sha256'] and
                 evidence['initial_reward_fingerprint'] == discovery_plan['initial_reward_fingerprint'] and
                 evidence['resolved_config_sha256'] == discovery_plan['resolved_config_sha256'] and
                 canonical_bytes(evidence['identity']) == canonical_bytes(plan['identity']),
                 'Discovery declaration role/run/model/data/source differs')
        _require(type(evidence['ranks']) is list and all(type(rank) is int for rank in evidence['ranks']) and
                 evidence['ranks'] == list(range(8)) and
                 type(evidence['completed_outer_updates']) is int and evidence['completed_outer_updates'] == 1 and
                 type(evidence['optimizer_writes_per_rank']) is int and evidence['optimizer_writes_per_rank'] == 20,
                 'Discovery declaration lacks complete native budget')
        selection = previous['threshold_selection']
        _frozen(selection, 'threshold selection', 'policy_safety_threshold_selection')
        choice = selection['document']['criteria']
        _require(choice['discovery_evidence_sha256'] == previous['evidence_sha256'] and
                 choice['health_rule_sha256'] == plan['health_rule']['sha256'] and
                 canonical_bytes(choice['candidate_thresholds']) == canonical_bytes(thresholds),
                 'Threshold selection not bound to frozen discovery/rule/candidates')
        _require(plan['slurm_job_id'] != discovery_plan['slurm_job_id'] and
                 gate['proof_sha256'] != discovery_plan['initial_probability_gate']['proof_sha256'],
                 'Confirmation reused discovery native job')
    return dict(schema_version=SCHEMA_VERSION, role='policy_safety_calibration_plan_structural_verification',
                execution_role=plan['role'], run_id=plan['run_id'],
                parent_production_run_id=parent['run_id'], slurm_job_id=plan['slurm_job_id'], plan_sha256=expected_hash,
                structural_verified=True, production_admission=False, healthy_checkpoint_accepted=False,
                physical_execution_authenticated=False, actual_recipe_verified=False,
                actual_probe_identity_verified=False, actual_grpo_updates_proven=0)


def load_plan(base, run, *, expected_slurm_job_id=None, discovery_plan=None,
              discovery_evidence_bytes=None):
    """Load the child plan and its independently frozen parent; return(plan,sha).

    ``run`` is the actual child manifest, also checked against its on-disk file.
    Its explicit calibration role and external plan/parent hashes bind identity;
    they never imply healthy admission. ``expected_slurm_job_id`` comes from the
    native allocation, independently of the manifest (default: SLURM_JOB_ID).
    The caller must still
    collectively verify all actual model/probe/worker/health prerequisites.
    """
    try:
        base = Path(base)
        if expected_slurm_job_id is None:
            expected_slurm_job_id = os.environ.get('SLURM_JOB_ID')
        _require(isinstance(expected_slurm_job_id, (str, int)) and
                 type(expected_slurm_job_id) is not bool and
                 _JOB.fullmatch(str(expected_slurm_job_id)) is not None,
                 'Missing actual native Slurm allocation identity')
        actual_run = _strict_json((base / 'logs/run-manifest.json').read_bytes())
        _require(canonical_bytes(run) == canonical_bytes(actual_run), 'Child manifest changed or stale')
        parent_bytes = (base / 'preparation/parent-run-manifest.json').read_bytes()
        _hash(run['policy_calibration_parent_manifest_sha256'], 'external parent manifest')
        _require(hashlib.sha256(parent_bytes).hexdigest() == run['policy_calibration_parent_manifest_sha256'],
                 'Frozen original parent manifest bytes differ')
        parent = _strict_json(parent_bytes)
        _require(run['parent_production_run_id'] == parent['run_id'] and run['run_id'] != parent['run_id'],
                 'Child manifest parent identity differs')
        _require(canonical_bytes(_identity(run)) == canonical_bytes(_identity(parent)),
                 'Actual child source/model/data/mathematics differs from production parent')
        plan = _strict_json((base / 'logs/policy-safety-calibration-plan.json').read_bytes())
        if plan.get('role') == CONFIRMATION_ROLE:
            if discovery_plan is None:
                discovery_plan = _strict_json((base / 'preparation/policy-safety-discovery-plan.json').read_bytes())
            if discovery_evidence_bytes is None:
                discovery_evidence_bytes = (base / 'preparation/policy-safety-discovery-evidence.json').read_bytes()
        digest = run['policy_calibration_plan_sha256']
        verify_plan(plan, parent, expected_plan_sha256=digest,
                    discovery_plan=discovery_plan, discovery_evidence_bytes=discovery_evidence_bytes)
        _require(plan['run_id'] == run['run_id'] and plan['role'] == run['policy_calibration_role'] and
                 plan['slurm_job_id'] == str(expected_slurm_job_id) == run['policy_calibration_slurm_job_id'],
                 'Actual child calibration role/run/native job differs')
        return plan, digest
    except PlanError:
        raise
    except (OSError, KeyError, TypeError, OverflowError) as failure:
        raise PlanError('Missing or malformed frozen calibration files: ' + str(failure)) from failure
