"""Prepare confirmation v3 for explicitly approved runtimea09a5530.

This external orchestration is deliberately outside the immutable runtime.
Default/check mode is local and cannot start a service or submit a job. Submit
requires complete independently reread discovery evidence and owned cleanup.
Any partial/unknown attempt keeps its claims: inspect it, never resubmit.
"""
import argparse
import base64
import hashlib
import inspect
import json
import math
import os
import shlex
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path('/Users/shaoshitong/Documents/Search-G1')
REPO, PREP = ROOT / 'repo', ROOT / 'preparation'
FROZEN_COMMIT = 'a09a5530fd16fa0c04c0aa157ee56462aab825f4'
FROZEN_METHOD_SHA256 = 'ef2c88215d74fcb638954c86482ffef85e18e744b10f57eb3f3609642126627f'
DISCOVERY_LAUNCH_SHA256 = 'e390ba1ab67d149d8e19b7b59551dcea5b3ed5b5c3092997fb7e48b976df8f1a'
DISCOVERY_ROLE = 'policy_safety_calibration_discovery'
CONFIRMATION_ROLE = 'policy_safety_calibration_confirmation'
CANONICAL_FILES = ('manifest.json', 'train.parquet', 'validation.parquet',
                   'stage.parquet', 'final.parquet', 'exclusions-audit.json')
SELECTION_NAME = 'policy-confirmation-v3-selection.json'
METHOD_NAME = 'policy-calibration-v3-selection-method.json'
CPU_NAME = 'behavior-consistency-target-cpu-v1-result.json'
DISCOVERY_LAUNCH_NAME = 'policy-discovery-v3-native-launch.json'
ZERO_OBSERVATION_NAME = 'policy-discovery-v3-native-observation-1791245435798697.json'
PREREGISTRATION_NAME = 'policy-calibration-v3-method-preregistration.json'
CPU_MODULES = ('reproduction.common8.test_policy_calibration_plan', 'reproduction.common8.test_freeze_calibration_plan', 'reproduction.common8.test_safety_calibration_evidence', 'reproduction.common8.test_calibration_evaluation', 'reproduction.common8.test_calibration_runtime', 'reproduction.common8.test_prepare_calibration_run', 'reproduction.common8.test_adopt_initial_probes', 'reproduction.common8.test_production_safety_admission', 'reproduction.common8.test_pipeline_calibration', 'search_g1.test_policy_safety', 'reproduction.common8.test_temporary_cache', 'reproduction.common8.test_sampling_capture', 'reproduction.common8.test_behavior_trace', 'reproduction.common8.test_behavior_probability', 'reproduction.common8.test_behavior_runtime', 'reproduction.common8.test_generation_behavior')
METHOD_PATH = CPU_PATH = SOURCE_CLEANUP_PATH = None


def configure_contract(commit, method_path, cpu_path, cleanup_path):
    """Explicit local inputs, never infer a future method or reuse old claims."""
    global METHOD_PATH, CPU_PATH, SOURCE_CLEANUP_PATH
    require(commit == FROZEN_COMMIT, 'Explicit new fulla09a5530 commit is required')
    proposed = tuple(map(Path, (method_path, cpu_path, cleanup_path)))
    method_bytes = proposed[0].read_bytes()
    require(sha(method_bytes) == FROZEN_METHOD_SHA256, 'Explicit root-frozen v3 method bytes are required')
    method = strict_json(method_bytes)
    require(method['source_commit'] == commit and method['before_completed_discovery_update'] is True,
            'Missing new prospective method for this source')
    METHOD_PATH, CPU_PATH, SOURCE_CLEANUP_PATH = proposed
    return FROZEN_METHOD_SHA256


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def sha(content):
    return hashlib.sha256(content).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def strict_json(content):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(content, object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON: ' + value)))


def selection_schema(selection, launch):
    """Validate explicit frozen choices, without choosing thresholds or health."""
    names = {'schema_version', 'role', 'selected_by', 'selection_provenance',
             'parent_base', 'discovery_base', 'parent_production_run_id',
             'discovery_run_id', 'discovery_slurm_job_id', 'source_commit',
             'source_map_sha256', 'scientific_identity_sha256',
             'discovery_manifest_sha256', 'discovery_plan_file_sha256',
             'discovery_plan_sha256', 'discovery_evidence_sha256',
             'selection_method_sha256', 'negative_tail_limit', 'health_rule', 'threshold_selection'}
    require(isinstance(selection, dict) and set(selection) == names, 'Wrong frozen selection schema')
    require(type(selection['schema_version']) is int and selection['schema_version'] == 1 and
            selection['role'] == 'policy_safety_confirmation_orchestration_selection' and
            selection['selected_by'] == 'root_explicit_selection' and
            isinstance(selection['selection_provenance'], dict) and bool(selection['selection_provenance']),
            'Missing explicit root selection provenance')
    require(selection['source_commit'] == launch['source_commit'] == FROZEN_COMMIT,
            'Confirmation must use the frozen parent runtime, not current HEAD')
    require(selection['parent_base'] == launch['parent_base'] and
            selection['discovery_base'] == launch['base'] and
            selection['discovery_slurm_job_id'] == launch['job_id'], 'Wrong discovery handle')
    require(selection['parent_base'] != selection['discovery_base'], 'Parent and discovery are not separate')
    for name in ('parent_base', 'discovery_base'):
        path = Path(selection[name])
        require(path.is_absolute() and '..' not in path.parts and path.as_posix() == selection[name] and
                selection[name].startswith('/mnt/data/shaoshitong/'), 'Unbound native namespace')
    for name in ('parent_production_run_id', 'discovery_run_id'):
        value = selection[name]
        require(isinstance(value, str) and len(value) == 32 and all(c in '0123456789abcdef' for c in value),
                'Invalid actual run UUID')
    require(selection['parent_production_run_id'] != selection['discovery_run_id'], 'Child reused parent UUID')
    for name in ('source_map_sha256', 'scientific_identity_sha256', 'discovery_manifest_sha256',
                 'discovery_plan_file_sha256', 'discovery_plan_sha256', 'discovery_evidence_sha256'):
        value = selection[name]
        require(isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value),
                'Invalid frozen hash: ' + name)
    require(selection['source_map_sha256'] == sha(canonical(launch['source_sha256'])), 'Frozen source map changed')
    require(selection['selection_method_sha256'] == FROZEN_METHOD_SHA256, 'Root pre-discovery selection method changed')
    tail = selection['negative_tail_limit']
    require(type(tail) in (int, float) and math.isfinite(tail) and 0 <= tail <= 1,
            'Explicit actual finite tail candidate is required')
    for name, role in (('health_rule', 'policy_safety_health_rule'),
                       ('threshold_selection', 'policy_safety_threshold_selection')):
        frozen = selection[name]
        require(isinstance(frozen, dict) and set(frozen) == {'document', 'sha256'} and
                sha(canonical(frozen['document'])) == frozen['sha256'] and
                frozen['document']['role'] == role and
                isinstance(frozen['document']['provenance'], dict) and bool(frozen['document']['provenance']),
                'Wrong frozen rule/provenance: ' + name)
    choice = selection['threshold_selection']['document']['criteria']
    require(choice['discovery_evidence_sha256'] == selection['discovery_evidence_sha256'] and
            choice['health_rule_sha256'] == selection['health_rule']['sha256'] and
            canonical(choice['candidate_thresholds']) == canonical(dict(
                dual_clip=3.0, epsilon=0.2, target_old_policy_kl=0.01,
                kl_log_bound=30.0, negative_tail_limit=tail)), 'Frozen candidates differ from selected evidence/rule')
    # Shape/method checks only. Actual paired measurements are checked on target.
    criteria = selection['health_rule']['document']['criteria']
    require(criteria['method'] == 'paired_fixed512_absolute_allowed_drop' and
            set(criteria) <= {'method', 'allowed_drop', 'minimum_before', 'minimum_after'}, 'Unknown health rule')
    supported = {'weighted_em', 'weighted_f1', 'valid_action_rate',
                 'executed_search_question_rate', 'executed_search_calls_per_question'}
    require({'weighted_em', 'weighted_f1', 'valid_action_rate'} <= set(criteria['allowed_drop']) <= supported,
            'Frozen EM/F1/valid-action health criteria are mandatory')
    for name in ('allowed_drop', 'minimum_before', 'minimum_after'):
        values = criteria.get(name, {})
        require(isinstance(values, dict) and set(values) <= supported, 'Unknown health criterion')
        for value in values.values():
            require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                    'Invalid explicit health value')
    return selection


def method_binding(content, selection):
    require(sha(content) == selection['selection_method_sha256'] == FROZEN_METHOD_SHA256,
            'Original root selection-method bytes changed')
    method = strict_json(content)
    require(method['role'] == 'root_explicit_policy_calibration_selection_method' and
            method['source_commit'] == selection['source_commit'] and
            method['parent_production_run_id'] == selection['parent_production_run_id'] and
            method['discovery_run_id'] == selection['discovery_run_id'] and
            method['before_completed_discovery_update'] is True and method['before_any_optimizer_write'] is True and
            method['sampling_distribution'] == dict(temperature=1.0, top_p=1.0, denominator='actual_selected_sampling_log_probs_after_all_token_context_and_0.05_gate') and
            method['negative_tail_limit'] is None and
            method['production_admission'] is False and method['healthy_checkpoint_accepted'] is False and
            canonical(method['health_rule_document']) == canonical(selection['health_rule']['document']),
            'Selection changed the root-frozen health rule or scientific identity')
    choice = selection['threshold_selection']['document']['criteria']['candidate_thresholds']
    require(canonical({key: choice[key] for key in method['fixed_candidates']}) ==
            canonical(method['fixed_candidates']), 'Selection changed the original candidate mathematics')
    criteria = method['health_rule_document']['criteria']
    require(set(criteria) == {'method', 'allowed_drop'} and
            criteria['method'] == 'paired_fixed512_absolute_allowed_drop' and
            canonical(criteria['allowed_drop']) == canonical(dict(weighted_em=0.0, weighted_f1=0.0, valid_action_rate=0.0)),
            'The original explicit zero-allowed-drop rule cannot be relaxed')
    tail = method['negative_tail_selection']
    require(type(tail['required_global_observations']) is int and tail['required_global_observations'] == 40 and
            tail['arbitrary_margin_or_floor'] is False and
            type(tail['fraction_with_zero_negative_mass']) in (int, float) and tail['fraction_with_zero_negative_mass'] == 0.0 and
            method['fixed_candidates'] == dict(dual_clip=3.0, epsilon=0.2, target_old_policy_kl=0.01, kl_log_bound=30.0) and
            type(method['frozen_unix']) in (int, float) and math.isfinite(method['frozen_unix']) and method['frozen_unix'] > 0,
            'The original prospective40 exact-max method/candidates cannot be changed')
    binding = method['prospective_binding']
    require(binding['native_job_id'] == selection['discovery_slurm_job_id'] == '3283' and
            binding['discovery_base'] == selection['discovery_base'] == '/mnt/data/shaoshitong/g1d-2541e8f3' and
            binding['parent_base'] == selection['parent_base'] == '/mnt/data/shaoshitong/g1p-2541e8f3' and
            selection['parent_production_run_id'] == '32d57da04a8c4115b7cd19c47209eefb' and
            selection['discovery_run_id'] == 'd08276e6875d416c8d6f63772c88ff96' and
            binding['parent_actual_unstarted_verified'] is True,
            'Selected identity differs from the actual prospective3283 observation')
    return method


def prospective_binding(method, launch_bytes, observation_bytes):
    """Read original zero-update observation, without claiming later health.

    The whole original method has an explicit root-approved SHA. Its launch
    and observation bytes must also still match; later manifest operational
    metadata is checked by stable identity, not by this initial byte snapshot.
    """
    binding = method['prospective_binding']
    require(sha(launch_bytes) == DISCOVERY_LAUNCH_SHA256 and
            sha(observation_bytes) == binding['zero_update_observation_sha256'],
            'Prospective launch or zero-update observation bytes changed')
    launch, observed = strict_json(launch_bytes), strict_json(observation_bytes)
    require(launch['source_commit'] == method['source_commit'] == FROZEN_COMMIT and
            launch['job_id'] == observed['job_id'] == binding['native_job_id'] and
            launch['base'] == observed['base'] == binding['discovery_base'] and
            launch['parent_base'] == observed['parent_base'] == binding['parent_base'] and
            type(launch['returncode']) is int and launch['returncode'] == 0 and
            type(launch['actual_production_updates']) is int and launch['actual_production_updates'] == 0 and
            type(observed['job_returncode']) is int and observed['job_returncode'] == 0 and
            type(observed['actual_production_updates']) is int and observed['actual_production_updates'] == 0 and
            observed['actual_updates'] == [] and observed['source_hashes_match'] is True and
            observed['child_manifest_slurm_job_id'] == binding['native_job_id'] and
            observed['child_manifest_sha256'] == binding['child_manifest_sha256'] and
            observed['parent_actual_unstarted_verified'] is True and observed['child_actual_unstarted_verified'] is True,
            'Not the authenticated zero-update native discovery observation')
    child, parent = observed['run-manifest.json'], observed['parent_run_manifest']
    require(child['run_id'] == method['discovery_run_id'] and
            child['parent_production_run_id'] == method['parent_production_run_id'] == parent['run_id'] and
            child['source_commit'] == parent['source_commit'] == FROZEN_COMMIT and
            child['policy_calibration_role'] == DISCOVERY_ROLE and
            type(child['completed_updates']) is int and child['completed_updates'] == 0 and
            type(parent['completed_updates']) is int and parent['completed_updates'] == 0 and
            parent['manifest_sha256'] == binding['parent_manifest_sha256'],
            'Prospective parent/child scientific identity differs')
    empty = observed['parent_actual_unstarted']
    require(empty == dict(training_claim_exists=False, optimizer_evidence_paths=[], train_log_exists=False,
                          checkpoint_entries=[], actual_updates_exists=False, segment_exists=False, segment_complete_exists=False) and
            type(observed['observation_finished_unix']) in (int, float) and
            math.isfinite(observed['observation_finished_unix']) and
            observed['child_actual_unstarted'] == empty and
            binding['child_actual_unstarted_verified'] is True and
            binding['preregistered_unix'] <= observed['checked_unix'] <= observed['observation_finished_unix'] <= method['frozen_unix'],
            'Parent was not freshly unstarted when this method was prospectively frozen')
    return launch


def preregistration_binding(content, method, launch):
    binding = method['prospective_binding']
    require(sha(content) == binding['preregistration_sha256'], 'Root preregistration original bytes changed')
    prior = strict_json(content)
    require(prior['role'] == 'root_policy_calibration_method_preregistration_before_native_submission' and
            prior['source_commit'] == FROZEN_COMMIT and
            type(prior['created_unix']) in (int, float) and math.isfinite(prior['created_unix']) and
            prior['created_unix'] == binding['preregistered_unix'] and
            prior['created_unix'] <= launch['submitted_unix'] and
            type(prior['actual_discovery_observations_for_this_source']) is int and
            prior['actual_discovery_observations_for_this_source'] == 0 and
            prior['production_admission'] is False and prior['healthy_checkpoint_accepted'] is False and
            prior['target_cpu_proof_sha256'] == launch['target_cpu_proof_sha256'],
            'Not the prospective pre-submit mathematical preregistration')
    for key in ('health_rule_document', 'negative_tail_selection', 'fixed_candidates', 'negative_tail_limit', 'sampling_distribution'):
        require(canonical(prior[key]) == canonical(method[key]), 'Preregistered mathematics changed: ' + key)
    return prior


def measured_selection(selection, segment):
    """Check the frozen empirical-max method; never generate a candidate."""
    structural = segment['structural']
    require(structural['execution_role'] == DISCOVERY_ROLE and
            structural['run_id'] == selection['discovery_run_id'] and
            structural['calibration_sha256'] == selection['discovery_plan_sha256'] and
            type(structural['outer_count']) is int and structural['outer_count'] == 1 and
            type(structural['optimizer_writes_per_rank']) is int and structural['optimizer_writes_per_rank'] == 20 and
            structural['enforcement']['negative_tail'] == 'observe_only', 'Not the complete observe-only discovery')
    observations = structural['global_monitors']
    require(type(observations) is list and len(observations) == 40, 'Need exactly 40 coalesced global observations')
    tails = []
    for ordinal, observation in enumerate(observations):
        mini, stage = ordinal // 2, ('before_minibatch', 'after_minibatch')[ordinal % 2]
        expected = dict(logical_update=1, minibatch_index=mini, prior_optimizer_writes=mini,
                        behavior_probability_gate_sha256=segment['behavior_probability_gate']['proof_sha256'])
        require(canonical(observation['context']) == canonical(expected) and
                observation['stage'] == stage and observation['run_id'] == selection['discovery_run_id'] and
                observation['calibration_sha256'] == selection['discovery_plan_sha256'] and
                type(observation['optimizer_writes']) is int and observation['optimizer_writes'] == mini + ordinal % 2 and
                observation['execution_role'] == DISCOVERY_ROLE and
                observation['enforcement']['negative_tail'] == 'observe_only' and
                observation['allowed'] is True and observation['reasons'] == [], 'Incomplete/reordered global observation')
        value = observation['negative_tail_weight_fraction']
        require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
                'Actual weighted negative-tail fraction is invalid')
        tails.append(value)
    require(selection['negative_tail_limit'] == max(tails), 'Selected tail differs from exact actual maximum; no margin/floor')
    return max(tails)


def save(path, value):
    """Durable exclusive claim; a partial file deliberately blocks replay."""
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    descriptor = os.open(str(Path(path).parent), os.O_DIRECTORY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


def frozen_sources(launch, repo=REPO):
    """Read commit blobs and exact known map, never mutable HEAD files."""
    require(launch['source_commit'] == FROZEN_COMMIT, 'Wrong immutable source commit')
    # Reuse the independently reviewed exact full-tree/blob reader, while the
    # original failed61f6 identity remains immutable in that module.
    sys.path.insert(0, str(PREP))
    from submit_policy_discovery_v3_native import committed_sources
    return committed_sources(FROZEN_COMMIT, launch['source_sha256'], repo)


def cpu_contract(proof, launch):
    summary = strict_json(proof['stdout'])
    require(type(proof['returncode']) is int and proof['returncode'] == 0 and
            type(proof['shell_syntax_returncode']) is int and proof['shell_syntax_returncode'] == 0 and
            summary['successful'] is True and summary['cuda_available'] is False and
            summary['skipped'] == [] and type(summary['failures']) is int and summary['failures'] == 0 and
            type(summary['errors']) is int and summary['errors'] == 0 and
            type(summary['tests_run']) is int and summary['tests_run'] == 222 and
            tuple(proof['modules']) == CPU_MODULES and
            len(proof['source_sha256']) == 353 and proof['source_sha256'] == launch['source_sha256'],
            'The actual222/16-module/353-source immutable CPU proof is required')


def behavior_links(discovery, run, segment, updates):
    """Original public receipt/ledger hashes only; no raw/GPU reauthentication."""
    from reproduction.common8.behavior_runtime import read_gate_receipt, verify_write_receipt_links
    require(len(updates) == 1, 'Need the complete first actual outer')
    gate = read_gate_receipt(discovery, run, 1, update_record=updates[0])
    require(canonical(gate) == canonical(segment['behavior_probability_gate']),
            'Segment does not bind the original true-sampling .05 receipt')
    verify_write_receipt_links(discovery, {'1': gate}, 1, 1)
    return gate


def source_precheck(selection, launch, method_bytes, launch_bytes, observation_bytes):
    """Small-memory historical check; defer tensor audit to native confirmation.

    Only JSON/source bytes, the eight monitor ledgers and fixed512 arithmetic
    are read here. In particular this does not call verify_child/_checkpoint,
    load model tensors or declare a healthy checkpoint/production admission.
    """
    selection_schema(selection, launch)
    method = method_binding(method_bytes, selection)
    require(prospective_binding(method, launch_bytes, observation_bytes) == launch,
            'Source differs from its actual prospectively bound launch')
    parent_base, discovery = Path(selection['parent_base']), Path(selection['discovery_base'])
    require(parent_base.resolve() == parent_base and discovery.resolve() == discovery,
            'Namespace symlink rejected')
    sys.path.insert(0, str(parent_base / 'repo'))
    from reproduction.common8.policy_calibration_plan import _identity, load_plan, document_sha256
    from reproduction.common8.freeze_calibration_plan import _unstarted, _mathematics
    from reproduction.common8.production_safety_admission import _paired_evaluations, _cleanup, evaluate_health
    from reproduction.common8.safety_calibration_evidence import verify_segment
    from reproduction.common8.source_files import runtime_files
    parent = strict_json((parent_base / 'logs/run-manifest.json').read_bytes())
    require(parent.get('policy_calibration_role', 'production') == 'production' and
            parent['run_id'] == selection['parent_production_run_id'] and parent['source_commit'] == FROZEN_COMMIT,
            'Wrong original fresh parent/source')
    _unstarted(parent_base, parent)
    require(sha(canonical(_identity(parent))) == selection['scientific_identity_sha256'],
            'Parent science identity changed')
    for namespace in (parent_base, discovery):
        mapping = {str(path.relative_to(namespace / 'repo')): sha(path.read_bytes())
                   for path in runtime_files(namespace / 'repo')}
        require(mapping == launch['source_sha256'], 'Actual frozen namespace source changed')
    run = strict_json((discovery / 'logs/run-manifest.json').read_bytes())
    require(run['run_id'] == selection['discovery_run_id'] and
            run['parent_production_run_id'] == parent['run_id'] and run['policy_calibration_role'] == DISCOVERY_ROLE and
            canonical(_identity(run)) == canonical(_identity(parent)), 'Actual discovery scientific identity differs')
    plan, plan_sha = load_plan(discovery, run, expected_slurm_job_id=selection['discovery_slurm_job_id'])
    for name, relative in (('discovery_manifest_sha256', 'logs/run-manifest.json'),
                           ('discovery_plan_file_sha256', 'logs/policy-safety-calibration-plan.json'),
                           ('discovery_evidence_sha256', 'logs/policy-calibration-segment.json')):
        require(sha((discovery / relative).read_bytes()) == selection[name], 'Frozen source bytes changed: ' + name)
    require(plan_sha == selection['discovery_plan_sha256'], 'Actual canonical plan changed')
    config = strict_json((discovery / 'logs/resolved-training-config.json').read_bytes())
    require(config['run_id'] == run['run_id'] and config['slurm_job_id'] == plan['slurm_job_id'] and
            document_sha256(config['resolved_config']) == config['resolved_config_sha256'] == plan['resolved_config_sha256'],
            'Actual resolved mathematics bytes differ')
    _mathematics(config['resolved_config'], discovery, run)
    segment = strict_json((discovery / 'logs/policy-calibration-segment.json').read_bytes())
    require(segment['origin'] == 'actual_native_policy_segment' and segment['role'] == DISCOVERY_ROLE + '_evidence' and
            segment['run_id'] == run['run_id'] and segment['parent_production_run_id'] == parent['run_id'] and
            segment['slurm_job_id'] == plan['slurm_job_id'] and segment['plan_sha256'] == plan_sha and
            type(segment['completed_outer_updates']) is int and segment['completed_outer_updates'] == 1 and
            type(segment['optimizer_writes_per_rank']) is int and segment['optimizer_writes_per_rank'] == 20 and
            segment['production_admission'] is False and segment['healthy_checkpoint_accepted'] is False,
            'Not a complete actual discovery measurement segment')
    for key in ('identity', 'initial_reward_fingerprint', 'resolved_config_sha256'):
        require(canonical(segment[key]) == canonical(plan[key]), 'Segment frozen identity differs: ' + key)
    logs = {}
    for rank in range(8):
        content = (discovery / f'logs/optimizer-rank-{rank}.jsonl').read_bytes()
        require(sha(content) == segment['worker_log_sha256'][str(rank)], 'Actual monitor ledger bytes changed')
        logs[rank] = [strict_json(line) for line in content.splitlines()]
    structural = verify_segment(logs, dict(schema_version=1, role=DISCOVERY_ROLE, run_id=run['run_id'],
                                          calibration_sha256=plan_sha, outer_count=1, initial_optimizer_writes=0))
    require(canonical(structural) == canonical(segment['structural']), 'Actual monitor/write reconstruction differs')
    measured_selection(selection, segment)
    receipts = {}
    _paired_evaluations(discovery, run, plan, plan_sha, segment, receipts, 'source/')
    updates_path = discovery / 'logs/actual-updates.jsonl'
    updates = [strict_json(line) for line in updates_path.read_bytes().splitlines()]
    require(updates == segment['actual_updates'] and len(updates) == 1 and
            updates[0]['run_id'] == run['run_id'] and type(updates[0]['step']) is int and updates[0]['step'] == 1 and
            updates[0]['probe_version'] == run['initial_probe_version'] and
            type(updates[0]['metrics']['actor/optimizer_steps_this_update']) in (int, float) and
            updates[0]['metrics']['actor/optimizer_steps_this_update'] == 20 and
            all(type(value) in (int, float) and math.isfinite(value) for value in updates[0]['metrics'].values()) and
            method['frozen_unix'] <= updates_path.stat().st_mtime,
            'Missing complete actual outer or prospective timing')
    behavior_links(discovery, run, segment, updates)
    # This reads only the scalar producer receipt. Its finite/delta claims must
    # still be independently checked against all actual tensors in Slurm.
    checkpoint = Path(run['checkpoint_dir']) / 'actor/global_step_1'
    complete = strict_json((checkpoint / 'complete.json').read_bytes())
    require(complete == segment['checkpoint'] and complete['run_id'] == run['run_id'] and
            type(complete['step']) is int and complete['step'] == 1 and
            complete['finite_all_saved_tensors'] is True and bool(segment['changed_parameters']),
            'Missing producer finite/fresh/delta receipt; independent tensor audit is still pending')
    evaluate_health(selection['health_rule'], segment, segment)
    _cleanup(discovery, run, receipts, 'source/')
    marker = ('SEARCH_G1_BASE=' + str(discovery)).encode()
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit(): continue
        try:
            if proc.stat().st_uid == os.getuid() and marker in (proc / 'environ').read_bytes().split(b'\0'):
                require((proc / 'stat').read_text().rsplit(')', 1)[1].split()[0] == 'Z',
                        'Discovery namespace remains active: ' + proc.name)
        except (FileNotFoundError, ProcessLookupError): pass
    command = subprocess.run(['scontrol', 'show', 'job', '-o', selection['discovery_slurm_job_id']],
                             capture_output=True, text=True, timeout=15)
    from reproduction.common8.native_allocation import fields
    actual = fields(command.stdout)
    require(command.returncode == 0 and actual['JobId'] == selection['discovery_slurm_job_id'] and
            actual['JobState'] == 'COMPLETED' and actual['NodeList'] == 'h100-8' and
            actual['WorkDir'] == str(discovery / 'repo') and
            actual['Command'] == str(discovery / 'repo/reproduction/common8/launch_calibration_native.sh'),
            'Actual scoped discovery native job has not completed')
    return dict(role='confirmation_source_small_memory_precheck', run_id=run['run_id'],
                parent_production_run_id=parent['run_id'], source_commit=parent['source_commit'],
                discovery_evidence_sha256=selection['discovery_evidence_sha256'],
                actual_production_updates=0, full_checkpoint_verification_pending=True,
                production_admission=False, healthy_checkpoint_accepted=False,
                physical_execution_authenticated=False,
                scope='Source/JSON/math/40 ledger/fixed512 arithmetic/cleanup only; full tensor audit is deferred to native allocation')


def target_evidence(selection, launch, method_bytes, launch_bytes, observation_bytes):
    """Native-only full source audit, before any confirmation model or Adam.

    The real verifier reconstructs 40 global reports, 20 writes/rank, finite
    fresh1/Adam/delta, rank .05/configuration and exact fixed512 arithmetic.
    The selected health rule is then applied to before/after, twice to the
    same completed discovery. This does NOT compare a measurement to itself.
    """
    selection_schema(selection, launch)
    method = method_binding(method_bytes, selection)
    require(prospective_binding(method, launch_bytes, observation_bytes) == launch,
            'Target discovery differs from the original prospectively bound launch')
    parent_base, discovery = Path(selection['parent_base']), Path(selection['discovery_base'])
    require(parent_base.resolve() == parent_base and discovery.resolve() == discovery, 'Namespace symlink rejected')
    sys.path.insert(0, str(parent_base / 'repo'))
    from reproduction.common8.prepare_calibration_run import native_preparation
    require(native_preparation() != selection['discovery_slurm_job_id'],
            'Full tensor audit requires the fresh independent native eight-H100/512GiB allocation')
    from reproduction.common8.policy_calibration_plan import _identity, DISCOVERY_ROLE
    from reproduction.common8.freeze_calibration_plan import _unstarted
    from reproduction.common8.production_safety_admission import verify_child, evaluate_health
    from reproduction.common8.source_files import runtime_files
    parent = strict_json((parent_base / 'logs/run-manifest.json').read_bytes())
    require(parent.get('policy_calibration_role', 'production') == 'production' and
            parent['run_id'] == selection['parent_production_run_id'] and
            parent['source_commit'] == FROZEN_COMMIT, 'Wrong original fresh parent/source')
    _unstarted(parent_base, parent)
    require(sha(canonical(_identity(parent))) == selection['scientific_identity_sha256'], 'Parent science identity changed')
    for namespace in (parent_base, discovery):
        mapping = {str(path.relative_to(namespace / 'repo')): sha(path.read_bytes())
                   for path in runtime_files(namespace / 'repo')}
        require(mapping == launch['source_sha256'], 'Actual frozen namespace source changed')
    receipts = {}
    run, plan, segment = verify_child(discovery, parent, DISCOVERY_ROLE, receipts, 'discovery/')
    require(run['run_id'] == selection['discovery_run_id'] and
            plan['slurm_job_id'] == selection['discovery_slurm_job_id'], 'Wrong actual complete child')
    for name, relative in (('discovery_manifest_sha256', 'logs/run-manifest.json'),
                           ('discovery_plan_file_sha256', 'logs/policy-safety-calibration-plan.json'),
                           ('discovery_evidence_sha256', 'logs/policy-calibration-segment.json')):
        require(sha((discovery / relative).read_bytes()) == selection[name], 'Frozen actual artifact changed: ' + name)
    require(sha(canonical(plan)) == selection['discovery_plan_sha256'], 'Canonical plan hash changed')
    updates = [strict_json(line) for line in (discovery / 'logs/actual-updates.jsonl').read_bytes().splitlines()]
    behavior_links(discovery, run, segment, updates)
    measured_selection(selection, segment)
    # The method existed before the first completed-update record. This is a
    # prospective timing check, not a confidence/population-risk claim.
    require(method['frozen_unix'] <= (discovery / 'logs/actual-updates.jsonl').stat().st_mtime,
            'Method was not frozen before the actual discovery outer completed')
    health = evaluate_health(selection['health_rule'], segment, segment)
    require(health['accepted'] is True, 'Discovery failed the explicit frozen health rule')
    # Source verification has checked its owned ledger; independently census the
    # actual namespace too, so an unledgered owned child cannot remain active.
    marker = ('SEARCH_G1_BASE=' + str(discovery)).encode()
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit(): continue
        try:
            if proc.stat().st_uid != os.getuid(): continue
            if marker in (proc / 'environ').read_bytes().split(b'\0'):
                state = (proc / 'stat').read_text().rsplit(')', 1)[1].split()[0]
                require(state == 'Z', 'Discovery namespace remains active: ' + proc.name)
        except (FileNotFoundError, ProcessLookupError): pass
    command = subprocess.run(['scontrol', 'show', 'job', '-o', selection['discovery_slurm_job_id']],
                             capture_output=True, text=True, timeout=15)
    from reproduction.common8.native_allocation import fields
    actual = fields(command.stdout)
    require(command.returncode == 0 and actual['JobId'] == selection['discovery_slurm_job_id'] and
            actual['JobState'] == 'COMPLETED', 'Actual discovery native job has not completed')
    return dict(role='confirmation_source_prerequisite_verification', run_id=run['run_id'],
                parent_production_run_id=parent['run_id'], source_commit=parent['source_commit'],
                discovery_evidence_sha256=selection['discovery_evidence_sha256'],
            frozen_health_rule_satisfied=True, health_rule_sha256=health['rule_sha256'],
            discovery_owned_cleanup_verified=True, actual_production_updates=0,
            full_checkpoint_verification_pending=False,
                production_admission=False, healthy_checkpoint_accepted=False)


def remote(host, code, payload, timeout=90, interpreter='python3'):
    """Unknown remote outcome is never converted into permission to retry."""
    compile(code, '<remote>', 'exec')
    sys.path.insert(0, '/Users/shaoshitong/Documents/remote_codex_control')
    import control
    result = subprocess.run(control.route(host, shlex.quote(interpreter) + ' -c ' + shlex.quote(code)),
                            input=json.dumps(payload, allow_nan=False), capture_output=True,
                            text=True, timeout=timeout)
    require(result.returncode == 0, 'Observe the same saved intent/remote claims; do not retry: ' + result.stderr[-2500:])
    return strict_json(result.stdout)


def verification_code():
    return ('import base64,hashlib,json,math,os,subprocess,sys\nfrom pathlib import Path\n' +
            '\n'.join(inspect.getsource(function) for function in
                      (canonical, sha, require, strict_json, selection_schema, method_binding,
                       prospective_binding, measured_selection, behavior_links, source_precheck)) +
            '\nFROZEN_COMMIT=' + repr(FROZEN_COMMIT) + '\nFROZEN_METHOD_SHA256=' + repr(FROZEN_METHOD_SHA256) +
            '\nDISCOVERY_LAUNCH_SHA256=' + repr(DISCOVERY_LAUNCH_SHA256) +
            '\nDISCOVERY_ROLE=' + repr(DISCOVERY_ROLE) +
            '\np=json.load(sys.stdin);print(json.dumps(source_precheck(p["selection"],p["launch"],base64.b64decode(p["method_bytes"]),base64.b64decode(p["launch_bytes"]),base64.b64decode(p["observation_bytes"]))))\n')


def native_wrapper(files):
    """Preserve frozen native environment/horizon; insert only identity binding."""
    source = files['reproduction/common8/launch_calibration_native.sh'].decode()
    needle = 'exec python -m reproduction.common8.run_pipeline'
    require(source.count(needle) == 1, 'Frozen launcher changed')
    return source.replace(needle, 'python "$BASE/preparation/submit_policy_confirmation_v3_native.py" --native-bind --base "$BASE"\n' + needle).encode()


def ray_socket_budget(base):
    """Conservative fresh-session path for actual Ray's plasma_store basename.

    The recorded native error proves the date/time/microsecond/session PID and
    basename form. Ten PID digits are conservatively used before any service.
    This calculation is not a claim that Ray has started successfully.
    """
    path = (str(base) + '/ray/session_9999-12-31_23-59-59_999999_9999999999/sockets/plasma_store')
    count = len(path.encode('utf-8'))
    require(count <= 107, 'Conservative Ray AF_UNIX path exceeds the installed 107-byte limit')
    return dict(worst_plasma_socket_path=path, worst_plasma_socket_bytes=count,
                af_unix_limit_bytes=107, pid_digits=10, actual_ray_started=False)


def native_bind(base):
    """Bind frozen choices to the newly prepared actual child UUID, before Adam."""
    base = Path(base).resolve()
    selection_bytes = (base / 'preparation' / SELECTION_NAME).read_bytes()
    package = strict_json((base / 'preparation/source-package-intent.json').read_bytes())
    require(package['source_commit'] == FROZEN_COMMIT, 'Sealed newsource contract differs')
    require(package['selection_method_sha256'] == FROZEN_METHOD_SHA256,
            'Sealed package cannot replace the explicit root-frozen method')
    require(sha(selection_bytes) == package['selection_sha256'], 'Selection bytes changed after reservation')
    selection = strict_json(selection_bytes); launch = package['discovery_launch']
    method_bytes = (base / 'preparation' / METHOD_NAME).read_bytes()
    method_binding(method_bytes, selection)
    cpu_bytes = (base / 'preparation' / CPU_NAME).read_bytes()
    require(sha(cpu_bytes) == package['target_cpu_proof_sha256'] == launch['target_cpu_proof_sha256'],
            'Actual222 proof bytes changed')
    cpu = strict_json(cpu_bytes); cpu_contract(cpu, launch); summary = strict_json(cpu['stdout'])
    require(cpu['returncode'] == cpu['shell_syntax_returncode'] == 0 and
            summary['successful'] is True and summary['skipped'] == [] and summary['cuda_available'] is False and
            type(summary['tests_run']) is int and summary['tests_run'] == 222 and
            cpu['source_sha256'] == launch['source_sha256'], 'Not the actual newsource222 proof')
    preregistration_binding((base / 'preparation' / PREREGISTRATION_NAME).read_bytes(),
                            strict_json(method_bytes), launch)
    launch_bytes = (base / 'preparation' / DISCOVERY_LAUNCH_NAME).read_bytes()
    observation_bytes = (base / 'preparation' / ZERO_OBSERVATION_NAME).read_bytes()
    from reproduction.common8.prepare_calibration_run import native_preparation
    from reproduction.common8.policy_calibration_plan import _identity
    from reproduction.common8.freeze_calibration_plan import _unstarted
    job_id = native_preparation()
    # This allocation is real and independent of the historical discovery.
    # The heavy all-tensor source audit cannot execute on the SSH control path.
    require(job_id != selection['discovery_slurm_job_id'], 'Confirmation reused discovery native allocation')
    target_evidence(selection, launch, method_bytes, launch_bytes, observation_bytes)
    run = strict_json((base / 'logs/run-manifest.json').read_bytes())
    require(str(base) == package['base'] and run['policy_calibration_role'] == CONFIRMATION_ROLE and
            run['policy_calibration_slurm_job_id'] == job_id and job_id != selection['discovery_slurm_job_id'] and
            run['run_id'] not in (selection['parent_production_run_id'], selection['discovery_run_id']) and
            run['parent_production_run_id'] == selection['parent_production_run_id'] and
            run['policy_calibration_discovery_base'] == selection['discovery_base'] and
            run['initial_probe_version'] is None, 'Not a fresh independent actual confirmation')
    _unstarted(base, run)
    require(sha(canonical(_identity(run))) == selection['scientific_identity_sha256'], 'Child scientific identity differs')
    # Bound external code is not silently included in the immutable runtime map.
    require(sha(Path(__file__).read_bytes()) == package['external_orchestrator_sha256'], 'External binder changed')
    require(sha((base / 'preparation/launch-confirmation-native.sh').read_bytes()) ==
            package['external_wrapper_sha256'], 'External launcher changed')
    claim = base / 'preparation/confirmation-bind-claim.json'
    save(claim, dict(run_id=run['run_id'], slurm_job_id=job_id, selection_sha256=sha(selection_bytes),
                     started_unix=time.time(), scope='Frozen identity binding only; no health or model admission'))
    discovery = Path(selection['discovery_base'])
    for relative, name, expected in (
        ('logs/policy-safety-calibration-plan.json', 'policy-safety-discovery-plan.json', selection['discovery_plan_file_sha256']),
        ('logs/policy-calibration-segment.json', 'policy-safety-discovery-evidence.json', selection['discovery_evidence_sha256'])):
        content = (discovery / relative).read_bytes(); require(sha(content) == expected, 'Original source bytes changed')
        with (base / 'preparation' / name).open('xb') as stream:
            stream.write(content); stream.flush(); os.fsync(stream.fileno())
    candidate = dict(schema_version=1, role='policy_safety_confirmation_candidates',
                     parent_production_run_id=selection['parent_production_run_id'],
                     confirmation_run_id=run['run_id'], discovery_run_id=selection['discovery_run_id'],
                     discovery_evidence_sha256=selection['discovery_evidence_sha256'],
                     negative_tail_limit=selection['negative_tail_limit'],
                     health_rule=selection['health_rule'], threshold_selection=selection['threshold_selection'])
    save(base / 'preparation/policy-safety-confirmation-candidates.json', candidate)
    result = dict(role='confirmation_frozen_candidate_binding', run_id=run['run_id'], slurm_job_id=job_id,
                  selection_sha256=sha(selection_bytes), actual_production_updates=0,
                  production_admission=False, healthy_checkpoint_accepted=False,
                  child_model_probe_http_admission_required=True)
    save(base / 'preparation/confirmation-bind-complete.json', result)
    return result


PREPARE_CODE = r'''
import base64,hashlib,inspect,json,os,shutil,subprocess,sys,time
from pathlib import Path
p=json.load(sys.stdin);b=Path(p['base']);parent=Path(p['parent_base'])
assert not b.exists();b.mkdir(parents=True,exist_ok=False)
for name in ('repo','data','preparation','logs','tmp','ray','artifacts','cache/triton','cache/inductor','cache/hf'):(b/name).mkdir(parents=True,exist_ok=True)
for name in ('env','bootstrap-assets','bootstrap-tools','assets','retriever-gpu-lib'):
 source=parent/name;assert source.is_dir();(b/name).symlink_to(source.resolve(),target_is_directory=True)
for name,data in p['files'].items():
 relative=Path(name);assert not relative.is_absolute() and '..' not in relative.parts
 target=b/'repo'/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(base64.b64decode(data))
 assert hashlib.sha256(target.read_bytes()).hexdigest()==p['source_sha256'][name]
for name in ('manifest.json','train.parquet','validation.parquet','stage.parquet','final.parquet','exclusions-audit.json'):
 source=parent/'data'/name;target=b/'data'/name;shutil.copy2(source,target)
 assert hashlib.sha256(source.read_bytes()).hexdigest()==hashlib.sha256(target.read_bytes()).hexdigest()
(b/'preparation/official-model-manifest-original.json').write_bytes((parent/'preparation/official-model-manifest-original.json').read_bytes())
for name,key in (('submit_policy_confirmation_v3_native.py','external_orchestrator'),('launch-confirmation-native.sh','external_wrapper'),('policy-confirmation-v3-selection.json','selection_bytes'),('policy-calibration-v3-selection-method.json','method_bytes'),('policy-calibration-v3-method-preregistration.json','preregistration_bytes'),('behavior-consistency-target-cpu-v1-result.json','cpu_bytes'),('policy-discovery-v3-native-launch.json','launch_bytes'),('policy-discovery-v3-native-observation-1791245435798697.json','observation_bytes')):
 content=base64.b64decode(p[key]);expected=p[{'external_orchestrator':'external_orchestrator_sha256','external_wrapper':'external_wrapper_sha256','selection_bytes':'selection_sha256','method_bytes':'selection_method_sha256','preregistration_bytes':'preregistration_sha256','cpu_bytes':'target_cpu_proof_sha256','launch_bytes':'prospective_launch_sha256','observation_bytes':'prospective_observation_sha256'}[key]]
 assert hashlib.sha256(content).hexdigest()==expected;(b/'preparation'/name).write_bytes(content)
(b/'preparation/launch-confirmation-native.sh').chmod(0o700)
(b/'preparation/source-package-intent.json').write_text(json.dumps({k:v for k,v in p.items() if k not in ('files','external_orchestrator','external_wrapper','selection_bytes','method_bytes','preregistration_bytes','cpu_bytes','launch_bytes','observation_bytes')},indent=2))
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(b/'repo'),HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1')
code='from reproduction.common8.verify_data import verify;import pathlib,sys,json,inspect,hashlib;from types import SimpleNamespace;from ray._private.node import Node;s=inspect.getsource(Node._prepare_socket_file);good="/"+"a"*106;dummy=SimpleNamespace(_sockets_dir="/unused",_make_inc_temp=lambda **kwargs:good);assert sys.platform=="linux" and Node._prepare_socket_file(dummy,None,"plasma_store")==good\ntry:\n dummy._make_inc_temp=lambda **kwargs:"/"+"a"*107;Node._prepare_socket_file(dummy,None,"plasma_store")\nexcept OSError:pass\nelse:raise AssertionError("Installed Ray accepted108-byteAF_UNIX")\nprint(json.dumps(dict(data=verify(pathlib.Path(sys.argv[1])),installed_ray_socket_check_sha256=hashlib.sha256(s.encode()).hexdigest(),installed_limit_tested_bytes=[107,108],scope="Actual installed AF_UNIX method CPU boundary check; no Ray startup")))'
r=subprocess.run([str(b/'env/bin/python'),'-c',code,str(b/'data')],cwd=b/'repo',env=env,capture_output=True,text=True,timeout=70)
assert r.returncode==0,r.stderr
syntax=subprocess.run(['bash','-n',str(b/'preparation/launch-confirmation-native.sh')],capture_output=True,text=True);assert syntax.returncode==0,syntax.stderr
proof=dict(base=str(b),parent_base=str(parent),returncode=r.returncode,stdout=r.stdout,source_commit=p['source_commit'],ray_socket_budget=p['ray_socket_budget'],actual_production_updates=0,production_admission=False,healthy_checkpoint_accepted=False,scope='Canonical six-file target-side copy and CPU structure only')
with (b/'preparation/package-cpu-proof.json').open('x') as f:json.dump(proof,f,indent=2);f.flush();os.fsync(f.fileno())
print(json.dumps(proof))
'''

START_SOURCE_CODE = r'''
import base64,hashlib,json,os,socket,subprocess,sys,time
from pathlib import Path
p=json.load(sys.stdin);b=Path(p['export']);assert not b.exists()
with socket.socket() as sock:sock.bind(('10.33.0.13',19874))
b.mkdir(exist_ok=False);s=b/'serve_original_retrieval.py';s.write_bytes(base64.b64decode(p['source']))
assert hashlib.sha256(s.read_bytes()).hexdigest()==p['source_sha256']
with (b/'start-claim.json').open('x') as f:json.dump(dict(export=str(b),source_sha256=p['source_sha256'],created_unix=time.time()),f);f.flush();os.fsync(f.fileno())
env=dict(os.environ,CUDA_VISIBLE_DEVICES='');env.pop('LD_LIBRARY_PATH',None)
with (b/'server.log').open('x') as f:process=subprocess.Popen(['nice','-n','19','python3',str(s)],env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
stat=Path('/proc/'+str(process.pid)+'/stat').read_text().rsplit(')',1)[1].split()
with (b/'launcher-process.json').open('x') as f:json.dump(dict(pid=process.pid,start_ticks=int(stat[19]),sid=int(stat[3]),uid=os.getuid(),source_sha256=p['source_sha256']),f);f.flush();os.fsync(f.fileno())
h=None
for _ in range(100):
 assert process.poll() is None
 try:h=json.loads((b/'handle.json').read_text());break
 except (FileNotFoundError,json.JSONDecodeError):time.sleep(.1)
assert h is not None and h['pid']==process.pid
h.update(export=str(b),source_sha256=p['source_sha256']);print(json.dumps(h))
'''

SUBMIT_CODE = r'''
import hashlib,json,os,subprocess,sys,time,urllib.request
from pathlib import Path
p=json.load(sys.stdin);b=Path(p['base']);assert not (b/'logs/run-manifest.json').exists()
for name,expected in p['source_sha256'].items():assert hashlib.sha256((b/'repo'/name).read_bytes()).hexdigest()==expected
for name,key in (('submit_policy_confirmation_v3_native.py','external_orchestrator_sha256'),('launch-confirmation-native.sh','external_wrapper_sha256'),('policy-confirmation-v3-selection.json','selection_sha256'),('policy-calibration-v3-selection-method.json','selection_method_sha256'),('policy-calibration-v3-method-preregistration.json','preregistration_sha256'),('behavior-consistency-target-cpu-v1-result.json','target_cpu_proof_sha256'),('policy-discovery-v3-native-launch.json','prospective_launch_sha256'),('policy-discovery-v3-native-observation-1791245435798697.json','prospective_observation_sha256')):
 assert hashlib.sha256((b/'preparation'/name).read_bytes()).hexdigest()==p[key]
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
for name,size in p['handle']['files'].items():
 with opener.open(urllib.request.Request('http://10.33.0.13:19874/'+name,method='HEAD'),timeout=10) as r:assert r.status==200 and int(r.headers['Content-Length'])==size
args=['sbatch','--parsable','--partition=gpu','--nodelist=h100-8','--nodes=1','--ntasks=1','--cpus-per-task=64','--gres=gpu:h100:8','--mem=512G','--exclusive','--time=08:00:00','--job-name=search-g1-policy-confirmation-v3','--chdir='+str(b/'repo'),'--output='+str(b/'logs/native-confirmation-v3-%j.log'),'--export=ALL,SEARCH_G1_BASE='+str(b)+',SEARCH_G1_PRODUCTION_BASE='+p['parent_base']+',SEARCH_G1_CALIBRATION_ROLE=policy_safety_calibration_confirmation,SEARCH_G1_SOURCE_COMMIT='+p['source_commit']+',SEARCH_G1_DISCOVERY_BASE='+p['discovery_base'],str(b/'preparation/launch-confirmation-native.sh')]
with (b/'preparation/native-submit-claim.json').open('x') as f:json.dump(dict(argv=args,created_unix=time.time(),base=str(b),selection_sha256=p['selection_sha256'],scope='Independent one-outer confirmation; production0 remains fresh'),f);f.flush();os.fsync(f.fileno())
with (b/'preparation/retrieval-source-handle.json').open('x') as f:json.dump(p['handle'],f);f.flush();os.fsync(f.fileno())
r=subprocess.run(args,capture_output=True,text=True)
proof=dict(base=str(b),parent_base=p['parent_base'],discovery_base=p['discovery_base'],source_commit=p['source_commit'],source_sha256=p['source_sha256'],selection_sha256=p['selection_sha256'],argv=args,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,submitted_unix=time.time(),actual_production_updates=0)
if r.returncode==0:proof['job_id']=r.stdout.strip().split(';')[0];assert proof['job_id'].isdigit()
with (b/'preparation/native-launch.json').open('x') as f:json.dump(proof,f,indent=2);f.flush();os.fsync(f.fileno())
assert r.returncode==0,r.stderr;print(json.dumps(proof))
'''

OBSERVE_CODE = r'''
import json,os,subprocess,sys,time
from pathlib import Path
p=json.load(sys.stdin);b=Path(p['base']);out=dict(base=str(b),checked_unix=time.time(),actual_production_updates=0)
launch=b/'preparation/native-launch.json';out['submission_outcome_known']=launch.exists()
job=None
if launch.exists():
 out['launch']=json.loads(launch.read_text());job=out['launch'].get('job_id')
if job:
 r=subprocess.run(['scontrol','show','job','-o',job],capture_output=True,text=True,timeout=15)
 out.update(job_stdout=r.stdout,job_returncode=r.returncode,job_stderr=r.stderr)
for name in ('pipeline-status.json','cleanup-status.json','retrieval-staging-status.json'):
 path=b/'logs'/name
 if path.exists():out[name]=json.loads(path.read_text())
active=[];marker=('SEARCH_G1_BASE='+str(b)).encode()
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  if proc.stat().st_uid!=os.getuid() or marker not in (proc/'environ').read_bytes().split(b'\0'):continue
  row=(proc/'stat').read_text().rsplit(')',1)[1].split()
  if row[0]!='Z':active.append(dict(pid=int(proc.name),start_ticks=int(row[19]),sid=int(row[3]),uid=proc.stat().st_uid))
 except (FileNotFoundError,ProcessLookupError):pass
out['actual_active_namespace_processes']=active
out['native_submit_claim_exists']=(b/'preparation/native-submit-claim.json').exists()
print(json.dumps(out))
'''

SOURCE_HANDLE_CODE = r'''
import hashlib,json,sys
from pathlib import Path
p=json.load(sys.stdin);b=Path(p['export_base']);h=json.loads((b/'handle.json').read_text())
assert hashlib.sha256((b/'serve_original_retrieval.py').read_bytes()).hexdigest()==p['export_source_sha256']
h.update(export=str(b),source_sha256=p['export_source_sha256']);print(json.dumps(h))
'''

CLEANUP_SOURCE_CODE = r'''
import hashlib,json,os,signal,sys,time
from pathlib import Path
h=json.load(sys.stdin);b=Path(h['export']);source=b/'serve_original_retrieval.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()==h['source_sha256']
path=Path('/proc')/str(h['pid'])
if path.exists():
 descriptor=os.pidfd_open(h['pid'])
 try:
  row=(path/'stat').read_text().rsplit(')',1)[1].split()
  assert (int(row[19]),int(row[3]),path.stat().st_uid)==(h['start_ticks'],h['sid'],h['uid'])
  assert str(source).encode() in (path/'cmdline').read_bytes()
  if row[0]!='Z':
   signal.pidfd_send_signal(descriptor,signal.SIGTERM)
   for _ in range(50):
    if not path.exists() or (path/'stat').read_text().rsplit(')',1)[1].split()[0]=='Z':break
    time.sleep(.1)
  assert not path.exists() or (path/'stat').read_text().rsplit(')',1)[1].split()[0]=='Z'
 finally:os.close(descriptor)
proof=dict(cleaned=True,handle=h,checked_unix=time.time(),scope='Only this source-SHA/PID-birth/SID/UID service; fixed pidfd signal')
with (b/'cleanup.json').open('x') as f:json.dump(proof,f,indent=2);f.flush();os.fsync(f.fileno())
print(json.dumps(proof))
'''


def load_inputs(prep=PREP):
    """Missing root selection rejects before intent, target copy or service."""
    require(all(path is not None for path in (METHOD_PATH, CPU_PATH, SOURCE_CLEANUP_PATH)),
            'Explicit newsource/method/CPU/cleanup contract must be configured')
    selection_bytes = (prep / SELECTION_NAME).read_bytes()
    selection = strict_json(selection_bytes)
    launch_bytes = (prep / DISCOVERY_LAUNCH_NAME).read_bytes()
    launch = strict_json(launch_bytes)
    selection_schema(selection, launch)
    method = method_binding(METHOD_PATH.read_bytes(), selection)
    require(prospective_binding(method, launch_bytes, (prep / ZERO_OBSERVATION_NAME).read_bytes()) == launch,
            'Selection is not bound to the actual root prospective source/job/observation')
    preregistration_binding((prep / PREREGISTRATION_NAME).read_bytes(), method, launch)
    cleanup = strict_json(SOURCE_CLEANUP_PATH.read_bytes())
    handle = strict_json((prep / 'policy-discovery-v3-retrieval-export-handle.json').read_bytes())
    require(cleanup['cleaned'] is True and cleanup['handle'] == handle, 'Original discovery source cleanup not verified')
    cpu_bytes = CPU_PATH.read_bytes()
    require(sha(cpu_bytes) == launch['target_cpu_proof_sha256'], 'Actual launch-bound target CPU bytes changed')
    proof = strict_json(cpu_bytes)
    summary = strict_json(proof['stdout'])
    require(proof['returncode'] == proof['shell_syntax_returncode'] == 0 and
            summary['successful'] is True and not summary['skipped'] and not summary['cuda_available'] and
            type(summary['tests_run']) is int and summary['tests_run'] == 222 and
            proof['source_sha256'] == launch['source_sha256'], 'Not the immutable actual new222 target regression proof')
    sys.path.insert(0, str(PREP))
    cpu_contract(proof, launch)
    return selection_bytes, selection, launch


def submit(prep=PREP, repo=REPO):
    intent_path = prep / 'policy-confirmation-v3-native-submit-intent.json'
    require(not intent_path.exists(), 'Attempt already reserved; observe the same namespace, never repeat')
    selection_bytes, selection, launch = load_inputs(prep)
    files = frozen_sources(launch, repo)
    # Actual historical reread happens before any remote mutation/service.
    method_bytes = METHOD_PATH.read_bytes()
    launch_bytes, observation_bytes = (prep / DISCOVERY_LAUNCH_NAME).read_bytes(), (prep / ZERO_OBSERVATION_NAME).read_bytes()
    proof_payload = dict(selection=selection, launch=launch, method_bytes=base64.b64encode(method_bytes).decode(),
                         launch_bytes=base64.b64encode(launch_bytes).decode(),
                         observation_bytes=base64.b64encode(observation_bytes).decode())
    evidence = remote('h100-8', verification_code(), proof_payload, timeout=180,
                      interpreter=selection['parent_base'] + '/env/bin/python')
    token = uuid.uuid4().hex[:8]
    base = '/mnt/data/shaoshitong/g1c-' + token
    socket_proof = ray_socket_budget(base)
    export = '/dataset/shaoshitong/g1-common8-original-retrieval-export-confirmation-' + token
    orchestrator, wrapper = Path(__file__).read_bytes(), native_wrapper(files)
    source = files['reproduction/common8/assets_export.py'].decode()
    needle = "BASE=Path('/dataset/shaoshitong/g1-common8-original-retrieval-export')"
    require(source.count(needle) == 1, 'Frozen asset exporter changed')
    source = source.replace(needle, 'BASE=Path(' + repr(export) + ')').encode()
    intent = dict(base=base, parent_base=selection['parent_base'], discovery_base=selection['discovery_base'],
                  export_base=export, source_commit=FROZEN_COMMIT, source_sha256=launch['source_sha256'],
                  selection_sha256=sha(selection_bytes), external_orchestrator_sha256=sha(orchestrator),
                  selection_method_sha256=selection['selection_method_sha256'],
                  preregistration_sha256=sha((prep / PREREGISTRATION_NAME).read_bytes()),
                  target_cpu_proof_sha256=sha(CPU_PATH.read_bytes()),
                  prospective_launch_sha256=sha(launch_bytes), prospective_observation_sha256=sha(observation_bytes),
                  export_source_sha256=sha(source),
                  ray_socket_budget=socket_proof,
                  external_wrapper_sha256=sha(wrapper), discovery_launch=launch,
                  created_unix=time.time(), source_prerequisite_proof=evidence,
                  actual_production_updates=0, scope='Independent fresh official0 one-outer confirmation, never resume child weights/Adam')
    save(intent_path, intent)
    payload = dict(intent, files={name: base64.b64encode(value).decode() for name, value in files.items()},
                   external_orchestrator=base64.b64encode(orchestrator).decode(),
                   external_wrapper=base64.b64encode(wrapper).decode(), selection_bytes=base64.b64encode(selection_bytes).decode(),
                   method_bytes=base64.b64encode(method_bytes).decode(),
                   preregistration_bytes=base64.b64encode((prep / PREREGISTRATION_NAME).read_bytes()).decode(), cpu_bytes=base64.b64encode(CPU_PATH.read_bytes()).decode(),
                   launch_bytes=base64.b64encode(launch_bytes).decode(), observation_bytes=base64.b64encode(observation_bytes).decode())
    prepared = remote('h100-8', PREPARE_CODE, payload)
    save(prep / 'policy-confirmation-v3-native-package-proof.json', prepared)
    handle = remote('h100-9', START_SOURCE_CODE, dict(export=export,
                    source=base64.b64encode(source).decode(), source_sha256=sha(source)))
    save(prep / 'policy-confirmation-v3-retrieval-export-handle.json', handle)
    # Recheck the same source, health and fresh parent before the one submission.
    remote('h100-8', verification_code(), proof_payload, timeout=180,
           interpreter=selection['parent_base'] + '/env/bin/python')
    launched = remote('h100-8', SUBMIT_CODE, dict(intent, handle=handle))
    save(prep / 'policy-confirmation-v3-native-launch.json', launched)
    return {key: value for key, value in launched.items() if key not in ('source_sha256', 'argv')}


def cleanup_eligible(observation):
    staging = observation.get('retrieval-staging-status.json', {})
    terminal = observation.get('job_returncode') == 0 and any(
        'JobState=' + value in observation.get('job_stdout', '') for value in
        ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL'))
    return ((staging.get('stage') == 'COMPLETE' and staging.get('accepted') is True) or
            (terminal and not observation['actual_active_namespace_processes']))


def observe(prep=PREP, cleanup=False):
    """Observe one saved namespace, including a lost submission response."""
    intent = strict_json((prep / 'policy-confirmation-v3-native-submit-intent.json').read_bytes())
    observation = remote('h100-8', OBSERVE_CODE, intent)
    # Exclusive timestamped evidence preserves earlier observations/unknowns.
    suffix = str(time.time_ns())
    save(prep / ('policy-confirmation-v3-observation-' + suffix + '.json'), observation)
    if not cleanup:
        return observation
    require(cleanup_eligible(observation), 'Source may still be needed or native/namespace cleanup is unknown')
    handle_path = prep / 'policy-confirmation-v3-retrieval-export-handle.json'
    handle = strict_json(handle_path.read_bytes()) if handle_path.exists() else remote('h100-9', SOURCE_HANDLE_CODE, intent)
    require(handle['export'] == intent['export_base'] and handle['source_sha256'] == intent['export_source_sha256'],
            'Source handle differs from the one reserved namespace')
    proof = remote('h100-9', CLEANUP_SOURCE_CODE, handle)
    save(prep / ('policy-confirmation-v3-retrieval-source-cleanup-' + suffix + '.json'), proof)
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--submit', action='store_true')
    group.add_argument('--native-bind', action='store_true')
    group.add_argument('--check-selection', action='store_true')
    group.add_argument('--observe', action='store_true')
    group.add_argument('--cleanup-source', action='store_true')
    parser.add_argument('--base')
    parser.add_argument('--source-commit'); parser.add_argument('--prospective-method')
    parser.add_argument('--target-cpu-proof'); parser.add_argument('--discovery-source-cleanup-proof')
    args = parser.parse_args()
    if args.native_bind:
        require(args.base is not None, 'Explicit native child base required')
        result = native_bind(args.base)
    elif args.submit:
        require(args.base is None, 'Submission reserves a new namespace; do not reuse a base')
        require(all((args.source_commit, args.prospective_method, args.target_cpu_proof, args.discovery_source_cleanup_proof)),
                'Explicit fullnewcommit, prospective method, actual222 proof and completed source cleanup required')
        configure_contract(args.source_commit, args.prospective_method, args.target_cpu_proof, args.discovery_source_cleanup_proof)
        result = submit()
    elif args.observe or args.cleanup_source:
        require(args.base is None, 'Observation uses the same reserved handle')
        result = observe(cleanup=args.cleanup_source)
    else:
        require(args.base is None, 'Local check does not operate a target')
        require(all((args.source_commit, args.prospective_method, args.target_cpu_proof, args.discovery_source_cleanup_proof)),
                'Explicit fullnewcommit, prospective method, actual222 proof and completed source cleanup required')
        configure_contract(args.source_commit, args.prospective_method, args.target_cpu_proof, args.discovery_source_cleanup_proof)
        content, selection, launch = load_inputs()
        frozen_sources(launch)
        result = dict(role='local_confirmation_selection_structure_check', selection_sha256=sha(content),
                      source_commit=FROZEN_COMMIT, structural_verified=True,
                      actual_production_updates=0, production_admission=False,
                      healthy_checkpoint_accepted=False, physical_execution_authenticated=False,
                      scope='No remote call, service, job submission or actual evidence acceptance')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
