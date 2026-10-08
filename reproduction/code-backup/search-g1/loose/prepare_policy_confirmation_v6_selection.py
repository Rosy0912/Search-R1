"""Read-only, local scalar preparation for the root-frozen v6 selection.

No network, selection-file writer, model/Adam reader, or job launcher exists here.
The caller must supply SHA-bound original metadata and a native-produced scalar
projection. This checks the supplied scalar structure; it cannot authenticate a
projection as original segment bytes, loaded workers, raw2560, or finite tensors.
Confirmation must independently reread those originals in its native allocation.

``files`` transports only the closed metadata set below, as {base64, sha256}.
``segment`` is the explicit SCALAR_SEGMENT_FIELDS projection, never the full
segment (which contains training/optimizer arrays). ``segment_sha256`` is the
original remote segment-file SHA, not a hash of that projection. The source
snapshot receipt and canonical validation order must come from the actual
native metadata reader; caller booleans do not confer physical admission.
"""
import argparse
import ast
import base64
import copy
import hashlib
import importlib.util
import json
import math
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace


HERE = Path(__file__).resolve().parent
REPO = HERE.parent / 'repo'
METHOD_SHA256 = None  # Explicit future original input, never a v5 identity.
PREREGISTRATION_SHA256 = '6e77dafe7677571c846ae47b609e61bba302132c5eac8b25875cace4349974da'
SOURCE_COMMIT = 'f0445f4edf0dbea916ff63bedafe356d6d9ebcdb'
CPU_PROOF_SHA256 = '2a6ffec1a4a8c8c53d138a6975af6daadd90221cd0256879902002ea082ae24f'
SOURCE_MAP_SHA256 = '0caacf25e7eae533760b52139ee86b009a11b0c3ce1fa191834f362b94c9da3e'
LAUNCH_SHA256 = None  # Explicit future original input, never a v5 identity.
BIRTH_SHA256 = None  # Explicit future original input, never a v5 identity.
ACTUAL_CPU_TESTS = 264
CONFIRMATION_READER_SHA256 = '7245dc8ee6e1c5b84aa15b63c343f53edd8d5dee2e635f56c6dc781074834a3f'
SCALAR_SEGMENT_FIELDS = frozenset((
    'role', 'origin', 'run_id', 'parent_production_run_id', 'slurm_job_id',
    'plan_sha256', 'identity', 'initial_reward_fingerprint', 'resolved_config_sha256',
    'ranks', 'completed_outer_updates', 'optimizer_writes_per_rank', 'structural',
    'checkpoint', 'changed_parameters', 'worker_log_sha256', 'evaluation_before',
    'evaluation_after', 'evaluation_sha256', 'actual_updates',
    'behavior_probability_gate', 'production_admission', 'healthy_checkpoint_accepted',
))
INPUT_FIELDS = frozenset(('schema_version', 'role', 'segment_sha256', 'segment',
    'files', 'source_snapshot_verification', 'canonical_validation',
    'namespace_cleanup_snapshot', 'actual_updates_stat'))
METADATA_FILES = frozenset((
    'parent/logs/run-manifest.json', 'discovery/logs/run-manifest.json',
    'discovery/preparation/parent-run-manifest.json',
    'discovery/logs/policy-safety-calibration-plan.json',
    'discovery/logs/policy-calibration-evaluation-before.json',
    'discovery/logs/policy-calibration-evaluation-after.json',
    'discovery/logs/actual-updates.jsonl',
    'discovery/logs/behavior-probability-outer-1.json',
    'discovery/logs/pipeline-status.json', 'discovery/logs/cleanup-status.json',
    'discovery/logs/process-ledger.json',
    'discovery/logs/retrieval-staging-claim.json', 'discovery/logs/corpus-cache-claim.json',
    'discovery/logs/training-launch-claim.json', 'discovery/logs/resolved-training-config.json',
    'discovery/checkpoint-complete.json',
)) | frozenset('discovery/logs/optimizer-rank-' + str(rank) + '.jsonl' for rank in range(8))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode()


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON: ' + value)))


def digest(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def same(left, right, message):
    require(canonical(left) == canonical(right), message)


def configure_future_identity(method_bytes, method_sha256, launch_bytes, launch_sha256,
                              birth_bytes, birth_sha256):
    """Bind only caller-sealed future originals; this creates no native acceptance."""
    global METHOD_SHA256, LAUNCH_SHA256, BIRTH_SHA256
    for raw, value in ((method_bytes, method_sha256), (launch_bytes, launch_sha256), (birth_bytes, birth_sha256)):
        require(isinstance(raw, bytes) and digest(value) and sha(raw) == value,
                'Explicit original future v6 method/launch/zero bytes/SHA are required')
    method, launch = strict_json(method_bytes), strict_json(launch_bytes)
    binding = method['prospective_binding']
    require(method['source_commit'] == launch['source_commit'] == SOURCE_COMMIT and
            binding['launch_sha256'] == launch_sha256 and
            binding['zero_update_observation_sha256'] == birth_sha256 and
            binding['target_cpu_proof_sha256'] == CPU_PROOF_SHA256 and
            binding['preregistration_sha256'] == PREREGISTRATION_SHA256 and
            type(binding['actual_target_cpu_tests']) is int and binding['actual_target_cpu_tests'] == ACTUAL_CPU_TESTS and
            sha(canonical(launch['source_sha256'])) == SOURCE_MAP_SHA256,
            'Future original identity belongs to another source/CPU/preregistration')
    METHOD_SHA256, LAUNCH_SHA256, BIRTH_SHA256 = method_sha256, launch_sha256, birth_sha256


def confirmation_reader(path, expected_sha256, method_sha256):
    require(path is not None and digest(method_sha256) and expected_sha256 == CONFIRMATION_READER_SHA256 and method_sha256 == METHOD_SHA256,
            'Explicit independently reviewed v6 reader path/SHA and actual method SHA are required')
    path = Path(path)
    require(path.resolve() == HERE / 'submit_policy_confirmation_v6_native.py' and
            sha(path.read_bytes()) == expected_sha256,
            'Frozen external v6 confirmation reader changed; explicit rereview is required')
    spec = importlib.util.spec_from_file_location('_v6_selection_confirmation', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.FROZEN_METHOD_SHA256 = method_sha256
    require(module.FROZEN_COMMIT == SOURCE_COMMIT and module.CPU_PROOF_SHA256 == CPU_PROOF_SHA256 and
            module.PREREGISTRATION_SHA256 == PREREGISTRATION_SHA256,
            'Reviewed external v6 reader has another approved source/CPU/preregistration contract')
    return module


def pure_readers(mapping):
    """Execute only hash-pinned stdlib structural/arithmetic functions.

    The original evaluation module imports Torch for batch extraction. Select
    its unchanged aggregate/helpers by AST so this local scalar reader never
    imports Torch or accidentally calls model/checkpoint/optimizer functions.
    """
    values = {}
    for label, relative in (('plan', 'reproduction/common8/policy_calibration_plan.py'),
                            ('ledger', 'reproduction/common8/safety_calibration_evidence.py')):
        path = REPO / relative
        raw = path.read_bytes()
        require(sha(raw) == mapping[relative], 'Frozen pure reader changed: ' + relative)
        namespace = {'__name__': '_selection_pure_' + label}
        exec(compile(raw, str(path), 'exec'), namespace)
        values[label] = SimpleNamespace(**namespace)
    relative = 'reproduction/common8/calibration_evaluation.py'
    raw = (REPO / relative).read_bytes()
    require(sha(raw) == mapping[relative], 'Frozen evaluation arithmetic changed')
    names = {'_require', '_integer', '_actions', 'aggregate'}
    constants = {'QUESTION_COUNT', 'SOURCE_COUNTS', '_ROW_FIELDS'}
    tree = ast.parse(raw)
    selected = [node for node in tree.body if
        (isinstance(node, ast.FunctionDef) and node.name in names) or
        (isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and
         target.id in constants for target in node.targets))]
    namespace = dict(math=math, Counter=Counter, Mapping=Mapping, Sequence=Sequence)
    exec(compile(ast.Module(body=selected, type_ignores=[]), relative, 'exec'), namespace)
    values['aggregate'] = namespace['aggregate']
    relative = 'reproduction/common8/behavior_probability.py'
    raw = (REPO / relative).read_bytes()
    require(sha(raw) == mapping[relative], 'Frozen sampler semantics reader changed')
    tree = ast.parse(raw)
    selected = [node for node in tree.body if
        (isinstance(node, ast.FunctionDef) and node.name in {'_require', 'verify_sampling'}) or
        (isinstance(node, ast.ClassDef) and node.name == 'BehaviorProbabilityError')]
    namespace = dict(math=math, Mapping=Mapping)
    exec(compile(ast.Module(body=selected, type_ignores=[]), relative, 'exec'), namespace)
    values['verify_sampling'] = namespace['verify_sampling']
    return SimpleNamespace(**values)


def original_files(packet):
    files = packet['files']
    require(type(files) is dict, 'Original scalar metadata files are required')
    extras = set(files) - METADATA_FILES
    require(METADATA_FILES <= set(files) and len(extras) == 3 and
        all(re.fullmatch(r'discovery/repo/outputs/\d{4}-\d\d-\d\d/\d\d-\d\d-\d\d/\.hydra/(config|hydra|overrides)\.yaml', name)
            for name in extras) and len({str(Path(name).parent) for name in extras}) == 1 and
        {Path(name).name for name in extras} == {'config.yaml','hydra.yaml','overrides.yaml'},
        'Closed metadata set plus exactly three actual same-run Hydra snapshots is required')
    raw = {}
    for name, entry in files.items():
        require(type(entry) is dict and set(entry) == {'base64', 'sha256'} and
                digest(entry['sha256']) and isinstance(entry['base64'], str), 'Invalid original byte envelope: ' + name)
        content = base64.b64decode(entry['base64'], validate=True)
        require(0 < len(content) <= 4 * 1024**2 and sha(content) == entry['sha256'],
                'Original metadata bytes/hash/size differ: ' + name)
        raw[name] = content
    return raw, extras


def hydra_descriptor(raw, extras, run, launch):
    claim_raw = raw['discovery/logs/training-launch-claim.json']
    config_raw = raw['discovery/logs/resolved-training-config.json']
    claim, config = map(strict_json, (claim_raw, config_raw))
    require(claim['run_id'] == config['run_id'] == run['run_id'] and
            claim['slurm_job_id'] == config['slurm_job_id'] == launch['job_id'] and
            digest(config['resolved_config_sha256']) and
            sha(canonical(config['resolved_config'])) == config['resolved_config_sha256'] and
            number(claim['claimed_unix']) and number(config['checked_unix']) and
            claim['claimed_unix'] <= config['checked_unix'], 'Original actual training claim/resolved identity differs')
    return dict(schema_version=1, run_id=run['run_id'], slurm_job_id=launch['job_id'],
        source_map_sha256=sha(canonical(launch['source_sha256'])),
        training_launch_claim_sha256=sha(claim_raw),
        resolved_training_config_file_sha256=sha(config_raw),
        resolved_config_sha256=config['resolved_config_sha256'],
        files={name.removeprefix('discovery/repo/'): dict(size_bytes=len(raw[name]), sha256=sha(raw[name]))
               for name in sorted(extras)})


def check_snapshot_receipt(receipt, descriptor):
    fields = {'role', 'run_id', 'slurm_job_id', 'source_map_sha256', 'descriptor_sha256',
              'source_snapshot_kind', 'production_admission', 'healthy_checkpoint_accepted',
              'physical_execution_authenticated'}
    require(type(receipt) is dict and set(receipt) == fields and
            receipt['role'] == 'exact_generated_hydra_file_snapshot_verification' and
            receipt['run_id'] == descriptor['run_id'] and receipt['slurm_job_id'] == descriptor['slurm_job_id'] and
            receipt['source_map_sha256'] == descriptor['source_map_sha256'] and
            receipt['descriptor_sha256'] == sha(canonical(descriptor)) and
            receipt['source_snapshot_kind'] == 'current_disk_not_loaded_module_proof' and
            receipt['production_admission'] is False and receipt['healthy_checkpoint_accepted'] is False and
            receipt['physical_execution_authenticated'] is False,
            'Need the actual native source354/exact-three metadata reader receipt; it is not physical proof')


def check_ledgers(raw, segment, readers):
    ledger = readers.ledger
    binding = dict(schema_version=1, role=segment['role'].removesuffix('_evidence'),
        run_id=segment['run_id'], calibration_sha256=segment['plan_sha256'],
        outer_count=1, initial_optimizer_writes=0)
    common = {'kind', 'run_id', 'rank'}
    schemas = dict(
        POLICY_MONITOR=common | set(ledger.GLOBAL_FIELDS) | {'execution_role', 'enforcement', 'context',
            'stage', 'optimizer_writes', 'calibration_sha256', 'monitor_identity',
            'objective_weight_definition', 'monitoring_scope', 'rejection_count'},
        OPTIMIZER_WRITE=common | {'context', 'write_count', 'gradient_norm_before', 'gradient_norm_after', 'learning_rates'},
        OUTER_BUDGET=common | {'logical_update', 'local_trajectories', 'local_mini', 'local_micro',
            'optimizer_writes', 'cumulative_writes'})
    logs = {}
    for rank in range(8):
        name = 'discovery/logs/optimizer-rank-' + str(rank) + '.jsonl'
        require(sha(raw[name]) == segment['worker_log_sha256'][str(rank)], 'Original worker ledger SHA differs')
        rows = [strict_json(line) for line in raw[name].splitlines()]
        for row in rows:
            require(type(row) is dict and row.get('kind') in schemas and set(row) == schemas[row['kind']],
                    'Unknown/raw token/tensor payload in scalar worker ledger')
            if row['kind'] == 'OPTIMIZER_WRITE':
                require(type(row['learning_rates']) is list and 0 < len(row['learning_rates']) <= 16,
                        'Not a bounded scalar learning-rate list')
        logs[rank] = rows
    structural = ledger.verify_segment(logs, binding)
    same(structural, segment['structural'], 'Original eight-rank coalescing differs from scalar segment projection')
    for observation in structural['global_monitors']:
        require(number(observation['sampled_old_policy_kl']) and observation['sampled_old_policy_kl'] <= .01,
                'Actual finite monitor exceeded original target old-policy KL .01')
        require(observation['negative_log_ratio_max'] is None or observation['negative_log_ratio_max'] <= 30.,
                'Actual monitor exceeded the original log bound')
    return structural


def check_evaluations(raw, segment, run, canonical_validation, aggregate):
    require(type(canonical_validation) is dict and
        set(canonical_validation) == {'validation_sha256', 'question_ids', 'data_sources'} and
        canonical_validation['validation_sha256'] == run['input_sha256']['validation.parquet'],
        'Actual canonical ordered validation identity is required')
    ids, sources = canonical_validation['question_ids'], canonical_validation['data_sources']
    measured = {}
    for label, step in (('before', 0), ('after', 1)):
        name = 'discovery/logs/policy-calibration-evaluation-' + label + '.json'
        require(sha(raw[name]) == segment['evaluation_sha256'][label], 'Original evaluation SHA differs')
        value = strict_json(raw[name])
        same(value, segment['evaluation_' + label], 'Evaluation original bytes differ from projected scalar measurements')
        require(value['run_id'] == run['run_id'] and value['parent_production_run_id'] == run['parent_production_run_id'] and
            value['slurm_job_id'] == run['policy_calibration_slurm_job_id'] and value['plan_sha256'] == segment['plan_sha256'] and
            type(value['step']) is int and value['step'] == step and value['phase_role'] == segment['role'].removesuffix('_evidence') and
            value['validation_sha256'] == canonical_validation['validation_sha256'], 'Paired evaluation run/job/step differs')
        calculated = aggregate(value['rows'], ids, sources)
        for key, actual in calculated.items():
            same(value[key], actual, 'Ordered fixed512 measurement arithmetic differs: ' + key)
        measured[label] = calculated
    for name in ('weighted_em', 'weighted_f1', 'valid_action_rate'):
        require(measured['before'][name] <= measured['after'][name], 'Original zero-allowed-drop scalar rule failed: ' + name)
    return measured


def check_gate_and_update(raw, segment, run, method, updates_stat, verify_sampling):
    name = 'discovery/logs/behavior-probability-outer-1.json'
    gate_raw = raw[name]; gate = strict_json(gate_raw)
    fields = {'schema_version', 'role', 'accepted', 'run_id', 'source_sha256', 'slurm_job_id',
        'logical_update', 'policy_snapshot_id', 'sampling', 'trajectory_count', 'sampling_distribution',
        'trajectory_ids', 'actor_token_counts', 'included_actor_token_count', 'max_abs_discrepancy',
        'tolerance', 'trace_sha256', 'actor_rows_sha256', 'audit_sha256', 'tensor_sha256',
        'hf_dtype', 'sampling_dtype', 'actual_frozen_old_all_finite', 'scope', 'production_admission',
        'healthy_checkpoint_accepted', 'physical_execution_authenticated', 'row_tensor_sha256',
        'upstream_group_sha256', 'private_trace_sha256', 'denominator'}
    require(type(gate) is dict and set(gate) == fields and type(gate['schema_version']) is int and
        gate['schema_version'] == 1 and gate['role'] == 'common8_behavior_probability_gate',
        'Unknown/raw token/tensor arrays or wrong schema in original public gate receipt')
    verify_sampling(gate['sampling'])
    require(gate['accepted'] is True and gate['run_id'] == run['run_id'] and gate['source_sha256'] == run['source_sha256'] and
        gate['slurm_job_id'] == run['policy_calibration_slurm_job_id'] and type(gate['logical_update']) is int and gate['logical_update'] == 1 and
        type(gate['trajectory_count']) is int and gate['trajectory_count'] == 2560 and
        gate['actual_frozen_old_all_finite'] is True and number(gate['max_abs_discrepancy']) and
        0 <= gate['max_abs_discrepancy'] <= gate['tolerance'] == .05 and
        gate['sampling']['temperature'] == gate['sampling']['top_p'] == 1. and
        gate['denominator'] == 'actual_selected_sampling_log_probs_after_all_token_context_and_0.05_gate' and
        gate['hf_dtype'] == gate['sampling_dtype'] == 'torch.float32' and
        gate['production_admission'] is False and gate['healthy_checkpoint_accepted'] is False and
        gate['physical_execution_authenticated'] is False,
        'Original all2560 actual-sampling receipt identity or .05 scalar gate differs')
    ids = gate['trajectory_ids']
    require(type(ids) is list and len(ids) == len(set(ids)) == 2560 and
        set(gate['row_tensor_sha256']) == set(gate['upstream_group_sha256']) == set(ids) and
        all(digest(value) for value in list(gate['row_tensor_sha256'].values()) + list(gate['upstream_group_sha256'].values())),
        'Original behavior row/group hash coverage differs')
    count = gate['included_actor_token_count']
    require(type(count) is int and count > 0 and digest(gate['trace_sha256']) and
        gate['trace_sha256'] == gate['private_trace_sha256'], 'Missing original included-actor count or raw-trace pointer')
    counts = gate['actor_token_counts']
    require(type(counts) is list and len(counts) == 2560 and
        all(type(value) is int and value > 0 for value in counts) and sum(counts) == count,
        'Original all2560 included-actor scalar counts differ')
    summary = dict(proof_sha256=sha(gate_raw), private_trace_sha256=gate['private_trace_sha256'],
        included_actor_token_count=count, max_abs_discrepancy=gate['max_abs_discrepancy'])
    same(summary, segment['behavior_probability_gate'], 'Segment behavior receipt SHA/link differs')
    updates = [strict_json(line) for line in raw['discovery/logs/actual-updates.jsonl'].splitlines()]
    same(updates, segment['actual_updates'], 'Actual update original bytes differ')
    require(len(updates) == 1, 'Exactly one completed actual outer is required')
    update = updates[0]
    require(update['run_id'] == run['run_id'] and type(update['step']) is int and update['step'] == 1 and
        update['probe_version'] == run['initial_probe_version'] and
        update['metrics']['actor/optimizer_steps_this_update'] == 20 and
        all(number(value) for value in update['metrics'].values()) and
        update['behavior_probability_gate_sha256'] == summary['proof_sha256'] and
        update['behavior_trace_sha256'] == summary['private_trace_sha256'], 'Actual update/gate/raw-trace scalar links differ')
    update_raw = raw['discovery/logs/actual-updates.jsonl']
    require(type(updates_stat) is dict and set(updates_stat) == {'size_bytes', 'sha256', 'mtime_unix'} and
        type(updates_stat['size_bytes']) is int and updates_stat['size_bytes'] == len(update_raw) and
        updates_stat['sha256'] == sha(update_raw) and number(updates_stat['mtime_unix']) and
        method['frozen_unix'] <= updates_stat['mtime_unix'],
        'Need actual original completed-update file stat; prospective method cannot follow completion')
    for monitor in segment['structural']['global_monitors']:
        require(monitor['context'].get('behavior_probability_gate_sha256') == summary['proof_sha256'],
                'Actual40 monitor/write receipt context changed')
    return summary


def check_checkpoint(raw, segment, run, discovery_base):
    complete = strict_json(raw['discovery/checkpoint-complete.json'])
    same(complete, segment['checkpoint'], 'Original fresh1 checkpoint scalar receipt differs')
    expected = str(Path(run['checkpoint_dir']) / 'actor/global_step_1')
    require(Path(discovery_base) in Path(expected).parents and complete['checkpoint'] == expected and
        complete['run_id'] == run['run_id'] and type(complete['step']) is int and complete['step'] == 1 and
        type(complete['validated_tensor_count']) is int and complete['validated_tensor_count'] == 435 and
        complete['finite_all_saved_tensors'] is True and type(complete['files_sha256']) is dict and bool(complete['files_sha256']) and
        all(digest(value) for value in complete['files_sha256'].values()), 'Missing producer fresh1/canonical435/finite receipt')
    changed = segment['changed_parameters']
    require(type(changed) is list and 0 < len(changed) <= 435 and len(set(changed)) == len(changed) and
        all(isinstance(value, str) and value for value in changed), 'Missing producer canonical parameter delta scalar receipt')
    return dict(checkpoint_complete_sha256=sha(raw['discovery/checkpoint-complete.json']),
                producer_changed_parameter_count=len(changed), independent_tensor_verification_pending=True)


def check_cleanup(raw, packet, run, terminal):
    status, cleanup, ledger = (strict_json(raw['discovery/logs/' + name]) for name in
        ('pipeline-status.json', 'cleanup-status.json', 'process-ledger.json'))
    require(status['run_id'] == cleanup['run_id'] == ledger['run_id'] == run['run_id'] and
        status['stage'] == 'FINISHED_POLICY_CALIBRATION_MEASUREMENTS_AND_OWNED_CLEANUP' and
        cleanup['cleaned'] is True and type(ledger['jobs']) is dict and
        {'trainer', 'retriever', 'probe-server'} <= set(ledger['jobs']) and
        all(job['state'] == 'STOPPED' for job in ledger['jobs'].values()), 'Actual owned cleanup metadata is incomplete')
    for name in ('retrieval-staging-claim.json', 'corpus-cache-claim.json'):
        claim = strict_json(raw['discovery/logs/' + name])
        root = Path('/dev/shm') / ('search-g1-common8-' + run['run_id'] + '-' + run['policy_calibration_slurm_job_id'])
        if name == 'corpus-cache-claim.json': root /= 'datasets-cache'
        require(claim['run_id'] == run['run_id'] and claim['slurm_job_id'] == run['policy_calibration_slurm_job_id'] and
            claim['phase'] == 'CLEANED' and claim['root'] == str(root), 'Actual temporary-cache cleanup claim differs')
    observed = packet['namespace_cleanup_snapshot']
    require(type(observed) is dict and set(observed) == {'run_id', 'base', 'slurm_job_id', 'checked_unix',
        'active_owned_processes', 'temporary_roots_present'} and observed['run_id'] == run['run_id'] and
        observed['base'] == terminal['base'] and observed['slurm_job_id'] == terminal['job_id'] and
        number(observed['checked_unix']) and observed['checked_unix'] >= terminal['checked_unix'] and
        observed['active_owned_processes'] == [] and observed['temporary_roots_present'] == [],
        'Need actual completed namespace-empty/cache-absent metadata; no inferred cleanup')


def prepare_selection(evidence_bytes, *, evidence_sha256, method_bytes, method_sha256,
        launch_bytes, launch_sha256, birth_bytes, birth_sha256, terminal_bytes, terminal_sha256,
        preregistration_bytes, preregistration_sha256, cpu_bytes, cpu_sha256, expected_cpu_tests,
        confirmation_reader_path, confirmation_reader_sha256):
    """Return a candidate in memory only after the closed scalar checks succeed.

    Neither original-file hashes nor successful scalar reconstruction prove
    physical execution. The returned candidate still requires the existing
    confirmation small reader and independent native full reader before use.
    """
    configure_future_identity(method_bytes, method_sha256, launch_bytes, launch_sha256, birth_bytes, birth_sha256)
    for raw, actual, frozen, label in (
        (method_bytes,method_sha256,METHOD_SHA256,'method'),
        (preregistration_bytes,preregistration_sha256,PREREGISTRATION_SHA256,'preregistration'),
        (launch_bytes,launch_sha256,LAUNCH_SHA256,'launch'),
        (birth_bytes,birth_sha256,BIRTH_SHA256,'zero observation'),
        (cpu_bytes,cpu_sha256,CPU_PROOF_SHA256,'target CPU')):
        require(isinstance(raw,bytes) and digest(actual) and sha(raw) == actual == frozen,
                'Explicit original rootv6 '+label+' bytes/SHA are required')
    require(type(expected_cpu_tests) is int and expected_cpu_tests == ACTUAL_CPU_TESTS,
            'Explicit original actual264 CPU test count is required')
    require(digest(evidence_sha256) and sha(evidence_bytes) == evidence_sha256 and
            len(evidence_bytes) <= 64 * 1024**2, 'Explicit original scalar-input SHA/size is required')
    reader = confirmation_reader(confirmation_reader_path,confirmation_reader_sha256,method_sha256)
    method, launch, packet = map(strict_json, (method_bytes, launch_bytes, evidence_bytes))
    require(reader.prospective_binding(method, launch_bytes, birth_bytes) == launch,
            'Actual prospective launch/birth binding differs')
    require(launch['source_commit'] == SOURCE_COMMIT and sha(canonical(launch['source_sha256'])) == SOURCE_MAP_SHA256 and
            type(launch['actual_target_cpu_tests']) is int and launch['actual_target_cpu_tests'] == expected_cpu_tests,
            'Original approved full354 source map/commit/CPU count differs')
    reader.preregistration_binding(preregistration_bytes, method, launch)
    require(sha(cpu_bytes) == reader.CPU_PROOF_SHA256 == launch['target_cpu_proof_sha256'], 'Original actual264 CPU bytes differ')
    reader.cpu_contract(strict_json(cpu_bytes), launch)
    terminal = reader.read_terminal_observation(terminal_bytes, expected_sha256=terminal_sha256,
        launch_bytes=launch_bytes, expected_launch_sha256=method['prospective_binding']['launch_sha256'],
        expected_run_id=method['discovery_run_id'], expected_parent_run_id=method['parent_production_run_id'],
        expected_state='COMPLETED', expected_exit_code='0:0', birth_observation_bytes=birth_bytes,
        expected_birth_sha256=method['prospective_binding']['zero_update_observation_sha256'])
    require(type(packet) is dict and set(packet) == INPUT_FIELDS and type(packet['schema_version']) is int and
        packet['schema_version'] == 1 and packet['role'] == 'policy_confirmation_v6_local_scalar_input' and
        digest(packet['segment_sha256']), 'Closed original native scalar input is required')
    raw, extras = original_files(packet)
    parent = strict_json(raw['parent/logs/run-manifest.json'])
    run = strict_json(raw['discovery/logs/run-manifest.json'])
    original_parent_raw = raw['discovery/preparation/parent-run-manifest.json']
    original_parent = strict_json(original_parent_raw)
    require(parent['run_id'] == method['parent_production_run_id'] and run['run_id'] == method['discovery_run_id'] and
        run['parent_production_run_id'] == parent['run_id'] and run['policy_calibration_role'] == reader.DISCOVERY_ROLE and
        run['policy_calibration_slurm_job_id'] == launch['job_id'] and
        run['source_commit'] == parent['source_commit'] == reader.FROZEN_COMMIT and
        run['source_sha256'] == parent['source_sha256'] == launch['source_sha256'] and
        sha(original_parent_raw) == run['policy_calibration_parent_manifest_sha256'], 'Original parent/child manifest/source link differs')
    readers = pure_readers(launch['source_sha256'])
    identity = readers.plan._identity(parent)
    same(readers.plan._identity(run), identity, 'Child scientific identity differs')
    same(readers.plan._identity(original_parent), identity, 'Original parent snapshot scientific identity differs')
    plan_raw = raw['discovery/logs/policy-safety-calibration-plan.json']; plan = strict_json(plan_raw)
    plan_sha = sha(canonical(plan))
    readers.plan.verify_plan(plan, parent, expected_plan_sha256=run['policy_calibration_plan_sha256'])
    require(plan['role'] == reader.DISCOVERY_ROLE and plan['run_id'] == run['run_id'] and
        plan['slurm_job_id'] == launch['job_id'], 'Actual discovery plan handle differs')
    segment = packet['segment']
    require(type(segment) is dict and set(segment) == SCALAR_SEGMENT_FIELDS and
        segment['role'] == reader.DISCOVERY_ROLE + '_evidence' and segment['origin'] == 'actual_native_policy_segment' and
        segment['run_id'] == run['run_id'] and segment['parent_production_run_id'] == parent['run_id'] and
        segment['slurm_job_id'] == launch['job_id'] and segment['plan_sha256'] == plan_sha and
        type(segment['completed_outer_updates']) is int and segment['completed_outer_updates'] == 1 and
        type(segment['optimizer_writes_per_rank']) is int and segment['optimizer_writes_per_rank'] == 20 and
        segment['ranks'] == list(range(8)) and all(type(rank) is int for rank in segment['ranks']) and
        segment['production_admission'] is False and
        segment['healthy_checkpoint_accepted'] is False, 'Missing complete20/40 scalar discovery projection')
    for name in ('identity', 'initial_reward_fingerprint', 'resolved_config_sha256'):
        same(segment[name], plan[name], 'Projected segment frozen identity differs: ' + name)
    descriptor = hydra_descriptor(raw, extras, run, launch)
    require(descriptor['resolved_config_sha256'] == plan['resolved_config_sha256'], 'Actual resolved recipe differs from plan')
    check_snapshot_receipt(packet['source_snapshot_verification'], descriptor)
    structural = check_ledgers(raw, segment, readers)
    measurements = check_evaluations(raw, segment, run, packet['canonical_validation'], readers.aggregate)
    gate = check_gate_and_update(raw, segment, run, method, packet['actual_updates_stat'], readers.verify_sampling)
    checkpoint = check_checkpoint(raw, segment, run, launch['base'])
    check_cleanup(raw, packet, run, terminal)
    tail = max(row['negative_tail_weight_fraction'] for row in structural['global_monitors'])
    health_rule = dict(document=copy.deepcopy(method['health_rule_document']),
                       sha256=sha(canonical(method['health_rule_document'])))
    criteria = dict(discovery_evidence_sha256=packet['segment_sha256'], health_rule_sha256=health_rule['sha256'],
                    candidate_thresholds=dict(method['fixed_candidates'], negative_tail_limit=tail))
    threshold_doc = dict(role='policy_safety_threshold_selection', provenance=dict(
        method_sha256=METHOD_SHA256, scalar_input_sha256=evidence_sha256,
        method='Exact maximum across actual40 coalesced pre/post observations; no margin/floor',
        scope='Monitored32 empirical envelope only; independent native confirmation is required',
        production_admission=False, healthy_checkpoint_accepted=False), criteria=criteria)
    selection = dict(schema_version=1, role='policy_safety_confirmation_orchestration_selection',
        selected_by='root_explicit_selection', selection_provenance=dict(
            origin='Root-frozen prospectivev6 method applied to supplied original scalar evidence',
            scalar_input_sha256=evidence_sha256, launch_sha256=sha(launch_bytes),
            birth_observation_sha256=sha(birth_bytes), preregistration_sha256=sha(preregistration_bytes),
            production_admission=False, healthy_checkpoint_accepted=False, physical_execution_authenticated=False,
            independent_native_verification_required=True),
        parent_base=launch['parent_base'], discovery_base=launch['base'],
        parent_production_run_id=parent['run_id'], discovery_run_id=run['run_id'], discovery_slurm_job_id=launch['job_id'],
        source_commit=reader.FROZEN_COMMIT, source_map_sha256=sha(canonical(launch['source_sha256'])),
        scientific_identity_sha256=sha(canonical(identity)), discovery_manifest_sha256=sha(raw['discovery/logs/run-manifest.json']),
        discovery_plan_file_sha256=sha(plan_raw), discovery_plan_sha256=plan_sha,
        discovery_evidence_sha256=packet['segment_sha256'], discovery_terminal_observation_sha256=terminal_sha256,
        selection_method_sha256=METHOD_SHA256, discovery_generated_hydra_snapshots=descriptor,
        negative_tail_limit=tail, health_rule=health_rule,
        threshold_selection=dict(document=threshold_doc, sha256=sha(canonical(threshold_doc))))
    reader.selection_schema(selection, launch)
    reader.method_binding(method_bytes, selection)
    reader.measured_selection(selection, segment)
    return dict(role='policy_confirmation_v6_local_scalar_selection_candidate', selection=selection,
        scalar_input_sha256=evidence_sha256, unique_global_observation_count=40,
        optimizer_writes_per_rank=20, paired_question_count=512,
        scalar_metrics={phase: {key: value[key] for key in ('weighted_em', 'weighted_f1', 'valid_action_rate')}
                        for phase, value in measurements.items()},
        behavior_probability_receipt=gate, checkpoint_scalar_receipt=checkpoint,
        production_admission=False, healthy_checkpoint_accepted=False, physical_execution_authenticated=False,
        full_checkpoint_verification_pending=True, optimizer_tensor_verification_pending=True,
        raw2560_verification_pending=True, original_full_segment_reread_pending=True,
        scope='Supplied original small metadata/scalar/hash structure only; no independent tensor/raw/GPU authentication')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('evidence-input', 'method', 'launch', 'birth', 'terminal', 'preregistration', 'cpu'):
        parser.add_argument('--' + name, type=Path, required=True)
        if name != 'evidence-input': parser.add_argument('--'+name+'-sha256',required=True)
    parser.add_argument('--evidence-sha256', required=True)
    parser.add_argument('--expected-cpu-tests',required=True,type=int)
    parser.add_argument('--confirmation-reader',type=Path,required=True)
    parser.add_argument('--confirmation-reader-sha256',required=True)
    args = parser.parse_args()
    result = prepare_selection(args.evidence_input.read_bytes(), evidence_sha256=args.evidence_sha256,
        method_bytes=args.method.read_bytes(), method_sha256=args.method_sha256,
        launch_bytes=args.launch.read_bytes(), launch_sha256=args.launch_sha256,
        birth_bytes=args.birth.read_bytes(),birth_sha256=args.birth_sha256,
        terminal_bytes=args.terminal.read_bytes(), terminal_sha256=args.terminal_sha256,
        preregistration_bytes=args.preregistration.read_bytes(),preregistration_sha256=args.preregistration_sha256,
        cpu_bytes=args.cpu.read_bytes(),cpu_sha256=args.cpu_sha256,expected_cpu_tests=args.expected_cpu_tests,
        confirmation_reader_path=args.confirmation_reader,confirmation_reader_sha256=args.confirmation_reader_sha256)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
