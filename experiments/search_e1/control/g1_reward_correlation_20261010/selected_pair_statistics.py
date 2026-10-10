"""CPU-only consumer of accepted historical R artifacts at eight chosen steps.

This reads every C/R record of steps 1,20,40,60,80,100,120,140 on the original
remote node.  An accepted step from a stopped dense producer may be reused; an
unfinished step is never used.  It does not require that stopped producer's
whole-scope completion.  It never trains, generates, probes, transfers raw data,
loads tensors, or edits a producer.  No converted raw ledger is written.

The published pairwise_pearson implementation supplies exact sufficient moments.
Its contiguous coverage initializer is replaced with explicit selected-step
coverage; actual historical step/attempt/trajectory IDs are never renumbered.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import socket
import struct
import sys
from types import SimpleNamespace

SELECTED_STEPS = [1, 20, 40, 60, 80, 100, 120, 140]
CONSTANTS = dict(PENALTY=.1, MAX_SEARCH=2, wrong_floor=.05, bal_ground=.6,
    bal_nosearch_bonus=.5, w_cost=.02, wrong_ground_scale=0., probe_base=0.,
    probe_gate=True, probe_alpha_cost=False)


class Hold(Exception):
    """Controlled, non-private error codes only."""


def require(value, reason):
    if not value:
        raise Hold(reason)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def immutable_path(path):
    path = Path(path).absolute()
    require(path.resolve() == path and path.is_file() and not path.is_symlink(), 'missing_or_symlink_input')
    return path


def read_metadata(path, expected=None):
    path = immutable_path(path)
    require(path.stat().st_size <= 32 * 1024 * 1024, 'metadata_exceeds_32MiB')
    raw = path.read_bytes(); sha = hashlib.sha256(raw).hexdigest()
    require(expected is None or sha == expected, 'metadata_SHA_mismatch')
    return json.loads(raw), sha


def load_bound_module(name, path, expected):
    path = immutable_path(path)
    require(file_sha(path) == expected, 'published_CPU_dependency_SHA_mismatch')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    require('torch' not in sys.modules, 'Torch_import_forbidden')
    return module


def load_native_score(path, expected, constants):
    require(file_sha(immutable_path(path)) == expected and constants == CONSTANTS, 'native_reward_binding_mismatch')
    tree = ast.parse(Path(path).read_text())
    methods = [method for item in tree.body if isinstance(item, ast.ClassDef)
               for method in item.body if isinstance(method, ast.FunctionDef) and method.name == '_score_probe']
    require(len(methods) == 1, 'native_score_method_not_unique')
    module = ast.Module(body=methods, type_ignores=[]); ast.fix_missing_locations(module)
    namespace = {}; exec(compile(module, str(path), 'exec'), namespace)
    owner = SimpleNamespace(**constants)
    return lambda c, b, d: namespace['_score_probe'](owner, dict(
        answer='VALID' if c['final_action_valid'] else None, correct=bool(c['C_raw']),
        n_search=c['model_authored_search_count']), dict(boundary=b, flip=d))[0]


def selected_accumulator(pairwise, coverage):
    """Adapt coverage only; reuse unchanged published add/finish/exact math."""
    steps = coverage['selected_steps']
    require(steps and all(type(step) is int and 1 <= step <= 150 for step in steps) and
            steps == sorted(set(steps)), 'selected_steps_not_sorted_unique_TRAIN_steps')
    require([item['step'] for item in coverage['updates']] == steps, 'selected_coverage_binding_mismatch')
    instance = pairwise.CompletePairLedger.__new__(pairwise.CompletePairLedger)
    instance.run_id = coverage['run_id']
    instance.semantics = 'reconstructed_frozen_historical_input'
    instance.reward_field = 'R_reconstructed'
    instance.bindings, instance.steps = {}, {}
    attempts = set()
    for item in coverage['updates']:
        require(re.fullmatch('[0-9a-f]{32}', item['attempt_id']) is not None and
                re.fullmatch('[0-9a-f]{64}', item['trajectory_ids_sha256']) is not None and
                item['attempt_id'] not in attempts, 'invalid_selected_attempt_or_ID_set_binding')
        attempts.add(item['attempt_id'])
        instance.bindings[item['step']] = item
        instance.steps[item['step']] = dict(moments=pairwise.ExactBinaryMoments(), ids=set(), parents={})
    return instance


def source_steps(config):
    if 'selected_steps' in config:
        steps = config['selected_steps']
    else:
        steps = list(range(config['start_step'], config['end_step'] + 1))
    require(steps and all(type(step) is int and 1 <= step <= 150 for step in steps) and
            steps == sorted(set(steps)), 'invalid_producer_configured_steps')
    return steps


def stream_jsonl(path, expected, pairwise):
    path = immutable_path(path); before = path.stat()
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while True:
            raw = stream.readline(pairwise.MAX_PAIR_JSONL_BYTES + 1)
            if not raw:
                break
            require(len(raw) <= pairwise.MAX_PAIR_JSONL_BYTES and raw.endswith(b'\n') and raw.strip(),
                    'partial_or_oversized_record')
            h.update(raw); yield pairwise.parse_json(raw)
    after = path.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), 'input_changed_during_stream')
    require(h.hexdigest() == expected, 'complete_JSONL_SHA_mismatch')


def index_C(path, summary, gate, update, run_id, pairwise):
    indexed = {}
    for position, row in enumerate(stream_jsonl(path, summary['raw_C_fresh_sha256'], pairwise)):
        require(position < 2560 and row['trajectory_id'] == gate['trajectory_ids'][position], 'C_native_ID_order_mismatch')
        require(row['run_id'] == run_id and row['step'] == update['step'] and row['attempt_id'] == update['attempt_id'] and
                row['probe_version_for_future_R_replay'] == update['probe_version'] and
                row['policy_snapshot_id'] == gate['policy_snapshot_id'], 'C_row_native_identity_mismatch')
        require(type(row['C_raw']) is int and row['C_raw'] in (0, 1) and type(row['final_action_valid']) is bool and
                (not row['C_raw'] or row['final_action_valid']) and type(row['model_authored_search_count']) is int and
                row['model_authored_search_count'] >= 0 and row['reconstructed_strict_C'] is True and
                row['original_online_reward'] is False, 'invalid_C_or_provenance')
        require(row['upstream_group_sha256'] == gate['upstream_group_sha256'][row['trajectory_id']], 'C_upstream_hash_mismatch')
        require(row['trajectory_id'] not in indexed, 'duplicate_C_identity')
        indexed[row['trajectory_id']] = (row, position, hashlib.sha256(canonical(row)).hexdigest())
    require(len(indexed) == 2560 and sum(row['C_raw'] for row, _, _ in indexed.values()) == summary['C_hits_exact'],
            'incomplete_C_or_hit_count_mismatch')
    return indexed


def validate_R_row(row, position, indexed, update, run_id, native_score):
    identity = row['trajectory_id']; require(identity in indexed, 'unknown_R_trajectory')
    C, original_index, C_sha = indexed[identity]
    require(row['run_id'] == run_id and row['step'] == update['step'] and row['attempt_id'] == update['attempt_id'] and
            row['probe_version'] == update['probe_version'], 'R_native_identity_mismatch')
    require(type(row['original_sample_index']) is int and row['original_sample_index'] == original_index and
            type(row['balanced_request_index']) is int and row['balanced_request_index'] == position and
            row['C_record_sha256'] == C_sha and row['upstream_group_sha256'] == C['upstream_group_sha256'],
            'R_original_row_mapping_or_C_SHA_mismatch')
    require(row['original_online_R_available'] is False, 'R_mislabeled_online_provenance')
    require(all(row[name] == C[name] and type(row[name]) is type(C[name]) for name in
                ['C_raw', 'final_action_valid', 'model_authored_search_count']), 'R_C_outcome_mismatch')
    b, d = row['b_reconstructed'], row['d_reconstructed']
    require(all(type(x) in (int, float) and math.isfinite(x) and 0 <= x <= 1 for x in (b, d)), 'nonfinite_or_invalid_probe_probability')
    python_R, R = row['R_python_before_FP32'], row['R_reconstructed']
    require(all(type(x) in (int, float) and math.isfinite(x) for x in (python_R, R)), 'nonfinite_R')
    require(python_R == native_score(C, float(b), float(d)), 'R_record_native_score_mismatch')
    require(float(R) == struct.unpack('<f', struct.pack('<f', python_R))[0], 'R_not_native_final_FP32')
    return dict(run_id=run_id, step=update['step'], attempt_id=update['attempt_id'], trajectory_id=identity,
                C_raw=C['C_raw'], R_reconstructed=float(R), original_online_reward=False,
                reward_semantics='reconstructed_frozen_historical_input')


def check_comparison(summary, update, report, minimum, maximum, C_summary):
    comparison = summary['native_scalar_comparison']
    native = update['metrics']['critic/rewards/mean']
    require(comparison['passed'] is True and comparison['precise_native_update_R_mean'] == native and
            comparison['precise_mean_tolerance'] == 2e-6 and
            comparison['precise_mean_absolute_difference'] <= 2e-6 and
            abs(report['R_mean'] - native) <= 2e-6, 'native_full_precision_mean_gate_not_passed')
    require(comparison['native_scalar_agreement_does_not_prove_online_per_row_bitwise_identity'] is True,
            'missing_historical_numeric_identity_caveat')
    metrics = comparison['native_printed']
    actual = dict(zip(['critic/score/mean', 'critic/score/min', 'critic/score/max', 'train/perf/em'],
                      [report['R_mean'], minimum, maximum, report['C_mean']]))
    require(all(abs(value - metrics[key]) <= .0005001 for key, value in actual.items()) and
            report['sumC'] == C_summary['C_hits_exact'], 'native_printed_extrema_or_C_gate_mismatch')
    for legacy, exact in [('sum_C', 'sumC'), ('sum_R', 'sumR'), ('sum_R2', 'sumR2'), ('sum_CR', 'sumCR')]:
        require(summary[legacy] == report[exact], 'accepted_R_summary_moment_mismatch')


def analyze_scope(config, pairwise, C_helper):
    steps = config['selected_steps']
    require(steps == sorted(set(steps)) and steps and all(type(x) is int and 1 <= x <= 150 for x in steps), 'invalid_selected_scope')
    run_id = config['native_run_id']
    require(re.fullmatch('[0-9a-f]{32}', run_id) is not None, 'invalid_native_run_id')
    native_base = Path(config['original_base']); D = Path(config['diagnostic_base']); C_dir = Path(config['C_directory'])
    require(all(path.is_absolute() and path.resolve() == path for path in (native_base, D, C_dir)) and
            C_dir.is_relative_to(D), 'original_or_C_directory_path_binding_mismatch')
    run, _ = read_metadata(native_base / 'logs/run-manifest.json', config['run_manifest_sha256'])
    require(run['run_id'] == run_id, 'original_run_mismatch')
    C_inputs, C_inputs_sha = read_metadata(C_dir / 'inputs.json', config['C_inputs_sha256'])
    require(C_inputs['run_id'] == run_id and C_inputs['helper_sha256'] == config['C_helper_sha256'], 'C_producer_source_binding_mismatch')
    for relative, expected in C_inputs['sources'].items():
        require(run['source_sha256'][relative] == expected, 'C_original_source_identity_mismatch')
    raw = (native_base / 'logs/actual-updates.jsonl').read_bytes()
    require(raw.endswith(b'\n'), 'partial_native_update_ledger')
    all_updates = [pairwise.parse_json(line) for line in raw.splitlines()]
    require([u['step'] for u in all_updates] == list(range(1, len(all_updates) + 1)) and
            all(u['run_id'] == run_id for u in all_updates), 'native_ledger_not_unique_contiguous_original_run')
    updates = {u['step']: u for u in all_updates if u['step'] in steps}
    require(set(updates) == set(steps), 'selected_native_steps_not_committed')
    assignments, source_proofs = {}, []
    for source in config['R_sources']:
        selected = source['selected_steps']; require(selected == sorted(set(selected)) and selected, 'invalid_R_source_selected_steps')
        R_cfg, R_cfg_sha = read_metadata(source['config_path'], source['config_sha256'])
        R_dir = Path(R_cfg['output']); source_dir = Path(source['source_directory'])
        require(R_cfg['expected_run_id'] == run_id and R_cfg['base'] == str(native_base) and
                R_cfg['C_base'] == str(C_dir) and R_cfg['run_manifest_sha256'] == config['run_manifest_sha256'] and
                R_cfg['C_helper_sha256'] == config['C_helper_sha256'] and
                R_cfg['reward_constants'] == CONSTANTS and all(step in source_steps(R_cfg) for step in selected), 'R_config_native_scope_binding_mismatch')
        require(R_dir.resolve() == R_dir and R_dir.is_relative_to(D) and R_dir != C_dir, 'R_artifacts_not_remote_diagnostic_children')
        manifest, manifest_sha = read_metadata(source['source_manifest_path'], source['source_manifest_sha256'])
        for relative, expected in manifest['source_sha256'].items():
            path = source_dir / relative
            require(path.resolve().is_relative_to(source_dir) and file_sha(immutable_path(path)) == expected, 'R_published_source_SHA_mismatch')
        inputs, inputs_sha = read_metadata(R_dir / 'inputs.json', source['inputs_sha256'])
        require(inputs['original_run_id'] == run_id and inputs['config_sha256'] == R_cfg_sha and
                inputs['source_manifest_sha256'] == manifest_sha and inputs['raw_online_R_saved'] is False,
                'R_execution_input_binding_mismatch')
        for step in selected:
            require(step in steps and step not in assignments, 'duplicate_or_unrequested_R_source_assignment')
            assignments[step] = R_dir
        source_proofs.append(dict(selected_steps=selected, R_directory=str(R_dir), config_sha256=R_cfg_sha,
            source_manifest_sha256=manifest_sha, inputs_sha256=inputs_sha,
            diagnostic_run_id=R_cfg['diagnostic_run_id'], producer_whole_scope_completion_not_required=True))
    require(set(assignments) == set(steps), 'missing_selected_R_source_assignment')
    metadata, coverage = {}, dict(run_id=run_id, selected_steps=steps, updates=[])
    for step in steps:
        update = updates[step]; R_dir = assignments[step]
        R_summary, R_sha = read_metadata(R_dir / ('TRAIN-' + str(step) + '-summary.json'))
        C_summary, C_sha = read_metadata(C_dir / ('TRAIN-' + str(step) + '-summary.json'), R_summary['independent_C_summary_sha256'])
        gate, gate_sha = read_metadata(native_base / 'logs' / ('behavior-probability-attempt-' + update['attempt_id'] + '.json'), update['behavior_probability_gate_sha256'])
        C_helper.verify_native_gate(gate, update, run)
        require(R_summary['status'] == 'ACCEPTED_RECONSTRUCTED_R_STEP' and R_summary['step'] == step and
                R_summary['N'] == R_summary['trajectory_count'] == R_summary['actual_probe_rows'] == 2560 and
                R_summary['commit_micro4_batches'] == R_summary['boundary_micro4_batches'] == 640 and
                R_summary['probe_version'] == update['probe_version'] and
                R_summary['original_trace_sha256'] == update['behavior_trace_sha256'] and
                R_summary['actor_rows_sha256'] == gate['actor_rows_sha256'] and
                R_summary['original_online_row_rewards_available'] is False, 'R_step_not_native_complete_accepted')
        require(C_summary['step'] == step and C_summary['attempt_id'] == update['attempt_id'] and
                C_summary['trajectory_count'] == 2560 and C_summary['native_EM_check_pass'] is True and
                C_summary['probe_version_for_future_R_replay'] == update['probe_version'] and
                C_summary['complete_trace_fresh_sha256'] == update['behavior_trace_sha256'] and
                R_summary['independent_C_rows_sha256'] == C_summary['raw_C_fresh_sha256'], 'C_step_identity_or_acceptance_mismatch')
        R_path = Path(R_summary['output_path']); C_path = Path(C_summary['raw_C_remote_path'])
        require(R_path == R_dir / ('TRAIN-' + str(step) + '-pairs-R-reconstructed.jsonl') and
                C_path == C_dir / ('TRAIN-' + str(step) + '-raw-C.jsonl'), 'row_artifact_path_mismatch')
        coverage['updates'].append(dict(step=step, attempt_id=update['attempt_id'],
            trajectory_ids_sha256=pairwise.trajectory_ids_sha256(gate['trajectory_ids'])))
        metadata[step] = (R_summary, C_summary, gate, R_sha, C_sha, gate_sha, R_path, C_path)
    score = load_native_score(native_base / 'repo/verl/trainer/main_ppo_eco.py',
                              run['source_sha256']['verl/trainer/main_ppo_eco.py'], CONSTANTS)
    accumulator = selected_accumulator(pairwise, coverage); proofs = []
    for step in steps:
        R_summary, C_summary, gate, R_sha, C_sha, gate_sha, R_path, C_path = metadata[step]
        update = updates[step]; indexed = index_C(C_path, C_summary, gate, update, run_id, pairwise)
        order, minimum, maximum = [], math.inf, -math.inf
        for position, row in enumerate(stream_jsonl(R_path, R_summary['output_sha256'], pairwise)):
            adapted = validate_R_row(row, position, indexed, update, run_id, score)
            accumulator.add(adapted); order.append(row['original_sample_index'])
            minimum = min(minimum, adapted['R_reconstructed']); maximum = max(maximum, adapted['R_reconstructed'])
        require(sorted(order) == list(range(2560)) and
                hashlib.sha256(canonical(order)).hexdigest() == R_summary['post_balance_permutation_sha256'], 'R_original_row_permutation_mismatch')
        report = accumulator.steps[step]['moments'].report()
        check_comparison(R_summary, update, report, minimum, maximum, C_summary)
        require(file_sha(R_path) == R_summary['output_sha256'] and file_sha(C_path) == C_summary['raw_C_fresh_sha256'], 'row_artifact_changed_after_stream')
        require(read_metadata(assignments[step] / ('TRAIN-' + str(step) + '-summary.json'))[1] == R_sha and
                read_metadata(C_dir / ('TRAIN-' + str(step) + '-summary.json'))[1] == C_sha, 'step_receipt_changed_during_statistics')
        proofs.append(dict(step=step, attempt_id=update['attempt_id'], accepted_R_summary_sha256=R_sha,
            accepted_C_summary_sha256=C_sha, original_native_gate_sha256=gate_sha,
            original_trace_SHA_authenticated_by_replay_not_rehashed_here=update['behavior_trace_sha256'],
            C_row_artifact_fresh_sha256=C_summary['raw_C_fresh_sha256'], R_row_artifact_fresh_sha256=R_summary['output_sha256'],
            original_trajectory_ID_set_sha256=coverage['updates'][steps.index(step)]['trajectory_ids_sha256'],
            original_row_permutation_sha256=R_summary['post_balance_permutation_sha256'],
            native_full_mean_tolerance=2e-6, complete_individual_pairs=2560))
        del indexed
    report = accumulator.finish()
    report.update(status='ACCEPTED_SELECTED_TRAIN_STEPS_PAIR_CORRELATION', selected_steps=steps,
        selected_only=True, not_all_TRAIN_steps_1_to_150=True, no_interpolation_or_extrapolation=True,
        required_individual_pairs=len(steps) * 2560, original_online_R_available=False,
        original_online_R_bitwise_identity_not_claimed=True, converted_raw_ledger_created=False,
        CPU_only_no_model_or_probe=True, producer_process_terminal_not_verified_by_this_consumer=True,
        source_proofs=source_proofs, step_proofs=proofs, original_ledger_read_sha256=hashlib.sha256(raw).hexdigest(),
        C_inputs_sha256=C_inputs_sha)
    require(report['total_pairs'] == len(steps) * 2560 and 'torch' not in sys.modules, 'final_count_or_CPU_only_failure')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True); parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--helper-sha256', required=True); args = parser.parse_args()
    try:
        require(file_sha(__file__) == args.helper_sha256, 'unpublished_statistics_helper')
        config, _ = read_metadata(args.config, args.config_sha256)
        require(sys.platform.startswith('linux') and socket.gethostname() == config['expected_node'] and
                os.environ.get('CUDA_VISIBLE_DEVICES') == '' and 'torch' not in sys.modules, 'not_bound_remote_CPU_process')
        require(config['selected_steps'] == SELECTED_STEPS, 'not_latest_human_selected8_scope')
        D = Path(config['diagnostic_base']); out = Path(config['output_new'])
        require(D.is_relative_to('/data/shaoshitong') and D.resolve() == D and out.resolve() == out and
                out.parent == D and not out.exists(), 'output_not_fresh_remote_diagnostic_directory')
        pairwise = load_bound_module('published_pairwise_pearson', config['pairwise_source_path'], config['pairwise_source_sha256'])
        C_helper = load_bound_module('published_C_gate_dependency', config['C_helper_path'], config['C_helper_sha256'])
        report = analyze_scope(config, pairwise, C_helper)
        out.mkdir(mode=0o700)
        target = out / 'selected-pair-statistics.json'
        with target.open('xb') as stream:
            stream.write(canonical(report)); stream.flush(); os.fsync(stream.fileno())
        print(json.dumps(dict(status=report['status'], selected_steps=report['selected_steps'],
            total_individual_pairs=report['total_pairs'], selected_step_pooled_Pearson=report['pooled']['Pearson'],
            report_remote_path=str(target), report_sha256=file_sha(target), raw_records_not_exported=True,
            not_all_TRAIN_steps_1_to_150=True, historical_online_R_bitwise_identity_not_claimed=True)))
        return 0
    except Exception as error:
        print(json.dumps(dict(status='HOLD', reason=str(error) if isinstance(error, Hold) else type(error).__name__,
            selected_scope_accepted=False, raw_exception_not_exported=True)))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
