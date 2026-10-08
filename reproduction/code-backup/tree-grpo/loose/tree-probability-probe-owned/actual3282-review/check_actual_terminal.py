"""Independent saved source-bound JSON/log measurements; no Torch/PT/HF/live IO."""
import base64
import collections
import copy
import gzip
import hashlib
import importlib.util
import json
import math
import re
from pathlib import Path

BASE = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = BASE / 'outputs/tree-instruct-300'
LOCAL = OUT / 'gpu3282'
PKG = BASE / 'work/probability-probe-49a6b35-29822f9-r3'
REMOTE = Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
RUN = REMOTE / 'runs/tree-probability-probe-3282'
checks = []
sha = lambda data: hashlib.sha256(data).hexdigest()
read = lambda path: json.loads(path.read_bytes())

def check(name, condition):
    if not condition:
        raise AssertionError(name)
    checks.append(name)

def near(a, b):
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)

raw_path = OUT / 'PPO_PROBE_GPU3282_TERMINAL_RAW.json'
check('exact completed terminal capture', raw_path.stat().st_size == 12974950 and sha(raw_path.read_bytes()) ==
      '95b245c1f1c7f768868c74fd027c370336ebaac7c18f4fe6c0d1f927868ad909')
raw = read(raw_path)
spec = importlib.util.spec_from_file_location('actual_r3_pure_contract', PKG / 'probe_package.py')
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)
files = {}
for path, entry in raw['files'].items():
    data = gzip.decompress(base64.b64decode(entry['gzip_base64']))
    check('captured SHA/length ' + path, sha(data) == entry['sha256'] and len(data) == entry['bytes'])
    remote = Path(path)
    relative = remote.relative_to(RUN) if RUN in remote.parents else Path(remote.name)
    check('local complete bytes ' + str(relative), (LOCAL / relative).read_bytes() == data)
    check('metadata only no tensor assets ' + str(relative), remote.suffix not in ('.pt', '.safetensors'))
    files[str(relative)] = {'bytes': len(data), 'sha256': sha(data)}
check('exact700 metadata files', len(files) == 700)
manifest = read(PKG / 'package-manifest.json')
bundle, launch, result, initial, termination = [read(LOCAL / x) for x in
    ('bundle.json', 'launch.json', 'result.json', 'initial-allocation.json', 'termination-cost.json')]
check('exact manifest', sha((PKG / 'package-manifest.json').read_bytes()) ==
      '014751600fbc6336dca1fa2d78fcef2a1dcb69730e4f6e7dcbbf17fb38035660')
for record in (result, launch, initial, raw['journal']['identity']):
    check('actual manifest/release identity ' + str(len(checks)), record['package_manifest_sha256'] ==
          '014751600fbc6336dca1fa2d78fcef2a1dcb69730e4f6e7dcbbf17fb38035660' and
          record['release_sha256'] == '9b995c701afd4fa787b0040a1ace47c9a58890232535554649fcb4a339e1f03b')
check('actual source755 and runtime bundle', launch['source_manifest_files'] == manifest['files'] and
      len(manifest['files']) == 755 and launch['bundle_sha256'] == result['bundle_sha256'] ==
      files['bundle.json']['sha256'] and result['launch_sha256'] == files['launch.json']['sha256'])
check('actual normalized complete bundle', P.digest(P.bundle_payload(bundle, '3282')) ==
      manifest['bundle_payload_sha256'] == launch['bundle_payload_sha256'])
check('actual include compiler environment', launch['compiler_environment'] == {
    'C_INCLUDE_PATH': str(REMOTE / 'python-dev/usr/include/python3.10') + ':' + str(REMOTE / 'python-dev/usr/include')})
control = raw['commands']['scontrol show job 3282']
queue = raw['commands']['squeue -h -j 3282 -o %i %j %T %M %N']
check('actual scheduler completed0:0', control['rc'] == 0 and control['stderr'] == '' and
      'JobState=COMPLETED ' in control['stdout'] and 'ExitCode=0:0' in control['stdout'] and
      'RunTime=00:04:15 ' in control['stdout'])
check('queue actually empty', queue == {'rc': 0, 'stdout': '', 'stderr': ''})
watch = [P.decode(x.encode()) for x in (LOCAL / 'probe-scheduler-observation-3282.jsonl').read_text().splitlines()]
check('read-only watcher captured actual completed', len(watch) == 17 and
      all(x['job_id'] == '3282' and x['mutation'] is False for x in watch) and
      watch[-1]['returncode'] == 0 and watch[-1]['scheduler_state'] == 'COMPLETED' and
      watch[-1]['terminal_state_observed'] is True and 'ExitCode=0:0' in watch[-1]['stdout'])
check('body completion actual zero optimizer not probability acceptance',
      result['status'] == 'OWNED_ZERO_UPDATE_FSDP_DIAGNOSTICS_COMPLETE_PENDING_INDEPENDENT_REVIEW' and
      result['diagnostic_body_complete'] is True and result['probe_exit_code'] == 0 and
      type(result['actual_optimizer_writes']) is int and result['actual_optimizer_writes'] == 0 and
      result['actual_training_accepted'] is False and result['probability_consistency_accepted'] is False and
      result['gpu_training_capacity_proven'] is False)
check('eight completed rank summaries', len(raw['rank_summaries']) == 8 and
      {x['rank'] for x in raw['rank_summaries']} == set(range(8)) and
      all(x['ready'] is True and x['json_files'] == 86 and len(x['comparisons']) == 84 for x in raw['rank_summaries']))
selection = bundle['selection']['selections']
groups = {}
monitor = collections.defaultdict(collections.Counter)
cpu_seconds = []
rank_ready = []
logical = set(bundle['hf_index']['weight_map'])
all_reports = []

def include(kind, micro, key, stats):
    # Combine per-row micro contrasts by the explicit same contrast, retaining
    # every physical token numerator/denominator rather than averaging rows.
    row_contrast = re.fullmatch(r'row-[0-9]+-micro([12])-vs-original-micro4', key)
    if row_contrast:
        check('row contrast original micro identity ' + str(len(checks)), int(row_contrast.group(1)) == micro)
        key = f'original_actor_micro{micro}_vs_original_micro4'
    check('finite diagnostic metric ' + str(len(checks)), stats['acceptance'] is False and
          stats['trial_001_is_diagnostic_only'] is True and stats['physical_effective_token_occurrences'] > 0 and
          all(math.isfinite(stats[k]) and stats[k] >= 0 for k in
              ('max_abs', 'mean_abs', 'rms', 'tree_weight_denominator', 'tree_weighted_abs_numerator',
               'tree_weighted_mean_abs')))
    n = stats['physical_effective_token_occurrences']
    check('metric counter/mean denominator ' + str(len(checks)),
          0 <= stats['above_trial_001_count'] <= n and stats['tree_weight_denominator'] > 0 and
          near(stats['tree_weighted_mean_abs'], stats['tree_weighted_abs_numerator'] / stats['tree_weight_denominator']) and
          stats['mean_abs'] <= stats['max_abs'] + 1e-12 and stats['rms'] <= stats['max_abs'] + 1e-12 and
          stats['abs_quantiles']['1.0'] == stats['max_abs'])
    g = groups.setdefault((kind, micro, key), [])
    g.append(stats)

for rank in range(8):
    directory = LOCAL / f'gpu-records/rank-{rank:02d}'
    ready_path = directory / 'ready.json'
    ready = P.decode(ready_path.read_bytes())
    warm = P.decode((directory / 'load-and-warmup.json').read_bytes())
    w, hf = warm['actual_native_warmup'], warm['hf']
    record = result['rank_records'][rank]
    check('actual full rank receipt ' + str(rank), record['rank'] == rank and record['record'] == ready and
          record['sha256'] == sha(ready_path.read_bytes()) and ready['rank'] == rank and ready['world_size'] == 8 and
          ready['status'] == 'SPECIFIED_INPUT_NATIVE_FSDP_DIAGNOSTICS_COMPLETE_NO_TRAINING_ACCEPTANCE' and
          ready['source_and_input_bundle_identity'] == bundle['identity'] and ready['original_forward_calls'] == 85 and
          ready['duplicate_alignment_calls'] == 0 and len(ready['results']) == 84 and
          ready['full_state_unchanged_since_postwarm_baseline'] is True and ready['original_module_modes_restored'] is True and
          ready['optimizer_committed'] == 0)
    check('rank zero mutators and no overclaim ' + str(rank), all(ready[k] is False for k in
          ('backward_called', 'optimizer_instantiated', 'scheduler_instantiated', 'scheduler_advanced',
           'training_acceptance', 'probability_consistency_acceptance', 'gpu_training_capacity_proven',
           'original_3268_effective_actor_identity_proven')))
    check('actual warmup full435/434/effective435 ' + str(rank), hf['serialized_count'] == 435 and
          hf['unique_parameter_count'] == 434 and len(logical) == 435 and set(hf['logical_records']) == logical and
          set(hf['serialized_to_unique']) == logical and len(set(hf['serialized_to_unique'].values())) == 434 and
          w['effective_weights']['logical_tensor_count'] == 435 and set(w['effective_weights']['records']) == logical and
          w['effective_weights']['trace_completed_successfully'] is True and
          hf['original_3268_effective_identity_proven'] is False and
          w['effective_weights']['original_3268_effective_weight_identity_proven'] is False)
    for name in logical:
        h, eff = hf['logical_records'][name], w['effective_weights']['records'][name]
        check('full HF/master/cast SHA ' + str(rank) + '/' + name,
              h['serialized'] == h['loaded_master'] and h['loaded_master']['dtype'] == 'torch.float32' and
              h['unique_parameter_key'] == hf['serialized_to_unique'][name] and
              all(eff[k] == h['expected_effective_bf16'][k] for k in
                  ('dtype', 'shape', 'full_shape_dtype_bytes_sha256')) and eff['dtype'] == 'torch.bfloat16' and
              eff['device'] == f'cuda:{rank}')
    tied = w['same_window_tie']
    check('actual tied alias same gather window ' + str(rank), tied['tied_alias_identity_proven'] is True and
          tied['historical_3268_identity_proven'] is False and len(tied['records']) == 1 and
          tied['records'][0]['same_window_alias'] is True and tied['records'][0]['logical_names'] ==
          ['model.embed_tokens.weight', 'lm_head.weight'] and tied['records'][0]['dtype'] == 'torch.bfloat16' and
          tied['records'][0]['shape'] == [151936, 2048] and tied['records'][0]['device'] == f'cuda:{rank}')
    pre, post = w['prewarm_state'], w['postwarm_baseline']
    check('warmup nonparameter state unchanged ' + str(rank),
          all(pre[k] == post[k] for k in ('buffers', 'gradients', 'modes', 'dropout')) and
          all(x is None for x in post['gradients'].values()) and w['lazy_buffer_changes'] == {})
    lazy = w['lazy_private_local_shard_initialization']
    check('exact37 legal private lazy creation ' + str(rank), len(lazy) == 37 and set(lazy) == set(post['parameters']))
    for key in post['parameters']:
        before, after = copy.deepcopy(pre['parameters'][key]), copy.deepcopy(post['parameters'][key])
        local = after.pop('private_local_shard')
        check('lazy None becomes unchanged full master ' + str(rank) + '/' + key,
              before.pop('private_local_shard') is None and before == after == local and
              lazy[key]['before'] is None and lazy[key]['after'] == local)
    check('actual mixedprecision and original normalized micros ' + str(rank),
          w['actual_master_dtype'] == 'torch.float32' and w['actual_mixed_precision'] ==
          {'param_dtype': 'torch.bfloat16', 'reduce_dtype': 'torch.float32', 'buffer_dtype': 'torch.float32'} and
          w['normalized_raw_old_micro_batch'] == 4 and w['normalized_actor_micro_batch'] == 2 and
          w['normalized_actor_mini_batch'] == 48 and w['historical_monitor_mode_proven'] is False and
          w['rng_unchanged'] is True and w['optimizer_instantiated'] is False and
          w['scheduler_instantiated'] is False and w['backward_called'] is False)
    reports = [P.decode(path.read_bytes()) for path in sorted(directory.glob('micro*-call*.json'))]
    check('exact84 planned micro calls ' + str(rank), len(reports) == 84 and
          collections.Counter(x['micro'] for x in reports) == {4: 12, 2: 24, 1: 48})
    planned = {(x['micro'], x['label']) for x in ready['results']}
    check('ready matches saved exact report labels ' + str(rank), planned ==
          {(x['micro'], f"micro{x['micro']}-call{x['call_index']:03d}") for x in reports})
    for report in reports:
        native, same = report['native_forward_receipt'], report['same_logits']
        check('actual report same native input and RNG0 ' + str(len(checks)), report['rank'] == rank and
              report['acceptance'] is False and report['duplicate_excluded_from_occurrence_totals'] is False and
              native['input_identity'] == report['input_identity'] and native['original_actor_calls'] == 1 and
              native['rng_unchanged'] is True and all(native[k] is False for k in
                  ('optimizer_instantiated', 'scheduler_instantiated', 'backward_called')) and
              native['observed_forward_kwargs'] == ['attention_mask', 'input_ids', 'position_ids', 'use_cache'] and
              native['observed_forward_input_dtypes'] == {'input_ids': 'torch.int64', 'position_ids': 'torch.int64'} and
              native['observed_model_training'] is False)
        check('same logits original selected exact and real dtypes ' + str(len(checks)),
              same['retained_logits_same_tensor'] is True and same['original_actor_calls'] == 1 and
              same['selected_native_exactly_matches_original'] is True and
              same['selected_native_vs_original_max_abs_delta'] == 0 and
              same['original_forward_logits_dtype'] == same['scaled_logits_dtype'] == 'torch.bfloat16' and
              same['original_actor_log_probs_dtype'] == 'torch.float32' and
              same['normalization_actual_output_dtypes'] == {'fp32_logsoftmax': ['torch.float32'],
                  'fp64_logsoftmax': ['torch.float64'], 'native_selected_rows': ['torch.float32'],
                  'original_dtype_logsoftmax': ['torch.float32']} and
              same['native_logprob_source']['source_sha256'] == bundle['native_logprob_sha256'] and
              same['acceptance'] is False and report['same_call_extreme_logits']['second_model_forward'] is False)
        kind, micro = report['selection']['kind'], report['micro']
        check('source original call rows retained ' + str(len(checks)), report['call']['micro'] == micro and
              report['call']['exclude_duplicate'] is False and len(report['call']['rows']) == micro and
              set(report['call']['rows']) <= set(report['selection']['row_indices']) and
              report['input_identity']['input_ids']['shape'][0] == micro)
        for key, stats in report['metrics'].items():
            if key == 'monitor_vs_sampled_ranges':
                c = stats
                check('monitor matching unknowns and duplicate probabilities not averaged ' + str(len(checks)),
                      c['acceptance'] is False and c['unmatched_placeholders_excluded'] is True and
                      c['duplicate_probabilities_averaged'] is False and c['particular_raw_generation_provenance_proven'] is False and
                      c['effective_occurrences'] == c['matched_occurrences'] + c['unknown_occurrences'] and
                      c['matched_raw_occurrences_denominator'] >= c['matched_occurrences'])
                for k in ('effective_occurrences', 'matched_occurrences', 'unknown_occurrences',
                          'matched_contexts_with_duplicate_raw_occurrences', 'matched_raw_occurrences_denominator'):
                    monitor[micro][k] += c[k]
                for variant, diffs in c['variants'].items():
                    for label, q in diffs.items():
                        check('matched-only denominator ' + str(len(checks)),
                              q['physical_effective_token_occurrences'] == c['matched_occurrences'])
                        include(kind, micro, 'matched_monitor/' + variant + '/' + label, q)
            else:
                include(kind, micro, key, stats)
        all_reports.append(report)
    cpu_seconds.append(ready['own_cpu_seconds'])
    rank_ready.append({'rank': rank, 'original_forward_calls': 85, 'comparisons': 84,
                       'serialized_count': 435, 'unique_parameter_count': 434, 'same_window_alias': True,
                       'optimizer_committed': 0, 'own_cpu_seconds': ready['own_cpu_seconds']})
check('global672 comparisons 680 native calls', len(all_reports) == 672 and
      sum(x['original_forward_calls'] for x in rank_ready) == 680)
for micro in (1, 2, 4):
    check('monitor occurrence totals repeated not new sampling ' + str(micro), dict(monitor[micro]) == {
        'effective_occurrences': 281703, 'matched_occurrences': 154467, 'unknown_occurrences': 127236,
        'matched_contexts_with_duplicate_raw_occurrences': 19257, 'matched_raw_occurrences_denominator': 175445})
aggregated = []
for (kind, micro, metric), rows in sorted(groups.items()):
    n = sum(x['physical_effective_token_occurrences'] for x in rows)
    denom = math.fsum(x['tree_weight_denominator'] for x in rows)
    weighted = math.fsum(x['tree_weighted_abs_numerator'] for x in rows)
    above = sum(x['above_trial_001_count'] for x in rows)
    aggregated.append({'kind': kind, 'micro': micro, 'metric': metric, 'calls': len(rows),
        'physical_effective_token_occurrences': n, 'max_abs': max(x['max_abs'] for x in rows),
        'mean_abs': math.fsum(x['mean_abs'] * x['physical_effective_token_occurrences'] for x in rows) / n,
        'rms': math.sqrt(math.fsum(x['rms'] ** 2 * x['physical_effective_token_occurrences'] for x in rows) / n),
        'above_trial_001_count': above, 'above_trial_001_fraction': above / n,
        'tree_weight_denominator': denom, 'tree_weighted_abs_numerator': weighted,
        'tree_weighted_mean_abs': weighted / denom})
owner_path = OUT / 'PPO_PROBE_GPU3282_NUMERICAL_SUMMARY.json'
owner = read(owner_path)
check('exact owner summary reference', sha(owner_path.read_bytes()) ==
      '8fec892e6442d6d8dd67f876ca8b5c3418ab84e525f1e0d3f6189ef8e39cff69' and
      owner['input_capture_sha256'] == sha(raw_path.read_bytes()) and owner['probability_accepted'] is False and
      owner['training_accepted'] is False and owner['global_quantiles_reconstructed'] is False)
owner_groups = {(x['kind'], x['micro'], x['metric']): x for x in owner['groups']}
check('all109 aggregate keys independently reconstructed', len(aggregated) == 109 and len(owner_groups) == 109 and
      set(groups) == set(owner_groups))
for row in aggregated:
    expected = owner_groups[(row['kind'], row['micro'], row['metric'])]
    check('aggregate exact integer counts ' + str(len(checks)), all(row[k] == expected[k] for k in
          ('calls', 'physical_effective_token_occurrences', 'above_trial_001_count')))
    check('aggregate physical and Tree-weight global denominators ' + str(len(checks)),
          all(near(row[k], expected[k]) for k in ('max_abs', 'mean_abs', 'rms', 'above_trial_001_fraction',
              'tree_weight_denominator', 'tree_weighted_abs_numerator', 'tree_weighted_mean_abs')))
resources = [P.decode(x.encode()) for x in (LOCAL / 'resources.jsonl').read_text().splitlines()]
processes = [P.decode(x.encode()) for x in (LOCAL / 'process-resources.jsonl').read_text().splitlines()]
for x in resources:
    P.guard_record(x, '3282', launch['gpu_uuids'], x['epoch'])
check('50 historical guard rows same dual15', len(resources) == result['guard_samples'] == 50 and
      result['final_guard_healthy'] is True)
for x in processes:
    check('233 saved own live adapter records ' + str(len(checks)), x['adapter']['job_id'] == '3282' and
          x['adapter']['phase'] == 'supervisor_live' and x['adapter']['original_background_processes_preserved'] is True and
          0 <= x['epoch'] - x['adapter']['guard_epoch'] <= 15)
check('saved process233 count', len(processes) == 233)
check('full cleanup and original8 exact final GPU process list', result['cleanup_residual'] == {} and
      result['original_background_pid_start_ticks_preserved'] is True and
      launch['background_pid_start_ticks'] == initial['background_pid_start_ticks'] ==
      {str(k): v for k, v in P.PROTECTED.items()} and
      result['final_gpu'] == initial['initial_gpu'])
check('requested32 actual192/GPU8 same allocation', result['requested_cpus'] == initial['requested_cpus'] == 32 and
      result['allocated_cpus'] == initial['allocated_cpus'] == 192 and
      result['allocated_gpus'] == initial['allocated_gpus'] == 8)
duration = result['end_epoch'] - result['start_epoch']
check('supervisor253.2964 cost separate from scheduler255', abs(duration - result['wall_seconds']) < 1e-5 and
      abs(result['gpu_hours'] * 3600 / 8 - duration) < 1e-5 and
      abs(result['allocated_cpu_hours'] * 3600 / 192 - duration) < 1e-5 and
      termination['start_epoch'] == result['start_epoch'] and
      result['start_epoch'] < termination['termination_epoch'] < result['end_epoch'])
check('native no compiler failure or suppression visible in saved logs',
      all('fatal error: Python.h' not in (LOCAL / name).read_text() and
          'GlobalDiagnosticRefusal' not in (LOCAL / name).read_text() for name in
          ('probe.log', 'tree-probability-probe-3282.out')))
proof = {'status': 'ACTUAL3282_SAVED_NATIVE_MEASUREMENT_TERMINAL_ZERO_UPDATE_METADATA_INDEPENDENTLY_VERIFIED',
    'checks_passed': len(checks), 'checks': checks, 'metadata_files': len(files), 'file_inventory': files,
    'terminal_capture_sha256': sha(raw_path.read_bytes()), 'owner_summary_sha256': sha(owner_path.read_bytes()),
    'package_manifest_sha256': launch['package_manifest_sha256'], 'release_sha256': launch['release_sha256'],
    'bundle_payload_sha256': launch['bundle_payload_sha256'], 'compiler_environment': launch['compiler_environment'],
    'rank_summary': rank_ready, 'independent_aggregate_groups': aggregated,
    'global_quantiles_reconstructed': False, 'micro_variants_are_repeated_not_new_sampling': True,
    'monitor_counts_by_micro': {str(k): dict(v) for k, v in monitor.items()},
    'actual_optimizer_writes': 0, 'probability_gate_passed': False, 'training_accepted': False,
    'historical3268_effective_weight_identity_proven': False,
    'normalization_only_fix_supported': False,
    'terminal': {'state': 'COMPLETED', 'exit_code': '0:0', 'elapsed_reported_seconds': 255,
                 'watcher_observed_terminal': True},
    'cost': {'supervisor_seconds': result['wall_seconds'], 'supervisor_gpu_hours': result['gpu_hours'],
        'supervisor_allocated_cpu_hours': result['allocated_cpu_hours'], 'allocated_cpus': 192, 'requested_cpus': 32,
        'scheduler_reported_seconds': 255, 'scheduler_gpu_hours_from_reported_elapsed': 255 * 8 / 3600,
        'scheduler_allocated_cpu_hours_from_reported_elapsed': 255 * 192 / 3600,
        'sum_actual_eight_rank_cpu_seconds': math.fsum(cpu_seconds),
        'sum_actual_eight_rank_cpu_hours': math.fsum(cpu_seconds) / 3600,
        'rank_cpu_scope': 'Per-rank process CPU only, not compiler/helper descendants or whole-job total CPU',
        'before_cleanup_seconds': termination['termination_epoch'] - result['start_epoch'],
        'cleanup_interval_seconds': result['end_epoch'] - termination['termination_epoch']},
    'resource_scope': {'guard_samples': 50, 'process_samples': 233,
        'max_foreign_memory_fraction': max(g['foreign_memory_fraction'] for x in resources for g in x['gpus']),
        'max_foreign_sm_upper': max(g['foreign_sm_upper'] for x in resources for g in x['gpus']),
        'whole_card_sampled_peak_bytes': max(g['total_used_memory_bytes'] for x in resources for g in x['gpus']),
        'max_guard_gap_seconds': max(b['epoch'] - a['epoch'] for a, b in zip(resources, resources[1:])),
        'training_capacity_proven': False},
    'cleanup_residual': {}, 'original_background_pid_start_ticks_preserved': True,
    'reviewer_scope': 'Local saved metadata/log bytes and source-bound receipts. No reload of PT/HF model assets or live node/FD queries',
    'reviewer_pt_hf_or_gpu_scheduler_remote_calls': 0,
}
(HERE / 'ACTUAL_TERMINAL_METADATA_PROOF.json').write_text(json.dumps(proof, indent=2, sort_keys=True) + '\n')
print(json.dumps({'checks_passed': len(checks), 'proof_sha256': sha((HERE / 'ACTUAL_TERMINAL_METADATA_PROOF.json').read_bytes()),
                  'aggregate_groups': len(aggregated), 'native_comparisons': len(all_reports)}))
