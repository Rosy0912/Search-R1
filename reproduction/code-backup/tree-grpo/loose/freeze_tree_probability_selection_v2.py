"""Freeze CPU-selected original micro4 groups and exact-context monitor ranges.

All tensor assets stay on h100-9. This does no model forward or optimizer work.
Selection rules are applied before any new GPU result. Unknown monitor tokens
carry an explicit mask; zero placeholders must never enter comparisons.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import socket
import sys
import time

import torch
import analyze_tree_sampling_calibration as A

CONTEXT_SHA = '28e665eb822651ff44559b2e9c6110d01542c5e6433db8dd4d07c659af7131ac'
PACKAGE_SHA = '84e3e18682afa880f1053a0025eda595fca332d550ec1f6905bc1334049b5a48'
READY_SHA = 'ba948bdec0b71a3c8ff5c61db1360840134e57fccaf71fb154449f30c300c468'
PARENT_HF = '56df6f31c42decee1b2302875b54db32d5cfa07db5e858446502c549cb0243ef'
PARENT_STATE = 'e59d8011df3b1248c0c548afeb9d4ad6e564c5f9b0f84f2147bb3f99ad182366'
CORE_COMMIT = '3d10326fe8cde3bf7a62bc9c7567b308cc6afc3a'
CPU_TERMINAL_SHA = 'cda5c22fbbcff3ff6584a07c589175e20ba620b4013723f18a296f1d94220e78'
CPU_REPORTS = {'position-analysis-3273.json': '326fe29839606f1d52dfc7cb68e8858f92e3f12f1178965343d88dbdde4bfff3',
               'retained-actions-3273.json': '6a08df127b7f4cccdb0be3f88c16036966318b490cc4caa1da2fbb190a12b0c0'}


def verify_cpu_terminal(value):
    A.require(value['status'] == 'ACTUAL_H1009_CPU3273_REPORTS_AND_TERMINAL_INDEPENDENTLY_VERIFIED' and
        value['job_id'] == 3273 and value['parent_calibration_job_id'] == 3268 and
        value['slurm_state'] == 'COMPLETED' and value['slurm_exit_code'] == '0:0' and
        value['host'] == 'h100-9' and value['gpu_count'] == 0 and value['cpu_count'] == 4 and
        value['ready_sha256'] == READY_SHA and value['package_manifest_sha256'] == PACKAGE_SHA,
        'Actual CPU3273 terminal identity differs')
    A.require(type(value['actual_optimizer_writes']) is int and value['actual_optimizer_writes'] == 0 and
        type(value['actual_model_forwards']) is int and value['actual_model_forwards'] == 0,
        'Actual CPU3273 was not zero forward/write')
    A.require(value['position_report_sha256'] == CPU_REPORTS['position-analysis-3273.json'] and
        value['action_report_sha256'] == CPU_REPORTS['retained-actions-3273.json'],
        'Terminal receipt binds other CPU reports')


def tensor_sha(value):
    A.require(isinstance(value, torch.Tensor) and value.device.type == 'cpu', 'CPU tensor required')
    x = value.detach().contiguous()
    h = hashlib.sha256(json.dumps([str(x.dtype), list(x.shape)], separators=(',', ':')).encode())
    h.update(x.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def plain_distribution(value):
    expected = dict(temperature=1., top_p=1., top_k=-1, min_p=0., presence_penalty=0.,
                    frequency_penalty=0., repetition_penalty=1., n=1)
    A.require(isinstance(value, dict) and all(value.get(k) == v for k, v in expected.items()) and
              not value.get('logits_processors'), 'Only original plain temperature1 sampling is comparable')


def describe_rows(tensors, asset):
    layout = A.normalize(tensors, raw=True)
    effective = layout['attention_mask'][:, -layout['width']:].bool()
    delta = layout['old_log_probs'].double() - layout['sampled_log_probs'].double()
    A.require(bool(torch.isfinite(delta[effective]).all()), 'Nonfinite selected source probabilities')
    result = []
    for row in range(layout['rows']):
        ds = delta[row][effective[row]].abs()
        valid_pos = layout['position_ids'][row][layout['attention_mask'][row].bool()].tolist()
        prefix_mask = layout['attention_mask'][row, :layout['response_start']].tolist()
        anomaly = valid_pos != list(range(len(valid_pos))) or prefix_mask != sorted(prefix_mask)
        row_sha = hashlib.sha256()
        for name in ('input_ids', 'attention_mask', 'position_ids', 'responses', 'old_log_probs', 'sampled_log_probs'):
            row_sha.update(name.encode() + tensor_sha(layout[name][row]).encode())
        result.append(dict(asset=asset, row=row, source_group=row // 4,
                           max_abs=float(ds.max()) if ds.numel() else None,
                           above_dot01=int((ds > .01).sum()), effective_tokens=ds.numel(),
                           above_dot01_fraction=float((ds > .01).sum()) / ds.numel() if ds.numel() else None,
                           full_attention_tokens=len(valid_pos), position_anomaly=anomaly,
                           row_tensor_sha256=row_sha.hexdigest()))
    return result


def choose_groups(rows, inventory):
    """Stable ties and complete original neighbors; at most five groups/rank."""
    A.require(rows and all(r['asset'] in inventory for r in rows), 'Missing source inventory')
    valid = [r for r in rows if r['effective_tokens'] > 0]
    A.require(valid, 'No valid source response tokens')
    tie = lambda r: (r['asset'], r['row'])
    max_row = min(valid, key=lambda r: (-r['max_abs'], *tie(r)))
    other_groups = [r for r in valid if (r['asset'], r['source_group']) != (max_row['asset'], max_row['source_group'])]
    tail_row = min(other_groups, key=lambda r: (-r['above_dot01_fraction'], -r['above_dot01'], *tie(r))) if other_groups else None
    long_row = min(valid, key=lambda r: (-r['full_attention_tokens'], *tie(r)))
    anomalies = [r for r in valid if r['position_anomaly']]
    low = [r for r in valid if r['max_abs'] <= .01]
    choices = [('max_raw_abs_delta', max_row), ('highest_above_dot01_fraction', tail_row),
               ('longest_full_attention_context', long_row),
               ('position_or_prompt_padding_anomaly', min(anomalies, key=tie) if anomalies else None),
               ('hash_first_low_delta_control', min(low, key=lambda r: (r['row_tensor_sha256'], *tie(r))) if low else None)]
    groups = {}; categories = []
    for category, row in choices:
        if row is None:
            categories.append(dict(category=category, status='ABSENT', reason='No source row meets frozen criterion'))
            continue
        key = (row['asset'], row['source_group'])
        item = groups.setdefault(key, dict(asset=key[0], source_group=key[1],
            row_indices=inventory[key[0]]['original_micro4_groups'][key[1]], criteria=[]))
        item['criteria'].append(category)
        categories.append(dict(category=category, status='SELECTED', **row))
    result = [groups[key] for key in sorted(groups)]
    A.require(sum(len(g['row_indices']) for g in result) <= 20, 'Selection budget exceeded')
    return result, categories


def monitor_companion(index, tensors, identity, asset, asset_sha):
    layout = A.normalize(tensors, raw=False)
    shape = layout['responses'].shape
    count = torch.zeros(shape, dtype=torch.int64)
    lo = torch.zeros(shape, dtype=torch.float64); hi = torch.zeros_like(lo)
    keys = torch.zeros((*shape, 48), dtype=torch.uint8)
    effective = layout['attention_mask'][:, -layout['width']:].bool() & layout['loss_mask'].bool()
    for row in range(layout['rows']):
        for key, offset, _, _, _ in A.causal_keys(layout, row, supervised=True):
            keys[row, offset] = torch.tensor(list(key), dtype=torch.uint8)
            found = index.values.get(key)
            if found is None:
                continue
            A.require(found.sampled_finite == found.count and math.isfinite(found.sampled_min) and
                      math.isfinite(found.sampled_max), 'Invalid matched raw probabilities')
            count[row, offset] = found.count
            lo[row, offset], hi[row, offset] = found.sampled_min, found.sampled_max
    matched = (count > 0) & effective
    A.require(not bool(((count > 0) & ~effective).any()), 'Companion includes ineffective tokens')
    return dict(identity=identity, monitor_asset=asset, monitor_asset_sha256=asset_sha,
        prefix_version=A.PREFIX_VERSION.decode().replace('\x00', '\\0'),
        matched_count=count, sampled_min=lo, sampled_max=hi, exact_context_key=keys,
        effective_mask=effective, matched_mask=matched,
        scope='All bound raw occurrences at exact complete attention-valid token+position/current-token key. '
              'Duplicates retain min/max/count, no mean and no particular generation provenance claim. '
              'Unknown sampled_min/max=0 are placeholders excluded by matched_mask.'), dict(
        effective_tokens=int(effective.sum()), matched_tokens=int(matched.sum()),
        unknown_tokens=int((effective & ~matched).sum()),
        matches_with_duplicate_raw_occurrences=int(((count > 1) & effective).sum()))


def save_new_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True); p.add_argument('--package', type=Path, required=True)
    p.add_argument('--cpu-reports', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cpu-terminal', type=Path, required=True)
    args = p.parse_args()
    A.require(socket.gethostname().split('.')[0] == 'h100-9' and not os.environ.get('CUDA_VISIBLE_DEVICES', ''),
              'Proper same-node CPU allocation with GPU hidden is required')
    A.require(A.file_sha(A.__file__) == CONTEXT_SHA, 'Pinned context helper changed')
    output = args.output.resolve()
    A.require(output.is_relative_to(A.TASK_ROOT/'instruct300') and not output.exists(), 'New task-local output required')
    started, cpu = time.perf_counter(), time.process_time(); torch.set_num_threads(4)
    run, package, ready, parent_id, sources = A.verify_bundle(args.run, args.package, PACKAGE_SHA, READY_SHA)
    terminal = A.json_bound(args.cpu_terminal, CPU_TERMINAL_SHA); verify_cpu_terminal(terminal)
    cpu_results = {}
    for name, sha in CPU_REPORTS.items():
        path = args.cpu_reports/name; report = A.json_bound(path, sha)
        cpu_results['position' if name.startswith('position-') else 'action'] = dict(path=str(path.resolve()), sha256=sha)
    A.require(package['parent_manifest']['sha256'] == PARENT_STATE, 'Healthy110 parent changed')
    identity = dict(ready_sha256=READY_SHA, source_sha256=PACKAGE_SHA,
        config_sha256=parent_id['config_sha256'],
        parent_hf_sha256=PARENT_HF, parent_state_manifest_sha256=PARENT_STATE,
        core_commit=CORE_COMMIT)
    inventory = {}; ranks = []; index = A.RawIndex(); monitors = []; raw_metadata = []; selection_categories = []
    for rank, worker in enumerate(sorted(ready['workers'], key=lambda w: w['rank'])):
        A.require(worker['rank'] == rank and worker['world_size'] == 8 and worker['optimizer_committed'] == 0,
                  'Worker identity/rank mismatch')
        rows = []
        calls = []
        for entry in worker['diagnostic_entries']:
            raw_path = Path(entry['path']).resolve(); snap = Path(entry['snapshot_path']).resolve()
            A.require(raw_path.is_relative_to(run/'output/sampling-consistency') and
                      snap == raw_path.with_suffix('.pt'), 'Raw evidence escaped current run')
            raw = A.json_bound(raw_path, entry['sha256'])
            plain_distribution(raw['distribution'])
            A.require(raw['rank'] == rank and raw['world_size'] == 8 and raw['optimizer_committed'] == 0 and
                raw['scheduler_step'] == 110 and raw['saved_rows'] == raw['local_rows'] and raw['enforced'] is False and
                all(raw.get(k) == v for k, v in parent_id.items() if k != 'run_id'), 'Raw identity/count mismatch')
            A.require(raw['snapshot_path'] == str(snap) and raw['snapshot_sha256'] == entry['snapshot_sha256'],
                      'Raw snapshot metadata mismatch')
            call = raw['generation_call']; calls.append(call)
            A.require(raw_path.name == f'rank-{rank:02d}-call-{call:05d}.json', 'Raw name mismatch')
            value = A.load_verified(snap, entry['snapshot_sha256'])
            A.require(value['input_ids'].shape[0] == raw['saved_rows'] and value['distribution'] == raw['distribution'],
                      'Raw snapshot rows/distribution mismatch')
            asset = snap.name
            A.require(asset not in inventory, 'Duplicate asset')
            n = raw['saved_rows']
            inventory[asset] = dict(path=str(snap), sha256=entry['snapshot_sha256'], kind='raw', rank=rank,
                row_count=n, generation_call=call,
                original_micro4_groups=[list(range(start, min(start+4, n))) for start in range(0, n, 4)],
                distribution=raw['distribution'])
            index.add_snapshot(value, str(snap)); rows.extend(describe_rows(value, asset))
            raw_metadata.append(dict(path=str(raw_path), sha256=entry['sha256']))
            del value
        A.require(calls == list(range(1, 9)), 'Incomplete raw call set')
        groups, categories = choose_groups(rows, inventory)
        monitor = Path(worker['monitor_path']).resolve()
        A.require(monitor.is_relative_to(run/'output/probability-calibration') and
                  monitor.name == f'rank-{rank:02d}-monitor.pt', 'Monitor identity/path mismatch')
        value = A.load_verified(monitor, worker['monitor_sha256'])
        A.require(value['input_ids'].shape[0] == 32, 'Monitor count changed')
        asset = monitor.name
        inventory[asset] = dict(path=str(monitor), sha256=worker['monitor_sha256'], kind='monitor', rank=rank, row_count=32)
        ranks.append(dict(rank=rank, raw_groups=[{k: v for k, v in g.items() if k != 'criteria'} for g in groups],
                          monitor=dict(asset=asset, row_indices=list(range(32)))))
        selection_categories.append(dict(rank=rank, categories=categories, selected_group_criteria=groups))
        monitors.append((asset, value))
    A.require(len(ranks) == 8 and index.rows == 11528 and index.valid_tokens == 3186179, 'Raw complete counts changed')
    A.require(set(raw_metadata[i]['path'] for i in range(len(raw_metadata))) ==
              {str(path.resolve()) for path in (run/'output/sampling-consistency').glob('rank-*-call-*.json')},
              'Extra/missing raw source records')
    for name, sha in sorted(package['files'].items()):
        if name.startswith('source/repo/'):
            path = args.package/name
            A.require(A.file_sha(path) == sha, 'Frozen package source changed')
            sources.append(dict(path=str(path.resolve()), sha256=sha))
    sources = list({s['path']: dict(path=s['path'], sha256=s['sha256']) for s in sources}.values())
    identity['source_files'] = sources
    inventory_hashes = {asset['path']: asset['sha256'] for asset in inventory.values()}
    identity['assets_sha256'] = hashlib.sha256(json.dumps(inventory_hashes, sort_keys=True).encode()).hexdigest()
    output.mkdir()
    companions = []
    for rank, (asset, value) in enumerate(monitors):
        tensor_data, counts = monitor_companion(index, value, identity, asset, inventory[asset]['sha256'])
        path = output/f'rank-{rank:02d}-monitor-sampled-ranges.pt'
        with path.open('xb') as stream:
            torch.save(tensor_data, stream); stream.flush(); os.fsync(stream.fileno())
        companions.append(dict(rank=rank, path=str(path), sha256=A.file_sha(path), monitor_asset=asset,
                               monitor_asset_sha256=inventory[asset]['sha256'], **counts))
    A.require(not torch.cuda.is_initialized(), 'Unexpected CUDA initialization')
    result = dict(schema_version=1, asset_root=str(A.TASK_ROOT), frozen=True,
        status='SELECTION_FROZEN_FROM_CPU_RESULTS', identity=identity, inventory=inventory,
        selections=ranks, monitor_companions=companions, source_files=sources, raw_metadata=raw_metadata,
        selection_categories=selection_categories,
        provenance=dict(source_commit=parent_id['code_commit'], model_identity=parent_id['model_identity'],
            data_identity=parent_id['data_identity'], parent_calibration_job=3268),
        cpu_results=cpu_results, cpu_terminal=dict(path=str(args.cpu_terminal.resolve()), sha256=CPU_TERMINAL_SHA),
        helper_sha256=A.file_sha(__file__), context_helper_sha256=CONTEXT_SHA,
        raw_complete_counts=dict(rows=index.rows, effective_tokens=index.valid_tokens),
        selection_rule='Per rank choose maximum absolute delta, maximum fraction |delta|>.01, longest attention context, '
            'recorded position/prompt padding anomaly if present, and smallest row hash among rows max|delta|<=.01 if present; '
            'ties asset/row ascending, complete original adjacent micro4, deduplicate groups, <=20 raw rows. '
            'An absent category is not substituted. Monitor all32 rows/rank.',
        actual_model_forwards=0, actual_optimizer_writes=0, probability_consistency_accepted=False,
        long_training_ready=False, cost=dict(wall_seconds=time.perf_counter()-started,
            process_cpu_seconds=time.process_time()-cpu, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*
            (1 if sys.platform == 'darwin' else 1024), cuda_initialized=False,
            slurm_job_id=os.environ.get('SLURM_JOB_ID'), threads=4))
    manifest = output/'selection.json'; save_new_json(manifest, result)
    print(json.dumps(dict(status=result['status'], selection_path=str(manifest), selection_sha256=A.file_sha(manifest),
        rank_rows=[sum(len(g['row_indices']) for g in r['raw_groups']) for r in ranks],
        monitor_companions=companions, cost=result['cost'], actual_optimizer_writes=0), sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
