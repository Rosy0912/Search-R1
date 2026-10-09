"""Local reporting of closed P/J precision observations and preserved retries."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time
import types

sys.dont_write_bytecode = True
PINS = {
    'aggregate': 'b7a264119fd18b9ac744f452c86e8deb6db550c318050178e94adb38a1f39c8a',
    'trainer': '4dc352f5091594ab5bcae71ecc17516afd8607601b5932216cbda66befc5409c',
    'precision': '7dd23d6378c91cb33c730241fe5cd63618768aa95eea6136bf571a7cb64358bc',
    'retry_state': 'f3b4abe482b1dbc430959d4a189f282cdfeeca7372e57b78a9492c14e836c8fe',
    'dtype_launcher': '5b8bfe675ee5b2186aee45a598e56991b2738becb15b140cf8d0cdbb89cbeb7a',
}
ADAPTER_SHA = '8af9579f1663f1febcb6cac6f8d6e392e4e5aab55e58dbbb94858617a47217bd'
ORIGINAL_MANIFEST = '8b22e616d470d79022cc6a0b07dcb0774df21b8e35ebd7dbcaa9b19150236a41'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def same_ref(left, right):
    return all(left[k] == right[k] for k in ('path', 'sha256', 'bytes'))


def load_sources(refs):
    """Only the pinned stdlib backup reader is executed; producer bytes are evidence."""
    require(set(refs) == set(PINS), 'Exact five source references required')
    module = types.ModuleType('_pinned_precision_history_backup_reader')
    proofs = {}
    for name, expected in PINS.items():
        ref = refs[name]; path = Path(ref['path'])
        require(ref['sha256'] == expected and path.is_absolute() and '..' not in path.parts
                and all(not q.is_symlink() for q in (path, *path.parents)), 'Pinned source/path differs')
        before = path.stat(); raw = path.read_bytes(); after = path.stat()
        keys = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
        require(all(getattr(before, k) == getattr(after, k) for k in keys)
                and hashlib.sha256(raw).hexdigest() == expected, 'Source changed or SHA differs')
        proofs[name] = dict(path=str(path), sha256=expected, bytes=len(raw))
        if name == 'aggregate':
            module.__file__ = str(path)
            exec(compile(raw, str(path), 'exec'), module.__dict__)
    require('torch' not in sys.modules, 'This reader must remain stdlib CPU-only')
    return module, proofs


def bound(backups, ref):
    require(ref['path'] in backups.files, 'Referenced JSON/state absent from exact closed inventory')
    row = backups.files[ref['path']][0]
    require(same_ref(ref, row), 'Reference SHA/bytes differs from accepted inventory')
    return {k: row[k] for k in ('path', 'sha256', 'bytes')}


def scaler(value):
    require(value['version'] == 'synchronous-loss-scaler-v1' and value['enabled'] is True,
            'Expected P/J synchronized scaler state')
    require(type(value['scale']) in (int, float) and math.isfinite(value['scale']) and value['scale'] > 0
            and value['growth_factor'] == 2 and value['backoff_factor'] == .5
            and value['growth_interval'] == 2000 and type(value['growth_tracker']) is int
            and 0 <= value['growth_tracker'] < 2000, 'Invalid recorded scaler policy/state')
    return value


def precision(value):
    require(value['mode'] == 'fp32_master_fp16'
            and value['optimizer_master_dtypes'] == ['torch.float32']
            and value['actual_model_parameter_dtypes'] == ['torch.float32'], 'Recorded P/J master/storage dtype differs')
    events = value['events']; dropped = value['events_dropped']; counts = value['dtype_counts']
    require(isinstance(events, list) and len(events) <= 4096 and type(dropped) is int and dropped >= 0
            and (dropped == 0 or len(events) == 4096)
            and value['count_scope'] == 'All installed module observations counted; detailed events cap4096',
            'Invalid detailed-event coverage/cap')
    keys = {}; observed = Counter(); categories = Counter()
    for row in counts:
        key = (row['module'], row['role'], row['stage'], tuple(row['input_dtypes']), tuple(row['output_dtypes']))
        require(key not in keys and type(row['count']) is int and row['count'] > 0
                and row['role'] in ('teacher', 'outer_old', 'current', 'unassigned',
                                    'current_head', 'teacher_head', 'outer_old_head')
                and all(re.fullmatch(r'torch\.[a-zA-Z0-9_]+', d) for d in (*row['input_dtypes'], *row['output_dtypes']))
                and row['stage'] in ('forward', 'backward_recompute'), 'Invalid/duplicate dtype observation counter')
        keys[key] = row['count']
        categories[(row['role'], row['stage'], tuple(row['input_dtypes']), tuple(row['output_dtypes']))] += row['count']
    for event in events:
        key = (event['module'], event['role'], event['stage'], tuple(event['input_dtypes']), tuple(event['output_dtypes']))
        require(key in keys and type(event['explicit_activation_cast']) is bool, 'Detailed event/counter differs')
        observed[key] += 1
    require(sum(keys.values()) == len(events) + dropped
            and all(n <= keys[k] for k, n in observed.items()), 'Full observation count differs from capped events')
    return dict(detailed_events=len(events), events_dropped=dropped, observations=sum(keys.values()),
                counters=[dict(role=k[0], stage=k[1], input_dtypes=list(k[2]), output_dtypes=list(k[3]), count=n)
                          for k, n in sorted(categories.items())], scaler=scaler(value['scaler']))


def coverage(attempts, terminal_count, nonzero_terminals):
    """No absent rank/attempt can silently become zero historical overflow."""
    gaps = []
    for name, item in sorted(attempts.items()):
        missing = sorted(set(range(8)) - set(item['precision']))
        if missing:
            gaps.append(dict(attempt=name, missing_precision_ranks=missing))
        if not item['source_load']:
            gaps.append(dict(attempt=name, missing_source_load=True))
    if len(attempts) != terminal_count:
        gaps.append(dict(attempt_directories=len(attempts), training_invocation_terminals=terminal_count,
                         reason='No producer UUID-to-PID mapping; missing/pre-directory invocation possible'))
    if nonzero_terminals:
        gaps.append(dict(nonzero_training_invocation_terminals=nonzero_terminals,
                         reason='Failure history cannot be inferred complete from end-of-invocation precision files'))
    return gaps


def prepare_group(spec, facts, helper):
    variant = spec['variant']; require(variant in ('P', 'J'), 'Only explicit P/J closed groups supported')
    config, config_ref = helper.stable_json(spec['config']['path'], spec['config']['sha256'])
    run = config['run_id']; output = Path(config['paths']['output_root'])
    require(config['schema'] == 'opsd-ablation-config-v1' and config['variant'] == variant
            and config['precision']['mode'] == 'fp32_master_fp16' and config['scientific']['outer_iterations'] == 10
            and config['helper_source_manifest']['sha256'] == ORIGINAL_MANIFEST
            and config_ref['sha256'] in {r['sha256'] for r in facts['configs']}, 'Original P/J config/facts identity differs')
    outers = sorted([o for o in facts['outers'] if o['variant'] == variant], key=lambda o: o['outer'])
    require(len(outers) == 10 and [o['outer'] for o in outers] == list(range(10))
            and all(o['status'] == 'COMPLETE_FACTS' and o['run_id'] == run for o in outers), 'Full ten closed native outers required')
    require(len(spec['backups']) == 1 and spec['backups'][0]['kind'] == 'gzip', 'One explicit accepted generic closed10 gzip required')
    backup = spec['backups'][0]
    proof, proof_ref = helper.stable_json(backup['complete']['path'], backup['complete']['sha256'])
    require(proof['schema'] == 'opsd-ablation-closed-raw-reader-adapter-v1' and proof['variant'] == variant
            and proof['closed_outers'] == list(range(10)) and proof['config']['sha256'] == config_ref['sha256']
            and proof['adapter_source']['sha256'] == ADAPTER_SHA, 'Accepted generic adapter/closed config scope differs')
    require(len(proof['closures']['outers']) == 10, 'Closed backup omits an outer closure')
    for outer, closure in zip(outers, proof['closures']['outers']):
        require(closure['outer_index'] == outer['outer'] and same_ref(closure['complete'], outer['sources']['completion'])
                and same_ref(closure['checkpoint'], outer['sources']['checkpoint']), 'Raw backup/facts closed outer differs')
    backups = helper.FixedBackups(spec['backups'], run)
    wanted = {}; attempts = {}; terminals = []; before = exposure = 0
    for outer in outers:
        index = outer['outer']; c = outer['counts']; contract = dict(direction='student_to_target_reverse',
            target_version='opsd-ablation-reverse-v1', variant=variant, precision='fp32_master_fp16', mixture_alpha=.1 if variant == 'J' else 0.)
        require(outer['contract'] == contract and c['optimizer_updates_before'] == before
                and c['optimizer_updates_delta'] == len(outer['updates']) == math.ceil(c['K']/32)
                and c['optimizer_updates'] == before + len(outer['updates'])
                and c['cumulative_pair_exposure'] == exposure + c['K'], 'Closed counter/science chain differs')
        for n, update in enumerate(outer['updates'], 1):
            require(update['update'] == before+n, 'Committed update sequence differs')
            wanted[update['source']['path']] = bound(backups, update['source'])
        for key in ('outer_manifest', 'order', 'completion'):
            wanted[outer['sources'][key]['path']] = bound(backups, outer['sources'][key])
        before = c['optimizer_updates']; exposure += c['K']
    prefix = str(output / 'outers') + '/'
    for path, (row, _) in backups.files.items():
        if path.startswith(prefix):
            rel = path[len(prefix):]
            m = re.fullmatch(r'outer-(\d{5})/training/(attempt-[0-9a-f]{32})/(.+)', rel)
            if '/training/attempt-' in rel:
                require(m is not None and 0 <= int(m[1]) < 10, 'Unknown training attempt path')
            if m:
                outer, name, tail = int(m[1]), m[2], m[3]; key = (outer, name)
                attempts.setdefault(key, dict(precision={}, source_load=False, files=[]))['files'].append(path)
                if re.fullmatch(r'precision-rank-[0-7]\.json', tail) or tail == 'source100-load-proof.json' or re.fullmatch(r'update-\d{8}\.json', tail) or re.fullmatch(r'retry-update-\d{8}-attempt-\d{4}/manifest\.json', tail):
                    wanted[path] = {k: row[k] for k in ('path', 'sha256', 'bytes')}
        if path.startswith(str(output / 'phase-exits') + '/') and path.endswith('.json'):
            wanted[path] = {k: row[k] for k in ('path', 'sha256', 'bytes')}; terminals.append(path)
    history = str(output/'child-history.jsonl')
    require(history in backups.files, 'Original complete child-history inventory missing')
    wanted[history] = bound(backups, backups.files[history][0])
    require(attempts, 'Closed inventory contains no training attempt; zero is not assumed')
    return dict(variant=variant, config=config, config_ref=config_ref, run=run, output=output, outers=outers,
                backups=backups, wanted=wanted, attempts=attempts, terminals=terminals, proof=proof_ref,
                history=history, repair_receipts=spec.get('dtype_repair_receipts', []))


def analyze(group, values, helper):
    backups = group['backups']; run = group['run']; attempts = group['attempts']; private = []; gaps = []
    total = Counter(); dtype_counts = Counter(); failures = Counter(); retry_rows = []; updates = {}; committed = {}
    config_sha = group['config_ref']['sha256']; terminal_count = nonzero = 0
    for outer in group['outers']:
        index = outer['outer']; sources = outer['sources']; manifest = values[sources['outer_manifest']['path']]
        order = values[sources['order']['path']]; complete = values[sources['completion']['path']]
        require(manifest['run_id'] == run and manifest['outer_index'] == index and manifest['config_sha256'] == config_sha
                and manifest['contract'] == outer['contract'] and order['outer_index'] == index
                and complete['run_id'] == run and complete['contract'] == outer['contract'], 'Outer JSON identity differs')
        for update in outer['updates']:
            ref = update['source']; value = values[ref['path']]; step = value['scaler']
            require(value['schema'] == 'opsd-online-real-update-v1' and value['accepted'] is True
                    and value['run_id'] == run and value['outer_index'] == index and value['contract'] == outer['contract']
                    and value['optimizer_updates'] == update['update'] and value['outer_old_adapter'] == manifest['outer_old_adapter']
                    and step['accepted'] is True and step['overflow'] is False and step['adam_advanced'] is True
                    and type(value['retries']) is int and value['retries'] >= 0, 'Committed scaler/identity differs')
            ranks = value['all_rank_actual_token_metrics']; after = scaler(step['scaler_after'])
            require(len(ranks) == 8 and {r['rank'] for r in ranks} == set(range(8))
                    and all(r['scaler'] == after for r in ranks), 'Committed eight-rank scaler differs')
            committed[ref['path']] = (index, value['optimizer_updates']); updates[(index, value['optimizer_updates'])] = value
            total['committed_updates'] += 1; total['committed_recorded_retries'] += value['retries']
            private.append(dict(category='committed_scaler_step', source=ref, outer=index,
                update=value['optimizer_updates'], retries=value['retries'], scaler=step,
                pre_update_adapter_sha256=value['pre_update_adapter_sha256'],
                post_update_adapter_sha256=value['post_update_adapter_sha256'], outer_old_adapter=value['outer_old_adapter']))
        ends = complete['all_rank_scaler_state']
        require(len(ends) == 8 and all(scaler(x) == ends[0] for x in ends) and ends == outer['scaler_end'], 'Final eight-rank scaler/facts differs')
    for key, item in sorted(attempts.items()):
        index, name = key; outer = group['outers'][index]; base = group['output']/'outers'/('outer-%05d'%index)/'training'/name
        manifest = values[outer['sources']['outer_manifest']['path']]; order = values[outer['sources']['order']['path']]
        for path in item['files']:
            tail = str(Path(path).relative_to(base))
            if path not in values:
                continue
            value = values[path]; ref = group['wanted'][path]
            match = re.fullmatch(r'precision-rank-([0-7])\.json', tail)
            if match:
                rank = int(match[1]); observed = precision(value); item['precision'][rank] = observed
                total.update({k: observed[k] for k in ('detailed_events', 'events_dropped', 'observations')})
                for row in observed['counters']:
                    dtype_counts[(row['role'], row['stage'], tuple(row['input_dtypes']), tuple(row['output_dtypes']))] += row['count']
            elif tail == 'source100-load-proof.json':
                require(value['schema'] == 'opsd-online-source100-fresh-load-v1' and value['run_id'] == run
                        and value['outer_index'] == index and value['source_model_identity'] == group['config']['source_model_manifest']['sha256'], 'Attempt source-load identity differs')
                item['source_load'] = True
            elif re.fullmatch(r'update-\d{8}\.json', tail):
                require(value['run_id'] == run and value['outer_index'] == index and value['contract'] == outer['contract'], 'Attempt update identity differs')
                private.append(dict(category='update', source=ref, committed=path in committed,
                                    outer=index, attempt=name, update=value['optimizer_updates']))
                total['uncommitted_update_receipts'] += path not in committed
            else:
                match = re.fullmatch(r'retry-update-(\d{8})-attempt-(\d{4})/manifest\.json', tail)
                require(match is not None, 'Unknown selected attempt JSON')
                target, retry = int(match[1]), int(match[2]); cursor = value['cursor']; before = outer['counts']['optimizer_updates_before']
                require(value['schema'] == 'opsd-ablation-overflow-retry-v1' and value['accepted'] is True
                        and value['run_id'] == run and value['config_sha256'] == config_sha and value['contract'] == outer['contract']
                        and value['outer_old_adapter'] == manifest['outer_old_adapter']
                        and before < target <= outer['counts']['optimizer_updates']
                        and cursor['outer_index'] == index and cursor['next_pair_cursor'] == min(32*(target-before-1), outer['counts']['K'])
                        and cursor['usable_pair_count'] == outer['counts']['K'] and cursor['order_sha256'] == outer['sources']['order']['sha256']
                        and cursor['encoded_sha256'] == order['encoded_sha256']
                        and value['checkpoint']['path'] == str(group['output']/'checkpoints'/('update-%08d-outer-%05d'%(target-1,index))/'manifest.json'), 'Retry checkpoint/cursor/old/contract join differs')
                scaler(value['scaler_after']); ranks = value['ranks']
                require(len(ranks) == 8 and {r['rank'] for r in ranks} == set(range(8)), 'Retry manifest missing/duplicate rank')
                for row in ranks:
                    require(row['file']['path'] == str(Path(path).parent/('rank-%d.pt'%row['rank'])), 'Retry rank path differs')
                    bound(backups, row['file'])
                require(value['failure_kind'] in ('gradient_overflow', 'forward_or_logging_failure'), 'Unknown retry failure kind')
                failures[value['failure_kind']] += 1
                retry_rows.append(dict(outer=index, attempt=name, target_update=target, retry=retry,
                    failure_kind=value['failure_kind'], cursor=cursor, checkpoint=value['checkpoint'],
                    scaler_after=value['scaler_after'], ranks=ranks, source=ref))
        private.append(dict(category='attempt', outer=index, attempt=name, source_files=item['files'],
                            precision=item['precision'], source_load_present=item['source_load']))
    retry_by_attempt = Counter((r['outer'], r['attempt'], r['target_update']) for r in retry_rows if r['failure_kind'] == 'gradient_overflow')
    for path, (index, update) in committed.items():
        attempt = Path(path).parent.name; declared = updates[(index, update)]['retries']; observed = retry_by_attempt[(index, attempt, update)]
        require(declared == observed, 'Committed same-invocation retry declaration/inventory differs')
        require(sorted(r['retry'] for r in retry_rows if (r['outer'], r['attempt'], r['target_update']) == (index, attempt, update)
                       and r['failure_kind'] == 'gradient_overflow') == list(range(declared)), 'Same-invocation retry indices omitted/duplicate')
    retry_paths = {r['file']['path'] for x in retry_rows for r in x['ranks']}
    orphan = [p for item in attempts.values() for p in item['files'] if re.search(r'/retry-update-\d{8}-attempt-\d{4}/rank-[0-7]\.pt$', p) and p not in retry_paths]
    if orphan:
        gaps.append(dict(orphan_retry_rank_payload_refs=orphan))
    terminal_owners = {}
    for path in group['terminals']:
        value = values[path]
        if value.get('phase') != 'training-running':
            continue
        require(value['schema'] == 'opsd-ablation-actual-phase-terminal-v1' and value['run_id'] == run
                and value['variant'] == group['variant'] and value['config_sha256'] == config_sha
                and len(value['child_exit_evidence']) == 1, 'Original training terminal identity differs')
        row = value['child_exit_evidence'][0]; identity = row['identity']; rc = row['actual_wait_returncode']
        require(identity['uid'] == 1005 and identity['run_id'] == run and type(identity['starttick']) is int
                and type(rc) is int, 'Actual terminal child identity/returncode absent')
        bound(backups, row['log']); terminal_count += 1; nonzero += rc != 0
        owner = (identity['pid'], identity['uid'], identity['starttick'])
        require(owner not in terminal_owners, 'Duplicate original training terminal identity')
        terminal_owners[owner] = group['wanted'][path]
        private.append(dict(category='actual_training_terminal', source=group['wanted'][path], child=identity, actual_wait_returncode=rc))
    history_owners = set(); selected_retries = []; invocations_by_outer = Counter()
    for row in values[group['history']]:
        if row['phase'] != 'training-running':
            continue
        identity = row['identity']; command = row['command']; flags = {}
        for flag in ('--config', '--config-sha256', '--outer-manifest', '--outer-manifest-sha256',
                     '--resume-checkpoint', '--resume-checkpoint-sha256', '--retry-capsule-manifest', '--retry-capsule-sha256'):
            require(command.count(flag) <= 1, 'Duplicate original training CLI flag')
            if flag in command:
                n = command.index(flag); require(n+1 < len(command), 'Incomplete original training CLI')
                flags[flag] = command[n+1]
        candidates = [o for o in group['outers'] if o['sources']['outer_manifest']['path'] == flags.get('--outer-manifest')
                      and o['sources']['outer_manifest']['sha256'] == flags.get('--outer-manifest-sha256')]
        require(len(candidates) == 1 and flags.get('--config-sha256') == config_sha
                and '--nproc_per_node=8' in command and identity['uid'] == 1005 and identity['run_id'] == run,
                'Training history command/run/config/outer differs')
        owner = (identity['pid'], identity['uid'], identity.get('starttick')); history_owners.add(owner)
        invocations_by_outer[candidates[0]['outer']] += 1
        if owner not in terminal_owners:
            gaps.append(dict(training_history_owner_without_actual_terminal=owner))
        if '--retry-capsule-manifest' in flags:
            matches = [r for r in retry_rows if r['source']['path'] == flags['--retry-capsule-manifest']
                       and r['source']['sha256'] == flags.get('--retry-capsule-sha256')]
            require(len(matches) == 1 and matches[0]['checkpoint']['path'] == flags.get('--resume-checkpoint')
                    and matches[0]['checkpoint']['sha256'] == flags.get('--resume-checkpoint-sha256'),
                    'Recorded recovery CLI does not select its bound retry/accepted checkpoint')
            selected_retries.append(dict(outer=candidates[0]['outer'], child=identity, retry_manifest=matches[0]['source'],
                resume_checkpoint=matches[0]['checkpoint'], payload_restore_independently_verified=False))
        private.append(dict(category='training_invocation', outer=candidates[0]['outer'], child=identity,
            command=command, terminal=terminal_owners.get(owner), original_log=row['log']))
    if history_owners != set(terminal_owners):
        gaps.append(dict(reason='Original training history/terminal identity sets differ'))
    for index in range(10):
        actual_attempts = sum(i == index for i,_ in attempts)
        if actual_attempts != invocations_by_outer[index]:
            gaps.append(dict(outer=index, attempt_directories=actual_attempts,
                             recorded_invocations=invocations_by_outer[index]))
    gaps += coverage(attempts, terminal_count, nonzero)
    repairs = []
    for ref in group['repair_receipts']:
        value, proof = helper.stable_json(ref['path'], ref['sha256']); rank = value['rank']
        candidates = [o for o in group['outers'] if all(o['sources']['outer_manifest'][k] == value['outer_manifest'][k] for k in ('path','sha256'))]
        require(len(candidates) == 1 and value['schema'] == 'opsd-ablation-dtype-repair-execution-v1' and value['accepted'] is True
                and value['run_id'] == run and value['variant'] == group['variant'] and value['uid'] == 1005
                and value['config']['sha256'] == config_sha and rank in range(8) and value['world_size'] == 8
                and value['tensors'] == 504 and value['old_storage_dtypes_after'] == ['torch.float32']
                and value['promotion_before_original_setter'] is True and value['current_unchanged'] is True
                and value['repair_source_manifest']['sha256'] == 'eee5adf6621f85274584412c19be642d8a7e4a8f6f005c84b8a90a972f9803b9'
                and value['original_digest_and_frozen_guard_not_modified'] is True
                and value['original_outer_start'] == values[candidates[0]['sources']['outer_manifest']['path']]['outer_old_adapter']
                and value['actual_old_tensor_sha256'] == value['original_input_tensor_sha256'], 'External dtype-repair receipt differs')
        repairs.append(dict(source=proof, outer=candidates[0]['outer'], rank=rank, pid=value['pid'], original_outer_start=value['original_outer_start']))
    result = dict(variant=group['variant'], status='COMPLETE_RECORDED_ATTEMPT_COVERAGE' if not gaps else 'PARTIAL',
        closed_outers=10, attempts=len(attempts), precision_rank_files=sum(len(a['precision']) for a in attempts.values()),
        expected_precision_rank_files=8*len(attempts), recorded_training_history_invocations=len(history_owners),
        actual_training_invocation_terminals=terminal_count, nonzero_training_terminals=nonzero,
        counts=dict(total), preserved_failure_manifests=dict(failures),
        historical_gradient_overflow_manifests=failures['gradient_overflow'],
        recorded_cross_invocation_retry_selections=len(selected_retries),
        zero_preserved_gradient_overflow_proven=not gaps and failures['gradient_overflow'] == 0,
        dtype_observations=[dict(role=k[0], stage=k[1], input_dtypes=list(k[2]), output_dtypes=list(k[3]), count=n)
                            for k,n in sorted(dtype_counts.items())],
        dtype_repair_receipts_reviewed=len(repairs), dtype_repair_external_inventory_complete=False,
        missing_evidence_count=len(gaps), rng_or_CUDA_resume_verified=False)
    result['outers'] = []
    for outer in group['outers']:
        index = outer['outer']; selected = [a for (i,_),a in attempts.items() if i == index]
        observed = [p for a in selected for p in a['precision'].values()]
        states = [v['scaler']['scaler_after'] for (i,_),v in updates.items() if i == index]
        states += [r['scaler_after'] for r in retry_rows if r['outer'] == index]
        states += [p['scaler'] for p in observed]
        counters = [row for p in observed for row in p['counters']]
        result['outers'].append(dict(outer=index, attempts=len(selected), precision_rank_files=len(observed),
            detailed_events=sum(p['detailed_events'] for p in observed), events_dropped=sum(p['events_dropped'] for p in observed),
            module_observations=sum(p['observations'] for p in observed),
            backward_recompute_observations=sum(r['count'] for r in counters if r['stage'] == 'backward_recompute'),
            fp16_output_observations=sum(r['count'] for r in counters if 'torch.float16' in r['output_dtypes']),
            committed_updates=outer['counts']['optimizer_updates_delta'], cumulative_updates=outer['counts']['optimizer_updates'],
            preserved_failures=dict(Counter(r['failure_kind'] for r in retry_rows if r['outer'] == index)),
            recorded_after_scale_min=min(s['scale'] for s in states) if states else None,
            recorded_after_scale_max=max(s['scale'] for s in states) if states else None,
            final_scaler=outer['scaler_end'][0], failure_before_scale_payload_read=False))
    return result, dict(attempts=private, retries=retry_rows, recorded_retry_selections=selected_retries, external_dtype_repairs=repairs, gaps=gaps,
                        config=group['config_ref'], backups=backups.inputs, selected_reads=backups.reads)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'manifest-sha256', 'output'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--plan-only', action='store_true', help='Read small inputs/inventories; do not traverse archives')
    args = p.parse_args(); out = Path(args.output)
    require(out.is_absolute() and not out.exists() and '..' not in out.parts
            and all(not q.is_symlink() for q in out.parents), 'Fresh canonical output directory required')
    # Bootstrap source SHA is pinned before executing its stdlib JSON/file reader.
    raw = Path(args.manifest).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == args.manifest_sha256, 'Input manifest SHA differs')
    manifest = json.loads(raw); helper, sources = load_sources(manifest['sources'])
    manifest, mp = helper.stable_json(args.manifest, args.manifest_sha256)
    require(manifest['schema'] == 'opsd-ablation-recorded-precision-history-inputs-v1' and manifest['ready'] is True,
            'Explicit ready closed-group inputs required')
    facts, fp = helper.stable_json(manifest['facts']['path'], manifest['facts']['sha256'])
    require(facts['schema'] == 'opsd-ablation-outer-facts-v1' and facts['accepted'] is True, 'Accepted native facts required')
    specs = manifest['groups']; require(1 <= len(specs) <= 2 and len({x['variant'] for x in specs}) == len(specs), 'Explicit distinct P/J subset required')
    groups = [prepare_group(spec, facts, helper) for spec in specs]
    plan = dict(schema='opsd-ablation-precision-history-read-plan-v1', manifest=mp, facts=fp, sources=sources,
                groups=[dict(variant=g['variant'], run_id=g['run'], config=g['config_ref'], backup=g['proof'],
                             exact_selected_JSON_refs=list(g['wanted'].values()),
                             attempt_directories=[dict(outer=k[0], attempt=k[1]) for k in sorted(g['attempts'])]) for g in groups])
    out.mkdir(parents=True, exist_ok=False)
    def write(name, value):
        path = out/name; path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)+'\n')
        return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)
    artifacts = [write('read-plan.private.json', plan)]
    if not args.plan_only:
        public = {}; private = {}
        for group in groups:
            values = {}
            def read(ref):
                handle, row, local, path = group['backups'].opened(ref)
                with handle:
                    raw = handle.read(); group['backups'].finish(ref, handle, row, local, path,
                        hashlib.sha256(raw).hexdigest(), len(raw), 'precision_history_selected_JSON')
                values[ref['path']] = [json.loads(line) for line in raw.splitlines()] if ref['path'] == group['history'] else json.loads(raw)
            group['backups'].visit(list(group['wanted'].values()), read)
            public[group['variant']], private[group['variant']] = analyze(group, values, helper)
        summary = dict(schema='opsd-ablation-recorded-precision-history-summary-v1', groups=public,
            pending_variants=sorted({'P','J'}-set(public)), source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            limitations=['Only recorded installed-module input/output observations, not all kernels/activation tensors.',
                'Detailed events capped4096 per rank/attempt; full dtype_counts retained, dropped counts explicit.',
                'Capsule PT payloads are inventory-SHA references only, not loaded; Adam/cursor/RNG claims inside them are not independently verified.',
                'Retry JSON records scaler_after; failure scaler_before lives in PT payloads and is not inferred or validated here.',
                'No UUID-to-PID mapping is emitted by original trainer; invocation and directory counts compared, not a fabricated per-attempt mapping.',
                'External dtype-repair refs are explicitly supplied only, not a full external receipt inventory.',
                'Historical whole-archive/member acceptance plus current metadata continuity; fresh selected JSON SHA and gzip CRC only.',
                'No Node, model, token/logits, optimizer construction, Torch, CUDA, HTTP, locks, signals or GPU gate.'])
        artifacts += [write('summary.json', summary), write('details.private.json', private)]
    for ref in (mp, fp, *sources.values()):
        helper.stable_json(ref['path'], ref['sha256']) if ref in (mp, fp) else require(hashlib.sha256(Path(ref['path']).read_bytes()).hexdigest() == ref['sha256'], 'Source changed')
    derived = write('derivation-manifest.json', dict(schema='opsd-ablation-precision-history-derivation-v1',
        created_unix=time.time(), plan_only=args.plan_only, input=mp, facts=fp, sources=sources, artifacts=artifacts,
        cli_arguments=sys.argv, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), no_node_or_GPU_IO=True))
    print(json.dumps(dict(plan_only=args.plan_only, output=str(out), derivation_manifest=derived)))


if __name__ == '__main__':
    main()
