"""HKUST filesystem reporting of closed P/J precision observations and retries."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import stat
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
BASE = Path('/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008')
SNAPSHOT_SHA = '40e05812069c118b14df6605128204bd1a013286674c4cacab768a91ec48c211'
SCOPE_SOURCE_SHA = '34857f4a6d0cfea4da2e2c4b71a5af662aca9a49e7ee7134ab97cd3c88185222'
ORDER = ('C0', 'C1', 'M', 'R', 'W', 'P', 'J')
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


def read_transport_ref(helper, ref):
    require(set(('path', 'sha256', 'bytes')) <= set(ref), 'Transport reference needs exact path/SHA/bytes')
    value, proof = helper.stable_json(ref['path'], ref['sha256'])
    require(proof['bytes'] == ref['bytes'], 'Transport reference byte count differs')
    return value, proof


def source_bytes_ref(helper, ref, expected):
    path = Path(ref['path'])
    helper.FixedBackups.path_check(path)
    before = helper.metadata(path.stat()); raw = path.read_bytes()
    require(helper.metadata(path.stat()) == before and ref['bytes'] == len(raw)
            and ref['sha256'] == hashlib.sha256(raw).hexdigest() == expected,
            'Copy producer source changed or pinned SHA differs')
    return dict(path=str(path), sha256=expected, **before)


def prepare_direct(spec, config_ref, run, output, outers, helper):
    """Bind a captured actual copy process to independent same-HKUST members.

    The snapshot's full-file copy SHA acceptance is historical evidence; current
    file metadata/inodes and fresh selected native closure JSON are checked here.
    Neither absent PIDs nor a file named complete.json imply process success.
    """
    variant = spec['variant']
    require(output == BASE / ('group-' + variant) / 'runs', 'Original authorized group output required')
    require(len(spec['backups']) == 1 and spec['backups'][0]['kind'] == 'direct',
            'One explicit accepted independent HKUST closed10 direct copy required')
    t = spec['remote_copy']
    require(set(t) == {'deployment', 'launch', 'actual_wait', 'complete'},
            'Exact deployment/launch/actual-wait/complete transport refs required')
    deployment, dp = read_transport_ref(helper, t['deployment'])
    launch, lp = read_transport_ref(helper, t['launch'])
    waited, wp = read_transport_ref(helper, t['actual_wait'])
    all_copies, ap = read_transport_ref(helper, t['complete'])
    require(deployment['accepted'] is True and deployment['actual_ssh_returncode'] == 0
            and deployment['scientific_source_changed'] is False,
            'Actual source deployment/readback was not accepted')
    source_refs = deployment['source']
    require(set(source_refs) == {'snapshot.py', 'source-operation.py'}, 'Exact pinned copy producer references required')
    snapshot = source_bytes_ref(helper, source_refs['snapshot.py'], SNAPSHOT_SHA)
    scope_source = source_bytes_ref(helper, source_refs['source-operation.py'], SCOPE_SOURCE_SHA)
    for key, got in (('snapshot.py', snapshot), ('source-operation.py', scope_source)):
        evidence = [v for k, v in deployment['Git_FETCH_files'].items() if k.endswith('/' + key)]
        require(len(evidence) == 1 and evidence[0]['sha256'] == got['sha256']
                and evidence[0]['bytes'] == got['bytes'], 'Actual Git source readback differs')
    require(BASE / 'analysis' in Path(snapshot['path']).parents
            and Path(snapshot['path']).parent == Path(scope_source['path']).parent,
            'Copy source must remain in its independent authorized HKUST deployment')
    copy_output = Path(ap['path']).parent
    require(copy_output.is_relative_to(BASE / 'analysis')
            and all_copies['schema'] == 'opsd-ablation-remote-raw-copy-v1'
            and all_copies['accepted'] is True and all_copies['Mac_raw_payloads'] is False
            and all_copies['scientific_source_changed'] is False
            and all_copies['original_files_deleted'] is False
            and same_ref(all_copies['source_operation'], scope_source), 'Exact original remote-copy acceptance differs')
    selected = [v for v in ORDER if v in all_copies['groups']]
    require(selected and set(selected) == set(all_copies['groups']) and variant in selected,
            'Copy-complete group set differs')
    require(launch['output'] == waited['output'] == str(copy_output)
            and launch['run_id'] == waited['run_id'] and waited['actual_ssh_returncode'] == 0,
            'Captured actual copy wait0/output/run required; PID absence is not success')
    # Direct tool execution can capture an actual unified session but no local
    # SSH PID. Preserve that unknown PID; require whichever captured execution
    # identity was actually available to agree, never invent a PID.
    pid = launch.get('actual_ssh_pid'); wait_pid = waited.get('actual_ssh_pid')
    session = launch.get('actual_unified_session'); wait_session = waited.get('actual_unified_session')
    require((type(pid) is int and pid > 0 and pid == wait_pid)
            or (type(session) in (int, str) and str(session).isdigit() and int(session) > 0
                and session == wait_session), 'Actual copy PID/session capture is missing or contradictory')
    require((pid is None and wait_pid is None) or type(pid) is int and pid > 0 and pid == wait_pid,
            'Known copy PID contradicts actual wait capture')
    require(session is None or session == wait_session, 'Known copy session contradicts actual wait capture')
    require(launch['started_unix'] <= all_copies['started_unix'] <= all_copies['ended_unix'] <= waited['ended_unix'],
            'Copy launch/completion/actual-wait chronology differs')
    argv = launch['argv']
    require(isinstance(argv, list) and argv and all(isinstance(x, str) for x in argv)
            and Path(argv[0]).name == 'ssh' and 'hkust1' in argv[:-1], 'Actual HKUST1 SSH launch argv required')
    command = shlex.split(argv[-1])
    prefix = ['env', 'E1_RUN_ID=' + launch['run_id'], 'CUDA_VISIBLE_DEVICES=', 'PYTHONDONTWRITEBYTECODE=1']
    require(command[:4] == prefix and len(command) == 13 and Path(command[4]).name == 'python'
            and command[5:7] == ['-B', snapshot['path']]
            and command[7:] == ['--groups', ','.join(selected), '--source-operation', scope_source['path'],
                                '--output', str(copy_output)],
            'Actual copy launch did not execute these pinned sources/output/group subset')
    group_ref = all_copies['groups'][variant]
    require(group_ref['path'] == str(copy_output / variant / 'complete.json'), 'Group copy completion path differs')
    closed, cp = read_transport_ref(helper, group_ref)
    backup = spec['backups'][0]
    require(closed['accepted'] is True and closed['variant'] == variant and closed['run_id'] == run
            and same_ref(closed['config'], config_ref) and closed['backups'] == spec['backups']
            and closed['independent_inode_copy'] is True and closed['node_originals_retained'] is True
            and closed['no_GPU_or_Torch'] is True and closed['no_training_lock'] is True,
            'Accepted exact independent direct group copy identity differs')
    require(backup['source']['path'] == str(copy_output / variant / 'source-inventory.json')
            and backup['local']['path'] == str(copy_output / variant / 'copy-accepted.json')
            and same_ref(closed['source'], backup['source']) and same_ref(closed['local'], backup['local']),
            'Exact group source/local inventory references differ')
    source, sp = read_transport_ref(helper, backup['source'])
    local, bp = read_transport_ref(helper, backup['local'])
    require(source['accepted'] is True and source['all_source_pinned_metadata_stable'] is True
            and source['run_id'] == run and source['variant'] == variant and source['source_root'] == str(BASE)
            and same_ref(source['source_operation'], scope_source), 'Original copy source scope/producer differs')
    require(local['accepted'] is True and local['exact_inventory'] is True
            and local['local_metadata_stable'] is True and local['no_symlinks_or_special_files'] is True
            and local['source_sha_matched_all_files'] is True and local['storage_cluster'] == 'HKUST'
            and local['mac_payload_copy'] is False and local['destination'] == str(copy_output / variant / 'raw'),
            'Exact same-HKUST copy acceptance differs')
    scope = source['source_scope']; closure = source['original_closure']
    require(scope['schema'] == 'opsd-ablation-closed-raw-scope-v1' and scope['accepted'] is True
            and scope['cluster'] == 'HKUST' and scope['variant'] == variant and scope['run_id'] == run
            and scope['output_root'] == str(output) and scope['closed_outers'] == list(range(10))
            and scope['include_outer_raw'] is True and same_ref(scope['config'], config_ref)
            and same_ref(closure['config'], config_ref)
            and len(scope['outers']) == len(closure['outers']) == 10, 'Original exact native closed10 scope differs')
    require(scope['status']['path'] == str(output / 'status.json')
            and Path(scope['group_terminal']['path']).name == 'group-terminal.json'
            and Path(scope['group_terminal']['path']).parent.name == 'training'
            and BASE in Path(scope['group_terminal']['path']).parents
            and same_ref(scope['status'], closure['status'])
            and same_ref(scope['group_terminal'], closure['group_terminal']), 'Original native status/terminal scope differs')
    for outer, declared, accepted in zip(outers, scope['outers'], closure['outers']):
        require(declared['outer_index'] == accepted['outer_index'] == outer['outer']
                and declared['optimizer_updates'] == accepted['optimizer_updates'] == outer['counts']['optimizer_updates']
                and same_ref(declared['completion'], accepted['complete'])
                and same_ref(accepted['complete'], outer['sources']['completion'])
                and same_ref(declared['checkpoint'], accepted['checkpoint'])
                and same_ref(accepted['checkpoint'], outer['sources']['checkpoint']),
                'Direct-copy native outer/checkpoint closure differs from original facts')
    sm = {r['relative']: r for r in source['files']}; lm = {r['relative']: r for r in local['files']}
    require(len(sm) == len(source['files']) == source['file_count'] == len(lm) == len(local['files'])
            == local['file_count'] == closed['files'] and sm.keys() == lm.keys()
            and sum(r['bytes'] for r in sm.values()) == source['total_bytes'] == local['total_bytes'] == closed['bytes'],
            'Exact complete source/destination file set or byte total differs')
    fixed = {r['path']: r for r in scope['global_files']}
    require(len(fixed) == len(scope['global_files'])
            and {config_ref['path'], str(output / 'status.json'), str(output / 'child-history.jsonl'),
                 str(output / 'run-identity.json'), scope['group_terminal']['path']} <= set(fixed),
            'Required complete global native closure file set differs')
    terminal_dir = Path(scope['group_terminal']['path']).parent
    for path, r in fixed.items():
        q = Path(path)
        require(q == Path(config_ref['path']) or q in (output / 'status.json', output / 'child-history.jsonl', output / 'run-identity.json')
                or output / 'phase-exits' in q.parents or BASE / ('group-' + variant) / 'receipts/dtype-recovery' in q.parents
                or terminal_dir in q.parents, 'Global scope escapes exact closed native group')
    excluded = set(source['excluded_subtrees'])
    require(len(excluded) == len(source['excluded_subtrees'])
            and excluded <= {str(output / 'outers' / ('outer-%05d' % i) / 'merged-model-cache') for i in range(10)},
            'Only original derived model-cache subtree may be excluded')
    original_map = {}; local_map = {}; fixed_paths = {Path(x) for x in fixed}
    for relative, r in sm.items():
        l = lm[relative]; q = Path(r['path']); destination = Path(l['path'])
        safe = Path(relative)
        require(not safe.is_absolute() and '..' not in safe.parts and str(safe) == relative
                and q == BASE / safe and destination == Path(local['destination']) / safe
                and r['sha256'] == l['sha256'] and r['bytes'] == l['bytes'], 'Exact source/destination SHA/member path differs')
        eligible = q in fixed_paths or any(
            output / 'outers' / ('outer-%05d' % i) in q.parents
            and output / 'outers' / ('outer-%05d' % i) / 'merged-model-cache' not in q.parents for i in range(10))
        require(eligible, 'Member escapes complete noncache closed10 raw/global scope')
        helper.FixedBackups.path_check(q); helper.FixedBackups.path_check(destination)
        metadata = {k: r[k] for k in ('device', 'inode', 'bytes', 'mode', 'mtime_ns', 'ctime_ns')}
        require(helper.metadata(q.stat()) == metadata and helper.metadata(destination.stat()) == l['metadata']
                and (metadata['device'], metadata['inode']) != (l['metadata']['device'], l['metadata']['inode']),
                'Source/destination continuity differs or member is not an independent inode copy')
        if r['path'] in fixed:
            require(same_ref(r, fixed[r['path']]), 'Fixed native closure member SHA/bytes differs')
        original_map[r['path']] = r; local_map[r['path']] = l
    require(set(fixed) <= set(original_map), 'Complete global closure omitted from member inventory')
    native_proofs = []
    def native(ref):
        require(ref['path'] in original_map and same_ref(ref, original_map[ref['path']]),
                'Native closure reference absent from exact accepted copy')
        l = local_map[ref['path']]
        value, proof = helper.stable_json(l['path'], ref['sha256'])
        require(proof['bytes'] == ref['bytes'] and helper.metadata(Path(l['path']).stat()) == l['metadata'],
                'Fresh copied native closure JSON differs')
        native_proofs.append(dict(original={k: ref[k] for k in ('path', 'sha256', 'bytes')}, copy=proof))
        return value
    status = native(closure['status']); terminal = native(closure['group_terminal'])
    wait = native(closure['actual_child_exit']); release = native(closure['actual_release'])
    final = outers[-1]
    require(status['phase'] == 'TRAIN_COMPLETE' and status['outer_iterations'] == 10
            and status['run_id'] == run and status['variant'] == variant and status['config_sha256'] == config_ref['sha256']
            and status['actual_optimizer_updates'] == final['counts']['optimizer_updates']
            and same_ref(status['latest_checkpoint'], final['sources']['checkpoint']), 'Fresh copied status is not final native10')
    require(terminal['schema'] == 'opsd-ablation-group-foreground-terminal-v1' and terminal['accepted'] is True
            and terminal['run_id'] == run and terminal['config_sha256'] == config_ref['sha256']
            and terminal['actual_exit_code'] == 0 and terminal['signal_termination_not_exit0'] is False
            and terminal['controller'] == status['owner'] and same_ref(terminal['status'], scope['status']),
            'Fresh copied terminal does not prove original controller exit0')
    owner = terminal['controller']
    require(owner['uid'] == 1005 and owner['run_id'] == run and type(owner['pid']) is int and owner['pid'] > 0
            and type(owner['starttick']) is int and owner['starttick'] > 0 and not owner.get('capture_status')
            and same_ref(terminal['actual_child_exit'], closure['actual_child_exit'])
            and same_ref(terminal['actual_release'], closure['actual_release'])
            and wait['actual_wait_returncode'] == 0 and wait['identity'] == owner and wait['run_id'] == run
            and wait['signal_termination_not_exit0'] is False and release['accepted'] is True
            and release['owned']['accepted'] is True and release['after']['accepted'] is True,
            'Fresh copied actual parent wait/owned release/controller identity differs')
    return dict(deployment=dp, launch=lp, actual_wait=wp, all_copies_complete=ap,
                group_complete=cp, source_inventory=sp, copy_inventory=bp, snapshot_source=snapshot,
                scope_source=scope_source, files=len(sm), member_bytes=source['total_bytes'],
                independent_inode_copies_checked=len(sm), fresh_native_closure_JSON=native_proofs,
                processing_host='HKUST1', actual_copy_PID=pid, actual_copy_unified_session=session,
                whole_payload_fresh_SHA_in_this_reader=False)


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
    transport = prepare_direct(spec, config_ref, run, output, outers, helper)
    proof_ref = transport['group_complete']
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
                history=history, repair_receipts=spec.get('dtype_repair_receipts', []), transport=transport)


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
    p.add_argument('--plan-only', action='store_true', help='Read references/inventories and native closure JSON; do not read precision observations')
    args = p.parse_args(); out = Path(args.output)
    require(os.getuid() == os.geteuid() == 1005 and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
            and 'torch' not in sys.modules, 'Original HKUST1 UID and CPU-only environment required')
    require(out.is_relative_to(BASE / 'analysis'), 'Output must remain on authorized HKUST analysis filesystem')
    require(out.is_absolute() and not out.exists() and '..' not in out.parts
            and all(not q.is_symlink() for q in out.parents), 'Fresh canonical output directory required')
    # Bootstrap source SHA is pinned before executing its stdlib JSON/file reader.
    raw = Path(args.manifest).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == args.manifest_sha256, 'Input manifest SHA differs')
    manifest = json.loads(raw); helper, sources = load_sources(manifest['sources'])
    manifest, mp = helper.stable_json(args.manifest, args.manifest_sha256)
    require(manifest['schema'] == 'opsd-ablation-recorded-precision-history-remote-inputs-v2' and manifest['ready'] is True,
            'Explicit ready closed-group inputs required')
    facts, fp = helper.stable_json(manifest['facts']['path'], manifest['facts']['sha256'])
    require(facts['schema'] == 'opsd-ablation-outer-facts-v1' and facts['accepted'] is True, 'Accepted native facts required')
    specs = manifest['groups']; require(1 <= len(specs) <= 2 and len({x['variant'] for x in specs}) == len(specs), 'Explicit distinct P/J subset required')
    groups = [prepare_group(spec, facts, helper) for spec in specs]
    plan = dict(schema='opsd-ablation-precision-history-remote-read-plan-v2', manifest=mp, facts=fp, sources=sources,
                groups=[dict(variant=g['variant'], run_id=g['run'], config=g['config_ref'], backup=g['proof'],
                             remote_copy_transport=g['transport'], exact_selected_JSON_refs=list(g['wanted'].values()),
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
        summary = dict(schema='opsd-ablation-recorded-precision-history-remote-summary-v2', groups=public,
            pending_variants=sorted({'P','J'}-set(public)), source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            limitations=['Only recorded installed-module input/output observations, not all kernels/activation tensors.',
                'Detailed events capped4096 per rank/attempt; full dtype_counts retained, dropped counts explicit.',
                'Capsule PT payloads are inventory-SHA references only, not loaded; Adam/cursor/RNG claims inside them are not independently verified.',
                'Retry JSON records scaler_after; failure scaler_before lives in PT payloads and is not inferred or validated here.',
                'No UUID-to-PID mapping is emitted by original trainer; invocation and directory counts compared, not a fabricated per-attempt mapping.',
                'External dtype-repair refs are explicitly supplied only, not a full external receipt inventory.',
                'Original independent HKUST copy source/destination full-file SHA acceptance plus source/destination metadata continuity at preparation; selected JSON is freshly hashed before/after its actual read in this reporter.',
                'HKUST filesystem reads/writes only; no network, model, token/logits, optimizer construction, Torch, CUDA, HTTP, locks, signals or GPU gate.'])
        artifacts += [write('summary.json', summary), write('details.private.json', private)]
    for ref in (mp, fp, *sources.values()):
        helper.stable_json(ref['path'], ref['sha256']) if ref in (mp, fp) else require(hashlib.sha256(Path(ref['path']).read_bytes()).hexdigest() == ref['sha256'], 'Source changed')
    for group in groups:
        transport = group['transport']
        for key in ('deployment', 'launch', 'actual_wait', 'all_copies_complete', 'group_complete', 'source_inventory', 'copy_inventory'):
            read_transport_ref(helper, transport[key])
        source_bytes_ref(helper, transport['snapshot_source'], SNAPSHOT_SHA)
        source_bytes_ref(helper, transport['scope_source'], SCOPE_SOURCE_SHA)
    derived = write('derivation-manifest.json', dict(schema='opsd-ablation-precision-history-remote-derivation-v2',
        created_unix=time.time(), plan_only=args.plan_only, input=mp, facts=fp, sources=sources, artifacts=artifacts,
        cli_arguments=sys.argv, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        processing_host='HKUST1', Mac_raw_payloads=False, no_network_or_GPU_IO=True))
    print(json.dumps(dict(plan_only=args.plan_only, output=str(out), derivation_manifest=derived)))


if __name__ == '__main__':
    main()
