"""Collect closed native milestones without touching the active training process.

Only standard-library modules and the existing CPU report/contracts are imported.
No tensor files, process signals, model evaluation, network calls, or GPU queries.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time

from release.recursive_bran_a.ablation150_reports_v1 import report
from release.recursive_bran_a.ablation150_v1.contracts import planned_conditions, read_receipt, sha

MILESTONES = (5, 50, 100, 150)
TERMINAL_STUDY = 'ALL_PLANNED_LABELS_150_COMPLETE_OR_MATCHED_REUSE'
COMPLETE_TRAINING = 'COMPLETE_NATIVE_150_TRAINING_AND_FOUR_SCHEDULED_DEV'
A_COMMIT = 'a1ec638575268747468b1a14ff8fd5f858ca0e1a'
A_CONFIG = '43e28cc7ecc0d38a93324bf480aafb23a0c1ecdb979e9ea587d0358ae095233d'
REFERENCE_PACK = Path(__file__).with_name('a-reference-pack.json')
STEP = re.compile(r'\bstep:(\d+) - ')
_GROUP_CACHE = {}


def finite(value):
    """Validate existing scalar fields; this is not a native tensor inspection."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Nonfinite existing native scalar')
    if isinstance(value, dict):
        for child in value.values():
            finite(child)
    elif isinstance(value, list):
        for child in value:
            finite(child)
    return value


def read(path, optional=False):
    try:
        return finite(json.loads(Path(path).read_text()))
    except FileNotFoundError:
        if optional:
            return None
        raise


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + '.pending-' + str(os.getpid()))
    pending.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
    os.replace(pending, path)


def bounded_path(path, root):
    path, root = Path(path), Path(root).resolve()
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError('Evidence path escapes its own condition: ' + str(path))
    return path


def process(owner):
    """Read an exact local owner identity, without sending a signal."""
    if not owner:
        return None
    path = Path('/proc', str(owner['pid']))
    try:
        fields = (path/'stat').read_text().rsplit(')', 1)[1].split()
        value = dict(pid=owner['pid'], uid=path.stat().st_uid, tick=int(fields[19]), state=fields[0])
    except (FileNotFoundError, ProcessLookupError):
        return None
    if value['uid'] != owner['uid'] or value['tick'] != owner['tick'] or value['state'] == 'Z':
        return None
    return value


def identity(value, receipt):
    for key in ('run_id', 'config_sha256', 'source_commit', 'group'):
        if value.get(key) != receipt.get(key):
            raise ValueError('Native ' + key + ' differs from immutable receipt')


def complete_jsonl(path):
    """A final unterminated line is still being written, so never interpret it."""
    try:
        with Path(path).open() as stream:
            for line in stream:
                if line.endswith('\n'):
                    yield finite(json.loads(line))
    except FileNotFoundError:
        return


def closed_records(roots, receipt):
    rows, origins = {}, {}
    for root in roots:
        path = root/'completed-outers.jsonl'
        for value in complete_jsonl(path):
            identity(value, receipt)
            outer = value['completed_outer']
            if type(outer) is not int or not 1 <= outer <= 150 or value['next_outer'] != outer+1:
                raise ValueError('Invalid actual completed outer/cursor')
            if outer in rows and rows[outer] != value:
                raise ValueError('Conflicting completed native outer ' + str(outer))
            rows[outer] = value
            origins[outer] = dict(path=str(path), record_sha256=hashlib.sha256(
                json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest())
    return rows, origins


def boundary(value, receipt, group_base):
    outer = value['completed_outer']
    checkpoint = value['checkpoint']
    path = bounded_path(checkpoint['path'], group_base/'checkpoints')
    if path.name != 'manifest.json' or sha(path) != checkpoint['sha256']:
        raise ValueError('Closed native boundary manifest hash differs')
    saved = read(path)
    native = saved['identity']
    if (saved.get('status') != 'COMPLETE_SAME_WORLD_STATE_SAVED'
            or saved.get('boundary_phase') != 'post-evaluation'
            or native.get('completed_step') != outer
            or native.get('config_sha256') != receipt['config_sha256']
            or native.get('code_commit') != receipt['source_commit']
            or native.get('data_identity') != receipt['data_identity']
            or native.get('actual_update_limit') != 150
            or saved['driver'].get('next_step') != outer+1):
        raise ValueError('Incomplete or mismatched post-evaluation native boundary')
    workers = saved['worker_states']
    if len(workers) != 8 or sorted(v['identity']['rank'] for v in workers) != list(range(8)):
        raise ValueError('Native boundary omits an actual rank')
    counters = value['optimizer_calls_per_rank']
    if set(counters) != {str(rank) for rank in range(8)} or len(set(counters.values())) != 1:
        raise ValueError('Closed native optimizer counters differ across ranks')
    for worker in workers:
        rank = str(worker['identity']['rank'])
        if (any(worker['identity'].get(key) != item for key, item in native.items())
                or worker['identity'].get('world_size') != 8
                or worker.get('full_worker_state_finite') is not True
                or worker.get('scheduler_last_epoch') != outer
                or worker.get('actual_total_optimizer_calls') != counters[rank]):
            raise ValueError('Recorded native finite/scheduler/Adam state differs')
    dev = report.evaluation(value['metrics'])
    if dev is None:
        raise ValueError('Closed scheduled milestone lacks complete native dev512')
    return dict(path=str(path), sha256=checkpoint['sha256'],
                boundary_phase='post-evaluation', recorded_full_worker_state_finite=True,
                scheduler_each_rank=outer, native_tensor_reload_performed=False), dev


def native_lines(group_base, closed):
    lines, origins = {}, {}
    for path in sorted((group_base/'logs').glob('training*.log')):
        with path.open(errors='replace') as stream:
            for line in stream:
                if not line.endswith('\n'):
                    continue
                clean = report.ANSI.sub('', line)
                match = STEP.search(clean)
                if match is None or int(match.group(1)) not in closed:
                    continue
                outer = int(match.group(1))
                if 'timing_s/step:' not in clean:
                    continue
                # Normalize ANSI/process prefixes only, retaining actual native metrics.
                normalized = clean[match.start():]
                if outer in lines and lines[outer] != normalized:
                    raise ValueError('Conflicting complete native log records')
                lines[outer] = normalized
                origins[outer] = dict(path=str(path), captured_line_sha256=hashlib.sha256(
                    line.encode()).hexdigest(), normalization='ANSI/process prefix removed only')
    return lines, origins


def attempt_records(roots, receipt, closed):
    accepted, origins = {}, {}
    for root in roots:
        for path in sorted((root/'attempt-results').glob('attempt-*.json')):
            # An atomic attempt result can precede completed_round/checkpoint by minutes.
            value = read(path)
            result = value['all_rank_result']
            if result.get('decision') != 'COMPLETED_OUTER':
                continue
            outer = value['accepted_outer']
            if outer not in closed:
                continue
            if any(result['identity'].get(key) != receipt[key] for key in ('run_id', 'config_sha256')):
                raise ValueError('Accepted Adam receipt identity differs')
            if outer in accepted and accepted[outer] != value:
                raise ValueError('Conflicting accepted Adam receipt')
            accepted[outer] = value
            origins[outer] = dict(path=str(path), sha256=sha(path))
    return accepted, origins


def wait_matches(wait, entry):
    return bool(wait and entry and wait.get('pid') == entry['pid']
                and wait.get('group') == entry['group'] and wait.get('time', 0) >= entry['time'])


def group_terminal(group_base, receipt, entry, study, closed, read_process=process):
    wait = read(group_base/'receipts/actual-parent-wait.json', optional=True)
    active = read_process(entry)
    terminal = read(Path(entry['output'])/'entry-exit.json', optional=True) if entry else None
    complete = read(Path(entry['output'])/'training-complete.json', optional=True) if entry else None
    current_wait = wait if wait_matches(wait, entry) else None
    child = study.get('child', {})
    replacing_entry = bool(entry and study.get('current') == receipt['group']
        and child.get('group') == receipt['group'] and child.get('pid') != entry['pid'])
    if replacing_entry:
        return dict(accepted=False, owner_active=False, awaiting_current_entry=True,
            actual_parent_wait_rc=None, ignored_previous_parent_wait=bool(wait), entry_exit_rc=None, errors=[])
    errors = []
    if terminal:
        if any(terminal.get(key) != entry[key] for key in ('pid', 'uid', 'tick', 'run_id', 'group')):
            raise ValueError('Entry exit receipt is not the exact current owner')
        if terminal['rc'] != 0:
            errors.append('Current entry has actual native exit ' + str(terminal['rc']))
    if current_wait and current_wait['rc'] != 0:
        errors.append('Current owner has actual parent wait ' + str(current_wait['rc']))
    accepted = study.get('complete', {}).get(receipt['group'])
    status = dict(accepted=False, owner_active=bool(active), actual_parent_wait_rc=(
        current_wait['rc'] if current_wait else None), ignored_previous_parent_wait=bool(wait and not current_wait),
        entry_exit_rc=terminal['rc'] if terminal else None, errors=errors)
    if not accepted:
        return status
    complete_path = Path(entry['output'])/'training-complete.json'
    if not complete or not terminal or not current_wait:
        raise ValueError('Study group acceptance lacks complete/current exit/current parent wait')
    identity(complete, receipt)
    if (complete.get('status') != COMPLETE_TRAINING or complete.get('completed_outer') != 150
            or complete.get('evaluation_steps') != list(MILESTONES)
            or sorted(closed) != list(range(1, 151))
            or terminal['rc'] != 0 or current_wait['rc'] != 0
            or complete['checkpoint'] != closed[150]['checkpoint']
            or report.evaluation(complete['final_dev_metrics']) != report.evaluation(closed[150]['metrics'])
            or complete['actual_optimizer_calls_per_rank'] != closed[150]['optimizer_calls_per_rank']):
        raise ValueError('Study acceptance is not actual complete150 and four scheduled dev')
    if (Path(accepted['training_complete']) != complete_path
            or sha(complete_path) != accepted['complete_sha256']):
        raise ValueError('Accepted training completion path/hash differs')
    if accepted.get('actual_parent_wait') and Path(accepted['actual_parent_wait']).resolve() != (group_base/'receipts/actual-parent-wait.json').resolve():
        raise ValueError('Accepted parent wait is not this condition')
    if accepted.get('actual_parent_wait_rc', 0) != 0:
        raise ValueError('Accepted group stores a failed parent wait')
    if accepted.get('entry_exit_sha256') and sha(Path(entry['output'])/'entry-exit.json') != accepted['entry_exit_sha256']:
        raise ValueError('Accepted entry exit hash differs')
    release = accepted['owned_gpu_release']
    if release.get('complete') is not True or release.get('applications') != [] or release.get('query_rc') != 0:
        raise ValueError('Accepted group omits actual owned GPU context release receipt')
    report.number(release['time'], 'existing release time')
    if release['time'] < complete['time'] or release['time'] < current_wait['time']:
        raise ValueError('GPU release receipt precedes actual training/parent completion')
    status.update(accepted=True, proof=dict(training_complete_sha256=sha(complete_path),
        entry_exit_sha256=sha(Path(entry['output'])/'entry-exit.json'),
        parent_wait_sha256=sha(group_base/'receipts/actual-parent-wait.json'),
        release_receipt=release))
    return status


def inspect_group(base, row, study, read_process=process):
    group_base = base/row['group']
    receipt_path = group_base/'receipts/configuration.json'
    if not receipt_path.exists():
        if row['group'] in study.get('complete', {}):
            raise ValueError('Accepted group has no immutable configuration')
        return None
    receipt = finite(read_receipt(receipt_path))
    if (receipt['group'] != row['group'] or receipt['condition'] != row['options']
            or receipt['source_commit'] != study['source_commit']
            or receipt['actual_update_limit'] != 150 or receipt['evaluation_steps'] != list(MILESTONES)):
        raise ValueError('Prepared group differs from frozen study condition')
    entry = read(group_base/'receipts/current-entry.json', optional=True)
    if entry:
        identity(entry, receipt)
        bounded_path(entry['output'], group_base/'output')
    roots = sorted((group_base/'output').glob('attempt-*'))
    roots = [bounded_path(root, group_base/'output') for root in roots if root.is_dir()]
    closed, closed_origins = closed_records(roots, receipt)
    lines, log_origins = native_lines(group_base, closed)
    attempts, attempt_origins = attempt_records(roots, receipt, closed)
    milestones, pending = {}, []
    for outer in MILESTONES:
        if outer not in closed:
            continue
        required = set(range(1, outer+1))
        if not required.issubset(closed) or not required.issubset(attempts) or not required.issubset(lines):
            if read_process(entry):
                pending.append(outer)
                continue  # completed_round flush can precede the native complete scalar line.
            raise ValueError('Closed milestone lacks contiguous native log/Adam/completion coverage')
        proof, dev = boundary(closed[outer], receipt, group_base)
        milestones[outer] = dict(boundary=proof, dev512=dev,
            native_completion=closed_origins[outer], native_log=log_origins[outer],
            adam_receipts=[attempt_origins[index] for index in range(1, outer+1)])
    terminal = group_terminal(group_base, receipt, entry, study, closed, read_process)
    return dict(receipt=receipt, configuration_sha256=sha(receipt_path), entry=entry,
        completed_outer=max(closed, default=0), closed=closed, lines=lines, attempts=attempts,
        milestones=milestones, pending_milestones=pending, terminal=terminal,
        small_evidence_paths=[origin['path'] for origin in attempt_origins.values()]
            + [value['boundary']['path'] for value in milestones.values()],
        recovered_sampling_gap=any(read(root/'zero-write-recovered.json', optional=True) is not None for root in roots)
            or bool(entry and entry.get('zero_write_manifest'))
            or (study.get('explicit_recovery', {}).get('group') == row['group']
                and study.get('explicit_recovery', {}).get('request', {}).get('kind') == 'ZERO_WRITE_SAVED_BATCH'))


def stat_signature(path):
    try:
        value = Path(path).stat()
        return (str(path), value.st_ino, value.st_size, value.st_mtime_ns)
    except FileNotFoundError:
        return (str(path), None)


def cached_group(base, row, study, read_process):
    """A growing live log alone does not trigger a repeated scan of accepted work."""
    group_base = base/row['group']
    key = str(group_base)
    cached = _GROUP_CACHE.get(key)
    receipt_path = group_base/'receipts/configuration.json'
    receipt = read(receipt_path, optional=True)
    entry_path = group_base/'receipts/current-entry.json'
    entry = read(entry_path, optional=True)
    paths = [receipt_path, entry_path, group_base/'receipts/actual-parent-wait.json']
    roots = sorted((group_base/'output').glob('attempt-*'))
    paths.extend(root/'completed-outers.jsonl' for root in roots)
    if entry:
        paths.extend(Path(entry['output'])/name for name in ('entry-exit.json', 'training-complete.json'))
    if cached and cached['value']:
        paths.extend(cached['value']['small_evidence_paths'])
    signature = dict(stats=[stat_signature(path) for path in paths],
        config_sha256=sha(receipt['config']) if receipt else None, source_commit=study['source_commit'],
        accepted=study.get('complete', {}).get(row['group']),
        explicit_recovery=study.get('explicit_recovery') if study.get('explicit_recovery', {}).get('group') == row['group'] else None,
        child=study.get('child') if study.get('current') == row['group'] else None)
    # Preserve the signature including dependencies discovered by the first read.
    if cached and cached['signature'] == signature and not (cached['value'] and cached['value']['pending_milestones']):
        value = cached['value']
        if value:
            value['terminal'] = group_terminal(group_base, value['receipt'], value['entry'], study,
                value['closed'], read_process)
        return value
    value = inspect_group(base, row, study, read_process)
    if value:
        discovered = [path for path in value['small_evidence_paths'] if str(path) not in {str(p) for p in paths}]
        signature['stats'].extend(stat_signature(path) for path in discovered)
    _GROUP_CACHE[key] = dict(signature=signature, value=value)
    return value


def reference(pack_path):
    pack = read(pack_path)
    if pack.get('protocol') != 'retained_A_native_aggregate_pack_v1' or pack.get('new_inference') is not False:
        raise ValueError('Reference is not the retained native aggregate pack')
    if set(pack['milestones']) != {str(v) for v in MILESTONES}:
        raise ValueError('Reference pack omits required native milestones')
    for outer, value in pack['milestones'].items():
        if value['outer'] != int(outer):
            raise ValueError('Reference milestone identity differs')
        source = value['source']
        if len(source['captured_line_sha256']) != 64 or not source['native_log']:
            raise ValueError('Reference lacks origin line provenance hash')
        dev = value['dev512']
        groups = report.evaluation({'val/qa_count/all': dev['count'], 'val/em/all': dev['hits']/dev['count'],
            'val/qa_count/nq': dev['nq']['count'], 'val/em/nq': dev['nq']['hits']/dev['nq']['count'],
            'val/qa_count/hotpotqa': dev['hotpot']['count'], 'val/em/hotpotqa': dev['hotpot']['hits']/dev['hotpot']['count']})
        if groups is None:
            raise ValueError('Missing retained A complete dev')
    return pack


def collect_report(label, outer, group, pack, destination, reuse_of=None):
    """A temporary receipt/log view lets the existing collector read only closed work."""
    with tempfile.TemporaryDirectory(prefix='aggregate-read-', dir=destination.parent) as scratch:
        root = Path(scratch)
        ref_path = root/'A.json'
        ref_path.write_text(json.dumps(pack['milestones'][str(outer)], allow_nan=False))
        original = dict(id='A-original', run_id='retained-A-native-multiple-entries',
            config_sha256=None, world_size=8, aggregate_files={str(outer): str(ref_path)})
        condition = dict(original, id=label)
        if group:
            receipt = group['receipt']
            attempts = root/'attempt-results'
            attempts.mkdir()
            for index in range(1, outer+1):
                (attempts/f'attempt-{index:06d}.json').write_text(json.dumps(group['attempts'][index], allow_nan=False))
            log = root/'native-closed.log'
            log.write_text(''.join(group['lines'][index] for index in range(1, outer+1)))
            condition = dict(id=label, run_id=receipt['run_id'], config_sha256=receipt['config_sha256'],
                world_size=8, logs=[str(log)], output_roots=[str(root)])
        result = report.build(dict(reference_condition='A-original', milestones=[outer], conditions=[original, condition]))
    if not result['analysis_complete']:
        raise ValueError('Closed milestone collector failed complete aggregate coverage')
    result.pop('analysis_complete')
    result.update(protocol='recursive_bran_ablation150_closed_milestone_v1',
        milestone_report_complete=True, goal_complete=False, requested_label=label, milestone=outer,
        reuse_of=reuse_of, generated_new_training=False if reuse_of else True,
        reference_provenance=pack['milestones'][str(outer)]['source'])
    reference_row = result['conditions']['A-original']['milestones'][str(outer)]
    reference_row['recorded_boundary_scalars'] = pack['milestones'][str(outer)].get('recorded_boundary_scalars', {})
    reference_row['cost_coverage'] = 'Retained boundary scalars only; no complete prefix cost available in reference pack'
    target = result['conditions'][label]['milestones'][str(outer)]
    if group:
        expected = group['milestones'][outer]['dev512']
        if target['dev512'] != expected:
            raise ValueError('Native complete log dev differs from actual closed dev receipt')
        cumulative = group['closed'][outer]['optimizer_calls_per_rank']
        if target['receipt_adam_each_rank'] != next(iter(cumulative.values())):
            raise ValueError('Actual Adam receipt prefix differs from native closed counters')
        if target['native_actual_adam_each_rank'] != target['receipt_adam_each_rank']:
            raise ValueError('Native reported Adam differs from accepted receipt prefix')
        result['native_evidence'] = group['milestones'][outer]
        result['configuration_receipt_sha256'] = group['configuration_sha256']
        metrics = group['closed'][outer]['metrics']
        target['recorded_boundary_scalars'] = {key: value for key, value in metrics.items()
            if key.startswith(('actor/lr', 'actor/scheduler_', 'training/', 'timing_s/'))}
        target['cost_coverage'] = dict(recorded_only=True, end_to_end_coverage_complete=False,
            recovered_outer1_original_sampling_excluded=group['recovered_sampling_gap'],
            original_sampling_cost_unknown=group['recovered_sampling_gap'])
        if group['recovered_sampling_gap']:
            result['limitations'].append('Recovered outer1 timers exclude its original pre-fault sampling; '
                'recorded prefix wall/tokens are incomplete end-to-end cost, including all later milestones.')
    else:
        target['recorded_boundary_scalars'] = reference_row['recorded_boundary_scalars']
        target['cost_coverage'] = reference_row['cost_coverage']
    result['limitations'].append('Only existing recorded native finite/scheduler scalar evidence is checked; '
        'no tensor load, model replay, new inference, or independent Adam reload is performed.')
    report.write_report(result, destination)
    return dict(directory=str(destination), aggregate_sha256=sha(destination/'aggregate.json'),
                reuse_of=reuse_of, generated_new_training=False if reuse_of else True)


def validate_reuse(study, row):
    value = study.get('complete', {}).get(row['group'])
    if not value:
        return False
    if (value.get('reuse_of') != row['reuse'] or value.get('condition') != row['options']
            or value.get('generated_new_training') is not False):
        raise ValueError('Study reuse acceptance differs from matched frozen condition')
    accepted = value['accepted']
    if row['reuse'] == 'C1-root-only':
        if accepted != study['complete'].get('C1-root-only'):
            raise ValueError('C1 reuse is not the actual accepted source condition')
    else:
        path = Path(accepted['path'])
        original = read(path)
        native = original['identity']
        if (sha(path) != accepted['sha256'] or accepted.get('completed_outer') != 150
                or original.get('status') != 'COMPLETE_SAME_WORLD_STATE_SAVED'
                or native.get('completed_step') != 150 or native.get('code_commit') != A_COMMIT
                or native.get('config_sha256') != A_CONFIG
                or original.get('boundary_phase') != 'post-evaluation'):
            raise ValueError('A reuse lacks the actual accepted retained native150 boundary')
    return True


def poll(base, output, pack_path=REFERENCE_PACK, read_process=process):
    base, output = Path(base).resolve(), Path(output).resolve()
    if output == base or any(output.is_relative_to(base/row['group']) for row in planned_conditions()):
        raise ValueError('Reports need a separate analysis directory')
    output.mkdir(parents=True, exist_ok=True)
    state_path = base/'study-progress.json'
    study = read(state_path)
    planned = planned_conditions()
    if study['conditions'] != planned or study.get('requested_labels') != len(planned):
        raise ValueError('Study does not contain the full frozen eleven-label goal')
    pack = reference(pack_path)
    previous = read(output/'watch-state.json', optional=True) or {}
    if previous.get('reports') and (previous.get('scientific_source_commit') != study['source_commit']
            or previous.get('reference_pack_sha256') != sha(pack_path)):
        raise ValueError('Existing report publication belongs to a different immutable study/reference')
    reports = copy.deepcopy(previous.get('reports', {}))
    errors, groups, created = [], {}, []
    for row in planned:
        if row.get('reuse'):
            continue
        try:
            groups[row['group']] = cached_group(base, row, study, read_process)
            group = groups[row['group']]
            if group:
                errors.extend(dict(group=row['group'], error=v) for v in group['terminal']['errors'])
        except (ValueError, KeyError, FileNotFoundError, TypeError) as error:
            errors.append(dict(group=row['group'], error=str(error)))
    for row in planned:
        label, reused = row['group'], row.get('reuse')
        group = groups.get(reused or label)
        available = MILESTONES if reused == 'A-original' else sorted(group['milestones']) if group else []
        reports.setdefault(label, {})
        for outer in available:
            destination = output/'milestones'/label/f'outer-{outer:03d}'
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                existing = reports[label].get(str(outer))
                if existing:
                    existing_path = bounded_path(existing['directory'], output/'milestones')/'aggregate.json'
                    if sha(existing_path) != existing['aggregate_sha256']:
                        raise ValueError('Previously persisted milestone report changed')
                    continue
                reports[label][str(outer)] = collect_report(label, outer, group, pack, destination, reused)
                created.append(dict(label=label, outer=outer))
            except (ValueError, KeyError, FileNotFoundError, TypeError) as error:
                errors.append(dict(group=label, outer=outer, error=str(error)))
    accepted_labels = {}
    for row in planned:
        try:
            accepted_labels[row['group']] = validate_reuse(study, row) if row.get('reuse') else bool(
                groups.get(row['group']) and groups[row['group']]['terminal']['accepted'])
        except (ValueError, KeyError, FileNotFoundError, TypeError) as error:
            errors.append(dict(group=row['group'], error=str(error)))
            accepted_labels[row['group']] = False
    active = read(base/'active-study.json', optional=True)
    exit_path = Path(active['actual_exit_path']) if active else base/'study-actual-exit.json'
    actual_exit = read(exit_path, optional=True)
    # Only the current wrapper's exit path is considered, never preserved failure files.
    if actual_exit and active and actual_exit.get('time', 0) < active['time']:
        actual_exit = None
    if actual_exit and actual_exit.get('rc') != 0:
        errors.append(dict(group='study', error='Actual current study exit ' + str(actual_exit.get('rc'))))
    if study['status'] not in ('ACTIVE', TERMINAL_STUDY):
        errors.append(dict(group='study', error='Study requires attention: ' + study['status']))
    required = [(row['group'], str(outer)) for row in planned for outer in MILESTONES]
    coverage_complete = all(outer in reports[label] for label, outer in required)
    terminal_accepted = bool(study['status'] == TERMINAL_STUDY and study.get('current') is None
        and set(study.get('complete', {})) == {row['group'] for row in planned}
        and all(accepted_labels.values()) and actual_exit and actual_exit.get('rc') == 0)
    status = 'ATTENTION_REQUIRED' if errors else 'GOAL_COMPLETE' if coverage_complete and terminal_accepted else 'ACTIVE'
    result = dict(protocol='recursive_bran_ablation150_aggregate_watch_v1', time=time.time(), status=status,
        goal_complete=status == 'GOAL_COMPLETE', required_labels=[row['group'] for row in planned],
        required_milestones=list(MILESTONES), required_report_count=len(required),
        collected_report_count=sum(outer in reports[label] for label, outer in required),
        all_required_reports_present=coverage_complete, study_terminal_accepted=terminal_accepted,
        study_status=study['status'], current_group=study.get('current'), study_sha256=sha(state_path),
        scientific_source_commit=study['source_commit'], entry_source_commit=study.get('entry_source_commit'),
        reference_pack_sha256=sha(pack_path), reports=reports, new_reports=created,
        accepted_labels=accepted_labels, current_study_exit=actual_exit, current_study_exit_path=str(exit_path),
        groups={key: dict(completed_outer=value['completed_outer'], milestones=sorted(value['milestones']),
            pending_milestones=value['pending_milestones'], terminal=value['terminal'])
            for key, value in groups.items() if value}, errors=errors,
        scope='CPU/file/proc reads only; no new inference, paired rows, bootstrap, signals, GPU queries, or training edits')
    atomic_json(output/'watch-state.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-pack', type=Path, default=REFERENCE_PACK)
    parser.add_argument('--interval', type=float, default=60)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval < 1:
        parser.error('--interval must be a finite value >=1')
    while True:
        try:
            result = poll(args.base, args.output, args.reference_pack)
        except (ValueError, KeyError, FileNotFoundError, TypeError, OSError) as error:
            result = dict(time=time.time(), status='ATTENTION_REQUIRED', goal_complete=False,
                errors=[dict(group='watcher', error=str(error))])
            atomic_json(args.output/'watch-error.json', result)
            atomic_json(args.output/'watch-state.json', result)
        print(json.dumps({key: result.get(key) for key in ('time', 'status', 'goal_complete',
            'current_group', 'collected_report_count', 'required_report_count', 'new_reports', 'errors')},
            allow_nan=False), flush=True)
        if result['status'] == 'ATTENTION_REQUIRED':
            return 2
        if result['goal_complete'] or args.once:
            return 0
        time.sleep(args.interval)


if __name__ == '__main__':
    raise SystemExit(main())
