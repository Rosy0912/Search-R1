"""Read-only compact selected-R progress on the original Linux node.

Request arrives on stdin; all original raw/model inputs remain remote. This
does not import Torch, signal processes, perform HTTP, write Node files, or
infer successful exit from PID absence. The root publishes this helper before
executing its bytes with the original Node Python and CUDA_VISIBLE_DEVICES=''.
"""
import csv
import datetime
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

SELECTED = [1, 20, 40, 60, 80, 100, 120, 140]


class Hold(Exception):
    pass


def require(condition, reason):
    if not condition:
        raise Hold(reason)


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def endpoint(path):
    s = Path(path).stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns


def read_json(path, expected=None):
    p = Path(path)
    require(p.is_absolute() and p.resolve() == p and not p.is_symlink() and p.is_file(), 'missing_or_symlink_metadata')
    require(p.stat().st_size <= 32 * 1024 * 1024, 'metadata_over_32MiB')
    before = endpoint(p); raw = p.read_bytes(); after = endpoint(p)
    require(before == after, 'metadata_read_race')
    sha = hashlib.sha256(raw).hexdigest()
    require(expected is None or sha == expected, 'metadata_SHA_mismatch')
    return json.loads(raw), sha


def safe_file_tail(path, cap=65536):
    p = Path(path)
    if not p.is_file():
        return b'', 0
    require(p.resolve() == p and not p.is_symlink(), 'log_symlink_forbidden')
    with p.open('rb') as f:
        f.seek(0, 2); total = f.tell(); f.seek(max(0, total - cap)); raw = f.read(cap)
    return raw, total


def json_tail(path):
    raw, total = safe_file_tail(path)
    for line in reversed(raw.splitlines()):
        try:
            row = json.loads(line)
            if isinstance(row, dict):
                return row, total
        except (ValueError, UnicodeDecodeError):
            continue
    return None, total


def compact_log(path):
    raw, total = safe_file_tail(path)
    text = raw.decode(errors='replace')
    kinds = ['Traceback', 'CUDA out of memory', 'CUDA error', 'Hold:', 'RuntimeError:', 'KeyError:']
    # Return enums and hashes only, never arbitrary traceback/context text.
    return dict(bytes=total, tail_sha256=hashlib.sha256(raw).hexdigest(),
                error_markers=[s for s in kinds if s in text])


def compact_stdout(path):
    row, total = json_tail(path)
    safe = {}
    for key in ['status', 'exception_type', 'reason']:
        if row and key in row and isinstance(row[key], str) and re.fullmatch(r'[A-Za-z0-9_: .-]{1,160}', row[key]):
            safe[key] = row[key]
    if row:
        for key in ['step', 'trajectory_count', 'elapsed_seconds', 'observed_utc']:
            if key in row:
                safe[key] = row[key]
    return dict(bytes=total, last_safe_summary=safe)


def current_process(record):
    if record is None:
        return dict(state='NO_PERSISTED_OWN_IDENTITY')
    p = Path('/proc') / str(record['pid'])
    try:
        raw = (p / 'cmdline').read_bytes(); t = (p / 'stat').read_text(); tail = t[t.rfind(')') + 2:].split()
        actual = dict(pid=int(record['pid']), start_ticks=int(tail[19]), sid=int(tail[3]), uid=p.stat().st_uid,
            state=tail[0], command_sha256=hashlib.sha256(raw.replace(b'\0', b' ').decode(errors='replace').encode()).hexdigest())
        actual['matches_persisted_own_identity'] = all(actual[k] == record[k] for k in ['pid', 'start_ticks', 'sid', 'uid', 'command_sha256'])
        return actual
    except FileNotFoundError:
        return dict(pid=record['pid'], state='ABSENT', absent_is_not_exit_success=True)
    except (PermissionError, ProcessLookupError):
        return dict(pid=record['pid'], state='UNKNOWN_READ_RACE_OR_PERMISSION')


def pane_state(session):
    require(re.fullmatch(r'g1-pair-[A-Za-z0-9_-]+', session) is not None, 'session_name_not_diagnostic')
    r = subprocess.run(['tmux', 'list-panes', '-t', session, '-F', '#{pane_id}|#{pane_pid}|#{pane_dead}|#{pane_dead_status}|#{pane_dead_signal}'],
                       capture_output=True, text=True, timeout=15)
    if r.returncode != 0:
        return dict(state='SESSION_ABSENT_OR_UNKNOWN', success_not_inferred=True)
    lines = r.stdout.splitlines(); require(len(lines) == 1, 'diagnostic_session_not_single_pane')
    fields = lines[0].split('|'); require(len(fields) == 5, 'pane_schema_changed')
    return dict(pane_id=fields[0], pane_pid=int(fields[1]), pane_dead=fields[2] == '1',
        actual_dead_status=fields[3] or None, actual_dead_signal=fields[4] or None,
        wrapper_zero_is_not_scientific_acceptance=True)


def stable_rows_sha(path, expected):
    p = Path(path); require(p.is_file() and not p.is_symlink() and p.resolve() == p, 'accepted_rows_missing_or_symlink')
    before = endpoint(p); sha = sha_file(p); after = endpoint(p)
    require(before == after and sha == expected, 'accepted_rows_SHA_or_identity_changed')
    return dict(sha256=sha, bytes=after[2], fresh_SHA_pass=True)


def compact_accepted(summary_path, output, update, C_base):
    s, digest = read_json(summary_path)
    step = update['step']
    require(s['step'] == step and s['status'] == 'ACCEPTED_RECONSTRUCTED_R_STEP' and
            s['N'] == s['trajectory_count'] == s['actual_probe_rows'] == 2560 and
            s['original_online_row_rewards_available'] is False and
            s['probe_version'] == update['probe_version'] and
            s['original_trace_sha256'] == update['behavior_trace_sha256'], 'accepted_summary_native_binding_mismatch')
    target = Path(s['output_path']); require(target.parent == output and target.name == 'TRAIN-' + str(step) + '-pairs-R-reconstructed.jsonl', 'accepted_rows_wrong_directory')
    row_proof = stable_rows_sha(target, s['output_sha256'])
    c_path = C_base / ('TRAIN-' + str(step) + '-summary.json')
    c, c_sha = read_json(c_path, s['independent_C_summary_sha256'])
    require(c['attempt_id'] == update['attempt_id'] and c['step'] == step and c['trajectory_count'] == 2560 and
            c['native_EM_check_pass'] is True and c['raw_C_fresh_sha256'] == s['independent_C_rows_sha256'], 'accepted_C_binding_mismatch')
    require(endpoint(summary_path)[2] <= 32 * 1024 * 1024, 'summary_changed_to_large_file')
    gate = s['native_scalar_comparison']
    require(gate['passed'] is True and gate['precise_mean_absolute_difference'] <= gate['precise_mean_tolerance'] <= 2e-6 and
            abs(gate['precise_native_update_R_mean'] - update['metrics']['critic/rewards/mean']) <= 1e-15, 'native_full_precision_gate_mismatch')
    require(sha_file(summary_path) == digest and sha_file(c_path) == c_sha, 'accepted_metadata_changed_during_audit')
    return dict(step=step, attempt_id=update['attempt_id'], N=2560, probe_version=s['probe_version'],
        summary_path=str(summary_path), summary_sha256=digest, rows=row_proof,
        independent_C_summary_sha256=c_sha, native_full_precision_gate_pass=True,
        native_R_mean=gate['precise_native_update_R_mean'], reconstructed_R_mean=gate['reconstructed_metrics']['critic/score/mean'],
        precise_mean_absolute_difference=gate['precise_mean_absolute_difference'],
        observed_utc=s['observed_utc'], elapsed_seconds=s['elapsed_seconds'],
        sum_C=s['sum_C'], preliminary_per_pair_Pearson=s['individual_pair_Pearson'],
        Pearson_final_requires_independent_selected_statistics=True,
        original_online_row_rewards_available=False)


def report(request):
    require(sys.platform.startswith('linux') and socket.gethostname() == request['expected_node'] and
            os.environ.get('CUDA_VISIBLE_DEVICES') == '' and 'torch' not in sys.modules, 'wrong_remote_readonly_CPU_context')
    require(request['global_selected_steps'] == SELECTED, 'wrong_selected_scope')
    base = Path(request['original_base']); require(base.resolve() == base, 'base_realpath_mismatch')
    run, run_sha = read_json(base / 'logs/run-manifest.json', request['run_manifest_sha256'])
    require(run['run_id'] == request['native_run_id'] and run['execution_host'] == socket.gethostname(), 'native_run_identity_mismatch')
    raw = (base / 'logs/actual-updates.jsonl').read_bytes()
    require(raw.endswith(b'\n'), 'partial_native_ledger_snapshot')
    updates = [json.loads(x) for x in raw.splitlines()]
    require(len({x['step'] for x in updates}) == len(updates), 'duplicate_native_steps')
    update_map = {x['step']: x for x in updates}
    producer_reports = []; available_steps = set()
    for item in request['producers']:
        r = dict(producer_id=item['producer_id'], requested_steps=item['selected_steps'], GPU_index=item['gpu_index'])
        try:
            config, config_sha = read_json(item['config_path'], item['config_sha256'])
            source = Path(item['source_directory']); manifest, manifest_sha = read_json(source / 'source-manifest.json', item['source_manifest_sha256'])
            require(config['expected_run_id'] == run['run_id'] and config['selected_steps'] == item['selected_steps'] and
                    config['gpu_index'] == item['gpu_index'] and config['base'] == str(base), 'producer_config_binding_mismatch')
            require(all(sha_file(source / name) == sha for name, sha in manifest['source_sha256'].items()), 'producer_frozen_source_SHA_mismatch')
            out = Path(config['output']); require(out.resolve() == out and out.is_relative_to(Path(request['diagnostic_base'])), 'producer_output_not_remote_diagnostic')
            r.update(source_manifest_fresh_SHA_pass=True, config_fresh_SHA_pass=True, output=str(out), pane=pane_state(item['session']))
            inputs = None
            if (out / 'inputs.json').is_file():
                inputs, inp_sha = read_json(out / 'inputs.json')
                require(inputs['original_run_id'] == run['run_id'] and inputs['config_sha256'] == config_sha and
                    inputs['source_manifest_sha256'] == manifest_sha and inputs['requested_replay_steps'] == item['selected_steps'] and
                    inputs['raw_online_R_saved'] is False, 'producer_inputs_binding_mismatch')
                r.update(inputs_sha256=inp_sha, own_identity=inputs['own_identity'],
                    current_process=current_process(inputs['own_identity']))
            else:
                r.update(inputs_sha256=None, current_process=current_process(None), initialization_not_acceptance=True)
            if (out / 'status.json').is_file():
                status, status_sha = read_json(out / 'status.json')
                r['stage'] = {k: status[k] for k in ['status', 'step', 'observed_utc', 'elapsed_seconds', 'completed_replayed_steps'] if k in status}
                r['status_sha256'] = status_sha
            if (out / 'observer-failure.json').is_file():
                failure, failure_sha = read_json(out / 'observer-failure.json')
                r['observer_failure'] = dict(status=failure['status'], observed_utc=failure['observed_utc'],
                    exception_type=failure['exception_type'], reason=failure['reason'],
                    resource_retry_events=failure.get('resource_retry_events', []),
                    proof_sha256=failure_sha)
            mem, mem_bytes = json_tail(out / 'memory.jsonl')
            if mem:
                r['last_memory_observer'] = dict(observed_utc=mem['observed_utc'], actual_free_mib=mem['candidate']['actual_free_mib'],
                    process_memory_mib={str(p['pid']): p['memory_mib'] for p in mem['processes']}, bytes=mem_bytes,
                    stale_observer_is_not_fresh_GPU_query=True)
            r['stdout'] = compact_stdout(Path(request['logs_directory']) / ('gpu' + str(item['gpu_index']) + '.stdout.jsonl'))
            r['stderr'] = compact_log(Path(request['logs_directory']) / ('gpu' + str(item['gpu_index']) + '.stderr.log'))
            r['accepted_steps'] = []
            for step in item['selected_steps']:
                p = out / ('TRAIN-' + str(step) + '-summary.json')
                if p.is_file():
                    require(step in update_map, 'accepted_point_not_native_committed')
                    r['accepted_steps'].append(compact_accepted(p, out, update_map[step], Path(request['C_directory'])))
                    available_steps.add(step)
            pf_path = source / 'preflight_cpu.py'
            spec = importlib.util.spec_from_file_location('progress_pf_' + str(item['gpu_index']), pf_path)
            pf = importlib.util.module_from_spec(spec); spec.loader.exec_module(pf)
            snap = pf.resource_snapshot(base, run, item['gpu_index'])
            processes = []
            own = inputs['own_identity'] if inputs else None
            for proc in snap['compute_processes']:
                is_own = own is not None and pf.same_process(proc, own) and proc['command_sha256'] == own['command_sha256']
                processes.append(dict(pid=proc['pid'], start_ticks=proc['start_ticks'], sid=proc['sid'], uid=proc['uid'],
                    memory_mib=proc['memory_mib'], command_sha256=proc['command_sha256'],
                    recorded_command_sha256=proc.get('recorded_command_sha256'),
                    role='exact_diagnostic' if is_own else proc['recorded_role'],
                    allowed=bool(is_own or proc['matched_recorded_pid_tick_sid_uid'])))
            require(snap['candidate_gpu']['uuid'] == config['gpu_uuid'], 'producer_GPU_UUID_changed')
            r['fresh_GPU'] = dict(snap['candidate_gpu'], processes=processes,
                foreign_or_unbound_pids=[p['pid'] for p in processes if not p['allowed']],
                actual_running_free_floor_mib=config['running_min_free_mib'], own_memory_cap_mib=config['max_replay_memory_mib'],
                no_GPU_allocation_or_signals_by_reporter=True)
        except Exception as e:
            r['read_status'] = 'HOLD_OR_READ_ERROR'
            r['error_type'] = type(e).__name__
            r['error_code'] = str(e) if isinstance(e, Hold) else 'metadata_or_proc_read_error'
        producer_reports.append(r)
    # Retained step1 is counted only once here, never from each worker's total.
    retained, retained_sha = read_json(request['retained_step1_summary_path'], request['retained_step1_summary_sha256'])
    retained_out = Path(retained['output_path']).parent
    retained_report = compact_accepted(Path(request['retained_step1_summary_path']), retained_out, update_map[1], Path(request['C_directory']))
    require(retained_sha == retained_report['summary_sha256'], 'retained_step1_changed')
    available_steps.add(1)
    require('torch' not in sys.modules, 'reporter_imported_Torch')
    return dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), node=socket.gethostname(), uid=os.getuid(),
        native_run_id=run['run_id'], original_run_manifest_sha256=run_sha, native_last_committed_step=max(update_map),
        requested_selected_steps=SELECTED, accepted_unique_selected_steps=sorted(available_steps),
        accepted_unique_pair_count=2560 * len(available_steps), selected_scientific_step_coverage_complete=available_steps == set(SELECTED),
        producer_reports=producer_reports, retained_step1=retained_report,
        raw_payloads_exported=False, model_or_Torch_imported=False, reporter_readonly=True,
        all_pairs_statistics_and_actual_wrapper_exit_still_separate_acceptances=True)


def main():
    request = json.load(sys.stdin)
    result = report(request)
    print(json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False))


if __name__ == '__main__':
    main()
