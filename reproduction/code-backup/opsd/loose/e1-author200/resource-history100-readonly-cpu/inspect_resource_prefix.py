"""Read only the actual immutable local resource prefix; no network/GPU imports."""
import collections
import datetime
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import stat

HERE = Path(__file__).resolve().parent
W = HERE.parent
SNAP = W / 'actual-snapshots/3261-20261006T005604410916Z'
SOURCE = '188f4e5e7b63cf479290d0888a4a075499886750'
EXPECTED_GZIP = '0e2a365c309f8adce1db342e3e15429e74f59d76cc087e995cb1330e552c0674'
EXPECTED_PREFIX = '3c2bda6fd8eedf456875ab1e40adf708f8bf388fd6a31254bca55f40e7af9b82'
EXPECTED_BYTES = 648787899
STARTS = {1531763: '1829074494', 1533312: '1829085316', 1533910: '1829086072',
          1534229: '1829086582', 1534230: '1829086583', 1534231: '1829086584',
          1534232: '1829086585', 1534233: '1829086586', 1534234: '1829086587',
          1534235: '1829086588'}
WORKERS = set(STARTS) - {1531763, 1533312}


def need(value, message):
    if not value:
        raise ValueError(message)


def identity(value):
    return {k: getattr(value, k) for k in ('st_dev', 'st_ino', 'st_uid', 'st_mode',
                                          'st_size', 'st_mtime_ns', 'st_ctime_ns')}


def physical_fd(path):
    path = Path(path)
    need(path.is_absolute() and '..' not in path.parts, 'Absolute physical local input required')
    parent = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for name in path.parts[1:-1]:
            child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            os.close(parent)
            parent = child
        return os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
    finally:
        os.close(parent)


def own_regular(fd):
    value = os.fstat(fd)
    need(stat.S_ISREG(value.st_mode) and value.st_uid == os.getuid(), 'Owned regular local original required')
    return identity(value)


def stable_after(fd, path, before):
    again = physical_fd(path)
    try:
        need(own_regular(fd) == before == own_regular(again), 'Local original changed while reading')
    finally:
        os.close(again)


def exact_json(path):
    fd = physical_fd(path)
    try:
        before = own_regular(fd)
        with os.fdopen(fd, 'rb', closefd=False) as reader:
            raw = reader.read()
        stable_after(fd, path, before)
        def unique(items):
            result = {}
            for k, v in items:
                need(k not in result, 'Duplicate JSON field')
                result[k] = v
            return result
        value = json.loads(raw, object_pairs_hook=unique,
                           parse_constant=lambda value: need(False, 'Nonfinite JSON'))
        json.dumps(value, allow_nan=False)
        return value, {'physical_path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(),
                       'bytes': len(raw), 'stable_identity': before}
    finally:
        os.close(fd)


def quantile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil((len(ordered) - 1) * fraction))] if ordered else None


def numeric(value):
    return type(value) in (int, float) and math.isfinite(value)


def describe(values):
    return {'samples': len(values), 'min_mib': min(values) if values else None,
            'max_mib': max(values) if values else None, 'p50_mib': quantile(values, .5),
            'p95_mib': quantile(values, .95),
            'min_gib': min(values) / 1024 if values else None,
            'max_gib': max(values) / 1024 if values else None,
            'p50_gib': quantile(values, .5) / 1024 if values else None,
            'p95_gib': quantile(values, .95) / 1024 if values else None}


def main():
    summary, summary_record = exact_json(SNAP / 'observation-summary.json')
    previous, previous_record = exact_json(W / 'ACTUAL100_PROGRESS_AND_FIXED_SET_EFFECT_INDEPENDENT_REVIEW.json')
    history = summary['resource_history']
    need(history['included'] is True and history['complete_prefix_at_start'] is True
         and history['gzip_sha256'] == EXPECTED_GZIP and history['sha256'] == EXPECTED_PREFIX
         and type(history['prefix_bytes']) is int and history['prefix_bytes'] == EXPECTED_BYTES,
         'Actual snapshot prefix metadata differs')
    need(previous['resources_captured_prefix']['sha256'] == EXPECTED_PREFIX
         and previous['resources_captured_prefix']['gzip_sha256'] == EXPECTED_GZIP
         and previous['resources_captured_prefix']['bytes'] == EXPECTED_BYTES,
         'Independent100 prefix binding differs')
    need(summary['node'] == 'h100-10' and summary['job_id'] == '3261'
         and type(summary['reader_uid']) is int and summary['reader_uid'] == 0,
         'Actual local snapshot remote node/job/UID binding differs')
    current_processes = {}
    for pid, start in STARTS.items():
        process = summary['processes'][str(pid)]
        need(type(process['pid']) is int and process['pid'] == pid and process['start_ticks'] == start
             and process['uids'] == [0, 0, 0, 0] and all(type(v) is int for v in process['uids'])
             and process['in_observed_job'] is True
             and process['cgroup'] == '1:freezer:/slurm_h100-10/uid_0/job_3261/step_0\n0::/system.slice/slurmd.service\n',
             'Actual original retriever/driver/worker process identity differs')
        env = process['environment_allowlist']
        need(env['SLURM_JOB_ID'] == '3261' and env['SLURM_JOB_NODELIST'] == 'h100-10'
             and env['SEARCH_R1_REPO'].endswith('/repo-188f4e5e7b63')
             and env['OPSD_E1_RETRIEVER_URL'] == 'http://127.0.0.1:18084/retrieve',
             'Snapshot own-node/loopback environment differs')
        current_processes[str(pid)] = {k: process[k] for k in
            ('pid', 'start_ticks', 'uids', 'cgroup', 'in_observed_job', 'stdout_fd_target')}
    need(summary['commands']['source_head']['returncode'] == 0
         and summary['commands']['source_head']['stdout'].strip() == SOURCE
         and summary['commands']['source_porcelain']['returncode'] == 0
         and summary['commands']['source_porcelain']['stdout'] == '', 'Observed source188 clean binding differs')
    faiss, faiss_record = exact_json(SNAP / 'original-files/actual-faiss-runtime.json')
    path = SNAP / 'resources-prefix-original.ndjson.gz'
    fd = physical_fd(path)
    before = own_regular(fd)
    count = decoded = terminated = 0
    compressed_hash = hashlib.sha256()
    full_hash = hashlib.sha256()
    times, intervals, gaps, errors = [], [], [], collections.Counter()
    anomalies, unknown_apps = [], {}
    all_values = {str(i): [] for i in range(8)}
    active_values = {str(i): [] for i in range(8)}
    uuids, totals, peaks = {}, {}, {}
    sums, active_sums = [], []
    observed_identity = {str(pid): {'job_samples': 0, 'gpu_application_samples': 0,
                                  'first_job_time': None, 'last_job_time': None,
                                  'first_gpu_time': None, 'last_gpu_time': None,
                                  'gpu_mib': []} for pid in STARTS}
    all8_active = 0
    try:
        with os.fdopen(fd, 'rb', closefd=False) as reader:
            while True:
                block = reader.read(1024 * 1024)
                if not block:
                    break
                compressed_hash.update(block)
            need(reader.tell() == before['st_size'] == history['gzip_bytes']
                 and compressed_hash.hexdigest() == EXPECTED_GZIP, 'Whole gzip identity differs')
            reader.seek(0)
            with gzip.GzipFile(fileobj=reader, mode='rb') as uncompressed:
                for line in uncompressed:
                    count += 1
                    decoded += len(line)
                    full_hash.update(line)
                    terminated += int(line.endswith(b'\n'))
                    try:
                        row = json.loads(line)
                        need(isinstance(row, dict), 'Resource row is not object')
                        t = datetime.datetime.fromisoformat(row['time'])
                        need(t.utcoffset() == datetime.timedelta(0), 'Resource time not UTC')
                    except Exception as error:
                        errors['malformed_row'] += 1
                        if len(anomalies) < 20:
                            anomalies.append({'line': count, 'bytes': len(line), 'sha256': hashlib.sha256(line).hexdigest(), 'error': str(error)})
                        continue
                    if times:
                        delta = (t - times[-1]).total_seconds()
                        intervals.append(delta)
                        if delta <= 0:
                            errors['non_increasing_timestamp'] += 1
                        if delta > 10:
                            gaps.append({'from': times[-1].isoformat(), 'to': t.isoformat(), 'seconds': delta})
                    times.append(t)
                    if row.get('job_id') != '3261': errors['wrong_job_record'] += 1
                    if type(row.get('nominal_sleep_seconds')) is not int or row['nominal_sleep_seconds'] != 2:
                        errors['wrong_nominal_sleep'] += 1
                    if 'query_error' in row: errors['query_error:' + str(row['query_error'])] += 1
                    app_workers = set()
                    for role, field in [('job', 'job_processes'), ('gpu', 'gpu_processes')]:
                        for process in row.get(field, []):
                            pid = process.get('pid')
                            if process.get('unavailable') is True:
                                errors[role + '_process_unavailable'] += 1
                                continue
                            if pid in STARTS:
                                if type(pid) is not int or process.get('process_start_ticks') != STARTS[pid]:
                                    errors[role + '_original_pid_reused_or_wrong_start'] += 1
                                    continue
                                if process.get('uid') != '0' or process.get('in_slurm_job') is not True:
                                    errors[role + '_original_uid_or_job_differs'] += 1
                                    continue
                                record = observed_identity[str(pid)]
                                if role == 'job':
                                    record['job_samples'] += 1
                                    record['first_job_time'] = record['first_job_time'] or row['time']
                                    record['last_job_time'] = row['time']
                                else:
                                    record['gpu_application_samples'] += 1
                                    record['first_gpu_time'] = record['first_gpu_time'] or row['time']
                                    record['last_gpu_time'] = row['time']
                                    try:
                                        amount = float(process['gpu_used_mib'])
                                        need(math.isfinite(amount) and amount >= 0, 'Invalid app memory')
                                        record['gpu_mib'].append(amount)
                                    except Exception:
                                        errors['invalid_gpu_process_memory'] += 1
                                    if pid in WORKERS: app_workers.add(pid)
                            elif role == 'gpu':
                                key = (pid, process.get('process_start_ticks'), process.get('in_slurm_job'), process.get('uid'))
                                unknown_apps.setdefault(key, {'samples': 0, 'first': row['time'], 'last': row['time']})['samples'] += 1
                                unknown_apps[key]['last'] = row['time']
                                if process.get('in_slurm_job') is not True or process.get('uid') != '0':
                                    errors['nonowned_or_nonroot_gpu_application'] += 1
                    active = app_workers == WORKERS
                    all8_active += int(active)
                    gpu_rows = row.get('gpus', [])
                    indices = [v.get('index') for v in gpu_rows]
                    if sorted(indices) != [str(i) for i in range(8)] or len(set(indices)) != 8:
                        errors['missing_or_duplicate_gpu_inventory'] += 1
                    valid_memory, seen = [], set()
                    for gpu in gpu_rows:
                        index = gpu.get('index')
                        if index not in all_values or index in seen:
                            errors['foreign_or_duplicate_gpu_index'] += 1
                            continue
                        seen.add(index)
                        used, total = gpu.get('used_mib'), gpu.get('total_mib')
                        if not numeric(used) or not numeric(total) or not 0 <= used <= total:
                            errors['invalid_gpu_memory'] += 1
                            continue
                        if index in uuids and (uuids[index] != gpu.get('uuid') or totals[index] != total):
                            errors['gpu_uuid_or_capacity_changed'] += 1
                        uuids[index], totals[index] = gpu.get('uuid'), total
                        all_values[index].append(used)
                        valid_memory.append(used)
                        if active: active_values[index].append(used)
                        if index not in peaks or used > peaks[index]['used_mib']:
                            peaks[index] = {'used_mib': used, 'used_gib': used / 1024, 'time': row['time'], 'line': count}
                    if len(valid_memory) == 8:
                        sums.append(sum(valid_memory))
                        if active: active_sums.append(sum(valid_memory))
        stable_after(fd, path, before)
    finally:
        os.close(fd)
    need(decoded == EXPECTED_BYTES and full_hash.hexdigest() == EXPECTED_PREFIX, 'Decompressed exact prefix full SHA differs')
    for record in observed_identity.values():
        record['gpu_application_memory'] = describe(record.pop('gpu_mib'))
    output = {'version': 1, 'scope': 'ACTUAL100_CAPTURED_PREFIX_SAMPLED_RESOURCE_STATISTICS_NOT_CONTINUOUS_PEAK_TERMINAL_COST_OR_RELEASE',
        'source_commit': SOURCE, 'job_id': '3261', 'node': 'h100-10',
        'observed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'gzip_original': {'physical_path': str(path), 'bytes': before['st_size'], 'sha256': compressed_hash.hexdigest(), 'stable_identity': before},
        'decompressed_prefix': {'bytes': decoded, 'sha256': full_hash.hexdigest(), 'gzip_crc_eof_completed': True,
            'line_count': count, 'newline_terminated_lines': terminated, 'complete_final_line': count == terminated,
            'physical_remote_path': history['physical_path'], 'original_remote_metadata': history},
        'binding_originals': {'snapshot_summary': summary_record, 'independent_actual100_report': previous_record,
            'faiss_runtime_json': faiss_record},
        'time_coverage': {'first': times[0].isoformat(), 'last': times[-1].isoformat(),
            'sampled_span_seconds': (times[-1] - times[0]).total_seconds(), 'valid_timestamp_records': len(times),
            'nominal_sleep_seconds': 2, 'gap_threshold_seconds': 10,
            'interval_min_seconds': min(intervals), 'interval_p50_seconds': quantile(intervals, .5),
            'interval_p95_seconds': quantile(intervals, .95), 'interval_max_seconds': max(intervals),
            'gaps_over10_seconds': len(gaps), 'largest_gaps': sorted(gaps, key=lambda v: v['seconds'], reverse=True)[:20],
            'gap_interpretation': 'Timestamps precede NVML and /proc queries; intervals include measurement work plus two-second sleep.'},
        'per_gpu_all_samples': {i: {'uuid': uuids[i], 'total_mib': totals[i], **describe(v),
            'observed_peak_record': peaks[i]} for i, v in all_values.items()},
        'per_gpu_original8workers_present_samples': {i: describe(v) for i, v in active_values.items()},
        'eight_gpu_simultaneous_sum': describe(sums),
        'eight_gpu_simultaneous_sum_original8workers_present': describe(active_sums),
        'original8workers_present_records': all8_active,
        'record_or_query_errors': dict(errors), 'malformed_row_details_first20': anomalies,
        'original_process_history': observed_identity, 'actual_snapshot_original_process_identity': current_processes,
        'other_gpu_application_identities': [{'pid': key[0], 'start_ticks': key[1], 'in_slurm_job': key[2], 'uid': key[3], **value}
            for key, value in sorted(unknown_apps.items(), key=lambda item: str(item[0]))],
        'faiss_runtime_original': faiss,
        'limitations': [
            'Observed maxima are discrete sampled device memory, not allocator max_memory_allocated or continuous true peaks.',
            'Whole prefix SHA establishes exact copied bytes; the remote writer was still growing, so this is not writer closure or final accounting.',
            'All-sample minima include initialization and hardware checks. The second view requires all eight original policy-worker PID/start identities in that same GPU-app query.',
            'Resource timestamps are sampling timestamps and do not identify outer iterations or pair reward/effect measurements.',
            'The cgroup memory paths include system.slice/slurmd.service; they are not asserted as an exclusive per-job memory or OOM budget.',
            'GPU process lists bind sampled PID/start/UID/job telemetry, while h100-10/loopback binds the actual snapshot reader and process environment. No remote call was made by this analysis.',
            'Empty or unavailable process observations do not establish resource release. This report makes no terminal cost, true continuous peak, or final acceptance claim.']}
    raw = (json.dumps(output, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()
    target = HERE / 'ACTUAL100_SAMPLED_RESOURCE_PREFIX_INDEPENDENT_STATISTICS.json'
    with target.open('xb') as writer:
        writer.write(raw)
        writer.flush()
        os.fsync(writer.fileno())
    print(json.dumps({'report_path': str(target), 'report_sha256': hashlib.sha256(raw).hexdigest(),
        'actual_gzip_sha256': compressed_hash.hexdigest(), 'actual_prefix_sha256': full_hash.hexdigest(),
        'prefix_bytes': decoded, 'records': count, 'first': output['time_coverage']['first'],
        'last': output['time_coverage']['last'], 'gaps_over10_seconds': len(gaps),
        'record_or_query_errors': dict(errors),
        'per_gpu_sampled_peak_gib': {i: v['used_gib'] for i, v in peaks.items()},
        'scope': output['scope']}, sort_keys=True))


if __name__ == '__main__':
    main()
