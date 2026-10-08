"""Active leased-own occupancy accounting; initial absolute15% remains public preflight."""
import json
import math
import os
from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor
from eval64 import require

FRACTION = .40
RESERVE_MIB = 2048.


def pid_identity(pid):
    proc = Path('/proc') / str(pid)
    stat = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=int(pid), starttime_ticks=int(stat[19]), cgroup=(proc / 'cgroup').read_text().splitlines())


def nvml_snapshot():
    query = ['nvidia-smi', '--format=csv,noheader,nounits']
    # Both original read-only queries must finish before any snapshot is returned.
    # Executor exit joins the sibling even if either future raises.
    with ThreadPoolExecutor(max_workers=2) as executor:
        gpu_future = executor.submit(subprocess.check_output, query + ['--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu'], text=True, timeout=10)
        app_future = executor.submit(subprocess.check_output, query + ['--query-compute-apps=gpu_uuid,pid,used_gpu_memory'], text=True, timeout=10)
        gpus, apps = gpu_future.result(), app_future.result()
    gpu_rows = []
    for line in gpus.splitlines():
        i, uid, used, total, sm = [v.strip() for v in line.split(',')]
        gpu_rows.append(dict(index=i, uuid=uid, used_mib=float(used), total_mib=float(total), sm=float(sm)))
    app_rows = []
    for line in apps.splitlines():
        uid, pid, used = [v.strip() for v in line.split(',')]
        app_rows.append(dict(uuid=uid, pid=int(pid), used_mib=float(used)))
    return dict(gpus=gpu_rows, applications=app_rows,
                identities={str(row['pid']): pid_identity(row['pid']) for row in app_rows})


def sm_aggregate_bounds(snapshot, foreign, own, windows, now_us, *, actual_query_completed_us=None):
    """Return aggregate bounds, never fake numeric values for absent pmon PIDs.

    A SUCCESS sparse list is interpretable only with a fresh positive known-own
    sample proving that this device's supported process sampler returned actual positive activity.
    Foreign bound is the sum of per-PID maxima among this query's returned records.
    It is not a continuous/instantaneous bound over unreturned historical activity.
    Global utilization bounds foreign aggregate when it alone is <=15.
    """
    query_completed_us = now_us if actual_query_completed_us is None else actual_query_completed_us
    require(query_completed_us <= now_us, 'Query completion later than evaluation clock')
    bounds, evidence = {}, {}
    for gpu in snapshot['gpus']:
        uid, sm = gpu['uuid'], gpu['sm']
        if not any(u == uid for u, _ in foreign):
            continue
        require(math.isfinite(sm) and 0 <= sm <= 100, 'Unknown device utilization')
        if sm <= 15:
            bounds[uid] = sm
            evidence[uid] = dict(method='DEVICE_TOTAL_AGGREGATE_UPPER_BOUND', upper_bound=sm,
                                 absent_pid_values='NOT_INFERRED')
            continue
        w = windows.get(uid)
        require(w is not None and w['status'] == 'SUCCESS', 'High-device-SM needs supported fresh process samples')
        start, end = w['start_us'], w['end_us']
        require(end <= query_completed_us, 'Window end later than original query completion')
        require(0 <= now_us-end <= 2_000_000 and 1_000_000 <= end-start <= 3_000_000,
                'Stale or invalid process sampling window')
        rows = w['samples']
        require(rows, 'Empty high-SM sampling window')
        known = {pid for u, pid in foreign | own if u == uid}
        for r in rows:
            require(r['pid'] in known, 'Unknown PID in process utilization samples')
            require(start < r['timestamp_us'] <= end and 0 <= now_us-r['timestamp_us'] <= 1_500_000,
                    'Stale/future process sample')
            require(math.isfinite(r['sm']) and 0 <= r['sm'] and ((uid, r['pid']) in own or r['sm'] <= 100), 'Invalid process SM')
        positive_own = [r for r in rows if (uid, r['pid']) in own and r['sm'] > 0]
        require(positive_own, 'No positive known-own sample proves active process sampler')
        # NVML returns nonzero-filtered per-process records; recording times may differ.
        # Do not derive anything from NOT_FOUND, pmon '-', or a previous window.
        maxima = {}
        for r in rows:
            if (uid, r['pid']) in foreign:
                maxima[r['pid']] = max(maxima.get(r['pid'], 0), r['sm'])
        bound = sum(maxima.values())
        bounds[uid] = bound
        evidence[uid] = dict(method='FRESH_SUPPORTED_NONZERO_PROCESS_WINDOW', upper_bound=bound,
                            window=w, positive_own_pids=sorted({r['pid'] for r in positive_own}),
                            foreign_observed_maxima=maxima,
                            query_completed_us=query_completed_us, evaluation_now_us=now_us,
                            timestamp_min_us=min(r['timestamp_us'] for r in rows),
                            timestamp_max_us=max(r['timestamp_us'] for r in rows),
                            timestamp_span_us=max(r['timestamp_us'] for r in rows)-min(r['timestamp_us'] for r in rows),
                            aggregate_scope='SUM_OF_PER_FOREIGN_PID_MAXIMA_IN_RETURNED_RECORDS_NOT_CONTINUOUS_BOUND',
                            absent_interpretation='NONZERO_FILTER_WINDOW_INFERENCE_NOT_MEASURED',
                            foreign_absent_pids=sorted(pid for u,pid in foreign if u == uid and pid not in maxima))
    return bounds, evidence


def collect_process_windows(snapshot, foreign):
    """No CUDA initialization; only query NVML on devices needing process evidence."""
    import time
    import pynvml as nv
    targets = [g for g in snapshot['gpus'] if g['sm'] > 15 and any(u == g['uuid'] for u,_ in foreign)]
    if not targets:
        return {}, int(time.time()*1_000_000)
    nv.nvmlInit()
    try:
        start = int(time.time()*1_000_000)
        time.sleep(1.0)
        def query_device(g):
            query_start_us = int(time.time()*1_000_000)
            handle = nv.nvmlDeviceGetHandleByUUID(g['uuid'])
            try:
                rows = nv.nvmlDeviceGetProcessUtilization(handle, start)
                status = 'SUCCESS'
                samples = [dict(pid=int(r.pid), timestamp_us=int(r.timeStamp), sm=float(r.smUtil)) for r in rows]
            except nv.NVMLError as exc:
                status, samples = 'NVML_ERROR_' + str(exc.value), []
            query_end_us = int(time.time()*1_000_000)
            return g['uuid'], dict(start_us=start, end_us=query_end_us,
                                  query_start_us=query_start_us, query_end_us=query_end_us,
                                  status=status, samples=samples)
        # All query workers join before nvmlShutdown, including unexpected failures.
        with ThreadPoolExecutor(max_workers=len(targets)) as executor:
            futures = [executor.submit(query_device, g) for g in targets]
            windows = dict(future.result() for future in futures)
        return windows, int(time.time()*1_000_000)
    finally:
        nv.nvmlShutdown()


def accounting_gate(snapshot, baseline, uuid_set, service, relative, foreign_sm, pre_model=False, validate_sm=True):
    gpus = {g['uuid']: g for g in snapshot['gpus'] if g['uuid'] in uuid_set}
    require(len(gpus) == 8 and all(all(math.isfinite(g[k]) for k in ('used_mib', 'total_mib', 'sm'))
            and g['total_mib'] > 0 and 0 <= g['used_mib'] <= g['total_mib'] and 0 <= g['sm'] <= 100 for g in gpus.values()),
            'Unknown allocated GPU measurements')
    require(all(math.isfinite(p['used_mib']) and p['used_mib'] >= 0 for p in snapshot['applications']), 'Unknown process memory')
    original = {(p['uuid'], p['pid']) for p in baseline['applications'] if p['uuid'] in uuid_set}
    current = {(p['uuid'], p['pid']) for p in snapshot['applications'] if p['uuid'] in uuid_set}
    require(original <= current, 'Foreign baseline PID/UUID disappeared')
    own = set()
    for p in snapshot['applications']:
        ident = snapshot['identities'][str(p['pid'])]
        member = any(line.split(':', 2)[-1] == relative or line.split(':', 2)[-1].startswith(relative + '/') for line in ident['cgroup'])
        key = (p['uuid'], p['pid'])
        if key in original:
            old = baseline['identities'][str(p['pid'])]
            require(ident['starttime_ticks'] == old['starttime_ticks'] and ident['cgroup'] == old['cgroup'], 'Foreign PID reused or scope changed')
        else:
            require(member and p['uuid'] in uuid_set, 'Unknown/foreign GPU process or unleased UUID')
            own.add(key)
    actual_service = snapshot['identities'].get(str(service['pid']))
    require(actual_service == service, 'Retriever PID/starttime/cgroup changed')
    require({uid for uid, pid in own if pid == service['pid']} == uuid_set, 'Retriever must actually reside on all8 leased UUIDs')
    if pre_model:
        require(own == {(uid, service['pid']) for uid in uuid_set}, 'Unexpected GPU owner before model startup')
    for uid, g in gpus.items():
        foreign = [p for p in snapshot['applications'] if (p['uuid'], p['pid']) in original and p['uuid'] == uid]
        require(sum(p['used_mib'] for p in foreign) / g['total_mib'] <= .15, 'Foreign GPU memory exceeds15%')
        if foreign and validate_sm:
            bound = foreign_sm.get(uid)
            require(bound is not None and math.isfinite(bound) and 0 <= bound <= 15,
                    'Foreign aggregate SM exceeds15% or unknown')
        if pre_model:
            require(g['used_mib'] / g['total_mib'] <= .15, 'Pre-model absolute GPU memory exceeds15%')
            require(g['sm'] <= 15, 'Retriever must settle before profiling')
            require(g['total_mib'] - g['used_mib'] >= FRACTION * g['total_mib'] + RESERVE_MIB,
                    'Insufficient actual free memory for eval40% budget+2GiB reserve')
    return dict(foreign_pid_uuids=sorted(original), own_pid_uuids=sorted(own),
                gpu_memory_utilization=FRACTION, reserve_mib=RESERVE_MIB,
                gpu_rows=list(gpus.values()), foreign_sm_aggregate_upper_bounds=foreign_sm)


def active_gate(runtime, pre_model=False):
    from search_r1.baselines.topd_local_gpu import lease_ownership
    leases = json.loads(os.environ.get('SEARCH_R1_GPU_LEASES', '[]'))
    require(os.environ.get('SEARCH_R1_GPU_PREFLIGHT') == '1' and len(leases) == 8
            and len({r['uuid'] for r in leases}) == 8, 'Require actual8 inherited held UUID leases')
    for record in leases:
        lease_ownership(record)
    uuids = {r['uuid'] for r in leases}
    snapshot = nvml_snapshot()
    index_map = {g['index']: g['uuid'] for g in snapshot['gpus']}
    selected = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    require(len(selected) == 8 and {index_map.get(v, v) for v in selected} == uuids,
            'Actual visibleGPU allocation differs from leased UUIDs')
    foreign = {(p['uuid'], p['pid']) for p in runtime['baseline']['applications'] if p['uuid'] in uuids}
    # Establish identity ownership with the unchanged strict identity/memory gate
    # before any sparse-process samples can be interpreted as known-own activity.
    identity_record = accounting_gate(snapshot, runtime['baseline'], uuids, runtime['retriever_identity'],
                                      runtime['cgroup'], {}, pre_model, validate_sm=False)
    own = {tuple(row) for row in identity_record['own_pid_uuids']}
    raw_pmon = subprocess.check_output(['nvidia-smi', 'pmon', '-s', 'u', '-c', '1'], text=True, timeout=15)
    windows, now_us = collect_process_windows(snapshot, foreign)
    sm, evidence = sm_aggregate_bounds(snapshot, foreign, own, windows, now_us)
    initial_sm, initial_evidence = dict(sm), dict(evidence)
    # Re-read identities/memory/device SM after sampling, closing process reuse/exit
    # races. Low-SM shortcut must still be low in the final snapshot.
    final = nvml_snapshot()
    require(final['identities'] == snapshot['identities'], 'Process identities changed during SM window')
    final_gpus = {g['uuid']: g for g in final['gpus']}
    for uid, row in evidence.items():
        if row['method'] == 'DEVICE_TOTAL_AGGREGATE_UPPER_BOUND':
            require(final_gpus[uid]['sm'] <= 15, 'Device SM increased after upper-bound sample')
            sm[uid] = max(sm[uid], final_gpus[uid]['sm'])
    initial_snapshot = snapshot
    snapshot = final
    import time
    sm, evidence = sm_aggregate_bounds(snapshot, foreign, own, windows, int(time.time()*1_000_000),
                                       actual_query_completed_us=now_us)
    # pmon is retained for provenance; '-' is never interpreted by this gate.
    for uid, row in evidence.items():
        final_bound = sm[uid]
        sm[uid] = max(initial_sm[uid], final_bound)
        row.update(upper_bound=sm[uid], initial_foreign_upper_bound=initial_sm[uid],
                   final_foreign_upper_bound=final_bound,
                   initial_evidence=initial_evidence[uid], scope='TWO_FOREIGN_BOUND_SNAPSHOTS_MAXIMUM')
    record = accounting_gate(snapshot, runtime['baseline'], uuids, runtime['retriever_identity'], runtime['cgroup'], sm, pre_model)
    record.update(raw_snapshot=snapshot, foreign_sm_evidence=evidence, raw_foreign_pmon=raw_pmon, candidate_scope='UNVALIDATED_MULTI_PID_WINDOW_ON_GPU')
    return record
