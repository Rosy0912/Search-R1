"""Copied v6-v3 narrow startup control; not accepted for a new deployment by this copy."""
import json
import subprocess
import time
from pathlib import Path
from formal_protection_contract_v1 import require

def query_gpu_coexistence_snapshot(timeout_seconds, *, command_runner=subprocess.check_output,
                                  monotonic=time.monotonic):
    """Same source parser/checker; exact raw reads are bounded by one shared budget."""
    from search_r1.baselines.preflight import parse_gpu_inventory
    require(0 < timeout_seconds <= 60., 'Invalid GPU cooldown command budget')
    deadline = monotonic() + timeout_seconds
    query = ['nvidia-smi', '--format=csv,noheader,nounits']
    raw = dict(gpu_inventory_csv=None, compute_apps_csv=None)
    try:
        raw['gpu_inventory_csv'] = command_runner(
            query + ['--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu'],
            text=True, timeout=timeout_seconds)
        remaining = deadline - monotonic()
        require(remaining > 0, 'GPU inventory query exceeded cooldown command budget')
        raw['compute_apps_csv'] = command_runner(query + ['--query-compute-apps=gpu_uuid,pid'],
                                                text=True, timeout=remaining)
        require(monotonic() < deadline, 'GPU process query exceeded cooldown command budget')
        inventory = parse_gpu_inventory(raw['gpu_inventory_csv'])
        processes = {line.split(',')[0].strip() for line in raw['compute_apps_csv'].splitlines() if ',' in line}
        return inventory, processes, raw
    except BaseException as error:
        # Preserve malformed/partial originals even when parsing or a later read
        # fails. This is evidence only and cannot substitute missing gate data.
        partial = getattr(error, 'output', None)
        if partial is not None:
            raw['failed_command_partial_output'] = partial.decode(errors='replace') if isinstance(partial, bytes) else partial
        error.gpu_coexistence_raw_snapshot = raw
        raise

def wait_retriever_gpu_coexistence(run, selected, guard, retriever, *, snapshot, checker,
                                  monotonic=time.monotonic, sleep=time.sleep):
    """Keep both original total15% gates; allow own HTTP-probe activity to cool.

    The accepted guard-v2 continues enforcing foreign use throughout. No actor,
    Ray head, sampler or optimizer may start until this separate launch gate passes.
    Dependencies, unknown snapshots and the fixed60s deadline fail closed.
    """
    journal = Path(run) / 'gpu-retriever-coexistence-before-ray.jsonl'
    receipt = Path(run) / 'gpu-retriever-coexistence-before-ray.json'
    started = monotonic()
    deadline = started + 60.
    attempt = 0

    def append(row):
        # Preserve unknown/nonfinite readings without writing invalid JSON.
        import math
        def finite_json(value):
            if isinstance(value, float) and not math.isfinite(value):
                return {'nonfinite_float_repr': repr(value)}
            if isinstance(value, dict):
                return {str(k): finite_json(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [finite_json(v) for v in value]
            return value
        with journal.open('a') as stream:
            stream.write(json.dumps(finite_json(row), allow_nan=False, sort_keys=True) + '\n')
            stream.flush()
        if row['decision'] != 'WAIT_RETRY':
            receipt.write_text(json.dumps(finite_json(row), indent=2, allow_nan=False) + '\n')

    while True:
        attempt += 1
        row = dict(scope='POST_HTTP_LOCAL_RETRIEVER_TOTAL15_COOLDOWN', attempt=attempt,
                   selected_devices=list(selected), timeout_seconds=60., retry_seconds=3.,
                   elapsed_seconds=monotonic() - started, total_sm_limit_percent=15.,
                   total_memory_limit_percent=15., foreign_guard_v2_unchanged=True,
                   guard_pid=guard.pid, retriever_pid=retriever.pid,
                   dependency_exit_codes=dict(guard=guard.poll(), retriever=retriever.poll()),
                   gate_snapshot=None, raw_gpu_sample=None)
        try:
            require(all(v is None for v in row['dependency_exit_codes'].values()),
                    'Owned guard/retriever exited during GPU cooldown')
            require(monotonic() < deadline, 'Local GPU retriever exceeds existing launch eligibility after60s')
            # Exact raw CSV and its parsed gate inventory come from the same
            # bounded reads, not a later potentially quieter substitute snapshot.
            row['sample_started_elapsed_seconds'] = monotonic() - started
            inventory, processes, raw = snapshot(deadline - monotonic())
            row['raw_gpu_sample'] = raw
            row['gate_sample_elapsed_seconds'] = monotonic() - started
            row['gate_snapshot'] = dict(inventory=inventory, compute_gpu_uuids=sorted(processes))
            row['dependency_exit_codes'] = dict(guard=guard.poll(), retriever=retriever.poll())
            require(all(v is None for v in row['dependency_exit_codes'].values()),
                    'Owned guard/retriever exited while reading GPU cooldown snapshot')
            require(monotonic() < deadline, 'GPU cooldown snapshot exceeded60s deadline')
            try:
                row['selected_gpu_records'] = checker(selected, inventory, processes)
            except RuntimeError as error:
                # Retry ONLY the frozen checker's explicit total occupancy case.
                # NaN/unknown/invalid readings must not become a later admission.
                retry_messages = {f'Allocated GPU {device} exceeds the 15% idle thresholds'
                                  for device in selected}
                if str(error) not in retry_messages:
                    raise
                # The frozen global checker stops at its first overbusy GPU.
                # Validate EVERY selected GPU before treating that short-circuit
                # as a retry: later unknown data or duplicate UUIDs are fatal.
                resolved_uuids = []
                for device in selected:
                    matches = [v for index, v in inventory.items()
                               if device == index or device == v['uuid']]
                    require(len(matches) == 1, 'Cannot resolve every allocated GPU during cooldown retry')
                    resolved_uuids.append(matches[0]['uuid'])
                    try:
                        checker([device], inventory, processes)
                    except RuntimeError as device_error:
                        if str(device_error) != f'Allocated GPU {device} exceeds the 15% idle thresholds':
                            raise
                require(len(set(resolved_uuids)) == len(selected),
                        'Duplicate allocated GPU UUIDs during cooldown retry')
                row.update(decision='WAIT_RETRY', eligibility_error_type=type(error).__name__,
                           eligibility_error=str(error))
            else:
                row['dependency_exit_codes'] = dict(guard=guard.poll(), retriever=retriever.poll())
                require(all(v is None for v in row['dependency_exit_codes'].values()),
                        'Owned guard/retriever exited before GPU coexistence admission')
                require(monotonic() < deadline, 'GPU coexistence admission exceeded60s deadline')
                row.update(decision='PASS_ORIGINAL_TOTAL15', elapsed_seconds=monotonic() - started)
                append(row)
                return row
        except BaseException as error:
            if hasattr(error, 'gpu_coexistence_raw_snapshot'):
                row['raw_gpu_sample'] = error.gpu_coexistence_raw_snapshot
            row.update(decision='FAILED_CLOSED', error_type=type(error).__name__, error=str(error),
                       elapsed_seconds=monotonic() - started)
            append(row)
            raise
        append(row)
        sleep(min(3., max(0., deadline - monotonic())))
