"""Actual sampled eight-device occupancy over this process's first five outers.

The existing98% sampled occupancy criterion is retained. Acquisition gaps and
query durations are reported as observed, without inventing a new gap limit or
claiming continuous instantaneous peak coverage.
"""
import hashlib
import math
import os
import time
from pathlib import Path

from reproduction.common8.policy_calibration_plan import canonical_bytes, _strict_json


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _owned(path):
    path = Path(path)
    assert path.is_file() and not path.is_symlink() and path.resolve() == path
    assert path.stat().st_uid == os.getuid(), 'GPU evidence file owner differs'
    return path


def _identity(pid):
    from reproduction.a800.owned_processes import identity
    actual = identity(pid)
    assert actual['state'] != 'Z', 'GPU admission process has exited'
    return {key: actual[key] for key in ('pid', 'start_ticks', 'sid', 'uid')}


def _live_producer(base, claim, producer):
    assert all(type(producer[key]) is int for key in ('pid', 'start_ticks', 'sid', 'uid'))
    assert producer['pid'] > 0 and producer['start_ticks'] > 0 and producer['uid'] == os.getuid()
    assert _identity(producer['pid']) == producer, 'GPU sampler process birth changed'
    ledger = _strict_json(_owned(base / 'logs/process-ledger.json').read_bytes())
    assert ledger['run_id'] == claim['run_id']
    trainer = ledger['jobs']['trainer']
    assert trainer['state'] == 'STARTED', 'GPU admission trainer is no longer active'
    leader = trainer['leader']
    assert (leader['pid'], leader['start_ticks']) == (claim['shell_pid'], claim['shell_start_ticks'])
    assert _identity(leader['pid']) == {key: leader[key] for key in ('pid', 'start_ticks', 'sid', 'uid')}, \
        'Actual training shell birth changed'
    stat = (Path('/proc') / str(leader['pid']) / 'stat').read_text().rsplit(')', 1)[1].split()
    assert int(stat[1]) == producer['pid'], 'GPU sampler is not the claimed trainer supervisor'


def _complete_rows(path):
    raw = _owned(path).read_bytes()
    # The sampler appends while this reader runs. Ignore only an unfinished
    # final write; a malformed completed line remains a hard failure.
    complete = raw[:raw.rfind(b'\n') + 1]
    rows = [_strict_json(line) for line in complete.splitlines()]
    assert rows, 'No actual GPU occupancy samples'
    return complete, rows


def _sample_after_entry(base, run, claim, entered):
    path = base / 'logs' / ('gpu-samples-' + run['run_id'] + '.jsonl')
    raw, rows = _complete_rows(path)
    stat = path.stat()
    inode = (stat.st_dev, stat.st_ino)
    producer = rows[-1]['sampler_identity']
    deadline=time.monotonic()+60  # Operational wait bound, not a new scientific gap/peak criterion.
    while True:
        _live_producer(base, claim, producer)
        assert rows[-1]['sampler_identity'] == producer, 'GPU sampler identity changed'
        if _number(rows[-1]['query_started_unix']) and rows[-1]['query_started_unix'] >= entered:
            return path, raw, rows, producer
        assert time.monotonic()<deadline, 'GPU sampler made no fresh query progress within the bounded gate wait'
        # This is the existing sampler cadence, not a new allowable data gap.
        time.sleep(2)
        newer, rows = _complete_rows(path)
        current = path.stat()
        assert (current.st_dev, current.st_ino) == inode and newer.startswith(raw), \
            'Actual GPU sample ledger was replaced or rewritten'
        raw = newer


def _coverage(rows, run, claim, allocation, scheduler_receipts, *, entered):
    """Pure receipt/sample reader; caller separately authenticates live births."""
    assert allocation['accepted'] is True and allocation['run_id'] == claim['run_id'] == run['run_id']
    assert allocation['slurm_job_id'] == claim['slurm_job_id']
    physical = allocation['actual_allocated_indices']
    assert len(physical) == 8 and len(set(physical)) == 8 and all(type(i) is int for i in physical)
    assert len(allocation['physical_gpu_inventory']) == 8
    inventory = {int(row[0]): (row[1].strip(), int(row[2])) for row in allocation['physical_gpu_inventory']}
    assert set(inventory) == set(physical) and len({uuid for uuid, _ in inventory.values()}) == 8
    assert all(uuid and total > 0 for uuid, total in inventory.values())
    assert len(allocation['actual_cuda_runtime_uuids']) == 8
    assert set(allocation['actual_cuda_runtime_uuids']) == {uuid for uuid, _ in inventory.values()}
    source_sha = hashlib.sha256(canonical_bytes(run['source_sha256'])).hexdigest()
    claimed = claim['claimed_unix']
    assert _number(claimed) and _number(entered) and run['started_unix'] <= claimed <= entered
    windows = []
    for step in range(1, 6):
        begins, completes = [], []
        for rank in range(8):
            selected = [row for row in scheduler_receipts[str(rank)]['selected_rows']
                        if row['logical_update'] == step]
            begin = [row for row in selected if row['kind'] == 'BEGIN']
            complete = [row for row in selected if row['kind'] == 'COMPLETE']
            assert len(begin) == len(complete) == 1
            start, end = begin[0]['observed_unix'], complete[0]['observed_unix']
            assert _number(start) and _number(end) and claimed <= start <= end <= entered, \
                'Actual first-five scheduler time window differs'
            begins.append(start); completes.append(end)
        windows.append(dict(logical_update=step, begin_unix=min(begins), complete_unix=max(completes)))
    assert rows and rows[-1]['query_started_unix'] >= entered, 'GPU sample is historical at the five gate'
    prior = [index for index, row in enumerate(rows) if _number(row['unix']) and row['unix'] <= claimed]
    assert prior, 'GPU samples do not bracket this claimed training process'
    selected = rows[prior[-1]:]
    producer = selected[0]['sampler_identity']
    peaks = {index: 0 for index in physical}
    previous = None
    gaps, durations = [], []
    for row in selected:
        assert row['schema_version'] == 1 and row['role'] == 'actual_common8_gpu_occupancy'
        assert row['run_id'] == run['run_id'] and row['slurm_job_id'] == claim['slurm_job_id']
        assert row['source_map_sha256'] == source_sha and row['sampler_identity'] == producer
        assert row['sampling_interval_seconds'] == 2
        start, end = row['query_started_unix'], row['unix']
        assert _number(start) and _number(end) and run['started_unix'] <= start <= end
        if previous is not None:
            assert previous <= start, 'GPU samples are duplicated or reordered'
            gaps.append(end - previous)
        previous = end; durations.append(end - start)
        gpus = row['gpus']
        assert len(gpus) == 8 and sorted(gpu['index'] for gpu in gpus) == sorted(physical), \
            'Missing, duplicated or foreign actual sampled device'
        for gpu in gpus:
            index = gpu['index']
            assert all(type(gpu[key]) is int for key in ('index', 'used_mib', 'total_mib', 'util_percent'))
            assert (gpu['uuid'], gpu['total_mib']) == inventory[index], 'Actual sampled device identity differs'
            assert 0 <= gpu['used_mib'] <= gpu['total_mib'] and 0 <= gpu['util_percent'] <= 100
            assert gpu['used_mib'] / gpu['total_mib'] <= .98, 'Measured first-five GPU peak exceeds existing admission limit'
            peaks[index] = max(peaks[index], gpu['used_mib'])
    for window in windows:
        inside = [row for row in selected if window['begin_unix'] <= row['query_started_unix'] and
                  row['unix'] <= window['complete_unix']]
        assert inside, 'No actual GPU sample inside a completed first-five optimizer window'
        window['sample_count_inside_optimizer_window'] = len(inside)
    encoded = b'\n'.join(canonical_bytes(row) for row in selected) + b'\n'
    return dict(run_id=run['run_id'], slurm_job_id=claim['slurm_job_id'], accepted=True,
        first_actual_update=1, last_actual_update=5, sampling_interval_seconds=2,
        training_claim_unix=claimed, actual_optimizer_windows=windows,
        first_sample_unix=selected[0]['unix'], last_sample_unix=selected[-1]['unix'],
        acceptance_entered_unix=entered, sample_count=len(selected), sampler_identity=producer,
        actual_device_inventory={str(i): dict(uuid=inventory[i][0], total_mib=inventory[i][1]) for i in physical},
        sampled_peak_mib={str(i): peaks[i] for i in physical},
        sampled_peak_fraction={str(i): peaks[i] / inventory[i][1] for i in physical},
        existing_sampled_peak_fraction_limit=.98, maximum_sampling_gap_seconds=max(gaps, default=0.),
        maximum_query_duration_seconds=max(durations), selected_samples_sha256=hashlib.sha256(encoded).hexdigest(),
        semantics='Actual sampled occupancy from training claim through this same-process five gate, including '
                  'initialization, generation and optimizer writes; all eight actual devices in every selected sample. '
                  'Observed gaps and query durations are disclosed; instantaneous peaks between samples are unmeasured.')


def verify_training_gpu_samples(base, run, scheduler_receipts):
    entered = time.time()
    base = Path(base).resolve()
    claim = _strict_json(_owned(base / 'logs/training-launch-claim.json').read_bytes())
    allocation_path = _owned(base / 'logs/native-allocation-admission.json')
    allocation_raw = allocation_path.read_bytes()
    allocation = _strict_json(allocation_raw)
    assert claim['slurm_job_id'] == os.environ['SLURM_JOB_ID']
    path, raw, rows, producer = _sample_after_entry(base, run, claim, entered)
    result = _coverage(rows, run, claim, allocation, scheduler_receipts, entered=entered)
    _live_producer(base, claim, producer)
    result.update(receipt_path=str(path), observed_complete_prefix_sha256=hashlib.sha256(raw).hexdigest(),
                  allocation_receipt_sha256=hashlib.sha256(allocation_raw).hexdigest(),
                  semantics=result['semantics'] + ' A newly acquired sample after gate entry and live sampler/trainer births are mandatory.')
    return result
