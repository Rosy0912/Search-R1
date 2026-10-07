"""Independent observer: actual nvidia-smi free memory, preserving V1 evidence.

Run separately; never overwrite memory-continuous.json or stop its observer.
The legacy minimum_reserve_mib field is computed and is not an admission gate.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


QUERY = ['nvidia-smi',
         '--query-gpu=index,uuid,memory.used,memory.total,memory.free,memory.reserved,utilization.gpu',
         '--format=csv,noheader,nounits']
EXCLUDED_PHASES = {
    'WAITING_COMPLETE_OLD_STATE_AND_ARCHIVE100_TRANSFER',
    'STOPPING_ONLY_PRESERVED_OLD_ALIGNED8',
    'WAITING_OLD_OWNED_GPU_RELEASE',
}


def parse_rows(text):
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(',')]
        if len(fields) != 7:
            raise ValueError('Unexpected nvidia-smi column count; actual free/reserved cannot be certified')
        gpu = int(fields[0])
        used, total, free, reserved, utilization = map(int, fields[2:])
        if total <= 0 or any(value < 0 or value > total for value in (used, free, reserved)):
            raise ValueError('Invalid GPU memory measurements')
        rows.append(dict(gpu=gpu, uuid=fields[1], used_mib=used, total_mib=total,
            actual_free_mib=free, reserved_mib=reserved, utilization=utilization,
            reserve_mib=total-used, computed_reserve_mib=total-used,
            computed_minus_actual_free_mib=total-used-free))
    if sorted(row['gpu'] for row in rows) != list(range(8)):
        raise ValueError('Actual observer requires exactly GPUs 0 through 7')
    return sorted(rows, key=lambda row: row['gpu'])


def atomic_json(path, value):
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def historical_v1_snapshot(path):
    """Read a stable V1 snapshot, without rewriting or stopping the V1 writer."""
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, ValueError):
        return None
    return dict(path=str(path), captured_at_unix=time.time(),
        content_sha256=hashlib.sha256(raw).hexdigest(),
        metric='COMPUTED total_mib - peak(memory.used); includes driver-reserved memory; not actual free',
        time=value.get('time'), samples=value.get('samples'), peaks_mib=value.get('peaks_mib'),
        minimum_reserve_mib=value.get('minimum_reserve_mib'), phase=value.get('phase'),
        original_scope=value.get('scope'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path,
        default=Path('/data/shaoshitong/search-e1-opsd-from-archive100-8gpu-20261007'))
    args = parser.parse_args()
    base = args.base
    receipt = base / 'receipts/memory-continuous-v2.json'
    sample_path = base / 'logs/memory-v2-samples.jsonl'
    history_path = base / 'receipts/memory-v1-computed-snapshot-at-v2-start.json'
    if any(path.exists() for path in (receipt, sample_path, history_path)):
        raise RuntimeError('V2 evidence already exists: preserve it before an explicit observer restart')
    observer_started = time.time()
    observer = dict(pid=os.getpid(), started_at_unix=observer_started,
        source_path=str(Path(__file__).resolve()),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    peaks, free_minima, reserved_maxima, uuids = {}, {}, {}, {}
    samples = 0
    interval_start = None
    historical = None
    with sample_path.open('x') as sample_log:
        while True:
            try:
                sample_started = time.time()
                rows = parse_rows(subprocess.check_output(QUERY, text=True))
                sample_ended = time.time()
                phase = json.loads((base / 'receipts/pipeline-stage.json').read_text())
                if phase['phase'] in EXCLUDED_PHASES:
                    peaks, free_minima, reserved_maxima, uuids = {}, {}, {}, {}
                    samples, interval_start = 0, None
                    time.sleep(2)
                    continue
                if historical is None:
                    historical = historical_v1_snapshot(base / 'receipts/memory-continuous.json')
                    if historical is not None:
                        atomic_json(history_path, historical)
                if interval_start is None:
                    interval_start = sample_started
                for row in rows:
                    key = str(row['gpu'])
                    if key in uuids and uuids[key] != row['uuid']:
                        raise RuntimeError('Physical GPU UUID changed during V2 observation')
                    uuids[key] = row['uuid']
                    peaks[key] = max(peaks.get(key, 0), row['used_mib'])
                    free_minima[key] = min(free_minima.get(key, row['actual_free_mib']), row['actual_free_mib'])
                    reserved_maxima[key] = max(reserved_maxima.get(key, 0), row['reserved_mib'])
                samples += 1
                report = dict(schema_version=2, observer=observer,
                    time=sample_ended, samples=samples, phase=phase,
                    sample_start_unix=interval_start, sample_end_unix=sample_ended,
                    updated_at_unix=time.time(), current=rows,
                    actual_free_source='nvidia-smi memory.free; directly measured, no total-used substitution',
                    free_measurement='nvidia-smi memory.free',
                    minimum_actual_free_mib=min(free_minima.values()),
                    actual_free_minima_by_gpu_mib=free_minima,
                    reserved_maxima_by_gpu_mib=reserved_maxima,
                    peaks_mib=peaks,
                    minimum_reserve_mib=min(row['total_mib']-peaks[str(row['gpu'])] for row in rows),
                    computed_fields_note='peaks_mib and minimum_reserve_mib cover this V2 interval; minimum_reserve_mib is legacy COMPUTED total-used, not the admission metric',
                    historical_v1_computed_evidence=historical,
                    scope='HKUST1 allocated eight physical GPUs; actual sampled memory.free includes all processes and driver reservations; V1 files remain separate and untouched')
                sample_log.write(json.dumps(report) + '\n')
                sample_log.flush()
                atomic_json(receipt, report)
                if samples == 1 or samples % 30 == 0:
                    print(json.dumps(dict(samples=samples, phase=phase['phase'],
                        sample_start_unix=interval_start, sample_end_unix=sample_ended,
                        minimum_actual_free_mib=report['minimum_actual_free_mib'],
                        computed_minimum_reserve_mib=report['minimum_reserve_mib'])), flush=True)
            except Exception as error:
                print(json.dumps(dict(time=time.time(), error_type=type(error).__name__,
                    reason=str(error), scope='Invalid sample discarded; no new healthy memory receipt written')), flush=True)
            time.sleep(2)


if __name__ == '__main__':
    main()
