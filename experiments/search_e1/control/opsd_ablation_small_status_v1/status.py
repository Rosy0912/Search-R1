"""Read-only remote progress; emit small aggregates, never export raw records."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def read(path):
    if not path.exists():
        return None
    raw = path.read_bytes()
    return json.loads(raw), dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def identity(pid):
    try:
        p = Path('/proc') / str(pid)
        tail = (p / 'stat').read_text().rsplit(')', 1)[1].split()
        env = (p / 'environ').read_bytes().split(b'\0')
        run = next((v.split(b'=', 1)[1].decode() for v in env if v.startswith(b'E1_RUN_ID=')), None)
        return dict(pid=pid, uid=p.stat().st_uid, state=tail[0], starttick=int(tail[19]), run_id=run)
    except (OSError, ValueError) as error:
        return dict(pid=pid, status='ABSENT_OR_UNREADABLE', error_type=type(error).__name__)


def snapshot(base):
    value = dict(schema='opsd-ablation-small-status-v1', checked_unix=time.time(),
                 driver=identity(1894801), retriever=identity(2239529), groups={}, evaluation={},
                 raw_records_exported=False, node_writes=False, signals=False)
    driver_root = base / 'eval-study-driver-stats-v2-8b641cb51b4c478684f596f75c603851/runs'
    for name in ('C0', 'C1', 'M', 'R', 'W', 'P', 'J'):
        current = read(base / ('group-' + name) / 'runs/status.json')
        if current is None:
            value['groups'][name] = dict(status='NOT_PRESENT_AT_EXPECTED_PATH')
        else:
            row, ref = current
            value['groups'][name] = dict(reference=ref, **{k:row.get(k) for k in
                ('phase', 'run_id', 'config_sha256', 'outer', 'outer_iterations', 'updates', 'latest_checkpoint')})
        root = base / ('eval-stats-v2-72e1e12-8b641cb51b4c478684f596f75c603851/' + name + '/runs')
        versions = {}
        for version in ('raw10_bf16', 'raw10_fp16'):
            directory = root / version
            if not directory.exists():
                continue
            workers = []
            for rank in range(8):
                part = directory / ('worker-' + str(rank))
                status = read(part / 'status.json')
                records = part / 'results.jsonl'
                rows = 0
                if records.exists():
                    with records.open('rb') as handle:
                        rows = sum(line.endswith(b'\n') for line in handle)
                workers.append(dict(rank=rank, rows=rows, status=status[0].get('status') if status else None,
                    error=status[0].get('error') if status else None))
            terminal = read(directory / 'phase-complete.json')
            versions[version] = dict(rows=sum(w['rows'] for w in workers), workers=workers,
                phase_complete=None if terminal is None else dict(reference=terminal[1],
                    accepted=terminal[0].get('accepted'), actual_wait_codes=[x.get('actual_wait_returncode')
                        for x in terminal[0].get('child_exit_evidence', [])]))
        summary = read(root / 'summary.json')
        complete = read(root / 'controller-complete.json')
        value['evaluation'][name] = dict(versions=versions,
            native_summary=None if summary is None else dict(reference=summary[1], accepted=summary[0].get('accepted'),
                models=summary[0].get('models'), primary_slots=summary[0].get('primary_slots')),
            controller_complete=None if complete is None else dict(reference=complete[1], accepted=complete[0].get('accepted')))
    result = subprocess.run(['nvidia-smi', '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, timeout=15)
    value['gpu_free'] = dict(actual_returncode=result.returncode, rows=result.stdout.strip().splitlines())
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, timeout=15)
    value['gpu_processes'] = dict(actual_returncode=result.returncode, rows=result.stdout.strip().splitlines())
    finished = read(driver_root / 'driver-complete.json')
    value['driver_complete'] = None if finished is None else dict(reference=finished[1], accepted=finished[0].get('accepted'))
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    args = parser.parse_args()
    base = Path(args.base)
    if str(base.resolve()) != '/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008':
        raise ValueError('Explicit current study root required; refuse another resource scope')
    print(json.dumps(snapshot(base), ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
