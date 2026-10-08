"""Wrap an own GPU phase; occupancy changes stop only the wrapper, never foreign PIDs.

Caller MUST use set -e/pipefail and scoped marker cleanup on EXIT. Exit 42 means
resource pause; the child remains marked for that cleanup. No process is signalled
here. nvidia-smi pmon unsupported/unknown external occupancy is fail-closed.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def number(value, label):
    try:
        result = float(value)
    except (ValueError, TypeError):
        raise ValueError('Unknown/invalid ' + label + ': ' + repr(value)) from None
    if not math.isfinite(result) or result < 0:
        raise ValueError('Unknown/invalid ' + label)
    return result


def inventory(text, selected):
    rows = {}
    for row in csv.reader(text.splitlines(), skipinitialspace=True):
        if not row:
            continue
        fields = [v.strip() for v in row]
        index, uuid, total = fields[:3]
        rows[index] = {'uuid': uuid, 'total_mib': number(total, 'GPU memory total')}
        if len(fields) == 4:
            rows[index]['whole_gpu_sm_percent'] = number(fields[3], 'whole GPU SM')
    allocated = {}
    for token in selected:
        matches = [(i, r) for i, r in rows.items() if token in (i, r['uuid'])]
        if len(matches) != 1:
            raise ValueError('Allocated GPU cannot be resolved: ' + token)
        index, row = matches[0]
        if row['total_mib'] <= 0 or index in allocated:
            raise ValueError('Invalid/duplicate GPU selection')
        allocated[index] = row
    if len(allocated) != 8 or len({r['uuid'] for r in allocated.values()}) != 8:
        raise ValueError('Watcher requires eight distinct allocated GPUs')
    return allocated


def parse_pmon(text, allocated):
    # Default pmon -s u prints: gpu pid type sm mem enc dec [optional columns] command.
    # Parse header by name so added driver columns do not change SM interpretation.
    fields = None
    seen = set()
    result = {}
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == '#':
            if 'gpu' in parts and 'pid' in parts and 'sm' in parts:
                fields = parts[1:]
            continue
        if fields is None:
            raise ValueError('pmon has no recognized field header')
        if len(parts) < len(fields):
            raise ValueError('Truncated pmon row')
        gpu = parts[fields.index('gpu')]
        if gpu not in allocated:
            continue
        seen.add(gpu)
        pid = parts[fields.index('pid')]
        if pid == '-':
            continue  # Explicit no-process row, not an unknown utilization for a PID.
        pid = int(pid)
        if pid <= 0 or (gpu, pid) in result:
            raise ValueError('Invalid/duplicate pmon PID')
        raw_sm = parts[fields.index('sm')]
        result[(gpu, pid)] = raw_sm
    if seen != set(allocated):
        raise ValueError('pmon omitted allocated GPU(s)')
    return result


def parse_memory(text, allocated):
    uuid_to_index = {r['uuid']: i for i, r in allocated.items()}
    result = {}
    for row in csv.reader(text.splitlines(), skipinitialspace=True):
        if not row:
            continue
        uuid, pid, memory = [v.strip() for v in row]
        if uuid not in uuid_to_index:
            continue
        key = (uuid_to_index[uuid], int(pid))
        if key[1] <= 0 or key in result:
            raise ValueError('Invalid/duplicate compute PID')
        result[key] = memory
    return result


def own_pid(pid, run, proc_root=Path('/proc')):
    # Command substrings, same uid, and an unrelated Slurm allocation are NOT ownership.
    # Permission/read errors yield UNKNOWN, never fabricate own/external metrics.
    try:
        env = (proc_root / str(pid) / 'environ').read_bytes().split(b'\0')
    except OSError:
        return None
    return ('BAPO_RUN_ID=' + run).encode() in env


def assess(allocated, sm, memory, ownership):
    totals = {i: {'external_sm_percent': 0.0, 'external_memory_percent': 0.0} for i in allocated}
    external = []
    for gpu, pid in sorted(set(sm) | set(memory)):
        own = ownership(pid)
        if own is None:
            raise ValueError('GPU PID ownership unreadable: ' + str(pid))
        if own:
            continue
        # Missing SM (e.g. pmon does not support this device/process) or unavailable
        # graphics-process memory must stop our phase, not appear as zero.
        if (gpu, pid) not in sm or (gpu, pid) not in memory:
            raise ValueError('External GPU PID has incomplete pmon/compute metrics: ' + str(pid))
        used = number(memory[(gpu, pid)], 'external memory GPU=' + gpu + ' PID=' + str(pid))
        util = number(sm[(gpu, pid)], 'external SM GPU=' + gpu + ' PID=' + str(pid))
        if util > 100 or used > allocated[gpu]['total_mib']:
            raise ValueError('Out-of-range external GPU occupancy')
        percent = 100.0 * used / allocated[gpu]['total_mib']
        totals[gpu]['external_sm_percent'] += util
        totals[gpu]['external_memory_percent'] += percent
        external.append({'gpu_index': gpu, 'gpu_uuid': allocated[gpu]['uuid'], 'pid': pid,
                         'sm_percent': util, 'memory_mib': used, 'memory_percent': percent})
    exceeded = {i: t for i, t in totals.items() if any(v > 15.0 for v in t.values())}
    return {'external_processes': external, 'external_totals': totals,
            'pause': bool(exceeded), 'exceeded': exceeded}


NVML_PROBE = r"""
import json,sys,time
try:
 import pynvml as n
 n.nvmlInit()
 records={}
 for uuid in json.loads(sys.argv[1]):
  row={'query_start_us':time.time_ns()//1000}
  try:
   h=n.nvmlDeviceGetHandleByUUID(uuid)
   row['whole_gpu_sm_percent']=int(n.nvmlDeviceGetUtilizationRates(h).gpu)
   row['whole_gpu_sm_query_us']=time.time_ns()//1000
   samples=n.nvmlDeviceGetProcessUtilization(h,0)
   row.update(status='ok',samples=[{'pid':int(s.pid),'timestamp_us':int(s.timeStamp),
       'sm_percent':int(s.smUtil)} for s in samples])
  except Exception as e:
   row.update(status='error',error={'code':getattr(e,'value',None),'text':str(e)})
  row['query_end_us']=time.time_ns()//1000
  records[uuid]=row
 print(json.dumps(records,allow_nan=False))
 n.nvmlShutdown()
except Exception as e:
 print(json.dumps({'probe_error':type(e).__name__+': '+str(e)}))
"""


def collect_nvml(allocated):
    try:
        proc = subprocess.run([sys.executable, '-c', NVML_PROBE,
            json.dumps([r['uuid'] for r in allocated.values()])], check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=8)
        return json.loads(proc.stdout)
    except Exception as exc:
        return {'probe_error': type(exc).__name__ + ': ' + str(exc)}


def fresh_nvml(allocated, nvml, window_start_us):
    fresh = {}
    covered = set()
    for gpu, record in allocated.items():
        row = nvml.get(record['uuid'], {})
        if row.get('status') != 'ok':
            continue  # Error/NOT_FOUND/NOT_SUPPORTED is NOT an implicit zero.
        end = row.get('query_end_us')
        if not isinstance(end, int) or end < window_start_us:
            continue
        for item in row.get('samples', []):
            timestamp = item.get('timestamp_us')
            if not isinstance(timestamp, int) or timestamp > end:
                raise ValueError('Invalid/future NVML timestamp')
            if timestamp < max(window_start_us, end - 1_500_000):
                continue
            pid = int(item['pid'])
            util = number(item['sm_percent'], 'fresh NVML SM PID=' + str(pid))
            if pid <= 0 or util > 100:
                raise ValueError('Invalid NVML process sample')
            # Require actual nonzero samples to prove this current window is
            # represented by the supported API; successful empty output is not proof.
            if util > 0:
                covered.add(gpu)
            fresh[(gpu, pid)] = max(fresh.get((gpu, pid), 0.0), util)
    return fresh, covered


def assess_with_bounds(allocated, sm, memory, ownership, nvml, window_start_us):
    fresh, covered = fresh_nvml(allocated, nvml, window_start_us)
    external = []
    totals = {}
    for gpu, record in allocated.items():
        whole = number(record.get('whole_gpu_sm_percent'), 'whole GPU SM ' + gpu)
        if whole > 100:
            raise ValueError('Invalid whole GPU SM ' + gpu)
        rows = []
        for g, pid in sorted(set(sm) | set(memory) | set(fresh)):
            if g != gpu:
                continue
            own = ownership(pid)
            if own is None:
                raise ValueError('GPU PID ownership unreadable: ' + str(pid))
            if own:
                continue
            if (gpu, pid) not in memory:
                raise ValueError('External memory missing GPU=' + gpu + ' PID=' + str(pid))
            used = number(memory[(gpu, pid)], 'external memory GPU=' + gpu + ' PID=' + str(pid))
            if used > record['total_mib']:
                raise ValueError('Out-of-range external memory')
            raw_sm = sm.get((gpu, pid))
            try:
                known = number(raw_sm, 'pmon SM')
            except (ValueError, TypeError):
                known = None
            if known is not None and known > 100:
                raise ValueError('Invalid pmon SM')
            if known is None:
                if whole <= 15:
                    util, basis = None, 'whole_gpu_upper_bound'
                elif gpu in covered:
                    util = fresh.get((gpu, pid), 0.0)
                    basis = 'fresh_nvml_nonzero_sample' if (gpu, pid) in fresh else 'fresh_nvml_nonzero_samples_absence'
                else:
                    raise ValueError('External SM unknown without current NVML coverage GPU=' + gpu + ' PID=' + str(pid))
            else:
                util = max(known, fresh.get((gpu, pid), 0.0))
                basis = 'pmon_and_fresh_nvml_max' if (gpu, pid) in fresh else 'pmon'
            rows.append({'gpu_index': gpu, 'gpu_uuid': record['uuid'], 'pid': pid,
                'sm_percent': util, 'sm_evidence': basis, 'memory_mib': used,
                'memory_percent': 100.0 * used / record['total_mib']})
        known_sum = sum(r['sm_percent'] for r in rows if r['sm_percent'] is not None)
        bound_used = any(r['sm_percent'] is None for r in rows)
        bound = max(whole, known_sum) if bound_used else known_sum
        totals[gpu] = {'external_sm_percent': None if bound_used else known_sum,
            'external_sm_upper_bound_percent': bound, 'whole_gpu_sm_percent': whole,
            'external_memory_percent': sum(r['memory_percent'] for r in rows),
            'sm_evidence': 'whole_gpu_upper_bound' if bound_used else 'process_samples'}
        external.extend(rows)
    exceeded = {i: t for i, t in totals.items() if
        t['external_sm_upper_bound_percent'] > 15 or t['external_memory_percent'] > 15}
    return {'external_processes': external, 'external_totals': totals,
        'pause': bool(exceeded), 'exceeded': exceeded}


def query(args):
    return subprocess.run(['nvidia-smi'] + args, check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=8).stdout


class LifecycleChanged(ValueError):
    """A sampled PID exited, appeared, or was reused; discard the entire sample."""


class SamplingError(ValueError):
    def __init__(self, error, diagnostic):
        super().__init__(str(error))
        self.diagnostic = diagnostic


def proc_start_ticks(pid, proc_root=Path('/proc')):
    try:
        fields = (proc_root / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError:
        return None
    # Zombies have exited and cannot validate a live GPU ownership observation.
    return None if fields[0] == 'Z' else int(fields[19])


def process_starts(proc_root=Path('/proc')):
    starts = {}
    for path in proc_root.iterdir():
        if not path.name.isdigit():
            continue
        try:
            tick = proc_start_ticks(int(path.name), proc_root)
            if tick is not None:
                starts[int(path.name)] = tick
        except (OSError, ValueError, IndexError):
            # A relevant inaccessible PID will still fail the exact live check;
            # inaccessible unrelated /proc entries must not block the entire node.
            continue
    return starts


def sample(run, selected):
    attempts = []
    # Only proven PID lifecycle changes permit one complete replacement sample.
    # Stable live external unknown metrics/identity fail immediately, never pass
    # based on retries, and never become zero. No phase is launched during this.
    for attempt in range(2):
        diagnostic = {'attempt': attempt + 1}
        try:
            starts_before = process_starts()
            window_start_us = time.time_ns() // 1000
            diagnostic['window_start_us'] = window_start_us
            inventory_raw = query(['--query-gpu=index,uuid,memory.total,utilization.gpu', '--format=csv,noheader,nounits'])
            diagnostic['raw_inventory'] = inventory_raw
            allocated = inventory(inventory_raw, selected)
            diagnostic['allocated_gpus'] = allocated
            memory_raw = query(['--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader,nounits'])
            diagnostic['raw_compute'] = memory_raw
            sm_raw = query(['pmon', '-s', 'u', '-c', '1'])
            diagnostic['raw_pmon'] = sm_raw
            sm, memory = parse_pmon(sm_raw, allocated), parse_memory(memory_raw, allocated)
            nvml = collect_nvml(allocated)
            diagnostic['raw_nvml'] = nvml
            for record in allocated.values():
                row = nvml.get(record['uuid'], {})
                if 'whole_gpu_sm_percent' in row:
                    record['whole_gpu_sm_percent'] = number(row['whole_gpu_sm_percent'], 'NVML whole GPU SM')
                    record['whole_gpu_sm_source'] = 'nvml_query'
                    record['whole_gpu_sm_query_us'] = row.get('whole_gpu_sm_query_us')
                else:
                    record['whole_gpu_sm_source'] = 'nvidia_smi_inventory_query'
            diagnostic['window_end_us'] = time.time_ns() // 1000
            fresh, _ = fresh_nvml(allocated, nvml, window_start_us)
            starts_after = process_starts()
            keys = set(sm) | set(memory) | set(fresh)
            diagnostic['pid_start_ticks'] = {str(pid): {'before': starts_before.get(pid),
                'after': starts_after.get(pid)} for _, pid in keys}
            def checked_owner(pid):
                before, after = starts_before.get(pid), starts_after.get(pid)
                if before is None or after is None or before != after:
                    raise LifecycleChanged('GPU PID lifecycle changed: ' + str(pid))
                first = proc_start_ticks(pid)
                own = own_pid(pid, run)
                last = proc_start_ticks(pid)
                if first != after or last != after:
                    raise LifecycleChanged('GPU PID changed during ownership read: ' + str(pid))
                return own
            # Validate every referenced PID before using ANY SM/memory observation.
            # A reused PID cannot inherit an exited process's own/external status.
            owners = {pid: checked_owner(pid) for pid in {p for _, p in keys}}
            report = assess_with_bounds(allocated, sm, memory, lambda pid: owners[pid], nvml, window_start_us)
            report.update(diagnostic)
            if attempts:
                report['discarded_lifecycle_samples'] = attempts
            return report
        except LifecycleChanged as exc:
            diagnostic['error'] = str(exc)
            attempts.append(diagnostic)
            if attempt == 0:
                continue
            raise SamplingError(exc, {'samples': attempts}) from exc
        except Exception as exc:
            diagnostic['error'] = type(exc).__name__ + ': ' + str(exc)
            raise SamplingError(exc, {'samples': attempts + [diagnostic]}) from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--interval', type=float, default=10)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or not math.isfinite(args.interval) or args.interval < 1:
        parser.error('command and finite interval >= 1 required')
    if os.environ.get('BAPO_RUN_ID') != args.run or not os.environ.get('SLURM_JOB_ID'):
        parser.error('exact inherited BAPO_RUN_ID and own Slurm allocation required')
    run = Path(args.run)
    if not run.is_absolute() or not run.is_dir():
        parser.error('existing absolute own run directory required')
    selected = [v.strip() for v in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if v.strip()]
    child = None
    with (run / 'resource-watch.jsonl').open('a', buffering=1) as log:
        while True:
            try:
                event = sample(args.run, selected)
                event['reason'] = 'external_occupancy_exceeds_15_percent' if event['pause'] else 'within_threshold'
            except Exception as exc:
                event = {'pause': True, 'reason': 'monitor_unknown_fail_closed',
                         'error': type(exc).__name__ + ': ' + str(exc)}
                if isinstance(exc, SamplingError):
                    event['diagnostic'] = exc.diagnostic
            event.update({'time': time.time(), 'run': args.run, 'slurm_job_id': os.environ['SLURM_JOB_ID'],
                          'phase_pid': child.pid if child is not None else None})
            log.write(json.dumps(event, allow_nan=False) + '\n')
            if event['pause']:
                print('RESOURCE_PAUSE: ' + event['reason'] + '; own stage must exit for scoped cleanup', file=sys.stderr)
                return 42
            if child is None:
                # Preserve lease FDs held by an outer common preflight exec launcher.
                os.environ['SEARCH_R1_GPU_RESOURCE_GUARD'] = json.dumps(dict(
                    pid=os.getpid(), start_ticks=proc_start_ticks(os.getpid()), run=args.run))
                child = subprocess.Popen(command, close_fds=False)
            deadline = time.monotonic() + args.interval
            while time.monotonic() < deadline:
                code = child.poll()
                if code is not None:
                    return code if code >= 0 else 128 - code
                time.sleep(min(0.25, max(0, deadline - time.monotonic())))


if __name__ == '__main__':
    sys.exit(main())
