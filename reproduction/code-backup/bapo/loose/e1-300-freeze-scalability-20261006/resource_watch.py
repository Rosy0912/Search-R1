"""Wrap an own GPU phase; finite recovery freezes only certified own process epochs.

Caller MUST use set -e/pipefail and scoped marker cleanup on EXIT. Exit 42 means
resource pause; the child remains marked for that cleanup. Only certified own process epochs are STOP/CONT signalled during bounded recovery. nvidia-smi pmon unsupported/unknown external occupancy is fail-closed.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import signal
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
   mode=n.nvmlDeviceGetMigMode(h)[0]
   row['mig_mode']=int(mode)
   samples=n.nvmlDeviceGetProcessUtilization(h,0)
   row.update(status='ok',api='nvmlDeviceGetProcessUtilization',last_seen_timestamp_us=0,
       complete_query=True,returned_sample_count=len(samples),
       samples=[{'pid':int(s.pid),'timestamp_us':int(s.timeStamp),
       'sm_percent':int(s.smUtil),'mem_percent':int(s.memUtil),
       'enc_percent':int(s.encUtil),'dec_percent':int(s.decUtil)} for s in samples])
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


def nvml_absence_period(record, row, pid, window_start_us):
    """A conservative observed-period inference, never an instantaneous zero.

    NVIDIA documents complete returned per-process nonzero-period entries. No
    global period ID is promised: require one shared fresh nonzero epoch, complete
    successful buffer, non-MIG, and PID absence from the ENTIRE raw buffer.
    """
    samples=row.get('samples')
    if (row.get('status')!='ok' or row.get('api')!='nvmlDeviceGetProcessUtilization'
            or row.get('complete_query')is not True or type(row.get('last_seen_timestamp_us'))is not int or row['last_seen_timestamp_us']!=0
            or type(row.get('mig_mode'))is not int or row['mig_mode']!=0
            or not isinstance(samples,list) or type(row.get('returned_sample_count'))is not int
            or row['returned_sample_count']!=len(samples) or not samples):return None
    end=row.get('query_end_us');start=row.get('query_start_us')
    if type(end)is not int or type(start)is not int or not 0<window_start_us<=start<=end:return None
    epochs=set();newest=None;seen_pids=set()
    for item in samples:
        if type(item.get('pid'))is not int or item['pid']<=0 or item['pid'] in seen_pids or type(item.get('timestamp_us'))is not int or item['timestamp_us']<=0:return None
        seen_pids.add(item['pid'])
        if item['pid']==pid:return None  # An old foreign sample cannot be filtered into absence.
        stamp=item['timestamp_us']
        if stamp>end:return None
        values=[]
        for key in ('sm_percent','mem_percent','enc_percent','dec_percent'):
            try:value=number(item.get(key),'NVML complete-period '+key)
            except ValueError:return None
            if value>100:return None
            values.append(value)
        if stamp>=max(window_start_us,end-1_500_000):
            newest=stamp if newest is None else max(newest,stamp)
            if values[0]>0:epochs.add(stamp)  # SM0-only does not cover a whole-SM98 anomaly.
    if len(epochs)!=1 or newest not in epochs:return None
    return newest


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
                elif (gpu, pid) in fresh:
                    util = fresh[(gpu, pid)]
                    basis = 'fresh_nvml_exact_pid_sample'
                elif nvml_absence_period(record, nvml.get(record['uuid'],{}), pid, window_start_us) is not None:
                    # Official API complete list omits processes with no nonzero
                    # utilization in its observed sample period, not instant SM.
                    util = 0.0
                    basis = 'nvml_complete_observed_period_absence'

                else:
                    raise ValueError('External SM unknown without current NVML coverage GPU=' + gpu + ' PID=' + str(pid))
            else:
                util = max(known, fresh.get((gpu, pid), 0.0))
                basis = 'pmon_and_fresh_nvml_max' if (gpu, pid) in fresh else 'pmon'
            rows.append({'gpu_index': gpu, 'gpu_uuid': record['uuid'], 'pid': pid,
                'sm_percent': util, 'sm_evidence': basis,
                'nvml_observed_period_us': nvml_absence_period(record,nvml.get(record['uuid'],{}),pid,window_start_us) if basis=='nvml_complete_observed_period_absence' else None, 'memory_mib': used,
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


def lease_uuids():
    records = json.loads(os.environ.get('SEARCH_R1_GPU_LEASES', '[]'))
    if not isinstance(records, list) or len(records) != 8:
        raise ValueError('Missing eight inherited UUID leases')
    values = {r['uuid'] for r in records}
    if len(values) != 8 or any(not isinstance(v, str) or not v.startswith('GPU-') for v in values):
        raise ValueError('Invalid inherited UUID leases')
    return values


def snapshot_owned(starts, run, expected_uuids, proc_root=Path('/proc')):
    """Capture live own epochs BEFORE GPU queries; never reconstruct a dead owner.

    No environment contents are logged. Exact run, Slurm, allocated UUID set,
    caller UID and stable stat epoch are the only recorded ownership witnesses.
    """
    result = {}
    marker = ('BAPO_RUN_ID=' + run).encode()
    job = os.environ.get('SLURM_JOB_ID')
    if not job:
        raise ValueError('Missing own Slurm allocation')
    for pid, tick in starts.items():
        path = proc_root / str(pid)
        try:
            first = proc_start_ticks(pid, proc_root)
            env = (path / 'environ').read_bytes().split(b'\0')
            if marker not in env:
                continue
            uid = path.stat().st_uid
            fields = dict(item.split(b'=', 1) for item in env if b'=' in item)
            critical = (b'BAPO_RUN_ID', b'SLURM_JOB_ID', b'SEARCH_R1_GPU_LEASES')
            if any(sum(item.startswith(k + b'=') for item in env) != 1 for k in critical):
                continue  # Same-run CPU siblings may not inherit GPU leases.
            records = json.loads(fields.get(b'SEARCH_R1_GPU_LEASES', b'[]'))
            uuids = {r['uuid'] for r in records}
            last = proc_start_ticks(pid, proc_root)
        except FileNotFoundError:
            continue  # Not certified; a later referenced dead PID will fail closed.
        except (PermissionError, ValueError, TypeError, KeyError):
            continue
        if first != tick or last != tick:
            continue
        if (uid != os.getuid() or fields.get(b'BAPO_RUN_ID') != run.encode() or
                fields.get(b'SLURM_JOB_ID') != job.encode() or
                not isinstance(records, list) or len(records) != 8 or uuids != expected_uuids):
            continue  # Never certified; if GPU-referenced, checked_owner rejects.
        result[pid] = {'start_ticks': tick, 'run': run, 'slurm_job_id': job,
                       'uid': uid, 'lease_uuids': sorted(uuids)}
    return result


def sample(run, selected):
    attempts = []
    expected_uuids = lease_uuids()
    exited = {}  # Only this bounded call; no successful sample or metrics reused.
    # Four whole samples, with bounded backoff. Only a certified OWN exit allows
    # replacement; unknown/external exits and PID reuse remain fail-closed.
    for attempt in range(4):
        diagnostic = {'attempt': attempt + 1}
        try:
            starts_before = process_starts()
            owned_before = snapshot_owned(starts_before, run, expected_uuids)
            diagnostic['owned_before_query'] = {str(p): r for p, r in owned_before.items()}
            window_start_us = time.time_ns() // 1000
            diagnostic['window_start_us'] = window_start_us
            inventory_raw = query(['--query-gpu=index,uuid,memory.total,utilization.gpu', '--format=csv,noheader,nounits'])
            diagnostic['raw_inventory'] = inventory_raw
            allocated = inventory(inventory_raw, selected)
            if {r['uuid'] for r in allocated.values()} != expected_uuids:
                raise ValueError('Allocated UUIDs changed from inherited leases')
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
            pending_exits = []
            def own_exit(pid, before, after):
                gpu_uuids = {allocated[g]['uuid'] for g, p in keys if p == pid}
                if before is not None and after is not None:
                    raise ValueError('GPU PID reused or appeared: ' + str(pid))
                if before is not None and after is None and pid in owned_before:
                    prior = exited.get(pid)
                    if prior is not None and prior['proof']['start_ticks'] != before:
                        raise ValueError('Certified exited PID was reused: ' + str(pid))
                    exited[pid] = {'proof': owned_before[pid], 'gpu_uuids': sorted(gpu_uuids)}
                elif before is None and after is None and pid in exited:
                    if set(exited[pid]['gpu_uuids']) != gpu_uuids:
                        raise ValueError('Exited GPU PID UUID association changed: ' + str(pid))
                else:
                    raise ValueError('Uncertified GPU PID lifecycle change: ' + str(pid))
                # Every row remains in the discarded sample. No occupancy zeroing.
                diagnostic['certified_exited_pids'] = {str(p): r for p, r in exited.items()}
                pending_exits.append(pid)
                return True
            def checked_owner(pid):
                before, after = starts_before.get(pid), starts_after.get(pid)
                if pid in exited and (before is not None or after is not None):
                    raise ValueError('Certified exited GPU PID reappeared: ' + str(pid))
                if before is None or after is None or before != after:
                    return own_exit(pid, before, after)
                first = proc_start_ticks(pid)
                own = own_pid(pid, run)
                last = proc_start_ticks(pid)
                if ((first is not None and first != after) or
                        (last is not None and last != after)):
                    raise ValueError('GPU PID reused during ownership read: ' + str(pid))
                if first != after or last != after:
                    return own_exit(pid, before, last)
                if own and pid not in owned_before:
                    raise ValueError('Own GPU PID lacks prequery allocation proof: ' + str(pid))
                if own and snapshot_owned({pid: after}, run, expected_uuids).get(pid) != owned_before[pid]:
                    raise ValueError('Own GPU PID allocation proof changed: ' + str(pid))
                return own
            # Validate every referenced PID before using ANY SM/memory observation.
            # A reused PID cannot inherit an exited process's own/external status.
            owners = {pid: checked_owner(pid) for pid in {p for _, p in keys}}
            report = assess_with_bounds(allocated, sm, memory, lambda pid: owners[pid], nvml, window_start_us)
            if pending_exits:
                diagnostic['external_assessment_before_discard'] = report
                if report['pause']:
                    raise ValueError('External occupancy exceeds 15 percent during own exit')
                raise LifecycleChanged('Certified own GPU PID exited: ' + str(sorted(pending_exits)))
            report.update(diagnostic)
            if attempts:
                report['discarded_lifecycle_samples'] = attempts
            return report
        except LifecycleChanged as exc:
            diagnostic['error'] = str(exc)
            attempts.append(diagnostic)
            if attempt < 3:
                time.sleep((0.25, 0.5, 1.0)[attempt])
                continue
            raise SamplingError(exc, {'samples': attempts}) from exc
        except Exception as exc:
            diagnostic['error'] = type(exc).__name__ + ': ' + str(exc)
            raise SamplingError(exc, {'samples': attempts + [diagnostic]}) from exc



class RecoveryInterrupted(BaseException):
    def __init__(self, signum):
        self.signum = signum


def freeze_inventory(run, job, uuids, phase_pid, proc_root=Path('/proc')):
    """Certify epochs before signalling, including CPU workers without GPU leases.

    The watcher and its ancestors are excluded; same-run unrelated allocations,
    foreign descendants and unreadable descendant identities reject the freeze.
    """
    rows = {}
    for path in proc_root.iterdir():
        if not path.name.isdigit():
            continue
        pid = int(path.name)
        try:
            stat = (path / 'stat').read_text().rsplit(')', 1)[1].split()
            rows[pid] = dict(pid=pid, ppid=int(stat[1]), state=stat[0],
                start_ticks=int(stat[19]), uid=path.stat().st_uid)
        except FileNotFoundError:
            continue
        except (OSError, ValueError, IndexError):
            # Relevant subtree unreadability is checked below via ancestry or
            # exact environment markers; a totally inaccessible stat is unsafe.
            raise ValueError('Unreadable process stat during freeze: '+str(pid))
    excluded = {os.getpid()}
    cursor = os.getpid()
    while cursor in rows and rows[cursor]['ppid'] not in excluded:
        cursor = rows[cursor]['ppid']; excluded.add(cursor)
    children={}
    for pid,row in rows.items():children.setdefault(row['ppid'],[]).append(pid)
    def subtree(roots):
        found=set(roots);pending=list(roots)
        while pending:
            for pid in children.get(pending.pop(),[]):
                if pid not in found:found.add(pid);pending.append(pid)
        return found
    descendants=subtree({phase_pid})
    # Exact installed passive monitor command + exact run argument, never a
    # command substring. Exclude its entire sleep/nvidia-smi descendant subtree.
    tools=Path(run).parent.parent/'tools'
    passive_roots=set()
    commands={}
    for pid in rows:
        try:commands[pid]=(proc_root/str(pid)/'cmdline').read_bytes().decode(errors='replace').split('\0')
        except FileNotFoundError:continue
        except OSError:
            if pid in descendants:raise ValueError('Unreadable descendant command: '+str(pid))
            continue
        command=commands[pid]
        if len(command)>=3 and Path(command[0]).name in ('bash','sh')and command[1]==str(tools/'monitor.sh')and command[2]==run:
            passive_roots.add(pid)
    passive_descendants=subtree(passive_roots)
    excluded.update(passive_descendants)
    result = {}
    marker = ('BAPO_RUN_ID='+run).encode()
    for pid, row in rows.items():
        if pid in excluded or row['state'] in ('Z','X'):
            continue
        try:
            raw = (proc_root / str(pid) / 'environ').read_bytes().split(b'\0')
        except FileNotFoundError:
            continue
        except OSError:
            if pid in descendants:
                raise ValueError('Unreadable phase descendant identity: '+str(pid))
            continue
        if marker not in raw:
            if pid in descendants:
                raise ValueError('Foreign/unmarked phase descendant: '+str(pid))
            continue
        values = dict(x.split(b'=',1) for x in raw if b'=' in x)
        if any(sum(x.startswith(k+b'=') for x in raw)!=1 for k in (b'BAPO_RUN_ID',b'SLURM_JOB_ID')):
            raise ValueError('Ambiguous own process identity: '+str(pid))
        if values[b'SLURM_JOB_ID']!=job.encode() or row['uid']!=os.getuid():
            raise ValueError('Own marker has foreign UID/allocation: '+str(pid))
        inherited=[]
        if b'SEARCH_R1_GPU_LEASES' in values:
            if sum(x.startswith(b'SEARCH_R1_GPU_LEASES=') for x in raw)!=1:
                raise ValueError('Ambiguous own UUID leases: '+str(pid))
            leases=json.loads(values[b'SEARCH_R1_GPU_LEASES'])
            if not isinstance(leases,list) or len(leases)!=8 or {x['uuid']for x in leases}!=uuids:
                raise ValueError('Own process UUID allocation mismatch: '+str(pid))
            inherited=sorted(x['uuid'] for x in leases)
        if proc_start_ticks(pid,proc_root)!=row['start_ticks']:
            raise ValueError('Own process epoch changed during freeze: '+str(pid))
        command=commands.get(pid)
        if not command or not command[0]:raise ValueError('Unreadable/empty own command: '+str(pid))
        basenames={Path(token).name for token in command if token}
        title=command[0]
        if pid==phase_pid:role='phase'
        elif command[0].startswith('ray::') or Path(command[0]).name=='raylet':role='ray_worker'
        elif 'retriever_local.py' in basenames:role='retriever'
        elif any(token=='verl.trainer.main_ppo' for token in command):role='trainer'
        elif pid in descendants:role='phase_descendant'
        else:raise ValueError('Same-run process lacks approved compute role: '+str(pid))
        row.update(parent_start_ticks=rows.get(row['ppid'],{}).get('start_ticks'),
            run=run,slurm_job_id=job,role=role,command=command,
            allocated_uuid_set=sorted(uuids),gpu_leases_inherited=b'SEARCH_R1_GPU_LEASES' in values,
            inherited_gpu_lease_uuids=inherited)
        result[pid]=row
    if phase_pid not in result:
        raise ValueError('Phase child not live/certified for freeze')
    return result


FREEZE_IDENTITY_KEYS=('pid','ppid','start_ticks','parent_start_ticks','run','slurm_job_id','uid','role','command','allocated_uuid_set','gpu_leases_inherited','inherited_gpu_lease_uuids')

def local_freeze_recheck(proof, held, proc_root=Path('/proc'), *, deadline=None, clock=time.monotonic, sleep=time.sleep):
    """Constant-size /proc recheck after pidfd open, no whole-system rescan.

    A descendant is signalled only after its certified parent is held stopped.
    This permits induction from the phase root without rewalking every ancestry
    chain for each child. Parent epoch checks also cover approved reparented Ray
    roles and the live watcher parent of the phase root.
    """
    pid=proof['pid'];path=proc_root/str(pid)
    def stat(p):
        directory=proc_root/str(p);fields=(directory/'stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=p,ppid=int(fields[1]),state=fields[0],start_ticks=int(fields[19]),uid=directory.stat().st_uid)
    first=stat(pid)
    if first['state']in ('Z','X'):raise ValueError('Own process exited before STOP: '+str(pid))
    raw=(path/'environ').read_bytes().split(b'\0');values=dict(x.split(b'=',1)for x in raw if b'='in x)
    for k in (b'BAPO_RUN_ID',b'SLURM_JOB_ID'):
        if sum(x.startswith(k+b'=')for x in raw)!=1:raise ValueError('Ambiguous own process identity: '+str(pid))
    if values[b'BAPO_RUN_ID']!=proof['run'].encode()or values[b'SLURM_JOB_ID']!=proof['slurm_job_id'].encode()or first['uid']!=os.getuid():raise ValueError('PID epoch/identity changed before STOP: '+str(pid))
    inherited=[];leased=b'SEARCH_R1_GPU_LEASES'in values
    if leased:
        if sum(x.startswith(b'SEARCH_R1_GPU_LEASES=')for x in raw)!=1:raise ValueError('Ambiguous own UUID leases: '+str(pid))
        leases=json.loads(values[b'SEARCH_R1_GPU_LEASES'])
        if not isinstance(leases,list)or len(leases)!=8 or {x['uuid']for x in leases}!=set(proof['allocated_uuid_set']):raise ValueError('Own process UUID allocation mismatch: '+str(pid))
        inherited=sorted(x['uuid']for x in leases)
    command=(path/'cmdline').read_bytes().decode(errors='replace').split('\0')
    if not command or not command[0]:raise ValueError('Unreadable/empty own command: '+str(pid))
    parent=first['ppid'];parent_epoch=None
    if proof.get('parent_start_ticks')is not None:
        parent_stat=stat(parent);parent_epoch=parent_stat['start_ticks']
        if parent_epoch!=proof['parent_start_ticks']:raise ValueError('Certified parent epoch changed before STOP: '+str(pid))
    if proof['role']=='phase_descendant':
        if parent not in held:raise ValueError('CPU descendant parent not held before STOP: '+str(pid))
        original=held[parent]['proof']
        while True:
            parent_stat=stat(parent)
            if parent_stat['start_ticks']!=original['start_ticks']or parent_stat['uid']!=original['uid']or parent_stat['state']in ('Z','X'):raise ValueError('CPU descendant parent not original stopped epoch: '+str(pid))
            if parent_stat['state']in ('T','t'):break
            if deadline is None or clock()>=deadline:raise ValueError('CPU descendant parent not stopped before deadline: '+str(pid))
            sleep(min(.005,max(0,deadline-clock())))
    last=stat(pid)
    if any(first[k]!=last[k]for k in ('pid','ppid','start_ticks','uid'))or last['state']in ('Z','X'):raise ValueError('PID epoch/identity changed during local read: '+str(pid))
    now=dict(last,parent_start_ticks=parent_epoch,run=proof['run'],slurm_job_id=proof['slurm_job_id'],role=proof['role'],command=command,allocated_uuid_set=proof['allocated_uuid_set'],gpu_leases_inherited=leased,inherited_gpu_lease_uuids=inherited)
    if any(now[k]!=proof[k]for k in FREEZE_IDENTITY_KEYS):raise ValueError('PID epoch/identity changed before STOP: '+str(pid))
    return now

def freeze_parent_order(rows):
    """Linear parent-before-child topological order; a cycle rejects the batch."""
    children={};roots=[]
    for pid,row in rows.items():
        if row['ppid']in rows:children.setdefault(row['ppid'],[]).append(pid)
        else:roots.append(pid)
    pending=list(sorted(roots));ordered=[]
    for pid in pending:
        ordered.append(pid);pending.extend(sorted(children.get(pid,[])))
    if len(ordered)!=len(rows):raise ValueError('Owned freeze ancestry cycle')
    return ordered


class OwnedFreezer:
    """Signals only certified epochs through pidfd; refuses unsafe fallback kill."""
    def __init__(self, run, phase_pid, log, clock=time.monotonic, sleep=time.sleep,
                 inventory_fn=freeze_inventory, opener=None, sender=None,
                 closer=os.close, recheck_fn=None):
        self.run=run; self.job=os.environ['SLURM_JOB_ID']; self.uuids=lease_uuids()
        self.phase_pid=phase_pid; self.log=log; self.clock=clock; self.sleep=sleep
        self.inventory_fn=inventory_fn; self.opener=opener or getattr(os,'pidfd_open',None)
        self.sender=sender or getattr(signal,'pidfd_send_signal',None)
        self.closer=closer; self.held={}
        # Injected inventories in CPU tests may explicitly inject their own local
        # identity reader. Production always uses the bounded single-PID reader.
        self.recheck_fn=recheck_fn or (lambda proof,held:local_freeze_recheck(proof,held,deadline=self._deadline,clock=self.clock,sleep=self.sleep))
        if self.opener is None or self.sender is None:
            raise ValueError('Safe Linux pidfd signalling unavailable')

    def view(self):
        return self.inventory_fn(self.run,self.job,self.uuids,self.phase_pid)

    def freeze(self, timeout=5):
        deadline=self.clock()+timeout; self._deadline=deadline;stable=0
        while self.clock()<deadline:
            rows=self.view()
            new=set(rows)-set(self.held)
            for pid in freeze_parent_order(rows):
                if pid not in new:continue
                if self.clock()>=deadline:raise ValueError('Own process freeze deadline before STOP')
                proof=rows[pid]; fd=self.opener(pid,0)
                try:
                    # Opening a pidfd is followed by exact local identity/parent
                    # epoch recheck, rather than another full /proc inventory.
                    now=self.recheck_fn(proof,self.held)
                    if now is None or any(now.get(k)!=proof.get(k) for k in FREEZE_IDENTITY_KEYS):
                        raise ValueError('PID epoch/identity changed before STOP: '+str(pid))
                    if self.clock()>=deadline:raise ValueError('Own process freeze deadline during identity recheck')
                    # Register first: SIGTERM/log exceptions after STOP must still
                    # reach finally-thaw with the original safe pidfd.
                    self.held[pid]=dict(fd=fd,proof=proof)
                    self.sender(fd,signal.SIGSTOP,None,0)
                    self.log(dict(event='owned_freeze_signal',pid=pid,start_ticks=proof['start_ticks'],uid=proof['uid'],role=proof['role'],slurm_job_id=proof['slurm_job_id'],lease_uuids=sorted(self.uuids),certificate=dict(proof)))
                except BaseException:
                    if pid not in self.held: self.closer(fd)
                    raise
            if self.clock()>=deadline:raise ValueError('Own process freeze deadline before stable inventory')
            after=self.view()
            if self.clock()>=deadline:raise ValueError('Own process freeze deadline during stable inventory')
            # No resumed/new process may survive into recovery sampling. Stop
            # ancestors as well as workers, then require two identical snapshots.
            stopped=all(r['state'] in ('T','t') for r in rows.values())and all(r['state'] in ('T','t') for r in after.values())
            known=all(pid in self.held and all(self.held[pid]['proof'].get(k)==r.get(k)for k in FREEZE_IDENTITY_KEYS)for view in (rows,after)for pid,r in view.items())
            stable=stable+1 if stopped and known and set(after)==set(rows) and not new else 0
            if stable>=2:
                self.log(dict(event='owned_freeze_stable',pids=sorted(after),epochs={str(p):r['start_ticks'] for p,r in after.items()}))
                return
            self.sleep(.05)
        raise ValueError('Own process freeze failed to reach stable stopped snapshot')

    def thaw(self, abort=False):
        errors=[]
        def record(event):
            try:self.log(event)
            except Exception as exc:errors.append('resume audit write failed: '+str(exc))
        # Never let log/close failures prevent CONT for the remaining epochs.
        # Defer catchable termination only for this bounded cleanup section.
        mask={signal.SIGTERM,signal.SIGINT,signal.SIGHUP}
        previous=signal.pthread_sigmask(signal.SIG_BLOCK,mask)
        try:
            for pid,entry in sorted(self.held.items(),key=lambda item:item[0]==self.phase_pid):
                error=None
                try:
                    sig=signal.SIGCONT
                    if abort:
                        # Fail-closed compute shutdown precedes all phase CONT.
                        # The held pidfd is the original, frozen certified epoch.
                        sig=signal.SIGTERM if pid==self.phase_pid else signal.SIGKILL
                    for attempt in range(3):
                        try:
                            self.sender(entry['fd'],sig,None,0)
                            if abort and pid==self.phase_pid:
                                self.sender(entry['fd'],signal.SIGCONT,None,0)
                            error=None
                            record(dict(event='owned_abort_signal' if abort else 'owned_resume_signal',signal=int(sig),pid=pid,start_ticks=entry['proof']['start_ticks']))
                            break
                        except ProcessLookupError:
                            error=None
                            record(dict(event='owned_resume_already_exited',pid=pid,start_ticks=entry['proof']['start_ticks']))
                            break
                        except Exception as exc:
                            error=str(exc)
                            record(dict(event='owned_resume_retry',pid=pid,start_ticks=entry['proof']['start_ticks'],attempt=attempt+1,error=error))
                            if attempt<2:self.sleep(.05)
                    if error:errors.append(error)
                except Exception as exc:errors.append('resume cleanup failed: '+str(exc))
                finally:
                    try:self.closer(entry['fd'])
                    except Exception as exc:errors.append('pidfd close failed: '+str(exc))
                    self.held.pop(pid,None)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK,previous)
        if errors:raise ValueError('Own thaw audit/cleanup failed; scoped cleanup required: '+repr(errors))


def observed_sample(run, selected):
    try:
        event=sample(run,selected)
        event['reason']='external_occupancy_exceeds_15_percent' if event['pause'] else 'within_threshold'
        return event
    except Exception as exc:
        event=dict(pause=True,reason='monitor_unknown_fail_closed',error=type(exc).__name__+': '+str(exc))
        if isinstance(exc,SamplingError):event['diagnostic']=exc.diagnostic
        return event


def recover(run, selected, child, emit, grace=60, interval=2, required=3,
            sampler=observed_sample, factory=OwnedFreezer, clock=time.monotonic,
            sleep=time.sleep):
    """Finite pause; all unknown/exceed observations remain unsafe, never zero."""
    start=clock(); freezer=None; safe=0; recovered=False
    emit(dict(event='recovery_started',phase_pid=child.pid if child else None,grace_seconds=grace))
    try:
        if child is not None:
            freezer=factory(run,child.pid,emit); freezer.freeze(timeout=min(5,max(0,grace-(clock()-start))))
        while clock()-start<grace:
            if child is not None and child.poll() is not None:
                raise ValueError('Phase exited during recovery')
            # Catch unexpected child spawns/resumptions before each GPU sample.
            if freezer is not None:freezer.freeze(timeout=min(5,max(0,grace-(clock()-start))))
            if clock()-start>=grace:
                emit(dict(event='recovery_deadline_before_sample',pause=True))
                return False
            event=sampler(run,selected); event.update(event='recovery_sample',elapsed_seconds=clock()-start)
            # A slow bounded query can finish after the observation deadline;
            # its successful result must never authorize a late normal resume.
            if clock()-start>=grace:
                event.update(pause=True,reason='recovery_deadline_during_sample')
                emit(event)
                return False
            emit(event); safe=safe+1 if not event['pause'] else 0
            if safe>=required:
                emit(dict(event='recovery_validated',consecutive_safe_samples=safe))
                recovered=True
                return True
            sleep(min(interval,max(0,grace-(clock()-start))))
        emit(dict(event='recovery_exhausted',pause=True,consecutive_safe_samples=safe))
        return False
    finally:
        # Scoped EXIT cleanup cannot wait forever for STOPped own processes.
        if freezer is not None:
            # Unsafe is termination, not normal resume. Audit failure cannot
            # bypass owned shutdown; thaw itself keeps signalling all entries.
            try:emit(dict(event='recovery_cleanup_only' if not recovered else 'recovery_resume',pause=not recovered))
            finally:freezer.thaw(abort=not recovered)
        elapsed=clock()-start
        emit(dict(event='recovery_finished',elapsed_seconds=elapsed,reserved_gpu_hours=elapsed*8/3600))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    parser.add_argument('--interval',type=float,default=10)
    parser.add_argument('--recovery-seconds',type=float,default=60)
    parser.add_argument('--recovery-interval',type=float,default=2)
    parser.add_argument('--recovery-safe-samples',type=int,default=3)
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args();command=args.command[1:] if args.command[:1]==['--'] else args.command
    if (not command or not math.isfinite(args.interval) or args.interval<1 or
        not math.isfinite(args.recovery_seconds) or not 1<=args.recovery_seconds<=120 or
        not math.isfinite(args.recovery_interval) or not .25<=args.recovery_interval<=10 or
        not 2<=args.recovery_safe_samples<=10):
        parser.error('Finite bounded monitor/recovery settings and command required')
    if os.environ.get('BAPO_RUN_ID')!=args.run or not os.environ.get('SLURM_JOB_ID'):
        parser.error('Exact inherited BAPO_RUN_ID/Slurm allocation required')
    run=Path(args.run)
    if not run.is_absolute() or not run.is_dir():parser.error('Existing absolute own run required')
    selected=[v.strip() for v in os.environ.get('CUDA_VISIBLE_DEVICES','').split(',') if v.strip()]
    child=None
    def interrupted(signum,frame):raise RecoveryInterrupted(signum)
    previous={sig:signal.signal(sig,interrupted) for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP)}
    try:
        with (run/'resource-watch.jsonl').open('a',buffering=1) as log:
            def emit(event):
                event.update(time=time.time(),run=args.run,slurm_job_id=os.environ['SLURM_JOB_ID'],phase_pid=child.pid if child else None)
                log.write(json.dumps(event,allow_nan=False)+'\n')
            while True:
                event=observed_sample(args.run,selected);emit(event)
                if event['pause']:
                    try:
                        safe=recover(args.run,selected,child,emit,args.recovery_seconds,args.recovery_interval,args.recovery_safe_samples)
                    except Exception as exc:
                        emit(dict(event='recovery_failed',pause=True,error=type(exc).__name__+': '+str(exc)));safe=False
                    if not safe:
                        print('RESOURCE_PAUSE: finite recovery failed; own scoped cleanup required',file=sys.stderr);return 42
                if child is None:
                    os.environ['SEARCH_R1_GPU_RESOURCE_GUARD']=json.dumps(dict(pid=os.getpid(),start_ticks=proc_start_ticks(os.getpid()),run=args.run))
                    child=subprocess.Popen(command,close_fds=False)
                deadline=time.monotonic()+args.interval
                while time.monotonic()<deadline:
                    code=child.poll()
                    if code is not None:return code if code>=0 else 128-code
                    time.sleep(min(.25,max(0,deadline-time.monotonic())))
    except RecoveryInterrupted as exc:
        return 128+exc.signum
    finally:
        for sig,handler in previous.items():signal.signal(sig,handler)


if __name__=='__main__':sys.exit(main())
