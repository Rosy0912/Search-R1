"""Read-only scalar observation for the one frozen formal run (stdlib only).

No training modules are imported. Visible newline-terminated receipts describe
an observed prefix, not completed fsync, acceptance, or a healthy checkpoint.
"""
import csv
import hashlib
import io
import json
import math
import os
import re
import stat
import subprocess
import time
from pathlib import Path

BASE = '/mnt/data/shaoshitong/g1f-3ff5337a'
RUN = 'db5235708c354bffae6d0120c1cb5459'
JOB = '3309'
COMMIT = 'dec4e52b67042373348ef522ddc611d337eadf79'
MAP_SHA = '63c97052a028b4edafb4d2c4f5140f6aef40e0b0c835dd4fdcda04acb1cc5372'
ROLE = 'formal_inline_fresh200'
MAX_DOCUMENT = 2 * 1024 * 1024
MAX_ATTEMPTS = 8 * 1024 * 1024
MAX_UPDATES = 2 * 1024 * 1024
MAX_GPU = 128 * 1024 * 1024
MAX_SCHEDULER = 16 * 1024 * 1024
MAX_LINE = 256 * 1024
MAX_ATTEMPT_ROWS = 20000
MAX_GPU_ROWS = 100000
MAX_SCHEDULER_ROWS = 60000
REMOTE_SECONDS = 60
REQUIRED_METRICS = (
    'critic/rewards/mean', 'actor/pg_loss', 'actor/entropy_loss',
    'actor/kl_loss', 'actor/ppo_kl', 'actor/grad_norm',
    'actor/optimizer_steps_this_update',
    'actor/update_opportunities_this_attempt',
    'actor/skipped_update_opportunities',
)
ATTEMPT_FIELDS = frozenset((
    'run_id', 'slurm_job_id', 'status', 'logical_update', 'attempt_id',
    'attempt_index', 'optimizer_writes_before', 'optimizer_writes_after',
    'optimizer_writes_this_attempt', 'opportunities_processed',
    'opportunity_statuses', 'guard_stage', 'guard_reasons', 'complete_outer',
    'retained_partial_adam_writes', 'no_automatic_rollback',
    'scheduler_advances', 'completed_outer', 'all_ranks_agreed',
    'real_rank_count', 'cumulative_aborted_attempts',
))
UPDATE_FIELDS = frozenset((
    'run_id', 'step', 'metrics', 'probe_version',
    'behavior_probability_gate_sha256', 'behavior_trace_sha256',
    'attempt_id', 'attempt_index', 'optimizer_writes_before',
    'optimizer_writes_after', 'opportunities_processed',
    'actual_optimizer_writes',
))
GPU_FIELDS = frozenset((
    'schema_version', 'role', 'run_id', 'slurm_job_id', 'source_map_sha256',
    'sampler_identity', 'sampling_interval_seconds', 'query_started_unix',
    'unix', 'gpus',
))
SCHEDULER_HEADER = frozenset((
    'schema_version', 'role', 'contract_kind', 'rank', 'world_size',
    'process_pid', 'process_start_ticks', 'base_learning_rate', 'warmup_steps',
    'observed_unix', 'run_id', 'source_commit', 'slurm_job_id',
    'source_map_sha256', 'policy_safety_mode', 'scheduler_contract_sha256',
))
SCHEDULER_BODY = frozenset((
    'kind', 'logical_update', 'writes_before', 'writes_after',
    'actual_optimizer_writes', 'completed_outer_before', 'completed_outer_after',
    'consumed_learning_rates', 'next_learning_rates',
    'scheduler_last_epoch_before', 'scheduler_last_epoch_after',
    'scheduler_step_count_before', 'scheduler_step_count_after',
    'batch_gate_sha256', 'scheduler_advances', 'attempt_id', 'attempt_index',
    'opportunities_processed', 'optimizer_opportunities',
))
SCHEDULER_ABORT = frozenset((
    'guard_stage', 'guard_reasons', 'recoverable_guard',
    'retained_partial_adam_writes', 'no_automatic_rollback',
    'scheduler_unchanged_on_rejection',
))


class ObservationError(ValueError):
    pass


def need(ok, message):
    if not ok:
        raise ObservationError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def hash_value(value, length=64):
    need(type(value) is str and
         re.fullmatch('[0-9a-f]{' + str(length) + '}', value) is not None,
         'Malformed hash/UUID')
    return value


def integer(value, label, minimum=0, maximum=None):
    need(type(value) is int and value >= minimum and
         (maximum is None or value <= maximum), 'Invalid integer: ' + label)
    return value


def finite(value, label, minimum=None):
    need(type(value) in (int, float), 'Non-numeric scalar: ' + label)
    try:
        valid = math.isfinite(value)
    except OverflowError:
        valid = False
    need(valid and (minimum is None or value >= minimum),
         'Nonfinite/out-of-range scalar: ' + label)
    return value


def strict_json(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            need(key not in value, 'Duplicate JSON field: ' + key)
            value[key] = item
        return value
    def reject(value):
        raise ObservationError('Nonfinite JSON constant: ' + value)
    def number(value):
        return finite(float(value), 'JSON exponent')
    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=reject,
                          parse_float=number)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as error:
        raise ObservationError('Invalid JSON') from error


def keys(row, names, label):
    need(type(row) is dict and set(row) == set(names),
         'Unexpected ' + label + ' schema')


def receipt_identity(row, job=True):
    need(type(row) is dict and row.get('run_id') == RUN,
         'Receipt belongs to another run')
    if job:
        need(row.get('slurm_job_id') == JOB, 'Receipt belongs to another job')


def jsonl_prefix(raw, *, max_bytes, max_lines):
    """Only newline-terminated rows; even valid JSON at the tail is incomplete."""
    need(type(raw) is bytes and len(raw) <= max_bytes, 'JSONL byte bound exceeded')
    newline = raw.rfind(b'\n')
    prefix = raw[:newline + 1] if newline >= 0 else b''
    tail = raw[newline + 1:]
    need(len(tail) <= MAX_LINE, 'JSONL tail line bound exceeded')
    rows = []
    for line in prefix.splitlines():
        need(line and len(line) <= MAX_LINE, 'Empty/overlong complete JSONL row')
        need(len(rows) < max_lines, 'JSONL line bound exceeded')
        value = strict_json(line)
        need(type(value) is dict, 'JSONL row must be an object')
        rows.append(value)
    return rows, dict(captured_bytes=len(raw), captured_sha256=sha(raw),
        complete_prefix_bytes=len(prefix), complete_prefix_sha256=sha(prefix),
        complete_rows=len(rows), partial_tail_bytes=len(tail),
        partial_tail_sha256=sha(tail) if tail else None,
        observation_incomplete=bool(tail))


def validate_attempts(rows):
    completed = writes = skipped = 0
    ids = set()
    complete_rows = []
    for ordinal, row in enumerate(rows, 1):
        keys(row, ATTEMPT_FIELDS, 'driver attempt')
        receipt_identity(row)
        hash_value(row['attempt_id'], 32)
        need(row['attempt_id'] not in ids, 'Attempt UUID reused')
        ids.add(row['attempt_id'])
        for name in ('attempt_index', 'logical_update', 'optimizer_writes_before',
                     'optimizer_writes_after', 'opportunities_processed',
                     'scheduler_advances', 'completed_outer',
                     'cumulative_aborted_attempts', 'real_rank_count'):
            integer(row[name], name)
        count = integer(row['optimizer_writes_this_attempt'], 'attempt writes', 0, 20)
        need(row['attempt_index'] == ordinal and
             row['logical_update'] == completed + 1 and completed < 200,
             'Attempt prefix sequence differs')
        need(row['all_ranks_agreed'] is True and row['real_rank_count'] == 8,
             'Fatal eight-rank attempt disagreement')
        need(row['optimizer_writes_before'] == writes and
             row['optimizer_writes_after'] == writes + count and
             row['opportunities_processed'] == 20,
             'Attempt is partial or retained write counters differ')
        statuses = row['opportunity_statuses']
        need(type(statuses) is list and len(statuses) == 20 and
             all(type(item) is str and item in
                 ('written', 'written_with_post_guard', 'skipped_guard')
                 for item in statuses) and
             sum(item != 'skipped_guard' for item in statuses) == count,
             'Opportunity statuses disagree with actual writes')
        complete = count > 0
        need(type(row['complete_outer']) is bool and row['complete_outer'] is complete and
             row['status'] == ('complete' if complete else 'recoverable_guard') and
             row['scheduler_advances'] == int(complete) and
             row['completed_outer'] == completed + int(complete),
             'Completion/warmup accounting differs')
        need(row['guard_stage'] == (None if complete else 'all_update_opportunities_skipped') and
             row['guard_reasons'] == ([] if complete else ['no_successful_adam_write']) and
             row['retained_partial_adam_writes'] is (not complete) and
             row['no_automatic_rollback'] is True, 'Skip/rollback evidence differs')
        skipped += int(not complete)
        need(row['cumulative_aborted_attempts'] == skipped,
             'Fully skipped attempt counter differs')
        if complete:
            complete_rows.append(row)
        writes += count
        completed += int(complete)
    return dict(observed_driver_completed_outer=completed,
        observed_scheduler_advances=completed, observed_attempt_count=len(rows),
        observed_fully_skipped_attempts=skipped,
        observed_processed_opportunities_per_rank=20 * len(rows),
        agreed_writes_at_last_return_per_rank=writes,
        actual_total_writes_unavailable=True,
        actual_total_writes_unavailable_reason='optimizer ledgers are not collected; '
            'inflight/abandoned retained writes may exceed the agreed driver counter',
        observed_skipped_opportunities_per_rank=20 * len(rows) - writes,
        target_completed_outer=200, fixed_warmup_outer=57,
        scope='visible driver all-eight-rank agreement prefix; agreed write '
              'counter is per rank and is not multiplied by eight; no fsync/acceptance claim'), complete_rows


def validate_updates(rows, complete_attempts):
    need(len(rows) <= len(complete_attempts), 'Metrics ahead of driver completion prefix')
    for step, row in enumerate(rows, 1):
        keys(row, UPDATE_FIELDS, 'update metrics')
        receipt_identity(row, job=False)
        integer(row['step'], 'metrics step', 1, 200)
        need(row['step'] == step, 'Metrics prefix has a gap/duplicate/reorder')
        hash_value(row['attempt_id'], 32)
        hash_value(row['behavior_probability_gate_sha256'])
        hash_value(row['behavior_trace_sha256'])
        need(row['probe_version'] is None or
             (type(row['probe_version']) is str and 0 < len(row['probe_version']) <= 256),
             'Invalid probe version')
        attempt = complete_attempts[step - 1]
        for name in ('attempt_id', 'attempt_index', 'optimizer_writes_before',
                     'optimizer_writes_after', 'opportunities_processed'):
            need(type(row[name]) is type(attempt[name]) and row[name] == attempt[name],
                 'Metrics/driver attempt differs: ' + name)
        count = integer(row['actual_optimizer_writes'], 'metrics writes', 1, 20)
        need(count == attempt['optimizer_writes_this_attempt'], 'Metrics write count differs')
        keys(row['metrics'], REQUIRED_METRICS, 'full precision scalar metrics')
        for name, value in row['metrics'].items():
            finite(value, name)
        need(row['metrics']['actor/optimizer_steps_this_update'] == count and
             row['metrics']['actor/update_opportunities_this_attempt'] == 20 and
             row['metrics']['actor/skipped_update_opportunities'] == 20 - count,
             'Metrics opportunity/write accounting differs')
    latest = rows[-1] if rows else None
    return dict(last_metrics_completed_step=len(rows),
        metrics_lag_completed_outer=len(complete_attempts) - len(rows),
        latest_metrics=latest['metrics'] if latest else None,
        probe_version=latest['probe_version'] if latest else None,
        latest_attempt_id=latest['attempt_id'] if latest else None,
        metrics_state='available' if rows else 'no_completed_metrics_rows',
        scope='full precision actual-updates scalar prefix; absence may be valid driver-to-metrics lag')


def process_identity(row):
    keys(row, ('pid', 'start_ticks', 'sid', 'uid'), 'process birth')
    for name in ('pid', 'start_ticks', 'sid'):
        integer(row[name], 'process ' + name, 1)
    integer(row['uid'], 'process uid')
    return row


def validate_gpu_rows(rows, inventory):
    need(type(rows) is list and len(rows) == 8, 'Need eight physical GPU scalars')
    need(type(inventory) is dict and len(inventory) == 8, 'Invalid bound GPU inventory')
    seen = set()
    for row in rows:
        keys(row, ('index', 'uuid', 'used_mib', 'total_mib', 'util_percent'), 'GPU scalar')
        index = integer(row['index'], 'GPU index', 0)
        need(index not in seen and index in inventory and
             row['uuid'] == inventory[index]['uuid'], 'GPU UUID/index provenance differs')
        seen.add(index)
        used = integer(row['used_mib'], 'GPU used MiB', 0)
        total = integer(row['total_mib'], 'GPU total MiB', 1)
        integer(row['util_percent'], 'GPU utilization percent', 0, 100)
        need(total == inventory[index]['total_mib'] and used <= total,
             'GPU memory scalar differs from physical inventory')
    return rows


def validate_gpu_samples(rows, inventory, *, now):
    finite(now, 'observation clock', 0)
    peaks = {str(index): 0 for index in inventory}
    previous = None
    sampler = None
    intervals = []
    gaps = []
    for ordinal, row in enumerate(rows, 1):
        keys(row, GPU_FIELDS, 'GPU sample')
        receipt_identity(row)
        need(type(row['schema_version']) is int and row['schema_version'] == 1 and
             row['role'] == 'actual_common8_gpu_occupancy' and
             row['source_map_sha256'] == MAP_SHA and
             type(row['sampling_interval_seconds']) is int and
             row['sampling_interval_seconds'] == 2, 'GPU sampler source/role/interval differs')
        identity = process_identity(row['sampler_identity'])
        if sampler is None:
            sampler = identity
        need(identity == sampler, 'GPU sampler process birth changed inside this run')
        started = finite(row['query_started_unix'], 'GPU query started', 0)
        ended = finite(row['unix'], 'GPU query ended', 0)
        need(0 < started <= ended <= now + 1 and ended - started <= 31,
             'GPU sample query window differs')
        if previous is not None:
            need(started >= previous, 'GPU samples duplicated/reordered/overlapping')
            gap = started - previous
            intervals.append(ended - rows[ordinal - 2]['unix'])
            # A reported gap is a measurement, never a safety threshold.
            if gap > 3:
                gaps.append(dict(after_row=ordinal - 1, before_row=ordinal,
                                 idle_seconds=gap))
        validate_gpu_rows(row['gpus'], inventory)
        for gpu in row['gpus']:
            key = str(gpu['index'])
            peaks[key] = max(peaks[key], gpu['used_mib'])
        previous = ended
    return dict(sample_count=len(rows), nominal_sampling_interval_seconds=2,
        sampling_started_unix=rows[0]['query_started_unix'] if rows else None,
        sampling_finished_unix=rows[-1]['unix'] if rows else None,
        sampling_window_seconds=(rows[-1]['unix'] - rows[0]['query_started_unix']) if rows else None,
        age_of_last_sample_seconds=now - rows[-1]['unix'] if rows else None,
        observed_interval_min_seconds=min(intervals) if intervals else None,
        observed_interval_max_seconds=max(intervals) if intervals else None,
        observed_gap_count=len(gaps), largest_idle_gap_seconds=max((r['idle_seconds'] for r in gaps), default=None),
        gap_examples=gaps[:20], gap_examples_truncated=len(gaps) > 20,
        cumulative_sampled_peak_mib=peaks if rows else None,
        latest_sample_gpus=rows[-1]['gpus'] if rows else None,
        sampler_identity=sampler,
        sample_state='available' if rows else 'no_complete_samples',
        scope='entire captured newline-complete prefix for this run; sampled '
              'peak is not an unsampled peak or a full rollout peak')


def native_summary(observed, fields=None):
    """A missing/purged scheduler record stays unknown even at outer 200."""
    need(type(observed) is dict and observed.get('base') == BASE and
         observed.get('job_id') == JOB, 'Native observation has wrong saved handle')
    code = observed.get('job_returncode')
    need(type(code) is int, 'Native return code unavailable')
    if code != 0:
        return dict(state='unknown', exit_code=None, restart_count=None,
                    reason='native_record_unavailable_or_purged',
                    terminal_success='unknown')
    need(type(fields) is dict and fields.get('JobId') == JOB, 'Native record differs')
    return dict(state=fields.get('JobState', 'unknown'), exit_code=fields.get('ExitCode'),
        restart_count=fields.get('Restarts'), submit_time=fields.get('SubmitTime'),
        start_time=fields.get('StartTime'), end_time=fields.get('EndTime'),
        node=fields.get('NodeList'), terminal_success='not_evaluated',
        scope='actual native record captured by existing observe(); no terminal-success inference')


def deadline_check(deadline):
    need(time.monotonic() <= deadline, 'Read-only collection time bound exceeded')


def snapshot(path, max_bytes, deadline):
    """Capture initial append-only file extent; an appended suffix is deferred.

    Paths and inodes cannot be replaced, truncated, or linked. Metadata records
    whether an append occurred. Prefix hashes always hash the bytes parsed.
    """
    deadline_check(deadline)
    captured_started = time.time()
    path = Path(path)
    need(path.resolve() == path, 'Noncanonical/symlink evidence path')
    before = path.lstat()
    need(stat.S_ISREG(before.st_mode) and before.st_uid == os.getuid() and
         0 <= before.st_size <= max_bytes,
         'Nonregular/over-bound evidence file')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, 'O_CLOEXEC', 0))
    try:
        opened = os.fstat(fd)
        identity = lambda item: (item.st_dev, item.st_ino, item.st_mode, item.st_uid)
        need(identity(before) == identity(opened) and opened.st_size >= before.st_size,
             'Evidence inode changed before capture')
        size = before.st_size
        chunks = []
        remaining = size
        while remaining:
            deadline_check(deadline)
            block = os.read(fd, min(65536, remaining))
            need(block, 'Evidence capture was truncated')
            chunks.append(block)
            remaining -= len(block)
        after = os.fstat(fd)
        current = path.lstat()
        need(identity(after) == identity(current) == identity(opened) and
             after.st_size >= size and current.st_size >= size,
             'Evidence inode changed or file truncated during capture')
        grew = current.st_size > size
        need(grew or (before.st_size == current.st_size and
                      before.st_mtime_ns == current.st_mtime_ns and
                      before.st_ctime_ns == current.st_ctime_ns),
             'Evidence same-size content changed during capture')
        raw = b''.join(chunks)
        return raw, dict(path=str(path), capture_started_size=size,
            capture_started_unix=captured_started, capture_finished_unix=time.time(),
            observed_size_after=current.st_size, appended_after_capture=grew,
            dev=opened.st_dev, inode=opened.st_ino, owner_uid=opened.st_uid,
            captured_sha256=sha(raw), captured_bytes=len(raw))
    finally:
        os.close(fd)


def capture_json(path, deadline):
    raw, provenance = snapshot(path, MAX_DOCUMENT, deadline)
    return strict_json(raw), provenance


def capture_jsonl(path, maximum, rows, deadline):
    raw, provenance = snapshot(path, maximum, deadline)
    parsed, prefix = jsonl_prefix(raw, max_bytes=maximum, max_lines=rows)
    prefix['observation_incomplete'] = (prefix['observation_incomplete'] or
                                        provenance['appended_after_capture'])
    return parsed, dict(provenance, **prefix)


def unavailable(error, *, missing=False):
    # Do not export arbitrary error text containing raw source material.
    return dict(evidence_state='unavailable' if missing else 'invalid_or_unstable',
        reason_code='missing_file' if missing else type(error).__name__,
        observation_incomplete=True)


def optional_capture(path, maximum, rows, deadline):
    try:
        values, provenance = capture_jsonl(path, maximum, rows, deadline)
        return values, dict(provenance, evidence_state='observed')
    except FileNotFoundError as error:
        return None, unavailable(error, missing=True)
    except (ObservationError, OSError) as error:
        return None, unavailable(error)


def actual_birth(saved, deadline):
    process_identity(saved)
    deadline_check(deadline)
    proc = Path('/proc') / str(saved['pid'])
    try:
        before = proc.stat()
        raw = (proc / 'stat').read_bytes()
        need(len(raw) < 4096, 'Proc stat bound exceeded')
        fields = raw.rsplit(b')', 1)[1].split()
        current = dict(pid=saved['pid'], start_ticks=int(fields[19]),
                       sid=int(fields[3]), uid=before.st_uid)
        state = fields[0].decode('ascii')
        env_fd = os.open(proc / 'environ', os.O_RDONLY | os.O_NONBLOCK)
        try:
            env = os.read(env_fd, 65537)
            need(len(env) <= 65536, 'Process environment byte bound exceeded')
        finally:
            os.close(env_fd)
        second = (proc / 'stat').read_bytes().rsplit(b')', 1)[1].split()
        need(int(second[19]) == current['start_ticks'] and
             proc.stat().st_uid == current['uid'], 'Process changed during birth check')
        same = current == saved
        namespace = (('SEARCH_G1_BASE=' + BASE).encode() in env.split(b'\0') and
                     ('SLURM_JOB_ID=' + JOB).encode() in env.split(b'\0'))
        return dict(saved_identity=saved, actual_identity=current, process_state=state,
                    birth_matches=same, namespace_matches=namespace if same else False,
                    evidence_state='observed', checked_unix=time.time())
    except (FileNotFoundError, ProcessLookupError):
        return dict(saved_identity=saved, actual_identity=None, birth_matches=None,
                    namespace_matches=None, evidence_state='absent_now',
                    checked_unix=time.time())
    except (OSError, ObservationError, ValueError, IndexError):
        return dict(saved_identity=saved, actual_identity=None, birth_matches=None,
                    namespace_matches=None, evidence_state='unknown',
                    checked_unix=time.time())


def allocation_inventory(row):
    receipt_identity(row)
    need(row.get('node') == 'h100-8', 'Allocation belongs to another node')
    indices = row.get('actual_allocated_indices')
    need(type(indices) is list and len(indices) == 8 and
         all(type(index) is int and index >= 0 for index in indices) and
         len(set(indices)) == 8, 'Invalid allocated physical indices')
    physical = row.get('physical_gpu_inventory')
    need(type(physical) is list and len(physical) == 8, 'Invalid physical GPU inventory')
    result = {}
    for values in physical:
        need(type(values) is list and len(values) == 3 and
             all(type(item) is str for item in values), 'Malformed GPU inventory row')
        index = int(values[0].strip())
        uuid = values[1].strip()
        total = int(values[2].strip())
        need(index in indices and index not in result and
             re.fullmatch(r'GPU-[0-9a-fA-F-]{36}', uuid) is not None and total > 0,
             'Malformed physical GPU identity')
        result[index] = dict(uuid=uuid, total_mib=total)
    need(len({item['uuid'] for item in result.values()}) == 8,
         'Duplicated physical GPU UUID')
    return result


def current_gpu(inventory, deadline, capture):
    deadline_check(deadline)
    started = time.time()
    budget = min(10, max(0.01, deadline - time.monotonic()))
    try:
        completed = capture([
            'nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu',
            '--format=csv,noheader,nounits'], budget, 16384, 4096)
        need(completed.returncode == 0,
             'Current GPU query failed/bound exceeded')
        values = []
        for row in csv.reader(io.StringIO(completed.stdout)):
            need(len(row) == 5, 'Current GPU query schema differs')
            values.append(dict(index=int(row[0].strip()), uuid=row[1].strip(),
                used_mib=int(row[2].strip()), total_mib=int(row[3].strip()),
                util_percent=int(row[4].strip())))
        validate_gpu_rows(values, inventory)
        return dict(evidence_state='observed', query_started_unix=started,
            query_finished_unix=time.time(), stdout_sha256=sha(completed.stdout.encode()),
            gpus=values, scope='current physical node inventory; not attribution of all occupancy to this run')
    except (OSError, subprocess.TimeoutExpired, UnicodeDecodeError, ValueError) as error:
        return unavailable(error)


def stage_summary(row):
    receipt_identity(row, job=False)
    need(type(row.get('stage')) is str and 0 < len(row['stage']) <= 256,
         'Malformed stage receipt')
    finite(row.get('unix'), 'stage time', 0)
    return dict(stage=row['stage'], unix=row['unix'],
                scope='reported pipeline stage; no completion or success inference')


def evaluation_summary(row, expected_step):
    receipt_identity(row, job=False)
    need(type(row.get('schema_version')) is int and row['schema_version'] == 1 and
         row.get('role') == 'formal_inline_development_evaluation' and
         type(row.get('step')) is int and row['step'] == expected_step and
         row.get('policy_safety_mode') == 'inline_first_five' and
         row.get('source_map_sha256') == MAP_SHA, 'Evaluation provenance differs')
    hash_value(row['plan_sha256'])
    hash_value(row['validation_sha256'])
    finite(row['checked_unix'], 'evaluation time', 0)
    allowed = ('weighted_em', 'weighted_f1', 'valid_action_rate',
        'executed_search_question_rate', 'executed_search_calls_per_question',
        'correct_count', 'f1_sum')
    metrics = {}
    for name in allowed:
        if name in row:
            metrics[name] = finite(row[name], 'evaluation ' + name, 0)
    need('weighted_em' in metrics and 'weighted_f1' in metrics,
         'Evaluation aggregate scalars absent')
    return dict(step=expected_step, checked_unix=row['checked_unix'],
        plan_sha256=row['plan_sha256'], validation_sha256=row['validation_sha256'],
        metrics=metrics, scope='observed paired dev512 aggregate scalars; raw rows/IDs/answers not exported')


def inspect_scheduler_rows(rows, rank, attempts):
    """Crosscheck existing driver receipts; suffix never advances global count.

    Missing/partial rank files leave a separate crosscheck unavailable. Failure
    records or conflicting completed receipts invalidate the scalar snapshot.
    """
    expected = {}
    for attempt in attempts:
        expected[attempt['attempt_id']] = attempt
    groups = {}
    suffix_count = 0
    saved_birth = None
    seen_groups = []
    every_group = {}
    for row in rows:
        receipt_identity(row)
        need(type(row.get('rank')) is int and row['rank'] == rank and
             type(row.get('world_size')) is int and row['world_size'] == 8 and
             type(row.get('schema_version')) is int and row['schema_version'] == 1 and
             row.get('role') == 'fresh200_outer_scheduler' and
             row.get('source_commit') == COMMIT and row.get('source_map_sha256') == MAP_SHA and
             row.get('policy_safety_mode') == 'inline_first_five' and
             row.get('contract_kind') == 'fresh200_warmup57', 'Scheduler rank/source identity differs')
        for name in ('process_pid', 'process_start_ticks'):
            integer(row[name], name, 1)
        hash_value(row['scheduler_contract_sha256'])
        need(row.get('kind') not in ('REJECTED', 'POST_COMMIT_FAILURE', 'PRE_COMMIT_EVIDENCE_FAILURE'),
             'Fatal scheduler evidence record')
        need(row.get('kind') in ('BEGIN', 'COMMIT_PREPARED', 'COMPLETE', 'ABORTED_ATTEMPT'),
             'Unknown scheduler receipt kind')
        keys(row, SCHEDULER_HEADER | SCHEDULER_BODY |
             (SCHEDULER_ABORT if row['kind'] == 'ABORTED_ATTEMPT' else frozenset()),
             'scheduler complete row')
        hash_value(row['attempt_id'], 32)
        integer(row['attempt_index'], 'scheduler attempt index', 1)
        birth = (row['process_pid'], row['process_start_ticks'])
        if saved_birth is None:
            saved_birth = birth
        need(birth == saved_birth, 'Scheduler process birth changed')
        need(type(row['warmup_steps']) is int and row['warmup_steps'] == 57 and
             finite(row['base_learning_rate'], 'scheduler base LR') == 1e-6,
             'Scheduler fixed warmup/LR differs')
        finite(row['observed_unix'], 'scheduler receipt time', 0)
        for name in ('logical_update', 'writes_before', 'writes_after',
                     'actual_optimizer_writes', 'completed_outer_before',
                     'completed_outer_after', 'scheduler_last_epoch_before',
                     'scheduler_last_epoch_after', 'scheduler_step_count_before',
                     'scheduler_step_count_after', 'scheduler_advances',
                     'opportunities_processed', 'optimizer_opportunities'):
            integer(row[name], name)
        need(row['optimizer_opportunities'] == 20 and
             row['opportunities_processed'] <= 20 and
             row['writes_after'] - row['writes_before'] == row['actual_optimizer_writes'] and
             0 <= row['actual_optimizer_writes'] <= row['opportunities_processed'],
             'Scheduler write/opportunity counters differ')
        for name in ('consumed_learning_rates', 'next_learning_rates'):
            need(type(row[name]) is list and row[name], 'Missing scheduler LR groups')
            for value in row[name]:
                finite(value, name, 0)
        hash_value(row['batch_gate_sha256'])
        advanced = row['kind'] == 'COMPLETE'
        prepared = row['kind'] in ('COMPLETE', 'COMMIT_PREPARED')
        before = row['completed_outer_before']
        after = before + int(advanced)
        need(row['logical_update'] == before + 1 and before < 200 and
             row['completed_outer_after'] == after and row['scheduler_advances'] == int(advanced) and
             row['scheduler_last_epoch_before'] == before and row['scheduler_last_epoch_after'] == after and
             row['scheduler_step_count_before'] == before + 1 and row['scheduler_step_count_after'] == after + 1,
             'Scheduler outer/warmup states differ')
        expected_before_lr = 1e-6 * min(before / 57, 1)
        expected_after_lr = 1e-6 * min(after / 57, 1)
        need(all(value == expected_before_lr for value in row['consumed_learning_rates']) and
             len(row['next_learning_rates']) == len(row['consumed_learning_rates']) and
             all(value == expected_after_lr for value in row['next_learning_rates']),
             'Scheduler consumed/next LR differs')
        if prepared:
            need(row['opportunities_processed'] == 20 and row['actual_optimizer_writes'] >= 1,
                 'Prepared/complete rank needs all20 and at least one write')
        elif row['kind'] == 'BEGIN':
            need(row['opportunities_processed'] == row['actual_optimizer_writes'] == 0,
                 'Scheduler BEGIN cannot report writes')
        else:
            need(row['recoverable_guard'] is True and row['retained_partial_adam_writes'] is True and
                 row['no_automatic_rollback'] is True and row['scheduler_unchanged_on_rejection'] is True and
                 type(row['guard_stage']) is str and type(row['guard_reasons']) is list and
                 all(type(value) is str for value in row['guard_reasons']) and
                 (row['opportunities_processed'] < 20 or row['actual_optimizer_writes'] == 0),
                 'Abandoned/fully-skipped scheduler semantics differ')
        if not seen_groups or seen_groups[-1] != row['attempt_id']:
            need(row['attempt_id'] not in seen_groups and row['attempt_index'] == len(seen_groups) + 1,
                 'Scheduler prefix has a gap/duplicate/reorder')
            seen_groups.append(row['attempt_id'])
        every_group.setdefault(row['attempt_id'], []).append(row)
        if row['attempt_id'] not in expected:
            need(row['attempt_index'] == len(attempts) + 1,
                 'Scheduler receipt cannot bind driver prefix or single inflight attempt')
            suffix_count += 1
            continue
        groups.setdefault(row['attempt_id'], []).append(row)
    missing = 0
    terminals = []
    for attempt in attempts:
        selected = groups.get(attempt['attempt_id'], [])
        kinds = ['BEGIN', 'COMMIT_PREPARED', 'COMPLETE'] if attempt['complete_outer'] else ['BEGIN', 'ABORTED_ATTEMPT']
        if [row['kind'] for row in selected] != kinds:
            need([row['kind'] for row in selected] == kinds[:len(selected)],
                 'Scheduler prefix duplicated/reordered')
            missing += 1
            continue
        terminal = selected[-1]
        before = attempt['completed_outer'] - int(attempt['complete_outer'])
        for row in selected:
            for name in ('logical_update', 'attempt_index'):
                need(type(row[name]) is int and row[name] == attempt[name],
                     'Fatal rank/driver scheduler identity disagreement')
            need(row['attempt_id'] == attempt['attempt_id'], 'Scheduler UUID differs')
            hash_value(row['batch_gate_sha256'])
        for name, value in dict(writes_before=attempt['optimizer_writes_before'],
            writes_after=attempt['optimizer_writes_after'],
            actual_optimizer_writes=attempt['optimizer_writes_this_attempt'],
            opportunities_processed=20, optimizer_opportunities=20,
            completed_outer_before=before, completed_outer_after=attempt['completed_outer'],
            scheduler_advances=attempt['scheduler_advances'],
            scheduler_last_epoch_after=attempt['completed_outer'],
            scheduler_step_count_after=attempt['completed_outer'] + 1).items():
            need(type(terminal.get(name)) is int and terminal[name] == value,
                 'Fatal rank/driver terminal disagreement: ' + name)
        terminals.append(dict(attempt_id=attempt['attempt_id'],
            attempt_index=attempt['attempt_index'], completed_outer=attempt['completed_outer'],
            actual_optimizer_writes=attempt['optimizer_writes_this_attempt'],
            batch_gate_sha256=terminal['batch_gate_sha256'],
            scheduler_contract_sha256=terminal['scheduler_contract_sha256'],
            consumed_learning_rates=terminal['consumed_learning_rates'],
            next_learning_rates=terminal['next_learning_rates'],
            observed_scheduler_last_epoch=terminal['scheduler_last_epoch_after'],
            observed_scheduler_step_count=terminal['scheduler_step_count_after']))
    for selected in every_group.values():
        kinds = [row['kind'] for row in selected]
        need(kinds in (['BEGIN'], ['BEGIN', 'COMMIT_PREPARED'],
                       ['BEGIN', 'COMMIT_PREPARED', 'COMPLETE'], ['BEGIN', 'ABORTED_ATTEMPT']),
             'Scheduler group is incomplete/reordered/duplicated')
        gate = selected[0]['batch_gate_sha256']
        need(all(row['batch_gate_sha256'] == gate and
                 row['writes_before'] == selected[0]['writes_before'] and
                 row['completed_outer_before'] == selected[0]['completed_outer_before']
                 for row in selected), 'Scheduler group shared state differs')
    latest = rows[-1] if rows else None
    observed_state = {name: latest[name] for name in (
        'kind', 'attempt_id', 'attempt_index', 'logical_update',
        'consumed_learning_rates', 'next_learning_rates',
        'completed_outer_before', 'completed_outer_after',
        'scheduler_last_epoch_before', 'scheduler_last_epoch_after',
        'scheduler_step_count_before', 'scheduler_step_count_after',
        'writes_before', 'writes_after', 'actual_optimizer_writes',
        'opportunities_processed', 'scheduler_advances')} if latest else None
    return dict(rank=rank, matched_driver_attempts=len(terminals),
        unavailable_driver_attempts=missing, uncommitted_suffix_rows=suffix_count,
        last_driver_matched_state=terminals[-1] if terminals else None,
        latest_rank_receipt_state=observed_state,
        scope='rank receipt crosscheck only; suffix rows never increase global completion'), terminals


def collect_remote(payload, protocol):
    """The sole remote entry. Reads fixed files and runs only nvidia-smi."""
    deadline = time.monotonic() + REMOTE_SECONDS
    started_unix = time.time()
    need(payload.get('base') == BASE and payload.get('run_id') == RUN and
         payload.get('job_id') == JOB and payload.get('source_commit') == COMMIT and
         payload.get('source_map_sha256') == MAP_SHA, 'Wrong frozen collection identity')
    base = Path(BASE)
    need(base.resolve() == base and base.is_dir() and not base.is_symlink(),
         'Remote base is not the frozen ordinary directory')
    mapping = protocol.runtime_map(base / 'repo')
    deadline_check(deadline)
    need(mapping == payload.get('source_sha256') and len(mapping) == 372 and
         sha(canonical(mapping)) == MAP_SHA, 'Full reviewed runtime source map differs')
    run, run_provenance = capture_json(base / 'logs/run-manifest.json', deadline)
    receipt_identity(run, job=False)
    need(run.get('source_commit') == COMMIT and run.get('source_sha256') == mapping and
         run.get('policy_safety_mode') == 'inline_first_five' and
         run.get('contract_kind') == 'fresh200_warmup57' and
         type(run.get('target_updates')) is int and run['target_updates'] == 200,
         'Remote run manifest differs from exact formal identity')
    method_raw = protocol.safe_regular_bytes(base / 'preparation/formal-inline-method.json')
    need(sha(method_raw) == payload['method_sha256'], 'Remote frozen method differs')
    out = dict(schema_version=1, role='readonly_formal_progress_scalars',
        base=BASE, run_id=RUN, slurm_job_id=JOB, source_commit=COMMIT,
        source_count=len(mapping), source_map_sha256=MAP_SHA,
        runtime_source_sha256=mapping, run_manifest_provenance=run_provenance,
        collection_started_unix=started_unix, provenance_scope='exact run/job/source disk '
        'hash binding; disk identity does not prove loaded module identity',
        hourly_anchor_advanced=False)
    rows, provenance = optional_capture(base / 'logs/actual-attempts.jsonl',
        MAX_ATTEMPTS, MAX_ATTEMPT_ROWS, deadline)
    out['attempts_provenance'] = provenance
    complete_attempts = None
    if rows is not None:
        out['progress'], complete_attempts = validate_attempts(rows)
    else:
        out['progress'] = dict(observed_driver_completed_outer=None,
            agreed_writes_at_last_return_per_rank=None,
            actual_total_writes_unavailable=True,
            scope='driver evidence unavailable; no zero inference; optimizer ledger suffix writes unavailable')
    updates, provenance = optional_capture(base / 'logs/actual-updates.jsonl',
        MAX_UPDATES, 200, deadline)
    out['updates_provenance'] = provenance
    if updates is not None and complete_attempts is not None:
        out['metrics'] = validate_updates(updates, complete_attempts)
    else:
        out['metrics'] = dict(last_metrics_completed_step=None,
            latest_metrics=None, metrics_state='unavailable_or_driver_unbound')
    out['scheduler_crosscheck'] = []
    terminals_by_rank = []
    for rank in range(8):
        scheduler, provenance = optional_capture(base / ('logs/scheduler-rank-%d.jsonl' % rank),
            MAX_SCHEDULER, MAX_SCHEDULER_ROWS, deadline)
        if scheduler is not None and rows is not None:
            summary, terminals = inspect_scheduler_rows(scheduler, rank, rows)
            out['scheduler_crosscheck'].append(dict(summary, provenance=provenance))
            terminals_by_rank.append(terminals)
        else:
            out['scheduler_crosscheck'].append(dict(rank=rank, provenance=provenance,
                evidence_state='unavailable_or_driver_unbound'))
    if len(terminals_by_rank) == 8:
        for left in terminals_by_rank:
            for right in terminals_by_rank:
                lhs = {row['attempt_id']: row for row in left}
                rhs = {row['attempt_id']: row for row in right}
                need(all(lhs[key] == rhs[key] for key in lhs.keys() & rhs.keys()),
                     'Fatal eight-rank scheduler disagreement')
    states = [row.get('latest_rank_receipt_state') for row in out['scheduler_crosscheck']]
    if all(state is not None for state in states) and all(state == states[0] for state in states):
        out['agreed_latest_scheduler_receipt_state'] = dict(evidence_state='observed',
            state=states[0], scope='actual latest newline-complete receipt state agrees '
            'across eight ranks; consumed LR/warmup values come from receipts; '
            'a suffix COMPLETE is not a global driver-completed outer')
    else:
        out['agreed_latest_scheduler_receipt_state'] = dict(evidence_state='unavailable_or_inflight_disagreement',
            state=None, scope='no inference from driver completed_outer or console-rounded LR')
    try:
        stage, provenance = capture_json(base / 'logs/pipeline-status.json', deadline)
        out['pipeline'] = dict(stage_summary(stage), provenance=provenance)
    except FileNotFoundError as error:
        out['pipeline'] = unavailable(error, missing=True)
    out['development_evaluations'] = []
    for step, name in ((0, 'before'), (5, 'after')):
        try:
            evaluation, provenance = capture_json(base / ('logs/formal-inline-development-' + name + '.json'), deadline)
            out['development_evaluations'].append(dict(evaluation_summary(evaluation, step), provenance=provenance))
        except FileNotFoundError as error:
            out['development_evaluations'].append(dict(step=step, **unavailable(error, missing=True)))
    try:
        allocation, provenance = capture_json(base / 'logs/native-allocation-admission.json', deadline)
        inventory = allocation_inventory(allocation)
        out['gpu_allocation_provenance'] = provenance
        out['current_gpu'] = current_gpu(inventory, deadline, protocol.bounded_capture)
        samples, provenance = optional_capture(base / ('logs/gpu-samples-' + RUN + '.jsonl'),
            MAX_GPU, MAX_GPU_ROWS, deadline)
        out['gpu_samples_provenance'] = provenance
        if samples is not None:
            out['gpu_samples'] = validate_gpu_samples(samples, inventory, now=time.time())
            if out['gpu_samples']['sampler_identity'] is not None:
                out['gpu_sampler_current_birth'] = actual_birth(out['gpu_samples']['sampler_identity'], deadline)
                birth = out['gpu_sampler_current_birth']
                need(birth['birth_matches'] is not False, 'Sampler PID was reused or birth differs')
                need(birth['namespace_matches'] is not False, 'Sampler process belongs to another run/job')
        else:
            out['gpu_samples'] = dict(sample_state='unavailable', cumulative_sampled_peak_mib=None)
    except FileNotFoundError as error:
        out['gpu_allocation_provenance'] = unavailable(error, missing=True)
        out['current_gpu'] = dict(evidence_state='unavailable', reason_code='allocation_unbound')
        out['gpu_samples'] = dict(sample_state='unavailable', cumulative_sampled_peak_mib=None)
    # Only externally observed exact namespace births are queried; no census,
    # launch, signals, scheduler changes, cleanup, logs or private journals.
    births = payload.get('observed_namespace_births', [])
    need(type(births) is list and len(births) <= 256, 'Namespace birth count bound exceeded')
    out['current_namespace_births'] = [actual_birth(row, deadline) for row in births]
    deadline_check(deadline)
    out['collection_finished_unix'] = time.time()
    out['observation_incomplete'] = any(out[name].get('observation_incomplete', False)
        for name in ('attempts_provenance', 'updates_provenance', 'gpu_samples_provenance') if name in out)
    out['observation_incomplete'] = out['observation_incomplete'] or any(
        row['provenance'].get('observation_incomplete', False)
        for row in out['scheduler_crosscheck'])
    out['raw_export_policy'] = 'scalar/hash/GPU/birth/evaluation summaries only; no raw QA, token journals, or full training logs'
    return out
