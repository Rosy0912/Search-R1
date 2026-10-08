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
MAX_PROCESS_LEDGER = 16 * 1024 * 1024
MAX_PROCESS_COMMAND = 64 * 1024
MAX_ATTEMPTS = 8 * 1024 * 1024
MAX_UPDATES = 2 * 1024 * 1024
MAX_GPU = 128 * 1024 * 1024
MAX_SCHEDULER = 16 * 1024 * 1024
MAX_LINE = 256 * 1024
MAX_ATTEMPT_ROWS = 20000
MAX_GPU_ROWS = 100000
MAX_SCHEDULER_ROWS = 60000
MAX_PROCESS_BIRTHS = 4096
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


class CollectionDeadlineExceeded(ObservationError):
    pass


class MetricGateDisagreement(ObservationError):
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
        latest_behavior_probability_gate_sha256=latest['behavior_probability_gate_sha256'] if latest else None,
        metrics_state='available' if rows else 'no_completed_metrics_rows',
        scope='full precision actual-updates scalar prefix; absence may be valid driver-to-metrics lag')


def process_identity(row):
    keys(row, ('pid', 'start_ticks', 'sid', 'uid'), 'process birth')
    for name in ('pid', 'start_ticks', 'sid'):
        integer(row[name], 'process ' + name, 1)
    integer(row['uid'], 'process uid')
    return row


def validate_namespace_births(rows):
    """One finite bound shared by the local wrapper and remote reader."""
    need(type(rows) is list and len(rows) <= MAX_PROCESS_BIRTHS,
         'Namespace birth count bound exceeded')
    seen = set()
    for row in rows:
        process_identity(row)
        identity = tuple(row[name] for name in ('pid', 'start_ticks', 'sid', 'uid'))
        need(identity not in seen, 'Namespace birth duplicated')
        seen.add(identity)
    return rows


def namespace_birth_component(rows):
    """Auxiliary inventory over budget cannot erase independent driver evidence."""
    try:
        return validate_namespace_births(rows), dict(evidence_state='observed',
            observation_incomplete=False)
    except ObservationError as error:
        return [], unavailable(error)


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


def scheduler_contract_sha256(budget):
    """Match scheduler._canonical exactly, including its default ASCII encoding."""
    need(type(budget) is dict, 'Manifest scheduler budget unavailable')
    for name, value in dict(target_actual_outer_updates=200, lr_warmup_steps=57,
                           world_size=8, optimizer_steps_per_outer=20).items():
        need(type(budget.get(name)) is int and budget[name] == value,
             'Manifest scheduler budget differs: ' + name)
    need(finite(budget.get('learning_rate'), 'manifest base LR') == 1e-6 and
         finite(budget.get('lr_warmup_steps_ratio'), 'manifest warmup ratio') == 0,
         'Manifest scheduler LR budget differs')
    return sha(json.dumps(budget, sort_keys=True, separators=(',', ':'),
                          allow_nan=False).encode())


def historical_identity(row):
    """Validate owned_processes identity; export only four bounded birth scalars."""
    keys(row, ('pid', 'start_ticks', 'sid', 'uid', 'state', 'command'),
         'historical owned process')
    identity = {name: row[name] for name in ('pid', 'start_ticks', 'sid', 'uid')}
    process_identity(identity)
    need(type(row['state']) is str and len(row['state']) == 1 and
         type(row['command']) is str and len(row['command'].encode('utf-8')) <= MAX_PROCESS_COMMAND,
         'Historical process metadata bound differs')
    return identity


def validate_training_binding(claim, ledger, *, run_started_unix, now):
    """Authenticate a durable claim against the independently admitted trainer.

    The frozen run/source manifest is authenticated by collect_remote first.
    This historical join does not assert that any saved process is still live.
    Raw commands and other jobs' ledgers are never returned.
    """
    keys(claim, ('run_id', 'shell_pid', 'shell_start_ticks', 'slurm_job_id',
                 'claimed_unix', 'semantics'), 'training launch claim')
    receipt_identity(claim)
    integer(claim['shell_pid'], 'claimed shell PID', 1)
    integer(claim['shell_start_ticks'], 'claimed shell start ticks', 1)
    claimed = finite(claim['claimed_unix'], 'training claim time', 0)
    finite(run_started_unix, 'authenticated run start', 0)
    finite(now, 'binding observation time', 0)
    need(0 < run_started_unix <= claimed <= now + 1 and
         claim['semantics'] == 'exclusive launch claim, not evidence of a completed update',
         'Training claim timeline/semantics differs')
    keys(ledger, ('run_id', 'jobs'), 'historical process ledger')
    receipt_identity(ledger, job=False)
    jobs = ledger['jobs']
    need(type(jobs) is dict and len(jobs) <= 64 and 'trainer' in jobs,
         'Historical trainer job unavailable/bound exceeded')
    trainer = jobs['trainer']
    need(type(trainer) is dict and trainer.get('arguments') ==
         ['bash', 'reproduction/common8/train.sh'] and
         trainer.get('log') == BASE + '/logs/trainer.log' and
         trainer.get('state') in ('STARTED', 'STOPPED'),
         'Historical trainer source launch differs')
    leader = historical_identity(trainer.get('leader'))
    need(leader['uid'] == os.getuid() and leader['sid'] == leader['pid'] and
         (leader['pid'], leader['start_ticks']) ==
         (claim['shell_pid'], claim['shell_start_ticks']),
         'Training claim/trainer leader birth differs')
    recorded = trainer.get('ledger')
    need(type(recorded) is dict and 1 <= len(recorded) <= MAX_PROCESS_BIRTHS,
         'Historical trainer birth count bound exceeded')
    admitted = {}
    for pid_key, row in recorded.items():
        identity = historical_identity(row)
        need(type(pid_key) is str and pid_key == str(identity['pid']) and
             identity['sid'] == leader['sid'] and identity['uid'] == leader['uid'],
             'Historical trainer member/session differs')
        birth = (identity['pid'], identity['start_ticks'])
        need(birth not in admitted, 'Historical trainer birth duplicated')
        admitted[birth] = identity
    need(admitted.get((leader['pid'], leader['start_ticks'])) == leader,
         'Trainer leader absent from historical ledger')
    return dict(evidence_state='observed', claimed_unix=claimed,
        run_started_unix=run_started_unix, trainer_leader_birth=leader,
        admitted_trainer_birth_count=len(admitted), recorded_trainer_state=trainer['state'],
        scope='exclusive training claim bound to frozen run/job/source and '
              'historically admitted trainer births; boundary includes trainer '
              'initialization, generation and optimizer writes; no current liveness claim'), admitted


def validate_gpu_samples(rows, inventory, *, now, run_started_unix, training_claim_unix):
    finite(now, 'observation clock', 0)
    finite(run_started_unix, 'authenticated run start', 0)
    need(0 < run_started_unix <= now + 1, 'Authenticated run start differs')
    if training_claim_unix is not None:
        finite(training_claim_unix, 'authenticated training boundary', 0)
        need(run_started_unix <= training_claim_unix <= now + 1,
             'Authenticated training boundary differs')
    peaks = {str(index): 0 for index in inventory}
    previous = None
    sampler = None
    intervals = []
    gaps = []
    window_rows = {name: [] for name in ('setup', 'trainer_window', 'boundary_straddling')}
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
        need(run_started_unix <= started <= ended <= now + 1 and ended - started <= 31,
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
        if training_claim_unix is not None:
            if started >= training_claim_unix:
                window_rows['trainer_window'].append(row)
            elif ended <= training_claim_unix:
                window_rows['setup'].append(row)
            else:
                window_rows['boundary_straddling'].append(row)
        previous = ended
    windows = {}
    for name, selected in window_rows.items():
        sampled_peaks = {str(index): max(gpu['used_mib'] for row in selected
            for gpu in row['gpus'] if gpu['index'] == index) for index in inventory} if selected else None
        windows[name] = dict(sample_count=len(selected),
            query_started_unix=selected[0]['query_started_unix'] if selected else None,
            query_finished_unix=selected[-1]['unix'] if selected else None,
            sampled_peak_mib=sampled_peaks,
            evidence_state=('observed' if selected else 'no_complete_samples')
                if training_claim_unix is not None else 'boundary_unavailable')
    windows['setup'].update(window_start_unix=run_started_unix,
                            window_end_unix=training_claim_unix)
    windows['trainer_window'].update(window_start_unix=training_claim_unix,
        window_end_unix=now if training_claim_unix is not None else None,
        scope='from authenticated launch claim, including trainer startup/model '
              'initialization, generation and optimizer writes; not an optimizer-only peak')
    windows['boundary_straddling'].update(boundary_unix=training_claim_unix,
        scope='query crosses the claim boundary; excluded from both setup and trainer-window peaks')
    return dict(sample_count=len(rows), nominal_sampling_interval_seconds=2,
        authenticated_run_started_unix=run_started_unix,
        authenticated_training_claim_unix=training_claim_unix, windows=windows,
        unclassified_sample_count=len(rows) if training_claim_unix is None else 0,
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
        observation_incomplete=(not rows or training_claim_unix is None or
            windows['setup']['evidence_state'] != 'observed' or
            windows['trainer_window']['evidence_state'] != 'observed'),
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
    if time.monotonic() > deadline:
        raise CollectionDeadlineExceeded('Read-only collection time bound exceeded')


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


def capture_json(path, deadline, *, maximum=MAX_DOCUMENT):
    raw, provenance = snapshot(path, maximum, deadline)
    provenance['observation_incomplete'] = provenance['appended_after_capture']
    return strict_json(raw), provenance


def capture_jsonl(path, maximum, rows, deadline):
    raw, provenance = snapshot(path, maximum, deadline)
    parsed, prefix = jsonl_prefix(raw, max_bytes=maximum, max_lines=rows)
    prefix['observation_incomplete'] = (prefix['observation_incomplete'] or
                                        provenance['appended_after_capture'])
    return parsed, dict(provenance, **prefix)


def unavailable(error, *, missing=False):
    # Do not export arbitrary error text containing raw source material.
    if isinstance(error, CollectionDeadlineExceeded):
        return dict(evidence_state='unavailable', reason_code='time_budget_exhausted',
                    observation_incomplete=True)
    return dict(evidence_state='unavailable' if missing else 'invalid_or_unstable',
        reason_code='missing_file' if missing else type(error).__name__,
        observation_incomplete=True)


def optional_capture(path, maximum, rows, deadline):
    try:
        deadline_check(deadline)
        values, provenance = capture_jsonl(path, maximum, rows, deadline)
        return values, dict(provenance, evidence_state='observed')
    except FileNotFoundError as error:
        return None, unavailable(error, missing=True)
    except (ObservationError, OSError) as error:
        return None, unavailable(error)


def optional_json(path, deadline, *, maximum=MAX_DOCUMENT):
    try:
        deadline_check(deadline)
        value, provenance = capture_json(path, deadline, maximum=maximum)
        return value, dict(provenance, evidence_state='observed',
                           observation_incomplete=provenance['appended_after_capture'])
    except FileNotFoundError as error:
        return None, unavailable(error, missing=True)
    except (ObservationError, OSError) as error:
        return None, unavailable(error)


def actual_birth(saved, deadline):
    # Invalid supplied identities remain strict. Deadline and /proc availability
    # belong to this auxiliary current-birth measurement only.
    process_identity(saved)
    proc = Path('/proc') / str(saved['pid'])
    def stat_bytes():
        deadline_check(deadline)
        fd = os.open(proc / 'stat', os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW |
                     getattr(os, 'O_CLOEXEC', 0))
        try:
            deadline_check(deadline)
            raw = os.read(fd, 4097)
            need(len(raw) <= 4096, 'Proc stat bound exceeded')
            return raw
        finally:
            os.close(fd)
    try:
        deadline_check(deadline)
        before = proc.stat()
        raw = stat_bytes()
        fields = raw.rsplit(b')', 1)[1].split()
        current = dict(pid=saved['pid'], start_ticks=int(fields[19]),
                       sid=int(fields[3]), uid=before.st_uid)
        state = fields[0].decode('ascii')
        deadline_check(deadline)
        env_fd = os.open(proc / 'environ', os.O_RDONLY | os.O_NONBLOCK)
        try:
            deadline_check(deadline)
            env = os.read(env_fd, 65537)
            need(len(env) <= 65536, 'Process environment byte bound exceeded')
        finally:
            os.close(env_fd)
        second = stat_bytes().rsplit(b')', 1)[1].split()
        deadline_check(deadline)
        need(int(second[19]) == current['start_ticks'] and
             proc.stat().st_uid == current['uid'], 'Process changed during birth check')
        deadline_check(deadline)
        same = current == saved
        namespace = (('SEARCH_G1_BASE=' + BASE).encode() in env.split(b'\0') and
                     ('SLURM_JOB_ID=' + JOB).encode() in env.split(b'\0'))
        return dict(saved_identity=saved, actual_identity=current, process_state=state,
                    birth_matches=same, namespace_matches=namespace if same else False,
                    evidence_state='observed', checked_unix=time.time())
    except CollectionDeadlineExceeded as error:
        return dict(unavailable(error), saved_identity=saved, actual_identity=None,
                    birth_matches=None, namespace_matches=None, evidence_state='unknown',
                    checked_unix=time.time())
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
    need(row.get('accepted') is True and row.get('node') == 'h100-8',
         'Allocation is not this admitted native node')
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
    try:
        deadline_check(deadline)
        budget = min(10, deadline - time.monotonic())
        if budget <= 0:
            raise CollectionDeadlineExceeded('No current GPU query time remains')
        started = time.time()
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
        deadline_check(deadline)
        return dict(evidence_state='observed', query_started_unix=started,
            query_finished_unix=time.time(), stdout_sha256=sha(completed.stdout.encode()),
            gpus=values, scope='current physical node inventory; not attribution of all occupancy to this run')
    except RuntimeError as error:
        return dict(unavailable(error), evidence_state='unavailable')
    except (OSError, subprocess.TimeoutExpired, UnicodeDecodeError, ValueError) as error:
        return unavailable(error)


def stage_summary(row):
    receipt_identity(row, job=False)
    need(type(row.get('stage')) is str and 0 < len(row['stage']) <= 256,
         'Malformed stage receipt')
    finite(row.get('unix'), 'stage time', 0)
    return dict(evidence_state='observed', stage=row['stage'], unix=row['unix'],
                scope='reported pipeline stage; no completion or success inference')


def evaluation_summary(row, expected_step):
    receipt_identity(row, job=False)
    required = ('schema_version', 'role', 'run_id', 'step', 'policy_safety_mode',
        'source_map_sha256', 'plan_sha256', 'validation_sha256', 'checked_unix',
        'weighted_em', 'weighted_f1')
    # The producer also writes raw rows/IDs and other aggregates. Require the
    # selected schema without rejecting or exporting those legitimate extras.
    need(all(name in row for name in required), 'Required evaluation field absent')
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
    return dict(evidence_state='observed', step=expected_step, checked_unix=row['checked_unix'],
        plan_sha256=row['plan_sha256'], validation_sha256=row['validation_sha256'],
        metrics=metrics, scope='observed paired dev512 aggregate scalars; raw rows/IDs/answers not exported')


def inspect_scheduler_rows(rows, rank, attempts, *, contract_sha256,
                           admitted_births, run_started_unix, training_claim_unix,
                           metric_gates=None):
    """Crosscheck existing driver receipts; suffix never advances global count.

    Missing/partial rank files leave a separate crosscheck unavailable. Failure
    records or conflicting receipts invalidate that crosscheck; the caller
    retains the independently valid driver prefix.
    """
    expected = {}
    for attempt in attempts:
        expected[attempt['attempt_id']] = attempt
    groups = {}
    suffix_count = 0
    saved_birth = None
    previous_unix = run_started_unix
    hash_value(contract_sha256)
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
            integer(row.get(name), name, 1)
        hash_value(row.get('scheduler_contract_sha256'))
        need(row['scheduler_contract_sha256'] == contract_sha256,
             'Scheduler contract differs from actual manifest budget')
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
        if admitted_births is not None:
            need(birth in admitted_births,
                 'Scheduler birth absent from historical admitted trainer ledger')
        need(type(row['warmup_steps']) is int and row['warmup_steps'] == 57 and
             finite(row['base_learning_rate'], 'scheduler base LR') == 1e-6,
             'Scheduler fixed warmup/LR differs')
        observed_unix = finite(row['observed_unix'], 'scheduler receipt time', 0)
        need(observed_unix >= previous_unix and
             (training_claim_unix is None or observed_unix >= training_claim_unix),
             'Scheduler receipt predates authenticated run/claim or is reordered')
        previous_unix = observed_unix
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
        if metric_gates is not None and row['attempt_id'] in metric_gates:
            if row['batch_gate_sha256'] != metric_gates[row['attempt_id']]:
                raise MetricGateDisagreement('Visible same-attempt scheduler/metrics gate differs')
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
            driver_outer = attempts[-1]['completed_outer'] if attempts else 0
            driver_writes = attempts[-1]['optimizer_writes_after'] if attempts else 0
            need(driver_outer < 200 and row['attempt_index'] == len(attempts) + 1 and
                 row['logical_update'] == driver_outer + 1 and
                 row['completed_outer_before'] == driver_outer and
                 row['writes_before'] == driver_writes,
                 'Scheduler suffix start differs from driver prefix or target boundary')
            suffix_count += 1
            continue
        groups.setdefault(row['attempt_id'], []).append(row)
    missing = 0
    terminals = []
    for attempt in attempts:
        selected = groups.get(attempt['attempt_id'], [])
        kinds = ['BEGIN', 'COMMIT_PREPARED', 'COMPLETE'] if attempt['complete_outer'] else ['BEGIN', 'ABORTED_ATTEMPT']
        before = attempt['completed_outer'] - int(attempt['complete_outer'])
        # Authenticate every visible known row before allowing a shorter prefix.
        # BEGIN has no writes; prepared/terminal rows contain the agreed return.
        for row in selected:
            beginning = row['kind'] == 'BEGIN'
            advanced = row['kind'] == 'COMPLETE'
            for name, value in dict(logical_update=attempt['logical_update'],
                attempt_index=attempt['attempt_index'],
                writes_before=attempt['optimizer_writes_before'],
                writes_after=attempt['optimizer_writes_before'] if beginning else attempt['optimizer_writes_after'],
                actual_optimizer_writes=0 if beginning else attempt['optimizer_writes_this_attempt'],
                opportunities_processed=0 if beginning else 20, optimizer_opportunities=20,
                completed_outer_before=before, completed_outer_after=before + int(advanced),
                scheduler_advances=int(advanced), scheduler_last_epoch_before=before,
                scheduler_last_epoch_after=before + int(advanced),
                scheduler_step_count_before=before + 1,
                scheduler_step_count_after=before + int(advanced) + 1).items():
                need(type(row.get(name)) is int and row[name] == value,
                     'Fatal rank/driver visible-row disagreement: ' + name)
            need(row['attempt_id'] == attempt['attempt_id'], 'Scheduler UUID differs')
        if [row['kind'] for row in selected] != kinds:
            need([row['kind'] for row in selected] == kinds[:len(selected)],
                 'Scheduler prefix duplicated/reordered')
            missing += 1
            continue
        terminal = selected[-1]
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
                 row['scheduler_contract_sha256'] == selected[0]['scheduler_contract_sha256'] and
                 row['writes_before'] == selected[0]['writes_before'] and
                 row['completed_outer_before'] == selected[0]['completed_outer_before']
                 for row in selected), 'Scheduler group shared state differs')
        if kinds == ['BEGIN', 'COMMIT_PREPARED', 'COMPLETE']:
            prepared, complete = selected[1:]
            need(all(prepared[name] == complete[name] for name in (
                'logical_update', 'attempt_id', 'attempt_index', 'writes_before',
                'writes_after', 'actual_optimizer_writes', 'opportunities_processed',
                'optimizer_opportunities', 'consumed_learning_rates',
                'completed_outer_before', 'scheduler_last_epoch_before',
                'scheduler_step_count_before')),
                'Scheduler COMPLETE does not copy the prepared opportunity/write state')
    latest = rows[-1] if rows else None
    observed_state = {name: latest[name] for name in (
        'kind', 'attempt_id', 'attempt_index', 'logical_update',
        'consumed_learning_rates', 'next_learning_rates',
        'completed_outer_before', 'completed_outer_after',
        'scheduler_last_epoch_before', 'scheduler_last_epoch_after',
        'scheduler_step_count_before', 'scheduler_step_count_after',
        'writes_before', 'writes_after', 'actual_optimizer_writes',
        'opportunities_processed', 'scheduler_advances',
        'batch_gate_sha256', 'scheduler_contract_sha256')} if latest else None
    bound = saved_birth is not None and admitted_births is not None and training_claim_unix is not None
    return dict(rank=rank, matched_driver_attempts=len(terminals),
        evidence_state='observed' if bound else 'unavailable_or_birth_unbound',
        claimed_process_birth=dict(pid=saved_birth[0], start_ticks=saved_birth[1]) if saved_birth else None,
        historical_process_birth=admitted_births[saved_birth] if bound else None,
        historical_birth_binding_state='observed' if bound else 'unavailable',
        unavailable_driver_attempts=missing, uncommitted_suffix_rows=suffix_count,
        visible_metric_gate_attempts_checked=len(set(every_group) & set(metric_gates or {})),
        last_driver_matched_state=terminals[-1] if terminals else None,
        latest_rank_receipt_state=observed_state,
        scope='rank receipt crosscheck only; suffix rows never increase global completion'), terminals


def summarize_scheduler_crosscheck(summaries, terminals_by_rank):
    """All visible rank evidence agrees, including suffix input provenance."""
    need(type(summaries) is list and len(summaries) == 8,
         'Need eight scheduler component states')
    claimed = [row.get('claimed_process_birth') for row in summaries]
    births = [(row['pid'], row['start_ticks']) for row in claimed if row is not None]
    need(len(set(births)) == len(births),
         'Different scheduler ranks claim the same process birth')
    if len(terminals_by_rank) == 8:
        for left in terminals_by_rank:
            for right in terminals_by_rank:
                lhs = {row['attempt_id']: row for row in left}
                rhs = {row['attempt_id']: row for row in right}
                need(all(lhs[key] == rhs[key] for key in lhs.keys() & rhs.keys()),
                     'Fatal eight-rank scheduler disagreement')
    states = [row.get('latest_rank_receipt_state') for row in summaries]
    bound = all(row.get('historical_birth_binding_state') == 'observed' for row in summaries)
    present = all(state is not None for state in states)
    agrees = present and all(state == states[0] for state in states)
    if bound and agrees:
        return dict(evidence_state='observed', state=states[0],
            scope='historically bound latest complete receipt state agrees across eight '
                  'distinct ranks, including gate and manifest budget provenance; '
                  'a suffix COMPLETE is not a driver-completed outer or a liveness claim')
    return dict(evidence_state='inflight_disagreement' if present and not agrees else 'unavailable_or_birth_unbound',
        state=None, scope='no eight-rank inference from unbound/partial/disagreeing receipts')


def observation_completeness(out):
    """Completeness is capture availability, never a training acceptance gate."""
    components = {}
    def observed(value):
        return (type(value) is dict and
                value.get('evidence_state', 'observed' if 'captured_sha256' in value else None) == 'observed' and
                not value.get('appended_after_capture', False) and
                not value.get('observation_incomplete', False))
    def add(name, complete, not_yet=False):
        components[name] = dict(status='not_yet_applicable' if not_yet else
            ('complete' if complete else 'incomplete'), capture_complete=bool(complete or not_yet))
    def birth_ok(value):
        return observed(value) and value.get('birth_matches') is True and \
            value.get('namespace_matches') is True and value.get('process_state') != 'Z'
    add('run_manifest', observed(out.get('run_manifest_provenance')))
    add('driver_attempts', observed(out.get('attempts_provenance')) and
        out.get('progress', {}).get('observed_driver_completed_outer') is not None)
    metrics = out.get('metrics', {})
    completed_driver = out.get('progress', {}).get('observed_driver_completed_outer')
    add('update_metrics', observed(out.get('updates_provenance')) and
        metrics.get('last_metrics_completed_step') is not None and
        metrics.get('metrics_lag_completed_outer') == 0 and
        (metrics.get('latest_metrics') is not None or completed_driver == 0))
    crosscheck = out.get('scheduler_crosscheck', [])
    for rank in range(8):
        selected = [row for row in crosscheck if row.get('rank') == rank]
        row = selected[0] if len(selected) == 1 else {}
        add('scheduler_rank_%d' % rank, observed(row.get('provenance')) and
            observed(row) and row.get('unavailable_driver_attempts') == 0)
    add('latest_scheduler_agreement', observed(out.get('agreed_latest_scheduler_receipt_state')))
    add('training_launch_claim', observed(out.get('training_claim_provenance')))
    add('historical_trainer_ledger', observed(out.get('trainer_ledger_provenance')))
    add('authenticated_training_boundary', observed(out.get('training_boundary')))
    add('pipeline', observed(out.get('pipeline')) and
        observed(out.get('pipeline', {}).get('provenance')))
    evaluations = out.get('development_evaluations', [])
    completed = out.get('progress', {}).get('observed_driver_completed_outer')
    for step in (0, 5):
        selected = [row for row in evaluations if row.get('step') == step]
        row = selected[0] if len(selected) == 1 else {}
        not_yet = (step == 5 and type(completed) is int and completed < 5 and
                   row.get('evidence_state') == 'not_yet_applicable')
        add('development_evaluation_%d' % step, observed(row) and
            observed(row.get('provenance')), not_yet=not_yet)
    add('gpu_allocation', observed(out.get('gpu_allocation_provenance')))
    add('current_gpu', observed(out.get('current_gpu')))
    samples = out.get('gpu_samples', {})
    add('gpu_sample_prefix', observed(out.get('gpu_samples_provenance')) and
        samples.get('sample_state') == 'available' and not samples.get('observation_incomplete', True))
    for name in ('setup', 'trainer_window'):
        add('gpu_' + name, observed(samples.get('windows', {}).get(name)))
    add('gpu_sampler_current_birth', birth_ok(out.get('gpu_sampler_current_birth')))
    add('trainer_leader_current_birth', birth_ok(out.get('trainer_leader_current_birth')))
    scheduler_births = out.get('scheduler_current_births', [])
    add('scheduler_current_births', len(scheduler_births) == 8 and all(birth_ok(row) for row in scheduler_births))
    namespace = out.get('current_namespace_births', [])
    add('current_namespace_births', bool(namespace) and all(birth_ok(row) for row in namespace))
    add('collection_time_budget', observed(out.get('collection_time_budget')))
    complete = all(row['capture_complete'] for row in components.values())
    return dict(components=components, overall_capture_complete=complete,
        observation_incomplete=not complete,
        scope='requested scalar evidence capture only; valid driver completion '
              'prefix remains usable when another component is incomplete; no acceptance inference')


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
    run_started = finite(run.get('started_unix'), 'authenticated manifest run start', 0)
    need(0 < run_started <= started_unix + 1, 'Manifest run start differs')
    contract_sha = scheduler_contract_sha256(run.get('budget'))
    method_raw = protocol.safe_regular_bytes(base / 'preparation/formal-inline-method.json')
    need(sha(method_raw) == payload['method_sha256'], 'Remote frozen method differs')
    out = dict(schema_version=1, role='readonly_formal_progress_scalars',
        base=BASE, run_id=RUN, slurm_job_id=JOB, source_commit=COMMIT,
        source_count=len(mapping), source_map_sha256=MAP_SHA,
        runtime_source_sha256=mapping, run_manifest_provenance=run_provenance,
        authenticated_run_started_unix=run_started,
        actual_manifest_scheduler_contract_sha256=contract_sha,
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
        try:
            out['metrics'] = validate_updates(updates, complete_attempts)
        except ObservationError as error:
            out['metrics'] = dict(unavailable(error), last_metrics_completed_step=None,
                latest_metrics=None, metrics_state='invalid_or_driver_disagreement',
                scope='invalid metrics crosscheck; valid driver completion prefix retained')
    else:
        out['metrics'] = dict(last_metrics_completed_step=None,
            latest_metrics=None, metrics_state='unavailable_or_driver_unbound')
    metric_gates = {row['attempt_id']: row['behavior_probability_gate_sha256'] for row in updates} \
        if updates is not None and out['metrics'].get('last_metrics_completed_step') is not None else None
    metric_gate_disagreement = False
    claim, out['training_claim_provenance'] = optional_json(base / 'logs/training-launch-claim.json', deadline)
    ledger, out['trainer_ledger_provenance'] = optional_json(base / 'logs/process-ledger.json', deadline,
        maximum=MAX_PROCESS_LEDGER)
    admitted_births = None
    out['training_boundary'] = dict(evidence_state='unavailable', claimed_unix=None,
        reason_code='claim_or_historical_trainer_unavailable', observation_incomplete=True)
    if claim is not None and ledger is not None:
        try:
            out['training_boundary'], admitted_births = validate_training_binding(claim, ledger,
                run_started_unix=run_started, now=time.time())
        except ObservationError as error:
            out['training_boundary'] = dict(unavailable(error), claimed_unix=None)
    claimed_unix = out['training_boundary'].get('claimed_unix')
    if admitted_births is not None:
        out['trainer_leader_current_birth'] = actual_birth(out['training_boundary']['trainer_leader_birth'], deadline)
    out['scheduler_crosscheck'] = []
    terminals_by_rank = []
    for rank in range(8):
        scheduler, provenance = optional_capture(base / ('logs/scheduler-rank-%d.jsonl' % rank),
            MAX_SCHEDULER, MAX_SCHEDULER_ROWS, deadline)
        if scheduler is not None and rows is not None:
            try:
                summary, terminals = inspect_scheduler_rows(scheduler, rank, rows,
                    contract_sha256=contract_sha, admitted_births=admitted_births,
                    run_started_unix=run_started, training_claim_unix=claimed_unix,
                    metric_gates=metric_gates)
                out['scheduler_crosscheck'].append(dict(summary, provenance=provenance))
                terminals_by_rank.append(terminals)
            except ObservationError as error:
                metric_gate_disagreement = metric_gate_disagreement or isinstance(error, MetricGateDisagreement)
                out['scheduler_crosscheck'].append(dict(unavailable(error), rank=rank,
                    provenance=provenance, scope='invalid scheduler crosscheck; valid driver prefix retained'))
        else:
            out['scheduler_crosscheck'].append(dict(rank=rank, provenance=provenance,
                evidence_state='unavailable_or_driver_unbound'))
    if metric_gate_disagreement:
        out['metrics'].update(latest_metrics=None, metrics_state='invalid_scheduler_gate_binding',
            scheduler_gate_binding_state='invalid',
            scope='visible same-attempt scheduler gate contradicts metrics; metric values '
                  'suppressed, independently valid driver prefix retained')
    else:
        gate_bound = metric_gates is not None and all(
            row.get('evidence_state') == 'observed' and
            row.get('visible_metric_gate_attempts_checked') == len(metric_gates)
            for row in out['scheduler_crosscheck'])
        out['metrics']['scheduler_gate_binding_state'] = \
            'observed_all_eight_ranks' if gate_bound else 'unavailable_or_partial'
    try:
        out['agreed_latest_scheduler_receipt_state'] = summarize_scheduler_crosscheck(
            out['scheduler_crosscheck'], terminals_by_rank)
    except ObservationError as error:
        out['agreed_latest_scheduler_receipt_state'] = dict(unavailable(error), state=None,
            scope='invalid eight-rank crosscheck; valid driver prefix retained')
    out['scheduler_current_births'] = [actual_birth(row['historical_process_birth'], deadline)
        for row in out['scheduler_crosscheck'] if row.get('historical_process_birth') is not None]
    stage, provenance = optional_json(base / 'logs/pipeline-status.json', deadline)
    out['pipeline'] = dict(provenance)
    if stage is not None:
        try:
            out['pipeline'] = dict(stage_summary(stage), provenance=provenance)
        except ObservationError as error:
            out['pipeline'] = unavailable(error)
    out['development_evaluations'] = []
    for step, name in ((0, 'before'), (5, 'after')):
        evaluation, provenance = optional_json(base / ('logs/formal-inline-development-' + name + '.json'), deadline)
        if evaluation is not None:
            try:
                out['development_evaluations'].append(dict(evaluation_summary(evaluation, step), provenance=provenance))
            except ObservationError as error:
                out['development_evaluations'].append(dict(unavailable(error), step=step))
        else:
            completed = out['progress'].get('observed_driver_completed_outer')
            not_yet = (step == 5 and type(completed) is int and completed < 5 and
                       provenance.get('reason_code') == 'missing_file')
            out['development_evaluations'].append(dict(provenance, step=step,
                evidence_state='not_yet_applicable' if not_yet else provenance['evidence_state'],
                scope='future step5 evaluation not yet applicable' if not_yet else 'requested evaluation unavailable'))
    allocation, provenance = optional_json(base / 'logs/native-allocation-admission.json', deadline)
    out['gpu_allocation_provenance'] = provenance
    inventory = None
    if allocation is not None:
        try:
            inventory = allocation_inventory(allocation)
        except (ObservationError, ValueError, KeyError) as error:
            out['gpu_allocation_provenance'] = unavailable(error)
    if inventory is not None:
        out['current_gpu'] = current_gpu(inventory, deadline, protocol.bounded_capture)
        samples, provenance = optional_capture(base / ('logs/gpu-samples-' + RUN + '.jsonl'),
            MAX_GPU, MAX_GPU_ROWS, deadline)
        out['gpu_samples_provenance'] = provenance
        if samples is not None:
            try:
                out['gpu_samples'] = validate_gpu_samples(samples, inventory, now=time.time(),
                    run_started_unix=run_started, training_claim_unix=claimed_unix)
            except ObservationError as error:
                out['gpu_samples'] = dict(unavailable(error), sample_state='invalid_or_unstable',
                    cumulative_sampled_peak_mib=None)
            if out['gpu_samples'].get('sampler_identity') is not None:
                out['gpu_sampler_current_birth'] = actual_birth(out['gpu_samples']['sampler_identity'], deadline)
        else:
            out['gpu_samples'] = dict(sample_state='unavailable', cumulative_sampled_peak_mib=None)
    else:
        out['current_gpu'] = dict(evidence_state='unavailable', reason_code='allocation_unbound')
        out['gpu_samples_provenance'] = dict(evidence_state='unavailable',
            reason_code='allocation_unbound', observation_incomplete=True)
        out['gpu_samples'] = dict(sample_state='unavailable', cumulative_sampled_peak_mib=None)
    # Query only captured exact historical/namespace/sampler birth tuples above
    # and below; no census, launch, signals, cleanup, or private journals.
    births, namespace_provenance = namespace_birth_component(payload.get('observed_namespace_births', []))
    supplied_namespace = payload.get('namespace_births_provenance',
                                     dict(evidence_state='observed', observation_incomplete=False))
    if supplied_namespace.get('evidence_state') != 'observed':
        births = []
        namespace_provenance = dict(evidence_state='unavailable',
            reason_code='local_namespace_inventory_invalid_or_over_bound',
            observation_incomplete=True)
    out['namespace_births_provenance'] = namespace_provenance
    out['current_namespace_births'] = [actual_birth(row, deadline) for row in births]
    budget_complete = time.monotonic() <= deadline
    out['collection_time_budget'] = dict(
        evidence_state='observed' if budget_complete else 'unavailable',
        limit_seconds=REMOTE_SECONDS, observation_incomplete=not budget_complete,
        scope='finite read budget; expiration defers auxiliary queries and retains validated driver evidence')
    if not budget_complete:
        out['collection_time_budget']['reason_code'] = 'time_budget_exhausted'
    out['collection_finished_unix'] = time.time()
    out['capture_completeness'] = observation_completeness(out)
    out['observation_incomplete'] = out['capture_completeness']['observation_incomplete']
    out['raw_export_policy'] = 'scalar/hash/GPU/birth/evaluation summaries only; no raw QA, token journals, or full training logs'
    return out
