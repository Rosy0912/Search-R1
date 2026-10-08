"""TEXT-ONLY, UNEXECUTED bounded complete-prefix JSONL and GPU scanner.

Constant memory in total row count. The admitted sampler keeps appending its
original file; this reader never edits it. Supplied core is the sealed v8 module.
"""
import hashlib
import math
import os
import stat
import time
from pathlib import Path


def need(ok, message):
    if not ok:
        raise ValueError(message)


def validate_policy(policy, duration):
    need(type(policy) is dict and set(policy) == {'total_walltime_seconds', 'capture_allowance_seconds',
         'max_rows', 'max_bytes', 'max_line_bytes', 'max_tail_bytes', 'chunk_bytes', 'read_seconds', 'summary_bytes'},
         'Exact finite capture policy required')
    for name, value in policy.items():
        need(type(value) is int and value > 0, 'Capture policy exact positive integers required')
    need(policy['total_walltime_seconds'] == duration and policy['max_rows'] ==
         (duration + policy['capture_allowance_seconds']) // 2 + 1 and
         policy['max_bytes'] == policy['max_rows'] * (policy['max_line_bytes'] + 1) + policy['max_tail_bytes'] and
         policy['max_line_bytes'] <= 256 * 1024 and policy['max_tail_bytes'] <= policy['max_line_bytes'] and
         policy['chunk_bytes'] <= 65536 and policy['read_seconds'] <= 120 and policy['summary_bytes'] <= 2 * 1024 * 1024,
         'Capture budget not derived from exact finite horizon/nominal2s wait')
    return policy


def check_deadline(deadline):
    need(type(deadline) in (int, float) and math.isfinite(deadline) and time.monotonic() <= deadline,
         'Streaming read deadline exhausted')


def identity(item):
    return (item.st_dev, item.st_ino, item.st_mode, item.st_uid)


def scan_prefix(path, *, owner_uid, max_bytes, max_rows, max_line_bytes, max_tail_bytes,
                chunk_bytes, deadline, consume, parse, previous=None):
    """Read initial extent only, then rehash same extent without materializing it.

    Complete lines are consumed exactly once. Partial old tail may complete on
    the next capture. Returned hashes bind the bytes actually parsed. Prior
    complete prefix is rehashed in this capture, including empty prefix.
    """
    for value in (owner_uid, max_bytes, max_rows, max_line_bytes, max_tail_bytes, chunk_bytes):
        need(type(value) is int and value >= 0, 'Typed finite scanner bounds required')
    need(0 < chunk_bytes <= 65536 and max_bytes > 0 and max_rows > 0 and
         0 < max_tail_bytes <= max_line_bytes <= 256 * 1024, 'Finite nonzero scanner bounds required')
    check_deadline(deadline)
    path = Path(path)
    need(path.is_absolute() and path.resolve() == path and not path.is_symlink(), 'Canonical ordinary append file required')
    before = path.lstat()
    need(stat.S_ISREG(before.st_mode) and before.st_uid == owner_uid and 0 <= before.st_size <= max_bytes,
         'Nonordinary/unowned/over-cap append extent')
    started_unix = time.time()
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, 'O_CLOEXEC', 0))
    try:
        opened = os.fstat(fd)
        need(identity(opened) == identity(before) and opened.st_size >= before.st_size, 'Append inode changed before open')
        size = before.st_size
        if previous is not None:
            need(type(previous) is dict and all(previous[key] == value for key, value in
                 dict(path=str(path), dev=opened.st_dev, inode=opened.st_ino, mode=opened.st_mode, owner_uid=owner_uid).items()) and
                 type(previous['complete_prefix_bytes']) is int and 0 <= previous['complete_prefix_bytes'] <= size,
                 'Previous append inode/owner/prefix changed')
            old_size = previous['complete_prefix_bytes']
        else:
            old_size = 0
        captured = hashlib.sha256(); complete = hashlib.sha256(); old = hashlib.sha256()
        pending = b''; offset = 0; complete_bytes = 0; rows = 0
        while offset < size:
            check_deadline(deadline)
            block = os.read(fd, min(chunk_bytes, size - offset))
            need(block, 'Append extent truncated while scanning')
            captured.update(block)
            if offset < old_size:
                old.update(block[:min(len(block), old_size - offset)])
            offset += len(block)
            pending += block
            while b'\n' in pending:
                check_deadline(deadline)
                line, pending = pending.split(b'\n', 1)
                need(line and len(line) <= max_line_bytes and rows < max_rows, 'Empty/over-cap complete JSONL row')
                consume(parse(line))
                complete.update(line + b'\n')
                complete_bytes += len(line) + 1; rows += 1
            need(len(pending) <= max_line_bytes, 'Overlong partial JSONL line')
        need(len(pending) <= max_tail_bytes, 'Partial-tail byte bound exceeded')
        if previous is not None:
            need(old.hexdigest() == previous['complete_prefix_sha256'], 'Original complete-prefix content changed')
        # A second bounded pass catches equal-size rewrite, and rewrite combined
        # with growth, of the same extent whose lines supplied the summary.
        os.lseek(fd, 0, os.SEEK_SET)
        verified = hashlib.sha256(); offset = 0
        while offset < size:
            check_deadline(deadline)
            block = os.read(fd, min(chunk_bytes, size - offset))
            need(block, 'Append extent truncated while verifying')
            verified.update(block); offset += len(block)
        need(verified.digest() == captured.digest(), 'Captured extent rewritten during scan')
        after, current = os.fstat(fd), path.lstat()
        need(identity(after) == identity(current) == identity(opened) and
             after.st_size >= size and current.st_size >= size, 'Append replaced/truncated during scan')
        grew = current.st_size > size
        need(grew or (before.st_size == current.st_size and before.st_mtime_ns == current.st_mtime_ns and
                      before.st_ctime_ns == current.st_ctime_ns), 'Same-size append metadata changed')
        check_deadline(deadline)
        provenance = dict(path=str(path), dev=opened.st_dev, inode=opened.st_ino, mode=opened.st_mode,
            owner_uid=opened.st_uid, capture_started_size=size, captured_bytes=size,
            captured_sha256=captured.hexdigest(), complete_prefix_bytes=complete_bytes,
            complete_prefix_sha256=complete.hexdigest(), complete_rows=rows,
            partial_tail_bytes=len(pending), partial_tail_sha256=hashlib.sha256(pending).hexdigest() if pending else None,
            observed_size_after=current.st_size, appended_after_capture=grew,
            capture_started_unix=started_unix, capture_finished_unix=time.time(), evidence_state='observed',
            observation_incomplete=bool(pending) or grew, verified_second_pass=True)
        recheck = dict(prefix_bytes=old_size, sha256=old.hexdigest()) if previous is not None else None
        return provenance, recheck
    finally:
        os.close(fd)


class GPUAccumulator:
    """No raw rows/window-row arrays/interval arrays survive consumption."""
    def __init__(self, core, inventory, *, now, run_started_unix, training_claim_unix, expected_sampler):
        self.core, self.inventory = core, inventory
        self.now = core.finite(now, 'observation clock', 0)
        self.run_started = core.finite(run_started_unix, 'authenticated run start', 0)
        need(0 < self.run_started <= self.now + 1, 'Authenticated run start differs')
        self.boundary = training_claim_unix
        if self.boundary is not None:
            core.finite(self.boundary, 'authenticated training boundary', 0)
            need(self.run_started <= self.boundary <= self.now + 1, 'Authenticated claim differs')
        self.expected_sampler = core.process_identity(expected_sampler)
        self.count = 0; self.first = None; self.last = None; self.latest = None
        self.peaks = {str(index): 0 for index in inventory}
        self.interval_min = None; self.interval_max = None; self.gap_count = 0; self.largest_gap = None
        self.examples = []; self.sampler = None
        self.windows = {name: dict(sample_count=0, query_started_unix=None, query_finished_unix=None,
            sampled_peak_mib=None) for name in ('setup', 'trainer_window', 'boundary_straddling')}

    def consume(self, row):
        c = self.core
        c.keys(row, c.GPU_FIELDS, 'GPU sample'); c.receipt_identity(row)
        need(type(row['schema_version']) is int and row['schema_version'] == 1 and
             row['role'] == 'actual_common8_gpu_occupancy' and row['source_map_sha256'] == c.MAP_SHA and
             type(row['sampling_interval_seconds']) is int and row['sampling_interval_seconds'] == 2,
             'GPU sample exact role/source/interval differs')
        sampler = c.process_identity(row['sampler_identity'])
        need(sampler == self.expected_sampler, 'GPU original sampler birth changed')
        started = c.finite(row['query_started_unix'], 'GPU query start', 0)
        ended = c.finite(row['unix'], 'GPU query end', 0)
        need(self.run_started <= started <= ended <= self.now + 1 and ended - started <= 31, 'GPU query window differs')
        if self.last is not None:
            need(started >= self.last, 'GPU query duplicated/reordered/overlapping')
            interval, gap = ended - self.last, started - self.last
            self.interval_min = interval if self.interval_min is None else min(interval, self.interval_min)
            self.interval_max = interval if self.interval_max is None else max(interval, self.interval_max)
            if gap > 3:
                self.gap_count += 1; self.largest_gap = gap if self.largest_gap is None else max(gap, self.largest_gap)
                if len(self.examples) < 20:
                    self.examples.append(dict(after_row=self.count, before_row=self.count + 1, idle_seconds=gap))
        c.validate_gpu_rows(row['gpus'], self.inventory)
        for gpu in row['gpus']:
            key = str(gpu['index']); self.peaks[key] = max(self.peaks[key], gpu['used_mib'])
        if self.boundary is not None:
            name = ('trainer_window' if started >= self.boundary else
                    'setup' if ended <= self.boundary else 'boundary_straddling')
            window = self.windows[name]
            if not window['sample_count']:
                window['query_started_unix'] = started
                window['sampled_peak_mib'] = {str(gpu['index']): gpu['used_mib'] for gpu in row['gpus']}
            else:
                for gpu in row['gpus']:
                    key = str(gpu['index']); window['sampled_peak_mib'][key] = max(window['sampled_peak_mib'][key], gpu['used_mib'])
            window['sample_count'] += 1; window['query_finished_unix'] = ended
        if self.first is None:
            self.first = started
        self.last, self.latest, self.sampler = ended, row['gpus'], sampler
        self.count += 1

    def summary(self):
        windows = {name: dict(value, evidence_state=('observed' if value['sample_count'] else 'no_complete_samples')
            if self.boundary is not None else 'boundary_unavailable') for name, value in self.windows.items()}
        windows['setup'].update(window_start_unix=self.run_started, window_end_unix=self.boundary)
        windows['trainer_window'].update(window_start_unix=self.boundary,
            window_end_unix=self.now if self.boundary is not None else None,
            scope='from authenticated launch claim, including trainer startup/model initialization, generation and optimizer writes; not an optimizer-only peak')
        windows['boundary_straddling'].update(boundary_unix=self.boundary,
            scope='query crosses the claim boundary; excluded from both setup and trainer-window peaks')
        return dict(sample_count=self.count, nominal_sampling_interval_seconds=2,
            authenticated_run_started_unix=self.run_started, authenticated_training_claim_unix=self.boundary,
            windows=windows, unclassified_sample_count=self.count if self.boundary is None else 0,
            sampling_started_unix=self.first, sampling_finished_unix=self.last,
            sampling_window_seconds=self.last - self.first if self.count else None,
            age_of_last_sample_seconds=self.now - self.last if self.count else None,
            observed_interval_min_seconds=self.interval_min, observed_interval_max_seconds=self.interval_max,
            observed_gap_count=self.gap_count, largest_idle_gap_seconds=self.largest_gap,
            gap_examples=self.examples, gap_examples_truncated=self.gap_count > 20,
            cumulative_sampled_peak_mib=self.peaks if self.count else None,
            latest_sample_gpus=self.latest, sampler_identity=self.sampler,
            sample_state='available' if self.count else 'no_complete_samples',
            observation_incomplete=(not self.count or self.boundary is None or
                windows['setup']['evidence_state'] != 'observed' or windows['trainer_window']['evidence_state'] != 'observed'),
            scope='entire captured newline-complete prefix for this run; sampled peak is not an unsampled peak or a full rollout peak')


def scan_gpu(path, core, inventory, policy, *, duration, now, run_started_unix,
             training_claim_unix, expected_sampler, deadline, previous=None):
    validate_policy(policy, duration)
    accumulator = GPUAccumulator(core, inventory, now=now, run_started_unix=run_started_unix,
        training_claim_unix=training_claim_unix, expected_sampler=expected_sampler)
    provenance, recheck = scan_prefix(path, owner_uid=expected_sampler['uid'],
        max_bytes=policy['max_bytes'], max_rows=policy['max_rows'], max_line_bytes=policy['max_line_bytes'],
        max_tail_bytes=policy['max_tail_bytes'], chunk_bytes=policy['chunk_bytes'], deadline=deadline,
        consume=accumulator.consume, parse=core.strict_json, previous=previous)
    summary = accumulator.summary()
    summary['observation_incomplete'] |= provenance['observation_incomplete']
    need(len(core.canonical(dict(summary=summary, provenance=provenance, recheck=recheck))) <= policy['summary_bytes'],
         'GPU summary export bound exceeded')
    return summary, provenance, recheck
