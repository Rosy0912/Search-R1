"""Preserve unreturned rank evidence without promoting it to a complete outer.

This is a read-only scalar audit. It neither proves process exit nor admits a
restore, and it cannot replace complete probability/monitor/checkpoint audits.
"""
import hashlib
import math
import os
import re
from pathlib import Path

from reproduction.common8.actor_probe_chain import account_segment
from reproduction.common8.policy_calibration_plan import _strict_json, document_sha256
from search_g1.actor_probe_namespace import namespace_origin


def _sealed_bytes(base, relative, *, limit=512 * 1024**2):
    path = base / relative
    assert path == path.resolve() and path.is_file() and path.stat().st_uid == os.getuid()
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        assert before.st_size <= limit, 'Bounded scalar file limit exceeded'
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            data = stream.read(before.st_size + 1)
        after = os.fstat(fd)
        assert (before.st_ino, before.st_size, before.st_mtime_ns) == (
            after.st_ino, after.st_size, after.st_mtime_ns), 'Log changed during sealed read'
        assert len(data) == before.st_size and path.stat().st_ino == before.st_ino
    finally:
        os.close(fd)
    return data, dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def _sealed_log(base, relative):
    data, metadata = _sealed_bytes(base, relative)
    end = data.rfind(b'\n') + 1
    fragment = data[end:]
    rows = [_strict_json(line) for line in data[:end].splitlines()]
    return rows, dict(metadata,
        newline_terminated_rows=len(rows), trailing_fragment_bytes=len(fragment),
        trailing_fragment_sha256=hashlib.sha256(fragment).hexdigest() if fragment else None)


def _context(row):
    # Fatal scheduler snapshots keep the pending/prepared identity separately.
    if row['kind'] in ('POST_COMMIT_FAILURE','PRE_COMMIT_EVIDENCE_FAILURE'):
        assert 'context' not in row, 'Actual fatal scheduler schema cannot override its retained identity'
        assert 'attempt_index' not in row and 'attempt_id' not in row
        candidates=[]
        for name in ('retained_pending','retained_prepared'):
            value=row.get(name)
            assert value is None or isinstance(value,dict)
            if value is not None:candidates.append(value)
        if not candidates:return row
        selected=candidates[0]
        fields=('attempt_id','attempt_index','logical_update')
        assert type(row['logical_update']) is int
        for candidate in candidates:
            assert type(candidate['attempt_index']) is int and candidate['attempt_index']>0
            assert type(candidate['logical_update']) is int and 1<=candidate['logical_update']<=200
            assert isinstance(candidate['attempt_id'],str) and re.fullmatch('[0-9a-f]{32}',candidate['attempt_id'])
        assert all(all(candidate[key]==selected[key] for key in fields) for candidate in candidates)
        assert row['logical_update']==selected['logical_update'], 'Fatal top-level logical update differs from retained identity'
        return selected
    if row['kind']=='REJECTED':
        assert 'context' not in row, 'Actual rejected scheduler schema uses its top-level attempt'
    return row.get('context') or row


def _gate(row):
    context = _context(row)
    return context.get('behavior_probability_gate_sha256', context.get('batch_gate_sha256', row.get('batch_gate_sha256')))


def _tail(rows, run, rank, cursor, outer, writes):
    selected = []; failures = []
    for row in rows:
        assert row['run_id'] == run['run_id'] and type(row['rank']) is int and row['rank'] == rank
        if row['kind'] == 'RESTORE':
            assert row is rows[0] and run.get('actor_probe_recovery') is not None
            assert row['last_attempt_index'] == namespace_origin(run)['attempt_index']
            continue
        context = _context(row)
        if row['kind'] in ('POST_COMMIT_FAILURE', 'PRE_COMMIT_EVIDENCE_FAILURE', 'REJECTED'):
            assert isinstance(row['reason'], str) and row['reason']
            if row['kind'] != 'REJECTED':
                assert row['fatal_no_automatic_resume'] is True
            failures.append(row)
        if 'attempt_index' not in context:
            assert row is rows[-1] and row['kind'] in ('POST_COMMIT_FAILURE', 'PRE_COMMIT_EVIDENCE_FAILURE')
            continue
        index = context['attempt_index']
        assert type(index) is int and index > namespace_origin(run)['attempt_index']
        if index <= cursor:
            assert not selected, 'Sealed prefix appended after an open tail'
            continue
        assert index == cursor + 1 and context['logical_update'] == outer
        assert 1 <= outer <= 200 and re.fullmatch('[0-9a-f]{32}', context['attempt_id'])
        gate = _gate(row)
        if gate is None:
            # The real OUTER_BUDGET writer records the attempt/counters, but
            # not a gate hash. Its preceding monitor prefix carries that hash.
            assert row['kind'] == 'OUTER_BUDGET' and selected and _gate(selected[0])
        else:
            assert isinstance(gate, str) and re.fullmatch('[0-9a-f]{64}', gate)
        if 'prior_optimizer_writes' in context:
            assert type(context['prior_optimizer_writes']) is int and writes <= context['prior_optimizer_writes'] <= writes + 20
        selected.append(row)
    return selected, failures


def _worker_prefix(rows, start):
    counter = start; cursor = 0; processed = 0; writes = []; signatures = []; terminal = None
    for mini in range(20):
        if cursor == len(rows): break
        pre = rows[cursor]; cursor += 1
        assert pre['kind'] == 'POLICY_MONITOR' and pre['stage'] == 'before_minibatch'
        assert pre['context']['minibatch_index'] == mini and pre['context']['prior_optimizer_writes'] == counter
        assert pre['optimizer_writes'] == counter
        if cursor == len(rows): break
        wrote = rows[cursor]['kind'] == 'OPTIMIZER_WRITE'
        if wrote:
            row = rows[cursor]; cursor += 1
            assert row['context'] == pre['context'] and type(row['write_count']) is int and row['write_count'] == counter + 1
            from reproduction.common8.formal_contract import consumed_lr
            assert row['learning_rates'] and all(type(rate) in (float, int) and
                rate == consumed_lr(pre['context']['logical_update']) for rate in row['learning_rates'])
            writes.append(dict(write_count=row['write_count'], minibatch_index=mini))
            counter += 1
            if cursor == len(rows): break
        post = rows[cursor]; cursor += 1
        assert post['kind'] == 'POLICY_MONITOR' and post['stage'] == 'after_minibatch'
        assert post['context'] == pre['context'] and post['optimizer_writes'] == counter
        if cursor == len(rows): break
        decision = rows[cursor]; cursor += 1
        assert decision['kind'] == 'OPPORTUNITY' and decision['context'] == pre['context']
        assert type(decision['opportunity_index']) is int and decision['opportunity_index'] == mini
        assert decision['processed'] is True and decision['optimizer_write_succeeded'] is wrote
        assert decision['optimizer_writes_before'] == counter - int(wrote) and decision['optimizer_writes_after'] == counter
        assert decision['status'] in (('written', 'written_with_post_guard') if wrote else ('skipped_guard',))
        processed += 1
        signatures.append(dict(minibatch_index=mini, status=decision['status'], before=counter-int(wrote), after=counter))
    if cursor < len(rows):
        assert processed == 20 and cursor == len(rows) - 1
        terminal = rows[cursor]
        assert terminal['kind'] == ('OUTER_BUDGET' if counter > start else 'ABORTED_ATTEMPT')
        assert terminal['opportunities_processed'] == 20 and terminal['optimizer_writes'] == counter - start
        assert terminal['cumulative_writes'] == counter
        cursor += 1
    assert cursor == len(rows)
    return dict(recorded_adam_writes=counter-start, recorded_write_positions=writes,
        processed_opportunities=processed, dispositions=signatures,
        terminal_kind=terminal['kind'] if terminal else None)


def read_unreturned_tail(base, run, *, retained_through):
    """Derive the driver origin from its real rows, then report rank-local tails.

    Caller must separately prove the selected checkpoint, original immutable
    source, complete retained prefix and parent trainer/session exit.
    """
    base = Path(base).resolve()
    raw, manifest_file = _sealed_bytes(base, 'logs/run-manifest.json', limit=32*1024**2)
    assert _strict_json(raw) == run
    from search_g1.actor_probe_contract import actor_probes
    assert actor_probes(run)
    assert run['training_world_size'] == 4 and run['execution_backend'] == 'ssh_local'
    origin = namespace_origin(run)
    updates, updates_file = _sealed_log(base, 'logs/actual-updates.jsonl')
    attempts, attempts_file = _sealed_log(base, 'logs/actual-attempts.jsonl')
    segment = account_segment(run['run_id'], updates, attempts, retained_through=retained_through,
        start_outer=origin['completed_outer'], start_writes=origin['actual_optimizer_writes'],
        start_attempt_index=origin['attempt_index'])
    cursor = attempts[-1]['attempt_index']; outer = segment['observed_completed_through'] + 1
    writes = segment['observed_writes']
    raw, ledger_file = _sealed_bytes(base, 'logs/process-ledger.json', limit=256*1024**2)
    ledger = _strict_json(raw)
    assert ledger['run_id'] == run['run_id']
    job = ledger['jobs']['trainer']; leader = job['leader']
    births = {(r['pid'], r['start_ticks']) for r in job['ledger'].values()
              if r['sid'] == leader['sid'] and r['uid'] == leader['uid']}
    per_rank = {}; identities = set(); process_births = set()
    files = [manifest_file, ledger_file, updates_file, attempts_file]
    source_sha = document_sha256(run['source_sha256']); budget_sha = document_sha256(run['budget'])
    for rank in range(4):
        workers, worker_file = _sealed_log(base, f'logs/optimizer-rank-{rank}.jsonl')
        schedulers, scheduler_file = _sealed_log(base, f'logs/scheduler-rank-{rank}.jsonl')
        files.extend((worker_file, scheduler_file))
        # Bind all historical scheduler rows, including the fault tail, to one
        # owned rank birth. A live current /proc identity is not asserted here.
        process = None; previous = 0.
        for row in schedulers:
            assert row['kind'] in ('RESTORE', 'BEGIN', 'COMMIT_PREPARED', 'COMPLETE', 'ABORTED_ATTEMPT',
                'REJECTED', 'POST_COMMIT_FAILURE', 'PRE_COMMIT_EVIDENCE_FAILURE')
            assert row['schema_version'] == 1 and row['role'] == 'fresh200_outer_scheduler'
            assert row['contract_kind'] == 'fresh200_warmup57' and row['warmup_steps'] == 57 and row['base_learning_rate'] == 1e-6
            assert row['run_id'] == run['run_id'] and row['source_commit'] == run['source_commit']
            assert row['source_map_sha256'] == source_sha and row['scheduler_contract_sha256'] == budget_sha
            assert row['execution_id'] == run['execution_id'] and row['execution_backend'] == 'ssh_local'
            assert row['slurm_job_id'] is None and row['hostname'] == run['execution_host']
            assert type(row['rank']) is int and row['rank'] == rank and row['world_size'] == 4
            birth = (row['process_pid'], row['process_start_ticks'])
            assert all(type(value) is int and value > 0 for value in birth)
            assert birth in births and (process is None or birth == process)
            process = birth
            assert type(row['observed_unix']) in (int, float) and math.isfinite(row['observed_unix']) and row['observed_unix'] > 0
            assert row['observed_unix'] >= previous; previous = row['observed_unix']
        assert process is not None and process not in process_births
        process_births.add(process)
        selected_workers, worker_failures = _tail(workers, run, rank, cursor, outer, writes)
        selected_schedulers, scheduler_failures = _tail(schedulers, run, rank, cursor, outer, writes)
        assert not worker_failures, 'Scheduler fault kind in worker receipts'
        for row in selected_workers + selected_schedulers:
            context = _context(row)
            if _gate(row) is not None:
                identities.add((context['attempt_id'], context['attempt_index'], context['logical_update'], _gate(row)))
        per_rank[str(rank)] = dict(_worker_prefix(selected_workers, writes),
            selected_worker_rows_sha256=document_sha256(selected_workers),
            selected_scheduler_rows_sha256=document_sha256(selected_schedulers),
            scheduler_tail_kinds=[row['kind'] for row in selected_schedulers],
            scheduler_fatal_failure_count=len(scheduler_failures),
            scheduler_unbound_failure_count=sum('attempt_index' not in _context(row) for row in scheduler_failures),
            scheduler_fatal_rows_sha256=document_sha256(scheduler_failures),
            process_pid=process[0], process_start_ticks=process[1])
    assert len(identities) <= 1, 'Unreturned rank attempt/gate identities disagree'
    common = []
    for positions in zip(*(per_rank[str(rank)]['recorded_write_positions'] for rank in range(4))):
        assert all(value == positions[0] for value in positions), 'Recorded rank write positions disagree'
        common.append(positions[0])
    for decisions in zip(*(per_rank[str(rank)]['dispositions'] for rank in range(4))):
        assert all(value == decisions[0] for value in decisions), 'Recorded rank dispositions disagree'
    return dict(role='actor_probe_unreturned_rank_tail', evidence_scope='scalar_receipts_only',
        segment=segment, after_driver_attempt_index=cursor, after_driver_writes=writes,
        unreturned_attempt=list(next(iter(identities))) if identities else None,
        per_rank=per_rank, all_rank_recorded_adam_writes=len(common),
        counted_new_complete_outers=0, files=files, production_admission=False,
        current_process_exit_proven=False, full_training_or_final_acceptance=False,
        requires_fault_review=any(row['scheduler_fatal_failure_count'] for row in per_rank.values()),
        scope='Newline-terminated recorded writes are lower-bound evidence; a write may occur before its receipt. '
              'No normal driver return means no new complete outer, even with four scheduler COMPLETE rows. '
              'Retained prefix, numerical/probability safeguards, source bytes, checkpoint/RNG/Head binding '
              'and actual parent process/session exit require independent acceptance.')
