"""Read a retained driver/rank prefix without erasing discarded fault evidence.

This is scalar evidence only. A caller must separately verify checkpoint bytes,
full state, source, actual process exit and the complete final recovery chain.
"""
import re
from pathlib import Path

from reproduction.common8.actor_probe_chain import account_segment
from reproduction.common8.actor_probe_rank_tail import _sealed_bytes, _sealed_log, _context
from reproduction.common8.policy_calibration_plan import _strict_json, document_sha256
from search_g1.actor_probe_contract import actor_probes
from search_g1.actor_probe_namespace import namespace_origin
from search_g1.actor_probe_recovery import identity

ROLE = 'actor_probe_retained_driver_prefix'
FATAL = ('POST_COMMIT_FAILURE', 'PRE_COMMIT_EVIDENCE_FAILURE')


def load_prefix(base, run, last):
    base = Path(base).resolve(); origin = namespace_origin(run)
    assert actor_probes(run) and type(last) is int and origin['completed_outer'] < last < 200
    raw, manifest = _sealed_bytes(base, 'logs/run-manifest.json', limit=32*1024**2)
    assert _strict_json(raw) == run
    updates, update_file = _sealed_log(base, 'logs/actual-updates.jsonl')
    attempts, attempt_file = _sealed_log(base, 'logs/actual-attempts.jsonl')
    segment = account_segment(run['run_id'], updates, attempts, retained_through=last,
        start_outer=origin['completed_outer'], start_writes=origin['actual_optimizer_writes'],
        start_attempt_index=origin['attempt_index'])
    cutoff = segment['checkpoint_attempt_index']
    assert segment['retained_attempts'][-1]['attempt_index'] == cutoff
    return dict(role=ROLE, evidence_scope='scalar_prefix_only', identity=identity(run),
        last=last, cutoff_attempt_index=cutoff, origin=origin, segment=segment,
        selected_updates=segment['retained_updates'], selected_attempts=segment['retained_attempts'],
        files=[manifest, update_file, attempt_file], production_admission=False,
        full_training_or_final_acceptance=False)


def verify_prefix(base, run, prefix, last):
    assert prefix == load_prefix(base, run, last), 'Prefix differs from the sealed actual driver bytes'
    return prefix


def discarded_identity(context, prefix):
    """Exclude only a real returned attempt or the single next open identity."""
    index=context.get('attempt_index'); step=context.get('logical_update'); attempt=context.get('attempt_id')
    assert type(index) is int and index>prefix['cutoff_attempt_index']
    assert type(step) is int and prefix['last']<step<=200
    assert isinstance(attempt,str) and re.fullmatch('[0-9a-f]{32}',attempt)
    actual=prefix['segment']['retained_attempts']+prefix['segment']['discarded_attempts']
    by_index={row['attempt_index']:row for row in actual}
    if index in by_index:
        row=by_index[index]
        assert (attempt,step)==(row['attempt_id'],row['logical_update']), 'Discarded identity differs from the actual returned driver'
    else:
        assert index==actual[-1]['attempt_index']+1
        assert step==prefix['segment']['observed_completed_through']+1
        assert attempt not in {row['attempt_id'] for row in actual}, 'Retained UUID was rebound to an open tail'
    return attempt,index,step


def rank_prefix(base, run, prefix, rank, ledger):
    """Returned prefix only; terminal fragments and later rows remain hashed."""
    assert ledger in ('optimizer', 'scheduler') and type(rank) is int and 0 <= rank < 4
    verify_prefix(base, run, prefix, prefix['last'])
    rows, metadata = _sealed_log(Path(base).resolve(), f'logs/{ledger}-rank-{rank}.jsonl')
    selected = []; discarded = []; tail_started = False; open_identities=set()
    cutoff = prefix['cutoff_attempt_index']; origin = prefix['origin']['attempt_index']
    by_index = {row['attempt_index']:row for row in prefix['selected_attempts']}
    for position, row in enumerate(rows):
        assert row['run_id'] == run['run_id'] and type(row['rank']) is int and row['rank'] == rank
        if row['kind'] == 'RESTORE':
            # The actual restore writer emits only a scheduler anchor.
            assert ledger == 'scheduler' and position == 0 and run.get('actor_probe_recovery') is not None
            assert row['last_attempt_index'] == origin
            selected.append(row); continue
        context = _context(row)
        index = context.get('attempt_index')
        assert type(index) is int and index > origin, 'Unbound fault cannot be assigned outside a retained prefix'
        assert isinstance(context.get('attempt_id'), str) and re.fullmatch('[0-9a-f]{32}', context['attempt_id'])
        if index > cutoff:
            value=discarded_identity(context,prefix)
            if index>prefix['segment']['retained_attempts'][-1]['attempt_index'] and \
                    index not in {row['attempt_index'] for row in prefix['segment']['discarded_attempts']}:
                open_identities.add(value)
                assert len(open_identities)==1, 'Multiple unreturned tail identities on one rank'
            tail_started = True; discarded.append(row); continue
        assert not tail_started and index in by_index, 'Retained row reordered or absent from actual driver'
        attempt = by_index[index]
        assert context['attempt_id'] == attempt['attempt_id'] and context['logical_update'] == attempt['logical_update']
        assert row['kind'] not in (*FATAL, 'REJECTED'), 'Fatal evidence inside the retained attempt prefix'
        selected.append(row)
    return selected, dict(metadata, selected_rows_sha256=document_sha256(selected),
        discarded_rows_sha256=document_sha256(discarded), discarded_newline_rows=len(discarded),
        discarded_fatal_rows=sum(row['kind'] in FATAL for row in discarded),
        discarded_rejected_rows=sum(row['kind']=='REJECTED' for row in discarded),
        requires_fault_review=any(row['kind'] in (*FATAL,'REJECTED') for row in discarded),
        unreturned_identity=list(next(iter(open_identities))) if open_identities else None,
        discarded_complete_outers_counted=0, scope='retained driver cutoff only; full fault-tail audit remains required')


def console_evidence(base, run, prefix):
    """A fatal fallback after the saved cutoff is lost work, never completion."""
    verify_prefix(base, run, prefix, prefix['last'])
    base = Path(base).resolve(); path = Path(run['train_log'])
    assert path.is_absolute() and path == path.resolve() and path.is_relative_to(base)
    raw, metadata = _sealed_bytes(base, path.relative_to(base).as_posix())
    marker = 'SEARCH_G1_FATAL_SCHEDULER_EVIDENCE='
    failures = []; open_by_rank={}; rank_files={}
    for line in raw.decode('utf-8').splitlines():
        if marker not in line: continue
        payload = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', line.split(marker, 1)[1])
        row = _strict_json(payload.encode('utf-8')); context = _context(row)
        assert row['kind'] in FATAL and row['fatal_no_automatic_resume'] is True
        assert row['run_id'] == run['run_id'] and type(row['rank']) is int and 0 <= row['rank'] < 4
        value=discarded_identity(context,prefix)
        if context['attempt_index']>(prefix['segment']['retained_attempts']+prefix['segment']['discarded_attempts'])[-1]['attempt_index']:
            rank=row['rank']
            assert rank not in open_by_rank or open_by_rank[rank]==value, 'Multiple console unreturned identities on one rank'
            open_by_rank[rank]=value
            if rank not in rank_files:
                rank_files[rank]=rank_prefix(base,run,prefix,rank,'scheduler')[1]
            recorded=rank_files[rank]['unreturned_identity']
            assert recorded is None or recorded==list(value), 'Console unreturned identity differs from the sealed rank tail'
        assert all(type(row.get(k)) is int and row[k] > 0 for k in ('process_pid','process_start_ticks'))
        assert isinstance(row.get('reason'), str) and row['reason']
        failures.append(row)
    return dict(console_path=str(path), observed_sha256=metadata['sha256'],
        discarded_fatal_records=failures, discarded_fatal_records_sha256=document_sha256(failures),
        requires_fault_review=bool(failures),
        discarded_rank_tail_files={str(rank):metadata for rank,metadata in rank_files.items()},
        scope='owned stable console; only bound fatal attempts strictly after retained cutoff are excluded; no rollback claimed')
