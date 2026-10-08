"""Local read-only 3315 numerical-history replay; never a launch/release tool."""
import hashlib
import json
import math
from pathlib import Path

SOURCE = '60f566b3ddabc88ce9de56ead5ad30a3d6795dd2'
JOB = 3315
RUN = '/dataset/shaoshitong/search-r1-rl/bapo-fresh200-controller-initfix-20261006/runs/formal200-bab8981d9fd14afc98923eb4ebb89e33-3315'
MARKERS = ('ACTOR_RANK_UPDATE_JSON=', 'FRESH200_ATTEMPT_JSON=', 'BAPO_TRAIN_METRICS_JSON=')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def finite(value):
    if isinstance(value, dict):
        for item in value.values(): finite(item)
    elif isinstance(value, list):
        for item in value: finite(item)
    elif isinstance(value, float):
        require(math.isfinite(value), 'nonfinite metric')


def lr(completed):
    return 1e-6 * min(completed + 1, 57) / 57


def close(a, b, message):
    require(type(a) in (int, float) and type(b) in (int, float)
            and math.isfinite(a) and math.isfinite(b)
            and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-14), message)


def check_rank_metrics(records, driver, completed_before, accepted, events):
    from search_r1.baselines.actor_metrics import merge_actor_metric_records
    require(len(records) == 8, 'exactly eight rank updates required')
    require(all(type(x.get('rank')) is int for x in records)
            and {x['rank'] for x in records} == set(range(8)), 'missing/duplicate rank')
    by_rank = {x['rank']: x for x in records}
    coverage_gaps = []
    metric_coverage = []
    for event in events:
        raw = by_rank[event['rank']]
        finite(raw)
        require(all(type(n) in (int, float) for value in raw['metrics'].values()
                    for n in (value if isinstance(value, list) else [value])), 'metric is not strictly numeric')
        require(type(raw.get('optimizer_steps')) is int
                and raw['optimizer_steps'] == event['optimizer_steps'], 'rank write count mismatch')
        require(type(raw.get('scheduler_last_epoch')) is int
                and raw['scheduler_last_epoch'] == completed_before + int(accepted), 'scheduler advanced on skip/abort')
        close(raw['lr_before'], lr(completed_before), 'incorrect before LR')
        close(raw['lr_after'], lr(completed_before + int(accepted)), 'incorrect after LR')
        require(raw['metrics'].get('actor/optimizer_steps') == raw['optimizer_steps'], 'metric/write mismatch')
        # Runtime appends loss before finite-gradient/Adam admission. Therefore
        # loss vector length is NOT the count of committed optimizer writes.
        # PartialPolicyUpdate drops the local loss/grad vectors entirely.
        for key in ('actor/pg_loss', 'actor/entropy_loss', 'actor/pg_clipfrac', 'actor/ppo_kl', 'actor/grad_norm'):
            values = raw['metrics'].get(key)
            coverage = dict(rank=event['rank'], metric=key, observed_count=(len(values) if isinstance(values,list) else None),
                committed_writes=event['optimizer_steps'], processed_opportunities=event['processed_opportunities'],
                skipped_opportunities=event['skipped_opportunities'],
                opportunity_index_mapping=('implicit_sequential_no_skips' if accepted and not event['skipped_opportunities'] else 'NOT_RECORDED_IN_RUNTIME_VECTOR'))
            metric_coverage.append(coverage)
            if values is None:
                require(not accepted, 'accepted update lacks numeric vector: ' + key)
                coverage_gaps.append(dict(rank=event['rank'], metric=key, reason='runtime_partial_return_has_no_vector'))
                continue
            require(isinstance(values, list), 'numeric mini metric must be a list')
            count = len(values)
            if key == 'actor/grad_norm' and accepted:
                require(count == event['optimizer_steps'], 'completed grad count differs from writes')
            else:
                require(event['optimizer_steps'] <= count <= event['processed_opportunities'], 'numeric opportunity count differs')
            if count < event['processed_opportunities']:
                coverage_gaps.append(dict(coverage, reason='fewer_observed_values_than_processed_opportunities',
                    missing_processed_values=event['processed_opportunities']-count))
            if event['skipped_opportunities']:
                coverage_gaps.append(dict(coverage, reason='runtime_vector_has_no_explicit_opportunity_index_mapping'))
            if not values:
                coverage_gaps.append(dict(coverage, reason='empty_vector'))
        for key, expected in (('actor/lr', raw['lr_after']), ('actor/lr_before_outer', raw['lr_before']),
                              ('actor/scheduler_outer_steps', raw['scheduler_last_epoch'])):
            close(raw['metrics'][key], expected, 'rank metric clock mismatch')
    merged = merge_actor_metric_records([by_rank[r]['metrics'] for r in range(8)])
    finite(driver)
    for key, value in merged.items():
        expected = math.fsum(value) / len(value) if isinstance(value, list) and value else value
        # Empty mini lists reduce to NaN in the runtime reducer; do not manufacture finite acceptance.
        require(not isinstance(expected, list), 'empty list has no finite driver mean')
        require(key in driver, 'driver lacks all-rank metric: ' + key)
        close(driver[key], expected, 'driver all-rank aggregate mismatch: ' + key)
    return merged, coverage_gaps, metric_coverage


def replay(train_log, ledger_documents, source=SOURCE, job=JOB, run=RUN):
    """Full ordered log + independently copied durable ledgers keyed by original ledger_path.

    Eight records must precede each attempt outcome; per-rank order is preserved,
    but rank arrival order inside an attempt is unrestricted. Delayed/interleaved
    ambiguous transport is rejected, rather than guessed from scheduler alone.
    """
    from search_r1.baselines.fresh200_driver import Fresh200Progress
    require((source, job, run) == (SOURCE, JOB, RUN), 'foreign run identity')
    progress = Fresh200Progress()
    ranks, waiting, used = [], None, set()
    outcomes = []
    driver_pid = None
    for number, line in enumerate(train_log.splitlines(), 1):
        hits = [m for m in MARKERS if m in line]
        if not hits: continue
        require(len(hits) == 1, 'ambiguous multi-marker log line')
        marker = hits[0]
        record = strict_json(line.split(marker, 1)[1]); finite(record)
        if marker == MARKERS[0]:
            require(waiting is None, 'rank record crossed driver outcome boundary')
            ranks.append(record)
            require(len(ranks) <= 8, 'duplicate or reordered rank record')
        elif marker == MARKERS[1]:
            require(waiting is None and len(ranks) == 8, 'missing ranks or reordered attempt')
            require(record.get('source_commit') == SOURCE, 'attempt source mismatch')
            require(type(record.get('driver_pid')) is int and record['driver_pid'] > 0, 'missing/invalid original driver PID')
            if driver_pid is None: driver_pid = record['driver_pid']
            require(driver_pid == record['driver_pid'], 'driver PID changed across attempts')
            candidate = record['candidate_outer']
            attempt_id = progress.begin_attempt(candidate)
            require(type(record['attempt_id']) is int and record['attempt_id'] == attempt_id, 'attempt order mismatch')
            path = record['ledger_path']
            expected_path = RUN + '/checkpoints/diagnostics/attempt_%d_candidate_outer_%d/all-rank-write-ledger.json' % (attempt_id, candidate)
            require(path == expected_path and path not in used, 'noncanonical/foreign/duplicate ledger path')
            require(path in ledger_documents and ledger_documents[path] == record, 'durable ledger missing/differs from log')
            used.add(path)
            before = progress.completed_outers
            accepted = progress.record(record['rank_write_ledger'], candidate)
            require(type(record['accepted_complete_outer']) is bool and record['accepted_complete_outer'] is accepted, 'accepted flag mismatch')
            require(record['progress'] == progress.state_dict(), 'cumulative progress mismatch')
            waiting = (record, ranks, before, accepted, number)
            ranks = []
        else:
            require(waiting is not None, 'driver metrics without attempt')
            outcome, raw, before, accepted, line_number = waiting
            require(type(record.get('attempt_id')) is int and record['attempt_id'] == outcome['attempt_id']
                    and type(record.get('step')) is int and record['step'] == outcome['candidate_outer']
                    and type(record.get('accepted_complete_outer')) is bool
                    and record['accepted_complete_outer'] is accepted, 'driver attempt identity mismatch')
            merged, gaps, coverage = check_rank_metrics(raw, record['metrics'], before, accepted, outcome['rank_write_ledger'])
            outcomes.append(dict(attempt_id=outcome['attempt_id'], accepted=accepted,
                                 candidate_outer=outcome['candidate_outer'], rank_metrics=merged,
                                 attempt_line=line_number, driver_line=number, numerical_coverage_gaps=gaps, per_rank_metric_coverage=coverage))
            waiting = None
    require(not ranks and waiting is None, 'incomplete trailing attempt')
    require(outcomes and used == set(ledger_documents), 'missing/extra durable attempt ledger')
    return dict(status=('READ_ONLY_ACCOUNTING_REPLAY_NUMERICAL_GAPS_NOT_ACCEPTANCE' if any(x['numerical_coverage_gaps'] for x in outcomes) else 'READ_ONLY_NUMERICAL_HISTORY_REPLAY_COMPLETE_NOT_FINAL_ACCEPTANCE'),
                source_commit=SOURCE, job_id=JOB, run_root=RUN, driver_pid=driver_pid,
                train_log_sha256=hashlib.sha256(train_log.encode()).hexdigest(),
                progress=progress.state_dict(), attempts=outcomes,
                native_acceptance=False, release_authorized=False,
                pending=['native initial/current fullstate', 'sampling chain', 'first5 actual receipt',
                         'final HF and 51713 evaluation', 'terminal resources/cleanup/cost'])


def main():
    import argparse
    import subprocess
    import sys
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--train-log', type=Path, required=True)
    parser.add_argument('--ledger-index', type=Path, required=True,
                        help='Local copied entries: original_path/local_path/sha256 (not an acceptance receipt)')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    require(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==SOURCE,
            'repository is not frozen source60f')
    require(not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip(),
            'repository dirty')
    sys.path.insert(0,str(repo))
    # Verify the exact shared implementations before importing any runtime reducers.
    for relative in ('search_r1/baselines/fresh200_driver.py','search_r1/baselines/fresh200_contract.py',
                     'search_r1/baselines/actor_metrics.py'):
        expected=subprocess.check_output(['git','-C',str(repo),'show',SOURCE+':'+relative])
        require((repo/relative).read_bytes()==expected,'shared helper source differs')
    entries=strict_json(args.ledger_index.read_text()); documents={}
    require(isinstance(entries,list),'ledger index must be list')
    for entry in entries:
        origin=entry['original_path'];local=Path(entry['local_path'])
        require(origin not in documents and local.is_absolute() and local.is_file() and not local.is_symlink() and local.resolve() == local,
                'duplicate or noncanonical ledger copy')
        raw=local.read_bytes()
        require(hashlib.sha256(raw).hexdigest()==entry['sha256'],'copied ledger digest mismatch')
        documents[origin]=strict_json(raw)
    raw_log=args.train_log.read_bytes()
    result=replay(raw_log.decode('utf-8'),documents)
    result['train_log_sha256']=hashlib.sha256(raw_log).hexdigest()
    result['ledger_index_sha256']=hashlib.sha256(args.ledger_index.read_bytes()).hexdigest()
    with args.output.open('x') as handle:
        json.dump(result,handle,indent=2,allow_nan=False)
    print(result['status'])


if __name__ == '__main__':
    main()
