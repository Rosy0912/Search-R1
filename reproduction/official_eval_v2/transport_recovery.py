"""Inspected same-repeat replacement with one allocation-wide transport replay.

The original worker, request encoder, response parser and 180s timeout are
unchanged. Only one idempotent retrieval POST may be repeated after a transport
exception. Every HTTP attempt is journaled; extra transport work is separate
from logical search actions and from failed-allocation costs.
"""
import argparse
import http.client
import json
from pathlib import Path
import socket
import time
import urllib.error

from . import driver, port_recovery
from .protocol import digest, sha

ORIGINAL_SEARCH = driver.search
ORIGINAL_VERIFY = driver.verify
NEW_JOB = 'zhangjunming-7b-v2t-searchr1-r3'
ROOT = None
RANK = None
CALL_INDEX = 0


def transport_exception(error):
    if isinstance(error, urllib.error.HTTPError):
        return False
    if isinstance(error, urllib.error.URLError):
        error = error.reason
    return isinstance(error, (socket.timeout, ConnectionResetError,
                              http.client.IncompleteRead))


def perform_search(url, queries, root, rank, call_index, request_fn):
    frozen_queries = list(queries)
    identity = digest(dict(url=url, queries=frozen_queries, topk=3,
                           return_scores=True))
    out = Path(root) / 'output'
    for attempt in (1, 2):
        prefix = out / ('http-rank-' + str(rank) + '-call-' +
                        str(call_index) + '-attempt-' + str(attempt))
        start = time.time()
        driver.once(str(prefix) + '.start.json', dict(
            at=start, rank=rank, call_index=call_index, attempt=attempt,
            request_identity=identity, url=url, queries=frozen_queries,
            logical_query_count=len(frozen_queries), timeout_seconds=180,
            original_request_encoder_and_parser=True))
        try:
            result = request_fn(url, list(frozen_queries))
        except Exception as error:
            driver.once(str(prefix) + '.error.json', dict(
                at=time.time(), rank=rank, call_index=call_index,
                attempt=attempt, request_identity=identity,
                exception_type=type(error).__name__, message=str(error),
                backend_completion_unknown=True))
            if attempt != 1 or not transport_exception(error):
                raise
            try:
                driver.once(out / 'http-single-replay-claim.json', dict(
                    at=time.time(), rank=rank, call_index=call_index,
                    request_identity=identity, logical_query_count=len(frozen_queries),
                    failed_attempt_backend_cost_unknown=True,
                    maximum_replays_for_entire_allocation=1,
                    no_job_resubmission=True))
            except FileExistsError:
                raise RuntimeError('Single transport replay budget exhausted; preserve output') from error
            time.sleep(1)
            continue
        driver.once(str(prefix) + '.complete.json', dict(
            at=time.time(), rank=rank, call_index=call_index, attempt=attempt,
            request_identity=identity, response_identity=result[1],
            logical_query_count=len(frozen_queries),
            elapsed_seconds=time.time() - start))
        return result
    raise AssertionError('Unreachable transport state')


def controlled_search(url, queries):
    global CALL_INDEX
    assert ROOT is not None and RANK in range(8)
    CALL_INDEX += 1
    return perform_search(url, queries, ROOT, RANK, CALL_INDEX, ORIGINAL_SEARCH)


def verify_recovery(plan):
    recovery = plan['transport_recovery']
    assert (plan['kind'], plan['repeat'], plan['job']) == ('searchr1', 3, NEW_JOB)
    assert recovery['failed_job'] == '139152'
    assert recovery['maximum_http_replays_per_allocation'] == 1
    assert recovery['no_auto_job_retry'] is True and recovery['partial_mixing_allowed'] is False
    oldpath = Path(recovery['original_plan'])
    assert sha(oldpath) == recovery['original_plan_sha256']
    old = json.loads(oldpath.read_text())
    assert old['source_commit'] == 'd9915b4243fabefe07acaac2e6d09badb396877b'
    mutable = {'root', 'job', 'source', 'source_commit', 'source_files', 'transport_recovery'}
    assert {k: v for k, v in old.items() if k not in mutable} == {
        k: v for k, v in plan.items() if k not in mutable}
    for name in ['reproduction/official_eval_v2/driver.py',
                 'reproduction/official_eval_v2/protocol.py']:
        assert plan['source_files'][name] == old['source_files'][name]
    for name, expected in recovery['evidence_files'].items():
        assert sha(name) == expected
    failure = json.loads(Path(recovery['failure_receipt']).read_text())
    assert failure['sub']['job_id'] == '139152'
    assert failure['sacct'][2:5] == ['FAILED', '388', '1:0']
    assert failure['raw_prediction_count'] == 2560 and failure['local_sha_backup_verified']
    health = json.loads(Path(recovery['batch_health_proof']).read_text())
    assert len(health['records']) == 2
    assert all(r['actual_original_saved_batch_exactly_matched'] for r in health['records'])
    return ORIGINAL_VERIFY(dict(plan, job=old['job']))


def worker(plan, rank, gpu_uuid, memory_mib, gpu_index=None):
    global ROOT, RANK, CALL_INDEX
    ROOT = plan['root']; RANK = rank; CALL_INDEX = 0
    driver.search = controlled_search
    try:
        return port_recovery.startup_worker(plan, rank, gpu_uuid, memory_mib, gpu_index)
    finally:
        driver.search = ORIGINAL_SEARCH


def run(planfile):
    original_worker = driver.worker
    driver.verify = verify_recovery
    driver.worker = worker
    try:
        return driver.run(planfile)
    finally:
        driver.verify = ORIGINAL_VERIFY
        driver.worker = original_worker


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('plan')
    run(parser.parse_args().plan)
