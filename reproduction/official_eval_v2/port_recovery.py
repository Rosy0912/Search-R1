"""One independently inspected zero-prediction startup replacement.

Only LLM construction is serialized. The frozen worker and all inference
arguments, RNG seeding, model inputs, generation and scoring remain unchanged.
"""
import argparse
import contextlib
import fcntl
import json
import os
from pathlib import Path
import time

from . import driver
from .protocol import sha

ORIGINAL_VERIFY = driver.verify
ORIGINAL_WORKER = driver.worker
OLD_JOB = '139151'
OLD_SOURCE = 'd9915b4243fabefe07acaac2e6d09badb396877b'
NEW_JOB = 'zhangjunming-7b-v2p-searchr1-r3'
SCIENCE_FILES = [
    'reproduction/official_eval_v2/driver.py',
    'reproduction/official_eval_v2/protocol.py',
]


@contextlib.contextmanager
def startup_lock(path, timeout=900):
    deadline = time.monotonic() + timeout
    with Path(path).open('a') as stream:
        while True:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Startup lock timed out; no automatic retry')
                time.sleep(.05)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def construct_serialized(constructor, root, rank, args, kwargs):
    """Call the constructor exactly once, with unchanged arguments."""
    out = Path(root) / 'output'
    with startup_lock(out / 'llm-startup.lock'):
        driver.once(out / ('startup-rank-' + str(rank) + '.json'), dict(
            at=time.time(), rank=rank, pid=os.getpid(),
            vllm_port=os.environ['VLLM_PORT'],
            serialization='constructor_only', constructor_kwargs=kwargs,
            no_constructor_retry=True))
        result = constructor(*args, **kwargs)
        driver.once(out / ('startup-rank-' + str(rank) + '.complete.json'),
                    dict(at=time.time(), rank=rank, pid=os.getpid(),
                         vllm_port=os.environ['VLLM_PORT']))
    return result


def verify_recovery(plan):
    recovery = plan['startup_recovery']
    assert (plan['kind'], plan['repeat'], plan['job']) == ('searchr1', 3, NEW_JOB)
    assert recovery['old_job'] == OLD_JOB and recovery['no_auto_retry'] is True
    oldpath = Path(recovery['old_plan'])
    assert sha(oldpath) == recovery['old_plan_sha256']
    old = json.loads(oldpath.read_text())
    assert old['source_commit'] == OLD_SOURCE and old['repeat'] == 3
    mutable = {'root', 'job', 'source', 'source_commit', 'source_files', 'startup_recovery'}
    assert {k: v for k, v in old.items() if k not in mutable} == {
        k: v for k, v in plan.items() if k not in mutable}
    for name in SCIENCE_FILES:
        assert old['source_files'][name] == plan['source_files'][name]
    for name, digest in recovery['failure_evidence_files'].items():
        assert sha(name) == digest
    failure = json.loads(Path(recovery['failure_receipt']).read_text())
    assert failure['sub']['job_id'] == OLD_JOB
    assert failure['zero_predictions_proved'] and failure['zero_initialized_proved']
    projection = dict(plan, job=old['job'])
    return ORIGINAL_VERIFY(projection)


def startup_worker(plan, rank, gpu_uuid, memory_mib, gpu_index=None):
    index = rank if gpu_index is None else int(gpu_index)
    assert 0 <= rank < 8 and 0 <= index < 8
    # Keep the original mask ordering before vLLM or CUDA imports.
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(index)
    for name in ['ROCR_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'RAY_ADDRESS']:
        os.environ.pop(name, None)
    os.environ['VLLM_PORT'] = str(20000 + rank * 2048)
    import vllm
    original = vllm.LLM

    def serial_llm(*args, **kwargs):
        return construct_serialized(original, plan['root'], rank, args, kwargs)

    vllm.LLM = serial_llm
    try:
        return ORIGINAL_WORKER(plan, rank, gpu_uuid, memory_mib, gpu_index)
    finally:
        vllm.LLM = original


def run(planfile):
    driver.verify = verify_recovery
    driver.worker = startup_worker
    try:
        return driver.run(planfile)
    finally:
        driver.verify = ORIGINAL_VERIFY
        driver.worker = ORIGINAL_WORKER


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('plan')
    run(parser.parse_args().plan)
