"""Monotonic observation only; no worker RPC or fabricated server attribution."""
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path


def record_duration(trainer, stage, wall_seconds, category='checkpoint_or_independent_review'):
    if not hasattr(trainer, 'continuation_timings'): trainer.continuation_timings = []
    trainer.continuation_timings.append(dict(stage=stage, category=category,
        wall_seconds=wall_seconds, original_call_returned=True, attempt_id=None,
        completed_outer_updates=getattr(trainer, 'actual_outer_updates', None)))


@contextmanager
def observe_time(trainer, stage, category='ordinary_training'):
    started, returned = time.monotonic(), False
    try:
        yield
        returned = True
    finally:
        try:
            progress = getattr(trainer, '_calibration_progress', None)
            row = dict(stage=stage, category=category, wall_seconds=time.monotonic() - started,
                original_call_returned=returned,
                attempt_id=None if progress is None else progress.get('attempt_id'),
                completed_outer_updates=getattr(trainer, 'actual_outer_updates', None))
            if progress is not None:
                progress.setdefault('timings', []).append(row)
            else:
                if not hasattr(trainer, 'continuation_timings'): trainer.continuation_timings = []
                trainer.continuation_timings.append(row)
        except BaseException:
            if returned: raise  # Never replace an original unknown exception.


def publish_timing(trainer, run, outcome=None):
    progress = getattr(trainer, '_calibration_progress', None)
    rows = list(getattr(trainer, 'continuation_timings', []))
    if progress is not None: rows += list(progress.get('timings', []))
    ledger = getattr(trainer, 'fresh200_attempt_ledger', {})
    value = dict(job_id=os.environ['SLURM_JOB_ID'], run_id=os.environ['SEARCH_R1_BASELINE_RUN_ID'],
        source_commit=os.environ['SEARCH_R1_BASELINE_COMMIT'],
        config_sha256=os.environ['SEARCH_R1_BASELINE_CONFIG_SHA256'],
        completed_outer_updates=ledger.get('completed_outer_updates', 0), actual_adam_total=ledger.get('actual_adam_total', 0),
        attempts=ledger.get('attempts', 0), outcome=outcome, timings=rows,
        generation_retrieval_is_client_combined=True, e5_server_cpu_or_network_attribution=None,
        benchmark_or_gpu_acceptance_claim=False)
    with (Path(run) / 'timing-progress.jsonl').open('a') as stream:
        stream.write(json.dumps(value, allow_nan=False) + '\n'); stream.flush(); os.fsync(stream.fileno())
    trainer.continuation_timings = []
    if progress is not None: progress['timings'] = []
