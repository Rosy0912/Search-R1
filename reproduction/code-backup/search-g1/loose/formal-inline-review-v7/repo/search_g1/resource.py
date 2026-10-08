"""Cross-process exclusion for probe inference/reload and calibration on one GPU.

Resident model memory remains allocated; this lock is not evidence of memory fit.
"""
import contextlib
import fcntl
import os
from pathlib import Path

@contextlib.contextmanager
def evaluator_gpu_lock():
    base=Path(os.environ.get('SEARCH_G1_BASE','/data/search-g1'))
    path=base/'logs/evaluator-gpu.lock'
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as handle:
        fcntl.flock(handle.fileno(),fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(),fcntl.LOCK_UN)
