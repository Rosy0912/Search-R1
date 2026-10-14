"""Render a C2 script from the executable condition registry before submission.

This module does not contact the scheduler. Callers must publish/deploy source,
preserve a once-only submission receipt, and inspect the returned allocation.
"""
from pathlib import PurePosixPath
import shlex

from .prepare_ablation import condition_for


GROUP = 'C2-random-final'


def script(base, source_commit, c1_job):
    condition = condition_for(GROUP)
    if condition['variant'] != 'random_final':
        raise ValueError('C2 must use the published random-final mechanism')
    if (len(source_commit) != 40 or any(c not in '0123456789abcdef' for c in source_commit)
            or not str(c1_job).isdigit() or int(c1_job) < 1):
        raise ValueError('Need a full published source SHA and exact prerequisite allocation')
    base = PurePosixPath(base)
    if not base.is_absolute() or '..' in base.parts:
        raise ValueError('Use an absolute study root')
    source = base / 'sources' / source_commit
    q = lambda value: shlex.quote(str(value))
    # GROUP is a configuration key, not a human-readable scheduler job label.
    # Keep its validation before sbatch so a typo cannot reserve eight GPUs.
    lines = ['#!/bin/bash', 'set -euo pipefail',
        'export RECURSIVE_LUMIA_BASE=' + q(base),
        'export RECURSIVE_GROUP=' + q(GROUP),
        'export RECURSIVE_SOURCE_COMMIT=' + q(source_commit),
        'export RECURSIVE_SOURCE_DIR=' + q(source),
        'export PYTHONPATH=' + q(source) + ' PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1',
        shlex.join(['python3', '-m', 'reproduction.recursive_lumia.gate_c1_complete',
            '--run', str(base / 'runs/C1-root-only'),
            '--log', str(base / ('logs/C1-root-only-4090-' + str(c1_job) + '.log')),
            '--job', str(c1_job), '--output']) + ' ' + q(base / 'receipts')
            + '/C2-prerequisite-${SLURM_JOB_ID}.json',
        'exec bash ' + q(source / 'reproduction/recursive_lumia/ablation.sh')]
    return '\n'.join(lines) + '\n'
