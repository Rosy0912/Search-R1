"""One explicitly planned replacement of a failed, zero-prediction repeat.

No scheduler submission or automatic retry occurs here. The prior attempt is
immutable, and the replacement retains its original repeat index and science.
"""
import json
from pathlib import Path
import re
import subprocess

from .contracts import require, sha


def validate_recovery(plan, previous, submission, previous_root):
    recovery = plan['repeat_recovery']
    require(set(recovery) == {'previous_plan', 'previous_plan_sha256',
        'previous_submission', 'previous_submission_sha256', 'reason'},
        'Unexpected repeat recovery fields')
    require(recovery['reason'] == 'failed_startup_zero_predictions',
            'Only inspected startup failures can be replaced')
    require('repeat_recovery' not in previous and 'repeat' in previous,
            'Only one replacement of an ordinary repeat is permitted')
    changed = {'evaluation_code_commit', 'evaluation_run_id', 'job', 'output', 'repeat_recovery'}
    require(set(plan) == set(previous) | {'repeat_recovery'}
            and {k:v for k,v in plan.items() if k not in changed}
            == {k:v for k,v in previous.items() if k not in changed},
            'Replacement changes frozen science or repeat identity')
    require(re.fullmatch(r'[0-9a-f]{40}', plan['evaluation_code_commit']) is not None
            and re.fullmatch(r'[0-9a-f]{32}', plan['evaluation_run_id']) is not None
            and plan['evaluation_run_id'] not in (previous['evaluation_run_id'],
                Path(previous_root).parent.name), 'Invalid replacement execution identity')
    require(Path(previous['output']) == previous_root/'output'
            and Path(recovery['previous_submission']) == previous_root/'submission.json',
            'Failed repeat namespace differs')
    require(str(submission['job_id']).isdigit()
            and submission['plan'] == recovery['previous_plan']
            and submission['plan_sha256'] == recovery['previous_plan_sha256']
            and all(submission[k] == previous[k] for k in
                    ('job', 'output', 'checkpoint_identity', 'group', 'outer')),
            'Failed repeat submission binding differs')
    root = previous_root/('startup-recovery-'+plan['evaluation_code_commit'][:12])
    require(plan['output'] == str(root/'output')
            and plan['job'] == previous['job']+'-startup-r1',
            'Replacement output or job is not uniquely scoped')
    return root


def inspect_failed_startup(previous, submission):
    job = submission['job_id']
    raw = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', job,
        '--format=JobIDRaw,User,State'], text=True)
    rows = [r.split('|')[:3] for r in raw.splitlines() if r.strip()]
    require(rows == [[job, 'jmzhang', 'FAILED']], 'Previous repeat is not uniquely FAILED')
    live = subprocess.run(['squeue', '-h', '-j', job, '-o', '%i|%T'],
                          text=True, capture_output=True)
    require(not live.stdout.strip() and (live.returncode == 0 or
            (live.returncode == 1 and 'Invalid job id specified' in live.stderr)),
            'Previous repeat is live or scheduler status is uncertain')
    out = Path(previous['output'])
    require((out.parent/'exit.code').read_text().strip() == '1',
            'Failed attempt has no definitive unsuccessful exit')
    exit_receipt = json.loads((out/'entry-exit.json').read_text())
    require(exit_receipt.get('status') == 'FAILED'
            and exit_receipt.get('actual_job_id') == job
            and exit_receipt.get('job') == previous['job'],
            'Failed evaluator exit receipt differs')
    require(not (out/'progress.json').exists()
            and not (out/'summary.json').exists()
            and (not (out/'predictions.jsonl').exists()
                 or (out/'predictions.jsonl').stat().st_size == 0),
            'Prediction-producing attempt requires result continuation')


def inspect_recovery(plan):
    recovery = plan['repeat_recovery']
    require(sha(recovery['previous_plan']) == recovery['previous_plan_sha256']
            and sha(recovery['previous_submission']) == recovery['previous_submission_sha256'],
            'Failed attempt identity changed')
    previous = json.loads(Path(recovery['previous_plan']).read_text())
    require('repeat_recovery' not in previous, 'Recursive retry is forbidden')
    submission = json.loads(Path(recovery['previous_submission']).read_text())
    from .repeats import inspect_repeat
    previous_root = inspect_repeat(previous)
    root = validate_recovery(plan, previous, submission, previous_root)
    inspect_failed_startup(previous, submission)
    return root
