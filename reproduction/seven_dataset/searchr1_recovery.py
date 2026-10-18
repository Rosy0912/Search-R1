"""An inspected zero-prediction startup replacement of one Search-R1 slot."""
import json
from pathlib import Path
import re
from .contracts import require, sha
from .repeat_recovery import inspect_failed_startup


def inspect_recovery(plan, template):
    recovery = plan['repeat_recovery']
    require(set(recovery) == {'previous_plan', 'previous_plan_sha256',
        'previous_submission', 'previous_submission_sha256', 'reason'},
        'Unexpected Search-R1 recovery fields')
    require(recovery['reason'] == 'failed_startup_zero_predictions',
            'Only an inspected zero-prediction startup failure can be replaced')
    require(sha(recovery['previous_plan']) == recovery['previous_plan_sha256']
            and sha(recovery['previous_submission']) == recovery['previous_submission_sha256'],
            'Failed Search-R1 attempt identity changed')
    previous = json.loads(Path(recovery['previous_plan']).read_text())
    require('repeat_recovery' not in previous, 'Recursive retry is forbidden')
    from .searchr1_contracts import validate_searchr1_plan
    previous_root = validate_searchr1_plan(previous, template)
    submission = json.loads(Path(recovery['previous_submission']).read_text())
    changed = {'evaluation_code_commit', 'evaluation_run_id', 'job', 'output', 'repeat_recovery'}
    require(set(plan) == set(previous) | {'repeat_recovery'}
            and {k:v for k,v in plan.items() if k not in changed}
            == {k:v for k,v in previous.items() if k not in changed},
            'Replacement changes frozen science or repeat identity')
    require(re.fullmatch(r'[0-9a-f]{40}', plan['evaluation_code_commit']) is not None
            and plan['evaluation_code_commit'] != previous['evaluation_code_commit']
            and re.fullmatch(r'[0-9a-f]{32}', plan['evaluation_run_id']) is not None
            and plan['evaluation_run_id'] != previous['evaluation_run_id'],
            'Replacement requires new pinned source and execution identities')
    require(Path(recovery['previous_submission']) == previous_root/'submission.json'
            and str(submission['job_id']).isdigit()
            and submission['plan'] == recovery['previous_plan']
            and submission['plan_sha256'] == recovery['previous_plan_sha256']
            and all(submission[k] == previous[k] for k in
                    ('job', 'output', 'checkpoint_identity', 'group', 'outer')),
            'Failed Search-R1 submission binding differs')
    root = previous_root/('startup-recovery-'+plan['evaluation_code_commit'][:12])
    require(plan['job'] == previous['job']+'-startup-r1'
            and plan['output'] == str(root/'output'),
            'Replacement namespace is not uniquely scoped')
    inspect_failed_startup(previous, submission)
    return root
