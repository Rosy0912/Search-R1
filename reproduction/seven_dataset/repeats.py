"""Bind requested original200 repeats to an existing first full evaluation.

Only submission identity and output location differ. This does not change
generation, scoring, model views, data, or the original milestone entry.
"""
import json
from pathlib import Path
import re

from .contracts import require, sha


def validate_repeat(plan, parent, submission):
    repeat = plan['repeat']
    require(set(repeat) == {'index', 'count', 'parent_plan', 'parent_plan_sha256',
                           'parent_submission', 'parent_submission_sha256'},
            'Unexpected repeat authority fields')
    index = repeat['index']
    require(type(index) is int and index in (2, 3)
            and type(repeat['count']) is int and repeat['count'] == 3,
            'Only requested repeats two and three are authorized')
    require(parent['protocol'] == 'tree_seven_milestone_v1'
            and parent['group'] == 'original' and parent['outer'] == 200
            and parent['backend'] == 'slurm_lumia' and 'repeat' not in parent,
            'Repeat parent must be the original200 first evaluation')
    require(not any(k in parent for k in ('replaces_job_id', 'pj_retired_submission')),
            'Repeat parent cannot be a replacement plan')
    require(parent.get('repeat_context', {}).get('requested_repeats') == 3
            and parent['repeat_context'].get('candidate_repeat_index') == 1,
            'Parent lacks the fixed three-repeat request')
    changed = {'evaluation_code_commit', 'evaluation_run_id', 'job', 'output', 'repeat'}
    require(set(plan) == set(parent) | {'repeat'}, 'Repeat schema differs')
    require({k: v for k, v in plan.items() if k not in changed}
            == {k: v for k, v in parent.items() if k not in changed},
            'Repeat changes model, data, configuration, view, or other frozen fields')
    require(re.fullmatch(r'[0-9a-f]{40}', plan['evaluation_code_commit']) is not None,
            'Invalid immutable source identity')
    require(re.fullmatch(r'[0-9a-f]{32}', plan['evaluation_run_id']) is not None
            and plan['evaluation_run_id'] != parent['evaluation_run_id'],
            'Repeat must have an independent execution identity')
    root = Path(parent['base']) / 'seven-evaluations' / parent['checkpoint_identity']
    require(Path(parent['output']) == root / 'output', 'First evaluation namespace differs')
    require(Path(repeat['parent_submission']) == root / 'submission.json',
            'Repeat references an unrelated submission receipt')
    require(submission['checkpoint_identity'] == parent['checkpoint_identity']
            and submission['group'] == 'original' and submission['outer'] == 200
            and submission['job'] == parent['job']
            and submission['output'] == parent['output']
            and submission['plan'] == repeat['parent_plan']
            and submission['plan_sha256'] == repeat['parent_plan_sha256']
            and str(submission['job_id']).isdigit(), 'First submission binding differs')
    expected = root / 'repeats' / parent['evaluation_run_id'] / ('r' + str(index))
    require(Path(plan['output']) == expected / 'output', 'Repeat output namespace differs')
    require(plan['job'] == 'zhangjunming-ttt-e-orig-200-r' + str(index)
            + '-' + parent['checkpoint_identity'][:4], 'Repeat job identity differs')
    return expected


def inspect_repeat(plan):
    repeat = plan['repeat']
    require(sha(repeat['parent_plan']) == repeat['parent_plan_sha256'],
            'First evaluation plan changed')
    require(sha(repeat['parent_submission']) == repeat['parent_submission_sha256'],
            'First evaluation submission changed')
    parent = json.loads(Path(repeat['parent_plan']).read_text())
    submission = json.loads(Path(repeat['parent_submission']).read_text())
    return validate_repeat(plan, parent, submission)
