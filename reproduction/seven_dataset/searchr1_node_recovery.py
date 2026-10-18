"""One inspected node4 failure replacement of Search-R1 repeat1.

This entry admits a new full execution only after the former allocation has
been retired and its partial predictions and cost have immutable backups.
It never appends/merges partial results, submits work, or automatically retries.
Scientific inference is performed by the unchanged common runner.
"""
import json
from pathlib import Path
import re
import subprocess

from .contracts import require, sha


def inspect_node_recovery(plan, template):
    proof = plan['node_recovery']
    fields = {'previous_plan', 'previous_submission', 'incident', 'backup', 'retirement'}
    require(set(proof) == fields | {k+'_sha256' for k in fields} | {'reason'},
            'Unexpected node-failure recovery fields')
    require(proof['reason'] == 'node4_reboot_139033_partial_preserved',
            'Uninspected node failure')
    for key in fields:
        require(sha(proof[key]) == proof[key+'_sha256'], 'Recovery evidence changed: '+key)
    previous = json.loads(Path(proof['previous_plan']).read_text())
    require('node_recovery' not in previous and 'repeat_recovery' in previous,
            'Only the inspected139033 node failure is eligible; no recursive node retry')
    from .searchr1_contracts import validate_searchr1_plan
    previous_root = validate_searchr1_plan(previous, template)
    sub = json.loads(Path(proof['previous_submission']).read_text())
    require(sub['job_id'] == '139033' and sub['repeat_index'] == 1
            and sub['repeat_count'] == 3 and previous['repeat_index'] == 1
            and sub['plan'] == proof['previous_plan']
            and sub['plan_sha256'] == proof['previous_plan_sha256']
            and Path(proof['previous_submission']) == previous_root/'submission.json'
            and all(sub[k] == previous[k] for k in ('job', 'output', 'checkpoint_identity', 'group', 'outer')),
            'Node-failure submission binding differs')
    changed = {'evaluation_code_commit', 'evaluation_run_id', 'job', 'output',
               'repeat_recovery', 'node_recovery'}
    require(set(plan) == (set(previous)-{'repeat_recovery'}) | {'node_recovery'}
            and {k:v for k,v in plan.items() if k not in changed}
            == {k:v for k,v in previous.items() if k not in changed},
            'Node replacement changes frozen science or repeat identity')
    require(re.fullmatch('[0-9a-f]{40}', plan['evaluation_code_commit']) is not None
            and plan['evaluation_code_commit'] != previous['evaluation_code_commit']
            and re.fullmatch('[0-9a-f]{32}', plan['evaluation_run_id']) is not None
            and plan['evaluation_run_id'] != previous['evaluation_run_id'],
            'Replacement requires new pinned execution identities')
    root = previous_root/('node-recovery-'+plan['evaluation_code_commit'][:12])
    require(plan['output'] == str(root/'output')
            and plan['job'] == 'zhangjunming-ttt-sr1-r1-nodefix',
            'Unbound node-failure output/name')
    incident = json.loads(Path(proof['incident']).read_text())
    backup = json.loads(Path(proof['backup']).read_text())
    retirement = json.loads(Path(proof['retirement']).read_text())
    require(incident['output'] == previous['output']
            and 'BootTime=2026-10-18T16:35:12' in incident['scheduler']['node']['stdout']
            and 'Prolog error' in incident['scheduler']['node']['stdout']
            and '139033|REQUEUED|2657|' in incident['scheduler']['accounting']['stdout'],
            'Node reboot/cost evidence differs')
    require(backup['source'] == previous['output'] and backup['preserved_only'] is True
            and backup['accepted_complete'] is False
            and set(backup['files']) == {'predictions.jsonl', 'progress.json'}
            and backup['files'] == incident['files'], 'Partial backup not bound to incident')
    require(retirement['previous_job_id'] == '139033'
            and retirement['prior_state'] == 'PENDING_HELD_NO_ALLOCATION'
            and retirement['retired_state'] == 'CANCELLED'
            and retirement['incident_sha256'] == proof['incident_sha256']
            and retirement['backup_sha256'] == proof['backup_sha256'],
            'Held allocation was not independently retired')
    rows = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', '139033',
        '--format=JobIDRaw,User,State%40'], text=True, timeout=20)
    rows = [r.split('|')[:3] for r in rows.splitlines() if r.strip()]
    require(len(rows) == 1 and rows[0][:2] == ['139033', 'jmzhang']
            and rows[0][2] in ('CANCELLED', 'CANCELLED by 1066'),
            'Former allocation is not uniquely cancelled')
    live = subprocess.run(['squeue', '-h', '-j', '139033', '-o', '%i|%T'],
                          text=True, capture_output=True, timeout=20)
    require(not live.stdout.strip() and (live.returncode == 0 or
            (live.returncode == 1 and 'Invalid job id specified' in live.stderr)),
            'Former allocation is live or scheduler uncertain')
    old_out = Path(previous['output'])
    require(not (old_out/'summary.json').exists(), 'Completed repeat cannot be replaced')
    for name, meta in backup['files'].items():
        require(name in ('predictions.jsonl', 'progress.json')
                and (old_out/name).stat().st_size == meta['bytes']
                and sha(old_out/name) == meta['sha256'], 'Preserved partial changed')
    progress = json.loads((old_out/'progress.json').read_text())
    require(progress['completed'] == 7168 and progress['total'] == 51713
            and progress['job'] == previous['job']
            and progress['checkpoint_identity'] == previous['checkpoint_identity'],
            'Partial progress identity differs')
    return root
