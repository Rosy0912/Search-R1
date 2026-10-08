"""Driver records only completed outer updates, after actual worker writes return."""
import json
import math
import os
from pathlib import Path
from reproduction.common8.formal_contract import (
    CONTRACT_KIND, SAFETY_MODE, TARGET_UPDATES, assert_formal_run,
)
from reproduction.common8.topology import world_size


LEGACY_METRICS=('critic/rewards/mean','actor/pg_loss','actor/entropy_loss',
         'actor/kl_loss','actor/ppo_kl','actor/grad_norm','actor/optimizer_steps_this_update')
METRICS=LEGACY_METRICS+('actor/update_opportunities_this_attempt','actor/skipped_update_opportunities')


def record(step,metrics,probe_version,attempt_result=None):
    base=Path(os.environ['SEARCH_G1_BASE'])
    run=json.loads((base/'logs/run-manifest.json').read_text())
    inline=run.get('policy_safety_mode')==SAFETY_MODE
    if inline:
        assert_formal_run(run)
        maximum=TARGET_UPDATES
    else:
        # A legacy child keeps its actual frozen 300/constant-LR mathematics
        # and old gate interface. It cannot acquire the new formal identity.
        from reproduction.common8.policy_calibration_plan import (
            CONFIRMATION_ROLE, DISCOVERY_ROLE, _identity, load_plan,
        )
        assert run.get('contract_kind')!=CONTRACT_KIND and run.get('policy_safety_mode') is None
        assert attempt_result is None
        role=run.get('policy_calibration_role','production')
        assert role in (DISCOVERY_ROLE,CONFIRMATION_ROLE,'production')
        _identity(run)
        if role!='production':
            plan,_=load_plan(base,run)
            maximum=plan['phase_budget']['outer_count']
        else:
            maximum=run['budget']['target_actual_outer_updates']
        assert maximum in (1,300)
    assert type(step) is int and 1<=step<=maximum
    scalar={key:float(metrics[key]) for key in (METRICS if inline else LEGACY_METRICS)}
    assert all(math.isfinite(v) for v in scalar.values())
    if inline:
        assert isinstance(attempt_result,dict) and attempt_result['status']=='complete'
        assert attempt_result['all_ranks_agreed'] is True
        assert type(attempt_result['real_rank_count']) is int and attempt_result['real_rank_count']==world_size(run), \
            'Completed attempt rank count differs from the actual run topology'
        assert attempt_result['logical_update']==step and attempt_result['completed_outer']==step
        assert attempt_result['complete_outer'] is True and attempt_result['scheduler_advances']==1
        successes=attempt_result['optimizer_writes_this_attempt']
        before,after=attempt_result['optimizer_writes_before'],attempt_result['optimizer_writes_after']
        assert type(before) is int and type(after) is int and before>=0
        assert type(successes) is int and 1<=successes<=20 and after-before==successes
        assert attempt_result['opportunities_processed']==20 and scalar['actor/update_opportunities_this_attempt']==20
        assert scalar['actor/optimizer_steps_this_update']==successes
        assert scalar['actor/skipped_update_opportunities']==20-successes
    else:
        assert scalar['actor/optimizer_steps_this_update']==20
    path=base/'logs/actual-updates.jsonl'
    prior=[json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    assert [row['step'] for row in prior]==list(range(1,step))
    assert all(row['run_id']==run['run_id'] for row in prior)
    from reproduction.common8.behavior_runtime import read_gate_receipt
    gate_args=dict(attempt_id=attempt_result['attempt_id'],attempt_index=attempt_result['attempt_index']) if inline else {}
    gate = read_gate_receipt(base, run, step,**gate_args)
    row=dict(run_id=run['run_id'],step=step,metrics=scalar,probe_version=probe_version,
             behavior_probability_gate_sha256=gate['proof_sha256'],
             behavior_trace_sha256=gate['private_trace_sha256'])
    if inline:
        row.update(attempt_id=attempt_result['attempt_id'],attempt_index=attempt_result['attempt_index'],
             optimizer_writes_before=before,optimizer_writes_after=after,opportunities_processed=20,
             actual_optimizer_writes=successes)
    with path.open('a') as f:
        f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
        parent_fd=os.open(path.parent,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
        try:os.fsync(parent_fd)
        finally:os.close(parent_fd)
    return row
