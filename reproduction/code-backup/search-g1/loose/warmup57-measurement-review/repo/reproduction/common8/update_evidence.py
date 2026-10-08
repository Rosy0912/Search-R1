"""Driver records only completed outer updates, after actual worker writes return."""
import json
import math
import os
from pathlib import Path
from reproduction.common8.formal_contract import MEASUREMENT_ROLE,TARGET_UPDATES,assert_scientific_run


METRICS=('critic/rewards/mean','actor/pg_loss','actor/entropy_loss',
         'actor/kl_loss','actor/ppo_kl','actor/grad_norm','actor/optimizer_steps_this_update')


def record(step,metrics,probe_version):
    base=Path(os.environ['SEARCH_G1_BASE'])
    run=json.loads((base/'logs/run-manifest.json').read_text())
    assert_scientific_run(run)
    assert type(step) is int and 1<=step<=TARGET_UPDATES
    if run.get('policy_calibration_role')==MEASUREMENT_ROLE:assert step<=2
    scalar={key:float(metrics[key]) for key in METRICS}
    assert all(math.isfinite(v) for v in scalar.values())
    assert scalar['actor/optimizer_steps_this_update']==20
    path=base/'logs/actual-updates.jsonl'
    prior=[json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    assert [row['step'] for row in prior]==list(range(1,step))
    assert all(row['run_id']==run['run_id'] for row in prior)
    from reproduction.common8.behavior_runtime import read_gate_receipt
    gate = read_gate_receipt(base, run, step)
    row=dict(run_id=run['run_id'],step=step,metrics=scalar,probe_version=probe_version,
             execution_role=run.get('policy_calibration_role','production'),
             behavior_probability_gate_sha256=gate['proof_sha256'],
             behavior_trace_sha256=gate['private_trace_sha256'])
    with path.open('a') as f:
        f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
    return row
