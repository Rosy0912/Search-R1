"""Durable per-rank scalar evidence, independent of Ray console forwarding."""
import json
import os
from pathlib import Path


def append(kind, payload):
    base=Path(os.environ['SEARCH_G1_BASE'])
    run=json.loads((base/'logs/run-manifest.json').read_text())
    assert run['profile']=='common8'
    rank=payload['rank'];assert rank in range(8)
    row=dict(kind=kind,run_id=run['run_id'],**payload)
    path=base/'logs'/f'optimizer-rank-{rank}.jsonl'
    fd=os.open(path,os.O_WRONLY|os.O_APPEND|os.O_CREAT,0o600)
    try:
        data=(json.dumps(row)+'\n').encode()
        written=os.write(fd,data)
        assert written==len(data)
        os.fsync(fd)
    finally:os.close(fd)


def read_console_form(base, run_id):
    run=json.loads((Path(base)/'logs/run-manifest.json').read_text())
    lines=[]
    for rank in range(8):
        path=Path(base)/'logs'/f'optimizer-rank-{rank}.jsonl'
        monitors=[];budgets=[]
        for line in path.read_text().splitlines():
            row=json.loads(line)
            assert row.pop('run_id')==run_id and row['rank']==rank
            kind=row.pop('kind')
            if kind=='POLICY_MONITOR':
                assert isinstance(row['allowed'],bool) and row['optimizer_writes']>=0
                monitors.append(row)
                continue
            if kind=='OUTER_BUDGET':budgets.append(row)
            assert kind in ('OPTIMIZER_WRITE','OUTER_BUDGET')
            lines.append('SEARCH_G1_'+kind+'='+json.dumps(row))
        if run.get('policy_safety_required',False):
            for budget in budgets:
                selected=[x for x in monitors if x['context']['logical_update']==budget['logical_update']]
                assert len(selected)==40,'Missing pre/post policy monitors for completed outer'
                start=budget['cumulative_writes']-20
                identity=selected[0]['monitor_identity'];calibration=selected[0]['calibration_sha256']
                assert len(identity['selection_indices'])==len(set(identity['selection_indices']))==4
                assert len(identity['actor_token_lengths'])==4 and all(x>0 for x in identity['actor_token_lengths'])
                for index,row in enumerate(selected):
                    mini=index//2;post=index%2
                    assert row['context']['minibatch_index']==mini
                    assert row['stage']==('after_minibatch' if post else 'before_minibatch')
                    assert row['optimizer_writes']==start+mini+post
                    assert row['allowed'] is True and row['kl_estimate_complete'] is True
                    assert row['calibration_sha256']==calibration and row['monitor_identity']==identity
                    assert row['objective_weight_definition']=='equal two-row micro masked_mean; weighted global numerator/denominator'
    return '\n'.join(lines)
