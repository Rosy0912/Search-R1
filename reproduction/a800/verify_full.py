"""Verify actual G1 updates1..300, fresh finite full weights and fixed64 final EM."""
import json
import math
import os
from pathlib import Path
import re
import time
from reproduction.a800.prepare_run import digest
from verl.utils.checkpoint_integrity import validate_saved_checkpoint


def actual_updates(text):
    text=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',text)
    rows=[];version=None
    for line in text.splitlines():
        if 'SEARCH_G1_PROBE_VERSION=' in line:
            version=line.split('SEARCH_G1_PROBE_VERSION=',1)[1].strip()
        match=re.search(r'\bstep:(\d+) - (.*)',line)
        if match and 'critic/rewards/mean:' in match[2]:
            values={k:float(v) for k,v in re.findall(r'([\w/]+):(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|nan|-?inf)',match[2])}
            rows.append(dict(step=int(match[1]),metrics=values,probe_version=version))
    assert [r['step'] for r in rows]==list(range(1,301)), 'Missing/duplicate/extra actual update'
    for row in rows:
        for key in ('critic/rewards/mean','actor/pg_loss','actor/entropy_loss','actor/kl_loss','actor/ppo_kl','actor/grad_norm'):
            assert key in row['metrics'] and math.isfinite(row['metrics'][key]), 'Invalid metric at '+str(row['step'])+':'+key
    return rows


def final_validation(text, expected):
    lines=[x.split('SEARCH_G1_FINAL_METRICS_JSON=',1)[1] for x in text.splitlines() if 'SEARCH_G1_FINAL_METRICS_JSON=' in x]
    assert len(lines)==1, 'Missing/duplicate final validation'
    metrics=json.loads(lines[0])
    raw_counts={k.split('/')[-1]:v for k,v in metrics.items() if k.startswith('val/evaluated_count/')}
    assert all(not isinstance(v,bool) and isinstance(v,(int,float)) and math.isfinite(v) and v>=0 and v==int(v) for v in raw_counts.values()), 'Invalid noninteger validation count'
    counts={k:int(v) for k,v in raw_counts.items()}
    assert counts==expected and sum(counts.values())==64, 'Incomplete fixed64 final validation'
    scores={k.split('/')[-1]:float(v) for k,v in metrics.items() if k.startswith('val/test_score/')}
    assert set(scores)==set(expected)
    assert all(math.isfinite(v) and 0<=v<=1 for v in scores.values())
    return dict(counts=counts,source_em=scores,weighted_em=sum(scores[k]*counts[k] for k in scores)/64)


def bind_probe_versions(rows, run_id, initial, adopted):
    assert len(adopted)==5
    assert [v.split(':')[1] for v in adopted]==['50','100','150','200','250']
    assert all(v.startswith(run_id+':') for v in adopted)
    mapping={0:initial,**{int(v.split(':')[1]):v for v in adopted}}
    for row in rows:
        round_step=((row['step']-1)//50)*50
        assert row['probe_version']==mapping[round_step], 'Wrong exact reward evaluator at update '+str(row['step'])


def verify(base):
    run=json.loads((base/'logs/run-manifest.json').read_text())
    log=Path(run['train_log']);text=log.read_text(errors='replace')
    assert text.splitlines()[0]=='SEARCH_G1_RUN '+run['run_id']
    initial=json.loads((base/'logs/initial-probes-accepted.json').read_text())
    five=json.loads((base/'logs/five-updates-acceptance.json').read_text())
    assert initial['accepted'] is True and five['accepted'] is True
    assert initial['run_id']==five['run_id']==run['run_id']
    assert initial['version']==run['initial_probe_version']
    assert [x['step'] for x in five['actual_updates']]==[1,2,3,4,5]
    assert all(x['probe_version']==run['initial_probe_version'] for x in five['actual_updates'])
    for name,expected in run['input_sha256'].items():assert digest(base/'data'/name)==expected
    for name,expected in run['source_sha256'].items():assert digest(base/'repo'/name)==expected
    rows=actual_updates(text)
    adopted=re.findall(r'SEARCH_G1_REFIT_ADOPTED=([^\s]+)',text)
    bind_probe_versions(rows,run['run_id'],run['initial_probe_version'],adopted)
    validation=final_validation(text,run['validation_source_counts'])
    checkpoint=Path(run['checkpoint_dir'])/'actor/global_step_300'
    proof=json.loads((checkpoint/'complete.json').read_text())
    assert proof['run_id']==run['run_id'] and proof['step']==300
    for name,h in proof['files_sha256'].items():assert digest(checkpoint/name)==h
    count=validate_saved_checkpoint(checkpoint,Path(run['initial_model']),run['started_unix'])
    assert count==proof['validated_tensor_count']
    # Preserve the original checkpoint marker; final validation never replaces its hashes.
    # Every requested saved boundary must have a complete same-run proof.
    saved=[]
    for step in (5,50,100,150,200,250,300):
        folder=Path(run['checkpoint_dir'])/'actor'/f'global_step_{step}'
        complete=json.loads((folder/'complete.json').read_text())
        assert complete['run_id']==run['run_id'] and complete['step']==step and complete['finite_all_saved_tensors'] is True
        for name,h in complete['files_sha256'].items():assert digest(folder/name)==h
        saved.append(step)
    result=dict(run_id=run['run_id'],accepted=True,actual_update_count=len(rows),saved_steps=saved,
                fresh_final_checkpoint=proof,fixed64_final_validation=validation,adopted_refit_versions=adopted,
                five_update_acceptance_sha256=digest(base/'logs/five-updates-acceptance.json'),
                training_console_precision='three decimal places; not raw full precision',checked_unix=time.time())
    p=base/'logs/full-run-acceptance.json';temp=p.with_suffix('.json.tmp');temp.write_text(json.dumps(result,indent=2));os.replace(temp,p)
    print('SEARCH_G1_FULL300_ACCEPTED='+run['run_id'],flush=True)
    return result


if __name__=='__main__':verify(Path(os.environ.get('SEARCH_G1_BASE','/data/search-g1')))
