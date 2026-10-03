"""Validated G1 checkpoint and deterministic evaluator refresh boundaries."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def checkpoint_complete(step, checkpoint):
    from verl.utils.checkpoint_integrity import validate_saved_checkpoint
    base=Path(os.environ['SEARCH_G1_BASE']);manifest=json.loads((base/'logs/run-manifest.json').read_text())
    count=validate_saved_checkpoint(Path(checkpoint),Path(manifest['initial_model']),manifest['started_unix'])
    index=json.loads((Path(checkpoint)/'model.safetensors.index.json').read_text())
    hashes={}
    for name in set(index['weight_map'].values())|{'config.json','model.safetensors.index.json'}:
        h=hashlib.sha256()
        with (Path(checkpoint)/name).open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
        hashes[name]=h.hexdigest()
    result=dict(run_id=manifest['run_id'],step=step,validated_tensor_count=count,
                finite_all_saved_tensors=True,checkpoint=str(checkpoint),files_sha256=hashes)
    path=Path(checkpoint)/'complete.json';tmp=path.with_suffix('.json.tmp');tmp.write_text(json.dumps(result,indent=2));os.replace(tmp,path)
    return result


def refresh_after_checkpoint(step,checkpoint):
    # Fresh5 is acceptance only; no extra calibration after the terminal300.
    if step not in (50,100,150,200,250):return
    base=Path(os.environ['SEARCH_G1_BASE'])
    print('SEARCH_G1_REFIT_PAUSE='+str(step),flush=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='6')
    subprocess.run([str(base/'env/bin/python'),str(base/'repo/reproduction/a800/refit.py'),
                    '--step',str(step),'--model',str(checkpoint)],env=env,check=True)
    import requests
    session=requests.Session();session.trust_env=False
    expected=json.loads((base/'artifacts/live/manifest.json').read_text())
    for _ in range(120):
        status=session.get('http://127.0.0.1:18012/health',timeout=10).json()
        if status.get('version')==expected['version'] and status.get('round')==step:
            print('SEARCH_G1_REFIT_ADOPTED='+expected['version'],flush=True);return expected['version']
        time.sleep(2)
    raise RuntimeError('Calibrated evaluator not adopted; refuse next GRPO update')


def accept_five_updates(checkpoint, records):
    """Block update6 until real metrics, fixed64 and changed full weights pass."""
    import math
    import torch
    from safetensors import safe_open
    base=Path(os.environ['SEARCH_G1_BASE'])
    run=json.loads((base/'logs/run-manifest.json').read_text())
    from reproduction.a800.prepare_run import digest
    for name in ('validation-64.parquet','audit.json'):
        assert digest(base/'data'/name)==run['input_sha256'][name], 'Validation inputs changed since admission'
    proof=json.loads((Path(checkpoint)/'complete.json').read_text())
    assert proof['run_id']==run['run_id'] and proof['step']==5
    assert [r['step'] for r in records]==[1,2,3,4,5], 'Missing actual GRPO update'
    for r in records:
        for field in ('critic/rewards/mean','actor/pg_loss','actor/entropy_loss','actor/kl_loss','actor/ppo_kl','actor/grad_norm'):
            assert math.isfinite(r['metrics'][field]), 'Nonfinite five-update metric: '+field
        assert r['probe_version']==run['initial_probe_version'], 'Wrong evaluator identity'
    counts={k.split('/')[-1]:int(v) for k,v in records[-1]['metrics'].items() if k.startswith('val/evaluated_count/')}
    expected=json.loads((base/'data/audit.json').read_text())['validation_source_counts']
    assert counts==expected and sum(counts.values())==64, 'Fixed validation coverage mismatch'
    initial=Path(run['initial_model']);saved=Path(checkpoint)
    old=json.loads((initial/'model.safetensors.index.json').read_text())['weight_map']
    new=json.loads((saved/'model.safetensors.index.json').read_text())['weight_map']
    changed=[]
    for key in old:
        with safe_open(str(initial/old[key]),framework='pt',device='cpu') as f: a=f.get_tensor(key)
        with safe_open(str(saved/new[key]),framework='pt',device='cpu') as f: z=f.get_tensor(key)
        if not torch.equal(a,z):changed.append(key)
        del a,z
    assert changed, 'Checkpoint5 has no parameter update'
    result=dict(run_id=run['run_id'],accepted=True,actual_updates=records,validation_counts=counts,
                changed_parameters=changed,checkpoint=proof,checked_unix=time.time())
    target=base/'logs/five-updates-acceptance.json';tmp=target.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(result,indent=2,default=lambda x:x.item()));os.replace(tmp,target)
    print('SEARCH_G1_FIVE_UPDATES_ACCEPTED='+run['run_id'],flush=True)
    return result
