"""Common-data runtime gates, separate from the immutable old A800 experiment."""
import json
import math
import os
import re
import time
from pathlib import Path
from reproduction.a800.runtime import checkpoint_complete
from reproduction.common8.verify_data import digest, verify


def budget_rows(text, first, last):
    rows = []
    writes = []
    for line in text.splitlines():
        if 'SEARCH_G1_OPTIMIZER_WRITE=' in line:
            payload = line.split('SEARCH_G1_OPTIMIZER_WRITE=', 1)[1]
            writes.append(json.loads(re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', payload)))
        if 'SEARCH_G1_OUTER_BUDGET=' in line:
            payload = line.split('SEARCH_G1_OUTER_BUDGET=', 1)[1]
            payload = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', payload)
            rows.append(json.loads(payload))
    selected = [r for r in rows if first <= r['logical_update'] <= last]
    assert len(selected) == (last-first+1)*8
    for step in range(first, last+1):
        group = [r for r in selected if r['logical_update'] == step]
        assert sorted(r['rank'] for r in group) == list(range(8))
        for r in group:
            assert (r['local_trajectories'], r['local_mini'], r['local_micro'],
                    r['optimizer_writes']) == (320, 16, 2, 20)
            assert r['cumulative_writes'] == step*20
            actual = [w for w in writes if w['rank'] == r['rank']
                      and w['context']['logical_update'] == step]
            assert len(actual) == 20, 'Missing or duplicate actual optimizer write records'
            assert sorted(w['context']['minibatch_index'] for w in actual) == list(range(20))
            assert sorted(w['write_count'] for w in actual) == list(range((step-1)*20+1, step*20+1))
            assert all(w['learning_rates'] and all(lr==1e-6 for lr in w['learning_rates']) for w in actual)
    return selected


def accept_five_updates(checkpoint, records):
    import torch
    from safetensors import safe_open
    base = Path(os.environ['SEARCH_G1_BASE'])
    run = json.loads((base/'logs/run-manifest.json').read_text())
    assert run['profile'] == 'common8' and run.get('completed_updates', 0) == 0
    assert run['acceptance_update'] == 5
    data_proof = verify(base/'data')
    proof = json.loads((Path(checkpoint)/'complete.json').read_text())
    assert proof['run_id'] == run['run_id'] and proof['step'] == 5
    assert proof['finite_all_saved_tensors'] is True
    for name,value in proof['files_sha256'].items():
        assert digest(Path(checkpoint)/name) == value
    assert [r['step'] for r in records] == list(range(1,6))
    actual = [json.loads(line) for line in (base/'logs/actual-updates.jsonl').read_text().splitlines()]
    assert [r['step'] for r in actual] == list(range(1,6))
    from reproduction.common8.behavior_runtime import read_gate_receipt, verify_write_receipt_links
    behavior_gates = {str(r['step']): read_gate_receipt(base, run, r['step'], update_record=r) for r in actual}
    verify_write_receipt_links(base, behavior_gates, 1, 5)
    for r in records:
        assert r['probe_version'] == run['initial_probe_version']
        for key in ('critic/rewards/mean','actor/pg_loss','actor/entropy_loss',
                    'actor/kl_loss','actor/ppo_kl','actor/grad_norm'):
            assert math.isfinite(r['metrics'][key])
        assert r['metrics']['actor/optimizer_steps_this_update'] == 20
        saved_record = actual[r['step']-1]
        assert saved_record['run_id'] == run['run_id']
        assert saved_record['probe_version'] == r['probe_version']
        assert all(saved_record['metrics'][key] == float(r['metrics'][key]) for key in saved_record['metrics'])
    from reproduction.common8.worker_evidence import read_console_form
    budgets = budget_rows(read_console_form(base, run['run_id']), 1, 5)
    validation = json.loads((base/'logs/latest-stage-identity.json').read_text())
    assert validation['run_id'] == run['run_id'] and validation['step'] == 5
    assert validation['counts'] == dict(nq=32, hotpotqa=32)
    assert validation['validation_sha256'] == run['input_sha256']['stage.parquet']
    initial, saved = Path(run['initial_model']), Path(checkpoint)
    old = json.loads((initial/'model.safetensors.index.json').read_text())['weight_map']
    new = json.loads((saved/'model.safetensors.index.json').read_text())['weight_map']
    changed = []
    for name in old:
        with safe_open(str(initial/old[name]), framework='pt', device='cpu') as f:
            a = f.get_tensor(name)
        with safe_open(str(saved/new[name]), framework='pt', device='cpu') as f:
            z = f.get_tensor(name)
        if not torch.equal(a,z):
            changed.append(name)
        del a,z
    assert changed
    result = dict(run_id=run['run_id'], accepted=True, actual_updates=records,
        checkpoint=proof, changed_parameters=changed, budget_rows=budgets,
        shared_data=data_proof, stage_validation=validation, checked_unix=time.time(),
        behavior_probability_gates=behavior_gates,
        semantics='structural training gate, not accuracy reproduction success')
    with (base/'logs/five-updates-acceptance.json').open('x') as f:
        json.dump(result,f,default=lambda x:x.item())
    print('SEARCH_G1_FIVE_UPDATES_ACCEPTED='+run['run_id'],flush=True)
    return result


def refresh_after_checkpoint(step, checkpoint):
    # Common8 has an explicitly admitted probe device; never borrow A800 GPU6.
    if step not in (50,100,150,200,250):
        return
    import subprocess
    import requests
    base = Path(os.environ['SEARCH_G1_BASE'])
    run = json.loads((base/'logs/run-manifest.json').read_text())
    session = requests.Session()
    session.trust_env = False
    previous = json.loads((base/'artifacts/live/manifest.json').read_text())
    response = session.post('http://127.0.0.1:18012/suspend',
        json={'version':previous['version']},timeout=120)
    response.raise_for_status()
    released = response.json()
    assert released['suspended'] and released['version'] == previous['version']
    assert released['allocated_bytes_after_release'] < 256*1024*1024
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(run['probe_visible_device']))
    subprocess.run([str(base/'env/bin/python'), '-m', 'reproduction.common8.refit',
        '--step',str(step),'--model',str(checkpoint)],env=env,check=True)
    expected = json.loads((base/'artifacts/live/manifest.json').read_text())
    for _ in range(120):
        status = session.get('http://127.0.0.1:18012/health',timeout=10).json()
        if status.get('version') == expected['version'] and status.get('round') == step:
            print('SEARCH_G1_REFIT_ADOPTED='+expected['version'],flush=True)
            return expected['version']
        time.sleep(2)
    raise RuntimeError('New frozen evaluator not adopted; refuse next update')
