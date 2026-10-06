"""Common-data runtime gates, separate from the immutable old A800 experiment."""
import json
import math
import os
import re
import time
import hashlib
from pathlib import Path
from reproduction.a800.runtime import checkpoint_complete
from reproduction.common8.verify_data import digest, verify
from reproduction.common8.formal_contract import (
    CONTRACT_KIND, PROBE_ROUNDS, TARGET_UPDATES, WARMUP_STEPS,
    assert_formal_run, consumed_lr,
)
from reproduction.common8.policy_calibration_plan import canonical_bytes


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
            expected_lr = consumed_lr(step)
            assert all(w['learning_rates'] and all(type(lr) in (int,float) and lr==expected_lr
                       for lr in w['learning_rates']) for w in actual), 'Consumed warmup LR differs'
    return selected


def verify_no_fatal_scheduler_console(base, run):
    """Reject a console fallback even if readable ledger bytes look complete.

    A failed fsync may leave a COMPLETE line visible. If the fatal ledger
    append also fails, the real worker emits this independent console signal.
    The observed hash is historical; it cannot prove future log contents.
    """
    base=Path(base).resolve();path=Path(run['train_log'])
    assert path.is_absolute() and path.is_file() and not path.is_symlink()
    assert path.resolve().is_relative_to(base), 'Training console is outside this owned run'
    raw=path.read_bytes()
    assert b'SEARCH_G1_FATAL_SCHEDULER_EVIDENCE=' not in raw, \
        'Fatal scheduler console fallback invalidates readable completion receipts'
    return dict(console_path=str(path),observed_sha256=hashlib.sha256(raw).hexdigest(),
                scope='no fatal scheduler fallback in the current owned console snapshot')


def verify_scheduler_receipts(base, run, first, last, behavior_gates):
    """Bind each complete outer to actual durable rank counters and consumed LR."""
    assert_formal_run(run)
    assert 1 <= first <= last <= TARGET_UPDATES
    base = Path(base)
    console_check=verify_no_fatal_scheduler_console(base,run)
    claim = json.loads((base/'logs/training-launch-claim.json').read_text())
    assert claim['run_id'] == run['run_id']
    owned = json.loads((base/'logs/process-ledger.json').read_text())
    assert owned['run_id']==run['run_id']
    trainer=owned['jobs']['trainer']
    leader=trainer['leader']
    assert (leader['pid'],leader['start_ticks'])==(claim['shell_pid'],claim['shell_start_ticks'])
    admitted_births={(r['pid'],r['start_ticks']) for r in trainer['ledger'].values()
                     if r['sid']==leader['sid'] and r['uid']==leader['uid']}
    source_map_sha = hashlib.sha256(canonical_bytes(run['source_sha256'])).hexdigest()
    contract_sha = hashlib.sha256(canonical_bytes(run['budget'])).hexdigest()
    result = {}
    rank_processes=set()
    for rank in range(8):
        path = base/'logs'/f'scheduler-rank-{rank}.jsonl'
        raw = path.read_bytes()
        rows = [json.loads(line) for line in raw.splitlines()]
        selected = [r for r in rows if first <= r['logical_update'] <= last]
        assert [(r['logical_update'], r['kind']) for r in selected] == [
            (step, kind) for step in range(first,last+1) for kind in ('BEGIN','COMMIT_PREPARED','COMPLETE')], \
            'Missing, duplicate, rejected or reordered scheduler outer receipts'
        process = None
        previous_unix = 0.
        for row in selected:
            step, complete = row['logical_update'], row['kind']=='COMPLETE'
            has_writes = row['kind'] in ('COMMIT_PREPARED','COMPLETE')
            for name in ('logical_update','rank','world_size','warmup_steps','completed_outer_before',
                         'completed_outer_after','scheduler_last_epoch_before','scheduler_last_epoch_after',
                         'scheduler_step_count_before','scheduler_step_count_after','writes_before',
                         'writes_after','actual_optimizer_writes'):
                assert type(row[name]) is int, 'Noninteger actual scheduler counter: '+name
            assert row['schema_version']==1 and row['role']=='fresh200_outer_scheduler'
            assert row['contract_kind']==CONTRACT_KIND and row['run_id']==run['run_id']
            assert row['source_commit']==run['source_commit'] and row['source_map_sha256']==source_map_sha
            assert row['scheduler_contract_sha256']==contract_sha
            assert row['slurm_job_id']==claim['slurm_job_id']
            assert row['rank']==rank and row['world_size']==8
            assert row['warmup_steps']==WARMUP_STEPS and row['base_learning_rate']==1e-6
            assert type(row['observed_unix']) in (int,float) and math.isfinite(row['observed_unix'])
            assert row['observed_unix'] >= previous_unix and row['observed_unix'] > 0
            previous_unix = row['observed_unix']
            identity = (row['process_pid'], row['process_start_ticks'])
            assert all(type(v) is int and v>0 for v in identity)
            if process is None: process=identity
            assert identity==process, 'Scheduler rank process changed inside the admitted segment'
            assert identity in admitted_births, 'Scheduler process birth absent from independent owned trainer ledger'
            assert row['batch_gate_sha256']==behavior_gates[str(step)]['proof_sha256']
            expected_after = step if complete else step-1
            assert row['completed_outer_before']==step-1 and row['completed_outer_after']==expected_after
            assert row['scheduler_last_epoch_before']==step-1 and row['scheduler_last_epoch_after']==expected_after
            # Torch LambdaLR performs its constructor initial step. Record the
            # actual internal counter rather than pretending it starts at zero.
            assert row['scheduler_step_count_before']==step and row['scheduler_step_count_after']==expected_after+1
            assert row['writes_before']==(step-1)*20 and row['writes_after']==((step if has_writes else step-1)*20)
            assert row['actual_optimizer_writes']==(20 if has_writes else 0)
            assert type(row['scheduler_advances']) is int and row['scheduler_advances']==int(complete)
            consumed = consumed_lr(step)
            next_lr = 1e-6*min(expected_after/WARMUP_STEPS,1.0)
            rates, following = row['consumed_learning_rates'], row['next_learning_rates']
            assert rates and len(rates)==len(following)
            assert all(type(lr) in (int,float) and lr==consumed for lr in rates)
            assert all(type(lr) in (int,float) and lr==next_lr for lr in following)
        prefix = b'\n'.join(canonical_bytes(r) for r in selected)+b'\n'
        assert process not in rank_processes, 'Different scheduler ranks claim the same actual process birth'
        rank_processes.add(process)
        result[str(rank)] = dict(receipt_path=str(path), selected_rows=selected,
            fatal_console_check=console_check,
            selected_rows_sha256=hashlib.sha256(prefix).hexdigest(),
            observed_file_sha256=hashlib.sha256(raw).hexdigest(),
            owned_process_ledger_sha256=digest(base/'logs/process-ledger.json'),
            semantics='selected completed outer receipts bound to independently recorded trainer births; '
                      'observed files may subsequently append; historical birth binding is not a current liveness claim')
    return result


def accept_five_updates(checkpoint, records):
    import torch
    from safetensors import safe_open
    base = Path(os.environ['SEARCH_G1_BASE'])
    run = json.loads((base/'logs/run-manifest.json').read_text())
    assert_formal_run(run)
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
    scheduler_receipts = verify_scheduler_receipts(base, run, 1, 5, behavior_gates)
    from reproduction.common8.gpu_training_admission import verify_training_gpu_samples
    gpu_samples = verify_training_gpu_samples(base, run, scheduler_receipts)
    validation = json.loads((base/'logs/latest-stage-identity.json').read_text())
    assert validation['run_id'] == run['run_id'] and validation['step'] == 5
    assert validation['counts'] == dict(nq=32, hotpotqa=32)
    assert validation['validation_sha256'] == run['input_sha256']['stage.parquet']
    initial, saved = Path(run['initial_model']), Path(checkpoint)
    from verl.utils.checkpoint_integrity import validate_saved_checkpoint
    fresh_tensor_count = validate_saved_checkpoint(saved, initial, run['started_unix'])
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
        independently_verified_saved_tensor_count=fresh_tensor_count,
        shared_data=data_proof, stage_validation=validation, checked_unix=time.time(),
        behavior_probability_gates=behavior_gates,
        scheduler_receipts=scheduler_receipts,
        actual_first_five_gpu_occupancy=gpu_samples,
        first_outer_semantics='20 real optimizer writes at consumed LR0; fresh5 actual parameter delta verified',
        semantics='structural training gate, not accuracy reproduction success')
    with (base/'logs/five-updates-acceptance.json').open('x') as f:
        json.dump(result,f,default=lambda x:x.item())
    print('SEARCH_G1_FIVE_UPDATES_ACCEPTED='+run['run_id'],flush=True)
    return result


def refresh_after_checkpoint(step, checkpoint):
    # Common8 has an explicitly admitted probe device; never borrow A800 GPU6.
    if step not in PROBE_ROUNDS or step == 0:
        return
    import subprocess
    import requests
    base = Path(os.environ['SEARCH_G1_BASE'])
    run = json.loads((base/'logs/run-manifest.json').read_text())
    assert_formal_run(run)
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
