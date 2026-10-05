"""Bounded child measurements and complete-write evidence, never health admission."""
import json
import os
import time
import hashlib
from collections import Counter
from pathlib import Path

from reproduction.common8.policy_calibration_plan import load_plan
from reproduction.common8.verify_data import digest


def training_input_rows(batch):
    """Hash actual loaded parent rows without publishing question or token text."""
    from reproduction.common8.policy_calibration_plan import canonical_bytes
    rows = []
    for index in range(len(batch)):
        tensors = {}
        for key in ('input_ids', 'attention_mask', 'position_ids'):
            value = batch.batch[key][index].detach().cpu().contiguous()
            metadata = canonical_bytes(dict(dtype=str(value.dtype), shape=list(value.shape)))
            tensors[key] = hashlib.sha256(metadata + value.numpy().tobytes()).hexdigest()
        # Dataset index is the unchanged upstream GRPO group key. No new UID,
        # deduplication or repartitioning is introduced for this measurement.
        parent = batch.non_tensor_batch['index'][index]
        if hasattr(parent, 'item'):
            parent = parent.item()
        source = batch.non_tensor_batch['data_source'][index]
        rows.append(dict(data_source=str(source), tensor_sha256=tensors,
                         upstream_group_sha256=hashlib.sha256(canonical_bytes(parent)).hexdigest()))
    return rows


def training_inputs(batch, phase, parents=None):
    """Observe the unchanged 512-parent, interleaved G5 expansion before generation."""
    base, run, plan, sha = phase
    rows = training_input_rows(batch)
    if parents is None:
        assert len(rows) == 512, 'Actual child outer does not contain 512 loaded parents'
        return rows
    assert len(parents) == 512 and len(rows) == 2560
    assert rows == [row for row in parents for _ in range(5)], 'Actual G5 parent expansion differs'
    result = dict(run_id=run['run_id'], plan_sha256=sha, slurm_job_id=plan['slurm_job_id'],
                  actual_parent_count=512, actual_trajectory_count=2560, n_agent=5,
                  train_sha256=run['input_sha256']['train.parquet'], parents=parents,
                  repeated_group_sha256=[row['upstream_group_sha256'] for row in rows],
                  production_admission=False, checked_unix=time.time())
    write_once(base / 'logs/policy-calibration-training-inputs.json', result)
    return result


def training_groups_before_update(batch, phase):
    """Confirm balancing retained every repeated upstream group before Adam."""
    from reproduction.common8.policy_calibration_plan import canonical_bytes
    base, run, plan, sha = phase
    saved = json.loads((base / 'logs/policy-calibration-training-inputs.json').read_text())
    assert saved['run_id'] == run['run_id'] and saved['plan_sha256'] == sha and len(batch) == 2560
    actual = []
    for value in batch.non_tensor_batch['uid']:
        if hasattr(value, 'item'):
            value = value.item()
        actual.append(hashlib.sha256(canonical_bytes(value)).hexdigest())
    assert Counter(actual) == Counter(saved['repeated_group_sha256']), 'Actual balanced GRPO groups changed'
    write_once(base / 'logs/policy-calibration-training-groups.json',
               dict(run_id=run['run_id'], plan_sha256=sha, actual_trajectory_count=len(batch),
                    balanced_group_sha256=actual, upstream_group_multiset_unchanged=True,
                    production_admission=False))


def active_plan():
    """Only an explicitly bound child manifest changes the production flow."""
    if os.environ.get('SEARCH_G1_PROFILE') != 'common8':
        return None
    base = Path(os.environ['SEARCH_G1_BASE'])
    run = json.loads((base / 'logs/run-manifest.json').read_text())
    if 'policy_calibration_role' not in run:
        return None
    plan, sha = load_plan(base, run)
    return base, run, plan, sha


def write_once(path, document):
    path = Path(path)
    with path.open('x') as f:
        json.dump(document, f, allow_nan=False, indent=2)
        f.flush(); os.fsync(f.fileno())
    fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def verify_optimizer_states(proofs, run, plan, sha, allocation):
    """Bind eight independently written, actual post-Adam summaries.

    This checks the summaries' identity and counters, not JSON as a substitute
    for a GPU finite check. Their producer checks the actual tensors collectively
    before it fsyncs these files and before it emits a completed outer.
    """
    assert set(proofs) == set(range(8)) and all(type(r) is int for r in proofs)
    assert allocation['accepted'] is True and allocation['run_id'] == run['run_id']
    assert allocation['slurm_job_id'] == plan['slurm_job_id']
    expected = allocation['actual_cuda_runtime_uuids']
    assert len(expected) == len(set(expected)) == 8
    devices, pids = [], []
    for rank in range(8):
        row = proofs[rank]
        assert row['run_id'] == run['run_id'] and row['execution_role'] == plan['role']
        assert row['parent_production_run_id'] == plan['parent_production_run_id']
        assert row['plan_sha256'] == sha and row['slurm_job_id'] == plan['slurm_job_id']
        assert type(row['rank']) is int and row['rank'] == rank
        for key, value in (('initial_optimizer_writes', 0), ('initial_adam_state_entries', 0),
                           ('final_optimizer_writes', 20)):
            assert type(row[key]) is int and row[key] == value, 'Wrong actual Adam counter: ' + key
        for key in ('pid', 'start_ticks', 'parameter_tensor_count', 'parameter_numel',
                    'adam_state_entries', 'adam_state_tensor_count', 'adam_state_numel'):
            assert type(row[key]) is int and row[key] > 0, 'Missing actual state count: ' + key
        assert row['adam_state_tensor_count'] >= 3 * row['adam_state_entries']
        steps = row['adam_steps']
        assert isinstance(steps, list) and len(steps) == row['adam_state_entries']
        assert all(type(n) in (int, float) and n == 20 for n in steps), 'Actual Adam steps differ'
        assert row['all_parameters_and_adam_finite'] is True
        assert row['cuda_uuid'] in expected
        devices.append(row['cuda_uuid']); pids.append(row['pid'])
    assert set(devices) == set(expected) and len(set(pids)) == 8, 'Duplicate actual worker/device'
    return [proofs[r] for r in range(8)]


def verify_live_optimizer_workers(states, base, job_id):
    """The returned Ray actors must still have these actual process births."""
    from reproduction.a800.owned_processes import identity
    observed = []
    for row in states:
        current = identity(row['pid'])
        assert current['start_ticks'] == row['start_ticks'] and current['state'] != 'Z'
        assert current['uid'] == os.getuid(), 'Actual calibration worker user differs'
        environment = (Path('/proc') / str(row['pid']) / 'environ').read_bytes().split(b'\0')
        assert ('SEARCH_G1_BASE=' + str(base)).encode() in environment
        assert ('SLURM_JOB_ID=' + job_id).encode() in environment
        observed.append(dict(pid=current['pid'], start_ticks=current['start_ticks'],
                             sid=current['sid'], uid=current['uid'], state=current['state']))
    return observed


def evaluation(base, run, plan, sha, step, rows, original_rewards):
    """Independent scalar measurement must agree with the unchanged strict EM."""
    from reproduction.common8.calibration_evaluation import aggregate
    from reproduction.common8.validation_identity import expected
    assert type(step) is int and step in (0, 1)
    record, frame, ids = expected(base, 'validation')
    result = aggregate(rows, ids, frame['data_source'].tolist())
    assert [r['em'] for r in rows] == list(original_rewards), 'Calibration measurement differs from original EM'
    result.update(run_id=run['run_id'], parent_production_run_id=plan['parent_production_run_id'],
                  slurm_job_id=plan['slurm_job_id'], plan_sha256=sha, step=step,
                  phase_role=plan['role'], validation_sha256=record['sha256'], checked_unix=time.time())
    name = 'before' if step == 0 else 'after'
    write_once(base / 'logs' / ('policy-calibration-evaluation-' + name + '.json'), result)
    return result


def finish_after_update(trainer, phase):
    """Save and evaluate the one completed child outer before final51713 logic.

    This function reports measurements and structural execution evidence. It
    never selects thresholds, interprets a health rule or admits production.
    Failed post-evaluation/save retains real worker writes and exits with error.
    """
    base, run, plan, sha = phase
    assert trainer.total_training_steps == 301, 'Keep the original scheduler horizon'
    assert trainer.global_steps == plan['phase_budget']['outer_count'] == 1
    assert plan['phase_budget']['final_evaluation_in_this_phase'] is False
    before_path = base / 'logs/policy-calibration-evaluation-before.json'
    before = json.loads(before_path.read_text())
    assert before['run_id'] == run['run_id'] and before['step'] == 0 and before['plan_sha256'] == sha
    trainer._save_checkpoint()
    from reproduction.common8.runtime import checkpoint_complete
    checkpoint = Path(trainer.config.trainer.default_local_dir) / 'actor/global_step_1'
    saved = checkpoint_complete(1, str(checkpoint))
    trainer._validate()  # Phase dispatch uses all512; never final=True.
    after_path = base / 'logs/policy-calibration-evaluation-after.json'
    after = json.loads(after_path.read_text())
    assert after['question_ids'] == before['question_ids'] and after['data_sources'] == before['data_sources']
    assert after['run_id'] == run['run_id'] and after['step'] == 1 and after['plan_sha256'] == sha
    assert after['validation_sha256'] == before['validation_sha256'] == run['input_sha256']['validation.parquet']
    state_paths = {rank: base / 'logs' / f'policy-calibration-optimizer-rank-{rank}.json'
                   for rank in range(8)}
    states = {rank: json.loads(path.read_text()) for rank, path in state_paths.items()}
    allocation = json.loads((base / 'logs/native-allocation-admission.json').read_text())
    states = verify_optimizer_states(states, run, plan, sha, allocation)
    live_workers = verify_live_optimizer_workers(states, base, plan['slurm_job_id'])
    input_path = base / 'logs/policy-calibration-training-inputs.json'
    group_path = base / 'logs/policy-calibration-training-groups.json'
    actual_inputs, actual_groups = json.loads(input_path.read_text()), json.loads(group_path.read_text())
    for document in (actual_inputs, actual_groups):
        assert document['run_id'] == run['run_id'] and document['plan_sha256'] == sha
        assert document['actual_trajectory_count'] == 2560
    assert actual_inputs['actual_parent_count'] == 512 and actual_inputs['n_agent'] == 5
    assert len(actual_inputs['parents']) == 512
    assert actual_inputs['train_sha256'] == run['input_sha256']['train.parquet']
    expected_groups = [row['upstream_group_sha256'] for row in actual_inputs['parents'] for _ in range(5)]
    assert actual_inputs['repeated_group_sha256'] == expected_groups
    assert actual_groups['upstream_group_multiset_unchanged'] is True
    assert Counter(actual_groups['balanced_group_sha256']) == Counter(expected_groups)
    from reproduction.common8.safety_calibration_evidence import verify_segment
    logs = {rank: [json.loads(line) for line in
                  (base / 'logs' / f'optimizer-rank-{rank}.jsonl').read_text().splitlines()]
            for rank in range(8)}
    structural = verify_segment(logs, dict(schema_version=1, role=plan['role'], run_id=run['run_id'],
                              calibration_sha256=sha, outer_count=1, initial_optimizer_writes=0))
    updates = [json.loads(line) for line in (base / 'logs/actual-updates.jsonl').read_text().splitlines()]
    assert len(updates) == 1 and updates[0]['step'] == 1 and updates[0]['run_id'] == run['run_id']
    assert updates[0]['probe_version'] == run['initial_probe_version']
    assert updates[0]['metrics']['actor/optimizer_steps_this_update'] == 20
    # Current frozen probe remains the one used to generate the actual rewards.
    live = json.loads((base / 'artifacts/live/manifest.json').read_text())
    assert live['version'] == run['initial_probe_version'] and live['round'] == 0
    assert live['head_sha256'] == plan['initial_reward_fingerprint']['head_sha256']
    assert live['encoder_sha256'] == plan['initial_reward_fingerprint']['encoder_sha256']
    for name, value in saved['files_sha256'].items():
        assert digest(checkpoint / name) == value
    assert saved['finite_all_saved_tensors'] is True
    assert saved['run_id'] == run['run_id'] and saved['step'] == 1
    # The canonical checker above covers every serialized key/shape/finite value.
    import torch
    from safetensors import safe_open
    initial = Path(run['initial_model'])
    old = json.loads((initial / 'model.safetensors.index.json').read_text())['weight_map']
    new = json.loads((checkpoint / 'model.safetensors.index.json').read_text())['weight_map']
    changed = []
    for key in old:
        with safe_open(str(initial / old[key]), framework='pt', device='cpu') as f:
            a = f.get_tensor(key)
        with safe_open(str(checkpoint / new[key]), framework='pt', device='cpu') as f:
            z = f.get_tensor(key)
        if not torch.equal(a, z): changed.append(key)
        del a, z
    assert changed, 'Completed child checkpoint has no actual parameter delta'
    result = dict(role=plan['role'] + '_evidence', origin='actual_native_policy_segment',
                  run_id=run['run_id'], parent_production_run_id=plan['parent_production_run_id'],
                  slurm_job_id=plan['slurm_job_id'], plan_sha256=sha, identity=plan['identity'],
                  initial_reward_fingerprint=plan['initial_reward_fingerprint'],
                  resolved_config_sha256=plan['resolved_config_sha256'], ranks=list(range(8)),
                  completed_outer_updates=1, optimizer_writes_per_rank=20,
                  structural=structural, checkpoint=saved, changed_parameters=changed,
                  actual_optimizer_states=states,
                  actual_worker_births_at_segment_close=live_workers,
                  optimizer_state_proof_sha256={str(r): digest(path) for r, path in state_paths.items()},
                  actual_training_inputs=actual_inputs, actual_training_groups=actual_groups,
                  training_input_proof_sha256=dict(parents=digest(input_path), groups=digest(group_path)),
                  worker_log_sha256={str(r): digest(base / 'logs' / f'optimizer-rank-{r}.jsonl') for r in range(8)},
                  evaluation_before=before, evaluation_after=after,
                  evaluation_sha256=dict(before=digest(before_path), after=digest(after_path)),
                  actual_updates=updates, production_admission=False,
                  healthy_checkpoint_accepted=False, checked_unix=time.time(),
                  semantics='complete child measurements; health and threshold decision require separate frozen review')
    write_once(base / 'logs/policy-calibration-segment.json', result)
    print('SEARCH_G1_CALIBRATION_SEGMENT_MEASURED=' + run['run_id'], flush=True)
    return result
