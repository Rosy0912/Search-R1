"""Native driver/worker binding for the actual common8 behavior denominator.

The context reader runs before any old/reference forward. The probability
reader runs on every retained model token before Adam. Published receipts contain
only scalar/hash evidence; raw token journals remain private on the native node.
"""
import hashlib
import json
import os
import uuid
from pathlib import Path

import numpy as np
import torch

from reproduction.common8.behavior_trace import (
    audit_outer, make_actor_row, make_trace, write_private_evidence,
)
from reproduction.common8.policy_calibration_plan import canonical_bytes, document_sha256


TOKEN_FIELDS = frozenset(('input_ids', 'prompts', 'responses', 'responses_with_info_mask',
                          'attention_mask', 'position_ids', 'info_mask', 'loss_mask'))


def cast_token_fields(batch):
    """Do not integer-cast actual selected probabilities or numerical metrics."""
    for name in TOKEN_FIELDS.intersection(batch.batch.keys()):
        batch.batch[name] = batch.batch[name].long()


def rollout_logprob_metadata(rollout):
    """Read actual per-rank config after the worker's global32 / world8 split."""
    values = dict(micro_batch_size=rollout.log_prob_micro_batch_size,
                  temperature=rollout.temperature,
                  use_dynamic_bsz=rollout.log_prob_use_dynamic_bsz,
                  max_token_len=rollout.log_prob_max_token_len_per_gpu)
    assert type(values['micro_batch_size']) is int and values['micro_batch_size'] == 4
    assert type(values['temperature']) in (int, float) and values['temperature'] == 1.0
    assert values['use_dynamic_bsz'] is False and type(values['max_token_len']) is int
    return values


def _native_run():
    base = Path(os.environ['SEARCH_G1_BASE'])
    assert base.is_absolute() and base.resolve() == base and str(base).startswith('/mnt/data/shaoshitong/')
    run = json.loads((base / 'logs/run-manifest.json').read_text())
    assert run['profile'] == 'common8' and run['completed_updates'] == 0
    job = os.environ['SLURM_JOB_ID']
    assert job.isdigit() and int(job) > 0
    # This is a current disk-source identity check, not proof of loaded modules.
    for name, expected in run['source_sha256'].items():
        path = base / 'repo' / name
        assert path.resolve() == path and hashlib.sha256(path.read_bytes()).hexdigest() == expected
    return base, run, job


def prepare_identity(parent_batch, logical_update, *, repeats, validation=False):
    base, run, job = _native_run()
    count = len(parent_batch)
    assert type(logical_update) is int and logical_update >= 0
    assert repeats == (1 if validation else 5) and (validation or count == 512)
    assert validation or logical_update >= 1
    parent_ids = []
    for index, group in enumerate(parent_batch.non_tensor_batch['index']):
        if hasattr(group, 'item'):
            group = group.item()
        parent_ids.append(str(index) + '-' + document_sha256(group))
    ids = [run['run_id'] + '-' + str(logical_update) + '-' + parent + '-sample-' + str(sample)
           for parent in parent_ids for sample in range(repeats)]
    # A validation nonce distinguishes repeated before/after calls at one step.
    snapshot = run['run_id'] + ':' + str(logical_update) + ':' + ('validation-' if validation else 'policy-') + uuid.uuid4().hex
    return dict(run_id=run['run_id'], source_sha256=run['source_sha256'], slurm_job_id=job,
                logical_update=logical_update, policy_snapshot_id=snapshot,
                parent_group_ids=parent_ids, trajectory_ids=ids, validation=validation)


def attach_generation_identity(gen_batch, identity, *, do_sample):
    assert len(gen_batch) == len(identity['trajectory_ids'])
    gen_batch.non_tensor_batch['behavior_row_id'] = np.array(identity['trajectory_ids'], dtype=object)
    gen_batch.meta_info.update(behavior_identity=identity, policy_snapshot_id=identity['policy_snapshot_id'],
                               recompute_log_prob=False, do_sample=do_sample)


def actor_rows(batch):
    width = batch.batch['prompts'].shape[1]
    ids = batch.non_tensor_batch['behavior_trajectory_id'].tolist()
    values = {name: batch.batch[name].detach().cpu().tolist() for name in
              ('input_ids', 'attention_mask', 'info_mask', 'position_ids')}
    return [make_actor_row(trajectory_id=identity, prompt_width=width,
                           **{name: rows[index] for name, rows in values.items()})
            for index, identity in enumerate(ids)]


def _bindings(identity, sampling):
    return {'expected_' + name: identity[name] for name in
            ('run_id', 'source_sha256', 'slurm_job_id', 'logical_update', 'parent_group_ids',
             'trajectory_ids', 'policy_snapshot_id')} | {'expected_sampling': sampling}


def context_before_forward(batch):
    identity = batch.meta_info['behavior_identity']
    assert identity['validation'] is False and len(batch) == 2560
    base, run, job = _native_run()
    assert (run['run_id'], run['source_sha256'], job) == (
        identity['run_id'], identity['source_sha256'], identity['slurm_job_id'])
    rows = actor_rows(batch)
    ids = identity['trajectory_ids']
    actual_ids = batch.non_tensor_batch['behavior_trajectory_id'].tolist()
    assert actual_ids == ids
    turns = batch.non_tensor_batch['behavior_turns'].tolist()
    trajectories = [dict(trajectory_id=value, parent_occurrence_index=index // 5,
                         sample_index=index % 5, turns=turns[index]) for index, value in enumerate(ids)]
    sampling = batch.meta_info['behavior_sampling']
    trace = make_trace(run_id=identity['run_id'], source_sha256=identity['source_sha256'],
        slurm_job_id=job, logical_update=identity['logical_update'],
        policy_snapshot_id=identity['policy_snapshot_id'], sampling=sampling,
        parent_group_ids=identity['parent_group_ids'], trajectories=trajectories, actor_rows=rows)
    bindings = _bindings(identity, sampling)
    private = write_private_evidence(base, trace)
    assert private['serialization_transformed'] is False
    audit = audit_outer(trace, actor_rows=rows, **bindings)
    assert audit['trace_sha256'] == private['private_trace_sha256']
    return dict(trace=trace, actor_rows=rows, audit=audit, bindings=bindings, private=private)


def _tensor_hash(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(canonical_bytes(dict(dtype=str(value.dtype), shape=list(value.shape))) +
                          value.view(torch.uint8).numpy().tobytes()).hexdigest()


def row_hash(batch, index):
    fields = ('input_ids', 'responses', 'attention_mask', 'position_ids', 'info_mask',
              'old_log_probs', 'sampling_log_probs', 'sampling_mask')
    return document_sha256({name: _tensor_hash(batch.batch[name][index]) for name in fields})


def freeze_after_probability_gate(batch, context, hf_output):
    """HF recomputation is a diagnostic; actual passed sampling LP becomes old."""
    from reproduction.common8.behavior_probability import verify
    assert hf_output.meta_info['temperature'] == batch.meta_info['behavior_sampling']['temperature'] == 1.0
    assert hf_output.meta_info['micro_batch_size'] == 4 and hf_output.meta_info['use_dynamic_bsz'] is False
    width = batch.batch['responses'].shape[1]
    actual = batch.batch['sampling_log_probs'].detach().cpu()
    authored = batch.batch['info_mask'][:, -width:].detach().cpu()
    attention = batch.batch['attention_mask'][:, -width:].detach().cpu()
    assert torch.equal(batch.batch['sampling_mask'].detach().cpu(), authored.bool())
    proof = verify(hf_output.batch['old_log_probs'].detach().cpu(), actual, authored, attention,
        trace=context['trace'], actor_rows=context['actor_rows'], audit=context['audit'], **context['bindings'])
    batch.batch['old_log_probs'] = batch.batch['sampling_log_probs'].clone()
    batch.meta_info['temperature'] = hf_output.meta_info['temperature']
    base, run, job = _native_run()
    proof.update(row_tensor_sha256={value: row_hash(batch, index) for index, value in
                 enumerate(batch.non_tensor_batch['behavior_trajectory_id'].tolist())},
                 upstream_group_sha256={value: batch.meta_info['behavior_identity']['parent_group_ids'][index // 5].split('-', 1)[1]
                     for index, value in enumerate(batch.non_tensor_batch['behavior_trajectory_id'].tolist())},
                 private_trace_sha256=context['private']['private_trace_sha256'],
                 denominator='actual_selected_sampling_log_probs_after_all_token_context_and_0.05_gate')
    path = base / 'logs' / ('behavior-probability-outer-' + str(proof['logical_update']) + '.json')
    content = canonical_bytes(proof)
    descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, 'wb') as stream:
        stream.write(content); stream.flush(); os.fsync(stream.fileno())
    directory = os.open(str(path.parent), os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    batch.meta_info['behavior_probability_gate'] = dict(path=str(path), sha256=hashlib.sha256(content).hexdigest(),
        run_id=run['run_id'], slurm_job_id=job, logical_update=proof['logical_update'])
    return proof


def failed_generation_evidence(identity, records, reason):
    if identity.get('validation'):
        # Validation failures are still failures, but never masquerade as an
        # actual training outer with update >=1 in the private writer namespace.
        return
    base, run, job = _native_run()
    trace = {name: identity[name] for name in ('run_id', 'source_sha256', 'slurm_job_id',
                                               'logical_update', 'policy_snapshot_id')}
    trace.update(role='common8_failed_partial_generation', actual_sampling_records=records,
                 failure_reason=reason, complete_outer=False)
    write_private_evidence(base, trace)


def read_gate_receipt(base, run, step, *, update_record=None):
    """Light scalar/hash binding; never claim this rereads raw trajectories."""
    base = Path(base)
    path = base / 'logs' / ('behavior-probability-outer-' + str(step) + '.json')
    assert path.resolve() == path
    raw = path.read_bytes()
    proof = json.loads(raw)
    sha = hashlib.sha256(raw).hexdigest()
    allocation = json.loads((base / 'logs/native-allocation-admission.json').read_text())
    assert allocation['accepted'] is True and allocation['run_id'] == run['run_id']
    assert proof['accepted'] is True and proof['run_id'] == run['run_id']
    assert proof['source_sha256'] == run['source_sha256']
    assert proof['slurm_job_id'] == allocation['slurm_job_id']
    assert proof['logical_update'] == step and proof['trajectory_count'] == 2560
    assert proof['actual_frozen_old_all_finite'] is True
    assert 0 <= proof['max_abs_discrepancy'] <= proof['tolerance'] == 0.05
    assert proof['sampling']['temperature'] == proof['sampling']['top_p'] == 1.0
    assert proof['denominator'] == 'actual_selected_sampling_log_probs_after_all_token_context_and_0.05_gate'
    assert len(proof['row_tensor_sha256']) == len(proof['upstream_group_sha256']) == 2560
    if update_record is not None:
        assert update_record['behavior_probability_gate_sha256'] == sha
        assert update_record['behavior_trace_sha256'] == proof['trace_sha256'] == proof['private_trace_sha256']
    return dict(proof_sha256=sha, private_trace_sha256=proof['private_trace_sha256'],
                included_actor_token_count=proof['included_actor_token_count'],
                max_abs_discrepancy=proof['max_abs_discrepancy'])


def verify_write_receipt_links(base, gates, first, last):
    """One scalar-ledger scan binds every pre/post monitor and Adam write."""
    base = Path(base)
    for rank in range(8):
        rows = [json.loads(line) for line in (base / 'logs' / ('optimizer-rank-' + str(rank) + '.jsonl')).read_text().splitlines()]
        for row in rows:
            if row['kind'] not in ('OPTIMIZER_WRITE', 'POLICY_MONITOR'):
                continue
            step = row['context']['logical_update']
            if first <= step <= last:
                assert row['context']['behavior_probability_gate_sha256'] == gates[str(step)]['proof_sha256']


def verify_worker_input(data, *, logical_update, group):
    """Every rank binds actual local bytes to the driver gate before any write.

    Failure is collective: a local file/row problem cannot leave other ranks in
    a later model collective. This check does not turn a JSON file into a GPU
    execution or healthy-policy proof.
    """
    failure = None
    try:
        base, run, job = _native_run()
        receipt = data.meta_info['behavior_probability_gate']
        path = base / 'logs' / ('behavior-probability-outer-' + str(logical_update) + '.json')
        assert receipt['path'] == str(path) and path.resolve() == path and path.stat().st_uid == os.getuid()
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == receipt['sha256']
        proof = json.loads(raw)
        assert proof['accepted'] is True and proof['production_admission'] is False
        assert proof['run_id'] == receipt['run_id'] == run['run_id']
        assert proof['source_sha256'] == run['source_sha256']
        assert proof['slurm_job_id'] == receipt['slurm_job_id'] == job
        assert type(logical_update) is int and proof['logical_update'] == receipt['logical_update'] == logical_update
        assert proof['trajectory_count'] == 2560 and len(proof['row_tensor_sha256']) == 2560
        assert proof['max_abs_discrepancy'] <= 0.05 and data.meta_info['temperature'] == 1.0
        assert len(data) == 320 and torch.equal(data.batch['old_log_probs'], data.batch['sampling_log_probs'])
        width = data.batch['responses'].shape[1]
        authored = data.batch['info_mask'][:, -width:]
        assert torch.equal(data.batch['loss_mask'], authored)
        assert torch.equal(data.batch['sampling_mask'], authored.bool())
        for index, identity in enumerate(data.non_tensor_batch['behavior_trajectory_id']):
            assert proof['row_tensor_sha256'][identity] == row_hash(data, index)
            upstream = data.non_tensor_batch['uid'][index]
            if hasattr(upstream, 'item'):
                upstream = upstream.item()
            assert proof['upstream_group_sha256'][identity] == document_sha256(upstream)
    except Exception as error:
        failure = type(error).__name__
    device = torch.device('cuda', torch.cuda.current_device()) if torch.distributed.get_backend(group) == 'nccl' else torch.device('cpu')
    flag = torch.tensor(int(failure is None), dtype=torch.int32, device=device)
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN, group=group)
    if not bool(flag.item()):
        failures = [None] * torch.distributed.get_world_size(group)
        torch.distributed.all_gather_object(failures, failure, group=group)
        raise RuntimeError('Common8 actual behavior gate rejected before Adam on ranks: ' + repr(failures))
    all_ids = [None] * torch.distributed.get_world_size(group)
    torch.distributed.all_gather_object(all_ids, data.non_tensor_batch['behavior_trajectory_id'].tolist(), group=group)
    flattened = [value for rank_ids in all_ids for value in rank_ids]
    assert len(flattened) == len(set(flattened)) == 2560 and set(flattened) == set(proof['trajectory_ids']), \
        'All-rank actual behavior trajectory coverage changed before Adam'
