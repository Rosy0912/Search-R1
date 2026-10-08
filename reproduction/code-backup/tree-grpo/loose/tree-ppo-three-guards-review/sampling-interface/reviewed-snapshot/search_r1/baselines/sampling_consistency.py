"""Read-only raw-turn sampler/FSDP diagnostics, never a replacement PPO old policy."""
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist

from .training_state import atomic_save, cpu_copy


def reduce_pair_statistics(sampled, actor, mask, distribution, temperature, limit):
    """Compare identical emitted tokens before any decoding or rolling truncation."""
    device = actor.device
    valid = mask.bool()
    local_error = int(not math.isfinite(limit) or limit < 0)
    expected = dict(temperature=temperature, top_p=1., top_k=-1, min_p=0.,
                    presence_penalty=0., frequency_penalty=0., repetition_penalty=1., n=1)
    local_error |= int(any(distribution.get(k) != v for k, v in expected.items()))
    # A filtered sampling distribution cannot be compared to plain actor softmax.
    local_error |= int(bool(distribution.get('logits_processors')))
    delta = torch.where(valid, actor, 0.).double() - torch.where(valid, sampled, 0.).double()
    finite = torch.isfinite(actor) & torch.isfinite(sampled) & torch.isfinite(delta)
    bad = (valid & ~finite).sum().double()
    safe = torch.where(valid & finite, delta, 0.)
    over_bound = (valid & finite & (safe > 30.)).sum().double()
    k3 = torch.expm1(torch.where(safe <= 30., safe, 0.)) - torch.where(safe <= 30., safe, 0.)
    sums = torch.stack([valid.sum().double(), bad, over_bound, safe.abs().sum(),
                        safe.square().sum(), k3.sum(), torch.tensor(float(local_error), device=device)])
    maximum = safe.abs().max() if safe.numel() else torch.zeros((), device=device, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(sums)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    count, invalid, excessive, abs_sum, square_sum, kl_sum, metadata_invalid = sums.tolist()
    reasons = []
    if metadata_invalid: reasons.append('sampling_distribution_not_plain_actor_softmax_or_invalid_limit')
    if invalid: reasons.append('nonfinite_valid_sampling_or_actor_logprob')
    if excessive: reasons.append('sampling_difference_exceeds_numeric_bound')
    if not count: reasons.append('empty_sampling_token_set')
    if maximum.item() > limit: reasons.append('raw_turn_sampling_actor_max_abs_delta')
    return dict(allowed=not reasons, reasons=reasons, valid_tokens=int(count),
                invalid_tokens=int(invalid), numeric_bound_exceeded=int(excessive),
                max_abs_delta=maximum.item(), mean_abs_delta=abs_sum/count if count else None,
                rms_delta=math.sqrt(square_sum/count) if count else None,
                sampled_k3_mean=kl_sum/count if count and not invalid and not excessive else None,
                max_abs_delta_limit=limit,
                scope='All raw emitted response tokens in this distributed generation call; '
                      'same raw turn context, not postprocessed final Tree leaves')


def record_raw_turn(worker, output):
    """Consume the temporary sampler field before it can reach Tree concatenation."""
    sampled = output.batch.pop('__sampling_log_probs')
    distribution = output.meta_info.pop('__sampling_distribution')
    config = worker.config.rollout.sampling_consistency
    actor = output.batch['old_log_probs']
    mask = output.batch['attention_mask'][:, -output.batch['responses'].shape[1]:]
    decision = reduce_pair_statistics(sampled, actor, mask, distribution,
                                      worker.config.rollout.temperature, float(config.max_abs_delta))
    rank = dist.get_rank() if dist.is_initialized() else 0
    call = getattr(worker, '_sampling_consistency_calls', 0) + 1
    worker._sampling_consistency_calls = call
    root = Path(os.environ['SEARCH_R1_BASELINE_OUTPUT'])/'sampling-consistency'
    record = dict(rank=rank, world_size=dist.get_world_size() if dist.is_initialized() else 1,
                  generation_call=call, scheduler_step=worker.actor_lr_scheduler.last_epoch,
                  distribution=distribution, decision=decision, enforced=bool(config.enforce),
                  code_commit=os.environ['SEARCH_R1_BASELINE_COMMIT'],
                  config_sha256=os.environ['SEARCH_R1_BASELINE_CONFIG_SHA256'],
                  data_identity=os.environ['SEARCH_R1_BASELINE_DATA_IDENTITY'],
                  local_rows=len(output), saved_rows=min(4, len(output)), optimizer_committed=0,
                  context_scope='Before response decode/retokenization/rolling-window truncation')
    failed = False
    try:
        snapshot = {key: cpu_copy(value[:4]) for key, value in output.batch.items()}
        snapshot['sampled_log_probs'] = cpu_copy(sampled[:4])
        snapshot['distribution'] = distribution
        snapshot_path = root/f'rank-{rank:02d}-call-{call:05d}.pt'
        record['snapshot_path'] = str(snapshot_path)
        record['snapshot_sha256'] = atomic_save(snapshot, snapshot_path)
        path = root/f'rank-{rank:02d}-call-{call:05d}.json'
        atomic_save(record, path, json_format=True)
    except Exception:
        failed = True
    flag = torch.tensor(int(failed), device=actor.device)
    if dist.is_initialized(): dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    if flag.item(): raise RuntimeError('Global raw-turn sampling evidence-write failure')
    worker._sampling_consistency_records = getattr(worker, '_sampling_consistency_records', []) + [str(path)]
    if bool(config.enforce) and not decision['allowed']:
        raise RuntimeError('Global raw-turn sampling/FSDP consistency refusal: '+repr(decision['reasons']))


def calibrate_actor(worker, data):
    """No optimizer call or scheduler advance; diagnostics only in a dedicated stage."""
    from .policy_safety_runtime import PolicySafetyRuntime
    if os.environ.get('TREE_PPO_CALIBRATION_ONLY') != '1':
        raise RuntimeError('Read-only calibration endpoint requires dedicated calibration stage')
    safety = PolicySafetyRuntime(worker.actor, data.batch, data.meta_info, worker.config.actor.policy_safety)
    with torch.no_grad():
        decision = safety.measure('before_optimizer', publish=False)
    steps = sorted({int(state['step'].item()) if isinstance(state['step'], torch.Tensor) else int(state['step'])
                    for state in worker.actor_optimizer.state.values() if 'step' in state})
    rank = dist.get_rank()
    record = dict(rank=rank, world_size=dist.get_world_size(), optimizer_committed=0,
                  scheduler_advanced=False, parent_completed_outer=int(data.meta_info['calibration_parent_outer']),
                  observed_outer=int(data.meta_info['baseline_outer_step']), restored_optimizer_steps=steps,
                  scheduler_step=worker.actor_lr_scheduler.last_epoch, decision=decision,
                  monitor_local_row_indices=safety.indices.tolist(),
                  diagnostic_records=getattr(worker, '_sampling_consistency_records', []),
                  code_commit=os.environ['SEARCH_R1_BASELINE_COMMIT'],
                  config_sha256=os.environ['SEARCH_R1_BASELINE_CONFIG_SHA256'])
    failed = False
    try:
        root = Path(os.environ['SEARCH_R1_BASELINE_OUTPUT'])/'probability-calibration'
        atomic_save({key: cpu_copy(value) for key, value in safety.monitor.items()},
                    root/f'rank-{rank:02d}-monitor.pt')
        atomic_save(record, root/f'rank-{rank:02d}.json', json_format=True)
    except Exception:
        failed = True
    flag = torch.tensor(int(failed), device=safety.device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    if flag.item(): raise RuntimeError('Global zero-write calibration evidence-write failure')
    return record


def finalize_calibration(trainer, batch):
    """Driver-only terminal observation, before the ordinary actor update path."""
    import json
    parent = trainer.recovery_provenance['parent_completed_step']
    if (trainer.global_steps != parent + 1 or trainer.actual_outer_updates != parent
            or not trainer.config.actor_rollout_ref.actor.policy_safety.enabled
            or not trainer.config.actor_rollout_ref.rollout.sampling_consistency.enabled
            or trainer.config.actor_rollout_ref.rollout.sampling_consistency.enforce):
        raise RuntimeError('Dedicated zero-write calibration configuration/boundary changed')
    batch.meta_info['baseline_outer_step'] = trainer.global_steps
    batch.meta_info['calibration_parent_outer'] = parent
    # Exactly the existing update temperature, but no update_policy endpoint.
    batch.meta_info['temperature'] = trainer.config.actor_rollout_ref.rollout.temperature
    trainer.actor_rollout_wg.calibrate_policy_safety(batch)
    root = Path(os.environ['SEARCH_R1_BASELINE_OUTPUT'])/'probability-calibration'
    workers = [json.loads((root/f'rank-{rank:02d}.json').read_text()) for rank in range(8)]
    if any(record['rank'] != rank or record['world_size'] != 8
           or record['optimizer_committed'] != 0 or record['scheduler_advanced']
           or record['scheduler_step'] != parent or not record['diagnostic_records']
           for rank, record in enumerate(workers)):
        raise RuntimeError('Zero-write calibration rank proof incomplete')
    ready = dict(status='CALIBRATION_OBSERVED_ZERO_WRITES', workers=workers,
                 parent_completed_outer=parent, observed_outer=trainer.global_steps,
                 new_completed_outer=0, optimizer_committed=0, scheduler_advanced=False,
                 sampler_cursor_before_outer=trainer.recovery_provenance['parent_cursor'],
                 sampler_cursor_after_observation=trainer.checkpoint_sampler.cursor,
                 job_id=os.environ['SLURM_JOB_ID'], code_commit=os.environ['SEARCH_R1_BASELINE_COMMIT'],
                 config_sha256=os.environ['SEARCH_R1_BASELINE_CONFIG_SHA256'],
                 data_identity=os.environ['SEARCH_R1_BASELINE_DATA_IDENTITY'],
                 model_identity=os.environ['SEARCH_R1_MODEL_IDENTITY'],
                 rollout_reward=float(batch.batch['token_level_rewards'].sum(-1).mean()),
                 scope='Healthy checkpoint observation without optimizer, scheduler, outer credit or final evaluation')
    atomic_save(ready, root/'ready.json', json_format=True)
    print('TREE_PPO_CALIBRATION_OBSERVED_ZERO_WRITES', flush=True)
    return ready
