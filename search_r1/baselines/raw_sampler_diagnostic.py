"""Bounded raw-context HF/native diagnostics; never an update admission guard.

The selected native tokens may be discarded or retokenized by the training
controller. Recompute HF probabilities on their original request+raw completion,
not on rewritten training actions. Differences include cross-backend drift.
"""
from dataclasses import dataclass
import hashlib
import json
import math

import torch

from .policy_safety import consensus
from . import policy_safety


@dataclass(frozen=True)
class RawRecord:
    request: tuple
    response: tuple
    native: tuple
    negative_advantage: object


@dataclass(frozen=True)
class RawDiagnosticCache:
    records: tuple
    reason: object
    rows_per_rank: int
    slots: int
    source_sha256: str


def _untransformed_full_support(sampling):
    required = dict(n=1, temperature=1., top_p=1., top_k=-1, min_p=0.,
                    frequency_penalty=0., presence_penalty=0., repetition_penalty=1.)
    if any(sampling.get(key) != value for key, value in required.items()):
        raise ValueError('native_temperature_support_or_penalties_unverified')
    if 'best_of' not in sampling or sampling['best_of'] not in (None, 1, {'field_unavailable': True}):
        raise ValueError('native_best_of_selection_unverified')
    if 'logits_processors' not in sampling or sampling['logits_processors'] not in (None, [], {'field_unavailable': True}):
        raise ValueError('native_logits_processors_unverified')
    unsupported = sampling.get('unsupported_fields', [])
    for key in ('logit_bias', 'allowed_token_ids', 'guided_decoding'):
        if key not in sampling and key not in unsupported:
            raise ValueError('native_optional_support_field_unverified:' + key)
        if key in sampling and sampling[key] == {'field_unavailable': True}:
            continue  # Actual runtime API absence, never a fabricated neutral value.
        if key in sampling and sampling[key] not in (None, [], {}):
            raise ValueError('native_support_transform:' + key)
    for key, neutral in (('ignore_eos', False), ('min_tokens', 0)):
        value = sampling.get(key)
        if value != neutral and value != {'field_unavailable': True}:
            raise ValueError('native_eos_support_transform:' + key)
    if sampling.get('logprob_semantics') != 'native_request_output_only':
        raise ValueError('native_logprob_provenance_unverified')


def prepare_raw_diagnostic(data, config):
    """Cache at most4 rows/rank,5 raw turns/row and20 forward slots/rank."""
    rows = config.get('raw_sampler_diagnostic_rows_per_rank', 4)
    if type(rows) is not int or not 0 < rows <= 4:
        return RawDiagnosticCache((), 'invalid_bounded_row_budget', 4, 20, '')
    records, subset = [], []
    try:
        journals = data.non_tensor_batch.get('raw_sampling_diagnostics')
        if journals is None:
            raise ValueError('native_raw_journals_unavailable')
        width = data.batch['responses'].shape[1]
        masks = data.batch['loss_mask'] if config.state_masking else data.batch['attention_mask'][:, -width:]
        advantages = data.batch['advantages']
        for row in range(min(rows, len(data))):
            turns = journals[row]
            if not turns or len(turns) > 5:
                raise ValueError('raw_turn_count_unavailable_or_out_of_bounds')
            observed_adv = advantages[row][masks[row].bool()]
            sign = None
            if observed_adv.numel() and bool(torch.isfinite(observed_adv).all()) and bool(
                    torch.equal(observed_adv, observed_adv[:1].expand_as(observed_adv))):
                sign = bool(observed_adv[0] < 0)
            for journal in turns:
                trace = journal.get('native_trace')
                if not trace or journal.get('capture_provenance') != 'native_request_output_only':
                    raise ValueError('native_request_trace_unavailable')
                from search_r1.llm_agent.sampling_trace import sha
                body = {key: value for key, value in trace.items() if key != 'native_sha256'}
                if trace.get('native_sha256') != sha(body):
                    raise ValueError('native_journal_hash_mismatch')
                if trace['actual_request_token_ids'] != journal['actual_request_token_ids']:
                    raise ValueError('native_request_context_mismatch')
                if trace['row_id'] != journal['row_id'] or trace['policy_snapshot_id'] != journal['policy_snapshot_id']:
                    raise ValueError('native_row_or_snapshot_mismatch')
                request, response, native = trace['actual_request_token_ids'], trace['raw_generated_token_ids'], trace['selected_logprobs']
                if (not request or len(request) > 4096 or not response or len(response) > 500
                        or len(response) != len(native)
                        or any(type(token) is not int or token < 0 for token in request + response)
                        or any(not math.isfinite(float(lp)) or float(lp) > 0 for lp in native)):
                    raise ValueError('native_tokens_or_probabilities_out_of_bounds')
                transported = journal['raw_generated_token_ids']
                if transported[:len(response)] != response:
                    raise ValueError('native_raw_token_transport_mismatch')
                _untransformed_full_support(trace['sampling'])
                proof = trace.get('effective_sampler_provenance')
                if not proof:
                    raise ValueError('actual_engine_group_sampler_provenance_unavailable')
                proof_body = {key: value for key, value in proof.items() if key != 'sha256'}
                if (proof.get('sha256') != sha(proof_body)
                        or proof.get('source') != 'native_enqueued_sequence_group'
                        or proof.get('observed_stage') != 'after_actual_add_request_before_scheduling'
                        or not isinstance(trace.get('request_id'), str) or not trace['request_id']
                        or proof.get('request_id') != trace.get('request_id')
                        or proof.get('actual_request_token_ids') != request):
                    raise ValueError('actual_engine_group_sampler_provenance_mismatch')
                _untransformed_full_support(proof['sampling'])
                # Bind verified native journals, not unrelated padded wrapper
                # scores whose padding can be nonfinite or unavailable.
                subset.append(trace)
                records.append(RawRecord(tuple(request), tuple(response), tuple(float(x) for x in native), sign))
        if not records:
            raise ValueError('empty_native_raw_subset')
        digest = hashlib.sha256(json.dumps(subset, sort_keys=True, allow_nan=False).encode()).hexdigest()
        return RawDiagnosticCache(tuple(records), None, rows, rows * 5, digest)
    except (KeyError, ValueError, TypeError, AttributeError) as error:
        return RawDiagnosticCache((), str(error), rows, rows * 5, '')


def raw_microbatch(records, device, pad_id=0):
    """Left-pad native requests and right-pad original raw completions explicitly."""
    request_width = max(len(record.request) for record in records)
    response_width = max(len(record.response) for record in records)
    ids = torch.full((len(records), request_width + response_width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    native = torch.zeros((len(records), response_width), dtype=torch.float64, device=device)
    selected = torch.zeros_like(native, dtype=torch.bool)
    for row, record in enumerate(records):
        start = request_width - len(record.request)
        ids[row, start:request_width] = torch.tensor(record.request, device=device)
        ids[row, request_width:request_width + len(record.response)] = torch.tensor(record.response, device=device)
        mask[row, start:request_width + len(record.response)] = 1
        native[row, :len(record.native)] = torch.tensor(record.native, dtype=torch.float64, device=device)
        selected[row, :len(record.native)] = True
    positions = (mask.cumsum(-1) - 1).clamp_min(0) * mask
    return dict(input_ids=ids, attention_mask=mask, position_ids=positions,
                responses=ids[:, request_width:]), native, selected


@torch.no_grad()
def evaluate_raw_diagnostic(actor, cache):
    """Fixed10 micro forwards/scan at most; all ranks follow the same FSDP calls."""
    group = actor._safety_group
    report = dict(diagnostic_only=True, scope='fixed_first_rows_per_rank_raw_native_contexts',
                  rows_per_rank=cache.rows_per_rank,
                  global_row_budget=cache.rows_per_rank * (torch.distributed.get_world_size(group) if policy_safety.distributed(group) else 8),
                  max_raw_turns_per_row=5, max_forward_calls_per_scan=10,
                  max_native_request_tokens=4096, max_raw_completion_tokens=500,
                  hf_recomputation_temperature=1., objective_weight_definition='uniform_raw_token_mean_on_bounded_subset',
                  denominator_role='native_raw_sampler_diagnostic_not_ppo_denominator',
                  interpretation='cross_backend_raw_context_gap_includes_backend_drift_not_pure_policy_kl',
                  raw_context_cross_backend_k3=None, negative_advantage_raw_token_tail_fraction=None,
                  source_journal_sha256=cache.source_sha256, available=False)
    slot_bound_ok = 0 < len(cache.records) <= cache.slots <= 20
    if not consensus(cache.reason is None and slot_bound_ok, group):
        report['unavailable_reason'] = cache.reason or (
            'raw_subset_slot_bound_exceeded' if not slot_bound_ok else 'native_raw_journals_unavailable_on_another_rank')
        return report
    records = list(cache.records)
    # Every rank executes an equal fixed slot count, independent of early
    # terminal answers. Duplicate slots are padding and excluded from all sums.
    slots = records + [records[0]] * (cache.slots - len(records))
    training = actor.actor_module.training
    device = torch.device('cuda', torch.cuda.current_device()) if torch.cuda.is_available() else next(actor.actor_module.parameters()).device
    totals = torch.zeros(6, dtype=torch.float64, device=device)
    numeric_ok = True
    tail_complete = all(record.negative_advantage is not None for record in records)
    try:
        actor.actor_module.eval()
        for start in range(0, cache.slots, 2):
            micro, native, selected = raw_microbatch(slots[start:start + 2], totals.device)
            _, current = actor._forward_micro_batch(micro, temperature=1.)
            for offset, record in enumerate(slots[start:start + 2]):
                if start + offset >= len(records):
                    continue
                valid = selected[offset]
                now = current[offset][valid].double()
                delta = now - native[offset][valid]
                if not bool(torch.isfinite(now).all() and torch.isfinite(delta).all()):
                    numeric_ok = False
                    continue
                oversized = delta > 30.
                safe_delta = torch.where(oversized, 0., delta)
                totals[0] += delta.numel()
                totals[1] += (safe_delta.expm1() - safe_delta).sum()
                totals[2] += oversized.sum()
                if record.negative_advantage is True:
                    totals[3] += delta.numel()
                    totals[4] += (delta > math.log(3.)).sum()
                totals[5] += 1
    finally:
        actor.actor_module.train(training)
    numeric_ok = consensus(numeric_ok and bool(torch.isfinite(totals).all()), group)
    tail_complete = consensus(tail_complete, group)
    if policy_safety.distributed(group):
        torch.distributed.all_reduce(totals, group=group)
    if not numeric_ok:
        report['unavailable_reason'] = 'nonfinite_hf_raw_diagnostic_or_totals'
        return report
    count, numerator, oversized, negative_count, tail_count, turn_count = totals.tolist()
    complete = oversized == 0 and count > 0 and all(math.isfinite(x) for x in totals.tolist())
    report.update(available=complete, kl_estimate_complete=complete, raw_token_count=int(count),
                  raw_turn_count=int(turn_count), oversized_log_ratio_count=int(oversized),
                  raw_context_cross_backend_k3=numerator / count if complete else None,
                  negative_advantage_raw_token_tail_fraction=(tail_count / negative_count if negative_count else 0.) if tail_complete else None,
                  negative_advantage_assignment='constant_supervised_advantage_per_original_trajectory',
                  tail_estimate_complete=tail_complete)
    if not complete:
        report['unavailable_reason'] = 'raw_kl_log_ratio_above_diagnostic_bound_or_empty'
    if not tail_complete:
        report['tail_unavailable_reason'] = 'nonconstant_or_missing_original_trajectory_advantage'
    return report
