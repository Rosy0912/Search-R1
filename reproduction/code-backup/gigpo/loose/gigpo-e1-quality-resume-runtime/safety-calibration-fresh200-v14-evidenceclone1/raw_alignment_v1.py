"""Strict actual raw-action identity, with probabilities from existing forwards.

No thresholds are selected here. Raw row timeline positions are deliberately
unused: each scored action must provide its actual independently forwarded
canonical sampling prefix and the whole sampled actor action.
"""
import copy
import torch
from raw_hf_observation_v1 import require, validate_actual_model_inputs
from raw_runner_observation_v1 import validate_actual_runner_observation


def validate_real_sample(raw, pad, eos):
    ids, lengths, mask = raw['sampled_ids'], raw['sampled_lengths'], raw['sampled_mask']
    require(ids.ndim == 2 and lengths.dtype == torch.long and lengths.shape == (ids.shape[0],),
            'Actual CompletionOutput lengths absent')
    require(mask.ndim == 2 and mask.shape[0] == ids.shape[0] and mask.shape[1] >= ids.shape[1],
            'Actual sampled mask absent')
    require(bool(((lengths > 0) & (lengths <= ids.shape[1])).all()), 'Invalid actual sample lengths')
    require(torch.equal(mask.bool(), torch.arange(mask.shape[1])[None, :] < lengths[:, None]),
            'Sample mask is not the real contiguous sampling support')
    require(raw['actual_pad_token_id'] == pad and raw['actual_eos_token_id'] == eos,
            'Actual worker tokenizer identity differs')
    require(torch.equal(raw['tensors']['response_sampled_lengths'], lengths)
            and torch.equal(raw['tensors']['response_sample_mask'], mask), 'Worker/engine sampling lengths differ')
    require(torch.equal(raw['tensors']['responses'][:, :ids.shape[1]], ids), 'Raw HF sample tokens differ from engine tokens')
    require(torch.equal(raw['tensors']['attention_mask'][:, -mask.shape[1]:].bool(), mask.bool()),
            'Actual sampled PAD token lost its HF attention support')
    require(torch.isfinite(raw['sampled_log_probs'][mask[:, :ids.shape[1]].bool()]).all(),
            'Nonfinite actual sampled token probability')
    runner_rows = validate_actual_runner_observation(raw['actual_runner_observation'],
        raw['actual_engine_prompt_token_ids'], ids, lengths, raw['sampled_log_probs'])
    raw['validated_actual_runner_rows'] = runner_rows
    for field in ('perturn_hf_flat_calls', 'repeated_perturn_hf_flat_calls'):
        offset = 0
        for call in raw[field]:
            validate_actual_model_inputs(call)
            n = len(call['flat_micro']['responses'])
            for key in ('responses', 'input_ids', 'attention_mask', 'position_ids'):
                require(torch.equal(call['flat_micro'][key], raw['tensors'][key][offset:offset+n]),
                        'Observed actual perturn HF input differs: ' + key)
            expected = raw['tensors']['old_log_probs'] if field == 'perturn_hf_flat_calls' else raw['actor_repeated_log_probs']
            require(torch.equal(call['flat_log_probs'][:, :ids.shape[1]], expected[offset:offset+n, :ids.shape[1]]),
                    'Observed actual perturn HF probability output differs')
            offset += n
        require(offset == len(ids), 'Incomplete actual perturn HF forward coverage')


def collect_raw_actions(turns, raw_turns, hf_roles, batch, sources, *, question_ids, pad, eos, code_commit, binding_sha256):
    """Return structural records and token/joint errors; never admit numerics."""
    expected_rows = len(batch['responses'])
    require(len(question_ids) == expected_rows, 'Missing actual five-response question identity')
    require(set(sources) == set(range(expected_rows)) and len(set(sources)) == len(sources), 'Actual scored trajectory mapping differs')
    require(set(hf_roles) == {'old', 'current', 'reference'}, 'Missing actual HF role')
    by_source = {source: dict(source=source, actions=[]) for source in range(expected_rows)}
    probability_records, checks = [], []
    expected_action_keys = set()
    scored_rows = {source: row for row, source in enumerate(sources)}
    question_groups = {}
    for source, question in zip(sources, question_ids):
        group = source//5
        require(group not in question_groups or question_groups[group] == question,
                'Five-response question grouping changed')
        question_groups[group] = question
    for serial, turn in enumerate(turns):
        rank_raw = [raw_turns[(serial, rank)] for rank in range(8)]
        chunk = len(rank_raw[0]['sampled_ids'])
        require(all(len(raw['sampled_ids']) == chunk for raw in rank_raw), 'Unexpected eight-rank turn partition')
        require(len(turn) <= chunk * 8 and 0 <= chunk * 8 - len(turn) < 8, 'Unexpected duplicated GPU padding')
        projected = []
        for rank, raw in enumerate(rank_raw):
            require(raw['serial'] == serial and raw['rank'] == rank and raw['phase'] == 'train'
                    and raw['source_commit'] == code_commit and raw['binding_sha256'] == binding_sha256,
                    'Raw turn source/binding mapping differs')
            validate_real_sample(raw, pad, eos)
            # The producer records the actual full typed identity and neutrality;
            # driver contract independently requires these fields before pause.
            projected.append(dict(serial=serial, rank=rank, source_commit=code_commit,
                comparison_scope='SAME_PREFIX_SAME_TOKENS',
                effective_probability_scope='RETAINED_ACTOR_ACTION_PREFIX_EXCLUDES_DISCARDED_AND_PADDED_ROWS',
                temperature=raw['actual_sampling_parameters']['temperature'],
                top_p=raw['actual_sampling_parameters']['top_p'], top_k=raw['actual_sampling_parameters']['top_k'],
                n=raw['actual_sampling_parameters']['n'], valid_tokens=0, nonfinite_effective_count=0,
                actor_old_max_abs_delta=0., actor_vllm_max_abs_delta=0.,
                repeated_probability_probe_rng_unchanged=raw['probe_rng_unchanged'],
                repeated_probability_probe_complete_state_unchanged=raw['probe_complete_unchanged'],
                repeated_probe_identity_before=raw['repeated_probe_state_before']['identity'],
                repeated_probe_identity_after=raw['repeated_probe_state_after']['identity']))
        for index, observed in enumerate(turn):
            rank, local = divmod(index, chunk)
            raw, record = rank_raw[rank], projected[rank]
            source, step = observed['source'], observed['step_id']
            key = (source, step)
            require(key not in expected_action_keys, 'Duplicate sampled trajectory/action identity')
            expected_action_keys.add(key)
            require(all(key in hf_roles[role] for role in hf_roles), 'Missing actually forwarded complete action')
            old, current, reference = (hf_roles[role][key] for role in ('old', 'current', 'reference'))
            retained = observed['retained_action_tokens']
            length = len(retained)
            real_length = int(raw['sampled_lengths'][local])
            sampled = raw['sampled_ids'][local, :real_length].tolist()
            require(retained == sampled[:length] and length <= real_length, 'Raw actor action differs from actual sampled token prefix')
            require(observed['raw_actor_record']['sampled_length'] == real_length, 'Raw source sampling support differs')
            require(observed['raw_actor_record']['actor_action_tokens'] == retained, 'Source raw actor provenance differs')
            require(sampled.count(eos) == int(bool(sampled and sampled[-1] == eos)), 'Interior/duplicate EOS sampling support is not valid')
            require(not (retained and retained[-1] == eos), 'Genuine terminal EOS cannot receive action loss mass')
            prefix = observed['generation_prefix']
            prompt_width = raw['tensors']['prompts'].shape[1]
            attention = raw['tensors']['attention_mask'][local]
            positions = raw['tensors']['position_ids'][local]
            prompt_mask = attention[:prompt_width].bool()
            actual_hf_prefix = raw['tensors']['prompts'][local][prompt_mask].tolist()
            require(actual_hf_prefix == prefix == raw['actual_engine_prompt_token_ids'][local],
                    'Actual perturn HF/engine prefix differs from sampled source prefix')
            require(positions[:prompt_width][prompt_mask].tolist() == list(range(len(prefix))),
                    'Actual perturn HF prefix position changed')
            for role, value in [('old', old), ('current', current), ('reference', reference)]:
                require(value['prefix'] == prefix and value['tokens'] == retained,
                        'Actual complete ' + role + ' HF input differs from sampler')
                require(value['question_id'] == question_ids[scored_rows[source]]
                        and value['response_group'] == source//5 and value['response_slot'] == source%5,
                        'Actual complete ' + role + ' HF response grouping/source identity differs')
                require(value['positions'] == positions[prompt_width:prompt_width+length].tolist()
                        == list(range(len(prefix), len(prefix)+length)) and value['mask'] == [1]*length,
                        'Actual complete ' + role + ' HF positions/mask differs')
            row = scored_rows[source]
            selected = batch['step_ids'][row].eq(step) & batch['loss_mask'][row].bool()
            require(batch['responses'][row][selected].tolist() == retained, 'Effective raw action mask/token mapping differs')
            require(torch.equal(batch['old_log_probs'][row][selected].double(), old['log_probs'])
                    and torch.equal(batch['ref_log_prob'][row][selected].double(), reference['log_probs']),
                    'Scored batch differs from actual old/reference forwards')
            values = [old['log_probs'], current['log_probs'], reference['log_probs'],
                      raw['tensors']['old_log_probs'][local, :length].double(),
                      raw['actor_repeated_log_probs'][local, :length].double(), raw['sampled_log_probs'][local, :length].double()]
            finite = torch.stack([torch.isfinite(value) for value in values]).all(0)
            record['valid_tokens'] += length
            record['nonfinite_effective_count'] += int((~finite).sum())
            require(bool(finite.all()), 'Nonfinite actual effective action probability')
            perturn, repeated, sampler = values[3:]
            if length:
                record['actor_old_max_abs_delta'] = max(record['actor_old_max_abs_delta'], float((perturn-repeated).abs().max()))
                record['actor_vllm_max_abs_delta'] = max(record['actor_vllm_max_abs_delta'], float((perturn-sampler).abs().max()))
            def maximum(delta):
                return float(delta.abs().max()) if length else 0.
            check = dict(source=source, step_id=step, effective_tokens=length, sampled_length=real_length,
                raw_response_offset=old['offset'], raw_trajectory_effective_tokens=int(batch['loss_mask'][row].sum()),
                effective_trajectory_token_weight=1./int(batch['loss_mask'][row].sum()) if bool(batch['loss_mask'][row].any()) else 0.,
                full_trajectory_vs_sample_prefix_old_max_abs=maximum(old['log_probs']-perturn),
                full_vs_prefix_old_joint_sum_delta=float((old['log_probs']-perturn).sum()),
                full_vs_vllm_joint_sum_delta=float((old['log_probs']-sampler).sum()),
                prefix_old_vs_vllm_joint_sum_delta=float((perturn-sampler).sum()),
                complete_current_vs_old_max_abs=maximum(current['log_probs']-old['log_probs']),
                complete_current_vs_old_joint_sum_delta=float((current['log_probs']-old['log_probs']).sum()),
                same_position_ids=True, same_effective_mask=True, engine_prefix_matches_generation=True,
                actual_complete_hf_inputs_observed=True, actual_vllm_runner_positions_observed=True,
                sampler_position_evidence_scope='ACTUAL_RUNNER_AND_MODEL_FORWARD_INPUTS_WITH_REQUEST_MAPPING',
                actual_runner_request_evidence=raw['validated_actual_runner_rows'][local],
                probability_accepted=False)
            checks.append(check)
            action = copy.deepcopy(observed)
            action.update(scoring_prefix=old['prefix'], sampled_tokens=sampled,
                          engine_prompt_token_ids=raw['actual_engine_prompt_token_ids'][local],
                          actual_hf_position_ids=old['positions'], actual_hf_mask=old['mask'])
            by_source[source]['actions'].append(action)
        probability_records += projected
    require(all(set(hf_roles[role]) == expected_action_keys for role in hf_roles),
            'Incomplete/extra complete HF action coverage')
    require(all(record['actions'] for record in by_source.values()), 'Missing actual source trajectory')
    alignment = dict(schema='RAW_ACTION_PREFIX_ALIGNMENT_V1', trajectory_rows=expected_rows,
        recorded_actions=len(expected_action_keys), prefix_mismatch_count=0, action_token_mismatch_count=0,
        position_ids_mismatch_count=0, effective_mask_mismatch_count=0, engine_prefix_mismatch_count=0,
        actual_complete_hf_inputs_observed=True, actual_vllm_runner_positions_observed=True,
        sampler_position_evidence_scope='ACTUAL_RUNNER_AND_MODEL_FORWARD_INPUTS_WITH_REQUEST_MAPPING', probability_accepted=False)
    return alignment, probability_records, checks, list(by_source.values())
