"""Raw native context construction, provenance refusal and bounded actor wiring."""
import copy
from dataclasses import FrozenInstanceError
import math
import pytest
import torch

from search_r1.baselines.raw_sampler_diagnostic import (prepare_raw_diagnostic, raw_microbatch,
                                                        evaluate_raw_diagnostic)
from search_r1.llm_agent.sampling_trace import sha
from verl.workers.actor.dp_actor import DataParallelPPOActor
from test_baseline_aligned_actor import aligned_actor


def attach_raw(data):
    import numpy as np
    journals = np.empty(len(data), dtype=object)
    examples = [([10, 11], [30, 31]), ([12, 13, 14], [40]), ([15], [50, 51, 52]), ([16], [60, 61])]
    sampling = dict(n=1, best_of=1, temperature=1., top_p=1., top_k=-1, min_p=0., frequency_penalty=0.,
                    presence_penalty=0., repetition_penalty=1., logits_processors=[],
                    logit_bias={'field_unavailable': True}, allowed_token_ids=None,
                    guided_decoding={'field_unavailable': True}, ignore_eos=False, min_tokens=0,
                    logprob_semantics='native_request_output_only')
    for row in range(len(data)):
        request, response = examples[row % 4]
        trace = dict(actual_request_token_ids=request, raw_generated_token_ids=response,
                     selected_logprobs=[-2.] * len(response), row_id=str(row), policy_snapshot_id='raw-snapshot',
                     sampling=copy.deepcopy(sampling))
        trace['request_id'] = 'native-request-' + str(row)
        proof = dict(source='native_enqueued_sequence_group',
                     observed_stage='after_actual_add_request_before_scheduling',
                     request_id=trace['request_id'], actual_request_token_ids=request,
                     sampling=copy.deepcopy(sampling))
        proof['sha256'] = sha(proof)
        trace['effective_sampler_provenance'] = proof
        trace['native_sha256'] = sha(trace)
        journals[row] = [dict(native_trace=trace, capture_provenance='native_request_output_only',
                              raw_generated_token_ids=response + [0] * (3 - len(response)),
                              actual_request_token_ids=request, row_id=str(row), policy_snapshot_id='raw-snapshot')]
    data.non_tensor_batch['raw_sampling_diagnostics'] = journals
    return journals


def test_raw_context_left_padding_positions_and_original_completions(monkeypatch):
    actor, data, _, _ = aligned_actor(monkeypatch)
    attach_raw(data)
    cache = prepare_raw_diagnostic(data, actor.config)
    assert cache.reason is None and len(cache.records) == 4
    with pytest.raises(FrozenInstanceError):
        cache.slots = 999
    micro, native, selected = raw_microbatch(cache.records[:2], torch.device('cpu'))
    assert micro['input_ids'].tolist() == [[0, 10, 11, 30, 31], [12, 13, 14, 40, 0]]
    assert micro['responses'].tolist() == [[30, 31], [40, 0]]
    assert micro['attention_mask'].tolist() == [[0, 1, 1, 1, 1], [1, 1, 1, 1, 0]]
    assert micro['position_ids'].tolist() == [[0, 0, 1, 2, 3], [0, 1, 2, 3, 0]]
    assert native.tolist() == [[-2., -2.], [-2., 0.]]
    assert selected.tolist() == [[True, True], [True, False]]
    # These original sampler actions are absent from retokenized training IDs.
    assert data.batch['responses'].max().item() == 1


@pytest.mark.parametrize('fault', ['hash', 'context', 'temperature', 'support', 'penalty',
                                   'processor', 'bias', 'guided', 'provenance', 'missing_penalty', 'eos', 'missing_group', 'effective_group_transform'])
def test_invalid_native_trace_or_transform_is_unavailable_not_update_skip(monkeypatch, fault):
    actor, data, _, _ = aligned_actor(monkeypatch)
    journals = attach_raw(data)
    journal = journals[0][0]; trace = journal['native_trace']
    if fault == 'hash':
        trace['selected_logprobs'][0] = -3.
    elif fault == 'context':
        journal['actual_request_token_ids'] = [999]
    elif fault == 'provenance':
        journal['capture_provenance'] = 'wrapped_output_tensor_only'
    elif fault in ('missing_group', 'effective_group_transform'):
        if fault == 'missing_group':
            trace.pop('effective_sampler_provenance')
        else:
            proof = trace['effective_sampler_provenance']
            proof['sampling']['temperature'] = .7
            proof['sha256'] = sha({k: v for k, v in proof.items() if k != 'sha256'})
        trace['native_sha256'] = sha({k: v for k, v in trace.items() if k != 'native_sha256'})
    else:
        sampling = trace['sampling']
        if fault == 'temperature': sampling['temperature'] = .7
        elif fault == 'support': sampling['top_p'] = .9
        elif fault == 'penalty': sampling['repetition_penalty'] = 1.2
        elif fault == 'processor': sampling['logits_processors'] = [dict(python_type='custom.Processor')]
        elif fault == 'bias': sampling['logit_bias'] = {'30': 2.}
        elif fault == 'guided': sampling['guided_decoding'] = dict(regex='restricted')
        elif fault == 'missing_penalty': sampling['frequency_penalty'] = {'field_unavailable': True}
        elif fault == 'eos': sampling['ignore_eos'] = True
        trace['native_sha256'] = sha({k: v for k, v in trace.items() if k != 'native_sha256'})
    cache = prepare_raw_diagnostic(data, actor.config)
    assert cache.reason and not cache.records
    actor._safety_group = None
    report = evaluate_raw_diagnostic(actor, cache)
    assert not report['available'] and report['diagnostic_only']
    assert report['raw_context_cross_backend_k3'] is None


def test_bounded_raw_diagnostic_actual_actor_outer_wiring(monkeypatch):
    actor, data, model, optimizer = aligned_actor(monkeypatch)
    journals = attach_raw(data)
    # Rows outside the first4/rank are not scanned or admitted to this diagnostic.
    journals[4] = [dict(deliberately_invalid=True)]
    original = actor._forward_micro_batch
    raw_calls, training_calls = [], []
    def forward(micro_batch, temperature):
        if 'old_log_probs' in micro_batch:
            training_calls.append(model.training)
            return original(micro_batch, temperature)
        raw_calls.append({k: v.clone() for k, v in micro_batch.items()})
        assert temperature == 1. and not model.training
        values = torch.full(micro_batch['responses'].shape, -.5, dtype=torch.float64) + model.weight.reshape(())
        return torch.zeros_like(values), values
    actor._forward_micro_batch = forward
    result = DataParallelPPOActor.update_policy(actor, data)
    assert optimizer.state[model.weight]['step'].item() == 20
    assert len(raw_calls) == 20  #10 fixed-slot micro forwards at each outer boundary.
    assert training_calls.count(False) == 320 and training_calls.count(True) == 160
    reports = [r['raw_sampler_diagnostic'] for r in result['fresh200_policy_monitor']]
    assert len(reports) == 2 and all(r['available'] for r in reports)
    assert all(r['raw_token_count'] == 8 and r['raw_turn_count'] == 4 for r in reports)
    assert reports[0]['raw_context_cross_backend_k3'] == pytest.approx(math.expm1(1.5) - 1.5)
    assert reports[0]['negative_advantage_raw_token_tail_fraction'] == 1.
    assert reports[1]['raw_context_cross_backend_k3'] != reports[0]['raw_context_cross_backend_k3']
    assert all(r['global_row_budget'] == 32 and r['max_forward_calls_per_scan'] == 10 for r in reports)
    assert all('not_pure_policy_kl' in r['interpretation'] for r in reports)
    assert all(r['sampling_policy_kl'] is None for r in result['fresh200_policy_monitor'])


def test_nonfinite_raw_hf_and_nonconstant_advantage_remain_diagnostic(monkeypatch):
    actor, data, model, optimizer = aligned_actor(monkeypatch)
    attach_raw(data)
    original = actor._forward_micro_batch
    def forward(micro_batch, temperature):
        if 'old_log_probs' in micro_batch:
            return original(micro_batch, temperature)
        values = torch.full(micro_batch['responses'].shape, float('nan'), dtype=torch.float64)
        return torch.zeros_like(values), values
    actor._forward_micro_batch = forward
    result = DataParallelPPOActor.update_policy(actor, data)
    assert result['fresh200_update_state']['status'] == 'COMPLETE'
    assert optimizer.state[model.weight]['step'].item() == 20
    reports = [r['raw_sampler_diagnostic'] for r in result['fresh200_policy_monitor']]
    assert all(not r['available'] and r['unavailable_reason'] == 'nonfinite_hf_raw_diagnostic_or_totals' for r in reports)


def raw_gloo_worker(rank, rendezvous):
    from types import SimpleNamespace
    from omegaconf import OmegaConf
    from verl import DataProto
    torch.distributed.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2)
    model = torch.nn.Linear(1, 1).double()
    data = DataProto.from_dict(dict(responses=torch.ones(1, 1, dtype=torch.long),
                                   loss_mask=torch.ones(1, 1), advantages=-torch.ones(1, 1)))
    journals = attach_raw(data)
    if rank == 1:
        journals[0] = journals[0] + copy.deepcopy(journals[0])
    cfg = OmegaConf.create(dict(state_masking=True))
    calls = []
    phase = ['good']
    def forward(micro_batch, temperature):
        calls.append(True)
        value = float('nan') if rank == 1 and phase[0] == 'nonfinite' else -.5 - rank * .1
        current = torch.full(micro_batch['responses'].shape, value, dtype=torch.float64)
        return torch.zeros_like(current), current
    actor = SimpleNamespace(actor_module=model, _safety_group=None, _forward_micro_batch=forward)
    cache = prepare_raw_diagnostic(data, cfg)
    report = evaluate_raw_diagnostic(actor, cache)
    assert len(calls) == 10 and report['raw_token_count'] == 6 and report['raw_turn_count'] == 3
    expected = ((math.expm1(1.5) - 1.5) * 2 + (math.expm1(1.4) - 1.4) * 4) / 6
    assert report['raw_context_cross_backend_k3'] == pytest.approx(expected)
    calls.clear(); phase[0] = 'nonfinite'
    report = evaluate_raw_diagnostic(actor, cache)
    assert len(calls) == 10 and not report['available']
    assert report['unavailable_reason'] == 'nonfinite_hf_raw_diagnostic_or_totals'
    calls.clear()
    if rank == 1:
        data.non_tensor_batch.pop('raw_sampling_diagnostics')
    report = evaluate_raw_diagnostic(actor, prepare_raw_diagnostic(data, cfg))
    assert not calls and not report['available']
    torch.distributed.destroy_process_group()


def test_real_gloo_equal_raw_forward_slots_and_collective_unavailability(tmp_path):
    torch.multiprocessing.spawn(raw_gloo_worker, args=(str(tmp_path / 'raw_gloo'),), nprocs=2, join=True)


@pytest.mark.parametrize('values', [[-1., 1.], [-1e-10, 1e-10]])
def test_nonconstant_trajectory_advantage_disables_only_raw_tail(monkeypatch, values):
    actor, data, _, _ = aligned_actor(monkeypatch)
    attach_raw(data)
    data.batch['responses'] = torch.ones(320, 2, dtype=torch.long)
    data.batch['loss_mask'] = torch.ones(320, 2)
    data.batch['advantages'] = torch.tensor([values]).repeat(320, 1)
    actor._safety_group = None
    actor._forward_micro_batch = lambda micro_batch, temperature: (
        torch.zeros(micro_batch['responses'].shape), torch.full(micro_batch['responses'].shape, -.5))
    report = evaluate_raw_diagnostic(actor, prepare_raw_diagnostic(data, actor.config))
    assert report['available'] and report['raw_context_cross_backend_k3'] is not None
    assert report['negative_advantage_raw_token_tail_fraction'] is None
    assert not report['tail_estimate_complete']
    assert report['tail_unavailable_reason'] == 'nonconstant_or_missing_original_trajectory_advantage'


def test_real_actor_forward_recomputes_raw_selected_token_probabilities(monkeypatch):
    from types import SimpleNamespace
    actor, data, _, _ = aligned_actor(monkeypatch)
    attach_raw(data)
    class ToyHF(torch.nn.Module):
        def __init__(self):
            super().__init__()
            table = torch.arange(80 * 80, dtype=torch.float64).reshape(80, 80)
            self.table = torch.nn.Parameter(torch.sin(table / 17.))
        def forward(self, input_ids, attention_mask, position_ids, use_cache):
            assert not use_cache
            return SimpleNamespace(logits=self.table[input_ids])
    actor.actor_module = ToyHF()
    actor.use_remove_padding = False
    actor._safety_group = None
    actor._forward_micro_batch = lambda micro_batch, temperature: DataParallelPPOActor._forward_micro_batch(
        actor, micro_batch, temperature)
    cache = prepare_raw_diagnostic(data, actor.config)
    expected = []
    for record in cache.records:
        preceding = (record.request[-1],) + record.response[:-1]
        for previous, selected, native in zip(preceding, record.response, record.native):
            logp = torch.log_softmax(actor.actor_module.table[previous], dim=-1)[selected].item()
            delta = logp - native
            expected.append(math.expm1(delta) - delta)
    report = evaluate_raw_diagnostic(actor, cache)
    assert report['available'] and report['raw_token_count'] == len(expected)
    assert report['raw_context_cross_backend_k3'] == pytest.approx(sum(expected) / len(expected), rel=1e-12, abs=1e-12)
    assert actor.actor_module.training  # Optional diagnostic restores actor mode.
