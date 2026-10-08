"""Actual vLLM rollout method with a CPU engine fixture: capture is observational."""
import ast
import copy
from contextlib import contextmanager, nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace as NS
from typing import List
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

class Proto:
    def __init__(self, batch, non_tensor_batch=None, meta_info=None):
        self.batch, self.non_tensor_batch, self.meta_info = batch, non_tensor_batch or {}, meta_info or {}
    def __len__(self):
        return self.batch['responses'].shape[0] if 'responses' in self.batch else self.batch['input_ids'].shape[0]

class TensorDict(dict):
    def __init__(self, value, batch_size):
        super().__init__(value)

def pad(tensor, length, pad):
    return torch.cat((tensor, torch.full((len(tensor), length-tensor.shape[1]), pad, dtype=tensor.dtype)), 1)

def eos_mask(response_id, eos_token, dtype):
    eos = response_id == eos_token
    return (eos.cumsum(1) - eos.long() == 0).to(dtype)

source = ast.parse((ROOT / 'verl/workers/rollout/vllm_rollout/vllm_rollout.py').read_text())
ns = dict(torch=torch, np=np, contextmanager=contextmanager, nullcontext=nullcontext,
          List=List, BaseRollout=object, nn=NS(Module=object), DictConfig=object,
          DataProto=Proto, TensorDict=TensorDict, pad_sequence_to_length=pad, get_eos_mask=eos_mask)
exec(compile(ast.Module(body=[n for n in source.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))],
                        type_ignores=[]), 'vllm_rollout.py', 'exec'), ns)
Rollout = ns['vLLMRollout']

class NativeGroups:
    def __init__(self, changes=None, inaccessible=False):
        self.scheduler = NS(waiting=[])
        self.changes, self.inaccessible = changes or {}, inaccessible
        self.calls = []
    def add_request(self, request_id, request, sampling):
        self.calls.append((request_id, list(request)))
        effective = copy.deepcopy(sampling)
        for key, value in self.changes.items(): setattr(effective, key, value)
        if not self.inaccessible:
            self.scheduler.waiting.append(NS(request_id=request_id, prompt_token_ids=list(request),
                sampling_params=effective, get_seqs=lambda: [NS(seq_id=0)]))

class Engine:
    def __init__(self, hook=True, missing_probabilities=False, actual_groups=False, group_fields=None, group_error=False):
        self.calls = []
        self.missing_probabilities = missing_probabilities
        if actual_groups:
            self.llm_engine = NativeGroups(group_fields, group_error)
        if hook:
            self._post_process_outputs = self.process
    def process(self, outputs):
        return torch.tensor([[7, 0, 9]] * len(outputs)), torch.tensor([[-.2, -.3, -.4]] * len(outputs))
    def generate(self, **kwargs):
        self.calls.append(kwargs['prompt_token_ids'])
        if hasattr(self, 'llm_engine'):
            for i, request in enumerate(kwargs['prompt_token_ids']):
                self.llm_engine.add_request(str(i), request, kwargs['sampling_params'])
        outputs = [NS(prompt_token_ids=request, request_id=str(i), outputs=[NS(
            token_ids=[7, 0, 9], logprobs=None if self.missing_probabilities else
            [{t: NS(logprob=p)} for t, p in zip([7, 0, 9], [-.2, -.3, -.4])],
            finish_reason='length', stop_reason=None)]) for i, request in enumerate(kwargs['prompt_token_ids'])]
        return self._post_process_outputs(outputs) if hasattr(self, '_post_process_outputs') else self.process(outputs)

class RawCaptureTests(unittest.TestCase):
    def run_capture(self, capture, hook=True, missing=False, sample=True, sampler_fields=None,
                    actual_groups=False, group_fields=None, group_error=False):
        rollout = Rollout.__new__(Rollout)
        rollout.config = NS(free_cache_engine=False, n=1, response_length=5)
        rollout.pad_token_id = 0
        rollout.sampling_params = NS(n=1, temperature=.7, top_p=.9, top_k=10, min_p=0., max_tokens=5,
                                     best_of=1, logprobs=1)
        for key, value in (sampler_fields or {}).items():
            setattr(rollout.sampling_params, key, value)
        rollout.inference_engine = Engine(hook, missing, actual_groups, group_fields, group_error)
        ids = torch.tensor([[0, 11, 0, 12]]) # Archive preprocessing preserves the embedded pad.
        batch = Proto(dict(input_ids=ids, attention_mask=(ids != 0).long(),
                           position_ids=torch.tensor([[0, 0, 0, 1]])),
                      {'fresh200_row_id': np.array(['real-row'], dtype=object)},
                      dict(eos_token_id=0, capture_raw_sampling=capture, policy_snapshot_id='snapshot',
                           fresh200_call_id='call', fresh200_turn_index=2, do_sample=sample))
        original = getattr(rollout.inference_engine, '_post_process_outputs', None)
        group_original = getattr(getattr(rollout.inference_engine, 'llm_engine', None), 'add_request', None)
        result = rollout.generate_sequences(batch)
        self.assertEqual(getattr(rollout.inference_engine, '_post_process_outputs', None), original)
        self.assertEqual(getattr(getattr(rollout.inference_engine, 'llm_engine', None), 'add_request', None), group_original)
        return result, rollout

    def test_capture_changes_no_archived_tensor_request_or_sampler(self):
        for sample in (True, False):
            ordinary, baseline = self.run_capture(False, sample=sample)
            captured, diagnostic = self.run_capture(True, sample=sample)
            for key, tensor in ordinary.batch.items():
                self.assertTrue(torch.equal(tensor, captured.batch[key]), key)
            self.assertEqual(baseline.inference_engine.calls, diagnostic.inference_engine.calls)
            self.assertEqual(diagnostic.inference_engine.calls, [[[11, 0, 12]]])
            self.assertEqual(diagnostic.sampling_params.temperature, .7)
            self.assertNotIn('sampling_mask', captured.batch)
            self.assertNotIn('sampling_log_probs', captured.batch)
            self.assertEqual(captured.batch['raw_sampling_log_probs'].tolist()[0][:3],
                             torch.tensor([-.2, -.3, -.4]).tolist())
            record = captured.non_tensor_batch['fresh200_sampling'][0]
            self.assertEqual(record['raw_generated_token_ids'], [7, 0, 9])
            self.assertEqual(record['actual_request_token_ids'], [11, 0, 12])
            self.assertEqual(record['row_id'], 'real-row')
            self.assertEqual(record['policy_snapshot_id'], 'snapshot')
            self.assertEqual(record['turn_index'], 2)
            self.assertEqual(captured.meta_info['raw_sampling_capture_provenance'], 'native_request_output_only')
            self.assertEqual(record['sampling']['frequency_penalty'], {'field_unavailable': True})

    def test_unavailable_capture_falls_back_without_native_probability_claim(self):
        for hook, missing in ((False, False), (True, True)):
            ordinary, _ = self.run_capture(False, hook=hook, missing=missing)
            captured, _ = self.run_capture(True, hook=hook, missing=missing)
            self.assertNotIn('fresh200_sampling', captured.non_tensor_batch)
            self.assertTrue(captured.meta_info['raw_sampling_capture_provenance'].startswith('wrapped_output_tensor_only'))
            for key, tensor in ordinary.batch.items():
                self.assertTrue(torch.equal(tensor, captured.batch[key]), key)
            self.assertIn('raw_sampling_log_probs', captured.batch)

    def test_actual_transform_metadata_is_serialized_without_neutral_defaults(self):
        processor = lambda value: value
        fields = dict(frequency_penalty=.4, presence_penalty=.2, repetition_penalty=1.1,
                      logits_processors=[processor], logit_bias={7: -.6}, allowed_token_ids=[7, 9])
        plain, _ = self.run_capture(False, sampler_fields=fields)
        captured, _ = self.run_capture(True, sampler_fields=fields)
        sampling = captured.non_tensor_batch['fresh200_sampling'][0]['sampling']
        for name in ('frequency_penalty', 'presence_penalty', 'repetition_penalty', 'allowed_token_ids'):
            self.assertEqual(sampling[name], fields[name])
        self.assertEqual(sampling['logit_bias'], {'7': -.6})
        self.assertEqual(sampling['logits_processors'][0]['python_type'], 'builtins.function')
        self.assertIn('function', sampling['logits_processors'][0]['representation'])
        for key, tensor in plain.batch.items():
            self.assertTrue(torch.equal(tensor, captured.batch[key]), key)

    def diagnostic_cache(self, output):
        from search_r1.baselines.raw_sampler_diagnostic import prepare_raw_diagnostic
        trace = output.non_tensor_batch['fresh200_sampling'][0]
        # Model actions can differ from the raw sampler tokens: the actual
        # callback journal is bound to its raw context, not these training IDs.
        data = Proto({'responses': torch.tensor([[20, 21]]), 'loss_mask': torch.ones(1, 2),
                      'advantages': -torch.ones(1, 2)},
                     {'raw_sampling_diagnostics': [[dict(native_trace=trace,
                        capture_provenance=output.meta_info['raw_sampling_capture_provenance'],
                        actual_request_token_ids=trace['actual_request_token_ids'],
                        raw_generated_token_ids=output.batch['responses'][0].tolist(),
                        row_id='real-row', policy_snapshot_id='snapshot')]]})
        class Config(dict):
            state_masking = True
        return prepare_raw_diagnostic(data, Config(raw_sampler_diagnostic_rows_per_rank=1))

    def test_actual_group_callback_to_raw_helper_produces_computed_diagnostic(self):
        from search_r1.baselines.raw_sampler_diagnostic import evaluate_raw_diagnostic
        fields = dict(temperature=1., top_p=1., top_k=-1, min_p=0.,
            frequency_penalty=0., presence_penalty=0., repetition_penalty=1.,
            logits_processors=[], logit_bias=None, allowed_token_ids=None, guided_decoding=None,
            ignore_eos=False, min_tokens=0)
        plain, ordinary = self.run_capture(False, sampler_fields=fields, actual_groups=True)
        captured, diagnostic = self.run_capture(True, sampler_fields=fields, actual_groups=True)
        for key, tensor in plain.batch.items():
            self.assertTrue(torch.equal(tensor, captured.batch[key]), key)
        self.assertEqual(ordinary.inference_engine.llm_engine.calls, diagnostic.inference_engine.llm_engine.calls)
        trace = captured.non_tensor_batch['fresh200_sampling'][0]
        self.assertEqual(trace['sampling']['logprob_semantics'], 'native_request_output_only')
        self.assertEqual(trace['effective_sampler_provenance']['sampling']['temperature'], 1.)
        cache = self.diagnostic_cache(captured)
        self.assertIsNone(cache.reason)
        self.assertEqual(cache.records[0].request, (11, 0, 12))
        self.assertEqual(cache.records[0].response, (7, 0, 9))
        torch.manual_seed(7)
        model = torch.nn.Sequential(torch.nn.Embedding(32, 8), torch.nn.Linear(8, 32))
        def forward(micro, temperature):
            width = micro['responses'].shape[1]
            logits = model(micro['input_ids'])[:, -width-1:-1] / temperature
            selected = logits.log_softmax(-1).gather(-1, micro['responses'].unsqueeze(-1)).squeeze(-1)
            return torch.zeros_like(selected), selected
        actor = NS(actor_module=model, _safety_group=None, _forward_micro_batch=forward)
        report = evaluate_raw_diagnostic(actor, cache)
        self.assertTrue(report['available'])
        self.assertEqual(report['raw_token_count'], 3)
        self.assertIsInstance(report['raw_context_cross_backend_k3'], float)

    def test_effective_group_transform_or_missing_proof_fails_diagnostic_only(self):
        fields = dict(temperature=1., top_p=1., top_k=-1, min_p=0.,
            frequency_penalty=0., presence_penalty=0., repetition_penalty=1.,
            logits_processors=[], logit_bias=None, allowed_token_ids=None, guided_decoding=None,
            ignore_eos=False, min_tokens=0)
        for options in (dict(actual_groups=True, group_fields=dict(temperature=.5)),
                        dict(actual_groups=True, group_fields=dict(allowed_token_ids=[7])),
                        dict(actual_groups=True, group_error=True), dict(actual_groups=False)):
            plain, ordinary = self.run_capture(False, sampler_fields=fields, **options)
            captured, diagnostic = self.run_capture(True, sampler_fields=fields, **options)
            for key, tensor in plain.batch.items():
                self.assertTrue(torch.equal(tensor, captured.batch[key]), key)
            self.assertEqual(ordinary.inference_engine.calls, diagnostic.inference_engine.calls)
            cache = self.diagnostic_cache(captured)
            self.assertIsNotNone(cache.reason)
            self.assertFalse(cache.records)

if __name__ == '__main__':
    unittest.main()
