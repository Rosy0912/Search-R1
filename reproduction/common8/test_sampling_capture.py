"""CPU protocol checks of actual production capture helpers, without vLLM/GPU imports.

AST extraction executes the real pure helper bodies, not a copied implementation.
It does not authenticate native sampling or prove model probability parity.
"""
import ast
import copy
import math
import os
from pathlib import Path
from types import SimpleNamespace as S
from typing import List, Tuple
import torch
from torch.nn.utils.rnn import pad_sequence
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


def helper(relative, name):
    tree = ast.parse((ROOT / relative).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = dict(math=math, copy=copy)
    exec(compile(ast.Module(body=[node], type_ignores=[]), relative, 'exec'), namespace)
    return namespace[name]


capture = helper('verl/third_party/vllm/vllm_v_0_6_3/llm.py', 'capture_request_behavior')
sampler = helper('verl/workers/rollout/vllm_rollout/vllm_rollout.py', 'actual_sampling_metadata')


def actual_output_method():
    relative = 'verl/third_party/vllm/vllm_v_0_6_3/llm.py'
    tree = ast.parse((ROOT / relative).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LLM')
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_post_process_outputs')
    namespace = dict(torch=torch, pad_sequence=pad_sequence, capture_request_behavior=capture,
                     List=List, Tuple=Tuple, RequestOutput=object)
    exec(compile(ast.Module(body=[node], type_ignores=[]), relative, 'exec'), namespace)
    return namespace['_post_process_outputs']


def request(tokens=(5, 7), values=(-.2, -.3), name='actual-1'):
    return S(request_id=name, prompt_token_ids=[7, 8], finished=True,
             outputs=[S(token_ids=list(tokens), logprobs=[{token: S(logprob=value)}
                          for token, value in zip(tokens, values)], finish_reason='stop', stop_reason=7)])


def parameters():
    return S(n=1, best_of=1, temperature=1.0, top_p=1.0, top_k=-1, min_p=0.,
             presence_penalty=0., frequency_penalty=0., repetition_penalty=1.,
             logprobs=1, min_tokens=0, max_tokens=500, logits_processors=None,
             truncate_prompt_tokens=None, ignore_eos=False, stop=['</answer>'], stop_token_ids=[7])


class SamplingCaptureTests(unittest.TestCase):
    def test_actual_fsdp_guard_rejects_hidden_recompute_and_sampled_smoke(self):
        relative = 'verl/workers/fsdp_workers.py'
        tree = ast.parse((ROOT / relative).read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ActorRolloutRefWorker')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'generate_sequences')
        guard = method.body[0]
        self.assertIsInstance(guard, ast.If)
        namespace = dict(os=os)
        code = compile(ast.Module(body=[guard], type_ignores=[]), relative, 'exec')
        with patch.dict(os.environ, {'SEARCH_G1_PROFILE': 'common8'}):
            for meta in ({}, {'recompute_log_prob': True}, {'common8_model_smoke': True, 'do_sample': True}):
                namespace['prompts'] = S(meta_info=meta)
                with self.assertRaises(ValueError): exec(code, namespace)
            for meta in ({'recompute_log_prob': False},
                         {'common8_model_smoke': True, 'do_sample': False, 'recompute_log_prob': True}):
                namespace['prompts'] = S(meta_info=meta); exec(code, namespace)
        with patch.dict(os.environ, {'SEARCH_G1_PROFILE': 'a800'}):
            namespace['prompts'] = S(meta_info={}); exec(code, namespace)

    def test_real_postprocess_legacy_tuple_and_common8_prepad_records(self):
        method = actual_output_method()
        worker = S(llm_engine=S(tokenizer=S(pad_token_id=7, eos_token_id=7)))
        requests = [request(), request(tokens=(7,), values=(-1.25,), name='actual-2')]
        legacy = method(worker, requests)
        self.assertEqual(len(legacy), 2)
        worker.capture_behavior_trace = True
        tokens, probabilities, records = method(worker, requests)
        self.assertEqual(tokens.tolist(), [[5, 7], [7, 7]])
        self.assertEqual(records[1]['raw_generated_token_count'], 1)
        self.assertEqual(records[1]['selected_logprobs'], [-1.25])
        self.assertEqual(probabilities[1, 0].item(), -1.25)
        self.assertTrue(torch.equal(tokens, legacy[0]))
        self.assertTrue(torch.equal(probabilities, legacy[1]))

    def test_raw_eos_equal_padding_is_retained_by_actual_length(self):
        rows = capture([request(), request(tokens=(7,), values=(-1.25,), name='actual-2')])
        self.assertEqual([r['raw_generated_token_count'] for r in rows], [2, 1])
        self.assertEqual(rows[1]['raw_generated_token_ids'], [7])
        self.assertEqual(rows[1]['selected_logprobs'], [-1.25])
        self.assertEqual(rows[0]['actual_request_token_ids'], [7, 8])
        self.assertEqual(rows[0]['finish_reason'], 'stop')
        self.assertEqual(rows[0]['stop_reason'], 7)

    def test_missing_truncated_nonfinite_logprob_reject(self):
        for modify in (lambda r: setattr(r.outputs[0], 'logprobs', None),
                       lambda r: r.outputs[0].logprobs.pop(),
                       lambda r: r.outputs[0].logprobs.__setitem__(0, {}),
                       lambda r: setattr(r.outputs[0].logprobs[0][5], 'logprob', float('nan'))):
            r = request(); modify(r)
            with self.assertRaises(ValueError): capture([r])

    def test_incomplete_n2_invalid_tokens_duplicate_request_reject(self):
        for modify in (lambda r: setattr(r, 'finished', False),
                       lambda r: r.outputs.append(r.outputs[0]),
                       lambda r: setattr(r, 'prompt_token_ids', [True])):
            r = request(); modify(r)
            with self.assertRaises(ValueError): capture([r])
        with self.assertRaises(ValueError): capture([request(), request()])

    def test_capture_does_not_alias_mutable_request(self):
        r = request(); row = capture([r])[0]
        r.outputs[0].token_ids[0] = 99; r.prompt_token_ids[0] = 99
        self.assertEqual(row['raw_generated_token_ids'], [5, 7]); self.assertEqual(row['actual_request_token_ids'], [7, 8])

    def test_effective_neutral_train_and_greedy_metadata(self):
        p = parameters(); result = sampler(p, True)
        self.assertEqual(result['temperature'], 1.0)
        self.assertEqual(result['logprob_semantics'], 'selected_token_logprob_at_actual_sampling_call')
        p.stop.append('changed'); self.assertEqual(result['stop'], ['</answer>'])
        p.temperature = 0.; self.assertEqual(sampler(p, False)['temperature'], 0.)

    def test_actual_transformed_policy_reject(self):
        for key, value in [('top_p', .95), ('temperature', .9), ('n', 2),
                           ('logits_processors', [object()]), ('repetition_penalty', 1.01),
                           ('min_tokens', 1), ('ignore_eos', True), ('truncate_prompt_tokens', 20),
                           ('logit_bias', {5: 1.}), ('allowed_token_ids', [5])]:
            p = parameters(); setattr(p, key, value)
            with self.subTest(key=key), self.assertRaises(ValueError): sampler(p, True)


if __name__ == '__main__':
    unittest.main()
