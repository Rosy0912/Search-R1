"""CPU parity between frozen Search-E1 generation semantics and BAPO rollout.

Uses actual production methods with only RPC/DataProto/tokenizer fixtures. No
model download or distributed runtime is required by this controller test.
"""
import ast
from dataclasses import dataclass
from pathlib import Path
import re
import runpy
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]


class Proto:
    def __init__(self, batch, non_tensors=None, meta_info=None):
        self.batch = batch
        self.non_tensor_batch = non_tensors or {}
        self.meta_info = meta_info or {}

    def __len__(self):
        return self.batch['input_ids'].shape[0]

    @classmethod
    def from_dict(cls, tensors, non_tensors=None, meta_info=None):
        return cls(tensors, non_tensors, meta_info)


helper = runpy.run_path(str(ROOT / 'search_r1/llm_agent/tensor_helper.py'))
source = ast.parse((ROOT / 'search_r1/llm_agent/generation.py').read_text())
namespace = dict(torch=torch, re=re, dataclass=dataclass, List=List, Dict=Dict,
                 Any=Any, Tuple=Tuple, DataProto=Proto,
                 TensorHelper=helper['TensorHelper'], TensorConfig=helper['TensorConfig'])
exec(compile(ast.Module(body=[n for n in source.body if isinstance(n, ast.ClassDef)],
                        type_ignores=[]), 'generation.py', 'exec'), namespace)
GenerationConfig, Manager = namespace['GenerationConfig'], namespace['LLMGenerationManager']
SearchRollout = runpy.run_path(str(ROOT / 'search_r1/baselines/rollout.py'))['SearchRollout']


class CharTokenizer:
    pad_token_id = 0
    pad_token = '<pad>'

    def __call__(self, texts, **kwargs):
        rows = [[ord(c) for c in text] for text in texts]
        width = max(map(len, rows), default=0)
        return {'input_ids': torch.tensor([r + [0] * (width - len(r)) for r in rows], dtype=torch.long)}

    def batch_decode(self, rows, **kwargs):
        return [''.join(chr(t) for t in row if t) for row in rows.tolist()]


class ScriptedWorker:
    def __init__(self, tokenizer):
        self.tokenizer, self.calls = tokenizer, []
        self.identities = []
        self.raw_token_count = 0

    def generate_sequences(self, batch):
        self.calls.append(({k: v.clone() for k, v in batch.batch.items()}, dict(batch.meta_info)))
        self.identities.append(batch.non_tensor_batch.get('fresh200_row_id'))
        turn = len(self.calls)
        # First row terminates immediately; remaining rows survive four turns.
        texts = (['<answer>done</answer>ignored', 'invalid', '<search>q</search>ignored'] if turn == 1
                 else ['<search>q</search>ignored', '<search>q</search>ignored'] if turn < 5
                 else ['<search>q</search>ignored', 'invalid'])
        # GPU padding always duplicates the first active row.
        texts += [texts[0]] * (len(batch) - len(texts))
        tokens = self.tokenizer(texts)['input_ids']
        self.raw_token_count += tokens.numel()
        return Proto.from_dict({'responses': tokens,
                                'sampling_log_probs': torch.full(tokens.shape, -.5)},
                               non_tensors={'fresh200_sampling': [dict(request_id=str(i),
                                   raw_generated_token_ids=tokens[i].tolist()) for i in range(len(texts))]},
                               meta_info={'temperature': 0.7})


class AlignedGenerationParity(unittest.TestCase):
    def manager(self, cap, obs_cap, aligned=True):
        tokenizer = CharTokenizer()
        worker = ScriptedWorker(tokenizer)
        cfg = GenerationConfig(max_turns=4, max_start_length=16, max_prompt_length=cap,
                               max_response_length=32, max_obs_length=obs_cap,
                               num_gpus=4, baseline_aligned=aligned)
        manager = Manager(tokenizer, worker, cfg)
        queries = []
        def retrieve(batch_queries):
            queries.append(list(batch_queries))
            return ['Doc 1(Title: title) passage\n'] * len(batch_queries)
        manager.batch_search = retrieve
        return manager, worker, queries

    def input(self, meta):
        ids = torch.tensor([[0] * 15 + [65], [0] * 14 + [66, 67], [0] * 13 + [68, 69, 70]])
        attention = (ids != 0).long()
        return Proto.from_dict(dict(input_ids=ids, attention_mask=attention,
                                    position_ids=helper['TensorHelper'].create_position_ids(None, attention)),
                               meta_info=dict(meta))

    def test_raw_only_tensordict_sampling_probabilities(self):
        from tensordict import TensorDict
        manager, worker, _ = self.manager(1000, 80)
        original = worker.generate_sequences
        def generate(batch):
            output = original(batch)
            values = dict(output.batch)
            values['raw_sampling_log_probs'] = values.pop('sampling_log_probs')
            output.batch = TensorDict(values, batch_size=[len(batch)])
            return output
        worker.generate_sequences = generate
        engine = SearchRollout(manager)
        with patch.dict(sys.modules, {'verl': SimpleNamespace(DataProto=Proto)}):
            output, rows = engine.run(self.input(dict(do_sample=False, validate=True)))
        self.assertEqual(rows.tolist(), [0, 1, 2])
        self.assertEqual(len(worker.calls), 5)
        self.assertEqual(output.meta_info['turns_stats'], [1, 5, 5])

    def test_train_and_eval_tensors_requests_and_retrieval_match(self):
        # Both right-side truncation and rolling context truncation are exercised.
        for cap, obs_cap in ((1000, 80), (90, 24)):
            for do_sample in (True, False):
                with self.subTest(cap=cap, do_sample=do_sample):
                    metadata = dict(do_sample=do_sample, validate=not do_sample, sentinel='same')
                    eval_manager, eval_worker, eval_queries = self.manager(cap, obs_cap)
                    eval_batch = self.input(metadata)
                    evaluation = eval_manager.run_llm_loop(eval_batch, eval_batch.batch['input_ids'])
                    train_manager, train_worker, train_queries = self.manager(cap, obs_cap)
                    train_batch = self.input(metadata)
                    engine = SearchRollout(train_manager)
                    with patch.dict(sys.modules, {'verl': SimpleNamespace(DataProto=Proto)}):
                        training, source_rows = engine.run(train_batch)
                    self.assertEqual(source_rows.tolist(), [0, 1, 2])
                    for key, expected in evaluation.batch.items():
                        self.assertTrue(torch.equal(training.batch[key], expected), key)
                    for key in ('turns_stats', 'active_mask', 'valid_action_stats', 'valid_search_stats'):
                        self.assertEqual(training.meta_info[key], evaluation.meta_info[key], key)
                    self.assertEqual(evaluation.meta_info['turns_stats'], [1, 5, 5])
                    self.assertEqual(len(eval_worker.calls), 5)
                    self.assertEqual(len(eval_queries), 4) # Fifth generation never retrieves.
                    self.assertEqual(eval_queries, train_queries)
                    for eval_call, train_call in zip(eval_worker.calls, train_worker.calls):
                        self.assertEqual(eval_call[1], train_call[1])
                        for key in eval_call[0]:
                            self.assertTrue(torch.equal(eval_call[0][key], train_call[0][key]), key)
                    self.assertEqual(evaluation.batch['prompts'].shape[1], 16)
                    self.assertNotIn('sampling_log_probs', training.batch)
                    self.assertEqual(training.meta_info['raw_sampling_probability_alignment'], 'raw_sampler_only')
                    diagnostics = training.non_tensor_batch['raw_sampling_diagnostics'][0]
                    self.assertEqual(len(diagnostics), 1)
                    raw_text = ''.join(chr(t) for t in diagnostics[0]['raw_generated_token_ids'] if t)
                    self.assertEqual(raw_text, '<answer>done</answer>ignored')
                    self.assertEqual(diagnostics[0]['native_trace']['request_id'], '0')
                    self.assertEqual(diagnostics[0]['raw_selected_logprobs'][0], -.5)
                    if cap == 1000:
                        # Final search is counted by the historical executor but
                        # its empty information observation is never appended.
                        decoded = train_manager.tokenizer.batch_decode(training.batch['responses'])
                        self.assertTrue(decoded[1].endswith('<search>q</search>'))
                        self.assertTrue(decoded[2].endswith('invalid'))

    def test_archive_truncation_prioritizes_search_and_retokenizes(self):
        manager, _, _ = self.manager(1000, 80)
        raw = manager.tokenizer(['<answer>a</answer>suffix<search>q</search>discard',
                                 '<search>q</search>suffix', '<answer>a</answer>suffix'])['input_ids']
        ids, text = manager._postprocess_responses(raw)
        self.assertEqual(text, ['<answer>a</answer>suffix<search>q</search>',
                                '<search>q</search>', '<answer>a</answer>'])
        self.assertEqual(manager.tokenizer.batch_decode(ids), text)

    def test_aligned_profile_rejects_budget_token_and_retrieval_drift(self):
        base = dict(max_turns=4, max_start_length=16, max_prompt_length=1000,
                    max_response_length=32, max_obs_length=80, num_gpus=1, baseline_aligned=True)
        for change in (dict(max_turns=3), dict(topk=4), dict(force_final_answer=False),
                       dict(preserve_sampling_tokens=True)):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'baseline_aligned'):
                GenerationConfig(**(base | change))
        # Existing legacy settings continue to be accepted without this profile.
        GenerationConfig(**(base | dict(baseline_aligned=False, max_turns=3,
                                       force_final_answer=False, preserve_sampling_tokens=True)))

    def test_topk_three_and_reference_format_match_archive(self):
        manager, _, _ = self.manager(1000, 80)
        payloads = []
        def post(url, json):
            payloads.append(json)
            return SimpleNamespace(json=lambda: {'result': [[{'document': {'contents': 'title\nline1\nline2'}}]]})
        namespace['requests'] = SimpleNamespace(post=post)
        # Use the unmodified production batch_search instead of the loop fixture.
        actual = Manager.batch_search(manager, ['q'])
        self.assertEqual(payloads, [{'queries': ['q'], 'topk': 3, 'return_scores': True}])
        self.assertEqual(actual, ['Doc 1(Title: title) line1\nline2\n'])

    def test_raw_capture_request_metadata_preserves_training_and_eval_outputs(self):
        for sample in (True, False):
            outputs = []
            for enabled in (False, True):
                manager, worker, _ = self.manager(1000, 80)
                manager.config.capture_raw_sampling = enabled
                batch = self.input(dict(do_sample=sample, policy_snapshot_id='snapshot'))
                with patch.dict(sys.modules, {'verl': SimpleNamespace(DataProto=Proto)}):
                    output, _ = SearchRollout(manager).run(batch)
                outputs.append(output)
                if enabled:
                    for ordinal, (_, meta) in enumerate(worker.calls):
                        self.assertTrue(meta['capture_raw_sampling'])
                        self.assertEqual(meta['fresh200_turn_index'], ordinal)
                        self.assertEqual(meta['policy_snapshot_id'], 'snapshot')
                        self.assertEqual(meta['fresh200_evaluation'], not sample)
                        self.assertEqual(len(worker.identities[ordinal]), 4)
                        self.assertEqual(len(set(worker.identities[ordinal])), 4)
                    self.assertEqual(len({meta['fresh200_call_id'] for _, meta in worker.calls}), 5)
                    self.assertNotEqual(worker.identities[0][0], worker.identities[1][0])
            for key, tensor in outputs[0].batch.items():
                self.assertTrue(torch.equal(tensor, outputs[1].batch[key]), key)


if __name__ == '__main__':
    unittest.main()
