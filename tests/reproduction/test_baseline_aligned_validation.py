"""Aligned validator uses the training engine and counts physical work once."""
import ast
from pathlib import Path
import sys
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch
from test_baseline_aligned_generation import (ROOT, Manager, GenerationConfig, SearchRollout,
                                               CharTokenizer, ScriptedWorker, Proto)
import test_baseline_aligned_generation as fixtures

qa = __import__('runpy').run_path(str(ROOT / 'search_r1/qa_protocol.py'))

def legal(records):
    return all(qa['parse_model_action'](r['text']) is not None for r in records)

def decode(tokenizer, ids, mask):
    spans, current = [], []
    for token, authored in zip(ids.tolist(), mask.tolist()):
        if authored:
            current.append(token)
        elif current:
            spans.append(''.join(chr(t) for t in current if t)); current = []
    if current:
        spans.append(''.join(chr(t) for t in current if t))
    return spans

source = ast.parse((ROOT / 'search_r1/baselines/bapo_validation.py').read_text())
ns = dict(__package__='search_r1.baselines', time=time, LLMGenerationManager=Manager,
          answer_from_spans=qa['extract_final_answer'], is_idk=lambda a: a == 'I do not know',
          legal_action_records=legal)
exec(compile(ast.Module(body=[n for n in source.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))],
                        type_ignores=[]), 'bapo_validation.py', 'exec'), ns)
ValidationManager = ns['BapoValidationManager']

class AlignedValidationTests(unittest.TestCase):
    def patches(self):
        return patch.dict(sys.modules, {'verl': NS(DataProto=Proto),
            'search_r1.baselines.rollout': NS(SearchRollout=SearchRollout),
            'verl.trainer.main_ppo': NS(decode_model_response_spans=decode)})

    def manager(self):
        tokenizer = CharTokenizer(); worker = ScriptedWorker(tokenizer)
        cfg = GenerationConfig(max_turns=4, max_start_length=16, max_prompt_length=1000,
            max_response_length=32, max_obs_length=80, num_gpus=4, baseline_aligned=True,
            capture_raw_sampling=True)
        manager = ValidationManager(tokenizer, worker, cfg, True)
        manager.batch_search = lambda queries: ['Doc 1(Title: title) passage\n'] * len(queries)
        return manager, worker

    def test_validation_uses_greedy_shared_flow_and_counts_padding_once(self):
        manager, worker = self.manager()
        batch = fixtures.AlignedGenerationParity().input(dict(do_sample=False, policy_snapshot_id='snapshot'))
        with self.patches():
            output = manager.run_llm_loop(batch, batch.batch['input_ids'])
        self.assertEqual(len(worker.calls), 5)
        self.assertEqual(manager.cost['generation_rows'], 20)
        expected_prefill = sum(int(call[0]['attention_mask'].sum()) for call in worker.calls)
        self.assertEqual(manager.cost['prefill_tokens'], expected_prefill)
        self.assertEqual(manager.cost['generated_tokens'], worker.raw_token_count)
        # All observed RPC work is counted once; diagnostics include padding rows.
        self.assertGreater(manager.cost['generated_tokens'], int(output.batch['info_mask'].sum()))
        self.assertEqual(manager.cost['trajectories'], 3)
        self.assertEqual(manager.answer_counts, dict(idk_count=0, answered_count=1, invalid_count=2))
        for _, meta in worker.calls:
            self.assertFalse(meta['do_sample']); self.assertTrue(meta['fresh200_evaluation'])
            self.assertTrue(meta['capture_raw_sampling'])
        self.assertIn('raw_sampling_diagnostics', output.non_tensor_batch)
        self.assertTrue(output.meta_info['baseline_aligned'])
        self.assertNotIn('sampling_log_probs', output.batch)

    def test_validation_refuses_stochastic_or_unidentified_capture(self):
        for meta in (dict(do_sample=True, policy_snapshot_id='snapshot'), dict(do_sample=False)):
            manager, _ = self.manager(); batch = fixtures.AlignedGenerationParity().input(meta)
            with self.patches(), self.assertRaises(ValueError):
                manager.run_llm_loop(batch, batch.batch['input_ids'])

    def test_aligned_answer_counts_grade_terminal_span_after_prior_invalid_action(self):
        tokenizer = CharTokenizer()
        texts = ['invalid<information>x</information><answer>Paris</answer>']
        ids = tokenizer(texts)['input_ids']; mask = (ids != 0).long()
        start = len('invalid'); end = len('invalid<information>x</information>')
        mask[:, start:end] = 0
        output = Proto.from_dict(dict(responses=ids, attention_mask=(ids != 0).long(), info_mask=mask))
        with self.patches():
            self.assertEqual(ns['validation_answer_counts'](output, tokenizer, 'baseline_aligned'),
                             dict(idk_count=0, answered_count=1, invalid_count=0))
            self.assertEqual(ns['validation_answer_counts'](output, tokenizer, 'legacy'),
                             dict(idk_count=0, answered_count=0, invalid_count=1))

if __name__ == '__main__':
    unittest.main()
