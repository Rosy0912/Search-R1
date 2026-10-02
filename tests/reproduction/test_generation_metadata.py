"""CPU regression for metadata across active filtering and GPU-size padding.

Run: python -m unittest discover -s tests/reproduction -p test_generation_metadata.py
The deployed environment imports the real DataProto. A lightweight fallback
executes the identical source class methods with a DataProto stand-in when the
review workstation lacks the optional distributed training dependencies.
"""
import ast
from dataclasses import dataclass
import os
from pathlib import Path
import re
import runpy
import sys
import unittest
from typing import Any, Dict, List, Tuple

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
try:
    from search_r1.llm_agent.generation import GenerationConfig, LLMGenerationManager
    from verl import DataProto
    REAL_DATAPROTO = True
except ImportError:
    if os.environ.get('SEARCH_R1_REQUIRE_REAL_DATAPROTO') == '1':
        raise
    REAL_DATAPROTO = False

    class DataProto:
        def __init__(self, batch, meta_info=None):
            self.batch = batch
            self.meta_info = {} if meta_info is None else meta_info

        @classmethod
        def from_dict(cls, tensors, meta_info=None):
            return cls(tensors, meta_info)

    helper = runpy.run_path(str(ROOT / 'search_r1/llm_agent/tensor_helper.py'))
    source = ast.parse((ROOT / 'search_r1/llm_agent/generation.py').read_text())
    classes = [node for node in source.body if isinstance(node, ast.ClassDef)]
    namespace = dict(torch=torch, re=re, dataclass=dataclass, List=List,
                     Dict=Dict, Any=Any, Tuple=Tuple, DataProto=DataProto,
                     TensorHelper=helper['TensorHelper'], TensorConfig=helper['TensorConfig'])
    exec(compile(ast.Module(body=classes, type_ignores=[]), 'generation.py', 'exec'), namespace)
    GenerationConfig = namespace['GenerationConfig']
    LLMGenerationManager = namespace['LLMGenerationManager']


class Tokenizer:
    pad_token_id = 0
    pad_token = '<pad>'

    def __call__(self, strings, **kwargs):
        return {'input_ids': torch.tensor([[7] if text else [0] for text in strings])}


class RecordingWorker:
    def __init__(self):
        self.calls = []

    def generate_sequences(self, data):
        self.calls.append((data.batch['input_ids'].shape[0], data.meta_info.copy()))
        # Workers may add bookkeeping. Do not mutate the caller's dictionary.
        data.meta_info['worker_bookkeeping'] = True
        return DataProto.from_dict(
            {'responses': torch.full((data.batch['input_ids'].shape[0], 1), 9)},
            meta_info={'temperature': 1.0},
        )


class MetadataRegression(unittest.TestCase):
    def manager(self, num_gpus):
        worker = RecordingWorker()
        config = GenerationConfig(max_turns=1, max_start_length=8,
                                  max_prompt_length=32, max_response_length=4,
                                  max_obs_length=4, num_gpus=num_gpus)
        manager = LLMGenerationManager(Tokenizer(), worker, config)
        # Deterministic search keeps all trajectories active into final rollout.
        manager._postprocess_responses = lambda responses: (
            responses, ['<search>France</search>'] * responses.shape[0])
        manager.batch_search = lambda queries: ['Paris'] * len(queries)
        return manager, worker

    def batch(self, size, meta):
        ids = torch.ones((size, 2), dtype=torch.long)
        return DataProto.from_dict({'input_ids': ids, 'attention_mask': ids.clone(),
                                    'position_ids': torch.tensor([[0, 1]] * size)},
                                   meta_info=meta.copy())

    def test_entire_loop_preserves_validation_and_training_flags(self):
        for num_gpus in (1, 2):
            for do_sample in (False, True):
                with self.subTest(num_gpus=num_gpus, do_sample=do_sample):
                    manager, worker = self.manager(num_gpus)
                    expected = dict(do_sample=do_sample, validate=not do_sample,
                                    recompute_log_prob=do_sample, sentinel='preserved')
                    batch = self.batch(3, expected)
                    manager.run_llm_loop(batch, batch.batch['input_ids'])
                    self.assertEqual(len(worker.calls), 2)
                    for size, meta in worker.calls:
                        self.assertEqual(size, 3 if num_gpus == 1 else 4)
                        self.assertEqual({key: meta[key] for key in expected}, expected)
                    self.assertNotIn('worker_bookkeeping', batch.meta_info)

    def test_padding_and_exactly_divisible_batches(self):
        for size in (2, 3):
            with self.subTest(size=size):
                manager, worker = self.manager(2)
                batch = self.batch(size, {'do_sample': False, 'validate': True})
                output = manager._generate_with_gpu_padding(batch)
                self.assertEqual(worker.calls[0][0], 2 if size == 2 else 4)
                self.assertFalse(worker.calls[0][1]['do_sample'])
                self.assertTrue(worker.calls[0][1]['validate'])
                self.assertEqual(output.batch['responses'].shape[0], size)
                if size == 3:
                    self.assertNotIn('worker_bookkeeping', batch.meta_info)

    def test_paper_action_budget_has_no_extra_fifth_generation(self):
        manager, worker = self.manager(2)
        manager.config.max_turns = 4
        manager.config.force_final_answer = False
        batch = self.batch(3, {'do_sample': True})
        output = manager.run_llm_loop(batch, batch.batch['input_ids'])
        self.assertEqual(len(worker.calls), 4)
        self.assertEqual(output.meta_info['active_mask'], [True, True, True])
        self.assertEqual(output.meta_info['valid_search_stats'], [4, 4, 4])
        self.assertEqual(output.meta_info['turns_stats'], [4, 4, 4])

    def test_legacy_default_retains_extra_final_generation(self):
        manager, worker = self.manager(2)
        manager.config.max_turns = 4
        self.assertTrue(manager.config.force_final_answer)
        batch = self.batch(3, {'do_sample': True})
        output = manager.run_llm_loop(batch, batch.batch['input_ids'])
        self.assertEqual(len(worker.calls), 5)
        self.assertEqual(output.meta_info['turns_stats'], [5, 5, 5])

    def test_nine_validation_samples_keep_greedy_flags_in_eight_plus_one_tail(self):
        manager, worker = self.manager(8)
        manager.config.force_final_answer = False
        batch = self.batch(9, {'do_sample': False, 'validate': True,
                               'recompute_log_prob': False})
        sizes = []
        for start in (0, 8):
            part = DataProto.from_dict({key: value[start:start + 8] for key, value in batch.batch.items()},
                                      meta_info=batch.meta_info.copy())
            output = manager.run_llm_loop(part, part.batch['input_ids'])
            sizes.append(output.batch['responses'].shape[0])
        self.assertEqual(sizes, [8, 1])
        self.assertEqual([size for size, meta in worker.calls], [8, 8])
        for size, meta in worker.calls:
            self.assertFalse(meta['do_sample'])
            self.assertTrue(meta['validate'])
            self.assertFalse(meta['recompute_log_prob'])


if __name__ == '__main__':
    print('REAL_DATAPROTO=' + str(REAL_DATAPROTO))
    unittest.main()
