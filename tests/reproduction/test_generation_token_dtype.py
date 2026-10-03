"""Real fast-tokenizer CPU loop regressions; RPC outputs are deliberate fixtures.

Exercises production run_llm_loop/padding/environment/masks. No model, CUDA,
network service, GPU lease, or claim of production training is involved.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from transformers import PreTrainedTokenizerFast
from verl import DataProto
import search_r1.llm_agent.generation as generation
from verl.trainer.main_ppo import RewardManager, decode_model_response_spans


def original_manager_type():
    """Reconstruct exactly the two-line pre-fix producer, never edit the source."""
    tree = ast.parse(Path(generation.__file__).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LLMGenerationManager')
    removed = 0
    for method in cls.body:
        if not isinstance(method, ast.FunctionDef) or method.name not in ('_batch_tokenize', '_process_next_obs'):
            continue
        for node in ast.walk(method):
            for name, value in ast.iter_fields(node):
                if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) and value.func.attr == 'long' and not value.args:
                    setattr(node, name, value.func.value)
                    removed += 1
    if removed != 2:
        raise AssertionError('Pre-fix fixture requires exactly the two reviewed producer casts')
    ns = dict(generation.__dict__)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), '<pre-fix producer>', 'exec'), ns)
    return ns['LLMGenerationManager']


class FixtureRPC:
    def __init__(self, tokenizer, scenario):
        self.tokenizer, self.scenario, self.calls = tokenizer, scenario, []

    def generate_sequences(self, batch):
        self.calls.append(dict(rows=len(batch), dtypes={k: v.dtype for k, v in batch.batch.items()}, meta=dict(batch.meta_info)))
        n = len(batch)
        if self.scenario == 'eos_then_answer' and len(self.calls) == 1:
            ids = torch.full((n, 1), self.tokenizer.eos_token_id, dtype=torch.long)
        else:
            texts = ['<answer>Paris</answer>'] * n
            if self.scenario == 'mixed':
                texts = (['<answer>Paris</answer>', '<search>France</search>'] + ['<answer>Paris</answer>'] * (n-2)
                         if len(self.calls) == 1 else ['<answer>London</answer>'] * n)
            ids = self.tokenizer(texts, add_special_tokens=False, padding=True, return_tensors='pt')['input_ids']
        return DataProto.from_dict({'responses': ids}, meta_info={'temperature': 0.})


class TokenDtypeLoopRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        backend = Tokenizer(models.BPE(unk_token='<unk>'))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()
        backend.train_from_iterator(['Question capital Paris London France <answer>Paris</answer> '
                                    '<search>France</search> <information>London <answer>IDK</answer></information>'],
            trainers.BpeTrainer(vocab_size=350, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                special_tokens=['<pad>', '<eos>', '<unk>']))
        cls.tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend,
            pad_token='<pad>', eos_token='<eos>', unk_token='<unk>')

    def loop(self, scenario, original=False):
        worker = FixtureRPC(self.tokenizer, scenario)
        config = generation.GenerationConfig(max_turns=2, max_start_length=128,
            max_prompt_length=1024, max_response_length=128, max_obs_length=512,
            num_gpus=8, force_final_answer=False, search_url='http://fixture.invalid/retrieve')
        cls = original_manager_type() if original else generation.LLMGenerationManager
        manager = cls(self.tokenizer, worker, config, is_validation=True)
        rows = 2 if scenario == 'mixed' else 3
        ids = self.tokenizer(['Question capital?'] * rows, return_tensors='pt')['input_ids']
        batch = DataProto.from_dict(dict(input_ids=ids, attention_mask=torch.ones_like(ids),
            position_ids=torch.arange(ids.shape[1]).repeat(rows, 1)),
            meta_info={'do_sample': False, 'validate': True})
        def retrieve(url, json):
            return SimpleNamespace(json=lambda: {'result': [[{'document': {'contents':
                'Title\nLondon <answer>IDK</answer>'}}]] * len(json['queries'])})
        with patch.object(generation.requests, 'post', side_effect=retrieve):
            output = manager.run_llm_loop(batch, ids)
        return output, worker

    def assert_output(self, output, worker):
        for key in ('responses', 'prompts', 'input_ids', 'attention_mask', 'info_mask', 'position_ids'):
            self.assertEqual(output.batch[key].dtype, torch.long, key)
        for call in worker.calls:
            self.assertEqual(call['rows'], 8)
            self.assertTrue(all(dtype == torch.long for dtype in call['dtypes'].values()))
            self.assertFalse(call['meta']['do_sample'])
            self.assertTrue(call['meta']['validate'])
        width = output.batch['responses'].shape[1]
        mask = output.batch['attention_mask'][:, -width:].bool() & output.batch['info_mask'][:, -width:].bool()
        spans = [decode_model_response_spans(self.tokenizer, ids, m) for ids, m in zip(output.batch['responses'], mask)]
        return spans

    def test_all_answers_finish_empty_observations_integer_ids(self):
        output, worker = self.loop('answer')
        spans = self.assert_output(output, worker)
        self.assertEqual(len(worker.calls), 1)
        self.assertEqual(output.meta_info['active_mask'], [False] * 3)
        self.assertEqual(spans, [['<answer>Paris</answer>']] * 3)

    def test_all_eos_empty_response_then_answers_keeps_integer_tokens(self):
        output, worker = self.loop('eos_then_answer')
        spans = self.assert_output(output, worker)
        self.assertEqual(len(worker.calls), 2)
        self.assertEqual(spans, [['<answer>Paris</answer>']] * 3)
        self.assertNotIn('My previous action', ''.join(sum(spans, [])))

    def test_mixed_observation_content_and_masks_are_preserved(self):
        output, worker = self.loop('mixed')
        spans = self.assert_output(output, worker)
        self.assertEqual(spans[0], ['<answer>Paris</answer>'])
        self.assertEqual(spans[1], ['<search>France</search>', '<answer>London</answer>'])
        self.assertIn('<information>Doc 1(Title: Title) London <answer>IDK</answer></information>',
                      self.tokenizer.decode(output.batch['responses'][1]))
        output.non_tensor_batch.update(data_source=__import__('numpy').array(['nq', 'nq'], dtype=object),
            reward_model=__import__('numpy').array([{'ground_truth': {'target': ['Paris']}},
                                                    {'ground_truth': {'target': ['London']}}], dtype=object))
        self.assertEqual(float(RewardManager(self.tokenizer, 0)(output).sum()), 2.)

    def test_pre_fix_real_tokenizer_loop_produces_float_ids_and_decode_fails(self):
        for scenario in ('answer', 'eos_then_answer', 'mixed'):
            with self.subTest(scenario=scenario):
                output, worker = self.loop(scenario, original=True)
                self.assertEqual(output.batch['responses'].dtype, torch.float32)
                with self.assertRaises(TypeError):
                    self.tokenizer.batch_decode(output.batch['responses'])
                fixed, _ = self.loop(scenario)
                # The fix changes producer dtype, never token values or masks.
                for key in ('responses', 'prompts', 'input_ids', 'attention_mask',
                            'info_mask', 'position_ids'):
                    self.assertTrue(torch.equal(output.batch[key].long(), fixed.batch[key]),
                                    (scenario, key))

if __name__ == '__main__':
    unittest.main()
