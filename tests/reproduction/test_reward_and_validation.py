"""CPU reward provenance and complete validation denominator regressions."""
import ast
import os
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
try:
    from verl import DataProto
    from verl.trainer.main_ppo import RewardManager
except ImportError:
    if os.environ.get('SEARCH_R1_REQUIRE_REAL_DATAPROTO') == '1':
        raise

    class DataProto:
        def __init__(self, batch, non_tensors):
            self.batch = batch
            self.non_tensor_batch = non_tensors

        @classmethod
        def from_dict(cls, tensors, non_tensors=None):
            return cls(tensors, non_tensors or {})

        def __len__(self):
            return self.batch['responses'].shape[0]

        def __getitem__(self, index):
            return SimpleNamespace(batch={k: v[index] for k, v in self.batch.items()},
                                   non_tensor_batch={k: v[index] for k, v in self.non_tensor_batch.items()})

    source = ast.parse((ROOT / 'verl/trainer/main_ppo.py').read_text())
    selected = [node for node in source.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                and node.name in ('_select_rm_score_fn', 'decode_model_response_spans', 'RewardManager')]
    qa = SimpleNamespace(**runpy.run_path(str(ROOT / 'verl/utils/reward_score/qa_em.py')))
    namespace = dict(torch=torch, DataProto=DataProto, qa_em=qa)
    exec(compile(ast.Module(body=selected, type_ignores=[]), 'main_ppo.py', 'exec'), namespace)
    RewardManager = namespace['RewardManager']


class CharTokenizer:
    def decode(self, ids):
        return ''.join(chr(value) for value in ids.tolist())


class RewardProvenance(unittest.TestCase):
    def reward(self, segments, answers=None, include_info_mask=True):
        prompt = 'Example <answer>Paris</answer>. Question: capital?'
        response = ''.join(text for text, model in segments)
        prompt_ids = torch.tensor([[ord(char) for char in prompt]], dtype=torch.long)
        response_ids = torch.tensor([[ord(char) for char in response]], dtype=torch.long)
        attention = torch.ones((1, len(prompt) + len(response)), dtype=torch.long)
        tensors = dict(prompts=prompt_ids, responses=response_ids, attention_mask=attention)
        if include_info_mask:
            mask = [1] * len(prompt)
            for text, model in segments:
                mask.extend([int(model)] * len(text))
            tensors['info_mask'] = torch.tensor([mask], dtype=torch.long)
        data = DataProto.from_dict(tensors, non_tensors={
            'data_source': np.array(['nq'], dtype=object),
            'reward_model': np.array([{'ground_truth': {'target': answers or ['Paris']}}], dtype=object),
        })
        result = RewardManager(CharTokenizer(), num_examine=0)(data)
        self.assertEqual(result.shape, response_ids.shape)
        if response:
            self.assertEqual(int(torch.count_nonzero(result[:, :-1])), 0)
        return float(result.sum())

    def test_prompt_and_observation_answers_are_not_outcomes(self):
        self.assertEqual(self.reward([('<search>France</search>', True)]), 0)
        self.assertEqual(self.reward([('<search>France</search>', True),
                                      ('<information><answer>Paris</answer></information>', False),
                                      ('<search>more</search>', True)]), 0)

    def test_complete_generated_answer_and_alias_normalization(self):
        self.assertEqual(self.reward([('<search>France</search>', True),
                                      ('<information>Paris</information>', False),
                                      ('<answer>The PARIS!</answer>', True)], ['Lutetia', 'Paris']), 1)
        self.assertEqual(self.reward([('<answer>Paris</answer>', True)], include_info_mask=False), 1)

    def test_answer_cannot_be_stitched_across_observation(self):
        self.assertEqual(self.reward([('<answer>Pa', True), ('external', False), ('ris</answer>', True)]), 0)
        self.assertEqual(self.reward([('<ans', True), ('external', False), ('wer>Paris</answer>', True)]), 0)

    def test_last_complete_model_answer_and_incomplete_answer(self):
        self.assertEqual(self.reward([('<answer>Paris</answer><answer>London</answer>', True)]), 0)
        self.assertEqual(self.reward([('<answer>Paris', True)]), 0)
        self.assertEqual(self.reward([]), 0)

    def test_legacy_standalone_api_is_preserved(self):
        qa = runpy.run_path(str(ROOT / 'verl/utils/reward_score/qa_em.py'))
        self.assertEqual(qa['compute_score_em']('<answer>Paris</answer>', {'target': ['Paris']}), 0)
        self.assertEqual(qa['compute_score_generated_em'](['<answer>Paris</answer>'], {'target': ['Paris']}), 1)


class ValidationCoverage(unittest.TestCase):
    def methods(self):
        tree = ast.parse((ROOT / 'verl/trainer/ppo/ray_trainer.py').read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'RayPPOTrainer')
        return {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}

    def test_actual_validation_loader_call_keeps_entire_tail(self):
        method = self.methods()['_create_dataloader']
        assignment = next(node for node in method.body if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Attribute) and target.attr == 'val_dataloader'
                                  for target in node.targets))
        instance = SimpleNamespace(val_dataset=list(range(3610)),
                                   config=SimpleNamespace(data=SimpleNamespace(val_batch_size=256)))
        loader = eval(compile(ast.Expression(assignment.value), 'ray_trainer.py', 'eval'),
                      dict(DataLoader=DataLoader, self=instance, collate_fn=lambda items: items))
        evaluated = [index for batch in loader for index in batch]
        self.assertEqual(evaluated, list(range(3610)))
        self.assertEqual(len(list(loader)[-1]), 26)

    def test_actual_metric_aggregation_reports_per_dataset_count(self):
        method = self.methods()['_validate']
        start = next(index for index, node in enumerate(method.body)
                     if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                        and target.id == 'data_source_reward' for target in node.targets))
        function = ast.FunctionDef(name='summarize',
            args=ast.arguments(posonlyargs=[], args=[ast.arg(arg='reward_tensor'), ast.arg(arg='data_sources')],
                               kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=method.body[start:], decorator_list=[])
        module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
        namespace = dict(np=np, validation_manager_class=None,
                         LLMGenerationManager=None)
        exec(compile(module, 'ray_trainer.py', 'exec'), namespace)
        metrics = namespace['summarize'](torch.tensor([1., 0., 1., 1., 0.]),
                                         np.array(['nq', 'nq', 'hotpotqa', 'hotpotqa', 'hotpotqa']))
        self.assertEqual(metrics['val/evaluated_count/nq'], 2)
        self.assertEqual(metrics['val/evaluated_count/hotpotqa'], 3)
        self.assertEqual(metrics['val/test_score/nq'], .5)
        self.assertAlmostEqual(metrics['val/test_score/hotpotqa'], 2 / 3)


if __name__ == '__main__':
    unittest.main()
