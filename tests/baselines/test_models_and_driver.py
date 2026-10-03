"""Offline real tiny Qwen model updates and real Ray trainer control-flow tests.

No downloaded model, CUDA, Ray cluster, external search, or production data used.
The distributed GPU kernels remain a separate acceptance gate.
"""
import copy
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from transformers import Qwen2Config, Qwen2ForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests/baselines'))
from test_algorithms import CharTokenizer, Worker, padded
from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.main_ppo import RewardManager
from search_r1.baselines.controller import Controller, validate_config, canonical_paths
from search_r1.baselines.bapo import BoundaryStage, BoundaryReward
from search_r1.baselines.distillation import FrozenTeacher, opsd_loss
from search_r1.baselines.cosearch import GenerativeRanker


def config(method='gigpo'):
    with initialize_config_dir(config_dir=str(ROOT / 'verl/trainer/config'), version_base=None):
        c = compose(config_name='ppo_baselines', overrides=['baselines.method=' + method])
    OmegaConf.resolve(c)
    # Tiny control-flow fixtures use one question; production count is tested separately.
    c.baselines.bapo.stage_validation_size = None
    return c


class TinyModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.tmp = tempfile.TemporaryDirectory()
        tokenizer = Tokenizer(models.BPE(unk_token='<unk>'))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.train_from_iterator(['Question Paris Rome Berlin capital teacher accepted answers '
                                      '<ranking>[1,2,3]</ranking> <answer>Paris</answer>\n'],
                                      trainers.BpeTrainer(vocab_size=300,
                                          initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                          special_tokens=['<pad>', '<eos>', '<unk>']))
        cls.tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer,
                                               pad_token='<pad>', eos_token='<eos>', unk_token='<unk>')
        cls.tokenizer.save_pretrained(cls.tmp.name)
        model = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(cls.tokenizer), hidden_size=16,
                                intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                                num_key_value_heads=2, max_position_embeddings=1024,
                                pad_token_id=cls.tokenizer.pad_token_id, eos_token_id=cls.tokenizer.eos_token_id,
                                attention_dropout=0.))
        model.save_pretrained(cls.tmp.name)
    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()
    def test_real_teacher_logprob_alignment_and_student_only_opsd_update(self):
        teacher = FrozenTeacher(self.tmp.name, self.tokenizer)
        student = Qwen2ForCausalLM.from_pretrained(self.tmp.name)
        student.eval()
        prompt = torch.tensor([self.tokenizer.encode('Question capital?')])
        response = torch.tensor([self.tokenizer.encode('<answer>Paris</answer>')])
        ids = torch.cat((prompt, response), 1)
        attention = torch.ones_like(ids)
        response_attention = torch.ones_like(response)
        actual = teacher.log_probs(prompt, response, response_attention, chunk=3)
        with torch.no_grad():
            logits = student(ids).logits[:, -response.shape[1] - 1:-1]
            expected = logits.log_softmax(-1).gather(-1, response[..., None]).squeeze(-1)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        tensors = dict(prompts=prompt, responses=response, input_ids=ids, attention_mask=attention,
                       position_ids=torch.arange(ids.shape[1])[None, :], loss_mask=torch.ones_like(response))
        settings = SimpleNamespace(chunk_size=3, beta=0., temperature=1., component_clip=.05)
        before = student.lm_head.weight.detach().clone()
        optimizer = torch.optim.AdamW(student.parameters(), lr=.001)
        loss = opsd_loss(student, teacher, tensors, self.tokenizer, [['Paris']], settings, response.numel())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(p.grad is None for p in teacher.model.parameters()))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in student.parameters() if p.grad is not None), 0)
        optimizer.step()
        self.assertFalse(torch.equal(before, student.lm_head.weight))
    def test_real_ranker_parameters_update_and_invalid_main_filter(self):
        c = config('cosearch').baselines.ranker
        c.model_path = self.tmp.name
        c.min_group_size = 3
        ranker = GenerativeRanker(c)
        ranker.question_uids = ['q'] * 3
        docs = [{'document': {'contents': d}} for d in ['Paris', 'Rome', 'Berlin']]
        manager = SimpleNamespace(tokenizer=CharTokenizer())
        # Deterministic valid actions still use actual HF forward/scoring/optimizer.
        def generate(input_ids, **kwargs):
            self.assertEqual(kwargs['generation_config'].repetition_penalty, 1.)
            self.assertEqual(kwargs['generation_config'].no_repeat_ngram_size, 0)
            response = torch.tensor([ranker.tokenizer.encode('<ranking>[1,2,3]</ranking>')])
            return torch.cat((input_ids, response), 1)
        ranker.model.generation_config.repetition_penalty = 1.05
        with patch.object(ranker.model, 'generate', side_effect=generate):
            for i in range(3):
                ranker.rank('Question capital?', 'capital', docs, SimpleNamespace(source=i, trajectory=str(i)))
        before = ranker.model.lm_head.weight.detach().clone()
        rewards = {str(i): (float(i == 0), ['Paris'], True) for i in range(3)}
        metrics = ranker.update(rewards)
        self.assertEqual(metrics['ranker/training_events'], 3)
        self.assertEqual(ranker.cost['optimizer_steps'], 1)
        self.assertFalse(torch.equal(before, ranker.model.lm_head.weight))
        with patch.object(ranker.model, 'generate', side_effect=generate):
            ranker.rank('Question capital?', 'capital', docs, SimpleNamespace(source=0, trajectory='bad'))
        ranker.update({'bad': (1., ['Paris'], False)})
        self.assertEqual(ranker.cost['optimizer_steps'], 1)
        saved = os.path.join(self.tmp.name, 'ranker-checkpoint')
        ranker.save(saved)
        self.assertTrue(Path(saved, 'optimizer.pt').is_file())


class Driver(unittest.TestCase):
    def test_configs_preserve_actor_data_and_reject_test_holdout_alias(self):
        for method in ('gigpo', 'tree_grpo', 'bapo', 'opsd', 'topd', 'cosearch'):
            c = config(method)
            self.assertEqual(c.actor_rollout_ref.model.path, '/data/search-r1/assets/qwen2.5-3b')
            self.assertEqual(c.max_turns, 4)
            self.assertEqual(c.retriever.topk, 3)
        c = config('bapo')
        c.baselines.bapo.stage_validation_files = [c.data.val_files]
        with self.assertRaises(ValueError):
            validate_config(c)
        c = config('topd')
        c.actor_rollout_ref.actor.use_kl_loss = False
        c.actor_rollout_ref.actor.entropy_coeff = 0.
        c.algorithm.kl_ctrl.kl_coef = 0.
        c.baselines.distillation.teacher_path = c.actor_rollout_ref.model.path + '/.'
        with self.assertRaises(ValueError):
            validate_config(c)
    def test_real_fit_driver_gigpo_tree_bapo_topd_opsd(self):
        # Actual fit(), with only GPU transport/model replaced. Checks unions,
        # repeated source rows, scoring metadata, hooks, balancing and final tail.
        for method in ('gigpo', 'tree_grpo', 'bapo', 'topd', 'opsd', 'cosearch'):
            with self.subTest(method=method):
                c = config(method)
                c.actor_rollout_ref.rollout.n_agent = 2
                c.baselines.tree.roots = 2
                c.trainer.total_epochs = 1
                c.trainer.test_freq = -1
                c.trainer.save_freq = -1
                c.trainer.critic_warmup = 0
                c.actor_rollout_ref.actor.use_kl_loss = True
                t = RayPPOTrainer.__new__(RayPPOTrainer)
                t.config, t.tokenizer = c, CharTokenizer()
                t.logger = SimpleNamespace(log=lambda **kwargs: None)
                worker = Worker(t.tokenizer)
                worker.world_size = 1
                updates = []
                def update_actor(data):
                    self.assertIsInstance(data, DataProto)
                    self.assertEqual(data.batch['advantages'].dtype, torch.float32)
                    self.assertEqual(data.meta_info['temperature'], 1.)
                    self.assertEqual(data.batch['loss_mask'].shape, data.batch['responses'].shape)
                    updates.append(data)
                    return DataProto(meta_info={'metrics': {'actor/fake_gpu_update': 1., 'actor/optimizer_steps': 1.}})
                worker.update_actor = update_actor
                t.actor_rollout_wg = worker
                ids = padded([t.tokenizer.encode('Question: capital?')], 0, True)
                t.train_dataloader = [dict(input_ids=ids, attention_mask=(ids != 0).long(),
                    position_ids=torch.arange(ids.shape[1])[None, :], index=np.array(['q'], dtype=object),
                    reward_model=np.array([{'ground_truth': {'target': ['Paris']}}], dtype=object),
                    data_source=np.array(['nq'], dtype=object))]
                t.use_reference_policy = t.use_rm = t.use_critic = False
                t.total_training_steps = 2
                t.val_reward_fn = None
                t.reward_fn = RewardManager(t.tokenizer, 0)
                t.kl_ctrl = SimpleNamespace(value=0.)
                controller = Controller.__new__(Controller)
                controller.trainer, controller.config, controller.method = t, c.baselines, method
                controller.stage = BoundaryStage()
                controller.ranker = controller.teacher = None
                if method == 'bapo':
                    t.val_dataloader = ['final']
                    t.val_dataset = 'final'
                    controller.stage_loader = ['holdout']
                    controller.stage_dataset = 'holdout'
                    t._validate = lambda: {'val/test_score/nq': 0.2, 'val/evaluated_count/nq': 1}
                    t.reward_fn = BoundaryReward(t.tokenizer, controller.stage)
                elif method == 'topd':
                    class Scorer:
                        parameter_count = 1
                        scored_tokens = 0
                        def log_probs(self, prompts, responses, attention, *args):
                            return torch.ones_like(responses, dtype=torch.float32)
                    controller.teacher = Scorer()
                elif method == 'cosearch':
                    class RankerTransport:
                        validation = False
                        events = []
                        def execute(self, manager, predictions, states, do_search):
                            return manager.execute_predictions(predictions, manager.tokenizer.pad_token,
                                        torch.ones(len(states), dtype=torch.bool), do_search)
                        def update(self, rewards):
                            assert len(rewards) == 2
                            return {'ranker/transport_update': 1}
                    controller.ranker = RankerTransport()
                    t.reward_fn = controller.main_f1_reward
                t.baseline_controller = controller
                with patch('search_r1.llm_agent.generation.LLMGenerationManager.batch_search',
                           lambda self, queries: ['Paris'] * len(queries)):
                    t.fit()
                self.assertEqual(len(updates), 1)
                if method == 'bapo':
                    self.assertEqual(controller.stage.best, 0.2)
                    self.assertEqual(t.val_dataloader, ['final'])
                    self.assertIsNone(t.val_reward_fn)
                self.assertEqual(len(updates[0]), 6 if method == 'tree_grpo' else 2)


if __name__ == '__main__':
    unittest.main()
