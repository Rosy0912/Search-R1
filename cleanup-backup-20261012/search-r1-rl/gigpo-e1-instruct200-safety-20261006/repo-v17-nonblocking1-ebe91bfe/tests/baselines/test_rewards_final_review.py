"""Independent regression checks of final integration, not implementation mirrors."""
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
import torch
from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from search_r1.baselines.bapo import BoundaryStage, BoundaryReward
from search_r1.baselines.controller import Controller, validate_config
from search_r1.baselines.holdout import split_frames
from test_models_and_driver import config
from test_algorithms import CharTokenizer, Worker, padded


@pytest.mark.parametrize('missing', ['column', 'index', 'null'])
def test_holdout_rejects_missing_question_identity(missing):
    train = pd.DataFrame([dict(prompt=[dict(content=f'q{i}')], extra_info={'index': i}) for i in range(100)])
    if missing == 'column':
        train = train.drop(columns=['extra_info'])
    else:
        train.at[5, 'extra_info'] = {} if missing == 'index' else {'index': None}
    final = pd.DataFrame([dict(prompt=[dict(content='unseen final')])])
    with pytest.raises(ValueError, match='index'):
        split_frames(train, final)


@pytest.mark.parametrize('warmup,actor_steps', [(10, 1), (0, 0), (0, 1)])
def test_real_fit_stage_only_observes_updated_actor(warmup, actor_steps):
    c = config('bapo')
    c.actor_rollout_ref.rollout.n_agent = 2
    c.trainer.total_epochs = 1
    c.trainer.test_freq = c.trainer.save_freq = -1
    c.trainer.critic_warmup = warmup
    c.baselines.bapo.stage_validation_freq = 1
    t = RayPPOTrainer.__new__(RayPPOTrainer)
    t.config, t.tokenizer = c, CharTokenizer()
    t.logger = SimpleNamespace(log=lambda **kwargs: None)
    worker = Worker(t.tokenizer)
    worker.world_size = 1
    updates = []
    def update_actor(data):
        updates.append(data)
        return DataProto(meta_info={'metrics': {'actor/optimizer_steps': float(actor_steps)}})
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
    t.kl_ctrl = SimpleNamespace(value=0.)
    ctrl = Controller.__new__(Controller)
    ctrl.trainer, ctrl.config, ctrl.method = t, c.baselines, 'bapo'
    ctrl.stage = BoundaryStage()
    # A stalled model would switch after the next observation: skipped iterations must not switch it.
    ctrl.stage.best = .2
    ctrl.stage.stalled = 4
    ctrl.ranker = ctrl.teacher = None
    ctrl.stage_loader = ['holdout']; ctrl.stage_dataset = 'holdout'
    t.val_dataloader = ['final']; t.val_dataset = 'final'
    validations = []
    def validate():
        validations.append(1)
        assert t.val_dataloader == ['holdout']
        return {'val/test_score/nq': .2, 'val/evaluated_count/nq': 1}
    t._validate = validate
    t.reward_fn = BoundaryReward(t.tokenizer, ctrl.stage)
    t.baseline_controller = ctrl
    with patch('search_r1.llm_agent.generation.LLMGenerationManager.batch_search',
               lambda self, queries: ['Paris'] * len(queries)):
        t.fit()
    expected_update = warmup <= 1
    expected_observation = expected_update and actor_steps > 0
    assert len(updates) == int(expected_update)
    assert len(validations) == int(expected_observation)
    assert ctrl.stage.stalled == 4 + int(expected_observation)
    assert ctrl.stage.early == (not expected_observation)
    assert t.val_dataloader == ['final'] and t.val_reward_fn is None


@pytest.mark.parametrize('field,value', [('batch_size', 0), ('epochs', 0), ('max_new_tokens', 0),
                                       ('clip_ratio', 0), ('grad_clip', 0), ('similarity', 1.1)])
def test_ranker_invalid_overrides_fail_before_model_load(field, value):
    c = config('cosearch')
    c.baselines.ranker.model_path = '/unused/checkpoint'
    c.baselines.ranker[field] = value
    with pytest.raises(ValueError, match='ranker'):
        validate_config(c)
