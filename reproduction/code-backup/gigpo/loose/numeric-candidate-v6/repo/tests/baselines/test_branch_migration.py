"""Independent migration checks of retained search-r1 terminal evaluation hooks.

Actual trainer terminal control flow and final DataLoader/RLHFDataset are executed.
GPU transport, actor updates and model-generated answers are outside this CPU gate.
"""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import tempfile

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tests/reproduction'))
sys.path.insert(0, str(ROOT / 'tests/baselines'))
from test_final_checkpoint import real_iteration_tail, FakeTrainer
from test_models_and_driver import TinyModels, config
from search_r1.baselines.bapo import BoundaryStage
from search_r1.baselines.controller import Controller
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, _baseline_validation


@pytest.mark.parametrize('method', ['gigpo', 'tree_grpo', 'bapo', 'opsd', 'topd', 'cosearch'])
def test_terminal_checkpoint_periodic_stage_and_final_data_order(method):
    t = FakeTrainer(5, 6, test_freq=1)
    t.tokenizer = SimpleNamespace()
    t.val_dataloader, t.val_dataset = 'periodic64', 'periodic64'
    original_reward = t.val_reward_fn
    controller = Controller.__new__(Controller)
    controller.trainer = t
    controller.method = method
    controller.config = config(method).baselines
    controller.stage = BoundaryStage()
    controller.stage_loader, controller.stage_dataset = 'stage', 'stage'
    controller.after_actor = lambda *args: t.events.append(('after_actor', t.global_steps))
    t.baseline_controller = controller
    t.update = lambda batch: SimpleNamespace(meta_info={'metrics': {'actor/optimizer_steps': 1}})
    t.actor_rollout_wg.update_actor = t.update
    def validate():
        t.events.append(('validation_data', t.val_dataset, t.global_steps))
        return {'val/test_score/nq': .5, 'val/evaluated_count/nq': 1}
    def final_data():
        t.val_dataloader = t.val_dataset = 'final7'
        t.events.append(('final_data', t.global_steps))
    t._validate, t._prepare_final_validation = validate, final_data
    real_iteration_tail()(t)
    selected = [e[1] for e in t.events if e[0] == 'validation_data']
    assert selected == (['periodic64', 'stage', 'final7'] if method == 'bapo'
                         else ['periodic64', 'final7'])
    assert t.val_reward_fn is original_reward
    assert t.val_dataset == 'final7'
    assert t.events.index(('save', 5, 0)) < t.events.index(('final_data', 6))
    if method == 'bapo':
        assert controller.stage.best == .5


def test_actual_final_rlhf_loader_preserves_all_seven_sources_and_tail():
    TinyModels.setUpClass()
    try:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'final.parquet'
            rows = [dict(prompt=[dict(role='user', content='Question capital?')],
                         data_source='dataset' + str(i % 7),
                         extra_info={'index': i},
                         reward_model={'ground_truth': {'target': ['Paris']}})
                    for i in range(65)]
            pd.DataFrame(rows).to_parquet(path, index=False)
            t = RayPPOTrainer.__new__(RayPPOTrainer)
            t.config = config()
            t.config.data.final_val_files = str(path)
            t.config.data.val_batch_size = 64
            t.config.data.max_prompt_length = 32
            t.config.data.val_data_num = 1  # final helper must ignore periodic subsampling.
            t.tokenizer = TinyModels.tokenizer
            RayPPOTrainer._prepare_final_validation(t)
            batches = list(t.val_dataloader)
            assert [len(batch['index']) for batch in batches] == [64, 1]
            assert set(int(i) for batch in batches for i in batch['index']) == set(range(65))
            assert set(x for batch in batches for x in batch['data_source']) == {'dataset' + str(i) for i in range(7)}
    finally:
        TinyModels.tearDownClass()


@pytest.mark.parametrize('initial', [False, True])
def test_cosearch_validation_flag_restored_on_terminal_evaluation_failure(initial):
    ranker = SimpleNamespace(validation=initial, events=[])
    t = SimpleNamespace(baseline_controller=SimpleNamespace(ranker=ranker))
    @_baseline_validation
    def failing(self):
        assert ranker.validation is True
        raise RuntimeError('test evaluation failed')
    with pytest.raises(RuntimeError, match='test evaluation failed'):
        failing(t)
    assert ranker.validation is initial
    assert ranker.events == []
