"""Runtime audit exercises the actual TOP-D actor, not a copied loss formula."""
import json
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from verl import DataProto
from verl.workers.actor.dp_actor import DataParallelPPOActor
from search_r1.baselines.objectives import attach_policy_weights
from search_r1.baselines.topd_runtime_audit import (teacher_boundary, require_actual_update,
                                                  shard_sample, tensor_hash, ActorAudit)


def run_actor(directory, rows=20, empty=False):
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    actor = DataParallelPPOActor.__new__(DataParallelPPOActor)
    actor.actor_module = model
    actor.actor_optimizer = torch.optim.SGD(model.parameters(), lr=.001)
    actor.config = OmegaConf.create(dict(ppo_mini_batch_size=2, ppo_micro_batch_size=1, ppo_epochs=2,
        state_masking=True, use_kl_loss=False, use_dynamic_bsz=False, baseline_method='topd',
        entropy_coeff=0., clip_ratio=.2, grad_clip=100.))
    gradients = []
    model.weight.register_hook(lambda grad: gradients.append(grad.detach().clone()))
    def forward(self, micro_batch, temperature):
        x = self.actor_module(torch.ones(len(micro_batch), 1))
        return x * 0 + 3, x
    actor._forward_micro_batch = MethodType(forward, actor)
    data = DataProto.from_dict(dict(responses=torch.zeros(rows, 1).long(), input_ids=torch.ones(rows, 2).long(),
        attention_mask=torch.ones(rows, 2).long(), info_mask=torch.ones(rows, 2).long(),
        position_ids=torch.zeros(rows, 2).long(), loss_mask=torch.ones(rows, 1) * int(not empty),
        old_log_probs=torch.zeros(rows, 1), advantages=torch.ones(rows, 1)),
        non_tensors={'uid': np.asarray(['q'] * rows)}, meta_info={'temperature': 1., 'topd_outer_step': 5})
    attach_policy_weights(data, 'topd')
    with patch.dict('os.environ', {'SEARCH_R1_TOPD_AUDIT_DIR': str(directory) if directory else ''}), \
         patch.object(TensorDict, 'cuda', lambda self, *a, **k: self):
        metrics = actor.update_policy(data)
    return metrics, model.weight.detach().clone(), gradients


def test_actual_actor_twenty_steps_audit_is_read_only(tmp_path):
    off, off_weights, off_grads = run_actor(None)
    on, on_weights, on_grads = run_actor(tmp_path)
    assert on['actor/optimizer_steps'] == off['actor/optimizer_steps'] == 20
    torch.testing.assert_close(off_weights, on_weights, rtol=0, atol=0)
    assert len(off_grads) == len(on_grads) == 40
    for left, right in zip(off_grads, on_grads):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    record = json.loads((tmp_path / '0.jsonl').read_text())
    assert record['outer_step'] == 5 and record['rank'] == 0
    assert record['optimizer_steps'] == record['grad_finite'] == record['grad_nonzero'] == 20
    assert record['fixed_unchanged'] and len(record['fixed_epochs']) == 2
    assert record['shard_layout_unchanged'] and record['shard_delta']['nonzero'] > 0
    assert len(record['micro_stats']) == 40
    assert all(v['current_log_probs']['finite'] == 1 for v in record['micro_stats'])


@pytest.mark.parametrize('enabled', [False, True])
def test_empty_actual_actor_cannot_consume_outer_budget(tmp_path, enabled):
    metrics, weight, grads = run_actor(tmp_path if enabled else None, empty=True)
    assert metrics['actor/optimizer_steps'] == 0
    assert weight.item() == 0 and grads == []
    with pytest.raises(RuntimeError, match='zero actor optimizer steps'):
        require_actual_update(metrics)


def test_guard_is_independent_of_audit():
    with pytest.raises(RuntimeError):
        require_actual_update({'actor/optimizer_steps': 0})
    require_actual_update({'actor/optimizer_steps': 20})


def test_teacher_frozen_hash_and_bounded_shard_sample(tmp_path, monkeypatch):
    monkeypatch.setenv('SEARCH_R1_TOPD_AUDIT_DIR', str(tmp_path))
    model = torch.nn.Linear(3, 2).eval().requires_grad_(False)
    teacher = SimpleNamespace(model=model)
    start = teacher_boundary(teacher, 0, 'loaded')
    for _ in range(5):
        with torch.no_grad():
            model(torch.ones(1, 3))
    finish = teacher_boundary(teacher, 5, 'after_actor')
    assert start['parameter_sha256'] == finish['parameter_sha256']
    assert finish['requires_grad_parameters'] == finish['grad_parameters'] == 0
    assert not finish['training'] and finish['devices'] == ['cpu']
    big = torch.nn.Linear(100, 100, bias=False)
    layout, sample = shard_sample(big)
    assert sample.numel() == 8192
    assert layout[0]['indices'][0] == 0 and layout[0]['indices'][-1] == 9999
    # BF16 must hash without numpy BF16 support or expanding entire tensors to FP32.
    assert len(tensor_hash(torch.ones(10, dtype=torch.bfloat16))) == 64


def test_controller_scores_without_gold_and_binds_observation_sample(tmp_path, monkeypatch):
    from search_r1.baselines.controller import Controller
    monkeypatch.setenv('SEARCH_R1_TOPD_AUDIT_DIR', str(tmp_path))
    monkeypatch.setenv('SEARCH_R1_TOPD_DATA_IDENTITY', 'fixture-split-sha')
    controller = Controller.__new__(Controller)
    controller.method = 'topd'
    controller.config = OmegaConf.create(dict(distillation=dict(privileged_answers=False, chunk_size=2),
                                             topd=dict(alpha=.1)))
    controller.trainer = SimpleNamespace(config=OmegaConf.create(dict(actor_rollout_ref=dict(rollout=dict(temperature=1.)))))
    calls = []
    class Teacher:
        scored_tokens = 0
        parameter_count = 6
        def log_probs(self, prompts, responses, attention, temperature, targets, chunk_size):
            calls.append(targets)
            self.scored_tokens += int(attention.sum())
            return torch.full(responses.shape, -.3 * len(calls))
    controller.teacher = Teacher()
    data = DataProto.from_dict(dict(prompts=torch.ones(2, 1).long(), responses=torch.ones(2, 2).long(),
        attention_mask=torch.ones(2, 3).long(), info_mask=torch.tensor([[1, 0, 1], [1, 1, 1]]),
        old_log_probs=torch.zeros(2, 2), token_level_rewards=torch.zeros(2, 2)),
        non_tensors={'uid': np.asarray(['q', 'q']), 'reward_model': np.asarray([
            {'ground_truth': {'target': 'SECRET_ACCEPTED_ANSWER'}},
            {'ground_truth': {'target': 'SECRET_ACCEPTED_ANSWER'}}], dtype=object)},
        meta_info={'topd_outer_step': 3})
    controller.compute_advantages(data, {})
    text = (tmp_path / 'driver.jsonl').read_text()
    assert 'SECRET_ACCEPTED_ANSWER' not in text and calls == [None, None]
    record = json.loads(text)
    assert record['outer_step'] == 3 and record['teacher_targets_none']
    assert record['teacher_scored_tokens_delta'] == 4
    assert record['data_identity'] == 'fixture-split-sha'
    assert record['action_tokens'] == 3 and record['observation_tokens'] == 1
    assert record['sample']['info_mask'] == [0, 1]
    assert record['sample']['loss_mask'] == [0, 1]
    assert record['teacher_scoring_wall_seconds'] >= 0
    assert record['tensors']['advantages']['finite'] == 3


def load_real_driver_tail():
    # Reuse the meaningful CPU seam that compiles the real fit() tail, replacing
    # only worker RPCs/timers. It retains guard/save/counter ordering verbatim.
    import importlib.util
    from pathlib import Path
    source = Path(__file__).resolve().parents[1] / 'reproduction/test_final_checkpoint.py'
    spec = importlib.util.spec_from_file_location('topd_driver_cpu_seam', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.real_iteration_tail(), module.FakeTrainer


@pytest.mark.parametrize('warmup', [False, True])
def test_real_driver_zero_or_warmup_refuses_save_and_counter(warmup):
    run, factory = load_real_driver_tail()
    trainer = factory(5, 6, save_freq=1, test_freq=0)
    trainer.baseline_controller = SimpleNamespace(method='topd', after_actor=lambda *args: None)
    trainer.config.trainer.critic_warmup = 100 if warmup else 0
    trainer.actor_rollout_wg.update_actor = lambda batch: SimpleNamespace(meta_info={'metrics': {'actor/optimizer_steps': 0}})
    with pytest.raises(RuntimeError, match='zero actor optimizer steps'):
        run(trainer)
    assert trainer.global_steps == 5 and trainer.events == []


def test_real_driver_guard_keeps_other_algorithms_unchanged_and_accepts_actual_update():
    run, factory = load_real_driver_tail()
    for method, count in [('grpo', 0), ('topd', 20)]:
        trainer = factory(5, 100, save_freq=1, test_freq=0)
        trainer.baseline_controller = SimpleNamespace(method=method, after_actor=lambda *args: None,
                                                     stage_validation=lambda *args: {})
        trainer.actor_rollout_wg.update_actor = lambda batch: SimpleNamespace(meta_info={'metrics': {'actor/optimizer_steps': count}})
        run(trainer)
        assert trainer.global_steps == 6
        assert trainer.events == [('save', 5, 0)]


def test_rank_bound_log_detects_mutated_rollout_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv('SEARCH_R1_TOPD_AUDIT_DIR', str(tmp_path))
    tensors = TensorDict(dict(old_log_probs=torch.zeros(2, 1), advantages=torch.ones(2, 1),
                              policy_weights=torch.ones(2, 1), loss_mask=torch.ones(2, 1)), [2])
    with patch.object(torch.distributed, 'is_initialized', return_value=True), \
         patch.object(torch.distributed, 'get_rank', return_value=7):
        audit = ActorAudit(torch.nn.Linear(1, 1), tensors, {'topd_outer_step': 2}, 2)
    audit.epoch()
    tensors['advantages'][0, 0] = 3
    audit.epoch()
    audit.finish(20, 0)
    record = json.loads((tmp_path / '7.jsonl').read_text())
    assert record['rank'] == 7 and record['outer_step'] == 2
    assert not record['fixed_unchanged']
    tensor = torch.arange(12).reshape(3, 4).T
    assert tensor_hash(tensor) == tensor_hash(tensor.contiguous())


@pytest.mark.parametrize('count', [float('nan'), float('inf'), -float('inf')])
def test_real_driver_nonfinite_optimizer_count_refuses_save(count):
    run, factory = load_real_driver_tail()
    trainer = factory(5, 6, save_freq=1, test_freq=0)
    trainer.baseline_controller = SimpleNamespace(method='topd', after_actor=lambda *args: None)
    trainer.actor_rollout_wg.update_actor = lambda batch: SimpleNamespace(meta_info={'metrics': {'actor/optimizer_steps': count}})
    with pytest.raises(RuntimeError):
        run(trainer)
    assert trainer.global_steps == 5 and trainer.events == []


def test_teacher_boundary_detects_child_training_despite_parent_eval(tmp_path, monkeypatch):
    monkeypatch.setenv('SEARCH_R1_TOPD_AUDIT_DIR', str(tmp_path))
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(.1)).eval().requires_grad_(False)
    model[1].train()
    record = teacher_boundary(SimpleNamespace(model=model), 0, 'loaded')
    assert not record['training'] and record['training_modules_count'] == 1
    assert record['parameter_bytes'] == sum(p.numel() * p.element_size() for p in model.parameters()) == 24


def audit_environment(monkeypatch, enabled):
    variables = {f'SEARCH_R1_TOPD_{key}': value for key, value in
                 [('AUDIT_DIR', '/audit/run'), ('RUN_ID', 'run42'), ('COMMIT', 'commit42'),
                  ('CONFIG_SHA256', 'config42'), ('DATA_IDENTITY', 'data42')]}
    for key, value in variables.items():
        monkeypatch.setenv(key, value)
    if not enabled:
        monkeypatch.delenv('SEARCH_R1_TOPD_AUDIT_DIR')
    return variables


@pytest.mark.parametrize('enabled', [False, True])
def test_actual_main_task_call_explicitly_propagates_opt_in_env(monkeypatch, enabled):
    import sys
    from unittest.mock import MagicMock
    from search_r1.baselines import main as entry
    from search_r1.baselines import preflight, holdout
    variables = audit_environment(monkeypatch, enabled)
    task = MagicMock()
    remote_task = task.options.return_value if enabled else task
    remote_task.remote.return_value = 'result-ref'
    monkeypatch.setitem(sys.modules, 'verl.trainer.main_ppo', SimpleNamespace(main_task=task))
    monkeypatch.setattr(preflight, 'validate_gpu_lease', lambda config: None)
    monkeypatch.setattr(preflight, 'validate_assets_and_retriever', lambda config: None)
    monkeypatch.setattr(holdout, 'prepare_common_holdout', lambda config: {})
    monkeypatch.setattr(entry, 'validate_config', lambda config: None)
    monkeypatch.setattr(entry.ray, 'is_initialized', lambda: True)
    monkeypatch.setattr(entry.ray, 'get', lambda ref: ref)
    config = OmegaConf.create(dict(trainer=dict(smoke_only=False), baselines=dict(holdout_eval_only=False)))
    entry.main.__wrapped__(config)
    remote_task.remote.assert_called_once_with(config)
    if enabled:
        task.options.assert_called_once_with(runtime_env={'env_vars': variables})
    else:
        task.options.assert_not_called()


@pytest.mark.parametrize('enabled', [False, True])
def test_actual_worker_creation_propagates_all_eight_rank_envs(monkeypatch, enabled):
    from copy import deepcopy
    from verl.single_controller.ray import base
    variables = audit_environment(monkeypatch, enabled)
    captured = []
    class CIA:
        cls = object
        def update_options(self, options):
            captured.append(deepcopy(options))
        def __call__(self, **kwargs):
            return object()
    group = SimpleNamespace(name_prefix='topd-test', _workers=[], _worker_names=[])
    pool = SimpleNamespace(use_gpu=True, world_size=8, max_collocate_count=1, store=[8],
                           get_placement_groups=lambda **kwargs: [SimpleNamespace(bundle_count=8)])
    center = SimpleNamespace(get_rank_zero_info=SimpleNamespace(remote=lambda: 'rank-zero-ref'))
    monkeypatch.setattr(base, 'list_named_actors', lambda: ['topd-test_register_center'])
    monkeypatch.setattr(base.ray, 'get_actor', lambda name: center)
    monkeypatch.setattr(base.ray, 'get', lambda ref: {'MASTER_ADDR': 'localhost', 'MASTER_PORT': '1234'})
    base.RayWorkerGroup._init_with_resource_pool(group, pool, CIA(), True, False)
    assert len(captured) == len(group._workers) == 8
    for rank, options in enumerate(captured):
        env = options['runtime_env']['env_vars']
        assert env['RANK'] == str(rank) and env['WORLD_SIZE'] == '8'
        propagated = {key: value for key, value in env.items() if key.startswith('SEARCH_R1_TOPD_')}
        assert propagated == (variables if enabled else {})


@pytest.mark.parametrize('value', ['0', '17', '-1', '1.0', '', 'true', '１６'])
def test_teacher_thread_budget_rejects_invalid_before_mutation(monkeypatch, value):
    from search_r1.baselines.topd_runtime_audit import configure_teacher_cpu_threads
    monkeypatch.setenv('SEARCH_R1_TOPD_TEACHER_CPU_THREADS', value)
    with patch.object(torch, 'set_num_threads') as setter:
        with pytest.raises(ValueError, match='1..16'):
            configure_teacher_cpu_threads()
        setter.assert_not_called()


def test_teacher_thread_budget_default_is_unchanged(monkeypatch):
    from search_r1.baselines.topd_runtime_audit import configure_teacher_cpu_threads
    monkeypatch.delenv('SEARCH_R1_TOPD_TEACHER_CPU_THREADS', raising=False)
    with patch.object(torch, 'set_num_threads') as setter:
        assert configure_teacher_cpu_threads() == torch.get_num_threads()
        setter.assert_not_called()


def test_real_controller_sets_only_driver_budget_before_teacher_load(monkeypatch):
    from search_r1.baselines.controller import Controller
    from search_r1.baselines import distillation
    monkeypatch.setenv('SEARCH_R1_TOPD_TEACHER_CPU_THREADS', '16')
    monkeypatch.delenv('SEARCH_R1_TOPD_AUDIT_DIR', raising=False)
    events = []
    monkeypatch.setattr(torch, 'set_num_threads', lambda count: events.append(('driver_threads', count)))
    def teacher(path, tokenizer, device):
        assert events == [('driver_threads', 16)]
        events.append(('teacher', device))
        return object()
    monkeypatch.setattr(distillation, 'FrozenTeacher', teacher)
    trainer = SimpleNamespace(tokenizer=object(), config=OmegaConf.create(dict(baselines=dict(
        seed=42, method='topd', bapo=dict(patience=5), distillation=dict(teacher_path='fixture')))))
    Controller(trainer)
    assert events == [('driver_threads', 16), ('teacher', 'cpu')]


def test_teacher_resource_env_propagated_with_audit_without_changing_torch(monkeypatch):
    from search_r1.baselines.topd_runtime_audit import runtime_env_vars
    variables = audit_environment(monkeypatch, True)
    monkeypatch.setenv('SEARCH_R1_TOPD_TEACHER_CPU_THREADS', '16')
    with patch.object(torch, 'set_num_threads') as setter:
        assert runtime_env_vars() == dict(variables, SEARCH_R1_TOPD_TEACHER_CPU_THREADS='16')
        setter.assert_not_called()


@pytest.mark.parametrize('bad_config,bad_data', [(False, False), (True, False), (False, True)])
def test_actual_entry_config_capture_rejects_drift_before_dispatch(tmp_path, monkeypatch, bad_config, bad_data):
    import hashlib
    from search_r1.baselines.topd_runtime_audit import config_boundary
    config = OmegaConf.create(dict(trainer=dict(total_training_steps=6), data=dict(train_files=['local/train.parquet'])))
    text = OmegaConf.to_yaml(OmegaConf.create(OmegaConf.to_container(config, resolve=True))).encode()
    expected = hashlib.sha256(text).hexdigest()
    monkeypatch.setenv('SEARCH_R1_TOPD_AUDIT_DIR', str(tmp_path))
    monkeypatch.setenv('SEARCH_R1_TOPD_CONFIG_SHA256', 'wrong' if bad_config else expected)
    monkeypatch.setenv('SEARCH_R1_TOPD_DATA_IDENTITY', 'wrong' if bad_data else 'actual-split')
    split = dict(content_identity=dict(sha256='actual-split'))
    if bad_config or bad_data:
        with pytest.raises(RuntimeError):
            config_boundary(config, split)
        assert not (tmp_path/'entry-config.yaml').exists()
    else:
        config_boundary(config, split)
        assert (tmp_path/'entry-config.yaml').read_bytes() == text
        row = json.loads((tmp_path/'entry.jsonl').read_text())
        assert row['actual_config_sha256'] == expected and row['actual_data_identity'] == 'actual-split'
        assert row['entry_total_training_steps'] == 6
        with pytest.raises(FileExistsError):
            config_boundary(config, split)


def test_entry_config_capture_default_off_has_no_side_effects(tmp_path, monkeypatch):
    from search_r1.baselines.topd_runtime_audit import config_boundary
    monkeypatch.delenv('SEARCH_R1_TOPD_AUDIT_DIR', raising=False)
    config_boundary(None, None)
    assert not list(tmp_path.iterdir())


def test_real_main_refuses_config_drift_before_ray_training_dispatch(tmp_path, monkeypatch):
    import sys
    from unittest.mock import MagicMock
    from search_r1.baselines import main as entry, preflight, holdout
    task = MagicMock()
    monkeypatch.setitem(sys.modules, 'verl.trainer.main_ppo', SimpleNamespace(main_task=task))
    monkeypatch.setattr(preflight, 'validate_gpu_lease', lambda config: None)
    monkeypatch.setattr(preflight, 'validate_assets_and_retriever', lambda config: None)
    monkeypatch.setattr(holdout, 'prepare_common_holdout', lambda config: dict(content_identity=dict(sha256='actual-split')))
    monkeypatch.setattr(entry, 'validate_config', lambda config: None)
    monkeypatch.setenv('SEARCH_R1_TOPD_AUDIT_DIR', str(tmp_path))
    monkeypatch.setenv('SEARCH_R1_TOPD_CONFIG_SHA256', 'changed-config')
    monkeypatch.setenv('SEARCH_R1_TOPD_DATA_IDENTITY', 'actual-split')
    config = OmegaConf.create(dict(trainer=dict(smoke_only=True, total_training_steps=6),
                                  baselines=dict(method='topd', holdout_eval_only=False)))
    with pytest.raises(RuntimeError, match='differs from frozen launch config'):
        entry.main.__wrapped__(config)
    task.options.assert_not_called()
    task.remote.assert_not_called()
    assert not list(tmp_path.iterdir())
