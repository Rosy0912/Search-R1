"""Execute the real fit loop's gate-before-fetch prefix and real final tail."""
import ast,os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from omegaconf import OmegaConf
from search_r1.baselines.training_state import CheckpointBatchSampler
from .test_actual_update_budget import real_iteration_tail,FakeTrainer


def real_gate_prefix():
    source=Path(__file__).resolve().parents[2]/'verl/trainer/ppo/ray_trainer.py'
    module=ast.parse(source.read_text());trainer=next(n for n in module.body if isinstance(n,ast.ClassDef) and n.name=='RayPPOTrainer')
    fit=next(n for n in trainer.body if isinstance(n,ast.FunctionDef) and n.name=='fit')
    epoch=next(n for n in fit.body if isinstance(n,ast.For));loop=next(n for n in epoch.body if isinstance(n,ast.While))
    stop=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and getattr(n.value.func,'id',None)=='print')
    body=loop.body[:stop]+[ast.Break()]
    function=ast.FunctionDef(name='fetch',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='self'),ast.arg(arg='batch_iterator')],kwonlyargs=[],kw_defaults=[],defaults=[]),body=[ast.While(test=ast.Constant(True),body=body,orelse=[])],decorator_list=[])
    namespace={'os':os};exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),str(source),'exec'),namespace)
    return namespace['fetch']


def test_real_sixth_batch_is_not_requested_until_gate_releases(tmp_path,monkeypatch):
    monkeypatch.setenv('SEARCH_R1_BASELINE_OUTPUT',str(tmp_path))
    sampler=CheckpointBatchSampler(40,4,42,'data');iterator=iter(sampler)
    for _ in range(5):next(iterator)
    t=SimpleNamespace(global_steps=6,acceptance_ready={'completed_outer_updates':5},config=OmegaConf.create({'trainer':{'acceptance_gate_enabled':True,'acceptance_step':5,'acceptance_gate_timeout_seconds':0}}))
    def hold(*args):
        assert sampler.cursor==20
        raise TimeoutError('independent review pending')
    fetch=real_gate_prefix()
    with patch('search_r1.baselines.training_state.wait_for_acceptance',side_effect=hold):
        with pytest.raises(TimeoutError):fetch(t,iterator)
    assert sampler.cursor==20 and not getattr(t,'acceptance_released',False)
    with patch('search_r1.baselines.training_state.wait_for_acceptance',return_value={'status':'PASS'}) as accept:
        fetch(t,iterator);assert sampler.cursor==24 and t.acceptance_released
        fetch(t,iterator);assert sampler.cursor==28 and accept.call_count==1


def test_real_final_tail_completes300_and_returns_before301():
    t=FakeTrainer(300,301);t.config.baselines=SimpleNamespace(method='tree_grpo')
    t.actual_outer_updates=299
    t.actor_rollout_wg.update_actor=lambda batch:SimpleNamespace(meta_info={'metrics':{'actor/optimizer_steps':24}})
    real_iteration_tail()(t)
    assert t.actual_outer_updates==300 and t.global_steps==301
    assert any(event[0]=='save' for event in t.events)
