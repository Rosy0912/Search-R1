"""Run the real driver tail: empty updates cannot consume the outer update budget."""
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduction'))
from test_final_checkpoint import real_iteration_tail, FakeTrainer


@pytest.mark.parametrize('method',['grpo','gigpo','tree_grpo'])
def test_zero_actor_update_refuses_budget_and_checkpoint(method):
    t=FakeTrainer(5,6)
    t.config.baselines=SimpleNamespace(method=method)
    with pytest.raises(RuntimeError,match='no actor update'):real_iteration_tail()(t)
    assert t.global_steps==5
    assert not any(e[0] in ('save','final_data') for e in t.events)


@pytest.mark.parametrize('method',['grpo','gigpo','tree_grpo'])
def test_multiple_optimizer_steps_count_as_one_outer_update(method):
    t=FakeTrainer(1,501)
    t.config.baselines=SimpleNamespace(method=method)
    t.actor_rollout_wg.update_actor=lambda batch:SimpleNamespace(meta_info={'metrics':{'actor/optimizer_steps':2}})
    run=real_iteration_tail()
    run(t);run(t)
    assert t.actual_outer_updates==2
    assert t.global_steps==3


@pytest.mark.parametrize('method',['grpo','gigpo','tree_grpo'])
def test_warmup_skip_refuses_budget_and_checkpoint(method):
    t=FakeTrainer(5,6)
    t.config.baselines=SimpleNamespace(method=method)
    t.config.trainer.critic_warmup=6
    with pytest.raises(RuntimeError,match='no actor update'):real_iteration_tail()(t)
    assert t.global_steps==5
    assert getattr(t,'actual_outer_updates',0)==0
    assert not any(e[0] in ('save','final_data') for e in t.events)
