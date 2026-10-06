"""Unexecuted CPU controls for v6 diagnostic-I/O continuation regression.

Actual emitter/handler/worker AST bodies use a deterministic common MAX fixture.
Eight CPU invocations are synthetic ranks, not proof of distributed GPU behavior.
Unknown original errors never enter the matched diagnostic MAX in these bodies.
"""
import ast
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from search_r1.baselines.partial_update import _adam_steps, rejection_disposition
from tests.baselines.test_fresh200_worker_skip_static import (
    KnownRefusal, ROOT, source_function, worker_fixture)


class IORefusal(KnownRefusal):
    def __init__(self, reason, record):
        super().__init__(record)
        self.reason = reason


class GradientRefusal(ValueError):
    def __init__(self, message, diagnostic_error=None):
        super().__init__(message)
        self.diagnostic_error = diagnostic_error


def source_handler(method, selector, namespace, *, whole_optimizer_handler=False):
    tree = ast.parse((ROOT / 'verl/workers/actor/dp_actor.py').read_text())
    function = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name == method)
    handler = next(node for node in ast.walk(function) if isinstance(node, ast.ExceptHandler)
                   and selector in ast.get_source_segment((ROOT / 'verl/workers/actor/dp_actor.py').read_text(), node))
    if whole_optimizer_handler:
        skeleton = ast.parse('def invoke(self, supplied, safety_enabled, scaled_joint, batch_idx, updates, outer_identity):\n'
            '    try:\n        raise supplied\n    except BaseException as optimizer_error:\n        pass\n').body[0]
        skeleton.body[0].handlers[0].body = copy.deepcopy(handler.body)
    else:
        skeleton = ast.parse('def invoke(record, error, supplied):\n'
            '    try:\n        raise supplied\n    except ValueError as matched_error:\n        pass\n').body[0]
        skeleton.body[0].handlers[0].body = copy.deepcopy(handler.body)

    class FixtureImports(ast.NodeTransformer):
        def visit_Import(self, node): return ast.copy_location(ast.Pass(), node)
        def visit_ImportFrom(self, node): return ast.copy_location(ast.Pass(), node)

    skeleton = FixtureImports().visit(skeleton)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[skeleton],type_ignores=[])),
                 'actual_actor_refusal_handler_cpu_fixture', 'exec'), namespace)
    return namespace['invoke']


def common_transport(rank, decisions):
    calls = []
    def reduce(flag, op, group):
        calls.append(int(flag))
        flag.fill_(decisions[len(calls)-1])
    return SimpleNamespace(int32=torch.int32,
        tensor=lambda value, device, dtype: torch.tensor(value,dtype=dtype),
        distributed=SimpleNamespace(is_initialized=lambda:True, get_rank=lambda:rank,
            all_reduce=reduce, ReduceOp=SimpleNamespace(MAX='MAX'))), calls


def actual_io_emitter(kind, rank, worker, reason=None, phase='before', global_io_failed=True):
    local_error = 'OSError: only rank3 diagnostic path failed' if rank==3 and global_io_failed else None
    proxy, calls = common_transport(rank, [1,int(global_io_failed)] if kind=='invalid'
                                    else [int(global_io_failed)])
    namespace = dict(torch=proxy, os=os, json=json, Path=Path, PolicySafetyRejected=IORefusal,
        FSDP=type('UnusedCPUFSDP',(),{}), MatchedGradientRejected=GradientRefusal)
    record = dict(outer_step=1, mini_index=2, phase=phase,
        actual_optimizer_writes=2, rejection_reason=reason, allowed=True)
    if kind=='monitor':
        handler = source_handler('_policy_safety_guard','matched monitor evidence IO refusal',namespace)
        handler(record,local_error,ValueError('original common monitor IO MAX refusal'))
    if kind=='invalid':
        def diagnostic_save(state,path):
            if local_error is not None: raise OSError(local_error)
        namespace.update(atomic_torch_save=diagnostic_save, cpu_snapshot=lambda value:value)
        emitter = source_function('verl/workers/actor/dp_actor.py','_safety_sync_invalid',namespace)
        emitter(SimpleNamespace(actor_module=worker.actor_module_fsdp,_fresh200_attempt_id=1),
            rank==3,'cpu',dict(outer_step=1,mini_index=2),phase,2)
    if kind=='gradient':
        def diagnostic_failure(*args):
            if local_error is not None: raise OSError(local_error)
        namespace.update(_failure=diagnostic_failure)
        matched = source_function('search_r1/baselines/gradient_norm.py',
                                  '_matched_gradient_rejection',namespace)
        try:
            matched([],torch.tensor(float('nan')),dict(attempt_id=1))
        except GradientRefusal as original:
            handler=source_handler('update_policy','if safety_enabled and isinstance(optimizer_error, MatchedGradientRejected)',
                                   namespace,whole_optimizer_handler=True)
            handler(SimpleNamespace(actor_module=worker.actor_module_fsdp,_fresh200_attempt_id=1),
                    original,True,False,2,2,dict(baseline_outer_step=1))
    raise AssertionError('Actual matched emitter unexpectedly returned')


@pytest.mark.parametrize('kind,reason,phase', [
    ('monitor',None,'before'),('monitor',None,'after'),
    ('monitor','OLD_POLICY_TOKEN_KL','before'),
    ('monitor','NEGATIVE_JOINT_ACTION_TAIL_COUNT','after'),
    ('invalid',None,'gradient_micro0'),('invalid',None,'gradient_action_prefix_outputs'),
    ('gradient',None,'global_gradient_before_adam')])
def test_one_rank_diagnostic_failure_is_common_terminal_even_when_later_snapshot_succeeds(
        tmp_path,monkeypatch,kind,reason,phase):
    monkeypatch.setenv('SEARCH_R1_BASELINE_AUDIT_DIR',str(tmp_path/'diagnostics'))
    records=[]
    for rank in range(8):
        worker,data,method,calls,stages,saved=worker_fixture(tmp_path/str(rank),{})
        def refused_update(data):
            # Real CPU Adam writes precede the failed evidence operation.
            for mini in range(2):
                calls.append(mini)
                worker.actor_optimizer.zero_grad(set_to_none=True)
                worker.actor_module_fsdp.weight.grad=torch.ones_like(worker.actor_module_fsdp.weight)
                worker.actor_optimizer.step()
                worker.actor._fresh200_attempt_adam_writes+=1
                worker.actor._fresh200_attempt_gradient_micros+=8
            calls.append(2)
            actual_io_emitter(kind,rank,worker,reason,phase)
        worker.actor.update_policy=refused_update
        with pytest.raises(IORefusal) as caught: method(worker,data)
        record=caught.value.record
        records.append(record)
        local_key={'monitor':'local_io_error','invalid':'local_diagnostic_io_error',
                   'gradient':'gradient_diagnostic_error'}[kind]
        assert (record[local_key] is not None)==(rank==3)
        assert record['global_diagnostic_io_failed'] is True
        assert rejection_disposition(record)=='fail_closed'
        assert calls==[0,1,2] and stages==['attempt/entry_actual_counters']
        assert worker.actor_lr_scheduler.last_epoch==0 and worker.actor_lr_scheduler.writes==0
        assert not hasattr(worker,'fresh200_worker_ledger')
        assert _adam_steps({'optimizer':worker.actor_optimizer.state_dict()})==2
        assert not saved  # No safe continuation or next opportunity was entered.
        # Even successful later partial persistence must not reinterpret this
        # terminal error. Retain actual model/Adam/pending-gradient evidence.
        preserved=method.__globals__['preserve_worker'](worker,data,caught.value,1)
        assert preserved['partial_update']['actual_adam_total']==2
        assert saved[preserved['state_receipt']['path']]['partial_update']['rejection']==record
    assert {rejection_disposition(record) for record in records}=={'fail_closed'}


def test_healthy_matched_gradient_diagnostic_remains_numerical_skip(tmp_path):
    worker,_,_,_,_,_=worker_fixture(tmp_path,{})
    with pytest.raises(IORefusal) as caught:
        actual_io_emitter('gradient',0,worker,global_io_failed=False)
    assert caught.value.record['global_diagnostic_io_failed'] is False
    assert rejection_disposition(caught.value.record)=='skip_opportunity'


def test_unknown_original_optimizer_error_never_enters_diagnostic_max(tmp_path):
    worker,_,_,_,_,_=worker_fixture(tmp_path,{})
    proxy,calls=common_transport(0,[])
    namespace=dict(torch=proxy,os=os,Path=Path,PolicySafetyRejected=IORefusal,
        FSDP=type('UnusedCPUFSDP',(),{}),MatchedGradientRejected=GradientRefusal)
    handler=source_handler('update_policy','if safety_enabled and isinstance(optimizer_error, MatchedGradientRejected)',
                           namespace,whole_optimizer_handler=True)
    with pytest.raises(OSError,match='original optimizer/collective'):
        handler(SimpleNamespace(actor_module=worker.actor_module_fsdp),
                OSError('original optimizer/collective'),True,False,2,2,dict(baseline_outer_step=1))
    assert calls==[]
