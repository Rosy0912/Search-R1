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
from types import MethodType, SimpleNamespace

import pytest
import torch

from search_r1.baselines.partial_update import _adam_steps, rejection_disposition
from tests.baselines.test_fresh200_worker_skip_static import (
    KnownRefusal, ROOT, source_function, worker_fixture)


class IORefusal(KnownRefusal):
    def __init__(self, reason, record):
        super().__init__(reason, record)


class GradientRefusal(ValueError):
    def __init__(self, message, diagnostic_error=None):
        super().__init__(message)
        self.diagnostic_error = diagnostic_error


def source_handler(method, selector, namespace, *, whole_optimizer_handler=False):
    source = (ROOT / 'verl/workers/actor/dp_actor.py').read_text()
    tree = ast.parse(source)
    function = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name == method)
    if method == '_policy_safety_guard':
        index = next(index for index, node in enumerate(function.body)
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
            and target.id == 'io_bad' for target in node.targets))
        skeleton = ast.parse('def invoke(record,error,supplied=None,device="cpu",group=None):\n    pass\n').body[0]
        # Keep the original diagnostic IO MAX, record update, and typed branch;
        # the removed broad ValueError handler must not be resurrected here.
        skeleton.body = copy.deepcopy(function.body[index:index + 4])
    elif whole_optimizer_handler:
        handler = next(node for node in ast.walk(function) if isinstance(node, ast.ExceptHandler)
                       and selector in ast.get_source_segment(source, node))
        skeleton = ast.parse('def invoke(self, supplied, safety_enabled, scaled_joint, batch_idx, updates, outer_identity):\n'
            '    try:\n        raise supplied\n    except BaseException as optimizer_error:\n        pass\n').body[0]
        skeleton.body[0].handlers[0].body = copy.deepcopy(handler.body)
    else:
        handler = next(node for node in ast.walk(function) if isinstance(node, ast.ExceptHandler)
                       and selector in ast.get_source_segment(source, node))
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
    proxy, calls = common_transport(rank, [2,int(global_io_failed)] if kind=='invalid'
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


@pytest.mark.parametrize('error_type', [ValueError, RuntimeError, torch.cuda.OutOfMemoryError])
def test_unknown_original_monitor_io_max_is_not_a_typed_refusal(error_type):
    original = error_type('unknown original monitor diagnostic IO MAX')
    proxy, calls = common_transport(0, [])
    def original_failure(flag, op, group):
        calls.append(int(flag))
        raise original
    proxy.distributed.all_reduce = original_failure
    namespace = dict(torch=proxy, PolicySafetyRejected=IORefusal)
    handler = source_handler('_policy_safety_guard', 'matched monitor evidence IO refusal', namespace)
    record = dict(outer_step=1, mini_index=2, phase='before')
    with pytest.raises(error_type) as caught:
        handler(record, None)
    assert caught.value is original and not isinstance(caught.value, KnownRefusal)
    assert calls == [0] and 'matched_guard' not in record


def finite_loss_fixture(worker,rank,decisions,*,local_io_failure=False,unknown=None):
    """Actual admission plus immediately following backward/counter AST bodies."""
    source=(ROOT/'verl/workers/actor/dp_actor.py').read_text()
    function=next(node for node in ast.walk(ast.parse(source))
                  if isinstance(node,ast.FunctionDef) and node.name=='update_policy')
    body=None
    for parent in ast.walk(function):
        statements=getattr(parent,'body',None)
        if not isinstance(statements,list): continue
        for index,node in enumerate(statements):
            if (isinstance(node,ast.If) and isinstance(node.test,ast.Name)
                    and node.test.id=='safety_enabled'
                    and 'NONFINITE_TRAINING_LOSS' in ast.get_source_segment(source,node)):
                body=copy.deepcopy(statements[index:index+4])
                break
        if body is not None: break
    assert body is not None and len(body)==4
    skeleton=ast.parse('def admit(self,loss,log_prob,outer_identity,batch_idx,micro_index,updates,data,'
        'safety_enabled=True,scaled_joint=False,loss_scale=1.,audit=None,fresh_attempt=True):\n    pass\n').body[0]
    skeleton.body=body
    class FixtureImports(ast.NodeTransformer):
        def visit_Import(self,node): return ast.copy_location(ast.Pass(),node)
        def visit_ImportFrom(self,node): return ast.copy_location(ast.Pass(),node)
    skeleton=FixtureImports().visit(skeleton)
    proxy,reductions=common_transport(rank,decisions)
    proxy.isfinite=torch.isfinite
    if unknown is not None:
        def original_collective_failure(flag,op,group):
            reductions.append(int(flag))
            raise unknown
        proxy.distributed.all_reduce=original_collective_failure
    diagnostics=[]
    def diagnostic_save(state,path):
        diagnostics.append(copy.deepcopy(state))
        if local_io_failure: raise OSError('one rank finite-loss diagnostic disk failure')
    namespace=dict(torch=proxy,dist=proxy.distributed,os=os,json=json,Path=Path,
        PolicySafetyRejected=IORefusal,atomic_torch_save=diagnostic_save,cpu_snapshot=lambda value:value)
    sync=source_function('verl/workers/actor/dp_actor.py','_safety_sync_invalid',namespace)
    namespace['coordinated_badflag']=source_function('search_r1/baselines/loss_scaling.py',
                                                   'coordinated_badflag',namespace)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[skeleton],type_ignores=[])),
                 'actual_finite_loss_admission_and_backward_cpu_fixture','exec'),namespace)
    actor=worker.actor
    actor.actor_module=worker.actor_module_fsdp
    actor._fresh200_attempt_id=1
    actor._fresh200_attempt_adam_writes=0
    actor._fresh200_attempt_gradient_micros=0
    actor._safety_sync_invalid=MethodType(sync,actor)
    return actor,namespace['admit'],reductions,diagnostics


@pytest.mark.parametrize('safety_enabled,scaled_joint',[(True,False),(False,True)])
@pytest.mark.parametrize('error_type',[ValueError,RuntimeError,OSError])
def test_unknown_finite_loss_collective_error_is_unchanged_no_new_max_or_save(
        tmp_path,safety_enabled,scaled_joint,error_type):
    worker,data,method,calls,stages,saved=worker_fixture(tmp_path,{})
    original=error_type('unknown original finite-loss all_reduce failure')
    actor,admit,reductions,diagnostics=finite_loss_fixture(worker,0,[],unknown=original)
    pending=data.meta_info.copy()
    def update(data):
        for mini in range(2):
            calls.append(mini)
            worker.actor_optimizer.zero_grad(set_to_none=True)
            worker.actor_module_fsdp.weight.grad=torch.ones_like(worker.actor_module_fsdp.weight)
            worker.actor_optimizer.step()
            actor._fresh200_attempt_adam_writes+=1
            actor._fresh200_attempt_gradient_micros+=8
        calls.append(2)
        admit(actor,worker.actor_module_fsdp.weight.square().sum(),torch.zeros(1),
              dict(baseline_outer_step=1),2,0,2,dict(pending=pending),
              safety_enabled=safety_enabled,scaled_joint=scaled_joint)
    actor.update_policy=update
    with pytest.raises(error_type) as caught: method(worker,data)
    assert caught.value is original and not isinstance(caught.value,KnownRefusal)
    assert reductions==[0] and diagnostics==[] and saved=={}
    assert calls==[0,1,2] and stages==['attempt/entry_actual_counters']
    assert actor._fresh200_attempt_adam_writes==2 and actor._fresh200_attempt_gradient_micros==16
    assert _adam_steps({'optimizer':worker.actor_optimizer.state_dict()})==2
    assert torch.equal(worker.actor_module_fsdp.weight.grad,torch.ones_like(worker.actor_module_fsdp.weight))
    assert data.meta_info['fresh200_processed_opportunities']==0
    assert worker.actor_lr_scheduler.last_epoch==0 and not hasattr(worker,'fresh200_worker_ledger')
    assert pending['fresh200_adam_total_before']==0


def test_healthy_finite_loss_keeps_one_max_original_backward_and_no_adam(tmp_path):
    worker,_,_,_,_,_=worker_fixture(tmp_path,{})
    actor,admit,reductions,diagnostics=finite_loss_fixture(worker,0,[0])
    audit=SimpleNamespace(gradient_micro_batches=0)
    weight=worker.actor_module_fsdp.weight.detach().clone()
    admit(actor,worker.actor_module_fsdp.weight.square().sum(),torch.zeros(1),
          dict(baseline_outer_step=1),2,0,0,{},audit=audit)
    assert reductions==[0] and diagnostics==[]
    assert torch.equal(worker.actor_module_fsdp.weight.grad,2*weight)
    assert actor._fresh200_attempt_gradient_micros==audit.gradient_micro_batches==1
    assert _adam_steps({'optimizer':worker.actor_optimizer.state_dict()})==0


@pytest.mark.parametrize('global_io_failure',[False,True])
def test_known_finite_loss_refusal_matches_all_ranks_and_io_failure_is_terminal(
        tmp_path,monkeypatch,global_io_failure):
    monkeypatch.setenv('SEARCH_R1_BASELINE_AUDIT_DIR',str(tmp_path/'loss-diagnostics'))
    records=[]
    for rank in range(8):
        worker,data,method,calls,stages,saved=worker_fixture(tmp_path/str(rank),{})
        normal_update=worker.actor.update_policy
        actor,admit,reductions,diagnostics=finite_loss_fixture(worker,rank,
            [1,int(global_io_failure)],local_io_failure=global_io_failure and rank==3)
        def update(data):
            if data.meta_info['fresh200_processed_opportunities']:
                return normal_update(data)  # Actual worker re-entry resumes at mini3.
            for mini in range(2):
                calls.append(mini)
                worker.actor_optimizer.zero_grad(set_to_none=True)
                worker.actor_module_fsdp.weight.grad=torch.ones_like(worker.actor_module_fsdp.weight)
                worker.actor_optimizer.step()
                actor._fresh200_attempt_adam_writes+=1
                actor._fresh200_attempt_gradient_micros+=8
            calls.append(2)
            loss=worker.actor_module_fsdp.weight.square().sum()
            if rank==3: loss=loss*float('nan')
            try:
                admit(actor,loss,torch.zeros(1),dict(baseline_outer_step=1),2,0,2,
                      dict(real_batch=data))
            except IORefusal as refusal:
                records.append(refusal.record)
                assert actor._fresh200_attempt_gradient_micros==16
                raise
        actor.update_policy=update
        if global_io_failure:
            with pytest.raises(IORefusal): method(worker,data)
            assert calls==[0,1,2] and stages==['attempt/entry_actual_counters']
            assert saved=={} and worker.actor_lr_scheduler.last_epoch==0
            assert _adam_steps({'optimizer':worker.actor_optimizer.state_dict()})==2
        else:
            _,receipt=method(worker,data)
            assert calls==list(range(20))
            assert receipt['counted_complete'] is True and receipt['actual_successful_adam_writes']==19
            assert receipt['gradient_micro_batches']==152
            assert len(receipt['refusals'])==1
            assert _adam_steps({'optimizer':worker.actor_optimizer.state_dict()})==19
        assert len(reductions)==2 and len(diagnostics)==1
        assert records[-1]['local_invalid']==(rank==3)
        assert records[-1]['micro_index_in_mini']==0
        assert records[-1]['global_diagnostic_io_failed'] is global_io_failure
        assert (records[-1]['local_diagnostic_io_error'] is not None)==(global_io_failure and rank==3)
    assert {rejection_disposition(record) for record in records}=={
        'fail_closed' if global_io_failure else 'skip_opportunity'}
