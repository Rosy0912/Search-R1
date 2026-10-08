"""Actual modified source seam + actual CPU release/ACK files, never GPU math."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import types
from contextlib import nullcontext
import pytest
from omegaconf import OmegaConf
ROOT=Path(__file__).parents[1]
REPO=ROOT/'topd-shared32b-integration-candidate'
sys.path[:0]=[str(REPO),str(ROOT/'topd-32b-pipeline-ipc-candidate')]
from search_r1.baselines.controller import Controller
from topd32b_shared_pipeline import backend as backend
spec=importlib.util.spec_from_file_location('ipc_contract_fixture',ROOT/'topd-32b-pipeline-ipc-candidate/test_backend.py')
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)


def real_update_tail(batch):
    source=REPO/'verl/trainer/ppo/ray_trainer.py';tree=ast.parse(source.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RayPPOTrainer')
    fit=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='fit')
    epoch=next(n for n in fit.body if isinstance(n,ast.For));batches=next(n for n in epoch.body if isinstance(n,ast.For))
    timed=next(n for n in batches.body if isinstance(n,ast.With))
    update=next(n for n in timed.body if isinstance(n,ast.If) and 'critic_warmup' in ast.unparse(n.test))
    method=ast.FunctionDef(name='run',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='self')],kwonlyargs=[],kw_defaults=[],defaults=[]),body=ast.parse('batch=fixture_batch; metrics={}; timing_raw={}').body+[update],decorator_list=[])
    module=ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[]))
    namespace=dict(fixture_batch=batch,metrics={},timing_raw={},_timer=lambda *args:nullcontext(),reduce_metrics=lambda x:x)
    exec(compile(module,str(source),'exec'),namespace)
    return namespace['run']


@pytest.mark.parametrize('bad',['none','missing_ack','tamper_ack','wrong_step','no_controller'])
def test_actual_driver_update_waits_for_fresh_global_barrier(tmp_path,monkeypatch,bad):
    monkeypatch.delenv('SEARCH_R1_TOPD_AUDIT_DIR',raising=False)
    monkeypatch.delenv('SEARCH_R1_BASELINE_AUDIT_DIR',raising=False)
    c,_,manifest,directory=fixture.setup.__wrapped__(tmp_path)
    _,errors=fixture.phase(c);assert not errors
    receipt=json.loads((directory/'global-release-step1.json').read_text())
    client=types.SimpleNamespace(failed=False,step=2,last_receipts=[receipt],m=manifest,directory=directory)
    controller=Controller.__new__(Controller)
    controller.method='topd';controller.teacher=client;controller.ranker=None
    controller.config=OmegaConf.create(dict(topd=dict(teacher_backend='local_gpu_fp32_shared_pipeline32b')))
    batch=types.SimpleNamespace(meta_info={'topd_outer_step':2 if bad=='wrong_step' else 1})
    updates=[]
    trainer=types.SimpleNamespace(global_steps=1,config=types.SimpleNamespace(do_search=False,
        trainer=types.SimpleNamespace(critic_warmup=0),baselines=types.SimpleNamespace(method='topd',topd={'teacher_backend':'local_gpu_fp32_shared_pipeline32b'})),
        baseline_controller=None if bad=='no_controller' else controller,
        actor_rollout_wg=types.SimpleNamespace(update_actor=lambda data:(updates.append(1) or types.SimpleNamespace(meta_info={'metrics':{'actor/optimizer_steps':20}}))))
    if bad=='missing_ack':(directory/'global-ack-step1.json').unlink()
    if bad=='tamper_ack':
        path=directory/'global-ack-step1.json';ack=json.loads(path.read_text());ack['nonce']='drift';path.write_text(json.dumps(ack))
    run=real_update_tail(batch)
    if bad=='none':run(trainer);assert updates==[1]
    else:
        with pytest.raises((RuntimeError,FileNotFoundError)):run(trainer)
        assert not updates
