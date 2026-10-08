"""Frozen source normalization + actual tiny Gloo entrypoints; no runtime edits."""
import ast
import copy
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace as N

SOURCE=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/work/tree-behavior-mu-ppo')
sys.path.insert(0,str(SOURCE))
import torch.multiprocessing as mp
import test_tree_guard_continue_cpu as fixture


def normalized():
    tree=ast.parse((SOURCE/'verl/workers/fsdp_workers.py').read_bytes())
    klass=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='ActorRolloutRefWorker')
    init=next(node for node in klass.body if isinstance(node,ast.FunctionDef) and node.name=='__init__')
    nodes=[copy.deepcopy(node) for node in init.body if isinstance(node,ast.If)
           and any(isinstance(child,ast.AugAssign) and isinstance(child.op,ast.FloorDiv) for child in ast.walk(node))]
    assert len(nodes)==3
    function=ast.FunctionDef(name='normalize',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='self')],
        kwonlyargs=[],kw_defaults=[],defaults=[]),body=nodes,decorator_list=[])
    module=ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[]));scope={}
    exec(compile(module,'actual-frozen-worker-normalization','exec'),scope)
    worker=N(_is_actor=True,_is_rollout=True,_is_ref=True,ulysses_sequence_parallel_size=1,
        device_mesh=N(shape=(8,1)),config=N(actor=N(ppo_mini_batch_size=128,ppo_micro_batch_size=16),
        rollout=N(n=1,log_prob_micro_batch_size=32),ref=N(log_prob_micro_batch_size=32)))
    scope['normalize'](worker)
    assert (worker.config.actor.ppo_mini_batch_size,worker.config.actor.ppo_micro_batch_size)==(16,2)
    assert worker.config.rollout.log_prob_micro_batch_size==worker.config.ref.log_prob_micro_batch_size==4
    return dict(global_mini=128,global_actor_micro=16,global_old_ref_micro=32,dp=8,sp=1,
        local_mini=16,local_actor_micro=2,local_old_ref_micro=4,actual_normalization_ast=True)


def worker(rank,rendezvous,run,mode):
    value=normalized();original=fixture.cfg
    def config():
        result=original();result['ppo_micro_batch_size']=value['local_actor_micro'];return result
    fixture.cfg=config
    fixture.gloo_worker(rank,rendezvous,run,mode)


if __name__=='__main__':
    owner=Path(__file__).resolve().parent;started=time.monotonic();results=[]
    for mode in ('post1','allskip','gradient_after1'):
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory)/'run';run.mkdir()
            mp.spawn(worker,args=(str(run/'gloo-init'),str(run),mode),nprocs=2,join=True)
            proof=json.loads((run/'cpu-proof.json').read_text())
            (owner/f'normalized-micro2-{mode}.json').write_text(json.dumps(
                dict(scope='Real Gloo/Adam/source AST; explicitly synthetic FSDP layout and emission factory',
                normalization=normalized(),proof=proof),sort_keys=True,allow_nan=False))
            results.append(dict(mode=mode,success=True))
    print(json.dumps(dict(normalization=normalized(),scenarios=results,seconds=time.monotonic()-started,
        actual_gpu_executed=False,actual_native_vllm_executed=False),sort_keys=True))
