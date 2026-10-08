"""Read-only actual-method counterexample; tiny CPU Adam and real two-rank Gloo.

The FSDP object layout is synthetic and lazy initialization is bypassed. The
unmodified clip function, actor optimizer method and worker control method run.
This proves control-flow refusal typing, never real FSDP/NCCL/GPU acceptance.
"""
import ast
from contextlib import nullcontext
import copy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace as N
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

SOURCE = Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/work/tree-behavior-mu-ppo')
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE))


class Config(dict):
    __getattr__ = dict.__getitem__


class SyntheticFSDP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.]))
        self._is_root = True
        self._all_handles = []
        self.compute_device = torch.device('cpu')
        self.process_group = dist.group.WORLD
        self.cpu_offload = N(offload_params=False)

    def _assert_state(self, state):
        pass


class TinyBatch:
    def cuda(self):
        return self


class Timer:
    def __init__(self, *args, **kwargs):
        self.last = 1.

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class Sharding:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def preprocess_data(self, data):
        return data

    def postprocess_data(self, data):
        return data


def actual_method(relative, klass, name, scope):
    file = SOURCE / relative
    tree = ast.parse(file.read_bytes())
    body = next(n.body for n in tree.body if isinstance(n, ast.ClassDef) and n.name == klass)
    node = copy.deepcopy(next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name))
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(file), 'exec'), scope)
    return scope[name]


def rank_main(rank, rendezvous):
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2)
    from search_r1.baselines import gradient_norm
    from search_r1.baselines.policy_safety_runtime import PolicySafetyRefusal
    try:
        run = OUT / 'f04-artifacts'
        env = dict(SLURM_JOB_ID='cpu-owned-f04', SEARCH_R1_BASELINE_RUN_ID='strict-review-f04',
            SEARCH_R1_BASELINE_COMMIT='8a3bc24db05ee1fd67660e2f89088d6f56bc0875',
            SEARCH_R1_BASELINE_CONFIG_SHA256='cpu-config', SEARCH_R1_BASELINE_DATA_IDENTITY='cpu-data',
            SEARCH_R1_MODEL_IDENTITY='tiny-cpu-model', SEARCH_R1_BASELINE_OUTPUT=str(run/'output'))
        with patch.dict(os.environ, env):
            model = SyntheticFSDP()
            optimizer = torch.optim.Adam(model.parameters(), lr=.01)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.)
            sync_calls = []
            safety = N(committed=0, sync_valid=lambda *a: sync_calls.append(a), publish=lambda *a: None)
            actor = N(actor_module=model, actor_optimizer=optimizer, config=Config(grad_clip=1), _policy_safety=safety)
            step = actual_method('verl/workers/actor/dp_actor.py', 'DataParallelPPOActor', '_optimizer_step', dict(torch=torch))
            def update_policy(data):
                model.weight.grad = torch.ones_like(model.weight)
                step(actor)
                model.weight.grad = torch.full_like(model.weight, float('nan') if rank == 1 else 1.)
                step(actor)
                raise AssertionError('Second optimizer call must refuse')
            actor.update_policy = update_policy
            worker = N(actor=actor, actor_optimizer=optimizer, actor_lr_scheduler=scheduler,
                config=Config(actor=Config(fresh200_protocol='tree_fresh200_five_leaf_v1')),
                _is_actor=True, _is_offload_param=False, _is_offload_optimizer=False,
                ulysses_sharding_manager=Sharding())
            identity = {key: env[name] for key,name in dict(job_id='SLURM_JOB_ID', run_id='SEARCH_R1_BASELINE_RUN_ID',
                code_commit='SEARCH_R1_BASELINE_COMMIT', config_sha256='SEARCH_R1_BASELINE_CONFIG_SHA256',
                data_identity='SEARCH_R1_BASELINE_DATA_IDENTITY', model_identity='SEARCH_R1_MODEL_IDENTITY').items()}
            identity.update(training_protocol='tree_fresh200_five_leaf_v1', attempted_outer=1, completed_outer=0)
            data = N(batch=TinyBatch(), meta_info=dict(baseline_outer_step=1,
                tree_partial_context=dict(protocol='tree_partial_state_v1', identity=identity,
                directory=str(run/'training_state/partial_outer_000001'))))
            data.to = lambda device: data
            update = actual_method('verl/workers/fsdp_workers.py', 'ActorRolloutRefWorker', 'update_actor',
                dict(torch=torch, os=os, DataProto=N, Timer=Timer, logger=None,
                    log_gpu_memory_usage=lambda *a, **k: None))
            caught = None
            with patch.object(gradient_norm, 'FSDP', SyntheticFSDP), \
                 patch('torch.distributed.fsdp._runtime_utils._lazy_init', lambda *a: None), \
                 patch.object(torch.cuda, 'empty_cache', lambda: None):
                try:
                    update(worker, data)
                except Exception as error:
                    caught = error
            record = dict(rank=rank, exception_type=type(caught).__name__, exception=str(caught),
                policy_safety_refusal=isinstance(caught, PolicySafetyRefusal),
                matched_all_ranks=getattr(caught, 'matched_all_ranks', False),
                actual_optimizer_returns=actor._actual_committed_optimizer_calls,
                actual_adam_step=int(next(iter(optimizer.state.values()))['step'].item()),
                actual_weight=model.weight.item(), scheduler_last_epoch=scheduler.last_epoch,
                safety_sync_calls=len(sync_calls), worker_snapshots=list(map(str, run.rglob('worker-*.pt'))),
                scope='Unmodified source methods, tiny CPU Adam, real Gloo, synthetic FSDP layout; no GPU/FSDP training claim')
            assert isinstance(caught, ValueError) and not isinstance(caught, PolicySafetyRefusal)
            assert record['actual_optimizer_returns'] == record['actual_adam_step'] == 1
            assert record['actual_weight'] != 1. and record['scheduler_last_epoch'] == 0
            assert record['worker_snapshots'] == []
            (OUT / f'F04_RANK_{rank}.json').write_text(json.dumps(record, indent=2, allow_nan=False))
            print(json.dumps(record, allow_nan=False), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    mp.spawn(rank_main, args=(str(OUT / 'f04-gloo-init'),), nprocs=2, join=True)
