#!/usr/bin/env python3
"""Eight-rank native state oracle; synthetic updates, never RL acceptance.

Default profiles cover FP32 baseline and production-like nested decoder FSDP
with BF16 parameter compute, FP32 reductions/buffers, mesh, manual offload.
CPUOffload is an explicit optional extra, not the production offload recipe.
Each case executes 20 initial Adam calls plus two four-call sixth-update
branches (28 actual calls); the second branch reloads the SAME objects.
"""
import argparse
import functools
import hashlib
import json
import os
import random
import subprocess
import sys
import types
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extra-cpu-offload', action='store_true')
    args = parser.parse_args()
    repo = Path(os.environ['SEARCH_R1_REPO']).resolve()
    expected = os.environ['E1_EXPECTED_SOURCE']
    source = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    assert source == expected and not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip()
    sys.path.insert(0, str(repo))
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import (FullyShardedDataParallel as FSDP, CPUOffload,
        MixedPrecision, ShardingStrategy, StateDictType, ShardedStateDictConfig)
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from search_r1.baselines.full_state import state_digest, restore_rng, resolve_warmup_steps
    from verl.workers.fsdp_workers import ActorRolloutRefWorker
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    from verl.workers.sharding_manager.fsdp_vllm import FSDPVLLMShardingManager
    from verl.utils.fsdp_utils import (load_fsdp_param_and_grad, load_fsdp_optimizer,
        offload_fsdp_param_and_grad, offload_fsdp_optimizer)
    from verl.utils.torch_functional import get_constant_schedule_with_warmup
    assert Path(__import__('search_r1.baselines.full_state', fromlist=['x']).__file__).resolve() == repo / 'search_r1/baselines/full_state.py'
    rank = int(os.environ['RANK']); local = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local); dist.init_process_group('nccl')
    assert dist.get_world_size() == 8
    root = Path(os.environ['E1_STATE_FIXTURE_OUTPUT']).resolve(); root.mkdir(parents=True, exist_ok=True)
    mesh = init_device_mesh('cuda', (8,), mesh_dim_names=('fsdp',))

    class DecoderBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.up = torch.nn.Linear(16, 32)
            self.down = torch.nn.Linear(32, 16)
            self.dropout = torch.nn.Dropout(.1)
            self.register_buffer('scale', torch.tensor(.125, dtype=torch.float32))
        def forward(self, x):
            return x + self.dropout(self.down(torch.nn.functional.gelu(self.up(x)))) * self.scale

    class NestedDecoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input = torch.nn.Linear(8, 16)
            self.layers = torch.nn.ModuleList([DecoderBlock(), DecoderBlock()])
            self.output = torch.nn.Linear(16, 2)
        def forward(self, x):
            x = self.input(x)
            for layer in self.layers: x = layer(x)
            return self.output(x)

    profiles = [('baseline-fp32', False, False, False),
                ('production-nested-bf16-manual', True, True, False)]
    if args.extra_cpu_offload:
        profiles.append(('extra-nested-bf16-native-cpu-offload', True, False, True))
    records = []
    try:
        for name, nested, manual, native_cpu in profiles:
            torch.manual_seed(42); torch.cuda.manual_seed(42)
            module = (NestedDecoder() if nested else torch.nn.Sequential(
                torch.nn.Linear(8, 8), torch.nn.Tanh(), torch.nn.Linear(8, 2))).cuda()
            kwargs = dict(device_id=local, use_orig_params=False,
                sharding_strategy=ShardingStrategy.FULL_SHARD, device_mesh=mesh,
                sync_module_states=True, forward_prefetch=False)
            if nested:
                kwargs.update(auto_wrap_policy=functools.partial(transformer_auto_wrap_policy,
                    transformer_layer_cls={DecoderBlock}), mixed_precision=MixedPrecision(
                    param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32))
            if native_cpu: kwargs['cpu_offload'] = CPUOffload(offload_params=True)
            model = FSDP(module, **kwargs)
            wrapper_count = sum(isinstance(m, FSDP) for m in model.modules())
            assert wrapper_count == (3 if nested else 1)
            optim = torch.optim.AdamW(model.parameters(), lr=1e-6)
            assert resolve_warmup_steps(dict(total_training_steps=301,
                lr_warmup_steps_ratio=.285, num_warmup_steps=114)) == 114
            scheduler = get_constant_schedule_with_warmup(optim, 114)
            worker = types.SimpleNamespace(_is_actor=True, _is_offload_param=manual,
                _is_offload_grad=manual, _is_offload_optimizer=manual,
                actor_module_fsdp=model, actor_module=model, actor_optimizer=optim,
                actor_lr_scheduler=scheduler, rank=rank, world_size=8,
                config=types.SimpleNamespace(grad_clip=1.))
            # Execute real production manager save/load methods on actual CUDA
            # RNG tensors; this synthetic fixture does not instantiate vLLM.
            manager=FSDPVLLMShardingManager.__new__(FSDPVLLMShardingManager)
            manager.device_mesh=mesh
            manager.torch_random_states=torch.cuda.get_rng_state(local).cpu().clone()
            manager.gen_random_states=torch.Generator(device=torch.device('cuda',local)).manual_seed(1000+rank).get_state().cpu().clone()
            worker.rollout_sharding_manager=manager
            random.seed(42 + rank); np.random.seed(42 + rank)
            torch.manual_seed(42 + rank); torch.cuda.manual_seed(42 + rank)
            calls = 0; norms = []; outer_lrs = []

            def load_manual():
                if manual:
                    load_fsdp_param_and_grad(model, local, load_grad=True)
                    load_fsdp_optimizer(optim, local)
            def offload_manual():
                if manual:
                    offload_fsdp_param_and_grad(model, offload_grad=True)
                    offload_fsdp_optimizer(optim)
            def update(outer):
                nonlocal calls
                load_manual(); before = optim.param_groups[0]['lr']
                for _ in range(4):
                    optim.zero_grad(set_to_none=True)
                    # Exercise Python, NumPy, CPU and rank-local CUDA RNG; dropout
                    # additionally exercises stochastic nested decoder computation.
                    # Independently advance generation RNG, then restore actor
                    # RNG, matching the real manager enter/exit state exchange.
                    actor_rng=torch.cuda.get_rng_state(local)
                    torch.cuda.set_rng_state(manager.gen_random_states,local)
                    torch.rand(7,device=local)
                    manager.gen_random_states=torch.cuda.get_rng_state(local).cpu().clone()
                    torch.cuda.set_rng_state(actor_rng,local)
                    manager.torch_random_states=actor_rng.cpu().clone()
                    shift = random.random() + float(np.random.random()) + float(torch.rand(()))
                    x = torch.randn(4, 8, device=local) + shift * .01
                    model(x).float().square().mean().backward()
                    norm = DataParallelPPOActor._optimizer_step(worker)
                    calls += 1; assert torch.isfinite(norm); norms.append(float(norm))
                scheduler.step()
                lr = dict(outer=outer, before=before, after=scheduler.get_last_lr()[0])
                offload_manual(); return lr
            def snapshot(directory, step, roundtrip):
                return ActorRolloutRefWorker.save_training_state(worker, str(directory), step, roundtrip)
            def read(report):
                return torch.load(report['path'], map_location='cpu', weights_only=False)

            directory = root / name
            identities = (id(model), id(optim), id(scheduler))
            for outer in range(1, 6): outer_lrs.append(update(outer))
            assert calls == 20 and scheduler.last_epoch == 5
            assert abs(scheduler.get_last_lr()[0] - 1e-6 * 5 / 114) < 1e-18
            initial = snapshot(directory / 'initial-step5', 5, False)
            oracle_lr = update(6)
            oracle = snapshot(directory / 'no-roundtrip-step6', 6, False)
            # Restore the original step5 into the same live objects, then exercise
            # the ACTUAL production worker roundtrip method before the next update.
            loaded = read(initial); load_manual()
            with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT,
                                     ShardedStateDictConfig(offload_to_cpu=True)):
                model.load_state_dict(loaded['model'], strict=True)
                optim.load_state_dict(loaded['optimizer'])
                scheduler.load_state_dict(loaded['scheduler'])
                assert loaded['version']==3 and 'generation_rng'in loaded
                restore_rng(loaded['rng'])
                manager.load_generation_rng_state(loaded['generation_rng'])
            offload_manual()
            roundtrip = snapshot(directory / 'roundtrip-step5', 5, True)
            assert roundtrip['status'] == 'EXACT_FULL_STATE_RELOAD_OK'
            assert roundtrip['model_state_dict_type'] == 'SHARDED_STATE_DICT'
            assert roundtrip['model_dtensor_entries'] > 0
            assert roundtrip['state_digest'] == initial['state_digest']
            reload_lr = update(6)
            reloaded = snapshot(directory / 'reloaded-step6', 6, False)
            left, right = read(oracle), read(reloaded)
            components = {}
            for key in ('model', 'optimizer', 'scheduler', 'rng','generation_rng'):
                a, b = state_digest(left[key]), state_digest(right[key])
                assert a == b, 'Sixth-update oracle differs: ' + key
                components[key] = dict(no_roundtrip_sha256=a, reloaded_sha256=b, exact_equal=True)
            # Every tensor/key, including every Adam moment and every local model
            # shard, contributes to the component digests. Metadata also must match.
            assert state_digest(left) == state_digest(right)
            assert identities == (id(model), id(optim), id(scheduler))
            assert calls == 28 and scheduler.last_epoch == 6 and oracle_lr == reload_lr
            assert roundtrip['optimizer_state_entries'] > 0
            adam_dtypes = sorted({str(v.dtype) for state in right['optimizer']['state'].values()
                for key, v in state.items() if key in ('exp_avg', 'exp_avg_sq') and torch.is_tensor(v)})
            assert adam_dtypes == ['torch.float32']
            storage_dtypes = sorted({str(p.dtype) for p in model.parameters()})
            buffer_dtypes = sorted({str(b.dtype) for b in model.buffers()})
            assert storage_dtypes == ['torch.float32']
            assert not nested or buffer_dtypes == ['torch.float32']
            records.append(dict(profile=name, production_like=nested and manual and not native_cpu,
                native_cpu_offload=native_cpu, manual_param_grad_optimizer_offload=manual,
                nested_fsdp_wrappers=wrapper_count, sharding='FULL_SHARD', mesh_shape=[8],
                param_compute_dtype='bfloat16' if nested else 'float32',
                reduction_dtype='float32', buffer_dtype='float32', adam_moment_dtypes=adam_dtypes,
                persistent_parameter_dtypes=storage_dtypes, actual_buffer_dtypes=buffer_dtypes,
                initial_synthetic_outer_updates=5, initial_optimizer_calls=20,
                oracle_sixth_optimizer_calls=4, reloaded_sixth_optimizer_calls=4,
                total_actual_optimizer_calls=calls, num_warmup_steps=114,
                scheduler_last_epoch=6, outer_lrs=outer_lrs, oracle_sixth_lr=oracle_lr,
                reloaded_sixth_lr=reload_lr, gradient_norms=norms,
                native_state_version=3,generation_rng_saved_and_restored=True,
                generation_rng_scope='Real CUDA RNG tensors and real production manager methods; synthetic no-vLLM-engine fixture',
                same_model_adam_scheduler_objects=True, initial_snapshot=initial,
                actual_worker_roundtrip=roundtrip, oracle_snapshot=oracle,
                reloaded_snapshot=reloaded, next_update_components=components,
                next_update_full_state_exact_equal=True))
            dist.barrier(); del worker, model, optim, scheduler, module, loaded, left, right
            torch.cuda.empty_cache()
        # Independent scalar warmup oracle, separate from model branch counts.
        p = torch.nn.Parameter(torch.ones((), device=local)); o = torch.optim.AdamW([p], lr=1e-6)
        sch = get_constant_schedule_with_warmup(o, 114); rates = []
        for _ in range(114):
            rates.append(o.param_groups[0]['lr']); p.grad = torch.zeros_like(p); o.step(); sch.step()
        assert rates[0] == 0 and abs(rates[-1] - 1e-6 * 113 / 114) < 1e-18
        assert sch.get_last_lr()[0] == 1e-6 and sch.last_epoch == 114
        out = dict(status='NATIVE_EIGHT_RANK_FULL_STATE_FIXTURE_PASS_NOT_RL_TRAINING',
            source=source, rank=rank, world_size=8, device=local, torch_version=torch.__version__,
            optional_cpu_offload_requested=args.extra_cpu_offload, records=records,
            independent_scalar_warmup_oracle=dict(optimizer_calls=114, scheduler_calls=114,
                first_lr=rates[0], last_before_lr=rates[-1], after114_lr=sch.get_last_lr()[0]),
            reviewed_sources={str(repo / p): hashlib.sha256((repo / p).read_bytes()).hexdigest()
                for p in ['verl/workers/fsdp_workers.py', 'search_r1/baselines/full_state.py',
                    'search_r1/baselines/checkpoint_gate.py', 'search_r1/baselines/gradient_norm.py',
                    'verl/workers/sharding_manager/fsdp_vllm.py']})
        (root / f'rank-{rank}.json').write_text(json.dumps(out, indent=2, allow_nan=False))
        print('E1_NATIVE_STATE_FIXTURE_JSON=' + json.dumps(out, sort_keys=True, allow_nan=False), flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == '__main__': main()
