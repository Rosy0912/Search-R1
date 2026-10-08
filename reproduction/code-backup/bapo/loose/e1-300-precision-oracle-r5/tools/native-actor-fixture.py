"""Actual eight-rank FSDP actor sequence_mean gate, never a CPU surrogate.

Launch with torchrun by the owned hardware gate. CPU self-check only verifies
the independent whole-mini reference and fixture construction; it cannot PASS
the native gate. No no_sync or manual gradient averaging is added to the actor.
"""
import argparse
import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import torch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_data(rows=128, empty_rank=False, all_empty=False):
    ids = torch.arange(rows, dtype=torch.float64)
    width = 7
    token = torch.arange(width, dtype=torch.float64)
    features = torch.stack([torch.sin(ids[:, None] * .17 + token * .31 + k * .13)
                            for k in range(16)], -1) * .4
    mask = token[None, :] < (ids.long() % 8)[:, None]
    if empty_rank:
        # For 128 global trajectories rank0 receives the first16.
        mask[:rows // 8] = False
    if all_empty:
        mask[:] = False
    old = torch.full((rows, width), -3., dtype=torch.float64)
    ref = old + torch.cos(ids[:, None] * .1 + token * .3) * .8
    advantages = torch.sin(ids[:, None] * .27 + token * .4) * 2
    # Ratio clipping must see both signs and both PPO clipping branches.
    old += torch.cos(ids[:, None] * .21 + token * .5) * .35
    return DataProto.from_dict({
        'responses': torch.zeros(rows, width, dtype=torch.long),
        'input_ids': features,
        'attention_mask': torch.ones(rows, width + 1, dtype=torch.long),
        'position_ids': torch.zeros(rows, width + 1, dtype=torch.long),
        'loss_mask': mask.double(), 'old_log_probs': old,
        'ref_log_prob': ref, 'advantages': advantages,
    }, meta_info={'temperature': 1.})


def reference_gradient(data, pg_on=True, kl_on=True):
    """Independent FP64 whole-global-mini formula, without actor helpers."""
    weight = torch.linspace(-.04, .05, 16, dtype=torch.float64, requires_grad=True)
    b = data.batch
    mask = b['loss_mask'].bool()
    current = -3 + b['input_ids'].double().matmul(weight)
    old = torch.where(mask, b['old_log_probs'], 0.)
    safe_current = torch.where(mask, current, 0.)
    adv = torch.where(mask, b['advantages'], 0.) if pg_on else torch.zeros_like(current)
    ratio = (safe_current - old).exp()
    pg = torch.maximum(-adv * ratio, -adv * ratio.clamp(.8, 1.2))
    # Stable FP64 oracle of the original saturated KL, not the implementation's
    # clamp-input trick. Inputs here are finite and moderate.
    x = torch.where(mask, b['ref_log_prob'], 0.) - safe_current
    kl = (x.exp() - x - 1).clamp(-10, 10)
    rows = torch.where(mask, pg + (.001 if kl_on else 0.) * kl, 0.).sum(-1)
    rows = rows / mask.sum(-1).clamp_min(1)
    loss = rows.mean()
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(weight.grad).all()
    return weight.grad.detach(), float(loss.detach())


def quantized_toy_reference_gradient(data, micro, pg_on=True, kl_on=True, ranks=8):
    """Independent analytic VJP of this toy's BF16 forward, FP32 reduction.

    This is NOT a production log-softmax oracle: the original toy emits BF16
    (-3 + linear), whereas actor._forward_micro_batch has a log-probability kernel.
    No actor/FSDP loss helper or autograd is used by this independent derivative.
    Whole-global-mini rows remain in the complete row denominator, even empty.
    """
    b = data.batch
    local_rows = len(data) // ranks
    assert len(data) % ranks == 0 and micro > 0
    contributions = []
    weight = torch.linspace(-.04, .05, 16, dtype=torch.float64).float().bfloat16()
    for r in range(ranks):
        total = torch.zeros(16, dtype=torch.float32)
        for start in range(r * local_rows, (r + 1) * local_rows, micro):
            sl = slice(start, min(start + micro, (r + 1) * local_rows))
            mask = b['loss_mask'][sl].bool()
            features = b['input_ids'][sl].bfloat16()
            current = (-3 + features.matmul(weight)).double()
            old = torch.where(mask, b['old_log_probs'][sl], 0.)
            safe_current = torch.where(mask, current, 0.)
            adv = torch.where(mask, b['advantages'][sl], 0.) if pg_on else torch.zeros_like(current)
            ratio = (safe_current - old).exp()
            l1, l2 = -adv * ratio, -adv * ratio.clamp(.8, 1.2)
            g1 = -adv * ratio
            g2 = g1 * ((ratio >= .8) & (ratio <= 1.2))
            pg_derivative = torch.where(l1 > l2, g1, torch.where(l1 < l2, g2, (g1 + g2) / 2))
            x = torch.where(mask, b['ref_log_prob'][sl], 0.) - safe_current
            kl_derivative = (.001 if kl_on else 0.) * (1 - x.exp())
            # Original saturated KL has zero derivative beyond upper clamp10.
            kl_derivative = torch.where((x.exp() - x - 1) <= 10, kl_derivative, 0.)
            denominator = local_rows * mask.sum(-1).clamp_min(1)[:, None]
            # Each branch reaches a BF16 current tensor separately before addition.
            pg_upstream = torch.where(mask, pg_derivative / denominator, 0.).bfloat16()
            kl_upstream = torch.where(mask, kl_derivative / denominator, 0.).bfloat16()
            upstream = pg_upstream + kl_upstream
            total += features.reshape(-1, 16).T.matmul(upstream.reshape(-1)).float()
        contributions.append(total)
    return torch.stack(contributions).mean(0).double()


def select(data, rows):
    return DataProto(batch=data.batch[rows], meta_info=dict(data.meta_info))


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 1, bias=False).double()
        self.linear.weight.data.copy_(torch.linspace(-.04, .05, 16, dtype=torch.float64).reshape(1, 16))

    def forward(self, features):
        return -3 + self.linear(features).squeeze(-1)


def run_actor(global_data, micro, offload, pg_on=True, kl_on=True, mini=16, production_precision=False):
    local_rows = len(global_data) // world
    assert len(global_data) % world == 0
    local = select(global_data, slice(rank * local_rows, (rank + 1) * local_rows))
    if not pg_on:
        local.batch['advantages'] = torch.zeros_like(local.batch['advantages'])
    # Test externally stored old/ref NaNs only on unsupervised positions.
    # The actual current forward uses only input features, never old/ref.
    for key in ('old_log_probs', 'ref_log_prob'):
        local.batch[key] = local.batch[key].clone()
        local.batch[key][local.batch['loss_mask'] == 0] = float('nan')
    precision = (MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                               buffer_dtype=torch.float32) if production_precision else None)
    base = Toy().float() if production_precision else Toy()
    model = FSDP(base, device_id=device, use_orig_params=False,
                 sharding_strategy=ShardingStrategy.FULL_SHARD,
                 mixed_precision=precision, sync_module_states=True, forward_prefetch=False)
    # Production offload is verl's outer-level manual transfer, NOT FSDP
    # CPUOffload(offload_params=True). Exercise the actual helpers and nonempty
    # AdamW state without changing parameters (lr0).
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.)
    model(torch.zeros(1, 7, 16, dtype=torch.float32 if production_precision else torch.float64,
                      device=device)).sum().backward()
    optimizer.step(); optimizer.zero_grad()
    from verl.utils.fsdp_utils import (offload_fsdp_param_and_grad, load_fsdp_param_and_grad,
                                      offload_fsdp_optimizer, load_fsdp_optimizer)
    if offload:
        offload_fsdp_param_and_grad(model, offload_grad=True)
        offload_fsdp_optimizer(optimizer)
        torch.cuda.synchronize()
        assert all(p.device.type == 'cpu' for p in model.parameters())
        assert all(value.device.type == 'cpu' for state in optimizer.state.values()
                   for value in state.values() if isinstance(value, torch.Tensor))
        load_fsdp_param_and_grad(model, device_id=device, load_grad=True)
        load_fsdp_optimizer(optimizer, device_id=device)
        torch.cuda.synchronize()
        assert all(p.device.type == 'cuda' for p in model.parameters())
    cfg = OmegaConf.create(dict(ppo_mini_batch_size=mini, ppo_micro_batch_size=micro,
        ppo_epochs=1, state_masking=True, use_kl_loss=True, use_dynamic_bsz=False,
        loss_reduction='sequence_mean', baseline_method='bapo', clip_ratio=.2,
        kl_loss_type='low_var_kl', kl_loss_coef=.001 if kl_on else 0.,
        entropy_coeff=0., grad_clip=1.e9))
    actual_gradients = []
    records = []

    assert all(value.device.type=='cpu' for value in local.batch.values() if isinstance(value,torch.Tensor))
    forward_devices=[]
    def forward(micro_batch, temperature):
        assert torch.cuda.current_device()==device.index
        assert all(value.device==device for value in micro_batch.values() if isinstance(value,torch.Tensor))
        assert all(parameter.device==device for parameter in model.parameters())
        forward_devices.append(str(device))
        current = model(micro_batch['input_ids'])
        return current * 0., current

    def step():
        # Flat one-parameter model:16 coordinates sharded equally on8 ranks.
        # Gather only for audit; actor backward/FSDP performs all averaging.
        parameters = list(model.parameters())
        assert len(parameters) == 1 and parameters[0].numel() == 2
        gradient = parameters[0].grad
        assert gradient is not None and gradient.numel() == 2
        assert torch.isfinite(gradient).all()
        parts = [torch.empty(2, dtype=torch.float64, device=device) for _ in range(world)]
        dist.all_gather(parts, gradient.detach().reshape(-1).to(device, torch.float64))
        actual_gradients.append(torch.cat(parts).cpu())
        norm = DataParallelPPOActor._optimizer_step(instance)
        assert torch.isfinite(norm)
        records.append(float(norm))
        return norm

    instance = SimpleNamespace(config=cfg, actor_module=model, actor_optimizer=optimizer,
        _forward_micro_batch=forward, _optimizer_step=step, ulysses_sequence_parallel_size=1)
    metrics = DataParallelPPOActor.update_policy(instance, local)
    steps = (local_rows + mini - 1) // mini
    assert len(actual_gradients) == steps == metrics['actor/optimizer_steps']
    assert metrics['actor/skipped_empty_minibatches'] == 0
    max_error = 0.
    max_fp64_deviation = 0.
    for step_id, actual in enumerate(actual_gradients):
        rows = [r * local_rows + i for r in range(world)
                for i in range(step_id * mini, min((step_id + 1) * mini, local_rows))]
        mathematical, _ = reference_gradient(select(global_data, rows), pg_on, kl_on)
        expected = (quantized_toy_reference_gradient(select(global_data, rows), micro, pg_on, kl_on, world)
                    if production_precision else mathematical)
        max_fp64_deviation = max(max_fp64_deviation, float((actual - mathematical).abs().max()))
        torch.testing.assert_close(actual, expected, rtol=.03 if production_precision else 2e-5,
                                   atol=2e-4 if production_precision else 2e-7)
        max_error = max(max_error, float((actual - expected).abs().max()))
    # Every rank must observe identical20-step counting in the schedule case.
    counts = torch.tensor(steps, device=device)
    minimum = counts.clone(); maximum = counts.clone()
    dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    assert minimum.item() == maximum.item() == steps
    if offload:
        offload_fsdp_param_and_grad(model, offload_grad=True)
        offload_fsdp_optimizer(optimizer)
        torch.cuda.synchronize()
        assert all(p.device.type == 'cpu' for p in model.parameters())
        assert all(value.device.type == 'cpu' for state in optimizer.state.values()
                   for value in state.values() if isinstance(value, torch.Tensor))
    result = dict(micro_per_rank=micro, mini_per_rank=mini, manual_outer_offload=offload,
                  native_cpu_offload=False, production_bf16_fp32=production_precision,
                  pg_on=pg_on, kl_on=kl_on, global_rows=len(global_data),
                  optimizer_steps=steps, max_gradient_error=max_error,
                  precision_oracle='ANALYTIC_BF16_TOY_VJP_FP32_REDUCTION' if production_precision else 'FP64_MATHEMATICAL',
                  max_fp64_approximation_deviation=max_fp64_deviation,
                  toy_output_dtype='torch.bfloat16' if production_precision else 'torch.float64',
                  input_device_before_update='cpu', forward_device_checks=len(forward_devices),
                  actual_forward_devices=sorted(set(forward_devices)),
                  gradient_norm_min=min(records), gradient_norm_max=max(records),
                  gradients=[g.tolist() for g in actual_gradients])
    del instance, model, optimizer
    gc.collect(); torch.cuda.empty_cache(); dist.barrier()
    return result


def production_forward_precision_probe(repo, output):
    """Real actor autocast/logprob kernel, separate from the BF16-offset toy."""
    import inspect
    from transformers.modeling_outputs import CausalLMOutputWithPast
    import verl.utils.torch_functional as functional
    import torch.distributed.utils as distributed_utils
    witness_tensor = torch.tensor([1.], requires_grad=True)
    namespace_visits, hf_visits = [], []
    distributed_utils._apply_to_tensors(lambda t: namespace_visits.append(t) or t, SimpleNamespace(logits=witness_tensor))
    distributed_utils._apply_to_tensors(lambda t: hf_visits.append(t) or t, CausalLMOutputWithPast(logits=witness_tensor))
    assert namespace_visits == [] and len(hf_visits) == 1 and hf_visits[0] is witness_tensor
    traversal_path = Path(distributed_utils.__file__).resolve()
    assert digest(traversal_path) == '33b72b820b44a3b96d89a386737f753d54a96339368aa18efaf09115f207a4db'
    owned_traversal = output / ('production-output-traversal-source-rank-' + str(rank) + '.py')
    with owned_traversal.open('xb') as f: f.write(traversal_path.read_bytes())
    output_traversal = dict(torch=torch.__version__, namespace_tensor_visits=0, HF_tensor_visits=1,
        output_class='transformers.modeling_outputs.CausalLMOutputWithPast',
        utils_file=str(traversal_path), utils_sha256=digest(traversal_path),
        owned_utils_source_path=str(owned_traversal))
    class TinyCausalLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(16, 1, bias=False)
            self.linear.weight.data.copy_(torch.linspace(-.04, .05, 16).reshape(1, 16))
            self.observed = []
        def forward(self, input_ids, attention_mask, position_ids, use_cache=False):
            features = torch.sin(input_ids.float().unsqueeze(-1) * .17 + torch.arange(16, device=input_ids.device) * .13) * .4
            linear = self.linear(features)
            vocab = torch.linspace(-1., 1., 8, device=input_ids.device, dtype=linear.dtype)
            bias = torch.linspace(-.2, .3, 8, device=input_ids.device, dtype=linear.dtype)
            logits = linear * vocab + bias
            self.observed.append(logits.detach())
            return CausalLMOutputWithPast(logits=logits)
    base = TinyCausalLM()
    precision = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32)
    model = FSDP(base, device_id=device, use_orig_params=False,
                 sharding_strategy=ShardingStrategy.FULL_SHARD,
                 mixed_precision=precision, sync_module_states=True, forward_prefetch=False)
    input_ids = torch.arange(16).reshape(2, 8) % 8
    cpu_data = DataProto.from_dict({'input_ids': input_ids,
        'responses': input_ids[:, -7:].clone(),
        'attention_mask': torch.ones(2, 8, dtype=torch.long),
        'position_ids': torch.arange(8).expand(2, 8)}, meta_info={'temperature': 1.})
    assert all(t.device.type == 'cpu' for t in cpu_data.batch.values())
    assert torch.cuda.current_device() == device.index
    batch = cpu_data.batch.cuda(device=torch.cuda.current_device())
    assert all(t.device == device for t in batch.values())
    assert all(p.device == device for p in model.parameters())
    config_path = repo / 'verl/trainer/config/ppo_bapo_e1_300.yaml'
    production_cfg = OmegaConf.load(config_path)
    assert production_cfg.actor_rollout_ref.model.use_remove_padding is True
    branch_records = []
    for remove_padding in (False, True):
        base.observed.clear(); model.zero_grad()
        config = OmegaConf.create({'use_remove_padding': remove_padding, 'ulysses_sequence_parallel_size': 1})
        # Constructor uses the same torch.compile entropy path as actual worker.
        instance = DataParallelPPOActor(config, model)
        entropy, logprob = instance._forward_micro_batch(batch, 1.)
        assert len(base.observed) == 1
        observed = base.observed[0]
        logits = (observed.reshape(2, 8, 8) if remove_padding else observed)[:, :-1]
        assert logits.dtype == torch.bfloat16 and torch.isfinite(logits).all()
        reference = torch.log_softmax(logits.double(), dim=-1).gather(-1, batch['responses'].unsqueeze(-1)).squeeze(-1)
        assert torch.isfinite(entropy).all() and torch.isfinite(logprob).all()
        torch.testing.assert_close(logprob.double(), reference, rtol=.03, atol=2e-4)
        (-logprob.mean()).backward()
        gradients = [p.grad for p in model.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert any(torch.count_nonzero(g) > 0 for g in gradients)
        branch_records.append(dict(use_remove_padding=remove_padding, use_ulysses_sp=False,
            ulysses_sequence_parallel_size=1, production_branch=remove_padding,
            logits_dtype=str(logits.dtype), logprob_dtype=str(logprob.dtype), entropy_dtype=str(entropy.dtype),
            gradient_dtypes=sorted(set(str(g.dtype) for g in gradients)),
            finite_forward_backward=True, nonzero_gradient=True,
            max_logprob_fp64_error=float((logprob.double()-reference).abs().max()),
            input_device_before_forward='cpu', actual_device=str(device),
            actual_parameter_devices=sorted(set(str(p.device) for p in model.parameters())),
            actual_tensor_devices=sorted(set(str(t.device) for t in batch.values()))))
    kernel = functional.cross_entropy_loss if functional.FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE else functional.logprobs_from_logits_naive
    kernel_file = Path(inspect.getsourcefile(kernel)).resolve()
    kernel_source = kernel_file.read_bytes()
    owned_kernel = output / ('production-logprob-kernel-source-rank-' + str(rank) + '.py')
    owned_kernel.write_bytes(kernel_source)
    record = dict(status='PASS_REAL_ACTOR_FORWARD_PRECISION_PROBE',rank=rank, world=world,
        source_commit=os.environ['E1_SOURCE_COMMIT'], actor_source_sha256=digest(repo/'verl/workers/actor/dp_actor.py'),
        fixture_sha256=digest(__file__), input_device_before_forward='cpu', device=str(device),
        config_path=str(config_path), config_sha256=digest(config_path), branch_records=branch_records,
        actual_output_traversal=output_traversal,
        kernel_branch='flash_cross_entropy' if functional.FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE else 'naive_log_softmax',
        kernel_module=kernel.__module__, kernel_source_path=str(kernel_file),
        kernel_source_sha256=hashlib.sha256(kernel_source).hexdigest(),
        kernel_owned_source_path=str(owned_kernel), torch=torch.__version__, sys_prefix=sys.prefix)
    (output / ('production-forward-rank-' + str(rank) + '.json')).write_text(json.dumps(record, indent=2, allow_nan=False))
    del instance, model
    gc.collect(); torch.cuda.empty_cache(); dist.barrier()
    return record


def main():
    global torch, dist, DataProto, FSDP, MixedPrecision, ShardingStrategy, OmegaConf, DataParallelPPOActor
    global rank, world, device
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--cpu-self-check', action='store_true')
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))
    import torch as torch_module
    import torch.distributed as distributed
    torch, dist = torch_module, distributed
    from verl import DataProto as proto
    from omegaconf import OmegaConf as config
    from verl.workers.actor.dp_actor import DataParallelPPOActor as actor
    from torch.distributed.fsdp import FullyShardedDataParallel, MixedPrecision as precision, ShardingStrategy as strategy
    DataProto, OmegaConf, DataParallelPPOActor = proto, config, actor
    FSDP, MixedPrecision, ShardingStrategy = FullyShardedDataParallel, precision, strategy
    source = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    files = ['verl/workers/actor/dp_actor.py', 'search_r1/baselines/objectives.py',
             'search_r1/baselines/gradient_norm.py', 'verl/trainer/ppo/core_algos.py',
             'verl/utils/fsdp_utils.py', 'verl/workers/fsdp_workers.py',
             'verl/protocol.py', 'verl/utils/torch_functional.py',
             'search_r1/baselines/distillation.py']
    bindings = {name: digest(repo / name) for name in files}
    assert Path(sys.modules[DataParallelPPOActor.__module__].__file__).resolve() == repo / files[0]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.cpu_self_check:
        results = []
        for empty in (False, True):
            for pg_on, kl_on in ((True, False), (False, True), (True, True)):
                gradient, loss = reference_gradient(make_data(all_empty=empty), pg_on, kl_on)
                results.append(dict(empty=empty, pg_on=pg_on, kl_on=kl_on, loss=loss,
                                    gradient_norm=float(gradient.norm())))
        b = make_data().batch
        current = -3 + b['input_ids'].matmul(torch.linspace(-.04, .05, 16, dtype=torch.float64))
        ratio = (current - b['old_log_probs']).exp()
        valid = b['loss_mask'].bool()
        branches = {'positive_advantage_upper_clip': int(((b['advantages'] > 0) & (ratio > 1.2) & valid).sum()),
                    'negative_advantage_lower_clip': int(((b['advantages'] < 0) & (ratio < .8) & valid).sum()),
                    'ordinary_ratio': int(((ratio >= .8) & (ratio <= 1.2) & valid).sum())}
        assert all(value > 0 for value in branches.values())
        (output / 'cpu-self-check.json').write_text(json.dumps(dict(
            status='PASS_CPU_REFERENCE_ONLY_NOT_NATIVE_ACCEPTANCE', records=results,
            ppo_branches=branches,
            source_commit=source, source_files_sha256=bindings,
            fixture_sha256=digest(__file__)), indent=2))
        return
    expected = os.environ['E1_SOURCE_COMMIT']
    assert len(expected) == 40 and source == expected
    assert repo == Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-precision-r5-20261004/repo')
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip()
    assert torch.__version__.startswith('2.4.0')
    rank = int(os.environ['RANK']); device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    assert output.resolve().is_relative_to(repo.parent / 'runs')
    assert not (output / f'rank-{rank}.json').exists()
    torch.cuda.set_device(device)
    dist.init_process_group('nccl', timeout=datetime.timedelta(seconds=180))
    world = dist.get_world_size()
    assert world == 8
    records = []
    for cpu_offload in (False, True):
        for empty_rank in (False, True):
            for pg_on, kl_on in ((True, False), (False, True), (True, True)):
                baseline = None
                for micro in (1, 2, 4, 16):
                    result = run_actor(make_data(empty_rank=empty_rank), micro, cpu_offload, pg_on, kl_on)
                    if baseline is None:
                        baseline = torch.tensor(result['gradients'], dtype=torch.float64)
                    torch.testing.assert_close(torch.tensor(result['gradients'], dtype=torch.float64),
                                               baseline, rtol=2e-5, atol=2e-7)
                    records.append(dict(case='actual_actor_micro_invariance', empty_rank=empty_rank, **result))
        records.append(dict(case='actual_actor_global_empty_step',
                            **run_actor(make_data(all_empty=True), 2, cpu_offload)))
        schedule = run_actor(make_data(rows=2560), 2, cpu_offload)
        assert schedule['optimizer_steps'] == 20
        records.append(dict(case='actual_actor_2560_rows_20_updates', **schedule))
    assert len(records) == 52
    # Actual worker's BF16 parameter/FP32 gradient-reduction configuration.
    # Compare to independent FP64 reference with stated mixed precision bounds.
    for manual_offload in (False, True):
        baseline = None
        for micro in (1, 2, 4, 16):
            result = run_actor(make_data(), micro, manual_offload, production_precision=True)
            current = torch.tensor(result['gradients'], dtype=torch.float64)
            if baseline is None:
                baseline = current
            torch.testing.assert_close(current, baseline, rtol=.03, atol=2e-4)
            records.append(dict(case='actual_actor_production_bf16_fp32_micro_invariance', **result))
        schedule = run_actor(make_data(rows=2560), 2, manual_offload, production_precision=True)
        assert schedule['optimizer_steps'] == 20
        records.append(dict(case='actual_actor_production_bf16_20_updates', **schedule))
    assert len(records) == 62
    # Persist successful actor components before the distinct production probe.
    # This can never substitute for the final rankJSON or full hardware receipt.
    component = dict(status='PASS_COMPONENT_ACTOR_62_CASES_ONLY_NOT_FULL_HARDWARE',
        scope='COMPONENT_ONLY_PRODUCTION_FORWARD_PROBE_NOT_YET_RUN',
        rank=rank, world=world, torch=torch.__version__, sys_prefix=sys.prefix,
        source_commit=source, repo=str(repo), actor_file=sys.modules[actor.__module__].__file__,
        source_files_sha256=bindings, fixture_sha256=digest(__file__), records=records)
    with (output / f'component-only-rank-{rank}.json').open('x') as f:
        json.dump(component, f, indent=2, allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    production_forward_record = production_forward_precision_probe(repo, output)
    (output / f'rank-{rank}.json').write_text(json.dumps(dict(status='PASS_NATIVE_ACTOR_FSDP8',
        rank=rank, world=world, torch=torch.__version__, sys_prefix=sys.prefix,
        source_commit=source, repo=str(repo), actor_file=sys.modules[actor.__module__].__file__,
        source_files_sha256=bindings, fixture_sha256=digest(__file__), records=records,
        production_forward_precision=production_forward_record), indent=2,
        allow_nan=False))
    dist.barrier(); dist.destroy_process_group()
    print('REAL_SEQUENCE_MEAN_ACTOR_FSDP8_PASS', rank, flush=True)


if __name__ == '__main__':
    main()
