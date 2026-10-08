"""UNRUN paired short-width 3B gate; no rollout/training acceptance.

Each case runs one torchrun8 group and one persistent FSDP object.
Reference writes complete shards; after exact full reset candidate compares them.
Only owned Slurm/cgroup + inherited leases and pinned external proposal allowed.
"""
import argparse
from datetime import timedelta
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import traceback

HERE = Path(__file__).resolve().parent
SOURCE = 'e43fbfa73088ed70ed3efef79ba561507c0afc33'
EXPECTED_PARAMETERS = 3_085_938_688


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_external(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pinned_files():
    assert sha(HERE / 'paired-gate-manifest.json') == os.environ['OPSD_PROPOSAL_MANIFEST_SHA256']
    manifest = json.loads((HERE / 'paired-gate-manifest.json').read_text())
    assert manifest['source_commit'] == SOURCE
    for filename, expected in manifest['sha256'].items():
        assert sha(HERE / filename) == expected, filename
    repo = Path(os.environ['SEARCH_R1_REPO'])
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip() == SOURCE
    assert sha(repo / 'search_r1/baselines/distillation.py') == sha(HERE / 'baseline_distillation_e43.py')
    # Require a clean production source, excluding untracked reports.
    assert not subprocess.check_output(['git', 'diff', '--name-only', 'HEAD'], cwd=repo, text=True).strip()
    return manifest


def compare_tensor(expected, actual):
    """All elements checked; no hash/sample used as a substitute for comparison."""
    import torch
    if expected is None or actual is None:
        assert expected is None and actual is None
        return dict(elements=0, max_abs=0., l2=0.)
    assert expected.shape == actual.shape and expected.dtype == actual.dtype
    actual = actual.detach().cpu().contiguous()
    assert torch.isfinite(expected).all() and torch.isfinite(actual).all()
    maximum, squared = 0., 0.
    for left, right in zip(expected.flatten().split(1_048_576), actual.flatten().split(1_048_576)):
        difference = left.double() - right.double()
        maximum = max(maximum, float(difference.abs().max()) if difference.numel() else 0.)
        squared += float(difference.square().sum())
    assert torch.equal(expected, actual), f'Exact tensor comparison failed max={maximum}, L2={squared**.5}'
    return dict(elements=expected.numel(), max_abs=maximum, l2=squared**.5)


def cpu_payload(value):
    import torch
    if torch.is_tensor(value):
        return value.detach().cpu().contiguous().clone()
    if isinstance(value, dict):
        return {key: cpu_payload(item) for key, item in value.items()}
    return value


def snapshot(variant, directory, label, payload):
    import torch
    path = directory / (label + '.pt')
    if variant == 'baseline':
        assert not path.exists()
        torch.save(cpu_payload(payload), path)
        return dict(label=label, comparison_variant=variant, baseline_snapshot_sha256=sha(path), elements=0)
    expected = torch.load(path, map_location='cpu', weights_only=True)
    assert expected.keys() == payload.keys()
    result = dict(label=label, comparison_variant=variant, baseline_snapshot_sha256=sha(path), elements=0, max_abs=0., l2=0.)
    for key, value in payload.items():
        if torch.is_tensor(value) or value is None:
            stats = compare_tensor(expected[key], value)
            result['elements'] += stats['elements']
            result['max_abs'] = max(result['max_abs'], stats['max_abs'])
            result['l2'] += stats['l2'] ** 2
        else:
            assert expected[key] == value, key
    result['l2'] **= .5
    return result


TEACHER_COUNTERS = ('scored_tokens', 'wall_seconds', 'cpu_seconds', 'cuda_seconds',
                   'transfer_wall_seconds', 'cuda_update_entries')


def tensor_digest(tensor):
    import torch
    value = tensor.detach().cpu().contiguous().flatten()
    digest = hashlib.sha256()
    for piece in value.split(1_048_576):
        digest.update(piece.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def teacher_parameter_digest(teacher):
    return {name: tensor_digest(param) for name, param in teacher.model.named_parameters()}


def capture_rng(device):
    import random
    import numpy as np
    import torch
    return dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state().clone(),
                cuda=torch.cuda.get_rng_state(device).clone() if device.type == 'cuda' else None)


def restore_rng(state, device):
    import random
    import numpy as np
    import torch
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'], device)
    current = capture_rng(device)
    assert current['python'] == state['python']
    assert current['numpy'][0] == state['numpy'][0] and np.array_equal(current['numpy'][1], state['numpy'][1])
    assert current['numpy'][2:] == state['numpy'][2:]
    compare_tensor(state['cpu'], current['cpu'])
    compare_tensor(state['cuda'], current['cuda'])
    return dict(python=True, numpy=True, torch_cpu=True, torch_local_cuda=state['cuda'] is not None)


def saved_buffers(module):
    return {name: cpu_payload(value) for name, value in module.named_buffers()}


def restore_buffers(module, expected):
    import torch
    current = dict(module.named_buffers())
    assert current.keys() == expected.keys()
    results = {}
    with torch.no_grad():
        for name, buffer in current.items():
            assert buffer.shape == expected[name].shape and buffer.dtype == expected[name].dtype
            buffer.copy_(expected[name].to(buffer.device))
            results[name] = compare_tensor(expected[name], buffer)
    return results


def restore_initial(named, directory, model, student_buffers, teacher, teacher_buffers,
                    teacher_counters, teacher_digest, rng, device, decoder_counts, model_modes):
    """Every master element including padding and all buffers restored, no sample."""
    import torch
    assert str(teacher.device) == 'cpu' and not teacher._cuda_events
    assert teacher_parameter_digest(teacher) == teacher_digest
    restored = 0
    identities = [(name, id(param)) for name, param in named]
    with torch.no_grad():
        for index, (name, param) in enumerate(named):
            saved = torch.load(directory / f'initial-{index}.pt', map_location='cpu', weights_only=True)
            assert saved['name'] == name and saved['master'].shape == param.shape
            assert saved['master'].dtype == param.dtype
            param.copy_(saved['master'].to(param.device))
            param.grad = None
            restored += compare_tensor(saved['master'], param)['elements']
    assert identities == [(name, id(param)) for name, param in model.named_parameters()]
    assert all(param.grad is None for _, param in named)
    student_result = restore_buffers(model, student_buffers)
    teacher_result = restore_buffers(teacher.model, teacher_buffers)
    for key, value in teacher_counters.items():
        setattr(teacher, key, value)
    assert all(getattr(teacher, key) == value for key, value in teacher_counters.items())
    assert teacher._cuda_events == []
    assert all(not p.requires_grad and p.grad is None for p in teacher.model.parameters())
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()
    teacher.model.eval()
    assert {name: child.training for name, child in model.named_modules()} == model_modes
    assert all(not child.training for child in teacher.model.modules())
    for key in decoder_counts:
        decoder_counts[key] = 0
    rng_result = restore_rng(rng, device)
    return dict(master_elements_including_padding=restored, exact_all_master=True,
                parameter_identity_preserved=True, every_gradient_none=True,
                student_buffers=student_result, teacher_buffers=teacher_result,
                teacher_parameters_unchanged=True, teacher_counters_restored=dict(teacher_counters),
                no_pending_teacher_cuda_events=True, decoder_counts=dict(decoder_counts),
                modes_preserved=True, rng_restored=rng_result)


def observed_clip(model, named, record):
    """Observe original set order and norm scalars; never reorder/replace clipping."""
    import torch
    from unittest.mock import patch
    module = importlib.import_module('torch.distributed.fsdp.fully_sharded_data_parallel')
    original = module._get_grad_norm
    names = {id(param): name for name, param in named}
    def observe(params, norm_type):
        ordered = list(params)
        identifiers = [id(param) for param in ordered]
        observed = [param for param in ordered if param.grad is not None]
        with torch.no_grad():
            scalars = [torch.linalg.vector_norm(param.grad.detach(), norm_type, dtype=torch.float32)
                       for param in observed]
            scalar_values = torch.stack(scalars).detach().cpu().tolist() if scalars else []
        result = original(params, norm_type)  # The original iterable/order, once.
        assert identifiers == [id(param) for param in params]
        record.append(dict(param_iteration_order=[names[id(param)] for param in ordered],
                           parameter_object_ids=identifiers, norm_type=norm_type,
                           with_gradient_order=[names[id(param)] for param in observed],
                           per_flat_gradient_norms_fp32=scalar_values,
                           original_local_norm_fp32=float(result.detach().cpu())))
        return result
    with patch.object(module, '_get_grad_norm', side_effect=observe):
        norm = model.clip_grad_norm_(1.)
    assert module._get_grad_norm is original
    return norm

def main():
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import transformers
    from types import SimpleNamespace
    from verl.utils.fsdp_utils import get_fsdp_wrap_policy
    from search_r1.baselines.distillation import global_token_count
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=['control', 'nonempty', 'rank0-empty', 'global-empty'], required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    manifest = pinned_files()
    assert torch.__version__.split('+')[0] == '2.4.0' and transformers.__version__ == '4.46.3'
    support = load_external('proposal_support', HERE / 'microbatch_probe_support.py')
    scope = support.allocation_scope()
    rank, local_rank, world = (int(os.environ[key]) for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE'))
    assert world == 8 and rank == local_rank
    # Torchrun supervisor holds inherited lease FDs across all workers. Elastic
    # workers may close inherited FDs; job cgroup and eight-lease metadata are
    # checked here, full FD/PID/memory ownership is checked before torchrun.
    records = json.loads(os.environ['SEARCH_R1_GPU_LEASES'])
    assert len(records) == 8
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    snapshots = out / ('paired-snapshots-' + args.case) / str(rank)
    snapshots.mkdir(parents=True, exist_ok=True)
    report = dict(passed=False, case=args.case, rank=rank,
                  allocation=scope, source_commit=SOURCE, proposal_sha256=manifest['sha256']['proposal_distillation_a.py'],
                  input_width=128, response_width=96, local_micro=2, legs=[], comparisons=[],
                  torch_version=torch.__version__, transformers_version=transformers.__version__,
                  production_outer_iterations=0, worst_width_capacity_tested=False, vllm_coexistence_tested=False)
    try:
        torch.set_num_threads(4)
        torch.cuda.set_device(local_rank)
        assert os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '') == ''
        assert torch.cuda.get_allocator_backend() == 'native'
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        device = torch.device('cuda', local_rank)
        free, total_memory = torch.cuda.mem_get_info()
        external_usage = total_memory - free
        dist.init_process_group('nccl', timeout=timedelta(minutes=15))
        tok = AutoTokenizer.from_pretrained(args.model)
        base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32,
                                                   attn_implementation='flash_attention_2')
        parameter_count = sum(parameter.numel() for parameter in base.parameters())
        assert parameter_count == EXPECTED_PARAMETERS and base.config.num_hidden_layers == 36
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model = FSDP(base, device_id=local_rank, use_orig_params=False,
                     auto_wrap_policy=get_fsdp_wrap_policy(base), sharding_strategy=ShardingStrategy.FULL_SHARD,
                     mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                                    buffer_dtype=torch.float32), sync_module_states=True,
                     forward_prefetch=False)
        flags, decoder_counts, hooks = support.audited_checkpoint_training(base, model)
        assert flags['decoder_count'] == 36
        report['student_runtime'] = flags
        named = list(model.named_parameters())
        assert all(getattr(param, '_is_flat_param', False) and param.dtype == torch.float32 for _, param in named)
        # PyTorch2.4 FlatParameter metadata distinguishes shard-padding from
        # actual model elements; sum of valid shards must cover ALL 3.085B.
        valid_elements = sum(param.numel() - param._shard_numel_padded for _, param in named)
        unsharded_count = sum(param._unpadded_unsharded_size.numel() for _, param in named)
        assert unsharded_count == parameter_count
        coverage = torch.tensor(valid_elements, device=device, dtype=torch.int64)
        dist.all_reduce(coverage)
        assert int(coverage) == parameter_count
        report.update(full_parameter_count=parameter_count, valid_local_shard_elements=valid_elements,
                      global_covered_parameter_elements=int(coverage), flat_shards=len(named))
        baseline_module = load_external('search_r1.baselines.paired_baseline', HERE / 'baseline_distillation_e43.py')
        candidate_module = load_external('search_r1.baselines.paired_a', HERE / 'proposal_distillation_a.py')
        teacher = baseline_module.FrozenTeacher(args.model, tok, 'cuda:' + str(local_rank))
        head_calls, teacher_base_calls = [], []
        hooks += [teacher.model.get_output_embeddings().register_forward_pre_hook(lambda *_: head_calls.append(1)),
                  teacher.model.base_model.register_forward_pre_hook(lambda *_: teacher_base_calls.append(1))]
        pieces = [v for v in tok.encode('Question France capital Paris answer search information', add_special_tokens=False)
                  if v != tok.pad_token_id]
        assert pieces
        sequence = (pieces * ((128 + len(pieces) - 1) // len(pieces)))[:128]
        ids = torch.tensor(sequence, device=device)[None].repeat(2, 1)
        mask = torch.ones((2, 96), device=device)
        mask[:, 16:32] = 0
        mask[:, 64:] = 0
        mask[0, 32:48] = 0  # Mixed rows in a retained chunk.
        if args.case == 'global-empty' or (args.case == 'rank0-empty' and rank == 0):
            mask.zero_()
        attention = torch.ones_like(ids)  # Unsupervised observations stay context.
        tensors = dict(prompts=ids[:, :32], responses=ids[:, 32:], input_ids=ids,
                       attention_mask=attention, position_ids=attention.cumsum(-1) - 1, loss_mask=mask)
        settings = SimpleNamespace(chunk_size=16, beta=0., temperature=1., component_clip=.05,
                                   max_student_logits_gib=64., loss_checkpointing=True)
        parameter_identities = [(name, id(param)) for name, param in named]
        report['fsdp_object_id'] = id(model)
        import inspect
        report['original_fsdp_clip_source_sha256'] = hashlib.sha256(inspect.getsource(model.clip_grad_norm_).encode()).hexdigest()
        report['parameter_identities'] = parameter_identities
        initial_modes = {name: child.training for name, child in model.named_modules()}
        initial_student_buffers = saved_buffers(model)
        initial_teacher_buffers = saved_buffers(teacher.model)
        initial_teacher_digest = teacher_parameter_digest(teacher)
        initial_teacher_counters = {key: getattr(teacher, key) for key in TEACHER_COUNTERS}
        initial_rng = capture_rng(device)
        torch.save(dict(student=initial_student_buffers, teacher=initial_teacher_buffers,
                        torch_cpu_rng=initial_rng['cpu'], torch_local_cuda_rng=initial_rng['cuda']),
                   snapshots / 'initial-buffers-rng.pt')
        (snapshots / 'initial-python-numpy-rng.json').write_text(json.dumps(dict(
            python=initial_rng['python'], numpy_name=initial_rng['numpy'][0],
            numpy_state=initial_rng['numpy'][1].tolist(), numpy_tail=initial_rng['numpy'][2:])))
        report['initial_buffer_rng_snapshot_sha256'] = sha(snapshots / 'initial-buffers-rng.pt')
        report['initial_teacher_parameter_digest'] = initial_teacher_digest
        for index, (name, param) in enumerate(named):
            snapshot('baseline', snapshots, f'initial-{index}', dict(name=name, master=param))
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        # The exact original FSDP object and parameter identities remain alive
        # across both legs, including the e43/e43 control. Never replace its norm.
        for leg_index, leg_name in enumerate(('reference', 'candidate')):
            variant = 'baseline' if leg_index == 0 else 'proposal'
            if leg_index:
                optimizer.zero_grad(set_to_none=True)
                optimizer.state.clear()
                del optimizer
                report['restore'] = restore_initial(named, snapshots, model, initial_student_buffers,
                    teacher, initial_teacher_buffers, initial_teacher_counters, initial_teacher_digest,
                    initial_rng, device, decoder_counts, initial_modes)
                head_calls.clear()
                teacher_base_calls.clear()
            else:
                report['initial_rng_verified'] = restore_rng(initial_rng, device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=0.)
            assert not optimizer.state and all(p.grad is None for _, p in named)
            assert parameter_identities == [(name, id(p)) for name, p in model.named_parameters()]
            assert base.model.gradient_checkpointing and base.model._gradient_checkpointing_func.keywords.get('use_reentrant') is False
            assert {name: child.training for name, child in model.named_modules()} == initial_modes
            leg = dict(name=leg_name, loss_implementation='e43' if leg_index == 0 or args.case == 'control' else 'A',
                       optimizer_initial_state_empty=True, steps=[], norm_observations=[],
                       parameter_identities_unchanged=True, fsdp_object_id=id(model))
            report['legs'].append(leg)
            module = baseline_module if leg_index == 0 or args.case == 'control' else candidate_module
            with teacher.on_device():
                leg['teacher_active'] = support.teacher_runtime_audit(teacher, device)
                assert teacher.cuda_update_entries == initial_teacher_counters['cuda_update_entries'] + 1
                for step in (1, 2):
                    total, world = global_token_count(mask)
                    if total.item() == 0:
                        assert args.case == 'global-empty' and not optimizer.state
                        assert not any(decoder_counts.values()) and not head_calls and not teacher_base_calls
                        leg['steps'].append(dict(step=step, globally_empty_skipped=True, optimizer_steps=0))
                        for index, (name, param) in enumerate(named):
                            report['comparisons'].append(snapshot(variant, snapshots, f'step{step}-empty-{index}', dict(name=name, master=param)))
                        continue
                    optimizer.zero_grad(set_to_none=True)
                    loss = module.opsd_loss(model, teacher, tensors, tok, [['Paris'], ['Paris']], settings, total / world)
                    assert all(count == 2 * step - 1 for count in decoder_counts.values())
                    assert torch.isfinite(loss)
                    report['comparisons'].append(snapshot(variant, snapshots, f'step{step}-loss', dict(loss=loss)))
                    loss.backward()
                    assert all(count == 2 * step for count in decoder_counts.values())
                    assert all(p.grad is not None and torch.isfinite(p.grad).all() for _, p in named)
                    raw_covered = 0
                    for index, (name, param) in enumerate(named):
                        raw_covered += param.grad.numel() - param._shard_numel_padded
                        report['comparisons'].append(snapshot(variant, snapshots, f'step{step}-rawgrad-{index}', dict(name=name, grad=param.grad)))
                    assert raw_covered == valid_elements
                    observation = []
                    norm = observed_clip(model, named, observation)
                    assert torch.isfinite(norm) and norm > 0
                    leg['norm_observations'].append(dict(step=step, local_reductions=observation,
                                                        original_global_norm=float(norm.detach().cpu())))
                    if leg_index:
                        reference = report['legs'][0]['norm_observations'][step - 1]
                        assert observation == reference['local_reductions']
                    report['comparisons'].append(snapshot(variant, snapshots, f'step{step}-norm', dict(norm=norm)))
                    for index, (name, param) in enumerate(named):
                        report['comparisons'].append(snapshot(variant, snapshots, f'step{step}-clipgrad-{index}', dict(name=name, grad=param.grad)))
                    tracked = next(p for _, p in named if p.grad.abs().max() > 0)
                    tracked_index = int(tracked.grad.flatten().abs().argmax())
                    before = tracked.detach().flatten()[tracked_index].clone()
                    optimizer.step()
                    assert float((tracked.detach().flatten()[tracked_index] - before).abs()) > 0
                    for index, (name, param) in enumerate(named):
                        state = optimizer.state[param]
                        assert set(state) == {'step', 'exp_avg', 'exp_avg_sq'} and int(state['step']) == step
                        report['comparisons'].append(snapshot(variant, snapshots, f'step{step}-postadam-{index}',
                            dict(name=name, master=param, step=state['step'], exp_avg=state['exp_avg'], exp_avg_sq=state['exp_avg_sq'])))
                    del state
                    leg['steps'].append(dict(step=step, loss=float(loss.detach()), grad_norm=float(norm),
                        local_raw_gradient_elements=raw_covered, decoder_calls=dict(decoder_counts), optimizer_steps=step))
                assert all(p.grad is None for p in teacher.model.parameters())
            leg['teacher_returned'] = support.teacher_runtime_audit(teacher, 'cpu')
            leg['teacher_audit'] = teacher.audit_metrics(optimizer)
            assert teacher_parameter_digest(teacher) == initial_teacher_digest
            expected_bases = 0 if args.case == 'global-empty' else 2
            expected_heads = (0 if args.case == 'global-empty' else
                              12 if module is baseline_module else
                              0 if args.case == 'rank0-empty' and rank == 0 else 6)
            assert len(teacher_base_calls) == expected_bases and len(head_calls) == expected_heads
            leg.update(teacher_head_calls=len(head_calls), teacher_base_calls=len(teacher_base_calls),
                       teacher_scored_tokens=teacher.scored_tokens, decoder_counts=dict(decoder_counts),
                       teacher_cuda_update_entries=teacher.cuda_update_entries,
                       teacher_parameters_unchanged=True)
            report['comparisons'].append(snapshot(variant, snapshots, 'teacher-scored', dict(scored_tokens=teacher.scored_tokens)))
            for category, saved in (('student-buffer', saved_buffers(model)), ('teacher-buffer', saved_buffers(teacher.model))):
                for index, (name, value) in enumerate(saved.items()):
                    report['comparisons'].append(snapshot(variant, snapshots, f'{category}-{index}', dict(name=name, buffer=value)))
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_reserved()
        report.update(peak_reserved_bytes=peak, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      external_baseline_used_bytes=external_usage, gpu_total_bytes=total_memory,
                      estimated_peak_device_used_bytes=external_usage + peak + 1024**3,
                      minimum_required_free_fraction=.15)
        assert external_usage + peak + 1024**3 <= .85 * total_memory
        for hook in hooks:
            hook.remove()
        report['passed'] = True
        gathered = [None] * world
        dist.all_gather_object(gathered, report)
        assert len(gathered) == 8 and all(item['passed'] for item in gathered)
        assert sum(item['valid_local_shard_elements'] for item in gathered) == parameter_count
        if rank == 0:
            (out / f'{args.case}-paired.json').write_text(json.dumps(dict(passed=True, ranks=gathered,
                scope='Unapplied paired same-FSDP-identity/processgroup e43/e43 control or e43/A exact full-shard state comparison on short128/96 contexts; native FSDP clipping unchanged; not production, vLLM or worst-width capacity acceptance; previous2987 remains FAILED'), indent=2))
        dist.barrier()
        dist.destroy_process_group()
    except BaseException as error:
        report.update(error=repr(error), traceback=traceback.format_exc())
        (out / f'{args.case}-paired-rank{rank}-failure.json').write_text(json.dumps(report, indent=2))
        raise  # torchrun must terminate divergent peers, never retry this group.


if __name__ == '__main__':
    main()
