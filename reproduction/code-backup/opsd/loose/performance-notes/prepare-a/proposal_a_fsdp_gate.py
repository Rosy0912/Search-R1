"""UNRUN proposal-only short-width 3B gate; no rollout/training acceptance.

Each variant runs in its own torchrun8 group, with a fresh identical model.
Baseline writes complete local shards; proposal compares every element exactly.
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
    assert sha(HERE / 'gpu-gate-manifest.json') == os.environ['OPSD_PROPOSAL_MANIFEST_SHA256']
    manifest = json.loads((HERE / 'gpu-gate-manifest.json').read_text())
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
        return dict(label=label, baseline_snapshot_sha256=sha(path), elements=0)
    expected = torch.load(path, map_location='cpu', weights_only=True)
    assert expected.keys() == payload.keys()
    result = dict(label=label, baseline_snapshot_sha256=sha(path), elements=0, max_abs=0., l2=0.)
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
    parser.add_argument('--variant', choices=['baseline', 'proposal'], required=True)
    parser.add_argument('--case', choices=['nonempty', 'rank0-empty', 'global-empty'], required=True)
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
    snapshots = out / ('snapshots-' + args.case) / str(rank)
    snapshots.mkdir(parents=True, exist_ok=True)
    report = dict(passed=False, variant=args.variant, case=args.case, rank=rank,
                  allocation=scope, source_commit=SOURCE, proposal_sha256=manifest['sha256']['proposal_distillation_a.py'],
                  input_width=128, response_width=96, local_micro=2, steps=[], comparisons=[],
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
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=0.)
        module = load_external('search_r1.baselines.proposal_gate_' + args.variant,
                               HERE / ('baseline_distillation_e43.py' if args.variant == 'baseline' else 'proposal_distillation_a.py'))
        teacher = module.FrozenTeacher(args.model, tok, 'cuda:' + str(local_rank))
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
        for index, (name, param) in enumerate(named):
            report['comparisons'].append(snapshot(args.variant, snapshots, f'initial-{index}', dict(name=name, master=param)))
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with teacher.on_device():
            report['teacher_active'] = support.teacher_runtime_audit(teacher, device)
            for step in (1, 2):
                total, world = global_token_count(mask)
                if total.item() == 0:
                    # Exact production branch: no HF forward, backward or Adam.
                    assert args.case == 'global-empty' and not optimizer.state
                    assert not any(decoder_counts.values()) and not head_calls and not teacher_base_calls
                    report['steps'].append(dict(step=step, globally_empty_skipped=True, optimizer_steps=0))
                    for index, (name, param) in enumerate(named):
                        report['comparisons'].append(snapshot(args.variant, snapshots, f'step{step}-empty-{index}', dict(name=name, master=param)))
                    continue
                optimizer.zero_grad(set_to_none=True)
                loss = module.opsd_loss(model, teacher, tensors, tok, [['Paris'], ['Paris']], settings, total / world)
                assert all(count == 2 * step - 1 for count in decoder_counts.values())
                assert torch.isfinite(loss)
                report['comparisons'].append(snapshot(args.variant, snapshots, f'step{step}-loss', dict(loss=loss)))
                loss.backward()
                assert all(count == 2 * step for count in decoder_counts.values())
                assert all(param.grad is not None and torch.isfinite(param.grad).all() for _, param in named)
                raw_covered = 0
                for index, (name, param) in enumerate(named):
                    raw_covered += param.grad.numel() - param._shard_numel_padded
                    report['comparisons'].append(snapshot(args.variant, snapshots, f'step{step}-rawgrad-{index}', dict(name=name, grad=param.grad)))
                assert raw_covered == valid_elements
                norm = model.clip_grad_norm_(1.)
                assert torch.isfinite(norm) and norm > 0
                report['comparisons'].append(snapshot(args.variant, snapshots, f'step{step}-norm', dict(norm=norm)))
                for index, (name, param) in enumerate(named):
                    report['comparisons'].append(snapshot(args.variant, snapshots, f'step{step}-clipgrad-{index}', dict(name=name, grad=param.grad)))
                tracked = next(param for _, param in named if param.grad.abs().max() > 0)
                tracked_index = int(tracked.grad.flatten().abs().argmax())
                before = tracked.detach().flatten()[tracked_index].clone()
                optimizer.step()
                assert float((tracked.detach().flatten()[tracked_index] - before).abs()) > 0
                for index, (name, param) in enumerate(named):
                    state = optimizer.state[param]
                    assert set(state) == {'step', 'exp_avg', 'exp_avg_sq'} and int(state['step']) == step
                    report['comparisons'].append(snapshot(args.variant, snapshots, f'step{step}-postadam-{index}',
                        dict(name=name, master=param, step=state['step'], exp_avg=state['exp_avg'], exp_avg_sq=state['exp_avg_sq'])))
                report['steps'].append(dict(step=step, loss=float(loss.detach()), grad_norm=float(norm),
                    local_raw_gradient_elements=raw_covered, decoder_calls=dict(decoder_counts), optimizer_steps=step))
            assert all(param.grad is None for param in teacher.model.parameters())
        report['teacher_returned'] = support.teacher_runtime_audit(teacher, 'cpu')
        report['teacher_audit'] = teacher.audit_metrics(optimizer)
        expected_bases = 0 if args.case == 'global-empty' else 2
        expected_heads = (0 if args.case == 'global-empty' else
                          12 if args.variant == 'baseline' else
                          0 if args.case == 'rank0-empty' and rank == 0 else 6)
        assert len(teacher_base_calls) == expected_bases and len(head_calls) == expected_heads
        report.update(teacher_head_calls=len(head_calls), teacher_base_calls=len(teacher_base_calls),
                      teacher_scored_tokens=teacher.scored_tokens, decoder_counts=dict(decoder_counts))
        report['comparisons'].append(snapshot(args.variant, snapshots, 'teacher-scored', dict(scored_tokens=teacher.scored_tokens)))
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
            (out / f'{args.case}-{args.variant}.json').write_text(json.dumps(dict(passed=True, ranks=gathered,
                scope='Unapplied external A, exact full-shard loss/gradient/master/Adam comparison only on short128/96 synthetic contexts; not production, vLLM or worst-width capacity acceptance'), indent=2))
        dist.barrier()
        dist.destroy_process_group()
    except BaseException as error:
        report.update(error=repr(error), traceback=traceback.format_exc())
        (out / f'{args.case}-{args.variant}-rank{rank}-failure.json').write_text(json.dumps(report, indent=2))
        raise  # torchrun must terminate divergent peers, never retry this group.


if __name__ == '__main__':
    main()
