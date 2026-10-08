"""One isolated eight-rank candidate. torchrun supervises failures, never retry within NCCL group.

Launch separate subprocesses for local micro 1/2/4/8. Live same-job GPU E5/FAISS
must already be running. A failed rank exits immediately; torchrun kills peers.
This probes worst-width actor update residency, not vLLM rollout or whole-job acceptance.
"""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import socket
import time
import traceback


def allocation_scope():
    job = os.environ.get('SLURM_JOB_ID')
    if not job:
        raise RuntimeError('Requires an owned Slurm allocation')
    node = socket.gethostname().split('.')[0]
    nodelist = os.environ.get('SLURM_JOB_NODELIST')
    cgroup = Path('/proc/self/cgroup').read_text()
    import re
    if not re.search(r'(?:^|/)job_' + re.escape(job) + r'(?:/|$)', cgroup, re.MULTILINE):
        raise RuntimeError('Process is outside its declared Slurm job cgroup')
    if node != 'h100-10' or nodelist != 'h100-10':
        raise RuntimeError('Probe requires the single local h100-10 allocation')
    return dict(job_id=job, hostname=socket.gethostname(), node_short=node,
                slurm_job_nodelist=nodelist, own_job_cgroup=cgroup)


def teacher_runtime_audit(teacher, expected_device):
    import torch
    expected_device = torch.device(expected_device)
    dtypes = sorted({str(p.dtype) for p in teacher.model.parameters()})
    devices = sorted({str(p.device) for p in teacher.model.parameters()})
    attention = teacher.model.config._attn_implementation
    if dtypes != ['torch.float32'] or attention != 'eager':
        raise AssertionError('Teacher FP32/eager invariant failed')
    if devices != [str(expected_device)] or teacher.device != expected_device:
        raise AssertionError('Teacher active device invariant failed')
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise AssertionError('TF32 unexpectedly enabled')
    return dict(parameter_dtypes=dtypes, parameter_devices=devices, attention=attention,
                active_device=str(teacher.device), matmul_tf32=False, cudnn_tf32=False)



def audited_checkpoint_training(base, model):
    """Match the production actor train/dropout modes and count real recomputation."""
    import torch
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()
    decoder_layers = [module for module in base.modules() if isinstance(module, Qwen2DecoderLayer)]
    flags = dict(student_wrapper_training=model.training, student_base_training=base.training,
                 student_decoder_container_training=base.model.training,
                 student_gradient_checkpointing=base.model.gradient_checkpointing,
                 decoder_count=len(decoder_layers), all_decoder_training=all(layer.training for layer in decoder_layers),
                 checkpoint_use_reentrant=base.model._gradient_checkpointing_func.keywords.get('use_reentrant'))
    if (not all(flags[key] for key in ('student_wrapper_training', 'student_base_training',
            'student_decoder_container_training', 'student_gradient_checkpointing', 'all_decoder_training'))
            or flags['decoder_count'] != base.config.num_hidden_layers or flags['decoder_count'] < 1
            or flags['checkpoint_use_reentrant'] is not False):
        raise AssertionError('Actual training/non-reentrant decoder checkpoint mode is inactive')
    counts = {str(index): 0 for index in range(len(decoder_layers))}
    handles = []
    for index, layer in enumerate(decoder_layers):
        key = str(index)
        def count_call(_module, _args, key=key):
            counts[key] += 1
        handles.append(layer.register_forward_pre_hook(count_call))
    return flags, counts, handles


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=os.environ.get('OPSD_TEACHER'))
    p.add_argument('--micro', type=int, required=True, choices=[1, 2, 4, 8])
    p.add_argument('--input-width', type=int, default=6144)
    p.add_argument('--response-width', type=int, default=4096)
    p.add_argument('--output-dir', default=os.environ.get('OPSD_EVIDENCE', '.'))
    args = p.parse_args()
    allocation = allocation_scope()
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from types import SimpleNamespace
    from verl.utils.fsdp_utils import get_fsdp_wrap_policy
    from search_r1.baselines.distillation import FrozenTeacher, opsd_loss
    rank, local_rank, world = int(os.environ['RANK']), int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE'])
    if world != 8:
        raise RuntimeError('Capacity gate requires eight ranks')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result = dict(rank=rank, local_rank=local_rank, world_size=world, local_micro=args.micro,
                  allocation=allocation, job_id=allocation['job_id'], hostname=allocation['hostname'], passed=False,
                  input_width=args.input_width, response_width=args.response_width)
    started = time.perf_counter()
    try:
        torch.set_num_threads(4)
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
        baseline_free, gpu_total = torch.cuda.mem_get_info(device)
        external_baseline = gpu_total - baseline_free
        result.update(gpu_total_bytes=gpu_total, external_baseline_used_bytes=external_baseline)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(42)
        dist.init_process_group('nccl', timeout=timedelta(minutes=15))
        tok = AutoTokenizer.from_pretrained(args.model)
        base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32,
                                    attn_implementation='flash_attention_2')
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model = FSDP(base, device_id=local_rank, use_orig_params=False,
                     auto_wrap_policy=get_fsdp_wrap_policy(base), sharding_strategy=ShardingStrategy.FULL_SHARD,
                     mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                                    buffer_dtype=torch.float32), sync_module_states=True,
                     forward_prefetch=False)
        training_flags, decoder_calls, checkpoint_handles = audited_checkpoint_training(base, model)
        result['student_checkpoint_training_runtime'] = training_flags
        result['decoder_forward_pre_hook_counts'] = decoder_calls
        student_dtypes = sorted({str(p.dtype) for p in model.parameters()})
        if student_dtypes != ['torch.float32']:
            raise AssertionError('FSDP student master parameters must be FP32')
        result['student_master_parameter_dtypes'] = student_dtypes
        result['student_mixed_precision_parameter_dtype'] = str(model.mixed_precision.param_dtype)
        result['allocator_configuration'] = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')
        result['allocator_backend'] = torch.cuda.get_allocator_backend()
        phase_records = []
        def memory_phase(label):
            torch.cuda.synchronize()
            stats = torch.cuda.memory_stats()
            free, total = torch.cuda.mem_get_info()
            phase_records.append(dict(label=label,allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),max_allocated=torch.cuda.max_memory_allocated(),max_reserved=torch.cuda.max_memory_reserved(),device_free=free,device_total=total,inactive_split_current=stats.get('inactive_split_bytes.all.current'),inactive_split_peak=stats.get('inactive_split_bytes.all.peak'),active_current=stats.get('active_bytes.all.current'),num_alloc_retries=stats.get('num_alloc_retries'),num_ooms=stats.get('num_ooms')))
        result['memory_phases'] = phase_records
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=0.)
        teacher = FrozenTeacher(args.model, tok, f'cuda:{local_rank}')
        handles = [
            teacher.model.base_model.register_forward_pre_hook(lambda *_: memory_phase('teacher_base_before')),
            teacher.model.base_model.register_forward_hook(lambda *_: memory_phase('teacher_base_after')),
            model.register_forward_pre_hook(lambda *_: memory_phase('student_forward_before')),
            model.register_forward_hook(lambda *_: memory_phase('student_forward_after')),
        ]
        prompt_width = args.input_width - args.response_width
        if prompt_width < 1:
            raise ValueError('Response must leave a prompt position')
        pieces = tok.encode('Question: Paris France capital <search>France capital</search> <information>Paris</information> <answer>Paris</answer>', add_special_tokens=False)
        pieces = [v for v in pieces if v != tok.pad_token_id]
        sequence = (pieces * ((args.input_width + len(pieces)-1)//len(pieces)))[:args.input_width]
        ids = torch.tensor(sequence, device=device)[None, :].repeat(args.micro, 1)
        attention = torch.ones_like(ids)
        mask = torch.ones((args.micro, args.response_width), device=device)
        mask[:, 16:32] = 0
        tensors = dict(prompts=ids[:, :prompt_width], responses=ids[:, prompt_width:], input_ids=ids,
                       attention_mask=attention, position_ids=attention.cumsum(-1)-1, loss_mask=mask)
        settings = SimpleNamespace(chunk_size=16, beta=0., temperature=1., component_clip=.05,
                                   max_student_logits_gib=64., loss_checkpointing=True)
        dist.barrier()
        torch.cuda.synchronize()
        result['pre_cleanup_allocated_bytes'] = torch.cuda.memory_allocated()
        result['pre_cleanup_reserved_bytes'] = torch.cuda.memory_reserved()
        result['pre_cleanup_device_free_bytes'] = torch.cuda.mem_get_info()[0]
        # Match production init/rollout-transition empty_cache; live tensors stay.
        torch.cuda.empty_cache()
        result['post_cleanup_allocated_bytes'] = torch.cuda.memory_allocated()
        result['post_cleanup_reserved_bytes'] = torch.cuda.memory_reserved()
        result['post_cleanup_device_free_bytes'] = torch.cuda.mem_get_info()[0]
        if result['pre_cleanup_allocated_bytes'] != result['post_cleanup_allocated_bytes']:
            raise AssertionError('Clean-start cache release unexpectedly changed live allocation')
        torch.cuda.reset_peak_memory_stats()
        memory_phase("clean_update_start")
        update_started = time.perf_counter()
        with teacher.on_device():
            memory_phase('teacher_cuda_ready')
            result['teacher_active_runtime'] = teacher_runtime_audit(teacher, device)
            # Second update keeps Adam states resident during the next full graph.
            optimizer_step_records = []
            for optimizer_step in range(2):
                step_started = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                memory_phase(f"step{optimizer_step+1}_zero_grad")
                loss = opsd_loss(model, teacher, tensors, tok, [['Paris']] * args.micro, settings, mask.sum())
                memory_phase(f"step{optimizer_step+1}_loss_ready")
                expected_forward_calls = 2 * optimizer_step + 1
                if any(count != expected_forward_calls for count in decoder_calls.values()):
                    raise AssertionError('Decoder forward checkpoint-call count differs from one forward per step')
                loss.backward()
                memory_phase(f"step{optimizer_step+1}_backward_done")
                expected_recomputed_calls = 2 * (optimizer_step + 1)
                if any(count != expected_recomputed_calls for count in decoder_calls.values()):
                    raise AssertionError('Actual backward decoder checkpoint recomputation missing or repeated')
                grad = model.clip_grad_norm_(1.)
                if not torch.isfinite(loss) or not torch.isfinite(grad) or grad <= 0:
                    raise AssertionError('Nonfinite or zero loss/gradient')
                # A nonzero real shard gradient proves the optimizer changes this shard.
                tracked = next(v for v in model.parameters() if v.grad is not None and v.grad.abs().max() > 0)
                tracked_index = int(tracked.grad.detach().abs().flatten().argmax())
                before = tracked.detach().flatten()[tracked_index].clone()
                memory_phase(f"step{optimizer_step+1}_clip_done")
                optimizer.step()
                memory_phase(f"step{optimizer_step+1}_adam_done")
                delta = float((tracked.detach().flatten()[tracked_index]-before).abs())
                if delta <= 0:
                    raise AssertionError('No student shard parameter update')
                torch.cuda.synchronize()
                optimizer_step_records.append(dict(step=optimizer_step+1,seconds=time.perf_counter()-step_started,loss=float(loss.detach()),grad_norm=float(grad),tracked_parameter_delta=delta))
            torch.cuda.synchronize()
            result.update(passed=True, loss=float(loss.detach()), grad_norm=float(grad),
                tracked_max_gradient_component_delta_abs=delta, optimizer_steps=2, optimizer_step_records=optimizer_step_records, update_seconds=time.perf_counter()-update_started,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                teacher_audit=teacher.audit_metrics(optimizer),
                scope='Eight-rank actual 3B FP32-master FSDP/BF16 Flash/checkpointed full-vocab OPSD update with FP32/eager CUDA teacher; two consecutive optimizer updates with resident Adam states; synthetic worst-width trajectories. vLLM residency not tested here; must pass production outer separately')
        memory_phase('teacher_cpu_returned')
        for handle in handles + checkpoint_handles: handle.remove()
        result['teacher_returned_runtime'] = teacher_runtime_audit(teacher, 'cpu')
        result['teacher_cost'] = teacher.cost_metrics()
        result['estimated_peak_device_used_bytes'] = external_baseline + result['peak_reserved_bytes'] + 1024**3
        result['minimum_required_free_fraction'] = .15
        result['headroom_estimate_includes_nccl_slack_bytes'] = 1024**3
        result['passed'] = result['passed'] and result['estimated_peak_device_used_bytes'] <= .85 * gpu_total
        gathered = [None] * world
        dist.all_gather_object(gathered, result)
        if rank == 0:
            aggregate = dict(passed=all(r['passed'] for r in gathered), local_micro=args.micro, ranks=gathered,
                scope='All eight ranks complete full backward and optimizer; conservative PyTorch peak + pre-model external usage + 1GiB NCCL slack headroom estimate, not continuous NVML residency measurement or vLLM gate')
            (output / f'micro-{args.micro}.json').write_text(json.dumps(aggregate, indent=2))
        if not all(r['passed'] for r in gathered):
            raise RuntimeError('Eight-rank capacity headroom gate failed')
        dist.barrier()
        dist.destroy_process_group()
    except BaseException as exc:
        result.update(passed=False, error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        result['oom'] = 'out of memory' in str(exc).lower()
        # Do not barrier or destroy an already divergent NCCL group. Let torchrun
        # terminate remaining workers before the next isolated candidate begins.
        result['elapsed_seconds'] = time.perf_counter()-started
        (output / f'micro-{args.micro}-rank-{rank}.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
        raise
    result['elapsed_seconds'] = time.perf_counter()-started
    (output / f'micro-{args.micro}-rank-{rank}.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
