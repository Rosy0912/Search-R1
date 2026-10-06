# E1 hardware candidate. Two synthetic Adam calls; never counted as RL outer updates.
# Same official Instruct asset, null component clip, user-context teacher, local micro4.
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


def synthetic_tensors(tok, micro, input_width, response_width, device):
    """Conservative full-width ChatML inputs; synthetic, never a data split."""
    import torch
    from search_r1.baselines.distillation import teacher_inputs
    prompt_width = input_width - response_width
    suffix = tok.encode('<|im_end|>\n<|im_start|>assistant\n', add_special_tokens=False)
    encoded = tok.apply_chat_template([{'role':'user','content':
        'Question: What is the capital of France? Follow <think>, <search>, <answer>. ' +
        'Context: Paris France. ' * 2048}], tokenize=True, add_generation_prompt=True)
    if len(encoded) < prompt_width or encoded[-len(suffix):] != suffix:
        raise ValueError('Official ChatML synthetic prompt boundaries differ')
    prefix = encoded[:prompt_width-len(suffix)] + suffix
    pieces = tok.encode('<think>Paris</think><search>France capital</search><information>Paris</information><answer>Paris</answer>', add_special_tokens=False)
    pieces = [v for v in pieces if v != tok.pad_token_id]
    response = (pieces * ((response_width + len(pieces)-1)//len(pieces)))[:response_width]
    prompts=torch.tensor(prefix,device=device)[None,:].repeat(micro,1)
    responses=torch.tensor(response,device=device)[None,:].repeat(micro,1)
    ids=torch.cat([prompts,responses],-1);attention=torch.ones_like(ids)
    mask=torch.ones((micro,response_width),device=device);mask[:,16:32]=0
    tid,tmask,positions=teacher_inputs(tok,prompts,responses,attention[:,-response_width:],[['Paris']]*micro,prompt_mode='user_context')
    assert prompts.shape==(micro,prompt_width) and ids.shape==(micro,input_width)
    assert positions.shape==(micro,response_width) and int(tmask.sum())>int(attention.sum())
    return dict(prompts=prompts,responses=responses,input_ids=ids,attention_mask=attention,
        position_ids=attention.cumsum(-1)-1,loss_mask=mask),dict(prompt_width=prompt_width,
        full_input_width=input_width,response_width=response_width,verified_chat_suffix_tokens=len(suffix),
        teacher_context_width=tid.shape[-1],teacher_only_hint_tokens=int(tmask[0].sum())-input_width,
        actual_user_context_adapter_pass=True,scope='synthetic full-width capacity only')


def gpu_full_path_repeatability(model, teacher, tok, device, settings):
    """Repeat the same native full teacher/student path before diagnostic Adam.

    The short tensors are synthetic ChatML inputs, not an RL update or data
    split. Both calls use the native full-position student head and the full
    four-row teacher forward. This tests repeatability of one path, not
    equivalence of teacher row segmentation or a response-only head. Save all
    observed differences and reject substantial differences before training.
    """
    import math
    import torch
    from search_r1.baselines.distillation import opsd_loss
    tensors,_=synthetic_tensors(tok,4,96,32,device)
    denominator=tensors['loss_mask'].sum()
    if settings.response_only_logits is not False or type(settings.teacher_forward_micro_batch) is not int or settings.teacher_forward_micro_batch != 4:
        raise ValueError('Native full student head and exact full teacher row micro4 required')
    variants=[settings,settings]
    losses=[];old_gradients=None;squared_old=squared_new=squared_delta=dot=0.0
    maximum_abs_delta=0.0;members=0;parameter_statistics=[];teacher_outputs=[];teacher_logits=[]
    def capture_teacher(_module,_args,value):teacher_outputs.append(value.detach().cpu())
    teacher_hook=teacher.model.get_output_embeddings().register_forward_hook(capture_teacher)
    for variant in variants:
        model.zero_grad(set_to_none=True)
        loss=opsd_loss(model,teacher,tensors,tok,[['Paris']]*4,variant,denominator)
        if not torch.isfinite(loss):raise ValueError('Nonfinite short GPU repeatability loss')
        loss.backward();torch.cuda.synchronize();losses.append(float(loss.detach()))
        teacher_logits.append(torch.cat(teacher_outputs,dim=1));teacher_outputs.clear()
        if old_gradients is None:
            old_gradients=[]
            for name,parameter in model.named_parameters():
                value=parameter.grad
                if value is not None and not torch.isfinite(value).all():
                    raise ValueError('Nonfinite first native path GPU gradient')
                old_gradients.append((name,None if value is None else value.detach().cpu().clone()))
        else:
            parameters=list(model.named_parameters())
            if len(parameters)!=len(old_gradients):raise ValueError('FSDP gradient inventory changed')
            for (name,parameter),(old_name,before) in zip(parameters,old_gradients):
                if name!=old_name:raise ValueError('FSDP parameter names changed')
                value=parameter.grad
                if (value is None)!=(before is None):raise ValueError('FSDP gradient participation changed')
                if value is None:continue
                if not torch.isfinite(value).all():raise ValueError('Nonfinite repeated native path GPU gradient')
                after=value.detach().cpu()
                if before.shape!=after.shape:raise ValueError('FSDP gradient shard shape changed')
                # Each shard is checked in bounded blocks; no second full
                # FP64 model-sized buffer is held on CPU or GPU.
                parameter_old=parameter_new=parameter_delta=parameter_max=0.0
                for start in range(0,before.numel(),1048576):
                    a=before.flatten()[start:start+1048576].double()
                    b=after.flatten()[start:start+1048576].double();delta=b-a
                    parameter_old+=float((a*a).sum());parameter_new+=float((b*b).sum())
                    parameter_delta+=float((delta*delta).sum());dot+=float((a*b).sum())
                    if delta.numel():parameter_max=max(parameter_max,float(delta.abs().max()))
                squared_old+=parameter_old;squared_new+=parameter_new;squared_delta+=parameter_delta
                maximum_abs_delta=max(maximum_abs_delta,parameter_max)
                parameter_statistics.append(dict(name=name,shard_elements=before.numel(),
                    native_first_squared_norm=parameter_old,native_repeat_squared_norm=parameter_new,
                    difference_squared_norm=parameter_delta,maximum_abs_difference=parameter_max))
                members+=1
    teacher_hook.remove();model.zero_grad(set_to_none=True)
    import torch.distributed as dist
    statistics=torch.tensor([squared_old,squared_new,squared_delta,dot],device=device,dtype=torch.float64)
    dist.all_reduce(statistics,op=dist.ReduceOp.SUM)
    global_old,global_new,global_delta,global_dot=statistics.tolist()
    relative_l2=math.sqrt(global_delta/max(global_old,1e-300))
    cosine=global_dot/math.sqrt(max(global_old*global_new,1e-300))
    relative_loss=abs(losses[1]-losses[0])/max(abs(losses[0]),1e-12)
    if teacher_logits[0].shape!=teacher_logits[1].shape:raise ValueError('Teacher q projection shape changed')
    logits_max=float((teacher_logits[0]-teacher_logits[1]).abs().max())
    q_before=teacher_logits[0].float().softmax(-1);q_after=teacher_logits[1].float().softmax(-1)
    q_l1=float((q_after-q_before).abs().sum(-1).mean())
    result=dict(comparison_kind='SAME_NATIVE_FULL_HEAD_FULL_TEACHER4_PATH_REPEATABILITY',
        response_only_logits=False, first_teacher_forward_micro_batch=4,
        repeat_teacher_forward_micro_batch=4,
        native_first_full_head_loss=losses[0],native_repeat_full_head_loss=losses[1],
        loss_absolute_difference=abs(losses[1]-losses[0]),loss_relative_difference=relative_loss,
        shard_gradient_relative_l2=relative_l2,shard_gradient_cosine=cosine,
        shard_gradient_max_abs_difference=maximum_abs_delta,checked_gradient_shards=members,
        local_native_first_squared_gradient_norm=squared_old,local_native_repeat_squared_gradient_norm=squared_new,
        global_native_first_squared_gradient_norm=global_old,global_native_repeat_squared_gradient_norm=global_new,
        parameter_statistics=parameter_statistics,teacher_FP32_logits_max_absolute_difference=logits_max,
        teacher_q_mean_probability_l1_difference=q_l1,maximum_teacher_q_mean_l1_difference=1e-4,
        maximum_relative_gradient_l2=.02,minimum_gradient_cosine=.9999,maximum_relative_loss=.002,
        scope='Actual official-model eight-rank native full-head/full-teacher4 BF16 FSDP short synthetic same-path gradient repeatability; not teacher-row2 equivalence; no Adam calls or RL outer iterations')
    result['passed']=(members>0 and global_old>0 and global_new>0 and
        all(math.isfinite(v) for v in (relative_l2,cosine,relative_loss,q_l1,logits_max)) and
        relative_l2<=.02 and cosine>=.9999 and relative_loss<=.002 and q_l1<=1e-4)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=os.environ.get('OPSD_E1_OFFICIAL_MODEL'))
    p.add_argument('--micro', type=int, required=True, choices=[1, 2, 4, 8])
    p.add_argument('--input-width', type=int, default=6144)
    p.add_argument('--response-width', type=int, default=4096)
    p.add_argument('--output-dir', default=os.environ.get('OPSD_EVIDENCE', '.'))
    args = p.parse_args()
    if args.micro != 4 or args.input_width != 6144 or args.response_width != 4096:
        raise ValueError('Fixed E1 local micro4 / conservative 6144 input / 4096 response probe required')
    import subprocess, hashlib
    repo = Path(os.environ['SEARCH_R1_REPO']).resolve()
    if subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip() != os.environ['E1_EXPECTED_SOURCE'] or subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip():
        raise ValueError('Actual fixed source differs or is dirty')
    from search_r1.baselines.e1_profile import verify_official_assets
    verify_official_assets(args.model, os.environ['OPSD_E1_OFFICIAL_MANIFEST'])
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
                  input_width=args.input_width, response_width=args.response_width, source_commit=os.environ['E1_EXPECTED_SOURCE'], official_revision='aa8e72537993ba99e69dfaafa59ed015b17504d1', component_clip=None, teacher_prompt_mode='user_context', max_student_logits_gib=8.0,response_only_logits=False,teacher_forward_micro_batch=4)
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
        if result['allocator_configuration'] or os.environ.get('PYTORCH_ALLOC_CONF') or result['allocator_backend'] != 'native':
            raise ValueError('Native allocator with default configuration required')
        phase_records = []
        def memory_phase(label):
            torch.cuda.synchronize()
            stats = torch.cuda.memory_stats()
            free, total = torch.cuda.mem_get_info()
            phase_records.append(dict(label=label,allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),max_allocated=torch.cuda.max_memory_allocated(),max_reserved=torch.cuda.max_memory_reserved(),device_free=free,device_total=total,inactive_split_current=stats.get('inactive_split_bytes.all.current'),inactive_split_peak=stats.get('inactive_split_bytes.all.peak'),active_current=stats.get('active_bytes.all.current'),num_alloc_retries=stats.get('num_alloc_retries'),num_ooms=stats.get('num_ooms')))
        result['memory_phases'] = phase_records
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=1e-2)
        teacher = FrozenTeacher(args.model, tok, f'cuda:{local_rank}')
        # Match the production old-policy LP local8/global64 path with the
        # teacher on CPU. It computes the original packed-token probabilities,
        # including the native entropy implementation, without changing Adam.
        from omegaconf import OmegaConf
        from verl.workers.actor.dp_actor import DataParallelPPOActor
        lp_actor=DataParallelPPOActor(OmegaConf.create(dict(use_remove_padding=True,
            ulysses_sequence_parallel_size=1)),model,optimizer)
        lp_tensors,_=synthetic_tensors(tok,8,args.input_width,args.response_width,device)
        model.eval();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        lp_started=time.perf_counter()
        with torch.no_grad():lp_entropy,lp_values=lp_actor._forward_micro_batch(lp_tensors,temperature=1.)
        torch.cuda.synchronize()
        lp_result=dict(local_micro=8,global_micro=64,input_width=args.input_width,
            response_width=args.response_width,logprob_shape=list(lp_values.shape),
            entropy_shape=list(lp_entropy.shape),logprob_finite=bool(torch.isfinite(lp_values).all()),
            entropy_finite=bool(torch.isfinite(lp_entropy).all()),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            teacher_storage_runtime=teacher_runtime_audit(teacher,'cpu'),seconds=time.perf_counter()-lp_started,
            scope='Actual production DP actor packed old-logprob micro8, no reference KL/PG/Adam/RL outer')
        lp_result['estimated_peak_device_used_bytes']=external_baseline+lp_result['peak_reserved_bytes']+1024**3
        lp_result['passed']=(lp_result['logprob_shape']==[8,args.response_width] and
            lp_result['entropy_shape']==[8,args.response_width] and lp_result['logprob_finite'] and
            lp_result['entropy_finite'] and lp_result['estimated_peak_device_used_bytes']<=.85*gpu_total)
        result['old_logprob_capacity']=lp_result
        lp_pass=torch.tensor(int(lp_result['passed']),device=device);dist.all_reduce(lp_pass,op=dist.ReduceOp.MIN)
        (output/f'old-logprob-rank-{rank}.json').write_text(json.dumps(lp_result,indent=2)+'\n')
        if lp_pass.item()!=1:raise AssertionError('Actual requested old-logprob micro8 capacity/numeric gate failed')
        del lp_tensors,lp_values,lp_entropy,lp_actor
        model.train()
        for module in model.modules():
            if isinstance(module,torch.nn.Dropout):module.eval()
        for key in decoder_calls:decoder_calls[key]=0
        handles = [
            teacher.model.base_model.register_forward_pre_hook(lambda *_: memory_phase('teacher_base_before')),
            teacher.model.base_model.register_forward_hook(lambda *_: memory_phase('teacher_base_after')),
            model.register_forward_pre_hook(lambda *_: memory_phase('student_forward_before')),
            model.register_forward_hook(lambda *_: memory_phase('student_forward_after')),
        ]
        prompt_width = args.input_width - args.response_width
        if prompt_width < 1:
            raise ValueError('Response must leave a prompt position')
        tensors, synthetic_prompt = synthetic_tensors(tok, args.micro, args.input_width, args.response_width, device)
        mask = tensors['loss_mask']
        result['synthetic_prompt_runtime'] = synthetic_prompt
        settings = SimpleNamespace(chunk_size=16, beta=0., temperature=1., component_clip=None, teacher_prompt_mode='user_context',
                                   max_student_logits_gib=8., loss_checkpointing=True, response_only_logits=False,teacher_forward_micro_batch=4)
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
            result['short_native_full_path_repeatability']=gpu_full_path_repeatability(
                model,teacher,tok,device,settings)
            # All ranks must reject together so no peer waits inside a later
            # FSDP collective after another rank fails this numeric gate.
            repeatable=torch.tensor(int(result['short_native_full_path_repeatability']['passed']),device=device)
            dist.all_reduce(repeatable,op=dist.ReduceOp.MIN)
            (output/f'repeatability-rank-{rank}.json').write_text(json.dumps(
                result['short_native_full_path_repeatability'],indent=2)+'\n')
            if repeatable.item()!=1:raise AssertionError('Native full-head/full-teacher4 same-path gradient repeatability failed')
            for key in decoder_calls:decoder_calls[key]=0
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            memory_phase('worst_width_update_start_after_short_repeatability')
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
        result['estimated_peak_device_used_bytes'] = max(
            external_baseline + result['peak_reserved_bytes'] + 1024**3,
            result['old_logprob_capacity']['estimated_peak_device_used_bytes'])
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
