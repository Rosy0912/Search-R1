"""Actual-model CPU/GPU FP32 teacher equivalence. Run only inside owned Slurm allocation."""
import argparse
import json
import os
from pathlib import Path
import socket
import time


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=os.environ.get('OPSD_TEACHER'))
    p.add_argument('--output', default=str(Path(os.environ.get('OPSD_EVIDENCE', '.')) / 'gpu-teacher-equivalence.json'))
    p.add_argument('--threads', type=int, default=8)
    args = p.parse_args()
    allocation = allocation_scope()
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from types import SimpleNamespace
    from search_r1.baselines.distillation import FrozenTeacher, opsd_loss, teacher_inputs
    from search_r1.baselines.rollout import padded
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = padded([tok.encode('Question: What is the capital of France?\nAssistant:')], tok.pad_token_id, True)
    response = padded([tok.encode('<search>France capital</search>\n<information>Paris is the capital.</information>\n<answer>Paris</answer>')], tok.pad_token_id)
    ids = torch.cat([prompts, response], dim=1).cuda()
    attention = torch.ones_like(ids)
    mask = torch.ones_like(response, dtype=torch.float32).cuda()
    # Observations remain attended while their prediction positions are excluded.
    mask[:, 8:16] = 0
    tensors = dict(prompts=prompts.cuda(), responses=response.cuda(), input_ids=ids,
                   attention_mask=attention, position_ids=attention.cumsum(-1)-1, loss_mask=mask)
    targets = [['Paris']]
    settings = SimpleNamespace(chunk_size=16, beta=0., temperature=1., component_clip=.05,
                               max_student_logits_gib=16.)
    teacher = FrozenTeacher(args.model, tok, 'cpu')
    student = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                  attn_implementation='flash_attention_2').cuda().eval()
    student.requires_grad_(True)
    student_dtypes = sorted({str(p.dtype) for p in student.parameters()})
    if student_dtypes != ['torch.bfloat16']:
        raise AssertionError('Numerical student must have BF16 parameters')
    teacher_cpu_runtime = teacher_runtime_audit(teacher, 'cpu')
    tid, tm, pos = teacher_inputs(tok, prompts, response, torch.ones_like(response), targets,
                                 teacher.model.config.max_position_embeddings)

    def targets_now():
        with teacher.measure(), torch.no_grad():
            h = teacher.model.base_model(input_ids=tid.to(teacher.device),
                    attention_mask=tm.to(teacher.device), use_cache=False).last_hidden_state
            selected = h.gather(1, pos.to(teacher.device)[..., None].expand(-1, -1, h.shape[-1]))
            return teacher.model.get_output_embeddings()(selected).float().cpu()

    started = time.perf_counter()
    cpu_logits = targets_now()
    loss_cpu = opsd_loss(student, teacher, tensors, tok, targets, settings, mask.sum())
    loss_cpu.backward()
    reference_grads = {name: v.grad.detach().cpu().clone() for name, v in student.named_parameters() if v.grad is not None}
    student.zero_grad(set_to_none=True)
    teacher.execution_device = torch.device('cuda', torch.cuda.current_device())
    with teacher.on_device():
        teacher_gpu_runtime = teacher_runtime_audit(teacher, teacher.execution_device)
        gpu_logits = targets_now()
        loss_gpu = opsd_loss(student, teacher, tensors, tok, targets, settings, mask.sum())
        loss_gpu.backward()
        torch.cuda.synchronize()
    squared_diff = squared_reference = 0.
    max_grad_difference = 0.
    count = 0
    for name, param in student.named_parameters():
        if param.grad is None:
            if name in reference_grads:
                raise AssertionError('Gradient presence changed: ' + name)
            continue
        actual = param.grad.detach().cpu().float()
        expected = reference_grads.pop(name).float()
        difference = actual - expected
        squared_diff += float(difference.square().sum())
        squared_reference += float(expected.square().sum())
        max_grad_difference = max(max_grad_difference, float(difference.abs().max()))
        count += param.numel()
    if reference_grads:
        raise AssertionError('GPU gradient missing')
    cp, gp = cpu_logits.softmax(-1), gpu_logits.softmax(-1)
    probability_difference = float((cp-gp).abs().max())
    kl = (cp*(cpu_logits.log_softmax(-1)-gpu_logits.log_softmax(-1))).sum(-1)
    relative_gradient_l2 = (squared_diff / max(squared_reference, 1e-30)) ** .5
    loss_difference = abs(float(loss_cpu.detach())-float(loss_gpu.detach()))
    result = dict(allocation=allocation, job_id=allocation['job_id'], hostname=allocation['hostname'], model=args.model,
        teacher_cpu_runtime=teacher_cpu_runtime, teacher_gpu_runtime=teacher_gpu_runtime,
        teacher_after_return_runtime=teacher_runtime_audit(teacher, 'cpu'), student_parameter_dtypes=student_dtypes,
        teacher_dtype='float32', teacher_attention='eager', tf32=False,
        teacher_cpu_gpu_logits_max_abs=float((cpu_logits-gpu_logits).abs().max()),
        teacher_probability_max_abs=probability_difference, teacher_kl_max=float(kl.max()),
        cpu_teacher_loss=float(loss_cpu.detach()), gpu_teacher_loss=float(loss_gpu.detach()),
        loss_absolute_difference=loss_difference, all_student_gradient_parameters_compared=count,
        student_gradient_relative_l2=relative_gradient_l2, student_gradient_max_abs=max_grad_difference,
        tolerances=dict(probability_max_abs=1e-5, loss_absolute=1e-5, gradient_relative_l2=.01, gradient_max_abs=5e-4),
        elapsed_seconds=time.perf_counter()-started, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        scope='Actual 3B short-context full-vocabulary clipped forward KL and gradients of every student parameter; synthetic response observation mask, no optimizer step')
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-6)
    result['teacher_audit'] = teacher.audit_metrics(optimizer)
    result['passed'] = (probability_difference <= 1e-5 and loss_difference <= 1e-5
        and relative_gradient_l2 <= .01 and max_grad_difference <= 5e-4
        and torch.isfinite(loss_cpu) and torch.isfinite(loss_gpu))
    result['passed'] = bool(result['passed'])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    if not result['passed']:
        raise AssertionError('CPU/GPU FP32 teacher numerical equivalence gate failed')


if __name__ == '__main__':
    main()
